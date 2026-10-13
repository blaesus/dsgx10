// The DSA indexer: f32 scores I(r, j) = sum_h w[r][h] relu(f32(q_h . k_j) * 128^-0.5) over each row's keys, then
// the exact top-2048 by (score desc, position asc).
//
// Scores (k_score): a CTA takes a group of up to RG rows of one session and a chunk of 1024 keys. The rows'
// queries (RoPE'd on load) and head weights sit in shared memory; each warp streams 16-key tiles of the keys
// straight into MMA A fragments (keys as rows, 16 dims a k-step) and runs, per row, 4 n-tiles (8 heads each)
// x 8 k-steps of m16n8k16 against the queries as B. A lane (g, t) holds dims 32t .. 32t+31 of keys g and g+8:
// k-step kk takes dims 32t + 4kk + {0,1} (the MMA's k = 2t, 2t+1) and 32t + 4kk + {2,3} (k = 2t+8, 2t+9), and
// the queries use the same map, so every score is the same chain of MMAs whatever the call. The head sum:
// each lane adds its 8 heads (n-tile order, then head 2t before 2t+1), then the quad adds its four lanes
// (xor 1, then xor 2). Decode and prefill run this one kernel: their scores are identical.
//
// Selection: three histogram passes over the order keys (11, 11 and 10 bits; integer counts, so the atomics'
// order does not matter) find the 32-bit threshold u* and how many keys equal to it are taken (the lowest
// positions); a count pass gives each 4096-key chunk its numbers of keys above and equal to u*, and a write
// pass compacts each chunk in position order at its prefix offset. Every pass recomputes the thresholds from
// the histograms. The last pass leaves the scratch zeroed.
#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"

namespace glm {
namespace {

constexpr int kScoreChunk = 1024;  // keys a scoring CTA
constexpr int kSelChunk = 4096;    // keys a selection CTA
constexpr int kQBlk = 36;          // a 32-dim block of a staged query, padded (bank-conflict free uint2 reads)
constexpr int kQHead = 4 * kQBlk;  // 144 bf16 a head
constexpr int kMaxSelChunks = 256; // 1M keys
constexpr int kH1 = 2048, kH2 = 2048, kH3 = 1024;
constexpr int kRowWords = kH1 + kH2 + kH3 + 4 + 2 * kMaxSelChunks;
// row scratch: hist1 | hist2 | hist3 | params (thr, need_eq, n, -) | chunk counts [chunk][gt, eq]

__device__ __forceinline__ int row_n(int p, int lo, int hi) {  // keys of the row's scan; 0: no selection
  if (p < kTopK) return 0;
  const int end = hi > 0 ? min(hi, p + 1) : p + 1;
  return max(0, end - lo);
}

struct ScoreArgs {
  const uint16_t* q;
  const float* w;
  const int* pos;
  const int* slot;
  const uint32_t* rope;
  IdxCache ik;
  int lo, hi, rows;
  bool decode;  // groups by session (rows <= 16); else consecutive groups of RG (one session)
  float* scores;
  size_t ld;
};

// NT threads: 128 x RG (cap 512), so a CTA whose staged queries fill most of shared memory (RG 8: 74 KB, one CTA
// an SM) still has 16 warps an SM to hide MMA latency
template <int RG, bool FP8K>
constexpr int score_threads() { return RG >= 4 ? 512 : 128 * RG; }

template <int RG, bool FP8K>
__global__ void __launch_bounds__(score_threads<RG, FP8K>()) k_score(ScoreArgs a) {
  constexpr int NT = score_threads<RG, FP8K>(), NWARP = NT / 32;
  extern __shared__ __align__(16) uint8_t smem[];
  uint16_t* qs = reinterpret_cast<uint16_t*>(smem);                      // [RG][32][kQHead]
  float* ws = reinterpret_cast<float*>(qs + RG * kIdxHeads * kQHead);    // [RG][32]
  pdl_trigger();
  pdl_wait();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // the CTA's rows [ra, rb): one session's
  int ra, rb;
  if (a.decode) {
    ra = blockIdx.y;
    if (ra >= a.rows) return;
    const int sl = a.slot ? a.slot[ra] : 0;
    int s0 = ra;
    while (s0 > 0 && (a.slot ? a.slot[s0 - 1] : 0) == sl) --s0;
    if ((ra - s0) % RG) return;  // not the start of a group
    rb = ra + 1;
    while (rb < a.rows && rb - ra < RG && (a.slot ? a.slot[rb] : 0) == sl) ++rb;
  } else {
    ra = blockIdx.y * RG;
    rb = min(a.rows, ra + RG);
  }
  int end = a.lo;  // the group's scan end (positions)
  for (int r = ra; r < rb; ++r) end = max(end, a.lo + row_n(a.pos[r], a.lo, a.hi));
  const int j0 = a.lo + blockIdx.x * kScoreChunk, j1 = min(end, j0 + kScoreChunk);
  if (j0 >= j1) return;
  const size_t sbase = a.slot ? (size_t)a.slot[ra] * a.ik.slot_stride : 0;
  // stage the queries (RoPE on dims 0..63 of each head; logical dim d at block d / 32, offset d % 32) and weights
  int* prow = reinterpret_cast<int*>(ws + RG * kIdxHeads);  // [RG] each row's scan end
  for (int i = tid; i < RG * kIdxHeads * 32; i += NT) {    // (row, head, pair pi)
    const int rr = i / (kIdxHeads * 32), h = (i / 32) % kIdxHeads, pi = i % 32;
    uint16_t* dst = qs + (rr * kIdxHeads + h) * kQHead;
    uint16_t v0 = 0, v1 = 0;
    uint32_t pv = 0;
    if (ra + rr < rb) {
      const uint16_t* src = a.q + ((size_t)(ra + rr) * kIdxHeads + h) * kIdxDim;
      const uint32_t v = *reinterpret_cast<const uint32_t*>(src + 2 * pi);
      float lo, hi;
      rope_pair(bf2f(v), bf2f_hi(v), a.rope[(size_t)a.pos[ra + rr] * 32 + pi], lo, hi);
      v0 = f2bf(lo);
      v1 = f2bf(hi);
      pv = *reinterpret_cast<const uint32_t*>(src + 64 + 2 * pi);
    }
    dst[pi] = v0;                 // dim pi: block 0
    dst[kQBlk + pi] = v1;         // dim 32 + pi: block 1
    *reinterpret_cast<uint32_t*>(dst + (2 + pi / 16) * kQBlk + (2 * pi) % 32) = pv;  // dims 64 + 2pi, +1
  }
  for (int i = tid; i < RG * kIdxHeads; i += NT) {
    const int rr = i / kIdxHeads;
    ws[i] = ra + rr < rb ? a.w[(size_t)(ra + rr) * kIdxHeads + i % kIdxHeads] : 0.f;
  }
  if (tid < RG) prow[tid] = ra + tid < rb ? a.lo + row_n(a.pos[ra + tid], a.lo, a.hi) : a.lo;
  __syncthreads();

  // key tile loads: key g and g+8, dims 32t .. 32t+31
  constexpr int NW = FP8K ? 2 : 4;  // uint4 a key
  auto load = [&](uint4 (&ka)[NW], uint4 (&kb)[NW], float& sa, float& sb, int jt) {
    const int ja = jt + g, jb = jt + g + 8;
    const bool va = ja < j1, vb = jb < j1;
    if constexpr (FP8K) {
      const uint8_t* base = a.ik.k + (sbase + 0) * kIdxDim;
#pragma unroll
      for (int i = 0; i < NW; ++i) {
        ka[i] = va ? ldg_stream128(base + (size_t)ja * kIdxDim + 32 * t + 16 * i) : make_uint4(0, 0, 0, 0);
        kb[i] = vb ? ldg_stream128(base + (size_t)jb * kIdxDim + 32 * t + 16 * i) : make_uint4(0, 0, 0, 0);
      }
      sa = va ? __ldg(a.ik.ks + sbase + ja) : 0.f;
      sb = vb ? __ldg(a.ik.ks + sbase + jb) : 0.f;
    } else {
      const uint16_t* base = reinterpret_cast<const uint16_t*>(a.ik.k) + sbase * kIdxDim;
#pragma unroll
      for (int i = 0; i < NW; ++i) {
        ka[i] = va ? ldg_stream128(base + (size_t)ja * kIdxDim + 32 * t + 8 * i) : make_uint4(0, 0, 0, 0);
        kb[i] = vb ? ldg_stream128(base + (size_t)jb * kIdxDim + 32 * t + 8 * i) : make_uint4(0, 0, 0, 0);
      }
      sa = sb = 1.f;
    }
  };
  uint4 na[NW], nb[NW];
  float nsa, nsb;
  int jt = j0 + warp * 16;
  if (jt < j1) load(na, nb, nsa, nsb, jt);
  for (; jt < j1; jt += 16 * NWARP) {
    uint32_t A[16], B[16];  // key g, key g+8: dims 32t + 2i, +1 in word i
    if constexpr (FP8K) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        fp8x4_to_bf16(na[i].x, A[8 * i + 0], A[8 * i + 1]); fp8x4_to_bf16(na[i].y, A[8 * i + 2], A[8 * i + 3]);
        fp8x4_to_bf16(na[i].z, A[8 * i + 4], A[8 * i + 5]); fp8x4_to_bf16(na[i].w, A[8 * i + 6], A[8 * i + 7]);
        fp8x4_to_bf16(nb[i].x, B[8 * i + 0], B[8 * i + 1]); fp8x4_to_bf16(nb[i].y, B[8 * i + 2], B[8 * i + 3]);
        fp8x4_to_bf16(nb[i].z, B[8 * i + 4], B[8 * i + 5]); fp8x4_to_bf16(nb[i].w, B[8 * i + 6], B[8 * i + 7]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        A[4 * i] = na[i].x; A[4 * i + 1] = na[i].y; A[4 * i + 2] = na[i].z; A[4 * i + 3] = na[i].w;
        B[4 * i] = nb[i].x; B[4 * i + 1] = nb[i].y; B[4 * i + 2] = nb[i].z; B[4 * i + 3] = nb[i].w;
      }
    }
    const float sa = nsa, sb = nsb;
    const int jc = jt;
    if (jt + 16 * NWARP < j1) load(na, nb, nsa, nsb, jt + 16 * NWARP);
#pragma unroll 1
    for (int rr = 0; rr < RG; ++rr) {
      if (ra + rr >= rb) break;
      const uint16_t* qrow = qs + rr * kIdxHeads * kQHead;
      float c[4][4];
#pragma unroll
      for (int n = 0; n < 4; ++n) c[n][0] = c[n][1] = c[n][2] = c[n][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const uint2 b = *reinterpret_cast<const uint2*>(qrow + (8 * n + g) * kQHead + t * kQBlk + 4 * kk);
          mma_bf16(c[n], A[2 * kk], B[2 * kk], A[2 * kk + 1], B[2 * kk + 1], b.x, b.y);
        }
      }
      const float* wr = ws + rr * kIdxHeads;
      float va = 0.f, vb = 0.f;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const float w0 = wr[8 * n + 2 * t], w1 = wr[8 * n + 2 * t + 1];
        va = __fadd_rn(va, __fmul_rn(w0, fmaxf(__fmul_rn(__fmul_rn(c[n][0], sa), kIdxScale), 0.f)));
        va = __fadd_rn(va, __fmul_rn(w1, fmaxf(__fmul_rn(__fmul_rn(c[n][1], sa), kIdxScale), 0.f)));
        vb = __fadd_rn(vb, __fmul_rn(w0, fmaxf(__fmul_rn(__fmul_rn(c[n][2], sb), kIdxScale), 0.f)));
        vb = __fadd_rn(vb, __fmul_rn(w1, fmaxf(__fmul_rn(__fmul_rn(c[n][3], sb), kIdxScale), 0.f)));
      }
      va = __fadd_rn(va, __shfl_xor_sync(0xffffffffu, va, 1));
      vb = __fadd_rn(vb, __shfl_xor_sync(0xffffffffu, vb, 1));
      va = __fadd_rn(va, __shfl_xor_sync(0xffffffffu, va, 2));
      vb = __fadd_rn(vb, __shfl_xor_sync(0xffffffffu, vb, 2));
      float* srow = a.scores + (size_t)(ra + rr) * a.ld - a.lo;
      if (t == 0 && jc + g < prow[rr]) srow[jc + g] = va;
      if (t == 1 && jc + g + 8 < prow[rr]) srow[jc + g + 8] = vb;
    }
  }
}

// ---- selection -------------------------------------------------------------------------------------------
// Block-wide exclusive scan (NT threads) of v; total in *tot.
template <int NT>
__device__ __forceinline__ uint32_t block_excl_scan(uint32_t v, uint32_t* sm, uint32_t* tot) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  __syncthreads();
  if (lane == 31) sm[warp] = x;
  __syncthreads();
  uint32_t before = 0, all = 0;
#pragma unroll
  for (int w = 0; w < NT / 32; ++w) {
    const uint32_t s = sm[w];
    if (w < warp) before += s;
    all += s;
  }
  if (tot) *tot = all;
  return before + x - v;
}

// The bin (scanning from the top) where the count of keys reaches `need`, and the count above it.
template <int NT>
__device__ void find_bin(const uint32_t* hist, int nbins, uint32_t need, uint32_t* sm, int& bin, uint32_t& above) {
  const int per = nbins / NT, i = threadIdx.x;
  uint32_t s = 0;
  for (int d = i * per; d < (i + 1) * per; ++d) s += hist[nbins - 1 - d];
  const uint32_t before = block_excl_scan<NT>(s, sm, nullptr);
  __syncthreads();
  if (before < need && before + s >= need) {
    uint32_t acc = before;
    for (int d = i * per; d < (i + 1) * per; ++d) {
      const uint32_t h = hist[nbins - 1 - d];
      if (acc + h >= need) {
        sm[32] = nbins - 1 - d;
        sm[33] = acc;
        break;
      }
      acc += h;
    }
  }
  __syncthreads();
  bin = (int)sm[32];
  above = sm[33];
  __syncthreads();
}

// The threshold (u*, the count above it) after pass `pass` (1..3) from a row's histograms.
template <int NT>
__device__ void threshold(const uint32_t* rs, int n, int pass, uint32_t* sm, uint32_t& prefix, uint32_t& above) {
  const uint32_t need = (uint32_t)min(n, kTopK);
  int b;
  uint32_t ab;
  find_bin<NT>(rs, kH1, need, sm, b, ab);
  prefix = (uint32_t)b;
  above = ab;
  if (pass < 2) return;
  find_bin<NT>(rs + kH1, kH2, need - above, sm, b, ab);
  prefix = prefix << 11 | (uint32_t)b;
  above += ab;
  if (pass < 3) return;
  find_bin<NT>(rs + kH1 + kH2, kH3, need - above, sm, b, ab);
  prefix = prefix << 10 | (uint32_t)b;
  above += ab;
}

struct SelArgs {
  const float* scores;
  size_t ld;
  const int* pos;
  int lo, hi, rows;
  uint32_t* work;
  int* topk;
  uint32_t* cand_key;
};

// passes 1-3: histograms of bits 31..21, 20..10 (within u* so far), 9..0
template <int PASS>
__global__ void __launch_bounds__(256) k_sel_hist(SelArgs a) {
  __shared__ uint32_t h[kH1];
  __shared__ uint32_t sm[40];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y, c = blockIdx.x;
  const int n = row_n(a.pos[r], a.lo, a.hi);
  if (n <= kTopK || c * kSelChunk >= n) return;
  uint32_t* rs = a.work + (size_t)r * kRowWords;
  uint32_t prefix = 0, above;
  if (PASS > 1) threshold<256>(rs, n, PASS - 1, sm, prefix, above);
  constexpr int nb = PASS == 3 ? kH3 : kH1;
  for (int i = threadIdx.x; i < nb; i += 256) h[i] = 0;
  __syncthreads();
  const float* s = a.scores + (size_t)r * a.ld;
  const int e = min(n, (c + 1) * kSelChunk);
  for (int j = c * kSelChunk + threadIdx.x; j < e; j += 256) {
    const uint32_t u = order_key(s[j]);
    if (PASS == 1) atomicAdd(&h[u >> 21], 1u);
    else if (PASS == 2) { if ((u >> 21) == prefix) atomicAdd(&h[(u >> 10) & 0x7FF], 1u); }
    else { if ((u >> 10) == prefix) atomicAdd(&h[u & 0x3FF], 1u); }
  }
  __syncthreads();
  uint32_t* gh = rs + (PASS == 1 ? 0 : PASS == 2 ? kH1 : kH1 + kH2);
  for (int i = threadIdx.x; i < nb; i += 256)
    if (h[i]) atomicAdd(&gh[i], h[i]);
}

// pass 4: u*, the keys equal to it that are taken; each chunk's counts above / equal
__global__ void __launch_bounds__(256) k_sel_count(SelArgs a) {
  __shared__ uint32_t sm[40];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y, c = blockIdx.x;
  const int n = row_n(a.pos[r], a.lo, a.hi);
  if (n <= kTopK || c * kSelChunk >= n) return;
  uint32_t* rs = a.work + (size_t)r * kRowWords;
  uint32_t thr, above;
  threshold<256>(rs, n, 3, sm, thr, above);
  const float* s = a.scores + (size_t)r * a.ld;
  const int e = min(n, (c + 1) * kSelChunk);
  uint32_t gt = 0, eq = 0;
  for (int j = c * kSelChunk + threadIdx.x; j < e; j += 256) {
    const uint32_t u = order_key(s[j]);
    gt += u > thr;
    eq += u == thr;
  }
  uint32_t tg, te;
  block_excl_scan<256>(gt, sm, &tg);
  block_excl_scan<256>(eq, sm, &te);
  if (threadIdx.x == 0) {
    uint32_t* cc = rs + kH1 + kH2 + kH3 + 4 + 2 * c;
    cc[0] = tg;
    cc[1] = te;
    if (c == 0) {
      rs[kH1 + kH2 + kH3 + 0] = thr;
      rs[kH1 + kH2 + kH3 + 1] = kTopK - above;  // keys equal to u* taken
    }
  }
}

// pass 5: ordered compaction; rows with n <= 2048 take every key. Clears the histograms.
__global__ void __launch_bounds__(256) k_sel_write(SelArgs a) {
  __shared__ uint32_t sm[40];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
  const int p = a.pos[r];
  if (p < kTopK) return;
  const int n = row_n(p, a.lo, a.hi);
  int* out = a.topk + (size_t)r * kTopK;
  uint32_t* ck = a.cand_key ? a.cand_key + (size_t)r * kTopK : nullptr;
  uint32_t* rs = a.work + (size_t)r * kRowWords;
  const int nsel = min(n, kTopK);
  if (c == 0)
    for (int i = nsel + tid; i < kTopK; i += 256) {
      out[i] = -1;
      if (ck) ck[i] = 0;
    }
  if (c * kSelChunk >= n) return;
  const float* s = a.scores + (size_t)r * a.ld;
  uint32_t thr = 0, need_eq = 0, off = 0, quota = 0;
  bool all = n <= kTopK;
  if (!all) {
    thr = rs[kH1 + kH2 + kH3];
    need_eq = rs[kH1 + kH2 + kH3 + 1];
    uint32_t gtb = 0, eqb = 0;
    const uint32_t* cc = rs + kH1 + kH2 + kH3 + 4;
    for (int i = 0; i < c; ++i) { gtb += cc[2 * i]; eqb += cc[2 * i + 1]; }
    off = gtb + min(eqb, need_eq);
    quota = need_eq > eqb ? need_eq - eqb : 0;
  } else {
    off = c * kSelChunk;
  }
  // thread tid: keys c * 4096 + 16 tid .. +15
  const int jb = c * kSelChunk + 16 * tid;
  uint32_t u[16];
  uint32_t neq = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int j = jb + i;
    u[i] = j < n ? order_key(s[j]) : 0u;
    neq += j < n && !all && u[i] == thr;
  }
  const uint32_t eq_before = block_excl_scan<256>(neq, sm, nullptr);
  uint32_t nsel_t = 0, e = eq_before;
  bool take[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int j = jb + i;
    bool tk = false;
    if (j < n) {
      if (all || u[i] > thr) tk = true;
      else if (u[i] == thr) { tk = e < quota; ++e; }
    }
    take[i] = tk;
    nsel_t += tk;
  }
  uint32_t o = off + block_excl_scan<256>(nsel_t, sm, nullptr);
#pragma unroll
  for (int i = 0; i < 16; ++i)
    if (take[i]) {
      out[o] = a.lo + jb + i;
      if (ck) ck[o] = u[i];
      ++o;
    }
  // clear this row's histograms (no pass reads them after k_sel_count)
  if (c == 0 && !all)
    for (int i = tid; i < kH1 + kH2 + kH3; i += 256) rs[i] = 0;
}

// The merge: per row (p >= 2048) the top-2048 of the 4 ranks' candidates, in list order for ties (the parts are
// ascending position ranges in rank order). One CTA of 1024 threads, 8 candidates a thread.
__global__ void __launch_bounds__(1024) k_merge(const int* __restrict__ cpos, const uint32_t* __restrict__ ckey,
                                                const int* __restrict__ pos, int* __restrict__ topk) {
  constexpr int NC = kTP * kTopK, NT = 1024;
  __shared__ uint32_t h[kH1 + kH2 + kH3];
  __shared__ uint32_t sm[40];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, tid = threadIdx.x;
  if (pos[r] < kTopK) return;
  uint32_t u[8];
  bool v[8];
  uint32_t nv = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = 8 * tid + i;
    v[i] = cpos[(size_t)r * NC + k] >= 0;
    u[i] = ckey[(size_t)r * NC + k];
    nv += v[i];
  }
  for (int i = tid; i < kH1 + kH2 + kH3; i += NT) h[i] = 0;
  uint32_t n;
  block_excl_scan<NT>(nv, sm, &n);
  const bool all = n <= (uint32_t)kTopK;
  uint32_t thr = 0, above = 0;
  if (!all) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (v[i]) atomicAdd(&h[u[i] >> 21], 1u);
    __syncthreads();
    int b;
    uint32_t ab;
    find_bin<NT>(h, kH1, kTopK, sm, b, ab);
    thr = b;
    above = ab;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (v[i] && (u[i] >> 21) == thr) atomicAdd(&h[kH1 + ((u[i] >> 10) & 0x7FF)], 1u);
    __syncthreads();
    find_bin<NT>(h + kH1, kH2, kTopK - above, sm, b, ab);
    thr = thr << 11 | b;
    above += ab;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (v[i] && (u[i] >> 10) == thr) atomicAdd(&h[kH1 + kH2 + (u[i] & 0x3FF)], 1u);
    __syncthreads();
    find_bin<NT>(h + kH1 + kH2, kH3, kTopK - above, sm, b, ab);
    thr = thr << 10 | b;
    above += ab;
  }
  const uint32_t quota = kTopK - above;
  uint32_t neq = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) neq += v[i] && !all && u[i] == thr;
  uint32_t e = block_excl_scan<NT>(neq, sm, nullptr), nt = 0;
  bool take[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    bool tk = false;
    if (v[i]) {
      if (all || u[i] > thr) tk = true;
      else if (u[i] == thr) { tk = e < quota; ++e; }
    }
    take[i] = tk;
    nt += tk;
  }
  uint32_t o = block_excl_scan<NT>(nt, sm, nullptr);
  int* out = topk + (size_t)r * kTopK;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (take[i]) out[o++] = cpos[(size_t)r * NC + 8 * tid + i];
  const uint32_t nsel = all ? n : (uint32_t)kTopK;
  for (uint32_t i = nsel + tid; i < (uint32_t)kTopK; i += NT) out[i] = -1;
}

template <int RG, bool FP8K>
void launch_score(const ScoreArgs& a, int grid_x, int grid_y, cudaStream_t s, bool pdl) {
  const size_t smem = (size_t)RG * kIdxHeads * kQHead * 2 + RG * kIdxHeads * 4 + RG * 4;
  static bool once = [&] {
    CUDA_CHECK(cudaFuncSetAttribute(k_score<RG, FP8K>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    return true;
  }();
  (void)once;
  launch(k_score<RG, FP8K>, dim3(grid_x, grid_y), dim3(score_threads<RG, FP8K>()), smem, s, pdl, a);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

size_t index_scratch_words(int rows) { return (size_t)rows * kRowWords; }

void index_select(const IdxQuery& q, const int* pos, const int* slot, int rows, const uint32_t* rope, IdxCache ik,
                  int lo, int hi, IdxScratch sc, int* topk, uint32_t* cand_key, cudaStream_t s, bool pdl) {
  if (sc.ld % kScoreChunk || sc.ld > (size_t)kMaxSelChunks * kSelChunk) die("index_select: ld %zu", sc.ld);
  if (ik.fp8 && !ik.ks) die("index_select: FP8 keys without scales");
  ScoreArgs a{q.q, q.w, pos, slot, rope, ik, lo, hi, rows, rows <= kMaxRows, sc.scores, sc.ld};
  const int gx = (int)(sc.ld / kScoreChunk);
  // decode: grid.y = rows (a CTA per group start); groups of up to 8 rows (1 for a single row, 2, 4)
  if (a.decode) {
    const int rg = rows == 1 ? 1 : rows == 2 ? 2 : rows <= 4 ? 4 : 8;
#define LS(R) (ik.fp8 ? launch_score<R, true>(a, gx, rows, s, pdl) : launch_score<R, false>(a, gx, rows, s, pdl))
    if (rg == 1) LS(1);
    else if (rg == 2) LS(2);
    else if (rg == 4) LS(4);
    else LS(8);
  } else {
    const int gy = (rows + 7) / 8;
    ik.fp8 ? launch_score<8, true>(a, gx, gy, s, pdl) : launch_score<8, false>(a, gx, gy, s, pdl);
  }
#undef LS
  SelArgs b{sc.scores, sc.ld, pos, lo, hi, rows, sc.work, topk, cand_key};
  const dim3 grid((unsigned)(sc.ld / kSelChunk + (sc.ld % kSelChunk != 0)), rows);
  launch(k_sel_hist<1>, grid, dim3(256), 0, s, pdl, b);
  launch(k_sel_hist<2>, grid, dim3(256), 0, s, pdl, b);
  launch(k_sel_hist<3>, grid, dim3(256), 0, s, pdl, b);
  launch(k_sel_count, grid, dim3(256), 0, s, pdl, b);
  launch(k_sel_write, grid, dim3(256), 0, s, pdl, b);
  CUDA_CHECK(cudaGetLastError());
}

void index_merge(const int* cand_pos, const uint32_t* cand_key, const int* pos, int rows, int* topk, cudaStream_t s,
                 bool pdl) {
  launch(k_merge, dim3(rows), dim3(1024), 0, s, pdl, cand_pos, cand_key, pos, topk);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace glm
