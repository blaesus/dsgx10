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
// (xor 1, then xor 2). The key tiles' loads have variants (LDV, below) that leave the registers unchanged.
// Prefill runs k_score_pf (below): the same MMA chains and head sums with 256 keys held in registers and the rows'
// staged queries streamed, so its scores equal k_score's bit for bit; it also counts a 1-in-16 sample of each
// row's first-level bins (bits 31..19 of the order key).
//
// Selection at decode: five parallel passes (three histogram levels, a count, an ordered write; below, "decode
// selection"). At prefill, two launches: k_sel_thresh turns the sample into a first-level bin b (aiming at ~4096
// candidates); k_sel_compact has each 4096-key chunk compact its keys at bin >= b in position order at an offset
// from the row's counter, and the row's last CTA gathers the candidates (position order) and takes the exact
// top-2048 with three histogram levels (11, 11 and 10 bits) and an ordered pass for the keys equal to the
// threshold (the lowest positions). A row whose candidates number under 2048 (a sample miss) or over the cap
// selects from its scores directly with the same method. Every path gives the exact top-2048 by (score desc,
// position asc), and the selection leaves the histograms and counters zeroed.
#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"

namespace glm {
namespace {

constexpr int kScoreChunk = 1024;  // keys a decode scoring CTA
constexpr int kPfKeys = 256;       // keys a prefill scoring CTA (8 warps x 2 tiles of 16, held in registers)
constexpr int kPfRows = 256;       // rows a prefill scoring CTA
constexpr int kPfQS = 2;           // staged query rows a shared-memory buffer (two buffers)
constexpr int kSelChunk = 4096;    // keys a compaction CTA
constexpr int kQBlk = 36;          // a 32-dim block of a staged query, padded (bank-conflict free uint2 reads)
constexpr int kQHead = 4 * kQBlk;  // 144 bf16 a head
constexpr int kMaxSelChunks = 256; // 1M keys
constexpr int kH1 = 2048, kH2 = 2048, kH3 = 1024;  // the merge's histogram levels
constexpr int kHB = 8192;          // the selection's first level: bits 31..19 of the order key
constexpr int kCand = 8192;        // candidates a row: the keys in or above the threshold's first-level bin
constexpr int kQRowBytes = kIdxHeads * kQHead * 2 + kIdxHeads * 4;  // a staged query row: q (RoPE'd), weights
// row scratch (words): prefill's sampled first-level histogram [8192] (decode: its three histogram levels in words
// 0 .. 5119), both kept zero between calls | params [8] | the compaction chunks' (offset, count) [256][2] |
// candidates (position [8192], order key [8192]; decode: its params and chunk counts) | staged query row
constexpr int kOffP = kHB, kOffSeg = kOffP + 8, kOffCand = kOffSeg + 2 * kMaxSelChunks, kOffQ = kOffCand + 2 * kCand,
              kRowWords = kOffQ + kQRowBytes / 4;
// params: threshold bin, (unused x3), (unused), compaction CTAs done, candidates written; the counters are left zero
enum { kPBin = 0, kPCDone = 5, kPCCnt = 6 };
static_assert(kQRowBytes % 16 == 0 && (kOffQ * 4) % 16 == 0 && (kRowWords * 4) % 16 == 0, "staged rows 16-byte aligned");

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
  uint32_t* work;  // the rows' scratch: prefill's sampled first-level histogram (1 key in 16)
};

// NT threads: 128 x RG (cap 512), so a CTA whose staged queries fill most of shared memory (RG 8: 74 KB, one CTA
// an SM) still has 16 warps an SM to hide MMA latency
template <int RG, bool FP8K>
constexpr int score_threads() { return RG >= 4 ? 512 : 128 * RG; }

// LDV: how a warp loads its key tiles (the registers end up the same: the scores do not depend on it). 0: each
// lane its own 16-byte pieces (an instruction uses half of each 32-byte sector), L1 bypassed; 1: the same through
// L1; 2: lanes t and t ^ 1 load whole sectors (one piece each of the lane's and its partner's) and swap halves
// with a shuffle when the tile is used (fix_key); 3: as 2 with two tiles in flight a warp instead of one; 4: as 0
// with two tiles in flight.
__device__ __forceinline__ uint4 ldg_l1(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 shfl_x1(uint4 v) {
  v.x = __shfl_xor_sync(0xffffffffu, v.x, 1);
  v.y = __shfl_xor_sync(0xffffffffu, v.y, 1);
  v.z = __shfl_xor_sync(0xffffffffu, v.z, 1);
  v.w = __shfl_xor_sync(0xffffffffu, v.w, 1);
  return v;
}
// a lane's NW 16-byte pieces of one key (bytes off(t) + 16 i, off(t) = 16 NW t) from the key's row kr
template <int NW, int LDV>
__device__ __forceinline__ void load_key(uint4 (&k)[NW], const uint8_t* kr, bool v, int t) {
  if constexpr (LDV == 2 || LDV == 3) {
    const int p = t & 1;
#pragma unroll
    for (int s2 = 0; s2 < NW / 2; ++s2) {
      const uint8_t* own = kr + 16 * NW * t + 32 * s2;
      const uint8_t* par = kr + 16 * NW * (t ^ 1) + 32 * s2 + 16;
      const uint4 z = make_uint4(0, 0, 0, 0);
      k[2 * s2] = v ? ldg_stream128(p ? par : own) : z;
      k[2 * s2 + 1] = v ? ldg_stream128(p ? own : par) : z;
    }
  } else {
#pragma unroll
    for (int i = 0; i < NW; ++i)
      k[i] = v ? (LDV == 1 ? ldg_l1(kr + 16 * NW * t + 16 * i) : ldg_stream128(kr + 16 * NW * t + 16 * i))
               : make_uint4(0, 0, 0, 0);
  }
}

// LDV 2, 3: (own piece, partner's piece) as loaded -> the lane's two pieces of each sector
template <int NW, int LDV>
__device__ __forceinline__ void fix_key(uint4 (&k)[NW], int t) {
  if constexpr (LDV == 2 || LDV == 3) {
    const int p = t & 1;
#pragma unroll
    for (int s2 = 0; s2 < NW / 2; ++s2) {
      const uint4 r0 = k[2 * s2], r1 = k[2 * s2 + 1];
      k[2 * s2] = p ? r1 : r0;
      k[2 * s2 + 1] = shfl_x1(p ? r0 : r1);
    }
  }
}

template <int RG, bool FP8K, int LDV>
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
    constexpr size_t KB = FP8K ? kIdxDim : 2 * kIdxDim;  // bytes a key
    const uint8_t* base = a.ik.k + sbase * KB;
    load_key<NW, LDV>(ka, base + (size_t)ja * KB, va, t);
    load_key<NW, LDV>(kb, base + (size_t)jb * KB, vb, t);
    if constexpr (FP8K) {
      sa = va ? __ldg(a.ik.ks + sbase + ja) : 0.f;
      sb = vb ? __ldg(a.ik.ks + sbase + jb) : 0.f;
    } else {
      sa = sb = 1.f;
    }
  };
  constexpr int D = LDV >= 3 ? 2 : 1;  // tiles in flight a warp
  uint4 nq[D][2][NW];
  float nsc[D][2];
  int jt = j0 + warp * 16;
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (jt + d * 16 * NWARP < j1) load(nq[d][0], nq[d][1], nsc[d][0], nsc[d][1], jt + d * 16 * NWARP);
  for (; jt < j1; jt += 16 * NWARP) {
    uint32_t A[16], B[16];  // key g, key g+8: dims 32t + 2i, +1 in word i
    fix_key<NW, LDV>(nq[0][0], t);
    fix_key<NW, LDV>(nq[0][1], t);
    const uint4(&na)[NW] = nq[0][0];
    const uint4(&nb)[NW] = nq[0][1];
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
    const float sa = nsc[0][0], sb = nsc[0][1];
    const int jc = jt;
#pragma unroll
    for (int d = 0; d + 1 < D; ++d) {
#pragma unroll
      for (int i = 0; i < NW; ++i) { nq[d][0][i] = nq[d + 1][0][i]; nq[d][1][i] = nq[d + 1][1][i]; }
      nsc[d][0] = nsc[d + 1][0];
      nsc[d][1] = nsc[d + 1][1];
    }
    if (jt + D * 16 * NWARP < j1) load(nq[D - 1][0], nq[D - 1][1], nsc[D - 1][0], nsc[D - 1][1], jt + D * 16 * NWARP);
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
      const bool ok = (t == 0 && jc + g < prow[rr]) || (t == 1 && jc + g + 8 < prow[rr]);
      if (ok) srow[jc + g + 8 * t] = t ? vb : va;
    }
  }
}

// ---- prefill scores ----------------------------------------------------------------------------------------
// k_q_stage writes each row's RoPE'd query in k_score's staged layout, and its head weights, into the row's
// scratch. k_score_pf: a CTA holds 256 keys in registers as MMA A fragments (8 warps x 2 tiles of 16 keys, the
// lane map of k_score) and streams up to 256 rows' staged queries through shared memory (two buffers of two rows,
// cp.async); for each row a warp runs k_score's chain of MMAs (8 k-steps a head tile, from zero) on each of its
// tiles and k_score's head sum: the scores are k_score's, bit for bit.
__global__ void __launch_bounds__(256) k_q_stage(const uint16_t* __restrict__ q, const float* __restrict__ w,
                                                 const int* __restrict__ pos, const uint32_t* __restrict__ rope,
                                                 uint32_t* __restrict__ work) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, tid = threadIdx.x;
  uint16_t* d0 = reinterpret_cast<uint16_t*>(work + (size_t)r * kRowWords + kOffQ);
  for (int i = tid; i < kIdxHeads * 32; i += 256) {
    const int h = i / 32, pi = i % 32;
    const uint16_t* src = q + ((size_t)r * kIdxHeads + h) * kIdxDim;
    const uint32_t v = *reinterpret_cast<const uint32_t*>(src + 2 * pi);
    float lo, hi;
    rope_pair(bf2f(v), bf2f_hi(v), rope[(size_t)pos[r] * 32 + pi], lo, hi);
    uint16_t* dst = d0 + h * kQHead;
    dst[pi] = f2bf(lo);
    dst[kQBlk + pi] = f2bf(hi);
    *reinterpret_cast<uint32_t*>(dst + (2 + pi / 16) * kQBlk + (2 * pi) % 32) =
        *reinterpret_cast<const uint32_t*>(src + 64 + 2 * pi);
  }
  if (tid < kIdxHeads) reinterpret_cast<float*>(d0 + kIdxHeads * kQHead)[tid] = w[(size_t)r * kIdxHeads + tid];
}

template <bool FP8K>
__global__ void __launch_bounds__(256) k_score_pf(ScoreArgs a, const uint32_t* __restrict__ work) {
  __shared__ __align__(16) uint8_t qs[2][kPfQS][kQRowBytes];
  pdl_trigger();
  pdl_wait();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int ra = blockIdx.y * kPfRows, rb = min(a.rows, ra + kPfRows);
  const int kb0 = a.lo + blockIdx.x * kPfKeys;
  auto row_end = [&](int r) { return a.lo + row_n(a.pos[r], a.lo, a.hi); };  // nondecreasing in r
  const int endmax = row_end(rb - 1);
  if (kb0 >= endmax) return;
  int r0 = ra, r1 = rb - 1;  // the first row whose scan reaches kb0
  while (r0 < r1) {
    const int m = (r0 + r1) >> 1;
    if (row_end(m) > kb0) r1 = m;
    else r0 = m + 1;
  }
  const size_t sbase = a.slot ? (size_t)a.slot[ra] * a.ik.slot_stride : 0;
  // the warp's two key tiles: keys jt + g and jt + g + 8, dims 32t .. 32t + 31 (k_score's map)
  uint32_t A[2][16], B[2][16];
  float sa[2], sb[2];
#pragma unroll
  for (int kt = 0; kt < 2; ++kt) {
    const int ja = kb0 + warp * 32 + 16 * kt + g, jb = ja + 8;
    const bool va = ja < endmax, vb = jb < endmax;
    if constexpr (FP8K) {
      const uint8_t* base = a.ik.k + sbase * kIdxDim;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 ka = va ? ldg_stream128(base + (size_t)ja * kIdxDim + 32 * t + 16 * i) : make_uint4(0, 0, 0, 0);
        const uint4 kb = vb ? ldg_stream128(base + (size_t)jb * kIdxDim + 32 * t + 16 * i) : make_uint4(0, 0, 0, 0);
        fp8x4_to_bf16(ka.x, A[kt][8 * i + 0], A[kt][8 * i + 1]); fp8x4_to_bf16(ka.y, A[kt][8 * i + 2], A[kt][8 * i + 3]);
        fp8x4_to_bf16(ka.z, A[kt][8 * i + 4], A[kt][8 * i + 5]); fp8x4_to_bf16(ka.w, A[kt][8 * i + 6], A[kt][8 * i + 7]);
        fp8x4_to_bf16(kb.x, B[kt][8 * i + 0], B[kt][8 * i + 1]); fp8x4_to_bf16(kb.y, B[kt][8 * i + 2], B[kt][8 * i + 3]);
        fp8x4_to_bf16(kb.z, B[kt][8 * i + 4], B[kt][8 * i + 5]); fp8x4_to_bf16(kb.w, B[kt][8 * i + 6], B[kt][8 * i + 7]);
      }
      sa[kt] = va ? __ldg(a.ik.ks + sbase + ja) : 0.f;
      sb[kt] = vb ? __ldg(a.ik.ks + sbase + jb) : 0.f;
    } else {
      const uint16_t* base = reinterpret_cast<const uint16_t*>(a.ik.k) + sbase * kIdxDim;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 ka = va ? ldg_stream128(base + (size_t)ja * kIdxDim + 32 * t + 8 * i) : make_uint4(0, 0, 0, 0);
        const uint4 kb = vb ? ldg_stream128(base + (size_t)jb * kIdxDim + 32 * t + 8 * i) : make_uint4(0, 0, 0, 0);
        A[kt][4 * i] = ka.x; A[kt][4 * i + 1] = ka.y; A[kt][4 * i + 2] = ka.z; A[kt][4 * i + 3] = ka.w;
        B[kt][4 * i] = kb.x; B[kt][4 * i + 1] = kb.y; B[kt][4 * i + 2] = kb.z; B[kt][4 * i + 3] = kb.w;
      }
      sa[kt] = sb[kt] = 1.f;
    }
  }
  const int nbat = (rb - r0 + kPfQS - 1) / kPfQS;
  auto stage = [&](int bt) {
    const int q0 = r0 + bt * kPfQS, nr = min(kPfQS, rb - q0);
    for (int i = tid; i < nr * (kQRowBytes / 16); i += 256) {
      const int rr = i / (kQRowBytes / 16), pc = i % (kQRowBytes / 16);
      cp_async16(&qs[bt & 1][rr][pc * 16],
                 reinterpret_cast<const uint8_t*>(work + (size_t)(q0 + rr) * kRowWords + kOffQ) + pc * 16);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  stage(0);
  for (int bt = 0; bt < nbat; ++bt) {
    if (bt + 1 < nbat) {
      stage(bt + 1);
      asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
      cp_async_wait_all();
    }
    __syncthreads();
    const int q0 = r0 + bt * kPfQS, nr = min(kPfQS, rb - q0);
    for (int rr = 0; rr < nr; ++rr) {
      const int r = q0 + rr, end = row_end(r);
      const uint16_t* qrow = reinterpret_cast<const uint16_t*>(qs[bt & 1][rr]);
      const float* wr = reinterpret_cast<const float*>(qrow + kIdxHeads * kQHead);
      float c[2][4][4];
#pragma unroll
      for (int kt = 0; kt < 2; ++kt)
#pragma unroll
        for (int n = 0; n < 4; ++n) c[kt][n][0] = c[kt][n][1] = c[kt][n][2] = c[kt][n][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const uint2 b = *reinterpret_cast<const uint2*>(qrow + (8 * n + g) * kQHead + t * kQBlk + 4 * kk);
#pragma unroll
          for (int kt = 0; kt < 2; ++kt)
            mma_bf16(c[kt][n], A[kt][2 * kk], B[kt][2 * kk], A[kt][2 * kk + 1], B[kt][2 * kk + 1], b.x, b.y);
        }
      }
      float* srow = a.scores + (size_t)r * a.ld - a.lo;
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        float va = 0.f, vb = 0.f;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const float w0 = wr[8 * n + 2 * t], w1 = wr[8 * n + 2 * t + 1];
          va = __fadd_rn(va, __fmul_rn(w0, fmaxf(__fmul_rn(__fmul_rn(c[kt][n][0], sa[kt]), kIdxScale), 0.f)));
          va = __fadd_rn(va, __fmul_rn(w1, fmaxf(__fmul_rn(__fmul_rn(c[kt][n][1], sa[kt]), kIdxScale), 0.f)));
          vb = __fadd_rn(vb, __fmul_rn(w0, fmaxf(__fmul_rn(__fmul_rn(c[kt][n][2], sb[kt]), kIdxScale), 0.f)));
          vb = __fadd_rn(vb, __fmul_rn(w1, fmaxf(__fmul_rn(__fmul_rn(c[kt][n][3], sb[kt]), kIdxScale), 0.f)));
        }
        va = __fadd_rn(va, __shfl_xor_sync(0xffffffffu, va, 1));
        vb = __fadd_rn(vb, __shfl_xor_sync(0xffffffffu, vb, 1));
        va = __fadd_rn(va, __shfl_xor_sync(0xffffffffu, va, 2));
        vb = __fadd_rn(vb, __shfl_xor_sync(0xffffffffu, vb, 2));
        const int jc = kb0 + warp * 32 + 16 * kt;
        if (t == 0 && jc + g < end) srow[jc + g] = va;
        if (t == 1 && jc + g + 8 < end) srow[jc + g + 8] = vb;
        // the 1-in-16 sample of the first-level histogram: positions lo + 16i
        if (lane == 0 && jc < end && end - a.lo > kTopK) atomicAdd(a.work + (size_t)r * kRowWords + (order_key(va) >> 19), 1u);
      }
    }
    __syncthreads();  // buffer bt & 1 is restaged at bt + 2
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

struct SelArgs {
  const float* scores;
  size_t ld;
  const int* pos;
  int lo, hi, rows, cap;
  uint32_t* work;
  int* topk;
  uint32_t* cand_key;
};

constexpr int kSampleNeed = 2 * kTopK / 16;  // prefill: the threshold bin is where the sample's count reaches this

// Prefill: one CTA a row with more than 2048 keys turns the 1-in-16 sample's histogram into the first-level bin
// b (the keys at bin >= b are expected to number ~4096: 2x the 2048 taken) and clears the histogram.
__global__ void __launch_bounds__(256) k_sel_thresh(SelArgs a) {
  __shared__ uint32_t h[kHB];
  __shared__ uint32_t sm[40];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, tid = threadIdx.x;
  if (row_n(a.pos[r], a.lo, a.hi) <= kTopK) return;
  uint32_t* rs = a.work + (size_t)r * kRowWords;
  uint32_t tot = 0;
  for (int i = tid; i < kHB; i += 256) {
    h[i] = rs[i];
    tot += h[i];
    rs[i] = 0;
  }
  block_excl_scan<256>(tot, sm, &tot);
  __syncthreads();
  int b = 0;
  uint32_t ab;
  if (tot >= (uint32_t)kSampleNeed) find_bin<256>(h, kHB, kSampleNeed, sm, b, ab);
  if (tid == 0) rs[kOffP + kPBin] = (uint32_t)b;
}

constexpr int kCT = 1024;  // compaction threads: 4 keys each of a 4096-key chunk; the final holds 8 candidates each

// The exact top-2048 of a list of more than 2048 (order key, position) entries in position order, held 8 a thread
// (entry 8 tid + e): three histogram levels (bits 31..21, 20..10, 9..0) find the 2048th key u* and how many keys
// above it; an ordered pass takes every key above u* and the lowest positions equal to it (k_merge's method).
__device__ void select_regs(const uint32_t (&u)[8], const int (&p)[8], int L, int* out, uint32_t* ck, uint32_t* h,
                            uint32_t* sm) {
  const int tid = threadIdx.x;
  bool v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = 8 * tid + i < L;
  for (int i = tid; i < kH1 + kH2 + kH3; i += kCT) h[i] = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (v[i]) atomicAdd(&h[u[i] >> 21], 1u);
  __syncthreads();
  int b;
  uint32_t ab;
  find_bin<kCT>(h, kH1, kTopK, sm, b, ab);
  uint32_t thr = b, above = ab;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (v[i] && (u[i] >> 21) == thr) atomicAdd(&h[kH1 + ((u[i] >> 10) & 0x7FF)], 1u);
  __syncthreads();
  find_bin<kCT>(h + kH1, kH2, kTopK - above, sm, b, ab);
  thr = thr << 11 | b;
  above += ab;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (v[i] && (u[i] >> 10) == thr) atomicAdd(&h[kH1 + kH2 + (u[i] & 0x3FF)], 1u);
  __syncthreads();
  find_bin<kCT>(h + kH1 + kH2, kH3, kTopK - above, sm, b, ab);
  thr = thr << 10 | b;
  above += ab;
  const uint32_t quota = kTopK - above;
  uint32_t neq = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) neq += v[i] && u[i] == thr;
  uint32_t e = block_excl_scan<kCT>(neq, sm, nullptr), nt = 0;
  bool take[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    bool tk = false;
    if (v[i]) {
      if (u[i] > thr) tk = true;
      else if (u[i] == thr) { tk = e < quota; ++e; }
    }
    take[i] = tk;
    nt += tk;
  }
  uint32_t o = block_excl_scan<kCT>(nt, sm, nullptr);
#pragma unroll
  for (int i = 0; i < 8; ++i)
    if (take[i]) {
      out[o] = p[i];
      if (ck) ck[o] = u[i];
      ++o;
    }
}

// The same from a row's scores directly (n > 2048 keys; the candidates did not fit or were too few): each level
// and the ordered pass stream the row (tiles of 8 keys a thread).
__device__ void select_scores(const float* s, int n, int lo, int* out, uint32_t* ck, uint32_t* h, uint32_t* sm) {
  const int tid = threadIdx.x;
  uint32_t prefix = 0, above = 0;
  for (int lev = 0; lev < 3; ++lev) {
    const int nb = lev == 2 ? kH3 : kH1;
    for (int i = tid; i < nb; i += kCT) h[i] = 0;
    __syncthreads();
    for (int j = tid; j < n; j += kCT) {
      const uint32_t u = order_key(s[j]);
      if (lev == 0) atomicAdd(&h[u >> 21], 1u);
      else if (lev == 1) { if ((u >> 21) == prefix) atomicAdd(&h[(u >> 10) & 0x7FF], 1u); }
      else if ((u >> 10) == prefix) atomicAdd(&h[u & 0x3FF], 1u);
    }
    __syncthreads();
    int b;
    uint32_t ab;
    find_bin<kCT>(h, nb, kTopK - above, sm, b, ab);
    prefix = (prefix << (lev == 2 ? 10 : 11)) | (uint32_t)b;
    above += ab;
  }
  const uint32_t thr = prefix, quota = kTopK - above;
  uint32_t off = 0, eqs = 0;
  for (int t0 = 0; t0 < n; t0 += kCT * 8) {
    const int ib = t0 + 8 * tid;
    uint32_t u[8], neq = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      u[e] = ib + e < n ? order_key(s[ib + e]) : 0u;
      neq += ib + e < n && u[e] == thr;
    }
    uint32_t eqt, tt;
    uint32_t e0 = eqs + block_excl_scan<kCT>(neq, sm, &eqt), ntk = 0;
    bool take[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      bool tk = false;
      if (ib + e < n) {
        if (u[e] > thr) tk = true;
        else if (u[e] == thr) { tk = e0 < quota; ++e0; }
      }
      take[e] = tk;
      ntk += tk;
    }
    uint32_t o = off + block_excl_scan<kCT>(ntk, sm, &tt);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (take[e]) {
        out[o] = lo + ib + e;
        if (ck) ck[o] = u[e];
        ++o;
      }
    off += tt;
    eqs += eqt;
  }
}

// Each 4096-key chunk of a row (more than 2048 keys) compacts its candidates, the keys whose first-level bin is
// at or above b, in position order at an offset taken from the row's counter (writing those below the cap), and
// records (offset, count). b: decode, the exact bin of the 2048th key from the full histogram (every CTA finds
// it); prefill, k_sel_thresh's from the sample. The row's last CTA selects from the candidates (in position order,
// 8 a thread) when there are 2048 to cap of them, else from the scores; it resets the counters (decode: and the
// histogram). Rows with at most 2048 keys take every key here; rows with p < 2048 are left alone.
__global__ void __launch_bounds__(kCT) k_sel_compact(SelArgs a) {
  __shared__ uint32_t h[kH1 + kH2 + kH3];  // the final's levels
  __shared__ uint32_t sm[40];
  __shared__ uint32_t pre[kMaxSelChunks + 1], cbase[kMaxSelChunks];
  __shared__ uint32_t sbase;
  __shared__ int last;
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
  const int p = a.pos[r];
  if (p < kTopK) return;
  const int n = row_n(p, a.lo, a.hi);
  int* out = a.topk + (size_t)r * kTopK;
  uint32_t* ck = a.cand_key ? a.cand_key + (size_t)r * kTopK : nullptr;
  const float* s = a.scores + (size_t)r * a.ld;
  if (n <= kTopK) {  // every key, in order; -1 after them
    if (c == 0)
      for (int i = n + tid; i < kTopK; i += kCT) {
        out[i] = -1;
        if (ck) ck[i] = 0;
      }
    const int e = min(n, (c + 1) * kSelChunk);
    for (int j = c * kSelChunk + tid; j < e; j += kCT) {
      out[j] = a.lo + j;
      if (ck) ck[j] = order_key(s[j]);
    }
    return;
  }
  const int nch = (n + kSelChunk - 1) / kSelChunk;
  if (c >= nch) return;
  uint32_t* rs = a.work + (size_t)r * kRowWords;
  const uint32_t b1 = rs[kOffP + kPBin];
  const int j0 = c * kSelChunk + 4 * tid;
  const float4 v = j0 < n ? *reinterpret_cast<const float4*>(s + j0) : make_float4(0.f, 0.f, 0.f, 0.f);
  const uint32_t u[4] = {order_key(v.x), order_key(v.y), order_key(v.z), order_key(v.w)};
  bool cand[4];
  uint32_t nt = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    cand[i] = j0 + i < n && (u[i] >> 19) >= b1;
    nt += cand[i];
  }
  uint32_t tot;
  uint32_t o = block_excl_scan<kCT>(nt, sm, &tot);
  if (tid == 0) {
    sbase = atomicAdd(&rs[kOffP + kPCCnt], tot);
    rs[kOffSeg + 2 * c] = sbase;
    rs[kOffSeg + 2 * c + 1] = tot;
  }
  __syncthreads();
  o += sbase;
  int* cp = reinterpret_cast<int*>(rs + kOffCand);
  uint32_t* cu = rs + kOffCand + kCand;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    if (cand[i]) {
      if (o < (uint32_t)a.cap) {
        cp[o] = a.lo + j0 + i;
        cu[o] = u[i];
      }
      ++o;
    }
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(&rs[kOffP + kPCDone], 1u) == (uint32_t)(nch - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const uint32_t C = __ldcg(rs + kOffP + kPCCnt);
  if (C >= (uint32_t)kTopK && C <= (uint32_t)a.cap) {
    // the candidates in position order: chunk cc's at its offset, after the earlier chunks'
    const uint32_t cnt = tid < nch ? __ldcg(rs + kOffSeg + 2 * tid + 1) : 0;
    const uint32_t ex = block_excl_scan<kCT>(cnt, sm, nullptr);
    if (tid < nch) {
      pre[tid] = ex;
      cbase[tid] = __ldcg(rs + kOffSeg + 2 * tid);
    }
    if (tid == 0) pre[nch] = C;
    __syncthreads();
    uint32_t cu8[8];
    int cp8[8];
    int cc = 0;
    {
      int lo2 = 0, hi2 = nch - 1;  // the chunk of entry 8 tid: the last cc with pre[cc] <= 8 tid
      const uint32_t i0 = 8 * tid;
      while (lo2 < hi2) {
        const int m = (lo2 + hi2 + 1) >> 1;
        if (pre[m] <= i0) lo2 = m;
        else hi2 = m - 1;
      }
      cc = lo2;
    }
    uint32_t k8[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {  // the entries' slots first, then all 16 loads at once
      const uint32_t i = 8 * tid + e;
      if (i < C)
        while (pre[cc + 1] <= i) ++cc;
      k8[e] = i < C ? cbase[cc] + (i - pre[cc]) : 0u;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const bool v = 8 * tid + e < C;
      cu8[e] = v ? __ldcg(cu + k8[e]) : 0u;
      cp8[e] = v ? __ldcg(cp + k8[e]) : -1;
    }
    select_regs(cu8, cp8, (int)C, out, ck, h, sm);
  } else {
    select_scores(s, n, a.lo, out, ck, h, sm);
  }
  if (tid == 0) {
    rs[kOffP + kPCDone] = 0;
    rs[kOffP + kPCCnt] = 0;
  }
}

// ---- decode selection: five parallel passes ------------------------------------------------------------------
// Three histogram passes over the order keys (bits 31..21, 20..10 within the prefix so far, 9..0; integer counts,
// so the atomics' order does not matter; rows' words 0 .. 5119 of the scratch, kept zero) find the threshold u* and
// how many keys equal to it are taken (the lowest positions); a count pass gives each 4096-key chunk its numbers of
// keys above and equal to u*; a write pass compacts each chunk in position order at its prefix offset and clears
// the histograms. Every pass recomputes the thresholds from the histograms. (A single compaction launch with the
// first level counted while scoring was slower at decode: its last CTA's selection is a serial tail; see
// docs/attention.md.)
constexpr int kDecPrm = kOffCand;  // decode params: u*, keys equal to it taken; then the chunks' (above, equal)

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
    uint32_t* cc = rs + kDecPrm + 4 + 2 * c;
    cc[0] = tg;
    cc[1] = te;
    if (c == 0) {
      rs[kDecPrm + 0] = thr;
      rs[kDecPrm + 1] = kTopK - above;  // keys equal to u* taken
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
    thr = rs[kDecPrm];
    need_eq = rs[kDecPrm + 1];
    uint32_t gtb = 0, eqb = 0;
    const uint32_t* cc = rs + kDecPrm + 4;
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

template <int RG, bool FP8K, int LDV>
void launch_score_v(const ScoreArgs& a, int grid_x, int grid_y, cudaStream_t s, bool pdl) {
  const size_t smem = (size_t)RG * kIdxHeads * kQHead * 2 + RG * kIdxHeads * 4 + RG * 4;
  static bool once = [&] {
    CUDA_CHECK(cudaFuncSetAttribute(k_score<RG, FP8K, LDV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    return true;
  }();
  (void)once;
  launch(k_score<RG, FP8K, LDV>, dim3(grid_x, grid_y), dim3(score_threads<RG, FP8K>()), smem, s, pdl, a);
  CUDA_CHECK(cudaGetLastError());
}

int g_score_ld = -1;  // index_score_config (-1: by key format, measured on GB10: 2 for BF16 keys, 0 for FP8)

template <int RG, bool FP8K>
void launch_score(const ScoreArgs& a, int grid_x, int grid_y, cudaStream_t s, bool pdl) {
  const int ldv = g_score_ld >= 0 ? g_score_ld : FP8K ? 0 : 2;
  if (ldv == 1) launch_score_v<RG, FP8K, 1>(a, grid_x, grid_y, s, pdl);
  else if (ldv == 2) launch_score_v<RG, FP8K, 2>(a, grid_x, grid_y, s, pdl);
  else if (ldv == 3) launch_score_v<RG, FP8K, 3>(a, grid_x, grid_y, s, pdl);
  else if (ldv == 4) launch_score_v<RG, FP8K, 4>(a, grid_x, grid_y, s, pdl);
  else launch_score_v<RG, FP8K, 0>(a, grid_x, grid_y, s, pdl);
}

}  // namespace

static int g_cand_cap = kCand;  // index_candidate_cap

size_t index_scratch_words(int rows) { return (size_t)rows * kRowWords; }
void index_score_config(int cfg) { g_score_ld = cfg; }
void index_candidate_cap(int cap) { g_cand_cap = cap < 0 ? 0 : cap > kCand ? kCand : cap; }

void index_select(const IdxQuery& q, const int* pos, const int* slot, int rows, const uint32_t* rope, IdxCache ik,
                  int lo, int hi, IdxScratch sc, int* topk, uint32_t* cand_key, cudaStream_t s, bool pdl) {
  if (sc.ld % kScoreChunk || sc.ld > (size_t)kMaxSelChunks * kSelChunk) die("index_select: ld %zu", sc.ld);
  if (ik.fp8 && !ik.ks) die("index_select: FP8 keys without scales");
  ScoreArgs a{q.q, q.w, pos, slot, rope, ik, lo, hi, rows, rows <= kMaxRows, sc.scores, sc.ld, sc.work};
  const int gx = (int)(sc.ld / kScoreChunk);
  // decode: grid.y = rows (a CTA per group start); groups of up to 8 rows (1 for a single row, 2, 4)
  if (a.decode) {
    const int rg = rows == 1 ? 1 : rows == 2 ? 2 : rows <= 4 ? 4 : 8;
#define LS(R) (ik.fp8 ? launch_score<R, true>(a, gx, rows, s, pdl) : launch_score<R, false>(a, gx, rows, s, pdl))
    if (rg == 1) LS(1);
    else if (rg == 2) LS(2);
    else if (rg == 4) LS(4);
    else LS(8);
  } else {  // prefill: staged queries, then 256 keys x 256 rows a CTA
    launch(k_q_stage, dim3(rows), dim3(256), 0, s, pdl, q.q, q.w, pos, rope, sc.work);
    const dim3 grid((unsigned)(sc.ld / kPfKeys), (unsigned)((rows + kPfRows - 1) / kPfRows));
    if (ik.fp8) launch(k_score_pf<true>, grid, dim3(256), 0, s, pdl, a, (const uint32_t*)sc.work);
    else launch(k_score_pf<false>, grid, dim3(256), 0, s, pdl, a, (const uint32_t*)sc.work);
  }
#undef LS
  SelArgs b{sc.scores, sc.ld, pos, lo, hi, rows, g_cand_cap, sc.work, topk, cand_key};
  const dim3 grid((unsigned)((sc.ld + kSelChunk - 1) / kSelChunk), rows);
  if (a.decode) {
    launch(k_sel_hist<1>, grid, dim3(256), 0, s, pdl, b);
    launch(k_sel_hist<2>, grid, dim3(256), 0, s, pdl, b);
    launch(k_sel_hist<3>, grid, dim3(256), 0, s, pdl, b);
    launch(k_sel_count, grid, dim3(256), 0, s, pdl, b);
    launch(k_sel_write, grid, dim3(256), 0, s, pdl, b);
  } else {
    launch(k_sel_thresh, dim3(rows), dim3(256), 0, s, pdl, b);
    launch(k_sel_compact, grid, dim3(kCT), 0, s, pdl, b);
  }
  CUDA_CHECK(cudaGetLastError());
}

void index_merge(const int* cand_pos, const uint32_t* cand_key, const int* pos, int rows, int* topk, cudaStream_t s,
                 bool pdl) {
  launch(k_merge, dim3(rows), dim3(1024), 0, s, pdl, cand_pos, cand_key, pos, topk);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace glm
