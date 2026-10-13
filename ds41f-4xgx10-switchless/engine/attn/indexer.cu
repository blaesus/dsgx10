// The indexer: query preparation, scores with the block-scaled FP4 MMA, and torch.topk-exact selection.
#include <algorithm>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"
#include "common/check.h"

namespace attn {
namespace {

// D += A (16x64 e2m1, row-major) * B (64x8 e2m1) with one ue8m0 scale per 32 of K. Operand layout
// verified bit-exact in tests/mma_fp4_probe.cu (lane 4g+t: A rows g, g+8; B column g; C (g|g+8, 2t|2t+1)).
__device__ __forceinline__ void mma_fp4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// bf16 bits -> an unsigned key with the same order (0 stays free for "no candidate": it sits below -inf)
__device__ __forceinline__ uint32_t key_from_bf16(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }

struct Range {
  int lo, hi;
};
// The part of [0, n) a rank scores, n = the compressed length the last (newest) row of the session sees.
__device__ __forceinline__ Range dense_range(const int* pos, int last, int ratio, int part, int nparts) {
  int n = (pos[last] + 1) / ratio;
  int chunk = ((n + nparts - 1) / nparts + 63) & ~63;
  int lo = min(n, part * chunk);
  return {lo, min(n, lo + chunk)};
}

// ---- queries --------------------------------------------------------------------------------------

// One (row, head) query's inputs as one lane holds them: src its 128 bf16, p the row's position. Lane l holds
// elements 4l..4l+3 (so a block of 32 is 8 lanes) and, from lane 16 on (dims 64..127), the rotary table's
// entries for pairs 2l-32 and 2l-31.
struct QIn {
  uint2 raw;
  float2 cs[2];
};
__device__ __forceinline__ QIn q_load(const uint16_t* src, int p, const float2* table, int lane) {
  QIn in;
  in.raw = *reinterpret_cast<const uint2*>(src + 4 * lane);
  in.cs[0] = in.cs[1] = make_float2(0.f, 0.f);
  if (lane >= 16) {
    const float2* tb = table + (size_t)p * 32 + 2 * lane - 32;
    in.cs[0] = tb[0];
    in.cs[1] = tb[1];
  }
  return in;
}

// One (row, head) query by one warp from its inputs (q_load); out: the 64 code bytes, the scale word and the
// head weight.
__device__ __forceinline__ void q_prep_in(const QIn& in, float wproj, int lane, uint8_t* q_out, uint32_t* sc_out,
                                          float* w_out, uint16_t* debug) {
  const uint2 raw = in.raw;
  float v[4] = {__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u), __uint_as_float(raw.y << 16),
                __uint_as_float(raw.y & 0xffff0000u)};
  if (lane >= 16) {  // dims 64..127: rotary pairs 2l-32 and 2l-31
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float2 cs = in.cs[j];
      rope_pair(v[2 * j], v[2 * j + 1], cs.x, cs.y, false);
      v[2 * j] = bfround(v[2 * j]);
      v[2 * j + 1] = bfround(v[2 * j + 1]);
    }
  }
  float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
  am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
  am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
  const float s = fp4_e8m0_scale(am), inv = pow2_inv(s);
  uint32_t c[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) c[i] = fp4_code_inv(v[i], inv);
  // canonical bytes 2l and 2l+1 share a word, so they stay adjacent in the transposed order
  *reinterpret_cast<uint16_t*>(q_out + fp4x128_byte(2 * lane)) = (uint16_t)(c[0] | c[1] << 4 | c[2] << 8 | c[3] << 12);
  uint32_t e = f_to_ue8m0(s);
  uint32_t word = __shfl_sync(0xffffffffu, e, 0) | __shfl_sync(0xffffffffu, e, 8) << 8 |
                  __shfl_sync(0xffffffffu, e, 16) << 16 | __shfl_sync(0xffffffffu, e, 24) << 24;
  if (lane == 0) {
    *sc_out = word;
    *w_out = __fmul_rn(wproj, 0x1p-6f);  // softmax_scale * n_heads^-0.5 = 128^-0.5 * 32^-0.5, exact
  }
  if (debug) {
#pragma unroll
    for (int i = 0; i < 4; ++i) debug[4 * lane + i] = f2bf(__fmul_rn(e2m1_to_f(c[i]), s));
  }
}
__device__ __forceinline__ void q_prep_head(const uint16_t* src, float wproj, int p, const float2* table, int lane,
                                            uint8_t* q_out, uint32_t* sc_out, float* w_out, uint16_t* debug) {
  q_prep_in(q_load(src, p, table, lane), wproj, lane, q_out, sc_out, w_out, debug);
}

// one warp per (row, head)
__global__ void __launch_bounds__(128) q_prep_kernel(const uint16_t* iq_pre, const float* wproj, const int* pos,
                                                     const float2* table, IndexQ q, uint16_t* debug_q, int rows) {
  int wid = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (wid >= rows * kIdxHeads) return;
  q_prep_head(iq_pre + (size_t)wid * kIdxDim, wproj[wid], pos[wid / kIdxHeads], table, lane, q.q + (size_t)wid * 64,
              q.sc + wid, q.w + wid, debug_q ? debug_q + (size_t)wid * kIdxDim : nullptr);
}

// ---- scores ---------------------------------------------------------------------------------------
// score[r][p] = bf16(sum_h bf16(relu(bf16(q[r,h] . k[p])) * w[r,h])), as model.py's einsum, relu, mul
// and sum. The fp4 dot products are exact in fp32 (both here and in the reference) unless a vector's
// block scales spread over more than ~2^11; the head sum is fp32 in another order (bf16-exact in practice).
// A tile is 16 positions (MMA rows) x 8 heads (MMA columns); four column tiles cover the 32 heads.

constexpr int kScoreWarps = 4, kScoreTiles = 4, kScorePos = kScoreWarps * kScoreTiles * 16;
static_assert(kScorePos == kScoreAlign, "keys_stride padding");

struct ScoreArgs {
  const uint8_t* q;
  const uint32_t* qsc;
  const float* w;
  const int* pos;
  int rows, ratio;
  const uint8_t* k;
  const uint32_t* ksc;
  int part, nparts;
  const int* blocks;
  int nblocks;
  uint16_t* keys;
  int keys_stride;
  uint16_t* bkeys;
  int bkeys_stride;
  const int* slot;    // IndexKeys' session slots (attn.h): slot s's keys from entry s * slot_entries on
  int slot_entries;  // (slot_entries below)
};

// Row r's session's first key entry: an offset in entries, so that the key pointers stay kernel parameters
// (pointers held in registers cost index_dense spills)
__device__ __forceinline__ uint32_t keys_of(const ScoreArgs& a, int r) {
  return a.slot ? (uint32_t)a.slot[r] * a.slot_entries : 0u;
}

// A 16-position tile's keys as one lane holds them: the positions of MMA rows g and g + 8 (-1 for none),
// their code words for this lane, and the scales of MMA row g + 8(t&1), which this lane supplies.
struct TileKeys {
  int p0, p1;
  uint4 k0, k1;
  uint32_t sa;
};
__device__ __forceinline__ TileKeys load_tile(const ScoreArgs& a, uint32_t e0, const int* blist, int i0, int n, int lo,
                                              int lane) {
  int g = lane >> 2, t = lane & 3;
  TileKeys k;
  if (blist) {
    int b0 = blist[i0 >> 3], b1 = (i0 >> 3) + 1 < a.nblocks ? blist[(i0 >> 3) + 1] : -1;
    k.p0 = b0 < 0 ? -1 : b0 * 8 + g;
    k.p1 = b1 < 0 ? -1 : b1 * 8 + g;
  } else {
    k.p0 = i0 + g < n ? lo + i0 + g : -1;
    k.p1 = i0 + 8 + g < n ? lo + i0 + 8 + g : -1;
  }
  k.k0 = make_uint4(0, 0, 0, 0);
  k.k1 = k.k0;
  uint32_t s0 = 0x7f7f7f7fu, s1 = 0x7f7f7f7fu;
  if (k.p0 >= 0) {  // entry e0 + p0, unsigned: one wide multiply-add per address
    const uint32_t e = e0 + k.p0;
    k.k0 = __ldg(reinterpret_cast<const uint4*>(a.k + (size_t)e * 64) + t);
    s0 = __ldg(a.ksc + e);
  }
  if (k.p1 >= 0) {
    const uint32_t e = e0 + k.p1;
    k.k1 = __ldg(reinterpret_cast<const uint4*>(a.k + (size_t)e * 64) + t);
    s1 = __ldg(a.ksc + e);
  }
  k.sa = (t & 1) ? s1 : s0;
  return k;
}

// Adds a key to a histogram of key>>5 in packed u16 counters (bin b: half b&1 of word b>>1).
__device__ __forceinline__ void hist_add(uint32_t* h, uint32_t key) {
  atomicAdd(&h[key >> 6], 1u << ((key >> 1) & 16));
}

// One row's scores of a tile: its keys krow[i0 .. i0+16) and, with brow, the block maxima of blocks i0/8 and
// i0/8 + 1 (the block of the row's newest position, len - 1, pinned to 0xFFFF). sq/ssc/sw: the row's queries.
// hk (hb): histograms (hist_add) that also count the nonzero keys (block maxima) written.
__device__ __forceinline__ void score_tile_row(const TileKeys& k, const uint8_t (*sq)[64], const uint32_t* ssc,
                                               const float* sw, int len, int i0, int lo, uint16_t* krow,
                                               uint16_t* brow, int lane, uint32_t* hk = nullptr,
                                               uint32_t* hb = nullptr) {
  int g = lane >> 2, t = lane & 3;
  float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    int h = nt * 8 + g;
    uint4 qv = *reinterpret_cast<const uint4*>(&sq[h][16 * t]);
    uint32_t qs = ssc[h];
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_fp4(d, k.k0.x, k.k1.x, k.k0.y, k.k1.y, qv.x, qv.y, k.sa & 0xffff, qs & 0xffff);
    mma_fp4(d, k.k0.z, k.k1.z, k.k0.w, k.k1.w, qv.z, qv.w, k.sa >> 16, qs >> 16);
    float w0 = sw[nt * 8 + 2 * t], w1 = sw[nt * 8 + 2 * t + 1];
    acc0 = __fadd_rn(acc0, bfround(__fmul_rn(fmaxf(bfround(d[0]), 0.f), w0)));
    acc0 = __fadd_rn(acc0, bfround(__fmul_rn(fmaxf(bfround(d[1]), 0.f), w1)));
    acc1 = __fadd_rn(acc1, bfround(__fmul_rn(fmaxf(bfround(d[2]), 0.f), w0)));
    acc1 = __fadd_rn(acc1, bfround(__fmul_rn(fmaxf(bfround(d[3]), 0.f), w1)));
  }
  acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 1));
  acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 2));
  acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 1));
  acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 2));
  uint32_t key0 = (k.p0 >= 0 && k.p0 < len) ? key_from_bf16(f2bf(acc0)) : 0u;
  uint32_t key1 = (k.p1 >= 0 && k.p1 < len) ? key_from_bf16(f2bf(acc1)) : 0u;
  if (t == 0) krow[i0 + g] = (uint16_t)key0;
  if (t == 1) krow[i0 + 8 + g] = (uint16_t)key1;
  if (hk && t < 2 && (t ? key1 : key0)) hist_add(hk, t ? key1 : key0);
  if (brow) {  // block maxima over g (lanes 4g): blocks i0/8 and i0/8 + 1
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      key0 = max(key0, __shfl_xor_sync(0xffffffffu, key0, o));
      key1 = max(key1, __shfl_xor_sync(0xffffffffu, key1, o));
    }
    if (lane == 0) {
      int pin = len > 0 ? (len - 1) >> 3 : -1, b = (lo + i0) >> 3;
      uint32_t m0 = b == pin ? 0xFFFF : key0, m1 = b + 1 == pin ? 0xFFFF : key1;
      brow[i0 >> 3] = (uint16_t)m0;
      brow[(i0 >> 3) + 1] = (uint16_t)m1;
      if (hb && m0) hist_add(hb, m0);
      if (hb && m1) hist_add(hb, m1);
    }
  }
}

template <int RB>
__global__ void __launch_bounds__(128) score_kernel(ScoreArgs a) {
  __shared__ __align__(16) uint8_t sq[RB][kIdxHeads][64];
  __shared__ uint32_t ssc[RB][kIdxHeads];
  __shared__ float sw[RB][kIdxHeads];
  __shared__ int slen[RB];
  int r0 = blockIdx.y * RB, nr = min(RB, a.rows - r0);
  int lo = 0, n;
  if (a.blocks) {
    n = a.nblocks * 8;
  } else {
    Range d = dense_range(a.pos, a.rows - 1, a.ratio, a.part, a.nparts);
    lo = d.lo;
    n = d.hi - d.lo;
  }
  int base = blockIdx.x * kScorePos;
  if (base >= n) return;
  const uint4* qsrc = reinterpret_cast<const uint4*>(a.q + (size_t)r0 * kIdxHeads * 64);
  for (int i = threadIdx.x; i < nr * kIdxHeads * 4; i += blockDim.x) reinterpret_cast<uint4*>(&sq[0][0][0])[i] = qsrc[i];
  for (int i = threadIdx.x; i < nr * kIdxHeads; i += blockDim.x) {
    (&ssc[0][0])[i] = a.qsc[(size_t)r0 * kIdxHeads + i];
    (&sw[0][0])[i] = a.w[(size_t)r0 * kIdxHeads + i];
  }
  if (threadIdx.x < nr) slen[threadIdx.x] = (a.pos[r0 + threadIdx.x] + 1) / a.ratio;
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int* blist = a.blocks ? a.blocks + (size_t)r0 * a.nblocks : nullptr;
  for (int tile = 0; tile < kScoreTiles; ++tile) {
    int i0 = base + (warp * kScoreTiles + tile) * 16;
    if (i0 >= n) break;
    TileKeys k = load_tile(a, 0, blist, i0, n, lo, lane);  // one session (index_score)
    for (int rr = 0; rr < nr; ++rr)
      score_tile_row(k, sq[rr], ssc[rr], sw[rr], slen[rr], i0, lo, a.keys + (size_t)(r0 + rr) * a.keys_stride,
                     a.bkeys ? a.bkeys + (size_t)(r0 + rr) * a.bkeys_stride : nullptr, lane);
  }
}

// ---- selection ------------------------------------------------------------------------------------
// One CTA per row, three passes over the keys: a 2048-bin histogram of key>>5 finds the bin holding the
// k-th key, a 32-bin histogram inside it finds the exact threshold key, and an ordered pass writes the
// keys above it plus the first `need` keys equal to it, in scan order (= ascending position).

constexpr int kSelThreads = 1024, kSelPer = 16, kSelChunk = kSelThreads * kSelPer;

struct SelectArgs {
  const uint16_t* keys;
  int keys_stride;
  int n;  // scan length; -1: this part's dense length (from pos/ratio/part/nparts)
  const int* pos;
  int rows, ratio, part, nparts;
  int unit;  // 1: positions; 8: block maxima -> block indices
  const int* blocks;
  int nblocks;
  const int* cand;
  int cand_stride;
  int kmax;
  int* out_pos;
  uint16_t* out_key;
  int out_stride;
  const int* slot;  // session slots (attn.h): a dense part's length is the row's session's
};

__device__ __forceinline__ int scan_pos(const SelectArgs& a, int r, int i, int lo) {
  if (a.cand) return a.cand[(size_t)r * a.cand_stride + i];
  if (a.blocks) {
    int b = a.blocks[(size_t)r * a.nblocks + (i >> 3)];
    return b < 0 ? -1 : b * 8 + (i & 7);
  }
  return lo / a.unit + i;
}

// exclusive prefix over the 1024 threads; *total gets the sum. tmp: 64 words.
__device__ __forceinline__ uint32_t block_excl_scan(uint32_t v, uint32_t* tmp, uint32_t* total) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  uint32_t x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    uint32_t y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  __syncthreads();
  if (lane == 31) tmp[w] = x;
  __syncthreads();
  if (w == 0) {
    uint32_t s = tmp[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t y = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= o) s += y;
    }
    tmp[32 + lane] = s;
  }
  __syncthreads();
  *total = tmp[63];
  return (w ? tmp[32 + w - 1] : 0) + x - v;
}

// kL2: keys other CTAs of this launch wrote (L2 loads; L1 is not coherent)
template <bool kL2>
__device__ __forceinline__ void load16(const uint16_t* row, int i0, int n, uint32_t (&k)[16]) {
  if (i0 + 16 <= n) {
    const uint4* p = reinterpret_cast<const uint4*>(row + i0);
    uint4 a = kL2 ? __ldcg(p) : p[0], b = kL2 ? __ldcg(p + 1) : p[1];
    const uint32_t wd[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      k[2 * j] = wd[j] & 0xffff;
      k[2 * j + 1] = wd[j] >> 16;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 16; ++j) k[j] = i0 + j < n ? (kL2 ? __ldcg(row + i0 + j) : row[i0 + j]) : 0u;
  }
}

// The exact threshold key inside bin `bin`, from its histogram h2 of key & 31 (warp 0): info[2] gets the K-th
// key, info[3] how many keys equal to it to take; `above` keys lie above the bin.
__device__ __forceinline__ void threshold_key(const uint32_t* h2, uint32_t bin, uint32_t above, uint32_t K,
                                              uint32_t* info) {
  int tid = threadIdx.x;
  if (tid < 32) {  // lane j looks at sub-bin 31-j, counting from the top
    uint32_t c = h2[31 - tid], incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += y;
    }
    uint32_t excl = incl - c;
    if (above + excl < K && K <= above + incl) {
      info[2] = (bin << 5) | (31 - tid);
      info[3] = K - (above + excl);
    }
  }
  __syncthreads();
}

// The 8 keys row[i0 .. i0 + 8) (i0 a multiple of 8, all written)
template <bool kL2>
__device__ __forceinline__ void load8(const uint16_t* row, int i0, uint32_t (&k)[8]) {
  const uint4* p = reinterpret_cast<const uint4*>(row + i0);
  uint4 a = kL2 ? __ldcg(p) : p[0];
  const uint32_t wd[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    k[2 * j] = wd[j] & 0xffff;
    k[2 * j + 1] = wd[j] >> 16;
  }
}

// Row r's selection by the 1024 threads of a CTA, given the histogram of its keys' key>>5 (key 0, none, not
// counted; thread tid holds bins 2047 - 2 tid and 2046 - 2 tid): the bin holding the k-th key, a 32-bin
// histogram inside it finds the exact threshold key, and an ordered pass writes the keys above it plus the
// first `need` keys equal to it, in scan order (= ascending position). With room for `cap` candidates
// (ckey/cidx in shared memory), when the keys in or above the threshold bin fit there, one ordered pass
// gathers them instead and the rest works on those: over the keys, or with bmax (the maxima of the row's
// blocks of 8 keys) over those, listing the blocks that reach the bin in blist (room for cap), then over the
// keys of the listed blocks.
template <bool kL2>
__device__ __forceinline__ void select_bins(const SelectArgs& a, int r, int lo, int n, uint32_t top, uint32_t next,
                                            int cap = 0, uint16_t* ckey = nullptr, uint32_t* cidx = nullptr,
                                            const uint16_t* bmax = nullptr, uint16_t* blist = nullptr) {
  __shared__ uint32_t h2[32];
  __shared__ uint32_t tmp[64];
  __shared__ uint32_t info[5];  // threshold bin, keys above it, threshold key, ties to take, keys in the bin
  int tid = threadIdx.x;
  const uint16_t* row = a.keys + (size_t)r * a.keys_stride;
  if (tid < 32) h2[tid] = 0;
  uint32_t total, pre = block_excl_scan(top + next, tmp, &total);
  uint32_t K = min((uint32_t)a.kmax, total);
  if (K > 0) {
    if (pre < K && K <= pre + top) {
      info[0] = 2047 - 2 * tid;
      info[1] = pre;
      info[4] = top;
    } else if (pre + top < K && K <= pre + top + next) {
      info[0] = 2046 - 2 * tid;
      info[1] = pre + top;
      info[4] = next;
    }
  }
  __syncthreads();
  int* opos = a.out_pos + (size_t)r * a.out_stride;
  uint16_t* okey = a.out_key + (size_t)r * a.out_stride;
  if (K > 0 && info[1] + info[4] <= (uint32_t)cap) {
    // the candidates, in scan order, and the threshold bin's histogram
    uint32_t bin = info[0], above = info[1], nc = 0;
    auto take = [&](uint32_t k, int i, uint32_t& o) {
      if (k && (k >> 5) >= bin && o < (uint32_t)cap) {
        ckey[o] = (uint16_t)k;
        cidx[o++] = i;
        if ((k >> 5) == bin) atomicAdd(&h2[k & 31], 1u);
      }
    };
    if (bmax) {  // the blocks that reach the bin, in order (each holds a candidate, so at most cap)
      const int nb = (n + 7) / 8;
      uint32_t nl = 0;
      for (int base = 0; base < nb; base += kSelChunk) {
        const int b0 = base + tid * kSelPer;
        uint32_t m[16], sel = 0;
        load16<kL2>(bmax, b0, nb, m);
#pragma unroll
        for (int j = 0; j < 16; ++j) sel |= (uint32_t)(m[j] && (m[j] >> 5) >= bin) << j;
        uint32_t tot, o = nl + block_excl_scan(__popc(sel), tmp, &tot);
        for (uint32_t t = sel; t; t &= t - 1)
          if (o < (uint32_t)cap) blist[o++] = (uint16_t)(b0 + __ffs(t) - 1);
        nl += tot;
      }
      __syncthreads();
      nl = min(nl, (uint32_t)cap);
      for (uint32_t base = 0; base < nl; base += kSelThreads) {  // a thread: a listed block's keys
        const uint32_t i = base + tid;
        const int i0 = i < nl ? blist[i] * 8 : 0;
        uint32_t k[8] = {}, c = 0;
        if (i < nl) {
          load8<kL2>(row, i0, k);
#pragma unroll
          for (int j = 0; j < 8; ++j) c += k[j] && (k[j] >> 5) >= bin;
        }
        uint32_t tot, o = nc + block_excl_scan(c, tmp, &tot);
        if (i < nl) {
#pragma unroll
          for (int j = 0; j < 8; ++j) take(k[j], i0 + j, o);
        }
        nc += tot;
      }
    } else {
      for (int base = 0; base < n; base += kSelChunk) {
        int i0 = base + tid * kSelPer;
        uint32_t k[16], c = 0;
        load16<kL2>(row, i0, n, k);
#pragma unroll
        for (int j = 0; j < 16; ++j) c += k[j] && (k[j] >> 5) >= bin;
        uint32_t tot, o = nc + block_excl_scan(c, tmp, &tot);
#pragma unroll
        for (int j = 0; j < 16; ++j) take(k[j], i0 + j, o);
        nc += tot;
      }
    }
    __syncthreads();
    threshold_key(h2, bin, above, K, info);
    nc = min(nc, (uint32_t)cap);
    uint32_t theta = info[2], need = info[3], per = (nc + kSelThreads - 1) / kSelThreads;
    uint32_t c0 = min(nc, tid * per), c1 = min(nc, c0 + per), gt = 0, eq = 0;
    for (uint32_t i = c0; i < c1; ++i) {
      gt += ckey[i] > theta;
      eq += ckey[i] == theta;
    }
    uint32_t tot, p = block_excl_scan(gt << 16 | eq, tmp, &tot);
    uint32_t tie = p & 0xffff, o = (p >> 16) + min(tie, need);
    for (uint32_t i = c0; i < c1; ++i) {
      uint32_t k = ckey[i];
      if (k > theta || (k == theta && tie++ < need)) {
        opos[o] = scan_pos(a, r, cidx[i], lo);
        okey[o] = (uint16_t)k;
        ++o;
      }
    }
  } else if (K > 0) {
    uint32_t bin = info[0], above = info[1];
    for (int base = 0; base < n; base += kSelChunk) {
      uint32_t k[16];
      load16<kL2>(row, base + tid * kSelPer, n, k);
#pragma unroll
      for (int j = 0; j < 16; ++j)
        if (k[j] && (k[j] >> 5) == bin) atomicAdd(&h2[k[j] & 31], 1u);
    }
    __syncthreads();
    threshold_key(h2, bin, above, K, info);
    uint32_t theta = info[2], need = info[3], sel_base = 0, tie_base = 0;
    for (int base = 0; base < n; base += kSelChunk) {
      int i0 = base + tid * kSelPer;
      uint32_t k[16];
      load16<kL2>(row, i0, n, k);
      uint32_t gt = 0, eq = 0;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        gt += k[j] > theta;
        eq += k[j] == theta;
      }
      uint32_t tot, p = block_excl_scan(gt << 16 | eq, tmp, &tot);
      uint32_t tie = tie_base + (p & 0xffff);
      uint32_t o = sel_base + (p >> 16) + min(tie, need);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        bool take = k[j] > theta || (k[j] == theta && tie++ < need);
        if (take) {
          opos[o] = scan_pos(a, r, i0 + j, lo);
          okey[o] = (uint16_t)k[j];
          ++o;
        }
      }
      sel_base += tot >> 16;
      tie_base += tot & 0xffff;
    }
  }
  for (int i = K + tid; i < a.kmax; i += kSelThreads) {
    opos[i] = -1;
    okey[i] = 0;
  }
}

// Row r's selection by the 1024 threads of a CTA: a pass over its keys for the histogram, then select_bins.
template <bool kL2>
__device__ __forceinline__ void select_row(const SelectArgs& a, int r) {
  __shared__ uint32_t hist[2048];
  int tid = threadIdx.x;
  const uint16_t* row = a.keys + (size_t)r * a.keys_stride;
  int lo = 0, n = a.n;
  if (n < 0) {
    Range d = dense_range(a.pos, session_last(a.slot, a.rows, r), a.ratio, a.part, a.nparts);
    lo = d.lo;
    n = (d.hi - d.lo + a.unit - 1) / a.unit;
  }
  hist[tid] = 0;
  hist[tid + 1024] = 0;
  __syncthreads();
  for (int base = 0; base < n; base += kSelChunk) {
    uint32_t k[16];
    load16<kL2>(row, base + tid * kSelPer, n, k);
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (k[j]) atomicAdd(&hist[k[j] >> 5], 1u);
  }
  __syncthreads();
  select_bins<kL2>(a, r, lo, n, hist[2047 - 2 * tid], hist[2046 - 2 * tid]);
}

__global__ void __launch_bounds__(kSelThreads) select_kernel(SelectArgs a) { select_row<false>(a, blockIdx.x); }

// ---- the fused decode path ----------------------------------------------------------------------
// index_q_prep + index_score + index_topk in one launch: grid (CTAs of 2048 scan positions, rows). Every CTA
// prepares its row's 32 queries (warp h: head h, as q_prep_kernel) while its tiles' keys load, scores as
// score_kernel, and the last CTA of the row to finish runs the selection(s) over the row's keys.
constexpr int kFusedPos = kSelThreads / 32 * kScoreTiles * 16;
struct IndexFusedArgs {
  const uint16_t* iq_pre;
  const float* wproj;
  const float2* table;
  ScoreArgs sc;          // q, qsc and w unused
  SelectArgs top, pool;  // pool.out_pos null: no pool
  int* counters;
};
template <bool kSlots>  // a slot table (ScoreArgs::slot); without one, the code of one session
__global__ void __launch_bounds__(kSelThreads) index_fused_kernel(IndexFusedArgs a) {
  __shared__ __align__(16) uint8_t sq[kIdxHeads][64];
  __shared__ uint32_t ssc[kIdxHeads];
  __shared__ float sw[kIdxHeads];
  __shared__ int last;
  pdl_trigger();
  pdl_wait();
  const ScoreArgs& s = a.sc;
  int r = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int lo = 0, n;
  if (s.blocks) {
    n = s.nblocks * 8;
  } else {
    Range d = dense_range(s.pos, kSlots ? session_last(s.slot, s.rows, r) : s.rows - 1, s.ratio, s.part, s.nparts);
    lo = d.lo;
    n = d.hi - d.lo;
  }
  int base = blockIdx.x * kFusedPos, p = s.pos[r];
  if (base < n) {
    const int* blist = s.blocks ? s.blocks + (size_t)r * s.nblocks : nullptr;
    const uint32_t e0 = kSlots ? keys_of(s, r) : 0u;
    TileKeys k[kScoreTiles];
#pragma unroll
    for (int tile = 0; tile < kScoreTiles; ++tile) {
      int i0 = base + (warp * kScoreTiles + tile) * 16;
      if (i0 < n) k[tile] = load_tile(s, e0, blist, i0, n, lo, lane);
    }
    q_prep_head(a.iq_pre + ((size_t)r * kIdxHeads + warp) * kIdxDim, a.wproj[(size_t)r * kIdxHeads + warp], p,
                a.table, lane, sq[warp], &ssc[warp], &sw[warp], nullptr);
    __syncthreads();
    int len = (p + 1) / s.ratio;
    uint16_t* krow = s.keys + (size_t)r * s.keys_stride;
    uint16_t* brow = s.bkeys ? s.bkeys + (size_t)r * s.bkeys_stride : nullptr;
#pragma unroll
    for (int tile = 0; tile < kScoreTiles; ++tile) {
      int i0 = base + (warp * kScoreTiles + tile) * 16;
      if (i0 < n) score_tile_row(k[tile], sq, ssc, sw, len, i0, lo, krow, brow, lane);
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(a.counters + r, 1) == (int)gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  select_row<true>(a.top, r);
  if (a.pool.out_pos) {
    __syncthreads();
    select_row<true>(a.pool, r);
  }
  if (threadIdx.x == 0) a.counters[r] = 0;
}

// ---- the decode step's dense scans ------------------------------------------------------------------
// index_fused over a whole dense scan for all the rows of a decode or verify step in one grid of one CTA per
// SM, whatever the context: the CTAs split the work by the scan length the positions give, so one launch (one
// graph) serves every context. They take the scan's chunks of kDensePos positions in turn. A chunk's keys
// are loaded once (the next chunk's while this one is scored) and scored for every row of the CTA's row
// group: all the rows, unless the scan has too few chunks to give each SM two, when the rows are split into
// groups (dense_per; CTA b: group b % groups, chunks b / groups + i * stride, CTAs past groups * stride
// idle). Each CTA counts the keys (and for
// the pool the block maxima, which every scan writes) it writes per row in histograms of key>>5 in shared
// memory, then adds those to global ones. The last R CTAs to finish select a row each with select_bins from
// the global histogram, gathering the candidates in shared memory when they fit (kDenseCand): for a long
// scan, the top's from the keys of the blocks whose maxima reach the threshold bin.
// Session slots (attn.h): a dense scan's sessions each take a share of the CTAs by their rows x chunks (at
// least one) and split it as one session alone would; a block list's rows are on their own anyway. A row's
// scan length is its session's, its keys its slot's.
// 9..16 rows (B4): a group has at most kDenseRows rows, whose queries and histograms the CTA keeps in shared
// memory at the group's own offsets; a session of more rows reads its keys once per group.
constexpr int kDensePos = kSelThreads / 32 * 16;  // a chunk: one 16-position tile per warp
constexpr int kDenseCand = 10240, kDenseBmax = 2 * kSelChunk;  // bmax gathers past kDenseBmax keys
constexpr int kDenseHist = 32;  // scratch words: [0] CTAs done, [1] selecting CTAs done; then the histograms
constexpr int kDenseRows = kDecodeGroup;  // rows of a CTA's group
constexpr size_t kDenseSmemScore = (size_t)kDenseRows * kIdxHeads * (64 + 8) + 2 * (size_t)kDenseRows * 4096;
constexpr size_t kDenseSmem = kDenseSmemScore > (size_t)kDenseCand * 8 ? kDenseSmemScore : (size_t)kDenseCand * 8;

// Rows per group: as few as the G CTAs have room for with up to two chunks each (a CTA preps its rows'
// queries, about a chunk's scoring per row, so a second chunk costs less than more rows); all the rows once
// the chunks fill the CTAs.
__device__ __forceinline__ int dense_per(int rows, int chunks, int G) {
  int per = 1;
  while (per < rows && (chunks >= G || (rows + per - 1) / per * ((chunks + 1) / 2) > G)) ++per;
  return per;
}

struct IndexDenseArgs {
  const uint16_t* iq_pre;
  const float* wproj;
  const float2* table;
  ScoreArgs sc;          // q, qsc, w, part/nparts and blocks unused
  SelectArgs top, pool;  // pool.out_pos null: no pool
  uint32_t* scratch;     // kDenseHist words, then [kRows][2048] histograms of the keys, then of the maxima
  int cap;               // candidates select_bins may gather (kDenseCand; tests use fewer)
};

// A CTA's histogram word (bins 2j and 2j + 1, hist_add) into a global histogram at g = &hist[2j].
__device__ __forceinline__ void hist_flush(uint32_t w, uint32_t* g) {
  if (w & 0xffff) atomicAdd(g, w & 0xffff);
  if (w >> 16) atomicAdd(g + 1, w >> 16);
}

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// select_bins of row r (n keys) from its global histogram gh, which is left zero for the next launch.
__device__ __forceinline__ void select_dense(const SelectArgs& a, int r, uint32_t* gh, int n, int cap, uint16_t* ckey,
                                             uint32_t* cidx, const uint16_t* bmax, uint16_t* blist) {
  int tid = threadIdx.x;
  uint32_t top = __ldcg(gh + 2047 - 2 * tid), next = __ldcg(gh + 2046 - 2 * tid);
  gh[2047 - 2 * tid] = 0;
  gh[2046 - 2 * tid] = 0;
  select_bins<true>(a, r, 0, n, top, next, cap, ckey, cidx, bmax, blist);
}

// kSlots: a slot table (ScoreArgs::slot); without one, the code of one session. kRows: kDenseRows, or
// kMaxDecodeRows (groups of at most kDenseRows rows, each at the start of the shared memory)
template <bool kSlots, int kRows>
__global__ void __launch_bounds__(kSelThreads) index_dense_kernel(IndexDenseArgs a) {
  extern __shared__ __align__(16) uint8_t dsm[];
  __shared__ int slen[kRows], sslot[kRows];  // each row's own scan length, and its slot
  __shared__ int ticket;
  pdl_trigger();
  pdl_wait();
  const ScoreArgs& s = a.sc;
  const int R = s.rows, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool pool = a.pool.out_pos != nullptr;
  if (tid < R) {  // once, for the partition, the key offsets and the selections' lengths below
    slen[tid] = (s.pos[tid] + 1) / s.ratio;
    if (kSlots) sslot[tid] = s.slot[tid];
  }
  __syncthreads();
  auto last_of = [&](int r) {  // the last row of row r's session
    if (kSlots)
      while (r + 1 < R && sslot[r + 1] == sslot[r]) ++r;
    else
      r = R - 1;
    return r;
  };
  uint8_t(*sq)[kIdxHeads][64] = reinterpret_cast<uint8_t(*)[kIdxHeads][64]>(dsm);  // [row][head]
  uint32_t* ssc = reinterpret_cast<uint32_t*>(dsm + kDenseRows * kIdxHeads * 64);
  float* sw = reinterpret_cast<float*>(ssc + kDenseRows * kIdxHeads);
  uint32_t* hk = reinterpret_cast<uint32_t*>(sw + kDenseRows * kIdxHeads);  // [row][1024] (hist_add)
  uint32_t* hb = hk + kDenseRows * 1024;
  uint32_t* gk = a.scratch + kDenseHist;  // [row][2048]
  uint32_t* gb = gk + kRows * 2048;
  const int G = gridDim.x;
  int s0 = 0, s1 = R, n = 0, Gs = G, bl = blockIdx.x;  // this CTA's session (rows [s0, s1)), its CTAs, its index there
  if (s.blocks) {
    n = s.nblocks * 8;
  } else if (!kSlots || last_of(0) == R - 1) {  // one session: all the CTAs
    n = slen[R - 1];
  } else {
    int S = 0, W = 0;
    for (int e, a0 = 0; a0 < R; a0 = e) {
      e = last_of(a0) + 1;
      ++S;
      W += (e - a0) * ((slen[e - 1] + kDensePos - 1) / kDensePos);
    }
    Gs = 0;  // past every session's share: idle
    for (int e, a0 = 0, off = 0; a0 < R; a0 = e, off += Gs) {
      e = last_of(a0) + 1;
      n = slen[e - 1];
      Gs = 1 + (W ? (G - S) * ((e - a0) * ((n + kDensePos - 1) / kDensePos)) / W : 0);
      if ((int)blockIdx.x < off + Gs) {
        s0 = a0, s1 = e, bl = blockIdx.x - off;
        break;
      }
      if (e == R) Gs = 0;
    }
  }
  const int nchunks = (n + kDensePos - 1) / kDensePos, Rs = s1 - s0;
  int per = s.blocks || !Gs ? 1 : dense_per(Rs, nchunks, Gs);
  if (kRows > kDenseRows) per = min(per, kDenseRows);
  const int groups = (Rs + per - 1) / per;
  const int stride = max(1, min(nchunks, Gs / groups));  // CTAs per group
  const int r0 = s0 + bl % groups * per, r1 = min(s1, r0 + per);
  const int q0 = kRows > kDenseRows ? r0 : 0;  // row r's queries and histograms at r - q0
  int c = Gs && bl < groups * stride ? bl / groups : nchunks;
  if (c < nchunks && r0 < r1) {
    for (int i = (r0 - q0) * 1024 + tid; i < (r1 - q0) * 1024; i += kSelThreads) {
      hk[i] = 0;
      hb[i] = 0;
    }
    const int* rb = s.blocks ? s.blocks + (size_t)r0 * s.nblocks : nullptr;  // a block list: the group's one row
    const uint32_t e0 = kSlots ? (uint32_t)sslot[r0] * s.slot_entries : 0u;  // the group's session's keys
    TileKeys cur = load_tile(s, e0, rb, c * kDensePos + warp * 16, n, 0, lane);
    QIn qi = q_load(a.iq_pre + ((size_t)r0 * kIdxHeads + warp) * kIdxDim, s.pos[r0], a.table, lane);
    float wq = a.wproj[r0 * kIdxHeads + warp];
    for (int r = r0; r < r1; ++r) {  // the queries (warp h: head h of each row), the next row's loading
      QIn qn = qi;
      float wn = wq;
      if (r + 1 < r1) {
        qn = q_load(a.iq_pre + ((size_t)(r + 1) * kIdxHeads + warp) * kIdxDim, s.pos[r + 1], a.table, lane);
        wn = a.wproj[(r + 1) * kIdxHeads + warp];
      }
      q_prep_in(qi, wq, lane, sq[r - q0][warp], &ssc[(r - q0) * kIdxHeads + warp], &sw[(r - q0) * kIdxHeads + warp],
                nullptr);
      qi = qn;
      wq = wn;
    }
    __syncthreads();
    for (; c < nchunks; c += stride) {
      const int i0 = c * kDensePos + warp * 16;
      TileKeys nxt = cur;
      if (c + stride < nchunks) nxt = load_tile(s, e0, rb, i0 + stride * kDensePos, n, 0, lane);
      if (i0 < n)
        for (int r = r0; r < r1; ++r)
          score_tile_row(cur, sq[r - q0], ssc + (r - q0) * kIdxHeads, sw + (r - q0) * kIdxHeads, slen[r], i0, 0,
                         s.keys + (size_t)r * s.keys_stride, s.bkeys ? s.bkeys + (size_t)r * s.bkeys_stride : nullptr, lane,
                         hk + (r - q0) * 1024, pool ? hb + (r - q0) * 1024 : nullptr);
      cur = nxt;
    }
    __syncthreads();
    for (int r = r0; r < r1; ++r) {
      hist_flush(hk[(r - q0) * 1024 + tid], gk + r * 2048 + 2 * tid);
      if (pool) hist_flush(hb[(r - q0) * 1024 + tid], gb + r * 2048 + 2 * tid);
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) ticket = atomicAdd(a.scratch, 1u);
  __syncthreads();
  const int S = min(G, R), first = ticket - (G - S);
  if (first < 0) return;
  if (tid == 0)
    while (ld_acquire(a.scratch) < (uint32_t)G) __nanosleep(32);
  __syncthreads();
  __threadfence();
  uint16_t* ckey = reinterpret_cast<uint16_t*>(dsm);
  uint32_t* cidx = reinterpret_cast<uint32_t*>(dsm + kDenseCand * 2);
  uint16_t* blist = reinterpret_cast<uint16_t*>(dsm + kDenseCand * 6);
  for (int r = first; r < R; r += S) {
    const int nr = s.blocks ? s.nblocks * 8 : slen[last_of(r)];  // the session's
    select_dense(a.top, r, gk + r * 2048, nr, a.cap, ckey, cidx,
                 nr > kDenseBmax ? s.bkeys + (size_t)r * s.bkeys_stride : nullptr, blist);
    if (pool) {
      __syncthreads();
      select_dense(a.pool, r, gb + r * 2048, (nr + 7) / 8, a.cap, ckey, cidx, nullptr, nullptr);
    }
    __syncthreads();
  }
  if (tid == 0 && atomicAdd(a.scratch + 1, 1u) == (uint32_t)S - 1) {
    a.scratch[0] = 0;
    a.scratch[1] = 0;
  }
}

// index_merge of one or two candidate sets (grid.y) in one launch
__global__ void __launch_bounds__(kSelThreads) merge_kernel(SelectArgs a0, SelectArgs a1) {
  pdl_trigger();
  pdl_wait();
  if (blockIdx.y == 0)
    select_row<false>(a0, blockIdx.x);
  else
    select_row<false>(a1, blockIdx.x);
}

}  // namespace

void index_q_prep(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, IndexQ q,
                  uint16_t* debug_q, int rows, cudaStream_t s) {
  q_prep_kernel<<<(rows * kIdxHeads + 3) / 4, 128, 0, s>>>(iq_pre, wproj, pos, table, q, debug_q, rows);
  CUDA_CHECK(cudaGetLastError());
}

void index_score(const IndexQ& q, const int* pos, int rows, int ratio, IndexKeys ik, const IndexScan& scan,
                 int max_scan, uint16_t* keys, int keys_stride, uint16_t* bkeys, int bkeys_stride, cudaStream_t s) {
  int nscan = scan.blocks ? scan.nblocks * 8 : max_scan;
  int padded = (nscan + kScorePos - 1) / kScorePos * kScorePos;
  if (keys_stride < padded || (bkeys && (scan.blocks || bkeys_stride < padded / 8)) || ik.slot)
    die("index_score: strides %d/%d for %d (or a slot table)", keys_stride, bkeys_stride, nscan);
  ScoreArgs a{q.q, q.sc, q.w, pos, rows, ratio, ik.k, reinterpret_cast<const uint32_t*>(ik.sc), scan.part,
              scan.nparts, scan.blocks, scan.nblocks, keys, keys_stride, bkeys, bkeys_stride, nullptr, 0};
  if (scan.blocks || rows == 1) {
    score_kernel<1><<<dim3(padded / kScorePos, rows), 128, 0, s>>>(a);
  } else {
    score_kernel<8><<<dim3(padded / kScorePos, (rows + 7) / 8), 128, 0, s>>>(a);
  }
  CUDA_CHECK(cudaGetLastError());
}


namespace {

SelectArgs topk_args(const uint16_t* keys, int keys_stride, const int* pos, int rows, int ratio, const IndexScan& scan,
                     bool pool, int kmax, int* out_pos, uint16_t* out_key, int out_stride) {
  if (pool && scan.blocks) die("index_topk: the pool comes from a dense scan");
  SelectArgs a{};
  a.keys = keys;
  a.keys_stride = keys_stride;
  a.n = scan.blocks ? scan.nblocks * 8 : -1;
  a.pos = pos;
  a.rows = rows;
  a.ratio = ratio;
  a.part = scan.part;
  a.nparts = scan.nparts;
  a.unit = pool ? 8 : 1;
  a.blocks = scan.blocks;
  a.nblocks = scan.nblocks;
  a.kmax = kmax;
  a.out_pos = out_pos;
  a.out_key = out_key;
  a.out_stride = out_stride;
  return a;
}

SelectArgs merge_args(const int* cand_pos, const uint16_t* cand_key, int n, int kmax, int* out_pos, uint16_t* out_key,
                      int out_stride) {
  if (n % 8) die("index_merge: n=%d", n);
  SelectArgs a{};
  a.keys = cand_key;
  a.keys_stride = n;
  a.n = n;
  a.unit = 1;
  a.cand = cand_pos;
  a.cand_stride = n;
  a.kmax = kmax;
  a.out_pos = out_pos;
  a.out_key = out_key;
  a.out_stride = out_stride;
  return a;
}

}  // namespace

void index_topk(const uint16_t* keys, int keys_stride, const int* pos, int rows, int ratio, const IndexScan& scan,
                int max_scan, bool pool, int kmax, int* out_pos, uint16_t* out_key, int out_stride, cudaStream_t s) {
  (void)max_scan;
  SelectArgs a = topk_args(keys, keys_stride, pos, rows, ratio, scan, pool, kmax, out_pos, out_key, out_stride);
  select_kernel<<<rows, kSelThreads, 0, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

void index_merge(const int* cand_pos, const uint16_t* cand_key, int n, int rows, int kmax, int* out_pos,
                 uint16_t* out_key, int out_stride, cudaStream_t s) {
  SelectArgs a = merge_args(cand_pos, cand_key, n, kmax, out_pos, out_key, out_stride);
  select_kernel<<<rows, kSelThreads, 0, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

// IndexKeys' slot stride in whole entries: k_stride = 64 x entries = 16 x sc_stride (each slot [entries][64] and
// [entries][4])
static int slot_entries(const IndexKeys& ik, const char* who) {
  check_slots(ik, who);
  return ik.slot ? (int)(ik.k_stride / 64) : 0;
}

void index_fused(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, int rows, int ratio,
                 IndexKeys ik, const IndexScan& scan, int max_scan, uint16_t* keys, int keys_stride, uint16_t* bkeys,
                 int bkeys_stride, IndexSel top, IndexSel pool, int* counters, bool pdl, cudaStream_t s) {
  int nscan = scan.blocks ? scan.nblocks * 8 : max_scan;
  int padded = (nscan + kScorePos - 1) / kScorePos * kScorePos;
  if (keys_stride < padded || (bkeys && (scan.blocks || bkeys_stride < padded / 8)) || (pool.pos && !bkeys))
    die("index_fused: strides %d/%d for %d", keys_stride, bkeys_stride, nscan);
  IndexFusedArgs a{};
  a.iq_pre = iq_pre;
  a.wproj = wproj;
  a.table = table;
  a.sc = ScoreArgs{nullptr, nullptr, nullptr, pos, rows, ratio, ik.k, reinterpret_cast<const uint32_t*>(ik.sc),
                   scan.part, scan.nparts, scan.blocks, scan.nblocks, keys, keys_stride, bkeys, bkeys_stride,
                   ik.slot, slot_entries(ik, "index_fused")};
  a.top = topk_args(keys, keys_stride, pos, rows, ratio, scan, false, top.kmax, top.pos, top.key, top.stride);
  a.top.slot = ik.slot;
  if (pool.pos) {
    a.pool = topk_args(bkeys, bkeys_stride, pos, rows, ratio, scan, true, pool.kmax, pool.pos, pool.key, pool.stride);
    a.pool.slot = ik.slot;
  }
  a.counters = counters;
  launch(ik.slot ? index_fused_kernel<true> : index_fused_kernel<false>,
         dim3(std::max(1, (nscan + kFusedPos - 1) / kFusedPos), rows), dim3(kSelThreads), 0, s, pdl, a);
}

size_t index_dense_scratch() { return (kDenseHist + 2 * (size_t)kMaxDecodeRows * 2048) * sizeof(uint32_t); }

namespace {
using DenseKernel = void (*)(IndexDenseArgs);
DenseKernel dense_kernel(bool slots, int rows) {  // up to kDenseRows rows, the instances of before B4
  if (rows > kDenseRows)
    return slots ? index_dense_kernel<true, kMaxDecodeRows> : index_dense_kernel<false, kMaxDecodeRows>;
  return slots ? index_dense_kernel<true, kDenseRows> : index_dense_kernel<false, kDenseRows>;
}
int dense_sms() {  // the SM count; also the kernels' shared memory limit, once (outside any graph capture)
  static int sms = 0;
  if (!sms) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    for (bool slots : {false, true})
      for (int rows : {kDenseRows, kMaxDecodeRows})
        CUDA_CHECK(cudaFuncSetAttribute(dense_kernel(slots, rows), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)kDenseSmem));
  }
  return sms;
}
}  // namespace


void index_dense(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, int rows, int ratio,
                 IndexKeys ik, int max_scan, uint16_t* keys, int keys_stride, uint16_t* bkeys, int bkeys_stride,
                 IndexSel top, IndexSel pool, uint32_t* scratch, bool pdl, cudaStream_t s, int cand) {
  int padded = (max_scan + kScorePos - 1) / kScorePos * kScorePos;
  if (rows < 1 || rows > kMaxDecodeRows || keys_stride < padded || !bkeys || bkeys_stride < padded / 8 ||
      bkeys_stride > 65536)  // select_bins lists blocks by 16-bit index
    die("index_dense: %d rows, strides %d/%d for %d", rows, keys_stride, bkeys_stride, max_scan);
  const int sms = dense_sms();
  const IndexScan whole{};
  IndexDenseArgs a{};
  a.iq_pre = iq_pre;
  a.wproj = wproj;
  a.table = table;
  a.sc = ScoreArgs{nullptr, nullptr, nullptr, pos, rows, ratio, ik.k, reinterpret_cast<const uint32_t*>(ik.sc), 0, 1,
                   nullptr, 0, keys, keys_stride, bkeys, bkeys_stride, ik.slot, slot_entries(ik, "index_dense")};
  a.top = topk_args(keys, keys_stride, pos, rows, ratio, whole, false, top.kmax, top.pos, top.key, top.stride);
  if (pool.pos)
    a.pool = topk_args(bkeys, bkeys_stride, pos, rows, ratio, whole, true, pool.kmax, pool.pos, pool.key, pool.stride);
  a.scratch = scratch;
  a.cap = cand < 0 ? kDenseCand : std::min(cand, kDenseCand);
  launch(dense_kernel(ik.slot, rows), dim3(sms), dim3(kSelThreads), kDenseSmem, s, pdl, a);
}

void index_dense_blocks(const uint16_t* iq_pre, const float* wproj, const int* pos, const float2* table, int rows,
                        int ratio, IndexKeys ik, const int* blocks, int nblocks, uint16_t* keys, int keys_stride,
                        IndexSel top, uint32_t* scratch, bool pdl, cudaStream_t s, int cand) {
  if (rows < 1 || rows > kMaxDecodeRows || !blocks || keys_stride < nblocks * 8 || nblocks * 8 > kDenseBmax)
    die("index_dense_blocks: %d rows, %d blocks, stride %d", rows, nblocks, keys_stride);
  const int sms = dense_sms();
  IndexScan scan{};
  scan.blocks = blocks;
  scan.nblocks = nblocks;
  IndexDenseArgs a{};
  a.iq_pre = iq_pre;
  a.wproj = wproj;
  a.table = table;
  a.sc = ScoreArgs{nullptr, nullptr, nullptr, pos, rows, ratio, ik.k, reinterpret_cast<const uint32_t*>(ik.sc), 0, 1,
                   blocks, nblocks, keys, keys_stride, nullptr, 0, ik.slot, slot_entries(ik, "index_dense_blocks")};
  a.top = topk_args(keys, keys_stride, pos, rows, ratio, scan, false, top.kmax, top.pos, top.key, top.stride);
  a.scratch = scratch;
  a.cap = cand < 0 ? kDenseCand : std::min(cand, kDenseCand);
  launch(dense_kernel(ik.slot, rows), dim3(sms), dim3(kSelThreads), kDenseSmem, s, pdl, a);
}

void index_merge2(int rows, const int* cand_pos, const uint16_t* cand_key, int n, IndexSel top, const int* pcand_pos,
                  const uint16_t* pcand_key, int pn, IndexSel pool, bool pdl, cudaStream_t s) {
  SelectArgs a0 = merge_args(cand_pos, cand_key, n, top.kmax, top.pos, top.key, top.stride), a1 = a0;
  if (pool.pos) a1 = merge_args(pcand_pos, pcand_key, pn, pool.kmax, pool.pos, pool.key, pool.stride);
  launch(merge_kernel, dim3(rows, pool.pos ? 2 : 1), dim3(kSelThreads), 0, s, pdl, a0, a1);
}

}  // namespace attn
