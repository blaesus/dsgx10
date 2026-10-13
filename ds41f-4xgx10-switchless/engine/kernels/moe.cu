// MoE of one decode / verify step, per rank at TP=4: the routed MXFP4 experts (576-row slices of
// w1/w3, 576-column slices of w2) and the FP8 shared expert, for M <= 16 rows (from 9 rows the _r16
// kernels: the MMAs' upper half carries rows 8..15, the union holds up to 96 experts).
//
// Two kernels. moe_up: for every expert in the union of the rows' picks (each read once), w1 and w3
// in one task per 32 intermediate rows, then SwiGLU (up clamped to +-10, gate to <= 10), the routing
// weight, bf16, and the per-32 activation quantisation w2's Linear would do. moe_down: per 16 output
// rows, w2 of every expert in the union, each output rounded to bf16, summed in fp32 in expert-id
// order for the rows that picked it, then the shared expert's output added: the reference's order.
//
// Both compute the union themselves from the router's output, so moe_down can stream its weights
// while moe_up finishes (PDL), and moe_up's shared-expert tasks start before the router is done.
// moe_routed: both also run the router's selection themselves, from the gate's logits.
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

namespace dsk {

namespace {

constexpr int kDim = 5120, kInter = 576;  // per rank
constexpr int kJB = kInter / 32;          // 18 blocks of 32 intermediate rows
constexpr int kKB1 = kDim / 32, kKB2 = kInter / 32;
constexpr int kP1 = kDim / 64, kP2 = kInter / 64;  // steps of two k-blocks: 80 (w1/w3), 9 (w2)
constexpr size_t kW13 = (size_t)kInter * kDim / 2, kS13 = (size_t)kInter * kKB1;  // bytes per expert
constexpr size_t kW2 = (size_t)kDim * kInter / 2, kS2 = (size_t)kDim * kKB2;
constexpr int kT13 = kInter / 16, kT2 = kDim / 16;  // 16-row tiles a matrix: 36 (w1/w3), 320 (w2)
constexpr int kB13 = code_block_words(kP1), kB2 = code_block_words(kP2);  // scale_code's words a tile: 84, 12
constexpr int kMaskWords = 16;      // expert-id bitmap: up to 512 experts
// workspace of a call with row budget RB (8 up to 8 rows, 16 above): hq [RB * kTopK + 1][RB][kInter] (the
// union's slots, then the shared expert's), hs [..][..][kJB]; moe_up's task counters after the larger one
constexpr size_t kCtrOff = (size_t)(kMaxUnion + 1) * kMaxRows * (kInter + kJB);

struct Params {
  const uint8_t *xq, *xs;
  const int* idx;
  const float* wts;
  int M;
  MoeWeights w;
  uint8_t *hq, *hs;  // [RB * kTopK + 1][RB][kInter], [..][..][kJB]
  float* y;
  // moe_routed: select from logits [M][E] (fp32) + bias instead of reading idx/wts; moe_up's CTA 0
  // writes the selection to idx_out/wts_out when given
  const float *logits, *bias;
  int E;
  float route_scale;
  int* idx_out;
  float* wts_out;
  // moe_up's task queue in the workspace: [0] tasks claimed past the first wave, [1] CTAs done; both
  // zero between launches (the last CTA out resets them). Null: CTA b takes tasks b, b + grid, ...
  int* ctr;
};

// The sorted union of the experts picked by M <= RB rows, with each row's routing weight.
template <int RB>
struct Union {
  int n;
  int id[RB * kTopK];
  uint32_t rows[RB * kTopK];  // bit m: row m picked it
  float rw[RB * kTopK][RB];
  uint32_t mask[kMaskWords];
  int pre[kMaskWords];
  int sel_id[RB * kTopK];  // moe_routed: the rows' picks
  float sel_w[RB * kTopK];
};

// All threads of the CTA (>= 64, >= RB * kTopK). Reads idx/wts (or the logits) through L2: they come
// from an earlier kernel.
template <int RB>
__device__ void build_union(Union<RB>& U, const Params& a, int M) {
  const int tid = threadIdx.x, ne = M * kTopK;
  if (tid < kMaskWords) U.mask[tid] = 0;
  if (tid < RB * kTopK) U.rows[tid] = 0;
  if (a.logits)
    for (int r = tid >> 5; r < M; r += blockDim.x >> 5) {
      int id;
      float w;
      router_row(a.logits + (size_t)r * a.E, a.bias, a.E, a.route_scale, id, w);
      if ((tid & 31) < kTopK) U.sel_id[r * kTopK + (tid & 31)] = id, U.sel_w[r * kTopK + (tid & 31)] = w;
    }
  __syncthreads();
  int id = 0;
  float w = 0.f;
  if (tid < ne) {
    if (a.logits) {
      id = U.sel_id[tid], w = U.sel_w[tid];
    } else {
      id = __ldcg(a.idx + tid);
      w = __ldcg(a.wts + tid);
    }
    atomicOr(&U.mask[id >> 5], 1u << (id & 31));
  }
  __syncthreads();
  if (tid < 32) {
    int c = tid < kMaskWords ? __popc(U.mask[tid]) : 0, incl = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xffffffffu, incl, o);
      if (tid >= o) incl += v;
    }
    if (tid < kMaskWords) U.pre[tid] = incl - c;
    if (tid == 31) U.n = incl;
  }
  __syncthreads();
  if (tid < ne) {
    int r = U.pre[id >> 5] + __popc(U.mask[id >> 5] & ((1u << (id & 31)) - 1));
    int m = tid / kTopK;
    U.id[r] = id;
    U.rw[r][m] = w;
    atomicOr(&U.rows[r], 1u << m);
  }
  __syncthreads();
}

// One m16n8k32 step on a packed e2m1 word of a tiled FP4 row: word = 8 values (k = 8t..8t+7 of a
// block), A = the matching 8 activation bytes. The k order inside the block is permuted the same
// way for A and B, which the per-block scales make invisible.
__device__ __forceinline__ void mma4(float c[4], uint2 a, uint32_t word, uint32_t sfa, uint32_t sfb) {
  mma_e4m3_e2m1(c, a.x, 0u, a.y, 0u, e2m1x4_expand(word & 0xFFFFu), e2m1x4_expand(word >> 16), sfa, sfb);
}

// tile_fp8 layout (8-row tiles, 64-byte steps): trade halves with lane t^2, two MMAs (blocks 2p, 2p+1).
__device__ __forceinline__ void mma8(float c[4], const uint2 (&a)[2], const uint32_t (&sfa)[2], uint4 w, uint32_t sc,
                                     bool lo) {
  uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
  uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
  mma_e4m3_e4m3(c, a[0].x, 0u, a[0].y, 0u, lo ? w.x : r0, lo ? w.y : r1, sfa[0], sc & 0xFFu);
  mma_e4m3_e4m3(c, a[1].x, 0u, a[1].y, 0u, lo ? r0 : w.z, lo ? r1 : w.w, sfa[1], (sc >> 8) & 0xFFu);
}

// Two e2m1 values (one byte, low nibble first) -> fp32.
__device__ __forceinline__ float2 e2m1x2_to_f2(uint32_t byte) {
  uint32_t h2;
  asm("{ .reg .b8 b, z;\n mov.b16 {b, z}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b;\n}" : "=r"(h2) : "h"((uint16_t)byte));
  return __half22float2(*reinterpret_cast<__half2*>(&h2));
}
__device__ __forceinline__ float2 e4m3x2_to_f2(uint32_t v16) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v16, __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}

// Activation fragments of step p (blocks 2p, 2p+1) for this lane's row; zero rows past M.
__device__ __forceinline__ void act_frag(uint2 (&a)[2], uint32_t (&sfa)[2], const uint8_t* arow, const uint8_t* srow,
                                         int p, bool live, bool odd) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    a[h] = live ? __ldg(reinterpret_cast<const uint2*>(arow + (2 * p + h) * 32)) : make_uint2(0, 0);
    sfa[h] = (odd || !live) ? 127u : __ldg(srow + 2 * p + h);
  }
}

// 9..16 rows: rows g (a, the MMA's lower half) and g + 8 (a2, its upper half); the scale row srowm is the
// one this lane gives the MMA (row g + 8's for odd lanes), slive whether that row is live.
__device__ __forceinline__ void act_frag2(uint2 (&a)[2], uint2 (&a2)[2], uint32_t (&sfa)[2], const uint8_t* arow,
                                          const uint8_t* arow2, const uint8_t* srowm, int p, bool live, bool live2,
                                          bool slive) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    a[h] = live ? __ldg(reinterpret_cast<const uint2*>(arow + (2 * p + h) * 32)) : make_uint2(0, 0);
    a2[h] = live2 ? __ldg(reinterpret_cast<const uint2*>(arow2 + (2 * p + h) * 32)) : make_uint2(0, 0);
    sfa[h] = slive ? __ldg(srowm + 2 * p + h) : 127u;
  }
}
__device__ __forceinline__ void mma4h(float c[4], uint2 a, uint2 a2, uint32_t word, uint32_t sfa, uint32_t sfb) {
  mma_e4m3_e2m1(c, a.x, a2.x, a.y, a2.y, e2m1x4_expand(word & 0xFFFFu), e2m1x4_expand(word >> 16), sfa, sfb);
}
__device__ __forceinline__ void mma8h(float c[4], const uint2 (&a)[2], const uint2 (&a2)[2], const uint32_t (&sfa)[2],
                                      uint4 w, uint32_t sc, bool lo) {
  uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
  uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
  mma_e4m3_e4m3(c, a[0].x, a2[0].x, a[0].y, a2[0].y, lo ? w.x : r0, lo ? w.y : r1, sfa[0], sc & 0xFFu);
  mma_e4m3_e4m3(c, a[1].x, a2[1].x, a[1].y, a2[1].y, lo ? r0 : w.z, lo ? r1 : w.w, sfa[1], (sc >> 8) & 0xFFu);
}

// The 1-bit scale code (dsk.h, ScaleCode; made by scale_code below), read by the decode kernels' CS builds.
// moe_up: a warp's raw flags of steps p0 .. p0 + 31 (bit 0: step p0) from its tile's code block blk.
__device__ __forceinline__ uint32_t code_flags_up(const uint32_t* blk, int p0) {
  const uint4 f = __ldg(reinterpret_cast<const uint4*>(blk + kP1));
  const int f0 = p0 >> 5;
  return __funnelshift_r(f0 == 0 ? f.x : f0 == 1 ? f.y : f.z, f0 == 0 ? f.y : f0 == 1 ? f.z : f.w, p0 & 31);
}
// moe_down, a routed unit's scale loads beside its weights: x its code words (lane p: step p's; lanes past
// 8: step 8's), and with raw flags f, y: raw group lane / 8 of the unit's (its lowest flags dropped), word
// lane % 8, from sr, the tile's raw scales.
__device__ __forceinline__ void code_load_down(const uint32_t* blk, const uint8_t* sr, uint32_t f, int lane,
                                               uint32_t& x, uint32_t& y) {
  x = ldg_stream32(blk + min(lane, kP2 - 1));
  if (f) {
    const int r = lane >> 3;
    if (r > 0) f &= f - 1;
    if (r > 1) f &= f - 1;
    if (r > 2) f &= f - 1;
    if (f) y = ldg_stream32(sr + (__ffs(f) - 1) * 32 + (lane & 7) * 4);
  }
}
// moe_down: step(p, scale word) over a routed unit's 9 steps, each word rebuilt from its code word or,
// flagged in fi, its raw group: from y (the unit's first four) or loaded from sr (beyond). fi is
// warp-uniform: most units take the first branch.
template <class F>
__device__ __forceinline__ void code_steps_down(uint32_t fi, uint32_t x, uint32_t y, int g, uint32_t base4,
                                                const uint8_t* sr, F&& step) {
  if (fi == 0) {
#pragma unroll
    for (int p = 0; p < kP2; ++p) step(p, ((__shfl_sync(~0u, x, p) >> g) & 0x01010101u) + base4);
  } else {
#pragma unroll
    for (int p = 0; p < kP2; ++p) {
      const int r = __popc(fi & ((1u << p) - 1u));  // the unit's raw group r, if flagged
      const uint32_t cx = __shfl_sync(~0u, x, p), cy = __shfl_sync(~0u, y, (r & 3) * 8 + g);
      uint32_t sc = ((cx >> g) & 0x01010101u) + base4;
      if (fi >> p & 1u) sc = r < 4 ? cy : __ldg(reinterpret_cast<const uint32_t*>(sr + p * 32 + g * 4));
      step(p, sc);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// moe_up. Task = (expert, 32 intermediate rows): 4 tiles of 16 rows (w1 2jb, 2jb+1, w3 2jb, 2jb+1),
// all of K. Warps: WN = 4/NT along the tiles, WK along K; the partials meet in shared memory.
// Tasks 0..17 are the shared expert's (FP8, tile_fp8 layout), then 18 per routed expert. A
// persistent grid walks the tasks in a fixed order. CC: the routed experts' dot products on CUDA
// cores (fp32 FMAs) instead of tensor cores, for comparison. CS: the routed scales from their 1-bit
// code (a.w.c1, c3). Up to 8 rows.
template <int NT, int WK, int KU, bool CC, bool CS = false>
__global__ void __launch_bounds__(32 * (4 / NT) * WK) k_moe_up(Params a) {
  static_assert(!CS || (!CC && kP1 % WK == 0 && kP1 / WK <= 32 && 2 * KU < 32), "no scale-code build");
  constexpr int kMaxRows = kLowRows, kSlotSh = kLowRows * kTopK;  // up to 8 rows (k_moe_up_r16 below)
  constexpr int WN = 4 / NT, KH = KU / 2 > 0 ? KU / 2 : 1;
  __shared__ Union<kLowRows> U;
  __shared__ float red[WK][kMaxRows][64];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % WN, wi_k = warp / WN;
  const int p0 = kP1 * wi_k / WK, p1 = kP1 * (wi_k + 1) / WK;
  const int M = a.M;
  const bool live = g < M, odd = lane & 1, lo = t < 2;
  const uint8_t* arow4 = a.xq + (size_t)(live ? g : 0) * kDim + 8 * t;
  const uint8_t* arow8 = a.xq + (size_t)(live ? g : 0) * kDim + 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* srowa = a.xs + (size_t)(live ? g : 0) * kKB1;

  // shared expert (FP8): this warp's tiles of task jb
  uint4 c8a[KH][NT][2], c8b[KH][NT][2];
  uint32_t s8a[KH][NT], s8b[KH][NT];
  const uint8_t* w8[NT][2];
  const uint8_t* sr8[NT];
  auto setup8 = [&](int jb) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
      const uint8_t* W = ti < 2 ? a.w.sh_w1 : a.w.sh_w3;
      w8[i][0] = W + (size_t)(2 * tile) * 8 * kDim + g * 64 + 16 * t;
      w8[i][1] = w8[i][0] + 8 * kDim;
      sr8[i] = (ti < 2 ? a.w.sh_s1 : a.w.sh_s3) + (size_t)jb * kKB1;
    }
  };
  auto load8 = [&](uint4 (&b)[KH][NT][2], uint32_t (&s)[KH][NT], int p) {
#pragma unroll
    for (int u = 0; u < KH; ++u)
      if (p + u < p1)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          b[u][i][0] = ldg_stream128(w8[i][0] + (p + u) * 512);
          b[u][i][1] = ldg_stream128(w8[i][1] + (p + u) * 512);
          s[u][i] = __ldg(reinterpret_cast<const uint16_t*>(sr8[i] + 2 * (p + u)));
        }
  };
  float c[NT][2][4];
  float acc[NT][2][kMaxRows];  // CC: rows g, g+8 of each tile, per activation row (partial over t)
  auto comp8 = [&](const uint4 (&b)[KH][NT][2], const uint32_t (&s)[KH][NT], int p) {
#pragma unroll
    for (int u = 0; u < KH; ++u)
      if (p + u < p1) {
        uint2 av[2];
        uint32_t sfa[2];
        act_frag(av, sfa, arow8, srowa, p + u, live, odd);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          mma8(c[i][0], av, sfa, b[u][i][0], s[u][i], lo);
          mma8(c[i][1], av, sfa, b[u][i][1], s[u][i], lo);
        }
      }
  };

  int task = blockIdx.x;
  bool pre = task < kJB;  // the first task is the shared expert's: its weights need nothing upstream
  if (pre) {
    setup8(task);
    load8(c8a, s8a, p0);
  }
  pdl_wait();
  build_union(U, a, M);
  if (a.logits && a.idx_out && blockIdx.x == 0 && threadIdx.x < M * kTopK)
    a.idx_out[threadIdx.x] = U.sel_id[threadIdx.x], a.wts_out[threadIdx.x] = U.sel_w[threadIdx.x];
  pdl_launch_dependents();
  const int ntask = kJB * (1 + U.n);

  // The first wave is static (CTA b: task b, so the shared expert's tasks can load before the wait);
  // then each CTA claims the next task in order when it starts one, so a CTA that starts late (its SM
  // held by another kernel) takes fewer.
  __shared__ int s_next[2];
#pragma unroll 1
  for (int it = 0; task < ntask; ++it) {
    if (a.ctr && threadIdx.x == 0) s_next[it & 1] = (int)gridDim.x + atomicAdd(a.ctr, 1);
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        c[i][h][0] = c[i][h][1] = c[i][h][2] = c[i][h][3] = 0.f;
#pragma unroll
        for (int m = 0; m < kMaxRows; ++m) acc[i][h][m] = 0.f;
      }
    const bool sh = task < kJB;
    const int u = sh ? 0 : task / kJB - 1, jb = task % kJB;
    if (sh) {
      if (!pre) {
        setup8(jb);
        load8(c8a, s8a, p0);
      }
      for (int p = p0; p < p1; p += 2 * KH) {
        if (p + KH < p1) load8(c8b, s8b, p + KH);
        comp8(c8a, s8a, p);
        if (p + KH >= p1) break;
        if (p + 2 * KH < p1) load8(c8a, s8a, p + 2 * KH);
        comp8(c8b, s8b, p + KH);
      }
    } else {
      const int e = U.id[u];
      const uint8_t* w4[NT];
      const uint8_t* sr4[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
        w4[i] = (ti < 2 ? a.w.w1 : a.w.w3) + (size_t)e * kW13 + (size_t)tile * (kP1 * 512) + lane * 16;
        sr4[i] = (ti < 2 ? a.w.s1 : a.w.s3) + (size_t)e * kS13 + (size_t)tile * (kP1 * 32) + g * 4;
      }
      uint4 ba[KU][NT], bb[KU][NT];
      uint32_t sa[KU][NT], sb[KU][NT];
      auto load4 = [&](uint4 (&b)[KU][NT], uint32_t (&s)[KU][NT], int p) {
#pragma unroll
        for (int q = 0; q < KU; ++q)
          if (p + q < p1)
#pragma unroll
            for (int i = 0; i < NT; ++i) {
              b[q][i] = ldg_stream128(w4[i] + (p + q) * 512);
              s[q][i] = ldg_stream32(sr4[i] + (p + q) * 32);
            }
      };
      auto comp4 = [&](const uint4 (&b)[KU][NT], const uint32_t (&s)[KU][NT], int p) {
#pragma unroll
        for (int q = 0; q < KU; ++q)
          if (p + q < p1) {
            uint2 av[2];
            uint32_t sfa[2];
            act_frag(av, sfa, arow4, srowa, p + q, live, odd);
#pragma unroll
            for (int i = 0; i < NT; ++i) {
              const uint4 w = b[q][i];
              const uint32_t sc = s[q][i];
              mma4(c[i][0], av[0], w.x, sfa[0], sc & 0xFFu);
              mma4(c[i][0], av[1], w.y, sfa[1], (sc >> 8) & 0xFFu);
              mma4(c[i][1], av[0], w.z, sfa[0], (sc >> 16) & 0xFFu);
              mma4(c[i][1], av[1], w.w, sfa[1], sc >> 24);
            }
          }
      };
      // CUDA cores: weights decoded to fp32 once per step, then 8-term dot products per
      // (row, block, activation row), each scaled by both block scales and accumulated in fp32.
      auto comp4cc = [&](const uint4 (&b)[KU][NT], const uint32_t (&s)[KU][NT], int p) {
#pragma unroll
        for (int q = 0; q < KU; ++q)
          if (p + q < p1) {
            const int pp = p + q;
            float wf[NT][4][8], sw[NT][4];
#pragma unroll
            for (int i = 0; i < NT; ++i) {
              const uint32_t* wv = &b[q][i].x;
#pragma unroll
              for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                  float2 f = e2m1x2_to_f2(wv[j] >> (8 * v));
                  wf[i][j][2 * v] = f.x, wf[i][j][2 * v + 1] = f.y;
                }
                sw[i][j] = ue8m0_to_f32((s[q][i] >> (8 * j)) & 0xFFu);
              }
            }
#pragma unroll
            for (int m = 0; m < kMaxRows; ++m) {
              if (m >= M) break;
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                const uint2 xa = __ldg(reinterpret_cast<const uint2*>(a.xq + (size_t)m * kDim + 32 * (2 * pp + h) + 8 * t));
                const float sx = ue8m0_to_f32(__ldg(a.xs + (size_t)m * kKB1 + 2 * pp + h));
                float xf[8];
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                  float2 f = e4m3x2_to_f2((v < 2 ? xa.x : xa.y) >> (16 * (v & 1)));
                  xf[2 * v] = f.x, xf[2 * v + 1] = f.y;
                }
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                  for (int r = 0; r < 2; ++r) {
                    float d = 0.f;
#pragma unroll
                    for (int v = 0; v < 8; ++v) d = fmaf(xf[v], wf[i][2 * r + h][v], d);
                    acc[i][r][m] = fmaf(d, sx * sw[i][2 * r + h], acc[i][r][m]);
                  }
              }
            }
          }
      };
      if constexpr (CS) {
        // The 1-bit scale code: a step's group comes from its code word (scale_code) or, flagged raw, from sr4
        // as above, into today's MMAs in today's order. A tile's flags follow its kP1 code words; fl holds this
        // warp's steps p0 .. p1 - 1 as bits 0 ... The first two load groups go out before the flags are in,
        // every step taking its code word, and fetch their raw groups once the flags arrive (warp-uniform,
        // once a task); later loads take the group the flags name, so the loop has no branch.
        const uint32_t* cw[NT];
        uint32_t fl[NT], b4[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
          const ScaleCode& k = ti < 2 ? a.w.c1 : a.w.c3;
          cw[i] = k.code + ((size_t)e * kT13 + tile) * kB13;
          b4[i] = k.base4;
          fl[i] = code_flags_up(cw[i], p0);
        }
        // steps p .. p + KU - 1 with the flags f: the weights first, then each step's code word or raw group
        auto load4c = [&](uint4 (&b)[KU][NT], uint32_t (&s)[KU][NT], int p, const uint32_t (&f)[NT]) {
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1)
#pragma unroll
              for (int i = 0; i < NT; ++i) b[q][i] = ldg_stream128(w4[i] + (p + q) * 512);
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1)
#pragma unroll
              for (int i = 0; i < NT; ++i)
                s[q][i] = ldg_stream32_sel(f[i] >> (p - p0), 1u << q, sr4[i] + (p + q) * 32, cw[i] + p + q);
        };
        auto comp4c = [&](const uint4 (&b)[KU][NT], const uint32_t (&s)[KU][NT], int p) {
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1) {
              uint2 av[2];
              uint32_t sfa[2];
              act_frag(av, sfa, arow4, srowa, p + q, live, odd);
#pragma unroll
              for (int i = 0; i < NT; ++i) {
                const uint4 w = b[q][i];
                const uint32_t sc = fl[i] >> (p + q - p0) & 1u ? s[q][i] : ((s[q][i] >> g) & 0x01010101u) + b4[i];
                mma4(c[i][0], av[0], w.x, sfa[0], sc & 0xFFu);
                mma4(c[i][0], av[1], w.y, sfa[1], (sc >> 8) & 0xFFu);
                mma4(c[i][1], av[0], w.z, sfa[0], (sc >> 16) & 0xFFu);
                mma4(c[i][1], av[1], w.w, sfa[1], sc >> 24);
              }
            }
        };
        const uint32_t none[NT] = {};
        load4c(ba, sa, p0, none);
        if (p0 + KU < p1) load4c(bb, sb, p0 + KU, none);
#pragma unroll
        for (int i = 0; i < NT; ++i)
          if (fl[i] & ((1u << 2 * KU) - 1u))
#pragma unroll
            for (int q = 0; q < 2 * KU; ++q)
              if (p0 + q < p1 && (fl[i] >> q & 1u)) {
                const uint32_t v = ldg_stream32(sr4[i] + (p0 + q) * 32);
                if (q < KU) sa[q][i] = v;
                else sb[q - KU][i] = v;
              }
        for (int p = p0; p < p1; p += 2 * KU) {
          comp4c(ba, sa, p);
          if (p + KU >= p1) break;
          if (p + 2 * KU < p1) load4c(ba, sa, p + 2 * KU, fl);
          comp4c(bb, sb, p + KU);
          if (p + 3 * KU < p1) load4c(bb, sb, p + 3 * KU, fl);
        }
      } else {
        load4(ba, sa, p0);
        for (int p = p0; p < p1; p += 2 * KU) {
          if (p + KU < p1) load4(bb, sb, p + KU);
          if constexpr (CC) comp4cc(ba, sa, p); else comp4(ba, sa, p);
          if (p + KU >= p1) break;
          if (p + 2 * KU < p1) load4(ba, sa, p + 2 * KU);
          if constexpr (CC) comp4cc(bb, sb, p + KU); else comp4(bb, sb, p + KU);
        }
      }
    }
    pre = false;

    // partials -> smem; rows 0..31 of the task are w1 (gate), 32..63 w3 (up)
    if (CC && !sh) {
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
          for (int m = 0; m < kMaxRows; ++m) {
            if (m >= M) break;
            float v = acc[i][r][m];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            if (t == 0) red[wi_k][m][16 * (wi_n * NT + i) + 8 * r + g] = v;
          }
    } else if (live) {
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = 16 * (wi_n * NT + i) + 8 * h + 2 * t;
          red[wi_k][g][r] = c[i][h][0];
          red[wi_k][g][r + 1] = c[i][h][1];
        }
    }
    __syncthreads();
    if (warp < M && (sh || (U.rows[u] >> warp & 1))) {
      const int m = warp;
      float gs = 0.f, us = 0.f;
#pragma unroll
      for (int k = 0; k < WK; ++k) gs += red[k][m][lane], us += red[k][m][32 + lane];
      float gt = __bfloat162float(__float2bfloat16_rn(gs)), up = __bfloat162float(__float2bfloat16_rn(us));
      up = fminf(fmaxf(up, -10.f), 10.f);
      gt = fminf(gt, 10.f);
      float h = gt / (1.0f + expf(-gt)) * up;
      if (!sh) h = U.rw[u][m] * h;
      h = __bfloat162float(__float2bfloat16_rn(h));
      const uint32_t e = act_scale_byte(warp_max(fabsf(h)));
      const int slot = sh ? kSlotSh : u;
      a.hq[((size_t)slot * kMaxRows + m) * kInter + 32 * jb + lane] = (uint8_t)quant_e4m3(h, e);
      if (lane == 0) a.hs[((size_t)slot * kMaxRows + m) * kJB + jb] = (uint8_t)e;
    }
    __syncthreads();
    task = a.ctr ? s_next[it & 1] : task + (int)gridDim.x;  // s_next[it & 1] is rewritten two tasks on
  }
  if (a.ctr && threadIdx.x == 0) {
    __threadfence();  // this CTA's claims before its exit count
    if (atomicAdd(a.ctr + 1, 1) == (int)gridDim.x - 1) atomicExch(a.ctr, 0), atomicExch(a.ctr + 1, 0);
  }
}

// moe_up for 9..16 rows: k_moe_up's tasks and weight loads, rows g and g + 8 in the MMAs' two halves; in
// the epilogue one warp per row, or two rows per warp when there are fewer warps. Tensor cores only.
// srowm: the scale row this lane gives the MMAs (row g + 8's for odd lanes), slive whether it is live.
// CS: the routed scales from their 1-bit code, as in k_moe_up.
template <int NT, int WK, int KU, bool CS = false>
__global__ void __launch_bounds__(32 * (4 / NT) * WK, 1) k_moe_up_r16(Params a) {
  static_assert(!CS || (kP1 % WK == 0 && kP1 / WK <= 32 && 2 * KU < 32), "no scale-code build");
  constexpr int RB = kMaxRows, WN = 4 / NT, KH = KU / 2 > 0 ? KU / 2 : 1;
  __shared__ Union<RB> U;
  __shared__ float red[WK][RB][64];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % WN, wi_k = warp / WN;
  const int p0 = kP1 * wi_k / WK, p1 = kP1 * (wi_k + 1) / WK;
  const int M = a.M;
  const bool live = g < M, live2 = g + 8 < M, odd = lane & 1, lo = t < 2, slive = odd ? live2 : live;
  const size_t r1 = live ? g : 0, r2 = live2 ? g + 8 : 0;
  const uint8_t* arow4 = a.xq + r1 * kDim + 8 * t;
  const uint8_t* arow4_2 = a.xq + r2 * kDim + 8 * t;
  const uint8_t* arow8 = a.xq + r1 * kDim + 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* arow8_2 = a.xq + r2 * kDim + 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* srowm = a.xs + (odd ? r2 : r1) * kKB1;

  // shared expert (FP8): this warp's tiles of task jb
  uint4 c8a[KH][NT][2], c8b[KH][NT][2];
  uint32_t s8a[KH][NT], s8b[KH][NT];
  const uint8_t* w8[NT][2];
  const uint8_t* sr8[NT];
  auto setup8 = [&](int jb) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
      const uint8_t* W = ti < 2 ? a.w.sh_w1 : a.w.sh_w3;
      w8[i][0] = W + (size_t)(2 * tile) * 8 * kDim + g * 64 + 16 * t;
      w8[i][1] = w8[i][0] + 8 * kDim;
      sr8[i] = (ti < 2 ? a.w.sh_s1 : a.w.sh_s3) + (size_t)jb * kKB1;
    }
  };
  auto load8 = [&](uint4 (&b)[KH][NT][2], uint32_t (&s)[KH][NT], int p) {
#pragma unroll
    for (int u = 0; u < KH; ++u)
      if (p + u < p1)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          b[u][i][0] = ldg_stream128(w8[i][0] + (p + u) * 512);
          b[u][i][1] = ldg_stream128(w8[i][1] + (p + u) * 512);
          s[u][i] = __ldg(reinterpret_cast<const uint16_t*>(sr8[i] + 2 * (p + u)));
        }
  };
  float c[NT][2][4];
  auto comp8 = [&](const uint4 (&b)[KH][NT][2], const uint32_t (&s)[KH][NT], int p) {
#pragma unroll
    for (int u = 0; u < KH; ++u)
      if (p + u < p1) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow8, arow8_2, srowm, p + u, live, live2, slive);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          mma8h(c[i][0], av, av2, sfa, b[u][i][0], s[u][i], lo);
          mma8h(c[i][1], av, av2, sfa, b[u][i][1], s[u][i], lo);
        }
      }
  };

  int task = blockIdx.x;
  bool pre = task < kJB;  // the first task is the shared expert's: its weights need nothing upstream
  if (pre) {
    setup8(task);
    load8(c8a, s8a, p0);
  }
  pdl_wait();
  build_union(U, a, M);
  if (a.logits && a.idx_out && blockIdx.x == 0 && threadIdx.x < M * kTopK)
    a.idx_out[threadIdx.x] = U.sel_id[threadIdx.x], a.wts_out[threadIdx.x] = U.sel_w[threadIdx.x];
  pdl_launch_dependents();
  const int ntask = kJB * (1 + U.n);

  __shared__ int s_next[2];  // k_moe_up's schedule
#pragma unroll 1
  for (int it = 0; task < ntask; ++it) {
    if (a.ctr && threadIdx.x == 0) s_next[it & 1] = (int)gridDim.x + atomicAdd(a.ctr, 1);
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) c[i][h][0] = c[i][h][1] = c[i][h][2] = c[i][h][3] = 0.f;
    const bool sh = task < kJB;
    const int u = sh ? 0 : task / kJB - 1, jb = task % kJB;
    if (sh) {
      if (!pre) {
        setup8(jb);
        load8(c8a, s8a, p0);
      }
      for (int p = p0; p < p1; p += 2 * KH) {
        if (p + KH < p1) load8(c8b, s8b, p + KH);
        comp8(c8a, s8a, p);
        if (p + KH >= p1) break;
        if (p + 2 * KH < p1) load8(c8a, s8a, p + 2 * KH);
        comp8(c8b, s8b, p + KH);
      }
    } else {
      const int e = U.id[u];
      const uint8_t* w4[NT];
      const uint8_t* sr4[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
        w4[i] = (ti < 2 ? a.w.w1 : a.w.w3) + (size_t)e * kW13 + (size_t)tile * (kP1 * 512) + lane * 16;
        sr4[i] = (ti < 2 ? a.w.s1 : a.w.s3) + (size_t)e * kS13 + (size_t)tile * (kP1 * 32) + g * 4;
      }
      uint4 ba[KU][NT], bb[KU][NT];
      uint32_t sa[KU][NT], sb[KU][NT];
      auto load4 = [&](uint4 (&b)[KU][NT], uint32_t (&s)[KU][NT], int p) {
#pragma unroll
        for (int q = 0; q < KU; ++q)
          if (p + q < p1)
#pragma unroll
            for (int i = 0; i < NT; ++i) {
              b[q][i] = ldg_stream128(w4[i] + (p + q) * 512);
              s[q][i] = ldg_stream32(sr4[i] + (p + q) * 32);
            }
      };
      auto comp4 = [&](const uint4 (&b)[KU][NT], const uint32_t (&s)[KU][NT], int p) {
#pragma unroll
        for (int q = 0; q < KU; ++q)
          if (p + q < p1) {
            uint2 av[2], av2[2];
            uint32_t sfa[2];
            act_frag2(av, av2, sfa, arow4, arow4_2, srowm, p + q, live, live2, slive);
#pragma unroll
            for (int i = 0; i < NT; ++i) {
              const uint4 w = b[q][i];
              const uint32_t sc = s[q][i];
              mma4h(c[i][0], av[0], av2[0], w.x, sfa[0], sc & 0xFFu);
              mma4h(c[i][0], av[1], av2[1], w.y, sfa[1], (sc >> 8) & 0xFFu);
              mma4h(c[i][1], av[0], av2[0], w.z, sfa[0], (sc >> 16) & 0xFFu);
              mma4h(c[i][1], av[1], av2[1], w.w, sfa[1], sc >> 24);
            }
          }
      };
      if constexpr (CS) {  // k_moe_up's
        const uint32_t* cw[NT];
        uint32_t fl[NT], b4[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
          const ScaleCode& k = ti < 2 ? a.w.c1 : a.w.c3;
          cw[i] = k.code + ((size_t)e * kT13 + tile) * kB13;
          b4[i] = k.base4;
          fl[i] = code_flags_up(cw[i], p0);
        }
        auto load4c = [&](uint4 (&b)[KU][NT], uint32_t (&s)[KU][NT], int p, const uint32_t (&f)[NT]) {
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1)
#pragma unroll
              for (int i = 0; i < NT; ++i) b[q][i] = ldg_stream128(w4[i] + (p + q) * 512);
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1)
#pragma unroll
              for (int i = 0; i < NT; ++i)
                s[q][i] = ldg_stream32_sel(f[i] >> (p - p0), 1u << q, sr4[i] + (p + q) * 32, cw[i] + p + q);
        };
        auto comp4c = [&](const uint4 (&b)[KU][NT], const uint32_t (&s)[KU][NT], int p) {
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1) {
              uint2 av[2], av2[2];
              uint32_t sfa[2];
              act_frag2(av, av2, sfa, arow4, arow4_2, srowm, p + q, live, live2, slive);
#pragma unroll
              for (int i = 0; i < NT; ++i) {
                const uint4 w = b[q][i];
                const uint32_t sc = fl[i] >> (p + q - p0) & 1u ? s[q][i] : ((s[q][i] >> g) & 0x01010101u) + b4[i];
                mma4h(c[i][0], av[0], av2[0], w.x, sfa[0], sc & 0xFFu);
                mma4h(c[i][0], av[1], av2[1], w.y, sfa[1], (sc >> 8) & 0xFFu);
                mma4h(c[i][1], av[0], av2[0], w.z, sfa[0], (sc >> 16) & 0xFFu);
                mma4h(c[i][1], av[1], av2[1], w.w, sfa[1], sc >> 24);
              }
            }
        };
        const uint32_t none[NT] = {};
        load4c(ba, sa, p0, none);
        if (p0 + KU < p1) load4c(bb, sb, p0 + KU, none);
#pragma unroll
        for (int i = 0; i < NT; ++i)
          if (fl[i] & ((1u << 2 * KU) - 1u))
#pragma unroll
            for (int q = 0; q < 2 * KU; ++q)
              if (p0 + q < p1 && (fl[i] >> q & 1u)) {
                const uint32_t v = ldg_stream32(sr4[i] + (p0 + q) * 32);
                if (q < KU) sa[q][i] = v;
                else sb[q - KU][i] = v;
              }
        for (int p = p0; p < p1; p += 2 * KU) {
          comp4c(ba, sa, p);
          if (p + KU >= p1) break;
          if (p + 2 * KU < p1) load4c(ba, sa, p + 2 * KU, fl);
          comp4c(bb, sb, p + KU);
          if (p + 3 * KU < p1) load4c(bb, sb, p + 3 * KU, fl);
        }
      } else {
        load4(ba, sa, p0);
        for (int p = p0; p < p1; p += 2 * KU) {
          if (p + KU < p1) load4(bb, sb, p + KU);
          comp4(ba, sa, p);
          if (p + KU >= p1) break;
          if (p + 2 * KU < p1) load4(ba, sa, p + 2 * KU);
          comp4(bb, sb, p + KU);
        }
      }
    }
    pre = false;

    // partials -> smem (rows g and g + 8); rows 0..31 of the task are w1 (gate), 32..63 w3 (up)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = 16 * (wi_n * NT + i) + 8 * h + 2 * t;
        if (live) red[wi_k][g][r] = c[i][h][0], red[wi_k][g][r + 1] = c[i][h][1];
        if (live2) red[wi_k][g + 8][r] = c[i][h][2], red[wi_k][g + 8][r + 1] = c[i][h][3];
      }
    __syncthreads();
    for (int m = warp; m < M; m += WN * WK) {
      if (!sh && !(U.rows[u] >> m & 1)) continue;
      float gs = 0.f, us = 0.f;
#pragma unroll
      for (int k = 0; k < WK; ++k) gs += red[k][m][lane], us += red[k][m][32 + lane];
      float gt = __bfloat162float(__float2bfloat16_rn(gs)), up = __bfloat162float(__float2bfloat16_rn(us));
      up = fminf(fmaxf(up, -10.f), 10.f);
      gt = fminf(gt, 10.f);
      float h = gt / (1.0f + expf(-gt)) * up;
      if (!sh) h = U.rw[u][m] * h;
      h = __bfloat162float(__float2bfloat16_rn(h));
      const uint32_t e = act_scale_byte(warp_max(fabsf(h)));
      const int slot = sh ? RB * kTopK : u;
      a.hq[((size_t)slot * RB + m) * kInter + 32 * jb + lane] = (uint8_t)quant_e4m3(h, e);
      if (lane == 0) a.hs[((size_t)slot * RB + m) * kJB + jb] = (uint8_t)e;
    }
    __syncthreads();
    task = a.ctr ? s_next[it & 1] : task + (int)gridDim.x;
  }
  if (a.ctr && threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(a.ctr + 1, 1) == (int)gridDim.x - 1) atomicExch(a.ctr, 0), atomicExch(a.ctr + 1, 0);
  }
}

// ---------------------------------------------------------------------------------------------
// moe_down. CTA = R tiles of 16 output rows. Units: (routed expert, tile) with w2 FP4-tiled (one
// uint4 per step), then the shared expert as (tile, 8-row half) units in the tile_fp8 layout; each
// unit is 9 steps. Units go round-robin to the W warps, outputs (bf16-rounded) to shared memory,
// then one fixed-order fold per output. CS: w2's scales from their 1-bit code (a.w.c2).
template <int W, int R, bool CS = false>
__global__ void __launch_bounds__(32 * W) k_moe_down(Params a) {
  constexpr int kMaxRows = kLowRows, kSlotSh = kLowRows * kTopK;  // up to 8 rows (k_moe_down_r16 above)
  extern __shared__ __nv_bfloat16 outs[];  // [U.n + 1][M][16R]
  __shared__ Union<kLowRows> U;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int M = a.M;
  const bool live = g < M, odd = lane & 1, lo = t < 2;
  const int tile0 = blockIdx.x * R;
  build_union(U, a, M);  // the router (gate) finished before moe_up let this grid start
  const int nr = U.n * R, nunits = nr + 2 * R;

  // CS: a routed unit's step takes its group's code word (scale_code) or, flagged raw, the raw group. A unit's
  // code words sit one a lane (x: lane p holds step p's; a shared unit's u16 scales likewise) and its first
  // four raw groups eight lanes each (y), shuffled out per step: two registers a buffer instead of nine (with
  // nine, the 512-thread variant spills right after its loads, stalling on them). The raw groups' loads need
  // the unit's flags: each warp reads its units' flags into shared memory first (before pdl_wait, while
  // moe_up finishes).
  __shared__ uint32_t sfl[CS ? kLowRows * kTopK * R : 1];
  if constexpr (CS) {
#pragma unroll 1
    for (int q = warp + W * lane; q < nr; q += 32 * W)
      sfl[q] = __ldg(a.w.c2.code + ((size_t)U.id[q / R] * kT2 + tile0 + q % R) * kB2 + kP2);
    __syncwarp();
  }
  auto load_cs = [&](uint4 (&b)[kP2], uint32_t& x, uint32_t& y, int q) {
    if (q < nr) {
      const int u = q / R, tile = tile0 + q % R;
      const size_t e = U.id[u];
      const uint8_t* w = a.w.w2 + e * kW2 + (size_t)tile * (kP2 * 512) + lane * 16;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512);
      code_load_down(a.w.c2.code + (e * kT2 + tile) * kB2, a.w.s2 + e * kS2 + (size_t)tile * (kP2 * 32), sfl[q], lane,
                     x, y);
    } else {
      const int qq = q - nr, tile = tile0 + qq / 2, nt = 2 * tile + (qq & 1);
      const uint8_t* w = a.w.sh_w2 + (size_t)nt * 8 * kInter + g * 64 + 16 * t;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512);
      x = __ldg(reinterpret_cast<const uint16_t*>(a.w.sh_s2 + (size_t)(tile / 2) * kKB2) + min(lane, kP2 - 1));
    }
  };
  auto comp_cs = [&](const uint4 (&b)[kP2], uint32_t x, uint32_t y, int q) {
    float c[2][4] = {};
    int slot, col;
    if (q < nr) {
      const int u = q / R;
      slot = u;
      col = 16 * (q % R);
      const uint8_t* arow = a.hq + ((size_t)u * kMaxRows + (live ? g : 0)) * kInter + 8 * t;
      const uint8_t* srow = a.hs + ((size_t)u * kMaxRows + (live ? g : 0)) * kJB;
      code_steps_down(sfl[q], x, y, g, a.w.c2.base4,
                      a.w.s2 + (size_t)U.id[u] * kS2 + (size_t)(tile0 + q % R) * (kP2 * 32), [&](int p, uint32_t sc) {
                        uint2 av[2];
                        uint32_t sfa[2];
                        act_frag(av, sfa, arow, srow, p, live, odd);
                        mma4(c[0], av[0], b[p].x, sfa[0], sc & 0xFFu);
                        mma4(c[0], av[1], b[p].y, sfa[1], (sc >> 8) & 0xFFu);
                        mma4(c[1], av[0], b[p].z, sfa[0], (sc >> 16) & 0xFFu);
                        mma4(c[1], av[1], b[p].w, sfa[1], sc >> 24);
                      });
    } else {
      const int qq = q - nr;
      slot = U.n;
      col = 16 * (qq / 2) + 8 * (qq & 1);
      const uint8_t* arow =
          a.hq + ((size_t)kSlotSh * kMaxRows + (live ? g : 0)) * kInter + 16 * (t & 1) + 8 * (t >> 1);
      const uint8_t* srow = a.hs + ((size_t)kSlotSh * kMaxRows + (live ? g : 0)) * kJB;
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2];
        uint32_t sfa[2];
        act_frag(av, sfa, arow, srow, p, live, odd);
        mma8(c[0], av, sfa, b[p], __shfl_sync(~0u, x, p), lo);
      }
    }
    if (live) {
      __nv_bfloat16* o = outs + ((size_t)slot * M + g) * (16 * R) + col + 2 * t;
      o[0] = __float2bfloat16_rn(c[0][0]);
      o[1] = __float2bfloat16_rn(c[0][1]);
      if (q < nr) {
        o[8] = __float2bfloat16_rn(c[1][0]);
        o[9] = __float2bfloat16_rn(c[1][1]);
      }
    }
  };
  auto load = [&](uint4 (&b)[kP2], uint32_t (&s)[kP2], int q) {
    if (q < nr) {
      const int u = q / R, tile = tile0 + q % R;
      const size_t e = U.id[u];
      const uint8_t* w = a.w.w2 + e * kW2 + (size_t)tile * (kP2 * 512) + lane * 16;
      const uint8_t* sr = a.w.s2 + e * kS2 + (size_t)tile * (kP2 * 32) + g * 4;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512), s[p] = ldg_stream32(sr + p * 32);
    } else {
      const int qq = q - nr, tile = tile0 + qq / 2, nt = 2 * tile + (qq & 1);
      const uint8_t* w = a.w.sh_w2 + (size_t)nt * 8 * kInter + g * 64 + 16 * t;
      const uint8_t* sr = a.w.sh_s2 + (size_t)(tile / 2) * kKB2;
#pragma unroll
      for (int p = 0; p < kP2; ++p)
        b[p] = ldg_stream128(w + p * 512), s[p] = __ldg(reinterpret_cast<const uint16_t*>(sr + 2 * p));
    }
  };
  auto comp = [&](const uint4 (&b)[kP2], const uint32_t (&s)[kP2], int q) {
    float c[2][4] = {};
    int slot, col;
    if (q < nr) {
      const int u = q / R;
      slot = u;
      col = 16 * (q % R);
      const uint8_t* arow = a.hq + ((size_t)u * kMaxRows + (live ? g : 0)) * kInter + 8 * t;
      const uint8_t* srow = a.hs + ((size_t)u * kMaxRows + (live ? g : 0)) * kJB;
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2];
        uint32_t sfa[2];
        act_frag(av, sfa, arow, srow, p, live, odd);
        mma4(c[0], av[0], b[p].x, sfa[0], s[p] & 0xFFu);
        mma4(c[0], av[1], b[p].y, sfa[1], (s[p] >> 8) & 0xFFu);
        mma4(c[1], av[0], b[p].z, sfa[0], (s[p] >> 16) & 0xFFu);
        mma4(c[1], av[1], b[p].w, sfa[1], s[p] >> 24);
      }
    } else {
      const int qq = q - nr;
      slot = U.n;
      col = 16 * (qq / 2) + 8 * (qq & 1);
      const uint8_t* arow =
          a.hq + ((size_t)kSlotSh * kMaxRows + (live ? g : 0)) * kInter + 16 * (t & 1) + 8 * (t >> 1);
      const uint8_t* srow = a.hs + ((size_t)kSlotSh * kMaxRows + (live ? g : 0)) * kJB;
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2];
        uint32_t sfa[2];
        act_frag(av, sfa, arow, srow, p, live, odd);
        mma8(c[0], av, sfa, b[p], s[p], lo);
      }
    }
    if (live) {
      __nv_bfloat16* o = outs + ((size_t)slot * M + g) * (16 * R) + col + 2 * t;
      o[0] = __float2bfloat16_rn(c[0][0]);
      o[1] = __float2bfloat16_rn(c[0][1]);
      if (q < nr) {
        o[8] = __float2bfloat16_rn(c[1][0]);
        o[9] = __float2bfloat16_rn(c[1][1]);
      }
    }
  };

  uint4 ba[kP2], bb[kP2];
  int q = warp;
  if constexpr (CS) {
    uint32_t xa = 0, ya = 0, xb = 0, yb = 0;
    if (q < nunits) load_cs(ba, xa, ya, q);
    pdl_wait();
    pdl_launch_dependents();
    for (; q < nunits; q += 2 * W) {
      if (q + W < nunits) load_cs(bb, xb, yb, q + W);
      comp_cs(ba, xa, ya, q);
      if (q + W >= nunits) break;
      if (q + 2 * W < nunits) load_cs(ba, xa, ya, q + 2 * W);
      comp_cs(bb, xb, yb, q + W);
    }
  } else {
    uint32_t sa[kP2], sb[kP2];
    if (q < nunits) load(ba, sa, q);
    pdl_wait();
    pdl_launch_dependents();
    for (; q < nunits; q += 2 * W) {
      if (q + W < nunits) load(bb, sb, q + W);
      comp(ba, sa, q);
      if (q + W >= nunits) break;
      if (q + 2 * W < nunits) load(ba, sa, q + 2 * W);
      comp(bb, sb, q + W);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < M * 16 * R; i += blockDim.x) {
    const int m = i / (16 * R), r = i % (16 * R);
    float v = 0.f;
    for (int u = 0; u < U.n; ++u)
      if (U.rows[u] >> m & 1) v += __bfloat162float(outs[((size_t)u * M + m) * (16 * R) + r]);
    v += __bfloat162float(outs[((size_t)U.n * M + m) * (16 * R) + r]);
    a.y[(size_t)m * kDim + 16 * tile0 + r] = v;
  }
}

// moe_down for 9..16 rows: k_moe_down's units (the MMAs' upper half carrying rows 8..15) in rounds of W,
// one per warp, whose outputs (bf16) meet in shared memory and are folded into fp32 sums (one or two
// per thread) in unit order, i.e. in expert-id order with the shared expert last: k_moe_down's fold,
// so the same bits, without keeping every unit's outputs (97 experts x 16 rows would not fit). CS: w2's
// scales from their 1-bit code, as in k_moe_down.
template <int W, int R, bool CS = false>
__global__ void __launch_bounds__(32 * W, 1) k_moe_down_r16(Params a) {
  constexpr int RB = kMaxRows, kOut = (RB * 16 * R + 32 * W - 1) / (32 * W);  // outputs per thread
  static_assert(W % R == 0, "a round holds whole experts");
  __shared__ __nv_bfloat16 outs[2][W][RB][16];  // a round's outputs: [its units][row][the unit's tile's columns]
  __shared__ Union<RB> U;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int M = a.M;
  const bool live = g < M, live2 = g + 8 < M, odd = lane & 1, lo = t < 2, slive = odd ? live2 : live;
  const int tile0 = blockIdx.x * R;
  build_union(U, a, M);  // the router (gate) finished before moe_up let this grid start
  const int nr = U.n * R, nunits = nr + 2 * R, nround = (nunits + W - 1) / W;
  __shared__ uint32_t sfl[CS ? RB * kTopK * R : 1];  // CS: k_moe_down's flags
  if constexpr (CS) {
#pragma unroll 1
    for (int q = warp + W * lane; q < nr; q += 32 * W)
      sfl[q] = __ldg(a.w.c2.code + ((size_t)U.id[q / R] * kT2 + tile0 + q % R) * kB2 + kP2);
    __syncwarp();
  }

  auto load = [&](uint4 (&b)[kP2], uint32_t (&s)[kP2], int q) {
    if (q < nr) {
      const int u = q / R, tile = tile0 + q % R;
      const size_t e = U.id[u];
      const uint8_t* w = a.w.w2 + e * kW2 + (size_t)tile * (kP2 * 512) + lane * 16;
      const uint8_t* sr = a.w.s2 + e * kS2 + (size_t)tile * (kP2 * 32) + g * 4;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512), s[p] = ldg_stream32(sr + p * 32);
    } else {
      const int qq = q - nr, tile = tile0 + qq / 2, nt = 2 * tile + (qq & 1);
      const uint8_t* w = a.w.sh_w2 + (size_t)nt * 8 * kInter + g * 64 + 16 * t;
      const uint8_t* sr = a.w.sh_s2 + (size_t)(tile / 2) * kKB2;
#pragma unroll
      for (int p = 0; p < kP2; ++p)
        b[p] = ldg_stream128(w + p * 512), s[p] = __ldg(reinterpret_cast<const uint16_t*>(sr + 2 * p));
    }
  };
  // unit q's outputs (bf16) into o [row][column of its tile]
  auto comp = [&](const uint4 (&b)[kP2], const uint32_t (&s)[kP2], int q, __nv_bfloat16 (&o)[RB][16]) {
    float c[2][4] = {};
    const bool sh = q >= nr;
    const int slot = sh ? RB * kTopK : q / R;
    const uint8_t* base = a.hq + (size_t)slot * RB * kInter + (sh ? 16 * (t & 1) + 8 * (t >> 1) : 8 * t);
    const uint8_t* arow = base + (size_t)(live ? g : 0) * kInter;
    const uint8_t* arow2 = base + (size_t)(live2 ? g + 8 : 0) * kInter;
    const uint8_t* srowm = a.hs + ((size_t)slot * RB + (odd ? (live2 ? g + 8 : 0) : (live ? g : 0))) * kJB;
    if (!sh) {
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
        mma4h(c[0], av[0], av2[0], b[p].x, sfa[0], s[p] & 0xFFu);
        mma4h(c[0], av[1], av2[1], b[p].y, sfa[1], (s[p] >> 8) & 0xFFu);
        mma4h(c[1], av[0], av2[0], b[p].z, sfa[0], (s[p] >> 16) & 0xFFu);
        mma4h(c[1], av[1], av2[1], b[p].w, sfa[1], s[p] >> 24);
      }
    } else {
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
        mma8h(c[0], av, av2, sfa, b[p], s[p], lo);
      }
    }
    const int col = sh ? 8 * ((q - nr) & 1) + 2 * t : 2 * t;  // the shared expert: one 8-row half per unit
    if (live) {
      o[g][col] = __float2bfloat16_rn(c[0][0]), o[g][col + 1] = __float2bfloat16_rn(c[0][1]);
      if (!sh) o[g][col + 8] = __float2bfloat16_rn(c[1][0]), o[g][col + 9] = __float2bfloat16_rn(c[1][1]);
    }
    if (live2) {
      o[g + 8][col] = __float2bfloat16_rn(c[0][2]), o[g + 8][col + 1] = __float2bfloat16_rn(c[0][3]);
      if (!sh) o[g + 8][col + 8] = __float2bfloat16_rn(c[1][2]), o[g + 8][col + 9] = __float2bfloat16_rn(c[1][3]);
    }
  };
  auto load_cs = [&](uint4 (&b)[kP2], uint32_t& x, uint32_t& y, int q) {
    if (q < nr) {
      const int u = q / R, tile = tile0 + q % R;
      const size_t e = U.id[u];
      const uint8_t* w = a.w.w2 + e * kW2 + (size_t)tile * (kP2 * 512) + lane * 16;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512);
      code_load_down(a.w.c2.code + (e * kT2 + tile) * kB2, a.w.s2 + e * kS2 + (size_t)tile * (kP2 * 32), sfl[q], lane,
                     x, y);
    } else {
      const int qq = q - nr, tile = tile0 + qq / 2, nt = 2 * tile + (qq & 1);
      const uint8_t* w = a.w.sh_w2 + (size_t)nt * 8 * kInter + g * 64 + 16 * t;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512);
      x = __ldg(reinterpret_cast<const uint16_t*>(a.w.sh_s2 + (size_t)(tile / 2) * kKB2) + min(lane, kP2 - 1));
    }
  };
  auto comp_cs = [&](const uint4 (&b)[kP2], uint32_t x, uint32_t y, int q, __nv_bfloat16 (&o)[RB][16]) {
    float c[2][4] = {};
    const bool sh = q >= nr;
    const int slot = sh ? RB * kTopK : q / R;
    const uint8_t* base = a.hq + (size_t)slot * RB * kInter + (sh ? 16 * (t & 1) + 8 * (t >> 1) : 8 * t);
    const uint8_t* arow = base + (size_t)(live ? g : 0) * kInter;
    const uint8_t* arow2 = base + (size_t)(live2 ? g + 8 : 0) * kInter;
    const uint8_t* srowm = a.hs + ((size_t)slot * RB + (odd ? (live2 ? g + 8 : 0) : (live ? g : 0))) * kJB;
    if (!sh) {
      code_steps_down(sfl[q], x, y, g, a.w.c2.base4,
                      a.w.s2 + (size_t)U.id[q / R] * kS2 + (size_t)(tile0 + q % R) * (kP2 * 32), [&](int p, uint32_t sc) {
                        uint2 av[2], av2[2];
                        uint32_t sfa[2];
                        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
                        mma4h(c[0], av[0], av2[0], b[p].x, sfa[0], sc & 0xFFu);
                        mma4h(c[0], av[1], av2[1], b[p].y, sfa[1], (sc >> 8) & 0xFFu);
                        mma4h(c[1], av[0], av2[0], b[p].z, sfa[0], (sc >> 16) & 0xFFu);
                        mma4h(c[1], av[1], av2[1], b[p].w, sfa[1], sc >> 24);
                      });
    } else {
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
        mma8h(c[0], av, av2, sfa, b[p], __shfl_sync(~0u, x, p), lo);
      }
    }
    const int col = sh ? 8 * ((q - nr) & 1) + 2 * t : 2 * t;
    if (live) {
      o[g][col] = __float2bfloat16_rn(c[0][0]), o[g][col + 1] = __float2bfloat16_rn(c[0][1]);
      if (!sh) o[g][col + 8] = __float2bfloat16_rn(c[1][0]), o[g][col + 9] = __float2bfloat16_rn(c[1][1]);
    }
    if (live2) {
      o[g + 8][col] = __float2bfloat16_rn(c[0][2]), o[g + 8][col + 1] = __float2bfloat16_rn(c[0][3]);
      if (!sh) o[g + 8][col + 8] = __float2bfloat16_rn(c[1][2]), o[g + 8][col + 9] = __float2bfloat16_rn(c[1][3]);
    }
  };
  // round k's units folded into this thread's sums: per output, its tile's routed units of the round in
  // order (rows that picked the expert), then the shared expert's unit if it is in the round
  float v[kOut];
#pragma unroll
  for (int j = 0; j < kOut; ++j) v[j] = 0.f;
  auto fold = [&](int k, const __nv_bfloat16 (&ob)[W][RB][16]) {
#pragma unroll
    for (int j = 0; j < kOut; ++j) {
      const int o = threadIdx.x + 32 * W * j;
      if (o >= M * 16 * R) break;
      const int m = o / (16 * R), col = o % (16 * R), tile = col / 16, c16 = col % 16;
      float x = v[j];
      for (int jj = tile; jj < W; jj += R) {
        const int q = k * W + jj;
        if (q >= nr) break;
        if (U.rows[q / R] >> m & 1) x += __bfloat162float(ob[jj][m][c16]);
      }
      const int js = nr + 2 * tile + (c16 >> 3) - k * W;
      if (js >= 0 && js < W) x += __bfloat162float(ob[js][m][c16]);
      v[j] = x;
    }
  };

  uint4 ba[kP2], bb[kP2];
  int q = warp;
  if constexpr (CS) {
    uint32_t xa = 0, ya = 0, xb = 0, yb = 0;
    if (q < nunits) load_cs(ba, xa, ya, q);
    pdl_wait();
    pdl_launch_dependents();
    for (int k = 0; k < nround; k += 2, q += 2 * W) {
      if (q + W < nunits) load_cs(bb, xb, yb, q + W);
      if (q < nunits) comp_cs(ba, xa, ya, q, outs[0][warp]);
      __syncthreads();
      fold(k, outs[0]);
      if (k + 1 >= nround) break;
      if (q + 2 * W < nunits) load_cs(ba, xa, ya, q + 2 * W);
      if (q + W < nunits) comp_cs(bb, xb, yb, q + W, outs[1][warp]);
      __syncthreads();
      fold(k + 1, outs[1]);
    }
  } else {
    uint32_t sa[kP2], sb[kP2];
    if (q < nunits) load(ba, sa, q);
    pdl_wait();
    pdl_launch_dependents();
    for (int k = 0; k < nround; k += 2, q += 2 * W) {
      if (q + W < nunits) load(bb, sb, q + W);
      if (q < nunits) comp(ba, sa, q, outs[0][warp]);
      __syncthreads();
      fold(k, outs[0]);
      if (k + 1 >= nround) break;
      if (q + 2 * W < nunits) load(ba, sa, q + 2 * W);
      if (q + W < nunits) comp(bb, sb, q + W, outs[1][warp]);
      __syncthreads();  // also: every thread's fold of round k is done before round k + 2 rewrites outs[0]
      fold(k + 1, outs[1]);
    }
  }
#pragma unroll
  for (int j = 0; j < kOut; ++j) {
    const int o = threadIdx.x + 32 * W * j;
    if (o >= M * 16 * R) break;
    a.y[(size_t)(o / (16 * R)) * kDim + 16 * tile0 + o % (16 * R)] = v[j];
  }
}

// natural [N][K/2] packed e2m1 + [N][K/32] scales -> [N/16][K/64][32 lanes][16 B] + [N/16][K/64][8][4 B]
__global__ void k_tile_fp4(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s, uint8_t* __restrict__ wt,
                           uint8_t* __restrict__ st, int N, int K) {
  const int KB = K / 32, P = K / 64;
  const size_t n = (size_t)N / 16 * P * 32;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int lane = i % 32, g = lane / 4, t = lane % 4;
    const size_t p = i / 32 % P, n16 = i / 32 / P;
    uint32_t v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const size_t row = n16 * 16 + g + 8 * (j >> 1), blk = 2 * p + (j & 1);
      v[j] = *reinterpret_cast<const uint32_t*>(w + row * (K / 2) + blk * 16 + 4 * t);
    }
    reinterpret_cast<uint4*>(wt)[i] = make_uint4(v[0], v[1], v[2], v[3]);
    if (t == 0) {
      uint32_t sv = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const size_t row = n16 * 16 + g + 8 * (j >> 1), blk = 2 * p + (j & 1);
        sv |= (uint32_t)s[row * KB + blk] << (8 * j);
      }
      reinterpret_cast<uint32_t*>(st)[(n16 * P + p) * 8 + g] = sv;
    }
  }
}

// scale_code: the values' histogram (n % 4 == 0)
__global__ void k_scale_hist(const uint8_t* __restrict__ s, size_t n, unsigned long long* __restrict__ hist) {
  __shared__ unsigned int h[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
  __syncthreads();
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n / 4; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t v = reinterpret_cast<const uint32_t*>(s)[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) atomicAdd(&h[v >> 8 * j & 0xFFu], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    if (h[i]) atomicAdd(&hist[i], (unsigned long long)h[i]);
}

// scale_code: thread = (tile, flag word f): the code words of steps 32f .. 32f + 31 of an L3 tile [P][8 g][4 j]
// and their flags; code zeroed before. coded: the groups in {base, base + 1}.
__global__ void k_scale_code(const uint8_t* __restrict__ s, size_t tiles, int P, uint32_t base, uint32_t* __restrict__ code,
                             unsigned long long* __restrict__ coded) {
  const int FW = (P + 31) / 32;
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  unsigned n = 0;
  if (i < tiles * FW) {
    const size_t t = i / FW;
    const int f = (int)(i % FW);
    uint32_t* blk = code + t * code_block_words(P);
    uint32_t fl = 0;
    for (int p = 32 * f; p < P && p < 32 * f + 32; ++p) {
      const uint4* gp = reinterpret_cast<const uint4*>(s + (t * P + p) * 32);
      const uint4 x = gp[0], y = gp[1];
      const uint32_t wv[8] = {x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w};
      uint32_t cw = 0;
      bool in = true;
#pragma unroll
      for (int g = 0; g < 8; ++g)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t v = wv[g] >> 8 * j & 0xFFu;
          in = in && (v == base || v == base + 1);
          cw |= (v - base & 1u) << (8 * j + g);
        }
      if (in) blk[p] = cw, ++n;
      else fl |= 1u << (p - 32 * f);
    }
    blk[P + f] = fl;
  }
  n += __shfl_xor_sync(0xffffffffu, n, 16);
  n += __shfl_xor_sync(0xffffffffu, n, 8);
  n += __shfl_xor_sync(0xffffffffu, n, 4);
  n += __shfl_xor_sync(0xffffffffu, n, 2);
  n += __shfl_xor_sync(0xffffffffu, n, 1);
  if ((threadIdx.x & 31) == 0 && n) atomicAdd(coded, (unsigned long long)n);
}

template <class F>
int resident_ctas(F kernel, int threads, size_t smem) {
  int n = 0, sms = 0, dev = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, smem));
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return n * sms;
}

template <int NT, int WK, int KU, bool CC = false, int RB = kLowRows>
void launch_up(const Params& p, const MoeCfg& cfg, cudaStream_t st) {
  constexpr int threads = 32 * (4 / NT) * WK;
  // the builds that read the scale code (when the weights carry it): the engine's, variant 0 up to 8 rows
  // and 4 above (up16)
  constexpr bool kCs = !CC && (RB == kLowRows ? NT == 1 && WK == 4 && KU == 4 : NT == 2 && WK == 4 && KU == 2);
  constexpr auto kernel = [] {
    if constexpr (RB == kLowRows) return k_moe_up<NT, WK, KU, CC>;
    else return k_moe_up_r16<NT, WK, KU>;
  }();
  constexpr auto kernel_cs = [] {
    if constexpr (!kCs) return kernel;
    else if constexpr (RB == kLowRows) return k_moe_up<NT, WK, KU, CC, true>;
    else return k_moe_up_r16<NT, WK, KU, true>;
  }();
  const bool cs = kCs && p.w.c1.code && p.w.c3.code;
  static const int resident = resident_ctas(kernel, threads, 0);
  static const int resident_cs = resident_ctas(kernel_cs, threads, 0);
  const int max_tasks = kJB * (1 + kTopK * p.M);
  int grid = cfg.up_grid > 0 ? cfg.up_grid : cs ? resident_cs : resident;
  grid = grid < max_tasks ? grid : max_tasks;
  launch_pdl(cs ? kernel_cs : kernel, dim3(grid), dim3(threads), 0, st, cfg.pdl, p);
}

template <int W, int R, int RB = kLowRows>
void launch_down(const Params& p, const MoeCfg& cfg, cudaStream_t st) {
  if constexpr (RB == kLowRows) {
    // the builds that read the scale code: the engine's (variant 2 at one row, 4 above)
    constexpr bool kCs = (W == 4 && R == 1) || (W == 16 && R == 4);
    constexpr auto kernel_cs = [] {
      if constexpr (kCs) return k_moe_down<W, R, true>;
      else return k_moe_down<W, R>;
    }();
    const size_t smem = (size_t)(kTopK * p.M + 1) * p.M * 16 * R * sizeof(__nv_bfloat16);
    static bool attr = false;
    if (!attr) {
      const int most = (kLowRows * kTopK + 1) * kLowRows * 16 * R * sizeof(__nv_bfloat16);
      CUDA_CHECK(cudaFuncSetAttribute(k_moe_down<W, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, most));
      CUDA_CHECK(cudaFuncSetAttribute(kernel_cs, cudaFuncAttributeMaxDynamicSharedMemorySize, most));
      attr = true;
    }
    launch_pdl(kCs && p.w.c2.code ? kernel_cs : k_moe_down<W, R>, dim3(kDim / 16 / R), dim3(32 * W), smem, st, cfg.pdl,
               p);
  } else {
    constexpr bool kCs = W == 8 && R == 4;  // the engine's (down16 2)
    constexpr auto kernel_cs = [] {
      if constexpr (kCs) return k_moe_down_r16<W, R, true>;
      else return k_moe_down_r16<W, R>;
    }();
    launch_pdl(kCs && p.w.c2.code ? kernel_cs : k_moe_down_r16<W, R>, dim3(kDim / 16 / R), dim3(32 * W), 0, st, cfg.pdl,
               p);
  }
}

}  // namespace

void tile_fp4(const uint8_t* w, const uint8_t* s, uint8_t* wt, uint8_t* st_, int N, int K, cudaStream_t st) {
  if (N % 16 || K % 64) die("tile_fp4: N=%d K=%d", N, K);
  k_tile_fp4<<<1024, 256, 0, st>>>(w, s, wt, st_, N, K);
  CUDA_CHECK(cudaGetLastError());
}

size_t scale_code_words(int E, int N, int KB) { return (size_t)E * (N / 16) * code_block_words(KB / 2); }

double scale_code(const uint8_t* s, int E, int N, int KB, uint32_t* code, int* base, cudaStream_t st) {
  if (E < 1 || N % 16 || KB % 2) die("scale_code: E=%d N=%d KB=%d", E, N, KB);
  const size_t n = (size_t)E * N * KB, tiles = (size_t)E * (N / 16);
  const int P = KB / 2;
  unsigned long long* d;  // [256] the histogram, [256] the groups coded
  CUDA_CHECK(cudaMalloc(&d, 257 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemsetAsync(d, 0, 257 * sizeof(unsigned long long), st));
  k_scale_hist<<<256, 256, 0, st>>>(s, n, d);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long h[257];
  CUDA_CHECK(cudaMemcpyAsync(h, d, 256 * sizeof(unsigned long long), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  int b0 = 0;
  for (int v = 1; v < 256; ++v)
    if (h[v] > h[b0]) b0 = v;
  const int b = b0 > 0 && h[b0 - 1] > (b0 < 255 ? h[b0 + 1] : 0ull) ? b0 - 1 : b0;  // {b, b + 1}: wfix's pair
  CUDA_CHECK(cudaMemsetAsync(code, 0, scale_code_words(E, N, KB) * 4, st));
  const size_t threads = tiles * ((P + 31) / 32);
  k_scale_code<<<(unsigned)((threads + 255) / 256), 256, 0, st>>>(s, tiles, P, (uint32_t)b, code, d + 256);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(h + 256, d + 256, sizeof(unsigned long long), cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaFree(d));
  *base = b;
  return (double)h[256] / (double)(tiles * P);
}

size_t moe_workspace_bytes() { return kCtrOff + 2 * sizeof(int); }

namespace {

// 9..16 rows: MoeCfg::up16/down16 pick from these (tensor cores; the builds that keep to their registers)
void run_moe16(const Params& p, MoeCfg cfg, cudaStream_t st) {
  if (cfg.phases & 1) switch (cfg.up16) {
    case 0: launch_up<1, 4, 2, false, kMaxRows>(p, cfg, st); break;
    case 1: launch_up<2, 4, 4, false, kMaxRows>(p, cfg, st); break;
    case 2: launch_up<4, 8, 2, false, kMaxRows>(p, cfg, st); break;
    case 3: launch_up<2, 8, 1, false, kMaxRows>(p, cfg, st); break;
    case 4: launch_up<2, 4, 2, false, kMaxRows>(p, cfg, st); break;
    default: die("moe: up16 variant %d", cfg.up16);
  }
  if (cfg.phases & 2) switch (cfg.down16) {
    case 0: launch_down<16, 2, kMaxRows>(p, cfg, st); break;
    case 1: launch_down<8, 2, kMaxRows>(p, cfg, st); break;
    case 2: launch_down<8, 4, kMaxRows>(p, cfg, st); break;
    case 3: launch_down<8, 1, kMaxRows>(p, cfg, st); break;
    case 4: launch_down<4, 1, kMaxRows>(p, cfg, st); break;
    default: die("moe: down16 variant %d", cfg.down16);
  }
}

void run_moe(const Params& p, MoeCfg cfg, cudaStream_t st) {
  if (p.M > kLowRows) return run_moe16(p, cfg, st);
  if (cfg.phases & 1) switch (cfg.up) {
    case 0: launch_up<1, 4, 4>(p, cfg, st); break;
    case 1: launch_up<2, 4, 4>(p, cfg, st); break;
    case 2: launch_up<1, 8, 4>(p, cfg, st); break;
    case 3: launch_up<2, 8, 2>(p, cfg, st); break;
    case 4: launch_up<4, 8, 2>(p, cfg, st); break;
    case 5: launch_up<1, 4, 8>(p, cfg, st); break;
    case 6: launch_up<1, 4, 2, true>(p, cfg, st); break;  // CUDA cores
    case 7: launch_up<2, 8, 1, true>(p, cfg, st); break;
    case 8: launch_up<4, 8, 1, true>(p, cfg, st); break;
    case 9: launch_up<1, 8, 2, true>(p, cfg, st); break;
    default: die("moe: up variant %d", cfg.up);
  }
  if (cfg.phases & 2) switch (cfg.down) {
    case 0: launch_down<8, 1>(p, cfg, st); break;
    case 1: launch_down<8, 2>(p, cfg, st); break;
    case 2: launch_down<4, 1>(p, cfg, st); break;
    case 3: launch_down<16, 2>(p, cfg, st); break;
    case 4: launch_down<16, 4>(p, cfg, st); break;
    default: die("moe: down variant %d", cfg.down);
  }
}

}  // namespace

void moe(const uint8_t* xq, const uint8_t* xs, int M, const int* idx, const float* wts, const MoeWeights& w, void* ws,
         float* y, MoeCfg cfg, cudaStream_t st) {
  if (M < 1 || M > kMaxRows) die("moe: M=%d", M);
  uint8_t* hq = static_cast<uint8_t*>(ws);
  Params p{xq, xs, idx, wts, M, w, hq, hq + moe_hs_offset(M), y};
  p.ctr = cfg.dyn ? reinterpret_cast<int*>(hq + kCtrOff) : nullptr;
  run_moe(p, cfg, st);
}

void moe_routed(const uint8_t* xq, const uint8_t* xs, int M, const float* logits, const float* bias, int E,
                float route_scale, const MoeWeights& w, void* ws, float* y, MoeCfg cfg, cudaStream_t st, int* idx_out,
                float* wts_out) {
  if (M < 1 || M > kMaxRows || E < kTopK || E > 32 * kMaskWords) die("moe_routed: M=%d E=%d", M, E);
  if (!idx_out != !wts_out) die("moe_routed: idx_out and wts_out go together");
  uint8_t* hq = static_cast<uint8_t*>(ws);
  Params p{xq, xs, nullptr, nullptr, M, w, hq, hq + moe_hs_offset(M), y,
           logits, bias, E, route_scale, idx_out, wts_out};
  p.ctr = cfg.dyn ? reinterpret_cast<int*>(hq + kCtrOff) : nullptr;
  run_moe(p, cfg, st);
}

}  // namespace dsk
