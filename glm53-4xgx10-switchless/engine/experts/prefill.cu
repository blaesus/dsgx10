// The prefill expert GEMMs (docs/experts.md, "Prefill"): a chunk's picks bucketed by expert, gate/up and down
// as grouped GEMMs on the bf16 tensor cores (m16n8k16, f32 accumulate), the weights dequantised into shared
// memory in the B-fragment order of the decode kernels (the same decode functions, so the same exact bf16
// values), each pick's down output kept in f32 and combined per row in expert-id order with the decode step's
// arithmetic: so a row's result equals the decode path's up to the order of the K sums. The shared expert as a
// dense FP8 GEMM over the chunk.
//
// Tiles: 128 picks of one expert (the bucket's order; positions come from atomics, which changes nothing a row
// sees) x 64 channels of gate and of up (8 column blocks) / x 128 outputs of down. 8 warps as 2 (rows) x 4
// (columns), a warp 64 rows x 32 columns (gate/up: 16 channels of each, so SwiGLU runs in its registers).
// Each k-step of 64: A (the gathered activation rows) by cp.async into a swizzled tile, B decoded by the
// warp that owns two of the 16 strips from registers loaded a whole block (512 k) ahead; two stages.
#include <algorithm>

#include "common/check.h"
#include "experts/formats.cuh"

namespace gx {

namespace {

constexpr int kTM = 128;                         // rows a tile
constexpr int kMaxPicks = kMaxChunk * kTopK;     // 65536
constexpr int kMaxRTiles = kMaxPicks / kTM + kE;  // routed tiles: sum of ceil(count / 128)
constexpr int kUpNB = kI / 64;                   // gate/up column blocks (64 channels of each)
constexpr int kStageA = kTM * 128;               // bytes: 128 rows x 64 bf16
constexpr int kStageB = 16 * 32 * 32;            // bytes: 16 strips x 32 lanes x 8 words
constexpr int kNA = 3;                           // A stages (B: 2)
constexpr int kSmem = kNA * kStageA + 2 * kStageB;  // 80 KB

struct PTile { int e, pos0, n, pad; };

// The workspace of a chunk of T rows. cnt is zero between calls (the scan clears it).
struct Ws {
  int *cnt, *cur, *off, *ntile;
  PTile* tiles;
  int *perm, *rpos, *re;
  float* rw;
  __nv_bfloat16 *h, *hsh;
  uint8_t *xq, *xs, *hq, *hs;  // FP4 path: NVFP4 activations (values, scales) of x [T] and of h [P]
  float* yp;
  int npass, cp;
  // the chained down: per position the pick's rank in its row's expert-id order and its routing weight; the
  // rows' tickets per 32 columns [kH / 32][kMaxChunk] (zero between calls: the last pick of a row
  // resets them) and the task and exit counters (zero between calls), both at fixed offsets, so that calls
  // of any T share them
  int *prank, *tick, *dq;
  float* pw;
  size_t bytes;
};
Ws ws_layout(void* base, int T) {
  Ws w{};
  uint8_t* p = static_cast<uint8_t*>(base);
  size_t o = 0;
  auto take = [&](size_t b) { uint8_t* r = p ? p + o : nullptr; o += (b + 255) / 256 * 256; return r; };
  const size_t P = (size_t)T * kTopK;
  w.tick = (int*)take((size_t)(kH / 32) * kMaxChunk * 4);
  w.dq = (int*)take(2 * 4);
  w.cnt = (int*)take(kE * 4);
  w.cur = (int*)take(kE * 4);
  w.off = (int*)take((kE + 1) * 4);
  w.ntile = (int*)take(4);
  w.tiles = (PTile*)take(kMaxRTiles * sizeof(PTile));
  w.perm = (int*)take(P * 4);
  w.rpos = (int*)take(P * 4);
  w.re = (int*)take(P * 4);
  w.rw = (float*)take(P * 4);
  w.h = (__nv_bfloat16*)take(P * kI * 2);
  w.hsh = (__nv_bfloat16*)take((size_t)T * kI * 2);
  w.xq = take((size_t)T * kH / 2);
  w.xs = take((size_t)T * kH / 16);
  w.hq = take(P * kI / 2);
  w.hs = take(P * kI / 16);
  // the down outputs of a pass of cp columns: at most ~420 MB
  int np = 1;
  while (P * (kH / np) * 4 > (size_t)420 << 20 || kH % np || (kH / np) % 128) ++np;
  w.npass = np, w.cp = kH / np;
  w.yp = (float*)take(P * w.cp * 4);
  w.prank = (int*)take(P * 4);
  w.pw = (float*)take(P * 4);
  w.bytes = o;
  return w;
}

struct PArgs {
  LayerW L;
  const __nv_bfloat16* x;
  int T;
  Ws w;
  float *yr, *ys;
  int c0;  // down: the pass's first column
  bool fp4;
  float in_gu, in_dn;
  int cpc, npc, cq;  // the chained down: columns a pass, passes, passes in flight together
};

// ---- bucketing ----------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) k_count(const int* __restrict__ idx, int n, int* __restrict__ cnt) {
  __shared__ int hist[kE];
  if (threadIdx.x < kE) hist[threadIdx.x] = 0;
  __syncthreads();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&hist[idx[i]], 1);
  __syncthreads();
  if (threadIdx.x < kE && hist[threadIdx.x]) atomicAdd(&cnt[threadIdx.x], hist[threadIdx.x]);
}
// one CTA of 256: offsets, the tile list in expert order, the scatter cursors; clears cnt for the next call
__global__ void __launch_bounds__(256) k_scan(Ws w) {
  __shared__ int sc[kE], st[kE];
  const int e = threadIdx.x, c = w.cnt[e], nt = (c + kTM - 1) / kTM;
  sc[e] = c, st[e] = nt;
  __syncthreads();
  for (int o = 1; o < kE; o <<= 1) {  // inclusive scans
    const int a = e >= o ? sc[e - o] : 0, b = e >= o ? st[e - o] : 0;
    __syncthreads();
    sc[e] += a, st[e] += b;
    __syncthreads();
  }
  const int c0 = sc[e] - c, t0 = st[e] - nt;
  w.off[e] = c0;
  if (e == kE - 1) w.off[kE] = sc[e], *w.ntile = st[e];
  for (int j = 0; j < nt; ++j) w.tiles[t0 + j] = PTile{e, c0 + kTM * j, min(kTM, c - kTM * j), 0};
  w.cur[e] = 0;
  w.cnt[e] = 0;
}
// a thread a row: the row's picks get bucket positions; its list sorted by expert id for the combine
__global__ void k_scatter(const int* __restrict__ idx, const float* __restrict__ wts, int T, Ws w) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int e[kTopK], p[kTopK];
  float v[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    e[k] = idx[t * kTopK + k];
    v[k] = wts[t * kTopK + k];
    p[k] = w.off[e[k]] + atomicAdd(&w.cur[e[k]], 1);
    w.perm[p[k]] = t;
  }
#pragma unroll
  for (int i = 1; i < kTopK; ++i)
#pragma unroll
    for (int j = i; j > 0; --j)
      if (e[j] < e[j - 1]) {
        int a = e[j]; e[j] = e[j - 1]; e[j - 1] = a;
        a = p[j]; p[j] = p[j - 1]; p[j - 1] = a;
        const float b = v[j]; v[j] = v[j - 1]; v[j - 1] = b;
      }
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    w.re[t * kTopK + k] = e[k], w.rpos[t * kTopK + k] = p[k], w.rw[t * kTopK + k] = v[k];
    w.prank[p[k]] = k, w.pw[p[k]] = v[k];
  }
}

// ---- GEMM pieces ----------------------------------------------------------------------------------------------------
__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// A stage: 128 rows x 64 k (bf16), row r's 16-byte chunk c (k 8c .. 8c + 7) at byte 128 r + 16 (c ^ (r & 7)).
// Thread tid copies chunks 4 (tid & 1) .. + 3 of row tid >> 1 from src (its row's k-step start).
__device__ __forceinline__ void cp_async16a(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void load_a(uint8_t* sa, const __nv_bfloat16* src, int tid) {
  const int r = tid >> 1, h = tid & 1;
  uint8_t* d = sa + r * 128;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = 4 * h + j;
    cp_async16a(d + 16 * (c ^ (r & 7)), src + 8 * c);
  }
}
// The same stage, loaded coalesced: thread tid copies chunk tid & 7 of rows (tid >> 3) + (NT / 8) j, j < R (eight
// threads a row: a warp instruction covers 4 whole lines, against 16 part-lines for load_a). ri[j]: the row's
// index into base (row stride LD elements).
template <int R, int NT, int LD>
__device__ __forceinline__ void load_a_co(uint8_t* sa, const __nv_bfloat16* base, const uint32_t (&ri)[R], int k0, int tid) {
  const int c = tid & 7;
#pragma unroll
  for (int j = 0; j < R; ++j) {
    const int r = (tid >> 3) + (NT / 8) * j;
    cp_async16a(sa + r * 128 + 16 * (c ^ (r & 7)), base + (size_t)ri[j] * LD + k0 + 8 * c);
  }
}
// B stage: strip s, k16 m, lane l: words (o[2m], o[2m+1]) at byte ((s * 4 + m) * 32 + l) * 8
__device__ __forceinline__ void store_b(uint8_t* sb, int s, int lane, const uint32_t (&o)[8]) {
#pragma unroll
  for (int m = 0; m < 4; ++m) *reinterpret_cast<uint2*>(sb + ((s * 4 + m) * 32 + lane) * 8) = make_uint2(o[2 * m], o[2 * m + 1]);
}
// one k-step of a warp: MT m16 tiles from row 16 MT wm, four strips sj[]. A warp whose rows all lie at or past
// n skips (a warp-uniform branch); otherwise all its rows are computed (rows past n hold copies of the last
// row and are not stored), so the fragments load ahead of the MMAs without per-tile branches.
template <int MT>
__device__ __forceinline__ void mma_step(float (&acc)[MT][4][4], const uint8_t* sa, const uint8_t* sb, int wm, const int (&sj)[4],
                                         int n, int lane) {
  if (16 * MT * wm >= n) return;
  const int g = lane >> 2, t = lane & 3;
  uint2 b[4][4];
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int j = 0; j < 4; ++j) b[m][j] = *reinterpret_cast<const uint2*>(sb + ((sj[j] * 4 + m) * 32 + lane) * 8);
#pragma unroll
  for (int ti = 0; ti < MT; ++ti) {
    const int r0 = 16 * (MT * wm + ti) + g;
    // lane t's k (16 t .. 16 t + 15 of the step) = chunks 2 t, 2 t + 1; rows r0 and r0 + 8 have (row & 7) = g
    const uint4 p0 = *reinterpret_cast<const uint4*>(sa + r0 * 128 + 16 * ((2 * t) ^ g));
    const uint4 p1 = *reinterpret_cast<const uint4*>(sa + r0 * 128 + 16 * ((2 * t + 1) ^ g));
    const uint4 q0 = *reinterpret_cast<const uint4*>(sa + (r0 + 8) * 128 + 16 * ((2 * t) ^ g));
    const uint4 q1 = *reinterpret_cast<const uint4*>(sa + (r0 + 8) * 128 + 16 * ((2 * t + 1) ^ g));
    const uint2 a0[4] = {make_uint2(p0.x, p0.y), make_uint2(p0.z, p0.w), make_uint2(p1.x, p1.y), make_uint2(p1.z, p1.w)};
    const uint2 a1[4] = {make_uint2(q0.x, q0.y), make_uint2(q0.z, q0.w), make_uint2(q1.x, q1.y), make_uint2(q1.z, q1.w)};
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_bf16(acc[ti][j], a0[m].x, a1[m].x, a0[m].y, a1[m].y, b[m][j].x, b[m][j].y);
  }
}
__device__ __forceinline__ float silu_mul(float g, float u) {
  const float s = __bfloat162float(__float2bfloat16_rn(g / (1.f + expf(-g))));
  return s * u;
}

// ---- gate/up ------------------------------------------------------------------------------------------------------
// SH: the shared expert's tiles (rows of the chunk, FP8 weights with a scale per 128 x 128 block, so each
// 128-k unit is summed apart and added scaled, as the decode step does). MT: m16 tiles a warp (CTA rows 32 MT).
// DG (bench diagnostics, wrong results): 1 no weights (no B decode), 2 no MMAs, 4 no A loads
template <bool SH, int MT, int DG = 0>
__global__ void __launch_bounds__(256, 1) k_pup(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RT = 32 * MT;  // rows a tile
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int tile = blockIdx.x / kUpNB, nb = blockIdx.x % kUpNB;
  int e = 0, pos0, n;
  if (SH) {
    pos0 = tile * RT;
    n = min(RT, a.T - pos0);
    if (n <= 0) return;
  } else {
    if (tile >= *a.w.ntile) return;
    const PTile tt = a.w.tiles[tile];
    e = tt.e, pos0 = tt.pos0, n = tt.n;
  }
  // A rows (threads past the tile's rows copy its last row)
  const int ar = min(tid >> 1, n - 1);
  const __nv_bfloat16* asrc = a.x + (size_t)(SH ? pos0 + ar : a.w.perm[pos0 + ar]) * kH;
  // B: this warp decodes strips 2 warp, 2 warp + 1 of 16 (0-7 gate, 8-15 up), channels 8 (8 nb + s % 8) ..
  const int mat = warp >= 4;
  const int n80 = nb * 8 + (2 * warp) % 8;
  uint32_t fmt = 0, tab[8] = {};
  const uint8_t* wb = nullptr;
  int bs = 0;
  float kmat[2] = {1.f, 1.f};
  if (!SH) {
    const MatDesc* d = a.L.desc + e * 3;
    fmt = d[mat].fmt;
#pragma unroll
    for (int j = 0; j < 8; ++j) tab[j] = d[mat].tab[j];
    bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk;
    wb = a.L.pool + d[mat].off + (size_t)n80 * (kH / kBlkK) * bs;
    kmat[0] = d[0].s2 * (d[0].fmt == kNVFP4 ? 0x1p30f : 1.f);
    kmat[1] = d[1].s2 * (d[1].fmt == kNVFP4 ? 0x1p30f : 1.f);
  } else {
    wb = (mat ? a.L.sh.u : a.L.sh.g) + (size_t)n80 * (kH / 128) * 1024;
  }
  constexpr int NK = kH / 64;  // 96 k-steps
  Raw rb[2][2];                // routed: blocks b, b + 1 of the two strips
  Raw8 r8[2][2];               // shared: units u, u + 1
  auto load_raw = [&](int b, int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) load_blk(rb[buf][i], wb + ((size_t)i * (kH / kBlkK) + b) * bs, fmt, lane);
  };
  auto load_r8 = [&](int u, int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint8_t* p = wb + ((size_t)i * (kH / 128) + u) * 1024 + 32 * lane;
      r8[buf][i].q0 = ldg_stream128(p);
      r8[buf][i].q1 = ldg_stream128(p + 16);
    }
  };
  // B of k-step `step` (ph = step % 16, a compile-time constant at every call, so the buffers stay in
  // registers), then the raw loads it frees
  auto decode = [&](uint8_t* sb, int step, int ph) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t o[8];
      if (SH) dec8((ph & 1) ? r8[(ph >> 1) & 1][i].q1 : r8[(ph >> 1) & 1][i].q0, o);
      else dec_group(rb[(ph >> 3) & 1][i], (ph >> 1) & 3, ph & 1, fmt, tab, o);
      store_b(sb, 2 * warp + i, lane, o);
    }
    if (SH) {
      if ((ph & 1) && (step >> 1) + 2 < NK / 2) load_r8((step >> 1) + 2, (ph >> 1) & 1);
    } else {
      if ((ph & 7) == 7 && (step >> 3) + 2 < NK / 8) load_raw((step >> 3) + 2, (ph >> 3) & 1);
    }
  };
  // the warp's MMA strips: gate 2 wn, 2 wn + 1 and up 8 + 2 wn, 9 + 2 wn
  const int sj[4] = {2 * wn, 2 * wn + 1, 8 + 2 * wn, 9 + 2 * wn};
  float acc[MT][4][4], cu[SH ? MT : 1][4][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  // A: kNA stages, loaded kNA - 1 k-steps ahead (one commit group a k-step); B: 2 stages, decoded one ahead
  uint8_t* sA = smem;
  uint8_t* sB = smem + kNA * kStageA;
  if (SH) {
    load_r8(0, 0);
    load_r8(1, 1);
  } else {
    load_raw(0, 0);
    load_raw(1, 1);
  }
#pragma unroll
  for (int j = 0; j < kNA - 1; ++j) {
    if (tid < 2 * RT) load_a(sA + j * kStageA, asrc + 64 * j, tid);
    cp_commit();
  }
  decode(sB, 0, 0);
  cp_wait<kNA - 2>();
  __syncthreads();
  int sa_cur = 0;  // the A stage of k-step i
  const float* gsc = SH ? a.L.sh.gs + (nb / 2) * (kH / 128) : nullptr;  // the strip's 128-row scale block
  const float* usc = SH ? a.L.sh.us + (nb / 2) * (kH / 128) : nullptr;
  static_assert(NK % 16 == 0);
#pragma unroll 1
  for (int i0 = 0; i0 < NK; i0 += 16)
#pragma unroll
  for (int ii = 0; ii < 16; ++ii) {
    const int i = i0 + ii, s = ii & 1;
    const int sa_ld = sa_cur == 0 ? kNA - 1 : sa_cur - 1;  // (i + kNA - 1) % kNA
    if (!(DG & 4) && i + kNA - 1 < NK && tid < 2 * RT) load_a(sA + sa_ld * kStageA, asrc + 64 * (i + kNA - 1), tid);
    cp_commit();
    if (!(DG & 1) && i + 1 < NK) decode(sB + (s ^ 1) * kStageB, i + 1, (ii + 1) & 15);
    const uint8_t* sAc = sA + sa_cur * kStageA;
    sa_cur = sa_cur == kNA - 1 ? 0 : sa_cur + 1;
    if (SH) {
      if (!(i & 1))
#pragma unroll
        for (int x = 0; x < MT; ++x)
#pragma unroll
          for (int j = 0; j < 4; ++j) cu[x][j][0] = cu[x][j][1] = cu[x][j][2] = cu[x][j][3] = 0.f;
      mma_step<MT>(*reinterpret_cast<float(*)[MT][4][4]>(&cu), sAc, sB + s * kStageB, wm, sj, n, lane);
      if (i & 1) {
        const float sg = __ldg(gsc + (i >> 1)), su = __ldg(usc + (i >> 1));
#pragma unroll
        for (int x = 0; x < MT; ++x)
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[x][j][c] = fmaf(j < 2 ? sg : su, cu[x][j][c], acc[x][j][c]);
      }
    } else if (!(DG & 2)) {
      mma_step<MT>(acc, sAc, sB + s * kStageB, wm, sj, n, lane);
    }
    cp_wait<kNA - 2>();
    __syncthreads();
  }
  // SwiGLU: gate strip j with up strip j + 2
  __nv_bfloat16* hout = SH ? a.w.hsh : a.w.h;
#pragma unroll
  for (int x = 0; x < MT; ++x)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = 16 * (MT * wm + x) + g + 8 * hh;
        if (r >= n) continue;
        float hv[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const float gt = __bfloat162float(__float2bfloat16_rn(acc[x][j][2 * hh + c] * kmat[0]));
          const float ut = __bfloat162float(__float2bfloat16_rn(acc[x][j + 2][2 * hh + c] * kmat[1]));
          hv[c] = silu_mul(gt, ut);
        }
        const int ch = 8 * (nb * 8 + 2 * wn + j) + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(hout + (size_t)(pos0 + r) * kI + ch) = __floats2bfloat162_rn(hv[0], hv[1]);
      }
}

// ---- gate/up, warp-specialised (PrefillCfg::up = 1) ---------------------------------------------------------------
// The tile and the arithmetic of k_pup<false, 4> (the same MMAs on the same fragments in the same order, so the
// same h bits), with the work split by role: 8 consumer warps (2 rows x 4 columns, as k_pup's) only run the
// MMAs; 8 producer warps load the A rows by cp.async (kWsS - 1 k-steps ahead) and decode the B strips (each
// two, as k_pup's warps do, from raw blocks in registers loaded two blocks ahead). A ring of kWsS stages (A + B)
// handed over with named barriers: full[s] (producers arrive, consumers wait), empty[s] (the reverse).
constexpr int kWsS = 3;
// registers a thread by role (setmaxnreg; 256 x P + 256 x C = 64K), 0: both 128 (the default). 120 / 136 was
// measured (docs/experts.md, 2026-10-14): the MMA warps already fit in 120, the producers need 128.
constexpr int kWsRegP = 0, kWsRegC = 0;
constexpr int kWsSmem = kWsS * (kStageA + kStageB);  // 96 KB
constexpr int kWsSSh = 4;  // the shared expert's: 64-row A stages
constexpr int kWsSmemSh = kWsSSh * (kStageA / 2 + kStageB);  // 96 KB
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
// SH: the shared expert (rows of the chunk, FP8 with a scale per 128 x 128 block: each 128-k unit summed apart
// and added scaled, as k_pup<true, 2>); 64-row tiles there (MT = 2), as its per-unit sums double the
// consumers' accumulators.
template <bool SH, int DG = 0>
__global__ void __launch_bounds__(512, 1) k_pup_ws(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int MT = SH ? 2 : 4, RT = 32 * MT;
  constexpr int SA = RT * 128;  // A bytes a stage
  constexpr int NS = SH ? kWsSSh : kWsS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int tile = blockIdx.x / kUpNB, nb = blockIdx.x % kUpNB;
  int e = 0, pos0, n;
  if (SH) {
    pos0 = tile * RT;
    n = min(RT, a.T - pos0);
    if (n <= 0) return;
  } else {
    if (tile >= *a.w.ntile) return;
    const PTile tt = a.w.tiles[tile];
    e = tt.e, pos0 = tt.pos0, n = tt.n;
  }
  const MatDesc* d = a.L.desc + e * 3;
  constexpr int NK = kH / 64;  // 96 k-steps
  auto sA = [&](int s) { return smem + s * (SA + kStageB); };
  auto sB = [&](int s) { return smem + s * (SA + kStageB) + SA; };
  // barriers: full[s] = 1 + s, empty[s] = 1 + NS + s, each over all 512 threads
  if (warp >= 8) {
    if (kWsRegP) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kWsRegP));
    // ---- producer: A rows and B strips 2 pw, 2 pw + 1 (0-7 gate, 8-15 up), as k_pup's warp pw
    const int ptid = tid - 256, pw = warp - 8;
    constexpr int AR = RT / 32;  // rows a producer thread loads
    uint32_t ri[AR];
#pragma unroll
    for (int j = 0; j < AR; ++j) {
      const int r = min((ptid >> 3) + 32 * j, n - 1);
      ri[j] = SH ? pos0 + r : a.w.perm[pos0 + r];
    }
    const int mat = pw >= 4;
    const int n80 = nb * 8 + (2 * pw) % 8;
    uint32_t fmt = 0, tab[8] = {};
    int bs = 0;
    const uint8_t* wb;
    if (!SH) {
      fmt = d[mat].fmt;
#pragma unroll
      for (int j = 0; j < 8; ++j) tab[j] = d[mat].tab[j];
      bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk;
      wb = a.L.pool + d[mat].off + (size_t)n80 * (kH / kBlkK) * bs;
    } else {
      wb = (mat ? a.L.sh.u : a.L.sh.g) + (size_t)n80 * (kH / 128) * 1024;
    }
    Raw rb[SH ? 1 : 2][2];
    Raw8 r8[SH ? 2 : 1][2];
    auto load_raw = [&](int b, int buf) {
#pragma unroll
      for (int i = 0; i < 2; ++i) load_blk(rb[buf % (SH ? 1 : 2)][i], wb + ((size_t)i * (kH / kBlkK) + b) * bs, fmt, lane);
    };
    auto load_r8 = [&](int u, int buf) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint8_t* q = wb + ((size_t)i * (kH / 128) + u) * 1024 + 32 * lane;
        r8[buf % (SH ? 2 : 1)][i].q0 = ldg_stream128(q);
        r8[buf % (SH ? 2 : 1)][i].q1 = ldg_stream128(q + 16);
      }
    };
    if (!(DG & 1)) {
      if (SH) load_r8(0, 0), load_r8(1, 1);
      else load_raw(0, 0), load_raw(1, 1);
    }
    // A of the first NS - 1 k-steps (their stages are free)
#pragma unroll
    for (int j = 0; j < NS - 1; ++j) {
      if (!(DG & 4)) load_a_co<AR, 256, kH>(sA(j), a.x, ri, 64 * j, ptid);
      cp_commit();
    }
    int s_cur = 0;  // the stage of k-step i
#pragma unroll 1
    for (int i0 = 0; i0 < NK; i0 += 16)
#pragma unroll
      for (int ii = 0; ii < 16; ++ii) {
        const int i = i0 + ii;
        // B of k-step i first (its stage's last user, k-step i - NS, was waited for at iteration i - NS + 1),
        // while the consumers still run k-step i - 1
        if (!(DG & 1)) {
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            uint32_t o[8];
            if (SH) dec8((ii & 1) ? r8[(ii >> 1) & 1][x].q1 : r8[(ii >> 1) & 1][x].q0, o);
            else dec_group(rb[(ii >> 3) & 1][x], (ii >> 1) & 3, ii & 1, fmt, tab, o);
            store_b(sB(s_cur), 2 * pw + x, lane, o);
          }
          if (SH) {
            if ((ii & 1) && (i >> 1) + 2 < NK / 2) load_r8((i >> 1) + 2, (ii >> 1) & 1);
          } else {
            if ((ii & 7) == 7 && (i >> 3) + 2 < NK / 8) load_raw((i >> 3) + 2, (ii >> 3) & 1);
          }
        }
        cp_wait<NS - 2>();  // k-step i's A group is in (groups: one a k-step)
        bar_arrive(1 + s_cur, 512);
        // then A of k-step i + NS - 1 into the stage k-step i - 1 used, once the consumers have left it
        const int s_ld = s_cur == 0 ? NS - 1 : s_cur - 1;
        if (i + NS - 1 < NK) {
          if (i >= 1) bar_sync(1 + NS + s_ld, 512);
          if (!(DG & 4)) load_a_co<AR, 256, kH>(sA(s_ld), a.x, ri, 64 * (i + NS - 1), ptid);
        }
        cp_commit();
        s_cur = s_cur == NS - 1 ? 0 : s_cur + 1;
      }
    cp_wait0();
    return;
  }
  // ---- consumer: MMAs and the epilogue (k_pup's)
  if (kWsRegC) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kWsRegC));
  const int wm = warp >> 2, wn = warp & 3;
  float kg = 1.f, ku = 1.f;
  if (!SH) kg = d[0].s2 * (d[0].fmt == kNVFP4 ? 0x1p30f : 1.f), ku = d[1].s2 * (d[1].fmt == kNVFP4 ? 0x1p30f : 1.f);
  const float* gsc = SH ? a.L.sh.gs + (nb / 2) * (kH / 128) : nullptr;  // the strips' 128-row scale block
  const float* usc = SH ? a.L.sh.us + (nb / 2) * (kH / 128) : nullptr;
  const int sj[4] = {2 * wn, 2 * wn + 1, 8 + 2 * wn, 9 + 2 * wn};
  float acc[MT][4][4], cu[SH ? MT : 1][4][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  int s_cur = 0;
#pragma unroll 1
  for (int i = 0; i < NK; ++i) {
    bar_sync(1 + s_cur, 512);
    if (SH) {
      if (!(i & 1))
#pragma unroll
        for (int x = 0; x < MT; ++x)
#pragma unroll
          for (int j = 0; j < 4; ++j) cu[x][j][0] = cu[x][j][1] = cu[x][j][2] = cu[x][j][3] = 0.f;
      if (!(DG & 2)) mma_step<MT>(*reinterpret_cast<float(*)[MT][4][4]>(&cu), sA(s_cur), sB(s_cur), wm, sj, n, lane);
    } else if (!(DG & 2)) {
      mma_step<MT>(acc, sA(s_cur), sB(s_cur), wm, sj, n, lane);
    }
    // the stage is reloaded (A of k-step i + NS) only if that k-step exists: the producers wait just then
    if (i + NS < NK) bar_arrive(1 + NS + s_cur, 512);
    s_cur = s_cur == NS - 1 ? 0 : s_cur + 1;
    if (SH && (i & 1)) {
      const float sg = __ldg(gsc + (i >> 1)), su = __ldg(usc + (i >> 1));
#pragma unroll
      for (int x = 0; x < MT; ++x)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[x][j][c] = fmaf(j < 2 ? sg : su, cu[x][j][c], acc[x][j][c]);
    }
  }
  __nv_bfloat16* hout = SH ? a.w.hsh : a.w.h;
#pragma unroll
  for (int x = 0; x < MT; ++x)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = 16 * (MT * wm + x) + g + 8 * hh;
        if (r >= n) continue;
        float hv[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const float gt = __bfloat162float(__float2bfloat16_rn(acc[x][j][2 * hh + c] * kg));
          const float ut = __bfloat162float(__float2bfloat16_rn(acc[x][j + 2][2 * hh + c] * ku));
          hv[c] = silu_mul(gt, ut);
        }
        const int ch = 8 * (nb * 8 + 2 * wn + j) + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(hout + (size_t)(pos0 + r) * kI + ch) = __floats2bfloat162_rn(hv[0], hv[1]);
      }
}

// ---- gate/up, warp-specialised with 64 x 64 warp tiles (PrefillCfg::up = 2) ------------------------------------
// The tile and the arithmetic of k_pup<false, 4> again, now with 4 consumer warps of 64 rows x 64 columns
// (2 x 2; gate strips 4 wn .. 4 wn + 3 with up strips 8 + 4 wn ..) and 4 producer warps (A rows, a row a
// thread; B: four strips a warp, raw blocks two ahead in registers). Each accumulator still takes the k-step's
// four MMAs in the order m = 0..3 on the same fragments, so the same h bits. Half the shared-memory reads
// per MMA of k_pup_ws, one MMA warp a scheduler.
template <int DG = 0>
__global__ void __launch_bounds__(256, 1) k_pup_w2(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int NS = kWsS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int tile = blockIdx.x / kUpNB, nb = blockIdx.x % kUpNB;
  if (tile >= *a.w.ntile) return;
  const PTile tt = a.w.tiles[tile];
  const int e = tt.e, pos0 = tt.pos0, n = tt.n;
  const MatDesc* d = a.L.desc + e * 3;
  constexpr int NK = kH / 64;
  auto sA = [&](int s) { return smem + s * (kStageA + kStageB); };
  auto sB = [&](int s) { return smem + s * (kStageA + kStageB) + kStageA; };
  if (warp >= 4) {
    const int ptid = tid - 128, pw = warp - 4;
    uint32_t ri[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) ri[j] = a.w.perm[pos0 + min((ptid >> 3) + 16 * j, n - 1)];
    const int mat = pw >= 2;
    const int n80 = nb * 8 + (4 * pw) % 8;
    const uint32_t fmt = d[mat].fmt;
    uint32_t tab[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) tab[j] = d[mat].tab[j];
    const int bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk;
    const uint8_t* wb = a.L.pool + d[mat].off + (size_t)n80 * (kH / kBlkK) * bs;
    Raw rb[2][4];
    auto load_raw = [&](int b, int buf) {
#pragma unroll
      for (int i = 0; i < 4; ++i) load_blk(rb[buf][i], wb + ((size_t)i * (kH / kBlkK) + b) * bs, fmt, lane);
    };
    if (!(DG & 1)) load_raw(0, 0), load_raw(1, 1);
#pragma unroll
    for (int j = 0; j < NS - 1; ++j) {
      if (!(DG & 4)) load_a_co<8, 128, kH>(sA(j), a.x, ri, 64 * j, ptid);
      cp_commit();
    }
    int s_cur = 0;
#pragma unroll 1
    for (int i0 = 0; i0 < NK; i0 += 16)
#pragma unroll
      for (int ii = 0; ii < 16; ++ii) {
        const int i = i0 + ii;
        if (!(DG & 1)) {
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            uint32_t o[8];
            dec_group(rb[(ii >> 3) & 1][x], (ii >> 1) & 3, ii & 1, fmt, tab, o);
            store_b(sB(s_cur), 4 * pw + x, lane, o);
          }
          if ((ii & 7) == 7 && (i >> 3) + 2 < NK / 8) load_raw((i >> 3) + 2, (ii >> 3) & 1);
        }
        cp_wait<NS - 2>();
        bar_arrive(1 + s_cur, 256);
        const int s_ld = s_cur == 0 ? NS - 1 : s_cur - 1;
        if (i + NS - 1 < NK) {
          if (i >= 1) bar_sync(1 + NS + s_ld, 256);
          if (!(DG & 4)) load_a_co<8, 128, kH>(sA(s_ld), a.x, ri, 64 * (i + NS - 1), ptid);
        }
        cp_commit();
        s_cur = s_cur == NS - 1 ? 0 : s_cur + 1;
      }
    cp_wait0();
    return;
  }
  const int wm = warp >> 1, wn = warp & 1;
  const float kg = d[0].s2 * (d[0].fmt == kNVFP4 ? 0x1p30f : 1.f), ku = d[1].s2 * (d[1].fmt == kNVFP4 ? 0x1p30f : 1.f);
  float acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const bool live = 64 * wm < n;
  int s_cur = 0;
#pragma unroll 1
  for (int i = 0; i < NK; ++i) {
    bar_sync(1 + s_cur, 256);
    if (!(DG & 2) && live) {
      const uint8_t* sa = sA(s_cur);
      const uint8_t* sb = sB(s_cur);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        uint2 b[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int strip = j < 4 ? 4 * wn + j : 8 + 4 * wn + (j - 4);
          b[j] = *reinterpret_cast<const uint2*>(sb + ((strip * 4 + m) * 32 + lane) * 8);
        }
#pragma unroll
        for (int ti = 0; ti < 4; ++ti) {
          const int r0 = 16 * (4 * wm + ti) + g;
          // k-group m of lane t: chunk 2 t + (m >> 1), its half m & 1
          const int ch = (2 * t + (m >> 1)) ^ g;
          const uint2 a0 = *reinterpret_cast<const uint2*>(sa + r0 * 128 + 16 * ch + 8 * (m & 1));
          const uint2 a1 = *reinterpret_cast<const uint2*>(sa + (r0 + 8) * 128 + 16 * ch + 8 * (m & 1));
#pragma unroll
          for (int j = 0; j < 8; ++j) mma_bf16(acc[ti][j], a0.x, a1.x, a0.y, a1.y, b[j].x, b[j].y);
        }
      }
    }
    if (i + NS < NK) bar_arrive(1 + NS + s_cur, 256);
    s_cur = s_cur == NS - 1 ? 0 : s_cur + 1;
  }
#pragma unroll
  for (int x = 0; x < 4; ++x)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = 16 * (4 * wm + x) + g + 8 * hh;
        if (r >= n) continue;
        float hv[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const float gt = __bfloat162float(__float2bfloat16_rn(acc[x][j][2 * hh + c] * kg));
          const float ut = __bfloat162float(__float2bfloat16_rn(acc[x][j + 4][2 * hh + c] * ku));
          hv[c] = silu_mul(gt, ut);
        }
        const int chn = 8 * (nb * 8 + 4 * wn + j) + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(a.w.h + (size_t)(pos0 + r) * kI + chn) = __floats2bfloat162_rn(hv[0], hv[1]);
      }
}

// ---- routed down ---------------------------------------------------------------------------------------------------
// CTA: one tile x kDnCB column blocks of 128 outputs of the pass, one after another as one pipeline of
// kDnCB x 8 k-steps (K = 512 a block): the A ring runs on across the blocks (the same h rows), the next blocks'
// raw weights load while the current ones are decoded, and each block's sums go out when its last k-step is
// done: the raw sums c (NVFP4's carry 2^-30) into yp [pick][column - c0].
constexpr int kDnCB = 4;
template <int MT>
__global__ void __launch_bounds__(256, 1) k_pdown_r(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RT = 32 * MT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int ngrp = a.w.cp / 128 / kDnCB;
  const int tile = blockIdx.x / ngrp, cg = blockIdx.x % ngrp;
  if (tile >= *a.w.ntile) return;
  const PTile tt = a.w.tiles[tile];
  const int e = tt.e, pos0 = tt.pos0, n = tt.n;
  const __nv_bfloat16* asrc = a.w.h + (size_t)(pos0 + min(tid >> 1, n - 1)) * kI;
  const MatDesc& d = a.L.desc[e * 3 + 2];
  const uint32_t fmt = d.fmt;
  uint32_t tab[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) tab[j] = d.tab[j];
  const int bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk;
  const uint8_t* wb = a.L.pool + d.off + (size_t)(a.c0 / 8 + 16 * kDnCB * cg + 2 * warp) * bs;
  Raw rb[2][2];  // [column block & 1][strip]
  auto load_raw = [&](int c, int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) load_blk(rb[buf][i], wb + (size_t)(16 * c + i) * bs, fmt, lane);
  };
  constexpr int NK = kDnCB * 8;
  static_assert(NK % 16 == 0);
  auto decode = [&](uint8_t* sb, int step, int ph) {  // ph = step % 16, compile-time
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t o[8];
      dec_group(rb[ph >> 3][i], (ph & 7) >> 1, ph & 1, fmt, tab, o);
      store_b(sb, 2 * warp + i, lane, o);
    }
    if ((ph & 7) == 7 && (step >> 3) + 2 < kDnCB) load_raw((step >> 3) + 2, ph >> 3);
  };
  const int sj[4] = {4 * wn, 4 * wn + 1, 4 * wn + 2, 4 * wn + 3};
  float acc[MT][4][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  uint8_t* sA = smem;
  uint8_t* sB = smem + kNA * kStageA;
  load_raw(0, 0);
  load_raw(1, 1);
#pragma unroll
  for (int j = 0; j < kNA - 1; ++j) {
    if (tid < 2 * RT) load_a(sA + j * kStageA, asrc + 64 * j, tid);
    cp_commit();
  }
  decode(sB, 0, 0);
  cp_wait<kNA - 2>();
  __syncthreads();
  int sa_cur = 0;
#pragma unroll 1
  for (int i0 = 0; i0 < NK; i0 += 16)
#pragma unroll
    for (int ii = 0; ii < 16; ++ii) {
      const int i = i0 + ii, s = ii & 1;
      const int sa_ld = sa_cur == 0 ? kNA - 1 : sa_cur - 1;
      if (i + kNA - 1 < NK && tid < 2 * RT) load_a(sA + sa_ld * kStageA, asrc + 64 * ((i + kNA - 1) & 7), tid);
      cp_commit();
      if (i + 1 < NK) decode(sB + (s ^ 1) * kStageB, i + 1, (ii + 1) & 15);
      mma_step<MT>(acc, sA + sa_cur * kStageA, sB + s * kStageB, wm, sj, n, lane);
      sa_cur = sa_cur == kNA - 1 ? 0 : sa_cur + 1;
      if ((ii & 7) == 7) {  // column block i >> 3 done
        const int cb = kDnCB * cg + (i >> 3);
#pragma unroll
        for (int x = 0; x < MT; ++x)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int r = 16 * (MT * wm + x) + g + 8 * hh;
              if (r < n)
                *reinterpret_cast<float2*>(a.w.yp + (size_t)(pos0 + r) * a.w.cp + 128 * cb + 8 * sj[j] + 2 * t) =
                    make_float2(acc[x][j][2 * hh], acc[x][j][2 * hh + 1]);
            }
            acc[x][j][0] = acc[x][j][1] = acc[x][j][2] = acc[x][j][3] = 0.f;
          }
      }
      cp_wait<kNA - 2>();
      __syncthreads();
    }
}

// ---- the chained down (PrefillCfg::down = 1) -------------------------------------------------------------------
// The same sums as k_pdown_r (each pick's c over K = 512 in the same MMA order), added into yr at once in the
// combine's arithmetic, acc = fmaf(s2 fmt_k w, c, acc) over the row's picks in expert-id order, with acc read
// back from yr: so the same bits as k_pdown_r + k_combine, without the per-pick f32 buffer (1.6 GB written and
// read back at 8k rows).
// The order: a row's ticket (one per 32 columns, a warp's share of a column block) counts its picks already
// added there; each warp waits for and releases its own rows, so a wait holds only the rows that need it. Tasks (pass,
// tile) are claimed from a counter in the order pass-major, then the tiles' order (expert id), so a tile waits
// only on lower tasks, each held by a running CTA (no deadlock however many CTAs are resident). For each
// 32 columns, a pick of rank k waits for ticket k, adds, and releases k + 1 (the row's last pick: 0, so the
// tickets are clear for the next call).
// The passes keep a slice of yr (T x cpc f32, the L2 budget) hot while every tile adds into it.
__device__ __forceinline__ int ld_relaxed(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed(int* p, int v) {
  asm volatile("st.relaxed.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint64_t l2_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ float2 ld_yr(const float* p, uint64_t pol) {
  float2 v;
  asm volatile("ld.global.cg.L2::cache_hint.v2.f32 {%0,%1}, [%2], %3;" : "=f"(v.x), "=f"(v.y) : "l"(p), "l"(pol) : "memory");
  return v;
}
__device__ __forceinline__ void st_yr(float* p, float2 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v2.f32 [%0], {%1,%2}, %3;" ::"l"(p), "f"(v.x), "f"(v.y), "l"(pol) : "memory");
}
template <int MT>
__global__ void __launch_bounds__(256, 1) k_pdown_c(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ int s_task, s_row[kTM], s_rank[kTM];
  __shared__ float s_k0[kTM];
  constexpr int RT = 32 * MT;
  static_assert(RT == kTM);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int ntile = *a.w.ntile, total = ntile * a.npc;
  const int NK = a.cpc / 16;  // 8 k-steps a column block of 128, an even number of blocks
  const uint64_t pol = l2_last();
  const int sj[4] = {4 * wn, 4 * wn + 1, 4 * wn + 2, 4 * wn + 3};
  uint8_t* sA = smem;
  uint8_t* sB = smem + kNA * kStageA;
#pragma unroll 1
  for (;;) {
    if (tid == 0) s_task = atomicAdd(a.w.dq, 1);
    __syncthreads();
    const int task = s_task;
    if (task >= total) break;
    // Q passes claimed round-robin (each pass's tiles still in order)
    const int grp = task / (a.cq * ntile), rem = task - grp * a.cq * ntile;
    const int pass = grp * a.cq + rem % a.cq, tile = rem / a.cq;
    const PTile tt = a.w.tiles[tile];
    const int e = tt.e, pos0 = tt.pos0, n = tt.n, c0 = pass * a.cpc;
    const MatDesc& d = a.L.desc[e * 3 + 2];
    const uint32_t fmt = d.fmt;
    if (tid < n) {
      const float kk = __ldg(&d.s2) * (fmt == kNVFP4 ? 0x1p30f : 1.f);  // as k_combine
      s_row[tid] = a.w.perm[pos0 + tid];
      s_rank[tid] = a.w.prank[pos0 + tid];
      s_k0[tid] = kk * a.w.pw[pos0 + tid];
    }
    uint32_t ri[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) ri[j] = pos0 + min((tid >> 3) + 32 * j, n - 1);
    uint32_t tab[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) tab[j] = d.tab[j];
    const int bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk;
    const uint8_t* wb = a.L.pool + d.off + (size_t)(c0 / 8 + 2 * warp) * bs;
    const int ncb = a.cpc / 128;
    Raw rb[2][2];  // [column block & 1][strip]
    auto load_raw = [&](int c, int buf) {
#pragma unroll
      for (int i = 0; i < 2; ++i) load_blk(rb[buf][i], wb + (size_t)(16 * c + i) * bs, fmt, lane);
    };
    auto decode = [&](uint8_t* sb, int step, int ph) {  // ph = step % 16, compile-time
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t o[8];
        dec_group(rb[ph >> 3][i], (ph & 7) >> 1, ph & 1, fmt, tab, o);
        store_b(sb, 2 * warp + i, lane, o);
      }
      if ((ph & 7) == 7 && (step >> 3) + 2 < ncb) load_raw((step >> 3) + 2, ph >> 3);
    };
    float acc[MT][4][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    load_raw(0, 0);
    load_raw(1, 1);
#pragma unroll
    for (int j = 0; j < kNA - 1; ++j) {
      load_a_co<4, 256, kI>(sA + j * kStageA, a.w.h, ri, 64 * j, tid);
      cp_commit();
    }
    decode(sB, 0, 0);
    cp_wait<kNA - 2>();
    __syncthreads();
    int sa_cur = 0;
#pragma unroll 1
    for (int i0 = 0; i0 < NK; i0 += 16)
#pragma unroll
      for (int ii = 0; ii < 16; ++ii) {
        const int i = i0 + ii, s = ii & 1;
        const int sa_ld = sa_cur == 0 ? kNA - 1 : sa_cur - 1;
        if (i + kNA - 1 < NK) load_a_co<4, 256, kI>(sA + sa_ld * kStageA, a.w.h, ri, 64 * ((i + kNA - 1) & 7), tid);
        cp_commit();
        if (i + 1 < NK) decode(sB + (s ^ 1) * kStageB, i + 1, (ii + 1) & 15);
        mma_step<MT>(acc, sA + sa_cur * kStageA, sB + s * kStageB, wm, sj, n, lane);
        sa_cur = sa_cur == kNA - 1 ? 0 : sa_cur + 1;
        if ((ii & 7) == 7) {  // column block i >> 3 done: each warp adds its 64 rows x 32 columns into yr
          const int cb = i >> 3, r0 = 64 * wm;
          if (r0 < n) {
            int* tick = a.w.tick + (size_t)((c0 + 128 * cb) / 32 + wn) * kMaxChunk;
            // wait until the rows' lower picks are in (lane l polls rows r0 + l, r0 + l + 32; relaxed loads,
            // then a fence)
            uint32_t pend = 0;
#pragma unroll
            for (int m = 0; m < 2; ++m)
              if (r0 + lane + 32 * m < n && s_rank[r0 + lane + 32 * m] > 0) pend |= 1u << m;
            if (__any_sync(0xffffffffu, pend)) {
              uint32_t ns = 64;
              for (uint32_t spin = 0;; ++spin) {
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                  const int r = r0 + lane + 32 * m;
                  if ((pend >> m & 1) && ld_relaxed(tick + s_row[r]) == s_rank[r]) pend &= ~(1u << m);
                }
                if (!__any_sync(0xffffffffu, pend)) break;
                __nanosleep(ns);
                ns = min(ns * 2, 1024u);
                if (spin == (1u << 25)) __trap();  // ~30 s: a ticket left set (the workspace not zeroed)
              }
              __threadfence();
              __syncwarp();
            }
            // per m16 tile, its two rows' eight loads in flight together, then the adds and stores
#pragma unroll
            for (int x = 0; x < MT; ++x) {
              float2 v[2][4];
              float* yrow[2];
              bool live[2];
#pragma unroll
              for (int hh = 0; hh < 2; ++hh) {
                const int r = 16 * (MT * wm + x) + g + 8 * hh;
                live[hh] = r < n;
                yrow[hh] = a.yr + (size_t)s_row[live[hh] ? r : 0] * kH + c0 + 128 * cb + 2 * t;
                const bool rd = live[hh] && s_rank[live[hh] ? r : 0] > 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[hh][j] = rd ? ld_yr(yrow[hh] + 8 * sj[j], pol) : make_float2(0.f, 0.f);
              }
#pragma unroll
              for (int hh = 0; hh < 2; ++hh)
                if (live[hh]) {
                  const float k0 = s_k0[16 * (MT * wm + x) + g + 8 * hh];
#pragma unroll
                  for (int j = 0; j < 4; ++j)
                    st_yr(yrow[hh] + 8 * sj[j],
                          make_float2(fmaf(k0, acc[x][j][2 * hh], v[hh][j].x), fmaf(k0, acc[x][j][2 * hh + 1], v[hh][j].y)), pol);
                }
            }
            // the adds are out: release the rows (each lane's fence, the warp barrier, relaxed stores)
            __threadfence();
            __syncwarp();
#pragma unroll
            for (int m = 0; m < 2; ++m) {
              const int r = r0 + lane + 32 * m;
              if (r < n) st_relaxed(tick + s_row[r], s_rank[r] == kTopK - 1 ? 0 : s_rank[r] + 1);
            }
          }
#pragma unroll
          for (int x = 0; x < MT; ++x)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[x][j][0] = acc[x][j][1] = acc[x][j][2] = acc[x][j][3] = 0.f;
        }
        cp_wait<kNA - 2>();
        __syncthreads();
      }
  }
  if (tid == 0 && atomicAdd(a.w.dq + 1, 1) == (int)gridDim.x - 1) a.w.dq[0] = 0, a.w.dq[1] = 0;
}

// The chained down, warp-specialised (PrefillCfg::down = 2): k_pdown_c's tasks, order, sums and adds, with
// k_pup_ws's split: 8 producer warps load the h rows (coalesced) and decode the weights (two strips a warp,
// raw blocks two column blocks ahead), 8 consumer warps run the MMAs and, after each column block, wait for
// their rows' tickets and add into yr. A ring of kWsS stages per task; the CTA meets at each task's end.
__global__ void __launch_bounds__(512, 1) k_pdown_cw(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ int s_task, s_row[kTM], s_rank[kTM];
  __shared__ float s_k0[kTM];
  constexpr int MT = 4, NS = kWsS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int ntile = *a.w.ntile, total = ntile * a.npc;
  const int NK = a.cpc / 16;  // 8 k-steps a column block of 128, an even number of blocks
  const int ncb = a.cpc / 128;
  const uint64_t pol = l2_last();
  auto sA = [&](int s) { return smem + s * (kStageA + kStageB); };
  auto sB = [&](int s) { return smem + s * (kStageA + kStageB) + kStageA; };
  // the roles split before the task loop, so each loop runs under its own register budget (setmaxnreg)
  if (warp >= 8) {
    if (kWsRegP) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(kWsRegP));
#pragma unroll 1
    for (;;) {
      if (tid == 0) s_task = atomicAdd(a.w.dq, 1);
      __syncthreads();
      const int task = s_task;
      if (task >= total) break;
      const int grp = task / (a.cq * ntile), rem = task - grp * a.cq * ntile;
      const int pass = grp * a.cq + rem % a.cq, tile = rem / a.cq;
      const PTile tt = a.w.tiles[tile];
      const int e = tt.e, pos0 = tt.pos0, n = tt.n, c0 = pass * a.cpc;
      const MatDesc& d = a.L.desc[e * 3 + 2];
      const uint32_t fmt = d.fmt;
      if (tid < n) {
        const float kk = __ldg(&d.s2) * (fmt == kNVFP4 ? 0x1p30f : 1.f);  // as k_combine
        s_row[tid] = a.w.perm[pos0 + tid];
        s_rank[tid] = a.w.prank[pos0 + tid];
        s_k0[tid] = kk * a.w.pw[pos0 + tid];
      }
      __syncthreads();
      // ---- producer
      const int ptid = tid - 256, pw = warp - 8;
      uint32_t ri[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) ri[j] = pos0 + min((ptid >> 3) + 32 * j, n - 1);
      uint32_t tab[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) tab[j] = d.tab[j];
      const int bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk;
      const uint8_t* wb = a.L.pool + d.off + (size_t)(c0 / 8 + 2 * pw) * bs;
      Raw rb[2][2];  // [column block & 1][strip]
      auto load_raw = [&](int c, int buf) {
#pragma unroll
        for (int x = 0; x < 2; ++x) load_blk(rb[buf][x], wb + (size_t)(16 * c + x) * bs, fmt, lane);
      };
      load_raw(0, 0);
      load_raw(1, 1);
#pragma unroll
      for (int j = 0; j < NS - 1; ++j) {
        load_a_co<4, 256, kI>(sA(j), a.w.h, ri, 64 * j, ptid);
        cp_commit();
      }
      int s_cur = 0;
#pragma unroll 1
      for (int i0 = 0; i0 < NK; i0 += 16)
#pragma unroll
        for (int ii = 0; ii < 16; ++ii) {
          const int i = i0 + ii;
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            uint32_t o[8];
            dec_group(rb[ii >> 3][x], (ii & 7) >> 1, ii & 1, fmt, tab, o);
            store_b(sB(s_cur), 2 * pw + x, lane, o);
          }
          if ((ii & 7) == 7 && (i >> 3) + 2 < ncb) load_raw((i >> 3) + 2, ii >> 3);
          cp_wait<NS - 2>();
          bar_arrive(1 + s_cur, 512);
          const int s_ld = s_cur == 0 ? NS - 1 : s_cur - 1;
          if (i + NS - 1 < NK) {
            if (i >= 1) bar_sync(1 + NS + s_ld, 512);
            load_a_co<4, 256, kI>(sA(s_ld), a.w.h, ri, 64 * ((i + NS - 1) & 7), ptid);
          }
          cp_commit();
          s_cur = s_cur == NS - 1 ? 0 : s_cur + 1;
        }
      cp_wait0();
      __syncthreads();  // the task's stages and row tables are free
    }
  } else {
    if (kWsRegC) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(kWsRegC));
#pragma unroll 1
    for (;;) {
      if (tid == 0) s_task = atomicAdd(a.w.dq, 1);
      __syncthreads();
      const int task = s_task;
      if (task >= total) break;
      const int grp = task / (a.cq * ntile), rem = task - grp * a.cq * ntile;
      const int pass = grp * a.cq + rem % a.cq, tile = rem / a.cq;
      const PTile tt = a.w.tiles[tile];
      const int e = tt.e, pos0 = tt.pos0, n = tt.n, c0 = pass * a.cpc;
      const MatDesc& d = a.L.desc[e * 3 + 2];
      const uint32_t fmt = d.fmt;
      if (tid < n) {
        const float kk = __ldg(&d.s2) * (fmt == kNVFP4 ? 0x1p30f : 1.f);  // as k_combine
        s_row[tid] = a.w.perm[pos0 + tid];
        s_rank[tid] = a.w.prank[pos0 + tid];
        s_k0[tid] = kk * a.w.pw[pos0 + tid];
      }
      __syncthreads();
      const int wm = warp >> 2, wn = warp & 3;
      const int sj[4] = {4 * wn, 4 * wn + 1, 4 * wn + 2, 4 * wn + 3};
      float acc[MT][4][4];
#pragma unroll
      for (int x = 0; x < MT; ++x)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[x][j][0] = acc[x][j][1] = acc[x][j][2] = acc[x][j][3] = 0.f;
      int s_cur = 0;
#pragma unroll 1
      for (int i = 0; i < NK; ++i) {
        bar_sync(1 + s_cur, 512);
        mma_step<MT>(acc, sA(s_cur), sB(s_cur), wm, sj, n, lane);
        if (i + NS < NK) bar_arrive(1 + NS + s_cur, 512);
        s_cur = s_cur == NS - 1 ? 0 : s_cur + 1;
    if ((i & 7) == 7) {  // column block i >> 3 done: each warp adds its 64 rows x 32 columns into yr
      const int cb = i >> 3, r0 = 64 * wm;
      if (r0 < n) {
        int* tick = a.w.tick + (size_t)((c0 + 128 * cb) / 32 + wn) * kMaxChunk;
        // wait until the rows' lower picks are in (lane l polls rows r0 + l, r0 + l + 32; relaxed loads,
        // then a fence)
        uint32_t pend = 0;
#pragma unroll
        for (int m = 0; m < 2; ++m)
          if (r0 + lane + 32 * m < n && s_rank[r0 + lane + 32 * m] > 0) pend |= 1u << m;
        if (__any_sync(0xffffffffu, pend)) {
          uint32_t ns = 64;
          for (uint32_t spin = 0;; ++spin) {
#pragma unroll
            for (int m = 0; m < 2; ++m) {
              const int r = r0 + lane + 32 * m;
              if ((pend >> m & 1) && ld_relaxed(tick + s_row[r]) == s_rank[r]) pend &= ~(1u << m);
            }
            if (!__any_sync(0xffffffffu, pend)) break;
            __nanosleep(ns);
            ns = min(ns * 2, 1024u);
            if (spin == (1u << 25)) __trap();  // ~30 s: a ticket left set (the workspace not zeroed)
          }
          __threadfence();
          __syncwarp();
        }
        // per m16 tile, its two rows' eight loads in flight together, then the adds and stores
#pragma unroll
        for (int x = 0; x < MT; ++x) {
          float2 v[2][4];
          float* yrow[2];
          bool live[2];
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int r = 16 * (MT * wm + x) + g + 8 * hh;
            live[hh] = r < n;
            yrow[hh] = a.yr + (size_t)s_row[live[hh] ? r : 0] * kH + c0 + 128 * cb + 2 * t;
            const bool rd = live[hh] && s_rank[live[hh] ? r : 0] > 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[hh][j] = rd ? ld_yr(yrow[hh] + 8 * sj[j], pol) : make_float2(0.f, 0.f);
          }
#pragma unroll
          for (int hh = 0; hh < 2; ++hh)
            if (live[hh]) {
              const float k0 = s_k0[16 * (MT * wm + x) + g + 8 * hh];
#pragma unroll
              for (int j = 0; j < 4; ++j)
                st_yr(yrow[hh] + 8 * sj[j],
                      make_float2(fmaf(k0, acc[x][j][2 * hh], v[hh][j].x), fmaf(k0, acc[x][j][2 * hh + 1], v[hh][j].y)), pol);
            }
        }
        // the adds are out: release the rows (each lane's fence, the warp barrier, relaxed stores)
        __threadfence();
        __syncwarp();
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int r = r0 + lane + 32 * m;
          if (r < n) st_relaxed(tick + s_row[r], s_rank[r] == kTopK - 1 ? 0 : s_rank[r] + 1);
        }
      }
#pragma unroll
      for (int x = 0; x < MT; ++x)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[x][j][0] = acc[x][j][1] = acc[x][j][2] = acc[x][j][3] = 0.f;
    }
      }
      __syncthreads();  // the task's stages and row tables are free
    }
  }
  if (tid == 0 && atomicAdd(a.w.dq + 1, 1) == (int)gridDim.x - 1) a.w.dq[0] = 0, a.w.dq[1] = 0;
}

// ---- down ----------------------------------------------------------------------------------------------------------
// CTA: one tile x 128 outputs (16 strips) of the pass; K = 512, 8 k-steps; raw weights loaded once.
// Routed: the raw sums c (NVFP4's carry 2^-30) into yp [pick][column - c0]; shared: scaled per unit into ys.
template <bool SH, int MT>
__global__ void __launch_bounds__(256, 1) k_pdown(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RT = 32 * MT;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int ncb = a.w.cp / 128;
  const int tile = blockIdx.x / ncb, cb = blockIdx.x % ncb;
  int e = 0, pos0, n;
  if (SH) {
    pos0 = tile * RT;
    n = min(RT, a.T - pos0);
    if (n <= 0) return;
  } else {
    if (tile >= *a.w.ntile) return;
    const PTile tt = a.w.tiles[tile];
    e = tt.e, pos0 = tt.pos0, n = tt.n;
  }
  const int ar = min(tid >> 1, n - 1);
  const __nv_bfloat16* asrc = (SH ? a.w.hsh : a.w.h) + (size_t)(pos0 + ar) * kI;
  const int n80 = a.c0 / 8 + 16 * cb + 2 * warp;  // this warp's decode strips n80, n80 + 1
  uint32_t fmt = 0, tab[8] = {};
  float kd = 1.f;
  Raw rb[2];
  Raw8 r8[SH ? 2 : 1][4];
  if (!SH) {
    const MatDesc& d = a.L.desc[e * 3 + 2];
    fmt = d.fmt;
#pragma unroll
    for (int j = 0; j < 8; ++j) tab[j] = d.tab[j];
    const int bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk;
#pragma unroll
    for (int i = 0; i < 2; ++i) load_blk(rb[i], a.L.pool + d.off + (size_t)(n80 + i) * bs, fmt, lane);
    (void)kd;
  } else {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const uint8_t* p = a.L.sh.d + ((size_t)(n80 + i) * 4 + u) * 1024 + 32 * lane;
        r8[i % (SH ? 2 : 1)][u].q0 = ldg_stream128(p);
        r8[i % (SH ? 2 : 1)][u].q1 = ldg_stream128(p + 16);
      }
  }
  auto decode = [&](uint8_t* sb, int step) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t o[8];
      if (SH) dec8((step & 1) ? r8[i % (SH ? 2 : 1)][step >> 1].q1 : r8[i % (SH ? 2 : 1)][step >> 1].q0, o);
      else dec_group(rb[i], step >> 1, step & 1, fmt, tab, o);
      store_b(sb, 2 * warp + i, lane, o);
    }
  };
  const int sj[4] = {4 * wn, 4 * wn + 1, 4 * wn + 2, 4 * wn + 3};
  float acc[MT][4][4], cu[SH ? MT : 1][4][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  uint8_t* sA = smem;
  uint8_t* sB = smem + kNA * kStageA;
#pragma unroll
  for (int j = 0; j < kNA - 1; ++j) {
    if (tid < 2 * RT) load_a(sA + j * kStageA, asrc + 64 * j, tid);
    cp_commit();
  }
  decode(sB, 0);
  cp_wait<kNA - 2>();
  __syncthreads();
  constexpr int NK = kI / 64;  // 8
  const float* dsc = SH ? a.L.sh.ds + ((a.c0 + 128 * cb) / 128) * (kI / 128) : nullptr;
#pragma unroll
  for (int i = 0; i < NK; ++i) {
    const int s = i & 1;
    if (i + kNA - 1 < NK && tid < 2 * RT) load_a(sA + ((i + kNA - 1) % kNA) * kStageA, asrc + 64 * (i + kNA - 1), tid);
    cp_commit();
    if (i + 1 < NK) decode(sB + (s ^ 1) * kStageB, i + 1);
    const uint8_t* sAc = sA + (i % kNA) * kStageA;
    if (SH) {
      if (!(i & 1))
#pragma unroll
        for (int x = 0; x < MT; ++x)
#pragma unroll
          for (int j = 0; j < 4; ++j) cu[x][j][0] = cu[x][j][1] = cu[x][j][2] = cu[x][j][3] = 0.f;
      mma_step<MT>(*reinterpret_cast<float(*)[MT][4][4]>(&cu), sAc, sB + s * kStageB, wm, sj, n, lane);
      if (i & 1) {
        const float sd = __ldg(dsc + (i >> 1));
#pragma unroll
        for (int x = 0; x < MT; ++x)
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[x][j][c] = fmaf(sd, cu[x][j][c], acc[x][j][c]);
      }
    } else {
      mma_step<MT>(acc, sAc, sB + s * kStageB, wm, sj, n, lane);
    }
    cp_wait<kNA - 2>();
    __syncthreads();
  }
#pragma unroll
  for (int x = 0; x < MT; ++x)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = 16 * (MT * wm + x) + g + 8 * hh;
        if (r >= n) continue;
        const int col = 128 * cb + 8 * sj[j] + 2 * t;  // within the pass
        const float2 v = make_float2(acc[x][j][2 * hh], acc[x][j][2 * hh + 1]);
        if (SH) *reinterpret_cast<float2*>(a.ys + (size_t)(pos0 + r) * kH + a.c0 + col) = v;
        else *reinterpret_cast<float2*>(a.w.yp + (size_t)(pos0 + r) * a.w.cp + col) = v;
      }
}


// ---- the FP4-activation path (PrefillCfg::fp4) ----------------------------------------------------------------
// Activations as NVFP4 (formats.cuh, nvfp4_quant16_host: the same steps), the GEMMs on mxf4nvf4 (m16n8k64,
// e2m1 x e2m1 with ue4m3 scales per 16, exact products summed then rounded to f32 toward zero; the f32 k-steps
// summed in order). The weights go to shared memory as natural-order e2m1 nibbles with their scale bytes (S3:
// the code -> nibble table; the subset bit dropped from the scale). The shared expert stays on the bf16 path.
__device__ __forceinline__ uint32_t cvt_e4m3(float v) {
  uint32_t r;
  asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %1; cvt.u32.u16 %0, t; }" : "=r"(r) : "f"(v));
  return r & 0xFFu;
}
__device__ __forceinline__ uint32_t cvt_e2m1x2(float lo, float hi) {  // lo -> low nibble
  uint32_t r;
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(hi), "f"(lo));
  return r & 0xFFu;
}
__device__ __forceinline__ float e4m3_f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}
// a thread a group of 16: rows of K bf16 values -> q [R][K / 2], sf [R][K / 16]
template <int K>
__global__ void k_quant(const __nv_bfloat16* __restrict__ src, int R, float S, uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R * (K / 16)) return;
  const uint4* p = reinterpret_cast<const uint4*>(src + (size_t)i * 16);
  const uint4 u0 = __ldg(p), u1 = __ldg(p + 1);
  const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
  float x[16], amax = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[2 * j] = __uint_as_float(w[j] << 16), x[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
    amax = fmaxf(amax, fmaxf(fabsf(x[2 * j]), fabsf(x[2 * j + 1])));
  }
  const uint32_t sb = cvt_e4m3(__fdiv_rn(amax, __fmul_rn(6.f, S)));
  const float d = __fmul_rn(e4m3_f(sb), S);
  uint32_t o[2] = {0, 0};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float lo = d > 0.f ? __fdiv_rn(x[2 * j], d) : 0.f, hi = d > 0.f ? __fdiv_rn(x[2 * j + 1], d) : 0.f;
    o[j >> 2] |= cvt_e2m1x2(lo, hi) << (8 * (j & 3));
  }
  *reinterpret_cast<uint2*>(q + (size_t)i * 8) = make_uint2(o[0], o[1]);
  sf[i] = (uint8_t)sb;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem) : "memory");
}
__device__ __forceinline__ void mma_nvf4(float d[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
// eight S3 selectors (a nibble each, value order) -> eight e2m1 nibbles; tl0/tl1: the subset's nibbles of codes
// 0-3 / 4-7, a byte each
__device__ __forceinline__ uint32_t s3_nib8(uint32_t sel, uint32_t tl0, uint32_t tl1) {
  const uint32_t b0 = bperm(tl0, tl1, sel & 0xFFFFu), b1 = bperm(tl0, tl1, sel >> 16);
  const uint32_t s0 = bperm(tl0 << 4, tl1 << 4, sel & 0xFFFFu), s1 = bperm(tl0 << 4, tl1 << 4, sel >> 16);
  return bperm(b0, b1, 0x6420) | bperm(s0, s1, 0x7531);
}
__device__ __forceinline__ uint32_t nib_bytes(uint32_t nib, int c0) {
  uint32_t b = 0;
#pragma unroll
  for (int c = 0; c < 4; ++c) b |= ((nib >> (4 * (c0 + c))) & 15u) << (8 * c);
  return b;
}
// stage layout (k-step of 128): A 128 rows x 64 B, its scales 128 x 8 B, B 16 strips x 8 rows x 64 B, scales
// 16 x 8 x 8 B. A row's 16-byte chunk c sits at chunk c ^ ((row >> 1) & 3) (conflict-free fragment reads).
constexpr int kF4A = 128 * 64, kF4As = 128 * 8, kF4B = 16 * 8 * 64, kF4Bs = 16 * 8 * 8;
constexpr int kF4Stage = kF4A + kF4As + kF4B + kF4Bs;
constexpr int kF4Smem = 2 * kF4Stage;
__device__ __forceinline__ int f4_off(int row, int chunk) { return row * 64 + 16 * (chunk ^ ((row >> 1) & 3)); }

// a warp's decode of strips s0, s0 + 1: unit jj of the raw blocks r[] into the stage's B
struct F4Dec {
  uint32_t fmt, tl[2][2];
  __device__ void init(const MatDesc& d) {
    fmt = d.fmt;
#pragma unroll
    for (int s = 0; s < 2; ++s) tl[s][0] = nib_bytes(d.nib[s], 0), tl[s][1] = nib_bytes(d.nib[s], 4);
  }
  __device__ __forceinline__ void run(const Raw& r, int jj, uint8_t* sb, uint8_t* sbs, int strip, int lane) const {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      uint32_t sc = ((jj < 2 ? r.s.x : r.s.y) >> (8 * (2 * (jj & 1) + q))) & 0xFFu;
      uint2 v;
      if (fmt == kNVFP4) {
        v = make_uint2(word(r.v[jj], 2 * q), word(r.v[jj], 2 * q + 1));
      } else {
        const int hi = sc >> 7;
        const uint32_t P0 = word(r.v[0], jj), P1 = word(r.v[1], jj), Q = word(r.v[2], jj);
        v.x = s3_nib8(s3_sel(P0, P1, Q, 2 * q), tl[hi][0], tl[hi][1]);
        v.y = s3_nib8(s3_sel(P0, P1, Q, 2 * q + 1), tl[hi][0], tl[hi][1]);
        sc &= 0x7Fu;
      }
      const int row = strip * 8 + g;
      *reinterpret_cast<uint2*>(sb + f4_off(row, 2 * q + (t >> 1)) + 8 * (t & 1)) = v;
      sbs[row * 8 + 4 * q + t] = (uint8_t)sc;
    }
  }
};
// A of a k-step: thread tid copies row tid >> 1's chunks 2 (tid & 1), + 1 and (even tid) its 8 scale bytes
__device__ __forceinline__ void f4_load_a(uint8_t* st, const uint8_t* q, const uint8_t* s, int tid) {
  const int r = tid >> 1, h = tid & 1;
  cp_async16(st + f4_off(r, 2 * h), q + 32 * h);
  cp_async16(st + f4_off(r, 2 * h + 1), q + 32 * h + 16);
  if (!h) cp_async8(st + kF4A + 8 * r, s);
}
// one k-step of a warp: m16 tiles from row 16 MT wm (those at or past n skipped), strips sj[]
template <int MT>
__device__ __forceinline__ void f4_mma(float (&acc)[MT][4][4], const uint8_t* st, int wm, const int (&sj)[4], int n, int lane) {
  if (16 * MT * wm >= n) return;
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* sB = st + kF4A + kF4As;
  const uint8_t* sBs = sB + kF4B;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t b[4][2], bs[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = sj[j] * 8 + g;
      b[j][0] = *reinterpret_cast<const uint32_t*>(sB + f4_off(row, 2 * h) + 4 * t);
      b[j][1] = *reinterpret_cast<const uint32_t*>(sB + f4_off(row, 2 * h + 1) + 4 * t);
      bs[j] = *reinterpret_cast<const uint32_t*>(sBs + row * 8 + 4 * h);
    }
#pragma unroll
    for (int ti = 0; ti < MT; ++ti) {
      const int r0 = 16 * (MT * wm + ti);
      uint32_t af[4];
      af[0] = *reinterpret_cast<const uint32_t*>(st + f4_off(r0 + g, 2 * h) + 4 * t);
      af[1] = *reinterpret_cast<const uint32_t*>(st + f4_off(r0 + g + 8, 2 * h) + 4 * t);
      af[2] = *reinterpret_cast<const uint32_t*>(st + f4_off(r0 + g, 2 * h + 1) + 4 * t);
      af[3] = *reinterpret_cast<const uint32_t*>(st + f4_off(r0 + g + 8, 2 * h + 1) + 4 * t);
      const uint32_t as = *reinterpret_cast<const uint32_t*>(st + kF4A + (r0 + g + 8 * (t & 1)) * 8 + 4 * h);
#pragma unroll
      for (int j = 0; j < 4; ++j) mma_nvf4(acc[ti][j], af, b[j][0], b[j][1], as, bs[j]);
    }
  }
}

template <int MT>
__global__ void __launch_bounds__(256, 1) k_pup_f4(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int tile = blockIdx.x / kUpNB, nb = blockIdx.x % kUpNB;
  if (tile >= *a.w.ntile) return;
  const PTile tt = a.w.tiles[tile];
  const int e = tt.e, pos0 = tt.pos0, n = tt.n;
  const int row = a.w.perm[pos0 + min(tid >> 1, n - 1)];
  const uint8_t* aq = a.w.xq + (size_t)row * (kH / 2);
  const uint8_t* as = a.w.xs + (size_t)row * (kH / 16);
  const int mat = warp >= 4;
  const int n80 = nb * 8 + (2 * warp) % 8;
  const MatDesc* d = a.L.desc + e * 3;
  F4Dec dec;
  dec.init(d[mat]);
  const int bs = dec.fmt == kNVFP4 ? kQ4Blk : kS3Blk;
  const uint8_t* wb = a.L.pool + d[mat].off + (size_t)n80 * (kH / kBlkK) * bs;
  const float kg = d[0].s2 * a.in_gu, ku = d[1].s2 * a.in_gu;
  Raw rb[2][2];
  auto load_raw = [&](int b, int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) load_blk(rb[buf][i], wb + ((size_t)i * (kH / kBlkK) + b) * bs, dec.fmt, lane);
  };
  constexpr int NK = kH / 128;  // 48 k-steps, 4 a block
  auto decode = [&](uint8_t* st, int step, int ph) {  // ph = step % 8, compile-time at every call
#pragma unroll
    for (int i = 0; i < 2; ++i) dec.run(rb[ph >> 2][i], ph & 3, st + kF4A + kF4As, st + kF4A + kF4As + kF4B, 2 * warp + i, lane);
    if ((ph & 3) == 3 && (step >> 2) + 2 < NK / 4) load_raw((step >> 2) + 2, ph >> 2);
  };
  const int sj[4] = {2 * wn, 2 * wn + 1, 8 + 2 * wn, 9 + 2 * wn};
  float acc[MT][4][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  load_raw(0, 0);
  load_raw(1, 1);
  f4_load_a(smem, aq, as, tid);
  cp_commit();
  decode(smem, 0, 0);
  cp_wait0();
  __syncthreads();
#pragma unroll 1
  for (int i0 = 0; i0 < NK; i0 += 8)
#pragma unroll
    for (int ii = 0; ii < 8; ++ii) {
      const int i = i0 + ii;
      uint8_t* cur = smem + (ii & 1) * kF4Stage;
      uint8_t* nxt = smem + ((ii & 1) ^ 1) * kF4Stage;
      if (i + 1 < NK) {
        f4_load_a(nxt, aq + 64 * (i + 1), as + 8 * (i + 1), tid);
        cp_commit();
        decode(nxt, i + 1, (ii + 1) & 7);
      }
      f4_mma<MT>(acc, cur, wm, sj, n, lane);
      cp_wait0();
      __syncthreads();
    }
#pragma unroll
  for (int x = 0; x < MT; ++x)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = 16 * (MT * wm + x) + g + 8 * hh;
        if (r >= n) continue;
        float hv[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const float gt = __bfloat162float(__float2bfloat16_rn(acc[x][j][2 * hh + c] * kg));
          const float ut = __bfloat162float(__float2bfloat16_rn(acc[x][j + 2][2 * hh + c] * ku));
          hv[c] = silu_mul(gt, ut);
        }
        const int ch = 8 * (nb * 8 + 2 * wn + j) + 2 * t;
        *reinterpret_cast<__nv_bfloat162*>(a.w.h + (size_t)(pos0 + r) * kI + ch) = __floats2bfloat162_rn(hv[0], hv[1]);
      }
}

// down: K = 512, one raw block a strip loaded at the start, 4 k-steps; the sums times the down input scale into yp
template <int MT>
__global__ void __launch_bounds__(256, 1) k_pdown_f4(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int ncb = a.w.cp / 128;
  const int tile = blockIdx.x / ncb, cb = blockIdx.x % ncb;
  if (tile >= *a.w.ntile) return;
  const PTile tt = a.w.tiles[tile];
  const int e = tt.e, pos0 = tt.pos0, n = tt.n;
  const int ar = pos0 + min(tid >> 1, n - 1);
  const uint8_t* aq = a.w.hq + (size_t)ar * (kI / 2);
  const uint8_t* as = a.w.hs + (size_t)ar * (kI / 16);
  const int n80 = a.c0 / 8 + 16 * cb + 2 * warp;
  const MatDesc& d = a.L.desc[e * 3 + 2];
  F4Dec dec;
  dec.init(d);
  const int bs = dec.fmt == kNVFP4 ? kQ4Blk : kS3Blk;
  Raw rb[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) load_blk(rb[i], a.L.pool + d.off + (size_t)(n80 + i) * bs, dec.fmt, lane);
  auto decode = [&](uint8_t* st, int jj) {
#pragma unroll
    for (int i = 0; i < 2; ++i) dec.run(rb[i], jj, st + kF4A + kF4As, st + kF4A + kF4As + kF4B, 2 * warp + i, lane);
  };
  const int sj[4] = {4 * wn, 4 * wn + 1, 4 * wn + 2, 4 * wn + 3};
  float acc[MT][4][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  f4_load_a(smem, aq, as, tid);
  cp_commit();
  decode(smem, 0);
  cp_wait0();
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint8_t* cur = smem + (i & 1) * kF4Stage;
    uint8_t* nxt = smem + ((i & 1) ^ 1) * kF4Stage;
    if (i + 1 < 4) {
      f4_load_a(nxt, aq + 64 * (i + 1), as + 8 * (i + 1), tid);
      cp_commit();
      decode(nxt, i + 1);
    }
    f4_mma<MT>(acc, cur, wm, sj, n, lane);
    cp_wait0();
    __syncthreads();
  }
#pragma unroll
  for (int x = 0; x < MT; ++x)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = 16 * (MT * wm + x) + g + 8 * hh;
        if (r >= n) continue;
        const int col = 128 * cb + 8 * sj[j] + 2 * t;
        *reinterpret_cast<float2*>(a.w.yp + (size_t)(pos0 + r) * a.w.cp + col) =
            make_float2(acc[x][j][2 * hh] * a.in_dn, acc[x][j][2 * hh + 1] * a.in_dn);
      }
}

// a row's routed output for the pass's columns: its picks in expert-id order, acc += (s2 * w) * c, as k_down
__global__ void k_combine(PArgs a) {
  const int c4 = a.w.cp / 4;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.T * c4) return;
  const int t = i / c4, c = 4 * (i % c4);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const int e = a.w.re[t * kTopK + k];
    const MatDesc& d = a.L.desc[e * 3 + 2];
    const float kk = __ldg(&d.s2) * (__ldg(&d.fmt) == kNVFP4 && !a.fp4 ? 0x1p30f : 1.f);
    const float k0 = kk * a.w.rw[t * kTopK + k];
    const float4 v = __ldcs(reinterpret_cast<const float4*>(a.w.yp + (size_t)a.w.rpos[t * kTopK + k] * a.w.cp + c));
    acc.x = fmaf(k0, v.x, acc.x);
    acc.y = fmaf(k0, v.y, acc.y);
    acc.z = fmaf(k0, v.z, acc.z);
    acc.w = fmaf(k0, v.w, acc.w);
  }
  *reinterpret_cast<float4*>(a.yr + (size_t)t * kH + a.c0 + c) = acc;
}

template <class K>
void set_smem(K kernel, int bytes) {
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

}  // namespace

size_t prefill_ws_bytes(int T) { return ws_layout(nullptr, T).bytes; }
PrefillView prefill_view(void* ws, int T) {
  const Ws w = ws_layout(ws, T);
  return PrefillView{w.rpos, w.re, w.h, w.hsh, w.xq, w.xs, w.hq, w.hs};
}

// The chained down's pass: the widest multiple of 256 columns dividing kH whose yr slice (T x cpc f32) fits
// the L2 budget.
constexpr int kDefYrKB = 16 << 10;
static int chain_cols(int T, int yr_kb) {
  const size_t budget = (size_t)(yr_kb > 0 ? yr_kb : kDefYrKB) << 10;
  for (int m : {24, 12, 8, 6, 4, 3, 2})
    if ((size_t)T * 256 * m * 4 <= budget) return 256 * m;
  return 256;
}

void moe_prefill_phases(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws,
                        float* yr, float* ys, cudaStream_t st, const PrefillCfg& cfg, int ph) {
  if (T < 1 || T > kMaxChunk) die("moe_prefill: T=%d", T);
  static bool init = false;
  static int nsm = 0;
  static thread_local cudaEvent_t ev[2] = {nullptr, nullptr};
  if (!init) {
    set_smem(k_pup<false, 4>, kSmem);
    set_smem(k_pup<false, 4, 1>, kSmem);
    set_smem(k_pup<false, 4, 2>, kSmem);
    set_smem(k_pup<false, 4, 3>, kSmem);
    set_smem(k_pup<false, 4, 4>, kSmem);
    set_smem(k_pup<false, 4, 5>, kSmem);
    set_smem(k_pup<false, 4, 6>, kSmem);
    set_smem(k_pup<true, 2>, kSmem);
    set_smem(k_pdown_r<4>, kSmem);
    set_smem(k_pdown_c<4>, kSmem);
    set_smem(k_pdown_cw, kWsSmem);
    set_smem(k_pup_ws<false, 0>, kWsSmem);
    set_smem(k_pup_ws<true>, kWsSmemSh);
    set_smem(k_pup_w2<0>, kWsSmem);
    set_smem(k_pup_w2<1>, kWsSmem);
    set_smem(k_pup_w2<2>, kWsSmem);
    set_smem(k_pup_w2<3>, kWsSmem);
    set_smem(k_pup_w2<4>, kWsSmem);
    set_smem(k_pup_w2<5>, kWsSmem);
    set_smem(k_pup_w2<6>, kWsSmem);
    set_smem(k_pup_ws<false, 1>, kWsSmem);
    set_smem(k_pup_ws<false, 2>, kWsSmem);
    set_smem(k_pup_ws<false, 3>, kWsSmem);
    set_smem(k_pup_ws<false, 4>, kWsSmem);
    set_smem(k_pup_ws<false, 5>, kWsSmem);
    set_smem(k_pup_ws<false, 6>, kWsSmem);
    set_smem(k_pdown<true, 2>, kSmem);
    set_smem(k_pup_f4<4>, kF4Smem);
    set_smem(k_pdown_f4<4>, kF4Smem);
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    init = true;
  }
  const bool chain = !cfg.fp4 && cfg.down >= 1;
  PArgs a{L, x, T, ws_layout(ws, T), yr, ys, 0, cfg.fp4, cfg.in_scale_gu, cfg.in_scale_dn, 0, 0, 1};
  a.cq = std::max(1, cfg.chain_q);
  a.cpc = chain_cols(T, (cfg.yr_kb > 0 ? cfg.yr_kb : kDefYrKB) / a.cq);
  a.npc = kH / a.cpc;
  while (a.npc % a.cq) --a.cq;
  const int P = T * kTopK;
  const int maxt = std::min(kMaxRTiles, P / kTM + kE);
  // the shared expert: needs only x, so it may run beside the routed experts on cfg.side
  const bool sh = cfg.shared && (ph & kPhShared);
  const bool fork = sh && cfg.side;
  if (fork) {
    if (!ev[0])
      for (auto& e : ev) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ev[0], st));
    CUDA_CHECK(cudaStreamWaitEvent(cfg.side, ev[0], 0));
  }
  if (sh) {
    const cudaStream_t ss = fork ? cfg.side : st;
    PArgs b = a;
    if (cfg.up == 1) k_pup_ws<true><<<(T + 63) / 64 * kUpNB, 512, kWsSmemSh, ss>>>(b);
    else k_pup<true, 2><<<(T + 63) / 64 * kUpNB, 256, kSmem, ss>>>(b);
    for (int p = 0; p < b.w.npass; ++p) {
      b.c0 = p * b.w.cp;
      k_pdown<true, 2><<<(T + 63) / 64 * (b.w.cp / 128), 256, kSmem, ss>>>(b);
    }
    if (fork) CUDA_CHECK(cudaEventRecord(ev[1], cfg.side));
  }
  if (ph & kPhBucket) {
    k_count<<<(P + 1023) / 1024, 1024, 0, st>>>(idx, P, a.w.cnt);
    k_scan<<<1, 256, 0, st>>>(a.w);
    k_scatter<<<(T + 127) / 128, 128, 0, st>>>(idx, w, T, a.w);
  }
  if (ph & kPhUp) {
    if (cfg.fp4) {
      k_quant<kH><<<(T * (kH / 16) + 255) / 256, 256, 0, st>>>(x, T, cfg.in_scale_gu, a.w.xq, a.w.xs);
      k_pup_f4<4><<<maxt * kUpNB, 256, kF4Smem, st>>>(a);
      k_quant<kI><<<(P * (kI / 16) + 255) / 256, 256, 0, st>>>(a.w.h, P, cfg.in_scale_dn, a.w.hq, a.w.hs);
    } else if (cfg.up == 2) {
      const int gu = maxt * kUpNB;
      switch (cfg.diag) {
        case 0: k_pup_w2<0><<<gu, 256, kWsSmem, st>>>(a); break;
        case 1: k_pup_w2<1><<<gu, 256, kWsSmem, st>>>(a); break;
        case 2: k_pup_w2<2><<<gu, 256, kWsSmem, st>>>(a); break;
        case 3: k_pup_w2<3><<<gu, 256, kWsSmem, st>>>(a); break;
        case 4: k_pup_w2<4><<<gu, 256, kWsSmem, st>>>(a); break;
        case 5: k_pup_w2<5><<<gu, 256, kWsSmem, st>>>(a); break;
        case 6: k_pup_w2<6><<<gu, 256, kWsSmem, st>>>(a); break;
        default: die("moe_prefill: diag %d", cfg.diag);
      }
    } else if (cfg.up == 1) {
      const int gu = maxt * kUpNB;
      switch (cfg.diag) {
        case 0: k_pup_ws<false, 0><<<gu, 512, kWsSmem, st>>>(a); break;
        case 1: k_pup_ws<false, 1><<<gu, 512, kWsSmem, st>>>(a); break;
        case 2: k_pup_ws<false, 2><<<gu, 512, kWsSmem, st>>>(a); break;
        case 3: k_pup_ws<false, 3><<<gu, 512, kWsSmem, st>>>(a); break;
        case 4: k_pup_ws<false, 4><<<gu, 512, kWsSmem, st>>>(a); break;
        case 5: k_pup_ws<false, 5><<<gu, 512, kWsSmem, st>>>(a); break;
        case 6: k_pup_ws<false, 6><<<gu, 512, kWsSmem, st>>>(a); break;
        default: die("moe_prefill: diag %d", cfg.diag);
      }
    } else {
      switch (cfg.diag) {
        case 0: k_pup<false, 4><<<maxt * kUpNB, 256, kSmem, st>>>(a); break;
        case 1: k_pup<false, 4, 1><<<maxt * kUpNB, 256, kSmem, st>>>(a); break;
        case 2: k_pup<false, 4, 2><<<maxt * kUpNB, 256, kSmem, st>>>(a); break;
        case 3: k_pup<false, 4, 3><<<maxt * kUpNB, 256, kSmem, st>>>(a); break;
        case 4: k_pup<false, 4, 4><<<maxt * kUpNB, 256, kSmem, st>>>(a); break;
        case 5: k_pup<false, 4, 5><<<maxt * kUpNB, 256, kSmem, st>>>(a); break;
        case 6: k_pup<false, 4, 6><<<maxt * kUpNB, 256, kSmem, st>>>(a); break;
        default: die("moe_prefill: diag %d", cfg.diag);
      }
    }
  }
  if (chain) {
    if (ph & kPhDown) {
      if (cfg.down == 2) k_pdown_cw<<<std::min(nsm, maxt * a.npc), 512, kWsSmem, st>>>(a);
      else k_pdown_c<4><<<std::min(nsm, maxt * a.npc), 256, kSmem, st>>>(a);
    }
  } else {
    for (int p = 0; p < a.w.npass; ++p) {
      a.c0 = p * a.w.cp;
      if (ph & kPhDown) {
        if (cfg.fp4) k_pdown_f4<4><<<maxt * (a.w.cp / 128), 256, kF4Smem, st>>>(a);
        else k_pdown_r<4><<<maxt * (a.w.cp / 128 / kDnCB), 256, kSmem, st>>>(a);
      }
      if (ph & kPhCombine) k_combine<<<(T * (a.w.cp / 4) + 255) / 256, 256, 0, st>>>(a);
    }
  }
  if (fork) CUDA_CHECK(cudaStreamWaitEvent(st, ev[1], 0));
  CUDA_CHECK(cudaGetLastError());
}

void moe_prefill(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws, float* yr,
                 float* ys, cudaStream_t st, const PrefillCfg& cfg) {
  moe_prefill_phases(L, x, idx, w, T, ws, yr, ys, st, cfg, kPhAll);
}

}  // namespace gx
