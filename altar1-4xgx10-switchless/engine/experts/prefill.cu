// The prefill expert GEMMs (docs/experts.md, "Prefill"): a chunk's picks bucketed by expert, gate/up and down
// as grouped GEMMs on the bf16 tensor cores (m16n8k16, f32 accumulate), the weights dequantised into shared
// memory in the B-fragment order of the decode kernels (the same decode functions F::dec, so the same bf16
// values: INT4 dequantised by the contract's rule, BF16 copied), each pick's down output kept in f32 and
// combined per row in expert-id order with the decode step's arithmetic: so a row's result equals the decode
// path's up to the order of the K sums. A fused FP8 shared expert (GLM-5.3's; altar-1 has none) as a dense FP8
// GEMM over the chunk. Every routed kernel is templated on the layer's format F (formats.cuh); a raw block is
// F::kBK k, KS = F::kBK / 64 k-steps (8 INT4/NVFP4, 2 BF16), and the prefetch runs a group of KS k-steps ahead.
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
  float kmul;        // the format's epilogue factor besides s2 (F::kMul: NVFP4's 2^30, else 1)
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
// one CTA of kE threads (a thread an expert): offsets, the tile list in expert order, the scatter cursors;
// clears cnt for the next call
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
template <class F, bool SH, int MT, int DG = 0>
__global__ void __launch_bounds__(256, 1) k_pup(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RT = 32 * MT;  // rows a tile
  constexpr int KS = F::kBK / 64, NBK = kH / F::kBK;  // k-steps a raw block, raw blocks along K
  using Raw = typename F::Raw;
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
  const uint8_t* wb = nullptr;
  float kmat[2] = {1.f, 1.f};
  if (!SH) {
    const MatDesc* d = a.L.desc + e * 3;
    if (d[0].fmt != F::kFmt || d[1].fmt != F::kFmt) __trap();
    wb = a.L.pool + d[mat].off + (size_t)n80 * NBK * F::kBlk;
    kmat[0] = d[0].s2 * F::kMul;
    kmat[1] = d[1].s2 * F::kMul;
  } else {
    wb = (mat ? a.L.sh.u : a.L.sh.g) + (size_t)n80 * (kH / 128) * 1024;
  }
  constexpr int NK = kH / 64;  // 96 k-steps
  Raw rb[2][2];                // routed: blocks b, b + 1 of the two strips
  Raw8 r8[2][2];               // shared: units u, u + 1
  auto load_raw = [&](int b, int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) F::load(rb[buf][i], wb + ((size_t)i * NBK + b) * F::kBlk, lane);
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
      else F::dec(rb[(ph / KS) & 1][i], (ph % KS) >> 1, ph & 1, o);
      store_b(sb, 2 * warp + i, lane, o);
    }
    if (SH) {
      if ((ph & 1) && (step >> 1) + 2 < NK / 2) load_r8((step >> 1) + 2, (ph >> 1) & 1);
    } else {
      if (ph % KS == KS - 1 && step / KS + 2 < NK / KS) load_raw(step / KS + 2, (ph / KS) & 1);
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
template <class F, bool SH, int DG = 0>
__global__ void __launch_bounds__(512, 1) k_pup_ws(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int KS = F::kBK / 64, NBK = kH / F::kBK;  // k-steps a raw block, raw blocks along K
  using Raw = typename F::Raw;
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
    const uint8_t* wb;
    if (!SH) {
      if (d[0].fmt != F::kFmt || d[1].fmt != F::kFmt) __trap();
      wb = a.L.pool + d[mat].off + (size_t)n80 * NBK * F::kBlk;
    } else {
      wb = (mat ? a.L.sh.u : a.L.sh.g) + (size_t)n80 * (kH / 128) * 1024;
    }
    Raw rb[SH ? 1 : 2][2];
    Raw8 r8[SH ? 2 : 1][2];
    auto load_raw = [&](int b, int buf) {
#pragma unroll
      for (int i = 0; i < 2; ++i) F::load(rb[buf % (SH ? 1 : 2)][i], wb + ((size_t)i * NBK + b) * F::kBlk, lane);
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
            else F::dec(rb[(ii / KS) & 1][x], (ii % KS) >> 1, ii & 1, o);
            store_b(sB(s_cur), 2 * pw + x, lane, o);
          }
          if (SH) {
            if ((ii & 1) && (i >> 1) + 2 < NK / 2) load_r8((i >> 1) + 2, (ii >> 1) & 1);
          } else {
            if (ii % KS == KS - 1 && i / KS + 2 < NK / KS) load_raw(i / KS + 2, (ii / KS) & 1);
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
  if (!SH) kg = d[0].s2 * F::kMul, ku = d[1].s2 * F::kMul;
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

// ---- routed down ---------------------------------------------------------------------------------------------------
// CTA: one tile x kDnCB column blocks of 128 outputs of the pass, one after another as one pipeline of
// kDnCB x 8 k-steps (K = 512 a block): the A ring runs on across the blocks (the same h rows), the next blocks'
// raw weights load while the current ones are decoded, and each block's sums go out when its last k-step is
// done: the raw sums c (NVFP4's carry 2^-30) into yp [pick][column - c0].
constexpr int kDnCB = 4;
template <class F, int MT>
__global__ void __launch_bounds__(256, 1) k_pdown_r(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int RT = 32 * MT;
  constexpr int KS = F::kBK / 64, GPB = 8 / KS;  // k-steps a raw block, raw blocks a strip (K = 512)
  using Raw = typename F::Raw;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int ngrp = a.w.cp / 128 / kDnCB;
  const int tile = blockIdx.x / ngrp, cg = blockIdx.x % ngrp;
  if (tile >= *a.w.ntile) return;
  const PTile tt = a.w.tiles[tile];
  const int e = tt.e, pos0 = tt.pos0, n = tt.n;
  const __nv_bfloat16* asrc = a.w.h + (size_t)(pos0 + min(tid >> 1, n - 1)) * kI;
  const MatDesc& d = a.L.desc[e * 3 + 2];
  if (d.fmt != F::kFmt) __trap();
  const uint8_t* wb = a.L.pool + d.off + (size_t)(a.c0 / 8 + 16 * kDnCB * cg + 2 * warp) * GPB * F::kBlk;
  Raw rb[2][2];  // [group & 1][strip]; group G: column block G / GPB, block G % GPB of each strip
  auto load_raw = [&](int G, int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) F::load(rb[buf][i], wb + ((size_t)(16 * (G / GPB) + i) * GPB + G % GPB) * F::kBlk, lane);
  };
  constexpr int NK = kDnCB * 8;
  static_assert(NK % 16 == 0);
  auto decode = [&](uint8_t* sb, int step, int ph) {  // ph = step % 16, compile-time
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      uint32_t o[8];
      F::dec(rb[(ph / KS) & 1][i], (ph % KS) >> 1, ph & 1, o);
      store_b(sb, 2 * warp + i, lane, o);
    }
    if (ph % KS == KS - 1 && step / KS + 2 < NK / KS) load_raw(step / KS + 2, (ph / KS) & 1);
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
// DG (bench diagnostics, wrong results): 1 no weights, 2 no MMAs, 4 no A loads, 8 no adds into yr (no waits, loads,
// stores or tickets), 16 no waits (the adds unordered, no tickets), 32 no yr loads, 64 no yr stores
template <class F, int MT, int DG = 0>
__global__ void __launch_bounds__(256, 1) k_pdown_c(PArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int KS = F::kBK / 64, GPB = 8 / KS;  // k-steps a raw block, raw blocks a strip (K = 512)
  using Raw = typename F::Raw;
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
    if (d.fmt != F::kFmt) __trap();
    if (tid < n) {
      const float kk = __ldg(&d.s2) * F::kMul;  // as k_combine
      s_row[tid] = a.w.perm[pos0 + tid];
      s_rank[tid] = a.w.prank[pos0 + tid];
      s_k0[tid] = kk * a.w.pw[pos0 + tid];
    }
    uint32_t ri[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) ri[j] = pos0 + min((tid >> 3) + 32 * j, n - 1);
    const uint8_t* wb = a.L.pool + d.off + (size_t)(c0 / 8 + 2 * warp) * GPB * F::kBlk;
    const int ng = a.cpc / 128 * GPB;  // raw groups of the task
    Raw rb[2][2];  // [group & 1][strip]; group G: column block G / GPB, block G % GPB of each strip
    auto load_raw = [&](int G, int buf) {
#pragma unroll
      for (int i = 0; i < 2; ++i) F::load(rb[buf][i], wb + ((size_t)(16 * (G / GPB) + i) * GPB + G % GPB) * F::kBlk, lane);
    };
    auto decode = [&](uint8_t* sb, int step, int ph) {  // ph = step % 16, compile-time
      if (DG & 1) return;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        uint32_t o[8];
        F::dec(rb[(ph / KS) & 1][i], (ph % KS) >> 1, ph & 1, o);
        store_b(sb, 2 * warp + i, lane, o);
      }
      if (ph % KS == KS - 1 && step / KS + 2 < ng) load_raw(step / KS + 2, (ph / KS) & 1);
    };
    float acc[MT][4][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    if (!(DG & 1)) load_raw(0, 0), load_raw(1, 1);
#pragma unroll
    for (int j = 0; j < kNA - 1; ++j) {
      if (!(DG & 4)) load_a_co<4, 256, kI>(sA + j * kStageA, a.w.h, ri, 64 * j, tid);
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
        if (!(DG & 4) && i + kNA - 1 < NK) load_a_co<4, 256, kI>(sA + sa_ld * kStageA, a.w.h, ri, 64 * ((i + kNA - 1) & 7), tid);
        cp_commit();
        if (i + 1 < NK) decode(sB + (s ^ 1) * kStageB, i + 1, (ii + 1) & 15);
        if (!(DG & 2)) mma_step<MT>(acc, sA + sa_cur * kStageA, sB + s * kStageB, wm, sj, n, lane);
        sa_cur = sa_cur == kNA - 1 ? 0 : sa_cur + 1;
        if ((ii & 7) == 7) {  // column block i >> 3 done: each warp adds its 64 rows x 32 columns into yr
          const int cb = i >> 3, r0 = 64 * wm;
          if (!(DG & 8) && r0 < n) {
            int* tick = a.w.tick + (size_t)((c0 + 128 * cb) / 32 + wn) * kMaxChunk;
            // wait until the rows' lower picks are in (lane l polls rows r0 + l, r0 + l + 32; relaxed loads,
            // then a fence)
            uint32_t pend = 0;
#pragma unroll
            for (int m = 0; m < 2; ++m)
              if (r0 + lane + 32 * m < n && s_rank[r0 + lane + 32 * m] > 0) pend |= 1u << m;
            if (!(DG & 16) && __any_sync(0xffffffffu, pend)) {
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
                const bool rd = !(DG & 32) && live[hh] && s_rank[live[hh] ? r : 0] > 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[hh][j] = rd ? ld_yr(yrow[hh] + 8 * sj[j], pol) : make_float2(0.f, 0.f);
              }
#pragma unroll
              for (int hh = 0; hh < 2; ++hh)
                if (!(DG & 64) && live[hh]) {
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
              if (!(DG & 16) && r < n) st_relaxed(tick + s_row[r], s_rank[r] == kTopK - 1 ? 0 : s_rank[r] + 1);
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

// ---- the fused FP8 shared expert's down --------------------------------------------------------------------------
// CTA: 32 MT rows of the chunk x 128 outputs (16 strips) of the pass; K = 512, 8 k-steps; raw weights loaded
// once; scaled per unit into ys. (SH is always true: GLM-5.3's routed form of this kernel is not used.)
template <bool SH, int MT>
__global__ void __launch_bounds__(256, 1) k_pdown(PArgs a) {
  static_assert(SH, "k_pdown: the shared expert only");
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
  (void)e;
  Raw8 r8[SH ? 2 : 1][4];
  {
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
      dec8((step & 1) ? r8[i % (SH ? 2 : 1)][step >> 1].q1 : r8[i % (SH ? 2 : 1)][step >> 1].q0, o);
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
    const float kk = __ldg(&d.s2) * a.kmul;
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
  return PrefillView{w.rpos, w.re, w.h, w.hsh};
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

template <class F>
void set_smem_fmt() {
  set_smem(k_pup<F, false, 4>, kSmem);
  set_smem(k_pup_ws<F, false, 0>, kWsSmem);
  set_smem(k_pdown_r<F, 4>, kSmem);
  set_smem(k_pdown_c<F, 4>, kSmem);
}
// The routed phases for format F. diag / ddiag (bench diagnostics) only for INT4, the format they are measured on.
template <class F>
void prefill_routed(const PArgs& a0, const PrefillCfg& cfg, const int* idx, const float* w, int ph, int maxt, int nsm,
                    cudaStream_t st) {
  PArgs a = a0;
  a.kmul = F::kMul;
  const int T = a.T, P = T * kTopK;
  constexpr bool kI4 = F::kFmt == kINT4;
  if ((cfg.diag || cfg.ddiag) && !kI4) die("moe_prefill: diagnostics are built for INT4 layers only");
  if (ph & kPhBucket) {
    k_count<<<(P + 1023) / 1024, 1024, 0, st>>>(idx, P, a.w.cnt);
    k_scan<<<1, kE, 0, st>>>(a.w);
    k_scatter<<<(T + 127) / 128, 128, 0, st>>>(idx, w, T, a.w);
  }
  if (ph & kPhUp) {
    const int gu = maxt * kUpNB;
    if (cfg.up == 1) {
      switch (kI4 ? cfg.diag : 0) {
        case 0: k_pup_ws<F, false, 0><<<gu, 512, kWsSmem, st>>>(a); break;
        case 1: k_pup_ws<F, false, kI4 ? 1 : 0><<<gu, 512, kWsSmem, st>>>(a); break;
        case 2: k_pup_ws<F, false, kI4 ? 2 : 0><<<gu, 512, kWsSmem, st>>>(a); break;
        case 4: k_pup_ws<F, false, kI4 ? 4 : 0><<<gu, 512, kWsSmem, st>>>(a); break;
        default: die("moe_prefill: diag %d", cfg.diag);
      }
    } else if (cfg.up == 0) {
      if (cfg.diag) die("moe_prefill: diag with up 0");
      k_pup<F, false, 4><<<gu, 256, kSmem, st>>>(a);
    } else {
      die("moe_prefill: up variant %d (0 or 1)", cfg.up);
    }
  }
  if (cfg.down == 1) {
    if (ph & kPhDown) {
      const int g = std::min(nsm, maxt * a.npc);
      switch (kI4 ? cfg.ddiag : 0) {
        case 0: k_pdown_c<F, 4><<<g, 256, kSmem, st>>>(a); break;
        case 8: k_pdown_c<F, 4, kI4 ? 8 : 0><<<g, 256, kSmem, st>>>(a); break;
        case 16: k_pdown_c<F, 4, kI4 ? 16 : 0><<<g, 256, kSmem, st>>>(a); break;
        default: die("moe_prefill: ddiag %d", cfg.ddiag);
      }
    }
  } else if (cfg.down == 0) {
    for (int p = 0; p < a.w.npass; ++p) {
      a.c0 = p * a.w.cp;
      if (ph & kPhDown) k_pdown_r<F, 4><<<maxt * (a.w.cp / 128 / kDnCB), 256, kSmem, st>>>(a);
      if (ph & kPhCombine) k_combine<<<(T * (a.w.cp / 4) + 255) / 256, 256, 0, st>>>(a);
    }
  } else {
    die("moe_prefill: down variant %d (0 or 1)", cfg.down);
  }
}

void moe_prefill_phases(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws,
                        float* yr, float* ys, cudaStream_t st, const PrefillCfg& cfg, int ph) {
  if (T < 1 || T > kMaxChunk) die("moe_prefill: T=%d", T);
  if (!fmt_known(L.fmt)) die("moe_prefill: layer expert format %u is not NVFP4 (0), INT4 (2) or BF16 (3)", L.fmt);
  static bool init = false;
  static int nsm = 0;
  static thread_local cudaEvent_t ev[2] = {nullptr, nullptr};
  if (!init) {
    set_smem_fmt<FmtINT4>();
    set_smem_fmt<FmtBF16>();
    set_smem_fmt<FmtNVFP4>();
    set_smem(k_pup_ws<FmtINT4, false, 1>, kWsSmem);
    set_smem(k_pup_ws<FmtINT4, false, 2>, kWsSmem);
    set_smem(k_pup_ws<FmtINT4, false, 4>, kWsSmem);
    set_smem(k_pdown_c<FmtINT4, 4, 8>, kSmem);
    set_smem(k_pdown_c<FmtINT4, 4, 16>, kSmem);
    set_smem(k_pup<FmtINT4, true, 2>, kSmem);
    set_smem(k_pup_ws<FmtINT4, true>, kWsSmemSh);
    set_smem(k_pdown<true, 2>, kSmem);
    set_smem(k_pdown<true, 4>, kSmem);
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    init = true;
  }
  PArgs a{L, x, T, ws_layout(ws, T), yr, ys, 0, 1.f, 0, 0, 1};
  a.cq = std::max(1, cfg.chain_q);
  a.cpc = chain_cols(T, (cfg.yr_kb > 0 ? cfg.yr_kb : kDefYrKB) / a.cq);
  a.npc = kH / a.cpc;
  while (a.npc % a.cq) --a.cq;
  const int P = T * kTopK;
  const int maxt = std::min(kMaxRTiles, P / kTM + kE);
  // a fused FP8 shared expert: needs only x, so it may run beside the routed experts on cfg.side
  const bool sh = cfg.shared && L.sh.fused() && (ph & kPhShared);
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
    if (cfg.up == 1) k_pup_ws<FmtINT4, true><<<(T + 63) / 64 * kUpNB, 512, kWsSmemSh, ss>>>(b);
    else k_pup<FmtINT4, true, 2><<<(T + 63) / 64 * kUpNB, 256, kSmem, ss>>>(b);
    for (int p = 0; p < b.w.npass; ++p) {
      b.c0 = p * b.w.cp;
      if (cfg.sh_rows == 128) k_pdown<true, 4><<<(T + 127) / 128 * (b.w.cp / 128), 256, kSmem, ss>>>(b);
      else k_pdown<true, 2><<<(T + 63) / 64 * (b.w.cp / 128), 256, kSmem, ss>>>(b);
    }
    if (fork) CUDA_CHECK(cudaEventRecord(ev[1], cfg.side));
  }
  switch (L.fmt) {
    case kINT4: prefill_routed<FmtINT4>(a, cfg, idx, w, ph, maxt, nsm, st); break;
    case kBF16: prefill_routed<FmtBF16>(a, cfg, idx, w, ph, maxt, nsm, st); break;
    case kNVFP4: prefill_routed<FmtNVFP4>(a, cfg, idx, w, ph, maxt, nsm, st); break;
  }
  if (fork) CUDA_CHECK(cudaStreamWaitEvent(st, ev[1], 0));
  CUDA_CHECK(cudaGetLastError());
}

void moe_prefill(const LayerW& L, const __nv_bfloat16* x, const int* idx, const float* w, int T, void* ws, float* yr,
                 float* ys, cudaStream_t st, const PrefillCfg& cfg) {
  moe_prefill_phases(L, x, idx, w, T, ws, yr, ys, st, cfg, kPhAll);
}

}  // namespace gx
