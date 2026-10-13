// FP8 x FP8 block-scaled GEMV for a few activation rows (decode / verify).
//
// Tensor-core version: each warp owns MT tiles of 8 weight rows and a range of k (split-K across
// the warps of a CTA). One m16n8k32 block-scaled MMA per (tile, k-block): A = the activation rows
// (rows >= M are zero; up to 8 rows so are 8..15), B = 8 weight rows. The k order inside a block is permuted so that
// loads are 16 bytes wide and coalesced; the scales are per block, so the permutation is invisible
// in the result. Several weight matrices sharing an input can go in one launch (segments).
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

#include <algorithm>

namespace dsk {

namespace {

constexpr int kU = 4;  // 64-byte steps loaded per batch: MT * kU 16-byte loads in flight per lane

struct Segs {
  Fp8Seg s[kMaxSegs];
  int row0[kMaxSegs + 1];  // prefix sums of N
  int n;
};

// Activations and their scales are read through L1 (they are tiny and every warp reuses them);
// weights stream past L1. Shared memory holds the cross-warp (split-K) partial sums and, with NORM,
// the activations: each CTA computes RMSNorm + act_quant of x itself, while its first weights load.
// RB: the rows it is built for, 8 (k_gemv_fp8) or 16 (k_gemv_fp8_r16: rows 8..15 in the MMAs' upper
// half, lane group g has rows g and g + 8, the odd lanes give row g + 8's scales; the same weight loads
// and MMAs, the activations of half a batch of steps at a time).
template <int MT, bool TILED, bool NORM, int RB>
__device__ __forceinline__ void gemv_fp8_body(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, int M,
                                              const Segs& segs, int K, int wn, int wk, NormIn nin) {
  extern __shared__ float red[];  // [wk][RB][rows_cta] (>= 256 + RB); NORM: then q [M][K + 32], s [M][K/32]
  constexpr bool kHi = RB > kLowRows;
  const int rows_cta = wn * 8 * MT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % wn, wi_k = warp / wn;
  const int r_cta = blockIdx.x * rows_cta;
  int si = 0;
  while (si + 1 < segs.n && r_cta >= segs.row0[si + 1]) ++si;
  const uint8_t* __restrict__ W = segs.s[si].W;
  const uint8_t* __restrict__ S = segs.s[si].S;
  const int N = segs.s[si].N;
  const int n_cta = r_cta - segs.row0[si];
  const int n0 = n_cta + wi_n * 8 * MT;
  const int KB = K / 32;

  // 64-byte steps (two k-blocks): lane (g, t) loads 16 bytes of its row at 16t, then trades half
  // with lane t^2 so that it holds 8 bytes of each block at offset 16(t&1) + 8(t>>1).
  const int P = K / 64;
  const int p0 = (P * wi_k) / wk, p1 = (P * (wi_k + 1)) / wk;
  float c[MT][4];
  const uint8_t* wrow[MT];
  const uint8_t* srow[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
    int n = n0 + 8 * i;
    n = n < N ? n : 0;
    // natural: row-major [N][K]; tiled: [N/8][K/64][8 rows][64 bytes], so a warp's 16-byte loads
    // for one 64-byte step of 8 rows are 512 contiguous bytes
    wrow[i] = TILED ? W + (size_t)(n / 8) * 8 * K + g * 64 + 16 * t : W + (size_t)(n + g) * K + 16 * t;
    srow[i] = S + (size_t)(n / 32) * KB;
  }
  const int off = 16 * (t & 1) + 8 * (t >> 1);
  const bool arow_live = g < M, arow_live2 = kHi && g + 8 < M;
  const int red_n = max(wk * RB * rows_cta, 256 + RB), ldq = NORM ? K + 32 : K;  // +32: no bank conflicts
  uint8_t* sq = reinterpret_cast<uint8_t*>(red + red_n);
  uint8_t* ss = sq + (size_t)M * ldq;
  const uint8_t* arow = (NORM ? sq : xq) + (size_t)(arow_live ? g : 0) * ldq + off;
  const uint8_t* srowa = (NORM ? ss : xs) + (size_t)(arow_live ? g : 0) * KB;
  const bool odd = lane & 1, lo = t < 2;
  // kHi: row g + 8, and the scale row this lane gives the MMA (row g + 8 for odd lanes)
  const uint8_t* arow2 = (NORM ? sq : xq) + (size_t)(arow_live2 ? g + 8 : 0) * ldq + off;
  const uint8_t* srowm = odd ? (NORM ? ss : xs) + (size_t)(arow_live2 ? g + 8 : 0) * KB : srowa;
  const bool slive = odd ? arow_live2 : arow_live;

  // Software pipeline: the weights of the next batch are in flight while this one computes. The
  // first batch is requested before waiting on the previous kernel (PDL), since weights never
  // depend on it.
  // Two register buffers used in turn (the loop is unrolled by two so that both stay in registers).
  uint4 ba[kU][MT], bb[kU][MT];
  auto load_w = [&](uint4 (&b)[kU][MT], int p) {
#pragma unroll
    for (int u = 0; u < kU; ++u)
      if (p + u < p1)
#pragma unroll
        for (int i = 0; i < MT; ++i) b[u][i] = ldg_stream128(wrow[i] + (p + u) * (TILED ? 512 : 64));
  };
  auto comp8 = [&](const uint4 (&b)[kU][MT], int p) {  // up to 8 rows
    uint2 a[kU][2];
    uint32_t sfa[kU][2], sfb[kU][2][MT];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (p + u >= p1) break;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kb = 2 * (p + u) + h;
        if constexpr (NORM) {
          a[u][h] = arow_live ? *reinterpret_cast<const uint2*>(arow + kb * 32) : make_uint2(0, 0);
          sfa[u][h] = (odd || !arow_live) ? 127u : srowa[kb];
        } else {
          a[u][h] = arow_live ? __ldg(reinterpret_cast<const uint2*>(arow + kb * 32)) : make_uint2(0, 0);
          sfa[u][h] = (odd || !arow_live) ? 127u : __ldg(srowa + kb);
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) sfb[u][h][i] = __ldg(srow[i] + kb);
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (p + u >= p1) break;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const uint4 w = b[u][i];
        uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
        uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        mma_e4m3_e4m3(c[i], a[u][0].x, 0u, a[u][0].y, 0u, lo ? w.x : r0, lo ? w.y : r1, sfa[u][0], sfb[u][0][i]);
        mma_e4m3_e4m3(c[i], a[u][1].x, 0u, a[u][1].y, 0u, lo ? r0 : w.z, lo ? r1 : w.w, sfa[u][1], sfb[u][1][i]);
      }
    }
  };
  // 9..16 rows: rows g and g + 8 (the odd lanes' scales are row g + 8's), the activations of half the
  // batch's steps at a time
  auto comp16 = [&](const uint4 (&b)[kU][MT], int p) {
#pragma unroll
    for (int u0 = 0; u0 < kU; u0 += kU / 2) {
      uint2 a[kU / 2][2], a2[kU / 2][2];
      uint32_t sfa[kU / 2][2], sfb[kU / 2][2][MT];
#pragma unroll
      for (int du = 0; du < kU / 2; ++du) {
        const int u = u0 + du;
        if (p + u >= p1) break;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int kb = 2 * (p + u) + h;
          a[du][h] = a2[du][h] = make_uint2(0, 0);
          if constexpr (NORM) {
            if (arow_live) a[du][h] = *reinterpret_cast<const uint2*>(arow + kb * 32);
            if (arow_live2) a2[du][h] = *reinterpret_cast<const uint2*>(arow2 + kb * 32);
            sfa[du][h] = slive ? srowm[kb] : 127u;
          } else {
            if (arow_live) a[du][h] = __ldg(reinterpret_cast<const uint2*>(arow + kb * 32));
            if (arow_live2) a2[du][h] = __ldg(reinterpret_cast<const uint2*>(arow2 + kb * 32));
            sfa[du][h] = slive ? __ldg(srowm + kb) : 127u;
          }
#pragma unroll
          for (int i = 0; i < MT; ++i) sfb[du][h][i] = __ldg(srow[i] + kb);
        }
      }
#pragma unroll
      for (int du = 0; du < kU / 2; ++du) {
        const int u = u0 + du;
        if (p + u >= p1) break;
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const uint4 w = b[u][i];
          uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
          uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
          mma_e4m3_e4m3(c[i], a[du][0].x, a2[du][0].x, a[du][0].y, a2[du][0].y, lo ? w.x : r0, lo ? w.y : r1,
                        sfa[du][0], sfb[du][0][i]);
          mma_e4m3_e4m3(c[i], a[du][1].x, a2[du][1].x, a[du][1].y, a2[du][1].y, lo ? r0 : w.z, lo ? r1 : w.w,
                        sfa[du][1], sfb[du][1][i]);
        }
      }
    }
  };
  auto comp = [&](const uint4 (&b)[kU][MT], int p) {
    if constexpr (kHi) comp16(b, p);
    else comp8(b, p);
  };
  load_w(ba, p0);
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (NORM) {
    const bool first = blockIdx.x == 0;
    rmsnorm_rows<RB>(nin.x, K, nin.w, M, K, nin.eps, red, first ? nin.y : nullptr, K, sq, ldq, ss,
                     first ? nin.q : nullptr, first ? nin.qs : nullptr);
    __syncthreads();
  }
  for (int p = p0; p < p1; p += 2 * kU) {
    if (p + kU < p1) load_w(bb, p + kU);
    comp(ba, p);
    if (p + kU >= p1) break;
    if (p + 2 * kU < p1) load_w(ba, p + 2 * kU);
    comp(bb, p + kU);
  }

  // Partial sums per k-slice -> smem [wk][m][row], then a fixed-order sum over the slices.
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    int r = wi_n * 8 * MT + 8 * i + 2 * t;
    red[(wi_k * RB + g) * rows_cta + r] = c[i][0];
    red[(wi_k * RB + g) * rows_cta + r + 1] = c[i][1];
    if constexpr (kHi) {
      red[(wi_k * RB + g + 8) * rows_cta + r] = c[i][2];
      red[(wi_k * RB + g + 8) * rows_cta + r + 1] = c[i][3];
    }
  }
  __syncthreads();
  __nv_bfloat16* __restrict__ y = segs.s[si].y;
  for (int idx = threadIdx.x; idx < M * rows_cta; idx += blockDim.x) {
    int m = idx / rows_cta, r = idx - m * rows_cta;
    if (n_cta + r >= N) continue;
    float v = 0.f;
    for (int j = 0; j < wk; ++j) v += red[(j * RB + m) * rows_cta + r];
    y[(size_t)m * N + n_cta + r] = __float2bfloat16_rn(v);
  }
}

template <int MT, bool TILED, bool NORM>
__global__ void __launch_bounds__(512) k_gemv_fp8(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                   int M, Segs segs, int K, int wn, int wk, NormIn nin) {
  gemv_fp8_body<MT, TILED, NORM, kLowRows>(xq, xs, M, segs, K, wn, wk, nin);
}

// 9..16 rows, one tile per warp. (512, 1): up to 128 registers.
template <bool TILED, bool NORM>
__global__ void __launch_bounds__(512, 1) k_gemv_fp8_r16(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                          int M, Segs segs, int K, int wn, int wk, NormIn nin) {
  gemv_fp8_body<1, TILED, NORM, kMaxRows>(xq, xs, M, segs, K, wn, wk, nin);
}

template <int MT, bool TILED, bool NORM, int RB>
constexpr auto gemv_fp8_kernel() {
  if constexpr (RB == kLowRows) return k_gemv_fp8<MT, TILED, NORM>;
  else return k_gemv_fp8_r16<TILED, NORM>;
}

// Persistent version: k_gemv_fp8's arithmetic for mt = wn = 1 (tiles of 8 weight rows, wk warps along
// K) with tiled weights, on a grid of about one CTA per SM, each taking a contiguous run of tiles.
// Warp w streams slice w of every tile of its run through a ring of S 512-byte stages in shared
// memory, filled by cp.async: lane l copies (and later reads) bytes 16l..16l+15 of each 64-byte step
// of the 8 rows, the bytes k_gemv_fp8's lane l loads, so no lane waits on another's copies; lane 0
// also copies the step's two block scales, which the warp reads after a __syncwarp. The ring holds
// no registers, so a CTA requests S stages per warp before waiting on the previous kernel (PDL: during
// a collective, while the DRAM has nothing else to do) and keeps S in flight until its run ends,
// with no waves. Per tile the slices' partial sums meet in shared memory (two buffers, one barrier
// per tile) and are added in slice order: the outputs are k_gemv_fp8's with the same wk, bit for bit.
constexpr int kRingS[] = {16, 12, 8, 4};  // ring depths instantiated, deepest first
constexpr size_t kRingBudget = 80 * 1024;  // the auto ring's shared memory per CTA: hc_project's 16 KB
                                           // still fits beside it on the SM

size_t fp8p_smem(int S, int wk, int M, int K, bool norm) {
  return (size_t)wk * S * 516 + (size_t)2 * wk * 64 * sizeof(float) +
         (norm ? (size_t)M * (K + 32) + (size_t)M * (K / 32) : 0);
}

template <int S, bool NORM>
__global__ void __launch_bounds__(512) k_gemv_fp8p(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                    int M, Segs segs, int K, int ntile, NormIn nin) {
  // ring [wk][S][512] | the steps' scale words [wk][S] | red [2][wk][8][8] | NORM: q [M][K + 32], s [M][K/32]
  extern __shared__ __align__(16) uint8_t sm[];
  const int wk = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int P = K / 64, KB = K / 32, ldq = NORM ? K + 32 : K;
  uint8_t* const ring = sm + (size_t)warp * S * 512 + 16 * lane;
  uint32_t* const ssc = reinterpret_cast<uint32_t*>(sm + (size_t)wk * S * 512) + warp * S;
  float* const red = reinterpret_cast<float*>(sm + (size_t)wk * S * 516);
  uint8_t* const sq = reinterpret_cast<uint8_t*>(red + 2 * wk * 64);
  uint8_t* const ss = sq + (size_t)M * ldq;
  const int p0 = (P * warp) / wk, p1 = (P * (warp + 1)) / wk;
  const int t0 = (int)((long long)ntile * blockIdx.x / gridDim.x);
  const int t1 = (int)((long long)ntile * (blockIdx.x + 1) / gridDim.x);

  // the issue cursor: tile it of segment is, step ip, ring slot si
  int it = t0, ip = p0, is = 0, si = 0;
  while (is + 1 < segs.n && 8 * it >= segs.row0[is + 1]) ++is;
  auto issue = [&]() {
    if (it < t1) {
      const int n = 8 * it - segs.row0[is];
      cp_async16(ring + si * 512, segs.s[is].W + (size_t)n * K + (size_t)ip * 512 + 16 * lane);
      if (lane == 0) {  // the aligned word holding blocks 2 ip, 2 ip + 1 of the tile's scale row
        const uintptr_t a = reinterpret_cast<uintptr_t>(segs.s[is].S + (size_t)(n / 32) * KB + 2 * ip);
        cp_async4(ssc + si, reinterpret_cast<const void*>(a & ~(uintptr_t)3));
      }
      if (++ip == p1) {
        ip = p0;
        if (++it < t1)
          while (is + 1 < segs.n && 8 * it >= segs.row0[is + 1]) ++is;
      }
    }
    cp_async_commit();  // empty past the run's end: every stage is one group
    si = si + 1 == S ? 0 : si + 1;
  };
#pragma unroll 1
  for (int i = 0; i < S; ++i) issue();
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (NORM) {
    const bool first = blockIdx.x == 0;
    rmsnorm_rows(nin.x, K, nin.w, M, K, nin.eps, red, first ? nin.y : nullptr, K, sq, ldq, ss, first ? nin.q : nullptr,
                 first ? nin.qs : nullptr);
    __syncthreads();
  }
  const bool live = g < M, odd = lane & 1, lo = t < 2;
  const int off = 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* arow = (NORM ? sq : xq) + (size_t)(live ? g : 0) * ldq + off;
  const uint8_t* srowa = (NORM ? ss : xs) + (size_t)(live ? g : 0) * KB;
  int slot = 0, cs = 0;
  while (cs + 1 < segs.n && 8 * t0 >= segs.row0[cs + 1]) ++cs;
#pragma unroll 1
  for (int tl = t0; tl < t1; ++tl) {
    while (cs + 1 < segs.n && 8 * tl >= segs.row0[cs + 1]) ++cs;
    const int n = 8 * tl - segs.row0[cs];
    const int sb = (int)(reinterpret_cast<uintptr_t>(segs.s[cs].S + (size_t)(n / 32) * KB) & 3);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int p = p0; p < p1; ++p) {
      cp_async_wait<S - 1>();  // this stage's group, and lane 0's scale word once the warp syncs
      __syncwarp();
      const uint4 w = *reinterpret_cast<const uint4*>(ring + slot * 512);
      const uint32_t sw = ssc[slot] >> (8 * ((sb + 2 * p) & 3));
      __syncwarp();  // the slot is read: refill it
      issue();
      slot = slot + 1 == S ? 0 : slot + 1;
      uint2 a[2];
      uint32_t sfa[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kb = 2 * p + h;
        if constexpr (NORM) {
          a[h] = live ? *reinterpret_cast<const uint2*>(arow + kb * 32) : make_uint2(0, 0);
          sfa[h] = (odd || !live) ? 127u : srowa[kb];
        } else {
          a[h] = live ? __ldg(reinterpret_cast<const uint2*>(arow + kb * 32)) : make_uint2(0, 0);
          sfa[h] = (odd || !live) ? 127u : __ldg(srowa + kb);
        }
      }
      const uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
      const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
      mma_e4m3_e4m3(c, a[0].x, 0u, a[0].y, 0u, lo ? w.x : r0, lo ? w.y : r1, sfa[0], sw & 0xFFu);
      mma_e4m3_e4m3(c, a[1].x, 0u, a[1].y, 0u, lo ? r0 : w.z, lo ? r1 : w.w, sfa[1], (sw >> 8) & 0xFFu);
    }
    // the slices' partial sums, then k_gemv_fp8's fixed-order sum over them
    float* rd = red + (tl & 1) * wk * 64;
    rd[(warp * 8 + g) * 8 + 2 * t] = c[0];
    rd[(warp * 8 + g) * 8 + 2 * t + 1] = c[1];
    __syncthreads();
    if (threadIdx.x < M * 8) {
      const int m = threadIdx.x >> 3, r = threadIdx.x & 7;
      float v = 0.f;
      for (int j = 0; j < wk; ++j) v += rd[(j * 8 + m) * 8 + r];
      segs.s[cs].y[(size_t)m * segs.s[cs].N + n + r] = __float2bfloat16_rn(v);
    }
  }
}

// CUDA-core version: one warp per weight row pair, lanes across K in 16-byte chunks, fp32 FMAs.
template <int M_>
__global__ void __launch_bounds__(256) k_gemv_fp8_cc(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs,
                                                     const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                     int N, int K, __nv_bfloat16* __restrict__ y) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int KB = K / 32;
  uint8_t* sa = smem;                                     // [M_][K]
  float* ssc = reinterpret_cast<float*>(smem + M_ * K);   // [M_][KB] scales as fp32
  for (int i = threadIdx.x; i < M_ * K / 16; i += blockDim.x)
    reinterpret_cast<uint4*>(sa)[i] = __ldg(reinterpret_cast<const uint4*>(xq) + i);
  for (int i = threadIdx.x; i < M_ * KB; i += blockDim.x) ssc[i] = ue8m0_to_f32(xs[i]);
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (blockIdx.x * (blockDim.x >> 5) + warp) * 2;
  if (n >= N) return;
  float acc[2][M_];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int m = 0; m < M_; ++m) acc[r][m] = 0.f;
  const int nch = K / 16;
  for (int ch = lane; ch < nch; ch += 32) {
    const int kb = ch >> 1;
    uint4 w[2];
    w[0] = ldg_stream128(W + (size_t)n * K + ch * 16);
    w[1] = ldg_stream128(W + (size_t)(n + 1) * K + ch * 16);
    float wf[2][16];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint32_t* wv = &w[r].x;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(wv[j >> 1] >> (16 * (j & 1))), __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        wf[r][2 * j] = f.x;
        wf[r][2 * j + 1] = f.y;
      }
    }
    const float sw0 = ue8m0_to_f32(__ldg(S + (size_t)(n / 32) * KB + kb));
#pragma unroll
    for (int m = 0; m < M_; ++m) {
      uint4 av = *reinterpret_cast<const uint4*>(sa + m * K + ch * 16);
      const uint32_t* a32 = &av.x;
      float p0 = 0.f, p1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(a32[j >> 1] >> (16 * (j & 1))), __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
        p0 = fmaf(f.x, wf[0][2 * j], fmaf(f.y, wf[0][2 * j + 1], p0));
        p1 = fmaf(f.x, wf[1][2 * j], fmaf(f.y, wf[1][2 * j + 1], p1));
      }
      const float sc = ssc[m * KB + kb] * sw0;
      acc[0][m] = fmaf(p0, sc, acc[0][m]);
      acc[1][m] = fmaf(p1, sc, acc[1][m]);
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int m = 0; m < M_; ++m) {
      float v = warp_sum(acc[r][m]);
      if (lane == 0) y[(size_t)m * N + n + r] = __float2bfloat16_rn(v);
    }
}

template <int MT, bool TILED, bool NORM, int RB = kLowRows>
void launch_fp8(const uint8_t* xq, const uint8_t* xs, int M, const Segs& segs, int K, GemvCfg cfg, const NormIn& nin,
                cudaStream_t st) {
  const int rows_cta = cfg.wn * 8 * MT;
  for (int i = 0; i < segs.n; ++i)
    if (segs.s[i].N % rows_cta) die("gemv_fp8: segment %d has N=%d, not a multiple of %d", i, segs.s[i].N, rows_cta);
  size_t smem = (size_t)std::max(cfg.wk * RB * rows_cta, 256 + RB) * sizeof(float);
  if (NORM) smem += (size_t)M * (K + 32) + (size_t)M * (K / 32);
  if (smem > 96 * 1024) die("gemv_fp8: M=%d K=%d wk=%d needs %zu bytes of shared memory", M, K, cfg.wk, smem);
  static bool attr = false;
  if (NORM && !attr) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_fp8_kernel<MT, TILED, NORM, RB>(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    96 * 1024));
    attr = true;
  }
  dim3 grid(segs.row0[segs.n] / rows_cta), block(32 * cfg.wn * cfg.wk);
  launch_pdl(gemv_fp8_kernel<MT, TILED, NORM, RB>(), grid, block, smem, st, cfg.pdl, xq, xs, M, segs, K, cfg.wn, cfg.wk,
             nin);
}

int sm_count() {
  static const int n = [] {
    int d = 0, c = 0;
    CUDA_CHECK(cudaGetDevice(&d));
    CUDA_CHECK(cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, d));
    return c;
  }();
  return n;
}

template <int S, bool NORM>
void launch_fp8p(const uint8_t* xq, const uint8_t* xs, int M, const Segs& segs, int K, int wk, int ctas, size_t smem,
                 bool pdl, const NormIn& nin, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_gemv_fp8p<S, NORM>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(k_gemv_fp8p<S, NORM>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
    attr = true;
  }
  launch_pdl(k_gemv_fp8p<S, NORM>, dim3(ctas), dim3(32 * wk), smem, st, pdl, xq, xs, M, segs, K, segs.row0[segs.n] / 8,
             nin);
}

// The persistent kernel's grid and ring: as asked, or one CTA per SM (at most one per tile) and the
// deepest ring within kRingBudget, but no deeper than a warp's whole run. False when even the
// shallowest ring does not fit: the caller runs k_gemv_fp8.
template <bool NORM>
bool persist_fp8(const uint8_t* xq, const uint8_t* xs, int M, const Segs& segs, int K, GemvCfg cfg, const NormIn& nin,
                 cudaStream_t st) {
  const int ntile = segs.row0[segs.n] / 8, wk = cfg.wk;
  const int ctas = std::min(ntile, cfg.ctas > 0 ? cfg.ctas : sm_count());
  int S = cfg.ring;
  if (S == 0) {
    const int run = (ntile + ctas - 1) / ctas * ((K / 64 + wk - 1) / wk);  // a warp's stages at most
    for (int s : kRingS)
      if (fp8p_smem(s, wk, M, K, NORM) <= kRingBudget && (S == 0 || s >= run)) S = s;
    if (S == 0) S = 4;
  }
  const size_t smem = fp8p_smem(S, wk, M, K, NORM);
  if (smem > 99 * 1024) return false;
  if (wk > 16 || wk > K / 64) die("gemv_fp8: persistent kernel, wk=%d S=%d", wk, S);
  switch (S) {
    case 16: launch_fp8p<16, NORM>(xq, xs, M, segs, K, wk, ctas, smem, cfg.pdl, nin, st); break;
    case 12: launch_fp8p<12, NORM>(xq, xs, M, segs, K, wk, ctas, smem, cfg.pdl, nin, st); break;
    case 8: launch_fp8p<8, NORM>(xq, xs, M, segs, K, wk, ctas, smem, cfg.pdl, nin, st); break;
    case 4: launch_fp8p<4, NORM>(xq, xs, M, segs, K, wk, ctas, smem, cfg.pdl, nin, st); break;
    default: die("gemv_fp8: ring %d (4, 8, 12 or 16)", S);
  }
  return true;
}

template <bool NORM>
void dispatch_fp8(const uint8_t* xq, const uint8_t* xs, int M, const Fp8Seg* seg, int nseg, int K, GemvCfg cfg,
                  const NormIn& nin, cudaStream_t st) {
  if (M < 1 || M > kMaxRows || K % 64 || nseg < 1 || nseg > kMaxSegs || cfg.wn * cfg.wk > 16 || (NORM && K % 256))
    die("gemv_fp8: unsupported call M=%d K=%d nseg=%d", M, K, nseg);
  Segs segs{};
  segs.n = nseg;
  segs.row0[0] = 0;
  bool aligned = true;  // the persistent kernel copies scales as aligned words
  for (int i = 0; i < nseg; ++i) {
    segs.s[i] = seg[i], segs.row0[i + 1] = segs.row0[i] + seg[i].N;
    aligned = aligned && reinterpret_cast<uintptr_t>(seg[i].S) % 4 == 0 && reinterpret_cast<uintptr_t>(seg[i].W) % 16 == 0;
  }
  if (M > kLowRows) {  // 9..16 rows: one tile per warp (the same bits as any mt; mt 2 and 4 would spill)
    if (cfg.tiled) launch_fp8<1, true, NORM, kMaxRows>(xq, xs, M, segs, K, cfg, nin, st);
    else launch_fp8<1, false, NORM, kMaxRows>(xq, xs, M, segs, K, cfg, nin, st);
    return;
  }
  if (NORM && cfg.ring < 0 && cfg.tiled && M <= persist_rows()) cfg.ring = 0, cfg.mt = cfg.wn = 1;
  if (cfg.ring >= 0 && cfg.tiled && cfg.mt == 1 && cfg.wn == 1 && aligned) {
    for (int i = 0; i < nseg; ++i)
      if (seg[i].N % 8) die("gemv_fp8: segment %d has N=%d, not a multiple of 8", i, seg[i].N);
    if (persist_fp8<NORM>(xq, xs, M, segs, K, cfg, nin, st)) return;
  }
  switch (cfg.mt * 2 + cfg.tiled) {
    case 2: launch_fp8<1, false, NORM>(xq, xs, M, segs, K, cfg, nin, st); break;
    case 3: launch_fp8<1, true, NORM>(xq, xs, M, segs, K, cfg, nin, st); break;
    case 4: launch_fp8<2, false, NORM>(xq, xs, M, segs, K, cfg, nin, st); break;
    case 5: launch_fp8<2, true, NORM>(xq, xs, M, segs, K, cfg, nin, st); break;
    case 8: launch_fp8<4, false, NORM>(xq, xs, M, segs, K, cfg, nin, st); break;
    case 9: launch_fp8<4, true, NORM>(xq, xs, M, segs, K, cfg, nin, st); break;
    default: die("gemv_fp8: mt=%d", cfg.mt);
  }
}

template <int M_>
void launch_fp8_cc(const uint8_t* xq, const uint8_t* xs, const uint8_t* W, const uint8_t* S, int N, int K,
                   __nv_bfloat16* y, cudaStream_t st) {
  size_t smem = (size_t)M_ * K + (size_t)M_ * (K / 32) * 4;
  static bool attr_set = false;
  if (!attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(k_gemv_fp8_cc<M_>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    attr_set = true;
  }
  int warps = 8;
  k_gemv_fp8_cc<M_><<<(N / 2 + warps - 1) / warps, 32 * warps, smem, st>>>(xq, xs, W, S, N, K, y);
  CUDA_CHECK(cudaGetLastError());
}

// natural [N][K] -> tiled [N/8][K/64][8][64]
__global__ void k_tile_fp8(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int N, int K) {
  const int P = K / 64;
  size_t n16 = (size_t)N * K / 16;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    size_t row = i / (K / 16), c = i % (K / 16);  // source row, 16-byte chunk
    size_t p = c / 4, q = c % 4;
    size_t d = ((row / 8) * P + p) * 512 + (row % 8) * 64 + q * 16;
    reinterpret_cast<uint4*>(dst)[d / 16] = reinterpret_cast<const uint4*>(src)[i];
  }
}

}  // namespace

void tile_fp8(const uint8_t* src, uint8_t* dst, int N, int K, cudaStream_t st) {
  if (N % 8 || K % 64) die("tile_fp8: N=%d K=%d", N, K);
  k_tile_fp8<<<1024, 256, 0, st>>>(src, dst, N, K);
  CUDA_CHECK(cudaGetLastError());
}

void gemv_fp8_multi(const uint8_t* xq, const uint8_t* xs, int M, const Fp8Seg* seg, int nseg, int K, GemvCfg cfg,
                    cudaStream_t st) {
  dispatch_fp8<false>(xq, xs, M, seg, nseg, K, cfg, NormIn{}, st);
}

void gemv_fp8_norm(const __nv_bfloat16* x, const __nv_bfloat16* norm_w, float eps, int M, const Fp8Seg* seg, int nseg,
                   int K, GemvCfg cfg, cudaStream_t st, __nv_bfloat16* y_out, uint8_t* q_out, uint8_t* qs_out) {
  dispatch_fp8<true>(nullptr, nullptr, M, seg, nseg, K, cfg, NormIn{x, norm_w, eps, y_out, q_out, qs_out}, st);
}

void gemv_fp8(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* W, const uint8_t* S, int N, int K,
              __nv_bfloat16* y, GemvCfg cfg, cudaStream_t st) {
  Fp8Seg seg{W, S, y, N};
  gemv_fp8_multi(xq, xs, M, &seg, 1, K, cfg, st);
}

void gemv_fp8_cc(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* W, const uint8_t* S, int N, int K,
                 __nv_bfloat16* y, cudaStream_t st) {
  if (K % 32 || N % 2) die("gemv_fp8_cc: unsupported shape");
  switch (M) {
    case 1: launch_fp8_cc<1>(xq, xs, W, S, N, K, y, st); break;
    case 2: launch_fp8_cc<2>(xq, xs, W, S, N, K, y, st); break;
    case 4: launch_fp8_cc<4>(xq, xs, W, S, N, K, y, st); break;
    case 6: launch_fp8_cc<6>(xq, xs, W, S, N, K, y, st); break;
    case 8: launch_fp8_cc<8>(xq, xs, W, S, N, K, y, st); break;
    default: die("gemv_fp8_cc: M=%d not instantiated", M);
  }
}

}  // namespace dsk
