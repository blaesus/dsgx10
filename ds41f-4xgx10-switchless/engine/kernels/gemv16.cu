// Weight-streaming GEMVs with 16-bit activations, for a few rows (decode / verify):
//  - bf16 weights, fp32 math: the head (fp32 logits) and the router's gate;
//  - wo_a: FP8 weights widened to bf16 in registers (times their 32x32 block scale, exact), block-
//    diagonal over groups of rows that each read their own slice of the input, bf16 out.
// Weights are in the tile_fp8 byte layout ([N/8][row bytes/64][8 rows][64 B]), activations bf16
// [M][K] read through L1. Tensor cores (m16n8k16 bf16, fp32 accumulation) or, for comparison, CUDA
// cores (fp32 FMAs), with the same loads; split-K across the warps of a CTA, fixed-order reduction.
// NORM (bf16 weights, tensor cores): the input is RMSNorm(x), computed by every CTA into shared
// memory while its first weights load; CTA 0 also writes its act_quant (the MoE's input). From 9 rows
// (RB = 16) the normed rows would crowd out the reduction, so the CTAs compute only the rows' 1/rms and
// normalise x as they read it, with the same bits.
// kBF16X: bf16 weights coded exactly in fewer bytes (bf16x_encode, the vocab head): each warp step's tile of
// 8 rows x 32 values keeps its signs and mantissas, and its exponents as 4-bit offsets from the tile's least;
// a tile whose exponents span more than 16 values stays raw. A lane widens its coded step back to the raw
// step's 16 bytes and runs kBF16's MMAs on them: the same bits as kBF16 on the raw weights.
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

#include <algorithm>
#include <vector>

namespace dsk {

namespace {

enum Fmt { kBF16 = 0, kWOA = 1, kBF16X = 2 };

// Two e4m3 bytes (bits 7:0 and 23:16 of u) -> bf16x2 of value * 2^-120: the e4m3 bits moved into
// bf16's fields (subnormals included); the caller's multiply restores the exponent bias.
__device__ __forceinline__ uint32_t e4m3x2_bits_to_bf16x2(uint32_t u) {
  return ((u << 8) & 0x80008000u) | ((u << 4) & 0x07F007F0u);
}

// 16 e4m3 weights -> 8 bf16x2 words of w * 2^(s-127): the bit move, then one multiply by 2^(s-7)
// (exact; needs s <= 134, i.e. block scales <= 2^7, which holds for the checkpoint's weights).
__device__ __forceinline__ void widen16(const uint4& w, uint32_t s, uint32_t (&o)[8]) {
  const __nv_bfloat162 f = __bfloat162bfloat162(__ushort_as_bfloat16((uint16_t)((s + 120u) << 7)));
  const uint32_t* v = &w.x;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t lo = e4m3x2_bits_to_bf16x2(__byte_perm(v[j], 0, 0x4140));
    uint32_t hi = e4m3x2_bits_to_bf16x2(__byte_perm(v[j], 0, 0x4342));
    __nv_bfloat162 a = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&lo), f);
    __nv_bfloat162 b = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&hi), f);
    o[2 * j] = *reinterpret_cast<uint32_t*>(&a);
    o[2 * j + 1] = *reinterpret_cast<uint32_t*>(&b);
  }
}

// kBF16X's bytes, for T = N/8 * 2K/64 tiles (tile i: the i-th 512 bytes of the tile_fp8 layout; lane l
// holds its bytes 16l .. 16l + 15, 8 values):
//   [T][256 B]  sign and mantissa bytes (sign << 7 | mantissa): lane l's 8 values at 8l
//   [T][128 B]  exponent offsets: lane l's u32 at 4l, nibble j (bits 4j) value 2j, nibble 4 + j value 2j + 1
//   [T] u32     header: (b << 23) | (b << 7), b the tile's least exponent; raw tile r: 0x80000000 | r
//   [R][512 B]  the raw tiles as they were, from the first multiple of 256 bytes after the headers
__host__ __device__ constexpr size_t bf16x_raw_off(size_t T) { return (T * 388 + 255) & ~(size_t)255; }

// a * M + c as one multiply-add (nvcc would make it a shift, a mask and an add)
template <uint32_t M>
__device__ __forceinline__ uint32_t mad_imm(uint32_t a, uint32_t c) {
  uint32_t d;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "n"(M), "r"(c));
  return d;
}

// A lane's 8 values of a coded tile (s0, s1: their sign-mantissa bytes; e: their exponent offsets; hb: the
// tile's header) -> the raw step's 4 words (bf16 pairs)
__device__ __forceinline__ uint4 bf16x_dec(uint32_t s0, uint32_t s1, uint32_t e, uint32_t hb) {
  const uint32_t e2 = e >> 8;
  uint4 o;
  o.x = (__byte_perm(s0, 0, 0x1100) & 0x807F807Fu) | mad_imm<128>(e & 0x000F000Fu, hb);
  o.y = (__byte_perm(s0, 0, 0x3322) & 0x807F807Fu) | mad_imm<8>(e & 0x00F000F0u, hb);
  o.z = (__byte_perm(s1, 0, 0x1100) & 0x807F807Fu) | mad_imm<128>(e2 & 0x000F000Fu, hb);
  o.w = (__byte_perm(s1, 0, 0x3322) & 0x807F807Fu) | mad_imm<8>(e2 & 0x00F000F0u, hb);
  return o;
}

// A lane's load of a kBF16X step by the tile's header h: a raw tile's 16 bytes at r, or a coded one's 8 at s
// and 4 at e (x, y; z). Predicated rather than branched (h is warp-uniform).
__device__ __forceinline__ uint4 ldg_bf16x(uint32_t h, const uint8_t* r, const uint8_t* s, const uint8_t* e) {
  uint4 v;
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, %4, 0;\n\t"
      "@p ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%5];\n\t"
      "@!p ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%6];\n\t"
      "@!p ld.global.nc.L1::no_allocate.u32 %2, [%7];\n\t}"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "r"(h), "l"(r), "l"(s), "l"(e));
  return v;
}

// Per 64-byte step of a row: kBF16 (and kBF16X) = 32 values (lane t: k = 8t..8t+7), kWOA = 64 values
// (lane t: k = 16t..16t+15, one 32-block). The k order inside an MMA is permuted identically for A and B.
// RB: the rows it is built for, 8 (k_gemv16) or 16 (k_gemv16_r16: rows 8..15 in the MMAs' upper half,
// lane group g has rows g and g + 8; the same weight loads and MMAs).
template <int FMT, int MT, bool CC, bool NORM, int RB>
__device__ __forceinline__ void gemv16_body(const __nv_bfloat16* __restrict__ x, int M, const uint8_t* __restrict__ W,
                                            const uint8_t* __restrict__ S, int N, int K, int group_rows,
                                            float* __restrict__ yf, __nv_bfloat16* __restrict__ yb,
                                            uint8_t* __restrict__ q, uint8_t* __restrict__ qs, int wn, int wk,
                                            NormIn nin) {
  static_assert(!NORM || (FMT != kWOA && !CC), "the norm prologue is for the bf16 tensor-core path");
  static_assert(RB == kLowRows || (RB == kMaxRows && !CC), "9..16 rows: tensor cores");
  constexpr bool kHi = RB > kLowRows;     // rows 8..15 in the upper half
  constexpr bool kSx = NORM && !kHi;      // the normed rows in shared memory (otherwise normed as read)
  constexpr bool kB16 = FMT != kWOA;      // bf16 values (kBF16X: once decoded)
  constexpr int KU = kB16 ? 4 : 2;  // steps per batch
  constexpr int VPS = kB16 ? 32 : 64;  // k values per 64-byte step
  extern __shared__ float red[];  // [wk][RB][rows_cta] (NORM: >= 256 + RB), up to 8 rows NORM: then x [M][K + 32] bf16
  const int rows_cta = wn * 8 * MT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % wn, wi_k = warp / wn;
  const int n_cta = blockIdx.x * rows_cta, n0 = n_cta + wi_n * 8 * MT;
  const int rowb = kB16 ? 2 * K : K;  // weight bytes per row (kBF16X: before the coding)
  const int P = rowb / 64;
  const int p0 = (P * wi_k) / wk, p1 = (P * (wi_k + 1)) / wk;
  const int ldx = FMT == kWOA ? K * (N / group_rows) : K;
  const int xoff = FMT == kWOA ? (n_cta / group_rows) * K : 0;  // a CTA's rows share a group
  const bool live = g < M, live2 = kHi && g + 8 < M;

  const uint8_t* wrow[MT];   // kBF16X: the sign-mantissa bytes
  const uint8_t* srow[MT];   // kWOA: the block scales; kBF16X: the exponent offsets
  const uint32_t* hrow[MT];  // kBF16X: the headers, hrow[i][p] tile (i, p)'s (every lane reads it: L1)
  const size_t T = (size_t)(N / 8) * P;  // kBF16X's tiles
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int n = n0 + 8 * i;
    if constexpr (FMT == kBF16X) {
      wrow[i] = W + (size_t)(n / 8) * P * 256 + 8 * lane;
      srow[i] = W + T * 256 + (size_t)(n / 8) * P * 128 + 4 * lane;
      hrow[i] = reinterpret_cast<const uint32_t*>(W + T * 384) + (size_t)(n / 8) * P;
    } else {
      wrow[i] = W + (size_t)(n / 8) * 8 * rowb + g * 64 + 16 * t;
      srow[i] = FMT == kWOA ? S + (size_t)(n / 32) * (K / 32) + (t >> 1) : nullptr;
    }
  }
  // kBF16X: the lane's bytes of raw tile 0, less 2^40: raw tile r (header h = 2^31 + r) at rawq + 512 h
  const uint64_t rawq = FMT == kBF16X ? reinterpret_cast<uint64_t>(W + bf16x_raw_off(T) + 16 * lane) - (1ull << 40) : 0;
  // MMA: this lane's activation row (kHi: and row g + 8); CC: all rows, read per step
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(red + max(wk * RB * rows_cta, 256 + RB));
  const __nv_bfloat16* xrow = kSx ? sx + (live ? g : 0) * (K + 32) + 8 * t
                                  : (NORM ? nin.x : x) + (size_t)(live ? g : 0) * ldx + xoff + (kB16 ? 8 : 16) * t;
  const __nv_bfloat16* xrow2 = (NORM ? nin.x : x) + (size_t)(live2 ? g + 8 : 0) * ldx + xoff + (kB16 ? 8 : 16) * t;
  const __nv_bfloat16* nwrow = NORM ? nin.w + 8 * t : nullptr;  // kHi NORM: the norm weight, and the rows' 1/rms
  float rr = 0.f, rr2 = 0.f;

  float c[MT][4];
  float acc[MT][RB];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
#pragma unroll
    for (int m = 0; m < RB; ++m) acc[i][m] = 0.f;
  }

  uint4 ba[KU][MT], bb[KU][MT];
  uint32_t sa[KU][MT], sb[KU][MT];
  auto load = [&](uint4 (&b)[KU][MT], uint32_t (&s)[KU][MT], int p) {
#pragma unroll
    for (int u = 0; u < KU; ++u)
      if (p + u < p1)
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if constexpr (FMT == kBF16X) {
            const uint32_t h = __ldg(hrow[i] + p + u);
            b[u][i] = ldg_bf16x(h, reinterpret_cast<const uint8_t*>(rawq + (uint64_t)h * 512),
                                wrow[i] + (size_t)(p + u) * 256, srow[i] + (size_t)(p + u) * 128);
          } else {
            b[u][i] = ldg_stream128(wrow[i] + (p + u) * 512);
            if (FMT == kWOA) s[u][i] = __ldg(srow[i] + 2 * (p + u));
          }
        }
  };
  // a lane's 16 bytes of bf16 of the warp's tile (i, p), as kBF16 loads them: kBF16X decodes a coded tile
  // in place (the header read again rather than kept: 16 registers fewer)
  auto w16 = [&](uint4& b, int i, int p) -> const uint4& {
    if constexpr (FMT == kBF16X) {
      const uint32_t h = __ldg(hrow[i] + p);
      if ((int)h >= 0) b = bf16x_dec(b.x, b.y, b.z, h);
    }
    return b;
  };
  auto comp = [&](uint4 (&b)[KU][MT], const uint32_t (&s)[KU][MT], int p) {
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      if (p + u >= p1) break;
      const int k0 = (p + u) * VPS;
      if constexpr (!CC && !kHi) {
        if constexpr (kB16) {
          uint4 a = make_uint4(0, 0, 0, 0);
          if (live) a = NORM ? *reinterpret_cast<const uint4*>(xrow + k0) : __ldg(reinterpret_cast<const uint4*>(xrow + k0));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            const uint4& w = w16(b[u][i], i, p + u);
            mma_bf16(c[i], a.x, 0u, a.y, 0u, w.x, w.y);
            mma_bf16(c[i], a.z, 0u, a.w, 0u, w.z, w.w);
          }
        } else {
          uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0;
          if (live) a0 = __ldg(reinterpret_cast<const uint4*>(xrow + k0)), a1 = __ldg(reinterpret_cast<const uint4*>(xrow + k0 + 8));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            uint32_t wv[8];
            widen16(b[u][i], s[u][i], wv);
            mma_bf16(c[i], a0.x, 0u, a0.y, 0u, wv[0], wv[1]);
            mma_bf16(c[i], a0.z, 0u, a0.w, 0u, wv[2], wv[3]);
            mma_bf16(c[i], a1.x, 0u, a1.y, 0u, wv[4], wv[5]);
            mma_bf16(c[i], a1.z, 0u, a1.w, 0u, wv[6], wv[7]);
          }
        }
      } else if constexpr (!CC) {  // 9..16 rows: rows g and g + 8
        if constexpr (kB16) {
          uint4 a = make_uint4(0, 0, 0, 0), a2 = a;
          if constexpr (NORM) {
            const uint4 wn8 = __ldg(reinterpret_cast<const uint4*>(nwrow + k0));
            if (live) a = rmsnorm8(__ldg(reinterpret_cast<const uint4*>(xrow + k0)), wn8, rr);
            if (live2) a2 = rmsnorm8(__ldg(reinterpret_cast<const uint4*>(xrow2 + k0)), wn8, rr2);
          } else {
            if (live) a = __ldg(reinterpret_cast<const uint4*>(xrow + k0));
            if (live2) a2 = __ldg(reinterpret_cast<const uint4*>(xrow2 + k0));
          }
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            const uint4& w = w16(b[u][i], i, p + u);
            mma_bf16(c[i], a.x, a2.x, a.y, a2.y, w.x, w.y);
            mma_bf16(c[i], a.z, a2.z, a.w, a2.w, w.z, w.w);
          }
        } else {
          uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0, a2 = a0, a3 = a0;  // row g: a0, a1; row g + 8: a2, a3
          if (live) a0 = __ldg(reinterpret_cast<const uint4*>(xrow + k0)), a1 = __ldg(reinterpret_cast<const uint4*>(xrow + k0 + 8));
          if (live2)
            a2 = __ldg(reinterpret_cast<const uint4*>(xrow2 + k0)), a3 = __ldg(reinterpret_cast<const uint4*>(xrow2 + k0 + 8));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            uint32_t wv[8];
            widen16(b[u][i], s[u][i], wv);
            mma_bf16(c[i], a0.x, a2.x, a0.y, a2.y, wv[0], wv[1]);
            mma_bf16(c[i], a0.z, a2.z, a0.w, a2.w, wv[2], wv[3]);
            mma_bf16(c[i], a1.x, a3.x, a1.y, a3.y, wv[4], wv[5]);
            mma_bf16(c[i], a1.z, a3.z, a1.w, a3.w, wv[6], wv[7]);
          }
        }
      } else {
        constexpr int NV = kB16 ? 8 : 16;  // weight values per lane and step
        float wf[MT][NV];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if constexpr (kB16) {
            const uint4& w = w16(b[u][i], i, p + u);
            const uint32_t* v = &w.x;
#pragma unroll
            for (int j = 0; j < 4; ++j) wf[i][2 * j] = bf16_lo(v[j]), wf[i][2 * j + 1] = bf16_hi(v[j]);
          } else {
            uint32_t wv[8];
            widen16(b[u][i], s[u][i], wv);
#pragma unroll
            for (int j = 0; j < 8; ++j) wf[i][2 * j] = bf16_lo(wv[j]), wf[i][2 * j + 1] = bf16_hi(wv[j]);
          }
        }
#pragma unroll
        for (int m = 0; m < RB; ++m) {
          if (m >= M) break;
          const __nv_bfloat16* xr = x + (size_t)m * ldx + xoff + NV * t + k0;
          float xv[NV];
#pragma unroll
          for (int h = 0; h < NV / 8; ++h) {
            const uint4 a = __ldg(reinterpret_cast<const uint4*>(xr + 8 * h));
            const uint32_t* av = &a.x;
#pragma unroll
            for (int j = 0; j < 4; ++j) xv[8 * h + 2 * j] = bf16_lo(av[j]), xv[8 * h + 2 * j + 1] = bf16_hi(av[j]);
          }
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NV; ++j) acc[i][m] = fmaf(xv[j], wf[i][j], acc[i][m]);
        }
      }
    }
  };

  load(ba, sa, p0);
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (kSx) {
    const bool first = blockIdx.x == 0;
    rmsnorm_rows(nin.x, K, nin.w, M, K, nin.eps, red, sx, K + 32, first ? nin.q : nullptr, K, first ? nin.qs : nullptr,
                 nullptr, nullptr);
    __syncthreads();
  } else if constexpr (NORM) {  // the rows' 1/rms (CTA 0: and the act_quant of the normed rows)
    if (blockIdx.x == 0 && nin.q)
      rmsnorm_rows<RB>(nin.x, K, nin.w, M, K, nin.eps, red, nullptr, 0, nin.q, K, nin.qs, nullptr, nullptr);
    else
      rmsnorm_scale<RB>(nin.x, K, M, K, nin.eps, red);
    rr = red[256 + (live ? g : 0)], rr2 = red[256 + (live2 ? g + 8 : 0)];
    __syncthreads();  // read before the partial sums reuse red
  }
  for (int p = p0; p < p1; p += 2 * KU) {
    if (p + KU < p1) load(bb, sb, p + KU);
    comp(ba, sa, p);
    if (p + KU >= p1) break;
    if (p + 2 * KU < p1) load(ba, sa, p + 2 * KU);
    comp(bb, sb, p + KU);
  }

  if constexpr (!CC) {
    if (live)
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int r = wi_n * 8 * MT + 8 * i + 2 * t;
        red[(wi_k * RB + g) * rows_cta + r] = c[i][0];
        red[(wi_k * RB + g) * rows_cta + r + 1] = c[i][1];
      }
    if constexpr (kHi)
      if (live2)
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const int r = wi_n * 8 * MT + 8 * i + 2 * t;
          red[(wi_k * RB + g + 8) * rows_cta + r] = c[i][2];
          red[(wi_k * RB + g + 8) * rows_cta + r + 1] = c[i][3];
        }
  } else {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int m = 0; m < RB; ++m) {
        if (m >= M) break;
        float v = acc[i][m];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        if (t == 0) red[(wi_k * RB + m) * rows_cta + wi_n * 8 * MT + 8 * i + g] = v;
      }
  }
  __syncthreads();
  if (q) {  // bf16 out and its act_quant: one warp per (row, 32 outputs)
    const int nblk = rows_cta / 32;
    for (int pr = warp; pr < M * nblk; pr += blockDim.x >> 5) {
      const int m = pr / nblk, r = 32 * (pr % nblk) + lane;
      float v = 0.f;
      for (int j = 0; j < wk; ++j) v += red[(j * RB + m) * rows_cta + r];
      const __nv_bfloat16 vb = __float2bfloat16_rn(v);
      if (yb) yb[(size_t)m * N + n_cta + r] = vb;
      const float vf = __bfloat162float(vb);
      const uint32_t e = act_scale_byte(warp_max(fabsf(vf)));
      q[(size_t)m * N + n_cta + r] = (uint8_t)quant_e4m3(vf, e);
      if (lane == 0) qs[(size_t)m * (N / 32) + (n_cta + r) / 32] = (uint8_t)e;
    }
    return;
  }
  for (int idx = threadIdx.x; idx < M * rows_cta; idx += blockDim.x) {
    const int m = idx / rows_cta, r = idx - m * rows_cta;
    float v = 0.f;
    for (int j = 0; j < wk; ++j) v += red[(j * RB + m) * rows_cta + r];
    if (yf) yf[(size_t)m * N + n_cta + r] = v;
    else yb[(size_t)m * N + n_cta + r] = __float2bfloat16_rn(v);
  }
}

template <int FMT, int MT, bool CC, bool NORM>
__global__ void __launch_bounds__(512) k_gemv16(const __nv_bfloat16* __restrict__ x, int M, const uint8_t* __restrict__ W,
                                                 const uint8_t* __restrict__ S, int N, int K, int group_rows,
                                                 float* __restrict__ yf, __nv_bfloat16* __restrict__ yb,
                                                 uint8_t* __restrict__ q, uint8_t* __restrict__ qs, int wn, int wk,
                                                 NormIn nin) {
  gemv16_body<FMT, MT, CC, NORM, kLowRows>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, wn, wk, nin);
}

// 9..16 rows. (512, 1): ptxas would otherwise hold some at 64 registers and spill.
template <int FMT, int MT, bool NORM>
__global__ void __launch_bounds__(512, 1) k_gemv16_r16(const __nv_bfloat16* __restrict__ x, int M,
                                                        const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                        int N, int K, int group_rows, float* __restrict__ yf,
                                                        __nv_bfloat16* __restrict__ yb, uint8_t* __restrict__ q,
                                                        uint8_t* __restrict__ qs, int wn, int wk, NormIn nin) {
  gemv16_body<FMT, MT, false, NORM, kMaxRows>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, wn, wk, nin);
}

template <int FMT, int MT, bool CC, bool NORM, int RB>
constexpr auto gemv16_kernel() {
  if constexpr (RB == kLowRows) return k_gemv16<FMT, MT, CC, NORM>;
  else return k_gemv16_r16<FMT, MT, NORM>;
}

// Persistent version of the bf16 tensor-core path for mt = wn = 1 (tiles of 8 rows, wk warps along K),
// as gemv_fp8.cu's k_gemv_fp8p: a grid of about one CTA per SM, each a contiguous run of tiles; warp w
// streams slice w of every tile through a ring of S 512-byte stages in shared memory (cp.async, lane
// l's own 16 bytes of each step, no scales), S stages per warp requested before the PDL wait; per
// tile one barrier and k_gemv16's fixed-order sum over the slices, so the bits are k_gemv16's.
constexpr int kRingS[] = {16, 12, 8, 4};
constexpr size_t kRingBudget = 80 * 1024;  // as gemv_fp8.cu: hc_project's 16 KB still fits on the SM

size_t g16p_smem(int S, int wk, int M, int K, bool norm) {
  return (size_t)wk * S * 512 + (size_t)std::max(2 * wk * 64, 264) * sizeof(float) + (norm ? (size_t)M * (K + 32) * 2 : 0);
}

template <int S, bool NORM>
__global__ void __launch_bounds__(512) k_gemv16p(const __nv_bfloat16* __restrict__ x, int M, const uint8_t* __restrict__ W,
                                                  int N, int K, float* __restrict__ yf, __nv_bfloat16* __restrict__ yb,
                                                  NormIn nin) {
  extern __shared__ __align__(16) uint8_t sm[];  // ring [wk][S][512] | red [2][wk][8][8] | NORM: x [M][K + 32] bf16
  const int wk = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int rowb = 2 * K, P = rowb / 64, ntile = N / 8;
  uint8_t* const ring = sm + (size_t)warp * S * 512 + 16 * lane;
  float* const red = reinterpret_cast<float*>(sm + (size_t)wk * S * 512);
  __nv_bfloat16* const sx = reinterpret_cast<__nv_bfloat16*>(red + std::max(2 * wk * 64, 264));
  const int p0 = (P * warp) / wk, p1 = (P * (warp + 1)) / wk;
  const int t0 = (int)((long long)ntile * blockIdx.x / gridDim.x);
  const int t1 = (int)((long long)ntile * (blockIdx.x + 1) / gridDim.x);
  int it = t0, ip = p0, si = 0;  // the issue cursor
  auto issue = [&]() {
    if (it < t1) {
      cp_async16(ring + si * 512, W + (size_t)it * 8 * rowb + (size_t)ip * 512 + 16 * lane);
      if (++ip == p1) ip = p0, ++it;
    }
    cp_async_commit();
    si = si + 1 == S ? 0 : si + 1;
  };
#pragma unroll 1
  for (int i = 0; i < S; ++i) issue();
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (NORM) {
    rmsnorm_rows(nin.x, K, nin.w, M, K, nin.eps, red, sx, K + 32, blockIdx.x == 0 ? nin.q : nullptr, K,
                 blockIdx.x == 0 ? nin.qs : nullptr, nullptr, nullptr);
    __syncthreads();
  }
  const bool live = g < M;
  const __nv_bfloat16* xrow = NORM ? sx + (live ? g : 0) * (K + 32) + 8 * t : x + (size_t)(live ? g : 0) * K + 8 * t;
  int slot = 0;
#pragma unroll 1
  for (int tl = t0; tl < t1; ++tl) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int p = p0; p < p1; ++p) {
      cp_async_wait<S - 1>();  // lane l reads only its own copy: no warp sync
      const uint4 b = *reinterpret_cast<const uint4*>(ring + slot * 512);
      issue();
      slot = slot + 1 == S ? 0 : slot + 1;
      uint4 a = make_uint4(0, 0, 0, 0);
      if (live)
        a = NORM ? *reinterpret_cast<const uint4*>(xrow + p * 32) : __ldg(reinterpret_cast<const uint4*>(xrow + p * 32));
      mma_bf16(c, a.x, 0u, a.y, 0u, b.x, b.y);
      mma_bf16(c, a.z, 0u, a.w, 0u, b.z, b.w);
    }
    float* rd = red + (tl & 1) * wk * 64;
    if (live) rd[(warp * 8 + g) * 8 + 2 * t] = c[0], rd[(warp * 8 + g) * 8 + 2 * t + 1] = c[1];
    __syncthreads();
    if (threadIdx.x < M * 8) {
      const int m = threadIdx.x >> 3, r = threadIdx.x & 7;
      float v = 0.f;
      for (int j = 0; j < wk; ++j) v += rd[(j * 8 + m) * 8 + r];
      if (yf) yf[(size_t)m * N + 8 * tl + r] = v;
      else yb[(size_t)m * N + 8 * tl + r] = __float2bfloat16_rn(v);
    }
  }
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
void launch16p(const __nv_bfloat16* x, int M, const uint8_t* W, int N, int K, float* yf, __nv_bfloat16* yb, int wk,
               int ctas, size_t smem, bool pdl, const NormIn& nin, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(k_gemv16p<S, NORM>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(k_gemv16p<S, NORM>, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
    attr = true;
  }
  launch_pdl(k_gemv16p<S, NORM>, dim3(ctas), dim3(32 * wk), smem, st, pdl, x, M, W, N, K, yf, yb, nin);
}

// Grid and ring as gemv_fp8.cu's persist_fp8: as asked, or one CTA per SM and the deepest ring within
// kRingBudget, no deeper than a warp's run. False when even the shallowest ring does not fit beside
// the normed input (8 rows of 5120: 80 KB): the caller runs k_gemv16.
template <bool NORM>
bool persist16(const __nv_bfloat16* x, int M, const uint8_t* W, int N, int K, float* yf, __nv_bfloat16* yb, GemvCfg cfg,
               const NormIn& nin, cudaStream_t st) {
  if (N % 8 || reinterpret_cast<uintptr_t>(W) % 16) die("gemv16: persistent kernel, N=%d", N);
  const int ntile = N / 8, wk = cfg.wk;
  const int ctas = std::min(ntile, cfg.ctas > 0 ? cfg.ctas : sm_count());
  int S = cfg.ring;
  if (S == 0) {
    const int run = (ntile + ctas - 1) / ctas * ((2 * K / 64 + wk - 1) / wk);
    for (int s : kRingS)
      if (g16p_smem(s, wk, M, K, NORM) <= kRingBudget && (S == 0 || s >= run)) S = s;
    if (S == 0) S = 4;
  }
  const size_t smem = g16p_smem(S, wk, M, K, NORM);
  if (smem > 99 * 1024) return false;
  if (wk > 16 || wk > 2 * K / 64) die("gemv16: persistent kernel, wk=%d S=%d", wk, S);
  switch (S) {
    case 16: launch16p<16, NORM>(x, M, W, N, K, yf, yb, wk, ctas, smem, cfg.pdl, nin, st); break;
    case 12: launch16p<12, NORM>(x, M, W, N, K, yf, yb, wk, ctas, smem, cfg.pdl, nin, st); break;
    case 8: launch16p<8, NORM>(x, M, W, N, K, yf, yb, wk, ctas, smem, cfg.pdl, nin, st); break;
    case 4: launch16p<4, NORM>(x, M, W, N, K, yf, yb, wk, ctas, smem, cfg.pdl, nin, st); break;
    default: die("gemv16: ring %d (4, 8, 12 or 16)", S);
  }
  return true;
}

template <int FMT, int MT, bool CC, bool NORM = false, int RB = kLowRows>
void launch16(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, int group_rows, float* yf,
              __nv_bfloat16* yb, uint8_t* q, uint8_t* qs, GemvCfg cfg, cudaStream_t st, const NormIn& nin = NormIn{}) {
  const int rows_cta = cfg.wn * 8 * MT;
  if (N % rows_cta || (FMT == kWOA && group_rows % rows_cta)) die("gemv16: N=%d not a multiple of %d", N, rows_cta);
  if (q && rows_cta % 32) die("gemv16: fused act_quant needs 32 rows per CTA (mt*wn = 4), got %d", rows_cta);
  size_t smem = (size_t)cfg.wk * RB * rows_cta * sizeof(float);
  if (NORM) {
    smem = (size_t)std::max(cfg.wk * RB * rows_cta, 256 + RB) * sizeof(float) +
           (RB == kLowRows ? (size_t)M * (K + 32) * 2 : 0);
    static size_t attr = 48 * 1024;
    if (smem > attr) {
      CUDA_CHECK(cudaFuncSetAttribute(gemv16_kernel<FMT, MT, CC, NORM, RB>(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)smem));
      attr = smem;
    }
  }
  launch_pdl(gemv16_kernel<FMT, MT, CC, NORM, RB>(), dim3(N / rows_cta), dim3(32 * cfg.wn * cfg.wk), smem, st, cfg.pdl, x,
             M, W, S, N, K, group_rows, yf, yb, q, qs, cfg.wn, cfg.wk, nin);
}

template <int FMT>
void dispatch16(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, int group_rows, float* yf,
                __nv_bfloat16* yb, uint8_t* q, uint8_t* qs, GemvCfg cfg, bool cc, cudaStream_t st) {
  if (M < 1 || M > kMaxRows || cfg.wn * cfg.wk > 16 || (FMT != kWOA ? 2 * K : K) % 64)
    die("gemv16: unsupported call M=%d K=%d", M, K);
  if (M > kLowRows) {  // 9..16 rows: tensor cores, mt 1 or 2 (4 spills at 8 rows already)
    if (cc) die("gemv16: CUDA cores take up to %d rows, not %d", kLowRows, M);
    switch (cfg.mt) {
      case 1: launch16<FMT, 1, false, false, kMaxRows>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      case 2: launch16<FMT, 2, false, false, kMaxRows>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      default: die("gemv16: mt=%d at %d rows (1 or 2)", cfg.mt, M);
    }
    return;
  }
  if constexpr (FMT == kBF16X) {  // tensor cores, mt 1 or 2
    switch (cfg.mt * 2 + cc) {
      case 2: launch16<FMT, 1, false>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      case 4: launch16<FMT, 2, false>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      default: die("gemv_bf16x: mt=%d cc=%d (mt 1 or 2, tensor cores)", cfg.mt, cc);
    }
  } else {
    switch (cfg.mt * 2 + cc) {
      case 2: launch16<FMT, 1, false>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      case 3: launch16<FMT, 1, true>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      case 4: launch16<FMT, 2, false>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      case 5: launch16<FMT, 2, true>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      case 8: launch16<FMT, 4, false>(x, M, W, S, N, K, group_rows, yf, yb, q, qs, cfg, st); break;
      default: die("gemv16: mt=%d cc=%d", cfg.mt, cc);
    }
  }
}

// ---- kBF16X: the coding (a warp per tile)

// A tile's least and greatest exponent -> its header; a raw tile's is 0x80000000 until the host numbers it
__global__ void k_bf16x_scan(const uint8_t* __restrict__ W, size_t T, uint32_t* __restrict__ hdr) {
  const size_t tile = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (tile >= T) return;  // the whole warp
  const uint4 v = ldg_stream128(W + tile * 512 + 16 * lane);
  const uint32_t* w = &v.x;
  uint32_t lo = 255, hi = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t e = (w[j / 2] >> (7 + 16 * (j & 1))) & 0xFF;
    lo = min(lo, e), hi = max(hi, e);
  }
  lo = __reduce_min_sync(0xffffffffu, lo), hi = __reduce_max_sync(0xffffffffu, hi);
  if (lane == 0) hdr[tile] = hi - lo <= 15 ? (lo << 23) | (lo << 7) : 0x80000000u;
}

// A tile into X by its header: coded, or copied to its raw slot
__global__ void k_bf16x_pack(const uint8_t* __restrict__ W, size_t T, const uint32_t* __restrict__ hdr,
                             uint8_t* __restrict__ X) {
  const size_t tile = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (tile >= T) return;
  const uint4 v = ldg_stream128(W + tile * 512 + 16 * lane);
  const uint32_t h = hdr[tile];
  if (lane == 0) reinterpret_cast<uint32_t*>(X + T * 384)[tile] = h;
  if (h >> 31) {
    *reinterpret_cast<uint4*>(X + bf16x_raw_off(T) + (size_t)(h & 0x7FFFFFFFu) * 512 + 16 * lane) = v;
    return;
  }
  const uint32_t base = (h >> 7) & 0xFF;
  const uint32_t* w = &v.x;
  uint32_t s[2] = {0, 0}, e = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {  // value j: bits 16 (j & 1) of word j / 2
    const uint32_t b = (w[j / 2] >> (16 * (j & 1))) & 0xFFFF;
    s[j / 4] |= (((b >> 8) & 0x80) | (b & 0x7F)) << (8 * (j % 4));
    e |= (((b >> 7) & 0xFF) - base) << (j & 1 ? 16 + 4 * (j / 2) : 4 * (j / 2));
  }
  *reinterpret_cast<uint2*>(X + tile * 256 + 8 * lane) = make_uint2(s[0], s[1]);
  reinterpret_cast<uint32_t*>(X + T * 256 + tile * 128)[lane] = e;
}

// Every lane's 16 bytes of every tile, decoded as the GEMV does, against W: the differing ones into *bad
__global__ void k_bf16x_check(const uint8_t* __restrict__ X, const uint8_t* __restrict__ W, size_t T,
                              unsigned long long* bad) {
  const size_t tile = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (tile >= T) return;
  const uint32_t h = reinterpret_cast<const uint32_t*>(X + T * 384)[tile];
  uint4 d;
  if (h >> 31) {  // the GEMV's address
    const uint64_t rawq = reinterpret_cast<uint64_t>(X + bf16x_raw_off(T) + 16 * lane) - (1ull << 40);
    d = *reinterpret_cast<const uint4*>(rawq + (uint64_t)h * 512);
  } else {
    const uint2 sm = *reinterpret_cast<const uint2*>(X + tile * 256 + 8 * lane);
    d = bf16x_dec(sm.x, sm.y, reinterpret_cast<const uint32_t*>(X + T * 256 + tile * 128)[lane], h);
  }
  const uint4 v = *reinterpret_cast<const uint4*>(W + tile * 512 + 16 * lane);
  if (d.x != v.x || d.y != v.y || d.z != v.z || d.w != v.w) atomicAdd(bad, 1ull);
}

size_t bf16x_tiles(int N, int K) {
  if (N % 8 || (2 * K) % 64) die("bf16x: N=%d K=%d (N a multiple of 8, K of 32)", N, K);
  const size_t T = (size_t)(N / 8) * (2 * K / 64);
  if (T >= (1u << 31)) die("bf16x: %zu tiles, fewer than 2^31 needed", T);
  return T;
}

}  // namespace

uint8_t* bf16x_encode(const uint8_t* W, int N, int K, size_t* bytes, int* raw, cudaStream_t st) {
  const size_t T = bf16x_tiles(N, K);
  const unsigned blocks = (unsigned)((T * 32 + 255) / 256);
  uint32_t* hdr = nullptr;
  CUDA_CHECK(cudaMalloc(&hdr, T * 4));
  k_bf16x_scan<<<blocks, 256, 0, st>>>(W, T, hdr);
  CUDA_CHECK(cudaGetLastError());
  std::vector<uint32_t> h(T);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), hdr, T * 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  uint32_t r = 0;  // the raw tiles in tile order
  for (uint32_t& x : h)
    if (x >> 31) x = 0x80000000u | r++;
  CUDA_CHECK(cudaMemcpyAsync(hdr, h.data(), T * 4, cudaMemcpyHostToDevice, st));
  *bytes = bf16x_raw_off(T) + (size_t)r * 512;
  *raw = (int)r;
  uint8_t* X = nullptr;
  CUDA_CHECK(cudaMalloc(&X, *bytes));
  CUDA_CHECK(cudaMemsetAsync(X, 0, *bytes, st));  // the padding after the headers
  k_bf16x_pack<<<blocks, 256, 0, st>>>(W, T, hdr, X);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaFree(hdr));
  return X;
}

uint64_t bf16x_check(const uint8_t* X, const uint8_t* W, int N, int K, cudaStream_t st) {
  const size_t T = bf16x_tiles(N, K);
  unsigned long long* bad = nullptr;
  CUDA_CHECK(cudaMalloc(&bad, sizeof *bad));
  CUDA_CHECK(cudaMemsetAsync(bad, 0, sizeof *bad, st));
  k_bf16x_check<<<(unsigned)((T * 32 + 255) / 256), 256, 0, st>>>(X, W, T, bad);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long n = 0;
  CUDA_CHECK(cudaMemcpyAsync(&n, bad, sizeof n, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaFree(bad));
  return n;
}

void gemv_bf16(const __nv_bfloat16* x, int M, const uint8_t* W, int N, int K, float* yf, __nv_bfloat16* yb, GemvCfg cfg,
               bool cuda_cores, cudaStream_t st) {
  if (cfg.ring >= 0 && cfg.mt == 1 && cfg.wn == 1 && !cuda_cores && M >= 1 && M <= kLowRows && (2 * K) % 64 == 0 &&
      persist16<false>(x, M, W, N, K, yf, yb, cfg, NormIn{}, st))
    return;
  dispatch16<kBF16>(x, M, W, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, cuda_cores, st);
}

void gemv_bf16_norm(const __nv_bfloat16* x, const __nv_bfloat16* norm_w, float eps, int M, const uint8_t* W, int N, int K,
                    float* yf, __nv_bfloat16* yb, GemvCfg cfg, cudaStream_t st, uint8_t* q_out, uint8_t* qs_out) {
  if (M < 1 || M > kMaxRows || cfg.wn * cfg.wk > 16 || K % 256) die("gemv_bf16_norm: unsupported call M=%d K=%d", M, K);
  const NormIn nin{x, norm_w, eps, nullptr, q_out, qs_out};
  if (M > kLowRows) {  // 9..16 rows: x normalised as the kernel reads it
    switch (cfg.mt) {
      case 1: launch16<kBF16, 1, false, true, kMaxRows>(nullptr, M, W, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
      case 2: launch16<kBF16, 2, false, true, kMaxRows>(nullptr, M, W, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
      default: die("gemv_bf16_norm: mt=%d", cfg.mt);
    }
    return;
  }
  if (cfg.ring < 0 && cfg.mt == 1 && cfg.wn == 1 && M <= persist_rows()) cfg.ring = 0;
  if (cfg.ring >= 0 && cfg.mt == 1 && cfg.wn == 1 && persist16<true>(nullptr, M, W, N, K, yf, yb, cfg, nin, st)) return;
  switch (cfg.mt) {
    case 1: launch16<kBF16, 1, false, true>(nullptr, M, W, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
    case 2: launch16<kBF16, 2, false, true>(nullptr, M, W, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
    default: die("gemv_bf16_norm: mt=%d", cfg.mt);
  }
}

void gemv_bf16x(const __nv_bfloat16* x, int M, const uint8_t* X, int N, int K, float* yf, __nv_bfloat16* yb, GemvCfg cfg,
                cudaStream_t st) {
  dispatch16<kBF16X>(x, M, X, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, false, st);
}

void gemv_bf16x_norm(const __nv_bfloat16* x, const __nv_bfloat16* norm_w, float eps, int M, const uint8_t* X, int N, int K,
                     float* yf, __nv_bfloat16* yb, GemvCfg cfg, cudaStream_t st) {
  if (M < 1 || M > kMaxRows || cfg.wn * cfg.wk > 16 || K % 256) die("gemv_bf16x_norm: unsupported call M=%d K=%d", M, K);
  const NormIn nin{x, norm_w, eps, nullptr, nullptr, nullptr};
  if (M > kLowRows) {
    switch (cfg.mt) {
      case 1: launch16<kBF16X, 1, false, true, kMaxRows>(nullptr, M, X, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
      case 2: launch16<kBF16X, 2, false, true, kMaxRows>(nullptr, M, X, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
      default: die("gemv_bf16x_norm: mt=%d", cfg.mt);
    }
    return;
  }
  switch (cfg.mt) {
    case 1: launch16<kBF16X, 1, false, true>(nullptr, M, X, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
    case 2: launch16<kBF16X, 2, false, true>(nullptr, M, X, nullptr, N, K, N, yf, yb, nullptr, nullptr, cfg, st, nin); break;
    default: die("gemv_bf16x_norm: mt=%d", cfg.mt);
  }
}

void gemv_woa(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, int group_rows,
              __nv_bfloat16* y, GemvCfg cfg, bool cuda_cores, cudaStream_t st, uint8_t* q, uint8_t* qs) {
  dispatch16<kWOA>(x, M, W, S, N, K, group_rows, nullptr, y, q, qs, cfg, cuda_cores, st);
}

void gemv_fp8w_f32(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, float* yf,
                   GemvCfg cfg, cudaStream_t st) {
  dispatch16<kWOA>(x, M, W, S, N, K, N, yf, nullptr, nullptr, nullptr, cfg, false, st);
}

}  // namespace dsk
