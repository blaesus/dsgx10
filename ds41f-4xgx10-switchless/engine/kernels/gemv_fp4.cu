// Weight-streaming GEMV with MXFP4 weights (e2m1 under a power-of-two scale per 32 values of a row) and bf16
// activations, fp32 out, for a few rows: the drafter's head as MXFP4 (docs/engine.md, "The draft pass's
// approximations"). A lane widens its e2m1 values to bf16 in registers (times their group's scale, exact) and
// runs gemv16.cu's kWOA MMAs on them (m16n8k16 bf16, fp32 accumulation; split-K across the warps of a CTA,
// fixed-order reduction), the same steps in the same order: on weights that the FP8 head also holds, with the
// same wk, gemv_fp4w_f32 gives gemv_fp8w_f32's bits at half the weight bytes.
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

namespace dsk {

namespace {

// 16 e2m1 weights (8 bytes in lo, hi; value 2i in byte i's low nibble) -> 8 bf16x2 words of w * 2^(s-127),
// word i holding values 2i, 2i + 1 (widen16's order): each nibble's sign, exponent and mantissa bits moved into
// bf16's fields (the value times 2^-126, the subnormal 0.5 included), then one multiply by 2^(s-1) (exact;
// needs s <= 128, scales <= 2).
__device__ __forceinline__ void widen16_fp4(uint32_t lo, uint32_t hi, uint32_t s, uint32_t (&o)[8]) {
  const __nv_bfloat162 f = __bfloat162bfloat162(__ushort_as_bfloat16((uint16_t)((s + 126u) << 7)));
  const uint32_t v[2] = {lo, hi};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t w4 = v[h] >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // byte j's low nibble at bits 3:0 and its high nibble at bits 19:16 (the other bits are not read)
      const uint32_t y = __byte_perm(v[h], w4, (4 + j) << 12 | (4 + j) << 8 | j << 4 | j);
      uint32_t b = ((y << 12) & 0x80008000u) | ((y << 6) & 0x01C001C0u);
      __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&b), f);
      o[4 * h + j] = *reinterpret_cast<uint32_t*>(&r);
    }
  }
}

// W [N/8][K/128][8 rows][4][16 B]: lane (g, t) of tile (n/8, j) holds row g's values k = 128j + 16t + i (bytes
// 0-7, i = 0..15: kWOA's step 2j) and k = 128j + 64 + 16t + i (bytes 8-15: step 2j + 1); S [N/8][K/128][8 rows]
// [4]: row g's scales of the groups 4j .. 4j + 3. Warp wi_k takes kWOA's steps p0 .. p1 - 1 (even bounds: the
// host checks), in batches of KU tiles, double-buffered, the scales beside their tiles.
template <int MT>
__global__ void __launch_bounds__(512) k_gemv_fp4w(const __nv_bfloat16* __restrict__ x, int M, const uint8_t* __restrict__ W,
                                                    const uint8_t* __restrict__ S, int N, int K, float* __restrict__ yf,
                                                    int wn, int wk) {
  constexpr int RB = kLowRows, KU = 2;
  extern __shared__ float red[];  // [wk][RB][rows_cta]
  const int rows_cta = wn * 8 * MT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % wn, wi_k = warp / wn;
  const int n_cta = blockIdx.x * rows_cta, n0 = n_cta + wi_n * 8 * MT;
  const int J = K / 128, P = K / 64;
  const int j0 = (P * wi_k) / wk / 2, j1 = (P * (wi_k + 1)) / wk / 2;
  const bool live = g < M;

  const uint8_t* wrow[MT];
  const uint8_t* srow[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int n = n0 + 8 * i;
    wrow[i] = W + (size_t)(n / 8) * J * 512 + 16 * lane;
    srow[i] = S + (size_t)(n / 8) * J * 32 + 4 * g;
  }
  const __nv_bfloat16* xrow = x + (size_t)(live ? g : 0) * K + 16 * t;
  const int sh = 8 * (t >> 1);  // the lane's scale byte: bits sh of the tile's word (step 2j), sh + 16 (step 2j + 1)

  float c[MT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;

  uint4 ba[KU][MT], bb[KU][MT];
  uint32_t sa[KU][MT], sb[KU][MT];
  auto load = [&](uint4 (&b)[KU][MT], uint32_t (&s)[KU][MT], int j) {
#pragma unroll
    for (int u = 0; u < KU; ++u)
      if (j + u < j1)
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          b[u][i] = ldg_stream128(wrow[i] + (size_t)(j + u) * 512);
          s[u][i] = __ldg(reinterpret_cast<const uint32_t*>(srow[i] + (size_t)(j + u) * 32));
        }
  };
  auto comp = [&](const uint4 (&b)[KU][MT], const uint32_t (&s)[KU][MT], int j) {
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      if (j + u >= j1) break;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int k0 = (2 * (j + u) + h) * 64;
        uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0;
        if (live) a0 = __ldg(reinterpret_cast<const uint4*>(xrow + k0)), a1 = __ldg(reinterpret_cast<const uint4*>(xrow + k0 + 8));
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          uint32_t wv[8];
          widen16_fp4(h ? b[u][i].z : b[u][i].x, h ? b[u][i].w : b[u][i].y, (s[u][i] >> (sh + 16 * h)) & 0xFFu, wv);
          mma_bf16(c[i], a0.x, 0u, a0.y, 0u, wv[0], wv[1]);
          mma_bf16(c[i], a0.z, 0u, a0.w, 0u, wv[2], wv[3]);
          mma_bf16(c[i], a1.x, 0u, a1.y, 0u, wv[4], wv[5]);
          mma_bf16(c[i], a1.z, 0u, a1.w, 0u, wv[6], wv[7]);
        }
      }
    }
  };

  load(ba, sa, j0);
  pdl_launch_dependents();
  pdl_wait();
  for (int j = j0; j < j1; j += 2 * KU) {
    if (j + KU < j1) load(bb, sb, j + KU);
    comp(ba, sa, j);
    if (j + KU >= j1) break;
    if (j + 2 * KU < j1) load(ba, sa, j + 2 * KU);
    comp(bb, sb, j + KU);
  }

  // gemv16_body's reduction: the warps' partial sums over K, added in warp order
  if (live)
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int r = wi_n * 8 * MT + 8 * i + 2 * t;
      red[(wi_k * RB + g) * rows_cta + r] = c[i][0];
      red[(wi_k * RB + g) * rows_cta + r + 1] = c[i][1];
    }
  __syncthreads();
  for (int idx = threadIdx.x; idx < M * rows_cta; idx += blockDim.x) {
    const int m = idx / rows_cta, r = idx - m * rows_cta;
    float v = 0.f;
    for (int j = 0; j < wk; ++j) v += red[(j * RB + m) * rows_cta + r];
    yf[(size_t)m * N + n_cta + r] = v;
  }
}

template <int MT>
void launch_fp4w(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, float* yf, GemvCfg cfg,
                 cudaStream_t st) {
  const int rows_cta = cfg.wn * 8 * MT;
  if (N % rows_cta) die("gemv_fp4w_f32: N=%d not a multiple of %d", N, rows_cta);
  launch_pdl(k_gemv_fp4w<MT>, dim3(N / rows_cta), dim3(32 * cfg.wn * cfg.wk), (size_t)cfg.wk * kLowRows * rows_cta * 4,
             st, cfg.pdl, x, M, W, S, N, K, yf, cfg.wn, cfg.wk);
}

}  // namespace

void gemv_fp4w_f32(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, float* yf,
                   GemvCfg cfg, cudaStream_t st) {
  if (M < 1 || M > kLowRows || cfg.wn < 1 || cfg.wk < 1 || cfg.wn * cfg.wk > 16 || K % (128 * cfg.wk))
    die("gemv_fp4w_f32: unsupported call M=%d K=%d wn=%d wk=%d", M, K, cfg.wn, cfg.wk);
  switch (cfg.mt) {
    case 1: launch_fp4w<1>(x, M, W, S, N, K, yf, cfg, st); break;
    case 2: launch_fp4w<2>(x, M, W, S, N, K, yf, cfg, st); break;
    case 4: launch_fp4w<4>(x, M, W, S, N, K, yf, cfg, st); break;
    default: die("gemv_fp4w_f32: mt=%d (1, 2 or 4)", cfg.mt);
  }
}

}  // namespace dsk
