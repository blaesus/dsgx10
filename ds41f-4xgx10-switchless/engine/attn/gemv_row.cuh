// gemv_kernel's arithmetic (gemv.cu) for one output row over K = kDim with the row's weights held in
// registers, so the fused decode kernels can load them before their PDL wait.
#pragma once
#include "attn/attn.h"
#include "attn/attn_common.cuh"

namespace attn {

constexpr int kRowWords = kDim / 8 / 32;  // uint4 words of a weight row per lane: lane l holds l + 32i

__device__ __forceinline__ void gemv_row_load(const uint16_t* w, int lane, uint4 (&wv)[kRowWords]) {
#pragma unroll
  for (int i = 0; i < kRowWords; ++i) wv[i] = __ldg(reinterpret_cast<const uint4*>(w) + lane + 32 * i);
}

// acc[r] = x[r] . w for r < nr, x rows of kDim bf16 (shared or global); then warp_sum, as gemv_kernel.
__device__ __forceinline__ void gemv_row_dot(const uint4 (&wv)[kRowWords], const uint16_t* x, int nr, int lane,
                                             float (&acc)[8]) {
#pragma unroll
  for (int r = 0; r < 8; ++r) acc[r] = 0.f;
#pragma unroll
  for (int i = 0; i < kRowWords; ++i) {
    const uint32_t wd[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r >= nr) break;
      uint4 xv = reinterpret_cast<const uint4*>(x + (size_t)r * kDim)[lane + 32 * i];
      const uint32_t xd[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[r] = __fmaf_rn(__uint_as_float(xd[j] << 16), __uint_as_float(wd[j] << 16), acc[r]);
        acc[r] = __fmaf_rn(__uint_as_float(xd[j] & 0xffff0000u), __uint_as_float(wd[j] & 0xffff0000u), acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) acc[r] = warp_sum(acc[r]);
}

}  // namespace attn
