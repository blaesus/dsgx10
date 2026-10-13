// act_quant: bf16 -> e4m3 per 32 elements with a power-of-two (ue8m0) scale, as the reference.
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

namespace dsk {

// One warp per 32-element block, one element per lane.
__global__ void k_act_quant(const __nv_bfloat16* __restrict__ x, int n, uint8_t* __restrict__ q,
                            uint8_t* __restrict__ s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;  // n is a multiple of 32, so whole warps leave together
  float v = __bfloat162float(x[i]);
  uint32_t e = act_scale_byte(warp_max(fabsf(v)));
  q[i] = (uint8_t)quant_e4m3(v, e);
  if ((threadIdx.x & 31) == 0) s[i >> 5] = (uint8_t)e;
}

void act_quant(const __nv_bfloat16* x, int M, int K, uint8_t* q, uint8_t* s, cudaStream_t st) {
  int n = M * K;
  k_act_quant<<<(n + 255) / 256, 256, 0, st>>>(x, n, q, s);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace dsk
