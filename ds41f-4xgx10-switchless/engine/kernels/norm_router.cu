// RMSNorm (fp32 math, bf16 out, optionally fused with act_quant of the output) and the router's
// selection step (sqrt(softplus), bias for selection only, top-k, normalise, scale), as model.py.
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

namespace dsk {

namespace {

// One CTA per row (K/8 threads, at most 1024).
__global__ void __launch_bounds__(1024) k_rmsnorm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                                  int D, float eps, __nv_bfloat16* __restrict__ y, uint8_t* __restrict__ q,
                                                  uint8_t* __restrict__ s) {
  __shared__ float red[264];
  const size_t row = blockIdx.x;
  pdl_launch_dependents();
  pdl_wait();
  rmsnorm_rows(x + row * D, D, w, 1, D, eps, red, y + row * D, D, q ? q + row * D : nullptr, D,
               s ? s + row * (D / 32) : nullptr, nullptr, nullptr);
}

// One warp per row.
__global__ void k_router_topk(const float* __restrict__ logits, const float* __restrict__ bias, int E, int M,
                              int* __restrict__ idx, float* __restrict__ wts, float route_scale, const float* bias_vl, const int* image_mask) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  pdl_launch_dependents();
  pdl_wait();
  if (warp >= M) return;
  int id;
  float wv;
  router_row(logits + (size_t)warp * E, image_mask && image_mask[warp] ? bias_vl : bias, E, route_scale, id, wv);
  if (lane < kTopK) idx[warp * kTopK + lane] = id, wts[warp * kTopK + lane] = wv;
}

}  // namespace

void rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, int M, int D, float eps, __nv_bfloat16* y, uint8_t* q,
             uint8_t* s, cudaStream_t st) {
  if (D % 256 || D > 8192) die("rmsnorm: D=%d", D);
  launch_pdl(k_rmsnorm, dim3(M), dim3(D / 8), 0, st, true, x, w, D, eps, y, q, s);
}

void router_topk(const float* logits, const float* bias, int M, int E, int* idx, float* wts, float route_scale,
                 cudaStream_t st, const float* bias_vl, const int* image_mask) {
  if (E > 512) die("router_topk: E=%d", E);
  launch_pdl(k_router_topk, dim3((M + 3) / 4), dim3(128), 0, st, true, logits, bias, E, M, idx, wts, route_scale, bias_vl, image_mask);
}

}  // namespace dsk
