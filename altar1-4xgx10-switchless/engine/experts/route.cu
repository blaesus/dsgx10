// The router's selection (model.md 2.7): s = sigmoid(logits) in f32, sel = s + e_score_correction_bias,
// the top 8 of sel (descending, ties to the lower expert id; n_group = topk_group = 1 keeps all 168),
// w = s[idx] / (sum + 1e-20) * 2.5. One warp a row, six experts a lane (e = lane + 32 j < 168).
#include "common/check.h"
#include "experts/formats.cuh"

namespace gx {

namespace {

__global__ void __launch_bounds__(128) k_route(const float* __restrict__ logits, const float* __restrict__ bias, int M,
                                               int* __restrict__ idx, float* __restrict__ wout) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  pdl_launch_dependents();
  pdl_wait();
  if (row >= M) return;
  route_warp(logits + (size_t)row * kE, bias, lane, idx + row * kTopK, wout + row * kTopK);
}

}  // namespace

void route(const float* logits, const float* bias, int M, int* idx, float* w, cudaStream_t st, bool pdl) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3((M + 3) / 4);
  cfg.blockDim = dim3(128);
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_route, logits, bias, M, idx, w));
}

}  // namespace gx
