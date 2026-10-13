// The router's selection (model.md 2.7): s = sigmoid(logits) in f32, sel = s + e_score_correction_bias,
// the top 8 of sel (descending, ties to the lower expert id; n_group = topk_group = 1 keeps all 256),
// w = s[idx] / (sum + 1e-20) * 2.5. One warp a row, eight experts a lane (e = lane + 32 j).
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
  float s[8], key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    s[j] = route_sigmoid(__ldcg(logits + (size_t)row * kE + lane + 32 * j));
    key[j] = s[j] + __ldg(bias + lane + 32 * j);
  }
  float picked[kTopK];
  int ids[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float bv = -INFINITY;
    int bj = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (key[j] > bv) bv = key[j], bj = j;
    int be = bv == -INFINITY ? 1 << 30 : lane + 32 * bj;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ov > bv || (ov == bv && oe < be)) bv = ov, be = oe;
    }
    ids[k] = be;
    float sv = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (lane + 32 * j == be) sv = s[j], key[j] = -INFINITY;
    picked[k] = __shfl_sync(0xffffffffu, sv, be & 31);
  }
  const float sum = route_sum8(picked) + 1e-20f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k)
    if (lane == k) idx[row * kTopK + k] = ids[k], wout[row * kTopK + k] = picked[k] / sum * kRouteScale;
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
