// mHC device code shared with the runner's fused collectives (engine/model/fused.cu): one row's mixes
// from its summed partials, on one warp (hc_finish_kernel's arithmetic, op for op).
#pragma once
#include "attn/attn.h"
#include "attn/attn_common.cuh"

namespace attn {

__device__ __forceinline__ float sigmoidf_ref(float v) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v))); }

// sum4_last and sum4_seq across lanes: lane 8 + 4j + k holds comb[j][k]
__device__ __forceinline__ float row_sum4(float v) {  // (c0 + c2) + (c1 + c3) of the lane's row (commuted)
  float a = __shfl_xor_sync(0xffffffffu, v, 2), b = __shfl_xor_sync(0xffffffffu, v, 1);
  float c = __shfl_xor_sync(0xffffffffu, v, 3);
  return __fadd_rn(__fadd_rn(v, a), __fadd_rn(b, c));
}
__device__ __forceinline__ float col_sum4(float v, int lane) {  // ((c0 + c1) + c2) + c3 of the lane's column
  int k = lane & 3;
  float a0 = __shfl_sync(0xffffffffu, v, 8 + k), a1 = __shfl_sync(0xffffffffu, v, 12 + k);
  float a2 = __shfl_sync(0xffffffffu, v, 16 + k), a3 = __shfl_sync(0xffffffffu, v, 20 + k);
  return __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
}

// What lane i of hc_sinkhorn_warp needs from the sublayer's hc_scale [3] and hc_base [24].
__device__ __forceinline__ float hc_lane_scale(const float* scale, int lane) {
  return scale[lane < 4 ? 0 : lane < 8 ? 1 : 2];
}
__device__ __forceinline__ float hc_lane_base(const float* base, int lane) { return lane < kMix ? base[lane] : 0.f; }

// One row's mixes on one warp; every lane calls it. Lane i < 25 brings the row's summed partial i (24:
// the sum of squares), sc and bs from hc_lane_scale / hc_lane_base. Lane i < 24 gets mix i: pre (lanes
// 0-3), post (4-7) or comb[j][k] (8 + 4j + k), the Sinkhorn on 16 lanes.
__device__ __forceinline__ float hc_sinkhorn_warp(float mv, int lane, float sc, float bs) {
  float rs = rsqrtf(__fadd_rn(__fmul_rn(__shfl_sync(0xffffffffu, mv, 24), 1.0f / kHcDim), kNormEps));
  float a = __fadd_rn(__fmul_rn(__fmul_rn(mv, rs), sc), bs);
  float sg = sigmoidf_ref(a);
  float mx = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
  mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
  float e = expf(__fsub_rn(a, mx));
  float c = __fadd_rn(__fdiv_rn(e, row_sum4(e)), kHcEps);
  c = __fdiv_rn(c, __fadd_rn(col_sum4(c, lane), kHcEps));
  for (int it = 1; it < 20; ++it) {
    c = __fdiv_rn(c, __fadd_rn(row_sum4(c), kHcEps));
    c = __fdiv_rn(c, __fadd_rn(col_sum4(c, lane), kHcEps));
  }
  return lane < 4 ? __fadd_rn(sg, kHcEps) : lane < 8 ? __fmul_rn(2.0f, sg) : c;
}

}  // namespace attn
