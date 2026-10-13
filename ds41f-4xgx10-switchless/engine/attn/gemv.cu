// The small bf16-weight projections of the attention side (compressor wkv/wgate, indexer weights_proj).
#include <algorithm>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/gemv_row.cuh"
#include "attn/launch.h"
#include "common/check.h"

namespace attn {
namespace {

constexpr int kGemvRows = 8;  // rows per pass; more rows re-read the weights

// y[r][n] = sum_k x[r][k] * W[n][k] in fp32 (bf16 x bf16 products are exact, so FMA == mul + add).
// One warp per output n for up to kGemvRows rows; grid.y walks row groups. n < N0 reads W0 into y0,
// else W1 into y1. round_bf16 rounds the result to bf16 (a bf16 F.linear), else it stays fp32.
__global__ void __launch_bounds__(256) gemv_kernel(const uint16_t* x, int rows, int K, const uint16_t* W0,
                                                   const uint16_t* W1, int N0, int N, float* y0, float* y1,
                                                   int round_bf16) {
  extern __shared__ uint16_t xs[];  // [kGemvRows][K]
  int r0 = blockIdx.y * kGemvRows, nr = min(kGemvRows, rows - r0);
  for (int i = threadIdx.x; i < nr * K / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x + (size_t)r0 * K)[i];
  __syncthreads();
  int n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  if (n >= N) return;
  const uint16_t* w = n < N0 ? W0 + (size_t)n * K : W1 + (size_t)(n - N0) * K;
  float acc[kGemvRows];
#pragma unroll
  for (int r = 0; r < kGemvRows; ++r) acc[r] = 0.f;
  for (int c = lane; c < K / 8; c += 32) {
    uint4 wv = reinterpret_cast<const uint4*>(w)[c];
    const uint32_t wd[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int r = 0; r < kGemvRows; ++r) {
      if (r >= nr) break;
      uint4 xv = reinterpret_cast<const uint4*>(xs + (size_t)r * K)[c];
      const uint32_t xd[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[r] = __fmaf_rn(__uint_as_float(xd[j] << 16), __uint_as_float(wd[j] << 16), acc[r]);
        acc[r] = __fmaf_rn(__uint_as_float(xd[j] & 0xffff0000u), __uint_as_float(wd[j] & 0xffff0000u), acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kGemvRows; ++r) {
    if (r >= nr) break;
    float v = warp_sum(acc[r]);
    if (lane == 0) {
      if (round_bf16) v = bfround(v);
      if (n < N0)
        y0[(size_t)(r0 + r) * N0 + n] = v;
      else
        y1[(size_t)(r0 + r) * (N - N0) + n - N0] = v;
    }
  }
}

// The fused decode path's projection: output rows [n0, n0 + n) of W0 into y0 [rows][n] and, with W1, the same
// rows of W1 into y1, one warp per output; the weight row is loaded before the PDL wait. K = kDim.
__global__ void __launch_bounds__(256) gemv_rows_kernel(const uint16_t* x, int rows, const uint16_t* W0,
                                                        const uint16_t* W1, int n0, int n, int nout, float* y0,
                                                        float* y1, int round_bf16) {
  extern __shared__ uint16_t xs[];  // [rows][kDim]
  pdl_trigger();
  int o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  uint4 wv[kRowWords];
  if (o < nout) gemv_row_load(o < n ? W0 + (size_t)(n0 + o) * kDim : W1 + (size_t)(n0 + o - n) * kDim, lane, wv);
  pdl_wait();
  for (int i = threadIdx.x; i < rows * kDim / 8; i += blockDim.x)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  if (o >= nout) return;
  float acc[8];
  gemv_row_dot(wv, xs, rows, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      if (r >= rows) break;
      float v = round_bf16 ? bfround(acc[r]) : acc[r];
      if (o < n)
        y0[(size_t)r * n + o] = v;
      else
        y1[(size_t)r * n + o - n] = v;
    }
  }
}

// gemv_rows_kernel for 9..16 rows: the rows in groups of up to 8 through the same shared memory, the weight row
// held in registers across them; every row's sum is the one gemv_rows_kernel makes.
__global__ void __launch_bounds__(256) gemv_rows16_kernel(const uint16_t* x, int rows, const uint16_t* W0,
                                                          const uint16_t* W1, int n0, int n, int nout, float* y0,
                                                          float* y1, int round_bf16) {
  extern __shared__ uint16_t xs[];  // [kDecodeGroup][kDim]
  pdl_trigger();
  int o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
  uint4 wv[kRowWords];
  if (o < nout) gemv_row_load(o < n ? W0 + (size_t)(n0 + o) * kDim : W1 + (size_t)(n0 + o - n) * kDim, lane, wv);
  pdl_wait();
  for (int r0 = 0; r0 < rows; r0 += kDecodeGroup) {
    const int nr = min(kDecodeGroup, rows - r0);
    if (r0) __syncthreads();  // the previous group is read
    for (int i = threadIdx.x; i < nr * kDim / 8; i += blockDim.x)
      reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x + (size_t)r0 * kDim)[i];
    __syncthreads();
    if (o >= nout) continue;
    float acc[8];
    gemv_row_dot(wv, xs, nr, lane, acc);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        if (r >= nr) break;
        float v = round_bf16 ? bfround(acc[r]) : acc[r];
        if (o < n)
          y0[(size_t)(r0 + r) * n + o] = v;
        else
          y1[(size_t)(r0 + r) * n + o - n] = v;
      }
    }
  }
}

}  // namespace

void compressor_project_rows(const uint16_t* x, const CompressorWeights& w, int n0, int n, float* kv, float* score,
                             int rows, bool pdl, cudaStream_t s) {
  static bool once = [] {
    for (auto k : {gemv_rows_kernel, gemv_rows16_kernel})
      CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      kDecodeGroup * kDim * (int)sizeof(uint16_t)));
    return true;
  }();
  (void)once;
  if (rows < 1 || rows > kMaxDecodeRows || n0 < 0 || n0 + n > kHead)
    die("compressor_project_rows: %d rows, [%d, +%d)", rows, n0, n);
  int nout = w.ratio > 1 ? 2 * n : n;
  launch(rows > kDecodeGroup ? gemv_rows16_kernel : gemv_rows_kernel, dim3((nout + 7) / 8), dim3(256),
         (size_t)std::min(rows, kDecodeGroup) * kDim * sizeof(uint16_t), s, pdl, x, rows, w.wkv, w.wgate, n0, n, nout,
         kv, score, (int)(w.ratio == 1));
}

void gemv_bf16(const uint16_t* x, int rows, int K, const uint16_t* W0, const uint16_t* W1, int N0, int N, float* y0,
               float* y1, bool round_bf16, cudaStream_t s) {
  static bool once = [] {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kGemvRows * kDim * (int)sizeof(uint16_t)));
    return true;
  }();
  (void)once;
  if (K > kDim || K % 8) die("gemv_bf16: K=%d", K);
  dim3 grid((N + 7) / 8, (rows + kGemvRows - 1) / kGemvRows);
  size_t smem = (size_t)(rows < kGemvRows ? rows : kGemvRows) * K * sizeof(uint16_t);
  gemv_kernel<<<grid, 256, smem, s>>>(x, rows, K, W0, W1, N0, N, y0, y1, round_bf16);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace attn
