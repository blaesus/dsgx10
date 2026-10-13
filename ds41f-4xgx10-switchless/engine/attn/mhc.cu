// mHC: the hyper-connection mixing around each sublayer (model.py Block.hc_mixes / hc_pre / hc_post).
#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"
#include "attn/mhc_dev.cuh"
#include "common/check.h"

namespace attn {


namespace {

constexpr int kHcChunk = 512;  // stream columns per CTA of the partial projection
constexpr int kHcRows = 8;     // rows per CTA

// partial[chunk][r][0..23] = fn[o][cols] . x[r][cols], partial[chunk][r][24] = sum x^2, over this
// CTA's 512 columns of [k0, k1). Warp w handles outputs 3w..3w+2; lane l columns l + 32j.
__global__ void __launch_bounds__(256) hc_partial_kernel(const uint16_t* x, int rows, const float* fn, int k0, int k1,
                                                         float* chunks) {
  __shared__ __align__(16) float xs[kHcRows][kHcChunk];
  int chunk = blockIdx.x, r0 = blockIdx.y * kHcRows, nr = min(kHcRows, rows - r0);
  int c0 = k0 + chunk * kHcChunk, nc = min(kHcChunk, k1 - c0);
  for (int i = threadIdx.x; i < nr * kHcChunk; i += blockDim.x) {
    int r = i / kHcChunk, c = i % kHcChunk;
    xs[r][c] = c < nc ? bf2f(x[(size_t)(r0 + r) * kHcDim + c0 + c]) : 0.f;
  }
  __syncthreads();
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int oo = 0; oo < 3; ++oo) {
    int o = warp * 3 + oo;
    float w[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      int c = lane + 32 * j;
      w[j] = c < nc ? fn[(size_t)o * kHcDim + c0 + c] : 0.f;
    }
    for (int r = 0; r < nr; ++r) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) acc = __fmaf_rn(xs[r][lane + 32 * j], w[j], acc);
      acc = warp_sum(acc);
      if (lane == 0) chunks[((size_t)chunk * rows + r0 + r) * 25 + o] = acc;
    }
  }
  if (warp == 0) {
    for (int r = 0; r < nr; ++r) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) acc = __fmaf_rn(xs[r][lane + 32 * j], xs[r][lane + 32 * j], acc);
      acc = warp_sum(acc);
      if (lane == 0) chunks[((size_t)chunk * rows + r0 + r) * 25 + 24] = acc;
    }
  }
}

__global__ void hc_sum_chunks_kernel(const float* chunks, int nchunks, int rows, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 25) return;
  float s = 0.f;
  for (int c = 0; c < nchunks; ++c) s = __fadd_rn(s, chunks[(size_t)c * rows * 25 + i]);
  out[i] = s;
}

// torch's 4-element sums (measured, tests/probe_sinkhorn.py): over the last dim and inside softmax
// (a 4-lane butterfly) (a+c)+(b+d); over dim -2 sequential ((a+b)+c)+d.
__device__ __forceinline__ float sum4_last(float a, float b, float c, float d) {
  return __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
}
__device__ __forceinline__ float sum4_seq(float a, float b, float c, float d) {
  return __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
}

// torchref hc_split_sinkhorn, op for op, one thread per row.
__global__ void hc_finish_kernel(const float* partial, int rows, const float* scale, const float* base, float* pre,
                                 float* post, float* comb) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* p = partial + (size_t)r * 25;
  float rs = rsqrtf(__fadd_rn(__fmul_rn(p[24], 1.0f / kHcDim), kNormEps));
  float m[kMix];
#pragma unroll
  for (int i = 0; i < kMix; ++i) m[i] = __fmul_rn(p[i], rs);
  float s0 = scale[0], s1 = scale[1], s2 = scale[2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    pre[r * 4 + j] = __fadd_rn(sigmoidf_ref(__fadd_rn(__fmul_rn(m[j], s0), base[j])), kHcEps);
    post[r * 4 + j] = __fmul_rn(2.0f, sigmoidf_ref(__fadd_rn(__fmul_rn(m[4 + j], s1), base[4 + j])));
  }
  float c[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int k = 0; k < 4; ++k) c[j][k] = __fadd_rn(__fmul_rn(m[8 + 4 * j + k], s2), base[8 + 4 * j + k]);
    float mx = fmaxf(fmaxf(c[j][0], c[j][1]), fmaxf(c[j][2], c[j][3]));
    float e[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) e[k] = expf(__fsub_rn(c[j][k], mx));
    float sum = sum4_last(e[0], e[1], e[2], e[3]);
#pragma unroll
    for (int k = 0; k < 4; ++k) c[j][k] = __fadd_rn(__fdiv_rn(e[k], sum), kHcEps);
  }
  auto colnorm = [&]() {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float d = __fadd_rn(sum4_seq(c[0][k], c[1][k], c[2][k], c[3][k]), kHcEps);
#pragma unroll
      for (int j = 0; j < 4; ++j) c[j][k] = __fdiv_rn(c[j][k], d);
    }
  };
  colnorm();
  for (int it = 1; it < 20; ++it) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float d = __fadd_rn(sum4_last(c[j][0], c[j][1], c[j][2], c[j][3]), kHcEps);
#pragma unroll
      for (int k = 0; k < 4; ++k) c[j][k] = __fdiv_rn(c[j][k], d);
    }
    colnorm();
  }
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) comb[r * 16 + 4 * j + k] = c[j][k];
}

// y = bf16(((p0 x0 + p1 x1) + p2 x2) + p3 x3), products rounded as torch's materialised product;
// optionally RMSNorm(5120) of y after. One CTA (256 threads, 20 columns each) per row.
__global__ void __launch_bounds__(256) hc_pre_kernel(const uint16_t* x, const float* pre, const uint16_t* norm_w,
                                                     uint16_t* y, uint16_t* y_raw) {
  __shared__ float red[8];
  pdl_trigger();
  pdl_wait();
  int r = blockIdx.x;
  const uint16_t* xr = x + (size_t)r * kHcDim;
  float p0 = pre[r * 4], p1 = pre[r * 4 + 1], p2 = pre[r * 4 + 2], p3 = pre[r * 4 + 3];
  float v[kDim / 256];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kDim / 256; ++i) {
    int d = threadIdx.x + 256 * i;
    float a = __fadd_rn(__fmul_rn(p0, bf2f(xr[d])), __fmul_rn(p1, bf2f(xr[kDim + d])));
    a = __fadd_rn(a, __fmul_rn(p2, bf2f(xr[2 * kDim + d])));
    a = __fadd_rn(a, __fmul_rn(p3, bf2f(xr[3 * kDim + d])));
    v[i] = bfround(a);
    ss = __fadd_rn(ss, __fmul_rn(v[i], v[i]));
  }
  if (y_raw) {
#pragma unroll
    for (int i = 0; i < kDim / 256; ++i) y_raw[(size_t)r * kDim + threadIdx.x + 256 * i] = f2bf(v[i]);
  }
  if (!norm_w) {
#pragma unroll
    for (int i = 0; i < kDim / 256; ++i) y[(size_t)r * kDim + threadIdx.x + 256 * i] = f2bf(v[i]);
    return;
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, red[w]);
  float rs = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / kDim), kNormEps));
#pragma unroll
  for (int i = 0; i < kDim / 256; ++i) {
    int d = threadIdx.x + 256 * i;
    y[(size_t)r * kDim + d] = f2bf(__fmul_rn(bf2f(norm_w[d]), __fmul_rn(v[i], rs)));
  }
}

// y[j] = bf16(post[j] sub + (((c0j r0 + c1j r1) + c2j r2) + c3j r3)), cij = comb[i][j]
__global__ void __launch_bounds__(256) hc_post_kernel(const uint16_t* sub, const uint16_t* res, const float* post,
                                                      const float* comb, uint16_t* y) {
  pdl_trigger();
  pdl_wait();
  int r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
  const uint16_t* rr = res + (size_t)r * kHcDim;
  float s = bf2f(sub[(size_t)r * kDim + d]);
  float x0 = bf2f(rr[d]), x1 = bf2f(rr[kDim + d]), x2 = bf2f(rr[2 * kDim + d]), x3 = bf2f(rr[3 * kDim + d]);
  const float* cb = comb + r * 16;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float a = __fadd_rn(__fmul_rn(cb[j], x0), __fmul_rn(cb[4 + j], x1));
    a = __fadd_rn(a, __fmul_rn(cb[8 + j], x2));
    a = __fadd_rn(a, __fmul_rn(cb[12 + j], x3));
    y[(size_t)r * kHcDim + j * kDim + d] = f2bf(__fadd_rn(__fmul_rn(post[r * 4 + j], s), a));
  }
}

// ---- the fused decode path: two kernels per sublayer -------------------------------------------

// hc_partial_kernel's chunks of the stream x over columns [k0, k1) (multiples of 512), without the sum:
// grid ((k1 - k0) / 512, row groups of 8); fn is loaded before the PDL wait.
__global__ void __launch_bounds__(256) hc_project_kernel(const uint16_t* x, int rows, const float* fn, int ld,
                                                         int koff, int k0, float* chunks) {
  __shared__ __align__(16) float xs[kHcRows][kHcChunk];
  pdl_trigger();
  int chunk = blockIdx.x, r0 = blockIdx.y * kHcRows, nr = min(kHcRows, rows - r0);
  int c0 = k0 + chunk * kHcChunk, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float w[3][16];  // this warp's three rows of fn over the chunk
#pragma unroll
  for (int oo = 0; oo < 3; ++oo)
#pragma unroll
    for (int j = 0; j < 16; ++j) w[oo][j] = fn[(size_t)(warp * 3 + oo) * ld + c0 - koff + lane + 32 * j];
  pdl_wait();
  for (int i = threadIdx.x; i < nr * kHcChunk / 8; i += blockDim.x) {  // 8 columns per load
    int rr = i / (kHcChunk / 8), c = i % (kHcChunk / 8) * 8;
    uint4 u = *reinterpret_cast<const uint4*>(x + (size_t)(r0 + rr) * kHcDim + c0 + c);
    const uint32_t wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      xs[rr][c + 2 * k] = __uint_as_float(wd[k] << 16);
      xs[rr][c + 2 * k + 1] = __uint_as_float(wd[k] & 0xffff0000u);
    }
  }
  __syncthreads();
  float* out = chunks + (size_t)chunk * rows * 25;
#pragma unroll
  for (int oo = 0; oo < 3; ++oo) {
    int o = warp * 3 + oo;
    for (int r = 0; r < nr; ++r) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) acc = __fmaf_rn(xs[r][lane + 32 * j], w[oo][j], acc);
      acc = warp_sum(acc);
      if (lane == 0) out[(size_t)(r0 + r) * 25 + o] = acc;
    }
  }
  if (warp == 0) {
    for (int r = 0; r < nr; ++r) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) acc = __fmaf_rn(xs[r][lane + 32 * j], xs[r][lane + 32 * j], acc);
      acc = warp_sum(acc);
      if (lane == 0) out[(size_t)(r0 + r) * 25 + 24] = acc;
    }
  }
}

// One CTA per row: the partials summed in hc_sum_chunks_kernel's order, hc_finish_kernel's Sinkhorn on 16
// lanes (hc_sinkhorn_warp), hc_pre_kernel's pre-mix and RMSNorm, then act_quant of the result. The stream
// loads are in flight while warp 0 runs the Sinkhorn.
__global__ void __launch_bounds__(256) hc_mix_kernel(const uint16_t* x, const float* chunks, int nchunks, int rows,
                                                     const float* scale, const float* base, const uint16_t* norm_w,
                                                     float* pre, float* post, float* comb, uint16_t* y, uint8_t* q,
                                                     uint8_t* qs, const float* pre_in) {
  constexpr int kPer = kDim / 256;
  __shared__ float red[8];
  __shared__ float spre[4];
  pdl_trigger();
  int r = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
  float nw[kPer];
  if (norm_w) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) nw[i] = bf2f(norm_w[t + 256 * i]);
  }
  float sc = 0.f, bs = 0.f;  // lane i of warp 0 works on mix i: pre (0-3), post (4-7), comb (8-23)
  if (warp == 0) {
    sc = hc_lane_scale(scale, lane);
    bs = hc_lane_base(base, lane);
  }
  pdl_wait();
  float mv = 0.f;  // warp 0: the partials, summed in order (loads 8 at a time, ahead of the stream's)
  if (warp == 0 && lane < 25) {
    for (int c0 = 0; c0 < nchunks; c0 += 8) {
      float pc[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) pc[k] = c0 + k < nchunks ? chunks[((size_t)(c0 + k) * rows + r) * 25 + lane] : 0.f;
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if (c0 + k < nchunks) mv = __fadd_rn(mv, pc[k]);
    }
  }
  const uint16_t* xr = x + (size_t)r * kHcDim;
  uint16_t xv[kHc][kPer];
#pragma unroll
  for (int c = 0; c < kHc; ++c)
#pragma unroll
    for (int i = 0; i < kPer; ++i) xv[c][i] = xr[c * kDim + t + 256 * i];
  if (warp == 0) {
    const float m = hc_sinkhorn_warp(mv, lane, sc, bs);
    if (lane < 4) {
      float pin = pre_in ? pre_in[r * 4 + lane] : m;  // read before pre is written: they may alias
      pre[r * 4 + lane] = m;
      spre[lane] = pin;
    } else if (lane < 8) {
      post[r * 4 + lane - 4] = m;
    } else if (lane < 24) {
      comb[r * 16 + lane - 8] = m;
    }
  }
  __syncthreads();
  float p0 = spre[0], p1 = spre[1], p2 = spre[2], p3 = spre[3];
  float v[kPer];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    float a = __fadd_rn(__fmul_rn(p0, bf2f(xv[0][i])), __fmul_rn(p1, bf2f(xv[1][i])));
    a = __fadd_rn(a, __fmul_rn(p2, bf2f(xv[2][i])));
    a = __fadd_rn(a, __fmul_rn(p3, bf2f(xv[3][i])));
    v[i] = bfround(a);
    ss = __fadd_rn(ss, __fmul_rn(v[i], v[i]));
  }
  if (norm_w) {
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, red[w]);
    float rs = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / kDim), kNormEps));
#pragma unroll
    for (int i = 0; i < kPer; ++i) v[i] = bfround(__fmul_rn(nw[i], __fmul_rn(v[i], rs)));
  }
#pragma unroll
  for (int i = 0; i < kPer; ++i) y[(size_t)r * kDim + t + 256 * i] = f2bf(v[i]);
  if (q) {  // act_quant: warp w holds the 32 values of block 8i + w; warp_max of every block at once
    float am[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) am[i] = fabsf(v[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int i = 0; i < kPer; ++i) am[i] = fmaxf(am[i], __shfl_xor_sync(0xffffffffu, am[i], o));
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      float s = fp8_block_scale(am[i]);
      q[(size_t)r * kDim + t + 256 * i] = fp8_code(v[i], s);
      if (lane == 0) qs[(size_t)r * (kDim / 32) + 8 * i + warp] = f_to_ue8m0(s);
    }
  }
}

}  // namespace

int hc_chunks(int k0, int k1) { return (k1 - k0 + kHcChunk - 1) / kHcChunk; }

void hc_partial(const uint16_t* x, int rows, const float* fn, int k0, int k1, float* chunks, float* out,
                cudaStream_t s) {
  int nch = hc_chunks(k0, k1);
  hc_partial_kernel<<<dim3(nch, (rows + kHcRows - 1) / kHcRows), 256, 0, s>>>(x, rows, fn, k0, k1, chunks);
  CUDA_CHECK(cudaGetLastError());
  hc_sum_chunks_kernel<<<(rows * 25 + 127) / 128, 128, 0, s>>>(chunks, nch, rows, out);
  CUDA_CHECK(cudaGetLastError());
}

void hc_finish(const float* partial, int rows, const float* scale, const float* base, float* pre, float* post,
               float* comb, cudaStream_t s) {
  hc_finish_kernel<<<(rows + 63) / 64, 64, 0, s>>>(partial, rows, scale, base, pre, post, comb);
  CUDA_CHECK(cudaGetLastError());
}

void hc_pre(const uint16_t* x, const float* pre, int rows, const uint16_t* norm_w, uint16_t* y, uint16_t* y_raw,
            cudaStream_t s, bool pdl) {
  launch(hc_pre_kernel, dim3(rows), dim3(256), 0, s, pdl, x, pre, norm_w, y, y_raw);
}

void hc_post(const uint16_t* sub, const uint16_t* res, const float* post, const float* comb, int rows, uint16_t* y,
             cudaStream_t s, bool pdl) {
  launch(hc_post_kernel, dim3(kDim / 256, rows), dim3(256), 0, s, pdl, sub, res, post, comb, y);
}

void hc_project(const uint16_t* x, int rows, const float* fn, int k0, int k1, float* chunks, bool pdl,
                cudaStream_t s, int fn_ld) {
  if (k0 % kHcChunk || k1 % kHcChunk || k0 < 0 || k1 > kHcDim || k0 >= k1 || (fn_ld && fn_ld < k1 - k0))
    die("hc_project: [%d, %d), fn_ld %d", k0, k1, fn_ld);
  launch(hc_project_kernel, dim3((k1 - k0) / kHcChunk, (rows + kHcRows - 1) / kHcRows), dim3(256), 0, s, pdl, x,
         rows, fn, fn_ld ? fn_ld : kHcDim, fn_ld ? k0 : 0, k0, chunks);
}

void hc_mix(const uint16_t* x, const float* chunks, int nchunks, int rows, const float* scale, const float* base,
            const uint16_t* norm_w, float* pre, float* post, float* comb, uint16_t* y, uint8_t* q, uint8_t* qs,
            bool pdl, cudaStream_t s, const float* pre_in) {
  launch(hc_mix_kernel, dim3(rows), dim3(256), 0, s, pdl, x, chunks, nchunks, rows, scale, base, norm_w, pre, post,
         comb, y, q, qs, pre_in);
}

}  // namespace attn
