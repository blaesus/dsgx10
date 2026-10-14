// Norms and the per-row preparation after the x-projection's all-gather: q_a_layernorm, kv_a_layernorm with
// the cache write, RoPE of k_rope, the indexer key (LayerNorm with bias, RoPE on its first 64 dims) and the
// indexer's head weights. Statistics in f32 in one fixed order; the reference's roundings (RMSNorm: f32 normed
// value rounded to bf16, then a bf16 multiply by the weight; LayerNorm: one rounding of the f32 result).
#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"

namespace glm {
namespace {

// Sum over the CTA in a fixed order: each thread's value, a warp xor-tree, then the warps in order.
template <int NT>
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int w = 0; w < NT / 32; ++w) s = __fadd_rn(s, red[w]);
  return s;
}

__device__ __forceinline__ float rms_rstd(float ss, int D, float eps) {
  const float var = __fdiv_rn(ss, (float)D);
  return __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(var, eps)));
}
// weight * bf16(x * rstd), the second rounding a bf16 multiply
__device__ __forceinline__ float rms_out(float x, float rstd, float w) {
  return bfround(__fmul_rn(w, bfround(__fmul_rn(x, rstd))));
}

// one CTA (256 threads) per row; D a multiple of 2048
__global__ void __launch_bounds__(256) k_rmsnorm(const uint16_t* __restrict__ x, int ldx, const uint16_t* __restrict__ w,
                                                 float eps, int D, uint16_t* __restrict__ y, int ldy) {
  __shared__ float red[8];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x;
  const uint16_t* xr = x + (size_t)r * ldx;
  const int per = D / 256;  // contiguous elements a thread, in uint4s of 8
  float v[48];              // up to D = 12288
  float ss = 0.f;
  for (int i = 0; i < per; i += 8) {
    const uint4 q = *reinterpret_cast<const uint4*>(xr + threadIdx.x * per + i);
    const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[i + 2 * j] = bf2f(ww[j]);
      v[i + 2 * j + 1] = bf2f_hi(ww[j]);
    }
  }
  for (int i = 0; i < per; ++i) ss = __fadd_rn(ss, __fmul_rn(v[i], v[i]));
  const float rstd = rms_rstd(block_sum<256>(ss, red), D, eps);
  uint16_t* yr = y + (size_t)r * ldy;
  for (int i = 0; i < per; i += 8) {
    const uint4 q = *reinterpret_cast<const uint4*>(w + threadIdx.x * per + i);
    const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      o[j] = pack_bf16(rms_out(v[i + 2 * j], rstd, bf2f(ww[j])), rms_out(v[i + 2 * j + 1], rstd, bf2f_hi(ww[j])));
    *reinterpret_cast<uint4*>(yr + threadIdx.x * per + i) = make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// one CTA (256 threads) per row
__global__ void __launch_bounds__(256) k_prep(PrepArgs a) {
  __shared__ float red[8];
  __shared__ float kidx[kIdxDim];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint16_t* xr = a.xg + (size_t)r * kTP * a.xrows;
  const int p = a.pos[r];
  const size_t entry = (a.slot ? (size_t)a.slot[r] * a.kv.slot_stride : 0) + p;
  const uint32_t* cs = a.rope + (size_t)p * (kRope / 2);
  // q latent: element i at rank i / 512, offset i % 512; 8 contiguous a thread
  float q[8];
  {
    const int i0 = tid * 8;
    const uint4 v = *reinterpret_cast<const uint4*>(xr + (i0 / kXQa) * a.xrows + i0 % kXQa);
    const uint32_t ww[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { q[2 * j] = bf2f(ww[j]); q[2 * j + 1] = bf2f_hi(ww[j]); }
  }
  // kv latent: element i at rank i / 128, offset 512 + i % 128; 2 a thread
  float c[2];
  {
    const int i0 = tid * 2;
    const uint32_t v = *reinterpret_cast<const uint32_t*>(xr + (i0 / kXKv) * a.xrows + kXQa + i0 % kXKv);
    c[0] = bf2f(v);
    c[1] = bf2f_hi(v);
  }
  float ssq = 0.f, ssc = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) ssq = __fadd_rn(ssq, __fmul_rn(q[j], q[j]));
  ssc = __fadd_rn(__fmul_rn(c[0], c[0]), __fmul_rn(c[1], c[1]));
  const float rq = rms_rstd(block_sum<256>(ssq, red), kQLora, kLatEps);
  const float rc = rms_rstd(block_sum<256>(ssc, red), kKvLora, kLatEps);
  {
    const uint4 wv = *reinterpret_cast<const uint4*>(a.q_norm + tid * 8);
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = pack_bf16(rms_out(q[2 * j], rq, bf2f(ww[j])), rms_out(q[2 * j + 1], rq, bf2f_hi(ww[j])));
    *reinterpret_cast<uint4*>(a.qr + (size_t)r * kQLora + tid * 8) = make_uint4(o[0], o[1], o[2], o[3]);
  }
  uint16_t* ent = a.kv.kv + entry * kLat;
  {
    const uint32_t wv = *reinterpret_cast<const uint32_t*>(a.kv_norm + tid * 2);
    *reinterpret_cast<uint32_t*>(ent + tid * 2) = pack_bf16(rms_out(c[0], rc, bf2f(wv)), rms_out(c[1], rc, bf2f_hi(wv)));
  }
  // k_rope: element i at rank i / 16, offset 640 + i % 16; warp 0, lane i takes the pair (2i, 2i+1)
  if (warp == 0) {
    const int i0 = 2 * lane;
    const uint32_t v = *reinterpret_cast<const uint32_t*>(xr + (i0 / kXRope) * a.xrows + kXQa + kXKv + i0 % kXRope);
    float lo, hi;
    rope_pair(bf2f(v), bf2f_hi(v), cs[lane], lo, hi);
    ent[kKvLora + lane] = f2bf(lo);
    ent[kKvLora + 32 + lane] = f2bf(hi);
  }
  if (!a.k_norm_w) return;
  // the index key: LayerNorm over 128 (warp 1, 4 a lane: element i at rank i / 32, offset 656 + i % 32)
  if (warp == 1) {
    float k[4];
    const int i0 = 4 * lane;
    const uint2 v = *reinterpret_cast<const uint2*>(xr + (i0 / kXWk) * a.xrows + kXRowsShared + i0 % kXWk);
    k[0] = bf2f(v.x); k[1] = bf2f_hi(v.x); k[2] = bf2f(v.y); k[3] = bf2f_hi(v.y);
    float s = __fadd_rn(__fadd_rn(__fadd_rn(k[0], k[1]), k[2]), k[3]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    const float mean = __fdiv_rn(s, (float)kIdxDim);
    float d[4], vs = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) { d[j] = __fsub_rn(k[j], mean); vs = __fadd_rn(vs, __fmul_rn(d[j], d[j])); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) vs = __fadd_rn(vs, __shfl_xor_sync(0xffffffffu, vs, o));
    const float rstd = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(__fdiv_rn(vs, (float)kIdxDim), kIdxLnEps)));
    const uint2 wv = *reinterpret_cast<const uint2*>(a.k_norm_w + i0);
    const uint2 bv = *reinterpret_cast<const uint2*>(a.k_norm_b + i0);
    const float wf[4] = {bf2f(wv.x), bf2f_hi(wv.x), bf2f(wv.y), bf2f_hi(wv.y)};
    const float bf[4] = {bf2f(bv.x), bf2f_hi(bv.x), bf2f(bv.y), bf2f_hi(bv.y)};
#pragma unroll
    for (int j = 0; j < 4; ++j) kidx[i0 + j] = bfround(__fadd_rn(__fmul_rn(__fmul_rn(d[j], rstd), wf[j]), bf[j]));
    __syncwarp();
    // RoPE on dims 0..63 (lane i: the pair 2i, 2i+1); dims 64..127 pass
    float lo, hi;
    rope_pair(kidx[2 * lane], kidx[2 * lane + 1], cs[lane], lo, hi);
    const float pass0 = kidx[64 + 2 * lane], pass1 = kidx[65 + 2 * lane];
    __syncwarp();
    kidx[lane] = lo;
    kidx[32 + lane] = hi;
    kidx[64 + 2 * lane] = pass0;
    kidx[65 + 2 * lane] = pass1;
    __syncwarp();
    const size_t ie = (a.slot ? (size_t)a.slot[r] * a.ik.slot_stride : 0) + p;
    if (!a.ik.fp8) {
      uint16_t* kd = reinterpret_cast<uint16_t*>(a.ik.k) + ie * kIdxDim;
      *reinterpret_cast<uint2*>(kd + i0) = make_uint2(pack_bf16(kidx[i0], kidx[i0 + 1]), pack_bf16(kidx[i0 + 2], kidx[i0 + 3]));
    } else {  // 128 e4m3 + an f32 scale amax / 448 (FP8 cache: [cap][128] codes, then [cap] f32 scales)
      float am = fmaxf(fmaxf(fabsf(kidx[i0]), fabsf(kidx[i0 + 1])), fmaxf(fabsf(kidx[i0 + 2]), fabsf(kidx[i0 + 3])));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
      const float sc = __fdiv_rn(fmaxf(am, 1e-20f), 448.f);
      uint32_t code = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) code |= (uint32_t)f_to_e4m3(__fdiv_rn(kidx[i0 + j], sc)) << (8 * j);
      *reinterpret_cast<uint32_t*>(a.ik.k + ie * kIdxDim + i0) = code;
      if (lane == 0) a.ik.ks[ie] = sc;
    }
  }
}

// index weights: w[r][h] = f32(x[r] . Wp[h]) * 32^-0.5; grid (heads, ceil(rows / 16)), 256 threads, each thread
// 24 contiguous k of every row, then a fixed-order CTA sum
__global__ void __launch_bounds__(256) k_index_weights(const uint16_t* __restrict__ x, int rows,
                                                       const uint16_t* __restrict__ wp, float* __restrict__ w, int ldw) {
  __shared__ float red[16][8];
  pdl_trigger();
  const int h = blockIdx.x, r0 = blockIdx.y * 16, tid = threadIdx.x;
  constexpr int kPer = kDim / 256;  // 24
  float wv[kPer];
  const uint16_t* wr = wp + (size_t)h * kDim + tid * kPer;
#pragma unroll
  for (int i = 0; i < kPer; i += 8) {
    const uint4 q = *reinterpret_cast<const uint4*>(wr + i);
    const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { wv[i + 2 * j] = bf2f(ww[j]); wv[i + 2 * j + 1] = bf2f_hi(ww[j]); }
  }
  pdl_wait();
  const int nr = min(16, rows - r0);
  for (int rr = 0; rr < nr; ++rr) {
    const uint16_t* xr = x + (size_t)(r0 + rr) * kDim + tid * kPer;
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; i += 8) {
      const uint4 q = *reinterpret_cast<const uint4*>(xr + i);
      const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        s = __fadd_rn(s, __fmul_rn(bf2f(ww[j]), wv[i + 2 * j]));
        s = __fadd_rn(s, __fmul_rn(bf2f_hi(ww[j]), wv[i + 2 * j + 1]));
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
    if ((tid & 31) == 0) red[rr][tid >> 5] = s;
  }
  __syncthreads();
  if (tid < nr) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s = __fadd_rn(s, red[tid][i]);
    w[(size_t)(r0 + tid) * ldw + h] = __fmul_rn(s, kIdxWScale);
  }
}

}  // namespace

void rmsnorm(const uint16_t* x, int ldx, const uint16_t* w, float eps, int D, uint16_t* y, int ldy, int rows,
             cudaStream_t s, bool pdl) {
  if (D % 2048 || D > 12288) die("rmsnorm: D %d", D);
  launch(k_rmsnorm, dim3(rows), dim3(256), 0, s, pdl, x, ldx, w, eps, D, y, ldy);
  CUDA_CHECK(cudaGetLastError());
}

void attn_prep(const PrepArgs& a, int rows, cudaStream_t s, bool pdl) {
  if (a.k_norm_w && a.xrows != kXRows) die("attn_prep: a full layer needs %d gathered rows", kXRows);
  if (a.k_norm_w && a.ik.fp8 && !a.ik.ks) die("attn_prep: FP8 index keys need their scales");
  launch(k_prep, dim3(rows), dim3(256), 0, s, pdl, a);
  CUDA_CHECK(cudaGetLastError());
}

void index_weights(const uint16_t* x, int rows, const uint16_t* wproj, int nh, float* w, int ldw, cudaStream_t s,
                   bool pdl) {
  launch(k_index_weights, dim3(nh, (rows + 15) / 16), dim3(256), 0, s, pdl, x, rows, wproj, w, ldw);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace glm
