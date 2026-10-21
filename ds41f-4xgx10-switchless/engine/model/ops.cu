#include "model/ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "common/check.h"
#include "model/config.h"

namespace model::ops {
namespace {

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }

__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

// Block-wide sum of one float (blockDim.x a multiple of 32, <= 1024); every thread gets the result.
__device__ float block_sum(float v, float* red) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x / 32, nw = blockDim.x / 32;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[w] = v;
  __syncthreads();
  float s = 0.f;
  for (int i = 0; i < nw; ++i) s += red[i];
  return s;
}

__global__ void k_vision_rows(const int* pos, const VisionSpan* spans, int count,
                               const bf16* features, bf16* x, int* mask) {
  const int r = blockIdx.x, position = pos[r];
  int source = -1;
  for (int k = 0; k < count; ++k)
    if (position >= spans[k].start && position < spans[k].start + spans[k].rows)
      source = spans[k].offset + position - spans[k].start;
  if (threadIdx.x == 0) mask[r] = source >= 0;
  if (source < 0 || !x) return;
  const uint4* src = reinterpret_cast<const uint4*>(features + (size_t)source * kDim);
  uint4* dst = reinterpret_cast<uint4*>(x + (size_t)r * kHcDim);
  for (int i = threadIdx.x; i < kDim / 8; i += blockDim.x) {
    const uint4 v = src[i];
    for (int c = 0; c < kHc; ++c) dst[c * (kDim / 8) + i] = v;
  }
}

__global__ void k_embed(const bf16* table, const int* tokens, bf16* x, float* pre_mix) {
  const int r = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)tokens[r] * kDim);
  uint4* dst = reinterpret_cast<uint4*>(x + (size_t)r * kHcDim);
  for (int i = threadIdx.x; i < kDim / 8; i += blockDim.x) {
    const uint4 v = src[i];
#pragma unroll
    for (int c = 0; c < kHc; ++c) dst[c * (kDim / 8) + i] = v;
  }
  if (threadIdx.x < kHc) pre_mix[r * kHc + threadIdx.x] = threadIdx.x == 0 ? 1.f : 0.f;
}

__global__ void k_unpack_attn(const uint8_t* gath, int R, int ncomp, size_t per_rank, bf16* qa, bf16* kv, float* ckv,
                              float* csc) {
  const int p = blockIdx.x;
  for (int r = 0; r < kRanks; ++r) {
    const uint8_t* b = gath + per_rank * r;
    const bf16* bqa = reinterpret_cast<const bf16*>(b) + (size_t)p * kQaPart;
    const bf16* bkv = reinterpret_cast<const bf16*>(b + (size_t)R * kQaPart * 2) + (size_t)p * kKvPart;
    for (int j = threadIdx.x; j < kQaPart; j += blockDim.x) qa[(size_t)p * kQLora + kQaPart * r + j] = bqa[j];
    for (int j = threadIdx.x; j < kKvPart; j += blockDim.x) kv[(size_t)p * kHead + kKvPart * r + j] = bkv[j];
    if (ncomp > 0) {
      const float* bck = reinterpret_cast<const float*>(b + (size_t)R * (kQaPart + kKvPart) * 2) + (size_t)p * kCompPart;
      for (int j = threadIdx.x; j < kCompPart; j += blockDim.x) ckv[(size_t)p * kHead + kCompPart * r + j] = bck[j];
    }
    if (ncomp > 1) {
      const float* bcs =
          reinterpret_cast<const float*>(b + (size_t)R * (kQaPart + kKvPart) * 2 + (size_t)R * kCompPart * 4) +
          (size_t)p * kCompPart;
      for (int j = threadIdx.x; j < kCompPart; j += blockDim.x) csc[(size_t)p * kHead + kCompPart * r + j] = bcs[j];
    }
  }
}

__global__ void k_unpack_gate(const float* gath, int R, float* logits) {
  const int p = blockIdx.x;
  for (int i = threadIdx.x; i < kExperts; i += blockDim.x) {
    const int r = i / kGatePart, j = i % kGatePart;
    logits[(size_t)p * kExperts + i] = gath[((size_t)r * R + p) * kGatePart + j];
  }
}

__global__ void k_f32_to_bf16(const float* in, bf16* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = __float2bfloat16_rn(in[i]);
}

__global__ void k_round_bf16(float* x, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    x[i] = __bfloat162float(__float2bfloat16_rn(x[i]));
}

__global__ void k_iota(int* pos, int p0, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) pos[i] = p0 + i;
}

__global__ void k_mul_bf16(const bf16* a, const bf16* b, float* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = __fmul_rn(bf(a[i]), bf(b[i]));
}

// One block per row: 24 records of 264 bytes (8-byte aligned) -> 256 values + 8 scales each.
__global__ void k_engram_pack(const uint8_t* rows, const uint32_t* off, uint8_t* xq, uint8_t* xs) {
  const int p = blockIdx.x;
  for (int i = threadIdx.x; i < kEngramCols * 33; i += blockDim.x) {
    const int c = i / 33, w = i % 33;
    const uint2 v = reinterpret_cast<const uint2*>(rows + off[p * kEngramCols + c])[w];
    if (w < 32)
      reinterpret_cast<uint2*>(xq + (size_t)p * kEngramK + c * 256)[w] = v;
    else
      reinterpret_cast<uint2*>(xs + (size_t)p * (kEngramK / 32) + c * 8)[0] = v;
  }
}

// grid (R, 5): y < 4 -> copy y's partial dot and key sum of squares; y == 4 -> the value slice.
__global__ void k_engram_partial(const bf16* x, const bf16* kv, const float* w, int rank, uint8_t* out) {
  __shared__ float red[32];
  const int p = blockIdx.x, c = blockIdx.y;
  const bf16* kvp = kv + (size_t)p * kEngramN;
  uint8_t* o = out + (size_t)p * kEngramPartBytes;
  if (c == 4) {
    bf16* v = reinterpret_cast<bf16*>(o + 32);
    for (int j = threadIdx.x; j < kEngramSeg; j += blockDim.x) v[j] = kvp[4 * kEngramSeg + j];
    return;
  }
  const bf16* h = x + ((size_t)p * kHc + c) * kDim + kEngramSeg * rank;
  const float* wc = w + (size_t)c * kDim + kEngramSeg * rank;
  float dot = 0.f, sq = 0.f;
  for (int j = threadIdx.x; j < kEngramSeg; j += blockDim.x) {
    const float k = bf(kvp[c * kEngramSeg + j]);
    dot = fmaf(__fmul_rn(bf(h[j]), wc[j]), k, dot);
    sq = fmaf(k, k, sq);
  }
  dot = block_sum(dot, red);
  sq = block_sum(sq, red);
  if (threadIdx.x == 0) {
    reinterpret_cast<float*>(o)[c] = dot;
    reinterpret_cast<float*>(o)[4 + c] = sq;
  }
}

// grid (R, 4): one copy of one row.
__global__ void k_engram_combine(const bf16* x, const uint8_t* gath, int R, bf16* y, const int* image_mask) {
  __shared__ float red[32];
  const int p = blockIdx.x, c = blockIdx.y;
  const bf16* h = x + ((size_t)p * kHc + c) * kDim;
  if (image_mask && image_mask[p]) {
    bf16* yo = y + ((size_t)p * kHc + c) * kDim;
    if (yo != h) for (int j = threadIdx.x; j < kDim; j += blockDim.x) yo[j] = h[j];
    return;  // image spans take no Engram contribution, including their delimiters
  }
  float hs = 0.f;
  for (int j = threadIdx.x; j < kDim; j += blockDim.x) {
    const float v = bf(h[j]);
    hs = fmaf(v, v, hs);
  }
  hs = block_sum(hs, red);
  float dot = 0.f, ks = 0.f;
  for (int r = 0; r < kRanks; ++r) {  // rank order: identical on every rank
    const float* f = reinterpret_cast<const float*>(gath + ((size_t)r * R + p) * kEngramPartBytes);
    dot = r ? __fadd_rn(dot, f[c]) : f[c];
    ks = r ? __fadd_rn(ks, f[4 + c]) : f[4 + c];
  }
  const float rstd = __fmul_rn(rsqrtf(__fadd_rn(__fdiv_rn(hs, (float)kDim), kNormEps)),
                               rsqrtf(__fadd_rn(__fdiv_rn(ks, (float)kDim), kNormEps)));
  const float d = __fmul_rn(__fmul_rn(dot, rstd), 0.013975424859373685f);  // dim ** -0.5
  const float g = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d)));
  bf16* yo = y + ((size_t)p * kHc + c) * kDim;
  __syncthreads();  // every h read above happened before y (which may alias x) is written
  for (int j = threadIdx.x; j < kDim; j += blockDim.x) {
    const int r = j / kEngramSeg;
    const bf16* v = reinterpret_cast<const bf16*>(gath + ((size_t)r * R + p) * kEngramPartBytes + 32);
    yo[j] = __float2bfloat16_rn(__fadd_rn(bf(h[j]), __fmul_rn(g, bf(v[j % kEngramSeg]))));
  }
}

__global__ void k_step_in(const StepIn* ring, uint32_t* counter, const int32_t* fb, StepIn* d) {
  constexpr int n = sizeof(StepIn) / 4;
  static_assert(sizeof(StepIn) % 4 == 0 && offsetof(StepIn, tokens) == 0, "StepIn");
  const uint32_t slot = *counter % kInSlots;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    uint32_t v = reinterpret_cast<const volatile uint32_t*>(ring + slot)[i];
    if (i < kMaxRows && (int32_t)v < 0) v = (uint32_t)fb[-1 - (int32_t)v];
    reinterpret_cast<uint32_t*>(d)[i] = v;
  }
  __syncthreads();
  if (threadIdx.x == 0) *counter += 1;
}

__global__ void k_wait_seq(const uint64_t* flag, const uint64_t* seq) {
  const uint64_t want = *seq;
  while (ld_acquire_sys(flag) != want) __nanosleep(100);
}

}  // namespace

void step_in(const StepIn* ring, uint32_t* counter, const int32_t* fb, StepIn* dev, cudaStream_t s) {
  k_step_in<<<1, 128, 0, s>>>(ring, counter, fb, dev);
  CUDA_CHECK(cudaGetLastError());
}

void vision_rows(const int* pos, int R, const VisionSpan* spans, int count,
                 const bf16* features, bf16* x, int* mask, cudaStream_t s) {
  if (R > 0) k_vision_rows<<<R, 256, 0, s>>>(pos, spans, count, features, x, mask);
}

void embed(const bf16* table, const int* tokens, int R, bf16* x, float* pre_mix, cudaStream_t s) {
  k_embed<<<R, 256, 0, s>>>(table, tokens, x, pre_mix);
  CUDA_CHECK(cudaGetLastError());
}

size_t attn_pack_bytes(int R, int ncomp) { return (size_t)R * ((kQaPart + kKvPart) * 2 + (size_t)ncomp * kCompPart * 4); }

void unpack_attn(const uint8_t* gath, int R, int ncomp, bf16* qa, bf16* kv, float* ckv, float* csc, cudaStream_t s) {
  k_unpack_attn<<<R, 128, 0, s>>>(gath, R, ncomp, attn_pack_bytes(R, ncomp), qa, kv, ckv, csc);
  CUDA_CHECK(cudaGetLastError());
}

void unpack_gate(const float* gath, int R, float* logits, cudaStream_t s) {
  k_unpack_gate<<<R, 128, 0, s>>>(gath, R, logits);
  CUDA_CHECK(cudaGetLastError());
}

void f32_to_bf16(const float* in, bf16* out, int n, cudaStream_t s) {
  k_f32_to_bf16<<<(n + 1023) / 1024, 256, 0, s>>>(in, out, n);
  CUDA_CHECK(cudaGetLastError());
}

void round_bf16(float* x, size_t n, cudaStream_t s) {
  k_round_bf16<<<(unsigned)std::min<size_t>((n + 1023) / 1024, 4096), 256, 0, s>>>(x, n);
  CUDA_CHECK(cudaGetLastError());
}

void iota(int* pos, int p0, int n, cudaStream_t s) {
  k_iota<<<(n + 255) / 256, 256, 0, s>>>(pos, p0, n);
  CUDA_CHECK(cudaGetLastError());
}

void mul_bf16(const bf16* a, const bf16* b, float* out, int n, cudaStream_t s) {
  k_mul_bf16<<<(n + 1023) / 1024, 256, 0, s>>>(a, b, out, n);
  CUDA_CHECK(cudaGetLastError());
}

void engram_pack(const uint8_t* rows, const uint32_t* off, int R, uint8_t* xq, uint8_t* xs, cudaStream_t s) {
  k_engram_pack<<<R, 256, 0, s>>>(rows, off, xq, xs);
  CUDA_CHECK(cudaGetLastError());
}

void engram_partial(const bf16* x, const bf16* kv, const float* w, int rank, int R, uint8_t* out, cudaStream_t s) {
  k_engram_partial<<<dim3(R, 5), 256, 0, s>>>(x, kv, w, rank, out);
  CUDA_CHECK(cudaGetLastError());
}

void engram_combine(const bf16* x, const uint8_t* gath, int R, bf16* y, cudaStream_t s, const int* image_mask) {
  k_engram_combine<<<dim3(R, kHc), 256, 0, s>>>(x, gath, R, y, image_mask);
  CUDA_CHECK(cudaGetLastError());
}

void wait_seq(const uint64_t* flag, const uint64_t* seq, cudaStream_t s) {
  k_wait_seq<<<1, 1, 0, s>>>(flag, seq);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace model::ops
