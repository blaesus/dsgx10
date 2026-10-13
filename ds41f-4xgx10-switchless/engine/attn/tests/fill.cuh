// Deterministic random device fills for the synthetic tests and benches (a hash of index and seed).
#pragma once
#include <cstdint>

#include "testutil.h"

namespace attn_test {
namespace fill {

__device__ __forceinline__ uint32_t hash32(uint64_t i, uint32_t seed) {
  uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t)seed * 0xD1B54A32D192ED03ull;
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  return (uint32_t)(x >> 16);
}
__global__ void u8_kernel(uint8_t* p, size_t n, uint32_t seed, uint32_t mask, uint32_t add) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (uint8_t)((hash32(i, seed) & mask) + add);
}
// uniform in [lo, lo + span), rounded to bf16
__global__ void bf16_kernel(uint16_t* p, size_t n, uint32_t seed, float lo, float span) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float v = lo + (hash32(i, seed) & 0xFFFFFF) * (1.0f / 16777216.0f) * span;
    uint32_t b = __float_as_uint(v);
    p[i] = (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
  }
}
__global__ void f32_kernel(float* p, size_t n, uint32_t seed, float lo, float span) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = lo + (hash32(i, seed) & 0xFFFFFF) * (1.0f / 16777216.0f) * span;
}

inline uint32_t g_seed = 1;
// bytes (hash & mask) + add
inline void u8(Dev<uint8_t>& d, uint32_t mask, uint32_t add) { u8_kernel<<<1024, 256>>>(d.p, d.n, g_seed++, mask, add); }
// symmetric: uniform in [-scale, scale)
inline void bf16(Dev<uint16_t>& d, float scale) { bf16_kernel<<<1024, 256>>>(d.p, d.n, g_seed++, -scale, 2 * scale); }
inline void f32(Dev<float>& d, float scale) { f32_kernel<<<1024, 256>>>(d.p, d.n, g_seed++, -scale, 2 * scale); }
inline void bf16_range(Dev<uint16_t>& d, float lo, float hi) { bf16_kernel<<<1024, 256>>>(d.p, d.n, g_seed++, lo, hi - lo); }
inline void f32_range(Dev<float>& d, float lo, float hi) { f32_kernel<<<1024, 256>>>(d.p, d.n, g_seed++, lo, hi - lo); }

}  // namespace fill
}  // namespace attn_test
