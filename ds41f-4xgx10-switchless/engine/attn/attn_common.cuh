// Device helpers shared by the attention-side kernels: number formats, the reference's quantisers,
// reductions. Everything here mirrors ref/inference/kernel.py (and our torch stand-ins) bit for bit.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "attn/attn.h"
#include "common/check.h"

namespace attn {

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfround(float f) { return bf2f(f2bf(f)); }

// 2^e for an ue8m0 byte (e in 1..254 in practice).
__device__ __forceinline__ float ue8m0_to_f(uint8_t e) {
  return e ? __uint_as_float((uint32_t)e << 23) : __uint_as_float(0x00400000u);
}
// The ue8m0 byte of a power of two.
__device__ __forceinline__ uint8_t f_to_ue8m0(float p2) { return (uint8_t)(__float_as_uint(p2) >> 23); }

// kernel.py fast_round_scale: 2^ceil(log2(amax * inv)) via the exponent bits.
__device__ __forceinline__ float fast_round_scale(float amax, float inv) {
  uint32_t b = __float_as_uint(__fmul_rn(amax, inv));
  int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0);
  return __uint_as_float((uint32_t)(e + 127) << 23);
}

// e4m3 (fn): saturating round-to-nearest-even, and exact decode.
__device__ __forceinline__ uint8_t f_to_e4m3(float f) {
  return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float e4m3_to_f(uint8_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
  return __half2float(__half(h));
}

// e2m1: decode table and round-to-nearest-even encode of a value already clamped to [-6, 6].
__device__ __forceinline__ float e2m1_to_f(uint32_t nib) {
  uint32_t c = nib & 7;
  float v = c < 2 ? (float)c * 0.5f : __uint_as_float(((c >> 1) + 126) << 23) * (1.0f + (float)(c & 1) * 0.5f);
  return (nib & 8) ? -v : v;
}
__device__ __forceinline__ uint32_t f_to_e2m1(float x) {
  float a = fabsf(x);
  uint32_t c = 0;
  c = a > 0.25f ? 1u : c;
  c = a >= 0.75f ? 2u : c;
  c = a > 1.25f ? 3u : c;
  c = a >= 1.75f ? 4u : c;
  c = a > 2.5f ? 5u : c;
  c = a >= 3.5f ? 6u : c;
  c = a > 5.0f ? 7u : c;
  return c | (__float_as_uint(x) >> 31 << 3);
}

// --- the reference quantisers, per block; `v` holds this lane's share of a block --------------

// act_quant(inplace) scale for an fp8 block: amax clamped at 1e-4, power-of-two scale.
__device__ __forceinline__ float fp8_block_scale(float amax) {
  return fast_round_scale(fmaxf(amax, 1e-4f), 1.0f / 448.0f);
}
// fp4_act_quant scale with an e8m0 scale (indexer, blocks of 32).
__device__ __forceinline__ float fp4_e8m0_scale(float amax) {
  return fast_round_scale(fmaxf(amax, 6.0f * 0x1p-126f), 1.0f / 6.0f);
}
// fp4_act_quant scale with an e4m3 scale (compressed KV, blocks of 16): e4m3(amax / 6). The golden
// data's torch stand-in divides by the scalar as amax * fp32(1/6) (torch's scalar division); the
// tilelang kernel divides exactly. They differ only when amax/6 sits on an e4m3 rounding boundary.
__device__ __forceinline__ float fp4_e4m3_scale(float amax) {
  return e4m3_to_f(f_to_e4m3(__fmul_rn(fmaxf(amax, 6.0f * 0x1p-9f), 1.0f / 6.0f)));
}
__device__ __forceinline__ uint32_t fp4_code(float x, float s) {
  return f_to_e2m1(fminf(fmaxf(__fdiv_rn(x, s), -6.0f), 6.0f));
}
// 1/s for a power of two s in [2^-126, 2^126] or infinite (fp4_e8m0_scale's), exactly; fp4_code(x, s) is then
// fp4_code_inv(x, pow2_inv(s)), bit for bit: x / s and x * (1/s) round the same real.
__device__ __forceinline__ float pow2_inv(float s) {
  return __uint_as_float(s < INFINITY ? (254u << 23) - __float_as_uint(s) : 0u);
}
__device__ __forceinline__ uint32_t fp4_code_inv(float x, float inv) {
  return f_to_e2m1(fminf(fmaxf(__fmul_rn(x, inv), -6.0f), 6.0f));
}
// s: a power of two (fp8_block_scale's), so x / s is x times the exact 1/s, bit for bit (finite x)
__device__ __forceinline__ uint8_t fp8_code(float x, float s) {
  float inv = __uint_as_float((254u << 23) - __float_as_uint(s));
  return f_to_e4m3(fminf(fmaxf(__fmul_rn(x, inv), -448.0f), 448.0f));
}

// fp4 vectors of 128 (index keys and queries) keep their 16 words transposed 4x4: stored word 4t+j holds
// elements 8(4j+t)..8(4j+t)+7, so one 16-byte load at byte 16t gives MMA lane t its four operand words.
__device__ __forceinline__ int fp4x128_byte(int cb) {
  int cw = cb >> 2;
  return (((cw & 3) << 2) | (cw >> 2)) * 4 + (cb & 3);
}

// --- reductions --------------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// RoPE on one interleaved pair (x0, x1) with (c, s): the reference multiplies complex64 values, and
// torch's compiled complex multiply contracts to re = fma(a, c, -(b*d)), im = fma(b, c, a*d). Of four
// contractions this is the one that reproduces model.py bit for bit (tests/rope check on 9.8M values).
__device__ __forceinline__ void rope_pair(float& x0, float& x1, float c, float s, bool inverse) {
  if (inverse) s = -s;
  float r = __fmaf_rn(x0, c, -__fmul_rn(x1, s));
  float i = __fmaf_rn(x1, c, __fmul_rn(x0, s));
  x0 = r;
  x1 = i;
}

// --- programmatic dependent launch (the fused decode path) -------------------------------------
// trigger: the next kernel in the stream may start, its prologue overlapping this kernel; wait: block until
// the previous kernel has completed and its writes are visible. Every kernel of a chain waits before it
// completes, so completion stays transitive. Both are no-ops in a kernel launched without the attribute.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// a hint: the 128-byte line at p into L2 (e.g. weights while a kernel waits for its predecessor)
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// --- session slots (attn.h): row r's state, and the rows of its session ------------------------
__device__ __forceinline__ Window win_at(Window w, int r) {
  if (w.slot) {
    const size_t s = (size_t)w.slot[r];
    w.kv += s * w.kv_stride;
    w.sc += s * w.sc_stride;
    w.slot = nullptr;
  }
  return w;
}
__device__ __forceinline__ Compressed ckv_at(Compressed c, int r) {
  if (c.slot) {
    const size_t s = (size_t)c.slot[r];
    c.kv += s * c.kv_stride;
    c.sc += s * c.sc_stride;
    c.slot = nullptr;
  }
  return c;
}
__device__ __forceinline__ IndexKeys ik_at(IndexKeys k, int r) {
  if (k.slot) {
    const size_t s = (size_t)k.slot[r];
    k.k += s * k.k_stride;
    k.sc += s * k.sc_stride;
    k.slot = nullptr;
  }
  return k;
}
__device__ __forceinline__ CompressorCarry carry_at(CompressorCarry c, int r) {
  if (c.slot) {
    const size_t s = (size_t)c.slot[r];
    c.kv += s * c.stride;
    c.score += s * c.stride;
    c.slot = nullptr;
  }
  return c;
}
// rows a and b of one step are in one session (a null table: one session)
__device__ __forceinline__ bool same_session(const int* slot, int a, int b) { return !slot || slot[a] == slot[b]; }
// the first and the last row of row r's session among the step's `rows`
__device__ __forceinline__ int session_first(const int* slot, int r) {
  if (!slot) return 0;
  while (r > 0 && slot[r - 1] == slot[r]) --r;
  return r;
}
__device__ __forceinline__ int session_last(const int* slot, int rows, int r) {
  if (!slot) return rows - 1;
  while (r + 1 < rows && slot[r + 1] == slot[r]) ++r;
  return r;
}
// Host: the strides of a state struct with a slot table keep every slot as aligned as slot 0 (attn.h). The
// entry points check what they take.
inline void check_slots(const Window& w, const char* who) {
  if (w.slot && (w.kv_stride % 16 || w.sc_stride % 16))
    die("%s: window slot strides %zu/%zu (bytes, multiples of 16)", who, w.kv_stride, w.sc_stride);
}
inline void check_slots(const Compressed& c, const char* who) {
  if (c.slot && (c.kv_stride % 16 || c.sc_stride % 16))
    die("%s: compressed slot strides %zu/%zu (bytes, multiples of 16)", who, c.kv_stride, c.sc_stride);
}
inline void check_slots(const CompressorCarry& c, const char* who) {
  if (c.slot && c.stride % 4) die("%s: carry slot stride %zu (floats, a multiple of 4)", who, c.stride);
}
inline void check_slots(const IndexKeys& k, const char* who) {  // whole entries: the indexer adds entry offsets
  if (k.slot && (k.k_stride % 64 || k.k_stride != 16 * k.sc_stride || k.k_stride / 64 > (1u << 24)))
    die("%s: index key slot strides %zu/%zu are not whole entries (k_stride = 16 * sc_stride)", who, k.k_stride,
        k.sc_stride);
}

}  // namespace attn
