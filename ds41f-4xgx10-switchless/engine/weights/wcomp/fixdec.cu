// fixdec.cu: the fixed-length decoders' instruction counts (docs/engine.md, "Lossless weight compression",
// Part 3; SASS only, never run). Each kernel walks P warp-steps with one iteration per step, decodes what a
// GEMV lane would feed its MMA, and folds it into an accumulator so nothing is dead. Compare each loop's
// instruction count with its raw twin's:
//   nvcc -std=c++20 -O3 -gencode arch=compute_121a,code=sm_121a -cubin fixdec.cu -o fixdec.cubin
//   cuobjdump -sass fixdec.cubin > fixdec.sass && python3 sass_loops.py fixdec.sass
// CUDA 13.0: scales raw 9, 1 bit 19; head raw 8, 4 bits from a tile base 37, 5 bits 52; FP8 raw 8, 3 bits 93.
#include <cstdint>

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

// ---- expert scales (MoE: one 32-byte group per warp-step, lane g = lane >> 2 takes word g) ----
// raw: today's load
__global__ void k_scale_raw(const uint8_t* __restrict__ s, int P, uint32_t* out) {
  const int g = threadIdx.x >> 2;
  uint32_t acc = 0;
#pragma unroll 1
  for (int p = 0; p < P; ++p) acc ^= __ldg(reinterpret_cast<const uint32_t*>(s + p * 32 + g * 4));
  out[threadIdx.x] = acc;
}
// 1 bit a scale: bit 8j + g of a step's code word is byte j of word g minus 120 (base even, so OR adds);
// a set flag bit sends the step to the raw group (warp-uniform)
__global__ void k_scale_1bit(const uint32_t* __restrict__ code, const uint32_t* __restrict__ flags,
                             const uint8_t* __restrict__ s, int P, uint32_t* out) {
  const int g = threadIdx.x >> 2;
  uint32_t acc = 0;
  const uint32_t fl = __ldg(flags);
#pragma unroll 1
  for (int p = 0; p < P; ++p) {
    const bool raw = (fl >> (p & 31)) & 1u;
    const uint32_t* a = raw ? reinterpret_cast<const uint32_t*>(s + p * 32 + g * 4) : code + p;
    const uint32_t w = __ldg(a);
    acc ^= raw ? w : (((w >> g) & 0x01010101u) | 0x78787878u);
  }
  out[threadIdx.x] = acc;
}

// ---- bf16 head (16 bytes = 8 bf16 a lane a step) ----
__global__ void k_head_raw(const uint4* __restrict__ w, int P, uint32_t* out) {
  uint32_t acc = 0;
#pragma unroll 1
  for (int p = 0; p < P; ++p) {
    const uint4 v = __ldg(w + p * 32 + threadIdx.x);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  out[threadIdx.x] = acc;
}
// planes: sm (sign, 7 mantissa bits: 8 bytes a lane), e4 (4 low exponent bits; nibble b = weight 2b,
// nibble 4 + b = weight 2b + 1), e5 (the 5th bit: bit b, bit 4 + b); exponent = 98 + 5-bit offset
__device__ __forceinline__ uint32_t head_word(uint32_t src, uint32_t sel, uint32_t e, uint32_t base2) {
  const uint32_t t = prmt(src, 0, sel);  // bytes: sm0, sign(sm0), sm1, sign(sm1)
  return (t & 0x807F807Fu) | (e * 128u + base2);
}
template <bool FIVE>
__global__ void k_head_dec(const uint2* __restrict__ sm, const uint32_t* __restrict__ e4,
                           const uint8_t* __restrict__ e5, const uint8_t* __restrict__ tb, int P, uint32_t* out) {
  uint32_t acc = 0;
#pragma unroll 1
  for (int p = 0; p < P; ++p) {
    const uint2 s = __ldg(sm + p * 32 + threadIdx.x);
    const uint32_t n = __ldg(e4 + p * 32 + threadIdx.x);
    uint32_t h = 0, base2;
    if constexpr (FIVE) {
      h = __ldg(e5 + p * 32 + threadIdx.x);
      base2 = (98u << 7) | (98u << 23);
    } else {
      const uint32_t b = __ldg(tb + p);  // the tile's base (4-bit index tiles)
      base2 = (b << 7) | (b << 23);
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      uint32_t e = (n >> (4 * b)) & 0x000F000Fu;
      if constexpr (FIVE) e |= (((h >> b) & 0x11u) * 0x10010u) & 0x00100010u;
      acc ^= head_word(b < 2 ? s.x : s.y, (b & 1) ? 0xB3A2u : 0x9180u, e, base2);
    }
  }
  out[threadIdx.x] = acc;
}

// ---- FP8 e4m3 (16 bytes = 16 values a lane a step) ----
// planes: nib (sign, 3 mantissa bits: 8 bytes, nibbles ordered so & 0x0F0F0F0F yields a word's 4),
// c (3-bit exponent codes, 12 bits a word packed: 48 bits a lane as u32 + u16), flags (bit per lane:
// its 16 values include one outside the tensor's 8-exponent window), x (flagged lanes' 4th bit, u16,
// compacted by popc); exponent = base + code, or the code and the 4th bit for a flagged lane
__device__ __forceinline__ uint32_t spread3(uint32_t c12) {  // 4 codes of 3 bits -> bits 3..5 of each byte
  const uint32_t ev = ((c12 & 0x1C7u) * ((1u << 3) | (1u << 13))) & 0x00380038u;
  const uint32_t od = ((c12 & 0xE38u) * ((1u << 8) | (1u << 18))) & 0x38003800u;
  return ev | od;
}
__device__ __forceinline__ uint32_t spread1(uint32_t n4) {  // 4 bits -> bit 6 of each byte
  return ((n4 & 0xFu) * (0x00204081u << 6)) & 0x40404040u;
}
__global__ void k_fp8_raw(const uint4* __restrict__ w, int P, uint32_t* out) {
  uint32_t acc = 0;
#pragma unroll 1
  for (int p = 0; p < P; ++p) {
    const uint4 v = __ldg(w + p * 32 + threadIdx.x);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  out[threadIdx.x] = acc;
}
__global__ void k_fp8_3bit(const uint2* __restrict__ nib, const uint32_t* __restrict__ c32,
                           const uint16_t* __restrict__ c16, const uint32_t* __restrict__ flags,
                           const uint16_t* __restrict__ x, const uint32_t* __restrict__ xoff, uint32_t base8, int P,
                           uint32_t* out) {
  const int lane = threadIdx.x & 31;
  uint32_t acc = 0;
#pragma unroll 1
  for (int p = 0; p < P; ++p) {
    const uint2 s = __ldg(nib + p * 32 + lane);
    const uint32_t ca = __ldg(c32 + p * 32 + lane), cb = __ldg(c16 + p * 32 + lane);
    const uint32_t fl = __ldg(flags + p);
    const bool f = (fl >> lane) & 1u;
    uint32_t hi = 0;
    if (f) hi = __ldg(x + __ldg(xoff + p) + __popc(fl & ((1u << lane) - 1u)));
    const uint32_t b8 = f ? 0u : base8;
    const uint32_t cw[4] = {ca, (ca >> 12), __funnelshift_r(ca, cb, 24), cb >> 4};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const uint32_t src = w < 2 ? s.x : s.y;
      const uint32_t n = ((w & 1) ? src >> 4 : src) & 0x0F0F0F0Fu;
      const uint32_t sm = (n * 0x11u) & 0x87878787u;
      const uint32_t e = spread3(cw[w] & 0xFFFu) + b8 + spread1(hi >> (4 * w));
      acc ^= sm | e;
    }
  }
  out[threadIdx.x] = acc;
}
// the head's two layouts
template __global__ void k_head_dec<true>(const uint2*, const uint32_t*, const uint8_t*, const uint8_t*, int, uint32_t*);
template __global__ void k_head_dec<false>(const uint2*, const uint32_t*, const uint8_t*, const uint8_t*, int, uint32_t*);
