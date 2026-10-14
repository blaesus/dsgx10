// Shared pieces of the glm53 micro benchmarks: host/device decoders for the four routed-expert formats
// (NVFP4, the E2M1-subset 3-bit code, EXL3 trellis K3/K4, FP8 128x128) and the MMA wrappers. Every decoder
// is __host__ __device__ so the CPU self-check (gemv --cpu-check) runs the very code the kernels run.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)
#define HD __host__ __device__ __forceinline__

// ---------------------------------------------------------------------------------------------------------
// scalar helpers (host and device)
HD uint32_t bperm(uint32_t x, uint32_t y, uint32_t s) {  // prmt.b32 default mode; selector nibbles < 8 only
#ifdef __CUDA_ARCH__
  return __byte_perm(x, y, s);
#else
  uint64_t v = ((uint64_t)y << 32) | x;
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) r |= (uint32_t)((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
  return r;
#endif
}
static inline float bf16_bits_to_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f_to_bf16_bits(float f) {  // round to nearest even, f finite
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline float h_bits_to_f(uint16_t b) { __half_raw r; r.x = b; return __half2float(__half(r)); }
static inline uint16_t f_to_h_bits(float f) { __half h = __float2half_rn(f); return ((__half_raw)h).x; }
static inline uint16_t d_to_h_bits(double d) {  // round to nearest even, |d| < 65520
  if (d == 0) return std::signbit(d) ? 0x8000 : 0;
  int e; frexp(d, &e);              // |d| in [2^(e-1), 2^e)
  int ex = e - 1 < -14 ? -14 : e - 1;
  double q = nearbyint(ldexp(d, 10 - ex));  // the significand in units of the last place (ties to even)
  return f_to_h_bits((float)ldexp(q, ex - 10));  // exact: q has <= 11 bits
}
static inline float e4m3_to_f(uint8_t b) { __nv_fp8_e4m3 v; memcpy(&v, &b, 1); return float(v); }
static inline float e2m1_to_f(int c) {
  static const float t[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  return t[c & 15];
}

// bf16x2 multiply. Every use multiplies values whose product is exact in bf16 (the checks confirm it),
// so the host form only has to compute the product and round.
HD uint32_t bmul2(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
#else
  uint16_t lo = f_to_bf16_bits(bf16_bits_to_f(a & 0xFFFF) * bf16_bits_to_f(b & 0xFFFF));
  uint16_t hi = f_to_bf16_bits(bf16_bits_to_f(a >> 16) * bf16_bits_to_f(b >> 16));
  return lo | (uint32_t)hi << 16;
#endif
}
// half2 add, half2 fma (EXL3's codebooks)
HD uint32_t hadd2_bits(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  __half2 r = __hadd2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
#else
  uint16_t lo = f_to_h_bits(h_bits_to_f(a & 0xFFFF) + h_bits_to_f(b & 0xFFFF));  // exact in f32, one rounding
  uint16_t hi = f_to_h_bits(h_bits_to_f(a >> 16) + h_bits_to_f(b >> 16));
  return lo | (uint32_t)hi << 16;
#endif
}
HD uint32_t hfma2_bits(uint32_t a, uint32_t b, uint32_t c) {
#ifdef __CUDA_ARCH__
  __half2 r = __hfma2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b), *reinterpret_cast<__half2*>(&c));
  return *reinterpret_cast<uint32_t*>(&r);
#else
  auto one = [](uint16_t x, uint16_t y, uint16_t z) {
    double d = (double)h_bits_to_f(x) * h_bits_to_f(y) + h_bits_to_f(z);  // exact in double
    return d_to_h_bits(d);  // one rounding, as the fma
  };
  return one(a & 0xFFFF, b & 0xFFFF, c & 0xFFFF) | (uint32_t)one(a >> 16, b >> 16, c >> 16) << 16;
#endif
}
HD uint32_t dp4a_u(uint32_t a, uint32_t b, uint32_t c) {
#ifdef __CUDA_ARCH__
  return __dp4a(a, b, c);
#else
  uint32_t s = c;
  for (int i = 0; i < 4; ++i) s += ((a >> 8 * i) & 0xFF) * ((b >> 8 * i) & 0xFF);
  return s;
#endif
}

// ---------------------------------------------------------------------------------------------------------
// NVFP4: 16 e2m1 values (lo = values 0..7, hi = 8..15, value 2i in byte i's low nibble) -> 8 bf16x2 words,
// word m = values (2m, 2m + 1), each value v * s * 2^-30 where f2 = bf16(s * 2^96) in both halves (s = the
// group's e4m3 scale). The nibble's bits go into bf16's fields (v * 2^-126, the subnormal 0.5 included,
// as in ds41f's gemv_fp4.cu), then one exact multiply. The 2^30 comes back in the epilogue.
HD void q4_widen16(uint32_t lo, uint32_t hi, uint32_t f2, uint32_t (&o)[8]) {
  const uint32_t v[2] = {lo, hi};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t w4 = v[h] >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t y = bperm(v[h], w4, (4 + j) << 12 | (4 + j) << 8 | j << 4 | j);
      const uint32_t b = ((y << 12) & 0x80008000u) | ((y << 6) & 0x01C001C0u);
      o[4 * h + j] = bmul2(b, f2);
    }
  }
}
static inline uint32_t q4_scale_f2_host(uint8_t s) {  // bf16(e4m3(s) * 2^96), both halves
  uint16_t b = f_to_bf16_bits(e4m3_to_f(s) * 0x1p96f);
  return b | (uint32_t)b << 16;
}
#ifdef __CUDACC__
__device__ __forceinline__ uint32_t e4m3_to_bf16x2_scaled(uint32_t s, float mul) {  // bf16(e4m3(s) * mul), both halves
  __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(s & 0xFF), __NV_E4M3);
  __nv_bfloat162 r = __float2bfloat162_rn(__half2float(__half(hr)) * mul);
  return *reinterpret_cast<uint32_t*>(&r);
}
#endif

// ---------------------------------------------------------------------------------------------------------
// E2M1-subset 3-bit code ("S3"): 8 of the 16 e2m1 code points, an e4m3 scale per 16, 3.5 bits a weight.
// The quantisation study's best subset is {+-0.5, +-2, +-4, +-6}: code c = sign << 2 | idx, idx -> {0.5, 2, 4, 6}.
// Storage of a lane's 32 values v[0..31] (two groups of 16) in three words:
//   P0 nibble m: bits 1:0 = v[m] & 3, bits 3:2 = v[m + 8] & 3;  P1: the same for v[16 + m], v[24 + m];
//   Q  nibble m: bit 0 = v[m] >> 2, bit 1 = v[m + 8] >> 2, bit 2 = v[m + 16] >> 2, bit 3 = v[m + 24] >> 2.
// Decode: two lop3s make a word of eight 3-bit selectors, and prmt reads the bf16 value of each code from an
// 8-entry table (low bytes TL0:TL1, high bytes TH0:TH1), then one exact multiply by the group's scale.
static const uint8_t kS3Nibble[8] = {0x1, 0x4, 0x6, 0x7, 0x9, 0xC, 0xE, 0xF};  // code -> e2m1 nibble
struct S3Tab { uint32_t tl0, tl1, th0, th1; };
static inline S3Tab s3_tables() {
  S3Tab t{0, 0, 0, 0};
  for (int c = 0; c < 8; ++c) {
    uint16_t b = f_to_bf16_bits(e2m1_to_f(kS3Nibble[c]));
    uint32_t lo = b & 0xFF, hi = b >> 8;
    if (c < 4) { t.tl0 |= lo << 8 * c; t.th0 |= hi << 8 * c; }
    else { t.tl1 |= lo << 8 * (c - 4); t.th1 |= hi << 8 * (c - 4); }
  }
  return t;
}
HD uint32_t s3_sel(uint32_t P0, uint32_t P1, uint32_t Q, int which) {  // selectors of values 8*which .. +7
  switch (which) {
    case 0: return (P0 & 0x33333333u) | ((Q << 2) & 0x44444444u);
    case 1: return ((P0 >> 2) & 0x33333333u) | ((Q << 1) & 0x44444444u);
    case 2: return (P1 & 0x33333333u) | (Q & 0x44444444u);
    default: return ((P1 >> 2) & 0x33333333u) | ((Q >> 1) & 0x44444444u);
  }
}
// four values (selectors in sel's low 16 bits) -> two bf16x2 words (values 0,1 and 2,3) times f2
HD void s3_dec4(uint32_t sel, const S3Tab& T, uint32_t f2, uint32_t& w0, uint32_t& w1) {
  const uint32_t lo = bperm(T.tl0, T.tl1, sel), hi = bperm(T.th0, T.th1, sel);
  w0 = bmul2(bperm(lo, hi, 0x5140), f2);
  w1 = bmul2(bperm(lo, hi, 0x7362), f2);
}
static inline void s3_pack_lane(const uint8_t v[32], uint32_t& P0, uint32_t& P1, uint32_t& Q) {
  P0 = P1 = Q = 0;
  for (int m = 0; m < 8; ++m) {
    P0 |= (uint32_t)(v[m] & 3) << 4 * m | (uint32_t)(v[m + 8] & 3) << (4 * m + 2);
    P1 |= (uint32_t)(v[m + 16] & 3) << 4 * m | (uint32_t)(v[m + 24] & 3) << (4 * m + 2);
    Q |= (uint32_t)(v[m] >> 2) << 4 * m | (uint32_t)(v[m + 8] >> 2) << (4 * m + 1) |
         (uint32_t)(v[m + 16] >> 2) << (4 * m + 2) | (uint32_t)(v[m + 24] >> 2) << (4 * m + 3);
  }
}

// ---------------------------------------------------------------------------------------------------------
// FP8 e4m3 (zai's 128x128 block FP8): four bytes -> two bf16x2 words of exact values (bits moved into bf16's
// fields give v * 2^-120, then one exact multiply by 2^120).
HD void fp8x4_to_bf16(uint32_t v, uint32_t& o0, uint32_t& o1) {
  const uint32_t k = 0x7B807B80u;  // bf16 2^120 in both halves
  const uint32_t r0 = bperm(v, 0, 0x4140), r1 = bperm(v, 0, 0x4342);
  o0 = bmul2(((r0 << 4) & 0x07F007F0u) | ((r0 << 8) & 0x80008000u), k);
  o1 = bmul2(((r1 << 4) & 0x07F007F0u) | ((r1 << 8) & 0x80008000u), k);
}

// ---------------------------------------------------------------------------------------------------------
// EXL3 trellis (exllamav3 master as fetched 2026-10-13, quant/codebook.cuh, exl3_dq.cuh, pack.cu, quantize.py).
// A 16x16 tile of the rotated weight W_inner[k][n] is 256 K-bit codes in a circular, MSB-first bitstream of 8K
// u32 words (the trellis tensor's int16s, read as little-endian u32, as every EXL3 kernel reads them). Position
// p's 16-bit state is the window of the stream that ends at bit (p + 1) K. Position p holds tile element
// tensor_core_perm[p] = k * 16 + n. The state is turned into an f16 by the procedural codebook:
//   cb 0 "3INST": x = s * 89226354 + 64248484; x = (x & 0x8fff8fff) ^ 0x3b603b60; half(lo) + half(hi)
//   cb 1 "MCG":   x = s * 0xCBAC1FED;          same lop3 and add
//   cb 2 "MUL1":  x = s * 0x83DCD12D; t = bytesum(x) + 0x6400 as an f16 bit pattern; t * 0x1eee + 0xc931 (f16 fma)
template <int CB>
HD uint32_t tr_decode2(uint32_t s0, uint32_t s1) {  // two states (< 2^16) -> half2 bits (s0 in the low half)
  if (CB == 2) {
    const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
    const uint32_t t0 = dp4a_u(x0, 0x01010101u, 0x6400u), t1 = dp4a_u(x1, 0x01010101u, 0x6400u);
    return hfma2_bits((t0 & 0xFFFF) | (t1 << 16), 0x1eee1eeeu, 0xc931c931u);
  }
  uint32_t x0, x1;
  if (CB == 0) { x0 = s0 * 89226354u + 64248484u; x1 = s1 * 89226354u + 64248484u; }
  else { x0 = s0 * 0xCBAC1FEDu; x1 = s1 * 0xCBAC1FEDu; }
#ifdef __CUDA_ARCH__
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
#else
  x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
  x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
#endif
  const uint32_t lows = (x0 & 0xFFFF) | (x1 << 16), highs = (x0 >> 16) | (x1 & 0xFFFF0000u);
  return hadd2_bits(lows, highs);
}
// EXL3's own single-weight dequantisation (exl3_dq.cuh dq<bits, cb>), ported literally: the CPU reference.
template <int K, int CB>
static inline uint16_t exl3_dq_ref(const uint32_t* ptr, int t) {
  const int W = K * 256 / 32;
  int b0 = t * K + K - 16 + 256 * K, b1 = b0 + 16, i0 = b0 / 32, i1 = (b1 - 1) / 32, s0 = (i1 + 1) * 32 - b1;
  uint32_t a = ptr[i0 % W], b = ptr[i1 % W];
  uint32_t w0 = (uint32_t)((((uint64_t)a << 32) | b) >> (s0 & 31)) & 0xffff;  // __funnelshift_r(b, a, s0)
  uint32_t r = tr_decode2<CB>(w0, 0);
  return (uint16_t)(r & 0xFFFF);
}
static inline int exl3_perm(int p) {  // quantize.py tensor_core_perm: position p -> k * 16 + n
  int t = p / 8, j = p % 8, r0 = (t % 4) * 2, c0 = t / 4;
  int r = r0 + (j & 1) + ((j & 2) ? 8 : 0), c = c0 + ((j & 4) ? 8 : 0);
  return r * 16 + c;
}
// Lane l decodes positions 8l .. 8l + 7. Its windows lie in two consecutive words (i0, i0 + 1 mod 8K) for
// K = 2, 3, 4; o0 = where the first window ends, counted from word i0's MSB.
template <int K>
HD void tr_lane_consts(int lane, int& i0, int& i1, int& o0) {
  const int W = 8 * K, s = ((8 * lane + 1) * K - 16 + 256 * K) % (256 * K);
  i0 = s / 32;
  i1 = (i0 + 1) % W;
  o0 = s - 32 * i0 + 16;
}
// the lane's eight states from its two words, then the A fragment of m16n8k16 (m = n of the tile, k = k):
// a0 = (n g, k 2t..2t+1) = positions 0, 1; a1 = (g + 8, 2t..) = 4, 5; a2 = (g, 2t + 8..) = 2, 3; a3 = 6, 7
template <int K, int CB>
HD void tr_lane_frag(uint32_t wa, uint32_t wb, int o0, uint32_t (&a)[4]) {
  const uint64_t m = ((uint64_t)wa << 32) | wb;
  uint32_t s[8];
#pragma unroll
  for (int p = 0; p < 8; ++p) s[p] = (uint32_t)(m >> (64 - (o0 + p * K))) & 0xFFFFu;
  a[0] = tr_decode2<CB>(s[0], s[1]);
  a[2] = tr_decode2<CB>(s[2], s[3]);
  a[1] = tr_decode2<CB>(s[4], s[5]);
  a[3] = tr_decode2<CB>(s[6], s[7]);
}

// ---------------------------------------------------------------------------------------------------------
// MMA wrappers
#ifdef __CUDACC__
__device__ __forceinline__ void mma_bf16(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldg_stream128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldg_stream32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldg_stream16(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}
// FWHT over 128 values held 4 a lane (element 4 * lane + i), in place: stages 1, 2 in the lane, then 4 .. 64
// across lanes; lower element a, upper b -> (a + b, a - b). The CPU reference runs the same order.
__device__ __forceinline__ void had128_warp(float (&h)[4]) {
  const int lane = threadIdx.x & 31;
  float s0 = h[0] + h[1], d0 = h[0] - h[1], s1 = h[2] + h[3], d1 = h[2] - h[3];
  h[0] = s0 + s1; h[1] = d0 + d1; h[2] = s0 - s1; h[3] = d0 - d1;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float p = __shfl_xor_sync(0xffffffffu, h[i], m);
      h[i] = (lane & m) ? p - h[i] : h[i] + p;
    }
  }
}
#endif
static inline void had128_ref(float* v) {  // the same order as had128_warp
  for (int h = 1; h < 128; h <<= 1)
    for (int j = 0; j < 128; ++j)
      if (!(j & h)) { float a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
}
static const float kHadScale = 0.088388347648f;  // 1/sqrt(128), EXL3's r_scale

// splitmix64: the random data of every benchmark, generated on the GPU and re-read by the CPU references
HD uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
