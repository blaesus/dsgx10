// The routed formats' decoders, host and device (__host__ __device__ so the CPU checks run the very code the
// kernels run), adapted from micro/common.cuh (experiment 1), plus the device-only helpers the kernels share.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "experts/experts.h"

#define GX_HD __host__ __device__ __forceinline__

namespace gx {

// ---- scalar helpers -----------------------------------------------------------------------------------------
GX_HD uint32_t bperm(uint32_t x, uint32_t y, uint32_t s) {  // prmt.b32 default mode; selector nibbles < 8 only
#ifdef __CUDA_ARCH__
  return __byte_perm(x, y, s);
#else
  const uint64_t v = ((uint64_t)y << 32) | x;
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) r |= (uint32_t)((v >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
  return r;
#endif
}
inline float bf16_bits_to_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
inline uint16_t f_to_bf16_bits(float f) {  // round to nearest even (finite f)
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
inline float bf16r(float f) { return bf16_bits_to_f(f_to_bf16_bits(f)); }
inline float e4m3_to_f(uint8_t b) { __nv_fp8_e4m3 v; memcpy(&v, &b, 1); return float(v); }
inline float e2m1_to_f(int c) {
  static const float t[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  return t[c & 15];
}

// bf16x2 multiply; every use multiplies values whose product is exact in bf16.
GX_HD uint32_t bmul2(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
#else
  const uint16_t lo = f_to_bf16_bits(bf16_bits_to_f(a & 0xFFFF) * bf16_bits_to_f(b & 0xFFFF));
  const uint16_t hi = f_to_bf16_bits(bf16_bits_to_f(a >> 16) * bf16_bits_to_f(b >> 16));
  return lo | (uint32_t)hi << 16;
#endif
}

// ---- NVFP4 ----------------------------------------------------------------------------------------------------
// 16 e2m1 values (lo = values 0..7, hi = 8..15, value 2i in byte i's low nibble) -> 8 bf16x2 words, word m =
// values (2m, 2m + 1), each v * s * 2^-30 where f2 = bf16(s * 2^96) in both halves (s the group's e4m3
// scale): the nibble's bits go into bf16's fields (v * 2^-126), then one exact multiply. The kernels put the
// 2^30 back with the per-tensor scale.
GX_HD void q4_widen16(uint32_t lo, uint32_t hi, uint32_t f2, uint32_t (&o)[8]) {
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
inline uint32_t q4_scale_f2_host(uint8_t s) {  // bf16(e4m3(s) * 2^96), both halves
  const uint16_t b = f_to_bf16_bits(e4m3_to_f(s) * 0x1p96f);
  return b | (uint32_t)b << 16;
}
inline uint32_t s3_scale_f2_host(uint8_t s) {  // bf16(e4m3(s & 0x7f)), both halves
  const uint16_t b = f_to_bf16_bits(e4m3_to_f(s & 0x7F));
  return b | (uint32_t)b << 16;
}

// ---- E2M1-subset 3-bit code (S3) ----------------------------------------------------------------------------
// A tensor has one or two subsets of four E2M1 magnitudes m0 < m1 < m2 < m3 (m0 may be 0); code c = s << 2 | j
// means (-1)^s m_j; bit 7 of a group's scale byte picks the subset, its low 7 bits are the e4m3 scale.
// A lane's 32 values v[0..31] (two groups of 16) of one 128-k unit are three words:
//   P0 nibble m: bits 1:0 = v[m] & 3, bits 3:2 = v[m + 8] & 3;  P1: the same for v[16 + m], v[24 + m];
//   Q  nibble m: bit 0 = v[m] >> 2, bit 1 = v[m + 8] >> 2, bit 2 = v[m + 16] >> 2, bit 3 = v[m + 24] >> 2.
// Decode: two lop3s give a word of eight 3-bit selectors and prmt reads each code's bf16 from the subset's
// 8-entry table (low bytes T[0]:T[1], high bytes T[2]:T[3]), then one exact multiply by the group's scale.
struct S3Tab { uint32_t tl0, tl1, th0, th1; };
inline S3Tab s3_table(const float mags[4]) {  // mags ascending, E2M1 magnitudes
  S3Tab t{0, 0, 0, 0};
  for (int c = 0; c < 8; ++c) {
    const float v = (c & 4 ? -1.f : 1.f) * mags[c & 3];
    const uint16_t b = f_to_bf16_bits(v);
    const uint32_t lo = b & 0xFF, hi = b >> 8;
    if (c < 4) { t.tl0 |= lo << 8 * c; t.th0 |= hi << 8 * c; }
    else { t.tl1 |= lo << 8 * (c - 4); t.th1 |= hi << 8 * (c - 4); }
  }
  return t;
}
GX_HD uint32_t s3_sel(uint32_t P0, uint32_t P1, uint32_t Q, int which) {  // selectors of values 8 which .. + 7
  switch (which) {
    case 0: return (P0 & 0x33333333u) | ((Q << 2) & 0x44444444u);
    case 1: return ((P0 >> 2) & 0x33333333u) | ((Q << 1) & 0x44444444u);
    case 2: return (P1 & 0x33333333u) | (Q & 0x44444444u);
    default: return ((P1 >> 2) & 0x33333333u) | ((Q >> 1) & 0x44444444u);
  }
}
// four values (selectors in sel's low 16 bits) -> two bf16x2 words (values 0,1 and 2,3) times f2
GX_HD void s3_dec4(uint32_t sel, const S3Tab& T, uint32_t f2, uint32_t& w0, uint32_t& w1) {
  const uint32_t lo = bperm(T.tl0, T.tl1, sel), hi = bperm(T.th0, T.th1, sel);
  w0 = bmul2(bperm(lo, hi, 0x5140), f2);
  w1 = bmul2(bperm(lo, hi, 0x7362), f2);
}
// one group (16 values, q = 0: v[0..15], 1: v[16..31]) -> 8 bf16x2 words
GX_HD void s3_group(uint32_t P0, uint32_t P1, uint32_t Q, int q, const S3Tab& T, uint32_t f2, uint32_t (&o)[8]) {
  const uint32_t sa = s3_sel(P0, P1, Q, 2 * q), sb = s3_sel(P0, P1, Q, 2 * q + 1);
  s3_dec4(sa, T, f2, o[0], o[1]);
  s3_dec4(sa >> 16, T, f2, o[2], o[3]);
  s3_dec4(sb, T, f2, o[4], o[5]);
  s3_dec4(sb >> 16, T, f2, o[6], o[7]);
}
inline void s3_pack_lane(const uint8_t v[32], uint32_t& P0, uint32_t& P1, uint32_t& Q) {
  P0 = P1 = Q = 0;
  for (int m = 0; m < 8; ++m) {
    P0 |= (uint32_t)(v[m] & 3) << 4 * m | (uint32_t)(v[m + 8] & 3) << (4 * m + 2);
    P1 |= (uint32_t)(v[m + 16] & 3) << 4 * m | (uint32_t)(v[m + 24] & 3) << (4 * m + 2);
    Q |= (uint32_t)(v[m] >> 2) << 4 * m | (uint32_t)(v[m + 8] >> 2) << (4 * m + 1) |
         (uint32_t)(v[m + 16] >> 2) << (4 * m + 2) | (uint32_t)(v[m + 24] >> 2) << (4 * m + 3);
  }
}
// S3 code -> e2m1 nibble of the same value (for the FP4 tensor-core path): magnitude index of m_j in the
// E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, sign in bit 3. The table packs the 8 nibbles of a subset in a word.
inline uint32_t s3_nibble_table(const float mags[4]) {
  static const float g[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  uint32_t t = 0;
  for (int c = 0; c < 8; ++c) {
    int m = 0;
    for (int i = 0; i < 8; ++i)
      if (g[i] == mags[c & 3]) m = i;
    t |= (uint32_t)((c & 4 ? 8 : 0) | m) << (4 * c);
  }
  return t;
}

// ---- NVFP4 activations (the FP4 prefill path) ---------------------------------------------------------------
// A group of 16 values x with the static input scale S: amax = max |x|, sb = e4m3_rn_sat(amax / (6 S)),
// d = e4m3(sb) S (f32), q = e2m1_rn_sat(x / d) (0 when d = 0); x ~ e2m1(q) e4m3(sb) S. Every step is one
// IEEE f32 operation, so the host function below and the device kernel give the same bits.
inline void nvfp4_quant16_host(const float* x, float S, uint8_t* q8, uint8_t& sb) {
  float amax = 0;
  for (int i = 0; i < 16; ++i) amax = std::max(amax, fabsf(x[i]));
  const float t = amax / (6.f * S);
  sb = (uint8_t)__nv_cvt_float_to_fp8(t, __NV_SATFINITE, __NV_E4M3);
  const float d = e4m3_to_f(sb) * S;
  for (int i = 0; i < 8; ++i) {
    uint8_t b = 0;
    for (int h = 0; h < 2; ++h) {
      const float v = d > 0 ? x[2 * i + h] / d : 0.f;
      b |= (uint8_t)(__nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest) & 15) << (4 * h);
    }
    q8[i] = b;
  }
}

// ---- FP8 e4m3 (the shared expert): four bytes -> two bf16x2 words of the exact values (bits into bf16's
// fields give v * 2^-120, then one exact multiply by 2^120).
GX_HD void fp8x4_to_bf16(uint32_t v, uint32_t& o0, uint32_t& o1) {
  const uint32_t k = 0x7B807B80u;  // bf16 2^120 in both halves
  const uint32_t r0 = bperm(v, 0, 0x4140), r1 = bperm(v, 0, 0x4342);
  o0 = bmul2(((r0 << 4) & 0x07F007F0u) | ((r0 << 8) & 0x80008000u), k);
  o1 = bmul2(((r1 << 4) & 0x07F007F0u) | ((r1 << 8) & 0x80008000u), k);
}

// splitmix64: the tests' and benches' random data
GX_HD uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// The routing function, host and device (route.cu runs it a warp a row; the tests run it on the CPU), from
// IEEE operations only (explicit fmas, one division) so the CPU and the GPU give the same bits:
// s = 1 / (1 + e^-l), e^x = 2^n p(r), n = rint(x log2 e), r = x - n ln 2 in two fmas, p a degree-7 Taylor
// polynomial in Horner form (|r| <= 0.35: error < 1.5e-8 relative before rounding).
GX_HD float route_sigmoid(float l) {
  const float x = -l;
  if (x > 88.f) return 0.f;
  if (x < -87.f) return 1.f;
  const float n = rintf(x * 1.44269504f);
  float r = fmaf(-n, 0.693145751953125f, x);
  r = fmaf(-n, 1.428606765330187e-06f, r);
  float p = 1.f / 5040;
  p = fmaf(p, r, 1.f / 720);
  p = fmaf(p, r, 1.f / 120);
  p = fmaf(p, r, 1.f / 24);
  p = fmaf(p, r, 1.f / 6);
  p = fmaf(p, r, 0.5f);
  p = fmaf(p, r, 1.f);
  p = fmaf(p, r, 1.f);
  const float e = ldexpf(p, (int)n);
#ifdef __CUDA_ARCH__
  return __fdiv_rn(1.f, __fadd_rn(1.f, e));
#else
  return 1.f / (1.f + e);
#endif
}
// The sum of the eight picked scores (picks in descending selection order p0..p7): the order torch's
// CUDA reduction was found to use for DeepSeek's six (ds41f tests/probe_router_sum.py), extended to eight.
GX_HD float route_sum8(const float p[8]) {
  return ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));
}

#ifdef __CUDACC__
// ---- device-only ------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t e4m3_to_bf16x2(uint32_t s, float mul) {  // bf16(e4m3(s & 0xff) * mul), both halves
  const __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(s & 0xFF), __NV_E4M3);
  __nv_bfloat162 r = __float2bfloat162_rn(__half2float(__half(hr)) * mul);
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma_bf16(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldg_stream128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ldg_stream64(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// One row's routing by one warp (all 32 lanes), eight experts a lane (e = lane + 32 j): lane k < 8 writes pick k's
// expert to idx[k] and its weight to w[k]. k_route's body; the decode kernels run it on the router's logits
// themselves (DecodeCfg::logits) with the same bits.
__device__ __forceinline__ void route_warp(const float* __restrict__ logits_row, const float* __restrict__ bias, int lane,
                                           int* idx, float* wout) {
  float s[8], key[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    s[j] = route_sigmoid(__ldcg(logits_row + lane + 32 * j));
    key[j] = s[j] + __ldg(bias + lane + 32 * j);
  }
  float picked[8];
  int ids[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
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
  for (int k = 0; k < 8; ++k)
    if (lane == k) idx[k] = ids[k], wout[k] = picked[k] / sum * 2.5f;
}

// ---- a routed block in registers and its decode (decode and prefill kernels) -----------------------------------
struct Raw {
  uint4 v[4];  // NVFP4: unit jj's 16 B; S3: P0, P1, Q (each 4 words, one a unit)
  uint2 s;     // the eight scale bytes, byte 2 jj + q
};
__device__ __forceinline__ void load_blk(Raw& r, const uint8_t* blk, uint32_t fmt, int lane) {
  if (fmt == kNVFP4) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) r.v[jj] = ldg_stream128(blk + 512 * jj + 16 * lane);
    r.s = ldg_stream64(blk + 2048 + 8 * lane);
  } else {
#pragma unroll
    for (int p = 0; p < 3; ++p) r.v[p] = ldg_stream128(blk + 512 * p + 16 * lane);
    r.s = ldg_stream64(blk + 1536 + 8 * lane);
  }
}
__device__ __forceinline__ uint32_t word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
// group q of unit jj -> 8 bf16x2 words (B fragments of four MMAs); NVFP4 values carry 2^-30
__device__ __forceinline__ void dec_group(const Raw& r, int jj, int q, uint32_t fmt, const uint32_t (&tab)[8],
                                          uint32_t (&o)[8]) {
  const uint32_t sc = ((jj < 2 ? r.s.x : r.s.y) >> (8 * (2 * (jj & 1) + q))) & 0xFFu;
  if (fmt == kNVFP4) {
    q4_widen16(word(r.v[jj], 2 * q), word(r.v[jj], 2 * q + 1), e4m3_to_bf16x2(sc, 0x1p96f), o);
  } else {
    const bool hi = sc & 0x80u;
    const S3Tab T{hi ? tab[4] : tab[0], hi ? tab[5] : tab[1], hi ? tab[6] : tab[2], hi ? tab[7] : tab[3]};
    s3_group(word(r.v[0], jj), word(r.v[1], jj), word(r.v[2], jj), q, T, e4m3_to_bf16x2(sc & 0x7Fu, 1.f), o);
  }
}
__device__ __forceinline__ float fmt_k(uint32_t fmt) { return fmt == kNVFP4 ? 0x1p30f : 1.f; }

// A fragments of a 16-k group (k = k0 + 0..15 of rows r0 and r1) and the four MMAs against one tile
struct XF { uint4 a, c, a2, c2; };
template <bool R16>
__device__ __forceinline__ void load_x(XF& f, const __nv_bfloat16* x0, const __nv_bfloat16* x1, int k0) {
  f.a = __ldg(reinterpret_cast<const uint4*>(x0 + k0));
  f.c = __ldg(reinterpret_cast<const uint4*>(x0 + k0 + 8));
  if (R16) {
    f.a2 = __ldg(reinterpret_cast<const uint4*>(x1 + k0));
    f.c2 = __ldg(reinterpret_cast<const uint4*>(x1 + k0 + 8));
  } else {
    f.a2 = f.c2 = make_uint4(0, 0, 0, 0);
  }
}
__device__ __forceinline__ void mma4(float c[4], const XF& f, const uint32_t (&o)[8]) {
  mma_bf16(c, f.a.x, f.a2.x, f.a.y, f.a2.y, o[0], o[1]);
  mma_bf16(c, f.a.z, f.a2.z, f.a.w, f.a2.w, o[2], o[3]);
  mma_bf16(c, f.c.x, f.c2.x, f.c.y, f.c2.y, o[4], o[5]);
  mma_bf16(c, f.c.z, f.c2.z, f.c.w, f.c2.w, o[6], o[7]);
}

// FP8 unit (128 k of one strip): two 16 B per lane
struct Raw8 { uint4 q0, q1; };
__device__ __forceinline__ void dec8(const uint4& v, uint32_t (&o)[8]) {
  fp8x4_to_bf16(v.x, o[0], o[1]);
  fp8x4_to_bf16(v.y, o[2], o[3]);
  fp8x4_to_bf16(v.z, o[4], o[5]);
  fp8x4_to_bf16(v.w, o[6], o[7]);
}

#endif

}  // namespace gx
