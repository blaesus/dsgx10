// The routed formats' decoders, host and device (__host__ __device__ so the CPU checks run the very code the
// kernels run), plus the device-only helpers the kernels share and the format traits they are templated on.
// GLM-5.3's NVFP4 decoders unchanged; the INT4-g32 (compressed-tensors pack-quantized) and BF16 families new
// (docs/experts.md section 2).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
inline uint16_t f_to_bf16_bits(float f) {  // round to nearest even (finite f, subnormals included)
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

// bf16x2 multiply: the exact product of the two bf16 operands rounded once to nearest even (the device's
// mul.rn.bf16x2 / fma.rn with -0; on the host the f32 product of two bf16 values is exact, 16 significant bits,
// and f_to_bf16_bits rounds it). NVFP4's uses are exact in bf16; INT4's (u - z) * s is not always (up to 12
// significant bits) and rounds as compressed-tensors' dequantize does.
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
// bf16x2 subtract; INT4's use (128 + u) - (128 + z) is exact (an integer in [-15, 15])
GX_HD uint32_t bsub2(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
#else
  const uint16_t lo = f_to_bf16_bits(bf16_bits_to_f(a & 0xFFFF) - bf16_bits_to_f(b & 0xFFFF));
  const uint16_t hi = f_to_bf16_bits(bf16_bits_to_f(a >> 16) - bf16_bits_to_f(b >> 16));
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

// ---- INT4-g32, asymmetric (compressed-tensors pack-quantized) --------------------------------------------------
// The checkpoint stores a signed 4-bit value q and zero point z as nibbles u = q + 8, u_z = z + 8, and
// compressed-tensors' dequantize is bf16(bf16(q) - bf16(z)) * bf16 scale in BF16 arithmetic: the difference
// q - z = u - u_z is exact, the product is rounded once to nearest even (docs/model.md, docs/findings.md A0).
// Here: (0x4300 | u) is the bf16 128 + u exactly (ulp 1 in [128, 256)); subtracting 128 + u_z gives u - u_z
// exactly; one bf16 multiply by the scale rounds the exact product as the library does.
// 16 stored nibbles of one group (lo = values 0..7, hi = 8..15, value i at bits 4 (i % 8): the checkpoint's own
// order) -> 8 bf16x2 words, word m = values (2m, 2m + 1). zz = 0x4300 | u_z in both halves, s2 = the scale in both.
GX_HD void i4_dec16(uint32_t lo, uint32_t hi, uint32_t zz, uint32_t s2, uint32_t (&o)[8]) {
  const uint32_t v[2] = {lo, hi};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t w4 = v[h] >> 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t y = bperm(v[h], w4, (4 + j) << 8 | j);  // byte 0: values 2j (low nibble), byte 2: 2j + 1
      o[4 * h + j] = bmul2(bsub2((y & 0x000F000Fu) | 0x43004300u, zz), s2);
    }
  }
}
GX_HD uint32_t i4_zz(uint32_t uz) { return (uz & 15u) * 0x00010001u | 0x43004300u; }
GX_HD uint32_t bf16x2_bcast(uint32_t b16) { return (b16 & 0xFFFFu) * 0x00010001u; }
// The contract's weight, one value: bf16((u - u_z) * s) as bf16 bits (the host reference of every check)
inline uint16_t i4_value_bits(int u, int uz, uint16_t s) {
  return f_to_bf16_bits((float)(u - uz) * bf16_bits_to_f(s));
}

// ---- FP8 e4m3 (a fused FP8 shared expert): four bytes -> two bf16x2 words of the exact values (bits into bf16's
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
__device__ __forceinline__ uint32_t ldg_stream32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// One row's routing by one warp (all 32 lanes), the experts e = lane + 32 j, j < kRJ (kRJ * 32 >= kE; slots past
// kE hold key -inf and never win): lane k < 8 writes pick k's expert to idx[k] and its weight to w[k]. k_route's
// body; the decode kernels run it on the router's logits themselves (DecodeCfg::logits) with the same bits. Ties
// go to the lower expert id (torch.topk's order on the reference's keys, tested).
constexpr int kRJ = (kE + 31) / 32;  // 6 at 168
__device__ __forceinline__ void route_warp(const float* __restrict__ logits_row, const float* __restrict__ bias, int lane,
                                           int* idx, float* wout) {
  float s[kRJ], key[kRJ];
#pragma unroll
  for (int j = 0; j < kRJ; ++j) {
    const int e = lane + 32 * j;
    if (e < kE) {
      s[j] = route_sigmoid(__ldcg(logits_row + e));
      key[j] = s[j] + __ldg(bias + e);
    } else {
      s[j] = 0.f, key[j] = -INFINITY;
    }
  }
  float picked[8];
  int ids[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float bv = -INFINITY;
    int bj = 0;
#pragma unroll
    for (int j = 0; j < kRJ; ++j)
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
    for (int j = 0; j < kRJ; ++j)
      if (lane + 32 * j == be) sv = s[j], key[j] = -INFINITY;
    picked[k] = __shfl_sync(0xffffffffu, sv, be & 31);
  }
  const float sum = route_sum8(picked) + 1e-20f;
#pragma unroll
  for (int k = 0; k < 8; ++k)
    if (lane == k) idx[k] = ids[k], wout[k] = picked[k] / sum * 2.5f;
}

// ---- the routed formats as kernel template parameters ----------------------------------------------------------
// A block is 8 rows x kBK k (kU units of 128 k); lane l = 4 g + t holds row g of it and, in unit jj, the 32
// values k = 128 jj + 64 q + 16 t + i (q = 0, 1; i < 16), two groups of 16 that each feed four m16n8k16 MMAs as
// B fragments. Raw is a lane's share of one block in registers; dec(r, jj, q, o) makes group q of unit jj into
// 8 bf16x2 words (o[m] = values 2m, 2m + 1). kMul is the epilogue's factor besides MatDesc::s2 (NVFP4's 2^30).
// Every format feeds the MMAs the same lane map in the same k order, so a matrix's sums depend only on its
// decoded bf16 values: INT4's equal BF16's on the dequantised matrix bit for bit.
struct FmtNVFP4 {
  static constexpr uint32_t kFmt = kNVFP4;
  static constexpr int kBK = kBlkK, kBlk = kQ4Blk, kU = kBK / 128;
  static constexpr float kMul = 0x1p30f;
  struct Raw {
    uint4 v[4];  // unit jj's 16 B
    uint2 s;     // the eight scale bytes, byte 2 jj + q
  };
  __device__ static __forceinline__ void load(Raw& r, const uint8_t* blk, int lane) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) r.v[jj] = ldg_stream128(blk + 512 * jj + 16 * lane);
    r.s = ldg_stream64(blk + 2048 + 8 * lane);
  }
  __device__ static __forceinline__ uint32_t w(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
  __device__ static __forceinline__ void dec(const Raw& r, int jj, int q, uint32_t (&o)[8]) {
    const uint32_t sc = ((jj < 2 ? r.s.x : r.s.y) >> (8 * (2 * (jj & 1) + q))) & 0xFFu;
    q4_widen16(w(r.v[jj], 2 * q), w(r.v[jj], 2 * q + 1), e4m3_to_bf16x2(sc, 0x1p96f), o);
  }
};
// INT4 block (layout.h pack_int4): values as NVFP4's (unit jj at 512 jj + 16 l, byte 8 q + i / 2); the lane's
// eight groups (jj, q) have scale bf16 at 2048 + 32 g + 16 (t >> 1) + 2 (2 jj + q) and zero-point nibble 2 jj + q
// of the word at 2304 + 8 g + 4 (t >> 1) (lanes t = 0, 1 and t = 2, 3 read the same words).
struct FmtINT4 {
  static constexpr uint32_t kFmt = kINT4;
  static constexpr int kBK = kBlkK, kBlk = kI4Blk, kU = kBK / 128;
  static constexpr float kMul = 1.f;
  struct Raw {
    uint4 v[4];
    uint4 s;     // bf16 scales of groups (jj, q) at half 2 jj + q
    uint32_t z;  // zero-point nibbles, nibble 2 jj + q
  };
  __device__ static __forceinline__ void load(Raw& r, const uint8_t* blk, int lane) {
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) r.v[jj] = ldg_stream128(blk + 512 * jj + 16 * lane);
    const int g = lane >> 2, h = (lane >> 1) & 1;
    r.s = ldg_stream128(blk + 2048 + 32 * g + 16 * h);
    r.z = ldg_stream32(blk + 2304 + 8 * g + 4 * h);
  }
  __device__ static __forceinline__ void dec(const Raw& r, int jj, int q, uint32_t (&o)[8]) {
    const uint32_t sw = FmtNVFP4::w(r.s, jj);
    const uint32_t s2 = bperm(sw, 0, q ? 0x3232 : 0x1010);
    i4_dec16(FmtNVFP4::w(r.v[jj], 2 * q), FmtNVFP4::w(r.v[jj], 2 * q + 1), i4_zz(r.z >> (4 * (2 * jj + q))), s2, o);
  }
};
// BF16 block (layout.h pack_bf16): 8 rows x 128 k, the lane's 16 B at 512 p + 16 l = values 8p .. 8p + 7 of its
// 32 (k = 64 (p >> 1) + 16 t + 8 (p & 1) + 0..7): group q is v[2q], v[2q + 1] as they are.
struct FmtBF16 {
  static constexpr uint32_t kFmt = kBF16;
  static constexpr int kBK = kBlkKBF16, kBlk = kB16Blk, kU = kBK / 128;
  static constexpr float kMul = 1.f;
  struct Raw {
    uint4 v[4];
  };
  __device__ static __forceinline__ void load(Raw& r, const uint8_t* blk, int lane) {
#pragma unroll
    for (int p = 0; p < 4; ++p) r.v[p] = ldg_stream128(blk + 512 * p + 16 * lane);
  }
  __device__ static __forceinline__ void dec(const Raw& r, int jj, int q, uint32_t (&o)[8]) {
    (void)jj;
    const uint4 a = r.v[2 * q], b = r.v[2 * q + 1];
    o[0] = a.x, o[1] = a.y, o[2] = a.z, o[3] = a.w, o[4] = b.x, o[5] = b.y, o[6] = b.z, o[7] = b.w;
  }
};

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
