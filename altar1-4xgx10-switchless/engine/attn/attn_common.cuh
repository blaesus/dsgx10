// Device helpers shared by the attention-side kernels: bf16/e4m3 conversions, the bf16 MMA, ldmatrix,
// streaming loads, PDL, RoPE in the reference's bf16 arithmetic, the launcher.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "attn/attn.h"
#include "common/check.h"

namespace glm {

__device__ __forceinline__ float bf2f(uint32_t b) { return __uint_as_float((b & 0xFFFFu) << 16); }
__device__ __forceinline__ float bf2f_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfround(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { return (uint32_t)f2bf(lo) | (uint32_t)f2bf(hi) << 16; }

__device__ __forceinline__ float e4m3_to_f(uint32_t b) {
  __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(b & 0xFF), __NV_E4M3);
  return __half2float(__half(h));
}
__device__ __forceinline__ uint8_t f_to_e4m3(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }

// four e4m3 bytes -> two bf16x2 words, exactly: the e4m3 bits moved into bf16's fields (bf16 has the wider
// exponent), then one bf16 multiply by 2^120 restores the bias (e4m3 subnormals land on bf16 normals exactly).
// o0 = (byte0, byte1), o1 = (byte2, byte3).
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void fp8x4_to_bf16(uint32_t v, uint32_t& o0, uint32_t& o1) {
  const uint32_t k = 0x7B807B80u;  // bf16 2^120 in both halves
  const uint32_t r0 = prmt(v, 0, 0x4140), r1 = prmt(v, 0, 0x4342);
  o0 = bmul2(((r0 << 4) & 0x07F007F0u) | ((r0 << 8) & 0x80008000u), k);
  o1 = bmul2(((r1 << 4) & 0x07F007F0u) | ((r1 << 8) & 0x80008000u), k);
}

// ---- INT4 (compressed-tensors, group 32, asymmetric): the contract's weight per element -----------------------
// W = bf16_rne((nib - znib) * s) (docs/model.md § 4.2). A nibble u becomes the bf16 128 + u exactly (0x4300 | u: the
// ulp is 1 in [128, 256)); the bf16 subtraction of 128 + znib is exact (an integer in [-15, 15]); the one bf16
// multiply by the scale rounds the exact product to nearest even, as torch's bf16 multiply in compressed-tensors'
// _dequantize (attn_tests D compares every scale pattern with the host's f32-product rounding, which equals torch's:
// tools/int4_check.py).
__device__ __forceinline__ uint32_t bsub2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// (a & 0x00FF00FF) | 0x43004300
__device__ __forceinline__ uint32_t or4300(uint32_t a) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(0x00FF00FFu), "r"(0x43004300u));
  return r;
}
// zz: the zero point as bf16 128 + znib in both halves; ss: the bf16 scale in both halves
__device__ __forceinline__ uint32_t int4_zz(uint32_t znib) { return (znib | (znib << 16)) | 0x43004300u; }
// 16 nibbles (lo: values 0..7, hi: 8..15, value 2i in byte i's low nibble: the checkpoint's order) -> 8 bf16x2
// words, word m = W of values (2m, 2m + 1)
__device__ __forceinline__ void int4x16_to_bf16(uint32_t lo, uint32_t hi, uint32_t zz, uint32_t ss, uint32_t (&o)[8]) {
  const uint32_t v[2] = {lo, hi};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t e = v[h] & 0x0F0F0F0Fu, d = (v[h] >> 4) & 0x0F0F0F0Fu;  // even / odd values, a byte each
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const uint32_t w = or4300(prmt(e, d, (uint32_t)(m | (4 + m) << 8)));  // (128 + v[2m], 128 + v[2m+1])
      o[4 * h + m] = bmul2(bsub2(w, zz), ss);
    }
  }
}
__device__ __forceinline__ uint32_t ldg_stream32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ uint4 ldg_stream128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

// PDL: let the next kernel in the stream launch / wait for the previous one's results.
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// RoPE of one interleaved pair in the reference's bf16 arithmetic (apply_rotary_pos_emb_interleave: every
// product and the sum rounded to bf16). cs: bf16 cos (low half), bf16 sin (high half). x0 = x[2i], x1 = x[2i+1]
// -> out[i] = x0 c - x1 s, out[32 + i] = x1 c + x0 s.
__device__ __forceinline__ void rope_pair(float x0, float x1, uint32_t cs, float& lo, float& hi) {
  const float c = bf2f(cs), s = bf2f_hi(cs);
  lo = bfround(__fsub_rn(bfround(__fmul_rn(x0, c)), bfround(__fmul_rn(x1, s))));
  hi = bfround(__fadd_rn(bfround(__fmul_rn(x1, c)), bfround(__fmul_rn(x0, s))));
}

// fp32 -> u32 order key (larger value, larger key); -0 counts as +0 (torch compares them equal).
__device__ __forceinline__ uint32_t order_key(float v) {
  uint32_t u = __float_as_uint(v == 0.f ? 0.f : v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

}  // namespace glm
