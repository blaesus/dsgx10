// Device helpers of the prefill attention-side kernels (prefill_attn/index/comp/mhc.cu): cp.async,
// ldmatrix, the f16 / FP4 MMAs, and the e4m3 / e2m1 / ue8m0 -> f16 conversions. sm_121a only.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace dsk {
namespace pa {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// 16 bytes global -> shared, zero-filled when !pred (the source is then not read). No memory clobber:
// the copy lands asynchronously anyway; cp_wait / cp_commit order it against the shared-memory readers.
__device__ __forceinline__ void cp16(void* dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// named barrier `id` (1..15) over `n` threads (a multiple of 32)
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p))
               : "memory");
}

// D += A (16x16 f16, row) * B (16x8 f16, col), fp32 accumulate. Lane (g, t): a0 = A[g][2t,2t+1],
// a1 = A[g+8][2t,2t+1], a2 = A[g][2t+8,2t+9], a3 = A[g+8][2t+8,2t+9]; b0 = B[2t,2t+1][g], b1 = B[2t+8,2t+9][g];
// d = D[g][2t,2t+1], D[g+8][2t,2t+1].
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D += A (16x64 e2m1, row) * B (64x8 e2m1, col), one ue8m0 scale per 32 of K (as attn's indexer).
__device__ __forceinline__ void mma_fp4(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// two e4m3 (low byte first) -> f16x2, exact
__device__ __forceinline__ uint32_t e4m3x2_h2(uint32_t v) {
  uint32_t r;
  asm("{.reg .b16 lo; cvt.u16.u32 lo, %1; cvt.rn.f16x2.e4m3x2 %0, lo;}" : "=r"(r) : "r"(v));
  return r;
}
// two e2m1 (one byte, low nibble first) -> f16x2, exact
__device__ __forceinline__ uint32_t e2m1x2_h2(uint32_t v) {
  uint32_t r;
  asm("{.reg .b8 lo; cvt.u8.u32 lo, %1; cvt.rn.f16x2.e2m1x2 %0, lo;}" : "=r"(r) : "r"(v));
  return r;
}
// ue8m0 byte -> 2^(e-127) as f16 bits (exact down to 2^-24, else 0; e <= 142)
__device__ __forceinline__ uint32_t ue8m0_h(uint32_t e) {
  int k = (int)e - 127;
  return k >= -14 ? (uint32_t)(k + 15) << 10 : k >= -24 ? 1u << (k + 24) : 0u;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 r = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

}  // namespace pa
}  // namespace dsk
