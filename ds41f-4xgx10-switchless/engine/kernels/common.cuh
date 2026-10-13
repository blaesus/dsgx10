// Device helpers shared by the decode kernels: block-scaled MMA wrappers, format conversions and
// the reference's activation quantisation. GB10 only (compute_121a / sm_121a).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "kernels/dsk.h"

namespace dsk {

// D += A(16x32, e4m3) * B(32x8, e4m3), one ue8m0 scale per row of A and per column of B.
// Lane (g = lane>>2, t = lane&3): a0 = A[g][4t..], a1 = A[g+8][4t..], a2 = A[g][16+4t..],
// a3 = A[g+8][16+4t..]; b0 = B[col g][4t..], b1 = B[col g][16+4t..]; d = C[g][2t,2t+1], C[g+8][2t,2t+1].
// sfa is the scale of A row g + 8*(lane&1), sfb that of B column g (selectors 0).
__device__ __forceinline__ void mma_e4m3_e4m3(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// Same with B in e2m1 (one value per byte, in bits 5:2).
__device__ __forceinline__ void mma_e4m3_e2m1(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// D += A(16x16, bf16) * B(16x8, bf16). a0 = A[g][2t,2t+1], a1 = A[g+8][..], a2 = A[g][2t+8,2t+9],
// a3 = A[g+8][..]; b0 = B[col g][2t,2t+1], b1 = B[col g][2t+8,2t+9].
__device__ __forceinline__ void mma_bf16(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Four packed e2m1 nibbles (16 bits, low nibble first) -> four bytes with the value in bits 5:2.
__device__ __forceinline__ uint32_t e2m1x4_expand(uint32_t v16) {
  uint32_t p = __byte_perm(v16, 0, 0x1100);  // bytes: b0 b0 b1 b1
  return ((p << 2) & 0x003C003Cu) | ((p >> 2) & 0x3C003C00u);
}

// ue8m0 byte -> 2^(e-127) as fp32 (e >= 1).
__device__ __forceinline__ float ue8m0_to_f32(uint32_t e) { return __uint_as_float(e << 23); }

// The reference's per-32 activation scale: 2^ceil(log2(amax / 448)) with amax clamped at 1e-4,
// computed from fp32 bits like the tilelang kernel. Returns the ue8m0 byte.
__device__ __forceinline__ uint32_t act_scale_byte(float amax) {
  float v = fmaxf(amax, 1e-4f) * (1.0f / 448.0f);
  uint32_t b = __float_as_uint(v);
  return ((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0);
}

// x / 2^(e-127) rounded to e4m3 (values already within +-448 after the scale, but clamp anyway).
__device__ __forceinline__ uint32_t quant_e4m3(float x, uint32_t e) {
  float inv = __uint_as_float((254u - e) << 23);  // 2^(127-e), exact
  float q = fminf(fmaxf(x * inv, -448.0f), 448.0f);
  return (uint32_t)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Streaming loads that skip L1 allocation: weights are read once per step.
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
// ldg_stream32 of a when flags & mask, else of b: one load issues, into one register, both addresses base +
// offset, the predicate made inside (a select of two pointers, or a flag bit per load held ahead, costs
// registers the MoE kernels do not have). The mov first: two predicated writes alone would keep the
// register's old value alive for ptxas.
__device__ __forceinline__ uint32_t ldg_stream32_sel(uint32_t flags, uint32_t mask, const void* a, const void* b) {
  uint32_t r;
  asm volatile(
      "{\n .reg .pred q;\n .reg .b32 m;\n and.b32 m, %1, %2;\n setp.ne.b32 q, m, 0;\n mov.u32 %0, 0;\n"
      " @q ld.global.nc.L1::no_allocate.u32 %0, [%3];\n @!q ld.global.nc.L1::no_allocate.u32 %0, [%4];\n}"
      : "=r"(r)
      : "r"(flags), "r"(mask), "l"(a), "l"(b));
  return r;
}
__device__ __forceinline__ uint4 ldg_stream128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

// cp.async: global -> shared memory without holding registers (the persistent GEMVs' rings). 16 bytes
// bypass L1 (.cg); 4 bytes (scales) go through it. A thread's copies are visible to itself after the
// wait for their group, to the rest of its warp after a __syncwarp as well.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem)
               : "memory");
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// Programmatic dependent launch: let the next kernel in the stream start launching, and wait for
// the previous one (its memory included) before touching anything it produced. Both are no-ops for
// a kernel launched without the PDL attribute.
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Input and outputs of a GEMV's RMSNorm prologue (the NORM kernels).
struct NormIn {
  const __nv_bfloat16* x;  // [M][K]
  const __nv_bfloat16* w;  // [K]
  float eps;
  __nv_bfloat16* y;        // optional global copies, written by CTA 0: normed bf16 and its act_quant
  uint8_t *q, *qs;
};

// RMSNorm of M rows of K (a multiple of 256) by all threads of the CTA (blockDim a multiple of 32; at most
// 512 when RB = 16): y = bf16(w * (x * rsqrt(mean(x^2) + eps))) in fp32 math, as model.py's RMSNorm. M <= RB
// (8 or 16): rows in groups of 8, each group's loads in flight together (a row's sums are the same for
// either RB). Thread j owns the 8-element chunks j, j + blockDim, ...; four neighbouring threads share a
// 32-block for act_quant. rmsnorm_scale: only the rows' 1/rms, into red[256 + m] (visible to the CTA on
// return). rmsnorm_rows: then the outputs (null = skip; shared or global): yb [m*ldy + k] bf16; q [m*ldq + k]
// and qs [m*K/32 + k/32] = act_quant(y); q2/qs2 a second copy of those with ld K. red: >= 256 + RB floats of
// shared memory.
template <int RB = kLowRows>
__device__ __forceinline__ void rmsnorm_scale(const __nv_bfloat16* x, int ldx, int M, int K, float eps, float* red) {
  static_assert(RB == kLowRows || RB == kMaxRows, "8 or 16 rows");
  const int nc = K / 8, tid = threadIdx.x, nt = blockDim.x, warp = tid >> 5, lane = tid & 31;
#pragma unroll
  for (int m0 = 0; m0 < RB; m0 += 8) {
    if (m0 > 0 && m0 >= M) break;
    float ss[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) ss[m] = 0.f;
    for (int c = tid; c < nc; c += nt) {
      uint4 v[8];  // the group's loads in flight together
#pragma unroll
      for (int m = 0; m < 8; ++m)
        v[m] = m0 + m < M ? *reinterpret_cast<const uint4*>(x + (size_t)(m0 + m) * ldx + 8 * c) : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const uint32_t* vv = &v[m].x;
#pragma unroll
        for (int j = 0; j < 4; ++j) ss[m] = fmaf(bf16_lo(vv[j]), bf16_lo(vv[j]), ss[m]), ss[m] = fmaf(bf16_hi(vv[j]), bf16_hi(vv[j]), ss[m]);
      }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      if (m0 + m >= M) break;
      const float s = warp_sum(ss[m]);
      if (lane == 0) red[warp * RB + m0 + m] = s;
    }
  }
  __syncthreads();
  if (tid < M) {
    float s = 0.f;
    for (int i = 0; i < (nt >> 5); ++i) s += red[i * RB + tid];
    red[256 + tid] = rsqrtf(s * (1.0f / (float)K) + eps);  // torch's mean multiplies by 1/K
  }
  __syncthreads();
}

template <int RB = kLowRows>
__device__ __forceinline__ void rmsnorm_rows(const __nv_bfloat16* x, int ldx, const __nv_bfloat16* w, int M, int K,
                                             float eps, float* red, __nv_bfloat16* yb, int ldy, uint8_t* q, int ldq,
                                             uint8_t* qs, uint8_t* q2, uint8_t* qs2) {
  rmsnorm_scale<RB>(x, ldx, M, K, eps, red);
  const int nc = K / 8, tid = threadIdx.x, nt = blockDim.x;
  for (int c = tid; c < nc; c += nt) {  // whole warps: nc is a multiple of 32
    const uint4 wv = *reinterpret_cast<const uint4*>(w + 8 * c);
    const uint32_t* wp = &wv.x;
#pragma unroll
    for (int m0 = 0; m0 < RB; m0 += 8) {
      if (m0 > 0 && m0 >= M) break;
      uint4 v[8];
#pragma unroll
      for (int m = 0; m < 8; ++m)
        v[m] = m0 + m < M ? *reinterpret_cast<const uint4*>(x + (size_t)(m0 + m) * ldx + 8 * c) : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        if (m0 + m >= M) break;
        const int row = m0 + m;
        const float r = red[256 + row];
        const uint32_t* vv = &v[m].x;
        float y[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          y[2 * j] = __bfloat162float(__float2bfloat16_rn(bf16_lo(wp[j]) * (bf16_lo(vv[j]) * r)));
          y[2 * j + 1] = __bfloat162float(__float2bfloat16_rn(bf16_hi(wp[j]) * (bf16_hi(vv[j]) * r)));
        }
        if (yb) {
          uint4 o;
          o.x = pack_bf16x2(y[0], y[1]), o.y = pack_bf16x2(y[2], y[3]), o.z = pack_bf16x2(y[4], y[5]), o.w = pack_bf16x2(y[6], y[7]);
          *reinterpret_cast<uint4*>(yb + (size_t)row * ldy + 8 * c) = o;
        }
        if (q || q2) {
          float amax = 0.f;
#pragma unroll
          for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(y[j]));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
          const uint32_t e = act_scale_byte(amax);
          uint2 qv = make_uint2(0, 0);
#pragma unroll
          for (int j = 0; j < 8; ++j) (j < 4 ? qv.x : qv.y) |= quant_e4m3(y[j], e) << (8 * (j & 3));
          if (q) *reinterpret_cast<uint2*>(q + (size_t)row * ldq + 8 * c) = qv;
          if (q2) *reinterpret_cast<uint2*>(q2 + (size_t)row * K + 8 * c) = qv;
          if ((tid & 3) == 0) {
            if (qs) qs[(size_t)row * (K / 32) + c / 4] = (uint8_t)e;
            if (qs2) qs2[(size_t)row * (K / 32) + c / 4] = (uint8_t)e;
          }
        }
      }
    }
  }
}

// Eight values of a row normalised as rmsnorm_rows makes them, from x, w and the row's 1/rms (the bf16
// GEMVs at 9..16 rows normalise their input as they read it).
__device__ __forceinline__ uint4 rmsnorm8(const uint4& xv, const uint4& wv, float r) {
  const uint32_t *vv = &xv.x, *wp = &wv.x;
  uint32_t o[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) o[j] = pack_bf16x2(bf16_lo(wp[j]) * (bf16_lo(vv[j]) * r), bf16_hi(wp[j]) * (bf16_hi(vv[j]) * r));
  return make_uint4(o[0], o[1], o[2], o[3]);
}

// The router's selection for one row, by one warp: scores sqrt(softplus(l)) (torch's softplus:
// l > 20 ? l : log1p(exp(l))), top-6 of scores + bias in descending order (ties: lower id), weights
// = scores / (sum + 1e-20) * route_scale. Results in lanes 0..5 (id, weight); E <= 512. The logits
// come from an earlier kernel and are read through L2. Every load goes out before any math: one
// round trip, not one per group of 32 experts (that was 10 us of moe_up's start at one row).
__device__ __forceinline__ void router_row(const float* logits, const float* bias, int E, float route_scale, int& id_out,
                                           float& w_out) {
  constexpr int kPer = 16;
  const int lane = threadIdx.x & 31;
  float sc[kPer], key[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int e = lane + 32 * j;
    sc[j] = e < E ? __ldcg(logits + e) : 0.f;
    key[j] = e < E ? __ldg(bias + e) : -INFINITY;
  }
#pragma unroll
  for (int j = 0; j < kPer; ++j)
    if (lane + 32 * j < E) {
      const float l = sc[j];
      sc[j] = sqrtf(l > 20.f ? l : log1pf(expf(l)));
      key[j] = sc[j] + key[j];
    }
  float picked[kTopK];
  int ids[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float bv = -INFINITY;
    int bj = 0;
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (key[j] > bv) bv = key[j], bj = j;
    int be = bv == -INFINITY ? 1 << 30 : lane + 32 * bj;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ov > bv || (ov == bv && oe < be)) bv = ov, be = oe;
    }
    ids[k] = be;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (lane + 32 * j == be) s = sc[j], key[j] = -INFINITY;
    picked[k] = __shfl_sync(0xffffffffu, s, be & 31);
  }
  // torch's order for this 6-value sum (4 threads at stride 4, then a shuffle tree with offsets
  // 2, 1), found with tests/probe_router_sum.py: it reproduces all reference weights
  static_assert(kTopK == 6, "summation order is written out for top-6");
  const float sum = (((picked[0] + picked[4]) + picked[2]) + ((picked[1] + picked[5]) + picked[3])) + 1e-20f;
  id_out = 0, w_out = 0.f;
#pragma unroll
  for (int k = 0; k < kTopK; ++k)
    if (lane == k) id_out = ids[k], w_out = picked[k] / sum * route_scale;
}

}  // namespace dsk
