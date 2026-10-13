// Prefill GEMMs (M5): chunks of M = 128..8192 rows at per-rank TP=4 shapes, on tensor cores, reading
// the decode weight layouts unchanged (L1 tile_fp8, L2 = bf16 rows through tile_fp8, L3 tile_fp4):
//  - gemm_fp8_p: FP8 x FP8 block-scaled GEMM (mma.sync kind::mxf8f6f4.block_scale m16n8k32), a
//    multistage cp.async pipeline into XOR-swizzled shared memory, ldmatrix fragments; epilogues:
//    bf16 out, or SwiGLU (gate <= 10, up within +-10, silu in fp32) + bf16 + act_quant;
//  - gemm16_p: bf16 activations x bf16 weights (gate, fp32 logits) or x FP8 weights widened exactly
//    to bf16 in registers (wo_a, block-diagonal), m16n8k16 bf16 MMA, fp32 accumulation;
//  - moe_prefill: the routed experts as a grouped GEMM. Rows are bucketed by expert (count, scan,
//    scatter), 64-row tiles of one expert, experts by descending count. Up: gathered e4m3 rows x
//    MXFP4 w1/w3 streamed from global, SwiGLU, routing weight, bf16, act_quant fused. Down: x w2,
//    each expert's output rounded to bf16. A combine sums each row's experts in ascending id order
//    in fp32 and adds the shared expert (dense FP8 GEMMs) last, as the reference does. The routed
//    scales from their 1-bit code when the weights carry it (dsk.h, ScaleCode), as in the decode.
#include <algorithm>

#include "comm/ring_dev.cuh"
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/dsk.h"

namespace dsk {

namespace {

constexpr int kDimP = 5120, kInterP = 576;
constexpr int kKB1P = kDimP / 32, kKB2P = kInterP / 32;  // 160, 18 scale blocks per row
constexpr int kP1P = kDimP / 64, kP2P = kInterP / 64;    // 80, 9 steps of 64
constexpr size_t kW13P = (size_t)kInterP * kDimP / 2, kS13P = (size_t)kInterP * kKB1P;
constexpr size_t kW2P = (size_t)kDimP * kInterP / 2, kS2P = (size_t)kDimP * kKB2P;
constexpr int kTileRows = 64;  // rows (token, expert pairs) per grouped-GEMM tile
// the experts' scale code (scale_code): 16-row tiles a matrix, words of a tile's code block
constexpr int kT13P = kInterP / 16, kT2P = kDimP / 16, kB13P = code_block_words(kP1P), kB2P = code_block_words(kP2P);

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// 16 / 8 bytes global -> shared, zero-filled when !pred (the source is then not read).
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 8 : 0) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Fetch the 128-byte line holding p into L2 (no registers held).
__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// L2 eviction priority per access (MoE): 0 normal (the default), 1 evict_first for weights that
// stream once, 2 evict_last for data that later CTAs of the same launch read again.
__device__ __forceinline__ uint64_t l2_policy(int pri) {
  uint64_t p;
  if (pri == 1)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else if (pri == 2)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  else
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// With H the access carries the policy; without, it is the plain instruction (no cache-hint operand).
template <bool H>
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred, uint64_t pol) {
  if constexpr (H)
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(dst), "l"(src),
                 "r"(pred ? 16 : 0), "l"(pol)
                 : "memory");
  else
    cp_async16(dst, src, pred);
}
template <bool H>
__device__ __forceinline__ void cp_async8(uint32_t dst, const void* src, bool pred, uint64_t pol) {
  if constexpr (H)
    asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 8, %2, %3;\n" ::"r"(dst), "l"(src),
                 "r"(pred ? 8 : 0), "l"(pol)
                 : "memory");
  else
    cp_async8(dst, src, pred);
}
template <bool H>
__device__ __forceinline__ uint4 ldg128(const void* p, uint64_t pol) {
  if constexpr (!H) return ldg_stream128(p);
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p), "l"(pol));
  return r;
}
template <bool H>
__device__ __forceinline__ uint32_t ldg32(const void* p, uint64_t pol) {
  if constexpr (!H) return ldg_stream32(p);
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(pol));
  return r;
}
// ldg32 of a when flags & mask, else of b (common.cuh's ldg_stream32_sel), with the policy.
template <bool H>
__device__ __forceinline__ uint32_t ldg32_sel(uint32_t flags, uint32_t mask, const void* a, const void* b, uint64_t pol) {
  if constexpr (!H) return ldg_stream32_sel(flags, mask, a, b);
  uint32_t r;
  asm volatile(
      "{\n .reg .pred q;\n .reg .b32 m;\n and.b32 m, %1, %2;\n setp.ne.b32 q, m, 0;\n mov.u32 %0, 0;\n"
      " @q ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%3], %5;\n"
      " @!q ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%4], %5;\n}"
      : "=r"(r)
      : "r"(flags), "r"(mask), "l"(a), "l"(b), "l"(pol));
  return r;
}
template <bool H>
__device__ __forceinline__ void stg128(void* p, uint4 v, uint64_t pol) {
  if constexpr (H)
    asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w), "l"(pol)
                 : "memory");
  else
    *reinterpret_cast<uint4*>(p) = v;
}
template <bool H>
__device__ __forceinline__ void stg_f8(float* p, const float (&s)[8], uint64_t pol) {
  stg128<H>(p, make_uint4(__float_as_uint(s[0]), __float_as_uint(s[1]), __float_as_uint(s[2]), __float_as_uint(s[3])),
            pol);
  stg128<H>(p + 4,
            make_uint4(__float_as_uint(s[4]), __float_as_uint(s[5]), __float_as_uint(s[6]), __float_as_uint(s[7])), pol);
}

__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ float bf2f(__nv_bfloat16 v) { return __bfloat162float(v); }

// 8 e4m3 bytes (k = 0..7) times 2^(s-127) -> 4 bf16x2 words (k pairs 01, 23, 45, 67), exact for s <= 134.
__device__ __forceinline__ void widen8(uint2 u, uint32_t s, uint32_t (&o)[4]) {
  const __nv_bfloat162 f = __bfloat162bfloat162(__ushort_as_bfloat16((uint16_t)((s + 120u) << 7)));
  const uint32_t v[2] = {u.x, u.y};
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    uint32_t lo = __byte_perm(v[j], 0, 0x4140), hi = __byte_perm(v[j], 0, 0x4342);
    lo = ((lo << 8) & 0x80008000u) | ((lo << 4) & 0x07F007F0u);
    hi = ((hi << 8) & 0x80008000u) | ((hi << 4) & 0x07F007F0u);
    __nv_bfloat162 a = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&lo), f);
    __nv_bfloat162 b = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&hi), f);
    o[2 * j] = *reinterpret_cast<uint32_t*>(&a);
    o[2 * j + 1] = *reinterpret_cast<uint32_t*>(&b);
  }
}

// The reference's SwiGLU on bf16 gate/up values: up clamped to +-10, gate to <= 10, silu in fp32.
__device__ __forceinline__ float swiglu(float gt, float up) {
  up = fminf(fmaxf(up, -10.f), 10.f);
  gt = fminf(gt, 10.f);
  return gt / (1.0f + expf(-gt)) * up;
}

// act_quant of 8 consecutive values per lane, 4 lanes per 32-block (lanes 4j..4j+3, all 32 lanes of
// the warp present): the e4m3 bytes, and the block's ue8m0 byte in e.
__device__ __forceinline__ uint2 quant8(const float (&v)[8], uint32_t& e) {
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(v[i]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
  e = act_scale_byte(m);
  uint32_t b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = quant_e4m3(v[i], e);
  return make_uint2(b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24, b[4] | b[5] << 8 | b[6] << 16 | b[7] << 24);
}

__device__ __forceinline__ void bf16x8(const uint4& u, float (&v)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) v[2 * j] = bf16_lo(w[j]), v[2 * j + 1] = bf16_hi(w[j]);
}

template <class K>
void set_smem(K kernel, size_t bytes) {
  if (bytes > 48 * 1024) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}

// =============================================================================================
// FP8 x FP8 GEMM. CTA tile BM x BN, K in stages of 64 (one L1 block). Shared memory per stage: A
// [BM][64] and B [BN][64], 16-byte chunk c of row r stored at chunk c ^ ((r >> 1) & 3), which keeps
// both the cp.async writes and the ldmatrix reads free of bank conflicts.
enum { kEpiBf16 = 0, kEpiSwiglu = 1 };

struct GemmArgs {
  const uint8_t *xq, *xs;  // e4m3 [M][K], ue8m0 [M][K/32]
  int M, K;
  int nseg;
  Fp8Seg seg[kMaxSegs];     // SwiGLU: seg[0] = w1, seg[1] = w3
  int tile0[kMaxSegs + 1];  // first n-tile of each segment
  uint8_t *hq, *hs;         // SwiGLU out: e4m3 [M][n_inter], ue8m0 [M][n_inter / 32]
  int n_inter;
};

template <int BM, int BN, int WM, int WN, int STAGES, int EPI>
__global__ void __launch_bounds__(BM / WM * (BN / WN) * 32) k_gemm_fp8(GemmArgs a) {
  constexpr int NWM = BM / WM, NWN = BN / WN, NTH = NWM * NWN * 32;
  constexpr int MI = WM / 16, NI = WN / 8;
  constexpr int SA = BM * 64, SB = BN * 64, STG = SA + SB;
  constexpr int ACH = BM * 4 / NTH, BCH = BN * 4 / NTH;
  static_assert(NI % 2 == 0 && ACH * NTH == BM * 4 && BCH * NTH == BN * 4, "tile shape");
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp % NWM, wn = warp / NWM;
  const int M = a.M, K = a.K, KT = K / 64, KB = K / 32;
  // n-tiles along x: the CTAs that read the same activation rows run together
  const int m0 = blockIdx.y * BM, nt = blockIdx.x;
  int si = 0;
  if (EPI == kEpiBf16)
    while (si + 1 < a.nseg && nt >= a.tile0[si + 1]) ++si;
  const int n_base = EPI == kEpiSwiglu ? nt * (BN / 2) : (nt - a.tile0[si]) * BN;
  // weight row of tile row r (SwiGLU: rows [0, BN/2) of w1, then the same rows of w3)
  auto wrow = [&](int r, const Fp8Seg*& sg) {
    if (EPI == kEpiSwiglu) {
      sg = &a.seg[r >= BN / 2];
      return n_base + (r >= BN / 2 ? r - BN / 2 : r);
    }
    sg = &a.seg[si];
    return n_base + r;
  };

  const uint8_t* asrc[ACH];
  uint32_t adst[ACH];
  bool aok[ACH];
#pragma unroll
  for (int j = 0; j < ACH; ++j) {
    const int i = tid + j * NTH, r = i >> 2, c = i & 3, row = m0 + r;
    aok[j] = row < M;
    asrc[j] = a.xq + (size_t)(aok[j] ? row : 0) * K + 16 * c;
    adst[j] = r * 64 + 16 * (c ^ ((r >> 1) & 3));
  }
  const uint8_t* bsrc[BCH];
  uint32_t bdst[BCH];
#pragma unroll
  for (int j = 0; j < BCH; ++j) {
    const int i = tid + j * NTH, r = i >> 2, c = i & 3;
    const Fp8Seg* sg;
    const int n = wrow(r, sg);
    bsrc[j] = sg->W + (size_t)(n >> 3) * KT * 512 + (n & 7) * 64 + 16 * c;
    bdst[j] = SA + r * 64 + 16 * (c ^ ((r >> 1) & 3));
  }
  // scale bytes of both 32-blocks of a stage in one 16-bit load (KB is even), a stage ahead
  const uint16_t* sfa_p[MI];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
    const int row = min(m0 + wm * WM + mi * 16 + g + 8 * (lane & 1), M - 1);
    sfa_p[mi] = reinterpret_cast<const uint16_t*>(a.xs + (size_t)row * KB);
  }
  const uint16_t* sfb_p[NI];
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) {
    const Fp8Seg* sg;
    const int n = wrow(wn * WN + ni * 8, sg);
    sfb_p[ni] = reinterpret_cast<const uint16_t*>(sg->S + (size_t)(n >> 5) * KB);
  }
  uint32_t sfa_n[MI], sfb_n[NI];
  auto load_sf = [&](int kt) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) sfa_n[mi] = __ldg(sfa_p[mi] + kt);
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) sfb_n[ni] = __ldg(sfb_p[ni] + kt);
  };
  const uint32_t s_base = smem_addr(smem);
  auto load_stage = [&](int kt, int buf) {
    const uint32_t s0 = s_base + buf * STG;
#pragma unroll
    for (int j = 0; j < ACH; ++j) cp_async16(s0 + adst[j], asrc[j] + kt * 64, aok[j]);
#pragma unroll
    for (int j = 0; j < BCH; ++j) cp_async16(s0 + bdst[j], bsrc[j] + (size_t)kt * 512, true);
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

  pdl_launch_dependents();
  pdl_wait();
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }
  load_sf(0);
  for (int kt = 0; kt < KT; ++kt) {
    uint32_t sfa[2][MI], sfb[2][NI];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) sfa[0][mi] = sfa_n[mi], sfa[1][mi] = sfa_n[mi] >> 8;
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) sfb[0][ni] = sfb_n[ni], sfb[1][ni] = sfb_n[ni] >> 8;
    if (kt + 1 < KT) load_sf(kt + 1);
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = kt + STAGES - 1;
      if (nk < KT) load_stage(nk, nk % STAGES);
      cp_async_commit();
    }
    const uint32_t sa0 = s_base + (kt % STAGES) * STG, sb0 = sa0 + SA;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      uint32_t af[MI][4], bfr[NI][2];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r = wm * WM + mi * 16 + (lane & 7) + 8 * ((lane >> 3) & 1), c = 2 * kb + (lane >> 4);
        ldsm4(sa0 + r * 64 + 16 * (c ^ ((r >> 1) & 3)), af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
      }
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) {
        const int r = wn * WN + nj * 16 + (lane & 7) + 8 * (lane >> 4), c = 2 * kb + ((lane >> 3) & 1);
        ldsm4(sb0 + r * 64 + 16 * (c ^ ((r >> 1) & 3)), bfr[2 * nj][0], bfr[2 * nj][1], bfr[2 * nj + 1][0],
              bfr[2 * nj + 1][1]);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
          mma_e4m3_e4m3(acc[mi][ni], af[mi][0], af[mi][1], af[mi][2], af[mi][3], bfr[ni][0], bfr[ni][1], sfa[kb][mi],
                        sfb[kb][ni]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // bf16 C tile through shared memory: coalesced stores, and the SwiGLU pairs gate with up
  constexpr int LDC = BN + 8;
  __nv_bfloat16* sC = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int r = wm * WM + mi * 16 + g, c = wn * WN + ni * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(&sC[r * LDC + c]) = pack_bf16x2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<uint32_t*>(&sC[(r + 8) * LDC + c]) = pack_bf16x2(acc[mi][ni][2], acc[mi][ni][3]);
    }
  __syncthreads();
  if constexpr (EPI == kEpiBf16) {
    const Fp8Seg& sg = a.seg[si];
    for (int i = tid; i < BM * (BN / 8); i += NTH) {
      const int r = i / (BN / 8), c8 = i % (BN / 8), row = m0 + r;
      if (row < M)
        *reinterpret_cast<uint4*>(sg.y + (size_t)row * (sg.ld ? sg.ld : sg.N) + n_base + 8 * c8) =
            *reinterpret_cast<const uint4*>(&sC[r * LDC + 8 * c8]);
    }
  } else {
    constexpr int NB = BN / 64;  // 32-wide blocks of h per row; 8 blocks per warp and pass
    static_assert((BM * NB) % (NTH / 4) == 0, "epilogue passes");
    for (int u0 = 0; u0 < BM * NB; u0 += NTH / 4) {
      const int u = u0 + tid / 4, r = u / NB, jb = u % NB, row = m0 + r, c = 32 * jb + 8 * (lane & 3);
      float gt[8], up[8], h[8];
      bf16x8(*reinterpret_cast<const uint4*>(&sC[r * LDC + c]), gt);
      bf16x8(*reinterpret_cast<const uint4*>(&sC[r * LDC + BN / 2 + c]), up);
#pragma unroll
      for (int i = 0; i < 8; ++i) h[i] = __bfloat162float(__float2bfloat16_rn(swiglu(gt[i], up[i])));
      uint32_t e;
      const uint2 qv = quant8(h, e);
      if (row < M) {
        *reinterpret_cast<uint2*>(a.hq + (size_t)row * a.n_inter + n_base + c) = qv;
        if ((lane & 3) == 0) a.hs[(size_t)row * (a.n_inter / 32) + n_base / 32 + jb] = (uint8_t)e;
      }
    }
  }
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI>
void launch_gemm_fp8(const GemmArgs& a, int ntiles, cudaStream_t st) {
  constexpr int NTH = BM / WM * (BN / WN) * 32;
  const size_t smem = std::max<size_t>((size_t)STAGES * (BM + BN) * 64, (size_t)BM * (BN + 8) * 2);
  static bool attr = false;
  if (!attr) set_smem(k_gemm_fp8<BM, BN, WM, WN, STAGES, EPI>, smem), attr = true;
  launch_pdl(k_gemm_fp8<BM, BN, WM, WN, STAGES, EPI>, dim3(ntiles, (a.M + BM - 1) / BM), dim3(NTH), smem, st, true, a);
}

// =============================================================================================
// bf16 GEMM: A bf16 [M][lda] (block-diagonal: the CTA's rows' group selects a K-wide slice), B bf16
// (L2 layout) or FP8 widened to bf16 (L1 layout, [N/32][K/32] scales). Stages of 64 k. Each lane
// owns k = 8t..8t+7 of every 32 (lds.128 of A, of bf16 B, lds.64 of FP8 B); the k order inside the
// two MMAs that cover them is permuted alike for A and B.
struct G16Args {
  const __nv_bfloat16* x;
  int lda, M, K;  // K: per group
  const uint8_t *W, *S;
  int N, group_rows;
  float* yf;          // fp32 out [M][N], or
  __nv_bfloat16* yb;  // bf16 out [M][N] and/or
  uint8_t *q, *qs;    // act_quant(bf16 out)
  // gemm_bf16_rows_p: row-major bf16 weights, rows n < N0 of W (row stride ldw elements) into yf [M][N0],
  // the rest from W1 into yf1 [M][N - N0]; rnd rounds the fp32 outputs to bf16 values
  int ldw = 0, N0 = 0;
  const uint8_t* W1 = nullptr;
  float* yf1 = nullptr;
  bool rnd = false;
};

template <int BM, int BN, int WM, int WN, int STAGES, bool WIDEN>
__global__ void __launch_bounds__(BM / WM * (BN / WN) * 32) k_gemm16(G16Args a) {
  constexpr int NWM = BM / WM, NWN = BN / WN, NTH = NWM * NWN * 32;
  constexpr int MI = WM / 16, NI = WN / 8;
  constexpr int RB = WIDEN ? 64 : 128;  // B bytes per row and stage
  constexpr bool PART = !WIDEN;
  constexpr int SA = BM * 128, SB = BN * RB, STG = SA + SB;
  constexpr int ACH = BM * 8 / NTH, BCH = BN * (RB / 16) / NTH;
  static_assert(ACH * NTH == BM * 8 && BCH * NTH == BN * (RB / 16), "tile shape");
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp % NWM, wn = warp / NWM;
  const int M = a.M, K = a.K, KT = K / 64;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;  // n-tiles along x, as in k_gemm_fp8
  const int xoff = (n0 / a.group_rows) * K;

  const uint8_t* asrc[ACH];
  uint32_t adst[ACH];
  bool aok[ACH];
#pragma unroll
  for (int j = 0; j < ACH; ++j) {
    const int i = tid + j * NTH, r = i >> 3, c = i & 7, row = m0 + r;
    aok[j] = row < M;
    asrc[j] = reinterpret_cast<const uint8_t*>(a.x + (size_t)(aok[j] ? row : 0) * a.lda + xoff + 8 * c);
    adst[j] = r * 128 + 16 * (c ^ ((r & 1) << 2));
  }
  const uint8_t* bsrc[BCH];
  uint32_t bdst[BCH];
#pragma unroll
  for (int j = 0; j < BCH; ++j) {
    const int i = tid + j * NTH;
    if constexpr (WIDEN) {
      const int r = i >> 2, c = i & 3, n = n0 + r;
      bsrc[j] = a.W + (size_t)(n >> 3) * KT * 512 + (n & 7) * 64 + 16 * c;
      bdst[j] = SA + r * 64 + 32 * ((c >> 1) ^ ((r >> 1) & 1)) + 16 * (c & 1);
    } else {
      const int r = i >> 3, c = i & 7, n = n0 + r;
      if (a.ldw)  // row-major: the stage's 128 bytes of row n are contiguous
        bsrc[j] = (n0 < a.N0 ? a.W + (size_t)n * a.ldw * 2 : a.W1 + (size_t)(n - a.N0) * a.ldw * 2) + 16 * c;
      else
        bsrc[j] = a.W + (size_t)(n >> 3) * (2 * KT) * 512 + (n & 7) * 64 + (c >> 2) * 512 + 16 * (c & 3);
      bdst[j] = SA + r * 128 + 16 * (c ^ ((r & 1) << 2));
    }
  }
  const uint8_t* sfb_p[NI];
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) sfb_p[ni] = WIDEN ? a.S + (size_t)((n0 + wn * WN + ni * 8) >> 5) * (K / 32) : nullptr;
  uint32_t sb_n[NI];  // both scale bytes of the next stage (K / 32 is even)
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) sb_n[ni] = WIDEN ? __ldg(reinterpret_cast<const uint16_t*>(sfb_p[ni])) : 0;
  const uint32_t s_base = smem_addr(smem);
  const size_t bstep = WIDEN ? 512 : a.ldw ? 128 : 1024;  // B bytes per stage along a row's source
  auto load_stage = [&](int kt, int buf) {
    const uint32_t s0 = s_base + buf * STG;
#pragma unroll
    for (int j = 0; j < ACH; ++j) cp_async16(s0 + adst[j], asrc[j] + kt * 128, aok[j]);
#pragma unroll
    for (int j = 0; j < BCH; ++j) cp_async16(s0 + bdst[j], bsrc[j] + (size_t)kt * bstep, true);
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f;

  pdl_launch_dependents();
  pdl_wait();
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = kt + STAGES - 1;
      if (nk < KT) load_stage(nk, nk % STAGES);
      cp_async_commit();
    }
    uint32_t sb[2][NI];
    if constexpr (WIDEN) {
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) sb[0][ni] = sb_n[ni] & 0xFF, sb[1][ni] = sb_n[ni] >> 8;
      if (kt + 1 < KT)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) sb_n[ni] = __ldg(reinterpret_cast<const uint16_t*>(sfb_p[ni]) + kt + 1);
    }
    const uint8_t* sa0 = smem + (kt % STAGES) * STG;
    const uint8_t* sb0 = sa0 + SA;
    // fp32 outputs (the gate's logits): a stage's 4 MMAs per output sum into a fresh accumulator
    // that is then added with an fp32 add. One chain of tensor-core accumulations over K = 5120
    // loses ~1e-5 relative (torch: ~2e-7); per-stage partials keep it at fp32 rounding level.
    // bf16 outputs (wo_a) do not need it.
    float part[MI][NI][4];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) part[mi][ni][0] = part[mi][ni][1] = part[mi][ni][2] = part[mi][ni][3] = 0.f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      uint4 xa[MI], xb[MI];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int r = wm * WM + mi * 16 + g;
        const int off = 16 * ((4 * q + t) ^ ((r & 1) << 2));
        xa[mi] = *reinterpret_cast<const uint4*>(sa0 + r * 128 + off);
        xb[mi] = *reinterpret_cast<const uint4*>(sa0 + (r + 8) * 128 + off);
      }
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int r = wn * WN + ni * 8 + g;
        uint32_t w[4];
        if constexpr (WIDEN) {
          const uint2 u = *reinterpret_cast<const uint2*>(sb0 + r * 64 + 32 * (q ^ ((r >> 1) & 1)) + 8 * t);
          widen8(u, sb[q][ni], w);
        } else {
          const uint4 v = *reinterpret_cast<const uint4*>(sb0 + r * 128 + 16 * ((4 * q + t) ^ ((r & 1) << 2)));
          w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w;
        }
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
          float* d = PART ? part[mi][ni] : acc[mi][ni];
          mma_bf16(d, xa[mi].x, xb[mi].x, xa[mi].y, xb[mi].y, w[0], w[1]);
          mma_bf16(d, xa[mi].z, xb[mi].z, xa[mi].w, xb[mi].w, w[2], w[3]);
        }
      }
    }
    if constexpr (PART)
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni)
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[mi][ni][j] += part[mi][ni][j];
  }
  cp_async_wait<0>();

  if (a.yf) {
    float* yf = a.yf;
    int ldy = a.N, cb = 0;
    if (a.ldw) {
      ldy = n0 < a.N0 ? a.N0 : a.N - a.N0;
      if (n0 >= a.N0) yf = a.yf1, cb = a.N0;
    }
    auto out = [&](float v) { return a.rnd ? __bfloat162float(__float2bfloat16_rn(v)) : v; };
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int r = m0 + wm * WM + mi * 16 + g, c = n0 + wn * WN + ni * 8 + 2 * t - cb;
        if (r < M)
          *reinterpret_cast<float2*>(yf + (size_t)r * ldy + c) = make_float2(out(acc[mi][ni][0]), out(acc[mi][ni][1]));
        if (r + 8 < M)
          *reinterpret_cast<float2*>(yf + (size_t)(r + 8) * ldy + c) =
              make_float2(out(acc[mi][ni][2]), out(acc[mi][ni][3]));
      }
    return;
  }
  __syncthreads();
  constexpr int LDC = BN + 8;
  __nv_bfloat16* sC = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int r = wm * WM + mi * 16 + g, c = wn * WN + ni * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(&sC[r * LDC + c]) = pack_bf16x2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<uint32_t*>(&sC[(r + 8) * LDC + c]) = pack_bf16x2(acc[mi][ni][2], acc[mi][ni][3]);
    }
  __syncthreads();
  if (a.yb)
    for (int i = tid; i < BM * (BN / 8); i += NTH) {
      const int r = i / (BN / 8), c8 = i % (BN / 8), row = m0 + r;
      if (row < M)
        *reinterpret_cast<uint4*>(a.yb + (size_t)row * a.N + n0 + 8 * c8) =
            *reinterpret_cast<const uint4*>(&sC[r * LDC + 8 * c8]);
    }
  if constexpr (BN >= 32) if (a.q) {  // act_quant per 32 of the bf16 values, 8 values per lane (launch_gemm16:
                                      // none below 32 columns)
    static_assert((BM * (BN / 32)) % (NTH / 4) == 0, "epilogue passes");
    for (int u0 = 0; u0 < BM * (BN / 32); u0 += NTH / 4) {
      const int u = u0 + tid / 4, r = u / (BN / 32), jb = u % (BN / 32), row = m0 + r, c = 32 * jb + 8 * (lane & 3);
      float v[8];
      bf16x8(*reinterpret_cast<const uint4*>(&sC[r * LDC + c]), v);
      uint32_t e;
      const uint2 qv = quant8(v, e);
      if (row < M) {
        *reinterpret_cast<uint2*>(a.q + (size_t)row * a.N + n0 + c) = qv;
        if ((lane & 3) == 0) a.qs[(size_t)row * (a.N / 32) + n0 / 32 + jb] = (uint8_t)e;
      }
    }
  }
}

template <int BM, int BN, int WM, int WN, int STAGES, bool WIDEN>
void launch_gemm16(const G16Args& a, cudaStream_t st) {
  constexpr int NTH = BM / WM * (BN / WN) * 32;
  if (BN < 32 && a.q) die("launch_gemm16: act_quant needs 32 columns a tile (BN=%d)", BN);
  const size_t smem =
      std::max<size_t>((size_t)STAGES * (BM * 128 + BN * (WIDEN ? 64 : 128)), (size_t)BM * (BN + 8) * 2);
  static bool attr = false;
  if (!attr) set_smem(k_gemm16<BM, BN, WM, WN, STAGES, WIDEN>, smem), attr = true;
  launch_pdl(k_gemm16<BM, BN, WM, WN, STAGES, WIDEN>, dim3(a.N / BN, (a.M + BM - 1) / BM), dim3(NTH), smem, st, true,
             a);
}

// Four bf16 partials summed in fp32 in order, rounded once (comm's sum8_bf16 and reduce16).
__global__ void k_sum4_bf16(const uint4* __restrict__ part, size_t slot16, size_t n16, uint4* __restrict__ out) {
  pdl_launch_dependents();
  pdl_wait();
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    float s[8], f[8];
    bf16x8(part[i], s);
#pragma unroll
    for (int k = 1; k < 4; ++k) {
      bf16x8(part[k * slot16 + i], f);
#pragma unroll
      for (int j = 0; j < 8; ++j) s[j] += f[j];
    }
    out[i] = make_uint4(pack_bf16x2(s[0], s[1]), pack_bf16x2(s[2], s[3]), pack_bf16x2(s[4], s[5]),
                        pack_bf16x2(s[6], s[7]));
  }
}

// =============================================================================================
// Routed experts.
struct MoePArgs {
  const uint8_t *xq, *xs;  // e4m3 [M][5120], ue8m0 [M][160]
  const int* idx;          // [M][6] expert ids
  const float* wts;        // [M][6] routing weights
  int M, E;
  MoeWeights w;
  int *cnt, *off, *cursor, *ntiles;
  int4* tiles;     // per 64-row tile: expert, first bucket position, rows, the expert's picks (k_moe_bucket)
  int* perm_tok;   // [P] token of each bucketed pair
  float* perm_w;   // [P] its routing weight
  int* pos_of;     // [M][6] bucket position of each pick
  uint8_t *hq, *hs;           // [P][576], [P][18]: the re-quantised SwiGLU output
  __nv_bfloat16* dout;        // [P][5120]: each pick's expert output, bf16
  const __nv_bfloat16* shout; // [M][5120]: the shared expert's output, bf16
  float* y;                   // [M][5120]
  int* done;                  // [40][ceil(M / 64)] picks finished per (column block, row group): fused combine
  int cb0, ncb;               // down and combine: column blocks [cb0, cb0 + ncb) of 128
  int l2;  // L2 priorities, bits: 1 up's weights evict_first for experts with one tile, 2 down's
           // likewise (and the combine's single-use traffic evict_first), 4 x (up), h and the fused
           // down's outputs evict_last;
           // 8 alone: the hinted instructions with evict_normal everywhere; 0: no hints at all
  int gcb;  // k_moe_down_g: column blocks per CTA
  int upf;  // k_moe_up_p: weights prefetched into L2 this many chunks past the registers' (0: none)
};

constexpr int kColBlocks = kDimP / 128, kGroupRows = 64;

// Where the i-th of nt tiles in descending order runs when k > 0: the order takes one tile from the
// front (the experts with most picks, whose later tiles find the weights in L2 and compute) and then k
// from the back (one-tile experts, which stream their weights), so that DRAM and the tensor cores work
// side by side.
__device__ __forceinline__ int tile_pos(int i, int nt, int k) {
  if (k <= 0) return i;
  const int f = (nt + k) / (k + 1);  // front tiles
  if (i < f) return i * (k + 1);
  const int j = nt - 1 - i;
  return j / k * (k + 1) + 1 + j % k;
}

// One CTA of 1024 threads, E <= 1024: the picks per expert (shared-memory atomics, one per group
// of lanes with the same expert), offsets by expert id, cursors reset, and the tile list: experts by
// descending count, ties by id (the longest-running tiles first), then reordered by tile_pos(ord).
__global__ void __launch_bounds__(1024) k_moe_bucket(const int* __restrict__ idx, int P, int E, int* __restrict__ cnt,
                                                     int* __restrict__ off, int* __restrict__ cursor,
                                                     int* __restrict__ ntiles, int4* __restrict__ tiles,
                                                     int* __restrict__ done, int ndone, int ord) {
  __shared__ int sc[1024], snt[1024], wsum[32], stot;
  const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
  sc[e] = 0;
  pdl_launch_dependents();
  pdl_wait();
  for (int i = e; i < ndone; i += 1024) done[i] = 0;
  __syncthreads();
  for (int i = e; i < P; i += 1024) {
    const int x = idx[i];
    const unsigned peers = __match_any_sync(__activemask(), x);
    if (lane == __ffs(peers) - 1) atomicAdd(&sc[x], __popc(peers));
  }
  __syncthreads();
  const int c = sc[e];
  // block-wide exclusive scan of v (the same pattern twice)
  auto scan = [&](int v) {
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (lane == 31) wsum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
      int w = wsum[lane], wi = w;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, wi, o);
        if (lane >= o) wi += u;
      }
      wsum[lane] = wi - w;
    }
    __syncthreads();
    const int ex = incl - v + wsum[warp];
    __syncthreads();
    return ex;
  };
  const int ex = scan(c);
  if (e < E) {
    cnt[e] = c, off[e] = ex, cursor[e] = 0;
    if (e == E - 1) off[E] = ex + c;
  }
  int rank = e;
  if (e < E) {
    rank = 0;
    for (int j = 0; j < E; ++j) {
      const int cj = sc[j];
      rank += cj > c || (cj == c && j < e);
    }
  }
  const int nt = (c + kTileRows - 1) / kTileRows;
  snt[rank] = nt;
  __syncthreads();
  const int mine = snt[e];  // tiles of the expert ranked e
  const int start = scan(mine);
  snt[e] = start;
  if (e == 1023) *ntiles = stot = start + mine;
  __syncthreads();
  if (e < E)
    for (int j = 0; j < nt; ++j)
      tiles[tile_pos(snt[rank] + j, stot, ord)] = make_int4(e, ex + j * kTileRows, min(kTileRows, c - j * kTileRows), c);
}

__global__ void k_moe_scatter(const int* __restrict__ idx, const float* __restrict__ wts, int n,
                              const int* __restrict__ off, int* __restrict__ cursor, int* __restrict__ perm_tok,
                              float* __restrict__ perm_w, int* __restrict__ pos_of) {
  pdl_launch_dependents();
  pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31;
  if (i >= n) return;
  // one atomic per group of lanes with the same expert; a pick's place inside its bucket does not
  // change any result
  const int e = idx[i];
  const unsigned peers = __match_any_sync(__activemask(), e);
  const int leader = __ffs(peers) - 1;
  int pos = 0;
  if (lane == leader) pos = atomicAdd(&cursor[e], __popc(peers));
  pos = off[e] + __shfl_sync(peers, pos, leader) + __popc(peers & ((1u << lane) - 1));
  perm_tok[pos] = i / kTopK;
  perm_w[pos] = wts[i];
  pos_of[i] = pos;
}

// Each row's six bucket positions in ascending expert id (the order the combine sums them in).
__global__ void k_moe_sortpos(const int* __restrict__ idx, const int* __restrict__ pos_of, int M,
                              int* __restrict__ pos_srt) {
  pdl_launch_dependents();
  pdl_wait();
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;
  int ex[kTopK], pos[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) ex[k] = idx[m * kTopK + k], pos[k] = pos_of[m * kTopK + k];
#pragma unroll
  for (int i = 1; i < kTopK; ++i)
#pragma unroll
    for (int j = i; j > 0; --j)
      if (ex[j] < ex[j - 1]) {
        const int te = ex[j], tp = pos[j];
        ex[j] = ex[j - 1], pos[j] = pos[j - 1], ex[j - 1] = te, pos[j - 1] = tp;
      }
#pragma unroll
  for (int k = 0; k < kTopK; ++k) pos_srt[m * kTopK + k] = pos[k];
}

// Up: CTA = (64 intermediate rows q, one 64-row tile of an expert), 4 warps; warp w streams w1 and w3
// rows q*64 + 16w .. +15 (L3 layout: each lane loads its own 16 bytes per 64 k) straight from global
// into registers, KU steps ahead, while the tile's gathered activation rows pass through shared
// memory in chunks of 64*KU k (double-buffered cp.async). Activation fragments follow L3's k order.
// CS: w1's and w3's scales from their 1-bit code (a.w.c1, c3): a step's word is its code word or, flagged
// raw, the raw one (the decode's k_moe_up), into the same MMAs in the same order. The tiles' flags are read
// first, in by the time the weights' loads go out (they wait for the tile's token ids).
template <int KU, bool H, bool CS = false>
__global__ void __launch_bounds__(128) k_moe_up_p(MoePArgs a) {
  constexpr int KC = 64 * KU, NCH = kDimP / KC;
  static_assert(NCH % 2 == 0, "chunks come in pairs");
  static_assert(32 % KU == 0, "a chunk's flags in one word");
  __shared__ __align__(16) uint8_t sA[2][kTileRows * KC];
  __shared__ __align__(16) uint8_t sS[2][kTileRows * (KC / 32)];
  __shared__ __align__(16) __nv_bfloat16 sH[kTileRows][64 + 8];
  __shared__ int stok[kTileRows];
  __shared__ float srw[kTileRows];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  pdl_launch_dependents();
  pdl_wait();
  const int4 te = a.tiles[blockIdx.y];  // (read beside the count: entries past it are stale, never used)
  if ((int)blockIdx.y >= *a.ntiles) return;
  const int e = te.x, base = te.y, rows = te.z;
  const int q = blockIdx.x, nq = q * 4 + warp;  // this warp's n16 group of w1 / w3
  const int mt = (rows + 15) / 16;              // m16 tiles with rows
  // CS: the tiles' code blocks (cbo: their word offset) and raw flags: sfl[warp][3 mat + j], steps 32j .. 32j + 31
  // of w1 (mat 0) or w3, read beside the token ids (in shared memory: in registers they spill)
  __shared__ uint32_t sfl[CS ? 4 : 1][8];
  const uint32_t cbo = CS ? ((uint32_t)e * kT13P + nq) * kB13P : 0u;
  uint32_t fword = 0;
  if (CS && lane < 6) fword = __ldg((lane < 3 ? a.w.c1.code : a.w.c3.code) + cbo + kP1P + lane % 3);
  // weights evict_first unless another tile of the expert reads them again
  const uint64_t pol_w = l2_policy((a.l2 & 1) && te.w <= kTileRows), pol_x = l2_policy(a.l2 & 4 ? 2 : 0);
  if (tid < kTileRows) {
    stok[tid] = tid < rows ? a.perm_tok[base + tid] : 0;
    srw[tid] = tid < rows ? a.perm_w[base + tid] : 0.f;
  }
  if (CS && lane < 6) sfl[CS ? warp : 0][lane] = fword;
  __syncthreads();

  const uint32_t sA0 = smem_addr(&sA[0][0]), sS0 = smem_addr(&sS[0][0]);
  auto load_a = [&](int buf, int c) {
#pragma unroll
    for (int j = 0; j < kTileRows * (KC / 16) / 128; ++j) {
      const int i = tid + 128 * j, r = i / (KC / 16), c16 = i % (KC / 16);
      cp_async16<H>(sA0 + buf * kTileRows * KC + r * KC + 32 * ((c16 >> 1) ^ (r & 3)) + 16 * (c16 & 1),
                 a.xq + (size_t)stok[r] * kDimP + c * KC + 16 * c16, r < rows, pol_x);
    }
    if (tid < kTileRows)
      cp_async8<H>(sS0 + buf * kTileRows * (KC / 32) + tid * (KC / 32),
                   a.xs + (size_t)stok[tid] * kKB1P + c * (KC / 32), tid < rows, pol_x);
    cp_async_commit();
  };
  const uint8_t* w1p = a.w.w1 + (size_t)e * kW13P + ((size_t)nq * kP1P * 32 + lane) * 16;
  const uint8_t* w3p = a.w.w3 + (size_t)e * kW13P + ((size_t)nq * kP1P * 32 + lane) * 16;
  const uint8_t* s1p = a.w.s1 + (size_t)e * kS13P + ((size_t)nq * kP1P * 8 + g) * 4;
  const uint8_t* s3p = a.w.s3 + (size_t)e * kS13P + ((size_t)nq * kP1P * 8 + g) * 4;
  // the L2 prefetch of a chunk: lanes 0-15 the 16 lines of this warp's w1 rows, 16-31 of w3's, lanes 0
  // and 1 the line of s1's and s3's
  const uint8_t* wpf = (lane < 16 ? a.w.w1 : a.w.w3) + (size_t)e * kW13P + (size_t)nq * kP1P * 512 + (lane & 15) * 128;
  const uint8_t* spf = (lane & 1 ? a.w.s3 : a.w.s1) + (size_t)e * kS13P + (size_t)nq * kP1P * 32;
  uint4 wa[KU][2], wb[KU][2];
  uint32_t sa[KU][2], sb[KU][2];
  // CS: steps p0 .. p0 + KU - 1's raw flags of w1 (mat 0) or w3
  auto chunk_flags = [&](int p0, int mat) -> uint32_t {
    return (sfl[CS ? warp : 0][3 * mat + (p0 >> 5)] >> (p0 & 31)) & ((1u << KU) - 1u);
  };
  auto load_w = [&](uint4 (&w)[KU][2], uint32_t (&s)[KU][2], int p0) {
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      w[u][0] = ldg128<H>(w1p + (size_t)(p0 + u) * 512, pol_w);
      w[u][1] = ldg128<H>(w3p + (size_t)(p0 + u) * 512, pol_w);
      if constexpr (CS) {
        s[u][0] = ldg32_sel<H>(chunk_flags(p0, 0), 1u << u, s1p + (size_t)(p0 + u) * 32, a.w.c1.code + cbo + p0 + u,
                               pol_w);
        s[u][1] = ldg32_sel<H>(chunk_flags(p0, 1), 1u << u, s3p + (size_t)(p0 + u) * 32, a.w.c3.code + cbo + p0 + u,
                               pol_w);
      } else {
        s[u][0] = ldg32<H>(s1p + (size_t)(p0 + u) * 32, pol_w);
        s[u][1] = ldg32<H>(s3p + (size_t)(p0 + u) * 32, pol_w);
      }
    }
    if (const int pf = p0 + a.upf * KU; a.upf && pf < kP1P) {
      prefetch_l2(wpf + (size_t)pf * 512);
      if (lane < 2) prefetch_l2(spf + (size_t)pf * 32);
    }
  };
  float acc[4][4][4];  // [m16 tile][w1 lo, w1 hi, w3 lo, w3 hi][c]
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
  auto compute = [&](const uint4 (&w)[KU][2], const uint32_t (&s)[KU][2], int buf, int p0) {
    const uint8_t* A = &sA[buf][0];
    const uint8_t* S = &sS[buf][0];
    uint32_t sw[KU][2];  // the scale words (CS: rebuilt from the code)
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int mat = 0; mat < 2; ++mat) {
        sw[u][mat] = s[u][mat];
        if constexpr (CS)
          if (!(chunk_flags(p0, mat) >> u & 1u))
            sw[u][mat] = ((s[u][mat] >> g) & 0x01010101u) + (mat ? a.w.c3.base4 : a.w.c1.base4);
      }
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int blk = 0; blk < 2; ++blk) {
        const int unit = 2 * u + blk;  // 32-k block within the chunk
        uint32_t af[4][4], sfa[4];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          if (mi >= mt) break;  // m16 tiles without rows are skipped (uniform per CTA)
          const int r = mi * 16 + g;
          const uint2 lo = *reinterpret_cast<const uint2*>(A + r * KC + 32 * (unit ^ (r & 3)) + 8 * t);
          const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * KC + 32 * (unit ^ (r & 3)) + 8 * t);
          af[mi][0] = lo.x, af[mi][1] = hi.x, af[mi][2] = lo.y, af[mi][3] = hi.y;
          sfa[mi] = S[(mi * 16 + g + 8 * (lane & 1)) * (KC / 32) + unit];
        }
#pragma unroll
        for (int mat = 0; mat < 2; ++mat)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t word = (&w[u][mat].x)[2 * h + blk];
            const uint32_t b0 = e2m1x4_expand(word & 0xFFFFu), b1 = e2m1x4_expand(word >> 16);
            const uint32_t sfb = (sw[u][mat] >> (8 * (2 * h + blk))) & 0xFFu;
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
              if (mi < mt)
                mma_e4m3_e2m1(acc[mi][2 * mat + h], af[mi][0], af[mi][1], af[mi][2], af[mi][3], b0, b1, sfa[mi], sfb);
          }
      }
  };

  load_a(0, 0);
  load_w(wa, sa, 0);
  for (int c = 0; c < NCH; c += 2) {
    load_a(1, c + 1);
    load_w(wb, sb, (c + 1) * KU);
    cp_async_wait<1>();
    __syncthreads();
    compute(wa, sa, 0, c * KU);
    __syncthreads();
    if (c + 2 < NCH) {
      load_a(0, c + 2);
      load_w(wa, sa, (c + 2) * KU);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    compute(wb, sb, 1, (c + 1) * KU);
    __syncthreads();
  }

  // SwiGLU (gate and up of the same columns sit in this thread), routing weight, bf16 -> sH
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int hr = 0; hr < 2; ++hr) {
        const int r = mi * 16 + g + 8 * hr, col = 16 * warp + 8 * h + 2 * t;
        const float rw = srw[r];
        float v[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float gt = __bfloat162float(__float2bfloat16_rn(acc[mi][h][2 * hr + j]));
          const float up = __bfloat162float(__float2bfloat16_rn(acc[mi][2 + h][2 * hr + j]));
          v[j] = rw * swiglu(gt, up);
        }
        *reinterpret_cast<uint32_t*>(&sH[r][col]) = pack_bf16x2(v[0], v[1]);
      }
  __syncthreads();
  for (int u = warp; u < rows * 2; u += 4) {
    const int r = u >> 1, jb = u & 1;
    const float v = bf2f(sH[r][32 * jb + lane]);
    const uint32_t ex = act_scale_byte(warp_max(fabsf(v)));
    a.hq[(size_t)(base + r) * kInterP + q * 64 + 32 * jb + lane] = (uint8_t)quant_e4m3(v, ex);
    if (lane == 0) a.hs[(size_t)(base + r) * kKB2P + q * 2 + jb] = (uint8_t)ex;
  }
}

// The fused combine for one (column block, row group) whose picks are all done: per row, the outputs
// summed in ascending expert id in fp32, then the shared expert; 16 threads per row, 8 columns each.
// The dout lines are dead afterwards and leave L2 without a write-back.
template <bool H>
__device__ void combine_group(const MoePArgs& a, int cb, int grp) {
  const int sub = threadIdx.x & 15, c0 = cb * 128 + 8 * sub;
  const uint64_t pol_s = l2_policy(a.l2 & 2 ? 1 : 0);
  const int m_end = min(a.M, (grp + 1) * kGroupRows);
  for (int m0 = grp * kGroupRows; m0 < m_end; m0 += 8) {
    const int m = m0 + (threadIdx.x >> 4);
    const bool ok = m < m_end;
    int ex[kTopK], pos[kTopK];
    if (ok) {
#pragma unroll
      for (int k = 0; k < kTopK; ++k) ex[k] = a.idx[m * kTopK + k], pos[k] = a.pos_of[m * kTopK + k];
#pragma unroll
      for (int i = 1; i < kTopK; ++i)
#pragma unroll
        for (int j = i; j > 0; --j)
          if (ex[j] < ex[j - 1]) {
            const int te = ex[j], tp = pos[j];
            ex[j] = ex[j - 1], pos[j] = pos[j - 1], ex[j - 1] = te, pos[j - 1] = tp;
          }
      float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        const uint4 v = __ldcg(reinterpret_cast<const uint4*>(a.dout + (size_t)pos[k] * kDimP + c0));
        const uint32_t* vv = &v.x;
#pragma unroll
        for (int j = 0; j < 4; ++j) s[2 * j] += bf16_lo(vv[j]), s[2 * j + 1] += bf16_hi(vv[j]);
      }
      const uint4 v = ldg128<H>(a.shout + (size_t)m * kDimP + c0, pol_s);
      const uint32_t* vv = &v.x;
#pragma unroll
      for (int j = 0; j < 4; ++j) s[2 * j] += bf16_lo(vv[j]), s[2 * j + 1] += bf16_hi(vv[j]);
      stg_f8<H>(a.y + (size_t)m * kDimP + c0, s, pol_s);
    }
    __syncwarp();
    if (ok && (sub & 7) == 0)
#pragma unroll
      for (int k = 0; k < kTopK; ++k)
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(a.dout + (size_t)pos[k] * kDimP + c0) : "memory");
  }
}

// Down: CTA = (128 output columns, one 64-row tile of an expert), 4 warps of 2 n16 groups each; the
// tile's h rows (all of K = 576) sit in shared memory, w2 streams from global into registers.
// FUSED: column-block-major order (one column block of all picks' outputs, P x 256 B, stays in L2),
// and each CTA counts its rows into their (column block, 64-row group); the CTA that completes a
// group runs its combine. Otherwise k_moe_combine runs afterwards over outputs that went to DRAM.
// CS: w2's scales from their 1-bit code (a.w.c2), as in k_moe_up_p; the flags read before the h rows.
template <bool FUSED, bool H, bool CS = false>
__global__ void __launch_bounds__(128) k_moe_down_p(MoePArgs a) {
  constexpr int KU = 3, LDA = kInterP;  // 9 steps in 3 batches
  __shared__ __align__(16) uint8_t sA[kTileRows * LDA];  // also the bf16 output staging
  __shared__ uint8_t sS[kTileRows * kKB2P];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  pdl_launch_dependents();
  pdl_wait();
  const int tile = FUSED ? blockIdx.x : blockIdx.y;
  const int ng = a.cb0 + (FUSED ? blockIdx.y : blockIdx.x);  // output columns ng*128 .. +127
  const int4 te = a.tiles[tile];
  if (tile >= *a.ntiles) return;
  const int e = te.x, base = te.y, rows = te.z;
  const int mt = (rows + 15) / 16;  // m16 tiles with rows
  const uint64_t pol_w = l2_policy((a.l2 & 2) && te.w <= kTileRows), pol_h = l2_policy(a.l2 & 4 ? 2 : 0);
  const uint64_t pol_o = l2_policy(FUSED ? (a.l2 & 4 ? 2 : 0) : (a.l2 & 2 ? 1 : 0));
  const int nq0 = ng * 8 + 2 * warp;
  const uint32_t* cp[2] = {};  // CS: this warp's two tiles' code blocks and raw flags (bit p: step p)
  uint32_t fl[2] = {};
  if constexpr (CS)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      cp[i] = a.w.c2.code + ((size_t)e * kT2P + nq0 + i) * kB2P;
      fl[i] = __ldg(cp[i] + kP2P);
    }
  const uint32_t sA0 = smem_addr(sA);
  for (int i = tid; i < kTileRows * (LDA / 16); i += 128) {
    const int r = i / (LDA / 16), c16 = i % (LDA / 16);
    cp_async16<H>(sA0 + r * LDA + 32 * ((c16 >> 1) ^ ((r >> 1) & 1)) + 16 * (c16 & 1),
               a.hq + (size_t)(base + (r < rows ? r : 0)) * kInterP + 16 * c16, r < rows, pol_h);
  }
  cp_async_commit();
  for (int i = tid; i < kTileRows * kKB2P; i += 128) {
    const int r = i / kKB2P;
    sS[i] = r < rows ? a.hs[(size_t)(base + r) * kKB2P + i % kKB2P] : 127;
  }
  const uint8_t* wp[2];
  const uint8_t* sp[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    wp[i] = a.w.w2 + (size_t)e * kW2P + ((size_t)(nq0 + i) * kP2P * 32 + lane) * 16;
    sp[i] = a.w.s2 + (size_t)e * kS2P + ((size_t)(nq0 + i) * kP2P * 8 + g) * 4;
  }
  uint4 wa[KU][2], wb[KU][2];
  uint32_t sa[KU][2], sb[KU][2];
  auto load_w = [&](uint4 (&w)[KU][2], uint32_t (&s)[KU][2], int p0) {
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        w[u][i] = ldg128<H>(wp[i] + (size_t)(p0 + u) * 512, pol_w);
        if constexpr (CS)
          s[u][i] = ldg32_sel<H>(fl[i] >> p0, 1u << u, sp[i] + (size_t)(p0 + u) * 32, cp[i] + p0 + u, pol_w);
        else
          s[u][i] = ldg32<H>(sp[i] + (size_t)(p0 + u) * 32, pol_w);
      }
  };
  float acc[4][4][4];  // [m16 tile][group 0 lo, hi, group 1 lo, hi][c]
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
  auto compute = [&](const uint4 (&w)[KU][2], const uint32_t (&s)[KU][2], int p0) {
    uint32_t sw[KU][2];  // the scale words (CS: rebuilt from the code)
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        sw[u][i] = s[u][i];
        if constexpr (CS)
          if (!(fl[i] >> (p0 + u) & 1u)) sw[u][i] = ((s[u][i] >> g) & 0x01010101u) + a.w.c2.base4;
      }
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int blk = 0; blk < 2; ++blk) {
        const int unit = 2 * (p0 + u) + blk;
        uint32_t af[4][4], sfa[4];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          if (mi >= mt) break;
          const int r = mi * 16 + g;
          const int off = 32 * (unit ^ ((r >> 1) & 1)) + 8 * t;  // rows r and r + 8 share the swizzle
          const uint2 lo = *reinterpret_cast<const uint2*>(sA + r * LDA + off);
          const uint2 hi = *reinterpret_cast<const uint2*>(sA + (r + 8) * LDA + off);
          af[mi][0] = lo.x, af[mi][1] = hi.x, af[mi][2] = lo.y, af[mi][3] = hi.y;
          sfa[mi] = sS[(mi * 16 + g + 8 * (lane & 1)) * kKB2P + unit];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t word = (&w[u][i].x)[2 * h + blk];
            const uint32_t b0 = e2m1x4_expand(word & 0xFFFFu), b1 = e2m1x4_expand(word >> 16);
            const uint32_t sfb = (sw[u][i] >> (8 * (2 * h + blk))) & 0xFFu;
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
              if (mi < mt)
                mma_e4m3_e2m1(acc[mi][2 * i + h], af[mi][0], af[mi][1], af[mi][2], af[mi][3], b0, b1, sfa[mi], sfb);
          }
      }
  };
  load_w(wa, sa, 0);
  load_w(wb, sb, KU);
  cp_async_wait<0>();
  __syncthreads();
  compute(wa, sa, 0);
  load_w(wa, sa, 2 * KU);
  compute(wb, sb, KU);
  compute(wa, sa, 2 * KU);
  __syncthreads();

  // bf16 outputs through shared memory (over sA), then 16-byte stores of whole 256-byte row pieces
  constexpr int LDO = 128 + 8;
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sA);
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = mi * 16 + g, c = 32 * warp + 8 * j + 2 * t;
      *reinterpret_cast<uint32_t*>(&sO[r * LDO + c]) = pack_bf16x2(acc[mi][j][0], acc[mi][j][1]);
      *reinterpret_cast<uint32_t*>(&sO[(r + 8) * LDO + c]) = pack_bf16x2(acc[mi][j][2], acc[mi][j][3]);
    }
  __syncthreads();
  for (int i = tid; i < rows * 16; i += 128) {
    const int r = i >> 4, c8 = i & 15;
    stg128<H>(a.dout + (size_t)(base + r) * kDimP + ng * 128 + 8 * c8,
              *reinterpret_cast<const uint4*>(&sO[r * LDO + 8 * c8]), pol_o);
  }
  if constexpr (FUSED) {
    __shared__ int sgrp[kTileRows], nsgrp;
    if (tid == 0) nsgrp = 0;
    __syncthreads();  // every thread's stores precede the releases below
    const int ngroups = (a.M + kGroupRows - 1) / kGroupRows;
    if (tid < rows) {
      const int m = a.perm_tok[base + tid], grp = m / kGroupRows;
      const int target = kTopK * (min(a.M, (grp + 1) * kGroupRows) - grp * kGroupRows);
      __threadfence();
      if (atomicAdd(&a.done[ng * ngroups + grp], 1) == target - 1) {
        __threadfence();
        sgrp[atomicAdd(&nsgrp, 1)] = grp;
      }
    }
    __syncthreads();
    for (int i = 0; i < nsgrp; ++i) combine_group<H>(a, ng, sgrp[i]);
  }
}

// Down, a CTA per (tile, a.gcb column blocks of 128): the tile's h rows go to shared memory once, w2
// streams from global into registers D steps of 64 k ahead with no break between column blocks, and
// each block's bf16 outputs leave through shared memory, 32 rows at a time, while the next block's
// first weights are in flight (46.7 KB of shared memory: 2 CTAs per SM, as k_moe_down_p). FUSED as in
// k_moe_down_p: the grid runs column group by column group, and the CTA that completes a (column
// block, 64-row group) runs its combine. Every output is k_moe_down_p's sum, in its order.
constexpr int kDownGSmem = kTileRows * kInterP + kTileRows * kKB2P + 32 * (128 + 8) * 2;

template <int D, bool FUSED, bool H>
__global__ void __launch_bounds__(128) k_moe_down_g(MoePArgs a) {
  constexpr int LDA = kInterP, LDO = 128 + 8, NS = kP2P;  // 9 steps of 64 k per column block
  static_assert(NS % D == 0, "the ring holds whole steps of a column block");
  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* sA = dsm;                                                                          // [64][576] h
  uint8_t* sS = dsm + kTileRows * LDA;                                                        // [64][18]
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(dsm + kTileRows * LDA + kTileRows * kKB2P);  // [32][LDO]
  __shared__ int sgrp[kTileRows], nsgrp;  // FUSED: the row groups a block completed
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  pdl_launch_dependents();
  pdl_wait();
  const int tile = FUSED ? blockIdx.x : blockIdx.y;
  const int4 te = a.tiles[tile];
  if (tile >= *a.ntiles) return;
  const int cb1 = a.cb0 + (FUSED ? blockIdx.y : blockIdx.x) * a.gcb, nb = min(a.gcb, a.cb0 + a.ncb - cb1);
  const int e = te.x, base = te.y, rows = te.z;
  const int mt = (rows + 15) / 16;
  const uint64_t pol_w = l2_policy((a.l2 & 2) && te.w <= kTileRows), pol_h = l2_policy(a.l2 & 4 ? 2 : 0);
  const uint64_t pol_o = l2_policy(FUSED ? (a.l2 & 4 ? 2 : 0) : (a.l2 & 2 ? 1 : 0));
  const uint32_t sA0 = smem_addr(sA);
  for (int i = tid; i < kTileRows * (LDA / 16); i += 128) {
    const int r = i / (LDA / 16), c16 = i % (LDA / 16);
    cp_async16<H>(sA0 + r * LDA + 32 * ((c16 >> 1) ^ ((r >> 1) & 1)) + 16 * (c16 & 1),
                  a.hq + (size_t)(base + (r < rows ? r : 0)) * kInterP + 16 * c16, r < rows, pol_h);
  }
  cp_async_commit();
  for (int i = tid; i < kTileRows * kKB2P; i += 128) {
    const int r = i / kKB2P;
    sS[i] = r < rows ? a.hs[(size_t)(base + r) * kKB2P + i % kKB2P] : 127;
  }
  // FUSED: this thread's row (tid < rows), its 64-row group and the picks that complete the group
  const int ngroups = (a.M + kGroupRows - 1) / kGroupRows;
  int grp = 0, target = 0;
  if (FUSED && tid < rows) {
    grp = a.perm_tok[base + tid] / kGroupRows;
    target = kTopK * (min(a.M, (grp + 1) * kGroupRows) - grp * kGroupRows);
  }
  // this warp's first n16 group of the CTA's first column block; group i of block b is 8 b + i past it
  const uint8_t* w0 = a.w.w2 + (size_t)e * kW2P + ((size_t)(cb1 * 8 + 2 * warp) * kP2P * 32 + lane) * 16;
  const uint8_t* s0 = a.w.s2 + (size_t)e * kS2P + ((size_t)(cb1 * 8 + 2 * warp) * kP2P * 8 + g) * 4;
  uint4 w[D][2];
  uint32_t s[D][2];
  auto load = [&](int j, int b, int p) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const size_t st = (size_t)(8 * b + i) * kP2P + p;
      w[j][i] = ldg128<H>(w0 + st * 512, pol_w);
      s[j][i] = ldg32<H>(s0 + st * 32, pol_w);
    }
  };
#pragma unroll
  for (int j = 0; j < D; ++j) load(j, 0, j);
  cp_async_wait<0>();
  __syncthreads();
  float acc[4][4][4];  // [m16 tile][group 0 lo, hi, group 1 lo, hi][c]
  for (int b = 0; b < nb; ++b) {
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
#pragma unroll
    for (int p = 0; p < NS; ++p) {
      constexpr int kD = D;
      const int j = p % kD;
#pragma unroll
      for (int blk = 0; blk < 2; ++blk) {
        const int unit = 2 * p + blk;
        uint32_t af[4][4], sfa[4];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
          if (mi >= mt) break;
          const int r = mi * 16 + g;
          const int off = 32 * (unit ^ ((r >> 1) & 1)) + 8 * t;  // rows r and r + 8 share the swizzle
          const uint2 lo = *reinterpret_cast<const uint2*>(sA + r * LDA + off);
          const uint2 hi = *reinterpret_cast<const uint2*>(sA + (r + 8) * LDA + off);
          af[mi][0] = lo.x, af[mi][1] = hi.x, af[mi][2] = lo.y, af[mi][3] = hi.y;
          sfa[mi] = sS[(mi * 16 + g + 8 * (lane & 1)) * kKB2P + unit];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t word = (&w[j][i].x)[2 * h + blk];
            const uint32_t b0 = e2m1x4_expand(word & 0xFFFFu), b1 = e2m1x4_expand(word >> 16);
            const uint32_t sfb = (s[j][i] >> (8 * (2 * h + blk))) & 0xFFu;
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
              if (mi < mt)
                mma_e4m3_e2m1(acc[mi][2 * i + h], af[mi][0], af[mi][1], af[mi][2], af[mi][3], b0, b1, sfa[mi], sfb);
          }
      }
      // the slot's next step: D past this one, in this block or the next
      if (b + (p + D) / NS < nb) load(j, b + (p + D) / NS, (p + D) % NS);
    }
    const int ng = cb1 + b;
    // bf16 outputs through shared memory, 32 rows at a time, then 16-byte stores of whole 256-byte
    // row pieces
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      if (32 * hf >= rows) break;
      __syncthreads();  // sO is free, and the previous block's combines have read nsgrp
#pragma unroll
      for (int mi = 2 * hf; mi < 2 * hf + 2; ++mi)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int r = mi * 16 - 32 * hf + g, c = 32 * warp + 8 * j + 2 * t;
          *reinterpret_cast<uint32_t*>(&sO[r * LDO + c]) = pack_bf16x2(acc[mi][j][0], acc[mi][j][1]);
          *reinterpret_cast<uint32_t*>(&sO[(r + 8) * LDO + c]) = pack_bf16x2(acc[mi][j][2], acc[mi][j][3]);
        }
      __syncthreads();
      const int n = min(32, rows - 32 * hf);
      for (int i = tid; i < n * 16; i += 128) {
        const int r = i >> 4, c8 = i & 15;
        stg128<H>(a.dout + (size_t)(base + 32 * hf + r) * kDimP + ng * 128 + 8 * c8,
                  *reinterpret_cast<const uint4*>(&sO[r * LDO + 8 * c8]), pol_o);
      }
    }
    if constexpr (FUSED) {
      if (tid == 0) nsgrp = 0;
      __syncthreads();  // every thread's stores precede the releases below
      if (tid < rows) {
        __threadfence();
        if (atomicAdd(&a.done[ng * ngroups + grp], 1) == target - 1) {
          __threadfence();
          sgrp[atomicAdd(&nsgrp, 1)] = grp;
        }
      }
      __syncthreads();
      for (int i = 0; i < nsgrp; ++i) combine_group<H>(a, ng, sgrp[i]);
    }
  }
}

template <bool H>
__global__ void __launch_bounds__(128) k_moe_combine(MoePArgs a) {
  pdl_launch_dependents();
  pdl_wait();
  const int m = blockIdx.x;
  const uint64_t pol_s = l2_policy(a.l2 & 2 ? 1 : 0);
  int ex[kTopK], pos[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) ex[k] = a.idx[m * kTopK + k], pos[k] = a.pos_of[m * kTopK + k];
#pragma unroll
  for (int i = 1; i < kTopK; ++i)
#pragma unroll
    for (int j = i; j > 0; --j)
      if (ex[j] < ex[j - 1]) {
        const int te = ex[j], tp = pos[j];
        ex[j] = ex[j - 1], pos[j] = pos[j - 1], ex[j - 1] = te, pos[j - 1] = tp;
      }
  for (int c = a.cb0 * 16 + threadIdx.x; c < (a.cb0 + a.ncb) * 16; c += 128) {
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const uint4 v = ldg128<H>(a.dout + (size_t)pos[k] * kDimP + 8 * c, pol_s);
      const uint32_t* vv = &v.x;
#pragma unroll
      for (int j = 0; j < 4; ++j) s[2 * j] += bf16_lo(vv[j]), s[2 * j + 1] += bf16_hi(vv[j]);
    }
    const uint4 v = ldg128<H>(a.shout + (size_t)m * kDimP + 8 * c, pol_s);
    const uint32_t* vv = &v.x;
#pragma unroll
    for (int j = 0; j < 4; ++j) s[2 * j] += bf16_lo(vv[j]), s[2 * j + 1] += bf16_hi(vv[j]);
    stg_f8<H>(a.y + (size_t)m * kDimP + 8 * c, s, pol_s);
  }
}

// The combine fused into the reduce-scatter of its output (moe_combine_rs). A quarter (rank k's rows
// [k rq, (k + 1) rq) of the column group) is one stream of rq * cols fp32, posted in pieces of up to
// cap / 4 bytes as comm's bulk reduce-scatter posts them, with its pipeline: piece i + 1 summed into
// its send buffer while i travels, then i reduced and i + 2 summed in. A send buffer's 16 bytes are
// four columns of a row: its six outputs in ascending expert id, then the shared expert, in fp32 from
// 0 as k_moe_combine adds them (zeros past M). This rank's quarter of the four contributions is summed
// in rank order (comm's sum4_f32) and rounded to bf16.
struct CombRsArgs {
  const __nv_bfloat16 *dout, *shout;
  const int* pos_srt;
  int M, rq, c0, cols;
  __nv_bfloat16* out;
  size_t ld_out;
};

__global__ void __launch_bounds__(256) k_moe_combine_rs(comm::DevChannel ch, CombRsArgs a, uint32_t piece) {
  const uint64_t s0 = comm::next_seq(ch);
  const uint32_t t0 = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  const size_t qbytes = (size_t)a.rq * a.cols * 4, npieces = (qbytes + piece - 1) / piece;
  auto len = [&](size_t i) { return (uint32_t)(qbytes - i * piece < piece ? qbytes - i * piece : piece); };
  auto sums = [&](int m, int col) {
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    if (m < a.M) {
      int pos[kTopK];
#pragma unroll
      for (int k = 0; k < kTopK; ++k) pos[k] = __ldg(a.pos_srt + m * kTopK + k);
      uint2 v[kTopK + 1];
#pragma unroll
      for (int k = 0; k < kTopK; ++k) v[k] = __ldcs(reinterpret_cast<const uint2*>(a.dout + (size_t)pos[k] * kDimP + col));
      v[kTopK] = __ldcs(reinterpret_cast<const uint2*>(a.shout + (size_t)m * kDimP + col));
#pragma unroll
      for (int k = 0; k <= kTopK; ++k)
        s[0] += bf16_lo(v[k].x), s[1] += bf16_hi(v[k].x), s[2] += bf16_lo(v[k].y), s[3] += bf16_hi(v[k].y);
    }
    return make_uint4(__float_as_uint(s[0]), __float_as_uint(s[1]), __float_as_uint(s[2]), __float_as_uint(s[3]));
  };
  auto copy = [&](size_t i, uint64_t s) {
    const uint32_t pl = len(i) / 16;
    uint4* sb = (uint4*)comm::send_buf(ch, s);
    for (uint32_t j = t0; j < pl; j += stride) {
      const size_t e = (i * piece + (size_t)j * 16) / 4;  // element of the quarter's stream
      const int r = (int)(e / a.cols), col = a.c0 + (int)(e % a.cols);
#pragma unroll 1
      for (int k = 0; k < 4; k++) sb[(size_t)k * pl + j] = sums(k * a.rq + r, col);
    }
  };
  copy(0, s0);
  comm::post_grid(ch, s0, 4 * len(0), comm::kOpReduceScatter);
  if (npieces > 1) copy(1, s0 + 1);
  for (size_t i = 0; i < npieces; i++) {
    const uint64_t s = s0 + i;
    comm::wait_all(ch, s);
    if (i + 1 < npieces) comm::post_grid(ch, s + 1, 4 * len(i + 1), comm::kOpReduceScatter);
    const uint32_t pl = len(i) / 16;
    const size_t qoff = (size_t)ch.rank * pl * 16;
    for (uint32_t j = t0; j < pl; j += stride) {
      const float4 v = comm::sum4_f32(ch, s, qoff + (size_t)j * 16);
      const size_t e = (i * piece + (size_t)j * 16) / 4;
      const int r = (int)(e / a.cols), col = a.c0 + (int)(e % a.cols);
      *reinterpret_cast<uint2*>(a.out + (size_t)r * a.ld_out + col) = make_uint2(pack_bf16x2(v.x, v.y), pack_bf16x2(v.z, v.w));
    }
    if (i + 2 < npieces) {
      if (i > 0) comm::wait_posted(ch, s + 1);  // s - 1's send buffer, as in comm's reduce_scatter_bulk_kernel
      copy(i + 2, s + 2);
    }
  }
  comm::finish_grid(ch, s0 + npieces - 1);
}

size_t align256(size_t b) { return (b + 255) & ~(size_t)255; }

// Workspace carve-up (shared by moe_prefill and its size query).
struct MoePWs {
  int *cnt, *off, *cursor, *ntiles, *done;
  int4* tiles;
  int *perm_tok, *pos_of, *pos_srt;
  float* perm_w;
  uint8_t *hq, *hs, *shq, *shs;
  __nv_bfloat16 *dout, *shout;
  size_t bytes;
};

MoePWs carve(void* base, int M, int E) {
  const size_t P = (size_t)M * kTopK, maxt = E + P / kTileRows + 1;
  uint8_t* p = static_cast<uint8_t*>(base);
  size_t o = 0;
  auto take = [&](size_t b) {
    uint8_t* r = p ? p + o : nullptr;
    o += align256(b);
    return r;
  };
  MoePWs w;
  w.cnt = reinterpret_cast<int*>(take(E * 4));
  w.off = reinterpret_cast<int*>(take((E + 1) * 4));
  w.cursor = reinterpret_cast<int*>(take(E * 4));
  w.ntiles = reinterpret_cast<int*>(take(4));
  w.done = reinterpret_cast<int*>(take((size_t)kColBlocks * ((M + kGroupRows - 1) / kGroupRows) * 4));
  w.tiles = reinterpret_cast<int4*>(take(maxt * 16));
  w.perm_tok = reinterpret_cast<int*>(take(P * 4));
  w.pos_of = reinterpret_cast<int*>(take(P * 4));
  w.pos_srt = reinterpret_cast<int*>(take(P * 4));
  w.perm_w = reinterpret_cast<float*>(take(P * 4));
  w.hq = take(P * kInterP);
  w.hs = take(P * kKB2P);
  w.shq = take((size_t)M * kInterP);
  w.shs = take((size_t)M * kKB2P);
  w.dout = reinterpret_cast<__nv_bfloat16*>(take(P * kDimP * 2));
  w.shout = reinterpret_cast<__nv_bfloat16*>(take((size_t)M * kDimP * 2));
  w.bytes = o;
  return w;
}

PrefillTuning g_tune;
// k_moe_down_g's column blocks per CTA by default, for the separate and the fused combine (0: k_moe_down_p)
constexpr int kDownGroup = 0, kDownGroupF = 0;
constexpr int kTileOrder = 0;   // tile_pos's k by default (0: descending count)
constexpr int kUpPrefetch = 0;  // k_moe_up_p's L2 prefetch distance in chunks by default (0: none)

// k_moe_down_g by column groups of G blocks, with g_tune.dring (3 or 9) weight steps in flight
void launch_down_g(MoePArgs a, int G, int maxt, bool fused, bool h, cudaStream_t st) {
  using K = void (*)(MoePArgs);
  static const K ks[2][2][2] = {{{k_moe_down_g<3, false, false>, k_moe_down_g<3, false, true>},
                                 {k_moe_down_g<3, true, false>, k_moe_down_g<3, true, true>}},
                                {{k_moe_down_g<9, false, false>, k_moe_down_g<9, false, true>},
                                 {k_moe_down_g<9, true, false>, k_moe_down_g<9, true, true>}}};
  a.gcb = G;
  const int ncg = (a.ncb + G - 1) / G;
  launch_pdl(ks[g_tune.dring == 9][fused][h], fused ? dim3(maxt, ncg) : dim3(ncg, maxt), dim3(128), kDownGSmem, st,
             true, a);
}

void gemm_fp8_any(const uint8_t* xq, const uint8_t* xs, int M, const Fp8Seg* segs, int nseg, int K, int epi,
                  uint8_t* hq, uint8_t* hs, int n_inter, cudaStream_t st) {
  if (M < 1 || K % 64 || nseg < 1 || nseg > kMaxSegs) die("gemm_fp8_p: M=%d K=%d nseg=%d", M, K, nseg);
  GemmArgs a{};
  a.xq = xq, a.xs = xs, a.M = M, a.K = K, a.nseg = nseg, a.hq = hq, a.hs = hs, a.n_inter = n_inter;
  bool all128 = true;
  for (int i = 0; i < nseg; ++i) {
    a.seg[i] = segs[i];
    if (segs[i].N % 64) die("gemm_fp8_p: segment %d N=%d not a multiple of 64", i, segs[i].N);
    all128 &= segs[i].N % 128 == 0;
  }
  // Tiles (measured at M = 2048 / 4096, docs in README): 128x128 with 4 warps of 64x64 and 3 stages
  // (48 KB, two CTAs per SM) beats 8-warp and 128x256 / 256x128 tiles by 5-25 %.
  const int v = g_tune.fp8;
  if (epi == kEpiSwiglu) {
    if (n_inter % 64) die("gemm_fp8_swiglu_p: N=%d", n_inter);
    if (g_tune.small) {  // 32 columns of w1 and w3 a tile: one act_quant group
      launch_gemm_fp8<32, 64, 32, 32, 4, kEpiSwiglu>(a, n_inter / 32, st);
      return;
    }
    if (v == 4) launch_gemm_fp8<128, 128, 64, 32, 4, kEpiSwiglu>(a, n_inter / 64, st);
    else launch_gemm_fp8<128, 128, 64, 64, 3, kEpiSwiglu>(a, n_inter / 64, st);
    return;
  }
  if (g_tune.small) {  // small-M tiles: narrower N tiles for more CTAs; each output's K chain unchanged
    const int sb = g_tune.small == 1 ? 64 : 32;
    a.tile0[0] = 0;
    for (int i = 0; i < nseg; ++i) {
      if (segs[i].N % sb) die("gemm_fp8_p: small tiles: segment %d N=%d not a multiple of %d", i, segs[i].N, sb);
      a.tile0[i + 1] = a.tile0[i] + segs[i].N / sb;
    }
    if (sb == 64) launch_gemm_fp8<32, 64, 32, 32, 4, kEpiBf16>(a, a.tile0[nseg], st);
    else launch_gemm_fp8<32, 32, 32, 16, 4, kEpiBf16>(a, a.tile0[nseg], st);
    return;
  }
  int bn = v == 2 || v == 5 ? 256 : 128;
  for (int i = 0; i < nseg; ++i)
    if (segs[i].N % bn) bn = all128 ? 128 : 64;
  a.tile0[0] = 0;
  for (int i = 0; i < nseg; ++i) a.tile0[i + 1] = a.tile0[i] + segs[i].N / bn;
  const int nt = a.tile0[nseg];
  if (bn == 64) launch_gemm_fp8<128, 64, 64, 32, 4, kEpiBf16>(a, nt, st);
  else if (bn == 256 && v == 5) launch_gemm_fp8<128, 256, 64, 64, 4, kEpiBf16>(a, nt, st);
  else if (bn == 256) launch_gemm_fp8<128, 256, 64, 64, 3, kEpiBf16>(a, nt, st);
  else if (v == 1) launch_gemm_fp8<128, 128, 64, 32, 3, kEpiBf16>(a, nt, st);
  else if (v == 3) launch_gemm_fp8<256, 128, 64, 64, 3, kEpiBf16>(a, nt, st);
  else if (v == 4) launch_gemm_fp8<128, 128, 64, 32, 4, kEpiBf16>(a, nt, st);
  else launch_gemm_fp8<128, 128, 64, 64, 3, kEpiBf16>(a, nt, st);
}

}  // namespace

void gemm_fp8_p(const uint8_t* xq, const uint8_t* xs, int M, const Fp8Seg* segs, int nseg, int K, cudaStream_t st) {
  gemm_fp8_any(xq, xs, M, segs, nseg, K, kEpiBf16, nullptr, nullptr, 0, st);
}

void sum4_bf16_p(const __nv_bfloat16* part, size_t slot, size_t n, __nv_bfloat16* out, cudaStream_t st) {
  if (n % 8 || slot % 8) die("sum4_bf16_p: n=%zu slot=%zu", n, slot);
  if (!n) return;
  const size_t n16 = n / 8;
  const int blocks = (int)std::min<size_t>((n16 + 255) / 256, 48 * 8);
  launch_pdl(k_sum4_bf16, dim3(blocks), dim3(256), 0, st, true, reinterpret_cast<const uint4*>(part), slot / 8, n16,
             reinterpret_cast<uint4*>(out));
}

void gemm_fp8_swiglu_p(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w1, const uint8_t* s1,
                       const uint8_t* w3, const uint8_t* s3, int N, int K, uint8_t* hq, uint8_t* hs, cudaStream_t st) {
  const Fp8Seg segs[2] = {{w1, s1, nullptr, N}, {w3, s3, nullptr, N}};
  gemm_fp8_any(xq, xs, M, segs, 2, K, kEpiSwiglu, hq, hs, N, st);
}

void gemm_bf16_p(const __nv_bfloat16* x, int M, const uint8_t* W, int N, int K, float* y, cudaStream_t st) {
  if (M < 1 || K % 64 || N % 32) die("gemm_bf16_p: M=%d N=%d K=%d", M, N, K);
  G16Args a{x, K, M, K, W, nullptr, N, N, y, nullptr, nullptr, nullptr};
  // small-M tiles (the router gate: N = 384 in 12 or 24 CTAs per M tile instead of 3)
  if (g_tune.small == 1 && N % 32 == 0) return launch_gemm16<32, 32, 32, 16, 4, false>(a, st);
  if (g_tune.small == 2 && N % 16 == 0) return launch_gemm16<16, 16, 16, 16, 4, false>(a, st);
  if (N % 128 == 0) launch_gemm16<128, 128, 64, 32, 3, false>(a, st);
  else if (N % 64 == 0) launch_gemm16<128, 64, 64, 32, 3, false>(a, st);
  else launch_gemm16<128, 32, 32, 32, 3, false>(a, st);
}

void gemm_bf16_rows_p(const __nv_bfloat16* x, int M, int K, const uint16_t* W0, const uint16_t* W1, int N0, int N,
                      float* y0, float* y1, bool round_bf16, cudaStream_t st) {
  if (M < 1 || K % 64 || N0 % 64 || N % 64 || N0 > N || (N > N0 && !W1))
    die("gemm_bf16_rows_p: M=%d N0=%d N=%d K=%d", M, N0, N, K);
  G16Args a{x, K, M, K, reinterpret_cast<const uint8_t*>(W0), nullptr, N, N, y0, nullptr, nullptr, nullptr};
  a.ldw = K, a.N0 = N0, a.W1 = reinterpret_cast<const uint8_t*>(W1), a.yf1 = y1, a.rnd = round_bf16;
  // 128x128 tiles while they make two waves, else 64x64 (both divide every segment)
  if (N0 % 128 == 0 && N % 128 == 0 && N / 128 * ((M + 127) / 128) >= 96)
    launch_gemm16<128, 128, 64, 32, 3, false>(a, st);
  else
    launch_gemm16<64, 64, 32, 32, 4, false>(a, st);
}

void gemm_woa_p(const __nv_bfloat16* x, int M, const uint8_t* W, const uint8_t* S, int N, int K, int group_rows,
                __nv_bfloat16* y, uint8_t* q, uint8_t* qs, cudaStream_t st) {
  if (M < 1 || K % 64 || N % 128 || group_rows % 128) die("gemm_woa_p: M=%d N=%d K=%d", M, N, K);
  G16Args a{x, K * (N / group_rows), M, K, W, S, N, group_rows, nullptr, y, q, qs};
  // 128x256 tiles while they make two waves or more; smaller tiles for short chunks
  const int mtiles = (M + 127) / 128;
  const bool t256 = N % 256 == 0 && group_rows % 256 == 0;
  if (g_tune.small == 1) return launch_gemm16<32, 64, 32, 32, 4, true>(a, st);
  if (g_tune.small == 2) return launch_gemm16<32, 32, 32, 16, 4, true>(a, st);
  if (g_tune.w16 == 1 && t256) launch_gemm16<128, 256, 64, 64, 3, true>(a, st);
  else if (g_tune.w16 == 2) launch_gemm16<128, 128, 64, 32, 3, true>(a, st);
  else if (g_tune.w16 == 3) launch_gemm16<64, 64, 32, 32, 4, true>(a, st);
  else if (t256 && N / 256 * mtiles >= 96) launch_gemm16<128, 256, 64, 64, 3, true>(a, st);
  else if (N / 128 * mtiles >= 96) launch_gemm16<128, 128, 64, 32, 3, true>(a, st);
  else launch_gemm16<64, 64, 32, 32, 4, true>(a, st);
}

void prefill_tuning(const PrefillTuning& t) { g_tune = t; }

size_t moe_prefill_workspace_bytes(int M, int E) { return carve(nullptr, M, E).bytes; }

bool moe_prefill_fuses(int M) {
  const size_t P = (size_t)M * kTopK;
  return g_tune.down == 0 ? P * (256 + kInterP) <= (12u << 20) : g_tune.down == 2;
}

void moe_combine_rs(const comm::DevChannel& ch, void* ws, int M, int E, int rq, int col0, int cols,
                    __nv_bfloat16* out, size_t ld_out, cudaStream_t st, int blocks) {
  if (col0 % 128 || cols % 128 || cols < 128 || col0 + cols > kDimP || rq < 1 || 4 * rq < M || ld_out % 4)
    die("moe_combine_rs: columns %d +%d, rq %d, M %d", col0, cols, rq, M);
  const MoePWs b = carve(ws, M, E);
  CombRsArgs a{b.dout, b.shout, b.pos_srt, M, rq, col0, cols, out, ld_out};
  blocks = blocks < 1 ? 1 : blocks > 48 ? 48 : blocks;
  k_moe_combine_rs<<<blocks, 256, 0, st>>>(ch, a, (ch.cap / 4) & ~63u);
  CUDA_CHECK(cudaGetLastError());
}

void moe_prefill(const uint8_t* xq, const uint8_t* xs, int M, const int* idx, const float* wts, int E,
                 const MoeWeights& w, void* ws, float* y, cudaStream_t st, MoePCfg cfg) {
  if (M < 1 || E < kTopK || E > 1024) die("moe_prefill: M=%d E=%d", M, E);
  const MoePWs b = carve(ws, M, E);
  const int P = M * kTopK, maxt = E + P / kTileRows + 1;
  // L2 priorities (MoePArgs::l2): 7 = weights evict_first unless the expert has several tiles, the
  // combine's single-use traffic evict_first, re-read activations evict_last. At M = 8192 that is
  // 19.2 ms against 21.5 without hints (weights of multi-tile experts stay for their next tile).
  const int l2 = g_tune.l2 >= 0 ? g_tune.l2 : 7;
  if (cfg.col0 % 128 || cfg.cols % 128 || cfg.col0 + cfg.cols > kDimP) die("moe_prefill: columns %d +%d", cfg.col0, cfg.cols);
  const int cb0 = cfg.col0 / 128, ncb = cfg.cols ? cfg.cols / 128 : kColBlocks - cb0;
  MoePArgs a{xq, xs, idx, wts, M, E, w, b.cnt, b.off, b.cursor, b.ntiles, b.tiles, b.perm_tok, b.perm_w,
             b.pos_of, b.hq, b.hs, b.dout, b.shout, y, b.done, cb0, ncb, l2, 0,
             g_tune.upf >= 0 ? g_tune.upf : kUpPrefetch};
  const int ndone = kColBlocks * ((M + kGroupRows - 1) / kGroupRows);
  // Fused combine while one column block of all picks' outputs plus the h rows it re-reads per column
  // block (P x (256 + 576) B) fit L2 comfortably: up to M ~ 2.5k (saves ~0.7 ms at M = 2048); beyond,
  // the re-read h rows thrash L2 and a separate combine over outputs in DRAM is faster.
  const bool fused = moe_prefill_fuses(M);
  // The shared expert (compute-bound) can run on a side stream while the routed experts stream
  // their weights; the combine joins both.
  static thread_local cudaEvent_t ev[2] = {nullptr, nullptr};
  const bool fork = cfg.side && (cfg.phases & 2);
  if (fork && !ev[0])
    for (auto& e : ev) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  if (fork) {
    CUDA_CHECK(cudaEventRecord(ev[0], st));
    CUDA_CHECK(cudaStreamWaitEvent(cfg.side, ev[0], 0));
  }
  if (cfg.phases & 2) {
    const cudaStream_t s2 = fork ? cfg.side : st;
    gemm_fp8_swiglu_p(xq, xs, M, w.sh_w1, w.sh_s1, w.sh_w3, w.sh_s3, kInterP, kDimP, b.shq, b.shs, s2);
    const Fp8Seg sh2{w.sh_w2, w.sh_s2, b.shout, kDimP};
    gemm_fp8_p(b.shq, b.shs, M, &sh2, 1, kInterP, s2);
    if (fork) CUDA_CHECK(cudaEventRecord(ev[1], cfg.side));
  }
  if (cfg.phases & 1) {
    launch_pdl(k_moe_bucket, dim3(1), dim3(1024), 0, st, true, idx, P, E, b.cnt, b.off, b.cursor, b.ntiles, b.tiles,
               b.done, ndone, g_tune.tord >= 0 ? g_tune.tord : kTileOrder);
    launch_pdl(k_moe_scatter, dim3((P + 255) / 256), dim3(256), 0, st, true, idx, wts, P, (const int*)b.off,
               b.cursor, b.perm_tok, b.perm_w, b.pos_of);
    if (!fused)
      launch_pdl(k_moe_sortpos, dim3((M + 255) / 256), dim3(256), 0, st, true, idx, (const int*)b.pos_of, M, b.pos_srt);
  }
  const bool h = l2 != 0;
  // the routed scales from their code when the weights carry it (k_moe_down_g, a tuning, reads them raw)
  const bool cs13 = w.c1.code && w.c3.code, cs2 = w.c2.code != nullptr;
  if (cfg.phases & 4)
    launch_pdl(cs13 ? (h ? k_moe_up_p<4, true, true> : k_moe_up_p<4, false, true>)
                    : (h ? k_moe_up_p<4, true> : k_moe_up_p<4, false>),
               dim3(kInterP / 64, maxt), dim3(128), 0, st, true, a);
  if (fork) CUDA_CHECK(cudaStreamWaitEvent(st, ev[1], 0));
  if (cfg.phases & 8) {  // with the combine fused, after the shared expert (above)
    const int G = g_tune.dgrp >= 0 ? g_tune.dgrp : fused ? kDownGroupF : kDownGroup;
    if (G > 0)
      launch_down_g(a, G, maxt, fused, h, st);
    else if (fused)
      launch_pdl(cs2 ? (h ? k_moe_down_p<true, true, true> : k_moe_down_p<true, false, true>)
                     : (h ? k_moe_down_p<true, true> : k_moe_down_p<true, false>),
                 dim3(maxt, ncb), dim3(128), 0, st, true, a);
    else
      launch_pdl(cs2 ? (h ? k_moe_down_p<false, true, true> : k_moe_down_p<false, false, true>)
                     : (h ? k_moe_down_p<false, true> : k_moe_down_p<false, false>),
                 dim3(ncb, maxt), dim3(128), 0, st, true, a);
  }
  if (!fused && (cfg.phases & 16))
    launch_pdl(h ? k_moe_combine<true> : k_moe_combine<false>, dim3(M), dim3(128), 0, st, true, a);
}

}  // namespace dsk
