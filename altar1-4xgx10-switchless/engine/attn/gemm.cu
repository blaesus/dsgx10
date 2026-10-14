// FP8 GEMM for prefill chunks (any number of rows) on the bf16 tensor cores: the GEMV's arithmetic in the
// GEMV's order, so every output equals gemv()'s bit for bit at the same splitk (docs/attention.md § 3).
//
// A CTA owns a BM x BN output tile; stages of 64 k go global -> shared by cp.async (STAGES deep): the
// activations as [BM][64] bf16 (16-byte chunk c of row r at c ^ (r & 1): conflict-free 128-bit reads), the
// weights as the BN/8 engine tiles' 512-byte warp-steps, unchanged. A warp owns WM x WN: per stage it holds
// the A fragments of its MI m-tiles (lane (g, t): rows g and g + 8, k = 16t .. 16t + 15: the GEMV's map), and
// per n-tile reads the lane's 16 weight bytes, widens them exactly to bf16 and runs, for each m-tile, the
// GEMV's four MMAs from zero; the stage's f32 result is scaled and added to the accumulator (acc += s * c),
// k in order. With split-K the parts' sums are kept apart and added in part order, as the GEMV's warps do.
// BF16 weights (gemm_bf16; row-major [N][K], no scales): the BF16 GEMV's arithmetic instead, the four MMAs of a
// chunk accumulating straight into the sum (k in order), the weights staged as each lane's 32 bytes (row g,
// k = 16t .. 16t + 15 of the chunk: the GEMV's two 16-byte loads).
#include <algorithm>

#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"

namespace glm {
namespace {

struct GSeg {
  const uint8_t* w;
  const float* s;
  void* y;
  int n, ldy, lrs, lks, group_rows, x_group, cta0;  // cta0: the segment's first CTA column
  int vec;    // y rows 16-byte aligned: the epilogue stores 16-byte pieces
  int tiled;  // BF16: kBF16T's lane order (else rows)
};
struct GArgs {
  const uint16_t* x;
  int ldx, rows, K, splitk, nseg;
  int gm;  // the raster: groups of gm row tiles, row tiles fastest within a group (0: blockIdx as is)
  GSeg s[kMaxSegs];
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void mma_z(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {  // d = A B (from zero)
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}

template <int OUT, bool SPLIT, int BM, int BN, int WM, int WN, int STAGES, bool BW = false>
__global__ void __launch_bounds__(BM / WM * (BN / WN) * 32, 1) k_gemm(GArgs a) {  // 1 CTA an SM
  constexpr int NWM = BM / WM, NTH = NWM * (BN / WN) * 32;
  constexpr int MI = WM / 16, NI = WN / 8;
  constexpr int WB = BW ? 2 : 1;  // weight bytes a value
  constexpr int SA = BM * 128, SB = BN * 64 * WB, STG = SA + SB;
  constexpr int ACH = BM * 8 / NTH, BCH = BN * 4 * WB / NTH;
  static_assert(ACH * NTH == BM * 8 && BCH * NTH == BN * 4 * WB, "tile shape");
  extern __shared__ __align__(128) uint8_t smem[];
  pdl_trigger();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wm = warp % NWM, wn = warp / NWM;
  // the CTA's segment and tile
  // grouped raster: the CTAs in flight share a few row tiles' activations and a few column tiles' weights (in
  // blockIdx order the weights would be read once a row tile; at 2,600 rows that bound the BF16 GEMM by bandwidth).
  // Which CTA computes a tile changes nothing in its arithmetic.
  int bx = blockIdx.x, by = blockIdx.y;
  if (a.gm > 0) {
    const int gx = gridDim.x, gy = gridDim.y, id = by * gx + bx, grp = id / (a.gm * gx), first = grp * a.gm;
    const int gsz = min(a.gm, gy - first), in = id - grp * a.gm * gx;
    by = first + in % gsz;
    bx = in / gsz;
  }
  int si = 0;
#pragma unroll
  for (int i = 1; i < kMaxSegs; ++i)
    if (i < a.nseg && bx >= a.s[i].cta0) si = i;
  const GSeg sg = a.s[si];
  const int n0 = (bx - sg.cta0) * BN;  // first row of W in the segment
  const int m0 = by * BM;
  const int K = a.K, J = K >> 6;
  const uint16_t* xb = a.x + (sg.group_rows ? (size_t)(n0 / sg.group_rows) * sg.x_group : 0);

  const uint8_t* asrc[ACH];
  uint32_t adst[ACH];
  bool aok[ACH];
#pragma unroll
  for (int j = 0; j < ACH; ++j) {
    const int i = tid + j * NTH, r = i >> 3, c = i & 7, row = m0 + r;
    aok[j] = row < a.rows;
    asrc[j] = reinterpret_cast<const uint8_t*>(xb + (size_t)(aok[j] ? row : 0) * a.ldx) + 16 * c;
    adst[j] = r * 128 + 16 * (c ^ (r & 1));
  }
  const uint8_t* bsrc[BCH];
  uint32_t bdst[BCH];
  bool bok[BCH];
#pragma unroll
  for (int j = 0; j < BCH; ++j) {
    const int i = tid + j * NTH;
    if constexpr (BW) {  // 16-byte half hf of lane l's 32 bytes: row 8 nt + g, k = 16t + 8hf .. + 7
      const int nt = i >> 6, l = (i >> 1) & 31, hf = i & 1, n = n0 + 8 * nt + (l >> 2);
      bok[j] = n < sg.n;
      bsrc[j] = sg.tiled ? sg.w + ((size_t)(bok[j] ? n >> 3 : 0) * J) * 1024 + 512 * hf + 16 * l
                         : sg.w + ((size_t)(bok[j] ? n : 0) * K + 16 * (l & 3) + 8 * hf) * 2;
      bdst[j] = SA + nt * 1024 + 32 * l + 16 * hf;
    } else {
      const int nt = i >> 5, l = i & 31, n = n0 + 8 * nt;
      bok[j] = n < sg.n;
      bsrc[j] = sg.w + ((size_t)(bok[j] ? n >> 3 : 0) * J) * 512 + 16 * l;
      bdst[j] = SA + nt * 512 + 16 * l;
    }
  }
  const uint32_t s_base = smem_u32(smem);
  const size_t bstep = BW ? (sg.tiled ? 1024 : 128) : 512;  // a 64-k chunk's stride
  auto load_stage = [&](int kt, int buf) {
    const uint32_t s0 = s_base + buf * STG;
#pragma unroll
    for (int j = 0; j < ACH; ++j) cp16(s0 + adst[j], asrc[j] + kt * 128, aok[j]);
#pragma unroll
    for (int j = 0; j < BCH; ++j) cp16(s0 + bdst[j], bsrc[j] + (size_t)kt * bstep, bok[j]);
  };
  // scales: the warp's n-tiles (SwiGLU: lane t's intermediate's gate row, up at + K / ks)
  const float* sp[NI];
  bool nok[NI];
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) {
    const int n = n0 + wn * WN + ni * 8;
    nok[ni] = n < sg.n;
    const int nn = nok[ni] && !BW ? n : 0;
    sp[ni] = BW ? nullptr : OUT == (int)GemvOut::kSwiGLU ? sg.s + (size_t)(((nn >> 1) + t) >> sg.lrs) * 2 * (K >> sg.lks)
                                          : sg.s + (size_t)(nn >> sg.lrs) * (K >> sg.lks);
  }
  const int sup = K >> sg.lks;

  float acc[MI][NI][4];
  float tot[SPLIT ? MI : 1][SPLIT ? NI : 1][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0.f;
  int part = 0, pend = SPLIT ? J / a.splitk : J;  // the GEMV's part boundaries: J * p / splitk
  float sn0[NI], sn1[NI];  // the next stage's scales
  auto load_scales = [&](int kt) {
    if constexpr (!BW) {
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const float* b = sp[ni] + ((kt * 64) >> sg.lks);
        sn0[ni] = __ldg(b);
        sn1[ni] = OUT == (int)GemvOut::kSwiGLU ? __ldg(b + sup) : sn0[ni];
      }
    }
  };
  load_scales(0);

  pdl_wait();
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < J) load_stage(s, s);
    cp_commit();
  }
  for (int kt = 0; kt < J; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nk = kt + STAGES - 1;
      if (nk < J) load_stage(nk, nk % STAGES);
      cp_commit();
    }
    float s0[NI], s1[NI];
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) s0[ni] = sn0[ni], s1[ni] = sn1[ni];
    if (kt + 1 < J) load_scales(kt + 1);
    const uint8_t* sa0 = smem + (kt % STAGES) * STG;
    const uint8_t* sb0 = sa0 + SA;
    uint4 xa[MI][2], xc[MI][2];  // rows g, g + 8: chunks 2t, 2t + 1
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
      const int r = wm * WM + mi * 16 + g;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int off = 16 * ((2 * t + h) ^ (r & 1));
        xa[mi][h] = *reinterpret_cast<const uint4*>(sa0 + r * 128 + off);
        xc[mi][h] = *reinterpret_cast<const uint4*>(sa0 + (r + 8) * 128 + off);
      }
    }
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      if constexpr (BW) {
        const uint8_t* wb = sb0 + (wn * (WN / 8) + ni) * 1024 + 32 * lane;
        const uint4 w0 = *reinterpret_cast<const uint4*>(wb), w1 = *reinterpret_cast<const uint4*>(wb + 16);
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
          float(&c)[4] = acc[mi][ni];
          mma_bf16(c, xa[mi][0].x, xc[mi][0].x, xa[mi][0].y, xc[mi][0].y, w0.x, w0.y);
          mma_bf16(c, xa[mi][0].z, xc[mi][0].z, xa[mi][0].w, xc[mi][0].w, w0.z, w0.w);
          mma_bf16(c, xa[mi][1].x, xc[mi][1].x, xa[mi][1].y, xc[mi][1].y, w1.x, w1.y);
          mma_bf16(c, xa[mi][1].z, xc[mi][1].z, xa[mi][1].w, xc[mi][1].w, w1.z, w1.w);
        }
        continue;
      }
      const uint4 wv = *reinterpret_cast<const uint4*>(sb0 + (wn * (WN / 8) + ni) * 512 + 16 * lane);
      uint32_t o[8];
      fp8x4_to_bf16(wv.x, o[0], o[1]);
      fp8x4_to_bf16(wv.y, o[2], o[3]);
      fp8x4_to_bf16(wv.z, o[4], o[5]);
      fp8x4_to_bf16(wv.w, o[6], o[7]);
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        float c[4];
        mma_z(c, xa[mi][0].x, xc[mi][0].x, xa[mi][0].y, xc[mi][0].y, o[0], o[1]);
        mma_bf16(c, xa[mi][0].z, xc[mi][0].z, xa[mi][0].w, xc[mi][0].w, o[2], o[3]);
        mma_bf16(c, xa[mi][1].x, xc[mi][1].x, xa[mi][1].y, xc[mi][1].y, o[4], o[5]);
        mma_bf16(c, xa[mi][1].z, xc[mi][1].z, xa[mi][1].w, xc[mi][1].w, o[6], o[7]);
        acc[mi][ni][0] = __fadd_rn(acc[mi][ni][0], __fmul_rn(s0[ni], c[0]));
        acc[mi][ni][1] = __fadd_rn(acc[mi][ni][1], __fmul_rn(s1[ni], c[1]));
        acc[mi][ni][2] = __fadd_rn(acc[mi][ni][2], __fmul_rn(s0[ni], c[2]));
        acc[mi][ni][3] = __fadd_rn(acc[mi][ni][3], __fmul_rn(s1[ni], c[3]));
      }
    }
    if constexpr (SPLIT) {
      if (kt + 1 == pend) {  // a part ends: the GEMV's warp 0 adds part p's sum to its own, in order
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int ni = 0; ni < NI; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              tot[mi][ni][e] = part == 0 ? acc[mi][ni][e] : __fadd_rn(tot[mi][ni][e], acc[mi][ni][e]);
              acc[mi][ni][e] = 0.f;
            }
        ++part;
        pend = (int)(((long)J * (part + 1)) / a.splitk);
      }
    }
  }
  cp_wait<0>();
  if constexpr (SPLIT) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[mi][ni][e] = tot[mi][ni][e];
  }
  // epilogue: the GEMV's roundings, staged through shared memory for whole-line stores
  __syncthreads();
  constexpr int ES = OUT == (int)GemvOut::kF32 ? 4 : 2;
  constexpr int TC = OUT == (int)GemvOut::kSwiGLU ? BN / 2 : BN;  // output columns of the tile
  constexpr int LDS = TC * ES + (ES == 4 ? 32 : 16);                // bytes a staged row (bank-conflict free)
  static_assert(BM * LDS <= STAGES * STG, "epilogue staging");
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = wm * WM + mi * 16 + g + 8 * h, c = wn * WN + ni * 8 + 2 * t;
        const float a0 = acc[mi][ni][2 * h], a1 = acc[mi][ni][2 * h + 1];
        uint8_t* d = smem + r * LDS;
        if constexpr (OUT == (int)GemvOut::kBF16) {
          *reinterpret_cast<uint32_t*>(d + c * 2) = pack_bf16(a0, a1);
        } else if constexpr (OUT == (int)GemvOut::kF32) {
          *reinterpret_cast<float2*>(d + c * 4) = make_float2(a0, a1);
        } else {  // column 2t gate, 2t + 1 up of intermediate c / 2
          const float gt = bfround(a0), up = bfround(a1);
          const float sl = bfround(__fdiv_rn(gt, __fadd_rn(1.f, expf(-gt))));
          *reinterpret_cast<uint16_t*>(d + c) = f2bf(__fmul_rn(sl, up));
        }
      }
  __syncthreads();
  const int nv = min(BN, sg.n - n0), cv = OUT == (int)GemvOut::kSwiGLU ? nv / 2 : nv;  // valid output columns
  const int c0 = OUT == (int)GemvOut::kSwiGLU ? n0 / 2 : n0;
  constexpr int PR = TC * ES / 16;  // 16-byte pieces a row
  for (int i = tid; i < BM * PR; i += NTH) {
    const int r = i / PR, pc = i % PR, row = m0 + r, col = pc * 16 / ES;
    if (row >= a.rows || col >= cv) continue;
    uint8_t* dst = static_cast<uint8_t*>(sg.y) + ((size_t)row * sg.ldy + c0 + col) * ES;
    const uint8_t* src = smem + r * LDS + pc * 16;
    if (sg.vec && col + 16 / ES <= cv) {
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else {
      for (int e = 0; e < 16 / ES && col + e < cv; ++e)
        if constexpr (ES == 4) reinterpret_cast<float*>(dst)[e] = reinterpret_cast<const float*>(src)[e];
        else reinterpret_cast<uint16_t*>(dst)[e] = reinterpret_cast<const uint16_t*>(src)[e];
    }
  }
}

int ilog2g(int v, const char* what) {
  int l = 0;
  while ((1 << l) < v) ++l;
  if ((1 << l) != v) die("gemm: %s %d is not a power of two", what, v);
  return l;
}

template <int OUT, bool SPLIT, int BM, int BN, int WM, int WN, int STAGES, bool BW = false>
void launch_gemm(const GArgs& a, int gx, cudaStream_t s, bool pdl) {
  constexpr int NTH = BM / WM * (BN / WN) * 32;
  const size_t smem = (size_t)STAGES * (BM * 128 + BN * 64 * (BW ? 2 : 1));
  static bool once = [&] {
    CUDA_CHECK(cudaFuncSetAttribute(k_gemm<OUT, SPLIT, BM, BN, WM, WN, STAGES, BW>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    return true;
  }();
  (void)once;
  launch(k_gemm<OUT, SPLIT, BM, BN, WM, WN, STAGES, BW>, dim3(gx, (a.rows + BM - 1) / BM), dim3(NTH), smem, s, pdl, a);
  CUDA_CHECK(cudaGetLastError());
}

int g_gemm_cfg = -1;  // a bench knob: tile configurations (-1, the default: by shape)
int g_gemm_gm = -1;   // a bench knob: the raster (-1, the default: kGemmRaster)

// the launch arguments; returns the CTA columns (BN weight rows a CTA)
int gemm_args(GArgs& a, const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out,
              int splitk, int BN, bool fp8) {
  a = GArgs{};
  a.x = x;
  a.ldx = ldx;
  a.rows = rows;
  a.K = K;
  a.splitk = splitk;
  a.nseg = nseg;
  // the raster by the launch's weight bytes: grouped once they outgrow the L2 (measured: 2x on o_proj, the dense
  // MLP and the head; 5-30 % slower on the small shapes, whose weights stay in L2 in blockIdx order)
  size_t wbytes = 0;
  for (int i = 0; i < nseg; ++i) wbytes += (size_t)segs[i].n * K * (fp8 ? 1 : 2);
  a.gm = g_gemm_gm >= 0 ? g_gemm_gm : wbytes >= ((size_t)32 << 20) ? kGemmRaster : 0;
  int c0 = 0;
  for (int i = 0; i < nseg; ++i) {
    const GemvSeg& g = segs[i];
    if (g.n % 8) die("gemm: segment %d has %d rows", i, g.n);
    if (g.group_rows && g.group_rows % BN) die("gemm: groups of %d rows (tiles of %d)", g.group_rows, BN);
    if (fp8 && (g.rs < 8 || g.ks < 64)) die("gemm: scale block %dx%d", g.rs, g.ks);
    GSeg& d = a.s[i];
    d.w = static_cast<const uint8_t*>(g.w);
    d.s = fp8 ? g.s : nullptr;
    d.y = g.y;
    d.n = g.n;
    d.ldy = g.ldy;
    d.lrs = fp8 ? ilog2g(g.rs, "rs") : 0;
    d.lks = fp8 ? ilog2g(g.ks, "ks") : 0;
    d.group_rows = g.group_rows;
    d.x_group = g.x_group;
    d.cta0 = c0;
    d.tiled = g.fmt == WFmt::kBF16T;
    {
      const size_t es = out == GemvOut::kF32 ? 4 : 2;
      d.vec = (reinterpret_cast<uintptr_t>(g.y) % 16 == 0) && ((size_t)g.ldy * es % 16 == 0);
    }
    c0 += (g.n + BN - 1) / BN;
  }
  return c0;
}


// BF16 weights: one tile shape, 128 x 64, 3 stages (72 KiB: a stage's bf16 weights take twice the FP8 bytes); warps
// 64 x 16 for a split-K (the FP8 split-K default's), 32 x 32 otherwise. SwiGLU: the GEMV's epilogue (gate/up rows
// interleaved within each 8-row n-tile).
void gemm_bf16_run(const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
                   cudaStream_t s, bool pdl) {
  GArgs a;
  const int c0 = gemm_args(a, x, ldx, rows, K, segs, nseg, out, splitk, 64, false);
#define GB(O)                                                                       \
  if (splitk > 1) launch_gemm<(int)O, true, 128, 64, 64, 16, 3, true>(a, c0, s, pdl); \
  else launch_gemm<(int)O, false, 128, 64, 32, 32, 3, true>(a, c0, s, pdl);
  if (out == GemvOut::kBF16) { GB(GemvOut::kBF16) }
  else if (out == GemvOut::kF32) { GB(GemvOut::kF32) }
  else { GB(GemvOut::kSwiGLU) }
#undef GB
}

void gemm_fp8_run(const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
                  cudaStream_t s, bool pdl) {
  // tile configurations (CTA rows x columns, warp tile, stages): 0: 128 x 128, 64 x 32, 4 (split-K: 128 x 64,
  // 64 x 16); 1: 64 x 128, 32 x 32, 4; 2: 128 x 64, 32 x 32, 4; 3: 128 x 128, 32 x 64, 4 (split-K: 128 x 64,
  // 32 x 32, 3); 4: as 0 with 3 stages; 5: 128 x 128, 32 x 32 (16 warps), 4 (split-K: 3)
  const int cfg = g_gemm_cfg >= 0 ? g_gemm_cfg : splitk > 1 ? 4 : 0;  // by shape: measured on GB10
  const int BN = splitk > 1 ? (cfg == 1 || cfg == 5 ? 128 : 64) : (cfg == 2 ? 64 : 128);
  GArgs a;
  const int c0 = gemm_args(a, x, ldx, rows, K, segs, nseg, out, splitk, BN, true);
#define GO(O, SP, BM_, BN_, WM_, WN_, ST) launch_gemm<(int)O, SP, BM_, BN_, WM_, WN_, ST>(a, c0, s, pdl)
#define CFGS(O)                                                         \
  if (splitk > 1) {                                                     \
    if (cfg == 1) GO(O, true, 64, 128, 32, 32, 4);                      \
    else if (cfg == 2) GO(O, true, 128, 64, 32, 32, 4);                 \
    else if (cfg == 3) GO(O, true, 128, 64, 32, 32, 3);                 \
    else if (cfg == 4) GO(O, true, 128, 64, 64, 16, 3);                 \
    else if (cfg == 5) GO(O, true, 128, 128, 32, 32, 3);                \
    else GO(O, true, 128, 64, 64, 16, 4);                               \
  } else {                                                              \
    if (cfg == 1) GO(O, false, 64, 128, 32, 32, 4);                     \
    else if (cfg == 2) GO(O, false, 128, 64, 32, 32, 4);                \
    else if (cfg == 3) GO(O, false, 128, 128, 32, 64, 4);               \
    else if (cfg == 4) GO(O, false, 128, 128, 64, 32, 3);               \
    else if (cfg == 5) GO(O, false, 128, 128, 32, 32, 4);               \
    else GO(O, false, 128, 128, 64, 32, 4);                             \
  }
  if (out == GemvOut::kBF16) { CFGS(GemvOut::kBF16) }
  else if (out == GemvOut::kF32) { CFGS(GemvOut::kF32) }
  else { CFGS(GemvOut::kSwiGLU) }
#undef CFGS
#undef GO
}

void gemm_check(const char* who, int rows, int K, int nseg, int splitk) {
  if (rows < 1) die("%s: %d rows", who, rows);
  if (K % 64 || nseg < 1 || nseg > kMaxSegs) die("%s: K %d, %d segments", who, K, nseg);
  if (splitk != 1 && splitk != 2 && splitk != 4) die("%s: splitk %d", who, splitk);
  if (K / 64 < splitk) die("%s: K %d for split-K %d", who, K, splitk);
}

}  // namespace

void gemm_config(int cfg) { g_gemm_cfg = cfg; }
void gemm_raster(int gm) { g_gemm_gm = gm; }

void gemm_bf16(const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
               cudaStream_t s, bool pdl) {
  gemm_check("gemm_bf16", rows, K, nseg, splitk);
  for (int i = 0; i < nseg; ++i)
    if (segs[i].fmt != WFmt::kNone && segs[i].fmt != WFmt::kBF16 && segs[i].fmt != WFmt::kBF16T)
      die("gemm_bf16: segment %d is not BF16", i);
  gemm_bf16_run(x, ldx, rows, K, segs, nseg, out, splitk, s, pdl);
}

void gemm(const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
          cudaStream_t s, bool pdl, uint16_t* scratch, size_t scratch_elems) {
  gemm_check("gemm", rows, K, nseg, splitk);
  GemvSeg b[kMaxSegs], f[kMaxSegs];
  int nb = 0, nf = 0;
  for (int i = 0; i < nseg; ++i) {
    const GemvSeg& g = segs[i];
    if (g.fmt == WFmt::kBF16 || g.fmt == WFmt::kBF16T) b[nb++] = g;
    else if (g.fmt == WFmt::kFP8) f[nf++] = g;
    else if (g.fmt != WFmt::kINT4) die("gemm: segment %d has no weight format (%d)", i, (int)g.fmt);
  }
  if (nb) gemm_bf16_run(x, ldx, rows, K, b, nb, out, splitk, s, pdl);
  for (int i = 0; i < nseg; ++i) {  // INT4: the GEMV's dequantisation into the scratch, then the BF16 GEMM
    const GemvSeg& g = segs[i];
    if (g.fmt != WFmt::kINT4) continue;
    if (K % kInt4BlkK || (K / kInt4BlkK) % splitk) die("gemm: INT4 K %d for split-K %d", K, splitk);
    if (!scratch || scratch_elems < (size_t)g.n * K) die("gemm: INT4 [%d][%d] needs a bf16 scratch of %zu elements (%zu given)",
                                                         g.n, K, (size_t)g.n * K, scratch_elems);
    dequant_int4(g.w, g.n, K, scratch, K, s, pdl);
    GemvSeg d = g;
    d.fmt = WFmt::kBF16;
    d.w = scratch;
    gemm_bf16_run(x, ldx, rows, K, &d, 1, out, splitk, s, pdl);
  }
  if (nf) gemm_fp8_run(x, ldx, rows, K, f, nf, out, splitk, s, pdl);
}

}  // namespace glm
