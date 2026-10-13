// FP8 (zai's 128x128 block scales, re-cut to the engine tiles) and BF16 GEMVs for 1-16 rows on the bf16
// tensor cores. A warp owns one 8-row tile of W and a range of k: per 64-k chunk it loads its 16 weights (one
// 16-byte load for FP8: lane (g, t) holds row g, k = 16t..16t+15 of the chunk; two for BF16), widens e4m3 to
// bf16 exactly, and runs four m16n8k16 MMAs with A = x (rows g and g+8 of the activations), B = the weights.
// The chunk's f32 result is scaled (FP8) and added in k order. With split-K the warps of a CTA take parts of
// k, added in a fixed order through shared memory. Up to P chunks of weights are in flight a lane.
#include "attn/attn.h"
#include "attn/attn_common.cuh"
#include "attn/launch.h"

namespace glm {
namespace {

struct SegDev {
  const uint8_t* w;
  const float* s;
  void* y;
  int ldy, lrs, lks, group_rows, x_group, tile0;
};
struct Segs {
  SegDev s[kMaxSegs];
  int tile_end;  // total tiles
  int n;
};
// the index weights fused into the x-projection's launch: CTAs past the GEMV's run prep.cu's k_index_weights
// for one head and up to 16 rows, each of the 128 threads doing two of its 256 threads' work (the same sums in
// the same order)
struct IwDev {
  const uint16_t* wp;
  float* w;
  int ldw, nh, cta0;  // cta0: the first index-weights CTA
};

__device__ __forceinline__ void index_weights_cta(const uint16_t* __restrict__ x, int ldx, int rows, const IwDev& iw,
                                                  int blk) {
  __shared__ float red[16][8];
  constexpr int kPer = kDim / 256;  // 24 a (virtual) thread
  const int h = blk % iw.nh, r0 = (blk / iw.nh) * 16, tid = threadIdx.x;
  float wv[2][kPer];
#pragma unroll
  for (int v = 0; v < 2; ++v) {
    const uint16_t* wr = iw.wp + (size_t)h * kDim + (tid + 128 * v) * kPer;
#pragma unroll
    for (int i = 0; i < kPer; i += 8) {
      const uint4 q = *reinterpret_cast<const uint4*>(wr + i);
      const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { wv[v][i + 2 * j] = bf2f(ww[j]); wv[v][i + 2 * j + 1] = bf2f_hi(ww[j]); }
    }
  }
  pdl_wait();
  const int nr = min(16, rows - r0);
  for (int rr = 0; rr < nr; ++rr) {
#pragma unroll
    for (int v = 0; v < 2; ++v) {
      const uint16_t* xr = x + (size_t)(r0 + rr) * ldx + (tid + 128 * v) * kPer;
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < kPer; i += 8) {
        const uint4 q = *reinterpret_cast<const uint4*>(xr + i);
        const uint32_t ww[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          sum = __fadd_rn(sum, __fmul_rn(bf2f(ww[j]), wv[v][i + 2 * j]));
          sum = __fadd_rn(sum, __fmul_rn(bf2f_hi(ww[j]), wv[v][i + 2 * j + 1]));
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
      if ((tid & 31) == 0) red[rr][(tid >> 5) + 4 * v] = sum;
    }
  }
  __syncthreads();
  if (tid < nr) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sum = __fadd_rn(sum, red[tid][i]);
    iw.w[(size_t)(r0 + tid) * iw.ldw + h] = __fmul_rn(sum, kIdxWScale);
  }
}

template <bool FP8, int OUT, bool HI, int P, int NWP, bool IW>
__global__ void __launch_bounds__(NWP * 32) k_gemv(const uint16_t* __restrict__ x, int ldx, int rows, int K, Segs segs,
                                                  int splitk, IwDev iw) {
  __shared__ float red[NWP][32][4];
  pdl_trigger();
  if constexpr (IW) {
    if ((int)blockIdx.x >= iw.cta0) {
      index_weights_cta(x, ldx, rows, iw, blockIdx.x - iw.cta0);
      return;
    }
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int per_cta = NWP / splitk;
  const int tile = blockIdx.x * per_cta + warp / splitk, kpart = warp % splitk;
  const bool valid = tile < segs.tile_end;
  int si = 0;
#pragma unroll
  for (int i = 1; i < kMaxSegs; ++i)
    if (i < segs.n && tile >= segs.s[i].tile0) si = i;
  const SegDev sg = segs.s[si];
  const int lt = valid ? tile - sg.tile0 : 0, n0 = lt * 8;
  const int J = K >> 6;
  const int j0 = (J * kpart) / splitk, j1 = (J * (kpart + 1)) / splitk;
  // weights
  const uint8_t* wp = FP8 ? sg.w + ((size_t)lt * J) * 512 + lane * 16
                          : sg.w + ((size_t)(n0 + g) * K + 16 * t) * 2;
  // scale row: the tile's block; SwiGLU: lane t's intermediate (n0/2 + t)'s gate block row (up follows it)
  const float* sp = !FP8 ? nullptr
                    : OUT == (int)GemvOut::kSwiGLU ? sg.s + (size_t)(((n0 >> 1) + t) >> sg.lrs) * 2 * (K >> sg.lks)
                                                   : sg.s + (size_t)(n0 >> sg.lrs) * (K >> sg.lks);
  // activations
  const uint16_t* xb = x + (sg.group_rows ? (size_t)(n0 / sg.group_rows) * sg.x_group : 0) + 16 * t;
  const bool live0 = g < rows, live1 = HI && g + 8 < rows;
  const uint16_t* x0 = xb + (size_t)(live0 ? g : 0) * ldx;
  const uint16_t* x1 = xb + (size_t)(live1 ? g + 8 : 0) * ldx;

  constexpr int NW = FP8 ? 1 : 2;  // 16-byte loads a chunk
  uint4 buf[P][NW];
  auto load = [&](uint4 (&b)[NW], int j) {
    if constexpr (FP8) {
      b[0] = ldg_stream128(wp + (size_t)j * 512);
    } else {
      b[0] = ldg_stream128(wp + (size_t)j * 128);
      b[1] = ldg_stream128(wp + (size_t)j * 128 + 16);
    }
  };
  if (valid) {
#pragma unroll
    for (int u = 0; u < P; ++u)
      if (j0 + u < j1) load(buf[u], j0 + u);
  }
  pdl_wait();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  auto comp = [&](const uint4 (&b)[NW], int j) {
    uint32_t o[8];
    if constexpr (FP8) {
      fp8x4_to_bf16(b[0].x, o[0], o[1]);
      fp8x4_to_bf16(b[0].y, o[2], o[3]);
      fp8x4_to_bf16(b[0].z, o[4], o[5]);
      fp8x4_to_bf16(b[0].w, o[6], o[7]);
    } else {
      o[0] = b[0].x; o[1] = b[0].y; o[2] = b[0].z; o[3] = b[0].w;
      o[4] = b[1].x; o[5] = b[1].y; o[6] = b[1].z; o[7] = b[1].w;
    }
    uint4 xa = make_uint4(0, 0, 0, 0), xc = xa, ya = xa, yc = xa;
    if (live0) {
      xa = __ldg(reinterpret_cast<const uint4*>(x0 + j * 64));
      xc = __ldg(reinterpret_cast<const uint4*>(x0 + j * 64 + 8));
    }
    if (live1) {
      ya = __ldg(reinterpret_cast<const uint4*>(x1 + j * 64));
      yc = __ldg(reinterpret_cast<const uint4*>(x1 + j * 64 + 8));
    }
    if constexpr (FP8) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      mma_bf16(c, xa.x, ya.x, xa.y, ya.y, o[0], o[1]);
      mma_bf16(c, xa.z, ya.z, xa.w, ya.w, o[2], o[3]);
      mma_bf16(c, xc.x, yc.x, xc.y, yc.y, o[4], o[5]);
      mma_bf16(c, xc.z, yc.z, xc.w, yc.w, o[6], o[7]);
      float s0, s1;
      if constexpr (OUT == (int)GemvOut::kSwiGLU) {  // column 2t: gate i, 2t+1: up i; scales [inter/rs][2][K/ks]
        const float* b = sp + ((j * 64) >> sg.lks);
        s0 = __ldg(b);
        s1 = __ldg(b + (K >> sg.lks));
      } else {
        s0 = s1 = __ldg(sp + ((j * 64) >> sg.lks));
      }
      acc[0] = __fadd_rn(acc[0], __fmul_rn(s0, c[0]));
      acc[1] = __fadd_rn(acc[1], __fmul_rn(s1, c[1]));
      acc[2] = __fadd_rn(acc[2], __fmul_rn(s0, c[2]));
      acc[3] = __fadd_rn(acc[3], __fmul_rn(s1, c[3]));
    } else {
      mma_bf16(acc, xa.x, ya.x, xa.y, ya.y, o[0], o[1]);
      mma_bf16(acc, xa.z, ya.z, xa.w, ya.w, o[2], o[3]);
      mma_bf16(acc, xc.x, yc.x, xc.y, yc.y, o[4], o[5]);
      mma_bf16(acc, xc.z, yc.z, xc.w, yc.w, o[6], o[7]);
    }
  };
  if (valid) {
    for (int j = j0; j < j1; j += P) {
#pragma unroll
      for (int u = 0; u < P; ++u) {
        const int jj = j + u;
        if (jj < j1) {
          uint4 b[NW];
#pragma unroll
          for (int q = 0; q < NW; ++q) b[q] = buf[u][q];
          if (jj + P < j1) load(buf[u], jj + P);
          comp(b, jj);
        }
      }
    }
  }
  if (splitk > 1) {
    if (kpart) {
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
    }
    __syncthreads();
    if (kpart) return;
    for (int k = 1; k < splitk; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = __fadd_rn(acc[i], red[warp + k][lane][i]);
  }
  if (!valid) return;
  const int n = n0 + 2 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    if (h ? !live1 : !live0) continue;
    const float a0 = acc[2 * h], a1 = acc[2 * h + 1];
    if constexpr (OUT == (int)GemvOut::kBF16) {
      *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(sg.y) + (size_t)r * sg.ldy + n) = pack_bf16(a0, a1);
    } else if constexpr (OUT == (int)GemvOut::kF32) {
      *reinterpret_cast<float2*>(static_cast<float*>(sg.y) + (size_t)r * sg.ldy + n) = make_float2(a0, a1);
    } else {  // SwiGLU: column 2t is gate i, 2t+1 up i, i = n0/2 + t
      const float gt = bfround(a0), up = bfround(a1);
      const float sl = bfround(__fdiv_rn(gt, __fadd_rn(1.f, expf(-gt))));
      static_cast<uint16_t*>(sg.y)[(size_t)r * sg.ldy + n0 / 2 + t] = f2bf(__fmul_rn(sl, up));
    }
  }
}

int ilog2(int v, const char* what) {
  int l = 0;
  while ((1 << l) < v) ++l;
  if ((1 << l) != v) die("gemv: %s %d is not a power of two", what, v);
  return l;
}

template <bool FP8, int OUT, bool HI, int P, int NW, bool IW>
void launch_gemv_cfg(const uint16_t* x, int ldx, int rows, int K, const Segs& sd, int splitk, IwDev iw, cudaStream_t s,
                     bool pdl) {
  const int per_cta = NW / splitk;
  const int grid = (sd.tile_end + per_cta - 1) / per_cta;
  iw.cta0 = grid;
  const int extra = IW ? iw.nh * ((rows + 15) / 16) : 0;
  launch(k_gemv<FP8, OUT, HI, P, NW, IW>, dim3(grid + extra), dim3(NW * 32), 0, s, pdl, x, ldx, rows, K, sd, splitk, iw);
  CUDA_CHECK(cudaGetLastError());
}

int g_gemv_cfg = -1;  // a bench knob: loads in flight a lane and warps a CTA (bit-neutral); -1: by shape

template <bool FP8, int OUT, bool HI>
void launch_gemv(const uint16_t* x, int ldx, int rows, int K, const Segs& sd, int splitk, const IwDev* iw,
                 cudaStream_t s, bool pdl) {
  if (iw) {  // 128-thread CTAs (the index weights' emulation)
    if constexpr (FP8) launch_gemv_cfg<FP8, OUT, HI, 8, 4, true>(x, ldx, rows, K, sd, splitk, *iw, s, pdl);
    else die("gemv: index weights fuse with the FP8 x-projection only");
    return;
  }
  const IwDev none{};
  int want = g_gemv_cfg;
  if (want < 0) {  // by shape (measured on GB10): 16 loads a lane for the grouped W_UV (K 512), else 8 warps a CTA
    bool grouped = false;
    for (int i = 0; i < sd.n; ++i) grouped = grouped || sd.s[i].group_rows > 0;
    want = grouped && K >= 512 ? 1 : 4;
  }
  const int cfg = FP8 && splitk <= 2 ? want : FP8 && (want <= 2 || want == 4) ? want : 0;
  if constexpr (FP8) {
    switch (cfg) {
      case 1: launch_gemv_cfg<FP8, OUT, HI, 16, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 2: launch_gemv_cfg<FP8, OUT, HI, 24, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 3: launch_gemv_cfg<FP8, OUT, HI, 8, 2, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 4: launch_gemv_cfg<FP8, OUT, HI, 8, 8, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 5: launch_gemv_cfg<FP8, OUT, HI, 16, 2, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      default: break;
    }
  }
  launch_gemv_cfg<FP8, OUT, HI, FP8 ? 8 : 4, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl);
}

}  // namespace

void gemv_config(int cfg) { g_gemv_cfg = cfg; }

void gemv(bool fp8, const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
          cudaStream_t s, bool pdl, const GemvIw* iw) {
  if (rows < 1 || rows > kMaxRows) die("gemv: %d rows", rows);
  if (K % 64 || nseg < 1 || nseg > kMaxSegs) die("gemv: K %d, %d segments", K, nseg);
  if (splitk != 1 && splitk != 2 && splitk != 4) die("gemv: splitk %d", splitk);
  Segs sd{};
  int t0 = 0;
  for (int i = 0; i < nseg; ++i) {
    const GemvSeg& g = segs[i];
    if (g.n % 8) die("gemv: segment %d has %d rows", i, g.n);
    SegDev& d = sd.s[i];
    d.w = static_cast<const uint8_t*>(g.w);
    d.s = g.s;
    d.y = g.y;
    d.ldy = g.ldy;
    d.lrs = fp8 ? ilog2(g.rs, "rs") : 0;
    d.lks = fp8 ? ilog2(g.ks, "ks") : 0;
    if (fp8 && (g.rs < 8 || g.ks < 64)) die("gemv: scale block %dx%d", g.rs, g.ks);
    d.group_rows = g.group_rows;
    d.x_group = g.x_group;
    if (g.group_rows && g.group_rows % 8) die("gemv: group of %d rows", g.group_rows);
    d.tile0 = t0;
    t0 += g.n / 8;
  }
  sd.tile_end = t0;
  sd.n = nseg;
  const bool hi = rows > 8;
  IwDev iwd{};
  if (iw) {
    if (!fp8 || out != GemvOut::kBF16 || K != kDim) die("gemv: index weights with a %s GEMV of K %d", fp8 ? "FP8" : "BF16", K);
    iwd.wp = iw->wproj;
    iwd.w = iw->w;
    iwd.ldw = iw->ldw;
    iwd.nh = iw->nh;
  }
  const IwDev* ip = iw ? &iwd : nullptr;
#define GL(F, O)                                                                    \
  (hi ? launch_gemv<F, (int)O, true>(x, ldx, rows, K, sd, splitk, ip, s, pdl)     \
      : launch_gemv<F, (int)O, false>(x, ldx, rows, K, sd, splitk, ip, s, pdl))
  if (fp8) {
    if (out == GemvOut::kBF16) GL(true, GemvOut::kBF16);
    else if (out == GemvOut::kF32) GL(true, GemvOut::kF32);
    else GL(true, GemvOut::kSwiGLU);
  } else {
    if (out == GemvOut::kBF16) GL(false, GemvOut::kBF16);
    else if (out == GemvOut::kF32) GL(false, GemvOut::kF32);
    else GL(false, GemvOut::kSwiGLU);
  }
#undef GL
}

}  // namespace glm
