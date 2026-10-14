// GEMVs for 1-16 rows on the bf16 tensor cores, over BF16, INT4 (compressed-tensors, group 32, asymmetric) and FP8
// (zai's 128x128 block scales, re-cut to the engine tiles) weights. A warp owns one 8-row tile of W and a range of
// k: per 64-k chunk lane (g, t) holds row g, k = 16t..16t+15 of the chunk (BF16: two 16-byte loads; FP8: one;
// INT4: half of one, a 16-byte load holding two chunks), turns them into bf16 (BF16 as is; FP8 widened exactly;
// INT4 the contract's bf16_rne((nib - znib) * s)) and runs four m16n8k16 MMAs with A = x (rows g and g+8 of the
// activations), B = the weights. BF16 and INT4: the MMAs accumulate straight into the f32 sum, k in order (so the
// INT4 GEMV equals the BF16 GEMV on the dequantised matrix bit for bit); FP8: the chunk's f32 result is scaled and
// added in k order. With split-K the warps of a CTA take parts of k, added in a fixed order through shared memory.
// Up to P chunks (BF16, FP8) or 512-k blocks (INT4) of weights are in flight a lane.
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

// The warp's place: its tile (8 rows of one segment), k part, rows of x. Shared by both kernels.
struct Place {
  SegDev sg;
  int tile, kpart, lt, n0;
  bool valid, live0, live1;
  const uint16_t *x0, *x1;
};
template <bool HI, int NWP>
__device__ __forceinline__ Place place(const uint16_t* __restrict__ x, int ldx, int rows, const Segs& segs, int splitk) {
  Place p;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int per_cta = NWP / splitk;
  p.tile = blockIdx.x * per_cta + warp / splitk;
  p.kpart = warp % splitk;
  p.valid = p.tile < segs.tile_end;
  int si = 0;
#pragma unroll
  for (int i = 1; i < kMaxSegs; ++i)
    if (i < segs.n && p.tile >= segs.s[i].tile0) si = i;
  p.sg = segs.s[si];
  p.lt = p.valid ? p.tile - p.sg.tile0 : 0;
  p.n0 = p.lt * 8;
  const uint16_t* xb = x + (p.sg.group_rows ? (size_t)(p.n0 / p.sg.group_rows) * p.sg.x_group : 0) + 16 * t;
  p.live0 = g < rows;
  p.live1 = HI && g + 8 < rows;
  p.x0 = xb + (size_t)(p.live0 ? g : 0) * ldx;
  p.x1 = xb + (size_t)(p.live1 ? g + 8 : 0) * ldx;
  return p;
}

// x's fragments of chunk j (rows g, g + 8; k = 64j + 16t .. + 15)
__device__ __forceinline__ void load_x(const Place& p, int j, uint4& xa, uint4& xc, uint4& ya, uint4& yc) {
  xa = xc = ya = yc = make_uint4(0, 0, 0, 0);
  if (p.live0) {
    xa = __ldg(reinterpret_cast<const uint4*>(p.x0 + j * 64));
    xc = __ldg(reinterpret_cast<const uint4*>(p.x0 + j * 64 + 8));
  }
  if (p.live1) {
    ya = __ldg(reinterpret_cast<const uint4*>(p.x1 + j * 64));
    yc = __ldg(reinterpret_cast<const uint4*>(p.x1 + j * 64 + 8));
  }
}
// the four MMAs of a chunk accumulating into acc (BF16 and INT4; the order of the BF16 GEMV)
__device__ __forceinline__ void mma_chunk(float (&acc)[4], const uint32_t (&o)[8], const uint4& xa, const uint4& xc,
                                          const uint4& ya, const uint4& yc) {
  mma_bf16(acc, xa.x, ya.x, xa.y, ya.y, o[0], o[1]);
  mma_bf16(acc, xa.z, ya.z, xa.w, ya.w, o[2], o[3]);
  mma_bf16(acc, xc.x, yc.x, xc.y, yc.y, o[4], o[5]);
  mma_bf16(acc, xc.z, yc.z, xc.w, yc.w, o[6], o[7]);
}

// split-K partials added in warp order, then the outputs (bf16, f32 or SwiGLU)
template <int OUT, int NWP>
__device__ __forceinline__ void finish(float (&red)[NWP][32][4], float (&acc)[4], const Place& p, int splitk) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (splitk > 1) {
    if (p.kpart) {
#pragma unroll
      for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
    }
    __syncthreads();
    if (p.kpart) return;
    for (int k = 1; k < splitk; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = __fadd_rn(acc[i], red[warp + k][lane][i]);
  }
  if (!p.valid) return;
  const SegDev& sg = p.sg;
  const int n = p.n0 + 2 * t;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    if (h ? !p.live1 : !p.live0) continue;
    const float a0 = acc[2 * h], a1 = acc[2 * h + 1];
    if constexpr (OUT == (int)GemvOut::kBF16) {
      *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(sg.y) + (size_t)r * sg.ldy + n) = pack_bf16(a0, a1);
    } else if constexpr (OUT == (int)GemvOut::kF32) {
      *reinterpret_cast<float2*>(static_cast<float*>(sg.y) + (size_t)r * sg.ldy + n) = make_float2(a0, a1);
    } else {  // SwiGLU: column 2t is gate i, 2t+1 up i, i = n0/2 + t
      const float gt = bfround(a0), up = bfround(a1);
      const float sl = bfround(__fdiv_rn(gt, __fadd_rn(1.f, expf(-gt))));
      static_cast<uint16_t*>(sg.y)[(size_t)r * sg.ldy + p.n0 / 2 + t] = f2bf(__fmul_rn(sl, up));
    }
  }
}

// ---- FP8 and BF16 ---------------------------------------------------------------------------------------------
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
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const Place pl = place<HI, NWP>(x, ldx, rows, segs, splitk);
  const SegDev& sg = pl.sg;
  const int n0 = pl.n0;
  const int J = K >> 6;
  const int j0 = (J * pl.kpart) / splitk, j1 = (J * (pl.kpart + 1)) / splitk;
  // weights
  const uint8_t* wp = FP8 ? sg.w + ((size_t)pl.lt * J) * 512 + lane * 16
                          : sg.w + ((size_t)(n0 + g) * K + 16 * t) * 2;
  // scale row: the tile's block; SwiGLU: lane t's intermediate (n0/2 + t)'s gate block row (up follows it)
  const float* sp = !FP8 ? nullptr
                    : OUT == (int)GemvOut::kSwiGLU ? sg.s + (size_t)(((n0 >> 1) + t) >> sg.lrs) * 2 * (K >> sg.lks)
                                                   : sg.s + (size_t)(n0 >> sg.lrs) * (K >> sg.lks);

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
  if (pl.valid) {
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
    uint4 xa, xc, ya, yc;
    load_x(pl, j, xa, xc, ya, yc);
    if constexpr (FP8) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      mma_chunk(c, o, xa, xc, ya, yc);
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
      mma_chunk(acc, o, xa, xc, ya, yc);
    }
  };
  if (pl.valid) {
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
  finish<OUT, NWP>(red, acc, pl, splitk);
}

// ---- INT4 -----------------------------------------------------------------------------------------------------
// A warp's tile is a run of 512-k blocks (pack.py int4_tiles): per block lane (g, t) loads its four 16-byte value
// units (unit jj: chunks 8b + 2jj and 8b + 2jj + 1, 16 nibbles each), its 16 bytes of scales (the 8 groups of its
// 8 chunks, shared with lane t ^ 1) and its 4 bytes of zero-point nibbles. P blocks in flight a lane.
struct Int4Blk {
  uint4 v[4];
  uint4 s;
  uint32_t z;
};
template <int OUT, bool HI, int P, int NWP>
__global__ void __launch_bounds__(NWP * 32) k_gemv4(const uint16_t* __restrict__ x, int ldx, int rows, int K, Segs segs,
                                                   int splitk) {
  __shared__ float red[NWP][32][4];
  pdl_trigger();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const Place pl = place<HI, NWP>(x, ldx, rows, segs, splitk);
  const int JB = K >> 9;
  const int b0 = (JB * pl.kpart) / splitk, b1 = (JB * (pl.kpart + 1)) / splitk;
  const uint8_t* wp = pl.sg.w + (size_t)pl.lt * JB * kInt4Blk;
  const int vo = 16 * lane, so = 2048 + 32 * g + 16 * (t >> 1), zo = 2304 + 8 * g + 4 * (t >> 1);
  Int4Blk buf[P];
  auto load = [&](Int4Blk& B, int b) {
    const uint8_t* p = wp + (size_t)b * kInt4Blk;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) B.v[jj] = ldg_stream128(p + 512 * jj + vo);
    B.s = ldg_stream128(p + so);
    B.z = ldg_stream32(p + zo);
  };
  if (pl.valid) {
#pragma unroll
    for (int u = 0; u < P; ++u)
      if (b0 + u < b1) load(buf[u], b0 + u);
  }
  pdl_wait();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  auto comp = [&](const Int4Blk& B, int b) {
    const uint32_t sw[4] = {B.s.x, B.s.y, B.s.z, B.s.w};  // unit jj's scales: chunk q = 0 low, q = 1 high
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const uint32_t lov[2] = {B.v[jj].x, B.v[jj].z}, hiv[2] = {B.v[jj].y, B.v[jj].w};
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const uint32_t ss = prmt(sw[jj], 0, q ? 0x3232u : 0x1010u);
        const uint32_t zz = int4_zz((B.z >> (8 * jj + 4 * q)) & 0xFu);
        uint32_t o[8];
        int4x16_to_bf16(lov[q], hiv[q], zz, ss, o);
        uint4 xa, xc, ya, yc;
        load_x(pl, 8 * b + 2 * jj + q, xa, xc, ya, yc);
        mma_chunk(acc, o, xa, xc, ya, yc);
      }
    }
  };
  if (pl.valid) {
    for (int b = b0; b < b1; b += P) {
#pragma unroll
      for (int u = 0; u < P; ++u) {
        const int bb = b + u;
        if (bb < b1) {
          const Int4Blk B = buf[u];
          if (bb + P < b1) load(buf[u], bb + P);
          comp(B, bb);
        }
      }
    }
  }
  finish<OUT, NWP>(red, acc, pl, splitk);
}

// ---- INT4 -> bf16 (the prefill's weights; gemm.cu) ------------------------------------------------------------
// one thread a (lane, unit) of a block: 32 values, the GEMV's arithmetic; out row-major [N][ld]
__global__ void __launch_bounds__(256) k_dequant_int4(const uint8_t* __restrict__ w, int N, int K, uint16_t* __restrict__ out,
                                                      size_t ld) {
  pdl_trigger();
  const int JB = K >> 9;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // (block, jj, lane)
  const size_t nblk = (size_t)(N / 8) * JB;
  pdl_wait();
  if (i >= nblk * 128) return;
  const int lane = (int)(i & 31), jj = (int)((i >> 5) & 3);
  const size_t blk = i >> 7;
  const int n8 = (int)(blk / JB), b = (int)(blk % JB), g = lane >> 2, t = lane & 3;
  const uint8_t* p = w + blk * kInt4Blk;
  const uint4 v = ldg_stream128(p + 512 * jj + 16 * lane);
  const uint32_t sw = ldg_stream32(p + 2048 + 32 * g + 16 * (t >> 1) + 4 * jj);
  const uint32_t zw = ldg_stream32(p + 2304 + 8 * g + 4 * (t >> 1));
  const uint32_t lov[2] = {v.x, v.z}, hiv[2] = {v.y, v.w};
  uint16_t* row = out + (size_t)(8 * n8 + g) * ld;
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    uint32_t o[8];
    int4x16_to_bf16(lov[q], hiv[q], int4_zz((zw >> (8 * jj + 4 * q)) & 0xFu), prmt(sw, 0, q ? 0x3232u : 0x1010u), o);
    uint4* d = reinterpret_cast<uint4*>(row + 512 * b + 128 * jj + 64 * q + 16 * t);
    d[0] = make_uint4(o[0], o[1], o[2], o[3]);
    d[1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// ---- launches ---------------------------------------------------------------------------------------------------
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
template <int OUT, bool HI, int P, int NW>
void launch_gemv4_cfg(const uint16_t* x, int ldx, int rows, int K, const Segs& sd, int splitk, cudaStream_t s, bool pdl) {
  const int per_cta = NW / splitk;
  const int grid = (sd.tile_end + per_cta - 1) / per_cta;
  launch(k_gemv4<OUT, HI, P, NW>, dim3(grid), dim3(NW * 32), 0, s, pdl, x, ldx, rows, K, sd, splitk);
  CUDA_CHECK(cudaGetLastError());
}

int g_gemv_cfg = -1;  // a bench knob: loads in flight a lane and warps a CTA (bit-neutral); -1: by shape

bool grouped_of(const Segs& sd) {
  bool grouped = false;
  for (int i = 0; i < sd.n; ++i) grouped = grouped || sd.s[i].group_rows > 0;
  return grouped;
}

template <bool FP8, int OUT, bool HI>
void launch_gemv(const uint16_t* x, int ldx, int rows, int K, const Segs& sd, int splitk, const IwDev* iw,
                 cudaStream_t s, bool pdl) {
  if (iw) {  // 128-thread CTAs (the index weights' emulation)
    launch_gemv_cfg<FP8, OUT, HI, FP8 ? 8 : 4, 4, true>(x, ldx, rows, K, sd, splitk, *iw, s, pdl);
    return;
  }
  const IwDev none{};
  int want = g_gemv_cfg;
  if constexpr (FP8) {
    if (want < 0) want = grouped_of(sd) && K >= 512 ? 1 : 4;  // measured on GB10: 16 loads a lane for the grouped W_UV
    const int cfg = splitk <= 2 ? want : (want <= 2 || want == 4) ? want : 0;
    switch (cfg) {
      case 1: launch_gemv_cfg<true, OUT, HI, 16, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 2: launch_gemv_cfg<true, OUT, HI, 24, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 3: launch_gemv_cfg<true, OUT, HI, 8, 2, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 4: launch_gemv_cfg<true, OUT, HI, 8, 8, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 5: launch_gemv_cfg<true, OUT, HI, 16, 2, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      default: launch_gemv_cfg<true, OUT, HI, 8, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
    }
  } else {
    if (want < 0) {  // by shape, measured on GB10 (docs/attention.md § 5): bit-neutral
      const size_t N = (size_t)sd.tile_end * 8;
      want = K <= 512 ? 2 : N * K <= ((size_t)1 << 21) ? 4 : N >= 16384 ? 0 : 1;
    }
    switch (want) {
      case 1: launch_gemv_cfg<false, OUT, HI, 8, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 2: launch_gemv_cfg<false, OUT, HI, 4, 8, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 3: launch_gemv_cfg<false, OUT, HI, 8, 8, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 4: launch_gemv_cfg<false, OUT, HI, 16, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      case 5: launch_gemv_cfg<false, OUT, HI, 2, 8, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
      default: launch_gemv_cfg<false, OUT, HI, 4, 4, false>(x, ldx, rows, K, sd, splitk, none, s, pdl); return;
    }
  }
}

template <int OUT, bool HI>
void launch_gemv4(const uint16_t* x, int ldx, int rows, int K, const Segs& sd, int splitk, cudaStream_t s, bool pdl) {
  int want = g_gemv_cfg;
  if (want < 0) want = 1;  // until measured
  switch (want) {
    case 0: launch_gemv4_cfg<OUT, HI, 1, 4>(x, ldx, rows, K, sd, splitk, s, pdl); return;
    case 2: launch_gemv4_cfg<OUT, HI, 1, 8>(x, ldx, rows, K, sd, splitk, s, pdl); return;
    case 3: launch_gemv4_cfg<OUT, HI, 2, 8>(x, ldx, rows, K, sd, splitk, s, pdl); return;
    case 4: launch_gemv4_cfg<OUT, HI, 3, 4>(x, ldx, rows, K, sd, splitk, s, pdl); return;
    case 5: launch_gemv4_cfg<OUT, HI, 4, 4>(x, ldx, rows, K, sd, splitk, s, pdl); return;
    default: launch_gemv4_cfg<OUT, HI, 2, 4>(x, ldx, rows, K, sd, splitk, s, pdl); return;
  }
}

const char* fmt_name(WFmt f) {
  return f == WFmt::kBF16 ? "BF16" : f == WFmt::kINT4 ? "INT4" : f == WFmt::kFP8 ? "FP8" : "unset";
}

// one launch: segments of one format
void gemv_one(WFmt fmt, const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out,
              int splitk, cudaStream_t s, bool pdl, const GemvIw* iw) {
  const bool fp8 = fmt == WFmt::kFP8, int4 = fmt == WFmt::kINT4;
  if (int4 && (K % kInt4BlkK || (K / kInt4BlkK) % splitk))
    die("gemv: INT4 K %d (a multiple of %d, K / %d a multiple of split-K %d)", K, kInt4BlkK, kInt4BlkK, splitk);
  Segs sd{};
  int t0 = 0;
  for (int i = 0; i < nseg; ++i) {
    const GemvSeg& g = segs[i];
    if (g.n % 8) die("gemv: segment %d has %d rows", i, g.n);
    SegDev& d = sd.s[i];
    d.w = static_cast<const uint8_t*>(g.w);
    d.s = fp8 ? g.s : nullptr;
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
    if (int4 || out != GemvOut::kBF16 || K != kDim) die("gemv: index weights with a %s GEMV of K %d", fmt_name(fmt), K);
    iwd.wp = iw->wproj;
    iwd.w = iw->w;
    iwd.ldw = iw->ldw;
    iwd.nh = iw->nh;
  }
  const IwDev* ip = iw ? &iwd : nullptr;
#define GL(F, O)                                                                    \
  (hi ? launch_gemv<F, (int)O, true>(x, ldx, rows, K, sd, splitk, ip, s, pdl)     \
      : launch_gemv<F, (int)O, false>(x, ldx, rows, K, sd, splitk, ip, s, pdl))
#define GL4(O) (hi ? launch_gemv4<(int)O, true>(x, ldx, rows, K, sd, splitk, s, pdl) \
                   : launch_gemv4<(int)O, false>(x, ldx, rows, K, sd, splitk, s, pdl))
  if (int4) {
    if (out == GemvOut::kBF16) GL4(GemvOut::kBF16);
    else if (out == GemvOut::kF32) GL4(GemvOut::kF32);
    else GL4(GemvOut::kSwiGLU);
  } else if (fp8) {
    if (out == GemvOut::kBF16) GL(true, GemvOut::kBF16);
    else if (out == GemvOut::kF32) GL(true, GemvOut::kF32);
    else GL(true, GemvOut::kSwiGLU);
  } else {
    if (out == GemvOut::kBF16) GL(false, GemvOut::kBF16);
    else if (out == GemvOut::kF32) GL(false, GemvOut::kF32);
    else GL(false, GemvOut::kSwiGLU);
  }
#undef GL4
#undef GL
}

}  // namespace

void gemv_config(int cfg) { g_gemv_cfg = cfg; }

void gemv(const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
          cudaStream_t s, bool pdl, const GemvIw* iw) {
  if (rows < 1 || rows > kMaxRows) die("gemv: %d rows", rows);
  if (K % 64 || nseg < 1 || nseg > kMaxSegs) die("gemv: K %d, %d segments", K, nseg);
  if (splitk != 1 && splitk != 2 && splitk != 4) die("gemv: splitk %d", splitk);
  bool iw_done = false;
  for (WFmt f : {WFmt::kBF16, WFmt::kINT4, WFmt::kFP8}) {
    GemvSeg g[kMaxSegs];
    int n = 0;
    for (int i = 0; i < nseg; ++i) {
      if (segs[i].fmt != WFmt::kBF16 && segs[i].fmt != WFmt::kINT4 && segs[i].fmt != WFmt::kFP8)
        die("gemv: segment %d has no weight format (%d)", i, (int)segs[i].fmt);
      if (segs[i].fmt == f) g[n++] = segs[i];
    }
    if (!n) continue;
    const bool with_iw = iw && !iw_done && f != WFmt::kINT4;
    gemv_one(f, x, ldx, rows, K, g, n, out, splitk, s, pdl, with_iw ? iw : nullptr);
    iw_done = iw_done || with_iw;
  }
  if (iw && !iw_done) die("gemv: index weights need a BF16 or FP8 segment");
}

void gemv(bool fp8, const uint16_t* x, int ldx, int rows, int K, const GemvSeg* segs, int nseg, GemvOut out, int splitk,
          cudaStream_t s, bool pdl, const GemvIw* iw) {
  if (nseg < 1 || nseg > kMaxSegs) die("gemv: %d segments", nseg);
  GemvSeg g[kMaxSegs];
  for (int i = 0; i < nseg; ++i) {
    g[i] = segs[i];
    if (g[i].fmt == WFmt::kNone) g[i].fmt = fp8 ? WFmt::kFP8 : WFmt::kBF16;
  }
  gemv(x, ldx, rows, K, g, nseg, out, splitk, s, pdl, iw);
}

void dequant_int4(const void* tiles, int N, int K, uint16_t* out, size_t ld, cudaStream_t s, bool pdl) {
  if (N % 8 || K % kInt4BlkK || ld < (size_t)K || ld % 8) die("dequant_int4: [%d][%d], ld %zu", N, K, ld);
  const size_t threads = (size_t)(N / 8) * (K / kInt4BlkK) * 128;
  launch(k_dequant_int4, dim3((unsigned)((threads + 255) / 256)), dim3(256), 0, s, pdl, static_cast<const uint8_t*>(tiles),
         N, K, out, ld);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace glm
