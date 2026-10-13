// mHC for prefill chunks (M5): the residual stream's work at a sublayer boundary in one pass.
//
// Between two sublayers model.py's Block.forward does, per row: y = hc_post(sub, res, post, comb) (the
// sublayer output back into the 4-copy stream), x = norm(hc_pre(y, pre)) (the next sublayer's input) and
// the next coefficients, Sinkhorn(hc_fn . y * rsqrt(mean(y^2) + eps)). Single-pass mHC takes pre, post and
// comb from the previous boundary, so all three need only y and one pass over the stream does them all.
// A CTA owns R rows (32, or 16, 8, 4 for small chunks, whose few 32-row CTAs leave most SMs idle) and
// streams sub and res through shared memory in blocks of 64 columns (of each copy), 96 / R blocks deep
// (at most 12): it writes y and the unnormalised x, sums x^2 in attn::hc_pre's order and projects y on
// tensor cores, hc_fn's fp32 weights split into three bf16 parts (exact), bf16 MMA with fresh fp32 partial
// sums per block. Every CTA reads all 2.9 MB of parts from L2, which costs (16-row CTAs: +20 % on the
// boundary at 8192 rows), and each warp loads its parts for the next block into registers while the
// current one streams. Then the Sinkhorn per row (4 lanes per row) and the norm of x, re-read by the warp
// that wrote it, act_quant optional. A row's thread roles and sums are the same for every R: the same bits.
// Optionally the CTA prefetches into L2 the parts and its rows of the block pfd blocks ahead: a chunk of few
// CTAs pulls hc_fn's parts (cold: each boundary has its own) and res from DRAM only a block or two ahead.
// hc_post, hc_pre, the norm and the Sinkhorn are attn's op for op (bit-identical); the projection
// differs from attn::hc_partial (and cuBLAS) in summation order only.
#include "attn/attn_common.cuh"
#include "common/check.h"
#include "kernels/common.cuh"
#include "kernels/pa_common.cuh"
#include "kernels/prefill_attn.h"

namespace dsk {
namespace {

using attn::kDim;
using attn::kHcDim;

constexpr int kBCols = 64;                       // columns of each copy per block
constexpr int kBThreads = 256;
constexpr int kNBlk = kDim / kBCols;             // 80 blocks
constexpr int kEPerRow = kBCols / 8;             // elementwise threads per row, 8 columns each
constexpr int kYStride = 4 * kBCols * 2 + 16;    // y tile row (bytes): ldmatrix without bank conflicts
constexpr int kFnPairs = kHcDim / 32;            // pairs of k16 steps
constexpr int kFnU32 = 36;                       // per lane and pair: 2 steps x 3 n-tiles x 3 parts x 2
constexpr int kFnU4 = kFnU32 / 4;
static_assert(32 * kEPerRow == kBThreads && 4 * kBCols == 32 * (kBThreads / 32), "a pair of k16 steps per warp");
static_assert(kBCols == 64 && kNBlk % 4 == 0, "the norm's column classes (below)");

template <int R>
struct BSmem {
  static constexpr int S = R >= 32 ? 3 : R == 16 ? 6 : 12;  // stages: 60 KB of the stream in flight (R >= 8)
  static constexpr int MT = (R + 15) / 16;                   // m16 tiles of the projection
  uint8_t stage[S][5 * R * kBCols * 2];  // sub + the 4 copies of res; reused for the warps' projection partials
  uint8_t ytile[16 * MT * kYStride];
  float mix[R][24];                      // pre, post, comb of the previous boundary
  float raw[R][25];                      // projection, sum of squares
  static_assert(sizeof(float) * 8 * 16 * MT * 24 <= sizeof(stage), "partials alias");
};

// The stream is read and written once: evict_first, so that x (re-read at the end) and hc_fn's parts (read
// by every CTA) stay in L2.
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// (ptxas 13.0 gave cp.async with a cache hint here a misaligned uniform descriptor, desc[UR1]: illegal
// instruction. The loads go without the hint.)
__device__ __forceinline__ void cp16_pol(void* dst, const void* src, bool pred, uint64_t) { pa::cp16(dst, src, pred); }
__device__ __forceinline__ void st16_pol(void* p, uint4 v, uint64_t pol) {
  asm volatile("st.global.L2::cache_hint.v4.u32 [%0], {%1,%2,%3,%4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void unpack8(const uint4& u, float (&v)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) v[2 * i] = __uint_as_float(w[i] << 16), v[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
}
// 8 values that are bf16 already -> packed bf16 (exact)
__device__ __forceinline__ uint4 pack8(const float (&v)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) w[i] = (__float_as_uint(v[2 * i]) >> 16) | (__float_as_uint(v[2 * i + 1]) & 0xffff0000u);
  return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ float sigmoidf_ref(float v) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v))); }
// torch's 4-element sums (attn/mhc.cu): over the last dim (a+c)+(b+d), over dim -2 ((a+b)+c)+d
__device__ __forceinline__ float sum4_last(float a, float b, float c, float d) {
  return __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
}
__device__ __forceinline__ float sum4_seq(float a, float b, float c, float d) {
  return __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
}

// fn [24][20480] -> three bf16 parts (hi, mid, lo: their sum is fn exactly) as B fragments of
// m16n8k16 (n = output): per pair of k16 steps and lane 36 words [step u][n-tile][part][b0, b1], stored
// as [pair][uint4 of the 36 words][lane] so that a warp's loads coalesce.
__global__ void hc_fn_parts_kernel(const float* __restrict__ fn, uint32_t* __restrict__ parts) {
  const int n = kFnPairs * 32 * kFnU32;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
    const int i = idx % kFnU32, lane = (idx / kFnU32) % 32, pair = idx / (kFnU32 * 32);
    const int u = i / 18, nt = (i % 18) / 6, p = (i % 6) / 2, reg = i & 1;
    const int g = lane >> 2, t = lane & 3, o = 8 * nt + g, k = 16 * (2 * pair + u) + 2 * t + 8 * reg;
    uint32_t w = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float f = fn[(size_t)o * kHcDim + k + h];
      const float hi = attn::bfround(f), r = __fsub_rn(f, hi), mid = attn::bfround(r), lo = attn::bfround(__fsub_rn(r, mid));
      const float v = p == 0 ? hi : p == 1 ? mid : lo;
      w |= (__float_as_uint(v) >> 16) << (16 * h);
    }
    parts[(((size_t)pair * kFnU4 + i / 4) * 32 + lane) * 4 + (i & 3)] = w;
  }
}

template <int R, bool POST>
__global__ void __launch_bounds__(kBThreads) hc_boundary_kernel(HcBoundary a, int rows, int pfd) {
  constexpr int kBRows = R, kBStages = BSmem<R>::S, kMT = BSmem<R>::MT, kRowsP = 16 * kMT;
  constexpr int kSubBytes = kBRows * kBCols * 2;  // sub per block
  extern __shared__ __align__(16) uint8_t smem_raw[];
  BSmem<R>& sm = *reinterpret_cast<BSmem<R>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * kBRows, nr = min(kBRows, rows - r0);
  const bool proj = a.fn_parts != nullptr;
  for (int i = tid; i < kBRows * 24; i += kBThreads) {
    const int rr = i / 24, k = i % 24, r = r0 + rr;
    float v = 0.f;
    if (rr < nr) v = k < 4 ? a.pre[r * 4 + k] : !POST ? 0.f : k < 8 ? a.post[r * 4 + k - 4] : a.comb[r * 16 + k - 8];
    sm.mix[rr][k] = v;
  }
  if constexpr (kBRows < kRowsP)  // the m16 tile's rows past R: zeros, never read back
    for (int i = tid; i < (kRowsP - kBRows) * kYStride / 16; i += kBThreads)
      reinterpret_cast<uint4*>(sm.ytile + kBRows * kYStride)[i] = make_uint4(0, 0, 0, 0);
  const uint64_t pol_stream = policy_evict_first();
  auto load = [&](int cb, int st) {
    uint8_t* s0 = sm.stage[st];
    constexpr int kSubChunks = kBRows * kEPerRow;
#pragma unroll
    for (int i = POST ? tid : tid + kSubChunks; i < 5 * kSubChunks; i += kBThreads) {
      if (i < kSubChunks) {
        const int rr = i / kEPerRow, c = i % kEPerRow;
        const bool ok = rr < nr;
        cp16_pol(s0 + rr * (kBCols * 2) + c * 16, a.sub + (size_t)(ok ? r0 + rr : 0) * kDim + cb * kBCols + c * 8, ok,
                 pol_stream);
      } else {
        const int i2 = i - kSubChunks, rr = i2 / (4 * kEPerRow), j = (i2 / kEPerRow) & 3, c = i2 % kEPerRow;
        const bool ok = rr < nr;
        cp16_pol(s0 + kSubBytes + (rr * 4 + j) * (kBCols * 2) + c * 16,
                 a.res + (size_t)(ok ? r0 + rr : 0) * kHcDim + j * kDim + cb * kBCols + c * 8, ok, pol_stream);
      }
    }
  };

  // elementwise slot: row er (a warp has 4 rows), columns 8 ec .. 8 ec + 7 of each block; projection slot:
  // warp w takes copy w / 2, columns 32 (w & 1) .. + 31 of each block (a pair of k16 steps), both m16 tiles
  const int er = tid / kEPerRow, ec = tid % kEPerRow, pj = warp >> 1, ph = warp & 1;
  const bool erow = er < nr;
  uint4 fr[kFnU4];  // the warp's parts for the current block
  auto fn_load = [&](int cb) {
    const uint4* fp = reinterpret_cast<const uint4*>(a.fn_parts) +
                      (size_t)((pj * kDim + cb * kBCols + ph * 32) >> 5) * kFnU4 * 32 + lane;
#pragma unroll
    for (int i = 0; i < kFnU4; ++i) fr[i] = __ldg(fp + i * 32);
  };
  if (proj) fn_load(0);
  float acc[kMT][3][4] = {};
  float ssy = 0.f, sx[4][8] = {};  // y^2; x^2 by column class, the current block's first (the norm's order)
#pragma unroll
  for (int s = 0; s < kBStages - 1; ++s) {
    load(s, s);
    pa::cp_commit();
  }
#pragma unroll 1
  for (int cb = 0; cb < kNBlk; ++cb) {
    pa::cp_wait<kBStages - 2>();
    __syncthreads();
    if (cb + kBStages - 1 < kNBlk) load(cb + kBStages - 1, (cb + kBStages - 1) % kBStages);
    pa::cp_commit();
    if (pfd && cb + pfd < kNBlk) {  // block cb + pfd into L2: each copy's two pairs of parts (72 lines), the rows
      const int pb = cb + pfd;
      for (int i = tid; i < 4 * 72 + 5 * kBRows; i += kBThreads) {
        const void* p;
        if (i < 4 * 72) {
          if (!proj) continue;
          p = reinterpret_cast<const uint8_t*>(a.fn_parts) + (size_t)((i / 72) * (kDim / 32) + 2 * pb) * (kFnU4 * 512) +
              (i % 72) * 128;
        } else {
          const int rr = (i - 4 * 72) / 5, j = (i - 4 * 72) % 5;
          if (rr >= nr || (j == 4 && !POST)) continue;
          p = j < 4 ? a.res + (size_t)(r0 + rr) * kHcDim + j * kDim + pb * kBCols : a.sub + (size_t)(r0 + rr) * kDim + pb * kBCols;
        }
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
      }
    }
    const uint8_t* s0 = sm.stage[cb % kBStages];
    if (er < kBRows) {
      const float* m = sm.mix[er];
      float y[4][8];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        unpack8(*reinterpret_cast<const uint4*>(s0 + kSubBytes + (er * 4 + j) * (kBCols * 2) + ec * 16), y[j]);
      if constexpr (POST) {  // y[j] = bf16(post[j] sub + (((c0j x0 + c1j x1) + c2j x2) + c3j x3)), attn::hc_post
        float sf[8];
        unpack8(*reinterpret_cast<const uint4*>(s0 + er * (kBCols * 2) + ec * 16), sf);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float x0 = y[0][e], x1 = y[1][e], x2 = y[2][e], x3 = y[3][e];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float v = __fadd_rn(__fmul_rn(m[8 + j], x0), __fmul_rn(m[12 + j], x1));
            v = __fadd_rn(v, __fmul_rn(m[16 + j], x2));
            v = __fadd_rn(v, __fmul_rn(m[20 + j], x3));
            y[j][e] = attn::bfround(__fadd_rn(__fmul_rn(m[4 + j], sf[e]), v));
          }
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint4 pk = pack8(y[j]);
        if (POST && erow && a.y) st16_pol(a.y + (size_t)(r0 + er) * kHcDim + j * kDim + cb * kBCols + ec * 8, pk, pol_stream);
        *reinterpret_cast<uint4*>(sm.ytile + er * kYStride + (j * kBCols + ec * 8) * 2) = pk;
#pragma unroll
        for (int e = 0; e < 8; ++e) ssy = __fmaf_rn(y[j][e], y[j][e], ssy);
      }
      // x = bf16(((p0 y0 + p1 y1) + p2 y2) + p3 y3), attn::hc_pre (normalised at the end)
      float xv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float v = __fadd_rn(__fmul_rn(m[0], y[0][e]), __fmul_rn(m[1], y[1][e]));
        v = __fadd_rn(v, __fmul_rn(m[2], y[2][e]));
        v = __fadd_rn(v, __fmul_rn(m[3], y[3][e]));
        xv[e] = attn::bfround(v);
        sx[0][e] = __fadd_rn(sx[0][e], __fmul_rn(xv[e], xv[e]));
        const float s3 = sx[0][e];
        sx[0][e] = sx[1][e], sx[1][e] = sx[2][e], sx[2][e] = sx[3][e], sx[3][e] = s3;
      }
      if (erow) *reinterpret_cast<uint4*>(a.x_out + (size_t)(r0 + er) * kDim + cb * kBCols + ec * 8) = pack8(xv);
    }
    if (!proj) continue;
    __syncthreads();
    {
      float part[kMT][3][4] = {};
      const uint32_t* fw = reinterpret_cast<const uint32_t*>(fr);
#pragma unroll
      for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
          uint32_t af[4];
          pa::ldsm_x4(af, sm.ytile + ((lane & 15) + 16 * mt) * kYStride +
                              (pj * kBCols + ph * 32 + u * 16 + (lane >> 4) * 8) * 2);
#pragma unroll
          for (int nt = 0; nt < 3; ++nt)
#pragma unroll
            for (int p = 2; p >= 0; --p)  // smallest part first
              pa::mma_bf16(part[mt][nt], af, fw[((u * 3 + nt) * 3 + p) * 2], fw[((u * 3 + nt) * 3 + p) * 2 + 1]);
        }
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 3; ++nt)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[mt][nt][i] = __fadd_rn(acc[mt][nt][i], part[mt][nt][i]);
    }
    if (cb + 1 < kNBlk) fn_load(cb + 1);
  }
  pa::cp_wait_all();
  __syncthreads();

  // The norm's sum of squares, attn::hc_pre's order: its thread 32 w + l adds the columns 32 w + l + 256 i
  // over i, then warp_sum (lanes xor 16, 8, 4, 2, 1), then the 8 warps in order. This thread holds columns
  // 64 h + 8 ec + e of the blocks of class h = cb & 3, so w = 2 h + ec / 4 and l = 8 (ec & 3) + e: xor 16
  // and 8 are lanes xor 2 and 1 here, xor 4, 2 and 1 are within e. rsx: rs of the row (the warp's 4 rows).
  float rsx = 0.f;
  if (a.norm_w) {
    float r[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float sv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) sv[e] = sx[h][e];
#pragma unroll
      for (int o = 2; o > 0; o >>= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e) sv[e] = __fadd_rn(sv[e], __shfl_xor_sync(0xffffffffu, sv[e], o));
#pragma unroll
      for (int o = 4; o > 0; o >>= 1)
#pragma unroll
        for (int e = 0; e < o; ++e) sv[e] = __fadd_rn(sv[e], sv[e + o]);
      r[h] = sv[0];
    }
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, __shfl_sync(0xffffffffu, r[w >> 1], (lane & 24) + 4 * (w & 1)));
    rsx = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / kDim), attn::kNormEps));
  }

  if (proj) {
    // sum of squares per row (a row's 8 threads are 8 lanes); the warps' partials, summed in warp order
#pragma unroll
    for (int o = kEPerRow / 2; o > 0; o >>= 1) ssy += __shfl_xor_sync(0xffffffffu, ssy, o);
    if (ec == 0 && er < kBRows) sm.raw[er][24] = ssy;
    float(*red)[kRowsP][24] = reinterpret_cast<float(*)[kRowsP][24]>(&sm.stage[0][0]);
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        red[warp][16 * mt + g][8 * nt + 2 * t] = acc[mt][nt][0];
        red[warp][16 * mt + g][8 * nt + 2 * t + 1] = acc[mt][nt][1];
        red[warp][16 * mt + g + 8][8 * nt + 2 * t] = acc[mt][nt][2];
        red[warp][16 * mt + g + 8][8 * nt + 2 * t + 1] = acc[mt][nt][3];
      }
    __syncthreads();
    for (int i = tid; i < kBRows * 24; i += kBThreads) {
      const int rr = i / 24, o = i % 24;
      float s = red[0][rr][o];
#pragma unroll
      for (int w = 1; w < 8; ++w) s = __fadd_rn(s, red[w][rr][o]);
      sm.raw[rr][o] = s;
    }
    __syncthreads();
    // attn::hc_finish with the 4x4 matrix spread over a lane quad (lane j of the quad holds row j); whole
    // warps run (at least one), rows past nr only compute (rows past R: row R - 1's)
    if (tid < (kBRows < 8 ? 32 : 4 * kBRows)) {
      const int rr = tid >> 2, j = tid & 3, r = r0 + rr;
      const bool valid = rr < nr;
      const float* p = sm.raw[rr < kBRows ? rr : kBRows - 1];
      if (valid && a.proj_out && j == 0)
        for (int i = 0; i < 25; ++i) a.proj_out[(size_t)r * 25 + i] = p[i];
      const float rs = rsqrtf(__fadd_rn(__fmul_rn(p[24], 1.0f / kHcDim), attn::kNormEps));
      const float s0 = a.scale[0], s1 = a.scale[1], s2 = a.scale[2];
      const float pre = __fadd_rn(sigmoidf_ref(__fadd_rn(__fmul_rn(__fmul_rn(p[j], rs), s0), a.base[j])), attn::kHcEps);
      const float post = __fmul_rn(2.0f, sigmoidf_ref(__fadd_rn(__fmul_rn(__fmul_rn(p[4 + j], rs), s1), a.base[4 + j])));
      if (valid) a.pre_out[r * 4 + j] = pre, a.post_out[r * 4 + j] = post;
      float c[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] = __fadd_rn(__fmul_rn(__fmul_rn(p[8 + 4 * j + k], rs), s2), a.base[8 + 4 * j + k]);
      {
        const float mx = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));
        float e[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) e[k] = expf(__fsub_rn(c[k], mx));
        const float sum = sum4_last(e[0], e[1], e[2], e[3]);
#pragma unroll
        for (int k = 0; k < 4; ++k) c[k] = __fadd_rn(__fdiv_rn(e[k], sum), attn::kHcEps);
      }
      const int qb = lane & ~3;
      auto colnorm = [&]() {
        float col[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int k = 0; k < 4; ++k) col[i][k] = __shfl_sync(0xffffffffu, c[k], qb + i);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float d = __fadd_rn(sum4_seq(col[0][k], col[1][k], col[2][k], col[3][k]), attn::kHcEps);
          c[k] = __fdiv_rn(c[k], d);
        }
      };
      colnorm();
#pragma unroll 1
      for (int it = 1; it < 20; ++it) {
        const float d = __fadd_rn(sum4_last(c[0], c[1], c[2], c[3]), attn::kHcEps);
#pragma unroll
        for (int k = 0; k < 4; ++k) c[k] = __fdiv_rn(c[k], d);
        colnorm();
      }
      if (valid)
#pragma unroll
        for (int k = 0; k < 4; ++k) a.comb_out[r * 16 + 4 * j + k] = c[k];
    }
  }

  // RMSNorm of x (and act_quant): warp w re-reads its 4 rows, lane l columns 8 (l + 32 i) .. + 7, so a lane
  // quad is one act_quant group
  if (a.norm_w) {
    const uint4* nw = reinterpret_cast<const uint4*>(a.norm_w);
#pragma unroll 1
    for (int k = 0; k < 32 / kEPerRow; ++k) {
      const int rr = warp * (32 / kEPerRow) + k;
      const float rs = __shfl_sync(0xffffffffu, rsx, kEPerRow * k);
      if (rr >= nr) break;
      const size_t ro = (size_t)(r0 + rr) * kDim;
      uint4* xr = reinterpret_cast<uint4*>(a.x_out + ro);
      uint4 xv[kDim / 256];
#pragma unroll
      for (int i = 0; i < kDim / 256; ++i) xv[i] = __ldcg(xr + lane + 32 * i);
#pragma unroll
      for (int i = 0; i < kDim / 256; ++i) {
        const int c = lane + 32 * i;
        float v[8], w[8], o[8];
        unpack8(xv[i], v);
        unpack8(__ldg(nw + c), w);
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = attn::bf2f(attn::f2bf(__fmul_rn(w[e], __fmul_rn(v[e], rs))));
        xr[c] = pack8(o);
        if (a.xq) {
          float mx = 0.f;
#pragma unroll
          for (int e = 0; e < 8; ++e) mx = fmaxf(mx, fabsf(o[e]));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
          const uint32_t sb = act_scale_byte(mx);
          uint2 qv = make_uint2(0, 0);
#pragma unroll
          for (int e = 0; e < 8; ++e) (e < 4 ? qv.x : qv.y) |= quant_e4m3(o[e], sb) << (8 * (e & 3));
          *reinterpret_cast<uint2*>(a.xq + ro + 8 * c) = qv;
          if ((lane & 3) == 0) a.xs[(size_t)(r0 + rr) * (kDim / 32) + c / 4] = (uint8_t)sb;
        }
      }
    }
  }
}


// The split boundary (hc_boundary_split; up to 512 rows): the default kernel's sums on more SMs. A CTA takes
// R rows (4, 8 or 16) and one half of each block's columns (blockIdx.y), so 32 rows run on 16, 8 or 4 CTAs
// instead of one. Every chain stays in one thread, in the default kernel's order: the projection's (copy,
// half) warp slot over the blocks, y^2 and x^2 per row and 8-column slot. The stream and hc_fn's parts come
// through shared memory S blocks ahead (cp.async), so no block waits on a load issued one block before;
// y goes through a tile that the chain and projection roles read. Each CTA leaves its partials in a scratch
// (the 4 warp slots' sums, y^2 and x^2 per slot); the second CTA of a row group to finish sums them in the
// default kernel's orders (warps 0..7, the lanes' butterflies), runs the Sinkhorn and the norm of x.
constexpr int kSHalf = kBCols / 2;              // columns of each copy per block and CTA
constexpr int kSThreads = 128;
constexpr int kSYStride = 4 * kSHalf * 2 + 16;  // y tile row (bytes): ldmatrix without bank conflicts
constexpr int kSPartU4 = kFnU4 * 32;            // a warp slot's parts per block (uint4)
constexpr int kSScr = 8 * 24 + 8 + 8 * 32;      // scratch floats a row: 8 slots' sums, y^2 and x^2 per slot
constexpr int kSMaxRows = 512;

template <int R>
struct SSmem {
  static constexpr int S = R <= 8 ? 4 : 3;  // blocks in flight
  static constexpr int MT = (R + 15) / 16;
  uint4 parts[S][4][kSPartU4];
  uint8_t stream[S][5 * R * kSHalf * 2];  // sub + the 4 copies of res, this half's columns
  uint8_t ytile[16 * MT * kSYStride];
  float mix[R][24];
  float raw[R][25];
  float rsx[R];
  int last;
};

template <int R, bool POST>
__global__ void __launch_bounds__(kSThreads) hc_bsplit_kernel(HcBoundary a, int rows, float* scr, int* cnt) {
  constexpr int kS = SSmem<R>::S, kMT = SSmem<R>::MT, kRowsP = 16 * kMT, kSubB = R * kSHalf * 2;
  extern __shared__ __align__(16) uint8_t smem_raw[];
  SSmem<R>& sm = *reinterpret_cast<SSmem<R>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, ph = blockIdx.y;
  const int r0 = blockIdx.x * R, nr = min(R, rows - r0), c0 = ph * kSHalf;
  const bool proj = a.fn_parts != nullptr;
  for (int i = tid; i < R * 24; i += kSThreads) {
    const int rr = i / 24, k = i % 24, r = r0 + rr;
    float v = 0.f;
    if (rr < nr) v = k < 4 ? a.pre[r * 4 + k] : !POST ? 0.f : k < 8 ? a.post[r * 4 + k - 4] : a.comb[r * 16 + k - 8];
    sm.mix[rr][k] = v;
  }
  for (int i = tid; i < (kRowsP - R) * kSYStride / 16; i += kSThreads)
    reinterpret_cast<uint4*>(sm.ytile + R * kSYStride)[i] = make_uint4(0, 0, 0, 0);
  const uint64_t pol_stream = policy_evict_first();
  auto load = [&](int cb, int st) {
    uint8_t* s0 = sm.stream[st];
    constexpr int kCh = R * (kSHalf / 8);
    for (int i = POST ? tid : tid + kCh; i < 5 * kCh; i += kSThreads) {
      if (i < kCh) {
        const int rr = i / 4, c = i % 4;
        const bool ok = rr < nr;
        pa::cp16(s0 + rr * (kSHalf * 2) + c * 16, a.sub + (size_t)(ok ? r0 + rr : 0) * kDim + cb * kBCols + c0 + c * 8, ok);
      } else {
        const int i2 = i - kCh, rr = i2 / 16, j = (i2 / 4) & 3, c = i2 % 4;
        const bool ok = rr < nr;
        pa::cp16(s0 + kSubB + (rr * 4 + j) * (kSHalf * 2) + c * 16,
                 a.res + (size_t)(ok ? r0 + rr : 0) * kHcDim + j * kDim + cb * kBCols + c0 + c * 8, ok);
      }
    }
    if (proj)
      for (int i = tid; i < 4 * kSPartU4; i += kSThreads) {
        const int pj = i / kSPartU4, k = i % kSPartU4;
        pa::cp16(&sm.parts[st][pj][k],
                 reinterpret_cast<const uint4*>(a.fn_parts) + (size_t)((pj * kDim + cb * kBCols + c0) >> 5) * kSPartU4 + k, true);
      }
  };
  // chain slot: row cr, columns 8 cec .. + 7 of the half (slot 4 ph + cec of the default kernel's 8);
  // projection slot: warp pj takes copy pj (the default kernel's warp 2 pj + ph)
  const int cr = tid >> 2, cec = tid & 3, pj = warp;
  const bool chain = tid < 4 * R;
  float acc[kMT][3][4] = {};
  float ssy = 0.f, sx[4][8] = {};
#pragma unroll
  for (int s = 0; s < kS - 1; ++s) {
    load(s, s);
    pa::cp_commit();
  }
#pragma unroll 1
  for (int cb = 0; cb < kNBlk; ++cb) {
    pa::cp_wait<kS - 2>();
    __syncthreads();
    if (cb + kS - 1 < kNBlk) load(cb + kS - 1, (cb + kS - 1) % kS);
    pa::cp_commit();
    const int st = cb % kS;
    const uint8_t* s0 = sm.stream[st];
    for (int u = tid; u < R * 16; u += kSThreads) {  // y: row, slot, copy
      const int rr = u >> 4, ec = (u >> 2) & 3, j = u & 3;
      float y[8];
      unpack8(*reinterpret_cast<const uint4*>(s0 + kSubB + (rr * 4 + j) * (kSHalf * 2) + ec * 16), y);
      if constexpr (POST) {  // attn::hc_post, as the default kernel
        const float* m = sm.mix[rr];
        float xs[4][8], sf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) unpack8(*reinterpret_cast<const uint4*>(s0 + kSubB + (rr * 4 + i) * (kSHalf * 2) + ec * 16), xs[i]);
        unpack8(*reinterpret_cast<const uint4*>(s0 + rr * (kSHalf * 2) + ec * 16), sf);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float v = __fadd_rn(__fmul_rn(m[8 + j], xs[0][e]), __fmul_rn(m[12 + j], xs[1][e]));
          v = __fadd_rn(v, __fmul_rn(m[16 + j], xs[2][e]));
          v = __fadd_rn(v, __fmul_rn(m[20 + j], xs[3][e]));
          y[e] = attn::bfround(__fadd_rn(__fmul_rn(m[4 + j], sf[e]), v));
        }
      }
      const uint4 pk = pack8(y);
      if (POST && a.y && rr < nr) st16_pol(a.y + (size_t)(r0 + rr) * kHcDim + j * kDim + cb * kBCols + c0 + ec * 8, pk, pol_stream);
      *reinterpret_cast<uint4*>(sm.ytile + rr * kSYStride + (j * kSHalf + ec * 8) * 2) = pk;
    }
    __syncthreads();
    if (chain) {  // y^2 and x = hc_pre(y), x^2 by column class: the default kernel's chains
      const float* m = sm.mix[cr];
      float y[4][8];
#pragma unroll
      for (int j = 0; j < 4; ++j) unpack8(*reinterpret_cast<const uint4*>(sm.ytile + cr * kSYStride + (j * kSHalf + cec * 8) * 2), y[j]);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) ssy = __fmaf_rn(y[j][e], y[j][e], ssy);
      float xv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float v = __fadd_rn(__fmul_rn(m[0], y[0][e]), __fmul_rn(m[1], y[1][e]));
        v = __fadd_rn(v, __fmul_rn(m[2], y[2][e]));
        v = __fadd_rn(v, __fmul_rn(m[3], y[3][e]));
        xv[e] = attn::bfround(v);
        sx[0][e] = __fadd_rn(sx[0][e], __fmul_rn(xv[e], xv[e]));
        const float s3 = sx[0][e];
        sx[0][e] = sx[1][e], sx[1][e] = sx[2][e], sx[2][e] = sx[3][e], sx[3][e] = s3;
      }
      if (cr < nr) *reinterpret_cast<uint4*>(a.x_out + (size_t)(r0 + cr) * kDim + cb * kBCols + c0 + cec * 8) = pack8(xv);
    }
    if (proj) {
      uint4 fr[kFnU4];
#pragma unroll
      for (int i = 0; i < kFnU4; ++i) fr[i] = sm.parts[st][pj][i * 32 + lane];
      const uint32_t* fw = reinterpret_cast<const uint32_t*>(fr);
      float part[kMT][3][4] = {};
#pragma unroll
      for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
          uint32_t af[4];
          pa::ldsm_x4(af, sm.ytile + ((lane & 15) + 16 * mt) * kSYStride + (pj * kSHalf + u * 16 + (lane >> 4) * 8) * 2);
#pragma unroll
          for (int nt = 0; nt < 3; ++nt)
#pragma unroll
            for (int p = 2; p >= 0; --p)
              pa::mma_bf16(part[mt][nt], af, fw[((u * 3 + nt) * 3 + p) * 2], fw[((u * 3 + nt) * 3 + p) * 2 + 1]);
        }
#pragma unroll
      for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 3; ++nt)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[mt][nt][i] = __fadd_rn(acc[mt][nt][i], part[mt][nt][i]);
    }
  }
  pa::cp_wait_all();
  if (chain && cr < nr) {
    float* p = scr + (size_t)(r0 + cr) * kSScr;
    const int ec = 4 * ph + cec;
    p[192 + ec] = ssy;
#pragma unroll
    for (int h = 0; h < 4; ++h)
#pragma unroll
      for (int e = 0; e < 8; ++e) p[200 + ec * 32 + h * 8 + e] = sx[h][e];
  }
  if (proj) {
    const int w = 2 * pj + ph;
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
      for (int nt = 0; nt < 3; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int rr = 16 * mt + g + 8 * (i >> 1), o = 8 * nt + 2 * t + (i & 1);
          if (rr < nr) scr[(size_t)(r0 + rr) * kSScr + w * 24 + o] = acc[mt][nt][i];
        }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) sm.last = atomicAdd(cnt + blockIdx.x, 1) == 1;
  __syncthreads();
  if (!sm.last) return;
  __threadfence();
  if (tid == 0) cnt[blockIdx.x] = 0;
  if (proj)
    for (int i = tid; i < nr * 25; i += kSThreads) {  // warps 0..7 in order; y^2's butterfly (lanes xor 4, 2, 1)
      const int rr = i / 25, o = i % 25;
      const float* p = scr + (size_t)(r0 + rr) * kSScr;
      float s;
      if (o < 24) {
        s = __ldcg(p + o);
#pragma unroll
        for (int w = 1; w < 8; ++w) s = __fadd_rn(s, __ldcg(p + w * 24 + o));
      } else {
        float q[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = __ldcg(p + 192 + k);
        s = __fadd_rn(__fadd_rn(__fadd_rn(q[0], q[4]), __fadd_rn(q[2], q[6])), __fadd_rn(__fadd_rn(q[1], q[5]), __fadd_rn(q[3], q[7])));
      }
      sm.raw[rr][o] = s;
    }
  if (a.norm_w)
    for (int rr = tid; rr < nr; rr += kSThreads) {  // x^2: lanes xor 2, 1; within e; the 8 warps in order
      const float* p = scr + (size_t)(r0 + rr) * kSScr + 200;
      float r[8];
#pragma unroll
      for (int h = 0; h < 4; ++h)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int e0 = 4 * q;
          float v[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float s0 = __ldcg(p + (e0 + 0) * 32 + h * 8 + e), s1 = __ldcg(p + (e0 + 1) * 32 + h * 8 + e);
            const float s2 = __ldcg(p + (e0 + 2) * 32 + h * 8 + e), s3 = __ldcg(p + (e0 + 3) * 32 + h * 8 + e);
            v[e] = __fadd_rn(__fadd_rn(s0, s2), __fadd_rn(s1, s3));
          }
#pragma unroll
          for (int o = 4; o > 0; o >>= 1)
#pragma unroll
            for (int e = 0; e < o; ++e) v[e] = __fadd_rn(v[e], v[e + o]);
          r[2 * h + q] = v[0];
        }
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) tot = __fadd_rn(tot, r[w]);
      sm.rsx[rr] = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / kDim), attn::kNormEps));
    }
  __syncthreads();
  if (proj && tid < (R < 8 ? 32 : 4 * R)) {  // attn::hc_finish, as the default kernel
    const int rr = tid >> 2, j = tid & 3, r = r0 + rr;
    const bool valid = rr < nr;
    const float* p = sm.raw[rr < nr ? rr : nr - 1];
    if (valid && a.proj_out && j == 0)
      for (int i = 0; i < 25; ++i) a.proj_out[(size_t)r * 25 + i] = p[i];
    const float rs = rsqrtf(__fadd_rn(__fmul_rn(p[24], 1.0f / kHcDim), attn::kNormEps));
    const float s0 = a.scale[0], s1 = a.scale[1], s2 = a.scale[2];
    const float pre = __fadd_rn(sigmoidf_ref(__fadd_rn(__fmul_rn(__fmul_rn(p[j], rs), s0), a.base[j])), attn::kHcEps);
    const float post = __fmul_rn(2.0f, sigmoidf_ref(__fadd_rn(__fmul_rn(__fmul_rn(p[4 + j], rs), s1), a.base[4 + j])));
    if (valid) a.pre_out[r * 4 + j] = pre, a.post_out[r * 4 + j] = post;
    float c[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) c[k] = __fadd_rn(__fmul_rn(__fmul_rn(p[8 + 4 * j + k], rs), s2), a.base[8 + 4 * j + k]);
    {
      const float mx = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));
      float e[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) e[k] = expf(__fsub_rn(c[k], mx));
      const float sum = sum4_last(e[0], e[1], e[2], e[3]);
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] = __fadd_rn(__fdiv_rn(e[k], sum), attn::kHcEps);
    }
    const int qb = lane & ~3;
    auto colnorm = [&]() {
      float col[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) col[i][k] = __shfl_sync(0xffffffffu, c[k], qb + i);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float d = __fadd_rn(sum4_seq(col[0][k], col[1][k], col[2][k], col[3][k]), attn::kHcEps);
        c[k] = __fdiv_rn(c[k], d);
      }
    };
    colnorm();
#pragma unroll 1
    for (int it = 1; it < 20; ++it) {
      const float d = __fadd_rn(sum4_last(c[0], c[1], c[2], c[3]), attn::kHcEps);
#pragma unroll
      for (int k = 0; k < 4; ++k) c[k] = __fdiv_rn(c[k], d);
      colnorm();
    }
    if (valid)
#pragma unroll
      for (int k = 0; k < 4; ++k) a.comb_out[r * 16 + 4 * j + k] = c[k];
  }
  if (a.norm_w) {  // RMSNorm of x and act_quant, as the default kernel: a warp a row
    const uint4* nw = reinterpret_cast<const uint4*>(a.norm_w);
#pragma unroll 1
    for (int rr = warp; rr < nr; rr += kSThreads / 32) {
      const float rs = sm.rsx[rr];
      const size_t ro = (size_t)(r0 + rr) * kDim;
      uint4* xr = reinterpret_cast<uint4*>(a.x_out + ro);
      uint4 xv[kDim / 256];
#pragma unroll
      for (int i = 0; i < kDim / 256; ++i) xv[i] = __ldcg(xr + lane + 32 * i);
#pragma unroll
      for (int i = 0; i < kDim / 256; ++i) {
        const int c = lane + 32 * i;
        float v[8], w[8], o[8];
        unpack8(xv[i], v);
        unpack8(__ldg(nw + c), w);
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = attn::bf2f(attn::f2bf(__fmul_rn(w[e], __fmul_rn(v[e], rs))));
        xr[c] = pack8(o);
        if (a.xq) {
          float mx = 0.f;
#pragma unroll
          for (int e = 0; e < 8; ++e) mx = fmaxf(mx, fabsf(o[e]));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
          const uint32_t sb = act_scale_byte(mx);
          uint2 qv = make_uint2(0, 0);
#pragma unroll
          for (int e = 0; e < 8; ++e) (e < 4 ? qv.x : qv.y) |= quant_e4m3(o[e], sb) << (8 * (e & 3));
          *reinterpret_cast<uint2*>(a.xq + ro + 8 * c) = qv;
          if ((lane & 3) == 0) a.xs[(size_t)(r0 + rr) * (kDim / 32) + c / 4] = (uint8_t)sb;
        }
      }
    }
  }
}

}  // namespace

size_t hc_fn_parts_bytes() { return (size_t)kFnPairs * 32 * kFnU32 * 4; }

void hc_fn_parts(const float* fn, uint32_t* parts, cudaStream_t s) {
  hc_fn_parts_kernel<<<240, 256, 0, s>>>(fn, parts);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
// Rows per CTA by the chunk's rows (<= 128: 4, <= 768: 16, else 32) and the L2 prefetch distance in blocks:
// off (32 rows, no prefetch): with the MoE picks, the turns gained too little for a default (docs/engine.md);
// hc_boundary_tuning overrides both.
constexpr bool kHcRowsByChunk = false;
constexpr int kHcPrefetch = 0;
int g_hc_rows = 0, g_hc_pf = -1;  // hc_boundary_tuning: 0 / -1 the defaults

template <int R>
void launch_boundary(const HcBoundary& a, int rows, int pfd, cudaStream_t s) {
  const size_t smem = sizeof(BSmem<R>);
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(hc_boundary_kernel<R, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(hc_boundary_kernel<R, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr = true;
  }
  const dim3 grid((rows + R - 1) / R);
  if (a.sub)
    hc_boundary_kernel<R, true><<<grid, kBThreads, smem, s>>>(a, rows, pfd);
  else
    hc_boundary_kernel<R, false><<<grid, kBThreads, smem, s>>>(a, rows, pfd);
  CUDA_CHECK(cudaGetLastError());
}
float* g_split_scr = nullptr;  // hc_boundary_split: the partials (kSMaxRows rows) and a counter per row group
int* g_split_cnt = nullptr;
int g_split_max = 0, g_split_rows = 0;

template <int R>
void launch_bsplit(const HcBoundary& a, int rows, cudaStream_t s) {
  const size_t smem = sizeof(SSmem<R>);
  static bool attr = false;
  if (!attr) {
    CUDA_CHECK(cudaFuncSetAttribute(hc_bsplit_kernel<R, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaFuncSetAttribute(hc_bsplit_kernel<R, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr = true;
  }
  const dim3 grid((rows + R - 1) / R, 2);
  if (a.sub)
    hc_bsplit_kernel<R, true><<<grid, kSThreads, smem, s>>>(a, rows, g_split_scr, g_split_cnt);
  else
    hc_bsplit_kernel<R, false><<<grid, kSThreads, smem, s>>>(a, rows, g_split_scr, g_split_cnt);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace

void hc_boundary_split(int max_rows, int rows_per_cta) {
  if (max_rows < 0 || max_rows > kSMaxRows) die("hc_boundary_split: %d rows (at most %d)", max_rows, kSMaxRows);
  if (rows_per_cta != 0 && rows_per_cta != 4 && rows_per_cta != 8 && rows_per_cta != 16)
    die("hc_boundary_split: %d rows per CTA (4, 8 or 16)", rows_per_cta);
  if (max_rows > 0 && !g_split_scr) {
    CUDA_CHECK(cudaMalloc(&g_split_scr, sizeof(float) * kSMaxRows * kSScr));
    CUDA_CHECK(cudaMalloc(&g_split_cnt, sizeof(int) * kSMaxRows));
    CUDA_CHECK(cudaMemset(g_split_cnt, 0, sizeof(int) * kSMaxRows));
  }
  g_split_max = max_rows, g_split_rows = rows_per_cta;
}

void hc_boundary_tuning(int rows_per_cta, int prefetch_blocks) { g_hc_rows = rows_per_cta, g_hc_pf = prefetch_blocks; }

void hc_boundary_p(const HcBoundary& a, int rows, cudaStream_t s) {
  if (rows < 1 || !a.res || !a.pre || !a.x_out || (a.sub && (!a.post || !a.comb)) ||
      (a.fn_parts && (!a.scale || !a.base || !a.pre_out || !a.post_out || !a.comb_out)) || (a.xq && !a.norm_w))
    die("hc_boundary_p: bad arguments (rows %d)", rows);
  if (g_split_max > 0 && rows <= g_split_max) {  // the split kernel: rows per CTA by the chunk's rows
    const int r = g_split_rows ? g_split_rows : rows <= 32 ? 4 : rows <= 128 ? 8 : 16;
    if (r == 4) launch_bsplit<4>(a, rows, s);
    else if (r == 8) launch_bsplit<8>(a, rows, s);
    else launch_bsplit<16>(a, rows, s);
    return;
  }
  // small chunks: CTAs of fewer rows, so that more SMs stream
  const bool by_chunk = g_hc_rows < 0 || (g_hc_rows == 0 && kHcRowsByChunk);
  const int r = by_chunk ? (rows <= 128 ? 4 : rows <= 768 ? 16 : 32) : g_hc_rows ? g_hc_rows : 32;
  const int pfd = g_hc_pf >= 0 ? g_hc_pf : kHcPrefetch;
  if (pfd < 0 || pfd >= kNBlk) die("hc_boundary_p: prefetch distance %d", pfd);
  if (r == 4) launch_boundary<4>(a, rows, pfd, s);
  else if (r == 8) launch_boundary<8>(a, rows, pfd, s);
  else if (r == 16) launch_boundary<16>(a, rows, pfd, s);
  else if (r == 32) launch_boundary<32>(a, rows, pfd, s);
  else die("hc_boundary_p: %d rows per CTA (4, 8, 16 or 32)", r);
}

}  // namespace dsk
