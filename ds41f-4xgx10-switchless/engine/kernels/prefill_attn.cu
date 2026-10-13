// Sparse attention for prefill chunks (M5): attn::sparse_attn's math for thousands of rows.
//
// Each row is 16 heads (MMA M = 16) against its own 640 keys (2 window blocks of 64, fp8 + ue8m0/32,
// then kmax/64 blocks of selected entries, e2m1 + e4m3/16). With M = 16 the tensor cores starve on
// operand traffic if keys are widened to bf16 in shared memory (the decode kernel's 3 x 64 KB per block),
// so here a block's keys stay packed in shared memory (cp.async, two stages) and every warp widens just
// the operands it needs to f16 in registers (cvt e4m3x2 / e2m1x2 -> f16x2, times the scale in f16: exact
// for these formats; q and P are bf16 values, exact in f16 too). The 8 warps split a 64-key block twice:
//   QK  warp w = (kg = w>>1, dg = w&1): keys 16kg..+15 over dims 256dg..+255, q in registers (64 regs,
//       k-dims permuted so each lane reads 64 contiguous code bits per key); the two dim halves add their
//       partial scores through shared memory (a pair barrier, not the whole CTA).
//   PV  warp w: output dims 64w..+63 for all 64 keys, P (bf16-rounded) from shared memory.
// Online softmax as the reference (fp32 sums, P rounded to bf16, sink in the denominator) with a lazy
// running max: P is taken against the current reference max m and only when a block's max exceeds
// m + 8 does the CTA agree on a new max (one extra barrier), so most blocks need two CTA barriers.
// One CTA (99 KB of shared memory) per SM walks its rows; the next block's keys and the next row's q and
// topk are prefetched while the current block computes.
#include <algorithm>
#include <cmath>

#include "attn/attn_common.cuh"
#include "common/check.h"
#include "kernels/pa_common.cuh"
#include "kernels/prefill_attn.h"

namespace dsk {
namespace {

using attn::kHead;
using attn::kHeads;
using attn::kRope;
using attn::kWindow;

constexpr int kThreads = 256;
constexpr int kBlk = 64;      // keys per block
constexpr int kMaxK = 512;    // compressed entries per row
constexpr float kLazy = 8.0f;  // P may reach e^8 before the reference max moves

constexpr int kCodeB = kBlk * 512;  // a block's packed keys, 512 B (fp8) or 256 B (fp4) per key row
constexpr int kSclB = kBlk * 32;    // scales, 16 B (fp8) or 32 B (fp4) per key
constexpr int kPS = kBlk + 8;       // P row stride (halves)
constexpr int kOffCode = 0;
constexpr int kOffScl = kOffCode + 2 * kCodeB;
constexpr int kOffValid = kOffScl + 2 * kSclB;
constexpr int kOffQ = kOffValid + 2 * kBlk;
constexpr int kOffTk = kOffQ + kHeads * kHead * 2;
constexpr int kOffSx = kOffTk + kMaxK * 4;
constexpr int kOffRed = kOffSx + 8 * 32 * 8 * 4;
constexpr int kOffP = kOffRed + 2 * 4 * kHeads * 4;
constexpr int kOffTab = kOffP + kHeads * kPS * 2;  // RoPE cos/sin of dims 448..511 at a row's pos, 2 slots
constexpr int kSmem = kOffTab + 2 * (kRope / 2) * 8;
static_assert(kSmem <= 101376, "shared memory");

// 16-byte chunk c of key row k: the low three chunk bits XORed with k and with bit 3 of c, so the QK reads
// (2 keys x 4 dim groups per phase) and the PV reads (4 keys x 2 chunks per phase) hit distinct banks.
__device__ __forceinline__ int swz(int k, int c) { return c ^ ((k ^ ((c >> 2) & 2)) & 7); }
// q staging: chunk c (of 64) of head h
__device__ __forceinline__ int qswz(int h, int c) { return c ^ (((c >> 3) ^ (h << 2)) & 7); }

struct PArgs {
  const uint16_t* q;
  const int* pos;
  int rows;
  const float2* table;
  attn::Window win;
  attn::Compressed ckv;
  const int* topk;
  int kmax, nb, win_lo;
  const float* sink;
  float scale;
  uint16_t* out;
};

// block b of row r into stage st: b < 2 the window (positions pos-127+64b+k from win_lo on), else topk
// entries 64(b-2)+k.
// Invalid keys are zero-filled (their scores are masked; zero codes keep P*V finite).
__device__ __forceinline__ void issue_block(const PArgs& a, uint8_t* sm, int p, int b, int st) {
  const int tid = threadIdx.x;
  uint8_t* cod = sm + kOffCode + st * kCodeB;
  uint8_t* scl = sm + kOffScl + st * kSclB;
  uint8_t* val = sm + kOffValid + st * kBlk;
  if (b < 2) {
    const int c = tid & 31;
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int k = (tid >> 5) + 8 * it, kp = p - (kWindow - 1) + kBlk * b + k;
      const bool ok = kp >= a.win_lo;
      const size_t slot = ok ? (size_t)((kp - a.win.base) & a.win.mask) : 0;
      pa::cp16(cod + k * 512 + swz(k, c) * 16, a.win.kv + slot * kHead + c * 16, ok);
      if (c == 0) {
        pa::cp16(scl + k * 32, a.win.sc + slot * 16, ok);
        val[k] = ok;
      }
    }
  } else {
    const int* tk = reinterpret_cast<const int*>(sm + kOffTk) + kBlk * (b - 2);
    const int c = tid & 15;
    int kps[4];
#pragma unroll
    for (int it = 0; it < 4; ++it) kps[it] = tk[(tid >> 4) + 16 * it];
#pragma unroll
    for (int it = 0; it < 4; ++it) {
      const int k = (tid >> 4) + 16 * it, kp = kps[it];
      const bool ok = kp >= 0;
      const size_t e = ok ? (size_t)kp : 0;
      pa::cp16(cod + k * 512 + swz(k, c) * 16, a.ckv.kv + e * (kHead / 2) + c * 16, ok);
      if (c < 2) pa::cp16(scl + k * 32 + c * 16, a.ckv.sc + e * 32 + c * 16, ok);
      if (c == 0) val[k] = ok;
    }
  }
}
// row r's q, and its RoPE table row (pos p) into table slot `slot`
__device__ __forceinline__ void issue_q(const PArgs& a, uint8_t* sm, int r, int p, int slot) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(a.q + (size_t)r * kHeads * kHead);
  for (int i = threadIdx.x; i < kHeads * 64; i += kThreads)
    pa::cp16(sm + kOffQ + (i >> 6) * 1024 + qswz(i >> 6, i & 63) * 16, src + i * 16, true);
  if (threadIdx.x < kRope / 4)
    pa::cp16(sm + kOffTab + slot * kRope * 4 + threadIdx.x * 16,
             reinterpret_cast<const uint8_t*>(a.table + (size_t)p * 32) + threadIdx.x * 16, true);
}
__device__ __forceinline__ void issue_tk(const PArgs& a, uint8_t* sm, int r) {
  const int* src = a.topk + (size_t)r * a.kmax;
  for (int i = threadIdx.x; i < a.kmax / 4; i += kThreads) pa::cp16(sm + kOffTk + i * 16, src + i * 4, true);
}

// q fragments of warp half dg: k-step s holds dims 256dg + 64t + 4s .. +3 (pairs at MMA k 2t / 2t+8) of
// heads g (a0, a2) and g+8 (a1, a3); RoPE at pos on dims 448..511, rounded to bf16, then f16 (exact).
__device__ __forceinline__ void load_q(const uint8_t* sm, const float2* tb, int dg, uint32_t (&qa)[16][4]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int h = g + 8 * hh;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = 32 * dg + 8 * t + i, d0 = 8 * c;
      const uint4 v = *reinterpret_cast<const uint4*>(sm + kOffQ + h * 1024 + qswz(h, c) * 16);
      float x[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t wd = pa::word_of(v, j);
        x[2 * j] = __uint_as_float(wd << 16);
        x[2 * j + 1] = __uint_as_float(wd & 0xffff0000u);
      }
      if (d0 >= kHead - kRope) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 cs = tb[(d0 - (kHead - kRope)) / 2 + j];
          attn::rope_pair(x[2 * j], x[2 * j + 1], cs.x, cs.y, false);
          x[2 * j] = attn::bfround(x[2 * j]);
          x[2 * j + 1] = attn::bfround(x[2 * j + 1]);
        }
      }
      qa[2 * i][hh] = pa::pack_h2(x[0], x[1]);
      qa[2 * i][2 + hh] = pa::pack_h2(x[2], x[3]);
      qa[2 * i + 1][hh] = pa::pack_h2(x[4], x[5]);
      qa[2 * i + 1][2 + hh] = pa::pack_h2(x[6], x[7]);
    }
  }
}

// partial scores of keys 16kg..+15 (n-tiles j: keys 16kg + 8j + g) over dims 256dg..+255
template <bool FP8>
__device__ __forceinline__ void qk_block(const uint8_t* cod, const uint8_t* scl, int kg, int dg,
                                         const uint32_t (&qa)[16][4], float (&acc)[2][4]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const int k = 16 * kg + 8 * j + g;
    const uint8_t* row = cod + k * 512;
    if constexpr (FP8) {  // 64 bytes: dims 64t..+63 of the half; ue8m0 per 32
      uint4 c[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] = *reinterpret_cast<const uint4*>(row + swz(k, 16 * dg + 4 * t + i) * 16);
      const uint32_t e2 = *reinterpret_cast<const uint16_t*>(scl + k * 32 + 8 * dg + 2 * t);
      uint32_t h0 = pa::ue8m0_h(e2 & 0xff), h1 = pa::ue8m0_h(e2 >> 8);
      h0 |= h0 << 16;
      h1 |= h1 << 16;
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        const uint32_t wd = pa::word_of(c[s >> 2], s & 3), hs = s < 8 ? h0 : h1;
        pa::mma_f16(acc[j], qa[s], pa::hmul2(pa::e4m3x2_h2(wd), hs), pa::hmul2(pa::e4m3x2_h2(wd >> 16), hs));
      }
    } else {  // 32 bytes: dims 64t..+63 of the half, low nibble first; e4m3 per 16
      uint4 c[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) c[i] = *reinterpret_cast<const uint4*>(row + swz(k, 8 * dg + 2 * t + i) * 16);
      const uint32_t e4 = *reinterpret_cast<const uint32_t*>(scl + k * 32 + 16 * dg + 4 * t);
      uint32_t hs[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t b = (e4 >> (8 * q)) & 0xff;
        hs[q] = pa::e4m3x2_h2(b | (b << 8));
      }
#pragma unroll
      for (int s = 0; s < 16; ++s) {
        const uint32_t wd = pa::word_of(c[s >> 3], (s & 7) >> 1) >> (16 * (s & 1));
        pa::mma_f16(acc[j], qa[s], pa::hmul2(pa::e2m1x2_h2(wd), hs[s >> 2]), pa::hmul2(pa::e2m1x2_h2(wd >> 8), hs[s >> 2]));
      }
    }
  }
}

// O[16 heads][dims 64w..+63] += P[16][64] V[64][...]: n-tile n, column c <-> dim 64w + 8c + n
template <bool FP8>
__device__ __forceinline__ void pv_block(const uint8_t* cod, const uint8_t* scl, const uint16_t* ps, int w,
                                         float (&o)[8][4]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < kBlk / 16; ++s) {
    uint32_t af[4];
    pa::ldsm_x4(af, ps + (lane & 15) * kPS + 16 * s + (lane >> 4) * 8);
    const int k0 = 16 * s + 2 * t;
    const int ks[4] = {k0, k0 + 1, k0 + 8, k0 + 9};
    if constexpr (FP8) {  // 8 bytes per key: dims 64w + 8g .. +7
      const int c = 4 * w + (g >> 1), off = (g & 1) * 8;
      uint2 v[4];
      uint32_t sc[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[i] = *reinterpret_cast<const uint2*>(cod + ks[i] * 512 + swz(ks[i], c) * 16 + off);
        sc[i] = pa::ue8m0_h(scl[ks[i] * 32 + 2 * w + (g >> 2)]);
      }
      const uint32_t s01 = sc[0] | sc[1] << 16, s23 = sc[2] | sc[3] << 16;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const uint32_t sel = (n & 3) | ((4 + (n & 3)) << 4);
        const uint32_t x01 = __byte_perm(n < 4 ? v[0].x : v[0].y, n < 4 ? v[1].x : v[1].y, sel);
        const uint32_t x23 = __byte_perm(n < 4 ? v[2].x : v[2].y, n < 4 ? v[3].x : v[3].y, sel);
        pa::mma_f16(o[n], af, pa::hmul2(pa::e4m3x2_h2(x01), s01), pa::hmul2(pa::e4m3x2_h2(x23), s23));
      }
    } else {  // 4 bytes per key: dims 64w + 8g .. +7
      const int c = 2 * w + (g >> 2), off = (g & 3) * 4;
      uint32_t v[4], sb[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[i] = *reinterpret_cast<const uint32_t*>(cod + ks[i] * 512 + swz(ks[i], c) * 16 + off);
        sb[i] = scl[ks[i] * 32 + 4 * w + (g >> 1)];
      }
      const uint32_t s01 = pa::e4m3x2_h2(sb[0] | sb[1] << 8), s23 = pa::e4m3x2_h2(sb[2] | sb[3] << 8);
      // byte i of e: (nibble 2i of key a, nibble 2i of key b); of od: nibbles 2i+1
      const uint32_t e01 = (v[0] & 0x0F0F0F0Fu) | ((v[1] & 0x0F0F0F0Fu) << 4);
      const uint32_t o01 = ((v[0] >> 4) & 0x0F0F0F0Fu) | (v[1] & 0xF0F0F0F0u);
      const uint32_t e23 = (v[2] & 0x0F0F0F0Fu) | ((v[3] & 0x0F0F0F0Fu) << 4);
      const uint32_t o23 = ((v[2] >> 4) & 0x0F0F0F0Fu) | (v[3] & 0xF0F0F0F0u);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const uint32_t x01 = ((n & 1) ? o01 : e01) >> (8 * (n >> 1)), x23 = ((n & 1) ? o23 : e23) >> (8 * (n >> 1));
        pa::mma_f16(o[n], af, pa::hmul2(pa::e2m1x2_h2(x01), s01), pa::hmul2(pa::e2m1x2_h2(x23), s23));
      }
    }
  }
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return (uint32_t)attn::f2bf(lo) | (uint32_t)attn::f2bf(hi) << 16;
}
// o (bf16) of dims (d, d+1), the inverse RoPE on dims 448..511 applied to the bf16 values (as decode)
__device__ __forceinline__ uint32_t finish_pair(float x0, float x1, int d, const float2* tb) {
  x0 = attn::bfround(x0);
  x1 = attn::bfround(x1);
  if (d >= kHead - kRope) {
    const float2 cs = tb[(d - (kHead - kRope)) >> 1];
    attn::rope_pair(x0, x1, cs.x, cs.y, true);
  }
  return pack_bf16(x0, x1);
}

__global__ void __launch_bounds__(kThreads, 1) attn_p_kernel(PArgs a) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, kg = w >> 1, dg = w & 1, g = lane >> 2, t = lane & 3;
  float* sx = reinterpret_cast<float*>(sm + kOffSx);
  float* red_m = reinterpret_cast<float*>(sm + kOffRed);
  float* red_l = red_m + 4 * kHeads;
  uint16_t* ps = reinterpret_cast<uint16_t*>(sm + kOffP);
  const int G = gridDim.x;
  const int nrows = a.rows > (int)blockIdx.x ? (a.rows - 1 - (int)blockIdx.x) / G + 1 : 0;
  const int nitems = nrows * a.nb;
  if (nitems == 0) return;
  // positions of this row and the next (read a row ahead: the window loads and the q / table prefetch
  // need them before their row starts)
  int p_cur = __ldg(a.pos + blockIdx.x), p_nxt = (int)blockIdx.x + G < a.rows ? __ldg(a.pos + blockIdx.x + G) : 0;
  issue_q(a, sm, blockIdx.x, p_cur, 0);
  issue_block(a, sm, p_cur, 0, 0);
  pa::cp_commit();

  uint32_t qa[16][4];
  float o[8][4];
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;  // heads g, g+8 (every warp keeps the same copy)
  for (int it = 0; it < nitems; ++it) {
    const int ri = it / a.nb, b = it - ri * a.nb, r = blockIdx.x + ri * G, st = it & 1;
    const float2* tb = reinterpret_cast<const float2*>(sm + kOffTab + (ri & 1) * kRope * 4);
    if (b == 0 && ri > 0) {
      p_cur = p_nxt;
      p_nxt = r + G < a.rows ? __ldg(a.pos + r + G) : 0;
    }
    pa::cp_wait_all();
    __syncthreads();
    if (b == 0) {
      load_q(sm, tb, dg, qa);
#pragma unroll
      for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
      m0 = m1 = -1e30f;
      l0 = l1 = 0.f;
    }
    // prefetch the next block; with block 0 this row's topk (first read by block 1's prefetch), with
    // block 1 the next row's q (this row's was read at block 0)
    if (it + 1 < nitems) issue_block(a, sm, b + 1 < a.nb ? p_cur : p_nxt, b + 1 < a.nb ? b + 1 : 0, st ^ 1);
    if (b == 0 && a.kmax > 0) issue_tk(a, sm, r);
    if (b == 1 && r + G < a.rows) issue_q(a, sm, r + G, p_nxt, (ri + 1) & 1);
    pa::cp_commit();

    const uint8_t* cod = sm + kOffCode + st * kCodeB;
    const uint8_t* scl = sm + kOffScl + st * kSclB;
    const uint8_t* val = sm + kOffValid + st * kBlk;
    float s[2][4];
    if (b < 2)
      qk_block<true>(cod, scl, kg, dg, qa, s);
    else
      qk_block<false>(cod, scl, kg, dg, qa, s);
    {  // the two dim halves of a key group add their partial scores
      float* mine = sx + (w * 32 + lane) * 8;
      *reinterpret_cast<float4*>(mine) = make_float4(s[0][0], s[0][1], s[0][2], s[0][3]);
      *reinterpret_cast<float4*>(mine + 4) = make_float4(s[1][0], s[1][1], s[1][2], s[1][3]);
      pa::bar_sync(1 + kg, 64);
      const float* oth = sx + ((w ^ 1) * 32 + lane) * 8;
      const float4 o0 = *reinterpret_cast<const float4*>(oth), o1 = *reinterpret_cast<const float4*>(oth + 4);
      s[0][0] = __fadd_rn(s[0][0], o0.x);
      s[0][1] = __fadd_rn(s[0][1], o0.y);
      s[0][2] = __fadd_rn(s[0][2], o0.z);
      s[0][3] = __fadd_rn(s[0][3], o0.w);
      s[1][0] = __fadd_rn(s[1][0], o1.x);
      s[1][1] = __fadd_rn(s[1][1], o1.y);
      s[1][2] = __fadd_rn(s[1][2], o1.z);
      s[1][3] = __fadd_rn(s[1][3], o1.w);
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = 16 * kg + 8 * j + 2 * t + (e & 1);
        s[j][e] = val[key] ? __fmul_rn(s[j][e], a.scale) : -INFINITY;
        if (e < 2)
          mx0 = fmaxf(mx0, s[j][e]);
        else
          mx1 = fmaxf(mx1, s[j][e]);
      }
#pragma unroll
    for (int x = 1; x < 4; x <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x));
    }
    if (dg == 0 && t == 0) {
      red_m[kg * kHeads + g] = mx0;
      red_m[kg * kHeads + g + 8] = mx1;
    }
    // P = exp(s - m) rounded to bf16 into ps (dim half 0 writes), block sums of the fp32 p into red_l
    auto make_p = [&](float mm0, float mm1) {
      float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float p0 = attn::bfround(expf(__fsub_rn(s[j][0], mm0))), p1 = attn::bfround(expf(__fsub_rn(s[j][1], mm0)));
        const float p2 = attn::bfround(expf(__fsub_rn(s[j][2], mm1))), p3 = attn::bfround(expf(__fsub_rn(s[j][3], mm1)));
        sum0 = __fadd_rn(__fadd_rn(sum0, p0), p1);
        sum1 = __fadd_rn(__fadd_rn(sum1, p2), p3);
        if (dg == 0) {
          const int col = 16 * kg + 8 * j + 2 * t;
          *reinterpret_cast<uint32_t*>(ps + g * kPS + col) = pa::pack_h2(p0, p1);
          *reinterpret_cast<uint32_t*>(ps + (g + 8) * kPS + col) = pa::pack_h2(p2, p3);
        }
      }
#pragma unroll
      for (int x = 1; x < 4; x <<= 1) {
        sum0 = __fadd_rn(sum0, __shfl_xor_sync(0xffffffffu, sum0, x));
        sum1 = __fadd_rn(sum1, __shfl_xor_sync(0xffffffffu, sum1, x));
      }
      if (dg == 0 && t == 0) {
        red_l[kg * kHeads + g] = sum0;
        red_l[kg * kHeads + g + 8] = sum1;
      }
    };
    if (b > 0) make_p(m0, m1);
    if (__syncthreads_or(b == 0 || mx0 > m0 + kLazy || mx1 > m1 + kLazy)) {  // rare after block 0: move the max
      float bm0 = red_m[g], bm1 = red_m[g + 8];
#pragma unroll
      for (int q = 1; q < 4; ++q) {
        bm0 = fmaxf(bm0, red_m[q * kHeads + g]);
        bm1 = fmaxf(bm1, red_m[q * kHeads + g + 8]);
      }
      const float mn0 = fmaxf(m0, bm0), mn1 = fmaxf(m1, bm1);
      const float al0 = expf(__fsub_rn(m0, mn0)), al1 = expf(__fsub_rn(m1, mn1));
      l0 = __fmul_rn(l0, al0);
      l1 = __fmul_rn(l1, al1);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        o[n][0] = __fmul_rn(o[n][0], al0);
        o[n][1] = __fmul_rn(o[n][1], al0);
        o[n][2] = __fmul_rn(o[n][2], al1);
        o[n][3] = __fmul_rn(o[n][3], al1);
      }
      m0 = mn0;
      m1 = mn1;
      make_p(m0, m1);
      __syncthreads();
    }
    l0 = __fadd_rn(l0, __fadd_rn(__fadd_rn(__fadd_rn(red_l[g], red_l[kHeads + g]), red_l[2 * kHeads + g]),
                                 red_l[3 * kHeads + g]));
    l1 = __fadd_rn(l1, __fadd_rn(__fadd_rn(__fadd_rn(red_l[g + 8], red_l[kHeads + g + 8]), red_l[2 * kHeads + g + 8]),
                                 red_l[3 * kHeads + g + 8]));
    if (b < 2)
      pv_block<true>(cod, scl, ps, w, o);
    else
      pv_block<false>(cod, scl, ps, w, o);

    if (b == a.nb - 1) {  // finish the row: sink, normalise, bf16, inverse RoPE
      const float d0 = __fadd_rn(l0, expf(__fsub_rn(__ldg(a.sink + g), m0)));
      const float d1 = __fadd_rn(l1, expf(__fsub_rn(__ldg(a.sink + g + 8), m1)));
      const int dbase = 64 * w + 16 * t;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float dd = hh ? d1 : d0;
        uint32_t wv[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          wv[i] = finish_pair(__fdiv_rn(o[2 * i][2 * hh], dd), __fdiv_rn(o[2 * i + 1][2 * hh], dd), dbase + 2 * i, tb);
          wv[4 + i] = finish_pair(__fdiv_rn(o[2 * i][2 * hh + 1], dd), __fdiv_rn(o[2 * i + 1][2 * hh + 1], dd),
                                  dbase + 8 + 2 * i, tb);
        }
        uint4* dst = reinterpret_cast<uint4*>(a.out + ((size_t)r * kHeads + g + 8 * hh) * kHead + dbase);
        dst[0] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        dst[1] = make_uint4(wv[4], wv[5], wv[6], wv[7]);
      }
    }
  }
}

}  // namespace

void sparse_attn_p(const uint16_t* q_pre, const int* pos, int rows, const float2* table, attn::Window win,
                   attn::Compressed ckv, const int* topk, int kmax, const float* sink, uint16_t* out, cudaStream_t s,
                   int win_lo) {
  static const int nsm = [] {
    int d, n;
    CUDA_CHECK(cudaGetDevice(&d));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d));
    CUDA_CHECK(cudaFuncSetAttribute(attn_p_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    return n;
  }();
  if (kmax % kBlk || kmax > kMaxK || (kmax > 0 && !topk) || win_lo < 0) die("sparse_attn_p: kmax %d", kmax);
  if (rows <= 0) return;
  PArgs a{};
  a.q = q_pre;
  a.pos = pos;
  a.rows = rows;
  a.table = table;
  a.win = win;
  a.ckv = ckv;
  a.topk = topk;
  a.kmax = kmax;
  a.nb = 2 + kmax / kBlk;
  a.win_lo = win_lo;
  a.sink = sink;
  a.scale = (float)std::pow(512.0, -0.5);
  a.out = out;
  attn_p_kernel<<<std::min(rows, nsm), kThreads, kSmem, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace dsk
