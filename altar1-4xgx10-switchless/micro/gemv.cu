// glm53 experiment 1: decode GEMV microbenchmarks on GLM-5.3's per-rank shapes at TP=4 (one GB10).
//   routed experts: gate/up [512 x 6144] each (fused, one launch), down [6144 x 512]; the union of the step's
//   experts read once: 8 experts at 1 row, 28 at 6 rows; a pool of >= 2.2 GB cycled so L2 holds nothing.
//   formats: q4 (NVFP4 as published), s3 (E2M1-subset 3-bit), tk3/tk4 (EXL3 trellis K3/K4, 3INST; tk3m = MUL1)
//   attention: bf16 and fp8 (128x128 block scales) on q_b [4096 x 2048] and o_proj [6144 x 4096].
// Every kernel is checked against a CPU reference on the data it timed. Usage:
//   gemv --cpu-check                     host-only self-check of the decoders and layouts (no GPU)
//   gemv --bench <q4|s3|tk3|tk4|tk3m|dense|bw> --out results.tsv [--rows 1,6]
#include "common.cuh"
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <cstdarg>

static constexpr int H = 6144, I = 512, NCH = 1024;  // hidden, intermediate a rank, gate+up channels
enum Fmt { Q4 = 0, S3 = 1, TK = 2 };

// ---------------------------------------------------------------------------------------------------------
// data generation (GPU), modes: 0 raw, 1 e4m3 scale bytes in [0.5, 3.75], 2 f16 +-[0.5, 2), 3 bf16 +-[2^-8, 2^-4),
// 4 e4m3 weights (no NaN), 5 bf16 activations +-[2^-5, 2^-2)
__global__ void k_fill(uint32_t* p, size_t n, uint64_t seed, int mode) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t h = mix64(seed ^ (i * 0x9E3779B97F4A7C15ull));
    uint32_t r = (uint32_t)h, r2 = (uint32_t)(h >> 32), o = 0;
    switch (mode) {
      case 0: o = r; break;
      case 1: for (int b = 0; b < 4; ++b) o |= (0x30u + ((r >> 8 * b) & 0xFF) % 0x18u) << 8 * b; break;
      case 2: for (int b = 0; b < 2; ++b) { uint32_t q = r >> 16 * b;
                o |= ((q & 0x8000u) | ((14u + ((r2 >> b) & 1)) << 10) | (q & 0x3FFu)) << 16 * b; } break;
      case 3: for (int b = 0; b < 2; ++b) { uint32_t q = r >> 16 * b;
                o |= ((q & 0x8000u) | ((119u + ((r2 >> 2 * b) & 3)) << 7) | (q & 0x7Fu)) << 16 * b; } break;
      case 4: for (int b = 0; b < 4; ++b) { uint32_t q = (r >> 8 * b) & 0xFF; if ((q & 0x7F) == 0x7F) q ^= 1; o |= q << 8 * b; } break;
      case 5: for (int b = 0; b < 2; ++b) { uint32_t q = r >> 16 * b;
                o |= ((q & 0x8000u) | ((122u + ((r2 >> 2 * b) & 3) % 3) << 7) | (q & 0x7Fu)) << 16 * b; } break;
    }
    p[i] = o;
  }
}
__global__ void k_fill_f32(float* p, size_t n, uint64_t seed, float lo, float hi) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = lo + (hi - lo) * (float)(mix64(seed + i) >> 40) * 0x1p-24f;
}
static void fill(void* p, size_t bytes, uint64_t seed, int mode) {
  k_fill<<<480, 256>>>((uint32_t*)p, bytes / 4, seed, mode);
  CK(cudaGetLastError());
}
static void fill_f32(float* p, size_t n, uint64_t seed, float lo, float hi) {
  k_fill_f32<<<480, 256>>>(p, n, seed, lo, hi);
  CK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------
// the software pipeline every streaming warp runs: P units in flight in registers
#define RING(P, NUNITS, RAW, LOAD, COMP)                                                   \
  {                                                                                         \
    RAW buf_[P];                                                                            \
    _Pragma("unroll") for (int p_ = 0; p_ < P; ++p_) if (p_ < (NUNITS)) LOAD(buf_[p_], p_); \
    for (int u0_ = 0; u0_ < (NUNITS); u0_ += P) {                                           \
      _Pragma("unroll") for (int p_ = 0; p_ < P; ++p_) {                                    \
        const int u_ = u0_ + p_;                                                            \
        if (u_ < (NUNITS)) {                                                                \
          COMP(buf_[p_], u_);                                                               \
          if (u_ + P < (NUNITS)) LOAD(buf_[p_], u_ + P);                                    \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

// ---------------------------------------------------------------------------------------------------------
// Q4 / S3 tile layouts (per matrix [N][K], J = K / 128 units of 128 k):
//   Q4 values  [N/8][J][32 lanes][16 B]: lane (g, t) holds row g, k = 128j + 64q + 16t + i in byte 8q + i/2
//              (low nibble = even i), q = 0, 1
//   S3 codes   [N/8][J][3 words P0 P1 Q][32 lanes] (the lane's 32 values v[16q + i] as in common.cuh)
//   scales     [N/8][J][32 lanes][2 B]: byte q = the e4m3 scale of the lane's group q
// Bytes a matrix: Q4 N*K/2 + N*K/16, S3 N*K*3/8 + N*K/16.
HD size_t mat_vbytes(int f, int N, int K, int kb = 4) {
  return f == Q4 ? (size_t)N * K / 2 : f == S3 ? (size_t)N * K * 3 / 8 : (size_t)N * K * kb / 8;
}
HD size_t mat_sbytes(int f, int N, int K) { return f == TK ? (size_t)(N + K) * 2 : (size_t)N * K / 16; }

struct QArgs {
  const uint8_t* v; const uint8_t* s;  // pool bases
  size_t ve, se;                       // bytes an expert (values, scales): gate, up (and only down for dn)
  const int* ex;                       // the step's union, pool indices [U]
  const float* gs;                     // global scales [E][3] (gate, up, down)
  int U, M, S;
  const __nv_bfloat16* x;              // up: [M][H] activations; down: hb [U][M][I]
  float* y;                            // up: yp [S][U][M][NCH]; down: y [M][H]
  const float* wr;                     // down: routing weights [U][M]
  S3Tab T;
};
struct Q4Raw { uint4 v[4]; uint32_t s[4]; };
struct S3Raw { uint32_t p0[4], p1[4], q[4], s[4]; };

template <int F>
__device__ __forceinline__ void q_tile_mmas(float c[4], uint32_t xa0, uint32_t xa1, uint32_t xa2, uint32_t xa3, uint32_t xb0,
                                            uint32_t xb1, uint32_t xb2, uint32_t xb3, const uint32_t (&o)[8]) {
  mma_bf16(c, xa0, 0u, xa1, 0u, o[0], o[1]);
  mma_bf16(c, xa2, 0u, xa3, 0u, o[2], o[3]);
  mma_bf16(c, xb0, 0u, xb1, 0u, o[4], o[5]);
  mma_bf16(c, xb2, 0u, xb3, 0u, o[6], o[7]);
}
// one group (16 values) of a tile -> 8 bf16x2 words (the B fragments of four m16n8k16)
template <int F>
__device__ __forceinline__ void q_group(const uint4& v, uint32_t p0, uint32_t p1, uint32_t qq, uint32_t s, int q,
                                        const S3Tab& T, uint32_t (&o)[8]) {
  if constexpr (F == Q4) {
    const uint32_t f2 = e4m3_to_bf16x2_scaled(s >> 8 * q, 0x1p96f);
    q4_widen16(q ? v.z : v.x, q ? v.w : v.y, f2, o);
  } else {
    const uint32_t f2 = e4m3_to_bf16x2_scaled(s >> 8 * q, 1.0f);
    const uint32_t sa = s3_sel(p0, p1, qq, 2 * q), sb = s3_sel(p0, p1, qq, 2 * q + 1);
    s3_dec4(sa, T, f2, o[0], o[1]);
    s3_dec4(sa >> 16, T, f2, o[2], o[3]);
    s3_dec4(sb, T, f2, o[4], o[5]);
    s3_dec4(sb >> 16, T, f2, o[6], o[7]);
  }
}

// gate/up fused: warp = (expert, 16 intermediate rows: gate tiles 2rt, 2rt + 1 and up tiles 2rt, 2rt + 1, a K split s)
template <int F, int P>
__global__ void __launch_bounds__(128) k_up_q(QArgs a) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= a.U * 32 * a.S) return;
  const int s = gw % a.S, rt = (gw / a.S) % 32, ei = gw / (a.S * 32);
  const int e = a.ex[ei];
  constexpr int J = H / 128;
  const int j0 = s * J / a.S, n = (s + 1) * J / a.S - j0;
  const size_t vmat = mat_vbytes(F, I, H), smat = mat_sbytes(F, I, H);
  const uint8_t* vb[4];
  const uint8_t* sb[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int nt8 = 2 * rt + (i & 1);
    const size_t m = (i >> 1);
    vb[i] = a.v + e * a.ve + m * vmat + (size_t)nt8 * J * (F == Q4 ? 512 : 384) + lane * (F == Q4 ? 16 : 4);
    sb[i] = a.s + e * a.se + m * smat + (size_t)nt8 * J * 64 + lane * 2;
  }
  const bool live = g < a.M;
  const __nv_bfloat16* xr = a.x + (size_t)(live ? g : 0) * H + 16 * t;
  float c[4][4] = {};
  if constexpr (F == Q4) {
#define LOADQ(r, u) { _Pragma("unroll") for (int i = 0; i < 4; ++i) { const int j = j0 + (u); \
      r.v[i] = ldg_stream128(vb[i] + (size_t)j * 512); r.s[i] = ldg_stream16(sb[i] + (size_t)j * 64); } }
#define COMPQ(r, u) { const int j = j0 + (u); _Pragma("unroll") for (int q = 0; q < 2; ++q) {                \
      uint4 xa = make_uint4(0, 0, 0, 0), xc = xa;                                                            \
      if (live) { xa = __ldg((const uint4*)(xr + j * 128 + 64 * q)); xc = __ldg((const uint4*)(xr + j * 128 + 64 * q + 8)); } \
      _Pragma("unroll") for (int i = 0; i < 4; ++i) { uint32_t o[8];                                       \
        q_group<F>(r.v[i], 0, 0, 0, r.s[i], q, a.T, o);                                                      \
        q_tile_mmas<F>(c[i], xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w, o); } } }
    RING(P, n, Q4Raw, LOADQ, COMPQ);
#undef LOADQ
#undef COMPQ
  } else {
#define LOADS(r, u) { _Pragma("unroll") for (int i = 0; i < 4; ++i) { const int j = j0 + (u); \
      const uint8_t* b_ = vb[i] + (size_t)j * 384; r.p0[i] = ldg_stream32(b_); r.p1[i] = ldg_stream32(b_ + 128); \
      r.q[i] = ldg_stream32(b_ + 256); r.s[i] = ldg_stream16(sb[i] + (size_t)j * 64); } }
#define COMPS(r, u) { const int j = j0 + (u); _Pragma("unroll") for (int q = 0; q < 2; ++q) {                \
      uint4 xa = make_uint4(0, 0, 0, 0), xc = xa;                                                            \
      if (live) { xa = __ldg((const uint4*)(xr + j * 128 + 64 * q)); xc = __ldg((const uint4*)(xr + j * 128 + 64 * q + 8)); } \
      _Pragma("unroll") for (int i = 0; i < 4; ++i) { uint32_t o[8];                                       \
        q_group<F>(make_uint4(0, 0, 0, 0), r.p0[i], r.p1[i], r.q[i], r.s[i], q, a.T, o);                     \
        q_tile_mmas<F>(c[i], xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w, o); } } }
    RING(P, n, S3Raw, LOADS, COMPS);
#undef LOADS
#undef COMPS
  }
  if (live) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int m = i >> 1, ch = m * I + 16 * rt + 8 * (i & 1) + 2 * t;
      const float k = (F == Q4 ? 0x1p30f : 1.0f) * a.gs[e * 3 + m];
      float* yo = a.y + (((size_t)s * a.U + ei) * a.M + g) * NCH + ch;
      yo[0] = c[i][0] * k;
      yo[1] = c[i][1] * k;
    }
  }
}

// SwiGLU of the K-split partials -> bf16 h [U][M][I]
__global__ void k_mid_q(const float* yp, int U, int M, int S, __nv_bfloat16* hb) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= U * M * I) return;
  const int ch = idx % I, er = idx / I;
  float gt = 0.f, up = 0.f;
  for (int s = 0; s < S; ++s) {
    gt += yp[((size_t)s * U * M + er) * NCH + ch];
    up += yp[((size_t)s * U * M + er) * NCH + I + ch];
  }
  hb[idx] = __float2bfloat16_rn(gt / (1.f + expf(-gt)) * up);
}

// down: warp = 16 output channels (tiles 2w, 2w + 1 of [H][I]) over every expert of the union, the routing-weighted
// sum in registers in union order
template <int F, int P>
__global__ void __launch_bounds__(128) k_down_q(QArgs a) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= H / 16) return;
  constexpr int J = I / 128;
  const int n = a.U * J;
  const bool live = g < a.M;
  const size_t voff = (size_t)(2 * gw) * J * (F == Q4 ? 512 : 384) + lane * (F == Q4 ? 16 : 4);
  const size_t vstep = J * (F == Q4 ? 512 : 384);  // the second tile
  const size_t soff = (size_t)(2 * gw) * J * 64 + lane * 2;
  float c[2][4] = {}, acc[2][2] = {};
  auto vptr = [&](int u, int i) { return a.v + a.ex[u / J] * a.ve + voff + i * vstep + (size_t)(u % J) * (F == Q4 ? 512 : 384); };
  auto sptr = [&](int u, int i) { return a.s + a.ex[u / J] * a.se + soff + i * J * 64 + (size_t)(u % J) * 64; };
  auto flush = [&](int u) {
    if ((u % J) != J - 1) return;
    const int ei = u / J;
    const float k = (F == Q4 ? 0x1p30f : 1.0f) * a.gs[a.ex[ei] * 3 + 2] * (live ? a.wr[ei * a.M + g] : 0.f);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      acc[i][0] += k * c[i][0]; acc[i][1] += k * c[i][1];
      c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
    }
  };
  struct DQ4 { uint4 v[2]; uint32_t s[2]; };
  struct DS3 { uint32_t p0[2], p1[2], q[2], s[2]; };
  if constexpr (F == Q4) {
#define LOADQ(r, u) { _Pragma("unroll") for (int i = 0; i < 2; ++i) { r.v[i] = ldg_stream128(vptr(u, i)); r.s[i] = ldg_stream16(sptr(u, i)); } }
#define COMPQ(r, u) { const int ei = (u) / J, j = (u) % J;                                                   \
      const __nv_bfloat16* xr = a.x + ((size_t)ei * a.M + (live ? g : 0)) * I + 16 * t;                      \
      _Pragma("unroll") for (int q = 0; q < 2; ++q) {                                                        \
      uint4 xa = make_uint4(0, 0, 0, 0), xc = xa;                                                            \
      if (live) { xa = __ldg((const uint4*)(xr + j * 128 + 64 * q)); xc = __ldg((const uint4*)(xr + j * 128 + 64 * q + 8)); } \
      _Pragma("unroll") for (int i = 0; i < 2; ++i) { uint32_t o[8];                                       \
        q_group<F>(r.v[i], 0, 0, 0, r.s[i], q, a.T, o);                                                      \
        q_tile_mmas<F>(c[i], xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w, o); } } flush(u); }
    RING(P, n, DQ4, LOADQ, COMPQ);
#undef LOADQ
#undef COMPQ
  } else {
#define LOADS(r, u) { _Pragma("unroll") for (int i = 0; i < 2; ++i) { const uint8_t* b_ = vptr(u, i);          \
      r.p0[i] = ldg_stream32(b_); r.p1[i] = ldg_stream32(b_ + 128); r.q[i] = ldg_stream32(b_ + 256); r.s[i] = ldg_stream16(sptr(u, i)); } }
#define COMPS(r, u) { const int ei = (u) / J, j = (u) % J;                                                   \
      const __nv_bfloat16* xr = a.x + ((size_t)ei * a.M + (live ? g : 0)) * I + 16 * t;                      \
      _Pragma("unroll") for (int q = 0; q < 2; ++q) {                                                        \
      uint4 xa = make_uint4(0, 0, 0, 0), xc = xa;                                                            \
      if (live) { xa = __ldg((const uint4*)(xr + j * 128 + 64 * q)); xc = __ldg((const uint4*)(xr + j * 128 + 64 * q + 8)); } \
      _Pragma("unroll") for (int i = 0; i < 2; ++i) { uint32_t o[8];                                       \
        q_group<F>(make_uint4(0, 0, 0, 0), r.p0[i], r.p1[i], r.q[i], r.s[i], q, a.T, o);                     \
        q_tile_mmas<F>(c[i], xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w, o); } } flush(u); }
    RING(P, n, DS3, LOADS, COMPS);
#undef LOADS
#undef COMPS
  }
  if (live)
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float* yo = a.y + (size_t)g * H + 16 * gw + 8 * i + 2 * t;
      yo[0] = acc[i][0];
      yo[1] = acc[i][1];
    }
}

// dequantise one matrix through the kernels' own decode and fragment mapping (bit-exactness check)
template <int F>
__global__ void k_deq_q(const uint8_t* v, const uint8_t* s, int N, int K, S3Tab T, uint16_t* out) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), J = K / 128;
  if (gw >= N / 8 * J) return;
  const int nt8 = gw / J, j = gw % J;
  const uint8_t* vb = v + ((size_t)nt8 * J + j) * (F == Q4 ? 512 : 384) + lane * (F == Q4 ? 16 : 4);
  const uint32_t sc = *(const uint16_t*)(s + (((size_t)nt8 * J + j) * 32 + lane) * 2);
  uint4 vv = make_uint4(0, 0, 0, 0);
  uint32_t p0 = 0, p1 = 0, qq = 0;
  if constexpr (F == Q4) vv = *(const uint4*)vb;
  else { p0 = *(const uint32_t*)vb; p1 = *(const uint32_t*)(vb + 128); qq = *(const uint32_t*)(vb + 256); }
  for (int q = 0; q < 2; ++q) {
    uint32_t o[8];
    q_group<F>(vv, p0, p1, qq, sc, q, T, o);
    for (int m = 0; m < 8; ++m) {
      const size_t base = (size_t)(8 * nt8 + g) * K + 128 * j + 64 * q + 16 * t + 2 * m;
      out[base] = o[m] & 0xFFFF;
      out[base + 1] = o[m] >> 16;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// EXL3 trellis. Per matrix W_inner [Kin][N] (EXL3's [in][out]), tiles in N-strip order: tile (nt, kt) at
// (nt * Kin/16 + kt) * 8K words (EXL3 stores [kt][nt]; the reorder is a lossless permutation of whole tiles).
// suh [Kin], svh [N] f16. An expert: gate, up ([6144][512] each), down ([512][6144]); su: gate, up, down;
// sv: gate, up, down.
struct TArgs {
  const uint32_t* w;      // pool of tiles: expert e at e * we words: gate, up (each 6144*512*K/32 words), down
  size_t we;
  const __half* su; const __half* sv;  // [E][6144 + 6144 + 512], [E][512 + 512 + 6144]
  const int* ex;
  int U, M, S;
  const __nv_bfloat16* x;  // [M][H]
  __half* xh;              // pre: [U][2][M][H]; down's input: [U][M][I]
  float* yp;               // [S][U][M][NCH]
  float* y;                // [M][H]
  const float* wr;         // [U][M]
};
static constexpr size_t kSuE = 2 * H + I, kSvE = 2 * I + H;

// input transform: xh = f16(had(x * su) / sqrt(128)) per expert, matrix, row, 128-block; one warp each
__global__ void k_tr_pre(TArgs a) {
  const int lane = threadIdx.x & 31;
  const int items = a.U * 2 * a.M * (H / 128);
  for (int it = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); it < items; it += gridDim.x * (blockDim.x >> 5)) {
    const int blk = it % (H / 128), r = (it / (H / 128)) % a.M, mat = (it / (H / 128 * a.M)) % 2, ei = it / (H / 128 * a.M * 2);
    const int ch = blk * 128 + 4 * lane;
    const __half* su = a.su + (size_t)a.ex[ei] * kSuE + mat * H + ch;
    const __nv_bfloat16* xr = a.x + (size_t)r * H + ch;
    float h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __bfloat162float(xr[i]) * __half2float(su[i]);
    had128_warp(h);
    __half* o = a.xh + (((size_t)ei * 2 + mat) * a.M + r) * H + ch;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = __float2half_rn(h[i] * kHadScale);
  }
}

// gate/up: warp = (expert, n-tile nt of gate and of up, K split s); one u32 a lane a tile (lanes < 8K), two shuffles
template <int K, int CB, int P>
__global__ void __launch_bounds__(128) k_tr_up(TArgs a) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= a.U * 32 * a.S) return;
  const int s = gw % a.S, nt = (gw / a.S) % 32, ei = gw / (a.S * 32), e = a.ex[ei];
  constexpr int KT = H / 16, W = 8 * K;
  const int kt0 = s * KT / a.S, n = (s + 1) * KT / a.S - kt0;
  const size_t mw = (size_t)H * I * K / 32;
  const uint32_t* tb[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) tb[m] = a.w + e * a.we + m * mw + ((size_t)nt * KT + kt0) * W + (lane < W ? lane : 0);
  int i0, i1, o0;
  tr_lane_consts<K>(lane, i0, i1, o0);
  const bool live = g < a.M;
  const __half* xb[2];
#pragma unroll
  for (int m = 0; m < 2; ++m) xb[m] = a.xh + (((size_t)ei * 2 + m) * a.M + (live ? g : 0)) * H + kt0 * 16 + 2 * t;
  float c[2][4] = {};
  struct R { uint32_t w[2]; };
#define LOADT(r, u) { _Pragma("unroll") for (int m = 0; m < 2; ++m) r.w[m] = lane < W ? ldg_stream32(tb[m] + (size_t)(u) * W) : 0u; }
#define COMPT(r, u) { _Pragma("unroll") for (int m = 0; m < 2; ++m) {                                         \
      const uint32_t wa = __shfl_sync(0xffffffffu, r.w[m], i0), wb = __shfl_sync(0xffffffffu, r.w[m], i1); \
      uint32_t af[4]; tr_lane_frag<K, CB>(wa, wb, o0, af);                                                   \
      uint32_t b0 = 0, b1 = 0;                                                                               \
      if (live) { b0 = __ldg((const uint32_t*)(xb[m] + (u) * 16)); b1 = __ldg((const uint32_t*)(xb[m] + (u) * 16 + 8)); } \
      mma_f16(c[m], af, b0, b1); } }
  RING(P, n, R, LOADT, COMPT);
#undef LOADT
#undef COMPT
  // c: (channel g, rows 2t, 2t + 1), (channel g + 8, rows 2t, 2t + 1)
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int row = 2 * t + rr;
        if (row < a.M) a.yp[(((size_t)s * a.U + ei) * a.M + row) * NCH + m * I + nt * 16 + g + 8 * h] = c[m][2 * h + rr];
      }
}

// output transform of gate/up, SwiGLU, bf16 h, down's input transform: one warp a (expert, row, 128-block)
__global__ void k_tr_mid(TArgs a) {
  const int lane = threadIdx.x & 31;
  const int items = a.U * a.M * (I / 128);
  const int it = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (it >= items) return;
  const int blk = it % (I / 128), er = it / (I / 128), ei = er / a.M;
  const int ch = blk * 128 + 4 * lane;
  const __half* sv = a.sv + (size_t)a.ex[ei] * kSvE;
  const __half* su = a.su + (size_t)a.ex[ei] * kSuE + 2 * H;
  float gt[4], up[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    gt[i] = up[i] = 0.f;
    for (int s = 0; s < a.S; ++s) {
      gt[i] += a.yp[((size_t)s * a.U * a.M + er) * NCH + ch + i];
      up[i] += a.yp[((size_t)s * a.U * a.M + er) * NCH + I + ch + i];
    }
  }
  had128_warp(gt);
  had128_warp(up);
  float h[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float gg = gt[i] * kHadScale * __half2float(sv[ch + i]), uu = up[i] * kHadScale * __half2float(sv[I + ch + i]);
    h[i] = __bfloat162float(__float2bfloat16_rn(gg / (1.f + expf(-gg)) * uu)) * __half2float(su[ch + i]);
  }
  had128_warp(h);
#pragma unroll
  for (int i = 0; i < 4; ++i) a.xh[(size_t)er * I + ch + i] = __float2half_rn(h[i] * kHadScale);
}

// down: CTA = 128 output channels (one Hadamard block), 16 warps = 8 n-tiles x 2 K halves, every expert of the
// union in turn; per expert the partials meet in shared memory, warps 0..M-1 take one row each through the output
// transform (had, sv, routing weight) and keep the sum in registers. One barrier an expert (double-buffered).
template <int K, int CB, int P>
__global__ void __launch_bounds__(512, 1) k_tr_down(TArgs a) {
  __shared__ float part[2][2][128][8];
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, w = threadIdx.x >> 5;
  const int b = blockIdx.x, ntl = w & 7, kh = w >> 3, nt = b * 8 + ntl;
  constexpr int KT = I / 16, W = 8 * K;  // 32 k-tiles, 16 a half
  const size_t dw = 2 * ((size_t)H * I * K / 32);
  int i0, i1, o0;
  tr_lane_consts<K>(lane, i0, i1, o0);
  const bool live = g < a.M;
  const int n = a.U * 16;
  float c[4] = {}, acc[4] = {};
  auto tptr = [&](int u) {
    return a.w + a.ex[u >> 4] * a.we + dw + ((size_t)nt * KT + kh * 16 + (u & 15)) * W + (lane < W ? lane : 0);
  };
  auto flush = [&](int u) {
    if ((u & 15) != 15) return;
    const int ei = u >> 4, pb = ei & 1;
    part[pb][kh][ntl * 16 + g][2 * t] = c[0];
    part[pb][kh][ntl * 16 + g][2 * t + 1] = c[1];
    part[pb][kh][ntl * 16 + g + 8][2 * t] = c[2];
    part[pb][kh][ntl * 16 + g + 8][2 * t + 1] = c[3];
    c[0] = c[1] = c[2] = c[3] = 0.f;
    __syncthreads();
    if (w < a.M) {
      float h[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) h[i] = part[pb][0][4 * lane + i][w] + part[pb][1][4 * lane + i][w];
      had128_warp(h);
      const __half* sv = a.sv + (size_t)a.ex[ei] * kSvE + 2 * I + b * 128 + 4 * lane;
      const float wt = a.wr[ei * a.M + w];
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] += h[i] * kHadScale * __half2float(sv[i]) * wt;
    }
  };
  struct R { uint32_t w; };
#define LOADT(r, u) { r.w = lane < W ? ldg_stream32(tptr(u)) : 0u; }
#define COMPT(r, u) { const uint32_t wa = __shfl_sync(0xffffffffu, r.w, i0), wb = __shfl_sync(0xffffffffu, r.w, i1); \
      uint32_t af[4]; tr_lane_frag<K, CB>(wa, wb, o0, af);                                                        \
      uint32_t b0 = 0, b1 = 0;                                                                                    \
      if (live) { const __half* xr = a.xh + ((size_t)((u) >> 4) * a.M + g) * I + (kh * 16 + ((u) & 15)) * 16 + 2 * t; \
                  b0 = __ldg((const uint32_t*)xr); b1 = __ldg((const uint32_t*)(xr + 8)); }                         \
      mma_f16(c, af, b0, b1); flush(u); }
  RING(P, n, R, LOADT, COMPT);
#undef LOADT
#undef COMPT
  if (w < a.M)
#pragma unroll
    for (int i = 0; i < 4; ++i) a.y[(size_t)w * H + b * 128 + 4 * lane + i] = acc[i];
}

// the trellis decode through the kernels' fragment mapping: tile (nt, kt) of a [Kin][N] matrix -> f16 W_inner
template <int K, int CB>
__global__ void k_deq_tr(const uint32_t* w, int Kin, int N, uint16_t* out) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), KT = Kin / 16, Wd = 8 * K;
  if (gw >= KT * (N / 16)) return;
  const int nt = gw / KT, kt = gw % KT;
  const uint32_t* tp = w + (size_t)gw * Wd;
  int i0, i1, o0;
  tr_lane_consts<K>(lane, i0, i1, o0);
  uint32_t af[4];
  tr_lane_frag<K, CB>(tp[i0], tp[i1], o0, af);
  const int nn[4] = {g, g + 8, g, g + 8}, kk[4] = {2 * t, 2 * t, 2 * t + 8, 2 * t + 8};
  for (int r = 0; r < 4; ++r)
    for (int h = 0; h < 2; ++h) out[(size_t)(kt * 16 + kk[r] + h) * N + nt * 16 + nn[r]] = (af[r] >> 16 * h) & 0xFFFF;
}

// ---------------------------------------------------------------------------------------------------------
// dense attention weights. bf16: natural [N][K]; lane (g, t) reads row g, k = 64j + 16t .. + 15 (32 B).
// fp8: [N/8][K/128][32 lanes][32 B] (lane: row g, k = 128j + 64q + 16t + i at byte 16q + i) + f32 scales [N/128][K/128]
template <int P>
__global__ void __launch_bounds__(128) k_dense_bf16(const __nv_bfloat16* Wt, const __nv_bfloat16* x, int M, int N, int K, float* y) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= N / 8) return;
  const bool live = g < M;
  const __nv_bfloat16* wr = Wt + (size_t)(8 * gw + g) * K + 16 * t;
  const __nv_bfloat16* xr = x + (size_t)(live ? g : 0) * K + 16 * t;
  float c[4] = {};
  struct R { uint4 a, b; };
#define LOADB(r, u) { r.a = ldg_stream128(wr + (size_t)(u) * 64); r.b = ldg_stream128(wr + (size_t)(u) * 64 + 8); }
#define COMPB(r, u) { uint4 xa = make_uint4(0, 0, 0, 0), xc = xa;                                           \
      if (live) { xa = __ldg((const uint4*)(xr + (u) * 64)); xc = __ldg((const uint4*)(xr + (u) * 64 + 8)); } \
      mma_bf16(c, xa.x, 0u, xa.y, 0u, r.a.x, r.a.y); mma_bf16(c, xa.z, 0u, xa.w, 0u, r.a.z, r.a.w);           \
      mma_bf16(c, xc.x, 0u, xc.y, 0u, r.b.x, r.b.y); mma_bf16(c, xc.z, 0u, xc.w, 0u, r.b.z, r.b.w); }
  RING(P, K / 64, R, LOADB, COMPB);
#undef LOADB
#undef COMPB
  if (live) { y[(size_t)g * N + 8 * gw + 2 * t] = c[0]; y[(size_t)g * N + 8 * gw + 2 * t + 1] = c[1]; }
}
template <int P>
__global__ void __launch_bounds__(128) k_dense_fp8(const uint8_t* Wt, const float* Sc, const __nv_bfloat16* x, int M, int N, int K,
                                                   float* y) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (gw >= N / 8) return;
  const int J = K / 128;
  const bool live = g < M;
  const uint8_t* wr = Wt + (size_t)gw * J * 1024 + lane * 32;
  const float* sr = Sc + (size_t)(8 * gw / 128) * J;
  const __nv_bfloat16* xr = x + (size_t)(live ? g : 0) * K + 16 * t;
  float c[4] = {}, acc[2] = {};
  struct R { uint4 a, b; float s; };
#define LOADF(r, u) { r.a = ldg_stream128(wr + (size_t)(u) * 1024); r.b = ldg_stream128(wr + (size_t)(u) * 1024 + 16); r.s = __ldg(sr + (u)); }
#define COMPF(r, u) { _Pragma("unroll") for (int q = 0; q < 2; ++q) { const uint4 v = q ? r.b : r.a; uint32_t o[8]; \
      fp8x4_to_bf16(v.x, o[0], o[1]); fp8x4_to_bf16(v.y, o[2], o[3]); fp8x4_to_bf16(v.z, o[4], o[5]); fp8x4_to_bf16(v.w, o[6], o[7]); \
      uint4 xa = make_uint4(0, 0, 0, 0), xc = xa;                                                                  \
      if (live) { xa = __ldg((const uint4*)(xr + (u) * 128 + 64 * q)); xc = __ldg((const uint4*)(xr + (u) * 128 + 64 * q + 8)); } \
      mma_bf16(c, xa.x, 0u, xa.y, 0u, o[0], o[1]); mma_bf16(c, xa.z, 0u, xa.w, 0u, o[2], o[3]);                       \
      mma_bf16(c, xc.x, 0u, xc.y, 0u, o[4], o[5]); mma_bf16(c, xc.z, 0u, xc.w, 0u, o[6], o[7]); }                     \
      acc[0] += r.s * c[0]; acc[1] += r.s * c[1]; c[0] = c[1] = c[2] = c[3] = 0.f; }
  RING(P, J, R, LOADF, COMPF);
#undef LOADF
#undef COMPF
  if (live) { y[(size_t)g * N + 8 * gw + 2 * t] = acc[0]; y[(size_t)g * N + 8 * gw + 2 * t + 1] = acc[1]; }
}

// streaming read (ds41f micro/bw.cu's unrolled form): the window's own bandwidth baseline
__global__ void k_read(const int4* __restrict__ p, size_t n, int* out) {
  int4 acc = make_int4(0, 0, 0, 0);
  size_t stride = (size_t)gridDim.x * blockDim.x, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    int4 a = __ldcs(p + i), b = __ldcs(p + i + stride), c = __ldcs(p + i + 2 * stride), d = __ldcs(p + i + 3 * stride);
    acc.x ^= a.x ^ b.x ^ c.x ^ d.x; acc.y ^= a.y ^ b.y ^ c.y ^ d.y; acc.z ^= a.z ^ b.z ^ c.z ^ d.z; acc.w ^= a.w ^ b.w ^ c.w ^ d.w;
  }
  for (; i < n; i += stride) { int4 v = p[i]; acc.x ^= v.x; }
  if (acc.x == 0x12345678 && acc.y == 1) out[0] = acc.z + acc.w;
}

// =========================================================================================================
// host side
static FILE* g_out = nullptr;
static std::string g_cond;
static void result(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  printf("%s\n", buf);
  if (g_out) { fprintf(g_out, "%s\n", buf); fflush(g_out); }
}
template <class T>
static std::vector<T> d2h(const void* p, size_t n) {
  std::vector<T> v(n);
  CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
// time a recorded sequence: capture `steps` steps (each a lambda launching that step's kernels) in one graph,
// replay it reps times; the median per-step time in microseconds
template <class F>
static double time_graph(int steps, F launch, int reps = 7) {
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaGraph_t gr;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  for (int s = 0; s < steps; ++s) launch(s, st);
  CK(cudaStreamEndCapture(st, &gr));
  CK(cudaGraphInstantiate(&ge, gr, 0));
  CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<double> v;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(e0, st));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    v.push_back(ms * 1e3 / steps);
  }
  CK(cudaGetLastError());
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(gr);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaStreamDestroy(st);
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static double g_bw_window = 0;  // this window's streaming read, GB/s

struct ErrStat {  // |gpu - ref| / sum |terms| (the f32 accumulation's scale), worst case
  double worst = 0, worst_abs = 0; size_t n = 0, bad = 0;
  void add(double gpu, double ref, double mag, double tol) {
    double e = fabs(gpu - ref) / (mag > 0 ? mag : 1);
    if (e > worst) { worst = e; worst_abs = fabs(gpu - ref); }
    if (!(e <= tol)) ++bad;
    ++n;
  }
};
static const double kTolSum = 1e-4;  // f32 (tensor-core) accumulation over <= 6144 terms in another order: |err| well under K * 2^-24 * sum|terms|

// ---------------------------------------------------------------------------------------------------------
// canonical views of the device layouts, read independently of the kernels' decode
static float q_weight(int f, const uint8_t* v, const uint8_t* s, int N, int K, int n, int k) {  // e2m1/subset value * scale
  const int J = K / 128, nt8 = n / 8, g = n % 8, j = k / 128, q = (k % 128) / 64, t = (k % 64) / 16, i = k % 16, lane = 4 * g + t;
  const uint8_t sc = s[(((size_t)nt8 * J + j) * 32 + lane) * 2 + q];
  int nib;
  if (f == Q4) {
    const uint8_t by = v[(((size_t)nt8 * J + j) * 32 + lane) * 16 + 8 * q + i / 2];
    nib = (i & 1) ? by >> 4 : by & 15;
  } else {
    const uint32_t* wp = (const uint32_t*)(v + ((size_t)nt8 * J + j) * 384);
    const uint32_t P0 = wp[lane], P1 = wp[32 + lane], Qw = wp[64 + lane];
    const int idx = 16 * q + i, m = idx % 8, hf = (idx / 8) % 2;
    const int low2 = ((idx < 16 ? P0 : P1) >> (4 * m + 2 * hf)) & 3, hb = (Qw >> (4 * m + idx / 8)) & 1;
    nib = kS3Nibble[hb << 2 | low2];
  }
  return e2m1_to_f(nib) * e4m3_to_f(sc);
}
template <int K, int CB>
static void tr_matrix_ref(const uint32_t* w, int Kin, int N, std::vector<float>& out) {  // W_inner [Kin][N]
  out.assign((size_t)Kin * N, 0.f);
  const int KT = Kin / 16;
#pragma omp parallel for schedule(static)
  for (int nt = 0; nt < N / 16; ++nt)
    for (int kt = 0; kt < KT; ++kt) {
      const uint32_t* tp = w + ((size_t)nt * KT + kt) * 8 * K;
      for (int p = 0; p < 256; ++p) {
        const int e = exl3_perm(p);
        out[(size_t)(kt * 16 + e / 16) * N + nt * 16 + e % 16] = h_bits_to_f(exl3_dq_ref<K, CB>(tp, p));
      }
    }
}

// ---------------------------------------------------------------------------------------------------------
// the host self-check: the decoders and the lane mappings on host data, no GPU
template <int K, int CB>
static int cpu_check_tr() {
  int bad = 0;
  for (int tile = 0; tile < 64; ++tile) {
    uint32_t wd[8 * K];
    for (int i = 0; i < 8 * K; ++i) wd[i] = (uint32_t)mix64(tile * 1000 + i + K * 77 + CB * 7);
    uint16_t ref[256], got[256];
    for (int p = 0; p < 256; ++p) ref[exl3_perm(p)] = exl3_dq_ref<K, CB>(wd, p);
    for (int lane = 0; lane < 32; ++lane) {
      int i0, i1, o0;
      tr_lane_consts<K>(lane, i0, i1, o0);
      uint32_t af[4];
      tr_lane_frag<K, CB>(wd[i0], wd[i1], o0, af);
      const int g = lane >> 2, t = lane & 3, nn[4] = {g, g + 8, g, g + 8}, kk[4] = {2 * t, 2 * t, 2 * t + 8, 2 * t + 8};
      for (int r = 0; r < 4; ++r)
        for (int h = 0; h < 2; ++h) got[(kk[r] + h) * 16 + nn[r]] = (af[r] >> 16 * h) & 0xFFFF;
    }
    for (int i = 0; i < 256; ++i) bad += ref[i] != got[i];
  }
  // the perm is a permutation
  std::vector<int> seen(256, 0);
  for (int p = 0; p < 256; ++p) seen[exl3_perm(p)]++;
  for (int i = 0; i < 256; ++i) bad += seen[i] != 1;
  printf("cpu-check trellis K%d cb%d: lane fragments vs EXL3's dq + tensor_core_perm, 64 tiles: %s (%d mismatches)\n", K, CB,
         bad ? "FAIL" : "ok", bad);
  return bad;
}
static int cpu_check() {
  int bad = 0;
  // Q4: q4_widen16 against the nibble table, every byte value, a range of scales
  {
    int b = 0;
    for (int sc = 0x08; sc < 0x7F; sc += 3)
      for (int r = 0; r < 64; ++r) {
        uint32_t lo = (uint32_t)mix64(r * 7 + sc), hi = (uint32_t)mix64(r * 7 + sc + 1000), o[8];
        q4_widen16(lo, hi, q4_scale_f2_host(sc), o);
        for (int i = 0; i < 16; ++i) {
          const int nib = ((i < 8 ? lo : hi) >> 4 * (i % 8)) & 15;
          const uint16_t want = f_to_bf16_bits(e2m1_to_f(nib) * e4m3_to_f(sc) * 0x1p-30f);
          const uint16_t gotb = (o[i / 2] >> 16 * (i & 1)) & 0xFFFF;
          b += want != gotb && !(nib == 8 && gotb == 0x8000) ? 1 : 0;
        }
      }
    printf("cpu-check q4 widen16 (all scales 0x08..0x7e, 64 random words each): %s (%d)\n", b ? "FAIL" : "ok", b);
    bad += b;
  }
  // S3: pack a lane, decode through the selector/prmt path, compare with the code table
  {
    int b = 0;
    S3Tab T = s3_tables();
    for (int r = 0; r < 4096; ++r) {
      uint8_t v[32];
      for (int i = 0; i < 32; ++i) v[i] = (uint8_t)(mix64(r * 33 + i) & 7);
      uint32_t P0, P1, Q;
      s3_pack_lane(v, P0, P1, Q);
      const uint8_t sc = 0x30 + r % 0x18;
      const uint32_t f2 = f_to_bf16_bits(e4m3_to_f(sc)) * 0x10001u;
      for (int which = 0; which < 4; ++which) {
        const uint32_t sel = s3_sel(P0, P1, Q, which);
        uint32_t w[4];
        s3_dec4(sel, T, f2, w[0], w[1]);
        s3_dec4(sel >> 16, T, f2, w[2], w[3]);
        for (int i = 0; i < 8; ++i) {
          const uint16_t want = f_to_bf16_bits(e2m1_to_f(kS3Nibble[v[8 * which + i]]) * e4m3_to_f(sc));
          b += want != ((w[i / 2] >> 16 * (i & 1)) & 0xFFFF);
        }
      }
    }
    printf("cpu-check s3 pack + prmt decode (4096 lanes): %s (%d)\n", b ? "FAIL" : "ok", b);
    bad += b;
  }
  // FP8: every byte but the NaNs
  {
    int b = 0;
    for (int v = 0; v < 256; ++v) {
      if ((v & 0x7F) == 0x7F) continue;
      uint32_t o0, o1;
      fp8x4_to_bf16((uint32_t)v * 0x01010101u, o0, o1);
      const uint16_t want = f_to_bf16_bits(e4m3_to_f(v));
      b += (o0 & 0xFFFF) != want || (o0 >> 16) != want || (o1 & 0xFFFF) != want || (o1 >> 16) != want;
    }
    printf("cpu-check fp8 -> bf16 (all 254 values): %s (%d)\n", b ? "FAIL" : "ok", b);
    bad += b;
  }
  bad += cpu_check_tr<3, 0>() + cpu_check_tr<4, 0>() + cpu_check_tr<3, 1>() + cpu_check_tr<3, 2>() + cpu_check_tr<2, 0>();
  // the FWHT is orthogonal up to its scale: had(had(v)) = 128 v
  {
    float v[128], w[128];
    for (int i = 0; i < 128; ++i) v[i] = w[i] = (float)((int)(mix64(i) % 17) - 8);
    had128_ref(w);
    had128_ref(w);
    int b = 0;
    for (int i = 0; i < 128; ++i) b += w[i] != 128 * v[i];
    printf("cpu-check had128 twice = 128 x: %s\n", b ? "FAIL" : "ok");
    bad += b;
  }
  printf("cpu-check: %s\n", bad ? "FAILED" : "all ok");
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------
// MoE benchmark of one format
struct MoeCfg { int f, K, CB; const char* name; };
static const double kPoolBytes = 2.2e9;

static std::vector<int> union_of_step(int s, int U, int E) {
  std::vector<int> v(U);
  for (int i = 0; i < U; ++i) v[i] = (int)(((size_t)s * U + i) % E);
  return v;
}

template <int F, int P>
static void launch_up_q(const QArgs& a, cudaStream_t st) {
  k_up_q<F, P><<<(a.U * 32 * a.S + 3) / 4, 128, 0, st>>>(a);
}
template <int F, int P>
static void launch_down_q(const QArgs& a, cudaStream_t st) {
  k_down_q<F, P><<<H / 16 / 4, 128, 0, st>>>(a);
}

static void bench_moe_q(int F, const char* name, const std::vector<int>& rows_list) {
  const size_t vgu = 2 * mat_vbytes(F, I, H), sgu = 2 * mat_sbytes(F, I, H);
  const size_t vdn = mat_vbytes(F, H, I), sdn = mat_sbytes(F, H, I);
  const size_t per_e = vgu + sgu + vdn + sdn;
  const int E = (int)ceil(kPoolBytes / per_e);
  printf("== %s: %.3f MB an expert (gate+up %.3f, down %.3f), pool %d experts = %.2f GB\n", name, per_e / 1e6, (vgu + sgu) / 1e6,
         (vdn + sdn) / 1e6, E, (double)E * per_e / 1e9);
  uint8_t *gv, *gsb, *dv, *dsb;
  float *gsc, *yp, *y, *wr;
  __nv_bfloat16 *x, *hb;
  int* ex;
  CK(cudaMalloc(&gv, vgu * E)); CK(cudaMalloc(&gsb, sgu * E)); CK(cudaMalloc(&dv, vdn * E)); CK(cudaMalloc(&dsb, sdn * E));
  CK(cudaMalloc(&gsc, E * 3 * 4));
  const int Umax = 28, Mmax = 8, Smax = 4, NSmax = 512;
  CK(cudaMalloc(&yp, (size_t)Smax * Umax * Mmax * NCH * 4)); CK(cudaMalloc(&y, (size_t)Mmax * H * 4));
  CK(cudaMalloc(&wr, Umax * Mmax * 4)); CK(cudaMalloc(&x, (size_t)Mmax * H * 2)); CK(cudaMalloc(&hb, (size_t)Umax * Mmax * I * 2));
  CK(cudaMalloc(&ex, (size_t)NSmax * Umax * 4));
  fill(gv, vgu * E, 11 + F, 0); fill(dv, vdn * E, 12 + F, 0);
  fill(gsb, sgu * E, 13 + F, 1); fill(dsb, sdn * E, 14 + F, 1);
  fill_f32(gsc, E * 3, 15, 0.002f, 0.004f);
  fill(x, (size_t)Mmax * H * 2, 16, 5);
  fill_f32(wr, Umax * Mmax, 17, 0.05f, 0.25f);
  CK(cudaDeviceSynchronize());
  const S3Tab T = s3_tables();

  // bit-exactness of the decode: expert 0's gate and down through the kernels' fragment mapping
  {
    int bad = 0;
    for (int which = 0; which < 2; ++which) {
      const int N = which ? H : I, K = which ? I : H;
      const uint8_t* v = which ? dv : gv;
      const uint8_t* s = which ? dsb : gsb;
      uint16_t* out;
      CK(cudaMalloc(&out, (size_t)N * K * 2));
      const int warps = N / 8 * (K / 128);
      if (F == Q4) k_deq_q<Q4><<<(warps + 7) / 8, 256>>>(v, s, N, K, T, out);
      else k_deq_q<S3><<<(warps + 7) / 8, 256>>>(v, s, N, K, T, out);
      CK(cudaDeviceSynchronize());
      auto got = d2h<uint16_t>(out, (size_t)N * K);
      auto hv = d2h<uint8_t>(v, mat_vbytes(F, N, K));
      auto hs = d2h<uint8_t>(s, mat_sbytes(F, N, K));
      for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
          float want = q_weight(F, hv.data(), hs.data(), N, K, n, k) * (F == Q4 ? 0x1p-30f : 1.f);
          uint16_t wb = f_to_bf16_bits(want);
          if (got[(size_t)n * K + k] != wb && !(want == 0 && (got[(size_t)n * K + k] & 0x7FFF) == 0)) ++bad;
        }
      CK(cudaFree(out));
    }
    result("CHECK\t%s\tdecode bit-exact (expert 0 gate [512x6144] + down [6144x512], %s)\t%s\t%d mismatches", name,
           F == Q4 ? "bf16 of e2m1*e4m3*2^-30" : "bf16 of subset*e4m3", bad ? "FAIL" : "ok", bad);
  }

  for (int M : rows_list) {
    const int U = M == 1 ? 8 : 28;
    for (int S : M == 1 ? std::vector<int>{2, 3, 4} : std::vector<int>{1}) {
    const int NS = std::min(NSmax, std::max(16, (E + U - 1) / U));
    std::vector<int> hex;
    for (int s = 0; s < NS; ++s) { auto u = union_of_step(s, U, E); hex.insert(hex.end(), u.begin(), u.end()); }
    CK(cudaMemcpy(ex, hex.data(), hex.size() * 4, cudaMemcpyHostToDevice));
    auto args = [&](int s, bool up) {
      QArgs a;
      a.v = up ? gv : dv; a.s = up ? gsb : dsb; a.ve = up ? vgu : vdn; a.se = up ? sgu : sdn;
      a.ex = ex + (size_t)s * U; a.gs = gsc; a.U = U; a.M = M; a.S = S;
      a.x = up ? x : hb; a.y = up ? yp : y; a.wr = wr; a.T = T;
      return a;
    };
    for (int P : {2, 4}) {
      auto up = [&](int s, cudaStream_t st) {
        if (F == Q4) { if (P == 2) launch_up_q<Q4, 2>(args(s, true), st); else launch_up_q<Q4, 4>(args(s, true), st); }
        else { if (P == 2) launch_up_q<S3, 2>(args(s, true), st); else launch_up_q<S3, 4>(args(s, true), st); }
      };
      auto mid = [&](int s, cudaStream_t st) { k_mid_q<<<(U * M * I + 255) / 256, 256, 0, st>>>(yp, U, M, S, hb); };
      auto down = [&](int s, cudaStream_t st) {
        if (F == Q4) { if (P == 2) launch_down_q<Q4, 2>(args(s, false), st); else launch_down_q<Q4, 4>(args(s, false), st); }
        else { if (P == 2) launch_down_q<S3, 2>(args(s, false), st); else launch_down_q<S3, 4>(args(s, false), st); }
      };
      // correctness at step 0
      up(0, 0); mid(0, 0); down(0, 0);
      CK(cudaDeviceSynchronize());
      if (P == 2) {  // checked at every S
        auto hyp = d2h<float>(yp, (size_t)S * U * M * NCH);
        auto hhb = d2h<__nv_bfloat16>(hb, (size_t)U * M * I);
        auto hy = d2h<float>(y, (size_t)M * H);
        auto hx = d2h<__nv_bfloat16>(x, (size_t)M * H);
        auto hgs = d2h<float>(gsc, E * 3);
        auto hwr = d2h<float>(wr, U * M);
        ErrStat eu, ed;
        std::vector<double> yd((size_t)M * H, 0.0), ydm((size_t)M * H, 0.0);
        for (int ei = 0; ei < U; ++ei) {
          const int e = hex[ei];
          auto hv = d2h<uint8_t>(gv + e * vgu, vgu);
          auto hs = d2h<uint8_t>(gsb + e * sgu, sgu);
          auto dvv = d2h<uint8_t>(dv + e * vdn, vdn);
          auto dss = d2h<uint8_t>(dsb + e * sdn, sdn);
          std::vector<double> ref((size_t)M * NCH), mag((size_t)M * NCH);
#pragma omp parallel for schedule(static)
          for (int ch = 0; ch < NCH; ++ch) {
            const int m = ch / I, n = ch % I;
            const uint8_t* v = hv.data() + m * mat_vbytes(F, I, H);
            const uint8_t* s = hs.data() + m * mat_sbytes(F, I, H);
            std::vector<double> wrow(H);
            for (int k = 0; k < H; ++k) wrow[k] = q_weight(F, v, s, I, H, n, k) * (double)hgs[e * 3 + m];
            for (int r = 0; r < M; ++r) {
              double acc = 0, am = 0;
              for (int k = 0; k < H; ++k) { double tt = (double)__bfloat162float(hx[(size_t)r * H + k]) * wrow[k]; acc += tt; am += fabs(tt); }
              ref[(size_t)r * NCH + ch] = acc; mag[(size_t)r * NCH + ch] = am;
            }
          }
          for (int r = 0; r < M; ++r)
            for (int ch = 0; ch < NCH; ++ch) {
              double gsum = 0;
              for (int s = 0; s < S; ++s) gsum += hyp[(((size_t)s * U + ei) * M + r) * NCH + ch];
              eu.add(gsum, ref[(size_t)r * NCH + ch], mag[(size_t)r * NCH + ch], kTolSum);
            }
          // down from the GPU's h
#pragma omp parallel for schedule(static)
          for (int n = 0; n < H; ++n)
            for (int r = 0; r < M; ++r) {
              double acc = 0, am = 0;
              for (int k = 0; k < I; ++k) {
                double tt = (double)__bfloat162float(hhb[((size_t)ei * M + r) * I + k]) * q_weight(F, dvv.data(), dss.data(), H, I, n, k);
                acc += tt; am += fabs(tt);
              }
              const double kk = hgs[e * 3 + 2] * hwr[ei * M + r];
              yd[(size_t)r * H + n] += acc * kk; ydm[(size_t)r * H + n] += am * fabs(kk);
            }
        }
        for (size_t i = 0; i < yd.size(); ++i) ed.add(hy[i], yd[i], ydm[i], kTolSum);
        result("CHECK\t%s\tr%d up (gate+up, %d experts) vs CPU double\t%s\tmax |err|/sum|terms| %.2e (abs %.2e), %zu outputs", name, M, U,
               eu.bad ? "FAIL" : "ok", eu.worst, eu.worst_abs, eu.n);
        result("CHECK\t%s\tr%d down (%d experts, weighted sum) vs CPU double from the GPU's h\t%s\tmax |err|/sum|terms| %.2e (abs %.2e), %zu outputs",
               name, M, U, ed.bad ? "FAIL" : "ok", ed.worst, ed.worst_abs, ed.n);
      }
      const double bu = (double)U * (vgu + sgu), bd = (double)U * (vdn + sdn);
      const double tu = time_graph(NS, up), tm = time_graph(NS, mid), td = time_graph(NS, down);
      const double tt = time_graph(NS, [&](int s, cudaStream_t st) { up(s, st); mid(s, st); down(s, st); });
      auto line = [&](const char* part, double bytes, double us) {
        result("MOE\t%s\tr%d\tU%d\tP%d\tS%d\t%s\t%.0f\t%.2f\t%.1f\t%.1f\t%.1f", name, M, U, P, part == std::string("up") ? S : 1, part,
               bytes, us, bytes / us / 1e3, 100 * bytes / us / 1e3 / 235.0, 100 * bytes / us / 1e3 / g_bw_window);
      };
      line("up", bu, tu);
      line("mid", 0, tm);
      line("down", bd, td);
      line("step", bu + bd, tt);
    }
    }
  }
  CK(cudaFree(gv)); CK(cudaFree(gsb)); CK(cudaFree(dv)); CK(cudaFree(dsb)); CK(cudaFree(gsc)); CK(cudaFree(yp)); CK(cudaFree(y));
  CK(cudaFree(wr)); CK(cudaFree(x)); CK(cudaFree(hb)); CK(cudaFree(ex));
}

// ---------------------------------------------------------------------------------------------------------
template <int K, int CB, int P>
static void launch_tr_up(const TArgs& a, cudaStream_t st) { k_tr_up<K, CB, P><<<(a.U * 32 * a.S + 3) / 4, 128, 0, st>>>(a); }
template <int K, int CB, int P>
static void launch_tr_down(const TArgs& a, cudaStream_t st) { k_tr_down<K, CB, P><<<H / 128, 512, 0, st>>>(a); }

template <int K, int CB>
static void bench_moe_tr(const char* name, const std::vector<int>& rows_list) {
  const size_t mw = (size_t)H * I * K / 32;  // words a matrix
  const size_t we = 3 * mw;
  const size_t per_e = we * 4 + (kSuE + kSvE) * 2;
  const size_t bgu = 2 * mw * 4 + (2 * H + 2 * I) * 2, bdn = mw * 4 + (I + H) * 2;  // with su, sv
  const int E = (int)ceil(kPoolBytes / per_e);
  printf("== %s: %.3f MB an expert, pool %d experts = %.2f GB\n", name, per_e / 1e6, E, (double)E * per_e / 1e9);
  uint32_t* tw;
  __half *su, *sv, *xh, *xhd;
  float *yp, *y, *wr;
  __nv_bfloat16* x;
  int* ex;
  const int Umax = 28, Mmax = 8, Smax = 4, NSmax = 512;
  CK(cudaMalloc(&tw, we * 4 * E)); CK(cudaMalloc(&su, kSuE * 2 * E)); CK(cudaMalloc(&sv, kSvE * 2 * E));
  CK(cudaMalloc(&xh, (size_t)Umax * 2 * Mmax * H * 2)); CK(cudaMalloc(&xhd, (size_t)Umax * Mmax * I * 2));
  CK(cudaMalloc(&yp, (size_t)Smax * Umax * Mmax * NCH * 4)); CK(cudaMalloc(&y, (size_t)Mmax * H * 4));
  CK(cudaMalloc(&wr, Umax * Mmax * 4)); CK(cudaMalloc(&x, (size_t)Mmax * H * 2)); CK(cudaMalloc(&ex, (size_t)NSmax * Umax * 4));
  fill(tw, we * 4 * E, 21 + K, 0);
  fill(su, kSuE * 2 * E, 22, 2); fill(sv, kSvE * 2 * E, 23, 2);
  fill(x, (size_t)Mmax * H * 2, 16, 5);
  fill_f32(wr, Umax * Mmax, 17, 0.05f, 0.25f);
  CK(cudaDeviceSynchronize());

  {  // decode bit-exactness through the fragment mapping: expert 0's gate and down
    int bad = 0;
    for (int which = 0; which < 2; ++which) {
      const int Kin = which ? I : H, N = which ? H : I;
      const uint32_t* m = tw + (which ? 2 * mw : 0);
      uint16_t* out;
      CK(cudaMalloc(&out, (size_t)Kin * N * 2));
      const int warps = Kin / 16 * (N / 16);
      k_deq_tr<K, CB><<<(warps + 7) / 8, 256>>>(m, Kin, N, out);
      CK(cudaDeviceSynchronize());
      auto got = d2h<uint16_t>(out, (size_t)Kin * N);
      auto hw = d2h<uint32_t>(m, mw);
      const int KT = Kin / 16;
      for (int nt = 0; nt < N / 16; ++nt)
        for (int kt = 0; kt < KT; ++kt)
          for (int p = 0; p < 256; ++p) {
            const int e = exl3_perm(p);
            bad += got[(size_t)(kt * 16 + e / 16) * N + nt * 16 + e % 16] != exl3_dq_ref<K, CB>(hw.data() + ((size_t)nt * KT + kt) * 8 * K, p);
          }
      CK(cudaFree(out));
    }
    result("CHECK\t%s\tdecode bit-exact vs EXL3's dq<%d,%d> + tensor_core_perm (expert 0 gate + down)\t%s\t%d mismatches", name, K, CB,
           bad ? "FAIL" : "ok", bad);
  }

  for (int M : rows_list) {
    const int U = M == 1 ? 8 : 28;
    for (int S : M == 1 ? std::vector<int>{2, 3, 4} : std::vector<int>{1}) {
    const int NS = std::min(NSmax, std::max(16, (E + U - 1) / U));
    std::vector<int> hex;
    for (int s = 0; s < NS; ++s) { auto u = union_of_step(s, U, E); hex.insert(hex.end(), u.begin(), u.end()); }
    CK(cudaMemcpy(ex, hex.data(), hex.size() * 4, cudaMemcpyHostToDevice));
    auto args = [&](int s, bool up) {
      TArgs a;
      a.w = tw; a.we = we; a.su = su; a.sv = sv; a.ex = ex + (size_t)s * U; a.U = U; a.M = M; a.S = S; a.x = x;
      a.xh = up ? xh : xhd; a.yp = yp; a.y = y; a.wr = wr;
      return a;
    };
    for (int P : {8, 16}) {
      auto pre = [&](int s, cudaStream_t st) { k_tr_pre<<<192, 256, 0, st>>>(args(s, true)); };
      auto up = [&](int s, cudaStream_t st) {
        if (P == 8) launch_tr_up<K, CB, 8>(args(s, true), st); else launch_tr_up<K, CB, 16>(args(s, true), st);
      };
      auto mid = [&](int s, cudaStream_t st) { k_tr_mid<<<(U * M * 4 + 7) / 8, 256, 0, st>>>(args(s, false)); };
      auto down = [&](int s, cudaStream_t st) {
        if (P == 8) launch_tr_down<K, CB, 8>(args(s, false), st); else launch_tr_down<K, CB, 16>(args(s, false), st);
      };
      pre(0, 0); up(0, 0); mid(0, 0); down(0, 0);
      CK(cudaDeviceSynchronize());
      if (P == 8) {
        auto hx = d2h<__nv_bfloat16>(x, (size_t)M * H);
        auto hxh = d2h<uint16_t>(xh, (size_t)U * 2 * M * H);
        auto hyp = d2h<float>(yp, (size_t)S * U * M * NCH);
        auto hxhd = d2h<uint16_t>(xhd, (size_t)U * M * I);
        auto hy = d2h<float>(y, (size_t)M * H);
        auto hwr = d2h<float>(wr, U * M);
        size_t pre_bad = 0, mid_bad = 0;
        int mid_maxulp = 0;
        ErrStat eu, ed;
        std::vector<double> yd((size_t)M * H, 0.0), ydm((size_t)M * H, 0.0);
        for (int ei = 0; ei < U; ++ei) {
          const int e = hex[ei];
          auto hsu = d2h<uint16_t>(su + (size_t)e * kSuE, kSuE);
          auto hsv = d2h<uint16_t>(sv + (size_t)e * kSvE, kSvE);
          auto hw = d2h<uint32_t>(tw + (size_t)e * we, we);
          // pre: bit-exact
          for (int m = 0; m < 2; ++m)
            for (int r = 0; r < M; ++r)
              for (int blk = 0; blk < H / 128; ++blk) {
                float v[128];
                for (int j = 0; j < 128; ++j)
                  v[j] = __bfloat162float(hx[(size_t)r * H + blk * 128 + j]) * h_bits_to_f(hsu[m * H + blk * 128 + j]);
                had128_ref(v);
                for (int j = 0; j < 128; ++j)
                  pre_bad += f_to_h_bits(v[j] * kHadScale) != hxh[(((size_t)ei * 2 + m) * M + r) * H + blk * 128 + j];
              }
          // up from the GPU's xh
          std::vector<float> wg, wu, wd;
          tr_matrix_ref<K, CB>(hw.data(), H, I, wg);
          tr_matrix_ref<K, CB>(hw.data() + mw, H, I, wu);
          tr_matrix_ref<K, CB>(hw.data() + 2 * mw, I, H, wd);
          std::vector<double> ref((size_t)M * NCH), mag((size_t)M * NCH);
#pragma omp parallel for schedule(static)
          for (int ch = 0; ch < NCH; ++ch) {
            const int m = ch / I, n = ch % I;
            const std::vector<float>& W = m ? wu : wg;
            for (int r = 0; r < M; ++r) {
              double acc = 0, am = 0;
              for (int k = 0; k < H; ++k) {
                double tt = (double)h_bits_to_f(hxh[(((size_t)ei * 2 + m) * M + r) * H + k]) * W[(size_t)k * I + n];
                acc += tt; am += fabs(tt);
              }
              ref[(size_t)r * NCH + ch] = acc; mag[(size_t)r * NCH + ch] = am;
            }
          }
          for (int r = 0; r < M; ++r)
            for (int ch = 0; ch < NCH; ++ch) {
              double gsum = 0;
              for (int s = 0; s < S; ++s) gsum += hyp[(((size_t)s * U + ei) * M + r) * NCH + ch];
              eu.add(gsum, ref[(size_t)r * NCH + ch], mag[(size_t)r * NCH + ch], kTolSum);
            }
          // mid from the GPU's yp, in the kernel's f32 order (expf may differ by an ulp: counted, not failed)
          for (int r = 0; r < M; ++r)
            for (int blk = 0; blk < I / 128; ++blk) {
              float gt[128], upv[128], h[128];
              for (int j = 0; j < 128; ++j) {
                gt[j] = upv[j] = 0.f;
                for (int s = 0; s < S; ++s) {
                  gt[j] += hyp[(((size_t)s * U + ei) * M + r) * NCH + blk * 128 + j];
                  upv[j] += hyp[(((size_t)s * U + ei) * M + r) * NCH + I + blk * 128 + j];
                }
              }
              had128_ref(gt);
              had128_ref(upv);
              for (int j = 0; j < 128; ++j) {
                const float gg = gt[j] * kHadScale * h_bits_to_f(hsv[blk * 128 + j]), uu = upv[j] * kHadScale * h_bits_to_f(hsv[I + blk * 128 + j]);
                h[j] = bf16_bits_to_f(f_to_bf16_bits(gg / (1.f + expf(-gg)) * uu)) * h_bits_to_f(hsu[2 * H + blk * 128 + j]);
              }
              had128_ref(h);
              for (int j = 0; j < 128; ++j) {
                const uint16_t want = f_to_h_bits(h[j] * kHadScale), gotb = hxhd[((size_t)ei * M + r) * I + blk * 128 + j];
                if (want != gotb) { ++mid_bad; mid_maxulp = std::max(mid_maxulp, abs((int)(int16_t)want - (int)(int16_t)gotb)); }
              }
            }
          // down from the GPU's xhd: y'[n] = sum_k xhd[k] Wd[k][n], then had per 128 block, sv, routing weight
#pragma omp parallel for schedule(static)
          for (int b = 0; b < H / 128; ++b)
            for (int r = 0; r < M; ++r) {
              double yb[128], mb[128];
              for (int j = 0; j < 128; ++j) {
                double acc = 0, am = 0;
                for (int k = 0; k < I; ++k) {
                  double tt = (double)h_bits_to_f(hxhd[((size_t)ei * M + r) * I + k]) * wd[(size_t)k * H + b * 128 + j];
                  acc += tt; am += fabs(tt);
                }
                yb[j] = acc; mb[j] = am;
              }
              double msum = 0;
              for (int j = 0; j < 128; ++j) msum += mb[j];
              for (int hh = 1; hh < 128; hh <<= 1)
                for (int j = 0; j < 128; ++j)
                  if (!(j & hh)) { double a0 = yb[j], b0 = yb[j + hh]; yb[j] = a0 + b0; yb[j + hh] = a0 - b0; }
              for (int j = 0; j < 128; ++j) {
                const double kk = (double)kHadScale * h_bits_to_f(hsv[2 * I + b * 128 + j]) * hwr[ei * M + r];
                yd[(size_t)r * H + b * 128 + j] += yb[j] * kk;
                ydm[(size_t)r * H + b * 128 + j] += msum * fabs(kk);  // the had mixes the block: its whole magnitude
              }
            }
        }
        for (size_t i = 0; i < yd.size(); ++i) ed.add(hy[i], yd[i], ydm[i], kTolSum);
        result("CHECK\t%s\tr%d input transform had(x*su)/sqrt128 -> f16 vs CPU same order\t%s\t%zu of %zu differ", name, M,
               pre_bad ? "FAIL" : "ok", pre_bad, (size_t)U * 2 * M * H);
        result("CHECK\t%s\tr%d up (gate+up, %d experts) vs CPU double from the GPU's xh\t%s\tmax |err|/sum|terms| %.2e (abs %.2e), %zu outputs", name,
               M, U, eu.bad ? "FAIL" : "ok", eu.worst, eu.worst_abs, eu.n);
        result("CHECK\t%s\tr%d mid (had, sv, SwiGLU, bf16, su, had -> f16) vs CPU f32 same order\t%s\t%zu of %zu differ, max %d f16 ulp", name, M,
               mid_maxulp <= 1 ? "ok" : "FAIL", mid_bad, (size_t)U * M * I, mid_maxulp);
        result("CHECK\t%s\tr%d down (%d experts, had + sv + weighted sum) vs CPU double from the GPU's xh\t%s\tmax |err|/sum|terms| %.2e (abs %.2e), %zu outputs",
               name, M, U, ed.bad ? "FAIL" : "ok", ed.worst, ed.worst_abs, ed.n);
      }
      const double bu = (double)U * bgu, bd = (double)U * bdn;
      const double tp = time_graph(NS, pre), tu = time_graph(NS, up), tm = time_graph(NS, mid), td = time_graph(NS, down);
      const double tt = time_graph(NS, [&](int s, cudaStream_t st) { pre(s, st); up(s, st); mid(s, st); down(s, st); });
      auto line = [&](const char* part, double bytes, double us) {
        result("MOE\t%s\tr%d\tU%d\tP%d\tS%d\t%s\t%.0f\t%.2f\t%.1f\t%.1f\t%.1f", name, M, U, P, std::string(part) == "up" ? S : 1, part, bytes, us,
               bytes / us / 1e3, 100 * bytes / us / 1e3 / 235.0, 100 * bytes / us / 1e3 / g_bw_window);
      };
      line("pre", 0, tp);
      line("up", bu, tu);
      line("mid", 0, tm);
      line("down", bd, td);
      line("step", bu + bd, tt);
    }
    }
  }
  CK(cudaFree(tw)); CK(cudaFree(su)); CK(cudaFree(sv)); CK(cudaFree(xh)); CK(cudaFree(xhd)); CK(cudaFree(yp)); CK(cudaFree(y));
  CK(cudaFree(wr)); CK(cudaFree(x)); CK(cudaFree(ex));
}

// ---------------------------------------------------------------------------------------------------------
static void bench_dense(const std::vector<int>& rows_list) {
  struct Shape { const char* name; int N, K; };
  const Shape shapes[2] = {{"q_b", 16384 / 4, 2048}, {"o_proj", 6144, 16384 / 4}};
  for (const Shape& sh : shapes)
    for (int fp8 = 0; fp8 < 2; ++fp8) {
      const int N = sh.N, K = sh.K;
      const size_t wb = (size_t)N * K * (fp8 ? 1 : 2), sb = fp8 ? (size_t)(N / 128) * (K / 128) * 4 : 0, per = wb + sb;
      const int L = (int)ceil(kPoolBytes / per);
      uint8_t* w;
      float *sc = nullptr, *y;
      __nv_bfloat16* x;
      CK(cudaMalloc(&w, per * L));
      if (fp8) { CK(cudaMalloc(&sc, sb * L)); fill_f32(sc, sb / 4 * L, 31, 0.5f, 2.f); }
      CK(cudaMalloc(&x, (size_t)8 * K * 2)); CK(cudaMalloc(&y, (size_t)8 * N * 4));
      fill(w, per * L, 32 + fp8, fp8 ? 4 : 3);
      fill(x, (size_t)8 * K * 2, 33, 5);
      CK(cudaDeviceSynchronize());
      const char* fname = fp8 ? "fp8" : "bf16";
      for (int M : rows_list)
        for (int P : {4, 8}) {
          auto run = [&](int s, cudaStream_t st) {
            const int l = s % L;
            if (fp8) {
              if (P == 4) k_dense_fp8<4><<<N / 32, 128, 0, st>>>(w + per * l, sc + sb / 4 * l, x, M, N, K, y);
              else k_dense_fp8<8><<<N / 32, 128, 0, st>>>(w + per * l, sc + sb / 4 * l, x, M, N, K, y);
            } else {
              if (P == 4) k_dense_bf16<4><<<N / 32, 128, 0, st>>>((const __nv_bfloat16*)(w + per * l), x, M, N, K, y);
              else k_dense_bf16<8><<<N / 32, 128, 0, st>>>((const __nv_bfloat16*)(w + per * l), x, M, N, K, y);
            }
          };
          run(0, 0);
          CK(cudaDeviceSynchronize());
          if (P == 4) {
            auto hw = d2h<uint8_t>(w, wb);
            auto hy = d2h<float>(y, (size_t)M * N);
            auto hx = d2h<__nv_bfloat16>(x, (size_t)M * K);
            std::vector<float> hs;
            if (fp8) hs = d2h<float>(sc, sb / 4);
            ErrStat es;
            std::vector<double> ref((size_t)M * N), mag((size_t)M * N);
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
              for (int r = 0; r < M; ++r) {
                double acc = 0, am = 0;
                for (int k = 0; k < K; ++k) {
                  double wv;
                  if (fp8) {
                    const int J = K / 128, j = k / 128, q = (k % 128) / 64, t = (k % 64) / 16, i = k % 16, lane = 4 * (n % 8) + t;
                    wv = e4m3_to_f(hw[(((size_t)(n / 8) * J + j) * 32 + lane) * 32 + 16 * q + i]) * (double)hs[(size_t)(n / 128) * J + j];
                  } else wv = bf16_bits_to_f(((const uint16_t*)hw.data())[(size_t)n * K + k]);
                  double tt = (double)__bfloat162float(hx[(size_t)r * K + k]) * wv;
                  acc += tt; am += fabs(tt);
                }
                ref[(size_t)r * N + n] = acc; mag[(size_t)r * N + n] = am;
              }
            for (size_t i = 0; i < ref.size(); ++i) es.add(hy[i], ref[i], mag[i], kTolSum);
            result("CHECK\t%s\t%s r%d [%dx%d] vs CPU double\t%s\tmax |err|/sum|terms| %.2e (abs %.2e)", fname, sh.name, M, N, K,
                   es.bad ? "FAIL" : "ok", es.worst, es.worst_abs);
          }
          const double us = time_graph(std::max(L, 16), run);
          result("DENSE\t%s\t%s\tr%d\tP%d\t%.0f\t%.2f\t%.1f\t%.1f\t%.1f", fname, sh.name, M, P, (double)per, us, per / us / 1e3,
                 100 * per / us / 1e3 / 235.0, 100 * per / us / 1e3 / g_bw_window);
        }
      CK(cudaFree(w)); if (sc) CK(cudaFree(sc)); CK(cudaFree(x)); CK(cudaFree(y));
    }
}

static double bench_bw() {
  const size_t bytes = (size_t)2 << 30;
  int4* p;
  int* o;
  CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&o, 4));
  CK(cudaMemset(p, 1, bytes));
  double best = 0;
  for (int bps : {2, 4, 8})
    for (int thr : {256, 512}) {
      const double us = time_graph(4, [&](int, cudaStream_t st) { k_read<<<48 * bps, thr, 0, st>>>(p, bytes / 16, o); }, 5);
      const double gbs = bytes / us / 1e3;
      result("BW\tread 2 GiB\tgrid %d x %d\t%.1f GB/s", 48 * bps, thr, gbs);
      best = std::max(best, gbs);
    }
  CK(cudaFree(p)); CK(cudaFree(o));
  return best;
}

int main(int argc, char** argv) {
  std::string mode, what, out;
  std::vector<int> rows = {1, 6};
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--cpu-check") mode = "cpu";
    else if (a == "--bench" && i + 1 < argc) { mode = "bench"; what = argv[++i]; }
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--rows" && i + 1 < argc) {
      rows.clear();
      for (char* t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) rows.push_back(atoi(t));
    } else { fprintf(stderr, "usage: gemv --cpu-check | --bench <q4|s3|tk3|tk4|tk3m|dense|bw> --out file [--rows 1,6]\n"); return 2; }
  }
  if (mode == "cpu") return cpu_check();
  if (mode != "bench") { fprintf(stderr, "nothing to do\n"); return 2; }
  if (!out.empty()) g_out = fopen(out.c_str(), "a");
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("%s, %d SMs, L2 %d MB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20);
  g_bw_window = bench_bw();
  result("BWBEST\t%s\t%.1f", what.c_str(), g_bw_window);
  if (what == "q4") bench_moe_q(Q4, "q4", rows);
  else if (what == "s3") bench_moe_q(S3, "s3", rows);
  else if (what == "tk3") bench_moe_tr<3, 0>("tk3", rows);
  else if (what == "tk4") bench_moe_tr<4, 0>("tk4", rows);
  else if (what == "tk3m") bench_moe_tr<3, 2>("tk3m", rows);
  else if (what == "dense") bench_dense(rows);
  else if (what != "bw") { fprintf(stderr, "unknown bench %s\n", what.c_str()); return 2; }
  result("DONE\t%s", what.c_str());
  if (g_out) fclose(g_out);
  return 0;
}
