// Device side of the compressor's emit (compressor.cu), shared with the fused decode prep (prep.cu).
#pragma once
#include "attn/attn.h"
#include "attn/attn_common.cuh"

namespace attn {

__device__ __forceinline__ float block_sum128(float v, float* red) {
  v = warp_sum(v);
  int w = threadIdx.x / 32;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[w] = v;
  __syncthreads();
  return __fadd_rn(__fadd_rn(__fadd_rn(red[0], red[1]), red[2]), red[3]);
}

// Row r's entry, by a 128-thread CTA; rows that do not complete a group return at once. carry, ckv and ik are
// row r's session's (attn_common.cuh carry_at etc.); prev: row r - 1 is of the same session (ratio 2: the
// group's first position is that row's projection, else the carry's).
__device__ __forceinline__ void emit_row(int r, bool prev, const float* kv, const float* score, const int* pos,
                                         const CompressorWeights& w, const CompressorCarry& carry,
                                         const float2* table, const Compressed& ckv, const IndexKeys& ik,
                                         uint16_t* dbg_latent, int dbg_base) {
  __shared__ float lat[kHead];
  __shared__ float ikv[kIdxDim];
  __shared__ float red[4];
  int t = threadIdx.x, ratio = w.ratio;
  int p = pos[r];
  if ((p + 1) % ratio != 0) return;
  int j = p / ratio;
  // pool the group (ratio 2: per-channel softmax over its two positions), round to bf16
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int c = t + 128 * i;
    if (ratio == 1) {
      v[i] = kv[(size_t)r * kHead + c];
    } else {
      const float* kv0 = prev ? kv + (size_t)(r - 1) * kHead : carry.kv + (size_t)((p - 1) & 15) * kHead;
      const float* s0p = prev ? score + (size_t)(r - 1) * kHead : carry.score + (size_t)((p - 1) & 15) * kHead;
      float a0 = kv0[c], s0 = s0p[c];
      float a1 = kv[(size_t)r * kHead + c], s1 = score[(size_t)r * kHead + c];
      float m = fmaxf(s0, s1);
      float e0 = expf(__fsub_rn(s0, m)), e1 = expf(__fsub_rn(s1, m));
      float sum = __fadd_rn(e0, e1);
      v[i] = __fadd_rn(__fmul_rn(a0, __fdiv_rn(e0, sum)), __fmul_rn(a1, __fdiv_rn(e1, sum)));
    }
    v[i] = bfround(v[i]);
  }
  // compressor RMSNorm(512)
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) ss = __fadd_rn(ss, __fmul_rn(v[i], v[i]));
  ss = block_sum128(ss, red);
  float rs = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.0f / kHead), kNormEps));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int c = t + 128 * i;
    lat[c] = bfround(__fmul_rn(bf2f(w.norm[c]), __fmul_rn(v[i], rs)));
    if (dbg_latent) dbg_latent[(size_t)(j - dbg_base) * kHead + c] = f2bf(lat[c]);
  }
  __syncthreads();
  // index key: bf16 wk(latent) -> k_norm -> RoPE at j*ratio -> fp4 with e8m0 per 32
  int lane = t & 31, wid = t / 32;
  float lf[kHead / 32];
#pragma unroll
  for (int i = 0; i < kHead / 32; ++i) lf[i] = lat[lane + 32 * i];
  for (int n0 = wid * 32; n0 < wid * 32 + 32; n0 += 4) {  // four rows at a time, so their loads overlap
    float wv[4][kHead / 32], acc[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int i = 0; i < kHead / 32; ++i) wv[k][i] = bf2f(w.wk[(size_t)(n0 + k) * kHead + lane + 32 * i]);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      acc[k] = 0.f;
#pragma unroll
      for (int i = 0; i < kHead / 32; ++i) acc[k] = __fmaf_rn(lf[i], wv[k][i], acc[k]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)  // warp_sum of each
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], o);
    if (lane == 0)
#pragma unroll
      for (int k = 0; k < 4; ++k) ikv[n0 + k] = bfround(acc[k]);
  }
  __syncthreads();
  float kval = ikv[t];
  float kss = block_sum128(__fmul_rn(kval, kval), red);
  float krs = rsqrtf(__fadd_rn(__fmul_rn(kss, 1.0f / kIdxDim), kNormEps));
  kval = bfround(__fmul_rn(bf2f(w.knorm[t]), __fmul_rn(kval, krs)));
  __syncthreads();
  ikv[t] = kval;
  __syncthreads();
  int rpos = j * ratio;
  if (t < 32) {
    float x0 = ikv[64 + 2 * t], x1 = ikv[65 + 2 * t];
    float2 cs = table[(size_t)rpos * 32 + t];
    rope_pair(x0, x1, cs.x, cs.y, false);
    ikv[64 + 2 * t] = bfround(x0);
    ikv[65 + 2 * t] = bfround(x1);
  }
  __syncthreads();
  {
    float x = ikv[t];
    float s = fp4_e8m0_scale(warp_max(fabsf(x)));
    uint32_t code = fp4_code(x, s);
    uint32_t hi = __shfl_down_sync(0xffffffffu, code, 1);
    if ((lane & 1) == 0) ik.k[(size_t)j * (kIdxDim / 2) + fp4x128_byte(t / 2)] = (uint8_t)(code | (hi << 4));
    if (lane == 0) ik.sc[(size_t)j * 4 + wid] = f_to_ue8m0(s);
  }
  // KV entry: RoPE on the latent's last 64 dims at j*ratio, then fp4 with an e4m3 scale per 16.
  // Thread t owns channels [4t, 4t+4): four threads per 16-block, threads 112..127 hold the RoPE dims.
  float e[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) e[i] = lat[4 * t + i];
  if (t >= 112) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int pi = (4 * t + 2 * h - 448) / 2;
      float2 cs = table[(size_t)rpos * 32 + pi];
      rope_pair(e[2 * h], e[2 * h + 1], cs.x, cs.y, false);
      e[2 * h] = bfround(e[2 * h]);
      e[2 * h + 1] = bfround(e[2 * h + 1]);
    }
  }
  float am = fmaxf(fmaxf(fabsf(e[0]), fabsf(e[1])), fmaxf(fabsf(e[2]), fabsf(e[3])));
  am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
  am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
  float s = fp4_e4m3_scale(am);
  uint32_t c0 = fp4_code(e[0], s), c1 = fp4_code(e[1], s), c2 = fp4_code(e[2], s), c3 = fp4_code(e[3], s);
  uint8_t* dst = ckv.kv + (size_t)j * (kHead / 2) + 2 * t;
  dst[0] = (uint8_t)(c0 | (c1 << 4));
  dst[1] = (uint8_t)(c2 | (c3 << 4));
  if ((t & 3) == 0) ckv.sc[(size_t)j * 32 + t / 4] = f_to_e4m3(s);
}

// ratio 2: keep row r's projections in the carry slot of its position (carry: row r's session's)
__device__ __forceinline__ void carry_row(int r, const float* kv, const float* score, const int* pos,
                                          const CompressorCarry& carry) {
  int slot = pos[r] & 15;
  for (int c = threadIdx.x; c < kHead; c += blockDim.x) {
    carry.kv[slot * kHead + c] = kv[(size_t)r * kHead + c];
    carry.score[slot * kHead + c] = score[(size_t)r * kHead + c];
  }
}

}  // namespace attn
