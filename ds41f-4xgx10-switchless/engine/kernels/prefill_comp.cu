// The compressor for prefill chunks (M5): the projections as one tensor-core GEMM (gemm_bf16_rows_p on
// wkv / wgate as they are) and the entries a chunk completes, 16 per CTA, with the index key's wk
// projection on tensor cores (attn's emit runs one CTA per row and a warp per output, re-reading wk's
// 128 KB for every entry). Pooling, both RMSNorms, RoPE and the FP4 formats are attn::compressor_emit's
// op for op, so the KV entries are bit-identical; wk's sums run in another order before their bf16
// rounding, which can move an index-key element by one FP4 step in rare entries.
#include "attn/attn_common.cuh"
#include "common/check.h"
#include "kernels/dsk.h"
#include "kernels/pa_common.cuh"
#include "kernels/prefill_attn.h"

namespace dsk {
namespace {

using attn::kHead;
using attn::kIdxDim;

constexpr int kEE = 16;                  // entries per CTA (an m16 tile of wk's MMA)
constexpr int kEThreads = 128;
constexpr int kLatStride = kHead + 32;   // latent row in shared memory (1088 B: lds.128 conflict-free)

struct ESmem {
  uint16_t lat[kEE][kLatStride];   // the normalised latents (bf16 values)
  float ikv[kEE][kIdxDim + 4];     // wk's outputs, bf16-rounded
  float red[kEE][4];
  int list[32];                    // the rows completing an entry
  int nlist;
};

// CTA b scans rows [16 ratio b, 16 ratio (b + 1)) for the rows that complete a group ((pos + 1) % ratio
// == 0) and emits their entries 16 at a time. Row r's group starts at row r - 1 (ratio 2), or in the
// carry for r = 0, as attn::compressor_emit.
__global__ void __launch_bounds__(kEThreads) emit_p_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                                                           const int* __restrict__ pos, int rows,
                                                           attn::CompressorWeights w, attn::CompressorCarry carry,
                                                           const float2* __restrict__ table, attn::Compressed ckv,
                                                           attn::IndexKeys ik) {
  __shared__ __align__(16) ESmem sm;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, ratio = w.ratio;
  const int rb = blockIdx.x * kEE * ratio, re = min(rows, rb + kEE * ratio);
  if (warp == 0) {
    const int r = rb + lane;
    const bool c = r < re && (pos[r] + 1) % ratio == 0;
    const unsigned b = __ballot_sync(0xffffffffu, c);
    if (c) sm.list[__popc(b & ((1u << lane) - 1))] = r;
    if (lane == 0) sm.nlist = __popc(b);
  }
  __syncthreads();
  const int nlist = sm.nlist;
  for (int e0 = 0; e0 < nlist; e0 += kEE) {
    const int ne = min(kEE, nlist - e0);
    // pool (ratio 2: per-channel softmax over the group's two positions), bf16, RMSNorm(512); thread t
    // holds channels t + 128 i
    float v[kEE][4];
#pragma unroll
    for (int e = 0; e < kEE; ++e) {
      if (e >= ne) break;
      const int r = sm.list[e0 + e], p = pos[r];
      float ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int c = tid + 128 * i;
        float x;
        if (ratio == 1) {
          x = kv[(size_t)r * kHead + c];
        } else {
          const float* kv0 = r > 0 ? kv + (size_t)(r - 1) * kHead : carry.kv + (size_t)((p - 1) & 15) * kHead;
          const float* s0p = r > 0 ? score + (size_t)(r - 1) * kHead : carry.score + (size_t)((p - 1) & 15) * kHead;
          const float a0 = kv0[c], s0 = s0p[c], a1 = kv[(size_t)r * kHead + c], s1 = score[(size_t)r * kHead + c];
          const float m = fmaxf(s0, s1);
          const float x0 = expf(__fsub_rn(s0, m)), x1 = expf(__fsub_rn(s1, m));
          const float sum = __fadd_rn(x0, x1);
          x = __fadd_rn(__fmul_rn(a0, __fdiv_rn(x0, sum)), __fmul_rn(a1, __fdiv_rn(x1, sum)));
        }
        v[e][i] = attn::bfround(x);
        ss = __fadd_rn(ss, __fmul_rn(v[e][i], v[e][i]));
      }
      ss = attn::warp_sum(ss);
      if (lane == 0) sm.red[e][warp] = ss;
    }
    __syncthreads();
#pragma unroll
    for (int e = 0; e < kEE; ++e) {
      if (e >= ne) break;
      const float tot = __fadd_rn(__fadd_rn(__fadd_rn(sm.red[e][0], sm.red[e][1]), sm.red[e][2]), sm.red[e][3]);
      const float rs = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / kHead), attn::kNormEps));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int c = tid + 128 * i;
        sm.lat[e][c] = attn::f2bf(__fmul_rn(attn::bf2f(w.norm[c]), __fmul_rn(v[e][i], rs)));
      }
    }
    __syncthreads();
    // wk: [16 entries][512] x [128][512]^T on tensor cores, warp w -> outputs 32w..32w+31. Lane t holds
    // k = 8t..8t+7 of every 32 for A and B alike (one 16-byte load each), split over two MMAs.
    {
      float d[4][4] = {};
#pragma unroll 4
      for (int kc = 0; kc < kHead / 32; ++kc) {
        const uint4 xa = *reinterpret_cast<const uint4*>(&sm.lat[g][32 * kc + 8 * t]);
        const uint4 xb = *reinterpret_cast<const uint4*>(&sm.lat[g + 8][32 * kc + 8 * t]);
        const uint32_t a0[4] = {xa.x, xb.x, xa.y, xb.y}, a1[4] = {xa.z, xb.z, xa.w, xb.w};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const uint4 b =
              __ldg(reinterpret_cast<const uint4*>(w.wk + (size_t)(32 * warp + 8 * nt + g) * kHead + 32 * kc + 8 * t));
          pa::mma_bf16(d[nt], a0, b.x, b.y);
          pa::mma_bf16(d[nt], a1, b.z, b.w);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int n = 32 * warp + 8 * nt + 2 * t;
        sm.ikv[g][n] = attn::bfround(d[nt][0]);
        sm.ikv[g][n + 1] = attn::bfround(d[nt][1]);
        sm.ikv[g + 8][n] = attn::bfround(d[nt][2]);
        sm.ikv[g + 8][n + 1] = attn::bfround(d[nt][3]);
      }
    }
    __syncthreads();
    // index key: k_norm, RoPE on dims 64..127 at j * ratio (pairs on neighbouring lanes), fp4 + ue8m0 per 32
    float kx[kEE];
#pragma unroll
    for (int e = 0; e < kEE; ++e) {
      if (e >= ne) break;
      kx[e] = sm.ikv[e][tid];
      const float s2 = attn::warp_sum(__fmul_rn(kx[e], kx[e]));
      if (lane == 0) sm.red[e][warp] = s2;
    }
    __syncthreads();
#pragma unroll
    for (int e = 0; e < kEE; ++e) {
      if (e >= ne) break;
      const int r = sm.list[e0 + e], j = pos[r] / ratio, rpos = j * ratio;
      const float tot = __fadd_rn(__fadd_rn(__fadd_rn(sm.red[e][0], sm.red[e][1]), sm.red[e][2]), sm.red[e][3]);
      const float krs = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / kIdxDim), attn::kNormEps));
      float x = attn::bfround(__fmul_rn(attn::bf2f(w.knorm[tid]), __fmul_rn(kx[e], krs)));
      if (warp >= 2) {  // dims 64 + 2i, 65 + 2i on lanes 2i, 2i + 1
        const float o = __shfl_xor_sync(0xffffffffu, x, 1);
        const float2 cs = table[(size_t)rpos * 32 + ((tid - 64) >> 1)];
        float x0 = (lane & 1) ? o : x, x1 = (lane & 1) ? x : o;
        attn::rope_pair(x0, x1, cs.x, cs.y, false);
        x = attn::bfround((lane & 1) ? x1 : x0);
      }
      const float s = attn::fp4_e8m0_scale(attn::warp_max(fabsf(x)));
      const uint32_t code = attn::fp4_code(x, s);
      const uint32_t hi = __shfl_down_sync(0xffffffffu, code, 1);
      if ((lane & 1) == 0) ik.k[(size_t)j * (kIdxDim / 2) + attn::fp4x128_byte(tid / 2)] = (uint8_t)(code | (hi << 4));
      if (lane == 0) ik.sc[(size_t)j * 4 + warp] = attn::f_to_ue8m0(s);
    }
    // KV entry: RoPE on the latent's last 64 dims at j * ratio, fp4 with an e4m3 scale per 16; thread t
    // owns channels 4t..4t+3
#pragma unroll 1
    for (int e = 0; e < ne; ++e) {
      const int r = sm.list[e0 + e], j = pos[r] / ratio, rpos = j * ratio;
      const uint2 lv = *reinterpret_cast<const uint2*>(&sm.lat[e][4 * tid]);
      float ev[4] = {__uint_as_float(lv.x << 16), __uint_as_float(lv.x & 0xffff0000u), __uint_as_float(lv.y << 16),
                     __uint_as_float(lv.y & 0xffff0000u)};
      if (tid >= 112) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 cs = table[(size_t)rpos * 32 + (4 * tid + 2 * h - 448) / 2];
          attn::rope_pair(ev[2 * h], ev[2 * h + 1], cs.x, cs.y, false);
          ev[2 * h] = attn::bfround(ev[2 * h]);
          ev[2 * h + 1] = attn::bfround(ev[2 * h + 1]);
        }
      }
      float am = fmaxf(fmaxf(fabsf(ev[0]), fabsf(ev[1])), fmaxf(fabsf(ev[2]), fabsf(ev[3])));
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
      const float s = attn::fp4_e4m3_scale(am);
      const uint32_t c0 = attn::fp4_code(ev[0], s), c1 = attn::fp4_code(ev[1], s), c2 = attn::fp4_code(ev[2], s),
                     c3 = attn::fp4_code(ev[3], s);
      *reinterpret_cast<uint16_t*>(ckv.kv + (size_t)j * (kHead / 2) + 2 * tid) =
          (uint16_t)((c0 | (c1 << 4)) | ((c2 | (c3 << 4)) << 8));
      if ((tid & 3) == 0) ckv.sc[(size_t)j * 32 + tid / 4] = attn::f_to_e4m3(s);
    }
    __syncthreads();
  }
}

// keep the last (up to) 16 rows' ratio-2 projections, by position (attn's carry)
__global__ void carry_p_kernel(const float* kv, const float* score, const int* pos, int rows,
                               attn::CompressorCarry carry) {
  const int r = rows - 1 - blockIdx.x;
  if (r < 0) return;
  const int slot = pos[r] & 15;
  for (int c = threadIdx.x; c < kHead; c += blockDim.x) {
    carry.kv[slot * kHead + c] = kv[(size_t)r * kHead + c];
    carry.score[slot * kHead + c] = score[(size_t)r * kHead + c];
  }
}

}  // namespace

void compressor_project_p(const uint16_t* x, const attn::CompressorWeights& w, float* kv, float* score, int rows,
                          cudaStream_t s) {
  gemm_bf16_rows_p(reinterpret_cast<const __nv_bfloat16*>(x), rows, attn::kDim, w.wkv, w.wgate, kHead,
                   w.ratio > 1 ? 2 * kHead : kHead, kv, score, w.ratio == 1, s);
}

void compressor_emit_p(const float* kv, const float* score, const int* pos, int rows, const attn::CompressorWeights& w,
                       attn::CompressorCarry carry, const float2* table, attn::Compressed ckv, attn::IndexKeys ik,
                       cudaStream_t s) {
  if (rows < 1 || (w.ratio != 1 && w.ratio != 2)) die("compressor_emit_p: rows %d ratio %d", rows, w.ratio);
  const int per = kEE * w.ratio;
  emit_p_kernel<<<(rows + per - 1) / per, kEThreads, 0, s>>>(kv, score, pos, rows, w, carry, table, ckv, ik);
  CUDA_CHECK(cudaGetLastError());
  if (w.ratio > 1) {
    carry_p_kernel<<<rows < 16 ? rows : 16, 128, 0, s>>>(kv, score, pos, rows, carry);
    CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace dsk
