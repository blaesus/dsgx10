// The indexer for prefill chunks (M5): attn::index_score's dense scores for thousands of rows, and the
// top-k selection in L2-sized batches of rows.
//
// The O(n^2) part of prefill: every row scores every compressed position before it (32 heads x 128 dims of
// e2m1 with ue8m0 per 32, the mxf4nvf4 MMA). attn's kernel stages 8 rows of q and reloads the keys for every
// 8 rows; here a CTA keeps 256 positions of keys in registers as the MMA's A operand (16 positions per
// tile, two tiles per warp) and streams q through shared memory for all of its rows, so keys are read once
// per CTA and q once per position block. The epilogue is attn's in the same order (bf16 rounding of each
// head's dot product, relu, times the head weight, rounded, fp32 sum over heads in head order, then over
// the lane quad), with the relu-multiply-round done two heads at a time in bf16x2: keys are bit-identical.
#include <algorithm>

#include "attn/attn_common.cuh"
#include "common/check.h"
#include "kernels/pa_common.cuh"
#include "kernels/prefill_attn.h"

namespace dsk {
namespace {

using attn::kIdxHeads;

constexpr int kSThreads = 256, kSWarps = kSThreads / 32, kSTiles = 2;
constexpr int kSPos = kSWarps * kSTiles * 16;  // positions per CTA
constexpr int kSRows = 8;                      // rows of q per shared-memory stage
static_assert(kSPos % attn::kScoreAlign == 0 || attn::kScoreAlign % kSPos == 0, "keys_stride padding");

struct SArgs {
  const uint8_t* q;
  const uint32_t* qsc;
  const float* w;
  const int* pos;
  int rows, ratio, rows_per_cta;
  const uint8_t* k;
  const uint32_t* ksc;
  uint16_t* keys;
  int keys_stride;
  uint16_t* bkeys;
  int bkeys_stride;
};

__device__ __forceinline__ uint32_t key_from_bf16(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }
// (lo, hi) -> bf16x2, round to nearest even
__device__ __forceinline__ uint32_t bf16x2_rn(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// bf16(max(x, 0) * w) for two bf16 pairs (the product of two bf16 values is exact before the rounding)
__device__ __forceinline__ uint32_t relu_mul_bf16x2(uint32_t x, uint32_t w) {
  uint32_t r, z = 0;
  asm("{.reg .b32 t; max.bf16x2 t, %1, %3; mul.rn.bf16x2 %0, t, %2;}" : "=r"(r) : "r"(x), "r"(w), "r"(z));
  return r;
}

struct SSmem {
  uint8_t q[2][kSRows][kIdxHeads][64];
  uint32_t sc[2][kSRows][kIdxHeads];
  float w[2][kSRows][kIdxHeads];
  int len[2][kSRows];
};

template <bool POOL>
__global__ void __launch_bounds__(kSThreads) score_p_kernel(SArgs a) {
  __shared__ __align__(16) SSmem sm;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int n = (__ldg(a.pos + a.rows - 1) + 1) / a.ratio;  // the dense scan: the last row's length
  const int base = blockIdx.x * kSPos;
  const int r0 = blockIdx.y * a.rows_per_cta, r1 = min(a.rows, r0 + a.rows_per_cta);
  if (base >= n || r0 >= r1) return;

  auto stage = [&](int rc, int st) {  // rows rc.. of q, scales, weights and lengths into stage st
    const int nr = min(kSRows, r1 - rc);
    const uint8_t* qs = a.q + (size_t)rc * kIdxHeads * 64;
    for (int i = tid; i < nr * kIdxHeads * 4; i += kSThreads) pa::cp16(&sm.q[st][0][0][0] + i * 16, qs + i * 16, true);
    for (int i = tid; i < nr * kIdxHeads / 4; i += kSThreads) {
      pa::cp16(&sm.sc[st][0][0] + i * 4, a.qsc + (size_t)rc * kIdxHeads + i * 4, true);
      pa::cp16(&sm.w[st][0][0] + i * 4, a.w + (size_t)rc * kIdxHeads + i * 4, true);
    }
    if (tid < nr) sm.len[st][tid] = (__ldg(a.pos + rc + tid) + 1) / a.ratio;
  };
  stage(r0, 0);
  pa::cp_commit();

  // this warp's keys: tile j holds positions i0 + g (MMA rows g) and i0 + 8 + g (rows g + 8)
  uint4 k0[kSTiles], k1[kSTiles];
  uint32_t sa[kSTiles];
#pragma unroll
  for (int j = 0; j < kSTiles; ++j) {
    const int i0 = base + (warp * kSTiles + j) * 16, p0 = i0 + g, p1 = i0 + 8 + g;
    k0[j] = k1[j] = make_uint4(0, 0, 0, 0);
    uint32_t s0 = 0x7f7f7f7fu, s1 = 0x7f7f7f7fu;
    if (p0 < n) {
      k0[j] = __ldg(reinterpret_cast<const uint4*>(a.k + (size_t)p0 * 64) + t);
      s0 = __ldg(a.ksc + p0);
    }
    if (p1 < n) {
      k1[j] = __ldg(reinterpret_cast<const uint4*>(a.k + (size_t)p1 * 64) + t);
      s1 = __ldg(a.ksc + p1);
    }
    sa[j] = (t & 1) ? s1 : s0;  // a lane supplies the scales of MMA row g + 8(t&1)
  }
  // output words: lane L < 16 writes positions 2m, 2m+1 (m = L & 7) of tile L >> 3, gathered from the
  // lanes holding them (t = 0: positions g, t = 1: positions 8 + g)
  const int m = lane & 7, src_lo = m < 4 ? 8 * m : 8 * (m - 4) + 1, src_hi = src_lo + 4;

  for (int rc = r0, it = 0; rc < r1; rc += kSRows, ++it) {
    const int st = it & 1, nr = min(kSRows, r1 - rc);
    pa::cp_wait_all();
    __syncthreads();
    if (rc + kSRows < r1) stage(rc + kSRows, st ^ 1);
    pa::cp_commit();
    for (int rr = 0; rr < nr; ++rr) {
      const int len = sm.len[st][rr];
      uint32_t kv[kSTiles], bk0[kSTiles], bk1[kSTiles];
      if (base < len) {  // CTA-uniform: all tiles computed, positions past len masked
        float d[kSTiles][4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int h = nt * 8 + g;
          const uint4 qv = *reinterpret_cast<const uint4*>(&sm.q[st][rr][h][16 * t]);
          const uint32_t qs = sm.sc[st][rr][h];
#pragma unroll
          for (int j = 0; j < kSTiles; ++j) {
            d[j][nt][0] = d[j][nt][1] = d[j][nt][2] = d[j][nt][3] = 0.f;
            pa::mma_fp4(d[j][nt], k0[j].x, k1[j].x, k0[j].y, k1[j].y, qv.x, qv.y, sa[j] & 0xffff, qs & 0xffff);
            pa::mma_fp4(d[j][nt], k0[j].z, k1[j].z, k0[j].w, k1[j].w, qv.z, qv.w, sa[j] >> 16, qs >> 16);
          }
        }
        uint32_t wpk[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float2 ww = *reinterpret_cast<const float2*>(&sm.w[st][rr][nt * 8 + 2 * t]);
          wpk[nt] = (__float_as_uint(ww.y) & 0xffff0000u) | (__float_as_uint(ww.x) >> 16);
        }
#pragma unroll
        for (int j = 0; j < kSTiles; ++j) {
          const int i0 = base + (warp * kSTiles + j) * 16;
          float acc0 = 0.f, acc1 = 0.f;
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const uint32_t u01 = relu_mul_bf16x2(bf16x2_rn(d[j][nt][0], d[j][nt][1]), wpk[nt]);
            const uint32_t u23 = relu_mul_bf16x2(bf16x2_rn(d[j][nt][2], d[j][nt][3]), wpk[nt]);
            acc0 = __fadd_rn(acc0, __uint_as_float(u01 << 16));
            acc0 = __fadd_rn(acc0, __uint_as_float(u01 & 0xffff0000u));
            acc1 = __fadd_rn(acc1, __uint_as_float(u23 << 16));
            acc1 = __fadd_rn(acc1, __uint_as_float(u23 & 0xffff0000u));
          }
          acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 1));
          acc0 = __fadd_rn(acc0, __shfl_xor_sync(0xffffffffu, acc0, 2));
          acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 1));
          acc1 = __fadd_rn(acc1, __shfl_xor_sync(0xffffffffu, acc1, 2));
          const uint32_t key0 = i0 + g < len ? key_from_bf16(attn::f2bf(acc0)) : 0u;
          const uint32_t key1 = i0 + 8 + g < len ? key_from_bf16(attn::f2bf(acc1)) : 0u;
          kv[j] = t == 0 ? key0 : key1;
          bk0[j] = key0;
          bk1[j] = key1;
        }
      } else {
#pragma unroll
        for (int j = 0; j < kSTiles; ++j) kv[j] = bk0[j] = bk1[j] = 0;
      }
      const int r = rc + rr;
      uint32_t wd = 0;
#pragma unroll
      for (int j = 0; j < kSTiles; ++j) {
        const uint32_t lo = __shfl_sync(0xffffffffu, kv[j], src_lo), hi = __shfl_sync(0xffffffffu, kv[j], src_hi);
        if ((lane >> 3) == j) wd = lo | hi << 16;
      }
      if (lane < 8 * kSTiles)
        reinterpret_cast<uint32_t*>(a.keys + (size_t)r * a.keys_stride + base + warp * kSTiles * 16)[lane] = wd;
      if constexpr (POOL) {  // block maxima over g: blocks i0/8 and i0/8 + 1 of each tile
        const int pin = len > 0 ? (len - 1) >> 3 : -1;
#pragma unroll
        for (int j = 0; j < kSTiles; ++j) {
#pragma unroll
          for (int o = 4; o < 32; o <<= 1) {
            bk0[j] = max(bk0[j], __shfl_xor_sync(0xffffffffu, bk0[j], o));
            bk1[j] = max(bk1[j], __shfl_xor_sync(0xffffffffu, bk1[j], o));
          }
          if (lane == 0) {
            const int b = (base + (warp * kSTiles + j) * 16) >> 3;
            uint16_t* brow = a.bkeys + (size_t)r * a.bkeys_stride;
            brow[b] = b == pin ? 0xFFFF : (uint16_t)bk0[j];
            brow[b + 1] = b + 1 == pin ? 0xFFFF : (uint16_t)bk1[j];
          }
        }
      }
    }
  }
}

}  // namespace

void index_score_p(const attn::IndexQ& q, const int* pos, int rows, int ratio, attn::IndexKeys ik, int max_n,
                   uint16_t* keys, int keys_stride, uint16_t* bkeys, int bkeys_stride, cudaStream_t s) {
  static const int nsm = [] {
    int d, v;
    CUDA_CHECK(cudaGetDevice(&d));
    CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d));
    return v;
  }();
  if (rows <= 0 || max_n <= 0) return;
  const int nblk = (max_n + kSPos - 1) / kSPos;
  if (keys_stride < nblk * kSPos || (bkeys && bkeys_stride < nblk * kSPos / 8))
    die("index_score_p: strides %d/%d for %d positions", keys_stride, bkeys_stride, max_n);
  // enough CTAs for ~8 per SM: split the rows when the positions alone give too few
  const int rchunks = (rows + kSRows - 1) / kSRows;
  const int ysplit = std::min(rchunks, std::max(1, (8 * nsm + nblk - 1) / nblk));
  const int rows_per_cta = (rchunks + ysplit - 1) / ysplit * kSRows;
  SArgs a{q.q, q.sc, q.w, pos, rows, ratio, rows_per_cta, ik.k, reinterpret_cast<const uint32_t*>(ik.sc),
          keys, keys_stride, bkeys, bkeys_stride};
  dim3 grid(nblk, (rows + rows_per_cta - 1) / rows_per_cta);
  if (bkeys)
    score_p_kernel<true><<<grid, kSThreads, 0, s>>>(a);
  else
    score_p_kernel<false><<<grid, kSThreads, 0, s>>>(a);
  CUDA_CHECK(cudaGetLastError());
}

void index_topk_p(const attn::IndexQ& q, const int* pos, int rows, int ratio, attn::IndexKeys ik, int max_n, int batch,
                  uint16_t* keys, int keys_stride, int kmax, int* out_pos, uint16_t* out_key, uint16_t* bkeys,
                  int pool_k, int* pool_pos, uint16_t* pool_key, cudaStream_t s) {
  for (int r0 = 0; r0 < rows; r0 += batch) {
    const int nr = std::min(batch, rows - r0);
    const attn::IndexQ qb{q.q + (size_t)r0 * kIdxHeads * 64, q.sc + (size_t)r0 * kIdxHeads, q.w + (size_t)r0 * kIdxHeads};
    index_score_p(qb, pos + r0, nr, ratio, ik, max_n, keys, keys_stride, bkeys, keys_stride / 8, s);
    attn::index_topk(keys, keys_stride, pos + r0, nr, ratio, attn::IndexScan{}, max_n, false, kmax,
                     out_pos + (size_t)r0 * kmax, out_key + (size_t)r0 * kmax, kmax, s);
    if (bkeys)
      attn::index_topk(bkeys, keys_stride / 8, pos + r0, nr, ratio, attn::IndexScan{}, max_n, true, pool_k,
                       pool_pos + (size_t)r0 * pool_k, pool_key + (size_t)r0 * pool_k, pool_k, s);
  }
}

}  // namespace dsk
