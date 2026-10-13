// Collectives fused with the work around them (ops.h), from the ring_dev.cuh halves: each kernel
// copies this rank's contribution, posts it, lets the next kernel launch (PDL: the following GEMV
// loads its first weights while the contributions travel), waits, and consumes the four
// contributions in place, in rank order, exactly as the standalone collective plus the kernels it
// replaces would.
#include <type_traits>

#include "attn/launch.h"
#include "attn/mhc_dev.cuh"
#include "comm/ring_dev.cuh"
#include "common/check.h"
#include "model/ops.h"

namespace model::ops {
namespace {

__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ void launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
// Before next_seq: the kernel before this one (and so every earlier collective) has finished.
__device__ __forceinline__ void wait_prev() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// allreduce_hc's grid: block b takes columns [128 b, 128 b + 128) of every row, in items of 16 bytes of
// each contribution (8 columns of bf16 partial sums, 4 of fp32), one item per thread from warp 1 on. So
// the epilogue's 40 KB per row of hc_post and collapse is spread over 40 SMs (a kernel that waits may
// have at most 48 blocks), and a warp reads contiguous host memory (fp32 items of 8 columns, two loads
// 16 bytes apart, ran 2-4x slower). With the mixes riding along, warp w < R of block 0 works them out
// for row w before the other blocks are let go (wait_all's hook): 40 blocks reading the same few lines
// of host memory instead queue up for them (~11 µs at one row).
// The block holds the rows (an item per thread), so the kernels whose block size depends on the rows come in
// two instances by the row budget RB (ops.h row_budget): kLowRows, the kernels and launches of before B4, for
// steps of at most 8 rows; kMaxRows for 9..16.
constexpr int kArGrid = 40;
constexpr int kArCols = kDim / kArGrid;  // 128
// (kMixStride, ops.h: floats per row of the mix partials, after the partial sums)
constexpr int kMaxChunks = kDim / 512;   // attn::hc_project chunks of one copy
static_assert(kDim % kArGrid == 0, "allreduce_hc columns");
template <bool kF32, int RB>
struct ArShape {
  static constexpr int kCols = kF32 ? 4 : 8;      // columns per item
  static constexpr int kItems = kArCols / kCols;  // per row and block
  // >= 8 warps (up to kLowRows rows: as before B4); past them a warp per row and one more, for the mixes' hook
  static constexpr int kMin = RB <= kLowRows ? 256 : 32 * (RB + 1);
  static constexpr int kThreads = 32 + RB * kItems < kMin ? kMin : 32 + RB * kItems;
};
template <int N>
using BfVec = std::conditional_t<N == 8, uint4, uint2>;  // N bf16 values

template <int N>
__device__ __forceinline__ BfVec<N> pack_bf16(const float* f) {
  BfVec<N> o;
  uint32_t* w = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
  for (int j = 0; j < N / 2; ++j) w[j] = (uint32_t)f2bf(f[2 * j]) | ((uint32_t)f2bf(f[2 * j + 1]) << 16);
  return o;
}

// The sum of this rank's nchunks hc_project chunks [nchunks][R][25] of row r, value v < 25, in chunk order.
__device__ __forceinline__ float chunk_sum(const float* chunks, int nchunks, int R, int r, int v) {
  float pc[kMaxChunks], m = 0.f;
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) pc[c] = c < nchunks ? chunks[((size_t)c * R + r) * 25 + v] : 0.f;
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c)
    if (c < nchunks) m = __fadd_rn(m, pc[c]);
  return m;
}

// kMixes: chunks [nchunks][R][25] (this rank's hc_project of its copy) are summed in chunk order and sent
// along; pre, post and comb are outputs. Otherwise they are inputs.
template <bool kF32, bool kMixes, int RB>
__global__ void __launch_bounds__((ArShape<kF32, RB>::kThreads))
    k_allreduce_hc(comm::DevChannel ch, const uint8_t* part, int R, const float* chunks, int nchunks,
                   const float* scale, const float* base, const uint16_t* res, float* pre, float* post, float* comb,
                   uint16_t* stream, uint16_t* yraw, uint16_t* sub) {
  constexpr int N = ArShape<kF32, RB>::kCols, kItems = ArShape<kF32, RB>::kItems, ES = kF32 ? 4 : 2;
  using Vec = BfVec<N>;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, i = t - 32;
  const bool item = i >= 0 && i < R * kItems;
  const int r = item ? i / kItems : 0, d0 = blockIdx.x * kArCols + (item ? i % kItems : 0) * N;
  float sc = 0.f, bs = 0.f;  // weights, read before the PDL wait
  if (kMixes && blockIdx.x == 0 && warp < R) {
    sc = attn::hc_lane_scale(scale, lane);
    bs = attn::hc_lane_base(base, lane);
  }
  wait_prev();
  const uint64_t s = comm::next_seq(ch);
  uint8_t* sb = comm::send_buf(ch, s);
  const size_t off = ((size_t)r * kDim + d0) * ES;  // the item's 16 bytes of each contribution
  if (item) *reinterpret_cast<uint4*>(sb + off) = *reinterpret_cast<const uint4*>(part + off);
  const size_t moff = (size_t)R * kDim * ES;    // mix partials [R][kMixStride]
  if (kMixes && blockIdx.x < R && t < 25)  // row b's 25 partials over this rank's copy
    reinterpret_cast<float*>(sb + moff)[blockIdx.x * kMixStride + t] = chunk_sum(chunks, nchunks, R, blockIdx.x, t);
  comm::post_grid(ch, s, (uint32_t)(moff + (kMixes ? (size_t)R * kMixStride * 4 : 0)));
  launch_dependents();
  // the stream's four copies at this item (and the mixes, when they are inputs) while the contributions travel
  Vec xin[kHc];
  float po[kHc], pr[kHc], cb[kHc * kHc];
  if (item) {
#pragma unroll
    for (int c = 0; c < kHc; ++c) xin[c] = *reinterpret_cast<const Vec*>(res + (size_t)r * kHcDim + (size_t)c * kDim + d0);
    if (!kMixes) {
#pragma unroll
      for (int j = 0; j < kHc; ++j) po[j] = post[r * kHc + j], pr[j] = yraw ? pre[r * kHc + j] : 0.f;
#pragma unroll
      for (int j = 0; j < kHc * kHc; ++j) cb[j] = comb[r * kHc * kHc + j];
    }
  }
  if constexpr (kMixes) {
    // block 0, warp w < R: row w's partials, the four ranks' added in rank order, and attn::hc_finish's
    // arithmetic into pre, post and comb, which the other blocks read once their wait_all returns
    comm::wait_all(ch, s, [&] {
      if (warp >= R) return;
      float mv = 0.f;
      if (lane < 25) {
        const size_t o = moff + ((size_t)warp * kMixStride + lane) * 4;
        mv = __ldcv(reinterpret_cast<const float*>(comm::src(ch, s, 0) + o));
#pragma unroll
        for (int k = 1; k < kRanks; ++k)
          mv = __fadd_rn(mv, __ldcv(reinterpret_cast<const float*>(comm::src(ch, s, k) + o)));
      }
      const float m = attn::hc_sinkhorn_warp(mv, lane, sc, bs);
      if (lane < kHc) pre[warp * kHc + lane] = m;
      else if (lane < 2 * kHc) post[warp * kHc + lane - kHc] = m;
      else if (lane < kMix) comb[warp * kHc * kHc + lane - 2 * kHc] = m;
    });
    if (item) {
#pragma unroll
      for (int j = 0; j < kHc; ++j) po[j] = __ldcg(post + r * kHc + j), pr[j] = __ldcg(pre + r * kHc + j);
#pragma unroll
      for (int j = 0; j < kHc * kHc; ++j) cb[j] = __ldcg(comb + r * kHc * kHc + j);
    }
  } else {
    comm::wait_all(ch, s);
  }
  float v[N];  // the reduced sublayer output, rounded to bf16 as the standalone all-reduce (+ f32_to_bf16)
  if (item) {
    if constexpr (kF32) {
      const float4 a = comm::sum4_f32(ch, s, off);
      v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w;
    } else {
      comm::sum8_bf16(ch, s, off, v);
    }
  }
  comm::finish_grid(ch, s);  // thread 0 has no item
  if (!item) return;
#pragma unroll
  for (int j = 0; j < N; ++j) v[j] = bf2f(f2bf(v[j]));
  if (sub) *reinterpret_cast<Vec*>(sub + (size_t)r * kDim + d0) = pack_bf16<N>(v);
  // hc_post (attn::hc_post's arithmetic): y[i] = bf16(post[i] v + (((c0i x0 + c1i x1) + c2i x2) + c3i x3))
  float y[kHc][N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    float x[kHc];
#pragma unroll
    for (int c = 0; c < kHc; ++c) x[c] = bf2f(reinterpret_cast<const uint16_t*>(&xin[c])[j]);
#pragma unroll
    for (int c = 0; c < kHc; ++c) {
      float a = __fadd_rn(__fmul_rn(cb[c], x[0]), __fmul_rn(cb[4 + c], x[1]));
      a = __fadd_rn(a, __fmul_rn(cb[8 + c], x[2]));
      a = __fadd_rn(a, __fmul_rn(cb[12 + c], x[3]));
      y[c][j] = bf2f(f2bf(__fadd_rn(__fmul_rn(po[c], v[j]), a)));
    }
  }
  uint16_t* so = stream + (size_t)r * kHcDim;
#pragma unroll
  for (int c = 0; c < kHc; ++c) *reinterpret_cast<Vec*>(so + (size_t)c * kDim + d0) = pack_bf16<N>(y[c]);
  if (!yraw) return;
  // the next sublayer's collapse (attn::hc_pre without the norm): bf16(((p0 y0 + p1 y1) + p2 y2) + p3 y3)
  float a[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    float q = __fadd_rn(__fmul_rn(pr[0], y[0][j]), __fmul_rn(pr[1], y[1][j]));
    q = __fadd_rn(q, __fmul_rn(pr[2], y[2][j]));
    a[j] = __fadd_rn(q, __fmul_rn(pr[3], y[3][j]));
  }
  *reinterpret_cast<Vec*>(yraw + (size_t)r * kDim + d0) = pack_bf16<N>(a);
}

// The attention projections' all-gather, unpacked on the way out (ops::unpack_attn's layout).
// grid kRanks blocks (block k unpacks rank k's contribution). With chunks, the attention's mix partials
// ride along: their sums [R][kMixStride] follow the pack, and block k copies rank k's to mix[k].
__global__ void __launch_bounds__(256) k_allgather_attn(comm::DevChannel ch, const uint8_t* pack, int R, int ncomp,
                                                        uint32_t per_rank, bf16* qa, bf16* kv, float* ckv, float* csc,
                                                        const float* chunks, int nchunks, float* mix) {
  wait_prev();
  const uint64_t s = comm::next_seq(ch);
  uint8_t* sb = comm::send_buf(ch, s);
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < per_rank / 16; i += gridDim.x * blockDim.x)
    reinterpret_cast<uint4*>(sb)[i] = reinterpret_cast<const uint4*>(pack)[i];
  const int gi = blockIdx.x * blockDim.x + threadIdx.x;  // mix value gi % kMixStride of row gi / kMixStride
  if (chunks && gi < R * kMixStride) {
    const int v = gi % kMixStride;
    reinterpret_cast<float*>(sb + per_rank)[gi] = v < 25 ? chunk_sum(chunks, nchunks, R, gi / kMixStride, v) : 0.f;
  }
  comm::post_grid(ch, s, per_rank + (chunks ? (uint32_t)R * kMixStride * 4 : 0));
  launch_dependents();
  comm::wait_all(ch, s);
  const int k = blockIdx.x;
  const uint8_t* b = comm::src(ch, s, k);
  constexpr int cq = kQaPart / 8, ck = kKvPart / 8, cc = kCompPart / 4;  // 16-byte chunks per row
  const int per_row = cq + ck + ncomp * cc;
  for (int i = threadIdx.x; i < R * per_row; i += blockDim.x) {
    const int p = i / per_row;
    int j = i % per_row;
    const uint8_t* from;
    uint8_t* to;
    if (j < cq) {
      from = b + ((size_t)p * kQaPart + 8 * j) * 2;
      to = reinterpret_cast<uint8_t*>(qa + (size_t)p * kQLora + kQaPart * k + 8 * j);
    } else if ((j -= cq) < ck) {
      from = b + (size_t)R * kQaPart * 2 + ((size_t)p * kKvPart + 8 * j) * 2;
      to = reinterpret_cast<uint8_t*>(kv + (size_t)p * kHead + kKvPart * k + 8 * j);
    } else {
      j -= ck;
      const int which = j / cc;  // 0: ckv, 1: csc
      j %= cc;
      from = b + (size_t)R * (kQaPart + kKvPart) * 2 + (size_t)which * R * kCompPart * 4 + ((size_t)p * kCompPart + 4 * j) * 4;
      to = reinterpret_cast<uint8_t*>((which ? csc : ckv) + (size_t)p * kHead + kCompPart * k + 4 * j);
    }
    *reinterpret_cast<uint4*>(to) = comm::ld_cv16(from);
  }
  if (chunks)
    for (int i = threadIdx.x; i < R * kMixStride / 4; i += blockDim.x)
      reinterpret_cast<uint4*>(mix + (size_t)k * R * kMixStride)[i] = comm::ld_cv16(b + per_rank + 16 * i);
  comm::finish_grid(ch, s);
}

// A sublayer's mixes as a collective of their own, for a side stream: this rank's chunk sums [R][kMixStride]
// go out, and warp w < R adds row w's four in rank order and runs attn::hc_finish's arithmetic (as
// allreduce_hc_mix's hook does). One block of a warp per row of the budget RB (kLowRows: the 256 threads of
// before B4). Launched without PDL behind an event, after the kernel that finished the previous collective,
// so next_seq is this one's.
template <int RB>
__global__ void __launch_bounds__(32 * RB) k_mix_allreduce(comm::DevChannel ch, const float* chunks, int nchunks, int R,
                                                           const float* scale, const float* base, float* pre, float* post,
                                                           float* comb) {
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  float sc = 0.f, bs = 0.f;
  if (warp < R) sc = attn::hc_lane_scale(scale, lane), bs = attn::hc_lane_base(base, lane);
  const uint64_t s = comm::next_seq(ch);
  if (t < R * kMixStride) {
    const int v = t % kMixStride;
    reinterpret_cast<float*>(comm::send_buf(ch, s))[t] = v < 25 ? chunk_sum(chunks, nchunks, R, t / kMixStride, v) : 0.f;
  }
  comm::post_grid(ch, s, (uint32_t)R * kMixStride * 4);
  comm::wait_all(ch, s);
  if (warp < R) {
    float mv = 0.f;
    if (lane < 25) {
      const size_t o = ((size_t)warp * kMixStride + lane) * 4;
      mv = __ldcv(reinterpret_cast<const float*>(comm::src(ch, s, 0) + o));
#pragma unroll
      for (int k = 1; k < kRanks; ++k) mv = __fadd_rn(mv, __ldcv(reinterpret_cast<const float*>(comm::src(ch, s, k) + o)));
    }
    const float m = attn::hc_sinkhorn_warp(mv, lane, sc, bs);
    if (lane < kHc) pre[warp * kHc + lane] = m;
    else if (lane < 2 * kHc) post[warp * kHc + lane - kHc] = m;
    else if (lane < kMix) comb[warp * kHc * kHc + lane - 2 * kHc] = m;
  }
  comm::finish_grid(ch, s);
}

// The mixes from the four ranks' partial sums mix [kRanks][R][kMixStride], added in rank order (warp w:
// row w), with attn::hc_finish's arithmetic: allreduce_hc_mix's hook, as a kernel of its own. One block of a
// warp per row (R <= kMaxRows).
__global__ void __launch_bounds__(32 * kMaxRows) k_mix_ranks(const float* mix, int R, const float* scale,
                                                             const float* base, float* pre, float* post, float* comb) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp >= R) return;
  const float sc = attn::hc_lane_scale(scale, lane), bs = attn::hc_lane_base(base, lane);
  float mv = 0.f;
  if (lane < 25) {
    mv = mix[warp * kMixStride + lane];
#pragma unroll
    for (int k = 1; k < kRanks; ++k) mv = __fadd_rn(mv, mix[((size_t)k * R + warp) * kMixStride + lane]);
  }
  const float m = attn::hc_sinkhorn_warp(mv, lane, sc, bs);
  if (lane < kHc) pre[warp * kHc + lane] = m;
  else if (lane < 2 * kHc) post[warp * kHc + lane - kHc] = m;
  else if (lane < kMix) comb[warp * kHc * kHc + lane - 2 * kHc] = m;
}

template <bool kF32, bool kMixes>
void launch_allreduce_hc(const comm::DevChannel& ch, const void* part, int R, const float* chunks, int nchunks,
                         const float* scale, const float* base, const bf16* res, float* pre, float* post, float* comb,
                         bf16* stream, bf16* yraw, bf16* sub, cudaStream_t s) {
  const size_t bytes = (size_t)R * kDim * (kF32 ? 4 : 2) + (kMixes ? (size_t)R * kMixStride * 4 : 0);
  if (R < 1 || R > kMaxRows || bytes > ch.cap || (kMixes && (nchunks < 1 || nchunks > kMaxChunks)))
    die("allreduce_hc: %d rows, %d chunks, %zu bytes (cap %u)", R, nchunks, bytes, ch.cap);
  if (R <= kLowRows)
    attn::launch(k_allreduce_hc<kF32, kMixes, kLowRows>, dim3(kArGrid), dim3((ArShape<kF32, kLowRows>::kThreads)), 0, s,
                 true, ch, static_cast<const uint8_t*>(part), R, chunks, nchunks, scale, base,
                 reinterpret_cast<const uint16_t*>(res), pre, post, comb, reinterpret_cast<uint16_t*>(stream),
                 reinterpret_cast<uint16_t*>(yraw), reinterpret_cast<uint16_t*>(sub));
  else
    attn::launch(k_allreduce_hc<kF32, kMixes, kMaxRows>, dim3(kArGrid), dim3((ArShape<kF32, kMaxRows>::kThreads)), 0, s,
                 true, ch, static_cast<const uint8_t*>(part), R, chunks, nchunks, scale, base,
                 reinterpret_cast<const uint16_t*>(res), pre, post, comb, reinterpret_cast<uint16_t*>(stream),
                 reinterpret_cast<uint16_t*>(yraw), reinterpret_cast<uint16_t*>(sub));
}

}  // namespace

void allreduce_hc(const comm::DevChannel& ch, const void* part, bool f32, int R, const bf16* res, const float* post,
                  const float* comb, const float* pre, bf16* stream, bf16* yraw, bf16* sub, cudaStream_t s) {
  float *pr = const_cast<float*>(pre), *po = const_cast<float*>(post), *cb = const_cast<float*>(comb);
  if (f32) launch_allreduce_hc<true, false>(ch, part, R, nullptr, 0, nullptr, nullptr, res, pr, po, cb, stream, yraw, sub, s);
  else launch_allreduce_hc<false, false>(ch, part, R, nullptr, 0, nullptr, nullptr, res, pr, po, cb, stream, yraw, sub, s);
}

void allreduce_hc_mix(const comm::DevChannel& ch, const void* part, bool f32, int R, const float* chunks, int nchunks,
                      const float* scale, const float* base, const bf16* res, float* pre, float* post, float* comb,
                      bf16* stream, bf16* yraw, bf16* sub, cudaStream_t s) {
  if (f32) launch_allreduce_hc<true, true>(ch, part, R, chunks, nchunks, scale, base, res, pre, post, comb, stream, yraw, sub, s);
  else launch_allreduce_hc<false, true>(ch, part, R, chunks, nchunks, scale, base, res, pre, post, comb, stream, yraw, sub, s);
}

void allgather_attn(const comm::DevChannel& ch, const uint8_t* pack, int R, int ncomp, bf16* qa, bf16* kv, float* ckv,
                    float* csc, cudaStream_t s) {
  attn::launch(k_allgather_attn, dim3(kRanks), dim3(256), 0, s, true, ch, pack, R, ncomp,
               (uint32_t)attn_pack_bytes(R, ncomp), qa, kv, ckv, csc, nullptr, 0, nullptr);
}

void allgather_attn_mix(const comm::DevChannel& ch, const uint8_t* pack, int R, int ncomp, const float* chunks,
                        int nchunks, bf16* qa, bf16* kv, float* ckv, float* csc, float* mix, cudaStream_t s) {
  const size_t bytes = attn_pack_bytes(R, ncomp) + (size_t)R * kMixStride * 4;
  if (R < 1 || R > kMaxRows || nchunks < 1 || nchunks > kMaxChunks || bytes > ch.cap)
    die("allgather_attn_mix: %d rows, %d chunks, %zu bytes (cap %u)", R, nchunks, bytes, ch.cap);
  attn::launch(k_allgather_attn, dim3(kRanks), dim3(256), 0, s, true, ch, pack, R, ncomp,
               (uint32_t)attn_pack_bytes(R, ncomp), qa, kv, ckv, csc, chunks, nchunks, mix);
}

void mix_allreduce(const comm::DevChannel& ch, const float* chunks, int nchunks, int R, const float* scale,
                   const float* base, float* pre, float* post, float* comb, cudaStream_t s) {
  if (R < 1 || R > kMaxRows || nchunks < 1 || nchunks > kMaxChunks || (size_t)R * kMixStride * 4 > ch.cap)
    die("mix_allreduce: %d rows, %d chunks (cap %u)", R, nchunks, ch.cap);
  if (R <= kLowRows)
    attn::launch(k_mix_allreduce<kLowRows>, dim3(1), dim3(32 * kLowRows), 0, s, false, ch, chunks, nchunks, R, scale, base,
                 pre, post, comb);
  else
    attn::launch(k_mix_allreduce<kMaxRows>, dim3(1), dim3(32 * kMaxRows), 0, s, false, ch, chunks, nchunks, R, scale, base,
                 pre, post, comb);
}

void mix_ranks(const float* mix, int R, const float* scale, const float* base, float* pre, float* post, float* comb,
               cudaStream_t s) {
  if (R < 1 || R > kMaxRows) die("mix_ranks: %d rows", R);
  attn::launch(k_mix_ranks, dim3(1), dim3(32 * R), 0, s, false, mix, R, scale, base, pre, post, comb);
}

}  // namespace model::ops
