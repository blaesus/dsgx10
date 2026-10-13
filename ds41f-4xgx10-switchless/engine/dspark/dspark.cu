// DSpark draft module: kernels and host side (dspark.h, README.md).
#include "dspark/dspark.h"

#include <string>
#include <type_traits>

#include "attn/attn_common.cuh"
#include "attn/launch.h"
#include "common/check.h"
#include "weights/loader.h"

namespace dspark {
namespace {

using attn::bf2f;
using attn::f2bf;
using attn::launch;
using attn::pdl_trigger;
using attn::pdl_wait;

constexpr float kEps = 1e-20f;       // rms_norm_eps
constexpr float kRouteScale = 1.5f;  // route_scale
constexpr int kHcChunks = 40;        // hc_chunks(0, 20480): each rank's 10, gathered in rank order
constexpr int kChunkBytes = 10 * kDraft * 25 * 4;
constexpr int kSplits = 3;           // sparse_attn CTAs per row: the window's 3 blocks of 64 keys
constexpr int kMarkovCtas = kVocabRank / 64;  // 505 CTAs x 8 warps x 8 rows
constexpr int kStreamBytes = kDraft * kHc * kDim * 2;
constexpr int kTopKPad = dsk::kTopK;  // dsk::moe's picks per row

__nv_bfloat16* bf(uint16_t* p) { return reinterpret_cast<__nv_bfloat16*>(p); }
const __nv_bfloat16* bf(const uint16_t* p) { return reinterpret_cast<const __nv_bfloat16*>(p); }

__device__ __forceinline__ bool better(float v, int id, float bv, int bid) { return v > bv || (v == bv && id < bid); }

// ---- kernels ------------------------------------------------------------------------------------------------

// torch's bf16 mean over the 4 copies: an fp32 sum in copy order, times 1/4, rounded. Block (c, r): 256 columns.
__global__ void __launch_bounds__(256) main_hidden_kernel(const uint16_t* x, int slot, uint16_t* mh) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y, k = blockIdx.x * 256 + threadIdx.x;
  const uint16_t* p = x + (size_t)r * kHc * kDim + k;
  const float s = __fadd_rn(__fadd_rn(__fadd_rn(bf2f(p[0]), bf2f(p[kDim])), bf2f(p[2 * kDim])), bf2f(p[3 * kDim]));
  mh[(size_t)r * kMainK + slot * kDim + k] = f2bf(__fmul_rn(s, 0.25f));
}

// accept, kernel 1: act_quant of main_hidden's rows [0, count) for main_proj (later rows: zeros), and the anchor
// start = pos0 + count - 1. Block (c, r): 256 columns of row r; a warp is one scale block of 32.
// row0 (device, may be null): the rows start at main_hidden's row *row0 (a batch step's entry; docs/batching.md B3).
__global__ void __launch_bounds__(256) accept_prep_kernel(const uint16_t* mh, const int* count, const int* pos0,
                                                          const int* row0, uint8_t* hq, uint8_t* hs, int* start) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
  const int n = *count, off = row0 ? *row0 : 0;
  if (r == 0 && c == 0) *start = *pos0 + n - 1;
  const float v = r < n ? bf2f(mh[(size_t)(r + off) * kMainK + c]) : 0.f;
  const float sc = attn::fp8_block_scale(attn::warp_max(fabsf(v)));
  hq[(size_t)r * kMainK + c] = attn::fp8_code(v, sc);
  if ((threadIdx.x & 31) == 0) hs[(size_t)r * (kMainK / 32) + c / 32] = attn::f_to_ue8m0(sc);
}

// draft, kernel 1: the block's input stream (row 0 the token at start + 1, rows 1-4 the noise token; each embedded
// and copied 4 times), the rows' positions start + 1 .. start + 5 and window bounds, the identity pre-mix, ids[0].
// Block j: row j.
__global__ void __launch_bounds__(kDim / 8) draft_prep_kernel(const int* token, const int* start,
                                                              const uint16_t* embed, uint16_t* x, int* pos, int* lo,
                                                              int* hi, float* pre, int* ids) {
  pdl_trigger();
  pdl_wait();
  const int j = blockIdx.x, t = threadIdx.x;
  const int id = j == 0 ? *token : kNoiseToken;
  const uint4 v = reinterpret_cast<const uint4*>(embed + (size_t)id * kDim)[t];
  uint4* dst = reinterpret_cast<uint4*>(x + (size_t)j * kHc * kDim);
#pragma unroll
  for (int c = 0; c < kHc; ++c) dst[c * (kDim / 8) + t] = v;
  if (t < kHc) pre[j * kHc + t] = t == 0 ? 1.f : 0.f;
  if (t == 0) {
    const int p = *start;
    pos[j] = p + 1 + j;
    lo[j] = max(0, p - (attn::kWindow - 1));
    hi[j] = p + kDraft;
    if (j == 0) ids[0] = id;
  }
}

// draft_batch, kernel 1 (B3 phase 2a): draft_prep_kernel over a pass's rows. Block (j, q) is row kDraft q + j,
// entry ent[q]'s: its token (j = 0, tok[ent * tok_stride]) or the noise token, embedded and copied 4 times; its
// positions and window bounds from its own anchor start[ent]; the identity pre-mix; ids[8 ent] its token.
__global__ void __launch_bounds__(kDim / 8) draft_prep_batch_kernel(const int* ent, const int* tok, int tok_stride,
                                                                    const int* start, const uint16_t* embed,
                                                                    uint16_t* x, int* pos, int* lo, int* hi,
                                                                    float* pre, int* ids) {
  pdl_trigger();
  pdl_wait();
  const int j = blockIdx.x, q = blockIdx.y, r = q * kDraft + j, t = threadIdx.x;
  const int e = ent[q];
  const int id = j == 0 ? tok[e * tok_stride] : kNoiseToken;
  const uint4 v = reinterpret_cast<const uint4*>(embed + (size_t)id * kDim)[t];
  uint4* dst = reinterpret_cast<uint4*>(x + (size_t)r * kHc * kDim);
#pragma unroll
  for (int c = 0; c < kHc; ++c) dst[c * (kDim / 8) + t] = v;
  if (t < kHc) pre[r * kHc + t] = t == 0 ? 1.f : 0.f;
  if (t == 0) {
    const int p = start[e];
    pos[r] = p + 1 + j;
    lo[r] = max(0, p - (attn::kWindow - 1));
    hi[r] = p + kDraft;
    if (j == 0) ids[8 * e] = id;
  }
}

// The DSpark router (model.py Gate with 128 experts, top-3): scores sqrt(softplus(l)), top-3 of scores + bias in
// descending order (ties: the lower id), weights = score / (sum + 1e-20) * 1.5 with torch's order for a sum of
// three, (p0 + p2) + p1. dsk::moe takes 6 picks per row: picks 4-6 repeat 1-3 with the same weights (moe() builds
// each row's set of experts, so a repeat changes nothing). A warp per row. BATCH (draft_batch): a block per
// entry, block q's warps rows kDraft q .., the same arithmetic per row.
template <bool BATCH>
__global__ void __launch_bounds__(32 * kDraft) router3_kernel(const float* logits, const float* bias, int* idx,
                                                              float* wts) {
  pdl_trigger();
  const int r = (BATCH ? blockIdx.x * kDraft : 0) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  float b[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) b[j] = bias[lane + 32 * j];
  pdl_wait();
  float sc[4], key[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float l = logits[r * kExperts + lane + 32 * j];
    sc[j] = sqrtf(l > 20.f ? l : log1pf(expf(l)));
    key[j] = sc[j] + b[j];
  }
  float picked[kTopK];
  int ids[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float bv = -INFINITY;
    int bj = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (key[j] > bv) bv = key[j], bj = j;
    int be = bv == -INFINITY ? 1 << 30 : lane + 32 * bj;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (better(ov, oe, bv, be)) bv = ov, be = oe;
    }
    ids[k] = be;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      if (lane + 32 * j == be) s = sc[j], key[j] = -INFINITY;
    picked[k] = __shfl_sync(0xffffffffu, s, be & 31);
  }
  const float sum = __fadd_rn(__fadd_rn(__fadd_rn(picked[0], picked[2]), picked[1]), 1e-20f);
  if (lane < kTopKPad) {
    const int k = lane % kTopK;
    idx[r * kTopKPad + lane] = ids[k];
    wts[r * kTopKPad + lane] = __fmul_rn(__fdiv_rn(picked[k], sum), kRouteScale);
  }
}

// Step j of the Markov chain on this rank's vocab slice: the Markov embedding e of the step's token (ids[0] for
// j = 0, else the gathered argmax of step j - 1; forced[j] when given), bias = head slice . e (fp32), dl[j] =
// logits[j] + bias, and dl[j]'s argmax (ties: the lowest id). A warp per L2 row group of 8 rows: lane l holds 8
// values of row l/4 per 512-byte chunk; the last CTA to finish reduces the CTAs' candidates into cand[j].
// F8: the Markov head as markov_fp8() lays it out (mhead e4m3, mscale its 32x32 blocks' ue8m0), the same values
// per lane: each decoded to fp32 times its block's scale (exact) and multiplied in the same order, so the bias
// is sim_fp8_l2's bit for bit.
// BATCH (draft_batch): grid (CTAs, entries); entry q = blockIdx.y works on its own row kDraft q + j of logits, dl
// and emb, its ids (ids + 8 ent[q]), column q of the pass's candidates [j][4] (gcand, cand), its own CTA
// candidates and counter: the chain of each entry as alone, in the same order. Without BATCH every offset is
// zero and the kernel is the one-session one.
template <bool F8, bool BATCH>
__global__ void __launch_bounds__(256) markov_kernel(int j, const int* ids, const Cand* gcand, const int* forced,
                                                     const uint16_t* membed, const uint8_t* mhead,
                                                     const uint8_t* mscale, const float* logits, float* dl,
                                                     float* bias_out, uint16_t* emb, Cand* cta, Cand* cand,
                                                     unsigned* counter, int vbase, const int* ent) {
  __shared__ float e[kMarkov];
  __shared__ Cand red[8];
  __shared__ bool last;
  pdl_trigger();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = blockIdx.x * 8 + warp;
  const int q = BATCH ? blockIdx.y : 0, jr = BATCH ? q * kDraft + j : j, cs = BATCH ? 4 : 1;
  if constexpr (BATCH) {
    ids += 8 * ent[q];
    cta += q * kMarkovCtas;
    counter += q;
  }
  uint4 w[F8 ? 4 : 8];
  uint2 ws = make_uint2(0, 0);
  if constexpr (F8) {
#pragma unroll
    for (int p = 0; p < 4; ++p)
      w[p] = __ldg(reinterpret_cast<const uint4*>(mhead + (size_t)g * 2048 + p * 512 + lane * 16));
    ws = __ldg(reinterpret_cast<const uint2*>(mscale + (size_t)(g / 4) * 8));
  } else {
#pragma unroll
    for (int kc = 0; kc < 8; ++kc)
      w[kc] = __ldg(reinterpret_cast<const uint4*>(mhead + (size_t)g * 4096 + kc * 512 + lane * 16));
  }
  pdl_wait();
  const int tok = forced ? forced[j] : j == 0 ? ids[0] : gcand[(j - 1) * cs + q].id;
  if (threadIdx.x < kMarkov / 2) {
    const uint32_t v = reinterpret_cast<const uint32_t*>(membed + (size_t)tok * kMarkov)[threadIdx.x];
    e[2 * threadIdx.x] = bf2f(v & 0xFFFF);
    e[2 * threadIdx.x + 1] = bf2f(v >> 16);
    if (blockIdx.x == 0) reinterpret_cast<uint32_t*>(emb + jr * kMarkov)[threadIdx.x] = v;
  }
  __syncthreads();
  float acc = 0.f;
  const int e0 = (lane & 3) * 8;
  if constexpr (F8) {
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      const float sc = attn::ue8m0_to_f((uint8_t)((kc < 4 ? ws.x : ws.y) >> (8 * (kc & 3))));
      const uint32_t u[2] = {kc & 1 ? w[kc / 2].z : w[kc / 2].x, kc & 1 ? w[kc / 2].w : w[kc / 2].y};
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc = fmaf(__fmul_rn(attn::e4m3_to_f((uint8_t)(u[i / 4] >> (8 * (i % 4)))), sc), e[kc * 32 + e0 + i], acc);
    }
  } else {
#pragma unroll
    for (int kc = 0; kc < 8; ++kc) {
      const uint32_t u[4] = {w[kc].x, w[kc].y, w[kc].z, w[kc].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc = fmaf(bf2f(u[i] & 0xFFFF), e[kc * 32 + e0 + 2 * i], acc);
        acc = fmaf(bf2f(u[i] >> 16), e[kc * 32 + e0 + 2 * i + 1], acc);
      }
    }
  }
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);
  acc += __shfl_xor_sync(0xffffffffu, acc, 2);
  const int row = g * 8 + (lane >> 2);
  float v = __fadd_rn(logits[(size_t)jr * kVocabRank + row], acc);
  if ((lane & 3) == 0) {
    dl[(size_t)jr * kVocabRank + row] = v;
    if (bias_out) bias_out[(size_t)jr * kVocabRank + row] = acc;
  }
  int id = vbase + row;
#pragma unroll
  for (int o = 16; o >= 4; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oid = __shfl_xor_sync(0xffffffffu, id, o);
    if (better(ov, oid, v, id)) v = ov, id = oid;
  }
  if (lane == 0) red[warp] = {v, id};
  __syncthreads();
  if (threadIdx.x == 0) {
    Cand b = red[0];
    for (int i = 1; i < 8; ++i)
      if (better(red[i].v, red[i].id, b.v, b.id)) b = red[i];
    cta[blockIdx.x] = b;
    __threadfence();
    last = atomicAdd(counter, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  Cand b{-INFINITY, 1 << 30};
  for (int i = threadIdx.x; i < (int)gridDim.x; i += blockDim.x) {
    const float cv = __ldcg(&cta[i].v);
    const int ci = __ldcg(&cta[i].id);
    if (better(cv, ci, b.v, b.id)) b = {cv, ci};
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b.v, o);
    const int oid = __shfl_xor_sync(0xffffffffu, b.id, o);
    if (better(ov, oid, b.v, b.id)) b = {ov, oid};
  }
  if (lane == 0) red[warp] = b;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < 8; ++i)
      if (better(red[i].v, red[i].id, b.v, b.id)) b = red[i];
    cand[j * cs + q] = b;
    *counter = 0;
  }
}

// The confidence head: conf[j] = [x_j, e_j] . w in fp32 (x_j the head's input before its norm, e_j the Markov
// embedding of ids[j]); survival[j] = prod_{k <= j} sigmoid(conf[k]) at conf + 8; ids[1..5] = the drafts. A warp
// per row. BATCH (draft_batch): a block per entry, block q over rows kDraft q .. of hx and emb, column q of the
// candidates [j][4], into entry ent[q]'s ids (ids + 8 ent[q]) and conf (conf + 16 ent[q]).
template <bool BATCH>
__global__ void __launch_bounds__(32 * kDraft) conf_kernel(const uint16_t* hx, const uint16_t* emb,
                                                           const uint16_t* w, const Cand* gcand, int* ids,
                                                           float* conf, const int* ent) {
  __shared__ float c[kDraft];
  pdl_trigger();
  pdl_wait();
  const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int q = BATCH ? blockIdx.x : 0, jr = BATCH ? q * kDraft + j : j, cs = BATCH ? 4 : 1;
  if constexpr (BATCH) {
    const int e = ent[q];
    ids += 8 * e;
    conf += 16 * e;
  }
  float acc = 0.f;
  auto dot8 = [&](const uint16_t* a, const uint16_t* b) {
    const uint4 x = *reinterpret_cast<const uint4*>(a), y = *reinterpret_cast<const uint4*>(b);
    const uint32_t xs[4] = {x.x, x.y, x.z, x.w}, ys[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      acc = fmaf(bf2f(xs[i] & 0xFFFF), bf2f(ys[i] & 0xFFFF), acc);
      acc = fmaf(bf2f(xs[i] >> 16), bf2f(ys[i] >> 16), acc);
    }
  };
  for (int k = lane * 8; k < kDim; k += 256) dot8(hx + (size_t)jr * kDim + k, w + k);
  dot8(emb + jr * kMarkov + lane * 8, w + kDim + lane * 8);
  acc = attn::warp_sum(acc);
  if (lane == 0) {
    c[j] = acc;
    conf[j] = acc;
    ids[1 + j] = gcand[j * cs + q].id;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float p = 1.f;
    for (int i = 0; i < kDraft; ++i) {
      p *= 1.f / (1.f + expf(-c[i]));
      conf[8 + i] = p;
    }
  }
}

// ---- EXPERIMENT, bench only: the probe (Draft::probe; docs/engine.md, "Draft trees") ------------------------

// a rank's record per depth, in words: its top kProbeK (value, id), its max and sum of exp(v - max), path[j]'s
// logit and whether the rank's slice holds path[j]
constexpr int kProbeRec = 2 * kProbeK + 4;
constexpr int kProbeBytes = kDraft * kProbeRec * 4;

__device__ __forceinline__ void warp_best(float& v, int& id) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oid = __shfl_xor_sync(0xffffffffu, id, o);
    if (better(ov, oid, v, id)) v = ov, id = oid;
  }
}
// the block's best (value, id) by better(), in every thread; sv, si: 32 entries of shared memory
__device__ void block_best(float& v, int& id, float* sv, int* si) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  warp_best(v, id);
  __syncthreads();
  if (lane == 0) sv[warp] = v, si[warp] = id;
  __syncthreads();
  v = lane < nw ? sv[lane] : -INFINITY;
  id = lane < nw ? si[lane] : (1 << 30);
  warp_best(v, id);
}
// the block's sum, in every thread
__device__ float block_sum(float v, float* sv) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  v = attn::warp_sum(v);
  __syncthreads();
  if (lane == 0) sv[warp] = v;
  __syncthreads();
  return attn::warp_sum(lane < nw ? sv[lane] : 0.f);
}

// The Markov chain's tokens along the path: the anchor's token, then path[0 .. 3] (a missing one: any id; the
// depths after it have no path token).
__global__ void probe_prep_kernel(const int* ids, const int* path, int* forced) {
  pdl_trigger();
  pdl_wait();
  const int j = threadIdx.x;
  if (j < kDraft) forced[j] = j == 0 ? ids[0] : max(path[j - 1], 0);
}

// This rank's record of depth j = blockIdx.x from its biased logits dl[j]: the top kProbeK in kProbeK passes,
// each taking the best below the one before (better() is a strict order).
__global__ void __launch_bounds__(1024) probe_topk_kernel(const float* dl, const int* path, int vbase, float* rec) {
  __shared__ float sv[32];
  __shared__ int si[32];
  pdl_trigger();
  pdl_wait();
  const int j = blockIdx.x;
  const float* x = dl + (size_t)j * kVocabRank;
  float* out = rec + j * kProbeRec;
  float m = -INFINITY;
  int unused = 0;
  for (int i = threadIdx.x; i < kVocabRank; i += blockDim.x) m = fmaxf(m, x[i]);
  block_best(m, unused, sv, si);
  float s = 0.f;
  for (int i = threadIdx.x; i < kVocabRank; i += blockDim.x) s += expf(x[i] - m);
  s = block_sum(s, sv);
  float pv = INFINITY;
  int pi = -1;
  for (int k = 0; k < kProbeK; ++k) {
    float bv = -INFINITY;
    int bi = (1 << 30);
    for (int i = threadIdx.x; i < kVocabRank; i += blockDim.x) {
      const float v = x[i];
      const int id = vbase + i;
      if (better(pv, pi, v, id) && better(v, id, bv, bi)) bv = v, bi = id;
    }
    block_best(bv, bi, sv, si);
    if (threadIdx.x == 0) out[2 * k] = bv, out[2 * k + 1] = __int_as_float(bi);
    pv = bv, pi = bi;
  }
  if (threadIdx.x == 0) {
    const int t = path[j];
    const bool mine = t >= vbase && t < vbase + kVocabRank;
    out[2 * kProbeK] = m;
    out[2 * kProbeK + 1] = s;
    out[2 * kProbeK + 2] = mine ? x[t - vbase] : -INFINITY;
    out[2 * kProbeK + 3] = __int_as_float(mine ? 1 : 0);
  }
}

// Every rank's records (gathered in rank order) -> ProbeOut, a thread per depth: the top kProbeK of the ranks'
// candidates, the log-sum-exp over the ranks in rank order, path[j]'s probability and rank.
__global__ void probe_merge_kernel(const float* grec, const int* path, const float* conf, ProbeOut* out) {
  pdl_trigger();
  pdl_wait();
  const int j = threadIdx.x;
  if (j >= kDraft) return;
  auto rec = [&](int r) { return grec + (r * kDraft + j) * kProbeRec; };
  float m = -INFINITY;
  for (int r = 0; r < kRanks; ++r) m = fmaxf(m, rec(r)[2 * kProbeK]);
  float s = 0.f;
  for (int r = 0; r < kRanks; ++r) s += rec(r)[2 * kProbeK + 1] * expf(rec(r)[2 * kProbeK] - m);
  const float lse = m + logf(s);
  float pv = INFINITY;
  int pi = -1;
  for (int k = 0; k < kProbeK; ++k) {
    float bv = -INFINITY;
    int bi = (1 << 30);
    for (int r = 0; r < kRanks; ++r)
      for (int c = 0; c < kProbeK; ++c) {
        const float v = rec(r)[2 * c];
        const int id = __float_as_int(rec(r)[2 * c + 1]);
        if (better(pv, pi, v, id) && better(v, id, bv, bi)) bv = v, bi = id;
      }
    out->id[j][k] = bi, out->p[j][k] = expf(bv - lse);
    pv = bv, pi = bi;
  }
  const int t = path[j];
  int rank = -1;
  float pp = 0.f;
  if (t >= 0) {
    rank = kProbeK;
    for (int k = kProbeK - 1; k >= 0; --k)
      if (out->id[j][k] == t) rank = k;
    for (int r = 0; r < kRanks; ++r)
      if (__float_as_int(rec(r)[2 * kProbeK + 3])) pp = expf(rec(r)[2 * kProbeK + 2] - lse);
  }
  out->rank[j] = rank, out->p_path[j] = pp, out->conf[j] = conf[j], out->lse[j] = lse;
}

// ---- STAND-IN collectives -----------------------------------------------------------------------------------

struct GatherArgs {
  GatherSeg s[kMaxGatherSegs];
  int n;
};
// all-gather: this rank's rows into every rank's slot (4-byte words)
__global__ void standin_gather_kernel(GatherArgs a) {
  pdl_trigger();
  pdl_wait();
  for (int i = 0; i < a.n; ++i) {
    const GatherSeg& g = a.s[i];
    const int words = (int)(g.width / 4), per_row = words * kRanks;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < g.rows * per_row; t += gridDim.x * blockDim.x) {
      const int r = t / per_row, q = t % per_row / words, k = t % words;
      reinterpret_cast<uint32_t*>(static_cast<char*>(g.dst) + r * g.dst_ld + q * g.width)[k] =
          reinterpret_cast<const uint32_t*>(static_cast<const char*>(g.src) + r * g.src_ld)[k];
    }
  }
}
// all-reduce: this rank's partial as the sum
__global__ void standin_reduce_kernel(const void* src, bool f32, uint16_t* dst, int n) {
  pdl_trigger();
  pdl_wait();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    dst[i] = f32 ? f2bf(static_cast<const float*>(src)[i]) : static_cast<const uint16_t*>(src)[i];
}
// argmax: this rank's candidate as the winner
__global__ void standin_argmax_kernel(const Cand* src, Cand* dst) {
  pdl_trigger();
  pdl_wait();
  if (threadIdx.x == 0) *dst = *src;
}

class StandinComm final : public Comm {
 public:
  void all_gather(const GatherSeg* segs, int n, cudaStream_t s) override {
    if (n < 1 || n > kMaxGatherSegs) die("standin all_gather: %d segments", n);
    GatherArgs a{};
    for (int i = 0; i < n; ++i) {
      a.s[i] = segs[i];
      if (segs[i].width % 4) die("standin all_gather: width %zu", segs[i].width);
    }
    a.n = n;
    launch(standin_gather_kernel, dim3(8), dim3(256), 0, s, true, a);
  }
  void all_reduce(const void* src, bool f32, uint16_t* dst, size_t n, cudaStream_t s) override {
    launch(standin_reduce_kernel, dim3(20), dim3(256), 0, s, true, src, f32, dst, (int)n);
  }
  void argmax(const Cand* src, Cand* dst, cudaStream_t s) override {
    launch(standin_argmax_kernel, dim3(1), dim3(32), 0, s, true, src, dst);
  }
};

// ---- weights --------------------------------------------------------------------------------------------------

struct Finder {
  std::initializer_list<const dsw::WeightFile*> files;
  const void* get(const std::string& name, uint64_t bytes) const {
    for (const dsw::WeightFile* f : files) {
      const dsw::DeviceTensor* t = f->find(name);
      if (!t || !t->data) continue;
      if (t->bytes != bytes) die("dspark: %s has %llu bytes, expected %llu", name.c_str(), (unsigned long long)t->bytes,
                                 (unsigned long long)bytes);
      return t->data;
    }
    die("dspark: no loaded tensor %s", name.c_str());
  }
  const uint8_t* u8(const std::string& n, uint64_t b) const { return static_cast<const uint8_t*>(get(n, b)); }
  const uint16_t* u16(const std::string& n, uint64_t elems) const {
    return static_cast<const uint16_t*>(get(n, elems * 2));
  }
  const float* f32(const std::string& n, uint64_t elems) const { return static_cast<const float*>(get(n, elems * 4)); }
};

}  // namespace

Weights weights_from(std::initializer_list<const dsw::WeightFile*> files) {
  const Finder f{files};
  Weights w{};
  for (int i = 0; i < kBlocks; ++i) {
    const std::string p = "mtp." + std::to_string(i) + ".";
    const std::string a = p + "attn.", e = p + "ffn.experts.", sh = p + "ffn.shared_experts.";
    BlockW& b = w.b[i];
    b.wq_a = f.u8(a + "wq_a.weight", 320ull * kDim), b.wq_a_s = f.u8(a + "wq_a.scale", 10 * 160);
    b.wkv = f.u8(a + "wkv.weight", 128ull * kDim), b.wkv_s = f.u8(a + "wkv.scale", 4 * 160);
    b.wq_b = f.u8(a + "wq_b.weight", 8192ull * kQLora), b.wq_b_s = f.u8(a + "wq_b.scale", 256 * 40);
    b.wo_a = f.u8(a + "wo_a.weight", 2048ull * 4096), b.wo_a_s = f.u8(a + "wo_a.scale", 64 * 128);
    b.wo_b = f.u8(a + "wo_b.weight", 5120ull * 2048), b.wo_b_s = f.u8(a + "wo_b.scale", 160 * 64);
    b.q_norm = f.u16(a + "q_norm.weight", kQLora), b.kv_norm = f.u16(a + "kv_norm.weight", 512);
    b.attn_norm = f.u16(p + "attn_norm.weight", kDim), b.ffn_norm = f.u16(p + "ffn_norm.weight", kDim);
    b.sink = f.f32(a + "attn_sink", 16);
    b.attn_fn = f.f32(p + "hc_attn_fn", 24 * kDim), b.attn_base = f.f32(p + "hc_attn_base", 24);
    b.attn_scale = f.f32(p + "hc_attn_scale", 3);
    b.ffn_fn = f.f32(p + "hc_ffn_fn", 24 * kDim), b.ffn_base = f.f32(p + "hc_ffn_base", 24);
    b.ffn_scale = f.f32(p + "hc_ffn_scale", 3);
    b.gate = f.u8(p + "ffn.gate.weight", 32ull * kDim * 2), b.gate_bias = f.f32(p + "ffn.gate.bias", kExperts);
    const uint64_t up = 128ull * 576 * 2560, up_s = 128ull * 576 * 160, dn = 128ull * 5120 * 288,
                   dn_s = 128ull * 5120 * 18;
    b.moe = {f.u8(e + "w1.weight", up),           f.u8(e + "w1.scale", up_s),         f.u8(e + "w3.weight", up),
             f.u8(e + "w3.scale", up_s),          f.u8(e + "w2.weight", dn),          f.u8(e + "w2.scale", dn_s),
             f.u8(sh + "w1.weight", 576ull * kDim), f.u8(sh + "w1.scale", 18 * 160), f.u8(sh + "w3.weight", 576ull * kDim),
             f.u8(sh + "w3.scale", 18 * 160),     f.u8(sh + "w2.weight", 576ull * kDim), f.u8(sh + "w2.scale", 160 * 18)};
  }
  w.main_proj = f.u8("mtp.0.main_proj.weight", 1280ull * kMainK);
  w.main_proj_s = f.u8("mtp.0.main_proj.scale", 40 * 480);
  w.main_norm = f.u16("mtp.0.main_norm.weight", kDim);
  w.norm = f.u16("mtp.2.norm.weight", kDim);
  w.embed = f.u16("embed.weight", (uint64_t)kVocab * kDim);
  w.head = f.u8("head.weight", (uint64_t)kVocabRank * kDim * 2);
  w.markov_embed = f.u16("mtp.2.markov_head.embed.weight", (uint64_t)kVocab * kMarkov);
  w.markov_head = f.u8("mtp.2.markov_head.head.weight", (uint64_t)kVocabRank * kMarkov * 2);
  w.conf = f.u16("mtp.2.confidence_head.proj.weight", kDim + kMarkov);
  return w;
}

void main_hidden(const uint16_t* x, int rows, int slot, uint16_t* mh, bool pdl, cudaStream_t s) {
  if (rows < 1 || rows > kMaxHidden || slot < 0 || slot > 2) die("dspark::main_hidden: rows %d slot %d", rows, slot);
  launch(main_hidden_kernel, dim3(kDim / 256, rows), dim3(256), 0, s, pdl, x, slot, mh);
}

namespace {
// head_fp8: a warp per 32x32 block, lane = column; bf16 L2 tiles [N/8][2K/64][8][64 B] in, e4m3 L1 tiles
// [N/8][K/64][8][64 B] and ue8m0 [N/32][K/32] out
__global__ void __launch_bounds__(256) head_fp8_kernel(const uint8_t* head, uint8_t* q, uint8_t* s) {
  constexpr int N = kVocabRank, K = kDim;
  const int blk = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (blk >= N / 32 * (K / 32)) return;
  const int n0 = blk / (K / 32) * 32, k = blk % (K / 32) * 32 + lane;
  float v[32], amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    const size_t off = ((size_t)(n / 8) * (2 * K / 64) + k / 32) * 512 + (n % 8) * 64 + (k % 32) * 2;
    v[i] = bf2f(*reinterpret_cast<const uint16_t*>(head + off));
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float sc = attn::fp8_block_scale(attn::warp_max(amax));
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    q[((size_t)(n / 8) * (K / 64) + k / 64) * 512 + (n % 8) * 64 + k % 64] = attn::fp8_code(v[i], sc);
  }
  if (lane == 0) s[blk] = attn::f_to_ue8m0(sc);
}
}  // namespace

void head_fp8(const uint8_t* head, uint8_t* q, uint8_t* s, cudaStream_t st) {
  launch(head_fp8_kernel, dim3(kVocabRank / 32 * (kDim / 32) / 8), dim3(256), 0, st, false, head, q, s);
}

namespace {
// head_fp4: a warp per 32x32 block of the FP8 copy, lane = column, row i's 32 values one group (sim_fp4's
// walk): each value's simulated r (e2m1 under the group's g, halved into e4m3 under the block's scale doubled,
// read back), then the least power-of-two scale from g up (at most 8g) under which every r of the group is an
// e2m1 value; the codes and scales in dsk::gemv_fp4w_f32's layouts. cnt[0]: groups without one; cnt[1]: groups
// whose scale moved up from g; cnt[2]: groups whose scale is above 2 (gemv_fp4w_f32 takes ue8m0 bytes up to 128).
__global__ void __launch_bounds__(256) head_fp4_kernel(const uint8_t* q8, const uint8_t* s8, uint8_t* q4, uint8_t* s4,
                                                       unsigned* cnt) {
  constexpr int N = kVocabRank, K = kDim;
  const int blk = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (blk >= N / 32 * (K / 32)) return;
  const int n0 = blk / (K / 32) * 32, k = blk % (K / 32) * 32 + lane, kk = k % 128;
  const int e8 = s8[blk];
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    const float v = attn::e4m3_to_f(q8[((size_t)(n / 8) * (K / 64) + k / 64) * 512 + (n % 8) * 64 + k % 64]);
    const float g = attn::fp4_e8m0_scale(attn::warp_max(fabsf(v)));
    const float q = __fmul_rn(attn::e2m1_to_f(attn::fp4_code(v, g)), g);
    const float r = __fmul_rn(attn::e4m3_to_f(attn::f_to_e4m3(__fmul_rn(q, 0.5f))), 2.0f);
    float gs = g;
    uint32_t c = attn::fp4_code(r, gs);
    int up = 0;
    while (!__all_sync(0xffffffffu, __fmul_rn(attn::e2m1_to_f(c), gs) == r) && up < 3)
      gs = __fmul_rn(gs, 2.0f), c = attn::fp4_code(r, gs), ++up;
    const bool ok = __all_sync(0xffffffffu, __fmul_rn(attn::e2m1_to_f(c), gs) == r);
    const bool zero = __all_sync(0xffffffffu, r == 0.0f);  // any scale holds them: the block's own
    const int e4 = zero ? e8 : e8 + (int)(__float_as_uint(gs) >> 23) - 127;
    const uint32_t c1 = __shfl_down_sync(0xffffffffu, c, 1);
    const size_t tile = (size_t)(n / 8) * (K / 128) + k / 128;
    if (!(lane & 1)) q4[tile * 512 + (n % 8) * 64 + (kk % 64) / 16 * 16 + kk / 64 * 8 + (kk % 16) / 2] = (uint8_t)(c | c1 << 4);
    if (lane == 0) {
      s4[tile * 32 + (n % 8) * 4 + kk / 32] = (uint8_t)e4;
      if (!(ok || zero)) atomicAdd(cnt, 1u);
      else if (e4 < 0 || e4 > 128) atomicAdd(cnt + 2, 1u);
      if (up && !zero) atomicAdd(cnt + 1, 1u);
    }
  }
}

// markov_fp8: a warp per 32x32 block, lane = column, sim_fp8_l2's values, into markov_kernel<true>'s order
__global__ void __launch_bounds__(256) markov_fp8_kernel(const uint8_t* mh, uint8_t* q, uint8_t* s) {
  constexpr int N = kVocabRank, K = kMarkov;
  const int blk = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (blk >= N / 32 * (K / 32)) return;
  const int n0 = blk / (K / 32) * 32, kc = blk % (K / 32), k = kc * 32 + lane;
  float v[32], amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    v[i] = bf2f(*reinterpret_cast<const uint16_t*>(mh + ((size_t)(n / 8) * (2 * K / 64) + k / 32) * 512 + (n % 8) * 64 +
                                                   (k % 32) * 2));
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float sc = attn::fp8_block_scale(attn::warp_max(amax));
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    q[(size_t)(n / 8) * 2048 + (kc / 2) * 512 + ((n % 8) * 4 + lane / 8) * 16 + (kc % 2) * 8 + lane % 8] =
        attn::fp8_code(v[i], sc);
  }
  if (lane == 0) s[blk] = attn::f_to_ue8m0(sc);
}
}  // namespace

long head_fp4(const uint8_t* head8, const uint8_t* head8_s, uint8_t* q, uint8_t* s, cudaStream_t st) {
  unsigned* cnt;
  CUDA_CHECK(cudaMalloc(&cnt, 12));
  CUDA_CHECK(cudaMemsetAsync(cnt, 0, 12, st));
  launch(head_fp4_kernel, dim3(kVocabRank / 32 * (kDim / 32) / 8), dim3(256), 0, st, false, head8, head8_s, q, s, cnt);
  unsigned h[3];
  CUDA_CHECK(cudaMemcpyAsync(h, cnt, 12, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaFree(cnt));
  if (h[0] || h[2])
    die("dspark::head_fp4: %u groups without an MXFP4 scale for their simulated values, %u under a scale above 2",
        h[0], h[2]);
  return h[1];
}

void markov_fp8(const uint8_t* markov_head, uint8_t* q, uint8_t* s, cudaStream_t st) {
  launch(markov_fp8_kernel, dim3(kVocabRank / 32 * (kMarkov / 32) / 8), dim3(256), 0, st, false, markov_head, q, s);
}

// ---- EXPERIMENT, measurement only: the draft pass study's rounded weights ---------------------------------------

namespace {
// a warp per 32x32 block, lane = column; row i's 32 values are one group
__global__ void __launch_bounds__(256) sim_fp4_kernel(uint8_t* w, uint8_t* s, int N, int K, bool l1) {
  const int blk = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (blk >= N / 32 * (K / 32)) return;
  const int n0 = blk / (K / 32) * 32, k = blk % (K / 32) * 32 + lane;
#pragma unroll 4
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    const size_t off = l1 ? ((size_t)(n / 8) * (K / 64) + k / 64) * 512 + (n % 8) * 64 + k % 64 : (size_t)n * K + k;
    const float v = attn::e4m3_to_f(w[off]);  // in units of the block's scale
    const float g = attn::fp4_e8m0_scale(attn::warp_max(fabsf(v)));
    const float q = __fmul_rn(attn::e2m1_to_f(attn::fp4_code(v, g)), g);
    w[off] = attn::f_to_e4m3(__fmul_rn(q, 0.5f));
  }
  if (lane == 0) s[blk] = (uint8_t)(s[blk] + 1);
}

__global__ void __launch_bounds__(256) sim_fp8_l2_kernel(uint8_t* w, int N, int K) {
  const int blk = (blockIdx.x * 256 + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (blk >= N / 32 * (K / 32)) return;
  const int n0 = blk / (K / 32) * 32, k = blk % (K / 32) * 32 + lane;
  float v[32], amax = 0.f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    v[i] = bf2f(*reinterpret_cast<const uint16_t*>(w + ((size_t)(n / 8) * (2 * K / 64) + k / 32) * 512 + (n % 8) * 64 +
                                                   (k % 32) * 2));
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float sc = attn::fp8_block_scale(attn::warp_max(amax));
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int n = n0 + i;
    const float r = __fmul_rn(attn::e4m3_to_f(attn::fp8_code(v[i], sc)), sc);  // exact in bf16
    *reinterpret_cast<uint16_t*>(w + ((size_t)(n / 8) * (2 * K / 64) + k / 32) * 512 + (n % 8) * 64 + (k % 32) * 2) =
        attn::f2bf(r);
  }
}
}  // namespace

void sim_fp4(uint8_t* w, uint8_t* s, int N, int K, bool l1, cudaStream_t st) {
  if (N % 32 || K % (l1 ? 64 : 32)) die("dspark::sim_fp4: %d x %d", N, K);
  launch(sim_fp4_kernel, dim3((N / 32 * (K / 32) + 7) / 8), dim3(256), 0, st, false, w, s, N, K, l1);
}

void sim_fp8_l2(uint8_t* w, int N, int K, cudaStream_t st) {
  if (N % 32 || K % 32) die("dspark::sim_fp8_l2: %d x %d", N, K);
  launch(sim_fp8_l2_kernel, dim3((N / 32 * (K / 32) + 7) / 8), dim3(256), 0, st, false, w, N, K);
}

Comm* standin_comm() {
  static StandinComm c;
  return &c;
}

// ---- Draft -----------------------------------------------------------------------------------------------------

struct Draft::Buf {
  void* arena = nullptr;
  // accept
  uint8_t *hq, *hs;
  int* start;
  uint16_t *mp_r, *mp, *main_x, *mkv_r, *mkv;
  attn::Window ring[kBlocks];
  // draft
  int *pos, *lo, *hi;
  uint16_t *xa, *xb;
  float *chunks_r, *chunks, *pre, *post, *comb;
  uint16_t *y, *qa_r, *kvp_r, *qa, *kvp, *qb, *o, *part, *sub, *hx, *hxn, *emb;
  uint8_t *xq, *xs, *qq, *qs, *oq, *os;
  float *so, *sml, *lg_r, *lg, *wts, *ymoe, *logits, *dl;
  int* idx;
  void* moe_ws;
  Cand *cta, *cand, *gcand;
  unsigned* counter;
  int* ids;
  float* conf;
  // EXPERIMENT: the probe's, in their own arena (enable_probe)
  void* probe_arena = nullptr;
  int *pr_forced, *pr_ids;
  float *pr_dl, *pr_conf, *pr_rec, *pr_grec;
  uint16_t* pr_emb;
  Cand* pr_cand;
  ProbeOut* pr_out;
  // B3 phase 2a: a batch pass's buffers (enable_batch), draft()'s at kBatchRows rows, in their own arena; per
  // entry its anchor and results (bstart, bids [8], bconf [16]); the Markov chain's candidates [kDraft][4]
  void* batch_arena = nullptr;
  int *bstart, *bids;
  float* bconf;
  int *bpos, *blo, *bhi;
  uint16_t *bxa, *bxb;
  float *bchunks_r, *bchunks, *bpre, *bpost, *bcomb;
  uint16_t *by, *bqa_r, *bkvp_r, *bqa, *bkvp, *bqb, *bo, *bpart, *bsub, *bhx, *bhxn, *bemb;
  uint8_t *bxq, *bxs, *bqq, *bqs, *boq, *bos;
  float *bso, *bsml, *blg_r, *blg, *bwts, *bymoe, *blogits, *bdl;
  int* bidx;
  void* bmoe_ws;
  Cand *bcta, *bcand, *bgcand;
  unsigned* bcounter;
};

Draft::Draft(const Weights& w, int rank, Comm* comm, const float2* table, int slots)
    : w_(w), rank_(rank), comm_(comm), table_(table), slots_(slots), b_(new Buf) {
  if (rank < 0 || rank >= kRanks || !comm || !table || slots < 1) die("dspark::Draft: rank %d, %d slots", rank, slots);
  Buf& b = *b_;
  // one arena, carved in two passes: sizes, then pointers
  size_t off = 0;
  char* base = nullptr;
  auto take = [&](auto*& p, size_t bytes) {
    off = (off + 255) / 256 * 256;
    p = reinterpret_cast<std::remove_reference_t<decltype(p)>>(base ? base + off : nullptr);
    off += bytes;
  };
  constexpr size_t R = kDraft, A = kMaxAcc;
  for (int pass = 0; pass < 2; ++pass) {
    off = 0;
    take(b.hq, A * kMainK), take(b.hs, A * kMainK / 32), take(b.start, 4);
    take(b.mp_r, A * 1280 * 2), take(b.mp, A * kDim * 2), take(b.main_x, A * kDim * 2);
    take(b.mkv_r, kBlocks * A * 128 * 2), take(b.mkv, kBlocks * A * 512 * 2);
    for (attn::Window& r : b.ring) take(r.kv, slots_ * attn::kRing * 512), take(r.sc, slots_ * attn::kRing * 16);
    take(b.pos, R * 4), take(b.lo, R * 4), take(b.hi, R * 4);
    take(b.xa, kStreamBytes), take(b.xb, kStreamBytes);
    take(b.chunks_r, kChunkBytes), take(b.chunks, kChunkBytes * kRanks);
    take(b.pre, R * 4 * 4), take(b.post, R * 4 * 4), take(b.comb, R * 16 * 4);
    take(b.y, R * kDim * 2), take(b.xq, R * kDim), take(b.xs, R * kDim / 32);
    take(b.qa_r, R * 320 * 2), take(b.kvp_r, R * 128 * 2), take(b.qa, R * kQLora * 2), take(b.kvp, R * 512 * 2);
    take(b.qq, R * kQLora), take(b.qs, R * kQLora / 32), take(b.qb, R * 8192 * 2);
    take(b.so, R * kSplits * 16 * 512 * 4), take(b.sml, R * kSplits * 16 * 2 * 4), take(b.o, R * 8192 * 2);
    take(b.oq, R * 2048), take(b.os, R * 2048 / 32), take(b.part, R * kDim * 2), take(b.sub, R * kDim * 2);
    take(b.lg_r, R * 32 * 4), take(b.lg, R * kExperts * 4), take(b.idx, R * kTopKPad * 4), take(b.wts, R * kTopKPad * 4);
    take(b.ymoe, R * kDim * 4), take(b.moe_ws, dsk::moe_workspace_bytes());
    take(b.hx, R * kDim * 2), take(b.hxn, R * kDim * 2), take(b.logits, R * kVocabRank * 4), take(b.dl, R * kVocabRank * 4);
    take(b.emb, R * kMarkov * 2), take(b.cta, kMarkovCtas * sizeof(Cand)), take(b.cand, 8 * sizeof(Cand));
    take(b.gcand, 8 * sizeof(Cand)), take(b.counter, 4), take(b.ids, 8 * 4), take(b.conf, 16 * 4);
    if (pass == 0) {
      CUDA_CHECK(cudaMalloc(&b.arena, off));
      CUDA_CHECK(cudaMemset(b.arena, 0, off));
      base = static_cast<char*>(b.arena);
    }
  }
  ids_ = b.ids;
  conf_ = b.conf;
  if (const char* v = getenv("DSPARK_VROWS")) {  // EXPERIMENT, bench only
    vrows_ = atoi(v);
    if (vrows_ < 64 || vrows_ > kVocabRank || vrows_ % 64) die("DSPARK_VROWS=%s", v);
  }
}

Draft::~Draft() {
  cudaFree(b_->arena);
  cudaFree(b_->probe_arena);
  cudaFree(b_->batch_arena);
  delete b_;
}

const float* Draft::logits() const { return b_->dl; }
attn::Window Draft::window(int b) const { return win(b, nullptr); }

void Draft::use_slot(int s) {
  if (s < 0 || s >= slots_) die("dspark::Draft: slot %d of %d", s, slots_);
  cur_ = s;
}

attn::Window Draft::win(int b, const int* slot) const {
  constexpr size_t kv = attn::kRing * 512, sc = attn::kRing * 16;
  attn::Window w = b_->ring[b];  // slot 0's
  if (slot && slots_ > 1) {
    w.slot = slot, w.kv_stride = kv, w.sc_stride = sc;
  } else {
    w.kv += cur_ * kv, w.sc += cur_ * sc;
  }
  return w;
}

namespace {
// tests: a Taps pointer gets a copy out, an Inject pointer is copied in (null: nothing, so no graph node)
void tap(void* dst, const void* src, size_t bytes, cudaStream_t s) {
  if (dst) CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s));
}
void copy_in(void* dst, const void* src, size_t bytes, cudaStream_t s) {
  if (src) CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s));
}
void tap_mix(float* dst, const float* pre, const float* post, const float* comb, cudaStream_t s) {
  if (!dst) return;
  CUDA_CHECK(cudaMemcpy2DAsync(dst, 24 * 4, pre, 16, 16, kDraft, cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpy2DAsync(dst + 4, 24 * 4, post, 16, 16, kDraft, cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpy2DAsync(dst + 8, 24 * 4, comb, 64, 64, kDraft, cudaMemcpyDeviceToDevice, s));
}
}  // namespace

void Draft::accept(const uint16_t* mh, const int* count, const int* pos0, cudaStream_t s, const int* slot,
                   const int* row0, int* start_out) {
  Buf& b = *b_;
  dsk::GemvCfg c8{1, 1, 16}, c8k{1, 1, 8};
  c8.tiled = c8k.tiled = true;
  tap(taps.main_hidden, mh, (size_t)kMaxAcc * kMainK * 2, s);
  launch(accept_prep_kernel, dim3(kMainK / 256, kMaxAcc), dim3(256), 0, s, true, mh, count, pos0, row0, b.hq, b.hs,
         start_out ? start_out : b.start);
  dsk::gemv_fp8(b.hq, b.hs, kMaxAcc, w_.main_proj, w_.main_proj_s, kDim / kRanks, kMainK, bf(b.mp_r), c8, s);
  const GatherSeg g{b.mp_r, 1280 * 2, b.mp, kDim * 2, 1280 * 2, kMaxAcc};
  comm_->all_gather(&g, 1, s);  // COLLECTIVE: all-gather of main_proj's outputs
  dsk::Fp8Seg seg[kBlocks];
  GatherSeg gk[kBlocks];
  attn::KvSets sets;
  sets.n = kBlocks;
  for (int i = 0; i < kBlocks; ++i) {
    uint16_t *part = b.mkv_r + (size_t)i * kMaxAcc * 128, *full = b.mkv + (size_t)i * kMaxAcc * 512;
    seg[i] = {w_.b[i].wkv, w_.b[i].wkv_s, bf(part), 128};
    gk[i] = {part, 128 * 2, full, 512 * 2, 128 * 2, kMaxAcc};
    sets.s[i] = {full, w_.b[i].kv_norm, win(i, slot)};
  }
  dsk::gemv_fp8_norm(bf(b.mp), bf(w_.main_norm), kEps, kMaxAcc, seg, kBlocks, kDim, c8k, s, bf(b.main_x));
  tap(taps.main_x, b.main_x, (size_t)kMaxAcc * kDim * 2, s);
  comm_->all_gather(gk, kBlocks, s);  // COLLECTIVE: all-gather of the 3 blocks' wkv(main_x)
  attn::kv_write_sets(sets, kMaxAcc, count, pos0, table_, true, s);
}

// mHC before a sublayer: this rank's projection chunks, gathered (or, with the whole projection, all 40 here),
// then the mixes, the pre-mix with the previous sublayer's pre (pre_in = pre: read, then overwritten with this
// sublayer's), the norm and act_quant -> y, xq, xs
void Draft::mix(const uint16_t* x, const float* fn, const float* fn_all, const float* scale, const float* base,
                const uint16_t* norm, cudaStream_t s) {
  Buf& b = *b_;
  if (fn_all) {
    attn::hc_project(x, kDraft, fn_all, 0, kHc * kDim, b.chunks, true, s, 0);
  } else {
    attn::hc_project(x, kDraft, fn, kDim * rank_, kDim * (rank_ + 1), b.chunks_r, true, s, kDim);
    const GatherSeg g{b.chunks_r, kChunkBytes, b.chunks, kChunkBytes, kChunkBytes, 1};
    comm_->all_gather(&g, 1, s);  // COLLECTIVE: all-gather of the hc projection chunks (40 in rank order)
  }
  attn::hc_mix(x, b.chunks, kHcChunks, kDraft, scale, base, norm, b.pre, b.post, b.comb, b.y, b.xq, b.xs, true, s,
               b.pre);
}

void Draft::draft(const int* token, cudaStream_t s, const int* slot) {
  Buf& b = *b_;
  // wq_b and wo_b with two MMA tiles per warp: half the CTAs, each row's sums as with one (~10 us less per block)
  dsk::GemvCfg c8k{1, 1, 8}, cqb{2, 1, 8}, cob{2, 1, 16}, cw{1, 4, 4}, cg{1, 1, 8}, ch{2, 1, 16}, ch8{2, 1, 8};
  c8k.tiled = cqb.tiled = cob.tiled = true;
  dsk::MoeCfg mc{};
  mc.down = 4;
  const size_t rows_bytes = (size_t)kDraft * kDim * 2;
  launch(draft_prep_kernel, dim3(kDraft), dim3(kDim / 8), 0, s, true, token, (const int*)b.start, w_.embed, b.xa,
         b.pos, b.lo, b.hi, b.pre, ids_);
  for (int i = 0; i < kBlocks; ++i) {
    const BlockW& w = w_.b[i];
    // ---- attention sublayer: SWA over the window's last 128 main positions and the block itself
    copy_in(b.xa, inject.stream[i], kStreamBytes, s);
    copy_in(b.pre, inject.pre[i], kDraft * 16, s);
    mix(b.xa, w.attn_fn, w.attn_fn_all, w.attn_scale, w.attn_base, w.attn_norm, s);
    tap_mix(taps.mix[i][0], b.pre, b.post, b.comb, s);
    tap(taps.attn_in[i], b.y, rows_bytes, s);
    const dsk::Fp8Seg seg[2] = {{w.wq_a, w.wq_a_s, bf(b.qa_r), 320}, {w.wkv, w.wkv_s, bf(b.kvp_r), 128}};
    dsk::gemv_fp8_multi(b.xq, b.xs, kDraft, seg, 2, kDim, c8k, s);
    const GatherSeg g[2] = {{b.qa_r, 320 * 2, b.qa, kQLora * 2, 320 * 2, kDraft},
                            {b.kvp_r, 128 * 2, b.kvp, 512 * 2, 128 * 2, kDraft}};
    comm_->all_gather(g, 2, s);  // COLLECTIVE: all-gather of wq_a's and wkv's outputs
    tap(taps.qa[i], b.qa, (size_t)kDraft * kQLora * 2, s);
    attn::DecodePrep a;
    a.pos = b.pos, a.table = table_, a.qa = b.qa, a.q_norm = w.q_norm, a.qq = b.qq, a.qs = b.qs;
    a.kv_pre = b.kvp, a.kv_norm = w.kv_norm, a.win = win(i, slot);
    attn::decode_prep(a, kDraft, true, s);
    dsk::gemv_fp8(b.qq, b.qs, kDraft, w.wq_b, w.wq_b_s, 8192, kQLora, bf(b.qb), cqb, s);
    tap(taps.q_pre[i], b.qb, (size_t)kDraft * 8192 * 2, s);
    attn::sparse_attn(b.qb, b.pos, {b.lo, b.hi}, kDraft, table_, win(i, slot), attn::Compressed{}, nullptr, 0, w.sink,
                      b.o, kSplits, {b.so, b.sml, kSplits}, s, true);
    tap(taps.attn_o[i], b.o, (size_t)kDraft * 8192 * 2, s);
    dsk::gemv_woa(bf(b.o), kDraft, w.wo_a, w.wo_a_s, 2048, 4096, 1024, nullptr, cw, false, s, b.oq, b.os);
    dsk::gemv_fp8(b.oq, b.os, kDraft, w.wo_b, w.wo_b_s, kDim, 2048, bf(b.part), cob, s);
    comm_->all_reduce(b.part, false, b.sub, (size_t)kDraft * kDim, s);  // COLLECTIVE: all-reduce of wo_b's partial
    tap(taps.attn_out[i], b.sub, rows_bytes, s);
    attn::hc_post(b.sub, b.xa, b.post, b.comb, kDraft, b.xb, s, true);
    tap(taps.mid[i], b.xb, kStreamBytes, s);
    // ---- FFN sublayer: 128 experts, top-3, and the shared expert
    copy_in(b.xb, inject.mid[i], kStreamBytes, s);
    copy_in(b.pre, inject.attn_pre[i], kDraft * 16, s);
    mix(b.xb, w.ffn_fn, w.ffn_fn_all, w.ffn_scale, w.ffn_base, w.ffn_norm, s);
    tap_mix(taps.mix[i][1], b.pre, b.post, b.comb, s);
    tap(taps.ffn_in[i], b.y, rows_bytes, s);
    if (w.gate_all) {
      dsk::gemv_bf16(bf(b.y), kDraft, w.gate_all, kExperts, kDim, b.lg, nullptr, cg, false, s);
    } else {
      dsk::gemv_bf16(bf(b.y), kDraft, w.gate, 32, kDim, b.lg_r, nullptr, cg, false, s);
      const GatherSeg gl{b.lg_r, 32 * 4, b.lg, kExperts * 4, 32 * 4, kDraft};
      comm_->all_gather(&gl, 1, s);  // COLLECTIVE: all-gather of the router logits
    }
    launch(router3_kernel<false>, dim3(1), dim3(32 * kDraft), 0, s, true, (const float*)b.lg, w.gate_bias, b.idx,
           b.wts);
    tap(taps.route_idx[i], b.idx, kDraft * kTopKPad * 4, s);
    tap(taps.route_w[i], b.wts, kDraft * kTopKPad * 4, s);
    dsk::moe(b.xq, b.xs, kDraft, b.idx, b.wts, w.moe, b.moe_ws, b.ymoe, mc, s);
    comm_->all_reduce(b.ymoe, true, b.sub, (size_t)kDraft * kDim, s);  // COLLECTIVE: all-reduce of the MoE's partial
    tap(taps.ffn_out[i], b.sub, rows_bytes, s);
    attn::hc_post(b.sub, b.xb, b.post, b.comb, kDraft, b.xa, s, true);
    tap(taps.out[i], b.xa, kStreamBytes, s);
  }
  // ---- head: hc_pre with the last pre-mix, norm, the vocab slice; the Markov chain; the confidence head
  copy_in(b.xa, inject.head_stream, kStreamBytes, s);
  copy_in(b.pre, inject.head_pre, kDraft * 16, s);
  attn::hc_pre(b.xa, b.pre, kDraft, w_.norm, b.hxn, b.hx, s, true);
  tap(taps.head_x, b.hx, rows_bytes, s);
  tap(taps.head_xn, b.hxn, rows_bytes, s);
  if (w_.head4)  // ch8's wk: the FP8 kernel's sums, on the MXFP4 values
    dsk::gemv_fp4w_f32(bf(b.hxn), kDraft, w_.head4, w_.head4_s, vrows_, kDim, b.logits, ch8, s);
  else if (w_.head8)
    dsk::gemv_fp8w_f32(bf(b.hxn), kDraft, w_.head8, w_.head8_s, vrows_, kDim, b.logits, ch8, s);
  else if (w_.head_coded)
    dsk::gemv_bf16x(bf(b.hxn), kDraft, w_.head_coded, kVocabRank, kDim, b.logits, nullptr, ch, s);
  else
    dsk::gemv_bf16(bf(b.hxn), kDraft, w_.head, kVocabRank, kDim, b.logits, nullptr, ch, false, s);
  tap(taps.logits, b.logits, (size_t)kDraft * kVocabRank * 4, s);
  for (int j = 0; j < kDraft; ++j) {
    launch(w_.markov8 ? markov_kernel<true, false> : markov_kernel<false, false>, dim3(vrows_ / 64), dim3(256), 0, s,
           true, j, (const int*)ids_, (const Cand*)b.gcand, inject.ids, w_.markov_embed,
           w_.markov8 ? w_.markov8 : w_.markov_head, w_.markov8_s, (const float*)b.logits, b.dl, taps.bias, b.emb,
           b.cta, b.cand, b.counter, kVocabRank * rank_, (const int*)nullptr);
    comm_->argmax(b.cand + j, b.gcand + j, s);  // COLLECTIVE: argmax over the ranks' candidates
  }
  tap(taps.markov_embed, b.emb, kDraft * kMarkov * 2, s);
  launch(conf_kernel<false>, dim3(1), dim3(32 * kDraft), 0, s, true, (const uint16_t*)b.hx, (const uint16_t*)b.emb,
         w_.conf, (const Cand*)b.gcand, ids_, conf_, (const int*)nullptr);
}

// ---- B3 phase 2a: one pass over several entries (docs/batching.md) ------------------------------------------------
// draft() over a pass's rows, side by side, in the batch buffers: the same launches with rows = kDraft n and the
// kernels' row dimension (an activation row per MMA lane group; hc_mix, decode_prep and the attention a CTA per
// row; the MoE's union of the rows' experts with each row's own sum), so each row's arithmetic is draft()'s for it
// alone (dspark.h). No taps or injects here (the tests take draft()).

void Draft::enable_batch() {
  Buf& b = *b_;
  if (b.batch_arena) return;
  size_t off = 0;
  char* base = nullptr;
  auto take = [&](auto*& p, size_t bytes) {
    off = (off + 255) / 256 * 256;
    p = reinterpret_cast<std::remove_reference_t<decltype(p)>>(base ? base + off : nullptr);
    off += bytes;
  };
  constexpr size_t R = kBatchRows, E = kBatchEnts, C = kDraft * 4;  // C: the candidates of a pass, [kDraft][4]
  for (int pass = 0; pass < 2; ++pass) {
    off = 0;
    take(b.bstart, kMaxEnts * 4), take(b.bids, kMaxEnts * 8 * 4), take(b.bconf, kMaxEnts * 16 * 4);
    take(b.bpos, R * 4), take(b.blo, R * 4), take(b.bhi, R * 4);
    take(b.bxa, R * kHc * kDim * 2), take(b.bxb, R * kHc * kDim * 2);
    take(b.bchunks_r, 10 * R * 25 * 4), take(b.bchunks, kHcChunks * R * 25 * 4);
    take(b.bpre, R * 4 * 4), take(b.bpost, R * 4 * 4), take(b.bcomb, R * 16 * 4);
    take(b.by, R * kDim * 2), take(b.bxq, R * kDim), take(b.bxs, R * kDim / 32);
    take(b.bqa_r, R * 320 * 2), take(b.bkvp_r, R * 128 * 2), take(b.bqa, R * kQLora * 2), take(b.bkvp, R * 512 * 2);
    take(b.bqq, R * kQLora), take(b.bqs, R * kQLora / 32), take(b.bqb, R * 8192 * 2);
    take(b.bso, R * kSplits * 16 * 512 * 4), take(b.bsml, R * kSplits * 16 * 2 * 4), take(b.bo, R * 8192 * 2);
    take(b.boq, R * 2048), take(b.bos, R * 2048 / 32), take(b.bpart, R * kDim * 2), take(b.bsub, R * kDim * 2);
    take(b.blg_r, R * 32 * 4), take(b.blg, R * kExperts * 4), take(b.bidx, R * kTopKPad * 4), take(b.bwts, R * kTopKPad * 4);
    take(b.bymoe, R * kDim * 4), take(b.bmoe_ws, dsk::moe_workspace_bytes());
    take(b.bhx, R * kDim * 2), take(b.bhxn, R * kDim * 2), take(b.blogits, R * kVocabRank * 4), take(b.bdl, R * kVocabRank * 4);
    take(b.bemb, R * kMarkov * 2), take(b.bcta, E * kMarkovCtas * sizeof(Cand)), take(b.bcand, C * sizeof(Cand));
    take(b.bgcand, C * sizeof(Cand)), take(b.bcounter, E * 4);
    if (pass == 0) {
      CUDA_CHECK(cudaMalloc(&b.batch_arena, off));
      CUDA_CHECK(cudaMemset(b.batch_arena, 0, off));
      base = static_cast<char*>(b.batch_arena);
    }
  }
}

int* Draft::batch_start() const {
  if (!b_->batch_arena) die("dspark::Draft: batch_start() before enable_batch()");
  return b_->bstart;
}
const int* Draft::batch_ids(int k) const {
  if (!b_->batch_arena || k < 0 || k >= kMaxEnts) die("dspark::Draft: batch_ids(%d)", k);
  return b_->bids + 8 * k;
}
const float* Draft::batch_confidence(int k) const {
  if (!b_->batch_arena || k < 0 || k >= kMaxEnts) die("dspark::Draft: batch_confidence(%d)", k);
  return b_->bconf + 16 * k;
}

void Draft::bmix(const uint16_t* x, int rows, const float* fn, const float* fn_all, const float* scale,
                 const float* base, const uint16_t* norm, cudaStream_t s) {
  Buf& b = *b_;
  if (fn_all) {
    attn::hc_project(x, rows, fn_all, 0, kHc * kDim, b.bchunks, true, s, 0);
  } else {
    attn::hc_project(x, rows, fn, kDim * rank_, kDim * (rank_ + 1), b.bchunks_r, true, s, kDim);
    const size_t bytes = (size_t)10 * rows * 25 * 4;  // this rank's 10 chunks of the rows
    const GatherSeg g{b.bchunks_r, bytes, b.bchunks, bytes, bytes, 1};
    comm_->all_gather(&g, 1, s);  // COLLECTIVE: all-gather of the hc projection chunks (40 in rank order)
  }
  attn::hc_mix(x, b.bchunks, kHcChunks, rows, scale, base, norm, b.bpre, b.bpost, b.bcomb, b.by, b.bxq, b.bxs, true,
               s, b.bpre);
}

void Draft::draft_batch(const BatchPass& p, cudaStream_t s) {
  Buf& b = *b_;
  if (!b.batch_arena) die("dspark::Draft: draft_batch() before enable_batch()");
  if (p.n < 1 || p.n > kBatchEnts || !p.ent || !p.tok || p.tok_stride < 1 || !p.slot)
    die("dspark::Draft: draft_batch of %d entries (1..%d, with tokens and a slot table)", p.n, kBatchEnts);
  const int R = kDraft * p.n;
  // draft()'s configs: the same k order, split-K slices and folds per output
  dsk::GemvCfg c8k{1, 1, 8}, cqb{2, 1, 8}, cob{2, 1, 16}, cw{1, 4, 4}, cg{1, 1, 8}, ch{2, 1, 16}, ch8{2, 1, 8};
  c8k.tiled = cqb.tiled = cob.tiled = true;
  dsk::MoeCfg mc{};
  mc.down = 4;
  launch(draft_prep_batch_kernel, dim3(kDraft, p.n), dim3(kDim / 8), 0, s, true, p.ent, p.tok, p.tok_stride,
         (const int*)b.bstart, w_.embed, b.bxa, b.bpos, b.blo, b.bhi, b.bpre, b.bids);
  for (int i = 0; i < kBlocks; ++i) {
    const BlockW& w = w_.b[i];
    // ---- attention sublayer: each row's SWA over its own window's last 128 main positions and its block
    bmix(b.bxa, R, w.attn_fn, w.attn_fn_all, w.attn_scale, w.attn_base, w.attn_norm, s);
    const dsk::Fp8Seg seg[2] = {{w.wq_a, w.wq_a_s, bf(b.bqa_r), 320}, {w.wkv, w.wkv_s, bf(b.bkvp_r), 128}};
    dsk::gemv_fp8_multi(b.bxq, b.bxs, R, seg, 2, kDim, c8k, s);
    const GatherSeg g[2] = {{b.bqa_r, 320 * 2, b.bqa, kQLora * 2, 320 * 2, R},
                            {b.bkvp_r, 128 * 2, b.bkvp, 512 * 2, 128 * 2, R}};
    comm_->all_gather(g, 2, s);  // COLLECTIVE: all-gather of wq_a's and wkv's outputs
    attn::DecodePrep a;
    a.pos = b.bpos, a.table = table_, a.qa = b.bqa, a.q_norm = w.q_norm, a.qq = b.bqq, a.qs = b.bqs;
    a.kv_pre = b.bkvp, a.kv_norm = w.kv_norm, a.win = win(i, p.slot);
    attn::decode_prep(a, R, true, s);
    dsk::gemv_fp8(b.bqq, b.bqs, R, w.wq_b, w.wq_b_s, 8192, kQLora, bf(b.bqb), cqb, s);
    attn::sparse_attn(b.bqb, b.bpos, {b.blo, b.bhi}, R, table_, win(i, p.slot), attn::Compressed{}, nullptr, 0, w.sink,
                      b.bo, kSplits, {b.bso, b.bsml, kSplits}, s, true);
    dsk::gemv_woa(bf(b.bo), R, w.wo_a, w.wo_a_s, 2048, 4096, 1024, nullptr, cw, false, s, b.boq, b.bos);
    dsk::gemv_fp8(b.boq, b.bos, R, w.wo_b, w.wo_b_s, kDim, 2048, bf(b.bpart), cob, s);
    comm_->all_reduce(b.bpart, false, b.bsub, (size_t)R * kDim, s);  // COLLECTIVE: all-reduce of wo_b's partial
    attn::hc_post(b.bsub, b.bxa, b.bpost, b.bcomb, R, b.bxb, s, true);
    // ---- FFN sublayer: 128 experts, top-3 per row, and the shared expert
    bmix(b.bxb, R, w.ffn_fn, w.ffn_fn_all, w.ffn_scale, w.ffn_base, w.ffn_norm, s);
    if (w.gate_all) {
      dsk::gemv_bf16(bf(b.by), R, w.gate_all, kExperts, kDim, b.blg, nullptr, cg, false, s);
    } else {
      dsk::gemv_bf16(bf(b.by), R, w.gate, 32, kDim, b.blg_r, nullptr, cg, false, s);
      const GatherSeg gl{b.blg_r, 32 * 4, b.blg, kExperts * 4, 32 * 4, R};
      comm_->all_gather(&gl, 1, s);  // COLLECTIVE: all-gather of the router logits
    }
    launch(router3_kernel<true>, dim3(p.n), dim3(32 * kDraft), 0, s, true, (const float*)b.blg, w.gate_bias, b.bidx,
           b.bwts);
    dsk::moe(b.bxq, b.bxs, R, b.bidx, b.bwts, w.moe, b.bmoe_ws, b.bymoe, mc, s);
    comm_->all_reduce(b.bymoe, true, b.bsub, (size_t)R * kDim, s);  // COLLECTIVE: all-reduce of the MoE's partial
    attn::hc_post(b.bsub, b.bxb, b.bpost, b.bcomb, R, b.bxa, s, true);
  }
  // ---- head: hc_pre with the last pre-mix, norm, the vocab slice; each entry's Markov chain; the confidence head
  attn::hc_pre(b.bxa, b.bpre, R, w_.norm, b.bhxn, b.bhx, s, true);
  if (w_.head4) {  // at most 8 rows a call: each entry's 5, draft()'s call
    for (int q = 0; q < p.n; ++q)
      dsk::gemv_fp4w_f32(bf(b.bhxn) + (size_t)q * kDraft * kDim, kDraft, w_.head4, w_.head4_s, vrows_, kDim,
                         b.blogits + (size_t)q * kDraft * kVocabRank, ch8, s);
  } else if (w_.head8) {
    dsk::gemv_fp8w_f32(bf(b.bhxn), R, w_.head8, w_.head8_s, vrows_, kDim, b.blogits, ch8, s);
  } else if (w_.head_coded) {
    dsk::gemv_bf16x(bf(b.bhxn), R, w_.head_coded, kVocabRank, kDim, b.blogits, nullptr, ch, s);
  } else {
    dsk::gemv_bf16(bf(b.bhxn), R, w_.head, kVocabRank, kDim, b.blogits, nullptr, ch, false, s);
  }
  const int ncand = (p.n + 1) & ~1;  // the pass's candidates per step, an even count (argmax_rows's 16 bytes)
  for (int j = 0; j < kDraft; ++j) {
    launch(w_.markov8 ? markov_kernel<true, true> : markov_kernel<false, true>, dim3(vrows_ / 64, p.n), dim3(256), 0,
           s, true, j, (const int*)b.bids, (const Cand*)b.bgcand, (const int*)nullptr, w_.markov_embed,
           w_.markov8 ? w_.markov8 : w_.markov_head, w_.markov8_s, (const float*)b.blogits, b.bdl, (float*)nullptr,
           b.bemb, b.bcta, b.bcand, b.bcounter, kVocabRank * rank_, p.ent);
    comm_->argmax_rows(b.bcand + j * 4, b.bgcand + j * 4, ncand, s);  // COLLECTIVE: the entries' argmaxes
  }
  launch(conf_kernel<true>, dim3(p.n), dim3(32 * kDraft), 0, s, true, (const uint16_t*)b.bhx, (const uint16_t*)b.bemb,
         w_.conf, (const Cand*)b.bgcand, b.bids, b.bconf, p.ent);
}

// ---- EXPERIMENT, bench only: the probe --------------------------------------------------------------------------

void Draft::enable_probe() {
  Buf& b = *b_;
  if (b.probe_arena) return;
  size_t off = 0;
  char* base = nullptr;
  auto take = [&](auto*& p, size_t bytes) {
    off = (off + 255) / 256 * 256;
    p = reinterpret_cast<std::remove_reference_t<decltype(p)>>(base ? base + off : nullptr);
    off += bytes;
  };
  for (int pass = 0; pass < 2; ++pass) {
    off = 0;
    take(b.pr_forced, 8 * 4), take(b.pr_ids, 8 * 4), take(b.pr_dl, (size_t)kDraft * kVocabRank * 4);
    take(b.pr_conf, 16 * 4), take(b.pr_rec, kProbeBytes), take(b.pr_grec, kRanks * kProbeBytes);
    take(b.pr_emb, kDraft * kMarkov * 2), take(b.pr_cand, 8 * sizeof(Cand)), take(b.pr_out, sizeof(ProbeOut));
    if (pass == 0) {
      CUDA_CHECK(cudaMalloc(&b.probe_arena, off));
      CUDA_CHECK(cudaMemset(b.probe_arena, 0, off));
      base = static_cast<char*>(b.probe_arena);
    }
  }
}

const ProbeOut* Draft::probe_out() const { return b_->pr_out; }

// The Markov chain along the path into the probe's own buffers (the CTA candidates and the counter are
// draft()'s scratch, free again once its chain is done), then each rank's record, gathered, merged.
void Draft::probe(const int* path, cudaStream_t s) {
  Buf& b = *b_;
  if (!b.probe_arena) die("dspark::Draft: probe() before enable_probe()");
  const int vbase = kVocabRank * rank_;
  launch(probe_prep_kernel, dim3(1), dim3(32), 0, s, true, (const int*)ids_, path, b.pr_forced);
  for (int j = 0; j < kDraft; ++j)
    launch(w_.markov8 ? markov_kernel<true, false> : markov_kernel<false, false>, dim3(kMarkovCtas), dim3(256), 0, s,
           true, j, (const int*)ids_, (const Cand*)b.gcand, (const int*)b.pr_forced, w_.markov_embed,
           w_.markov8 ? w_.markov8 : w_.markov_head, w_.markov8_s, (const float*)b.logits, b.pr_dl, nullptr,
           b.pr_emb, b.cta, b.pr_cand, b.counter, vbase, (const int*)nullptr);
  launch(probe_topk_kernel, dim3(kDraft), dim3(1024), 0, s, true, (const float*)b.pr_dl, path, vbase, b.pr_rec);
  launch(conf_kernel<false>, dim3(1), dim3(32 * kDraft), 0, s, true, (const uint16_t*)b.hx, (const uint16_t*)b.pr_emb,
         w_.conf, (const Cand*)b.pr_cand, b.pr_ids, b.pr_conf, (const int*)nullptr);
  const GatherSeg g{b.pr_rec, kProbeBytes, b.pr_grec, kProbeBytes, kProbeBytes, 1};
  comm_->all_gather(&g, 1, s);  // COLLECTIVE: all-gather of the ranks' probe records
  launch(probe_merge_kernel, dim3(1), dim3(32), 0, s, true, (const float*)b.pr_grec, path, (const float*)b.pr_conf,
         b.pr_out);
}

}  // namespace dspark
