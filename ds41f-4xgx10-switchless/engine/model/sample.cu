// Probabilistic speculative-sampling additions (2026): reimplemented for
// dsgx10's CUDA/ring execution path after consulting DeepInfer, revision
// 8b16149741a4935e1dec8cea27853f273b374c5b. Relevant upstream portions retain
// their MIT notices. DeepInfer-derived additions are provided under Apache-2.0;
// see repository NOTICE, LICENSES/Apache-2.0-DeepInfer.txt, and
// docs/probabilistic-speculation.md for source mapping and changes.
// The step's sampling fused with its collective (ops.h sample). One block per row: the producer half
// reduces this rank's 32320 logits of the row to a RowCand, the consumer half picks the token from the
// four RowCands in rank order, so every rank picks the same one.
#include <climits>
#include <cmath>
#include <type_traits>

#include "attn/launch.h"
#include "comm/ring_dev.cuh"
#include "common/check.h"
#include "model/ops.h"
#include "model/spec_batch.h"

namespace model::ops {
namespace {

constexpr int kThreads = 1024;
constexpr int kVec = kVocabPart / 4;  // float4 per row
static_assert(kVocabPart % 4 == 0, "vocab part");

struct RowCand {   // one rank's view of one row
  float m;         // max logit
  int32_t mi;      // its lowest id
  float z;         // sum of exp((l - m) / T') over the rank's logits (T' = T, or 1 when greedy)
  float gv;        // Gumbel winner: l / T + g
  int32_t gi;
  float gl;        // its logit
  float t;         // every logit left out of the candidates is <= t (-inf: none)
  int32_t n;       // candidates
  float v[kCand];  // their logits, descending (ties: ascending id)
  int32_t id[kCand];
};
static_assert(sizeof(RowCand) % 16 == 0, "RowCand");
struct ProbRowCand : RowCand {
  float qm, qz, pl, ql;
  int32_t valid, pad[3];
};
struct ResidualCand { float g; int32_t id; float lp; int32_t valid; };
static_assert(sizeof(ProbRowCand) == sizeof(RowCand) + 32 && sizeof(ResidualCand) == 16, "prob row ABI");
constexpr int kHdr = 16;  // rank 0's control word, padded
// a rank's contribution: the control word and a RowCand per block (RB = sample's max_rows: kLowRows or kMaxRows)
template <bool PROB> using SampleCand = std::conditional_t<PROB, ProbRowCand, RowCand>;
template <int RB, bool PROB = false>
constexpr uint32_t kBytes = kHdr + RB * sizeof(SampleCand<PROB>);

__device__ __forceinline__ void wait_prev() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void st_release_sys_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 32;
  return x;
}
// 23-bit midpoints remain representable in float and strictly inside (0,1).
// With 24 bits, the upper midpoint rounds to 1, giving an infinite Gumbel.
__device__ __forceinline__ float unit(uint64_t h) { return ((float)(h >> 41) + 0.5f) * (1.0f / 8388608.0f); }

__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

// Block reductions (fixed order, the result in every thread).
__device__ void block_argmax(float& v, int& i, float* sv, int* si) {
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(ov, oi, v, i)) v = ov, i = oi;
  }
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) sv[w] = v, si[w] = i;
  __syncthreads();
  v = sv[0], i = si[0];
  for (int k = 1; k < nw; ++k)
    if (better(sv[k], si[k], v, i)) v = sv[k], i = si[k];
  __syncthreads();
}
__device__ float block_sum(float v, float* sv) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) sv[w] = v;
  __syncthreads();
  float s = sv[0];
  for (int k = 1; k < nw; ++k) s += sv[k];
  __syncthreads();
  return s;
}
__device__ float block_max(float v, float* sv) {
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) sv[w] = v;
  __syncthreads();
  float s = sv[0];
  for (int k = 1; k < nw; ++k) s = fmaxf(s, sv[k]);
  __syncthreads();
  return s;
}

// Candidate bins: a logit l falls in bin floor((m - l) * 4) when that is below 256 (within 64 of the max).
__device__ __forceinline__ int bin_of(float m, float l) {
  const float d = (m - l) * 4.f;
  return d < 256.f ? (int)d : 256;
}

// RB blocks (one per row of the step's at most RB): the instance of kLowRows is the kernel of before B4, launch
// and bytes alike, the one of kMaxRows serves steps of 9..16 rows.
template <int RB, bool PROB = false>
__global__ void __launch_bounds__(kThreads) k_sample(comm::DevChannel ch, const float* logits, int vocab0,
                                                     const StepIn* in, const uint32_t* ctl, SampleOut* outs,
                                                     int32_t* fb, uint32_t* counter, SpecState* ss,
                                                     const float* qlogits, const spec_sampling::ProposalMeta* qmeta,
                                                     int qslots) {
  using RC = SampleCand<PROB>;
  constexpr int kChunks = sizeof(RC) / 16;
  __shared__ float sv[32];
  __shared__ int si[32];
  __shared__ int hist[257];
  __shared__ int s_n, s_b;
  __shared__ float s_gl;
  __shared__ float cv[kCand];
  __shared__ int ci[kCand];
  __shared__ __align__(16) RC all[kRanks];
  __shared__ __align__(16) ResidualCand residual[kRanks];
  __shared__ float uv[kRanks * kCand];
  __shared__ int ui[kRanks * kCand];
  __shared__ uint32_t s_ctl;

  wait_prev();
  const int b = blockIdx.x, t = threadIdx.x;
  const int R = in->rows;
  const StepSamp sp = in->samp[min(max(b < R ? in->sess[b] : 0, 0), kMaxSess - 1)];  // row b's session's
  const float T = sp.temperature;
  const bool greedy = !(T > 0.f);
  const float invT = greedy ? 1.f : 1.f / T;
  const bool trunc = !greedy && (sp.top_k > 0 || sp.top_p < 1.f);
  const int n_top = min(max(sp.n_top, 0), kTopLogprobs);
  const bool cand = trunc || n_top > 0;
  const int pos = in->pos[b];

  const uint64_t s = comm::next_seq(ch);
  uint8_t* sb = comm::send_buf(ch, s);
  if (b == 0 && t == 0) {
    const uint32_t c = ch.rank == 0 ? *reinterpret_cast<const volatile uint32_t*>(ctl) : 0u;
    *reinterpret_cast<uint4*>(sb) = make_uint4(c, 0, 0, 0);
  }
  RC* mine = reinterpret_cast<RC*>(sb + kHdr) + b;
  if (b < R) {
    const float4* l4 = reinterpret_cast<const float4*>(logits + (size_t)b * kVocabPart);
    // the max and its lowest id; ids ascend within a thread (vector i covers ids 4i .. 4i + 3)
    float m = -INFINITY;
    int mi = INT_MAX;
    for (int i = t; i < kVec; i += kThreads) {
      const float4 q = l4[i];
      const float e[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int c = 0; c < 4; ++c)
        if (e[c] > m) m = e[c], mi = vocab0 + 4 * i + c;
    }
    block_argmax(m, mi, sv, si);
    // partition sum, and the Gumbel-max winner (noise hashed from the seed, position and id)
    float z = 0.f, gv = -INFINITY, gl = 0.f;
    int gi = INT_MAX;
    const uint64_t base = mix64(sp.seed) ^ ((uint64_t)(uint32_t)pos * 0x9E3779B97F4A7C15ull);
    for (int i = t; i < kVec; i += kThreads) {
      const float4 q = l4[i];
      const float e[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        z += expf((e[c] - m) * invT);
        if (!greedy) {
          const int id = vocab0 + 4 * i + c;
          const float g = e[c] * invT - logf(-logf(unit(mix64(base + (uint64_t)id * 0xD1B54A32D192ED03ull))));
          if (g > gv) gv = g, gi = id, gl = e[c];
        }
      }
    }
    z = block_sum(z, sv);
    if (!greedy) {
      const int my = gi;
      const float myl = gl;
      block_argmax(gv, gi, sv, si);
      if (my == gi) s_gl = myl;
    }
    // candidates: the logits of the best bins that together hold at most kCand
    float tcut = -INFINITY;
    int nc = 0;
    if (cand) {
      for (int k = t; k < 257; k += kThreads) hist[k] = 0;
      if (t == 0) s_n = 0;
      __syncthreads();
      for (int i0 = 0; i0 < kVec; i0 += kThreads) {  // whole warps in step (__match_any_sync)
        const int i = i0 + t;
        const float4 q = i < kVec ? l4[i] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        const float e[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int k = bin_of(m, e[c]);
          // warp-aggregated: one atomic per distinct bin of the warp
          const unsigned peers = __match_any_sync(0xffffffffu, k);
          if (k < 256 && (t & 31) == __ffs(peers) - 1) atomicAdd(&hist[k], __popc(peers));
        }
      }
      __syncthreads();
      if (t == 0) {
        int cum = 0, last = -1;
        for (int k = 0; k < 256 && cum + hist[k] <= kCand; ++k) cum += hist[k], last = k;
        s_b = last;
      }
      __syncthreads();
      const int last = s_b;
      float ex = -INFINITY;
      for (int i = t; i < kVec; i += kThreads) {
        const float4 q = l4[i];
        const float e[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (bin_of(m, e[c]) <= last) {
            const int k = atomicAdd(&s_n, 1);
            cv[k] = e[c];
            ci[k] = vocab0 + 4 * i + c;
          } else {
            ex = fmaxf(ex, e[c]);
          }
        }
      }
      tcut = block_max(ex, sv);  // its barriers also publish cv, ci, s_n
      nc = s_n;
      if (t < nc) {  // sort by counting: rank = the candidates that come before
        const float v = cv[t];
        const int id = ci[t];
        int rk = 0;
        for (int k = 0; k < nc; ++k) rk += better(cv[k], ci[k], v, id);
        mine->v[rk] = v;
        mine->id[rk] = id;
      }
    }
    __syncthreads();
    if (t == 0) {
      mine->m = m, mine->mi = mi, mine->z = z;
      mine->gv = gv, mine->gi = gi, mine->gl = greedy ? 0.f : s_gl;
      mine->t = tcut, mine->n = nc;
    }
  } else if (t == 0) {
    mine->m = -INFINITY, mine->mi = INT_MAX, mine->z = 0.f;
    mine->gv = -INFINITY, mine->gi = INT_MAX, mine->gl = 0.f;
    mine->t = -INFINITY, mine->n = 0;
  }
  // q belongs to a real cache slot, not a transient batch entry. It is consumed before
  // this graph's next draft pass overwrites it. Given steps and bonus rows never use q.
  const bool verify_row = b < R && ss && in->spec == kSpecVerify
      && b + 1 < R && in->sess[b + 1] == in->sess[b];
  const float* qrow = nullptr;
  if constexpr (PROB) {
    const int qs = b < R ? in->slot[b] : -1;
    if (verify_row && in->gate != kGateRatio && qmeta && qlogits && qs >= 0 && qs < qslots
        && spec_sampling::matches(qmeta[qs], sp, pos + 1))
      qrow = qlogits + ((size_t)qs * kDrafts + pos + 1 - qmeta[qs].first_pos) * kVocabPart;
    float qm = -INFINITY, qz = 0.f;
    if (qrow) {
      for (int i = t; i < kVocabPart; i += kThreads) qm = fmaxf(qm, qrow[i]);
      qm = block_max(qm, sv);
      // Match p's float4 traversal and within-thread addition order. In particular p == q
      // must not acquire a spurious residual solely because the normalisers sum differently.
      if (isfinite(qm)) {
        const float4* q4 = reinterpret_cast<const float4*>(qrow);
        for (int i = t; i < kVec; i += kThreads) {
          const float4 q = q4[i];
          const float e[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) qz += expf((e[c] - qm) * invT);
        }
      }
      qz = block_sum(qz, sv);
    }
    if (t == 0) {
      mine->qm = qm; mine->qz = qz;
      mine->valid = qrow ? 1 : (verify_row && qmeta && qs >= 0 && qs < qslots && qmeta[qs].valid == -1 ? -1 : 0);
      mine->pl = mine->ql = -INFINITY;
      if (qrow) {
        const int d = in->tokens[b + 1] - vocab0;
        if (d >= 0 && d < kVocabPart) {
          mine->pl = logits[(size_t)b * kVocabPart + d]; mine->ql = qrow[d];
        }
      }
      mine->pad[0] = mine->pad[1] = mine->pad[2] = 0;
    }
  }
  comm::post_grid(ch, s, kBytes<RB, PROB>);
  comm::wait_all(ch, s);
  if (b < R)
    for (int k = t; k < kRanks * kChunks; k += kThreads) {
      const int r = k / kChunks, c = k % kChunks;
      reinterpret_cast<uint4*>(&all[r])[c] = comm::ld_cv16(comm::src(ch, s, r) + kHdr + (size_t)b * sizeof(RC) + 16 * c);
    }
  if (t == 0) s_ctl = comm::ld_cv16(comm::src(ch, s, 0)).x;
  comm::finish_grid(ch, s);  // starts with a barrier: the copies above are done

  SampleOut& o = outs[in->step % kOutSlots];
  if (b < R) {
    // the union of the candidates, sorted (thread k: candidate k)
    int nu = 0;
    float tx = -INFINITY;
    if (cand) {
      int off[kRanks + 1];
      off[0] = 0;
      for (int r = 0; r < kRanks; ++r) off[r + 1] = off[r] + all[r].n, tx = fmaxf(tx, all[r].t);
      nu = off[kRanks];
      if (t < nu) {
        int r = 0;
        while (t >= off[r + 1]) ++r;
        const float v = all[r].v[t - off[r]];
        const int id = all[r].id[t - off[r]];
        int rk = 0;
        for (int q = 0; q < kRanks; ++q)
          for (int j = 0; j < all[q].n; ++j) rk += better(all[q].v[j], all[q].id[j], v, id);
        uv[rk] = v;
        ui[rk] = id;
      }
      __syncthreads();
    }
    if (t == 0) {
      float M = -INFINITY;
      int Mi = INT_MAX;
      for (int r = 0; r < kRanks; ++r)
        if (better(all[r].m, all[r].mi, M, Mi)) M = all[r].m, Mi = all[r].mi;
      float Z = 0.f;
      for (int r = 0; r < kRanks; ++r)
        if (all[r].z > 0.f) Z += all[r].z * expf((all[r].m - M) * invT);
      const float logZ = M * invT + logf(Z);
      int tok = Mi;
      float val = M, lp = M * invT - logZ;
      uint32_t fl = 0;
      if (!greedy && (!trunc || nu == 0)) {  // Gumbel-max over the whole vocabulary
        float g = -INFINITY, gl = 0.f;
        int gi = INT_MAX;
        for (int r = 0; r < kRanks; ++r)
          if (better(all[r].gv, all[r].gi, g, gi)) g = all[r].gv, gi = all[r].gi, gl = all[r].gl;
        tok = gi, val = g, lp = gl * invT - logZ;
        if (trunc) fl |= kSampApprox;
      } else if (trunc) {  // temperature -> top-k -> top-p over the union, one uniform from (seed, position)
        int exact = 0;
        while (exact < nu && uv[exact] > tx) ++exact;
        int K = nu;
        if (sp.top_k > 0 && sp.top_k < K) K = sp.top_k;
        float lz = logZ;
        if (sp.top_k > 0) {  // renormalised over the top k
          float zz = 0.f;
          for (int i = 0; i < K; ++i) zz += expf((uv[i] - uv[0]) * invT);
          lz = uv[0] * invT + logf(zz);
        }
        int keep = K;
        float cum = 0.f;
        for (int i = 0; i < K; ++i) {
          cum += expf(uv[i] * invT - lz);
          if (cum >= sp.top_p) {
            keep = i + 1;
            break;
          }
        }
        if (keep > exact) fl |= kSampApprox;
        float tot = 0.f;
        for (int i = 0; i < keep; ++i) tot += expf(uv[i] * invT - lz);
        const float target = unit(mix64(mix64(sp.seed ^ 0x5851F42D4C957F2Dull) + (uint64_t)(uint32_t)pos)) * tot;
        int pick = keep - 1;
        float acc = 0.f;
        for (int i = 0; i < keep; ++i) {
          acc += expf(uv[i] * invT - lz);
          if (acc > target) {
            pick = i;
            break;
          }
        }
        tok = ui[pick], val = uv[pick], lp = uv[pick] * invT - logZ;
      }
      o.row[b] = SampleRow{tok, lp, val, fl};
      fb[b] = tok;
      for (int i = 0; i < n_top; ++i) {
        o.top_id[b][i] = i < nu ? ui[i] : -1;
        o.top_lp[b][i] = i < nu ? uv[i] * invT - logZ : -INFINITY;
      }
    }
  }
  if constexpr (PROB) {
    __syncthreads();  // original target sample is the exact-match fallback and the bonus
    bool valid = verify_row && qrow;
    float pm = -INFINITY, qm = -INFINITY, pz = 0.f, qz = 0.f;
    float pl = -INFINITY, ql = -INFINITY;
    if (verify_row) {
      for (int r = 0; r < kRanks; ++r) {
        valid = valid && all[r].valid == 1;
        pm = fmaxf(pm, all[r].m); qm = fmaxf(qm, all[r].qm);
        pl = fmaxf(pl, all[r].pl); ql = fmaxf(ql, all[r].ql);
      }
      for (int r = 0; r < kRanks; ++r) {
        if (all[r].z > 0.f) pz += all[r].z * expf((all[r].m - pm) * invT);
        if (all[r].qz > 0.f) qz += all[r].qz * expf((all[r].qm - qm) * invT);
      }
      valid = valid && pz > 0.f && qz > 0.f && isfinite(pz) && isfinite(qz) && isfinite(ql);
    }
    const float lpz = logf(pz), lqz = logf(qz);
    const float lp = (pl - pm) * invT - lpz, lq = (ql - qm) * invT - lqz;
    const bool accepted = valid && logf(spec_sampling::uniform(sp.seed, pos + 1,
        spec_sampling::kAcceptStream)) < fminf(0.f, lp - lq);
    float rg = -INFINITY, rlp = -INFINITY;
    int ri = INT_MAX;
    if (valid && !accepted) {
      for (int i = t; i < kVocabPart; i += kThreads) {
        const float pi = (logits[(size_t)b * kVocabPart + i] - pm) * invT - lpz;
        const float qi = (qrow[i] - qm) * invT - lqz;
        const float lr = spec_sampling::log_positive_difference(pi, qi);
        if (!isfinite(lr)) continue;
        const int id = vocab0 + i;
        const float g = lr - logf(-logf(spec_sampling::uniform(sp.seed, pos + 1,
            spec_sampling::kResidualStream, id)));
        if (better(g, id, rg, ri)) rg = g, ri = id, rlp = pi;
      }
      const int my = ri; const float mylp = rlp;
      block_argmax(rg, ri, sv, si);
      if (my == ri && ri != INT_MAX) s_gl = mylp;
      __syncthreads();
      rlp = ri == INT_MAX ? -INFINITY : s_gl;
    }
    // Fixed collective count across ranks and batch layouts, including fallback/bonus rows.
    // finish_grid(s) is not a grid barrier: use the known successor s+1 (not next_seq).
    // wait_all(s+1) proves all blocks have finished consuming s before this finish_grid.
    const uint64_t rs = s + 1;
    auto* rm = reinterpret_cast<ResidualCand*>(comm::send_buf(ch, rs)) + b;
    if (t == 0) *rm = ResidualCand{rg, ri, rlp, ri != INT_MAX};
    comm::post_grid(ch, rs, RB * sizeof(ResidualCand));
    comm::wait_all(ch, rs);
    if (t < kRanks)
      reinterpret_cast<uint4*>(residual)[t] = comm::ld_cv16(comm::src(ch, rs, t) + b * sizeof(ResidualCand));
    comm::finish_grid(ch, rs);
    if (t == 0 && verify_row) {
      if (!valid) {
        o.row[b].flags |= spec_sampling::eligible(sp) && in->gate != kGateRatio ? kSampProbFallback : kSampProbSkipped;
        // Independent target draw + equality is still exact. Unsupported requests retain greedy proposals.
      } else if (accepted) {
        o.row[b] = SampleRow{in->tokens[b + 1], lp, pl, kSampProb};
        fb[b] = in->tokens[b + 1];
      } else {
        ResidualCand win{-INFINITY, INT_MAX, -INFINITY, 0};
        for (int r = 0; r < kRanks; ++r)
          if (residual[r].valid && better(residual[r].g, residual[r].id, win.g, win.id)) win = residual[r];
        if (win.valid) {
          o.row[b] = SampleRow{win.id, win.lp, win.g, kSampProb | kSampProbReject};
          fb[b] = win.id;
        } else {
          // An impossible mathematical event (reject with no positive residual) is observable.
          // Keep a finite target token but force rejection; qualification requires zero such rows.
          o.row[b].flags |= kSampProb | kSampProbReject | kSampResidualZero;
        }
      }
    }
  }
  // the last block to finish publishes the step
  __threadfence_system();
  __syncthreads();
  if (t == 0 && atomicAdd(counter, 1u) == gridDim.x - 1) {
    *counter = 0;
    __threadfence_system();
    o.ctl = s_ctl;
    int nt = 0;  // the most of any session's
    for (int i = 0; i < min(in->nsess, kMaxSess); ++i) nt = max(nt, min(max(in->samp[i].n_top, 0), kTopLogprobs));
    o.n_top = nt;
    const int spec = ss ? in->spec : kSpecOff;
    if (spec == kSpecOff) {
      o.rows = R;
      __threadfence_system();
      st_release_sys_u32(&o.seq, in->step + 1);
      return;
    }
    const volatile int32_t* vfb = fb;
    if (in->sbatch) {
      // A batch spec step (docs/batching.md, B3): the verdict per entry over its own rows, each by exactly the
      // one-session rule below (spec_batch.h; no row reads another entry's). Entry k's rows are consecutive,
      // sess[r] = k; the step is published after the entries' draft passes (spec.cu, k_spec_publish_batch).
      SpecVerdict v[kMaxSess];
      spec_batch_verdicts(*in, vfb, spec == kSpecVerify, v, o.row);
      for (int k = 0; k < kMaxSess; ++k) {
        o.sess_row0[k] = v[k].row0, o.sess_step_rows[k] = v[k].step_rows, o.sess_rows[k] = v[k].rows;
        o.sess_pos0[k] = v[k].pos0;
        o.next_srows[k] = 0, o.next_spos0[k] = 0;
        if (!v[k].step_rows) continue;
        ss[k].count = v[k].rows;
        ss[k].pos0 = v[k].pos0;
        ss[k].bonus = in->force >= 0 ? in->force : v[k].bonus;
      }
      o.rows = R;
      o.pos0 = in->pos[0];
      o.step_rows = R;
      return;
    }
    // A spec step's final rows: a verify step's up to the first draft that is not the sample of the row
    // before it (row[] then holds exactly the tokens it emits), and the token after them. Published
    // after the draft pass (spec.cu).
    int n = R - 1;
    if (spec == kSpecVerify) {
      n = 0;
      while (n < R - 1 && spec_row_accepted(vfb[n], in->tokens[n + 1], o.row[n].flags)) ++n;
    }
    o.rows = n + 1;
    o.pos0 = in->pos[0];
    o.step_rows = R;
    ss->count = n + 1;
    ss->pos0 = in->pos[0];
    ss->bonus = in->force >= 0 ? in->force : vfb[n];
  }
}

}  // namespace

template <int RB, bool PROB>
void launch_sample(const comm::DevChannel& ch, const float* logits, int vocab0, const StepIn* in,
                   const uint32_t* ctl, SampleOut* outs, int32_t* fb, uint32_t* counter, cudaStream_t s,
                   SpecState* ss, const float* q, const spec_sampling::ProposalMeta* qm, int qs) {
  if (kBytes<RB, PROB> > ch.cap) die("sample: %u bytes exceed the channel's %u", kBytes<RB, PROB>, ch.cap);
  attn::launch(k_sample<RB, PROB>, dim3(RB), dim3(kThreads), 0, s, true,
               ch, logits, vocab0, in, ctl, outs, fb, counter, ss, q, qm, qs);
}
void sample(const comm::DevChannel& ch, const float* logits, int vocab0, const StepIn* in, const uint32_t* ctl,
            SampleOut* outs, int32_t* fb, uint32_t* counter, cudaStream_t s, SpecState* ss, int max_rows,
            const float* q, const spec_sampling::ProposalMeta* qm, int qs) {
  if ((q == nullptr) != (qm == nullptr) || (q && qs < 1)) die("sample: incomplete proposal storage");
  if (max_rows == kLowRows) {
    if (q) launch_sample<kLowRows, true>(ch, logits, vocab0, in, ctl, outs, fb, counter, s, ss, q, qm, qs);
    else launch_sample<kLowRows, false>(ch, logits, vocab0, in, ctl, outs, fb, counter, s, ss, q, qm, qs);
  } else if (max_rows == kMaxRows) {
    if (q) launch_sample<kMaxRows, true>(ch, logits, vocab0, in, ctl, outs, fb, counter, s, ss, q, qm, qs);
    else launch_sample<kMaxRows, false>(ch, logits, vocab0, in, ctl, outs, fb, counter, s, ss, q, qm, qs);
  } else die("sample: max_rows %d (kLowRows %d or kMaxRows %d)", max_rows, kLowRows, kMaxRows);
}

}  // namespace model::ops
