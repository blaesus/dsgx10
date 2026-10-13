// The step's sampling fused with its collective (ops.h sample): the DeepSeek engine's sampler
// (ds41f-4xgx10-switchless/engine/model/sample.cu) with GLM's vocabulary split (38,720 logits a rank), the
// padding ids masked, and without its speculative half (G7). One block per row: the producer half reduces this
// rank's logits of the row to a RowCand, the consumer half picks the token from the four RowCands in rank order,
// so every rank picks the same one.
#include <climits>
#include <cmath>

#include "attn/launch.h"
#include "comm/ring_dev.cuh"
#include "common/check.h"
#include "model/ops.h"

namespace glm::model::ops {
namespace {

constexpr int kLowRows = 8;  // steps of at most 8 rows launch 8 blocks
using ::glm::model::kRanks;

constexpr int kThreads = 1024;
constexpr int kVocabPart = kVocabRank;  // 38,720 logits a rank
constexpr int kVec = kVocabPart / 4;  // float4 per row
static_assert(kVocabPart % 4 == 0, "vocab part");
constexpr int kCand = 64;  // candidates a rank and row for top-k / top-p / top logprobs

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
constexpr int kHdr = 16;  // a header word, padded (unused)
// a rank's contribution: the control word and a RowCand per block (RB = sample's max_rows: kLowRows or kMaxRows)
template <int RB>
constexpr uint32_t kBytes = kHdr + RB * sizeof(RowCand);
constexpr int kChunks = sizeof(RowCand) / 16;

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
__device__ __forceinline__ float unit(uint64_t h) { return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f); }

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
template <int RB>
__global__ void __launch_bounds__(kThreads) k_sample(comm::DevChannel ch, const float* logits_, int vocab0,
                                                     const StepIn* in, SampleOut* outs, int32_t* fb, uint32_t* counter,
                                                     int publish) {
  __shared__ float sv[32];
  __shared__ int si[32];
  __shared__ int hist[257];
  __shared__ int s_n, s_b;
  __shared__ float s_gl;
  __shared__ float cv[kCand];
  __shared__ int ci[kCand];
  __shared__ __align__(16) RowCand all[kRanks];
  __shared__ float uv[kRanks * kCand];
  __shared__ int ui[kRanks * kCand];

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
  if (b == 0 && t == 0) *reinterpret_cast<uint4*>(sb) = make_uint4(0, 0, 0, 0);
  RowCand* mine = reinterpret_cast<RowCand*>(sb + kHdr) + b;
  if (b < R) {
    // ids >= kVocabUsed (the head's padding rows, rank 3's last 24) never win: -inf
    const float4* l4raw = reinterpret_cast<const float4*>(logits_ + (size_t)b * kVocabPart);
    auto ld4 = [&](int i) {
      float4 q = l4raw[i];
      const int id = vocab0 + 4 * i;
      if (id + 3 >= kVocabUsed) {
        if (id + 0 >= kVocabUsed) q.x = -INFINITY;
        if (id + 1 >= kVocabUsed) q.y = -INFINITY;
        if (id + 2 >= kVocabUsed) q.z = -INFINITY;
        if (id + 3 >= kVocabUsed) q.w = -INFINITY;
      }
      return q;
    };
    // the max and its lowest id; ids ascend within a thread (vector i covers ids 4i .. 4i + 3)
    float m = -INFINITY;
    int mi = INT_MAX;
    for (int i = t; i < kVec; i += kThreads) {
      const float4 q = ld4(i);
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
      const float4 q = ld4(i);
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
        const float4 q = i < kVec ? ld4(i) : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
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
        const float4 q = ld4(i);
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
  comm::post_grid(ch, s, kBytes<RB>);
  comm::wait_all(ch, s);
  if (b < R)
    for (int k = t; k < kRanks * kChunks; k += kThreads) {
      const int r = k / kChunks, c = k % kChunks;
      reinterpret_cast<uint4*>(&all[r])[c] = comm::ld_cv16(comm::src(ch, s, r) + kHdr + (size_t)b * sizeof(RowCand) + 16 * c);
    }
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
        if (trunc) fl |= 1u;
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
        if (keep > exact) fl |= 1u;
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
  // the last block to finish publishes the step (or leaves its seq to ops::publish, after the draft passes)
  __threadfence_system();
  __syncthreads();
  if (t == 0 && atomicAdd(counter, 1u) == gridDim.x - 1) {
    *counter = 0;
    int nt = 0;  // the most of any session's
    for (int i = 0; i < min(in->nsess, kMaxSess); ++i) nt = max(nt, min(max(in->samp[i].n_top, 0), kTopLogprobs));
    o.n_top = nt;
    o.rows = R;
    o.drafts = 0;
    if (publish) {
      __threadfence_system();
      st_release_sys_u32(&o.seq, in->step + 1);
    }
  }
}

}  // namespace

void sample(const comm::DevChannel& ch, const float* logits, int vocab0, const StepIn* in, SampleOut* outs, int32_t* fb,
            uint32_t* counter, cudaStream_t s, bool publish, int max_rows) {
  if (kBytes<kMaxRows> > ch.cap) die("sample: %u bytes exceed the channel's %u", kBytes<kMaxRows>, ch.cap);
  if (max_rows <= kLowRows)
    ::glm::launch(k_sample<kLowRows>, dim3(kLowRows), dim3(kThreads), 0, s, true, ch, logits, vocab0, in, outs, fb,
                  counter, publish ? 1 : 0);
  else
    ::glm::launch(k_sample<kMaxRows>, dim3(kMaxRows), dim3(kThreads), 0, s, true, ch, logits, vocab0, in, outs, fb,
                  counter, publish ? 1 : 0);
}

}  // namespace glm::model::ops
