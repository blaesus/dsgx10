// The decode step's paced L2 prefetch (Options::l2pf; docs/engine.md, "The paced L2 prefetch"): the DeepSeek
// engine's design (ds41f-4xgx10-switchless/engine/kernels/l2pf.cu). While a collective, or the attention, leaves DRAM
// idle, one warp on a side stream issues cp.async.bulk.prefetch.L2 of the weights the step reads next, 8 KiB chunks
// at a token bucket's pace. It writes nothing the model reads, so no output changes.
#include <algorithm>

#include "common/check.h"
#include "model/impl.h"

namespace glm::model::detail {
namespace {

constexpr uint32_t kChunk = 8192;
constexpr int kRanges = 4 * kPfTargets;
constexpr float kGbps = 200;  // the pace (the DeepSeek engine's: no fill cost at ~175 in small bursts)

struct Queue {
  const uint8_t* p[kRanges];
  uint32_t n[kRanges];
  uint32_t end[kRanges];  // chunks up to and including range i
  int count;
};

__device__ __forceinline__ unsigned long long ld_volatile(const unsigned long long* p) {
  return *(volatile const unsigned long long*)p;
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// One warp: from chunk max(first, where the layer's earlier windows ended) of the queue, up to budget chunks, one
// due every ns_chunk ns. It ends early once the window's collective has arrived (*arrived changed since the start,
// or is ahead of *finished), and leaves where it ended in *state.
__global__ void __launch_bounds__(32) k_l2pf(const __grid_constant__ Queue q, int layer, uint32_t first, uint32_t budget,
                                             uint32_t ns_chunk, const unsigned long long* arrived,
                                             const unsigned long long* finished, PfState* state) {
  const int lane = threadIdx.x;
  uint32_t c0 = 0;
  unsigned long long a0 = 0;
  if (lane == 0) {
    if (state->layer == layer) c0 = state->chunk;
    if (arrived) a0 = ld_volatile(arrived);
  }
  c0 = max(__shfl_sync(~0u, c0, 0), first);
  const uint32_t total = q.count ? q.end[q.count - 1] : 0, last = min(total, c0 + budget);
  const unsigned long long t0 = __shfl_sync(~0u, gtimer(), 0);
  uint32_t done = min(c0, last);
  while (done < last) {
    unsigned long long t = 0;
    int quit = 0;
    if (lane == 0) {
      t = gtimer();
      if (arrived) {
        const unsigned long long a = ld_volatile(arrived);
        quit = a != a0 || a > ld_volatile(finished);
      }
    }
    if (__shfl_sync(~0u, quit, 0)) break;
    t = __shfl_sync(~0u, t, 0);
    const unsigned long long by_now = c0 + (t - t0) / ns_chunk + 1;
    const uint32_t cap = min(last, done + 32), due = by_now < cap ? (uint32_t)by_now : cap;
    const uint32_t c = done + lane;
    if (c < due) {
      int i = 0;
      while (c >= q.end[i]) ++i;
      const uint32_t off = (c - (i ? q.end[i - 1] : 0)) * kChunk, left = q.n[i] - off;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(q.p[i] + off), "r"(min(left, kChunk)) : "memory");
    }
    if (due > done) done = due;
    else __nanosleep(64);
  }
  if (lane == 0) state->layer = layer, state->chunk = done;
}

}  // namespace

void l2pf_window(const PfTarget (&t)[kPfTargets], int layer, int window, float budget_us,
                 const unsigned long long* arrived, const unsigned long long* finished, PfState* state, cudaStream_t s) {
  if (!arrived != !finished) die("l2pf_window: arrived and finished go together");
  // a window's first target not yet read by the step: 1 (before the layer) the x-projection; 2 (its all-gather)
  // q_b; 3 (the attention) W_UV; 4 (the o_proj all-reduce) the FFN's first weights
  static const int kFrom[5] = {0, kPfXproj, kPfQb, kPfUv, kPfFfn0};
  Queue q{};
  uint32_t c = 0, first = 0;
  for (int i = 0; i < kPfTargets; ++i) {
    if (i == kPfFfn0 && window < 4) break;  // the FFN's weights only in their own window: they would push out the attention's
    for (int j = 0; j < 4; ++j) {
      const size_t n = t[i].n[j] & ~(size_t)15;
      if (!t[i].p[j] || n < 16) continue;
      q.p[q.count] = static_cast<const uint8_t*>(t[i].p[j]);
      q.n[q.count] = (uint32_t)n;
      c += (uint32_t)((n + kChunk - 1) / kChunk);
      q.end[q.count++] = c;
    }
    if (i < kFrom[window]) first = c;
  }
  if (first >= c || budget_us <= 0) return;
  const uint32_t ns_chunk = (uint32_t)(kChunk / kGbps + 0.5f), budget = (uint32_t)(budget_us * 1000 / ns_chunk);
  k_l2pf<<<1, 32, 0, s>>>(q, layer, first, budget, ns_chunk, arrived, finished, state);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace glm::model::detail
