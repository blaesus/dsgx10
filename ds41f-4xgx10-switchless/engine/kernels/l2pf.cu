// The decode step's L2 prefetch (model::Options::l2pf; docs/engine.md, "A paced L2 prefetch in the step's windows"):
// while a collective, or the attention, leaves DRAM idle, one warp on a side stream issues
// cp.async.bulk.prefetch.L2 of the weights the step reads next, 8 KiB chunks at a token bucket's pace
// (TensorFold's paced one-block prefetch, as sudoingX/dgx-spark-deepseek-v41's exl3_l2pf.cu runs it;
// Apache-2.0, this code is ours). It writes nothing the model reads, so no output changes.
#include <algorithm>

#include "common/check.h"
#include "kernels/dsk.h"

namespace dsk {
namespace {

constexpr uint32_t kChunk = 8192;
constexpr int kRanges = 16;
constexpr float kGbps = 200;  // the pace (micro/l2_retention's lat test: no fill cost at ~175 in small bursts)

struct Queue {  // a layer's planned weights in the order the step reads them
  const uint8_t* p[kRanges];
  uint32_t n[kRanges];    // bytes (multiples of 16)
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
// due every ns_chunk ns (a poll issues those due, up to 32). It ends early once the window's collective has arrived
// (the next kernel is about to start): *arrived changed since the start, or is ahead of *finished (the collectives
// run one at a time, so the one in flight has arrived, perhaps before this started). It leaves where it ended in
// *state.
__global__ void __launch_bounds__(32) k_l2pf(const __grid_constant__ Queue q, int layer, uint32_t first,
                                             uint32_t budget, uint32_t ns_chunk, const unsigned long long* arrived,
                                             const unsigned long long* finished, L2pfState* state) {
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
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(q.p[i] + off), "r"(min(left, kChunk))
                   : "memory");
    }
    if (due > done) done = due;
    else __nanosleep(64);
  }
  if (lane == 0) state->layer = layer, state->chunk = done;
}

}  // namespace

void l2pf_window(const L2pfTarget (&t)[kL2pfTargets], int layer, int window, int R, const unsigned long long* arrived,
                 const unsigned long long* finished, L2pfState* state, cudaStream_t s) {
  if (!arrived != !finished) die("l2pf_window: arrived and finished go together");
  if (window < 1 || window > 4) die("l2pf_window: window %d", window);
  // the plan: wo_a, wo_b and the router gate; at one row wq_b too, whose persistent kernel gains from L2 (the
  // classic one, from two rows, does not)
  const unsigned plan = (R == 1 ? 1u << kL2pfWqb : 0u) | 1u << kL2pfWoa | 1u << kL2pfWob | 1u << kL2pfGate;
  static const int kFrom[5] = {0, kL2pfWqa, kL2pfWqb, kL2pfWoa, kL2pfGate};  // a window's first target not read
  Queue q{};
  uint32_t c = 0, first = 0;
  for (int i = 0; i < kL2pfTargets; ++i) {
    if (!(plan >> i & 1)) continue;
    for (int j = 0; j < 4; ++j) {
      const size_t n = t[i].n[j] & ~(size_t)15;
      if (!t[i].p[j] || n < 16) continue;
      if (q.count == kRanges) die("l2pf_window: more than %d ranges", kRanges);
      q.p[q.count] = static_cast<const uint8_t*>(t[i].p[j]);
      q.n[q.count] = (uint32_t)n;
      c += (uint32_t)((n + kChunk - 1) / kChunk);
      q.end[q.count++] = c;
    }
    if (i < kFrom[window]) first = c;
  }
  if (first >= c) return;
  // the budget: the window's time at the pace. The collectives' at about their 75th percentile on rank 0 of the
  // engine's 1k decode (a stop ends them earlier); the attention's at about its 5th (nothing does).
  const int r = std::min(R, 8);
  const bool idx = t[kL2pfWqb].n[2] > 0;
  const float us = window == 1   ? 18 + 5.1f * (r - 1)
                   : window == 2 ? 18
                   : window == 3 ? (idx ? 30 : 11)
                                 : 14 + 1.2f * (r - 1);
  const uint32_t ns_chunk = (uint32_t)(kChunk / kGbps + 0.5f), budget = (uint32_t)(us * 1000 / ns_chunk);
  k_l2pf<<<1, 32, 0, s>>>(q, layer, first, budget, ns_chunk, arrived, finished, state);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace dsk
