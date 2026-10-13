// The two-chunk wavefront's probe (dsgb10_rank --wave-probe MASK --wave-probe-at POS; docs/engine.md
// "What remains for prefill"): at layer L of the chunk at POS, layer L's FFN beside a second attention
// side of layer L on another stream, its collectives on a second ring, against the two in sequence.
// The probe treats x, mid, y, yraw, sub, the mixes and yq / ys as one chunk's activations (ActSet) and
// everything else in Prefill as one side's scratch: a buffer both attention_p and ffn_p come to touch
// belongs in ActSet, or the timings run on garbage.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "model/prefill.h"

namespace model {

using namespace detail;

namespace {
// The buffers both a chunk's attention side and another chunk's FFN touch: one chunk's activations.
struct ActSet {
  bf16 *x, *mid, *y, *yraw, *sub;
  float *a_pre, *a_post, *a_comb, *pm, *f_post, *f_comb;
  uint8_t *yq, *ys;
};
ActSet act_get(const Prefill& p) {
  return {p.x, p.mid, p.y, p.yraw, p.sub, p.a_pre, p.a_post, p.a_comb, p.pm, p.f_post, p.f_comb, p.yq, p.ys};
}
void act_put(Prefill& p, const ActSet& s) {
  p.x = s.x, p.mid = s.mid, p.y = s.y, p.yraw = s.yraw, p.sub = s.sub, p.a_pre = s.a_pre, p.a_post = s.a_post;
  p.a_comb = s.a_comb, p.pm = s.pm, p.f_post = s.f_post, p.f_comb = s.f_comb, p.yq = s.yq, p.ys = s.ys;
}
ActSet act_alloc(int Q) {
  const size_t M4 = 4 * (size_t)Q;
  ActSet s;
  s.x = dalloc<bf16>((size_t)Q * kHcDim), s.mid = dalloc<bf16>((size_t)Q * kHcDim);
  s.y = dalloc<bf16>((size_t)Q * kDim), s.yraw = dalloc<bf16>((size_t)Q * kDim), s.sub = dalloc<bf16>((size_t)Q * kDim);
  s.a_pre = dalloc<float>((size_t)Q * kHc), s.a_post = dalloc<float>((size_t)Q * kHc);
  s.a_comb = dalloc<float>((size_t)Q * kHc * kHc), s.pm = dalloc<float>((size_t)Q * kHc);
  s.f_post = dalloc<float>((size_t)Q * kHc), s.f_comb = dalloc<float>((size_t)Q * kHc * kHc);
  s.yq = dalloc<uint8_t>(M4 * kDim), s.ys = dalloc<uint8_t>(M4 * kDim / 32);
  return s;
}
void act_copy(const ActSet& d, const ActSet& s, int Q, cudaStream_t st) {
  const size_t M4 = 4 * (size_t)Q;
  auto cp = [&](void* dst, const void* src, size_t b) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, b, cudaMemcpyDeviceToDevice, st));
  };
  cp(d.x, s.x, (size_t)Q * kHcDim * 2), cp(d.mid, s.mid, (size_t)Q * kHcDim * 2);
  cp(d.y, s.y, (size_t)Q * kDim * 2), cp(d.yraw, s.yraw, (size_t)Q * kDim * 2), cp(d.sub, s.sub, (size_t)Q * kDim * 2);
  cp(d.a_pre, s.a_pre, (size_t)Q * kHc * 4), cp(d.a_post, s.a_post, (size_t)Q * kHc * 4);
  cp(d.a_comb, s.a_comb, (size_t)Q * kHc * kHc * 4), cp(d.pm, s.pm, (size_t)Q * kHc * 4);
  cp(d.f_post, s.f_post, (size_t)Q * kHc * 4), cp(d.f_comb, s.f_comb, (size_t)Q * kHc * kHc * 4);
  cp(d.yq, s.yq, M4 * kDim), cp(d.ys, s.ys, M4 * kDim / 32);
}
struct Lane {
  cudaStream_t st, cs;
  cudaEvent_t grp[kColGroups], ev_cs, end;
};
}  // namespace

// Layer L's FFN of this chunk beside a second attention side of layer L: a stand-in for the next
// chunk's, on the same positions, from a copy of this chunk's activations (its input the FFN's normed
// input). Each on a lane of its own (main stream, comm stream; the attention's collectives on the second
// ring), against the two in sequence. Every rank issues the same sequence on each channel. The stand-ins
// rewrite layer L's keys at these positions: the run's outputs are not the model's.
void Model::Impl::wave_probe(int L, int a, int M) {
  if (!has_ch2) die("model: --wave-probe without the second ring");
  Prefill& p = *pf;
  static bool init = false;  // the probe's buffers and streams: made on first use, kept
  static ActSet saved, other;
  static Lane lane[4];  // FFN at normal priority, FFN higher, attention normal, attention higher
  static cudaEvent_t t0, t1;
  if (!init) {
    init = true;
    saved = act_alloc(p.Q), other = act_alloc(p.Q);
    int lo = 0, hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    for (int i = 0; i < 4; ++i) {
      Lane& l = lane[i];
      CUDA_CHECK(cudaStreamCreateWithPriority(&l.st, cudaStreamNonBlocking, i & 1 ? (lo + hi) / 2 : lo));
      CUDA_CHECK(cudaStreamCreateWithPriority(&l.cs, cudaStreamNonBlocking, hi));
      for (auto& e : l.grp) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&l.ev_cs, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&l.end, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    std::printf("rank %d: wave probe: priorities %d (normal) %d (higher) of [%d, %d]\n", rank, lo, (lo + hi) / 2, lo, hi);
  }
  const ActSet live = act_get(p);
  cudaStream_t const st0 = st;
  act_copy(saved, live, p.Q, st0);
  // a lane's streams and events in place of the main ones while its work is issued
  auto enter = [&](Lane& l) {
    std::swap(st, l.st), std::swap(p.cs, l.cs), std::swap(p.ev_cs, l.ev_cs);
    for (int c = 0; c < kColGroups; ++c) std::swap(p.ev_grp[c], l.grp[c]);
  };
  auto ffn = [&](Lane& l) {
    enter(l);
    ffn_p(L, M);
    enter(l);
  };
  auto att = [&](Lane& l) {
    enter(l);
    std::swap(ch, ch2);
    act_put(p, other);
    attention_p(L, a, M, {});
    act_put(p, live);
    std::swap(ch, ch2);
    enter(l);
  };
  // configurations: 0 FFN alone, 1 attention alone, 2 FFN then attention, 3 both at once (equal
  // priorities), 4 both, the FFN's main stream higher, 5 both, the attention's higher
  constexpr int kCfg = 6, kRounds = 8;
  float ms[kCfg][kRounds];
  for (int r = 0; r < kRounds; ++r)
    for (int c = 0; c < kCfg; ++c) {
      act_copy(live, saved, p.Q, st0);
      act_copy(other, saved, p.Q, st0);
      CUDA_CHECK(cudaEventRecord(t0, st0));
      Lane& lf = lane[c == 4 ? 1 : 0];
      Lane& la = lane[c == 5 ? 3 : 2];
      CUDA_CHECK(cudaStreamWaitEvent(lf.st, t0, 0));
      CUDA_CHECK(cudaStreamWaitEvent(la.st, t0, 0));
      if (c != 1) ffn(lf);
      CUDA_CHECK(cudaEventRecord(lf.end, lf.st));
      if (c == 2) CUDA_CHECK(cudaStreamWaitEvent(la.st, lf.end, 0));
      if (c != 0) att(la);
      CUDA_CHECK(cudaEventRecord(la.end, la.st));
      CUDA_CHECK(cudaStreamWaitEvent(st0, lf.end, 0));
      CUDA_CHECK(cudaStreamWaitEvent(st0, la.end, 0));
      CUDA_CHECK(cudaEventRecord(t1, st0));
      CUDA_CHECK(cudaEventSynchronize(t1));
      CUDA_CHECK(cudaEventElapsedTime(&ms[c][r], t0, t1));
    }
  act_copy(live, saved, p.Q, st0);
  float med[kCfg];
  for (int c = 0; c < kCfg; ++c) {
    std::vector<float> v(ms[c] + 1, ms[c] + kRounds);  // round 0 warms up
    std::sort(v.begin(), v.end());
    med[c] = v[v.size() / 2];
  }
  std::printf("rank %d: wave probe L=%d M=%d a=%d: ffn %.2f attn %.2f seq %.2f conc %.2f (ffn hi %.2f, attn hi %.2f) ms;"
              " conc/seq %.3f %.3f %.3f\n",
              rank, L, M, a, med[0], med[1], med[2], med[3], med[4], med[5], med[3] / med[2], med[4] / med[2],
              med[5] / med[2]);
  std::fflush(stdout);
}

}  // namespace model
