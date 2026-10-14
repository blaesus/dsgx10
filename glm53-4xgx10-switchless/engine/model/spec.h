// The draft gate of speculative decoding (docs/engine.md, "Drafts"): how many of a session's drafts its next verify
// step feeds. Host code, the same on every rank: each rank updates it from the same SampleOuts with the same float
// operations, so every rank makes the same choice. Used by the server's spec loop (engine/server/cmd.cpp) and the
// driver's spec mode (tools/glm_run.cu).
#pragma once
#include <algorithm>

#include "model/model.h"

namespace glm::model {

// The modelled cost of a verify step of one session, ms: c1 for its first row with draft pass 1, drow for each row
// more with its chained draft pass. The defaults are the 2026-10-14 measurements at split-K 16 (docs/engine.md).
struct SpecCost {
  float c1 = 44.5f;
  float drow = 15.5f;
};

// Per session: p[j], an exponential average (1/8) of the acceptance of draft j + 1 given that the drafts before it were
// accepted. The next step verifies the k (0 .. kmax, at most the drafts there are) that maximises the expected tokens
// a step, 1 + p0 + p0 p1 + ..., over the step's modelled cost c1 + k drow. With k = 0 the step still runs pass 1, so
// draft 1 is judged anyway (against the step's sample) and p0 keeps moving: the gate opens again when it would pay.
struct SpecGate {
  float p[kMaxDrafts];
  int kmax = 3;
  SpecGate() : SpecGate(3) {}
  explicit SpecGate(int kmax_, float p0 = 0.6f) : kmax(kmax_) { std::fill(p, p + kMaxDrafts, p0); }
  int choose(const SpecCost& c, int avail) const {
    float cum = 1.f, e = 1.f, best = 1.f / c.c1;
    int bk = 0;
    for (int k = 1; k <= std::min(kmax, avail); ++k) {
      cum *= p[k - 1];
      e += cum;
      const float v = e / (c.c1 + (float)k * c.drow);
      if (v > best) best = v, bk = k;
    }
    return bk;
  }
  // after a verify of k drafts of which a were accepted; with k == 0, hit: whether the step's draft 1 equalled the
  // sample (-1: no draft to judge)
  void update(int k, int a, int hit) {
    constexpr float b = 1.f / 8;
    if (k == 0) {
      if (hit >= 0) p[0] += b * ((float)hit - p[0]);
      return;
    }
    for (int j = 0; j < k && j <= a; ++j) p[j] += b * ((j < a ? 1.f : 0.f) - p[j]);
  }
};

// Tests' oracle drafts (glm_run spec's k = -3, the server's --spec-oracle): the draft of token n (counted from the
// run's first sample) is changed when this says so, about one in four, by a hash of n (so the verify steps accept
// runs of every length 0 .. k).
inline bool oracle_changed(uint32_t n) {
  n ^= n >> 16, n *= 0x7feb352du, n ^= n >> 15, n *= 0x846ca68bu, n ^= n >> 16;
  return (n & 3u) == 0;
}

}  // namespace glm::model
