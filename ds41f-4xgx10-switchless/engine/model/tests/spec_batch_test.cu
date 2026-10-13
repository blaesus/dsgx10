// B3 (docs/batching.md): spec_batch.h's layout and verdicts against a host reference, on the host (no
// GPU; the CMake target spec_batch_test). Random entries on and off, random rows per entry within the
// budget, random samples with matches planted: the device's layout of a verify step (k_spec_in_batch)
// and the verdicts per entry (k_sample) must equal what each entry gets in a step of its own under the
// one-session rule; a given step's row0 and verdicts too.
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "model/spec_batch.h"

using namespace model;
using namespace model::ops;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33);
}
int fails = 0, checks = 0;
void check(bool ok, const char* what, int it) {
  ++checks;
  if (ok) return;
  if (++fails <= 20) std::printf("FAIL (iteration %d): %s\n", it, what);
}

// the one-session rule for a step of R rows: fb[r] row r's sample, tok[r] what it was fed; the rows it
// emits (sample.cu's k_sample for one session)
int lone_rows(int R, const int32_t* fb, const int32_t* tok, bool verify) {
  int n = R - 1;
  if (verify) {
    n = 0;
    while (n < R - 1 && fb[n] == tok[n + 1]) ++n;
  }
  return n + 1;
}

}  // namespace

int main() {
  constexpr int kIters = 20000;
  for (int it = 0; it < kIters; ++it) {
    StepIn d{};
    SpecState ss[kMaxSess]{};
    const int ns = 1 + (int)(rnd() % kMaxSess);
    d.nsess = ns;
    d.spec = kSpecVerify;
    d.sbatch = 1;
    // the entries: on or off (one on at least), each in a slot; their rows within the host's cap
    int S = 0;
    for (int k = 0; k < ns; ++k) {
      d.sess_on[k] = rnd() % 4 != 0;
      d.sess_slot[k] = (k * 3 + 1) % kMaxSlots;
      S += d.sess_on[k];
    }
    if (!S) d.sess_on[rnd() % ns] = 1, S = 1;
    const int cap = std::min(kDrafts, (kMaxRows - S) / S);
    int want_R = 0, row0_ref[kMaxSess] = {};
    for (int k = 0; k < ns; ++k) {
      ss[k].next_rows = 1 + (int)(rnd() % (cap + 1));
      ss[k].next_pos0 = 100 * (k + 1) + (int)(rnd() % 50);
      for (int j = 0; j < kMaxRows; ++j) ss[k].next_tok[j] = 1000 * (k + 1) + j;
      row0_ref[k] = want_R;
      if (d.sess_on[k]) want_R += ss[k].next_rows;
    }
    // the verify step's layout
    int row0[kMaxSess];
    const int R = spec_batch_layout(d, ss, row0);
    check(R == want_R && d.rows == R && R >= 1 && R <= kMaxRows, "layout: the row count is the entries' sum", it);
    for (int k = 0; k < ns; ++k) {
      check(row0[k] == row0_ref[k], "layout: row0", it);
      if (!d.sess_on[k]) continue;
      for (int j = 0; j < ss[k].next_rows; ++j) {
        const int r = row0[k] + j;
        check(d.tokens[r] == ss[k].next_tok[j] && d.pos[r] == ss[k].next_pos0 + j && d.slot[r] == d.sess_slot[k] &&
                  d.sess[r] == k,
              "layout: an entry's rows (its tokens, positions, slot and entry index)", it);
      }
    }
    for (int r = R; r < kMaxRows; ++r)
      check(d.tokens[r] == 0 && d.pos[r] == d.pos[R - 1] && d.slot[r] == d.slot[R - 1] && d.sess[r] == d.sess[R - 1],
            "layout: the entries past R repeat the last row's", it);
    // B3 phase 2a: the draft passes' compact list and slot tables (k_spec_in_batch, the switch over the entries on)
    int bent[kMaxSess], pstab[kMaxSess * kDrafts];
    const int Sp = spec_batch_passes(d, bent, pstab);
    check(Sp == S, "passes: the count is the entries on", it);
    int c = 0;
    for (int k = 0; k < ns; ++k) {
      if (!d.sess_on[k]) continue;
      check(c < Sp && bent[c] == k, "passes: the compact list is the entries on, in order", it);
      for (int j = 0; j < kDrafts; ++j)
        check(pstab[c * kDrafts + j] == d.sess_slot[k], "passes: an entry's rows take its slot", it);
      ++c;
    }
    for (; c < kMaxSess; ++c) {
      check(bent[c] == 0, "passes: the list past the entries on is 0", it);
      for (int j = 0; j < kDrafts; ++j) check(pstab[c * kDrafts + j] == 0, "passes: the tables past the entries on are 0", it);
    }
    // the verdicts: samples with matches planted (row r's sample is row r + 1's token, two times in three)
    int32_t fb[kMaxRows];
    for (int r = 0; r < kMaxRows; ++r)
      fb[r] = rnd() % 3 ? d.tokens[std::min(r + 1, kMaxRows - 1)] : (int32_t)(5000 + rnd() % 7);
    SpecVerdict v[kMaxSess];
    spec_batch_verdicts(d, fb, true, v);
    for (int k = 0; k < kMaxSess; ++k) {
      if (k >= ns || !d.sess_on[k]) {
        check(v[k].step_rows == 0 && v[k].rows == 0, "verdicts: an entry off has no rows", it);
        continue;
      }
      const int nk = ss[k].next_rows, r0 = row0[k];
      const int rows = lone_rows(nk, fb + r0, d.tokens + r0, true);
      check(v[k].row0 == r0 && v[k].step_rows == nk, "verdicts: the entry's rows", it);
      check(v[k].rows == rows && v[k].pos0 == d.pos[r0] && v[k].bonus == fb[r0 + rows - 1],
            "verdicts: the one-session rule over the entry's rows alone", it);
    }
    // a given step: one row per entry on, the host's layout; row0 and the verdicts (every row final)
    StepIn g{};
    g.nsess = ns;
    g.spec = kSpecGiven;
    g.sbatch = 1;
    int r = 0;
    for (int k = 0; k < ns; ++k) {
      g.sess_on[k] = d.sess_on[k], g.sess_slot[k] = d.sess_slot[k];
      if (!g.sess_on[k]) continue;
      g.tokens[r] = 7000 + k, g.pos[r] = 300 + 10 * k, g.slot[r] = g.sess_slot[k], g.sess[r] = k;
      ++r;
    }
    g.rows = r;
    for (int q = r; q < kMaxRows; ++q) g.tokens[q] = 0, g.pos[q] = g.pos[r - 1], g.slot[q] = g.slot[r - 1], g.sess[q] = g.sess[r - 1];
    int row0g[kMaxSess];
    spec_batch_row0(g, row0g);
    for (int q = 0; q < kMaxRows; ++q) fb[q] = 9000 + q;
    spec_batch_verdicts(g, fb, false, v);
    int rr = 0;
    for (int k = 0; k < ns; ++k) {
      if (!g.sess_on[k]) {
        check(v[k].step_rows == 0, "given: an entry off has no rows", it);
        continue;
      }
      check(row0g[k] == rr, "given: row0", it);
      check(v[k].row0 == rr && v[k].step_rows == 1 && v[k].rows == 1 && v[k].pos0 == g.pos[rr] && v[k].bonus == fb[rr],
            "given: one final row per entry", it);
      ++rr;
    }
  }
  std::printf("spec_batch_test: %d checks over %d iterations, %d failed\n", checks, kIters, fails);
  return fails ? 1 : 0;
}
