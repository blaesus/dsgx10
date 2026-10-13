// B3 (docs/batching.md): the batch spec step's layout and verdicts, one code for the device (spec.cu's
// k_spec_in_batch, sample.cu's k_sample) and the host (tests/spec_batch_test.cu checks them against a
// reference per entry, no GPU needed).
#pragma once
#include "model/ops.h"

namespace model::ops {

// A batch verify step's rows from the entries' spec states: the entries on (sess_on), in entry order,
// each its next_rows rows (its bonus token, then its drafts) at next_pos0 .., with its slot (sess_slot)
// and its entry index in sess[]; the entries past R repeat the last row's (tokens 0); row0[k] entry k's
// first row (an entry off: where it would have started). Returns R, which the host keeps within
// kMaxRows (the entries' caps) and above 0 (it launches with an entry on).
__host__ __device__ inline int spec_batch_layout(StepIn& d, const SpecState* ss, int* row0) {
  const int ns = d.nsess < 1 ? 1 : d.nsess > kMaxSess ? kMaxSess : d.nsess;
  int R = 0;
  for (int k = 0; k < ns; ++k) {
    row0[k] = R;
    if (!d.sess_on[k]) continue;
    int nk = ss[k].next_rows;
    nk = nk < 1 ? 1 : nk > kSessTok ? kSessTok : nk;
    for (int j = 0; j < nk && R < kMaxRows; ++j, ++R) {
      d.tokens[R] = ss[k].next_tok[j];
      d.pos[R] = ss[k].next_pos0 + j;
      d.slot[R] = d.sess_slot[k];
      d.sess[R] = k;
    }
  }
  for (int k = ns; k < kMaxSess; ++k) row0[k] = 0;
  if (R < 1) R = 1;  // never: the host launches with an entry on
  d.rows = R;
  for (int r = R; r < kMaxRows; ++r) {  // the entries past R repeat the last row's
    d.tokens[r] = 0;
    d.pos[r] = d.pos[R - 1], d.slot[r] = d.slot[R - 1], d.sess[r] = d.sess[R - 1];
  }
  return R;
}

// A batch given step's row0 from the host's layout: sess[r] the row's entry, the entries on in order.
__host__ __device__ inline void spec_batch_row0(const StepIn& d, int* row0) {
  const int ns = d.nsess < 1 ? 1 : d.nsess > kMaxSess ? kMaxSess : d.nsess;
  int r = 0;
  for (int k = 0; k < ns; ++k) {
    row0[k] = r;
    if (!d.sess_on[k]) continue;
    while (r < d.rows && d.sess[r] == k) ++r;
  }
  for (int k = ns; k < kMaxSess; ++k) row0[k] = 0;
}

// The verdicts per entry of a batch spec step: fb[r] row r's sample, in.tokens[r] what it was fed. Entry
// k's rows are its run in sess[]: its first row is final and, in a verify step, each row after it is
// accepted while its token is the sample of the row before it (the one-session rule, over its rows
// alone; no row reads another entry's). v[k]: step_rows 0 for an entry with no rows; else its row0,
// step_rows, rows (the tokens it emits: the accepted drafts, then the token after them), pos0 and bonus
// (the sample of its last final row).
struct SpecVerdict {
  int32_t row0, step_rows, rows, pos0, bonus;
};
__host__ __device__ inline void spec_batch_verdicts(const StepIn& in, const volatile int32_t* fb, bool verify,
                                                    SpecVerdict* v) {
  for (int k = 0; k < kMaxSess; ++k) v[k] = SpecVerdict{0, 0, 0, 0, 0};
  const int R = in.rows < 1 ? 1 : in.rows > kMaxRows ? kMaxRows : in.rows;
  int r0 = 0;
  while (r0 < R) {
    const int k = in.sess[r0] < 0 ? 0 : in.sess[r0] >= kMaxSess ? kMaxSess - 1 : in.sess[r0];
    int nk = 1;
    while (r0 + nk < R && in.sess[r0 + nk] == in.sess[r0]) ++nk;
    int n = nk - 1;
    if (verify) {
      n = 0;
      while (n < nk - 1 && fb[r0 + n] == in.tokens[r0 + n + 1]) ++n;
    }
    v[k] = SpecVerdict{r0, nk, n + 1, in.pos[r0], fb[r0 + n]};
    r0 += nk;
  }
}

}  // namespace model::ops
