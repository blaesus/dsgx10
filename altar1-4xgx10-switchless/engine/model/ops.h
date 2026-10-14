// The runner's own small kernels (ops.cu), the sampler (sample.cu): the glue between engine/attn,
// engine/experts and engine/comm.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#include "comm/glmcoll.h"
#include "model/config.h"
#include "model/model.h"

namespace glm::model::ops {

// What the host decides for a step, written to pinned memory before the launch (a ring of kInSlots, so the next
// step can be queued while one runs); step_in copies the next slot to device memory first thing, so one captured
// graph serves every step of its row count.
inline constexpr int kInSlots = 4;
inline constexpr int kOutSlots = 4;
struct StepSamp {
  uint64_t seed;
  float temperature;  // 0: greedy
  float top_p;        // 1: off
  int32_t top_k;      // 0: off
  int32_t n_top;      // 0..kTopLogprobs
};
struct StepIn {
  int32_t tokens[kMaxRows];  // >= 0: the token; -1 - j: row j of the previous step's sample
  int32_t pos[kMaxRows];     // rows past R repeat the last row's (kernels read them for R rows only)
  int32_t slot[kMaxRows];
  int32_t sess[kMaxRows];
  int32_t rows, nsess;
  uint32_t step;  // the host's step number: where the sample goes, and its seq
  int32_t spec;   // Batch::spec (prefill's one-row step may leave any value: one row accepts nothing either way)
  StepSamp samp[kMaxSess];
};
static_assert(sizeof(StepIn) % 4 == 0, "StepIn is copied word by word");
// slot = *counter % kInSlots of ring -> dev, tokens < 0 resolved from fb, ++*counter.
void step_in(const StepIn* ring, uint32_t* counter, const int32_t* fb, StepIn* dev, cudaStream_t s, bool pdl);

// The vocabulary-split embedding: ebuf [R][4][6144] bf16, this rank's block of row r = its embedding row when the
// rank holds tokens[r] (rows [38720 rank, +38720)), else zeros. After the row all-gather, pick: out[r] = the owner's
// block (zeros where zero_pos0 and pos[r] == 0: the MTP's rule at position 0).
void embed_local(const uint16_t* table, const int32_t* tokens, int R, int rank, uint16_t* ebuf, cudaStream_t s, bool pdl);
void embed_pick(const uint16_t* ebuf, const int32_t* tokens, const int32_t* pos, bool zero_pos0, int R, uint16_t* out,
                cudaStream_t s, bool pdl);

// The MTP pass 1's rows and each session's chain row. Session s (rows [b, e)) accepts a rows: with in->spec, while
// row b + a + 1's input equals row b + a's sample (a verify step's drafts), else e - b - 1 (every input given); its
// chain row sel[s] = b + a, acc[s] = a. Row r's token: the next row's input for r < sel[s] (the given or accepted
// token after it), else its sample (fb); pos[r] = in->pos[r].
void mtp_tokens(const StepIn* in, const int32_t* fb, int R, int32_t* tok, int32_t* pos, int32_t* sel, int32_t* acc,
                cudaStream_t s, bool pdl);
// The chain's rows, one a session: row sel[s] of h, xn (bf16 [.][6144]) and topk ([.][2048]; for a row at a position
// below 2048, which selects nothing, positions 0 .. 2047) into row s of ch, cxn, ctopk; cpos[s] = pos[sel[s]],
// cslot[s] = in->slot[sel[s]].
void mtp_select(const StepIn* in, const int32_t* sel, int nsess, const int32_t* pos, const uint16_t* h, const uint16_t* xn,
                const int* topk, uint16_t* ch, uint16_t* cxn, int* ctopk, int32_t* cpos, int32_t* cslot, cudaStream_t s,
                bool pdl);
// After a pass's head: tok[r] = its draft (best[r].i), pos[r] += 1, draft_k[r] = the draft.
void mtp_next(const comm::glm::Cand* best, int R, int32_t* tok, int32_t* pos, int32_t* draft_k, cudaStream_t s, bool pdl);
// local argmax of this rank's logits [R][38720] (ids >= kVocabUsed masked; ties to the lowest id) as global ids
void local_top1(const float* logits, int R, int vocab0, comm::glm::Cand* out, cudaStream_t s, bool pdl);
// The step's drafts (draft [kMaxDrafts][kMaxSess]) and accepted counts (acc [kMaxSess]) into its SampleOut, then its
// seq (the publish of a step with draft passes).
void publish(SampleOut* outs, const StepIn* in, const int32_t* draft, const int32_t* acc, int ndraft, int nsess,
             cudaStream_t s, bool pdl);

// Prefill's sublayer ends on a reduce-scattered quarter: h = bf16(res + a), one rounding (decision 8; round2:
// bf16(res + bf16(a)), the double rounding), or with b (the old MoE combine), h = bf16(res + bf16(bf16(a) + bf16(b)))
// (the experts' finalize); n elements; h may be res.
void add_rows(const uint16_t* res, const float* a, const float* b, size_t n, uint16_t* h, cudaStream_t s,
              bool round2 = false);
// pos[i] = p0 + i, slot[i] = sl (i < n)
void iota(int32_t* pos, int32_t* slot, int p0, int sl, int n, cudaStream_t s);
// rows of bf16 [n][cols] at in (stride ldi) -> out (stride ldo)
void copy_rows(const uint16_t* in, size_t ldi, uint16_t* out, size_t ldo, int n, int cols, cudaStream_t s);

// The sampling of rows [0, in->rows) of this rank's f32 logits [R][38720] (global ids from vocab0, ids >= kVocabUsed
// masked), fused with its collective (the DeepSeek engine's sampler): greedy (ties to the lowest id), Gumbel-max
// (temperature only) or temperature -> top-k -> top-p over the ranks' candidates' union, each row by its session's
// parameters; the same token on every rank. Out: outs[in->step % kOutSlots] (seq written last when publish, else
// left to ops::publish) and the tokens in fb.
void sample(const comm::DevChannel& ch, const float* logits, int vocab0, const StepIn* in, SampleOut* outs, int32_t* fb,
            uint32_t* counter, cudaStream_t s, bool publish, int max_rows, const uint32_t* ctl = nullptr);
// (ctl: rank 0's control word in mapped host memory, null: 0; every rank's SampleOut::ctl gets rank 0's)

}  // namespace glm::model::ops
