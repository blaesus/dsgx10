// The runner's own small kernels: the glue between engine/kernels, engine/attn and engine/comm
// (embedding, unpacking all-gathers, Engram's gate, the head's local argmax).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "comm/collectives.h"
#include "model/config.h"
#include "common/spec_sampling.h"

namespace model::ops {

using bf16 = __nv_bfloat16;

// What the host decides for a step, written to pinned memory before the launch (a ring of kInSlots,
// so the next step can be queued while one runs); step_in copies the next slot to device memory first
// thing, so one captured graph serves every step of its row count.
constexpr int kInSlots = 4;
constexpr int kDrafts = 5;  // DSpark's drafts per pass (dspark::kDraft)
enum : int32_t { kSpecOff = 0, kSpecGiven = 1, kSpecVerify = 2 };
// How a spec step chooses the next step's drafts (StepIn::gate; spec.cu k_spec_next):
// - kGateThresh: draft j while survival[j - 1] >= thresh;
// - kGateRatio: the prefix of n drafts with the most expected tokens per ms,
//   (1 + survival[0] + .. + survival[n - 1]) / step_ms[n];
// - kGateMarginal: draft j while survival[j - 1] >= thresh * (step_ms[j] - step_ms[j - 1]), thresh then
//   being the tokens per ms of the steps so far.
enum : int32_t { kGateThresh = 0, kGateRatio = 1, kGateMarginal = 2 };
constexpr int kMaxSess = kMaxRows;  // sessions in a step, at most (docs/batching.md, B2)
// The runner's kernels whose launch geometry holds the rows (sample's grid and posted bytes, fused.cu's block
// sizes) launch as they did before B4 up to kLowRows rows, and sized for kMaxRows past them (as dsk's
// row_budget): a step of at most 8 rows runs the kernels and launches it ran.
constexpr int kLowRows = 8;
constexpr int row_budget(int R) { return R <= kLowRows ? kLowRows : kMaxRows; }
constexpr int kSessTok = 1 + kDrafts;  // a session's rows in a spec step, at most: its bonus token and its drafts
static_assert(kSessTok <= kMaxSessRows, "a session's spec rows fit its rows of a step");
using StepSamp = spec_sampling::Params;  // unchanged per-session sampling ABI
struct StepIn {
  int32_t tokens[kMaxRows];  // >= 0: the token; -1 - j: row j of the previous step's sample (decode
                             // queued before the previous step ended)
  int32_t pos[kMaxRows];
  int32_t slot[kMaxRows];  // row r's session slot (Model::use_slot), in every entry, so that a spec step's draft
                           // pass finds its session's in any of its rows; read only with Options::slots > 1
  int32_t sess[kMaxRows];  // row r's session within the step (0 .. nsess - 1), in every entry: its sampling is
                           // samp[sess[r]]
  uint64_t e_wait[2];  // Engram: the store's completion seq to wait for, per Engram layer
  int32_t rows;        // R
  int32_t nsess;       // the step's sessions (a batch step's; 1 otherwise)
  uint32_t step;       // the host's step number: where the sample goes, and its seq
  // speculative steps (spec.cu): kSpecGiven, the host's rows, every one final; kSpecVerify, the rows the
  // previous spec step chose (its last token, then the drafts it decided to verify)
  int32_t spec;
  float thresh;        // the next step verifies draft j while survival[j - 1] >= thresh (kGateThresh)
  int32_t max_drafts;  // and at most this many (0..kDrafts)
  int32_t pos_end;     // and its rows stay below this position
  int32_t force;       // >= 0: draft from this token, not the sampled one (teacher-forced diagnostics)
  int32_t probe[kDrafts];  // EXPERIMENT (Options::tree_probe): the path of the drafter's probe
  int32_t gate;        // kGateThresh, kGateRatio or kGateMarginal
  float step_ms[kDrafts + 1];  // kGateRatio, kGateMarginal: the modelled ms of a step verifying n drafts
                               // (n + 1 rows), its draft pass included
  StepSamp samp[kMaxSess];     // each session's sampling
  // Batch spec steps (spec.cu, docs/batching.md B3: drafts across sessions). Entry k (0 .. nsess - 1) is one
  // session; its rows are consecutive (sess[r] = k), its bonus token then its drafts. Read only when
  // sbatch = 1; the one-session path never reads them.
  int32_t sbatch;               // 1: a batch spec step (k_spec_in_batch's layout, verdicts per entry)
  int32_t sess_on[kMaxSess];    // entry k takes part in this step (the host writes it before every launch)
  int32_t sess_slot[kMaxSess];  // entry k's session slot
  int32_t scap[kMaxSess];       // entry k's drafts for the next step, at most (0..kDrafts)
  int32_t spos_end[kMaxSess];   // entry k's rows stay below this position
  float sthresh[kMaxSess];      // entry k's threshold (kGateThresh)
};
static_assert(sizeof(StepIn) % 4 == 0, "StepIn is copied word by word");
// slot = *counter % kInSlots of ring -> dev, feedback tokens resolved from fb, ++*counter.
void step_in(const StepIn* ring, uint32_t* counter, const int32_t* fb, StepIn* dev, cudaStream_t s);

// ---- sampling (sample.cu) ----
constexpr int kTopLogprobs = 20;  // top logprobs per row, at most
constexpr int kCand = 64;         // candidates per rank and row for top-k / top-p / top logprobs
constexpr int kOutSlots = 4;
constexpr int kProbeWords = 100;  // sizeof(dspark::ProbeOut) / 4
struct SampleRow {
  int32_t token;
  float logprob;   // of token under softmax(logits / T) (T = 1 when greedy), before top-k / top-p
  float value;     // what won: the logit (greedy, top-k / top-p) or the Gumbel-perturbed logit / T
  uint32_t flags;  // kSampApprox: the kept candidates reached past the exactly known ones
};
enum : uint32_t {
  kSampApprox = 1, kSampProb = 2, kSampProbReject = 4,
  kSampProbFallback = 8, kSampResidualZero = 16, kSampProbSkipped = 32
};
__host__ __device__ inline bool spec_row_accepted(int sampled, int proposed, uint32_t flags) {
  return (flags & kSampProb) ? !(flags & kSampProbReject) : sampled == proposed;
}
// One step's result in pinned memory (a ring of kOutSlots, by StepIn::step), the same on every rank. Its
// rows are the step's, up to kMaxRows (a batch step's) or kMaxSessRows (one session's).
struct SampleOut {
  uint32_t seq;  // StepIn::step + 1, written last
  uint32_t ctl;  // rank 0's control word as the step read it
  int32_t rows;  // rows of row[]: the step's rows; a verify step's: the tokens it emits (accepted drafts,
                 // then the token after them), row[j] and top_*[j] from row j's distribution
  int32_t n_top;  // the most top logprobs of any session; row r has its own session's n_top
  SampleRow row[kMaxRows];
  int32_t top_id[kMaxRows][kTopLogprobs];  // the n_top most likely ids, logprobs as SampleRow's
  float top_lp[kMaxRows][kTopLogprobs];
  // speculative steps: this step's rows, the next step's (the device's choice), the draft pass
  int32_t pos0, step_rows;
  int32_t next_pos0, next_rows;
  int32_t next_tok[kMaxRows];
  int32_t drafts[kDrafts];
  float survival[kDrafts];
  uint32_t probe[kProbeWords];  // EXPERIMENT (Options::tree_probe): the draft pass's dspark::ProbeOut
  // Batch spec steps (StepIn::sbatch): entry k's rows of this step and the device's choice for its next
  int32_t sess_row0[kMaxSess], sess_step_rows[kMaxSess];  // its rows: row[sess_row0 .. sess_row0 + sess_step_rows)
  int32_t sess_rows[kMaxSess];  // of them, the tokens it emits (accepted drafts, then the token after them)
  int32_t sess_pos0[kMaxSess];  // its first row's position
  int32_t next_srows[kMaxSess], next_spos0[kMaxSess];  // its rows of the next step (0: it was off this step)
  int32_t next_stok[kMaxSess][kSessTok];
};
// The device's side of speculative steps (spec.cu), in device memory: what a spec step's sampling
// decided, and the rows of the verify step after it.
struct SpecState {
  int32_t count, pos0;  // the step's final rows: positions pos0 .. pos0 + count - 1
  int32_t bonus;        // the token after them (what the next draft pass starts from)
  int32_t next_rows, next_pos0;
  int32_t next_tok[kMaxRows];
};

// The step's sampling for rows [0, in->rows) of this rank's logits [R][32320] (global ids from vocab0),
// fused with its collective: every rank sends, per row, its max, partition sum, Gumbel winner and
// (for top-k / top-p / top logprobs) its best <= kCand logits; every rank then picks the same token:
// greedy (ties to the lowest id), Gumbel-max over the whole vocabulary (temperature only), or
// temperature -> top-k -> top-p over the candidates' union with one uniform from (seed, position). Each
// row by its own session's parameters, samp[sess[r]].
// Rank 0 also sends *ctl (host memory, read when the kernel runs), which comes back in SampleOut::ctl.
// Out: outs[in->step % kOutSlots] and the tokens in fb [kMaxRows] (device, the next step's feedback).
// Spec steps (in->spec, ss given): the draft in row j + 1 of a verify step is accepted when it is row j's
// sample. Greedy, that is the longest matching prefix; sampled, it is speculative sampling with the
// draft as a one-hot proposal q: accepting with probability min(1, p(d) / q(d)) = p(d) and else drawing
// from p without d is the same law as drawing from p and accepting when the draw is d, and the draw is
// the non-speculative sampler's (noise from the seed and position), so the tokens are its tokens. The
// step's final rows and the token after them go to ss and row[] (SampleOut::rows); the step is
// published later, after its draft pass (spec.cu).
// With proposal_logits/meta, eligible verify rows instead apply min(1,p(d)/q(d)); a rejection
// samples max(p-q,0) with an independent RNG stream. Explicit flags, not token equality, end the
// accepted prefix. Bonus rows and unsupported/stale proposals use the original target sampler.
// q is [real cache slot][5][kVocabPart], raw FP32 biased draft logits; metadata is never restored
// from a prefix cache. A probability launch has two fixed collectives, an off launch keeps one.
// max_rows (kLowRows or kMaxRows, row_budget): the grid, a block per row, and the bytes each rank posts; the
// step's rows must not exceed it (a captured launch serves every step of its graph: the one-session spec graph's
// kLowRows, a batch graph's kMaxRows).
void sample(const comm::DevChannel& ch, const float* logits, int vocab0, const StepIn* in, const uint32_t* ctl,
            SampleOut* outs, int32_t* fb, uint32_t* counter, cudaStream_t s, SpecState* ss = nullptr,
            int max_rows = kMaxRows, const float* proposal_logits = nullptr,
            const spec_sampling::ProposalMeta* proposal_meta = nullptr, int proposal_slots = 0);

// x [R][4][5120] = embed[tokens[r]] in every copy; pre_mix [R][4] = (1, 0, 0, 0).
void embed(const bf16* table, const int* tokens, int R, bf16* x, float* pre_mix, cudaStream_t s);

// The attention input projections' all-gather. Each rank's block (bytes_per_rank) is
// [qa R x 320 bf16 | kv R x 128 bf16 | ckv R x 128 f32 | csc R x 128 f32]; ncomp = 0 (no compressor),
// 1 (ckv only, ratio 1) or 2 (ckv and csc). Out: qa [R][1280], kv [R][512], ckv/csc [R][512].
size_t attn_pack_bytes(int R, int ncomp);
inline constexpr int kMixStride = 32;  // floats per row of a block of mix partials (25 used)
void unpack_attn(const uint8_t* gath, int R, int ncomp, bf16* qa, bf16* kv, float* ckv, float* csc, cudaStream_t s);
// The router logits' all-gather [4][R][96] -> [R][384].
void unpack_gate(const float* gath, int R, float* logits, cudaStream_t s);
void f32_to_bf16(const float* in, bf16* out, int n, cudaStream_t s);

// Engram. rows: the all-gathered row segments; off [R][24] byte offsets of each cell's 264-byte record.
// -> xq [R][6144] e4m3, xs [R][192] ue8m0: the dequantised rows' exact act_quant equivalent.
void engram_pack(const uint8_t* rows, const uint32_t* off, int R, uint8_t* xq, uint8_t* xs, cudaStream_t s);
// This rank's share of the gate: per row the 4 partial dots sum(h * w * key) and key sums of squares
// over its 1280 columns of each copy, then its 1280 value columns (kEngramPartBytes per row).
// kv [R][6400] from wkv; w [4][5120] = q_weight * k_weight (fp32).
void engram_partial(const bf16* x, const bf16* kv, const float* w, int rank, int R, uint8_t* out, cudaStream_t s);
// y = bf16(h + gate * value) from the gathered parts [4][R][kEngramPartBytes]; y may alias x.
void engram_combine(const bf16* x, const uint8_t* gath, int R, bf16* y, cudaStream_t s);

// ---- collectives fused with their neighbours (fused.cu) ----
// Each is a PDL launch: it waits (griddepcontrol.wait) for the kernel before it, posts its
// contribution, lets the next launch start loading weights, then waits for the other ranks.
// A sublayer's all-reduce and what follows it: sub = bf16(sum of the four partials in rank order)
// (part: bf16 [R][5120], or fp32 with f32 = true, as allreduce_bf16 / allreduce_f32 + f32_to_bf16);
// stream = attn::hc_post(sub, res, post, comb) [R][20480]; then optionally the next sublayer's input
// before its norm, yraw = attn::hc_pre(stream, pre) without norm_w, and sub itself. Bit-identical to
// the separate kernels.
void allreduce_hc(const comm::DevChannel& ch, const void* part, bool f32, int R, const bf16* res, const float* post,
                  const float* comb, const float* pre, bf16* stream, bf16* yraw, bf16* sub, cudaStream_t s);
// allreduce_hc with the sublayer's mixes riding along (single-pass mHC needs them only here): chunks
// [nchunks][R][25] is attn::hc_project of this rank's copy (columns [5120 rank, 5120 rank + 5120) of
// res); their sum in chunk order is sent with part, the four ranks' sums are added in rank order, and
// attn::hc_finish's arithmetic (scale [3], base [24]) gives pre, post and comb [R][4|4|16], written out
// and used as allreduce_hc uses its inputs. 128 bytes more per row; fixed order, so bit-identical on
// every rank.
void allreduce_hc_mix(const comm::DevChannel& ch, const void* part, bool f32, int R, const float* chunks, int nchunks,
                      const float* scale, const float* base, const bf16* res, float* pre, float* post, float* comb,
                      bf16* stream, bf16* yraw, bf16* sub, cudaStream_t s);
// comm::allgather of the attention pack (attn_pack_bytes(R, ncomp) per rank) and unpack_attn in one.
void allgather_attn(const comm::DevChannel& ch, const uint8_t* pack, int R, int ncomp, bf16* qa, bf16* kv, float* ckv,
                    float* csc, cudaStream_t s);
// The same with the attention's mixes riding along (allreduce_hc_mix's mix partials, for a sublayer whose
// mixes are wanted before its all-reduce): chunks [nchunks][R][25], attn::hc_project of this rank's copy,
// summed in chunk order, follow the pack; mix [4][R][kMixStride] gets the four ranks' sums, for mix_ranks.
void allgather_attn_mix(const comm::DevChannel& ch, const uint8_t* pack, int R, int ncomp, const float* chunks,
                        int nchunks, bf16* qa, bf16* kv, float* ckv, float* csc, float* mix, cudaStream_t s);
// A sublayer's mixes by a collective of their own, for a side stream whose work can overlap the sublayer
// (one block, no PDL; it must be the only collective between the one before it and the one after it,
// e.g. forked after the previous all-reduce and joined before the sublayer's own): chunks as in
// allreduce_hc_mix, pre, post and comb out, bit for bit allreduce_hc_mix's.
void mix_allreduce(const comm::DevChannel& ch, const float* chunks, int nchunks, int R, const float* scale,
                   const float* base, float* pre, float* post, float* comb, cudaStream_t s);
// pre, post and comb [R][4|4|16] from allgather_attn_mix's mix: the ranks' sums added in rank order and
// attn::hc_finish's arithmetic, bit for bit what allreduce_hc_mix makes of the same partials. One block;
// no PDL (a side-stream kernel).
void mix_ranks(const float* mix, int R, const float* scale, const float* base, float* pre, float* post, float* comb,
               cudaStream_t s);

// x[i] = bf16(x[i]) in place (fp32 GEMM outputs that model.py computes in bf16).
void round_bf16(float* x, size_t n, cudaStream_t s);
// pos[i] = p0 + i, i < n.
void iota(int* pos, int p0, int n, cudaStream_t s);

// q_weight * k_weight in fp32 ([4][5120] bf16 each).
void mul_bf16(const bf16* a, const bf16* b, float* out, int n, cudaStream_t s);

// Spin until *flag == *seq (flag: host memory written by another agent, e.g. the Engram store's worker).
void wait_seq(const uint64_t* flag, const uint64_t* seq, cudaStream_t s);

}  // namespace model::ops
