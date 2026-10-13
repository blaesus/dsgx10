// DSpark draft module (M7): the three DSpark blocks (mtp.0-2), main_proj, the shared embedding and vocab head,
// the Markov head and the confidence head, on one rank at TP=4, as model.py's DSpark (README.md: how model.py
// drives it, every launch, the collectives). Every launch is a PDL launch; counts, positions and tokens live on
// the device, so one CUDA graph of accept() + draft() serves every step.
//
// Per step, after the main model's verify: accept() the newly accepted positions (their main_hidden goes
// through main_proj and main_norm into each block's sliding-window KV), then draft() the block after the last
// of them: the next token and four noise tokens, through the three blocks, the head and a 5-step Markov chain,
// giving 5 greedy draft tokens and the confidence head's survival probabilities. A batch spec step of several
// sessions runs their accepts, then one draft_batch() over their blocks side by side (B3 phase 2a).
#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <cuda_runtime.h>

#include "attn/attn.h"
#include "kernels/dsk.h"

namespace dsw {
class WeightFile;
}

namespace dspark {

inline constexpr int kBlocks = 3;   // mtp.0 .. mtp.2
inline constexpr int kDraft = 5;    // block rows = drafts per pass (dspark_block_size)
inline constexpr int kMaxAcc = 8;   // positions per accept() call
inline constexpr int kMaxHidden = 16;  // rows of a main_hidden buffer, at most: a batch spec step's (docs/batching.md, B3)
// A batch spec step's draft pass over several entries (Draft::draft_batch, docs/batching.md B3 phase 2a): a pass
// takes at most kBatchEnts entries, kDraft rows each, within the kernels' 16 rows (dsk::kMaxRows,
// attn::kMaxDecodeRows); kMaxEnts entries keep an anchor and results (the runner's ops::kMaxSess).
inline constexpr int kBatchEnts = 3, kBatchRows = kDraft * kBatchEnts, kMaxEnts = 16;
inline constexpr int kRanks = 4;
inline constexpr int kDim = 5120, kHc = 4, kQLora = 1280;
inline constexpr int kVocab = 129280, kVocabRank = kVocab / kRanks, kMarkov = 256;
inline constexpr int kExperts = 128, kTopK = 3, kNoiseToken = 128799;
inline constexpr int kMainK = 3 * kDim;  // main_hidden: the inputs of layers 37, 38, 39, concatenated

// One block's tensors on this rank (docs/layout.md: names, cuts and tile layouts).
struct BlockW {
  const uint8_t *wq_a, *wq_a_s, *wkv, *wkv_s, *wq_b, *wq_b_s, *wo_a, *wo_a_s, *wo_b, *wo_b_s;  // e4m3 L1, ue8m0
  const uint16_t *q_norm, *kv_norm, *attn_norm, *ffn_norm;
  const float* sink;                               // [16]: this rank's heads
  const float *attn_fn, *attn_base, *attn_scale;  // mHC; fn [24][5120]: this rank's copy of the stream
  const float *ffn_fn, *ffn_base, *ffn_scale;
  const uint8_t* gate;     // bf16 [32][5120] L2: this rank's 32 experts' router rows
  const float* gate_bias;  // [128]
  dsk::MoeWeights moe;     // the 128 experts' and the shared expert's 576-row slices
  // Optional, whole on every rank (the runner gathers them once): the mHC projections over all 4 copies of
  // the stream, [24][20480], and all 128 router rows, bf16 [128][5120] L2. Given, the pass computes every
  // rank's projection chunks and router logits itself, bit for bit what the gathers would bring, without
  // their 9 collectives.
  const float *attn_fn_all = nullptr, *ffn_fn_all = nullptr;
  const uint8_t* gate_all = nullptr;
};
struct Weights {
  BlockW b[kBlocks];
  const uint8_t *main_proj, *main_proj_s;  // e4m3 [1280][15360] L1 (this rank's outputs), ue8m0 [40][480]
  const uint16_t *main_norm, *norm;        // mtp.0.main_norm; mtp.2.norm (the head's)
  const uint16_t* embed;                   // bf16 [129280][5120], the main model's (replicated)
  const uint8_t* head;                     // bf16 [32320][5120] L2, the main model's vocab slice
  const uint16_t* markov_embed;            // bf16 [129280][256] (replicated)
  const uint8_t* markov_head;              // bf16 [32320][256] L2, the same vocab slice
  const uint16_t* conf;                    // bf16 [5376]: confidence_head.proj
  // Optional: an FP8 copy of the head slice (head_fp8()), which the pass then reads instead, half the bytes.
  // The drafts' arithmetic changes with it; the tokens of a spec decode cannot.
  const uint8_t *head8 = nullptr, *head8_s = nullptr;  // e4m3 [32320][5120] L1, ue8m0 [1010][160]
  // Optional: the head slice coded (dsk::bf16x_encode), which the pass then reads instead of head (without
  // head8), the same bits; the runner codes its head in place (model::Model::code_head, Draft::set_head_coded).
  const uint8_t* head_coded = nullptr;
  // Optional (docs/engine.md, "The draft pass's approximations"; bundle B): the head as MXFP4 (head_fp4()), read
  // before head8 and head_coded, a quarter of the head's bytes; the Markov head as FP8 (markov_fp8()), half its
  // bytes. The drafts' arithmetic changes with them, the tokens cannot.
  const uint8_t *head4 = nullptr, *head4_s = nullptr;      // dsk::gemv_fp4w_f32's layouts, [32320][5120]
  const uint8_t *markov8 = nullptr, *markov8_s = nullptr;  // e4m3 [32320][256] (markov_fp8()), ue8m0 [1010][8]
};
// This rank's tensors from loaded weight files (the first file holding a loaded tensor of a name wins): the
// runner's rank file, or a test's subsets. Dies on a missing tensor or an unexpected size.
Weights weights_from(std::initializer_list<const dsw::WeightFile*> files);
// Weights::head8 from the bf16 head slice: e4m3 with a power-of-two scale per 32x32 block (amax / 448 rounded
// up, as act_quant), in the layouts dsk::gemv_fp8w_f32 reads. q: kVocabRank * kDim bytes, s: their / 1024.
void head_fp8(const uint8_t* head, uint8_t* q, uint8_t* s, cudaStream_t st);
// Weights::head4 from the FP8 copy (head8, head8_s): each row's groups of 32 to the values sim_fp4 gives them
// (MXFP4's, the FP8 copy's e4m3 rounding of the halved value included), each under the least power-of-two scale
// that holds all 32 (sim_fp4's own unless that rounding moved a value), in dsk::gemv_fp4w_f32's layouts, which
// then gives the simulation's logits bit for bit. q: kVocabRank * kDim / 2 bytes, s: kVocabRank * kDim / 32.
// Synchronous (bind time); dies if a group has no such scale up to 2. Returns the groups whose scale moved.
long head_fp4(const uint8_t* head8, const uint8_t* head8_s, uint8_t* q, uint8_t* s, cudaStream_t st);
// Weights::markov8 from the bf16 Markov head slice (L2 tiles): sim_fp8_l2's values (e4m3 under a power-of-two
// scale per 32x32 block, amax / 448 rounded up), each group of 8 rows as markov_kernel's lanes read it ([4]
// [32 lanes][16 B]: lane l's piece p, row l / 4's k = 64p + 8(l % 4) + i, then 64p + 32 + 8(l % 4) + i,
// i = 0..7), the scales [N/32][8]. q: kVocabRank * kMarkov bytes, s: their / 1024.
void markov_fp8(const uint8_t* markov_head, uint8_t* q, uint8_t* s, cudaStream_t st);

// EXPERIMENT, measurement only (the draft pass study, docs/engine.md "The draft pass's approximations"): the
// drafter's weights rounded in place to the values a cheaper format holds, so that spectf measures the drafts
// under it with today's kernels. sim_fp4: an e4m3 matrix [N][K] (L1 tiles or natural) with ue8m0 scales
// [N/32][K/32]; each row's groups of 32 to MXFP4's values (e2m1 under a power-of-two scale per group, amax / 6
// rounded up), kept as e4m3 under the block's scale doubled (an e2m1 value rounded up may pass 448).
// sim_fp8_l2: a bf16 matrix [N][K] in L2 tiles to e4m3's values under a power-of-two scale per 32x32 block
// (amax / 448 rounded up, head_fp8's rule).
void sim_fp4(uint8_t* w, uint8_t* s, int N, int K, bool l1, cudaStream_t st);
void sim_fp8_l2(uint8_t* w, int N, int K, cudaStream_t st);

// The runner's part, at the start of layers 37, 38 and 39 (their input stream x [rows][4][5120], after Engram
// where there is one): the mean of the 4 copies (torch's bf16 mean) into columns [5120 * slot, +5120) of
// main_hidden [rows][15360] (rows <= kMaxHidden: a step's; accept() reads at most kMaxAcc of them), slot = layer - 37.
void main_hidden(const uint16_t* x, int rows, int slot, uint16_t* mh, bool pdl, cudaStream_t s);

// A rank's argmax candidate over its vocab slice (id: the global token id).
struct Cand {
  float v;
  int id;
};
// Several matrices gathered in one collective: every rank's rows of src (width bytes each) side by side in rank
// order, dst + r * dst_ld + q * width <- rank q's src + r * src_ld (bytes; rows of the same r on every rank).
struct GatherSeg {
  const void* src;
  size_t src_ld;
  void* dst;
  size_t dst_ld;
  size_t width;
  int rows;
};
inline constexpr int kMaxGatherSegs = 4;
// The cross-rank points of a pass, called in the same order and with the same shapes on every rank (the runner
// implements them over the ring with engine/comm). Every rank must get the same bits back.
class Comm {
 public:
  virtual ~Comm() = default;
  virtual void all_gather(const GatherSeg* segs, int n, cudaStream_t s) = 0;
  // dst [n] = bf16(the fp32 sum over the ranks of src [n], in rank order); src bf16 (f32 false) or fp32 partials
  virtual void all_reduce(const void* src, bool f32, uint16_t* dst, size_t n, cudaStream_t s) = 0;
  // dst = the largest candidate over the ranks (ties: the lowest id)
  virtual void argmax(const Cand* src, Cand* dst, cudaStream_t s) = 0;
  // The same for `rows` candidates at once (dst[r] from src[r] of every rank, rows independent; a batch pass's
  // entries): by default one argmax() per row, in order; the runner's ring does them in one collective.
  virtual void argmax_rows(const Cand* src, Cand* dst, int rows, cudaStream_t s) {
    for (int r = 0; r < rows; ++r) argmax(src + r, dst + r, s);
  }
};
// STAND-INs for timing one rank alone: every collective is one local PDL kernel that puts this rank's data where
// the result would land (gathers fill every rank's slot with it, reductions take it as the sum).
Comm* standin_comm();

// Tests: copies of intermediates (every non-null pointer receives its value during the pass; device memory).
struct Taps {
  uint16_t* main_hidden = nullptr;     // accept: [kMaxAcc][15360], the input
  uint16_t* main_x = nullptr;          // accept: [kMaxAcc][5120], main_norm(main_proj(main_hidden))
  float* mix[kBlocks][2] = {};         // [5][24]: pre, post, comb of block b's attention (0) / FFN (1) sublayer
  uint16_t* attn_in[kBlocks] = {};     // [5][5120]
  uint16_t* qa[kBlocks] = {};          // [5][1280] (gathered)
  uint16_t* q_pre[kBlocks] = {};       // [5][16][512]: this rank's heads
  uint16_t* attn_o[kBlocks] = {};      // [5][16][512]: this rank's heads, inverse RoPE applied
  uint16_t* attn_out[kBlocks] = {};    // [5][5120] (all-reduced)
  uint16_t* mid[kBlocks] = {};         // [5][4][5120]
  uint16_t* ffn_in[kBlocks] = {};      // [5][5120]
  int* route_idx[kBlocks] = {};        // [5][6] (picks 4-6 repeat 1-3)
  float* route_w[kBlocks] = {};        // [5][6]
  uint16_t* ffn_out[kBlocks] = {};     // [5][5120] (all-reduced)
  uint16_t* out[kBlocks] = {};         // [5][4][5120]
  uint16_t *head_x = nullptr, *head_xn = nullptr;  // [5][5120]: hc_pre's output, then normed
  float* logits = nullptr;             // [5][32320]: this rank's slice, before the Markov bias
  float* bias = nullptr;               // [5][32320]: the Markov bias
  uint16_t* markov_embed = nullptr;    // [5][256]
};
// EXPERIMENT, bench only (Draft::probe): the drafter along a given path. Depth j (the draft for the anchor
// + 2 + j) takes its Markov bias from the anchor's token (ids()[0]) for j = 0, else from path[j - 1], as a tree
// node whose ancestors are the path's would; its kProbeK most likely tokens under the biased logits (ties:
// the lowest id, as draft()'s argmax), their softmax probabilities over the whole vocabulary, path[j]'s
// probability and rank among them, and the confidence head's logit with the path's Markov embedding.
inline constexpr int kProbeK = 8;
struct ProbeOut {
  int32_t id[kDraft][kProbeK];
  float p[kDraft][kProbeK];
  float p_path[kDraft];  // 0: no path token
  int32_t rank[kDraft];  // path[j]'s index in id[j]; kProbeK: below them; -1: no path token
  float conf[kDraft];    // conf[0] is draft()'s confidence()[0]
  float lse[kDraft];     // log-sum-exp of the biased logits
};

// Tests: reference values replacing intermediates before they are used (device memory; null = computed).
struct Inject {
  const uint16_t* stream[kBlocks] = {};  // [5][4][5120]: block b's input stream
  const float* pre[kBlocks] = {};        // [5][4]: the pre-mix block b's attention collapses the stream with
  const uint16_t* mid[kBlocks] = {};     // [5][4][5120]: the stream after block b's attention sublayer
  const float* attn_pre[kBlocks] = {};   // [5][4]: the pre-mix block b's FFN collapses with (its attention's)
  const uint16_t* head_stream = nullptr; // [5][4][5120]
  const float* head_pre = nullptr;       // [5][4]
  const int* ids = nullptr;              // [6]: the Markov chain's tokens (ids[j] gives step j its embedding)
};

class Draft {
 public:
  // table: the RoPE table without YaRN (attn::rope_table_build(.., compressed = false)), positions up to the
  // last one drafted (anchor + 5). slots: the windows of that many sessions side by side (model::Options::slots).
  Draft(const Weights& w, int rank, Comm* comm, const float2* table, int slots = 1);
  ~Draft();
  Draft(const Draft&) = delete;
  Draft& operator=(const Draft&) = delete;

  // Newly accepted positions: rows [0, *count) of main_hidden (bf16 [kMaxAcc][15360], from main_hidden()) at
  // positions *pos0 + r, 1 <= *count <= 8 (a prompt: its last 128 positions, 8 per call). Their main_kv goes
  // into each block's window; the last of them, *pos0 + *count - 1, is the next draft's anchor.
  // Session slots: the windows are the current slot's (use_slot), or with `slot` (device, at least kMaxAcc
  // entries; a step's table) row r's slot's: a step captured once serves sessions in any slot. row0 (device, one
  // int): the rows are main_hidden's [*row0, *row0 + *count), not [0, *count) (a batch step's entry; the buffer
  // then holds at least *row0 + *count rows). start_out (device, one int): where the anchor goes instead of
  // draft()'s (a batch step's entry k: batch_start() + k, which draft_batch() reads).
  void accept(const uint16_t* mh, const int* count, const int* pos0, cudaStream_t s, const int* slot = nullptr,
              const int* row0 = nullptr, int* start_out = nullptr);
  // The draft after the anchor p: *token (the token at p + 1) and four noise tokens at positions p + 1 .. p + 5
  // through the blocks and the head; then tokens() holds d1..d5, greedy drafts for positions p + 2 .. p + 6.
  // `slot` as accept()'s (kDraft entries, all the anchor's session's).
  void draft(const int* token, cudaStream_t s, const int* slot = nullptr);
  void use_slot(int s);

  // B3 phase 2a (docs/batching.md): one pass over the drafts of several entries (sessions), their rows side by
  // side, the weights read once. Rows [kDraft q, kDraft q + kDraft) are entry ent[q]'s, q < n <= kBatchEnts: its
  // token tok[ent[q] * tok_stride] (the token at its anchor + 1), its anchor batch_start()[ent[q]] (accept() with
  // start_out wrote it this step), its rows' slots slot[kDraft q + j] (the rows' session slots, as accept()'s
  // tables; required). The results of entry k: batch_ids(k) [6] (the token, then d1..d5), batch_tokens(k) [5],
  // batch_confidence(k) [5], batch_survival(k) [5], the same bits on every rank. Every row computes, bit for
  // bit, what draft() computes for it alone: the kernels take the rows as they take draft()'s (an activation
  // row per MMA lane group, its own k order, warp and split-K folds; the attention over its own window; its
  // own Markov chain and confidence), and nothing of one row enters another's (the MoE runs the union of the
  // rows' experts, each row summing its own picks in expert-id order). Buffers of its own, sized for kBatchRows
  // rows: enable_batch() once, before any capture; draft() and accept() keep theirs.
  struct BatchPass {
    const int* ent;    // device [n]: the entries, in row order
    int n;             // entries in the pass, 1..kBatchEnts
    const int* tok;    // device: entry k's token at tok[k * tok_stride]
    int tok_stride;    // in ints
    const int* slot;   // device [kDraft * n]: each row's session slot
  };
  void enable_batch();
  void draft_batch(const BatchPass& p, cudaStream_t s);
  int* batch_start() const;                      // device [kMaxEnts]: accept()'s start_out + k
  const int* batch_ids(int k) const;             // [6]
  const int* batch_tokens(int k) const { return batch_ids(k) + 1; }
  const float* batch_confidence(int k) const;    // [5]
  const float* batch_survival(int k) const { return batch_confidence(k) + 8; }
  // Weights::head_coded from now on (the runner's head coded in place after the module was made); before any
  // draft() a graph captures.
  void set_head_coded(const uint8_t* hc) { w_.head_coded = hc; }

  // EXPERIMENT, bench only (docs/engine.md, "Draft trees"; ProbeOut): enable_probe() once, before any capture
  // of probe(). After draft(), probe(path) runs the Markov chain again along path (kDraft tokens on the device,
  // the true tokens at the anchor + 2 .. + 6; < 0: none) into probe_out() (device), the same bits on every rank.
  // It leaves draft()'s results alone.
  void enable_probe();
  void probe(const int* path, cudaStream_t s);
  const ProbeOut* probe_out() const;

  // Results on the device, valid once draft()'s work is done; the same bits on every rank.
  const int* ids() const { return ids_; }              // [6]: the input token, then d1..d5
  const int* tokens() const { return ids_ + 1; }       // [5]: d1..d5
  const float* confidence() const { return conf_; }    // [5]: logit of P(d_j accepted | d_1 .. d_j-1 were)
  const float* survival() const { return conf_ + 8; }  // [5]: P(d_1 .. d_j all accepted) = prod sigmoid(conf)
  const float* logits() const;                         // [5][32320]: this rank's slice of the biased logits
  attn::Window window(int b) const;  // block b's window KV in the current slot (ring slot = position & 255)

  Taps taps;
  Inject inject;

 private:
  struct Buf;
  void mix(const uint16_t* x, const float* fn, const float* fn_all, const float* scale, const float* base,
           const uint16_t* norm, cudaStream_t s);
  // mix() over a batch pass's rows, in the batch buffers (hc_project's rows per CTA group and hc_mix's CTA per
  // row keep every row's arithmetic its own)
  void bmix(const uint16_t* x, int rows, const float* fn, const float* fn_all, const float* scale, const float* base,
            const uint16_t* norm, cudaStream_t s);
  attn::Window win(int b, const int* slot) const;  // the current slot's, or through a step's table
  Weights w_;
  int rank_;
  Comm* comm_;
  const float2* table_;
  int slots_ = 1, cur_ = 0;
  // EXPERIMENT, bench only (the draft pass study): DSPARK_VROWS, the vocabulary rows the FP8 head GEMV and the
  // Markov chain cover (a multiple of 64; the pass's time with a vocabulary subset of 4x that, not its drafts)
  int vrows_ = kVocabRank;
  Buf* b_;
  int* ids_;
  float* conf_;
};

}  // namespace dspark
