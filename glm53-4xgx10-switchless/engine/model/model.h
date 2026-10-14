// The TP=4 forward of GLM-5.3 on one rank (docs/engine.md): the layer runner over the kernels of engine/attn and
// engine/experts and the collectives of engine/comm. Every rank holds a Model and makes the same calls in the
// same order; the residual stream, the routing, the index choices and the sampled tokens are bit-identical on
// all four (the collectives sum in rank order).
//
// A decode step takes R <= 16 rows, each a (session slot, position, token); a session's rows are consecutive and
// ascend one position at a time (a verify step's or a teacher-forced run's), sessions in slots of their own. Each
// row computes, bit for bit, what it computes in any other step (no kernel's result depends on the other rows).
// The step is one CUDA graph per row count (PDL between its kernels), queued one ahead of the host: a row's token
// may be -1 - j, row j of the previous step's sample, which the device fills in.
//
// Prefill runs a prompt's positions [a, a + M) of one slot in chunks of up to 8192 rows through the same layers
// with the kernels' prefill paths; it leaves the state the decode steps would have left, up to summation order.
// The MTP layer (78) keeps its own cache: prefill fills it for the prompt (no MoE), and each decode step's draft
// pass (the MTP layer for each row: token after it, the row's final hidden) writes the row's entry and drafts one
// token; further passes chain on the MTP's own hidden (Options::drafts).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comm/ring.h"
#include "model/config.h"
#include "weights/loader.h"

namespace glm::model {

inline constexpr int kTopLogprobs = 20;  // top logprobs a row, at most
inline constexpr int kMaxDrafts = 4;     // chained MTP draft passes a step, at most
inline constexpr int kMaxPages = 65536;  // Options::slots with paged caches (Options::page), at most

// How a session's rows are sampled; every rank must pass the same.
struct Sampling {
  float temperature = 0.f;  // 0: greedy (ties to the lowest id)
  float top_p = 1.f;        // 1: off
  int32_t top_k = 0;        // 0: off
  int32_t n_top = 0;        // top logprobs a row, 0..kTopLogprobs
  uint64_t seed = 0;
};

struct SampleRow {
  int32_t token;
  float logprob;   // of token under softmax(logits / T) (T = 1 when greedy), before top-k / top-p
  float value;     // what won: the logit (greedy, top-k / top-p) or the Gumbel-perturbed logit / T
  uint32_t flags;  // 1: the kept candidates reached past the exactly known ones (top-k / top-p)
};
// One step's result in pinned memory, the same on every rank.
struct SampleOut {
  uint32_t seq;   // the step's number + 1, written last
  int32_t rows;
  int32_t n_top;  // the most top logprobs of any session's
  int32_t drafts;  // MTP draft passes run (Options::drafts)
  SampleRow row[kMaxRows];
  int32_t top_id[kMaxRows][kTopLogprobs];
  float top_lp[kMaxRows][kTopLogprobs];
  int32_t draft[kMaxRows][kMaxDrafts];  // pass k's greedy draft for row r (the token after position pos[r] + k + 1)
  uint32_t ctl;  // rank 0's control word as the step's sampler read it (Model::set_ctl), the same on every rank
};

// A step: R rows; row r is position pos[r] of the session in slot slot[r], fed tokens[r] (>= 0, or -1 - j: row j
// of the previous step's sample); sess[r] numbers the sessions in row order from 0, session k sampled by sp[k].
struct Batch {
  int R = 0, nsess = 1;
  int32_t tokens[kMaxRows] = {}, pos[kMaxRows] = {}, slot[kMaxRows] = {}, sess[kMaxRows] = {};
  Sampling sp[kMaxSess];
  uint32_t flags = 0;
};
enum : uint32_t {
  kFlagLogits = 1,  // eager: all-gather the f32 logits; rank 0 hands them to Sink::logits
  kFlagDump = 2,    // eager: rank 0 hands each layer's residual output to Sink::tensor
  kFlagMtpLogits = 4,  // eager: the first draft pass's logits too (Sink::logits with mtp = true)
};
// Rank 0's view into an eager step or chunk.
struct Sink {
  virtual ~Sink() = default;
  // rows R of row_bytes: the tensor `name` of layer L ("hidden": the residual after layer L, bf16; L = -1:
  // "final": after the final norm). pos: the rows' positions.
  virtual void tensor(int L, const char* name, const int32_t* pos, int R, const void* data, size_t row_bytes) {}
  virtual void logits(const int32_t* pos, int R, const float* full, bool mtp) {}  // [R][kVocab] f32
};

// The decode step's exact speed levers (docs/engine.md, "Closing the decode gap"): each changes how the step runs,
// never what it computes (the same bits; glm_run's ab mode checks it). Options::tune, Model::set_tune.
enum Tune : uint32_t {
  kTuneNone = 0,
  kTuneRoute = 1u << 0,  // the routing inside the expert kernels (no route() launch)
  kTunePost = 1u << 1,   // the experts' down kernel posts its partials to the ring itself (the all-reduce: consumer half)
  kTuneIw = 1u << 2,     // full layers: the indexer's head weights in the x-projection's launch (attn's GemvIw)
  kTuneOproj = 1u << 3,  // o_proj posts its partials to the ring itself (oproj.cu): measured neutral, off by default
  kTuneArRow = 1u << 4,  // the fused all-reduces with one block a row (comm::glm slices = 1)
  kTuneDefault = kTuneRoute | kTunePost | kTuneIw | kTuneArRow,
  kTuneAll = 0xffffffffu,
};

struct Options {
  uint32_t tune = kTuneDefault;  // the levers on (Tune bits)
  int max_ctx = 65536;   // positions a slot's caches hold (<= the RoPE table's 1,048,576)
  int slots = 1;         // session slots: each one a whole cache of max_ctx positions
  // Positions from one slot's caches to the next (0: max_ctx, a whole cache a slot). With page < max_ctx the caches
  // are a pool of `slots` pages (at most kMaxPages) of `page` positions: a session in slot s runs on into pages
  // s + 1, s + 2, ... as far as its positions go (position p of slot s is pool position s * page + p), so a server
  // gives each session a run of pages of its own (engine/server, docs/serving.md).
  int page = 0;
  bool fp8_index_keys = false;  // the indexer keys as e4m3 + a scale (132 B), not BF16 (docs/attention.md)
  bool graphs = true;    // replay steps as one CUDA graph per row count (eager steps: flags, profile)
  bool pdl = true;       // programmatic dependent launch between the step's kernels
  bool l2pf = true;      // the paced L2 prefetch of the next weights during collectives and attention
  int drafts = 0;        // MTP draft passes after each step's sample, 0..kMaxDrafts (0: no MTP work at all)
  int prefill_rows = 8192;  // rows a prefill chunk, at most (<= kMaxChunk)
  int attn_splits = 0;   // sparse attention's split-K at decode (0: 16 up to 2 rows, 8 above)
  size_t index_scratch = 512ull << 20;  // the prefill indexer's score scratch, bytes
};

// Synthetic weights (no file): every tensor in the kernels' layouts with hashed contents, scaled so a forward
// stays finite. expert_pools distinct routed-expert pools (NVFP4 or the S3 code), layer l using pool l % n.
struct SynthOptions {
  uint64_t seed = 1;
  int expert_pools = 16;
  bool s3 = false;  // the E2M1-subset 3-bit code instead of NVFP4
};

// Per-kernel-class timing of one eager step (Model::profile).
enum KClass : int {
  kKEmbed, kKXproj, kKAgX, kKPrep, kKQb, kKAgQ, kKIndex, kKUk, kKAttn, kKUv, kKOproj, kKAr1, kKMlp, kKAr2,
  kKHead, kKSample, kKMtp, kKClasses
};
const char* kclass_name(int c);
struct Profile {
  double ms[kKClasses] = {};          // summed over the step's layers
  double layer_ms[kCacheLayers] = {};  // each layer's span (main layers; kMtpLayer: the draft passes)
  double total_ms = 0;
};

class Model {
 public:
  // CUDA must be initialised. Allocates the caches and step buffers.
  Model(int rank, const Options& opt);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Binds a loaded rank file (routed experts filled: every MoE layer's pool and descriptors) and the ring.
  // fill_experts: a file whose routed experts are still reserved runs with synthetic pools instead (layers 0-2 and
  // the attention then real, everything after layer 3's MoE not)
  void bind(glw::WeightFile& wf, comm::Ring& ring, const SynthOptions* fill_experts = nullptr);
  // Synthetic weights instead (allocated here); every rank must pass the same options.
  void bind_synthetic(const SynthOptions& so, comm::Ring& ring);
  // Fill slot `slot`'s caches at positions [0, n) with hashed values (benches: a context without a prefill).
  void fill_cache(int slot, int n, uint64_t seed);

  // Queue a step without waiting for it; returns its number. Flags make it eager (it has finished on return).
  uint64_t launch(const Batch& b, Sink* sink = nullptr);
  // The same for one session: R rows at pos0 .. pos0 + R - 1 of `slot`.
  uint64_t launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, int slot = 0, uint32_t flags = 0,
                  Sink* sink = nullptr);
  // Spin until step n's sample is in (valid until step n + 4 is launched).
  const SampleOut& wait(uint64_t n);

  // Prefill positions [a, a + M) of `slot` (tokens[0 .. M)), chunk by chunk. With sp the last row is sampled (and
  // the draft passes run on it): returns that step's number (wait() for it), else ~0. A prompt may come in several
  // calls, each starting where the one before ended. flags: kFlagLogits / kFlagDump / kFlagMtpLogits hand every row's
  // results to sink (the head over every row: slow).
  uint64_t prefill(int slot, int a, int M, const int32_t* tokens, const Sampling* sp = nullptr, uint32_t flags = 0,
                   Sink* sink = nullptr);
  // Forget slot's MTP pending row (a new sequence in the slot).
  void reset_slot(int slot);

  // ---- serving (engine/server, docs/serving.md) ----
  // Rank 0's control word: every step's sampler reads it when it runs and hands it to every rank (SampleOut::ctl),
  // so that a decode loop running ahead of the host stops after the same step on all ranks (a cancel, a pause).
  void set_ctl(uint32_t v);
  // The prefill path's buffers now, not at the first prefill (a server sizes its cache pool after them).
  void prefill_init();
  // The caches again with `slots` slots (pages), contents zeroed: before anything is kept in them (drops the
  // captured graphs and every slot's MTP pending row). Every rank must make the same call.
  void resize_slots(int slots);
  // Bytes one position of a slot's caches takes on this rank: every layer's KV entry, the index keys (and scales).
  size_t cache_pos_bytes() const;
  // Positions [0, n) of slot src's caches to slot dst's (queued on the model's stream; the two ranges may overlap).
  void cache_copy(int dst, int src, int n);
  // Positions [a, a + n) of slot's caches to / from host memory (pinned for speed): cache by cache (layers 0-78's
  // KV, then the index keys, then their scales), each a block of `block` >= n positions starting at the block's
  // first position; cache_pos_bytes() * block bytes in all. Synchronous.
  void cache_get(int slot, int a, int n, void* host, int block) const;
  void cache_put(int slot, int a, int n, const void* host, int block);

  // The GPU time of step n (one of the last 4 launched; waits for it), and of the last prefill call.
  float gpu_ms(uint64_t n) const;
  float prefill_ms() const;
  // One eager step of R rows timed per kernel class (no PDL, an event between classes): the same work as the
  // graph, slower by the events.
  Profile profile(const Batch& b);
  // Hashes of slot's caches at positions [0, n): KV of every layer, index keys, the MTP layer's (state checks).
  std::vector<uint64_t> cache_hashes(int slot, int n) const;
  // The final residual and normed hidden of the last step's or chunk's rows (host copies; checks).
  std::vector<uint16_t> last_hidden(int R) const;
  void drain();
  int file_expert_layers() const;  // MoE layers whose routed experts came from the file (the rest synthetic)
  double launch_us() const;  // the last launch's host time (us), not counting a wait for a free input slot
  int graph_count() const;
  int rank() const;
  const Options& options() const;
  size_t weight_bytes() const;  // the device bytes the bound weights take
  // The weight bytes a decode step reads on this rank with `experts` distinct routed experts a MoE layer (the
  // rows' union), without and with the MTP passes: the floor's numerator.
  double step_bytes(double experts, bool with_mtp = false) const;
  // Turn the paced L2 prefetch on or off (drops the captured graphs; Options::l2pf must have made its stream).
  void set_l2pf(bool on);
  // Set the speed levers (Tune bits; each set's captured graphs are kept).
  void set_tune(uint32_t bits);

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

}  // namespace glm::model
