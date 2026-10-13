// The TP=4 forward of DeepSeek-V4.1-Flash on one rank (docs/engine.md). Every rank holds a Model and
// runs the same steps in the same order; the residual stream, the routing and the index choices are
// bit-identical on all four (collectives sum in rank order). A step takes R <= 8 consecutive positions:
// a decode step (R = 1), a verify step (R <= 6) or a prefill chunk (exact mode: every layer, every row).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "comm/ring.h"
#include "engram/engram.h"
#include "model/config.h"
#include "model/ops.h"
#include "weights/loader.h"

namespace model {

// How a step's rows are sampled (ops::StepIn); every rank must pass the same.
struct Sampling {
  float temperature = 0.f;  // 0: greedy
  float top_p = 1.f;        // 1: off
  int32_t top_k = 0;        // 0: off
  int32_t n_top = 0;        // top logprobs per row, 0..ops::kTopLogprobs
  uint64_t seed = 0;
};
using ops::SampleOut;

// A step of rows from several sessions (docs/batching.md, B2; Model::launch_batch). Row r is position
// pos[r] of the session in slot slot[r], fed tokens[r] (>= 0, or -1 - j: row j of the previous step's
// sample, as launch()'s). A session's rows are consecutive and ascend one position at a time; sess[r]
// numbers the step's sessions in row order from 0, each in a slot of its own, and session k is sampled
// by sp[k].
struct Batch {
  int R = 0, nsess = 0;
  int32_t tokens[kMaxRows] = {}, pos[kMaxRows] = {}, slot[kMaxRows] = {}, sess[kMaxRows] = {};
  Sampling sp[ops::kMaxSess];
};

// What a spec step (spec.cu) lets its draft pass choose for the step after it: by default draft j is
// verified while the confidence head's survival[j - 1] (P(d_1 .. d_j all accepted)) >= thresh, at most
// max_drafts of them, every row below pos_end; the other gates (ops.h kGate*) use step_ms. Every rank
// must pass the same.
struct SpecCfg {
  float thresh = 0.21f;
  int max_drafts = 5;  // 0..5
  int pos_end = 0;
  int force = -1;  // >= 0: draft from this token instead of the step's last sample (teacher-forced checks)
  int probe[ops::kDrafts] = {-1, -1, -1, -1, -1};  // EXPERIMENT (Options::tree_probe): the probe's path
  int gate = ops::kGateThresh;
  float step_ms[ops::kDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f};  // 0..5 drafts
};

// A batch spec step (docs/batching.md, B3: drafts across sessions; Model::launch_spec_batch and
// launch_verify_batch): n entries, each one session in a slot of its own, taking part when on[k]. In a
// given step entry k is one row, token[k] at pos0[k]; in a verify step its rows are what the step before
// chose for it (its bonus token, then at most cap[k] drafts, every row below pos_end[k], thresh[k] its
// kGateThresh threshold; cfg's gate, step_ms, force and probe serve every entry, its thresh, max_drafts
// and pos_end are unused). The entries on, in entry order, make the step's rows; the caller keeps their
// next rows (1 + cap[k] each) within kMaxRows. Every rank must pass the same.
struct SpecBatch {
  int n = 0;
  int32_t on[ops::kMaxSess] = {}, slot[ops::kMaxSess] = {}, token[ops::kMaxSess] = {}, pos0[ops::kMaxSess] = {};
  int32_t cap[ops::kMaxSess] = {}, pos_end[ops::kMaxSess] = {};
  float thresh[ops::kMaxSess] = {};
  Sampling sp[ops::kMaxSess];
  SpecCfg cfg;
  int pos_bound = 0;  // launch_verify_batch: the rows' last position at most (the context bucket)
};

// One step, as rank 0 decided it (the golden and bench modes: every rank receives the same).
struct Ctl {
  int32_t op = 0;     // kOpStep / kOpStop
  int32_t rows = 0;   // R: positions pos0 .. pos0 + R - 1
  int32_t pos0 = 0;
  uint32_t flags = 0;
  int32_t tokens[kMaxRows] = {};
  float temperature = 0.f;  // 0: greedy; else Gumbel-max with `seed`
  uint32_t seed = 0;
  int32_t pad[2] = {};
};
enum : int32_t { kOpStop = 0, kOpStep = 1 };
enum : uint32_t {
  kFlagInject = 1,     // before layers >= 1: residual and pre_mix := rank 0's Sink::fill (golden inputs)
  kFlagDump = 2,       // rank 0 hands each layer's results to Sink::tensor
  kFlagLogits = 4,     // all-gather the full logits; rank 0 hands them to Sink::logits
  kFlagInjectSub = 8,  // also the FFN's input stream (mid), after it was dumped
  kFlagProfile = 16,   // every rank starts the CUDA profiler's capture range (nsys) before this step
};

// Rank 0's view into a step (eager steps only), for the golden comparison.
struct Sink {
  virtual ~Sink() = default;
  // R rows of row_bytes: the golden `name` of layer L at positions pos0 .. : "x" (the block input, i.e.
  // layer L-1's out), "pre_mix" (L-1's), "attn_in", "mid", "ffn_in" (bf16 bits, pre_mix fp32).
  virtual void fill(int L, const char* name, int pos0, int R, void* host, size_t row_bytes) = 0;
  // A step tensor on the host, R rows of row_bytes. L = -1 for the final ones.
  virtual void tensor(int L, const char* name, int pos0, int R, const void* data, size_t row_bytes) = 0;
  virtual void logits(int pos0, int R, const float* full) = 0;  // [R][kVocab]
};

// Contexts in steps of kCtxBucket positions (the graph sweep, spec_graphs). The step graphs do not depend on
// the context: index_dense reads the scan's length from the positions (impl.h, scan_ctx).
inline constexpr int kCtxBucket = 4096;

struct Options {
  int max_ctx = 65536;      // positions the RoPE tables and compressed caches cover
  int n_layers = kLayers;   // run only blocks [first_layer, n_layers) (tests with golden injection; the
  int first_layer = 0;      // logits are then meaningless)
  bool graphs = true;       // replay steps without golden checks as CUDA graphs (one per row count and
                            // context bucket, captured on first use)
  int prefill_rows = 4096;  // rows per prefill chunk, at most
  int wave_probe = 0;       // probe: bit L - layer L's FFN beside a second attention side, in the chunk
  int wave_probe_at = -1;  // at this position (wave_probe())
  bool draft_fp8 = false;   // DSpark's head GEMV on an FP8 copy of the vocab head (bind_spec; +165 MB): the
                            // drafts' arithmetic changes, the tokens cannot
  // The draft pass's cheaper formats (docs/engine.md, "The draft pass's approximations"; bundle B), drafts only:
  bool draft_head4 = false;    // the head as MXFP4, made from the FP8 copy (dspark::head_fp4; +88 MB, the FP8
                               // copy freed): the study's simulated FP4 head bit for bit
  bool draft_markov8 = false;  // the Markov head as FP8 (dspark::markov_fp8; +8.3 MB)
  int slots = 1;            // session slots (docs/batching.md): live sessions' states side by side (each one's
                            // window rings, compressed caches, index keys, carries and draft windows), up to
                            // kMaxSlots; ~240 MB a slot at 262,144 positions
  bool l2pf = true;         // the decode step's paced L2 prefetch in its windows (kernels/l2pf.cu, docs/engine.md
                            // "A paced L2 prefetch"): a side stream, the same bits
  // EXPERIMENT, bench only (docs/engine.md, "Draft trees"): every spec step's draft pass also runs the
  // drafter's probe (dspark::Draft::probe) along SpecCfg::probe, into SampleOut::probe
  bool tree_probe = false;
};

class Model {
 public:
  // CUDA must be initialised. Allocates the state (do this before reading weights).
  Model(int rank, const Options& opt);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Binds the loaded weights: re-lays a version-1 file into the kernels' layouts (tile_fp8 / tile_fp4)
  // and all-gathers the mHC projections. Collective: every rank calls it.
  void bind(dsw::WeightFile& wf, comm::Ring& ring);
  void bind_ring2(comm::Ring& ring);  // --wave-probe
  // The vocab head coded in place (dsk::bf16x_encode: each tile of 8 rows x 32 values with its exponents as
  // 4-bit offsets, 3/4 of the bytes; checked against the head before it replaces it): every reader of the
  // head moves to the coded GEMVs, which give the same bits. After bind and bind_spec (DSpark's FP8 copy is
  // made from the raw head), before the first step. Returns the coded bytes (in the head's place; the rest
  // of its slot stays allocated), or 0 if coding would not make it smaller (the head stays raw);
  // *raw_tiles: the tiles kept raw.
  size_t code_head(int* raw_tiles = nullptr);
  // The Engram row store of this rank (owned by the caller).
  void set_engram(const engram::HashMap* map, engram::Store* store);
  // Session slots (Options::slots). From now on the eager paths (prefill chunks, the state operations,
  // snapshots) act on slot s, and the steps launched have their rows in slot s. Every rank must make the
  // same calls. Slot 0 at the start.
  void use_slot(int s);
  int slot() const;
  int slots() const;

  // Queue one step of R rows at positions pos0 .. pos0 + R - 1 on every rank, without waiting for it;
  // returns its number. tokens[r] >= 0 is the token; -1 - j stands for row j of the previous step's
  // sample, which the device fills in (a decode step queued before the one before it has ended): call
  // engram_late() with the real tokens once they are known. Steps with golden checks (flags) run
  // eagerly and finish before this returns.
  uint64_t launch(int R, int pos0, const int32_t* tokens, const Sampling& sp, uint32_t flags = 0, Sink* sink = nullptr);
  // The history and Engram lookups of a launched step whose tokens were left to the device (a verify
  // step: its positions and rows too, SampleOut::next_* of the step before it).
  void engram_late(int pos0, int R, const int32_t* tokens);
  // A step of rows from several sessions (Batch; Options::slots > 1 for more than one): as launch(), with
  // each row in its own session's slot and Engram history, and each session's own sampling. Every row
  // computes what it would in a step of its session alone: the same tokens and logits, bit for bit.
  uint64_t launch_batch(const Batch& b);
  // engram_late() of a batch step: its rows' tokens, in row order.
  void engram_late_batch(const int32_t* tokens);
  // DSpark, the model's draft module (spec.cu; wf must hold mtp.*, embed and head). A spec step runs its
  // rows with main_hidden taken at layers 37-39, samples them, accept()s its final rows into the draft
  // module's windows, drafts from the token after them, and chooses the next step's rows: that token,
  // then the drafts worth verifying (SpecCfg). launch_spec: R rows from the host, all final;
  // launch_verify: the rows the step before chose (queued before it ends; pos_bound: their last
  // position at most, for the context bucket). A verify step's SampleOut holds the tokens it emits:
  // rows (1 + the drafts accepted) of row[], then next_* for the step after it.
  void bind_spec(dsw::WeightFile& wf);
  // the spec step's graph of every context bucket, captured now rather than at its first step (serve's
  // startup: ~0.2 GiB of device memory each)
  void spec_graphs();
  bool spec() const;
  uint64_t launch_spec(int R, int pos0, const int32_t* tokens, const Sampling& sp, const SpecCfg& c);
  uint64_t launch_verify(const Sampling& sp, const SpecCfg& c, int pos_bound);
  // B3 (docs/batching.md): drafts across sessions, a spec step of several sessions' rows (SpecBatch;
  // Options::slots > 1). launch_spec_batch: a given step, one row per entry on; launch_verify_batch: the
  // rows the step before chose for each entry on (queued before it ends). The step samples its rows with
  // the verdicts per entry and runs each entry's draft pass in its own slot; every row computes what it
  // would in a step of its session alone, bit for bit. SampleOut: sess_* per entry (its rows, the tokens
  // it emits), next_s* for the step after (0 rows: the entry was off). spec_batch_entries(): entries a
  // batch step takes at most (0 without DSpark or slots).
  int spec_batch_entries() const;
  uint64_t launch_spec_batch(const SpecBatch& b);
  uint64_t launch_verify_batch(const SpecBatch& b);
  // engram_late() of a batch verify step: its rows as the device laid them out (the entries on in entry
  // order, each its next_s* of the step before): their tokens, positions and slots.
  void engram_late_rows(int R, const int32_t* tokens, const int32_t* pos, const int32_t* slot);
  // Spin until step n's sample is in. The result stays valid until step n + ops::kOutSlots is launched.
  const SampleOut& wait(uint64_t n);
  // Rank 0's control word: every step's sampling reads it when it runs and hands it to every rank
  // (SampleOut::ctl), e.g. to stop a decode loop that runs ahead of the host on all ranks at once.
  void set_ctl(uint32_t v);
  // Prefill: positions [a, a + M) (M <= Options::prefill_rows) through every layer, with the tokens of
  // set_tokens(). Window snapshots (state_snap) of session `handle` at the positions `snaps`, all in
  // (a, a + M], are taken on the way. With sp the last row is sampled: returns its step number (wait()
  // for it), else ~0. prefill_lookup() starts a chunk's Engram lookups early (the next chunk's, while
  // this one runs); prefill() starts them itself when they were not.
  // With ced (docs/ced.md) the chunk runs the encoder (layers 0-19) and layer 20's KV projection only;
  // the chunk that samples then replays layers 20-39 over its last 128 positions [P - 128, P) from
  // fresh decoder rings, and samples that replay's last row. Those 128 positions must have run through
  // CED chunks of one run (each chunk starting where the one before ended), P >= 128. Of the snapshots
  // only one at P covers the decoder layers; the others hold the encoder layers' rings only.
  void set_tokens(int pos0, int n, const int32_t* tokens);
  void prefill_lookup(int a, int M);
  // The prefill path's buffers and the weights it gathers, made now instead of by the first prefill:
  // every rank at the same point, after set_engram(). This rank's allocations come before ring's host
  // barrier and the weights' all-gathers after it, so a rank that cannot allocate stops every rank at
  // the barrier. Returns this rank's free device memory afterwards.
  size_t prefill_init(comm::Ring& ring);
  uint64_t prefill(int a, int M, const Sampling* sp = nullptr, int handle = -1, const std::vector<int>& snaps = {},
                   bool ced = false);
  // The last prefill chunk's GPU time in ms (waits for the chunk): [0] the encoder (layers 0-19; every
  // layer without ced), [1] CED's projection, keep and replay, [2] the head and the sampling, [3] the
  // draft module's windows.
  void prefill_split(float ms[4]) const;
  // Marks on the model's stream (i < 4), and the GPU time from mark i to mark j (waits for j).
  void mark(int i);
  float mark_ms(int i, int j) const;

  // Sessions' saved state: every rank keeps the same store (rank 0 decides; commands make every rank do
  // the same). Session h holds its compressed caches up to some position, window snapshots (ring
  // positions [pos - 128, pos) of every layer) and an end snapshot (the whole ring and the compressor
  // carries). Copies are queued on the model's stream, in order with the steps.
  // The store's memory is a pool of `bytes` (state_reserve, once, the same on every rank; none: nothing
  // is kept). An operation that finds it full evicts the least recently used session that the current
  // command does not use (state_pin: its operations' sessions and its handle), the same on every rank,
  // and skips what still does not fit: a snapshot not taken, pages saved only as far as they go.
  // state_evicted() returns the sessions evicted since it was last called.
  void state_reserve(size_t bytes);
  size_t state_capacity() const;
  void state_pin(const std::vector<int>& h);
  std::vector<int> state_evicted();
  void state_save(int h, int from, int to);    // compressed caches of positions [from, to) -> session h
  void state_load(int h, int to);              // session h's compressed caches of positions [0, to) -> working
  void state_snap(int h, int pos);             // ring -> a snapshot at pos (pos - 128 .. pos - 1 still in the ring)
  void state_snap_end(int h, int pos);         // ring and carries -> session h's end snapshot (at pos)
  void state_restore(int h, int pos);          // snapshot at pos -> ring
  void state_restore_end(int h);               // the end snapshot -> ring and carries
  void state_drop(int h, int from);            // forget what session h holds from position `from` on
  void state_free(int h);
  void state_copy_snaps(int src, int dst, int upto);  // session src's window snapshots at <= upto -> dst
  void state_drop_snap(int h, int pos);
  // The store's disk tier: base.spill of `bytes` (the same on every rank; an earlier file is removed
  // first, and this one at exit), with `staging` bytes of pinned memory for writes in flight. A session
  // the store evicts is then spilled to it: its blocks are copied out after the work queued on the
  // model's stream and freed at once, and a writer thread writes them. state_unspill, or an operation
  // that needs a spilled session's blocks, reads it back, as far as the store has room. When the file is
  // full, its least recently used sessions are dropped (state_evicted).
  // Kept sessions (docs/engine.md): state_keep(gen), on every rank at a planned stop, syncs the file and
  // writes base.index (the sessions in the file: their blocks and checksums, the run id, gen), and the
  // file stays at exit. The next start's state_kept_check reads this rank's index: gen when it is whole,
  // of this run_id (main.cpp: the binary, the weights, the options), at most max_age_s old and beside
  // its file (whose size goes to *bytes), else 0 (*why: the reason, "" when there is none). With reuse
  // (every rank's check returned the same gen) state_disk opens that file as it is, and its sessions
  // are spilled ones; either way the index is removed, so that a crash from then on starts clean. A
  // block of a kept session that comes back different from what went out stops the process.
  uint64_t state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes, std::string* why);
  void state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse = false, uint64_t run_id = 0);
  void state_keep(uint64_t gen);
  uint64_t state_kept() const;              // the gen whose sessions this run's store holds (0: none)
  std::vector<int> state_sessions() const;  // the sessions in the store
  size_t state_disk_capacity() const;
  size_t state_disk_bytes() const;  // the file's bytes in spilled sessions
  std::string state_disk_path() const;
  void state_spill(int h);    // session h's state to the disk tier (freed without one: state_evicted)
  void state_unspill(int h);  // and back
  struct StateInfo {
    int saved = 0;           // compressed caches of positions [0, saved) are kept
    int end_pos = -1;        // the end snapshot's position (-1: none)
    std::vector<int> snaps;  // window snapshot positions, ascending
    size_t bytes = 0;        // in the store's device pool
    bool on_disk = false;    // spilled: its blocks are in the disk tier (disk_bytes), none in the pool
    size_t disk_bytes = 0;
  };
  StateInfo state_info(int h) const;
  size_t state_bytes() const;                  // device memory the store holds
  // Wait for everything queued.
  void drain();

  // launch + wait: sampled [R] (host) gets the chosen token of each row, the same everywhere.
  void step(const Ctl& c, int32_t* sampled, Sink* sink = nullptr);
  // Forget the sequence (positions restart at 0).
  void reset();
  // Bench only: the rows of a step of R rows ending at P = pos0 + R - 1 sit at P - (R - 1 - r) * stride
  // instead of P - (R - 1 - r) (1, the default), so stride >= 128 puts each row's window apart. With
  // group G > 1 they come in runs of G consecutive positions (a session's verify rows), the runs stride
  // apart: row r at P - (d / G) * stride - d % G, d = R - 1 - r. The state the steps leave is not a
  // sequence's; use it for timing only.
  void set_pos_stride(int stride, int group = 1);
  // Bench only: row r's router logits rotated by shift * (r / group) experts before the top-6 (0, the
  // default: none), so the rows (or runs of group rows) of a step route about as apart as rows of
  // unrelated sequences could. Drops the captured graphs (they are captured again, with or without it).
  void set_route_shift(int shift, int group = 1);
  // The GPU time of the last step waited for, from its first kernel to its last.
  float last_gpu_ms() const;
  // The same of step n, one of the last ops::kOutSlots launched (waits for it to finish).
  float gpu_ms(uint64_t n) const;
  // The value that won row r's sampling in the last step waited for (comparing runs bit for bit).
  float last_logit(int r) const;
  // Step graphs instantiated so far.
  int graph_count() const;

  int rank() const;
  cudaStream_t stream() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> d_;
};

}  // namespace model
