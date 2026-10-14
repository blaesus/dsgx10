// Rank 0's commands to every rank (docs/serving.md): what the server (serve.cpp) tells the other ranks, and the
// parts every rank runs alike: prefill chunks, the decode loop over 1..16 sessions (plain, or speculative with the
// MTP layer's drafts: spec_loop), the cache pool's moves, the disk tier's spills and reads, a planned stop's kept
// sessions. The DeepSeek engine's cmd.h re-specialised: no window snapshots, no Engram; a session is a run of pages
// of the paged cache (Options::page).
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "comm/ring.h"
#include "model/model.h"
#include "model/spec.h"
#include "server/disk.h"

namespace srv {
namespace model = ::glm::model;

double now_s();
void pin_self(int cpu);

// Aborts the process when a step hangs (a collective waiting on a dead peer spins forever).
struct Watchdog {
  std::atomic<double> started{0};
  std::atomic<bool> stop{false};
  std::thread th;
  Watchdog(int timeout_s, comm::Ring& ring);
  ~Watchdog();
};

// Cache operations every rank applies before a command's own work, in order.
enum : int32_t {
  kOpCopy = 1,   // positions [0, n) of slot b's caches to slot a's (Model::cache_copy; overlapping runs allowed)
  kOpReset = 2,  // slot a starts a new sequence (Model::reset_slot)
};
struct Op {
  int32_t op = 0, a = 0, b = 0, n = 0;
};
constexpr int kMaxOps = 16;
constexpr int kMaxStop = 32;
constexpr int kMaxOracle = 512;

enum : int32_t {
  kCmdStop = 0,     // leave the command loop
  kCmdOps = 1,      // nothing but the operations
  kCmdPrefill = 2,  // positions [pos0, pos0 + rows) of `slot`: the tokens follow as a payload; kPfSample: sample the last
  kCmdBatch = 3,    // batch_loop: n sessions decoding together; their BatchEntry's follow as a payload
  kCmdSpill = 4,    // positions [0, rows) of `slot` to the disk tier's pages (n of them, a payload of int32)
  kCmdUnspill = 5,  // and back from them
  kCmdKeep = 6,     // a planned stop: the disk tier synced and indexed with generation gen (Disk::keep)
};
enum : uint32_t { kPfSample = 1 };
enum : uint32_t { kBatchSpec = 1 };  // kCmdBatch's flags: spec_loop (the BatchEntry's drafts and gates)
struct Cmd {
  int32_t op = kCmdStop;
  int32_t slot = 0, pos0 = 0, rows = 0;
  uint32_t flags = 0;
  int32_t n = 0;  // the payload's elements
  model::Sampling sp{};
  uint64_t gen = 0;
  int32_t n_ops = 0;
  Op ops[kMaxOps] = {};
};

// rank 0's c to every rank (Ring::allgather_host on the bootstrap sockets: ~0.1-2 ms, once per chunk or run)
Cmd bcast(comm::Ring& ring, const Cmd& mine);
// rank 0's payload of nbytes to every rank, in pieces of 64 KiB (the bootstrap sockets' limit)
void bcast_payload(comm::Ring& ring, void* data, size_t nbytes, int rank);
void apply_ops(model::Model& m, const Cmd& c);

// A spill or a read of one session on this rank: positions [0, positions) of `slot` and its disk pages, a page of
// the pool (Options::page positions) to a page of the file.
void spill(model::Model& m, Disk& d, int slot, int positions, const std::vector<int32_t>& pages);
void unspill(model::Model& m, Disk& d, int slot, int positions, const std::vector<int32_t>& pages);

// The decode loop (docs/serving.md; the DeepSeek engine's batch_loop, B1/B2): entries decoding together, each in
// its own slot with its own stop ids and sampling, one row each per step (Model::launch). Entry k's step j feeds
// tok_j at pos0 + j (tok_0 = token) and samples tok_{j+1}; the next step is queued before a step ends, with its
// tokens left to the device, and an entry that ends keeps its row in the step already queued, which only adds the
// KV of its last sampled token. Rank 0's control word (SampleOut::ctl, the same on every rank): bit k ends entry k
// after the step that reads it (a cancel); kCtlPause ends every entry that its stop ids and max_new do not end
// (kEndPause). A paused entry's row in the queued step is sampled too, within max_new.
enum : uint32_t { kEndStop, kEndLength, kEndCancel, kEndContext, kEndPause };
constexpr uint32_t kCtlPause = 1u << 31;
struct BatchEntry {
  int32_t slot = 0;
  int32_t pos0 = 0, token = 0;
  int32_t max_new = 0;  // tokens to sample (the loop may compute one more position)
  int32_t limit = 0;    // positions below this may be computed (the session's pages)
  int32_t n_stop = 0;
  int32_t stop[kMaxStop] = {};
  model::Sampling sp{};
  // spec_loop: the drafts of tok_0's successors (the step that sampled tok_0 drafted them) and the session's gate
  int32_t n_draft = 0;
  int32_t draft[model::kMaxDrafts] = {};
  model::SpecGate gate;
  // tests (--spec-oracle): tok_1, tok_2, ... as an earlier identical request gave them; spec_loop's drafts are these
  // (about one in four changed), so verify steps accept drafts whatever the MTP's quality
  int32_t n_oracle = 0;
  int32_t oracle[kMaxOracle] = {};
};
struct BatchEnd {
  int sampled = 0;  // tokens sampled: tok_1 .. tok_sampled
  int fed = 0;      // positions pos0 .. pos0 + fed - 1 hold the KV of tok_0 .. tok_{fed-1} now
  uint32_t why = kEndStop;
  // spec_loop: the drafts after tok_sampled (none when the run's last step emitted past what it kept) and the gate
  int32_t n_draft = 0;
  int32_t draft[model::kMaxDrafts] = {};
  model::SpecGate gate;
  long steps = 0, verified = 0, accepted = 0;  // spec_loop: verify steps the entry was in, drafts fed, accepted
};
using OnBatch = std::function<void(int k, const model::SampleOut& o, int row)>;  // entry k's next token: o.row[row]
std::vector<BatchEnd> batch_loop(model::Model& m, const std::vector<BatchEntry>& e, Watchdog& wd, const OnBatch& on,
                                 std::vector<double>* step_t = nullptr);

// Speculative decoding (docs/serving.md, "Drafts"; docs/engine.md): the same entries, ends and callbacks, but one
// verify step at a time (Batch::spec): entry k feeds tok_0 and the k_j drafts its gate chooses (SpecGate over
// cost) as consecutive rows, and emits the samples of its rows up to the first draft that differs from the sample
// before it (the accepted ones and the bonus), so the tokens are batch_loop's, bit for bit. The step drafts the
// next drafts (Batch::passes: the most any gate may want, kmax). Positions written stay below each entry's limit:
// rows at pos .. pos + k_j and the chained draft passes' MTP entries up to pos + k_j + passes - 1.
struct SpecConf;
std::vector<BatchEnd> spec_loop(model::Model& m, const std::vector<BatchEntry>& e, Watchdog& wd, const OnBatch& on,
                                const SpecConf& sc, std::vector<double>* step_t = nullptr);

// Ranks 1-3: run rank 0's commands until kCmdStop.
void follow(model::Model& m, comm::Ring& ring, Watchdog& wd, Disk* disk, uint64_t run_id);

// The speculative decoding the server runs (main.cpp; every rank the same): kmax drafts a session at most (0: off,
// --no-spec), the gate's cost model.
struct SpecConf {
  int kmax = 0;
  model::SpecCost cost;
  int force = 0;  // > 0 (tests, --spec-force): every step feeds min(force, drafts there are), whatever the gate says
  bool oracle = false;  // tests (--spec-oracle): BatchEntry::oracle's drafts where an earlier request left them
};
extern SpecConf g_spec;

}  // namespace srv
