// Rank 0's commands to every rank (docs/engine.md): what the golden modes and the server (serve.cpp)
// tell the followers, and the parts every rank runs alike (the decode loop, prefill chunks, the
// sessions' state operations).
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "comm/ring.h"
#include "model/model.h"

namespace srv {

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

// A session's state operation (model.h state_*), applied by every rank in order.
enum : int32_t {
  kStSave = 1,     // state_save(h, a, b)
  kStSnapEnd,      // state_snap_end(h, a)
  kStLoad,         // state_load(h, a)
  kStRestore,      // state_restore(h, a)
  kStRestoreEnd,   // state_restore_end(h)
  kStDrop,         // state_drop(h, a)
  kStFree,         // state_free(h)
  kStCopySnaps,    // state_copy_snaps(h, a, b): h's snapshots at <= b into session a
  kStDropSnap,     // state_drop_snap(h, a)
  kStSnap,         // state_snap(h, a)
  kStSpill,        // state_spill(h)
  kStUnspill,      // state_unspill(h)
  kStKeep,         // state_keep(gen): a planned stop's kept sessions (gen = a | b << 32)
};
struct StateOp {
  int32_t op = 0, h = -1, a = 0, b = 0;
};
void apply_state(model::Model& m, const StateOp& o);

// What rank 0 tells every rank to do next (Ring::allgather_host on the bootstrap sockets, ~0.1-2 ms:
// once per request or chunk, never per decode step).
constexpr int kMaxStop = 32;
constexpr int kMaxSnaps = 8;
constexpr int kMaxOps = 12;
enum : int32_t {
  kCmdStop = 0,     // leave the command loop
  kCmdStep = 1,     // a golden-mode step (model::Ctl)
  kCmdDecode = 2,   // decode_loop
  kCmdTokens = 3,   // set_tokens: n tokens of positions pos0 .. follow as a payload
  kCmdPrefill = 4,  // a prefill chunk (run_prefill)
  kCmdState = 5,    // nothing but the state operations (every command applies ops[0 .. n_ops) first)
  kCmdRows = 6,     // a step of `rows` rows with explicit tokens, sampled with sp, then snaps[] from the ring
  kCmdBatch = 7,    // batch_loop: n sessions decoding together, their BatchEntry's follow as a payload
};
enum : uint32_t { kPfSample = 1, kPfCed = 2 };  // kPfCed: a CED chunk (Model::prefill's ced, docs/ced.md)
struct Cmd {
  int32_t op = kCmdStop;
  int32_t rows = 0, pos0 = 0;  // kCmdStep, kCmdRows, kCmdPrefill: the rows; kCmdDecode: the first position
  uint32_t flags = 0;
  // >= 0: the session slot (Model::use_slot) this command and the ones after it act on, from its state
  // operations on; -1: the one before
  int32_t slot = -1;
  // the store's handles of the sessions live in the slots (0: none): pinned like the command's own, so
  // that no allocation evicts the stored part of a session whose working state a slot holds
  int32_t live[model::kMaxSlots] = {};
  int32_t tokens[model::kMaxRows] = {};  // kCmdDecode: tokens[0] is the first step's
  model::Sampling sp{};
  int32_t max_new = 0;  // kCmdDecode: tokens to sample (the loop may compute one more position)
  int32_t limit = 0;    // kCmdDecode: positions below this may be computed
  int32_t n_stop = 0;   // kCmdDecode: ids that end the loop
  int32_t stop[kMaxStop] = {};
  int32_t ahead = 1;       // kCmdDecode: queue the next step before the current one ends (0: diagnostics)
  int32_t n = 0;           // kCmdTokens; kCmdBatch: the entries
  int32_t next = 0;        // kCmdPrefill: rows of the chunk after this one (its Engram lookups start now)
  int32_t handle = -1;     // the session whose window snapshots are taken
  int32_t snap_every = 0;  // kCmdDecode: a window snapshot at every multiple of this position (0: none)
  int32_t n_snaps = 0;     // kCmdPrefill: snapshots inside the chunk; kCmdRows: after the step
  int32_t snaps[kMaxSnaps] = {};
  int32_t n_ops = 0;
  StateOp ops[kMaxOps] = {};
  // DSpark (every rank bound it): kCmdDecode runs the speculative loop, kCmdRows a spec step
  int32_t spec = 0;
  int32_t spec_drafts = 5;   // drafts verified per step, at most
  float spec_thresh = -1.f;  // verify d_j while survival[j - 1] >= this; < 0: lambda * dC (SpecRate)
  float spec_cost[3] = {19.4f, 3.6f, 3.3f};  // ms: a step of one row, each further row (dC), a draft pass
  int32_t spec_force = -1;   // kCmdRows: draft from this token, not the sampled one (teacher-forced checks)
  int32_t spec_probe[model::ops::kDrafts] = {-1, -1, -1, -1, -1};  // kCmdRows, EXPERIMENT: SpecCfg::probe
  // the gate (model::ops kGate*): kGateThresh uses spec_thresh / spec_cost, the others spec_row_ms, the
  // ms of a step verifying 0..5 drafts with its draft pass (measured at 1k of context, docs/engine.md)
  int32_t spec_gate = model::ops::kGateThresh;
  float spec_row_ms[model::ops::kDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f};
  // kCmdDecode: the drafts' rate (SpecRate) to start from: a request decoded in slices goes on with the
  // last slice's (DecodeEnd::spec_tok, spec_ms)
  double spec_tok = 1, spec_ms = 0;
};

Cmd bcast(comm::Ring& ring, const Cmd& mine);
// c's state operations, on every rank before c's own work, with the sessions c uses pinned in the
// store until the next command: its operations', its handle and the slots' (model.h state_pin)
void apply_ops(model::Model& m, const Cmd& c);
// rank 0's payload of nbytes, to every rank (in pieces of 64 KiB, the bootstrap sockets' limit)
void bcast_payload(comm::Ring& ring, void* data, size_t nbytes, int rank);
model::Ctl to_ctl(const Cmd& c);

// A prefill chunk on this rank (every rank alike): positions [pos0, pos0 + rows) of the tokens set
// before; with kPfSample the last row is sampled (the step's number), else ~0.
uint64_t run_prefill(model::Model& m, const Cmd& c);
// kCmdRows on this rank: the step's number (wait() for it); the snapshots are queued behind it.
uint64_t run_rows(model::Model& m, const Cmd& c);

// The decode loop, run by every rank alike. Step i feeds token tok_i at position pos0 + i (tok_0 =
// c.tokens[0]) and samples tok_{i+1}; step i + 1 is queued before step i ends, with its token left
// to the device. Every rank sees the same samples and the same control word (rank 0's, e.g. a cancel),
// so all stop after the same step; the step queued behind it still runs, which only adds the KV of the
// last sampled token.
// With c.spec, steps are DSpark's (model.h launch_spec / launch_verify): the first feeds tok_0, each
// emits its accepted drafts and the token after them, and chooses the next step's rows, which is queued
// before it ends with its rows left to the device. Stops act per token: what follows a stop token in
// its step is dropped.
enum : uint32_t { kEndStop, kEndLength, kEndCancel, kEndContext, kEndPause };
struct DecodeEnd {
  int sampled = 0;     // tokens sampled: tok_1 .. tok_sampled
  int fed = 0;         // positions pos0 .. pos0 + fed - 1 hold the KV of tok_0 .. tok_{fed-1} now
  int32_t last = 0;    // the token the last of them fed
  uint32_t why = kEndStop;
  int written = 0;     // positions [fed, written) past them were written too (the queued step, rejected
                       // drafts; the draft module's windows hold its block keys up to 5 past the anchor)
  // DSpark: verify steps, the drafts they verified and accepted, and how many verified 0..5
  int vsteps = 0, verified = 0, accepted = 0;
  int rows_hist[model::ops::kDrafts + 1] = {};
  double gpu_ms[model::kMaxRows + 1] = {};  // spec steps' GPU time (rows, draft pass) by rows, summed
  int gpu_n[model::kMaxRows + 1] = {};
  // the verify-length gate: the threshold each spec step was given (lambda * dC, or --spec-thresh)
  double gate_sum = 0;
  float gate_min = 0, gate_max = 0;
  int gate_n = 0;
  double spec_tok = 1, spec_ms = 0;  // the drafts' rate (SpecRate) at the end: the next slice's start
  void gate(float t) {
    gate_min = gate_n ? std::min(gate_min, t) : t, gate_max = gate_n ? std::max(gate_max, t) : t;
    gate_sum += t, ++gate_n;
  }
};
using OnSample = std::function<void(int i, const model::SampleOut& o, int row)>;  // tok_{i+1} is o.row[row]
DecodeEnd decode_loop(model::Model& m, const Cmd& c, Watchdog& wd, const OnSample& on);
// The threshold the loop gives a spec step: c.spec_thresh, or lambda * dC with lambda the tokens per ms
// of the steps so far (their emitted tokens over their modelled times, spec_cost; the same on every rank).
// With the other gates, lambda itself, the steps' times modelled by spec_row_ms (kGateRatio ignores it).
struct SpecRate {
  double tok = 1, ms = 0;
  void add(const Cmd& c, int emitted, int rows);
  float thresh(const Cmd& c) const;
};

// kCmdBatch (docs/batching.md, B2): sessions decoding together, each in its own session slot with its own
// stop ids, sampling and snapshots, one row each per step (Model::launch_batch). Entry k's step j feeds
// tok_j at pos0 + j (tok_0 = token) and samples tok_{j+1}; as in decode_loop, the next step is queued
// before a step ends, with its tokens left to the device, and a session that ends keeps its row in the
// step already queued, which only adds the KV of its last sampled token.
// Rank 0's control word: bit s ends the session in slot s after the step that reads it (a cancel);
// kCtlPause ends every session there that its stop ids and max_new do not end (kEndPause), and with
// kBatchSolo the last session decoding pauses too (for the speculative loop). A paused session's row in
// the queued step is sampled too, within max_new: its tok_sampled, not fed yet, is the next to feed, at
// pos0 + fed.
struct BatchEntry {
  int32_t slot = 0;
  int32_t handle = -1;     // the session whose window snapshots are taken (snap_every, as kCmdDecode's)
  int32_t pos0 = 0, token = 0;
  int32_t max_new = 0;     // tokens to sample (the loop may compute one more position)
  int32_t limit = 0;       // positions below this may be computed
  int32_t snap_every = 0;
  int32_t n_stop = 0;
  int32_t stop[kMaxStop] = {};
  model::Sampling sp{};
};
constexpr uint32_t kCtlPause = 1u << 31;
enum : uint32_t { kBatchSolo = 1 };  // kCmdBatch's flags
struct BatchEnd {
  int sampled = 0;  // tokens sampled: tok_1 .. tok_sampled
  int fed = 0;      // positions pos0 .. pos0 + fed - 1 hold the KV of tok_0 .. tok_{fed-1} now
  uint32_t why = kEndStop;
};
using OnBatch = std::function<void(int k, const model::SampleOut& o, int row)>;  // entry k's next token: o.row[row]
std::vector<BatchEnd> batch_loop(model::Model& m, const Cmd& c, const std::vector<BatchEntry>& e, Watchdog& wd,
                                 const OnBatch& on);

// Host timeline of the last decode loop (diagnostics): per step, when its sample arrived, when the
// next step's Engram lookups were submitted, and when the step after was launched.
extern std::vector<double> g_t_wait, g_t_late, g_t_launch;

// Ranks 1-3: run rank 0's commands until kCmdStop.
void follow(model::Model& m, comm::Ring& ring, Watchdog& wd, int rank);

}  // namespace srv
