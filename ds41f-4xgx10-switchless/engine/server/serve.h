// Rank 0 as a server (docs/protocol.md): the front-end's Unix socket, sessions that keep their state
// across requests with exact prefix reuse, and each request run on every rank through cmd.h's commands.
#pragma once
#include <cstddef>
#include <string>

#include "comm/ring.h"
#include "model/model.h"
#include "server/cmd.h"

namespace srv {

struct ServeOptions {
  std::string socket;              // the front-end's socket (mode 0600)
  int max_ctx = 65536;             // prompt + output positions per request
  int prefill_rows = 4096;         // rows per prefill chunk
  int small_rows = 8;              // a remainder of at most this many rows runs as one step (graphs)
  int snap_every = 1024;           // a window snapshot every this many positions (exact reuse)
  bool snap_thin = true;           // keep them dense near a session's end only (serve.cpp keep_snap); false
                                   // (--snap-all): every one kept
  size_t budget = 12ull << 30;     // --session-gib: at most this much of the store's pool (main.cpp reserves
                                   // it, model.h state_reserve) holds sessions' saved state; LRU eviction
  bool ced = true;                 // CED prefill (docs/ced.md) for turns of at least 128 new positions
  int slice = 64;                  // time-slicing (docs/batching.md, B1): a new request pauses the requests
                                   // decoding and is prefilled; while more of them decode than there are
                                   // slots (Model::slots), they take turns of this many tokens. 0: no turns
                                   // (a request that would need the slot of another in flight waits)
  // Kept sessions (docs/engine.md): a planned stop (SIGTERM) spills the sessions in device memory to the
  // disk tier, the most recently used first, up to keep_bytes (0: nothing is kept), and writes every
  // kept session's history to keep_index (beside the disk tier's files); the next start takes them back
  // when the store reused that stop's files. Sessions unused for keep_age_s are not kept.
  std::string keep_index;
  size_t keep_bytes = 0;
  double keep_age_s = 7 * 86400.0;
  std::string name = "deepseek-v4.1-flash";
  // DSpark drafts in decode (the model bound them): Cmd's spec_drafts, spec_thresh, spec_cost, spec_gate,
  // spec_row_ms
  bool spec = false;
  int spec_drafts = 5;
  float spec_thresh = -1.f;
  float spec_cost[3] = {19.4f, 3.6f, 3.3f};
  int spec_gate = model::ops::kGateThresh;
  float spec_row_ms[model::ops::kDrafts + 1] = {21.10f, 25.38f, 29.26f, 32.76f, 35.61f, 38.70f};
  // --profile-from-req N: request N's first command opens nsys's capture range on every rank
  // (model::kFlagProfile; nsys --capture-range=cudaProfilerApi), until the server stops. 0: never.
  uint64_t profile_from_req = 0;
};

// Serves until SIGINT / SIGTERM, then stops the other ranks (kCmdStop). Rank 0 only.
int serve(model::Model& m, comm::Ring& ring, Watchdog& wd, const ServeOptions& o);

}  // namespace srv
