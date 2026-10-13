// Rank 0 as a server (docs/serving.md, docs/protocol.md version 2): the front-end's Unix socket, sessions that keep
// their caches across requests with exact prefix reuse, the paged cache pool and its disk tier, kept sessions
// across a planned stop; every model call mirrored on the other ranks by a command (cmd.h).
#pragma once
#include <cstddef>
#include <string>

#include "comm/ring.h"
#include "model/model.h"
#include "server/cmd.h"
#include "server/disk.h"

namespace srv {

struct ServeOptions {
  std::string socket;          // the front-end's socket (mode 0600)
  std::string name = "glm-5.3";
  int max_ctx = 65536;         // positions a session may reach (prompt + output), at most
  int chunk = 8192;            // rows a prefill command (the model chunks within it by Options::prefill_rows)
  int slots = 4;               // sessions decoding together, a row each per step (B2), at most kMaxSess
  int slice = 64;              // time-slicing (B1): while more requests decode than `slots`, turns of this many tokens
  int grow = 1024;             // positions a session's pages are given beyond what it needs now (fewer moves)
  // Kept sessions: a planned stop (SIGTERM) spills the sessions in the pool to the disk tier, the most recently
  // used first, up to keep_bytes (0: nothing kept), and writes their histories to keep_index; the next start
  // takes them back when every rank reused that stop's files (kept_gen, main.cpp). Unused for keep_age_s: dropped.
  std::string keep_index;
  size_t keep_bytes = 0;
  double keep_age_s = 7 * 86400.0;
  uint64_t kept_gen = 0;
  uint64_t run_id = 0;
};

// Serves until SIGINT / SIGTERM, then stops the other ranks (kCmdStop). Rank 0 only; disk may be null (no tier).
int serve(model::Model& m, comm::Ring& ring, Watchdog& wd, Disk* disk, const ServeOptions& o);

}  // namespace srv
