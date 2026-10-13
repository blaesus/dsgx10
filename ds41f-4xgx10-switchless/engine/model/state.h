// Sessions' saved state (state.cu): what a session needs to continue from a position without
// recomputing what it already holds. The window rings and the compressor carries are per position mod
// 256 / 16, so they are snapshotted; the compressed caches are per position and append-only, so a
// session keeps a copy of them (by pages) and a continuation reuses entries below its divergence. The
// draft module's windows (DSpark, spec.cu) are per position mod 256 too and go with the snapshots.
//
// The store's device memory is one pool, reserved once at startup (Model::state_reserve) with the same
// size on every rank, and a best-fit allocator with coalescing hands out its blocks. Every rank runs
// the same operations in the same order, so every rank holds the same blocks at the same offsets. When
// the pool is full, the least recently used session that the current command does not use is evicted
// (the same one on every rank), and an allocation that still finds no room is skipped: a snapshot not
// taken, pages not saved, no end snapshot. Nothing in the store fails a request.
//
// With a disk tier (Model::state_disk, spill.h) an evicted session is spilled instead: its blocks go to
// a file of the same size on every rank, allocated by the same best fit, so every rank again holds the
// same blocks at the same offsets. When the file is full, its least recently used sessions are dropped.
// A spilled session comes back (state_unspill, or any operation that needs its blocks) as far as the
// store has room.
//
// Kept sessions (state.cu): a planned stop keeps the file and writes an index of the sessions in it
// beside it; the next start that finds both as this run would have left them (the same run id, not too
// old, on every rank alike) opens the file as it is, and its sessions sit in the disk tier as spilled
// ones until a request reads them back.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model/impl.h"
#include "model/prefill.h"

namespace model::detail {

constexpr int kPagePos = 4096;  // positions per page of compressed caches
constexpr size_t kEntryBytes = 256 + 32 + 64 + 4;  // KV entry + scales, index key + scales
constexpr size_t kPageBytes = (size_t)(kPagePos / 2 * 3 + kPagePos) * kEntryBytes;  // sources 2, 8, 14 (ratio 2), 20
constexpr size_t kRingBytes = (size_t)attn::kRing * (kHead + kHead / 32);
constexpr size_t kCarryBytes = (size_t)16 * kHead * 4 * 2;
constexpr size_t kEndBytes = kLayers * kRingBytes + kKvSlots * kCarryBytes;
constexpr size_t kPoolGrain = 64 << 10;  // the pool's blocks: multiples of 64 KiB, at 64 KiB offsets

// Best fit over [base, base + size); free ranges are coalesced, ties go to the lowest offset. The
// offsets alone (take_off, give_off) serve the disk tier's file and its staging buffer.
struct Pool {
  static constexpr size_t kNone = SIZE_MAX;
  uint8_t* base = nullptr;
  size_t size = 0;
  std::map<size_t, size_t> free;  // offset -> bytes
  static size_t round(size_t bytes) { return (bytes + kPoolGrain - 1) / kPoolGrain * kPoolGrain; }
  uint8_t* take(size_t bytes);  // nullptr when no free range holds round(bytes)
  void give(uint8_t* p, size_t bytes);
  size_t take_off(size_t bytes);  // kNone when no free range holds round(bytes)
  bool take_at(size_t off, size_t bytes);  // [off, off + round(bytes)) when it is free (kept sessions)
  void give_off(size_t off, size_t bytes);
  size_t largest() const;
};

class SpillFile;

struct Store {
  struct Sess {
    std::vector<uint8_t*> pages;    // compressed caches: page i holds positions [i, i + 1) * kPagePos
    int saved = 0;                  // positions [0, saved) are in the pages
    std::map<int, uint8_t*> snaps;  // window snapshots by position, the draft module's windows after them
    uint8_t* end = nullptr;         // every layer's ring and the carries (and the draft module's windows)
    int end_pos = -1;               // at this position
    uint64_t used = 0;              // the clock when an operation last used the session
    // spilled: the same blocks at offsets in the disk tier's file, none in the pool
    bool disk = false;
    std::vector<size_t> dpages;
    std::map<int, size_t> dsnaps;
    size_t dend = Pool::kNone;
    uint64_t dseq = 0;  // its last write (reading it back waits for that)
    bool kept = false;  // as an earlier run kept it, not read back since: a block that differs is fatal
  };
  // A session in a kept file's index: its blocks' offsets and their checksums as they went out.
  struct Kept {
    int h = 0, saved = 0, end_pos = -1;
    uint64_t used = 0;
    std::vector<std::pair<size_t, uint64_t>> pages;
    std::map<int, std::pair<size_t, uint64_t>> snaps;
    size_t dend = Pool::kNone;
    uint64_t dend_sum = 0;
  };
  std::map<int, Sess> sess;
  Pool pool;
  size_t bytes = 0;          // the pool's bytes in sessions' blocks
  std::unique_ptr<SpillFile> disk;  // the disk tier (none: evicted sessions are freed)
  Pool dpool;                       // its file's offsets
  size_t dbytes = 0;                // the file's bytes in spilled sessions' blocks
  std::string base;                 // the file is base.spill, its index base.index
  uint64_t run_id = 0;              // what the kept sessions' state depends on (main.cpp run_id)
  std::vector<Kept> kept;           // a kept index as state_kept_check read it, for state_disk
  uint64_t kept_gen = 0, kept_clock = 0;  // its stop's generation and the store's clock then
  uint64_t loaded_gen = 0;                // the generation this run's store came from (0: none)
  uint64_t clock = 0;
  std::vector<int> pinned;   // the sessions the current command uses: never evicted during it
  std::vector<int> evicted;  // sessions evicted to make room, since rank 0's server last asked
  Sess& use(int h) {         // session h (made if new), used now
    Sess& s = sess[h];
    s.used = ++clock;
    return s;
  }
  ~Store();  // state.cu: waits for the disk tier's writes and removes its file
};

}  // namespace model::detail
