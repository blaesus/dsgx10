// Engram: the n-gram hash lookups at layers 1 and 14 (docs/engram.md).
//
// Every position hashes the 2-, 3- and 4-gram ending there into 24 rows per Engram layer
// (3 n-gram sizes x 8 heads). The rows live on NVMe in the kit's per-rank shards: rank r owns the
// rows [rows*r/4, rows*(r+1)/4) of each layer's table, as 264-byte records (256 e4m3 values and 8
// ue8m0 scales, one per 32). Each rank reads the rows it owns through io_uring, behind a hot-row
// cache, packs them in a fixed order, and the comm all-gather gives every rank all 24 rows.
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engram {

constexpr int kLayers = 2;               // Engram layers; model layers 1 and 14
constexpr int kMaxNgram = 4;             // n-grams of size 2..4
constexpr int kOrders = kMaxNgram - 1;
constexpr int kHeads = 8;
constexpr int kCols = kOrders * kHeads;  // rows per position per layer: order-major, then head
constexpr int kRowBytes = 264;           // 256 e4m3 + 8 ue8m0
constexpr int kWeightBytes = 256;
constexpr int kWorld = 4;

// ---- hashing --------------------------------------------------------------------------------

// The model constants hashing needs, from engram_hashmap.bin (engine/engram/py/make_hashmap_stdlib.py,
// or make_hashmap.py with the reference).
struct HashMap {
  int64_t n_tokens = 0, vocab = 0;
  int64_t pad = 0;  // compressed id that fills n-gram slots with no history
  int64_t layer_ids[kLayers] = {}, num_embeddings[kLayers] = {};
  int64_t mult[kLayers][kMaxNgram] = {};
  int64_t prime[kLayers][kOrders][kHeads] = {};
  int64_t offset[kLayers][kCols] = {};
  std::vector<int32_t> token_map;  // token id -> compressed id

  // Loads the file and checks the primes and offsets against its own recomputation.
  static HashMap load(const std::string& path);
};

// Row ids for positions [first, first + n) of a sequence whose raw token ids are tokens[0 ..
// first + n). Only tokens[first - 3 ..] are read. `live` may be null; a position with live == 0
// (an image span; never for text) blocks look-back, as in the reference.
// out: [n][kLayers][kCols], each id within its layer's table.
void hash_positions(const HashMap& m, const int32_t* tokens, const uint8_t* live, int64_t first, int n,
                    uint32_t* out);

// ---- who owns which rows --------------------------------------------------------------------

// The kit's split: rank r owns [rows*r/4, rows*(r+1)/4) of each layer's table.
struct Split {
  int64_t lo[kLayers][kWorld + 1] = {};
  static Split of(const HashMap& m);
  int owner(int layer, uint32_t row) const {
    int r = 0;
    while (r < kWorld - 1 && row >= lo[layer][r + 1]) r++;
    return r;
  }
};

// The hand-off to the all-gather, for one Engram layer of one step. Every rank builds the same map
// from the same row ids: cell (p, c) is record `index` of its owner's packed segment, and each
// rank's segment lists the cells it owns in (p, c) order. After the all-gather, the row of cell
// (p, c) is at seg[owner] + index * kRowBytes.
struct CellMap {
  int n_pos = 0;
  std::vector<uint8_t> owner;   // [n_pos * kCols]
  std::vector<uint32_t> index;  // [n_pos * kCols]
  uint32_t count[kWorld] = {};  // records in each rank's segment
  void build(const Split& s, int layer, const uint32_t* ids, int n_pos);  // ids as hash_positions

  // comm::allgather moves one size from every rank: the largest segment, rounded up to 16 bytes.
  // Each rank sends its count[rank] records followed by padding.
  size_t seg_bytes() const;
  // Where each cell's row lands in the gathered [seg_0 | seg_1 | seg_2 | seg_3]:
  // owner * seg_bytes() + index * kRowBytes. out: [n_pos * kCols], for the Engram kernel.
  void offsets(uint32_t* out) const;
};

// ---- the row store --------------------------------------------------------------------------

enum class Source {
  KitShard,    // the kit's packed per-rank shard: one read per row
  Checkpoint,  // the safetensors checkpoint itself: weight and scale are two reads (single box, checks)
};

struct StoreConfig {
  int rank = 0;                       // rows of this rank; -1 serves every row (single-box mode)
  Source source = Source::KitShard;
  std::string shard[kLayers];         // KitShard: engram-l1-r{rank}of4.bin, engram-l14-r{rank}of4.bin
  std::string checkpoint;             // Checkpoint: the checkpoint directory
  size_t cache_bytes = size_t(4) << 30;  // hot-row cache; 0 disables it
  int queue_depth = 128;              // reads in flight
  int spin_us = 1000000;              // the worker busy-polls this long after its last work before it
                                      // sleeps (-1: never sleeps); waking a sleeping worker costs the
                                      // CPU's idle-exit latency (up to ~430 us on GB10)
  int keepalive_ms = 5;               // with no read for this long, read one record anyway, so the SSD
                                      // never idles into a slower state (ESL01TBTLCZ: +50 us after
                                      // ~10 ms; APST after 100 ms: 1.1 ms on the ESL01TBTLCZ, 9.5 ms on
                                      // the PM9C1a; after 2 s: 10-20 ms); 0: off
  int read_batch = 1;                 // submit reads in multiples of this, padded with discarded reads:
                                      // 8 where the SSD coalesces interrupts (the Samsung PM9C1a signals
                                      // a completion once 8 are pending or after 200 us). The engine takes
                                      // it from the fleet file's RANK<r>_ENGRAM_READ_BATCH
  int cpu = -1;                       // pin the worker to this CPU (best: an otherwise idle X925 core)
};

// One Engram layer's rows for n_pos positions. The store fills the records it owns, packed in
// CellMap order (its all-gather segment), or, with rank -1, a full grid [n_pos][kCols][kRowBytes].
// `ids` and `out` must stay valid until done; `done` (any memory, e.g. pinned for the GPU or the
// proxy) is set to `seq` with release order once every record has landed.
struct Request {
  int layer = 0;
  const uint32_t* ids = nullptr;  // [n_pos][kLayers][kCols], as from hash_positions
  int n_pos = 0;
  uint8_t* out = nullptr;
  std::atomic<uint64_t>* done = nullptr;
  uint64_t seq = 0;
};

struct StoreStats {
  uint64_t requests = 0, lookups = 0, hits = 0, joined = 0;  // joined: miss on a row already in flight
  uint64_t reads = 0, read_bytes = 0, evictions = 0;
  uint64_t pad_reads = 0;  // discarded reads that rounded a batch up to read_batch
};

class Store {
 public:
  explicit Store(const StoreConfig& cfg, const HashMap& map);
  ~Store();
  Store(const Store&) = delete;
  Store& operator=(const Store&) = delete;

  void submit(const Request& r);  // non-blocking, from one control thread
  static void wait(const Request& r) {
    while (r.done->load(std::memory_order_acquire) != r.seq) {
    }
  }
  StoreStats stats() const;  // approximate while requests are in flight
  void clear_cache();        // drop every cached row (between benchmarks); call when idle
  const Split& split() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace engram
