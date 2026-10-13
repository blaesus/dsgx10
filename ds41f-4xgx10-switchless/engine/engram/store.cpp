// The Engram row store: one worker thread per store owns an io_uring, a hot-row cache and the
// bookkeeping of reads in flight. The control thread hands it Requests through a ring; the worker
// resolves each owned cell from the cache, by joining a read already in flight, or by a new
// O_DIRECT read of the smallest 512-byte-aligned window around the record.
#include <fcntl.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "common/check.h"
#include "engram/cache.h"
#include "engram/engram.h"
#include "engram/uring.h"
#include "weights/safetensors.h"

namespace engram {

namespace {

constexpr uint32_t kNone = kNoSlot;
constexpr unsigned kWin = 1024;  // the largest aligned window around a record of <= 264 bytes
constexpr uint64_t kShardMagic = 0x31344E4531565344ull;
constexpr uint64_t kDiscard = uint64_t(1) << 63;  // user_data of reads whose data is not wanted:
constexpr uint64_t kKeepAlive = kDiscard | 1;       //   keeping the SSD awake
constexpr uint64_t kPad = kDiscard | 2;             //   rounding a batch up to read_batch

inline uint64_t key_of(int layer, uint32_t row) { return (uint64_t(layer) << 32) | row; }
inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// The hot-row cache: a CLOCK index over slots of kRowBytes.
struct RowCache {
  ClockIndex idx;
  uint8_t* data = nullptr;

  void init(size_t bytes) {
    const size_t slots = bytes / kRowBytes;
    if (!slots) return;
    data = static_cast<uint8_t*>(big_alloc(slots * kRowBytes));
    idx.init(slots);
  }
  ~RowCache() {
    if (data) munmap(data, idx.slots() * kRowBytes);
  }
  void clear() { idx.clear(); }
  const uint8_t* find(uint64_t key) {
    const uint32_t s = idx.find(key);
    return s == kNoSlot ? nullptr : data + size_t(s) * kRowBytes;
  }
  // Returns true if a row was evicted.
  bool insert(uint64_t key, const uint8_t* rec) {
    bool evicted;
    const uint32_t s = idx.insert(key, &evicted);
    if (s != kNoSlot) std::memcpy(data + size_t(s) * kRowBytes, rec, kRowBytes);
    return evicted;
  }
};

}  // namespace

struct Store::Impl {
  StoreConfig cfg;
  Split sp;
  int64_t table_rows[kLayers];

  // where a layer's rows are read from
  struct Src {
    int file = -1, sfile = -1;  // registered file indices (sfile: the scale's file, Checkpoint only)
    uint64_t base = 0, sbase = 0;
    uint64_t lo = 0, hi = 0;  // rows this source holds
  } src[kLayers];
  std::vector<int> fds;
  bool two_part = false;

  std::unique_ptr<Uring> ring;
  int qd = 0;
  uint8_t* bounce = nullptr;  // qd slots x 2 parts x kWin, then one kWin for keep-alive reads
  size_t bounce_bytes = 0;
  uint64_t last_io = 0;       // when the last read was issued
  bool ka_inflight = false;
  uint32_t unseen = 0;        // reads submitted whose completions have not been reaped
  uint64_t ka_rnd = 0x9E3779B97F4A7C15ull;

  struct Slot {
    uint32_t miss = kNone;
    uint8_t parts_left = 0;
    uint16_t rec_off[2] = {0, 0};
    uint16_t need[2] = {0, 0};  // bytes the read must return (a window may run past the end of the file)
  };
  std::vector<Slot> slots;
  std::vector<uint32_t> free_slots;

  struct Miss {
    uint64_t key = kNoKey;
    uint32_t waiters = kNone;  // head of the waiter list
    uint32_t next = kNone;     // pending FIFO link
  };
  std::vector<Miss> misses;
  std::vector<uint32_t> free_misses;
  uint32_t pend_head = kNone, pend_tail = kNone;
  SlotTable inflight;  // key -> miss
  uint32_t n_inflight_reads = 0;

  struct Waiter {
    uint8_t* dest;
    uint32_t req;
    uint32_t next;
  };
  std::vector<Waiter> waiters;
  std::vector<uint32_t> free_waiters;

  RowCache cache;

  // requests: a single-producer ring; the worker scans them in order and they complete in any order
  static constexpr uint32_t kQ = 256;
  struct Active {
    Request r;
    int p = 0, c = 0;
    uint32_t k = 0, pending = 0;
    bool scanned = false, complete = false;
  };
  Active q[kQ];
  std::atomic<uint32_t> q_tail{0}, q_head{0};
  uint32_t q_scan = 0;  // worker: next request to scan

  std::atomic<uint32_t> futex_word{0};
  std::atomic<bool> sleeping{false}, stop{false};
  std::atomic<int> clear_state{0};
  std::thread worker;

  std::atomic<uint64_t> st_requests{0}, st_lookups{0}, st_hits{0}, st_joined{0}, st_reads{0}, st_bytes{0},
      st_evictions{0}, st_pads{0};

  Impl(const StoreConfig& c, const HashMap& m) : cfg(c), sp(Split::of(m)) {
    for (int l = 0; l < kLayers; l++) table_rows[l] = m.num_embeddings[l];
    if (cfg.rank < -1 || cfg.rank >= kWorld) die("engram: bad rank %d", cfg.rank);
    open_sources(m);
    qd = cfg.queue_depth < 1 ? 1 : cfg.queue_depth;
    ring = std::make_unique<Uring>(unsigned(2 * qd));
    ring->register_files(fds.data(), unsigned(fds.size()));
    bounce_bytes = (size_t(qd) * 2 + 1) * kWin;
    bounce = static_cast<uint8_t*>(big_alloc(bounce_bytes));
    std::memset(bounce, 0, bounce_bytes);
    iovec iov{bounce, bounce_bytes};
    ring->register_buffers(&iov, 1);
    slots.resize(qd);
    for (int i = qd - 1; i >= 0; i--) free_slots.push_back(uint32_t(i));
    const size_t max_miss = size_t(1) << 17, max_wait = size_t(1) << 19;
    misses.resize(max_miss);
    for (size_t i = max_miss; i-- > 0;) free_misses.push_back(uint32_t(i));
    inflight.init(max_miss * 2);
    waiters.resize(max_wait);
    for (size_t i = max_wait; i-- > 0;) free_waiters.push_back(uint32_t(i));
    cache.init(cfg.cache_bytes);
    worker = std::thread([this] { run(); });
  }

  ~Impl() {
    stop.store(true, std::memory_order_seq_cst);
    wake();
    worker.join();
    if (bounce) munmap(bounce, bounce_bytes);
    for (int fd : fds) close(fd);
  }

  void open_sources(const HashMap& m) {
    if (cfg.source == Source::KitShard) {
      if (cfg.rank < 0) die("engram: the kit shards hold one rank each; single-box mode reads the checkpoint");
      for (int l = 0; l < kLayers; l++) {
        const int fd = open(cfg.shard[l].c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) die("engram: open %s: %s", cfg.shard[l].c_str(), strerror(errno));
        uint64_t h[6];
        if (pread(fd, h, sizeof(h), 0) != sizeof(h)) {
          // O_DIRECT may refuse a small unaligned read; fall back to a buffered descriptor
          const int bfd = open(cfg.shard[l].c_str(), O_RDONLY);
          if (bfd < 0 || pread(bfd, h, sizeof(h), 0) != sizeof(h)) die("engram: %s: header", cfg.shard[l].c_str());
          close(bfd);
        }
        if (h[0] != kShardMagic || h[5] != uint64_t(kRowBytes) || int64_t(h[1]) != m.layer_ids[l] ||
            int64_t(h[4]) != m.num_embeddings[l])
          die("engram: %s is not the layer-%lld shard of this model", cfg.shard[l].c_str(), (long long)m.layer_ids[l]);
        if (int64_t(h[2]) != sp.lo[l][cfg.rank] || int64_t(h[3]) != sp.lo[l][cfg.rank + 1])
          die("engram: %s holds rows [%llu, %llu), rank %d owns [%lld, %lld)", cfg.shard[l].c_str(),
              (unsigned long long)h[2], (unsigned long long)h[3], cfg.rank, (long long)sp.lo[l][cfg.rank],
              (long long)sp.lo[l][cfg.rank + 1]);
        src[l] = {int(fds.size()), -1, 4096, 0, h[2], h[3]};
        fds.push_back(fd);
      }
    } else {
      const dsw::Checkpoint ck = dsw::Checkpoint::open(cfg.checkpoint);
      two_part = true;
      std::vector<int> shard_file(ck.shards.size(), -1);
      auto file_of = [&](int shard) {
        if (shard_file[shard] < 0) {
          const int fd = open(ck.shards[shard].c_str(), O_RDONLY | O_DIRECT);
          if (fd < 0) die("engram: open %s: %s", ck.shards[shard].c_str(), strerror(errno));
          shard_file[shard] = int(fds.size());
          fds.push_back(fd);
        }
        return shard_file[shard];
      };
      for (int l = 0; l < kLayers; l++) {
        const std::string pre = "layers." + std::to_string(m.layer_ids[l]) + ".engram.embed.";
        const dsw::SrcTensor& w = ck.at(pre + "weight");
        const dsw::SrcTensor& s = ck.at(pre + "scale");
        if (int64_t(w.shape[0]) != m.num_embeddings[l] || w.shape[1] != uint64_t(kWeightBytes) ||
            s.shape[1] != uint64_t(kRowBytes - kWeightBytes))
          die("engram: checkpoint table shapes do not match");
        src[l] = {file_of(w.shard), file_of(s.shard), w.offset, s.offset, 0, uint64_t(m.num_embeddings[l])};
      }
    }
  }

  // ---- control thread side ----
  void submit(const Request& r) {
    const uint32_t t = q_tail.load(std::memory_order_relaxed);
    while (t - q_head.load(std::memory_order_acquire) >= kQ) {
    }
    Active& a = q[t % kQ];
    a.r = r;
    a.p = a.c = 0;
    a.k = a.pending = 0;
    a.scanned = a.complete = false;
    q_tail.store(t + 1, std::memory_order_seq_cst);
    if (sleeping.load(std::memory_order_seq_cst)) wake();
  }
  void wake() {
    futex_word.fetch_add(1, std::memory_order_seq_cst);
    syscall(SYS_futex, &futex_word, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
  }

  // ---- worker ----
  void complete(Active& a) {
    a.complete = true;
    st_requests.fetch_add(1, std::memory_order_relaxed);
    a.r.done->store(a.r.seq, std::memory_order_release);
  }

  // Resolves the cells of requests in order until resources run out. Returns true on progress.
  bool scan() {
    bool did = false;
    const uint32_t tail = q_tail.load(std::memory_order_acquire);
    uint64_t lookups = 0, hits = 0, joined = 0;
    while (q_scan != tail) {
      Active& a = q[q_scan % kQ];
      const Request& r = a.r;
      bool blocked = false;
      for (; a.p < r.n_pos && !blocked; a.p++, a.c = 0) {
        const uint32_t* ids = r.ids + (size_t(a.p) * kLayers + r.layer) * kCols;
        for (; a.c < kCols; a.c++) {
          const uint32_t row = ids[a.c];
          uint8_t* dest;
          if (cfg.rank >= 0) {
            if (sp.owner(r.layer, row) != cfg.rank) continue;
            dest = r.out + size_t(a.k) * kRowBytes;
          } else {
            dest = r.out + (size_t(a.p) * kCols + a.c) * kRowBytes;
          }
          const uint64_t key = key_of(r.layer, row);
          if (const uint8_t* hit = cache.find(key)) {
            std::memcpy(dest, hit, kRowBytes);
            hits++;
          } else {
            if (free_waiters.empty()) {
              blocked = true;
              break;
            }
            uint32_t mi = inflight.find(key, [this](uint32_t i) { return misses[i].key; });
            if (mi == kNone) {
              if (free_misses.empty()) {
                blocked = true;
                break;
              }
              mi = free_misses.back();
              free_misses.pop_back();
              Miss& m = misses[mi];
              m.key = key;
              m.waiters = kNone;
              m.next = kNone;
              inflight.insert(key, mi);
              if (pend_tail == kNone) pend_head = mi;
              else misses[pend_tail].next = mi;
              pend_tail = mi;
            } else {
              joined++;
            }
            const uint32_t wi = free_waiters.back();
            free_waiters.pop_back();
            waiters[wi] = {dest, q_scan, misses[mi].waiters};
            misses[mi].waiters = wi;
            a.pending++;
          }
          lookups++;
          a.k++;
          did = true;
        }
        if (blocked) break;
      }
      if (blocked) break;
      a.scanned = true;
      did = true;
      if (a.pending == 0) complete(a);
      q_scan++;
    }
    st_lookups.fetch_add(lookups, std::memory_order_relaxed);
    st_hits.fetch_add(hits, std::memory_order_relaxed);
    st_joined.fetch_add(joined, std::memory_order_relaxed);
    retire();
    return did;
  }

  void retire() {
    uint32_t h = q_head.load(std::memory_order_relaxed);
    while (h != q_scan && q[h % kQ].complete) h++;
    q_head.store(h, std::memory_order_release);
  }

  // Starts reads for pending misses while slots are free. Returns true if any were queued.
  bool issue() {
    bool did = false;
    while (pend_head != kNone && !free_slots.empty()) {
      const uint32_t mi = pend_head;
      pend_head = misses[mi].next;
      if (pend_head == kNone) pend_tail = kNone;
      const uint32_t si = free_slots.back();
      free_slots.pop_back();
      Slot& s = slots[si];
      s.miss = mi;
      const int layer = int(misses[mi].key >> 32);
      const uint64_t row = misses[mi].key & 0xffffffffu;
      const Src& sr = src[layer];
      uint8_t* buf = bounce + size_t(si) * 2 * kWin;
      auto part = [&](int idx, int file, uint64_t off, unsigned len) {
        const uint64_t a = off & ~uint64_t(511), e = (off + len + 511) & ~uint64_t(511);
        io_uring_sqe* q = ring->sqe();
        if (!q) die("engram: submission ring full");
        ring->prep_read_fixed(q, file, buf + idx * kWin, unsigned(e - a), a, 0, (uint64_t(si) << 1) | unsigned(idx));
        unseen++;
        s.rec_off[idx] = uint16_t(off - a);
        s.need[idx] = uint16_t(off - a + len);
        st_bytes.fetch_add(e - a, std::memory_order_relaxed);
      };
      if (!two_part) {
        part(0, sr.file, sr.base + (row - sr.lo) * kRowBytes, kRowBytes);
        s.parts_left = 1;
      } else {
        part(0, sr.file, sr.base + row * kWeightBytes, kWeightBytes);
        part(1, sr.sfile, sr.sbase + row * (kRowBytes - kWeightBytes), kRowBytes - kWeightBytes);
        s.parts_left = 2;
      }
      n_inflight_reads++;
      st_reads.fetch_add(1, std::memory_order_relaxed);
      did = true;
    }
    if (!did) return false;
    // With nothing more to issue, round the reads outstanding up to a multiple of read_batch, so the
    // last completion trips the SSD's interrupt threshold instead of its timer. While misses are
    // still pending (deep prefill queues) completions keep coming and no padding is needed.
    if (cfg.read_batch > 1 && pend_head == kNone)
      while (unseen % unsigned(cfg.read_batch) && discard_read(kPad)) st_pads.fetch_add(1, std::memory_order_relaxed);
    last_io = now_ns();
    return true;
  }

  // One 512-byte read of a random record window of layer 1's source, into a scratch window and
  // discarded on completion. Returns false if the submission ring is full.
  bool discard_read(uint64_t tag) {
    ka_rnd ^= ka_rnd << 13;
    ka_rnd ^= ka_rnd >> 7;
    ka_rnd ^= ka_rnd << 17;
    const Src& sr = src[0];
    const uint64_t rows = (sr.hi - sr.lo) - 2;  // keep the window inside the table
    const uint64_t off = two_part ? sr.base + (ka_rnd % rows) * kWeightBytes : sr.base + (ka_rnd % rows) * kRowBytes;
    io_uring_sqe* q = ring->sqe();
    if (!q) return false;
    ring->prep_read_fixed(q, sr.file, bounce + size_t(qd) * 2 * kWin, 512, off & ~uint64_t(511), 0, tag);
    unseen++;
    return true;
  }
  void keepalive() {
    if (!discard_read(kKeepAlive)) return;
    ka_inflight = true;
    last_io = now_ns();
  }

  // Handles completed reads. Returns true if any of the store's own reads completed.
  bool reap() {
    const unsigned n = ring->cq_ready();
    if (!n) return false;
    bool did = false;
    for (unsigned i = 0; i < n; i++) {
      const io_uring_cqe& c = ring->cqe(i);
      if (c.user_data & kDiscard) {
        if (c.user_data == kKeepAlive) ka_inflight = false;
        continue;
      }
      did = true;
      const uint32_t si = uint32_t(c.user_data >> 1);
      const unsigned idx = unsigned(c.user_data & 1);
      Slot& s = slots[si];
      if (c.res < int(s.need[idx])) die("engram: read returned %d, needed %u", c.res, unsigned(s.need[idx]));
      if (--s.parts_left == 0) finish(si);
    }
    ring->cq_advance(n);
    unseen -= n;
    return did;
  }

  void finish(uint32_t si) {
    Slot& s = slots[si];
    const uint8_t* buf = bounce + size_t(si) * 2 * kWin;
    uint8_t tmp[kRowBytes];
    const uint8_t* rec = buf + s.rec_off[0];
    if (two_part) {
      std::memcpy(tmp, buf + s.rec_off[0], kWeightBytes);
      std::memcpy(tmp + kWeightBytes, buf + kWin + s.rec_off[1], kRowBytes - kWeightBytes);
      rec = tmp;
    }
    Miss& m = misses[s.miss];
    if (cache.insert(m.key, rec)) st_evictions.fetch_add(1, std::memory_order_relaxed);
    for (uint32_t wi = m.waiters; wi != kNone;) {
      const Waiter w = waiters[wi];
      std::memcpy(w.dest, rec, kRowBytes);
      Active& a = q[w.req % kQ];
      if (--a.pending == 0 && a.scanned) complete(a);
      free_waiters.push_back(wi);
      wi = w.next;
    }
    inflight.erase(m.key, [this](uint32_t i) { return misses[i].key; });
    m.key = kNoKey;
    free_misses.push_back(s.miss);
    s.miss = kNone;
    free_slots.push_back(si);
    n_inflight_reads--;
    retire();
  }

  void run() {
    if (cfg.cpu >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cfg.cpu, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    const uint64_t spin_ns = cfg.spin_us < 0 ? ~uint64_t(0) : uint64_t(cfg.spin_us) * 1000;
    const uint64_t ka_ns = uint64_t(cfg.keepalive_ms < 0 ? 0 : cfg.keepalive_ms) * 1000000;
    uint64_t last = now_ns();
    last_io = last;
    for (;;) {
      bool did = scan();
      if (issue()) {
        ring->submit();
        did = true;
      }
      did |= reap();
      if (did) {
        last = now_ns();
        continue;
      }
      if (stop.load(std::memory_order_relaxed)) break;
      if (clear_state.load(std::memory_order_acquire) == 1 && n_inflight_reads == 0 &&
          q_head.load(std::memory_order_relaxed) == q_tail.load(std::memory_order_acquire)) {
        cache.clear();
        clear_state.store(2, std::memory_order_release);
        continue;
      }
      const uint64_t t = now_ns();
      if (ka_ns && !ka_inflight && !n_inflight_reads && t - last_io >= ka_ns) {
        keepalive();
        ring->submit();
        continue;
      }
      if (t - last < spin_ns) continue;  // spin: lowest latency for the next request or completion
      if (n_inflight_reads) {
        ring->submit(1);  // block until a read completes
        continue;
      }
      // idle: sleep until submit() wakes us or the next keep-alive read is due
      timespec to, *top = nullptr;
      if (ka_ns) {
        const uint64_t due = last_io + ka_ns;
        const uint64_t rem = ka_inflight || due <= t ? 1000000 : due - t;
        to = {time_t(rem / 1000000000), long(rem % 1000000000)};
        top = &to;
      }
      const uint32_t v = futex_word.load(std::memory_order_seq_cst);
      sleeping.store(true, std::memory_order_seq_cst);
      if (q_tail.load(std::memory_order_seq_cst) == q_scan && !stop.load(std::memory_order_seq_cst) &&
          clear_state.load(std::memory_order_seq_cst) != 1)
        syscall(SYS_futex, &futex_word, FUTEX_WAIT_PRIVATE, v, top, nullptr, 0);
      sleeping.store(false, std::memory_order_seq_cst);
      // `last` stays: a request that woke us counts as work on the next pass, a timeout does not
    }
  }
};

Store::Store(const StoreConfig& cfg, const HashMap& map) : impl_(std::make_unique<Impl>(cfg, map)) {}
Store::~Store() = default;
void Store::submit(const Request& r) { impl_->submit(r); }
const Split& Store::split() const { return impl_->sp; }

StoreStats Store::stats() const {
  StoreStats s;
  s.requests = impl_->st_requests.load(std::memory_order_relaxed);
  s.lookups = impl_->st_lookups.load(std::memory_order_relaxed);
  s.hits = impl_->st_hits.load(std::memory_order_relaxed);
  s.joined = impl_->st_joined.load(std::memory_order_relaxed);
  s.reads = impl_->st_reads.load(std::memory_order_relaxed);
  s.read_bytes = impl_->st_bytes.load(std::memory_order_relaxed);
  s.evictions = impl_->st_evictions.load(std::memory_order_relaxed);
  s.pad_reads = impl_->st_pads.load(std::memory_order_relaxed);
  return s;
}

void Store::clear_cache() {
  impl_->clear_state.store(1, std::memory_order_seq_cst);
  impl_->wake();
  while (impl_->clear_state.load(std::memory_order_acquire) != 2) {
  }
  impl_->clear_state.store(0, std::memory_order_release);
}

}  // namespace engram
