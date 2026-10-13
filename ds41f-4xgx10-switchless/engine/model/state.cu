// Sessions' saved state (model.h state_*, state.h). Every copy is queued on the model's stream, so it
// lands in order with the steps around it, and a block freed on the host can be handed out again at
// once: whatever reads it later is queued after whatever wrote it before. Blocks come from the pool
// reserved at startup (state.h), so taking a snapshot inside the decode loop costs no allocation.
#include "model/state.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iterator>
#include <set>

#include "common/keepfile.h"
#include "model/spec.h"
#include "model/spill.h"

namespace model {

using namespace detail;

namespace {

constexpr size_t kArr[4] = {256, 32, 64, 4};  // bytes per entry: KV, KV scales, index key, key scales

double wall_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Page offset of kv slot k's array a (entries of kPagePos / ratio positions).
size_t page_off(int k, int a) {
  size_t off = 0;
  for (int j = 0; j < k; ++j) off += (size_t)(kPagePos / (j < 3 ? 2 : 1)) * kEntryBytes;
  const int E = kPagePos / (k < 3 ? 2 : 1);
  for (int j = 0; j < a; ++j) off += (size_t)E * kArr[j];
  return off;
}

// A window snapshot's block and an end snapshot's: the draft module's windows follow the layers' rows
size_t snap_bytes(bool spec) { return kSnapBytes + (spec ? kSpecSnapBytes : 0); }
size_t end_bytes(bool spec) { return kEndBytes + (spec ? kSpecEndBytes : 0); }

// A block's checksum for the disk tier's round trip: every 8-byte word mixed with its index, summed.
// kSumCtas CTAs each sum a stride of it; the last one to finish adds their partial sums in order and
// stores the result in *out (pinned memory) as one store, so nothing is zeroed first and the sum does
// not depend on which CTA finished when. Launches are serialized on the model's stream, so they share
// the partials and the counter.
constexpr int kSumCtas = 16;
__device__ uint64_t g_sum_part[kSumCtas];
__device__ unsigned g_sum_done;

__global__ void block_sum(const uint64_t* p, size_t n, uint64_t* out) {
  __shared__ uint64_t part[32];
  uint64_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint64_t x = p[i] + i * 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    acc += x ^ (x >> 27);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x >= 32) return;
  acc = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0;
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (threadIdx.x) return;
  g_sum_part[blockIdx.x] = acc;
  __threadfence();
  if (atomicAdd(&g_sum_done, 1u) != gridDim.x - 1) return;
  __threadfence();  // the last CTA: every partial sum is in
  uint64_t s = 0;
  for (unsigned j = 0; j < gridDim.x; ++j) s += ((volatile uint64_t*)g_sum_part)[j];
  *out = s;
  g_sum_done = 0;
}

// over the bytes the disk tier copies: the block rounded to the pool's grain
void sum_block(const void* p, size_t bytes, uint64_t* out, cudaStream_t st) {
  block_sum<<<kSumCtas, 1024, 0, st>>>((const uint64_t*)p, Pool::round(bytes) / 8, out);
}

}  // namespace

size_t Pool::take_off(size_t bytes) {
  const size_t n = round(bytes);
  auto best = free.end();
  for (auto it = free.begin(); it != free.end(); ++it)
    if (it->second >= n && (best == free.end() || it->second < best->second)) best = it;
  if (best == free.end()) return kNone;
  const size_t off = best->first, len = best->second;
  free.erase(best);
  if (len > n) free.emplace(off + n, len - n);
  return off;
}

uint8_t* Pool::take(size_t bytes) {
  const size_t off = take_off(bytes);
  return off == kNone ? nullptr : base + off;
}

void Pool::give(uint8_t* p, size_t bytes) { give_off((size_t)(p - base), bytes); }

bool Pool::take_at(size_t off, size_t bytes) {
  const size_t n = round(bytes);
  auto it = free.upper_bound(off);
  if (it == free.begin()) return false;
  --it;
  const size_t fo = it->first, fl = it->second;
  if (off + n > fo + fl) return false;
  free.erase(it);
  if (off > fo) free.emplace(fo, off - fo);
  if (off + n < fo + fl) free.emplace(off + n, fo + fl - off - n);
  return true;
}

size_t Pool::largest() const {
  size_t n = 0;
  for (const auto& f : free) n = std::max(n, f.second);
  return n;
}

void Pool::give_off(size_t off, size_t bytes) {
  size_t len = round(bytes);
  auto next = free.lower_bound(off);
  if (next != free.end() && next->first == off + len) {
    len += next->second;
    next = free.erase(next);
  }
  if (next != free.begin()) {
    const auto prev = std::prev(next);
    if (prev->first + prev->second == off) {
      off = prev->first;
      len += prev->second;
      free.erase(prev);
    }
  }
  free.emplace(off, len);
}

Store::~Store() = default;

Store& Model::Impl::store_of() {
  if (!store_s) store_s = std::make_unique<Store>();
  return *store_s;
}

// ---- kept sessions ---------------------------------------------------------------------------------
//
// A planned stop's index of the disk tier's file (base.index beside base.spill; keepfile.h: mode 0600
// in a 0700 directory, whole or not at all): a header, then every session in the file with its blocks'
// offsets and their checksums as they went out. Every rank writes its own with the same gen; the next
// start reuses the files only when every rank's index is whole, of this run's id, young enough, and of
// the same gen (main.cpp).

namespace {

constexpr char kKeptMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'K', '1'};

struct KeptHeader {
  char magic[8];
  uint64_t gen, run_id, file_bytes;
  int64_t written;  // unix time, seconds
  int32_t rank, n_sess;
  uint64_t clock;                              // the store's
  uint64_t page_bytes, snap_bytes, end_bytes;  // its layout
  uint64_t body_bytes, body_hash;
};

}  // namespace

uint64_t Model::Impl::state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                       std::string* why) {
  Store& s = store_of();
  s.kept.clear();
  s.kept_gen = s.kept_clock = 0;
  *bytes = 0;
  std::vector<uint8_t> f;
  if (!keepfile::read_private(base + ".index", 256u << 20, &f, why)) return 0;
  auto bad = [&](const std::string& w) {
    *why = base + ".index: " + w;
    s.kept.clear();
    return (uint64_t)0;
  };
  KeptHeader hd;
  if (f.size() < sizeof hd) return bad("too short");
  std::memcpy(&hd, f.data(), sizeof hd);
  const bool sp = spec != nullptr;
  const double age = (double)std::time(nullptr) - (double)hd.written;
  if (std::memcmp(hd.magic, kKeptMagic, sizeof kKeptMagic) != 0) return bad("not a kept index");
  if (hd.run_id != run_id) return bad("of another engine binary, weights or options");
  if (hd.rank != rank) return bad("of another rank");
  if (age > max_age_s || age < -600) return bad("written " + std::to_string((long long)age) + " s ago");
  if (hd.page_bytes != kPageBytes || hd.snap_bytes != snap_bytes(sp) || hd.end_bytes != end_bytes(sp))
    return bad("of another layout");
  if (!hd.gen || hd.body_bytes != f.size() - sizeof hd || keepfile::fnv(f.data() + sizeof hd, hd.body_bytes) != hd.body_hash)
    return bad("incomplete or damaged");
  if (!hd.file_bytes || hd.file_bytes % kPoolGrain || hd.n_sess < 0) return bad("a bad header");
  // every block on the grain and inside the file, none overlapping another
  std::vector<std::pair<size_t, size_t>> blocks;
  auto block = [&](size_t off, size_t n) {
    blocks.emplace_back(off, Pool::round(n));
    return off % kPoolGrain == 0 && off < hd.file_bytes && Pool::round(n) <= hd.file_bytes - off;
  };
  keepfile::Reader r{f.data() + sizeof hd, (size_t)hd.body_bytes};
  std::set<int> hs;
  const int max_pages = opt.max_ctx / kPagePos + 1;
  for (int i = 0; i < hd.n_sess && r.ok; ++i) {
    Store::Kept k;
    k.h = r.get<int32_t>(), k.saved = r.get<int32_t>(), k.end_pos = r.get<int32_t>();
    const int np = r.get<int32_t>(), ns = r.get<int32_t>(), he = r.get<int32_t>();
    k.used = r.get<uint64_t>();
    if (!r.ok || k.h <= 0 || !hs.insert(k.h).second || np < 0 || np > max_pages || ns < 0 || ns > 4096 || (he != 0 && he != 1) ||
        k.saved < 0 || k.saved > np * kPagePos || k.end_pos < -1 || k.end_pos > opt.max_ctx)
      return bad("a bad session record");
    for (int j = 0; j < np; ++j) {
      const size_t off = r.get<uint64_t>();
      const uint64_t sum = r.get<uint64_t>();
      if (!block(off, kPageBytes)) return bad("a block outside the file");
      k.pages.emplace_back(off, sum);
    }
    for (int j = 0; j < ns; ++j) {
      const int pos = r.get<int32_t>();
      (void)r.get<int32_t>();
      const size_t off = r.get<uint64_t>();
      const uint64_t sum = r.get<uint64_t>();
      if (pos < kSnapRows || pos > opt.max_ctx || !k.snaps.emplace(pos, std::make_pair(off, sum)).second ||
          !block(off, snap_bytes(sp)))
        return bad("a bad snapshot record");
    }
    if (he) {
      k.dend = r.get<uint64_t>(), k.dend_sum = r.get<uint64_t>();
      if (!block(k.dend, end_bytes(sp))) return bad("a block outside the file");
    }
    s.kept.push_back(std::move(k));
  }
  if (!r.ok || r.at != r.n) return bad("a bad body");
  std::sort(blocks.begin(), blocks.end());
  for (size_t i = 1; i < blocks.size(); ++i)
    if (blocks[i - 1].first + blocks[i - 1].second > blocks[i].first) return bad("overlapping blocks");
  struct stat st;  // the file beside it: ours, 0600, of the size the index names
  if (lstat((base + ".spill").c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != getuid() ||
      (st.st_mode & 0777) != 0600 || (uint64_t)st.st_size != hd.file_bytes)
    return bad("its file " + base + ".spill is missing or changed");
  s.kept_gen = hd.gen, s.kept_clock = hd.clock;
  *bytes = hd.file_bytes;
  why->clear();
  return hd.gen;
}

void Model::Impl::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  Store& s = store_of();
  if (s.disk) die("state: the disk tier is opened once");
  if (reuse && !s.kept_gen) die("state: reusing a kept file that was not checked");
  bytes = bytes / kPoolGrain * kPoolGrain;
  s.base = base, s.run_id = run_id;
  if (!bytes) return;
  const double t0 = wall_s();
  s.disk = std::make_unique<SpillFile>(base, bytes, staging, reuse);
  s.dpool.size = s.disk->size();
  s.dpool.free.emplace(0, s.dpool.size);
  unlink((base + ".index").c_str());  // whatever happens from now on, a crash starts clean
  unlink((base + ".index.new").c_str());
  if (reuse) {  // the kept sessions, spilled ones: back on demand, each block checked (state_unspill)
    const bool sp = spec != nullptr;
    auto take = [&](size_t off, size_t n, uint64_t sum) {
      if (!s.dpool.take_at(off, n)) die("state: a kept block at %zu is not free", off);
      s.dbytes += Pool::round(n);
      *s.disk->sum(off, false) = sum;
    };
    s.clock = s.kept_clock;
    for (const Store::Kept& k : s.kept) {
      Store::Sess& ss = s.sess[k.h];
      ss.disk = ss.kept = true;
      ss.saved = k.saved, ss.end_pos = k.end_pos, ss.used = k.used;
      for (const auto& [off, sum] : k.pages) take(off, kPageBytes, sum), ss.dpages.push_back(off);
      for (const auto& [pos, os] : k.snaps) take(os.first, snap_bytes(sp), os.second), ss.dsnaps.emplace(pos, os.first);
      if (k.dend != Pool::kNone) take(k.dend, end_bytes(sp), k.dend_sum), ss.dend = k.dend;
      s.clock = std::max(s.clock, k.used);
    }
    s.loaded_gen = s.kept_gen;
    if (rank == 0)
      std::printf("state: %zu kept sessions in %s (%.2f of %.2f GiB), in the store in %.1f ms\n", s.kept.size(),
                  s.disk->path().c_str(), s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0, 1e3 * (wall_s() - t0));
  }
  s.kept.clear();
  s.kept.shrink_to_fit();
}

// A planned stop (every rank alike, after the server spilled what it keeps): the file synced, then its
// index, and the file stays at exit. A rank whose index cannot be written keeps nothing (the next
// start's gens differ: every rank starts clean).
void Model::Impl::state_keep(uint64_t gen) {
  Store& s = store_of();
  if (!s.disk || !gen) return;
  const double t0 = wall_s();
  CUDA_CHECK(cudaStreamSynchronize(st));  // the checksums of what went out are in
  s.disk->sync();
  const double t1 = wall_s();
  const bool sp = spec != nullptr;
  std::vector<uint8_t> f(sizeof(KeptHeader));
  using keepfile::put;
  int n = 0;
  for (const auto& [h, ss] : s.sess) {
    if (!ss.disk) continue;
    ++n;
    put<int32_t>(f, h), put<int32_t>(f, ss.saved), put<int32_t>(f, ss.end_pos);
    put<int32_t>(f, (int32_t)ss.dpages.size()), put<int32_t>(f, (int32_t)ss.dsnaps.size());
    put<int32_t>(f, ss.dend != Pool::kNone), put<uint64_t>(f, ss.used);
    for (const size_t o : ss.dpages) put<uint64_t>(f, o), put<uint64_t>(f, *s.disk->sum(o, false));
    for (const auto& [pos, o] : ss.dsnaps)
      put<int32_t>(f, pos), put<int32_t>(f, 0), put<uint64_t>(f, o), put<uint64_t>(f, *s.disk->sum(o, false));
    if (ss.dend != Pool::kNone) put<uint64_t>(f, ss.dend), put<uint64_t>(f, *s.disk->sum(ss.dend, false));
  }
  KeptHeader hd{};
  std::memcpy(hd.magic, kKeptMagic, sizeof kKeptMagic);
  hd.gen = gen, hd.run_id = s.run_id, hd.file_bytes = s.disk->size(), hd.written = (int64_t)std::time(nullptr);
  hd.rank = rank, hd.n_sess = n, hd.clock = s.clock;
  hd.page_bytes = kPageBytes, hd.snap_bytes = snap_bytes(sp), hd.end_bytes = end_bytes(sp);
  hd.body_bytes = f.size() - sizeof hd;
  hd.body_hash = keepfile::fnv(f.data() + sizeof hd, hd.body_bytes);
  std::memcpy(f.data(), &hd, sizeof hd);
  std::string why;
  if (!keepfile::write_atomic(s.base + ".index", f.data(), f.size(), &why)) {
    std::printf("state: rank %d keeps nothing: %s\n", rank, why.c_str());
    return;
  }
  s.disk->keep();
  if (rank == 0)
    std::printf("state: %d sessions kept for the next start (%.2f GiB of the file): synced in %.1f ms, index of %.1f KiB "
                "in %.1f ms\n", n, s.dbytes / 1073741824.0, 1e3 * (t1 - t0), f.size() / 1024.0, 1e3 * (wall_s() - t1));
}

void Model::Impl::state_reserve(size_t bytes) {
  Store& s = store_of();
  if (s.pool.base) die("state: the store's pool is reserved once");
  bytes = bytes / kPoolGrain * kPoolGrain;
  if (!bytes) return;
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  s.pool.base = static_cast<uint8_t*>(p);
  s.pool.size = bytes;
  s.pool.free.emplace(0, bytes);
}

// A block for session h. While none is free, the least recently used session that the current command
// does not use is evicted (every rank evicts the same): spilled to the disk tier, or freed without one;
// nullptr when none is left to evict.
uint8_t* Model::Impl::store_alloc(int h, size_t bytes) {
  Store& s = store_of();
  for (;;) {
    if (uint8_t* p = s.pool.take(bytes)) {
      s.bytes += Pool::round(bytes);
      return p;
    }
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (const auto& [k, ss] : s.sess)
      if (k != h && !ss.disk && ss.used < oldest && std::find(s.pinned.begin(), s.pinned.end(), k) == s.pinned.end())
        victim = k, oldest = ss.used;
    if (victim < 0) {
      if (rank == 0)
        std::printf("state: no room for %.1f MiB of session %d's state (%.2f of %.2f GiB in use)\n",
                    bytes / 1048576.0, h, s.bytes / 1073741824.0, s.pool.size / 1073741824.0);
      return nullptr;
    }
    state_spill(victim);
  }
}

// ---- the disk tier --------------------------------------------------------------------------------

namespace {

// a spilled session's blocks, released from the file
void disk_release(Store& s, Store::Sess& ss, bool spec) {
  for (const size_t o : ss.dpages) s.dpool.give_off(o, kPageBytes), s.dbytes -= Pool::round(kPageBytes);
  for (const auto& sn : ss.dsnaps) s.dpool.give_off(sn.second, snap_bytes(spec)), s.dbytes -= Pool::round(snap_bytes(spec));
  if (ss.dend != Pool::kNone) s.dpool.give_off(ss.dend, end_bytes(spec)), s.dbytes -= Pool::round(end_bytes(spec));
  ss.dpages.clear();
  ss.dsnaps.clear();
  ss.dend = Pool::kNone;
}

}  // namespace

// Room in the file for a block of session h: while there is none, the least recently used spilled
// session that the current command does not use is dropped (gone: state_evicted). kNone when none is left.
size_t Model::Impl::disk_take(int h, size_t bytes) {
  Store& s = store_of();
  for (;;) {
    const size_t off = s.dpool.take_off(bytes);
    if (off != Pool::kNone) {
      s.dbytes += Pool::round(bytes);
      return off;
    }
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (const auto& [k, ss] : s.sess)
      if (k != h && ss.disk && ss.used < oldest && std::find(s.pinned.begin(), s.pinned.end(), k) == s.pinned.end())
        victim = k, oldest = ss.used;
    if (victim < 0) return Pool::kNone;
    if (rank == 0) std::printf("state: session %d dropped from the disk tier to make room\n", victim);
    state_free(victim);
    s.evicted.push_back(victim);
  }
}

// Session h's blocks into the disk tier: each one's checksum is taken, then it is copied to staging after
// the work queued so far and released at once, and the writer thread writes them out. Without a disk tier, or without room in it,
// h is freed (gone: state_evicted).
void Model::Impl::state_spill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || it->second.disk) return;
  Store::Sess& ss = it->second;
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  std::vector<size_t> dp;
  std::map<int, size_t> ds;
  size_t de = Pool::kNone;
  bool room = s.disk != nullptr;
  for (size_t i = 0; room && i < ss.pages.size(); ++i)
    room = (dp.emplace_back(disk_take(h, kPageBytes)), dp.back() != Pool::kNone);
  for (auto i = ss.snaps.begin(); room && i != ss.snaps.end(); ++i)
    room = (ds[i->first] = disk_take(h, snap_bytes(sp))) != Pool::kNone;
  if (room && ss.end) room = (de = disk_take(h, end_bytes(sp))) != Pool::kNone;
  if (!room) {
    if (s.disk) {
      if (!dp.empty() && dp.back() == Pool::kNone) dp.pop_back();
      for (auto i = ds.begin(); i != ds.end();)
        i = i->second == Pool::kNone ? ds.erase(i) : std::next(i);
      Store::Sess tmp;
      tmp.dpages = dp, tmp.dsnaps = ds, tmp.dend = de;
      disk_release(s, tmp, sp);
    }
    state_free(h);
    s.evicted.push_back(h);
    return;
  }
  const size_t b0 = s.bytes;  // spilling only releases blocks
  s.disk->begin(st);  // the checksums of what goes out, then the copies (each timed: spill.cpp)
  for (size_t i = 0; i < ss.pages.size(); ++i) sum_block(ss.pages[i], kPageBytes, s.disk->sum(dp[i], false), st);
  for (const auto& sn : ss.snaps) sum_block(sn.second, snap_bytes(sp), s.disk->sum(ds.at(sn.first), false), st);
  if (ss.end) sum_block(ss.end, end_bytes(sp), s.disk->sum(de, false), st);
  for (size_t i = 0; i < ss.pages.size(); ++i) {
    ss.dseq = s.disk->write(ss.pages[i], kPageBytes, dp[i], st);
    store_release(ss.pages[i], kPageBytes);
  }
  for (const auto& sn : ss.snaps) {
    ss.dseq = s.disk->write(sn.second, snap_bytes(sp), ds.at(sn.first), st);
    store_release(sn.second, snap_bytes(sp));
  }
  if (ss.end) {
    ss.dseq = s.disk->write(ss.end, end_bytes(sp), de, st);
    store_release(ss.end, end_bytes(sp));
  }
  s.disk->flush(st);
  ss.pages.clear();
  ss.snaps.clear();
  ss.end = nullptr;
  ss.disk = true;
  ss.dpages = std::move(dp);
  ss.dsnaps = std::move(ds);
  ss.dend = de;
  if (rank == 0)
    std::printf("state: session %d spilled to disk (%.1f MiB; %.2f of %.2f GiB of the file in use; host %.2f ms)\n", h,
                (b0 - s.bytes) / 1048576.0, s.dbytes / 1073741824.0, s.dpool.size / 1073741824.0,
                1e3 * (wall_s() - t0));
}

// Session h back from the disk tier (after its writes), as far as the store has room: pages in order
// (saved shrinks to what came back), the end snapshot, then window snapshots from the latest down; what
// does not fit is dropped, as a save that finds no room would. Every block that comes back is checked
// against its checksum from the way out; every rank logs a mismatch, and rank 0 the time it took.
void Model::Impl::state_unspill(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || !it->second.disk) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  const bool sp = spec != nullptr;
  const double t0 = wall_s();
  s.disk->wait(ss.dseq);
  const double t1 = wall_s();
  size_t got = 0;
  std::vector<size_t> back;  // the file offsets of the blocks that came back
  const auto read = [&](uint8_t* dst, size_t bytes, size_t off) {
    s.disk->read(dst, bytes, off, st);
    sum_block(dst, bytes, s.disk->sum(off, true), st);
    back.push_back(off);
    got += Pool::round(bytes);
  };
  s.pinned.push_back(h);  // its own blocks' allocations never evict it
  for (size_t i = 0; i < ss.dpages.size(); ++i) {
    uint8_t* p = store_alloc(h, kPageBytes);
    if (!p) break;
    read(p, kPageBytes, ss.dpages[i]);
    ss.pages.push_back(p);
  }
  ss.saved = std::min(ss.saved, (int)ss.pages.size() * kPagePos);
  if (ss.dend != Pool::kNone && ss.end_pos <= ss.saved)
    if ((ss.end = store_alloc(h, end_bytes(sp)))) read(ss.end, end_bytes(sp), ss.dend);
  if (!ss.end) ss.end_pos = -1;
  for (auto i = ss.dsnaps.rbegin(); i != ss.dsnaps.rend(); ++i) {
    uint8_t* b = store_alloc(h, snap_bytes(sp));
    if (!b) break;
    read(b, snap_bytes(sp), i->second);
    ss.snaps.emplace(i->first, b);
  }
  s.pinned.pop_back();
  CUDA_CHECK(cudaStreamSynchronize(st));
  const double t2 = wall_s();
  size_t bad = 0;
  uint64_t fp = 0;  // the checksums' xor, logged: a check that compared nothing would show as 0
  for (const size_t o : back) bad += *s.disk->sum(o, false) != *s.disk->sum(o, true), fp ^= *s.disk->sum(o, true);
  if (bad)
    std::printf("state: session %d: ERROR: %zu of %zu blocks read back from the disk differ from what was spilled "
                "(rank %d)\n", h, bad, back.size(), rank);
  // a kept session is not continued from a state that is not the one it left: the next start is clean
  if (bad && ss.kept) die("state: session %d, kept by an earlier run, came back different from the disk (rank %d)", h, rank);
  ss.kept = false;
  disk_release(s, ss, sp);
  ss.disk = false;
  if (rank == 0)
    std::printf("state: session %d back from disk: %.1f MiB in %.1f ms (%.2f GB/s; %.1f ms waiting for its writes; "
                "%zu blocks checked, %016llx)\n", h, got / 1048576.0, 1e3 * (t2 - t0), got / std::max(1e-9, t2 - t1) / 1e9,
                1e3 * (t1 - t0), back.size(), (unsigned long long)fp);
}

namespace {

// session h (made if new) for an operation that needs its blocks: back from the disk tier first
Store::Sess& use_blocks(Model::Impl& m, int h) {
  Store& s = m.store_of();
  const auto it = s.sess.find(h);
  if (it != s.sess.end() && it->second.disk) m.state_unspill(h);
  return s.use(h);
}

}  // namespace

void Model::Impl::store_release(uint8_t* p, size_t bytes) {
  if (!p) return;
  Store& s = store_of();
  s.pool.give(p, bytes);
  s.bytes -= Pool::round(bytes);
}

// Compressed caches of positions [from, to): entries [from / ratio, to / ratio) of every kv source
// (an entry is complete once its last position is in). Without room for every page, as far as the
// pages go.
void Model::Impl::state_save(int h, int from, int to) {
  Store::Sess& ss = use_blocks(*this, h);
  if (from > ss.saved) die("state: session %d saves [%d, %d) with %d saved", h, from, to, ss.saved);
  while ((int)ss.pages.size() * kPagePos < to) {
    uint8_t* p = store_alloc(h, kPageBytes);
    if (!p) break;
    ss.pages.push_back(p);
  }
  to = std::min(to, (int)ss.pages.size() * kPagePos);
  for (int k = 0; k < kKvSlots; ++k) {
    const int ratio = k < 3 ? 2 : 1, E = kPagePos / ratio;
    const KvSrc& src = kvs[k];
    uint8_t* const arr[4] = {src.ckv, src.csc, src.ik, src.iksc};
    for (int j = from / ratio, j1 = to / ratio; j < j1;) {
      const int pi = j / E, n = std::min(j1, (pi + 1) * E) - j;
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemcpyAsync(ss.pages[pi] + page_off(k, a) + (size_t)(j - pi * E) * kArr[a], arr[a] + (size_t)j * kArr[a],
                                   (size_t)n * kArr[a], cudaMemcpyDeviceToDevice, st));
      j += n;
    }
  }
  ss.saved = std::max(ss.saved, to);
}

void Model::Impl::state_load(int h, int to) {
  if (!store_of().sess.count(h)) die("state: session %d loads with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (to > ss.saved) die("state: session %d loads %d positions of %d", h, to, ss.saved);
  for (int k = 0; k < kKvSlots; ++k) {
    const int ratio = k < 3 ? 2 : 1, E = kPagePos / ratio;
    const KvSrc& dst = kvs[k];
    uint8_t* const arr[4] = {dst.ckv, dst.csc, dst.ik, dst.iksc};
    for (int j = 0, j1 = to / ratio; j < j1;) {
      const int pi = j / E, n = std::min(j1, (pi + 1) * E) - j;
      for (int a = 0; a < 4; ++a)
        CUDA_CHECK(cudaMemcpyAsync(arr[a] + (size_t)j * kArr[a], ss.pages[pi] + page_off(k, a) + (size_t)(j - pi * E) * kArr[a],
                                   (size_t)n * kArr[a], cudaMemcpyDeviceToDevice, st));
      j += n;
    }
  }
}

namespace {

// positions [p0, p1) between a ring and a linear buffer (row bytes per position), at most 256 of them
void ring_rows(uint8_t* ring, uint8_t* lin, size_t row, int p0, int p1, bool to_lin, cudaStream_t st) {
  for (int q = p0; q < p1;) {
    const int slot = q & (attn::kRing - 1), n = std::min(p1 - q, attn::kRing - slot);
    uint8_t* r = ring + (size_t)slot * row;
    uint8_t* l = lin + (size_t)(q - p0) * row;
    CUDA_CHECK(cudaMemcpyAsync(to_lin ? l : r, to_lin ? r : l, (size_t)n * row, cudaMemcpyDeviceToDevice, st));
    q += n;
  }
}

// the draft module's windows, positions [p0, p1) or whole (p0 < 0), to or from b
void spec_windows(const Spec& sp, uint8_t* b, int p0, int p1, bool to_buf, cudaStream_t st) {
  const size_t n = p0 < 0 ? attn::kRing : (size_t)(p1 - p0);
  for (int k = 0; k < dspark::kBlocks; ++k) {
    const attn::Window w = sp.draft->window(k);
    uint8_t* kv = b + (size_t)k * n * kSpecSlot;
    uint8_t* sc = kv + n * kHead;
    if (p0 < 0) {
      CUDA_CHECK(cudaMemcpyAsync(to_buf ? kv : w.kv, to_buf ? w.kv : kv, n * kHead, cudaMemcpyDeviceToDevice, st));
      CUDA_CHECK(cudaMemcpyAsync(to_buf ? sc : w.sc, to_buf ? w.sc : sc, n * kHead / 32, cudaMemcpyDeviceToDevice, st));
    } else {
      ring_rows(w.kv, kv, kHead, p0, p1, to_buf, st);
      ring_rows(w.sc, sc, kHead / 32, p0, p1, to_buf, st);
    }
  }
}

}  // namespace

// A snapshot's block (the caller fills it), or nullptr when the store has no room for it.
uint8_t* Model::Impl::state_snap_buf(int h, int pos) {
  Store::Sess& ss = use_blocks(*this, h);
  const auto it = ss.snaps.find(pos);
  if (it != ss.snaps.end()) return it->second;
  uint8_t* b = store_alloc(h, snap_bytes(spec != nullptr));
  if (b) ss.snaps.emplace(pos, b);
  return b;
}

void Model::Impl::state_snap(int h, int pos) {
  if (pos < kSnapRows) die("state: snapshot at %d", pos);
  uint8_t* b = state_snap_buf(h, pos);
  if (!b) return;
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(ring_kv[L], snap_kv(b, L), kHead, pos - kSnapRows, pos, true, st);
    ring_rows(ring_sc[L], snap_sc(b, L), kHead / 32, pos - kSnapRows, pos, true, st);
  }
  if (spec) spec_windows(*spec, b + kSnapBytes, pos - kSnapRows, pos, true, st);
}

// The draft module's windows into the snapshot at pos, which the prefill took (none: not taken).
void Model::Impl::state_snap_spec(int h, int pos) {
  const Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end() || it->second.disk) return;
  const auto b = it->second.snaps.find(pos);
  if (b != it->second.snaps.end()) spec_windows(*spec, b->second + kSnapBytes, pos - kSnapRows, pos, true, st);
}

void Model::Impl::state_restore(int h, int pos) {
  if (!store_of().sess.count(h)) die("state: session %d restores with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (!ss.snaps.count(pos)) die("state: session %d has no snapshot at %d", h, pos);
  uint8_t* b = ss.snaps.at(pos);
  for (int L = 0; L < kLayers; ++L) {
    ring_rows(ring_kv[L], snap_kv(b, L), kHead, pos - kSnapRows, pos, false, st);
    ring_rows(ring_sc[L], snap_sc(b, L), kHead / 32, pos - kSnapRows, pos, false, st);
  }
  if (spec) spec_windows(*spec, b + kSnapBytes, pos - kSnapRows, pos, false, st);
}

void Model::Impl::state_snap_end(int h, int pos) {
  Store::Sess& ss = use_blocks(*this, h);
  if (!ss.end) ss.end = store_alloc(h, end_bytes(spec != nullptr));
  if (!ss.end) {
    ss.end_pos = -1;
    return;
  }
  ss.end_pos = pos;
  uint8_t* b = ss.end;
  for (int L = 0; L < kLayers; ++L, b += kRingBytes) {
    CUDA_CHECK(cudaMemcpyAsync(b, ring_kv[L], (size_t)attn::kRing * kHead, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(b + (size_t)attn::kRing * kHead, ring_sc[L], (size_t)attn::kRing * kHead / 32,
                               cudaMemcpyDeviceToDevice, st));
  }
  for (int k = 0; k < kKvSlots; ++k, b += kCarryBytes) {
    CUDA_CHECK(cudaMemcpyAsync(b, kvs[k].carry_kv, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(b + kCarryBytes / 2, kvs[k].carry_sc, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
  }
  if (spec) spec_windows(*spec, ss.end + kEndBytes, -1, 0, true, st);
}

void Model::Impl::state_restore_end(int h) {
  if (!store_of().sess.count(h)) die("state: session %d restores its end with nothing stored", h);
  Store::Sess& ss = use_blocks(*this, h);
  if (!ss.end) die("state: session %d has no end snapshot", h);
  const uint8_t* b = ss.end;
  for (int L = 0; L < kLayers; ++L, b += kRingBytes) {
    CUDA_CHECK(cudaMemcpyAsync(ring_kv[L], b, (size_t)attn::kRing * kHead, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ring_sc[L], b + (size_t)attn::kRing * kHead, (size_t)attn::kRing * kHead / 32,
                               cudaMemcpyDeviceToDevice, st));
  }
  for (int k = 0; k < kKvSlots; ++k, b += kCarryBytes) {
    CUDA_CHECK(cudaMemcpyAsync(kvs[k].carry_kv, b, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(kvs[k].carry_sc, b + kCarryBytes / 2, kCarryBytes / 2, cudaMemcpyDeviceToDevice, st));
  }
  if (spec) spec_windows(*spec, ss.end + kEndBytes, -1, 0, false, st);
}

// What session h holds from position `from` on no longer belongs to its history: snapshots above it,
// the end snapshot if it is above, and saved positions from it on.
void Model::Impl::state_drop(int h, int from) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  ss.used = ++s.clock;
  const bool sp = spec != nullptr;
  if (ss.disk) {  // the file's blocks
    for (auto i = ss.dsnaps.upper_bound(from); i != ss.dsnaps.end();) {
      s.dpool.give_off(i->second, snap_bytes(sp)), s.dbytes -= Pool::round(snap_bytes(sp));
      i = ss.dsnaps.erase(i);
    }
    if (ss.end_pos > from) {
      if (ss.dend != Pool::kNone) s.dpool.give_off(ss.dend, end_bytes(sp)), s.dbytes -= Pool::round(end_bytes(sp));
      ss.dend = Pool::kNone;
      ss.end_pos = -1;
    }
    ss.saved = std::min(ss.saved, from);
    while ((int)ss.dpages.size() > (ss.saved + kPagePos - 1) / kPagePos) {
      s.dpool.give_off(ss.dpages.back(), kPageBytes), s.dbytes -= Pool::round(kPageBytes);
      ss.dpages.pop_back();
    }
    return;
  }
  for (auto i = ss.snaps.upper_bound(from); i != ss.snaps.end();) {
    store_release(i->second, snap_bytes(sp));
    i = ss.snaps.erase(i);
  }
  if (ss.end_pos > from) {
    store_release(ss.end, end_bytes(sp));
    ss.end = nullptr;
    ss.end_pos = -1;
  }
  ss.saved = std::min(ss.saved, from);
  while ((int)ss.pages.size() > (ss.saved + kPagePos - 1) / kPagePos) {
    store_release(ss.pages.back(), kPageBytes);
    ss.pages.pop_back();
  }
}

void Model::Impl::state_free(int h) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  Store::Sess& ss = it->second;
  for (uint8_t* p : ss.pages) store_release(p, kPageBytes);
  for (auto& sn : ss.snaps) store_release(sn.second, snap_bytes(spec != nullptr));
  store_release(ss.end, end_bytes(spec != nullptr));
  disk_release(s, ss, spec != nullptr);  // a spilled session's writes may still be queued: harmless
  s.sess.erase(it);
}

// src's snapshots up to upto into dst (src is kept while they are copied); those without room are not.
void Model::Impl::state_copy_snaps(int src, int dst, int upto) {
  Store& s = store_of();
  const auto it = s.sess.find(src);
  if (it == s.sess.end() || src == dst) return;
  if (it->second.disk) {
    s.pinned.push_back(dst);
    state_unspill(src);
    s.pinned.pop_back();
  }
  it->second.used = ++s.clock;
  const std::vector<std::pair<int, uint8_t*>> from(it->second.snaps.begin(), it->second.snaps.upper_bound(upto));
  s.pinned.push_back(src);
  for (const auto& sn : from)
    if (uint8_t* b = state_snap_buf(dst, sn.first))
      CUDA_CHECK(cudaMemcpyAsync(b, sn.second, snap_bytes(spec != nullptr), cudaMemcpyDeviceToDevice, st));
  s.pinned.pop_back();
}

void Model::Impl::state_drop_snap(int h, int pos) {
  Store& s = store_of();
  const auto it = s.sess.find(h);
  if (it == s.sess.end()) return;
  it->second.used = ++s.clock;
  if (it->second.disk) {
    const auto i = it->second.dsnaps.find(pos);
    if (i == it->second.dsnaps.end()) return;
    s.dpool.give_off(i->second, snap_bytes(spec != nullptr)), s.dbytes -= Pool::round(snap_bytes(spec != nullptr));
    it->second.dsnaps.erase(i);
    return;
  }
  const auto i = it->second.snaps.find(pos);
  if (i == it->second.snaps.end()) return;
  store_release(i->second, snap_bytes(spec != nullptr));
  it->second.snaps.erase(i);
}

void Model::state_reserve(size_t bytes) { d_->state_reserve(bytes); }
size_t Model::state_capacity() const { return d_->store_s ? d_->store_s->pool.size : 0; }
void Model::state_pin(const std::vector<int>& h) { d_->store_of().pinned = h; }
std::vector<int> Model::state_evicted() {
  std::vector<int> v;
  if (d_->store_s) v.swap(d_->store_s->evicted);
  return v;
}
void Model::state_save(int h, int from, int to) { d_->state_save(h, from, to); }
void Model::state_load(int h, int to) { d_->state_load(h, to); }
void Model::state_snap(int h, int pos) { d_->state_snap(h, pos); }
void Model::state_snap_end(int h, int pos) { d_->state_snap_end(h, pos); }
void Model::state_restore(int h, int pos) { d_->state_restore(h, pos); }
void Model::state_restore_end(int h) { d_->state_restore_end(h); }
void Model::state_drop(int h, int from) { d_->state_drop(h, from); }
void Model::state_free(int h) { d_->state_free(h); }
void Model::state_copy_snaps(int src, int dst, int upto) { d_->state_copy_snaps(src, dst, upto); }
void Model::state_drop_snap(int h, int pos) { d_->state_drop_snap(h, pos); }
uint64_t Model::state_kept_check(const std::string& base, uint64_t run_id, double max_age_s, size_t* bytes,
                                 std::string* why) {
  return d_->state_kept_check(base, run_id, max_age_s, bytes, why);
}
void Model::state_disk(const std::string& base, size_t bytes, size_t staging, bool reuse, uint64_t run_id) {
  d_->state_disk(base, bytes, staging, reuse, run_id);
}
void Model::state_keep(uint64_t gen) { d_->state_keep(gen); }
uint64_t Model::state_kept() const { return d_->store_s ? d_->store_s->loaded_gen : 0; }
std::vector<int> Model::state_sessions() const {
  std::vector<int> v;
  if (d_->store_s)
    for (const auto& kv : d_->store_s->sess) v.push_back(kv.first);
  return v;
}
size_t Model::state_disk_capacity() const { return d_->store_s && d_->store_s->disk ? d_->store_s->dpool.size : 0; }
size_t Model::state_disk_bytes() const { return d_->store_s ? d_->store_s->dbytes : 0; }
std::string Model::state_disk_path() const { return d_->store_s && d_->store_s->disk ? d_->store_s->disk->path() : ""; }
void Model::state_spill(int h) { d_->state_spill(h); }
void Model::state_unspill(int h) { d_->state_unspill(h); }
Model::StateInfo Model::state_info(int h) const {
  StateInfo r;
  if (!d_->store_s) return r;
  const auto it = d_->store_s->sess.find(h);
  if (it == d_->store_s->sess.end()) return r;
  const Store::Sess& ss = it->second;
  const bool spec = d_->spec != nullptr;
  r.saved = ss.saved;
  r.on_disk = ss.disk;
  if (ss.disk) {
    r.end_pos = ss.dend != Pool::kNone ? ss.end_pos : -1;
    for (const auto& sn : ss.dsnaps) r.snaps.push_back(sn.first);
    r.disk_bytes = ss.dpages.size() * Pool::round(kPageBytes) + ss.dsnaps.size() * Pool::round(snap_bytes(spec)) +
                   (ss.dend != Pool::kNone ? Pool::round(end_bytes(spec)) : 0);
    return r;
  }
  r.end_pos = ss.end ? ss.end_pos : -1;
  for (const auto& sn : ss.snaps) r.snaps.push_back(sn.first);
  r.bytes = ss.pages.size() * Pool::round(kPageBytes) + ss.snaps.size() * Pool::round(snap_bytes(spec)) +
            (ss.end ? Pool::round(end_bytes(spec)) : 0);
  return r;
}
size_t Model::state_bytes() const { return d_->store_s ? d_->store_s->bytes : 0; }
void Model::drain() { d_->drain(); }

}  // namespace model
