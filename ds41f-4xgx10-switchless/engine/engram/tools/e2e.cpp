// End-to-end Engram timing on one box: token ids known -> this rank's rows packed and ready.
//   engram_e2e <hashmap.bin> <rank> <shard-l1> <shard-l14> <stream.i32> [cache-MiB=4096] [spin-us] [cpu=-1]
//              [keepalive-ms] [read-batch]   (spin-us, keepalive-ms and read-batch default to StoreConfig's)
// Warms the cache on the first 1M tokens of the stream (as prefill), then measures
//   decode: steps of 1 and of 6 positions, ~15 ms apart, layer 1's rows and layer 14's rows;
//   prefill: 4096-position chunks of fresh text, warm cache and cold cache.
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/check.h"
#include "engram/engram.h"

using namespace engram;

static uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static std::vector<int32_t> read_stream(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) die("open %s", path);
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f) / 4;
  std::fseek(f, 0, SEEK_SET);
  std::vector<int32_t> v(n);
  if (std::fread(v.data(), 4, n, f) != size_t(n)) die("read %s", path);
  std::fclose(f);
  return v;
}

struct Pct {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  double at(double q) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[size_t(q * double(v.size() - 1))];
  }
};

int main(int argc, char** argv) {
  if (argc < 6)
    die("usage: %s <hashmap> <rank> <shard-l1> <shard-l14> <stream.i32> [cache-MiB] [spin-us] [cpu] [keepalive-ms] "
        "[read-batch]",
        argv[0]);
  const HashMap map = HashMap::load(argv[1]);
  StoreConfig cfg;
  cfg.rank = std::atoi(argv[2]);
  cfg.shard[0] = argv[3];
  cfg.shard[1] = argv[4];
  const std::vector<int32_t> toks = read_stream(argv[5]);
  cfg.cache_bytes = (argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 4096) << 20;
  if (argc > 7) cfg.spin_us = std::atoi(argv[7]);
  cfg.cpu = argc > 8 ? std::atoi(argv[8]) : -1;
  if (argc > 9) cfg.keepalive_ms = std::atoi(argv[9]);
  if (argc > 10) cfg.read_batch = std::atoi(argv[10]);
  cfg.queue_depth = 128;
  Store store(cfg, map);
  const Split& sp = store.split();
  std::atomic<uint64_t> done[kLayers];
  done[0] = done[1] = 0;
  uint64_t seq = 0;
  std::vector<uint32_t> ids;
  std::vector<uint8_t> out[kLayers];

  // one request per layer for positions [pos, pos+n); returns the time each layer's rows were ready
  auto run = [&](size_t pos, int n, uint64_t* t_ready, uint64_t* t_hash) {
    const uint64_t t0 = now_ns();
    ids.resize(size_t(n) * kLayers * kCols);
    hash_positions(map, toks.data(), nullptr, int64_t(pos), n, ids.data());
    const uint64_t t1 = now_ns();
    Request r[kLayers];
    for (int l = 0; l < kLayers; l++) {
      CellMap cm;
      cm.build(sp, l, ids.data(), n);
      out[l].resize(size_t(cm.count[cfg.rank]) * kRowBytes + 1);
      r[l].layer = l;
      r[l].ids = ids.data();
      r[l].n_pos = n;
      r[l].out = out[l].data();
      r[l].done = &done[l];
      r[l].seq = ++seq;
      store.submit(r[l]);
    }
    for (int l = 0; l < kLayers; l++) {
      Store::wait(r[l]);
      t_ready[l] = now_ns() - t0;
    }
    *t_hash = t1 - t0;
  };

  const size_t warm = std::min<size_t>(1000000, toks.size() / 2);
  uint64_t tr[kLayers], th;
  {
    const uint64_t t0 = now_ns();
    for (size_t p = 0; p + 4096 <= warm; p += 4096) run(p, 4096, tr, &th);
    const StoreStats s = store.stats();
    std::printf("rank %d, cache %zu MiB, spin %d us, keep-alive %d ms, read batch %d: warmed on %zu tokens in %.1f s; "
                "hit rate %.1f%%, %llu reads\n",
                cfg.rank, cfg.cache_bytes >> 20, cfg.spin_us, cfg.keepalive_ms, cfg.read_batch, warm,
                double(now_ns() - t0) * 1e-9, 100.0 * double(s.hits) / double(s.lookups), (unsigned long long)s.reads);
  }
  size_t pos = warm;
  for (int k : {1, 6}) {
    Pct l1, l14, hash, served[2];  // served: the slower layer, steps served from the cache [0] or not [1]
    const StoreStats s0 = store.stats();
    for (int step = 0; step < 300; step++) {
      const uint64_t r0 = store.stats().reads;
      run(pos, k, tr, &th);
      pos += size_t(k);
      l1.add(double(tr[0]) * 1e-3);
      l14.add(double(tr[1]) * 1e-3);
      hash.add(double(th) * 1e-3);
      served[store.stats().reads != r0].add(double(std::max(tr[0], tr[1])) * 1e-3);
      timespec ts{0, 15 * 1000 * 1000};
      nanosleep(&ts, nullptr);
    }
    const StoreStats s1 = store.stats();
    std::printf("decode, %d position%s per step, 300 steps 15 ms apart: hash %.1f us; rows ready (us) layer 1 "
                "p50 %.0f p90 %.0f p99 %.0f max %.0f | layer 14 p50 %.0f p99 %.0f; hits %.1f%%, %.1f reads/step "
                "(+%.1f padding)\n",
                k, k > 1 ? "s" : "", hash.at(0.5), l1.at(0.5), l1.at(0.9), l1.at(0.99), l1.at(1.0), l14.at(0.5),
                l14.at(0.99), 100.0 * double(s1.hits - s0.hits) / double(s1.lookups - s0.lookups),
                double(s1.reads - s0.reads) / 300.0, double(s1.pad_reads - s0.pad_reads) / 300.0);
    for (int m = 0; m < 2; m++)
      std::printf("  steps %s: %zu, both layers ready (us) p50 %.0f p90 %.0f p99 %.0f max %.0f\n",
                  m ? "with reads" : "all hits", served[m].v.size(), served[m].at(0.5), served[m].at(0.9),
                  served[m].at(0.99), served[m].at(1.0));
  }
  for (int cold = 0; cold < 2; cold++) {
    if (cold) store.clear_cache();
    Pct chunk;
    const StoreStats s0 = store.stats();
    const uint64_t t0 = now_ns();
    int chunks = 0;
    for (; chunks < 8 && pos + 4096 <= toks.size(); chunks++, pos += 4096) {
      run(pos, 4096, tr, &th);
      chunk.add(double(std::max(tr[0], tr[1])) * 1e-6);
    }
    const double el = double(now_ns() - t0) * 1e-9;
    const StoreStats s1 = store.stats();
    std::printf("prefill, 4096-position chunks, %s cache: %d chunks, %.1f ms median per chunk (max %.1f) -> %.0f tok/s "
                "if not overlapped; hits %.1f%%, %.0f reads/s\n",
                cold ? "cold" : "warm", chunks, chunk.at(0.5), chunk.at(1.0), chunks * 4096.0 / el,
                100.0 * double(s1.hits - s0.hits) / double(s1.lookups - s0.lookups), double(s1.reads - s0.reads) / el);
  }
  return 0;
}
