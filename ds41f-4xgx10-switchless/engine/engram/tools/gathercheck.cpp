// The whole hand-off on one box: four stores, one per rank, read the rows they own from the
// checkpoint into their segments of one buffer, seg_bytes() apart, as comm::allgather lays them
// out; every cell's row, found through CellMap::offsets, must equal what a single-box store
// (rank -1) returns for that cell, and no store may write past its records.
//   engram_gathercheck <hashmap.bin> <checkpoint-dir> <stream.i32> [requests=20]
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/check.h"
#include "engram/engram.h"

using namespace engram;

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

int main(int argc, char** argv) {
  if (argc < 4) die("usage: %s <hashmap.bin> <checkpoint-dir> <stream.i32> [requests]", argv[0]);
  const HashMap map = HashMap::load(argv[1]);
  const std::vector<int32_t> toks = read_stream(argv[3]);
  const int n_req = argc > 4 ? std::atoi(argv[4]) : 20;
  StoreConfig base;
  base.source = Source::Checkpoint;
  base.checkpoint = argv[2];
  base.cache_bytes = size_t(64) << 20;
  base.queue_depth = 64;
  std::vector<std::unique_ptr<Store>> rank;
  for (int r = 0; r < kWorld; r++) {
    StoreConfig c = base;
    c.rank = r;
    rank.push_back(std::make_unique<Store>(c, map));
  }
  StoreConfig one = base;
  one.rank = -1;
  Store single(one, map);
  const Split& sp = single.split();

  std::atomic<uint64_t> done[kWorld + 1];
  for (auto& d : done) d = 0;
  uint64_t x = 0x9E3779B97F4A7C15ull, seq = 0, cells = 0, bad = 0;
  size_t most_seg = 0;
  std::vector<uint32_t> ids, off;
  std::vector<uint8_t> gathered, grid;
  const int sizes[] = {1, 6, 37, 512, 2048};
  for (int q = 0; q < n_req; q++) {
    const int n = sizes[q % 5];
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    const size_t pos = q < 5 ? 0 : x % (toks.size() - size_t(n));  // the first requests start the sequence
    ids.resize(size_t(n) * kLayers * kCols);
    hash_positions(map, toks.data(), nullptr, int64_t(pos), n, ids.data());
    for (int layer = 0; layer < kLayers; layer++) {
      CellMap cm;
      cm.build(sp, layer, ids.data(), n);
      const size_t seg = cm.seg_bytes();
      most_seg = std::max(most_seg, seg);
      gathered.assign(seg * kWorld + 1, 0x5A);
      grid.assign(size_t(n) * kCols * kRowBytes, 0);
      Request req[kWorld + 1];
      seq++;
      for (int r = 0; r <= kWorld; r++) {
        req[r].layer = layer;
        req[r].ids = ids.data();
        req[r].n_pos = n;
        req[r].out = r < kWorld ? gathered.data() + size_t(r) * seg : grid.data();
        req[r].done = &done[r];
        req[r].seq = seq;
        (r < kWorld ? *rank[r] : single).submit(req[r]);
      }
      for (int r = 0; r <= kWorld; r++) Store::wait(req[r]);
      off.resize(size_t(n) * kCols);
      cm.offsets(off.data());
      for (size_t i = 0; i < off.size(); i++, cells++)
        if (std::memcmp(gathered.data() + off[i], grid.data() + i * kRowBytes, kRowBytes) != 0 && bad++ < 5)
          std::printf("  request %d layer %d cell %zu: gathered row differs from the single-box row\n", q, layer, i);
      for (int r = 0; r < kWorld; r++)  // the padding after each rank's records is untouched
        for (size_t b = size_t(cm.count[r]) * kRowBytes; b < seg; b++)
          if (gathered[size_t(r) * seg + b] != 0x5A) die("rank %d wrote past its %u records", r, cm.count[r]);
    }
  }
  std::printf("%d requests x 2 layers, %llu cells: %s; largest segment %zu bytes\n", n_req,
              (unsigned long long)cells, bad ? "MISMATCH" : "gathered rows equal the single-box rows", most_seg);
  for (int r = 0; r < kWorld; r++) {
    const StoreStats s = rank[r]->stats();
    std::printf("  rank %d: %llu lookups, %llu reads\n", r, (unsigned long long)s.lookups, (unsigned long long)s.reads);
  }
  return bad ? 1 : 0;
}
