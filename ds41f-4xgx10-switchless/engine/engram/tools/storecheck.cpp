// Checks the row store end to end against plain pread: for many requests (1 .. 2048 positions,
// both layers, repeated n-grams so the cache-hit and join paths run), every record the store
// returns must equal the record read directly from the source.
//   engram_storecheck <hashmap.bin> rank <r> <shard-l1> <shard-l14> [requests]   kit shards of rank r
//   engram_storecheck <hashmap.bin> ckpt <r|-1> <checkpoint-dir> [requests]      the checkpoint (r = -1: every row)
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "engram/engram.h"
#include "weights/safetensors.h"

using namespace engram;

int main(int argc, char** argv) {
  if (argc < 5) die("usage: see the header of storecheck.cpp");
  const HashMap map = HashMap::load(argv[1]);
  const std::string mode = argv[2];
  StoreConfig cfg;
  cfg.rank = std::atoi(argv[3]);
  cfg.cache_bytes = size_t(64) << 20;  // small, so eviction runs too
  cfg.queue_depth = 64;
  cfg.read_batch = 8;  // padding reads run alongside the real ones
  int n_req = 300;
  // reference reader: record of (layer, row) by pread
  std::vector<int> ref_fd(kLayers), ref_sfd(kLayers);
  std::vector<uint64_t> ref_base(kLayers), ref_sbase(kLayers), ref_lo(kLayers);
  bool ref_two = false;
  if (mode == "rank") {
    if (argc < 6) die("rank mode needs both shards");
    cfg.source = Source::KitShard;
    cfg.shard[0] = argv[4];
    cfg.shard[1] = argv[5];
    if (argc > 6) n_req = std::atoi(argv[6]);
    for (int l = 0; l < kLayers; l++) {
      ref_fd[l] = open(cfg.shard[l].c_str(), O_RDONLY);
      uint64_t h[6];
      if (ref_fd[l] < 0 || pread(ref_fd[l], h, sizeof(h), 0) != sizeof(h)) die("open %s", cfg.shard[l].c_str());
      ref_base[l] = 4096;
      ref_lo[l] = h[2];
    }
  } else if (mode == "ckpt") {
    cfg.source = Source::Checkpoint;
    cfg.checkpoint = argv[4];
    if (argc > 5) n_req = std::atoi(argv[5]);
    const dsw::Checkpoint ck = dsw::Checkpoint::open(cfg.checkpoint);
    ref_two = true;
    for (int l = 0; l < kLayers; l++) {
      const std::string pre = "layers." + std::to_string(map.layer_ids[l]) + ".engram.embed.";
      const dsw::SrcTensor& w = ck.at(pre + "weight");
      const dsw::SrcTensor& s = ck.at(pre + "scale");
      ref_fd[l] = open(ck.shards[w.shard].c_str(), O_RDONLY);
      ref_sfd[l] = open(ck.shards[s.shard].c_str(), O_RDONLY);
      ref_base[l] = w.offset;
      ref_sbase[l] = s.offset;
    }
  } else {
    die("mode is rank or ckpt");
  }
  Store store(cfg, map);
  const Split& sp = store.split();

  // a token stream with repeats: random tokens, and every so often a replay of an earlier span
  uint64_t x = 0x1234567ull;
  auto rnd = [&]() {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
  };
  std::vector<int32_t> toks;
  while (toks.size() < 200000) {
    if (toks.size() > 1000 && rnd() % 4 == 0) {
      const size_t from = rnd() % (toks.size() - 500), len = 20 + rnd() % 400;
      for (size_t i = 0; i < len; i++) toks.push_back(toks[from + i]);
    } else {
      for (int i = 0; i < 200; i++) toks.push_back(int32_t(rnd() % map.n_tokens));
    }
  }
  const int sizes[] = {1, 6, 1, 64, 6, 512, 1, 2048};
  long checked = 0, bad = 0;
  size_t pos = 0;
  std::atomic<uint64_t> done{0};
  std::vector<uint32_t> ids;
  std::vector<uint8_t> out;
  for (int q = 0; q < n_req; q++) {
    const int n = sizes[q % 8];
    if (pos + n > toks.size()) pos = rnd() % 1000;
    ids.resize(size_t(n) * kLayers * kCols);
    hash_positions(map, toks.data(), nullptr, int64_t(pos), n, ids.data());
    for (int layer = 0; layer < kLayers; layer++) {
      CellMap cm;
      cm.build(sp, layer, ids.data(), n);
      const size_t recs = cfg.rank >= 0 ? cm.count[cfg.rank] : size_t(n) * kCols;
      out.assign(recs * kRowBytes + 1, 0xA5);
      Request r;
      r.layer = layer;
      r.ids = ids.data();
      r.n_pos = n;
      r.out = out.data();
      r.done = &done;
      r.seq = uint64_t(q) * 2 + layer + 1;
      store.submit(r);
      Store::wait(r);
      if (out[recs * kRowBytes] != 0xA5) die("store wrote past its segment");
      for (int p = 0; p < n; p++)
        for (int c = 0; c < kCols; c++) {
          if (cfg.rank >= 0 && cm.owner[size_t(p) * kCols + c] != cfg.rank) continue;
          const uint32_t row = ids[(size_t(p) * kLayers + layer) * kCols + c];
          const size_t at = cfg.rank >= 0 ? cm.index[size_t(p) * kCols + c] : size_t(p) * kCols + c;
          uint8_t want[kRowBytes];
          bool ok;
          if (!ref_two) {
            ok = pread(ref_fd[layer], want, kRowBytes, off_t(ref_base[layer] + (row - ref_lo[layer]) * kRowBytes)) ==
                 kRowBytes;
          } else {
            ok = pread(ref_fd[layer], want, kWeightBytes, off_t(ref_base[layer] + uint64_t(row) * kWeightBytes)) ==
                     kWeightBytes &&
                 pread(ref_sfd[layer], want + kWeightBytes, 8, off_t(ref_sbase[layer] + uint64_t(row) * 8)) == 8;
          }
          if (!ok) die("reference read failed");
          checked++;
          if (std::memcmp(want, out.data() + at * kRowBytes, kRowBytes) != 0 && bad++ < 5)
            std::printf("  request %d layer %d pos %d col %d row %u differs\n", q, layer, p, c, row);
        }
    }
    pos += size_t(n);
  }
  const StoreStats s = store.stats();
  std::printf("%s, rank %d: %d requests x 2 layers, %ld records checked: %s\n",
              mode == "rank" ? "kit shards" : "checkpoint", cfg.rank, n_req, checked, bad ? "MISMATCH" : "all equal");
  std::printf("  lookups %llu, cache hits %llu (%.1f%%), joined in-flight %llu, reads %llu, evictions %llu, padding reads %llu\n",
              (unsigned long long)s.lookups, (unsigned long long)s.hits, 100.0 * double(s.hits) / double(s.lookups),
              (unsigned long long)s.joined, (unsigned long long)s.reads, (unsigned long long)s.evictions,
              (unsigned long long)s.pad_reads);
  return bad ? 1 : 0;
}
