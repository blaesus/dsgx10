// Hit rates of the hot-row cache on real token streams, with the store's exact replacement policy:
// each rank's cache sees only the rows it owns. Reported per capacity over the whole stream and
// from a later position on (default: after the first quarter), plus each rank's working set.
//   engram_cachesim <hashmap.bin> [split=kit|col] [caps=MiB,...] [from=pos|from=last] <stream.i32> [more...]
// Several streams are concatenated in the order given (e.g. code then prose); from=last measures the
// last stream only, with the cache warmed by the ones before it.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "common/check.h"
#include "engram/cache.h"
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
  if (argc < 3)
    die("usage: %s <hashmap.bin> [split=kit|col] [caps=MiB,...] [from=pos|last] <stream.i32> [more...]", argv[0]);
  const HashMap map = HashMap::load(argv[1]);
  const Split sp = Split::of(map);
  std::vector<size_t> caps_mib = {256, 1024, 2048, 4096, 8192};
  std::vector<int32_t> toks;
  long from = -1;  // -1: the first quarter
  bool from_last = false;
  bool by_col = false;  // split=col: rank r owns columns c with c % 4 == r (two heads of every n-gram size)
  for (int a = 2; a < argc; a++) {
    std::string s = argv[a];
    if (s == "split=col" || s == "split=kit") {
      by_col = s == "split=col";
      continue;
    }
    if (s.rfind("caps=", 0) == 0) {
      caps_mib.clear();
      for (size_t i = 5; i < s.size();) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        caps_mib.push_back(std::strtoull(s.substr(i, j - i).c_str(), nullptr, 10));
        i = j + 1;
      }
      continue;
    }
    if (s.rfind("from=", 0) == 0) {
      from_last = s == "from=last";
      if (!from_last) from = std::atol(s.c_str() + 5);
      continue;
    }
    std::vector<int32_t> more = read_stream(argv[a]);
    if (from_last) from = long(toks.size());
    toks.insert(toks.end(), more.begin(), more.end());
  }
  const size_t n = toks.size(), warm_from = from < 0 ? n / 4 : size_t(from);
  std::vector<uint32_t> ids(n * kLayers * kCols);
  for (size_t p = 0; p < n; p += 4096) {
    const int k = int(n - p < 4096 ? n - p : 4096);
    hash_positions(map, toks.data(), nullptr, int64_t(p), k, ids.data() + p * kLayers * kCols);
  }
  // each rank's key stream, in order
  std::vector<std::vector<uint64_t>> keys(kWorld);
  std::vector<size_t> warm_at(kWorld);
  for (size_t p = 0; p < n; p++) {
    if (p == warm_from)
      for (int r = 0; r < kWorld; r++) warm_at[r] = keys[r].size();
    for (int l = 0; l < kLayers; l++)
      for (int c = 0; c < kCols; c++) {
        const uint32_t row = ids[(p * kLayers + l) * kCols + c];
        keys[by_col ? c % kWorld : sp.owner(l, row)].push_back((uint64_t(l) << 32) | row);
      }
  }
  std::vector<uint32_t>().swap(ids);
  std::printf("%zu positions, %s split; owned lookups per position:", n, by_col ? "column" : "kit (row range)");
  for (int r = 0; r < kWorld; r++) std::printf(" %.2f", double(keys[r].size()) / double(n));
  std::printf("\n%10s  %-31s  from position %zu on\n", "cache MiB", "hit % rank 0..3, all positions", warm_from);
  for (size_t mib : caps_mib) {
    double all[kWorld], warm[kWorld];
    std::vector<std::thread> th;
    for (int r = 0; r < kWorld; r++)
      th.emplace_back([&, r] {
        ClockIndex idx((size_t(mib) << 20) / kRowBytes);
        uint64_t hit = 0, hit_w = 0;
        const std::vector<uint64_t>& k = keys[r];
        for (size_t i = 0; i < k.size(); i++) {
          if (idx.find(k[i]) != kNoSlot) {
            hit++;
            hit_w += i >= warm_at[r];
          } else {
            bool ev;
            idx.insert(k[i], &ev);
          }
        }
        all[r] = 100.0 * double(hit) / double(k.size());
        warm[r] = 100.0 * double(hit_w) / double(k.size() - warm_at[r]);
      });
    for (auto& t : th) t.join();
    std::printf("%10zu  %7.1f %7.1f %7.1f %7.1f  %7.1f %7.1f %7.1f %7.1f\n", mib, all[0], all[1], all[2], all[3],
                warm[0], warm[1], warm[2], warm[3]);
  }
  // the working set: distinct rows each rank touches over the whole stream
  std::printf("distinct rows per rank (MiB of records):");
  for (int r = 0; r < kWorld; r++) {
    std::vector<uint64_t> k = keys[r];
    std::sort(k.begin(), k.end());
    const size_t d = size_t(std::unique(k.begin(), k.end()) - k.begin());
    std::printf(" %.0f (%.0f%% of lookups)", double(d) * kRowBytes / 1048576.0, 100.0 * double(d) / double(k.size()));
  }
  std::printf("\n");
  return 0;
}
