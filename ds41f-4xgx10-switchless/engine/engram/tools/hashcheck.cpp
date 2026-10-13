// Checks the C++ hashing against the reference's test cases (engine/engram/py/make_hashmap.py),
// whole and fed in chunks, and times it.
//   engram_hashcheck <engram_hashmap.bin> <hash_cases.bin>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "common/check.h"
#include "engram/engram.h"

using namespace engram;

int main(int argc, char** argv) {
  if (argc != 3) die("usage: %s <engram_hashmap.bin> <hash_cases.bin>", argv[0]);
  const HashMap m = HashMap::load(argv[1]);
  std::printf("hash map: %lld tokens -> %lld compressed ids, pad %lld, layers %lld,%lld; primes and offsets check\n",
              (long long)m.n_tokens, (long long)m.vocab, (long long)m.pad, (long long)m.layer_ids[0],
              (long long)m.layer_ids[1]);
  FILE* f = std::fopen(argv[2], "rb");
  if (!f) die("cannot open %s", argv[2]);
  std::mt19937 rng(7);
  int n_cases = 0, n_bad = 0;
  long long n_ids = 0;
  std::vector<int32_t> longest;
  for (;;) {
    char magic[4];
    int64_t L, has_mask;
    if (std::fread(magic, 1, 4, f) != 4) break;
    if (std::memcmp(magic, "ENGC", 4) != 0 || std::fread(&L, 8, 1, f) != 1 || std::fread(&has_mask, 8, 1, f) != 1)
      die("bad case header");
    std::vector<int32_t> tok(L);
    std::vector<uint8_t> live(L);
    std::vector<int64_t> want(size_t(L) * kLayers * kCols);
    if (std::fread(tok.data(), 4, L, f) != size_t(L) || std::fread(live.data(), 1, L, f) != size_t(L) ||
        std::fread(want.data(), 8, want.size(), f) != want.size())
      die("short case");
    const uint8_t* lv = has_mask ? live.data() : nullptr;
    std::vector<uint32_t> got(want.size());
    int bad = 0;
    // whole sequence, one position at a time, and in random chunks
    for (int mode = 0; mode < 3; mode++) {
      std::fill(got.begin(), got.end(), 0xffffffffu);
      for (int64_t pos = 0; pos < L;) {
        int n = mode == 0 ? int(L) : mode == 1 ? 1 : 1 + int(rng() % 97);
        if (pos + n > L) n = int(L - pos);
        hash_positions(m, tok.data(), lv, pos, n, got.data() + size_t(pos) * kLayers * kCols);
        pos += n;
      }
      for (size_t i = 0; i < want.size(); i++)
        if (int64_t(got[i]) != want[i]) {
          if (bad++ < 3)
            std::printf("  case %d mode %d: id %zu (pos %zu) got %u want %lld\n", n_cases, mode, i,
                        i / (kLayers * kCols), got[i], (long long)want[i]);
        }
    }
    std::printf("case %2d: %5lld positions%s: %s\n", n_cases, (long long)L, has_mask ? " (masked span)" : "",
                bad ? "MISMATCH" : "ok (whole, by 1, by random chunks)");
    n_bad += bad != 0;
    n_ids += (long long)want.size();
    if (L > (int64_t)longest.size()) longest = tok;
    n_cases++;
  }
  std::fclose(f);
  std::printf("%d cases, %lld row ids each checked 3 ways: %s\n", n_cases, n_ids, n_bad ? "FAILED" : "all equal");

  // throughput on the longest case
  std::vector<uint32_t> out(longest.size() * kLayers * kCols);
  const int reps = 200;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; r++) hash_positions(m, longest.data(), nullptr, 0, int(longest.size()), out.data());
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("hashing: %.0f ns per position (%zu positions x %d reps, one thread)\n", s * 1e9 / (reps * longest.size()),
              longest.size(), reps);
  return n_bad ? 1 : 0;
}
