// Prompt-lookup drafts, a feasibility study (docs/engine.md), Q3: what a host-side prompt-lookup matcher costs per
// step over a long history. pld_study.py's Index, in C++. The index: for each n in
// [2, 8], an open-addressing table from a hash of n tokens to the latest position where they end (the tokens are
// compared on a hit, so a collision never proposes wrongly). A step appends its emitted tokens (each adds its
// n-grams) and looks up the longest suffix match, n = 8 .. 2, whose continuation is the proposal.
//   g++ -O2 -o pld_match pld_match.cpp
//   pld_match TOKENS.bin [history] [steps]   (TOKENS.bin: int32 ids; repeated to the history's length)
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

constexpr int NMIN = 2, NMAX = 8;

struct Table {
  std::vector<uint64_t> key;  // hash | 1 (0: empty)
  std::vector<int32_t> pos;
  uint64_t mask;
  explicit Table(int bits) : key(1ull << bits, 0), pos(1ull << bits, -1), mask((1ull << bits) - 1) {}
};

struct Index {
  std::vector<int32_t> h;
  std::vector<Table> t;
  explicit Index(int bits) { for (int n = 0; n <= NMAX; ++n) t.emplace_back(n >= NMIN ? bits : 1); }
  static uint64_t hash(const int32_t* p, int n) {
    uint64_t x = 0x9e3779b97f4a7c15ull ^ (uint64_t)n;
    for (int i = 0; i < n; ++i) x = (x ^ (uint32_t)p[i]) * 0xff51afd7ed558ccdull, x ^= x >> 29;
    return x | 1;
  }
  void append(int32_t tok) {
    h.push_back(tok);
    index_last();
  }
  void index_last() {  // the n-grams that end at the history's last token
    const int e = (int)h.size() - 1;
    for (int n = NMIN; n <= NMAX && n <= e + 1; ++n) {
      Table& T = t[n];
      const uint64_t k = hash(&h[e - n + 1], n);
      for (uint64_t s = k & T.mask;; s = (s + 1) & T.mask) {
        if (T.key[s] == 0 || (T.key[s] == k && std::equal(&h[T.pos[s] - n + 1], &h[T.pos[s] + 1], &h[e - n + 1]))) {
          T.key[s] = k, T.pos[s] = e;
          break;
        }
      }
    }
  }
  // The longest suffix (n = NMAX .. NMIN) of suf that occurred in the indexed history: up to kmax tokens after its
  // latest occurrence (the history's last token is pushed but not indexed yet, so a match ends before it).
  int lookup(const int32_t* suf, int len, int kmax, int32_t* out) const {
    const int end = (int)h.size();
    for (int n = std::min(NMAX, len); n >= NMIN; --n) {
      const Table& T = t[n];
      const uint64_t k = hash(suf + len - n, n);
      for (uint64_t s = k & T.mask; T.key[s]; s = (s + 1) & T.mask) {
        if (T.key[s] == k && std::equal(&h[T.pos[s] - n + 1], &h[T.pos[s] + 1], suf + len - n)) {
          const int j = T.pos[s];
          const int m = std::min(kmax, end - 1 - j);
          if (m <= 0) break;
          std::memcpy(out, &h[j + 1], m * 4);
          return m;
        }
      }
    }
    return 0;
  }
};

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "pld_match TOKENS.bin [history] [steps]\n"), 2;
  std::FILE* f = std::fopen(argv[1], "rb");
  if (!f) return std::fprintf(stderr, "cannot read %s\n", argv[1]), 1;
  std::vector<int32_t> src;
  int32_t x;
  while (std::fread(&x, 4, 1, f) == 1) src.push_back(x);
  std::fclose(f);
  const int H = argc > 2 ? std::atoi(argv[2]) : 250000, S = argc > 3 ? std::atoi(argv[3]) : 20000;
  int bits = 1;
  while ((1 << bits) < 2 * (H + S)) ++bits;
  Index ix(bits);
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  for (int i = 0; i < H; ++i) ix.append(src[i % src.size()] + (int32_t)(i / src.size()) * 7 % 3);  // vary repeats a little
  const double build = std::chrono::duration<double>(clk::now() - t0).count();
  // steps: emit 1..16 tokens of the source (a lookup before the append, as the step that verifies next would)
  std::vector<double> dt;
  int32_t out[16];
  long hits = 0, prop = 0;
  size_t p = H;
  for (int s = 0; s < S; ++s) {
    const int e = 1 + s % 16;
    const auto a = clk::now();
    for (int j = 0; j < e - 1; ++j) ix.append(src[(p + j) % src.size()]);
    ix.h.push_back(src[(p + e - 1) % src.size()]);  // the step's last token: the next step's anchor
    p += e;
    const int m = ix.lookup(&ix.h[ix.h.size() - NMAX], NMAX, 15, out);
    ix.index_last();
    dt.push_back(std::chrono::duration<double>(clk::now() - a).count());
    hits += m > 0, prop += m;
  }
  std::sort(dt.begin(), dt.end());
  size_t bytes = ix.h.size() * 4;
  for (const Table& T : ix.t) bytes += T.key.size() * 12;
  std::printf("history %d: index built in %.1f ms (%.0f ns a token), %.1f MiB; a step (1..16 tokens appended, one "
              "lookup): median %.2f us, p99 %.2f us, max %.2f us; %ld of %d steps proposed (%.1f tokens each)\n",
              H, 1e3 * build, 1e9 * build / H, bytes / 1048576.0, 1e6 * dt[dt.size() / 2], 1e6 * dt[dt.size() * 99 / 100],
              1e6 * dt.back(), hits, S, hits ? (double)prop / hits : 0.0);
  return 0;
}
