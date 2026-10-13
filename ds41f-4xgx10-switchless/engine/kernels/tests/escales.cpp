// escales: the routed experts' ue8m0 block scales in a .dsw rank file, for a compact scale format in the MoE
// decode kernels (docs/engine.md, "Compact expert scales"). CPU only: streams each experts' scale tensor (pread,
// then posix_fadvise DONTNEED behind, so the page cache keeps nothing) in the order the kernels read it (L3), and
// counts the distinct values per (layer, expert, matrix), per (layer, expert) over w1, w3 and w2, and per 16-row
// tile.
//   escales FILE --list                        the experts' scale tensors: name, dtype, layout, shape
//   escales FILE [--threads T] [--json OUT]    the counts (stdout), and with --json one line per tensor
//   escales FILE [--threads T] --groups OUT    the 1-bit code's 32-scale groups (stdout), one line per tensor
// Build: g++ -O2 -std=c++17 -pthread -I engine -o escales engine/kernels/tests/escales.cpp
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "weights/dsw.h"

namespace {

int g_fd = -1;
std::atomic<uint64_t> g_read{0};

[[noreturn]] void die(const char* m, const char* a = "") {
  std::fprintf(stderr, "escales: %s%s\n", m, a);
  std::exit(1);
}

// [off, off + n) of the file into buf, then its pages dropped
void rd(uint64_t off, uint64_t n, uint8_t* buf) {
  uint64_t got = 0;
  while (got < n) {
    const ssize_t r = pread(g_fd, buf + got, n - got, (off_t)(off + got));
    if (r <= 0) die("short read");
    got += (uint64_t)r;
  }
  posix_fadvise(g_fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
  g_read += n;
}

std::vector<dsw::TocEntry> toc;

void load_toc(const char* path) {
  g_fd = open(path, O_RDONLY);
  if (g_fd < 0) die("cannot open ", path);
  dsw::Header h;
  rd(0, sizeof h, reinterpret_cast<uint8_t*>(&h));
  if (std::memcmp(h.magic, dsw::kMagic, 8)) die("not a .dsw file: ", path);
  toc.resize(h.n_tensors);
  rd(h.toc_offset, (uint64_t)h.n_tensors * sizeof(dsw::TocEntry), reinterpret_cast<uint8_t*>(toc.data()));
}

bool is_expert_scale(const dsw::TocEntry& e) {
  const std::string n(e.name, strnlen(e.name, sizeof e.name));
  return n.find(".ffn.experts.w") != std::string::npos && n.size() > 6 && n.compare(n.size() - 6, 6, ".scale") == 0;
}

// One tensor's counts: [E][N][KB] scales, tiled (L3) or natural.
struct Counts {
  std::string name;
  int E = 0, N = 0, KB = 0;
  uint32_t layout = 0;
  uint64_t hist[256] = {};                 // all values
  std::vector<std::array<uint64_t, 4>> m;  // per expert: a 256-bit set of the values present
  std::vector<int> tile_distinct;          // per (expert, 16-row tile): distinct values
  uint64_t tile_hist[257] = {};            // tiles by distinct count
};

int popc256(const std::array<uint64_t, 4>& s) {
  return __builtin_popcountll(s[0]) + __builtin_popcountll(s[1]) + __builtin_popcountll(s[2]) +
         __builtin_popcountll(s[3]);
}

void count(const dsw::TocEntry& e, Counts& c, std::vector<uint8_t>& buf) {
  c.name = std::string(e.name, strnlen(e.name, sizeof e.name));
  if (e.ndim != 3 || e.dtype != (uint32_t)dsw::DType::F8_E8M0) die("unexpected tensor: ", c.name.c_str());
  c.E = (int)e.shape[0], c.N = (int)e.shape[1], c.KB = (int)e.shape[2];
  c.layout = e.layout;
  const uint64_t per = (uint64_t)c.N * c.KB;  // bytes per expert: tiled per matrix, so expert e is [e*per, +per)
  if (per * c.E != e.bytes) die("size mismatch: ", c.name.c_str());
  c.m.assign(c.E, {0, 0, 0, 0});
  c.tile_distinct.assign((size_t)c.E * (c.N / 16), 0);
  buf.resize(per);
  for (int x = 0; x < c.E; ++x) {
    rd(e.offset + per * x, per, buf.data());
    auto& s = c.m[x];
    for (uint64_t i = 0; i < per; ++i) {
      const uint8_t v = buf[i];
      ++c.hist[v];
      s[v >> 6] |= 1ull << (v & 63);
    }
    // a 16-row tile is [16][KB] natural, or [KB/2 steps][8 g][4 j] contiguous in L3: the same bytes either way
    for (int t = 0; t < c.N / 16; ++t) {
      std::array<uint64_t, 4> ts = {0, 0, 0, 0};
      const uint8_t* p = buf.data() + (uint64_t)t * 16 * c.KB;
      for (int i = 0; i < 16 * c.KB; ++i) ts[p[i] >> 6] |= 1ull << (p[i] & 63);
      const int d = popc256(ts);
      c.tile_distinct[(size_t)x * (c.N / 16) + t] = d;
      ++c.tile_hist[d];
    }
  }
}

// --groups: the 1-bit code's groups, 32 consecutive bytes of L3 (one tile-step: the bytes a warp-step reads),
// against the tensor's best pair {b, b + 1} (wfix's rule), against {120, 121} and against the quad b - 1 .. b + 2;
// and where the groups outside the pair (read raw) fall for the kernels: per moe_up warp slice (20 steps of a
// w1/w3 tile, WK = 4) and per moe_down unit (a whole w2 tile, 9 steps).
struct Groups {
  std::string name;
  int base = 0, P = 0;
  uint64_t groups = 0, in_pair = 0, in_120 = 0, in_quad = 0;
  uint64_t slices = 0, sl_raw = 0, sl_raw8 = 0, sl_raw4 = 0;  // slices with a raw group; with one in the first 8, 4 steps
  uint64_t tiles = 0, t_raw = 0;
  uint64_t per_slice[33] = {};  // slices by their raw groups (32: 32 or more)
};

void groups(const dsw::TocEntry& e, Groups& r, std::vector<uint8_t>& buf) {
  r.name = std::string(e.name, strnlen(e.name, sizeof e.name));
  if (e.ndim != 3 || e.dtype != (uint32_t)dsw::DType::F8_E8M0 || e.layout != (uint32_t)dsw::Layout::L3)
    die("expected an L3 ue8m0 tensor: ", r.name.c_str());
  const uint64_t E = e.shape[0], T = e.shape[1] / 16;
  const int P = (int)e.shape[2] / 2;
  r.P = P;
  buf.resize(e.bytes);
  for (uint64_t off = 0; off < e.bytes; off += 64u << 20) rd(e.offset + off, std::min<uint64_t>(64u << 20, e.bytes - off), buf.data() + off);
  uint64_t hs[256] = {};
  for (uint8_t v : buf) ++hs[v];
  const int b0 = (int)(std::max_element(hs, hs + 256) - hs);
  r.base = b0 > 0 && hs[b0 - 1] > (b0 < 255 ? hs[b0 + 1] : 0) ? b0 - 1 : b0;
  const int b = r.base, sl = P >= 80 ? 20 : P;
  for (uint64_t x = 0; x < E; ++x)
    for (uint64_t t = 0; t < T; ++t) {
      bool traw = false;
      for (int s0 = 0; s0 < P; s0 += sl) {
        int nraw = 0;
        bool r8 = false, r4 = false;
        for (int p = s0; p < std::min(P, s0 + sl); ++p) {
          const uint8_t* g = buf.data() + ((x * T + t) * P + p) * 32;
          bool pair = true, p120 = true, quad = true;
          for (int j = 0; j < 32; ++j) {
            const int v = g[j];
            pair = pair && (v == b || v == b + 1);
            p120 = p120 && (v == 120 || v == 121);
            quad = quad && v >= b - 1 && v <= b + 2;
          }
          ++r.groups, r.in_pair += pair, r.in_120 += p120, r.in_quad += quad;
          if (!pair) ++nraw, r8 = r8 || p - s0 < 8, r4 = r4 || p - s0 < 4;
        }
        ++r.slices, r.sl_raw += nraw > 0, r.sl_raw8 += r8, r.sl_raw4 += r4, ++r.per_slice[std::min(nraw, 32)];
        traw = traw || nraw > 0;
      }
      ++r.tiles, r.t_raw += traw;
    }
}

std::string set_str(const std::array<uint64_t, 4>& s) {
  std::string o;
  for (int v = 0; v < 256; ++v)
    if (s[v >> 6] >> (v & 63) & 1) o += (o.empty() ? "" : " ") + std::to_string(v);
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) die("usage: escales FILE [--list] [--threads T] [--json OUT] [--groups OUT]");
  load_toc(argv[1]);
  int threads = 1;
  const char* json = nullptr;
  const char* gjson = nullptr;
  bool list = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--list")) list = true;
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::max(1, std::min(4, std::atoi(argv[++i])));
    else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
    else if (!std::strcmp(argv[i], "--groups") && i + 1 < argc) gjson = argv[++i];
    else die("unknown argument ", argv[i]);
  }
  std::vector<const dsw::TocEntry*> ts;
  for (const auto& e : toc)
    if (is_expert_scale(e)) ts.push_back(&e);
  if (list) {
    for (const auto* e : ts)
      std::printf("%-44s %-8s %-7s [%llu, %llu, %llu] %llu B\n", e->name, dsw::dtype_name((dsw::DType)e->dtype),
                  dsw::layout_name((dsw::Layout)e->layout), (unsigned long long)e->shape[0],
                  (unsigned long long)e->shape[1], (unsigned long long)e->shape[2], (unsigned long long)e->bytes);
    return 0;
  }
  if (gjson) {  // the 1-bit code's groups: one tensor (35 MB) in memory a thread
    std::vector<Groups> gs(ts.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&] {
        std::vector<uint8_t> buf;
        for (size_t i; (i = next++) < ts.size();) groups(*ts[i], gs[i], buf);
      });
    for (auto& t : pool) t.join();
    FILE* f = std::fopen(gjson, "w");
    if (!f) die("cannot write ", gjson);
    std::map<std::string, Groups> by;  // "w1", "w3", "w2", "all"
    std::map<int, int> bases;
    for (const auto& g : gs) {
      std::fprintf(f, "{\"name\":\"%s\",\"base\":%d,\"P\":%d,\"groups\":%llu,\"in_pair\":%llu,\"in_120\":%llu,\"in_quad\":%llu,"
                   "\"slices\":%llu,\"sl_raw\":%llu,\"sl_raw8\":%llu,\"sl_raw4\":%llu,\"tiles\":%llu,\"t_raw\":%llu,\"per_slice\":[",
                   g.name.c_str(), g.base, g.P, (unsigned long long)g.groups, (unsigned long long)g.in_pair,
                   (unsigned long long)g.in_120, (unsigned long long)g.in_quad, (unsigned long long)g.slices,
                   (unsigned long long)g.sl_raw, (unsigned long long)g.sl_raw8, (unsigned long long)g.sl_raw4,
                   (unsigned long long)g.tiles, (unsigned long long)g.t_raw);
      for (int k = 0; k <= 32; ++k) std::fprintf(f, "%s%llu", k ? "," : "", (unsigned long long)g.per_slice[k]);
      std::fprintf(f, "]}\n");
      ++bases[g.base];
      const std::string kind = g.name.substr(g.name.size() - 8, 2);  // "w1" of "...w1.scale"
      for (const char* k : {kind.c_str(), "all"}) {
        Groups& a = by[k];
        a.P = g.P;
        a.groups += g.groups, a.in_pair += g.in_pair, a.in_120 += g.in_120, a.in_quad += g.in_quad;
        a.slices += g.slices, a.sl_raw += g.sl_raw, a.sl_raw8 += g.sl_raw8, a.sl_raw4 += g.sl_raw4;
        a.tiles += g.tiles, a.t_raw += g.t_raw;
        for (int j = 0; j <= 32; ++j) a.per_slice[j] += g.per_slice[j];
      }
    }
    std::fclose(f);
    std::printf("%zu scale tensors, %.2f GB read; the tensors' best pairs {b, b + 1}, b:", gs.size(), g_read / 1e9);
    for (const auto& [b, n] : bases) std::printf(" %d (%d)", b, n);
    std::printf("\n\n%-5s %10s %9s %9s %9s | %-26s %9s %9s %9s | %9s\n", "kind", "groups", "in pair", "{120,121}", "quad",
                "slices (up: 20 steps)", "a raw", "in 1st 8", "in 1st 4", "tiles raw");
    for (const char* k : {"w1", "w3", "w2", "all"}) {
      const Groups& a = by[k];
      auto pc = [](uint64_t x, uint64_t n) { return 100.0 * x / (n ? n : 1); };
      std::printf("%-5s %10llu %8.3f%% %8.3f%% %8.3f%% | %-26llu %8.2f%% %8.2f%% %8.2f%% | %8.2f%%\n", k,
                  (unsigned long long)a.groups, pc(a.in_pair, a.groups), pc(a.in_120, a.groups), pc(a.in_quad, a.groups),
                  (unsigned long long)a.slices, pc(a.sl_raw, a.slices), pc(a.sl_raw8, a.slices), pc(a.sl_raw4, a.slices),
                  pc(a.t_raw, a.tiles));
    }
    return 0;
  }
  std::vector<Counts> cs(ts.size());
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      std::vector<uint8_t> buf;
      for (size_t i; (i = next++) < ts.size();) count(*ts[i], cs[i], buf);
    });
  for (auto& t : pool) t.join();

  // per matrix kind and over all: experts by distinct count, per matrix and per (layer, expert)
  std::map<std::string, std::array<uint64_t, 257>> by_kind;  // "w1", "w3", "w2", "all"
  std::map<std::string, std::array<uint64_t, 4>> uni;       // "prefix|expert" -> union over w1, w3, w2
  std::map<std::string, int> uni_parts;
  uint64_t hist[256] = {}, tiles[257] = {};
  std::map<std::string, uint64_t> pairs;  // the 2-value sets
  for (const auto& c : cs) {
    const std::string kind = c.name.substr(c.name.size() - 8, 2);  // "w1" of "...w1.scale"
    const std::string pre = c.name.substr(0, c.name.size() - 8);
    auto& k = by_kind[kind];
    auto& a = by_kind["all"];
    for (int x = 0; x < c.E; ++x) {
      const int d = popc256(c.m[x]);
      ++k[d], ++a[d];
      if (d == 2) ++pairs[set_str(c.m[x])];
      auto& u = uni[pre + "|" + std::to_string(x)];
      for (int w = 0; w < 4; ++w) u[w] |= c.m[x][w];
      ++uni_parts[pre + "|" + std::to_string(x)];
    }
    for (int v = 0; v < 256; ++v) hist[v] += c.hist[v];
    for (int d = 0; d <= 256; ++d) tiles[d] += c.tile_hist[d];
  }
  std::array<uint64_t, 257> ue{};
  std::map<std::string, uint64_t> upairs;
  for (const auto& [key, s] : uni) {
    if (uni_parts[key] != 3) continue;
    const int d = popc256(s);
    ++ue[d];
    if (d <= 2) ++upairs[set_str(s)];
  }
  auto row = [](const char* label, const std::array<uint64_t, 257>& h) {
    uint64_t n = 0, le2 = 0, le4 = 0;
    int mx = 0;
    for (int d = 0; d <= 256; ++d) {
      n += h[d];
      if (d <= 2) le2 += h[d];
      if (d <= 4) le4 += h[d];
      if (h[d]) mx = d;
    }
    std::printf("%-22s %7llu   %6.2f %%  %6.2f %%   %3d  |", label, (unsigned long long)n, 100.0 * le2 / n, 100.0 * le4 / n, mx);
    for (int d = 1; d <= std::min(mx, 8); ++d) std::printf(" %d:%llu", d, (unsigned long long)h[d]);
    if (mx > 8) {
      uint64_t more = 0;
      for (int d = 9; d <= 256; ++d) more += h[d];
      std::printf(" >8:%llu", (unsigned long long)more);
    }
    std::printf("\n");
  };
  std::printf("%zu scale tensors, %.2f GB read\n\n", cs.size(), g_read / 1e9);
  std::printf("%-22s %7s   %8s  %8s   %3s  | by distinct count\n", "unit", "units", "<= 2", "<= 4", "max");
  for (const char* k : {"w1", "w3", "w2", "all"}) row((std::string("(layer, expert) ") + k).c_str(), by_kind[k]);
  row("(layer, expert) w1-3-2", ue);
  std::array<uint64_t, 257> th{};
  for (int d = 0; d <= 256; ++d) th[d] = tiles[d];
  row("16-row tile", th);
  std::printf("\nvalues over all scales:");
  uint64_t tot = 0;
  for (int v = 0; v < 256; ++v) tot += hist[v];
  for (int v = 0; v < 256; ++v)
    if (hist[v]) std::printf(" %d:%.4f%%", v, 100.0 * hist[v] / tot);
  std::printf("\n\nthe commonest value pairs of 2-value (layer, expert, matrix) units:");
  std::vector<std::pair<uint64_t, std::string>> pv;
  for (const auto& [s, n] : pairs) pv.push_back({n, s});
  std::sort(pv.rbegin(), pv.rend());
  for (size_t i = 0; i < pv.size() && i < 8; ++i) std::printf(" {%s}:%llu", pv[i].second.c_str(), (unsigned long long)pv[i].first);
  std::printf("\n");
  if (json) {
    FILE* f = std::fopen(json, "w");
    if (!f) die("cannot write ", json);
    for (const auto& c : cs) {
      std::fprintf(f, "{\"name\":\"%s\",\"E\":%d,\"N\":%d,\"KB\":%d,\"layout\":%u,\"sets\":[", c.name.c_str(), c.E, c.N, c.KB,
                   c.layout);
      for (int x = 0; x < c.E; ++x) std::fprintf(f, "%s\"%s\"", x ? "," : "", set_str(c.m[x]).c_str());
      std::fprintf(f, "],\"hist\":{");
      bool first = true;
      for (int v = 0; v < 256; ++v)
        if (c.hist[v]) std::fprintf(f, "%s\"%d\":%llu", first ? "" : ",", v, (unsigned long long)c.hist[v]), first = false;
      std::fprintf(f, "},\"tiles\":{");
      first = true;
      for (int d = 0; d <= 256; ++d)
        if (c.tile_hist[d]) std::fprintf(f, "%s\"%d\":%llu", first ? "" : ",", d, (unsigned long long)c.tile_hist[d]), first = false;
      std::fprintf(f, "}}\n");
    }
    std::fclose(f);
  }
  return 0;
}
