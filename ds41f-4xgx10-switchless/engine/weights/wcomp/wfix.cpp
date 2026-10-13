// wfix: what fixed-length codes would leave as exceptions, per weight class (CPU only, streams like wcomp).
//   fp8  (L1 tiles): per lane chunk (the 16 bytes a GEMV lane reads per 64-byte step of 8 rows, i.e. 16
//        consecutive k of one row): exponent max - min, and (min, max) jointly; per value: exponent
//        histogram (a per-tensor window)
//   fp4 scales (L3): per (tile, step) group (32 scales: 16 rows x 2 blocks, the bytes one warp-step reads):
//        max - min, and the group's values against the tensor's two most common
//   bf16 (L2 tiles): per 512-byte tile (8 rows x 32 values): exponent max - min
// Output: one JSON line per tensor with histograms of those ranges.
// Usage: wfix FILE [--expert-stride S] NAME...
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct Ent {
  std::string name;
  uint32_t dtype = 0, layout = 0;
  uint64_t shape[4] = {}, offset = 0, bytes = 0;
};

int g_fd = -1;
std::map<std::string, Ent> toc;

void die(const char* m) {
  std::fprintf(stderr, "wfix: %s\n", m);
  std::exit(1);
}

void rd(uint64_t off, uint64_t n, uint8_t* buf) {
  for (uint64_t got = 0; got < n;) {
    ssize_t r = pread(g_fd, buf + got, n - got, (off_t)(off + got));
    if (r <= 0) die("short read");
    got += (uint64_t)r;
  }
  posix_fadvise(g_fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
}

void load_toc(const char* path) {
  g_fd = open(path, O_RDONLY);
  if (g_fd < 0) die("open");
  uint8_t h[4096];
  rd(0, sizeof h, h);
  if (std::memcmp(h, "DSGB10W1", 8)) die("magic");
  uint32_t n;
  uint64_t toc_off;
  std::memcpy(&n, h + 24, 4);
  std::memcpy(&toc_off, h + 32, 8);
  std::vector<uint8_t> t((size_t)n * 512);
  rd(toc_off, t.size(), t.data());
  for (uint32_t i = 0; i < n; ++i) {
    const uint8_t* e = t.data() + (size_t)i * 512;
    Ent x;
    x.name = std::string(reinterpret_cast<const char*>(e), strnlen(reinterpret_cast<const char*>(e), 128));
    std::memcpy(&x.dtype, e + 352, 4);
    std::memcpy(x.shape, e + 360, 32);
    std::memcpy(&x.offset, e + 392, 8);
    std::memcpy(&x.bytes, e + 400, 8);
    std::memcpy(&x.layout, e + 428, 4);
    toc[x.name] = x;
  }
}

const Ent& ent(const std::string& name) {
  auto it = toc.find(name);
  if (it == toc.end()) {
    std::fprintf(stderr, "wfix: no tensor %s\n", name.c_str());
    std::exit(1);
  }
  return it->second;
}

void put_hist(const char* key, const std::vector<uint64_t>& h) {
  std::printf("\"%s\":[", key);
  for (size_t i = 0; i < h.size(); ++i) std::printf("%s%llu", i ? "," : "", (unsigned long long)h[i]);
  std::printf("]");
}

std::vector<uint8_t> buf(64u << 20);

// FP8 e4m3, L1 tiles: a lane chunk is 16 bytes at a multiple of 16 inside a 64-byte row segment
void fp8(const Ent& w) {
  if (w.layout != 1) die("fp8: not L1");
  std::vector<uint64_t> hexp(16), range(16), top(16);  // top: the chunk's max exponent
  std::vector<uint64_t> mm(256);                         // the chunk's (min, max) exponents, min * 16 + max
  for (uint64_t c0 = 0; c0 < w.bytes; c0 += buf.size()) {
    const uint64_t n = std::min<uint64_t>(buf.size(), w.bytes - c0);
    rd(w.offset + c0, n, buf.data());
    for (uint64_t i = 0; i < n; i += 16) {
      int lo = 15, hi = 0;
      for (int j = 0; j < 16; ++j) {
        const int e = (buf[i + j] >> 3) & 15;
        ++hexp[e];
        lo = std::min(lo, e), hi = std::max(hi, e);
      }
      ++range[hi - lo];
      ++top[hi];
      ++mm[lo * 16 + hi];
    }
  }
  std::printf("{\"name\":\"%s\",\"class\":\"fp8\",\"bytes\":%llu,", w.name.c_str(), (unsigned long long)w.bytes);
  put_hist("hexp", hexp);
  std::printf(",");
  put_hist("range", range);
  std::printf(",");
  put_hist("top", top);
  std::printf(",");
  put_hist("mm", mm);
  std::printf("}\n");
  std::fflush(stdout);
}

// MXFP4 expert scales, L3: per expert [N/16][K/64][8][4] bytes; a group is 32 consecutive bytes
void fp4s(const Ent& w, int stride) {
  const Ent& se = ent(w.name.substr(0, w.name.size() - 6) + "scale");
  const uint64_t E = se.shape[0], sper = se.bytes / E;
  std::vector<uint64_t> hs(256), range(256), sc(sper);
  std::vector<uint8_t> s(sper);
  // pass 1 over the sampled experts: the histogram; pass 2: groups against the two most common values
  std::vector<std::vector<uint8_t>> kept;
  for (uint64_t e = 0; e < E; e += stride) {
    rd(se.offset + e * sper, sper, s.data());
    for (uint8_t v : s) ++hs[v];
    kept.push_back(s);
  }
  int b0 = (int)(std::max_element(hs.begin(), hs.end()) - hs.begin());
  const int base = hs[b0 - 1] > hs[b0 + 1] ? b0 - 1 : b0;  // {base, base + 1}: the best pair
  uint64_t groups = 0, in_pair = 0, in_quad = 0;
  int qbase = base - 1;  // {base - 1 .. base + 2}
  for (const auto& v : kept)
    for (uint64_t g = 0; g + 32 <= v.size(); g += 32) {
      int lo = 255, hi = 0;
      bool p = true, q = true;
      for (int j = 0; j < 32; ++j) {
        const int x = v[g + j];
        lo = std::min(lo, x), hi = std::max(hi, x);
        p = p && (x == base || x == base + 1);
        q = q && (x >= qbase && x <= qbase + 3);
      }
      ++range[hi - lo];
      ++groups;
      in_pair += p;
      in_quad += q;
    }
  std::printf("{\"name\":\"%s\",\"class\":\"fp4s\",\"experts\":%zu,\"groups\":%llu,\"base\":%d,\"in_pair\":%llu,"
              "\"in_quad\":%llu,",
              w.name.c_str(), kept.size(), (unsigned long long)groups, base, (unsigned long long)in_pair,
              (unsigned long long)in_quad);
  put_hist("hs", hs);
  std::printf(",");
  put_hist("range", range);
  std::printf("}\n");
  std::fflush(stdout);
}

// bf16, L2 tiles (bytes [N][2K] tiled as L1): a 512-byte tile holds 8 rows x 32 values
void bf16(const Ent& w) {
  std::vector<uint64_t> range(256), hexp(256);
  for (uint64_t c0 = 0; c0 < w.bytes; c0 += buf.size()) {
    const uint64_t n = std::min<uint64_t>(buf.size(), w.bytes - c0);
    rd(w.offset + c0, n, buf.data());
    const uint16_t* x = reinterpret_cast<const uint16_t*>(buf.data());
    for (uint64_t i = 0; i < n / 2; i += 256) {
      int lo = 255, hi = 0;
      for (int j = 0; j < 256; ++j) {
        const int e = (x[i + j] >> 7) & 255;
        ++hexp[e];
        lo = std::min(lo, e), hi = std::max(hi, e);
      }
      ++range[hi - lo];
    }
  }
  std::printf("{\"name\":\"%s\",\"class\":\"bf16\",\"layout\":%u,\"bytes\":%llu,", w.name.c_str(), w.layout,
              (unsigned long long)w.bytes);
  put_hist("hexp", hexp);
  std::printf(",");
  put_hist("range", range);
  std::printf("}\n");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) die("usage: wfix FILE [--expert-stride S] NAME...");
  load_toc(argv[1]);
  int stride = 8;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--expert-stride") {
      stride = std::atoi(argv[++i]);
      continue;
    }
    const Ent& w = ent(a);
    switch (w.dtype) {
      case 3: fp8(w); break;
      case 5: fp4s(w, stride); break;
      case 2: bf16(w); break;
      default: die("unsupported dtype");
    }
    std::fprintf(stderr, "wfix: %s done\n", a.c_str());
  }
  return 0;
}
