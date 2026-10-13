// wcomp: code statistics of a .dsw rank file's weights, for a lossless-compression estimate (CPU only).
// Streams the named tensors in chunks (pread, then posix_fadvise DONTNEED behind), so the page cache keeps
// nothing. Writes one JSON line per tensor: its class, the bytes it read, and histograms:
//   fp8  (F8_E4M3, L1 or natural): byte h[256]; per block scale hs{scale: h[256]}; the previous byte in the same
//        64-byte row segment as context: pair (sparse, index prev * 256 + byte)
//   fp4  (E2M1X2, L3 experts): nibble hn[16]; byte (two neighbouring values) hb[256]; per scale hns{s: h[16]};
//        the scales: sh[256] and the previous scale byte of the same 4-byte group as context: spair (sparse)
//   bf16 (BF16): word h16 (sparse)
//   f32  (F32): exponent byte hexp[256]
// Usage: wcomp FILE [--expert-stride S] [--max-mb M] NAME...   (M: bf16 tensors are sampled in M MiB, evenly)
//        wcomp FILE --list   (the TOC: name, dtype, layout, bytes)
// Build: g++ -O2 -std=c++17 -o wcomp wcomp.cpp   (README.md: the study's run)
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
  uint32_t dtype = 0, ndim = 0, layout = 0, nstack = 0;
  uint64_t shape[4] = {}, offset = 0, bytes = 0;
};

int g_fd = -1;
uint64_t g_read = 0;

void die(const char* m) {
  std::fprintf(stderr, "wcomp: %s\n", m);
  std::exit(1);
}

// [off, off + n) of the file into buf, then its pages dropped
void rd(uint64_t off, uint64_t n, uint8_t* buf) {
  uint64_t got = 0;
  while (got < n) {
    ssize_t r = pread(g_fd, buf + got, n - got, (off_t)(off + got));
    if (r <= 0) die("short read");
    got += (uint64_t)r;
  }
  posix_fadvise(g_fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
  g_read += n;
}

std::vector<Ent> toc;
std::map<std::string, size_t> by_name;

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
    std::memcpy(&x.ndim, e + 356, 4);
    std::memcpy(x.shape, e + 360, 32);
    std::memcpy(&x.offset, e + 392, 8);
    std::memcpy(&x.bytes, e + 400, 8);
    std::memcpy(&x.nstack, e + 412, 4);
    std::memcpy(&x.layout, e + 428, 4);
    by_name[x.name] = toc.size();
    toc.push_back(x);
  }
}

const Ent& ent(const std::string& name) {
  auto it = by_name.find(name);
  if (it == by_name.end()) {
    std::fprintf(stderr, "wcomp: no tensor %s\n", name.c_str());
    std::exit(1);
  }
  return toc[it->second];
}

void put_hist(const char* key, const uint64_t* h, int n) {
  std::printf("\"%s\":[", key);
  for (int i = 0; i < n; ++i) std::printf("%s%llu", i ? "," : "", (unsigned long long)h[i]);
  std::printf("]");
}
void put_sparse(const char* key, const std::vector<uint64_t>& h) {
  std::printf("\"%s\":{", key);
  bool first = true;
  for (size_t i = 0; i < h.size(); ++i)
    if (h[i]) std::printf("%s\"%zu\":%llu", first ? "" : ",", i, (unsigned long long)h[i]), first = false;
  std::printf("}");
}
// rows of a [256][n] table that have counts, as {"row": [n counts]}
void put_rows(const char* key, const std::vector<uint64_t>& t, int n) {
  std::printf("\"%s\":{", key);
  bool first = true;
  for (int s = 0; s < 256; ++s) {
    uint64_t tot = 0;
    for (int v = 0; v < n; ++v) tot += t[(size_t)s * n + v];
    if (!tot) continue;
    std::printf("%s\"%d\":[", first ? "" : ",", s);
    first = false;
    for (int v = 0; v < n; ++v) std::printf("%s%llu", v ? "," : "", (unsigned long long)t[(size_t)s * n + v]);
    std::printf("]");
  }
  std::printf("}");
}

std::vector<uint8_t> buf(64u << 20);

// FP8 e4m3 weights [N][K] (L1: [N/8][K/64][8][64]), ue8m0 scales [ceil(N/32)][ceil(K/32)]
void fp8(const Ent& w) {
  const Ent& se = ent(w.name.substr(0, w.name.size() - 6) + "scale");
  const uint64_t K = w.shape[1], SC = (K + 31) / 32, KT = K / 64;
  std::vector<uint8_t> sc(se.bytes);
  rd(se.offset, se.bytes, sc.data());
  std::vector<uint64_t> h(256), hs(256 * 256), pair(256 * 256), sh(256);
  for (uint8_t s : sc) ++sh[s];
  for (uint64_t c0 = 0; c0 < w.bytes; c0 += buf.size()) {
    const uint64_t n = std::min<uint64_t>(buf.size(), w.bytes - c0);
    rd(w.offset + c0, n, buf.data());
    for (uint64_t i = 0; i < n; ++i) {
      const uint64_t b = c0 + i;
      uint64_t r, k;
      if (w.layout == 1) {
        const uint64_t t = b / 512, q = b % 512;
        r = t / KT * 8 + q / 64, k = t % KT * 64 + q % 64;
      } else {
        r = b / K, k = b % K;
      }
      const uint8_t v = buf[i];
      ++h[v];
      ++hs[(size_t)sc[r / 32 * SC + k / 32] * 256 + v];
      if (b % 64) ++pair[(size_t)buf[i - 1] * 256 + v];  // i >= 1: chunks are multiples of 64 bytes
    }
  }
  std::printf("{\"name\":\"%s\",\"class\":\"fp8\",\"layout\":%u,\"bytes\":%llu,\"read\":%llu,\"scale_bytes\":%llu,",
              w.name.c_str(), w.layout, (unsigned long long)w.bytes, (unsigned long long)w.bytes,
              (unsigned long long)se.bytes);
  put_hist("h", h.data(), 256);
  std::printf(",");
  put_hist("sh", sh.data(), 256);
  std::printf(",");
  put_rows("hs", hs, 256);
  std::printf(",");
  put_sparse("pair", pair);
  std::printf("}\n");
  std::fflush(stdout);
}

// MXFP4 experts, L3: per expert [N][K/2 bytes] values and [N][K/32] scales, tiled (docs/layout.md)
void fp4(const Ent& w, int stride) {
  const Ent& se = ent(w.name.substr(0, w.name.size() - 6) + "scale");
  const uint64_t E = w.shape[0], N = w.shape[1], KB = w.shape[2];  // KB: bytes per row, K/2 values
  const uint64_t per = N * KB, sper = N * (KB / 16);
  if (se.shape[0] != E || se.shape[1] != N || se.shape[2] != KB / 16) die("fp4 scale shape");
  std::vector<uint64_t> hn(16), hb(256), hns(256 * 16), sh(256), spair(256 * 256);
  std::vector<uint8_t> sc(sper);
  uint64_t read = 0, sread = 0, experts = 0;
  for (uint64_t e = 0; e < E; e += stride) {
    ++experts;
    rd(se.offset + e * sper, sper, sc.data());
    sread += sper;
    for (uint64_t i = 0; i < sper; ++i) {
      ++sh[sc[i]];
      if (i % 4) ++spair[(size_t)sc[i - 1] * 256 + sc[i]];  // j - 1 -> j within a 4-byte group: (h, l)
    }
    for (uint64_t c0 = 0; c0 < per; c0 += buf.size()) {
      const uint64_t n = std::min<uint64_t>(buf.size(), per - c0);
      rd(w.offset + e * per + c0, n, buf.data());
      read += n;
      for (uint64_t i = 0; i < n; ++i) {
        const uint64_t o = c0 + i, T = o / 512, q = o % 512;
        const uint64_t g = q / 64, j = q % 16 / 4;
        const uint8_t s = sc[T * 32 + g * 4 + j], v = buf[i];
        ++hb[v];
        ++hn[v & 15];
        ++hn[v >> 4];
        ++hns[(size_t)s * 16 + (v & 15)];
        ++hns[(size_t)s * 16 + (v >> 4)];
      }
    }
  }
  std::printf("{\"name\":\"%s\",\"class\":\"fp4\",\"layout\":%u,\"bytes\":%llu,\"read\":%llu,\"scale_bytes\":%llu,"
              "\"scale_read\":%llu,\"experts\":%llu,",
              w.name.c_str(), w.layout, (unsigned long long)w.bytes, (unsigned long long)read,
              (unsigned long long)se.bytes, (unsigned long long)sread, (unsigned long long)experts);
  put_hist("hn", hn.data(), 16);
  std::printf(",");
  put_hist("hb", hb.data(), 256);
  std::printf(",");
  put_hist("sh", sh.data(), 256);
  std::printf(",");
  put_rows("hns", hns, 16);
  std::printf(",");
  put_sparse("spair", spair);
  std::printf("}\n");
  std::fflush(stdout);
}

void bf16(const Ent& w, uint64_t max_bytes) {
  std::vector<uint64_t> h(65536);
  uint64_t read = 0;
  const uint64_t chunk = 4u << 20;
  const uint64_t nchunks = (w.bytes + chunk - 1) / chunk;
  const uint64_t want = max_bytes ? std::max<uint64_t>(1, max_bytes / chunk) : nchunks;
  for (uint64_t c = 0; c < nchunks; ++c) {
    if (want < nchunks && c * want / nchunks == (c + 1) * want / nchunks) continue;  // evenly spaced
    const uint64_t c0 = c * chunk, n = std::min<uint64_t>(chunk, w.bytes - c0);
    rd(w.offset + c0, n, buf.data());
    read += n;
    const uint16_t* x = reinterpret_cast<const uint16_t*>(buf.data());
    for (uint64_t i = 0; i < n / 2; ++i) ++h[x[i]];
  }
  std::printf("{\"name\":\"%s\",\"class\":\"bf16\",\"layout\":%u,\"bytes\":%llu,\"read\":%llu,", w.name.c_str(),
              w.layout, (unsigned long long)w.bytes, (unsigned long long)read);
  put_sparse("h16", h);
  std::printf("}\n");
  std::fflush(stdout);
}

void f32(const Ent& w) {
  std::vector<uint64_t> h(256);
  std::vector<uint8_t> b(w.bytes);
  rd(w.offset, w.bytes, b.data());
  const uint32_t* x = reinterpret_cast<const uint32_t*>(b.data());
  for (uint64_t i = 0; i < w.bytes / 4; ++i) ++h[(x[i] >> 23) & 255];
  std::printf("{\"name\":\"%s\",\"class\":\"f32\",\"layout\":%u,\"bytes\":%llu,\"read\":%llu,", w.name.c_str(),
              w.layout, (unsigned long long)w.bytes, (unsigned long long)w.bytes);
  put_hist("hexp", h.data(), 256);
  std::printf("}\n");
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) die("usage: wcomp FILE [--expert-stride S] [--max-mb M] NAME...");
  load_toc(argv[1]);
  int stride = 8;
  uint64_t max_bytes = 0;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--expert-stride") {
      stride = std::atoi(argv[++i]);
      continue;
    }
    if (a == "--max-mb") {
      max_bytes = (uint64_t)std::atoll(argv[++i]) << 20;
      continue;
    }
    if (a == "--list") {  // the TOC: name dtype layout bytes
      for (const Ent& e : toc)
        std::printf("%s\t%u\t%u\t%llu\n", e.name.c_str(), e.dtype, e.layout, (unsigned long long)e.bytes);
      continue;
    }
    const Ent& w = ent(a);
    switch (w.dtype) {
      case 3: fp8(w); break;
      case 5: fp4(w, stride); break;
      case 2: bf16(w, max_bytes); break;
      case 1: f32(w); break;
      default: die("unsupported dtype");
    }
    std::fprintf(stderr, "wcomp: %s done, %.2f GB read so far\n", a.c_str(), g_read / 1e9);
  }
  return 0;
}
