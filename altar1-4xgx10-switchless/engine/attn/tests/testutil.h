// Test helpers for engine/attn: .npy loading (as gen_ref.py writes them), device buffers, comparisons.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/check.h"

namespace tu {

struct Npy {
  std::string descr;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
  template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
  template <class T> std::vector<T> vec() const { return std::vector<T>(as<T>(), as<T>() + numel()); }
};

inline bool exists(const std::string& p) { return std::ifstream(p).good(); }

inline Npy load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open %s", path.c_str());
  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) die("%s: not npy", path.c_str());
  uint8_t ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hl = 0;
  if (ver[0] == 1) {
    uint16_t h16;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hl = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hl), 4);
  }
  std::string h(hl, ' ');
  f.read(h.data(), hl);
  Npy n;
  auto dpos = h.find("'descr': '");
  n.descr = h.substr(dpos + 10, h.find('\'', dpos + 10) - dpos - 10);
  auto spos = h.find("'shape': (") + 10;
  std::string sh = h.substr(spos, h.find(')', spos) - spos);
  size_t i = 0;
  while (i < sh.size()) {
    while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) ++i;
    if (i >= sh.size()) break;
    n.shape.push_back(std::stoll(sh.substr(i)));
    while (i < sh.size() && sh[i] != ',') ++i;
  }
  size_t esz = n.descr.back() - '0';
  if (n.descr == "|b1" || n.descr == "|u1") esz = 1;
  n.data.resize((size_t)n.numel() * esz);
  f.read(reinterpret_cast<char*>(n.data.data()), n.data.size());
  if (!f) die("%s: short read", path.c_str());
  return n;
}

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  Dev() = default;
  explicit Dev(size_t count) : n(count) {
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(T)));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  explicit Dev(const std::vector<T>& v) : Dev(v.size()) { up(v); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  Dev(Dev&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; }
  Dev& operator=(Dev&& o) noexcept {
    std::swap(p, o.p);
    std::swap(n, o.n);
    return *this;
  }
  ~Dev() {
    if (p) cudaFree(p);
  }
  void up(const std::vector<T>& v) { CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
  void up(const void* src, size_t count) { CUDA_CHECK(cudaMemcpy(p, src, count * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> down(size_t count = 0) const {
    std::vector<T> v(count ? count : n);
    CUDA_CHECK(cudaMemcpy(v.data(), p, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
  void zero() { CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T))); }
};

template <class T>
inline Dev<T> to_dev(const Npy& a) {
  Dev<T> d((size_t)a.data.size() / sizeof(T));
  d.up(a.as<T>(), d.n);
  return d;
}

inline float bf(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f2bf(float f) {  // round to nearest even
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

// a check's result line; returns pass
struct Tally {
  int pass = 0, fail = 0;
  bool operator()(bool ok, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::printf("%s ", ok ? "PASS" : "FAIL");
    std::vprintf(fmt, ap);
    std::printf("\n");
    std::fflush(stdout);
    va_end(ap);
    (ok ? pass : fail)++;
    return ok;
  }
};

struct BfCmp {
  size_t n = 0, diff = 0, over1ulp = 0;
  double max_abs = 0;
};
// bf16 a vs b (strided): mismatches, and how many differ by more than one bf16 ulp of the larger magnitude while
// also exceeding 1e-3 of the reference row's RMS (a sum that cancels to near zero can flip sign or lose bits
// under any other summation order; those are not counted)
inline BfCmp cmp_bf16(const uint16_t* a, const uint16_t* b, size_t rows, size_t cols, size_t lda, size_t ldb) {
  BfCmp c;
  for (size_t r = 0; r < rows; ++r) {
    double ss = 0;
    for (size_t i = 0; i < cols; ++i) ss += (double)bf(b[r * ldb + i]) * bf(b[r * ldb + i]);
    const double rms = std::sqrt(ss / cols);
    for (size_t i = 0; i < cols; ++i) {
      const uint16_t x = a[r * lda + i], y = b[r * ldb + i];
      ++c.n;
      if (x == y) continue;
      ++c.diff;
      const double d = std::fabs((double)bf(x) - bf(y)), m = std::max(std::fabs(bf(x)), std::fabs(bf(y)));
      if (d > m * (1.0 / 128) && d > 1e-3 * rms) ++c.over1ulp;
      c.max_abs = std::max(c.max_abs, d);
    }
  }
  return c;
}

// relative L2 error of a against the reference r (both as doubles)
template <class FA, class FR>
inline double rel_l2(size_t n, FA a, FR r) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = a(i) - r(i);
    num += d * d;
    den += r(i) * r(i);
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace tu
