// Test helpers for engine/attn: .npy loading (as written by gen_attn.py), device buffers, comparisons.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/check.h"

namespace attn_test {

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
  template <class T> T* as() { return reinterpret_cast<T*>(data.data()); }
};

inline bool npy_exists(const std::string& path) { return std::ifstream(path).good(); }

inline Npy npy_load(const std::string& path) {
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
  auto send = h.find(')', spos);
  std::string sh = h.substr(spos, send - spos);
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
    // the zeroing runs on the legacy stream, which non-blocking streams do not wait for: without this
    // it can land after a kernel on one of them has written the buffer
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
  }
  Dev(const T* host, size_t count) : Dev(count) { CUDA_CHECK(cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice)); }
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
  std::vector<T> host() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

template <class T> Dev<T> to_dev(const Npy& a) { return Dev<T>(a.as<T>(), a.data.size() / sizeof(T)); }

inline float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Compare bf16 bit patterns: exact-match fraction, max ulp distance, max abs and relative error.
struct Cmp {
  int64_t n = 0, exact = 0, bad = 0;
  double max_abs = 0, max_rel = 0, sum_abs = 0, ref_abs = 0;
  int max_ulp = 0;
};

inline int bf_ulp(uint16_t a, uint16_t b) {
  auto key = [](uint16_t x) { return (x & 0x8000) ? -(int)(x & 0x7fff) : (int)(x & 0x7fff); };
  return std::abs(key(a) - key(b));
}

inline Cmp cmp_bf16(const uint16_t* got, const uint16_t* ref, int64_t n, double rel_tol = 1e-2) {
  Cmp c;
  c.n = n;
  for (int64_t i = 0; i < n; ++i) {
    float g = bf2f(got[i]), r = bf2f(ref[i]);
    if (got[i] == ref[i] || (g == 0.f && r == 0.f)) {
      ++c.exact;
      c.ref_abs += std::fabs(r);
      continue;
    }
    double d = std::fabs((double)g - r);
    c.max_abs = std::max(c.max_abs, d);
    c.sum_abs += d;
    c.ref_abs += std::fabs(r);
    double rel = d / std::max(std::fabs((double)r), 1e-6);
    c.max_rel = std::max(c.max_rel, rel);
    c.max_ulp = std::max(c.max_ulp, bf_ulp(got[i], ref[i]));
    if (rel > rel_tol && d > 1e-3) ++c.bad;
  }
  return c;
}

inline Cmp cmp_f32(const float* got, const float* ref, int64_t n, double rel_tol = 1e-5, double abs_tol = 0) {
  Cmp c;
  c.n = n;
  for (int64_t i = 0; i < n; ++i) {
    if (got[i] == ref[i]) {
      ++c.exact;
      continue;
    }
    double d = std::fabs((double)got[i] - ref[i]);
    c.max_abs = std::max(c.max_abs, d);
    double rel = d / std::max(std::fabs((double)ref[i]), 1e-30);
    c.max_rel = std::max(c.max_rel, rel);
    if (rel > rel_tol && d > abs_tol) ++c.bad;
  }
  return c;
}

inline int g_failures = 0;

inline void report(const std::string& name, const Cmp& c, bool must_be_exact = false, bool counts = true) {
  bool ok = c.bad == 0 && (!must_be_exact || c.exact == c.n);
  if (!ok && counts) ++g_failures;
  std::printf("%-44s n=%-9lld exact=%7.3f%%  max_ulp=%-3d max_abs=%-10.3g max_rel=%-10.3g mean_rel=%-9.3g %s\n",
              name.c_str(), (long long)c.n, 100.0 * c.exact / std::max<int64_t>(c.n, 1), c.max_ulp, c.max_abs,
              c.max_rel, c.ref_abs > 0 ? c.sum_abs / c.ref_abs : 0.0, !counts ? "info" : ok ? "ok" : "FAIL");
}

}  // namespace attn_test
