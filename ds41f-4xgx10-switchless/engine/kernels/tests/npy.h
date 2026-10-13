// Minimal .npy reader for the kernel tests (little-endian, C order).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"

struct Npy {
  std::string descr;
  std::vector<size_t> shape;
  std::vector<uint8_t> data;
  size_t count() const {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    return n;
  }
  template <class T>
  const T* as() const { return reinterpret_cast<const T*>(data.data()); }
};

inline Npy npy_load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) die("cannot open %s", path.c_str());
  char magic[6];
  uint8_t ver[2];
  if (std::fread(magic, 1, 6, f) != 6 || std::memcmp(magic, "\x93NUMPY", 6) || std::fread(ver, 1, 2, f) != 2)
    die("%s: not npy", path.c_str());
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h;
    if (std::fread(&h, 2, 1, f) != 1) die("%s: short", path.c_str());
    hlen = h;
  } else if (std::fread(&hlen, 4, 1, f) != 1) {
    die("%s: short", path.c_str());
  }
  std::string hdr(hlen, '\0');
  if (std::fread(hdr.data(), 1, hlen, f) != hlen) die("%s: short header", path.c_str());
  Npy n;
  auto d = hdr.find("'descr':");
  auto q1 = hdr.find('\'', d + 8), q2 = hdr.find('\'', q1 + 1);
  n.descr = hdr.substr(q1 + 1, q2 - q1 - 1);
  if (hdr.find("'fortran_order': True") != std::string::npos) die("%s: fortran order", path.c_str());
  auto s = hdr.find("'shape':");
  auto p1 = hdr.find('(', s), p2 = hdr.find(')', p1);
  std::string sh = hdr.substr(p1 + 1, p2 - p1 - 1);
  size_t pos = 0;
  while (pos < sh.size()) {
    while (pos < sh.size() && (sh[pos] == ' ' || sh[pos] == ',')) ++pos;
    if (pos >= sh.size()) break;
    n.shape.push_back(std::stoull(sh.substr(pos)));
    while (pos < sh.size() && sh[pos] != ',') ++pos;
  }
  size_t elem = std::stoul(n.descr.substr(2));
  n.data.resize(n.count() * elem);
  if (std::fread(n.data.data(), 1, n.data.size(), f) != n.data.size()) die("%s: short data", path.c_str());
  std::fclose(f);
  return n;
}
