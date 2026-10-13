// Read-only .npy files, memory-mapped (the golden records: '<f4', '<i4', '<i8', C order).
#pragma once
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"

struct Npy {
  std::string descr;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t bytes = 0;  // of the data
  void* map = nullptr;
  size_t map_bytes = 0;

  Npy() = default;
  explicit Npy(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) die("npy: cannot open %s", path.c_str());
    struct stat sb{};
    fstat(fd, &sb);
    map_bytes = (size_t)sb.st_size;
    map = mmap(nullptr, map_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) die("npy: mmap %s", path.c_str());
    const uint8_t* p = static_cast<const uint8_t*>(map);
    if (map_bytes < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0) die("npy: %s is not an npy file", path.c_str());
    const int major = p[6];
    size_t hlen, hoff;
    if (major == 1) {
      hlen = p[8] | (p[9] << 8);
      hoff = 10;
    } else {
      hlen = p[8] | (p[9] << 8) | ((size_t)p[10] << 16) | ((size_t)p[11] << 24);
      hoff = 12;
    }
    const std::string h(reinterpret_cast<const char*>(p + hoff), hlen);
    auto field = [&](const char* key) {
      const size_t k = h.find(key);
      if (k == std::string::npos) die("npy: %s: no %s", path.c_str(), key);
      return k + std::strlen(key);
    };
    size_t d = h.find('\'', field("'descr':")) + 1;
    descr = h.substr(d, h.find('\'', d) - d);
    if (h.find("'fortran_order': True") != std::string::npos) die("npy: %s is Fortran order", path.c_str());
    size_t s = h.find('(', field("'shape':")) + 1;
    const size_t e = h.find(')', s);
    while (s < e) {
      while (s < e && (h[s] == ' ' || h[s] == ',')) ++s;
      if (s >= e) break;
      shape.push_back(std::stoll(h.substr(s)));
      while (s < e && h[s] != ',') ++s;
    }
    data = p + hoff + hlen;
    bytes = map_bytes - hoff - hlen;
    if (bytes < count() * elem_bytes()) die("npy: %s is short", path.c_str());
  }
  Npy(const Npy&) = delete;
  Npy& operator=(const Npy&) = delete;
  Npy(Npy&& o) noexcept { *this = std::move(o); }
  Npy& operator=(Npy&& o) noexcept {
    std::swap(descr, o.descr);
    std::swap(shape, o.shape);
    std::swap(data, o.data);
    std::swap(bytes, o.bytes);
    std::swap(map, o.map);
    std::swap(map_bytes, o.map_bytes);
    return *this;
  }
  ~Npy() {
    if (map) munmap(map, map_bytes);
  }
  size_t count() const {
    size_t n = 1;
    for (int64_t s : shape) n *= (size_t)s;
    return n;
  }
  size_t elem_bytes() const { return descr == "<i8" ? 8 : descr == "|u1" || descr == "|b1" ? 1 : 4; }
  size_t row_elems() const {  // elements per index of dim 0
    size_t n = 1;
    for (size_t i = 1; i < shape.size(); ++i) n *= (size_t)shape[i];
    return n;
  }
  const float* f32(size_t row = 0) const {
    if (descr != "<f4") die("npy: dtype %s, not <f4", descr.c_str());
    return reinterpret_cast<const float*>(data) + row * row_elems();
  }
  const int32_t* i32(size_t row = 0) const {
    if (descr != "<i4") die("npy: dtype %s, not <i4", descr.c_str());
    return reinterpret_cast<const int32_t*>(data) + row * row_elems();
  }
  int64_t ival(size_t i) const {  // element i of an integer array, any width
    if (descr == "<i8") return reinterpret_cast<const int64_t*>(data)[i];
    if (descr == "<i4") return reinterpret_cast<const int32_t*>(data)[i];
    die("npy: dtype %s is not an integer", descr.c_str());
  }
  const int64_t* i64(size_t row = 0) const {
    if (descr != "<i8") die("npy: dtype %s, not <i8", descr.c_str());
    return reinterpret_cast<const int64_t*>(data) + row * row_elems();
  }
};
