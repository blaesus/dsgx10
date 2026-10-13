// Host-side helpers shared by the loader and the C++ checks (the DeepSeek engine's io.h): CRC32C, aligned buffers and
// O_DIRECT reads of arbitrary (unaligned) byte ranges.
#pragma once
#include <arm_acle.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common/check.h"
#include "weights/glw.h"

namespace glw {

// CRC32C (Castagnoli) with the ARMv8 CRC instructions. Chainable: crc32c(b, crc32c(a)) == crc32c(a+b).
__attribute__((target("+crc"))) inline uint32_t crc32c(const void* data, size_t n, uint32_t crc = 0) {
  auto* p = static_cast<const uint8_t*>(data);
  crc = ~crc;
  while (n && (reinterpret_cast<uintptr_t>(p) & 7)) {
    crc = __crc32cb(crc, *p++);
    n--;
  }
  while (n >= 32) {
    uint64_t v[4];
    std::memcpy(v, p, 32);
    crc = __crc32cd(crc, v[0]);
    crc = __crc32cd(crc, v[1]);
    crc = __crc32cd(crc, v[2]);
    crc = __crc32cd(crc, v[3]);
    p += 32;
    n -= 32;
  }
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    crc = __crc32cd(crc, v);
    p += 8;
    n -= 8;
  }
  while (n--) crc = __crc32cb(crc, *p++);
  return ~crc;
}

inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A kAlign-aligned host buffer (for O_DIRECT).
struct AlignedBuf {
  uint8_t* p = nullptr;
  size_t cap = 0;
  AlignedBuf() = default;
  explicit AlignedBuf(size_t n) { reserve(n); }
  AlignedBuf(const AlignedBuf&) = delete;
  AlignedBuf& operator=(const AlignedBuf&) = delete;
  ~AlignedBuf() { std::free(p); }
  void reserve(size_t n) {
    n = align_up(n);
    if (n <= cap) return;
    std::free(p);
    void* q = nullptr;
    if (posix_memalign(&q, kAlign, n)) die("posix_memalign(%zu) failed", n);
    p = static_cast<uint8_t*>(q);
    cap = n;
  }
};

inline int open_or_die(const std::string& path, int flags, mode_t mode = 0644) {
  int fd = ::open(path.c_str(), flags, mode);
  if (fd < 0) die("open %s: %s", path.c_str(), std::strerror(errno));
  return fd;
}

// pread/pwrite the whole range or die. With O_DIRECT, buf, off and n must be kAlign-aligned.
inline void pread_all(int fd, void* buf, size_t n, uint64_t off) {
  auto* p = static_cast<uint8_t*>(buf);
  while (n) {
    ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) die("pread(%zu at %llu): %s", n, (unsigned long long)off, r ? std::strerror(errno) : "EOF");
    p += r;
    off += r;
    n -= r;
  }
}
inline void pwrite_all(int fd, const void* buf, size_t n, uint64_t off) {
  auto* p = static_cast<const uint8_t*>(buf);
  while (n) {
    ssize_t r = ::pwrite(fd, p, n, static_cast<off_t>(off));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) die("pwrite(%zu at %llu): %s", n, (unsigned long long)off, std::strerror(errno));
    p += r;
    off += r;
    n -= r;
  }
}

// Reads [off, off+n) of an O_DIRECT fd into dst (any alignment) through `scratch`, in pieces of at
// most `chunk` bytes. The file must extend to at least align_up(off+n) or end exactly at off+n.
inline void direct_read(int fd, uint64_t file_size, uint64_t off, uint64_t n, uint8_t* dst, AlignedBuf& scratch,
                        uint64_t chunk = 64ull << 20) {
  while (n) {
    uint64_t a0 = off / kAlign * kAlign;
    uint64_t take = std::min<uint64_t>(n, chunk - (off - a0));
    uint64_t a1 = align_up(off + take);
    uint64_t len = a1 - a0;
    scratch.reserve(len);
    // the last block of a file may be short; O_DIRECT reads it fine as long as the request is aligned
    uint64_t want = std::min<uint64_t>(len, align_up(file_size) - a0);
    auto* p = scratch.p;
    uint64_t got = 0;
    while (got < want) {
      ssize_t r = ::pread(fd, p + got, want - got, static_cast<off_t>(a0 + got));
      if (r < 0 && errno == EINTR) continue;
      if (r < 0) die("pread O_DIRECT(%llu at %llu): %s", (unsigned long long)(want - got),
                     (unsigned long long)(a0 + got), std::strerror(errno));
      if (r == 0) break;  // EOF inside the final partial block
      got += r;
    }
    if (got < off - a0 + take) die("short read at %llu", (unsigned long long)off);
    std::memcpy(dst, p + (off - a0), take);
    dst += take;
    off += take;
    n -= take;
  }
}

inline uint64_t file_size_of(int fd) {
  off_t end = ::lseek(fd, 0, SEEK_END);
  if (end < 0) die("lseek: %s", std::strerror(errno));
  return static_cast<uint64_t>(end);
}

}  // namespace glw
