#pragma once
#include "common/frontend_protocol.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace srv {
// Only the same-service frontend's private immutable spool is a protocol capability.
// Open descriptors pin inodes, so unlink on cancellation/connection loss cannot redirect reads.
inline constexpr char kVisionFileRoot[] = "/tmp/dsgb10-vision-features";
inline bool vision_component(const std::string& s) {
  if (s.empty() || s.size() > 128) return false;
  for (unsigned char c : s)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
  return true;
}
inline bool vision_file_path(const std::string& path, std::string& generation, std::string& name) {
  const std::string prefix = std::string(kVisionFileRoot) + "/";
  if (path.size() > fe::kMaxFeaturePathBytes || path.compare(0, prefix.size(), prefix)) return false;
  const auto slash = path.find('/', prefix.size());
  if (slash == std::string::npos || path.find('/', slash + 1) != std::string::npos) return false;
  generation = path.substr(prefix.size(), slash - prefix.size());
  name = path.substr(slash + 1);
  return vision_component(generation) && vision_component(name);
}

class VisionFile {
 public:
  using Data = std::shared_ptr<const VisionFile>;
  static Data open(const fe::ImageSpan& image, const std::string& path, std::string& error) {
    std::string gen, name;
    if (!image.rows || image.rows > 1024 || !vision_file_path(path, gen, name)) {
      error = "invalid private image feature file path or shape"; return {};
    }
    const size_t cap = descriptor_limit();
    size_t old = descriptors_.load();
    do {
      if (old >= cap) { error = "image feature file descriptor budget exceeded"; return {}; }
    } while (!descriptors_.compare_exchange_weak(old, old + 1));
    struct Guard {
      int base = -1, root = -1, gen = -1, file = -1;
      bool reserved = true;
      ~Guard() {
        for (int f : {file, gen, root, base}) if (f >= 0) ::close(f);
        if (reserved) --descriptors_;
      }
    } g;
    const int dirs = O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC;
    g.base = ::open("/tmp", dirs);
    if (g.base >= 0) g.root = ::openat(g.base, "dsgb10-vision-features", dirs);
    if (g.root < 0 || !private_directory(g.root)) {
      error = "image feature root must be same-service owned mode 0700"; return {};
    }
    g.gen = ::openat(g.root, gen.c_str(), dirs);
    if (g.gen < 0 || !private_directory(g.gen)) {
      error = "image feature generation must be same-service owned mode 0700"; return {};
    }
    // O_NONBLOCK ensures an invalid FIFO/device cannot block before fstat rejects it.
    g.file = ::openat(g.gen, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    struct stat st{};
    const size_t bytes = size_t(image.rows) * 5120 * 2;
    if (g.file < 0 || ::fstat(g.file, &st) || !valid_stat(st, bytes) || st.st_nlink != 1) {
      error = "image feature file must be private immutable regular BF16 data of exact size"; return {};
    }
    auto p = std::shared_ptr<VisionFile>(new VisionFile(g.file, image.rows, st));
    g.file = -1; g.reserved = false;
    return p;
  }
  ~VisionFile() { ::close(fd_); --descriptors_; }
  VisionFile(const VisionFile&) = delete;
  VisionFile& operator=(const VisionFile&) = delete;
  size_t bytes() const { return size_t(rows_) * 5120 * 2; }
  uint32_t rows() const { return rows_; }
  bool read(uint16_t* target, const std::function<bool()>& cancelled, std::string& error,
            uint64_t* bytes_read = nullptr) const {
    if (cancelled()) { error = "image feature read cancelled"; return false; }
    struct stat before{}, after{};
    if (::fstat(fd_, &before) || !unchanged(before)) {
      error = "image feature file changed after submit"; return false;
    }
    auto* dst = reinterpret_cast<uint8_t*>(target);
    for (size_t off = 0; off < bytes();) {
      if (cancelled()) { error = "image feature read cancelled"; return false; }
      const size_t n = std::min(size_t(1) << 20, bytes() - off);
      const ssize_t got = ::pread(fd_, dst + off, n, off);
      if (got < 0 && errno == EINTR) continue;
      if (got <= 0) { error = "image feature file read failed or truncated"; return false; }
      if (bytes_read) *bytes_read += uint64_t(got);
      off += size_t(got);
    }
    if (::fstat(fd_, &after) || !unchanged(after)) {
      error = "image feature file changed while reading"; return false;
    }
    for (size_t i = 0; i < bytes() / 2; ++i)
      if ((target[i] & 0x7f80) == 0x7f80) { error = "non-finite image feature file"; return false; }
    return true;
  }
  static size_t live_descriptors() { return descriptors_.load(); }
  static size_t descriptor_limit() {
    struct rlimit r{};
    if (::getrlimit(RLIMIT_NOFILE, &r) || r.rlim_cur <= 128) return 0;
    return std::min<size_t>(fe::kMaxImageSpans, r.rlim_cur - 128);
  }
 private:
  int fd_;
  uint32_t rows_;
  struct stat original_;
  inline static std::atomic<size_t> descriptors_{0};
  VisionFile(int fd, uint32_t rows, const struct stat& st) : fd_(fd), rows_(rows), original_(st) {}
  static bool private_directory(int fd) {
    struct stat st{};
    return !::fstat(fd, &st) && S_ISDIR(st.st_mode) && st.st_uid == ::geteuid() &&
           (st.st_mode & 07777) == 0700;
  }
  static bool valid_stat(const struct stat& st, size_t bytes) {
    return S_ISREG(st.st_mode) && st.st_uid == ::geteuid() && (st.st_mode & 07777) == 0400 &&
           st.st_size >= 0 && uint64_t(st.st_size) == bytes;
  }
  bool unchanged(const struct stat& st) const {
    // An unlink changes ctime/nlink but not the already-open immutable payload.
    return valid_stat(st, bytes()) && st.st_dev == original_.st_dev && st.st_ino == original_.st_ino &&
           st.st_mtim.tv_sec == original_.st_mtim.tv_sec && st.st_mtim.tv_nsec == original_.st_mtim.tv_nsec;
  }
};
}  // namespace srv
