// Files that outlive the engine and hold what conversations left behind (kept sessions, docs/engine.md):
// mode 0600 in a directory of mode 0700, both ours; written under a temporary name, synced, then
// renamed into place (and the directory synced), so that a reader finds the whole file or none.
#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace keepfile {

inline uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

inline std::string dir_of(const std::string& path) {
  const size_t cut = path.rfind('/');
  return cut == std::string::npos ? "." : cut == 0 ? "/" : path.substr(0, cut);
}

// dir is ours and nobody else's to write or read (made 0700 if it was ours but open to others)
inline bool private_dir(const std::string& dir, std::string* why) {
  struct stat s;
  if (lstat(dir.c_str(), &s) != 0 || !S_ISDIR(s.st_mode) || s.st_uid != getuid()) {
    *why = dir + " is not a directory of ours";
    return false;
  }
  if ((s.st_mode & 077) && chmod(dir.c_str(), 0700) != 0) {
    *why = dir + " is open to others and cannot be made 0700";
    return false;
  }
  return true;
}

// a regular file of ours, mode 0600, not a link (fd from open(..., O_NOFOLLOW))
inline bool private_file(int fd, const std::string& path, std::string* why) {
  struct stat s;
  if (fstat(fd, &s) != 0 || !S_ISREG(s.st_mode) || s.st_uid != getuid() || (s.st_mode & 0777) != 0600) {
    *why = path + " is not a file of ours of mode 0600";
    return false;
  }
  return true;
}

inline bool write_atomic(const std::string& path, const void* data, size_t n, std::string* why) {
  const std::string dir = dir_of(path), tmp = path + ".new";
  if (!private_dir(dir, why)) return false;
  unlink(tmp.c_str());
  const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) {
    *why = tmp + ": " + std::strerror(errno);
    return false;
  }
  const auto* b = static_cast<const uint8_t*>(data);
  size_t done = 0;
  while (done < n) {
    const ssize_t r = write(fd, b + done, n - done);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) break;
    done += (size_t)r;
  }
  const bool ok = done == n && fsync(fd) == 0;
  *why = ok ? "" : tmp + ": " + std::strerror(errno);
  close(fd);
  if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
    if (ok) *why = path + ": " + std::strerror(errno);
    unlink(tmp.c_str());
    return false;
  }
  const int dfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dfd >= 0) fsync(dfd), close(dfd);
  return true;
}

// The whole of a private file (at most max bytes); false with why ("" when there is none)
inline bool read_private(const std::string& path, size_t max, std::vector<uint8_t>* out, std::string* why) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    *why = errno == ENOENT ? "" : path + ": " + std::strerror(errno);
    return false;
  }
  struct stat s;
  bool ok = private_file(fd, path, why) && fstat(fd, &s) == 0;
  if (ok && (size_t)s.st_size > max) *why = path + " is too large", ok = false;
  if (ok) {
    out->resize((size_t)s.st_size);
    size_t done = 0;
    while (done < out->size()) {
      const ssize_t r = read(fd, out->data() + done, out->size() - done);
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) break;
      done += (size_t)r;
    }
    if (done != out->size()) *why = path + ": a short read", ok = false;
  }
  close(fd);
  return ok;
}

// A little-endian record reader over a file's body: every get fails once the bytes run out.
struct Reader {
  const uint8_t* p;
  size_t n, at = 0;
  bool ok = true;
  template <class T>
  T get() {
    T v{};
    if (!ok || n - at < sizeof v) {
      ok = false;
      return v;
    }
    std::memcpy(&v, p + at, sizeof v);
    at += sizeof v;
    return v;
  }
};

template <class T>
void put(std::vector<uint8_t>& b, const T& v) {
  const auto* c = reinterpret_cast<const uint8_t*>(&v);
  b.insert(b.end(), c, c + sizeof v);
}

}  // namespace keepfile
