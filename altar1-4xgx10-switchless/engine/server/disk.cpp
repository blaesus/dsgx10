// The disk tier of one rank (disk.h).
#include "server/disk.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <ctime>

#include <cuda_runtime.h>

#include "common/check.h"
#include "server/keepfile.h"

namespace srv {
namespace {

constexpr char kMagic[8] = {'G', 'L', 'M', 'K', 'E', 'P', 'T', '1'};
struct IndexHeader {
  char magic[8];
  uint64_t gen, run_id, page_bytes;
  int64_t written;
  int32_t pages, pad;
  uint64_t body_hash;  // fnv of the hashes that follow
};

bool io_all(bool wr, int fd, void* buf, size_t n, off_t off) {
  auto* b = static_cast<char*>(buf);
  while (n > 0) {
    const ssize_t r = wr ? pwrite(fd, b, n, off) : pread(fd, b, n, off);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    b += r, n -= (size_t)r, off += r;
  }
  return true;
}

}  // namespace

uint64_t hash64(const void* p, size_t n) {
  const auto* w = static_cast<const uint64_t*>(p);
  uint64_t h[4] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull, 0x27D4EB2F165667C5ull};
  const size_t m = n / 32;
  for (size_t i = 0; i < m; ++i)
    for (int k = 0; k < 4; ++k) {
      uint64_t x = h[k] ^ w[4 * i + k];
      x *= 0xBF58476D1CE4E5B9ull;
      h[k] = x ^ (x >> 31);
    }
  uint64_t r = n;
  for (int k = 0; k < 4; ++k) r = (r ^ h[k]) * 0x94D049BB133111EBull, r ^= r >> 29;
  return r;
}

uint64_t Disk::kept_check(const std::string& base, uint64_t run_id, size_t page_bytes, double max_age_s, int* pages,
                          std::string* why) {
  std::vector<uint8_t> f;
  const std::string idx = base + ".index";
  *pages = 0;
  if (!keepfile::read_private(idx, 64ull << 20, &f, why)) return 0;
  IndexHeader h{};
  if (f.size() < sizeof h) {
    *why = idx + " is damaged";
    return 0;
  }
  std::memcpy(&h, f.data(), sizeof h);
  const size_t body = f.size() - sizeof h;
  struct stat s{};
  if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.pages < 0 || body != (size_t)h.pages * 8 ||
      keepfile::fnv(f.data() + sizeof h, body) != h.body_hash) {
    *why = idx + " is damaged";
  } else if (h.run_id != run_id || h.page_bytes != page_bytes) {
    *why = "another build, weights or options";
  } else if ((double)std::time(nullptr) - (double)h.written > max_age_s) {
    *why = "older than the kept sessions' age";
  } else if (stat((base + ".spill").c_str(), &s) != 0 || (uint64_t)s.st_size != (uint64_t)h.pages * page_bytes) {
    *why = base + ".spill is missing or of another size";
  } else {
    *pages = h.pages;
    why->clear();
    return h.gen ? h.gen : 0;
  }
  return 0;
}

Disk::Disk(const std::string& base, int pages, size_t page_bytes, bool reuse, uint64_t run_id)
    : base_(base), path_(base + ".spill"), pages_(pages), page_bytes_(page_bytes), run_id_(run_id), hash_(pages, 0) {
  std::string why;
  const std::string dir = keepfile::dir_of(path_);
  mkdir(dir.c_str(), 0700);
  if (!keepfile::private_dir(dir, &why)) die("disk tier: %s", why.c_str());
  if (reuse) {  // the hashes the index holds
    std::vector<uint8_t> f;
    if (!keepfile::read_private(base + ".index", 64ull << 20, &f, &why) || f.size() != sizeof(IndexHeader) + 8ull * pages)
      die("disk tier: %s.index went away", base.c_str());
    std::memcpy(hash_.data(), f.data() + sizeof(IndexHeader), 8ull * pages);
  }
  unlink((base + ".index").c_str());
  unlink((base + ".index.new").c_str());
  if (!reuse) unlink(path_.c_str());
  const int flags = O_RDWR | O_CLOEXEC | O_NOFOLLOW | (reuse ? 0 : O_CREAT | O_EXCL);
  direct_ = page_bytes % 4096 == 0;
  fd_ = open(path_.c_str(), flags | (direct_ ? O_DIRECT : 0), 0600);
  if (fd_ < 0 && direct_) direct_ = false, fd_ = open(path_.c_str(), flags, 0600);
  if (fd_ < 0) die("disk tier: %s: %s", path_.c_str(), std::strerror(errno));
  if (!keepfile::private_file(fd_, path_, &why)) die("disk tier: %s", why.c_str());
  if (!reuse) {
    const int rc = posix_fallocate(fd_, 0, (off_t)pages * (off_t)page_bytes);
    if (rc != 0) die("disk tier: %s: cannot reserve %.1f GiB: %s", path_.c_str(), pages * (double)page_bytes / (1 << 30),
                     std::strerror(rc));
  }
  CUDA_CHECK(cudaHostAlloc(&stage_, page_bytes, cudaHostAllocDefault));
}

Disk::~Disk() {
  if (stage_) cudaFreeHost(stage_);
  if (fd_ >= 0) close(fd_);
  // the file stays only when an index names it (keep)
  struct stat s{};
  if (stat((base_ + ".index").c_str(), &s) != 0) unlink(path_.c_str());
}

void Disk::write_page(int p) {
  if (p < 0 || p >= pages_) die("disk tier: page %d of %d", p, pages_);
  hash_[p] = hash64(stage_, page_bytes_);
  if (!io_all(true, fd_, stage_, page_bytes_, (off_t)p * (off_t)page_bytes_))
    die("disk tier: writing page %d of %s: %s", p, path_.c_str(), std::strerror(errno));
}

void Disk::read_page(int p) {
  if (p < 0 || p >= pages_) die("disk tier: page %d of %d", p, pages_);
  if (!io_all(false, fd_, stage_, page_bytes_, (off_t)p * (off_t)page_bytes_))
    die("disk tier: reading page %d of %s: %s", p, path_.c_str(), std::strerror(errno));
  const uint64_t h = hash64(stage_, page_bytes_);
  if (h != hash_[p])
    die("disk tier: page %d of %s came back different from what was written (%016llx, not %016llx)", p, path_.c_str(),
        (unsigned long long)h, (unsigned long long)hash_[p]);
}

void Disk::keep(uint64_t gen) {
  if (fdatasync(fd_) != 0) die("disk tier: syncing %s: %s", path_.c_str(), std::strerror(errno));
  std::vector<uint8_t> f(sizeof(IndexHeader) + 8ull * pages_);
  IndexHeader h{};
  std::memcpy(h.magic, kMagic, sizeof kMagic);
  h.gen = gen, h.run_id = run_id_, h.page_bytes = page_bytes_, h.written = (int64_t)std::time(nullptr);
  h.pages = pages_;
  std::memcpy(f.data() + sizeof h, hash_.data(), 8ull * pages_);
  h.body_hash = keepfile::fnv(f.data() + sizeof h, 8ull * pages_);
  std::memcpy(f.data(), &h, sizeof h);
  std::string why;
  if (!keepfile::write_atomic(base_ + ".index", f.data(), f.size(), &why)) std::printf("disk tier: index not written: %s\n", why.c_str());
}

}  // namespace srv
