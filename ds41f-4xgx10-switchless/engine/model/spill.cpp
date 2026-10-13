// The store's disk tier (spill.h).
#include "model/spill.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common/check.h"

namespace model::detail {

namespace {

constexpr size_t kReadBuf = 8 << 20;  // the largest block (an end snapshot with the drafter's) is ~6.1 MiB

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void mkdirs(const std::string& dir) {
  for (size_t i = 1; i <= dir.size(); ++i)
    if (i == dir.size() || dir[i] == '/') {
      const std::string d = dir.substr(0, i);
      if (mkdir(d.c_str(), 0700) != 0 && errno != EEXIST) die("spill: mkdir %s: %s", d.c_str(), std::strerror(errno));
    }
}

void io_all(bool wr, int fd, uint8_t* buf, size_t n, size_t off, const std::string& path) {
  while (n) {
    const ssize_t r = wr ? pwrite(fd, buf, n, (off_t)off) : pread(fd, buf, n, (off_t)off);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) die("spill: %s %zu bytes at %zu of %s: %s", wr ? "write" : "read", n, off, path.c_str(),
                    r < 0 ? std::strerror(errno) : "end of file");
    buf += r, off += (size_t)r, n -= (size_t)r;
  }
}

}  // namespace

SpillFile::SpillFile(const std::string& base, size_t bytes, size_t staging, bool reuse) {
  const size_t cut = base.rfind('/');
  const std::string dir = cut == std::string::npos ? "." : base.substr(0, cut);
  mkdirs(dir);
  struct stat ds;
  if (stat(dir.c_str(), &ds) == 0 && ds.st_uid == getuid() && (ds.st_mode & 077)) chmod(dir.c_str(), 0700);
  path_ = base + ".spill";
  size_ = bytes / kPoolGrain * kPoolGrain;
  if (reuse) {  // the sessions a planned stop kept: the file as it is (kept_check looked at it just now)
    fd_ = open(path_.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_DIRECT);
    if (fd_ < 0 && errno == EINVAL) direct_ = false, fd_ = open(path_.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    struct stat fs;
    if (fd_ < 0 || fstat(fd_, &fs) != 0 || !S_ISREG(fs.st_mode) || fs.st_uid != getuid() || (fs.st_mode & 0777) != 0600 ||
        (size_t)fs.st_size != size_) {
      unlink((base + ".index").c_str());  // the next start is a clean one
      die("spill: cannot reuse %s, which changed since it was checked", path_.c_str());
    }
  } else {
    unlink(path_.c_str());  // nothing persists from an earlier run
    fd_ = open(path_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_DIRECT, 0600);
    if (fd_ < 0 && errno == EINVAL) {  // a filesystem without O_DIRECT: the page cache, dropped after use
      direct_ = false;
      fd_ = open(path_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    }
    if (fd_ < 0) die("spill: open %s: %s", path_.c_str(), std::strerror(errno));
    const int e = posix_fallocate(fd_, 0, (off_t)size_);  // the space is reserved: no write finds the disk full
    if (e) die("spill: reserving %.1f GiB for %s: %s", size_ / 1073741824.0, path_.c_str(), std::strerror(e));
  }
  staging = std::max<size_t>(staging / kPoolGrain * kPoolGrain, 2 * kReadBuf);
  CUDA_CHECK(cudaHostAlloc(&stage_, staging, cudaHostAllocDefault));
  stage_pool_.size = staging;
  stage_pool_.free.emplace(0, staging);
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaHostAlloc(&rbuf_[i], kReadBuf, cudaHostAllocDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&rev_[i], cudaEventDisableTiming));
  }
  slots_ = size_ / kPoolGrain;
  CUDA_CHECK(cudaHostAlloc(&sums_, 2 * slots_ * sizeof(uint64_t), cudaHostAllocDefault));
  th_ = std::thread([this] { writer(); });
}

SpillFile::~SpillFile() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (th_.joinable()) th_.join();
  for (cudaEvent_t e : {pre_, start_})
    if (e) cudaEventDestroy(e);
  for (int i = 0; i < 2; ++i) {
    cudaEventSynchronize(rev_[i]);
    cudaEventDestroy(rev_[i]);
    cudaFreeHost(rbuf_[i]);
  }
  cudaFreeHost(stage_);
  cudaFreeHost(sums_);
  if (fd_ >= 0) close(fd_);
  if (!keep_) unlink(path_.c_str());
}

uint64_t SpillFile::write(const void* src, size_t bytes, size_t off, cudaStream_t st) {
  const size_t n = Pool::round(bytes);
  if (off + n > size_) die("spill: a write of %zu bytes at %zu past the file's %zu", n, off, size_);
  size_t at;
  for (;;) {
    {
      std::lock_guard<std::mutex> l(mu_);
      at = stage_pool_.take_off(n);
    }
    if (at != Pool::kNone) break;
    flush(st);  // what waits in staging goes to the writer, which then makes room
    std::unique_lock<std::mutex> l(mu_);
    cv_.wait(l, [&] { return q_.empty() || stage_pool_.largest() >= n; });
    if (q_.empty() && stage_pool_.largest() < n) die("spill: %zu bytes do not fit the staging buffer", n);
  }
  if (pending_.empty()) {  // a flush's first copy: timed from here
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventRecord(start_, st));
    batch_ = 0;
  }
  CUDA_CHECK(cudaMemcpyAsync(stage_ + at, src, n, cudaMemcpyDeviceToHost, st));
  pending_.push_back(Job{++seq_, at, off, n, nullptr, false});
  batch_ += n;
  return seq_;
}

void SpillFile::begin(cudaStream_t st) {
  if (pre_) return;
  CUDA_CHECK(cudaEventCreate(&pre_));
  CUDA_CHECK(cudaEventRecord(pre_, st));
}

void SpillFile::flush(cudaStream_t st) {
  if (pending_.empty()) return;
  cudaEvent_t ev;
  CUDA_CHECK(cudaEventCreate(&ev));
  CUDA_CHECK(cudaEventRecord(ev, st));
  for (auto& j : pending_) j.ev = ev;
  Job& l = pending_.back();
  l.last = true;
  l.pre = pre_, l.start = start_, l.batch = batch_;
  pre_ = start_ = nullptr;
  {
    std::lock_guard<std::mutex> l(mu_);
    q_.insert(q_.end(), pending_.begin(), pending_.end());
  }
  pending_.clear();
  cv_.notify_all();
}

void SpillFile::wait(uint64_t seq) {
  if (!pending_.empty() && seq >= pending_.front().seq) die("spill: waiting for write %llu before its flush", (unsigned long long)seq);
  std::unique_lock<std::mutex> l(mu_);
  cv_.wait(l, [&] { return done_ >= seq; });
}

void SpillFile::sync() {
  if (!pending_.empty()) die("spill: a sync with writes not yet flushed");
  wait(seq_);
  if (fdatasync(fd_) != 0) die("spill: fdatasync %s: %s", path_.c_str(), std::strerror(errno));
}

void SpillFile::read(void* dst, size_t bytes, size_t off, cudaStream_t st) {
  const size_t n = Pool::round(bytes);
  if (n > kReadBuf || off + n > size_) die("spill: a read of %zu bytes at %zu", n, off);
  const int b = rnext_;
  rnext_ ^= 1;
  CUDA_CHECK(cudaEventSynchronize(rev_[b]));  // its last copy has left the buffer
  io_all(false, fd_, rbuf_[b], n, off, path_);
  CUDA_CHECK(cudaMemcpyAsync(dst, rbuf_[b], n, cudaMemcpyHostToDevice, st));
  CUDA_CHECK(cudaEventRecord(rev_[b], st));
  if (!direct_) posix_fadvise(fd_, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
}

void SpillFile::writer() {
  cudaEvent_t waited = nullptr;
  double t_landed = 0;
  for (;;) {
    Job j;
    {
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait(l, [&] { return stop_ || !q_.empty(); });
      if (q_.empty()) return;  // stopping, and every flushed write is in the file
      j = q_.front();
    }
    if (j.ev != waited) {
      CUDA_CHECK(cudaEventSynchronize(j.ev));  // the copies to staging have landed
      waited = j.ev;
      t_landed = now_s();
    }
    io_all(true, fd_, stage_ + j.at, j.n, j.off, path_);
    if (!direct_) {
      fdatasync(fd_);
      posix_fadvise(fd_, (off_t)j.off, (off_t)j.n, POSIX_FADV_DONTNEED);
    }
    if (j.last) {
      float sum_ms = 0, copy_ms = 0;
      if (j.pre) CUDA_CHECK(cudaEventElapsedTime(&sum_ms, j.pre, j.start));
      CUDA_CHECK(cudaEventElapsedTime(&copy_ms, j.start, j.ev));
      const double w = now_s() - t_landed;
      std::printf("spill: %.1f MiB to staging in %.2f ms on the stream (%.1f GB/s; checksums before it %.2f ms), "
                  "then to the file in %.1f ms (%.2f GB/s)\n",
                  j.batch / 1048576.0, copy_ms, j.batch / std::max(1e-6, (double)copy_ms) / 1e6, sum_ms, 1e3 * w,
                  j.batch / std::max(1e-9, w) / 1e9);
      for (cudaEvent_t e : {j.pre, j.start, j.ev})
        if (e) CUDA_CHECK(cudaEventDestroy(e));
      waited = nullptr;
    }
    {
      std::lock_guard<std::mutex> l(mu_);
      q_.pop_front();
      stage_pool_.give_off(j.at, j.n);
      done_ = j.seq;
    }
    cv_.notify_all();
  }
}

}  // namespace model::detail
