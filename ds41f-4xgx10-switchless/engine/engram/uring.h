// A minimal io_uring on raw syscalls (the boxes have the kernel UAPI header but no liburing).
// One thread owns a ring. SQEs are used in ring order, so the SQ index array is the identity.
#pragma once
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "common/check.h"

namespace engram {

class Uring {
 public:
  explicit Uring(unsigned entries, unsigned flags = 0) {
    io_uring_params p;
    std::memset(&p, 0, sizeof(p));
    p.flags = flags;
    fd_ = int(syscall(__NR_io_uring_setup, entries, &p));
    if (fd_ < 0) die("io_uring_setup(%u): %s", entries, strerror(errno));
    sq_sz_ = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    cq_sz_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
    const bool single = p.features & IORING_FEAT_SINGLE_MMAP;
    if (single) sq_sz_ = cq_sz_ = sq_sz_ > cq_sz_ ? sq_sz_ : cq_sz_;
    sq_ptr_ = map(sq_sz_, IORING_OFF_SQ_RING);
    cq_ptr_ = single ? sq_ptr_ : map(cq_sz_, IORING_OFF_CQ_RING);
    sqes_sz_ = p.sq_entries * sizeof(io_uring_sqe);
    sqes_ = static_cast<io_uring_sqe*>(map(sqes_sz_, IORING_OFF_SQES));
    auto* sq = static_cast<uint8_t*>(sq_ptr_);
    auto* cq = static_cast<uint8_t*>(cq_ptr_);
    sq_head_ = reinterpret_cast<unsigned*>(sq + p.sq_off.head);
    sq_tail_ = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
    sq_mask_ = *reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask);
    sq_entries_ = p.sq_entries;
    auto* array = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
    for (unsigned i = 0; i < sq_entries_; i++) array[i] = i;
    cq_head_ = reinterpret_cast<unsigned*>(cq + p.cq_off.head);
    cq_tail_ = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
    cq_mask_ = *reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask);
    cq_entries_ = p.cq_entries;
    cqes_ = reinterpret_cast<io_uring_cqe*>(cq + p.cq_off.cqes);
    local_tail_ = published_ = *sq_tail_;
  }
  ~Uring() {
    munmap(sqes_, sqes_sz_);
    if (cq_ptr_ != sq_ptr_) munmap(cq_ptr_, cq_sz_);
    munmap(sq_ptr_, sq_sz_);
    close(fd_);
  }
  Uring(const Uring&) = delete;
  Uring& operator=(const Uring&) = delete;

  unsigned sq_entries() const { return sq_entries_; }
  unsigned cq_entries() const { return cq_entries_; }

  void register_files(const int* fds, unsigned n) {
    if (syscall(__NR_io_uring_register, fd_, IORING_REGISTER_FILES, fds, n) < 0)
      die("io_uring register files: %s", strerror(errno));
  }
  void register_buffers(const iovec* iov, unsigned n) {
    if (syscall(__NR_io_uring_register, fd_, IORING_REGISTER_BUFFERS, iov, n) < 0)
      die("io_uring register buffers: %s", strerror(errno));
  }

  // The next free SQE, zeroed, or nullptr when the ring is full.
  io_uring_sqe* sqe() {
    if (local_tail_ - __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE) >= sq_entries_) return nullptr;
    io_uring_sqe* s = &sqes_[local_tail_ & sq_mask_];
    local_tail_++;
    std::memset(s, 0, sizeof(*s));
    return s;
  }
  // A fixed-buffer, fixed-file read.
  void prep_read_fixed(io_uring_sqe* s, int file_index, void* buf, unsigned len, uint64_t off, unsigned buf_index,
                       uint64_t user_data) {
    s->opcode = IORING_OP_READ_FIXED;
    s->flags = IOSQE_FIXED_FILE;
    s->fd = file_index;
    s->addr = reinterpret_cast<uint64_t>(buf);
    s->len = len;
    s->off = off;
    s->buf_index = uint16_t(buf_index);
    s->user_data = user_data;
  }

  // Publishes the queued SQEs and enters the kernel to submit them, optionally waiting for
  // `wait_nr` completions. Returns the number submitted.
  int submit(unsigned wait_nr = 0) {
    const unsigned n = local_tail_ - published_;
    if (n) {
      __atomic_store_n(sq_tail_, local_tail_, __ATOMIC_RELEASE);
      published_ = local_tail_;
    }
    if (!n && !wait_nr) return 0;
    for (;;) {
      const long r = syscall(__NR_io_uring_enter, fd_, n, wait_nr, wait_nr ? IORING_ENTER_GETEVENTS : 0, nullptr, 0);
      if (r >= 0) return int(r);
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EBUSY) return 0;  // completions pending; the caller reaps and retries
      die("io_uring_enter: %s", strerror(errno));
    }
  }
  // Enters the kernel to run completion work without blocking.
  void poll() { syscall(__NR_io_uring_enter, fd_, 0, 0, IORING_ENTER_GETEVENTS, nullptr, 0); }

  unsigned cq_ready() const { return __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE) - *cq_head_; }
  const io_uring_cqe& cqe(unsigned i) const { return cqes_[(*cq_head_ + i) & cq_mask_]; }
  void cq_advance(unsigned n) { __atomic_store_n(cq_head_, *cq_head_ + n, __ATOMIC_RELEASE); }

 private:
  void* map(size_t sz, uint64_t off) {
    void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, off);
    if (p == MAP_FAILED) die("io_uring mmap: %s", strerror(errno));
    return p;
  }

  int fd_ = -1;
  void *sq_ptr_ = nullptr, *cq_ptr_ = nullptr;
  size_t sq_sz_ = 0, cq_sz_ = 0, sqes_sz_ = 0;
  unsigned *sq_head_, *sq_tail_, *cq_head_, *cq_tail_;
  unsigned sq_mask_, cq_mask_, sq_entries_, cq_entries_;
  io_uring_sqe* sqes_;
  io_uring_cqe* cqes_;
  unsigned local_tail_ = 0, published_ = 0;
};

}  // namespace engram
