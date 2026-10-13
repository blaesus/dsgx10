// The session store's disk tier (state.cu): one file per rank, preallocated, that evicted sessions'
// blocks are written to and read back from. Writes leave the model's stream through pinned staging: the
// copy to staging is queued on the stream (after whatever wrote the block before), and a writer thread
// writes it out once that copy is done, so a block can be handed out again at once. Reads run on the
// caller's thread, each into one of two pinned buffers and from there to the device on the stream.
// Offsets and sizes are multiples of 64 KiB (the store's grain), so the file is opened O_DIRECT and
// the page cache stays out of it. Each block's checksum, taken on the device before it goes out and
// after it comes back (state.cu), lives in pinned memory by its offset. The writer thread logs each
// flush's time on the stream (checksums, copies to staging) and in the file. A planned stop keeps the
// file for the next start (state.cu, "kept sessions"): its sessions' blocks stay where they are, and the
// next run opens it as it is.
#pragma once
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "model/state.h"

namespace model::detail {

class SpillFile {
 public:
  // <base>.spill of `bytes`, with `staging` bytes of pinned memory for writes in flight; its directory
  // made (mode 0700: kept sessions are conversations). An earlier run's file is removed first, unless
  // `reuse`: then it is opened as it is (checked before: state.cu kept_check). Dies when the directory or
  // the file cannot be made.
  SpillFile(const std::string& base, size_t bytes, size_t staging, bool reuse);
  // Waits for the writes in flight, then removes the file (unless keep()).
  ~SpillFile();
  SpillFile(const SpillFile&) = delete;
  SpillFile& operator=(const SpillFile&) = delete;

  // `bytes` of device memory at src to the file at `off`, after the work queued on st so far. Waits
  // while staging is full. Returns the write's number (wait() for it).
  uint64_t write(const void* src, size_t bytes, size_t off, cudaStream_t st);
  // Marks st before the work that precedes the next writes (their checksums), to time it apart.
  void begin(cudaStream_t st);
  // Hands the writes since the last flush to the writer (they wait for one event recorded on st now).
  void flush(cudaStream_t st);
  // Until write `seq` is in the file.
  void wait(uint64_t seq);
  // Every write so far in the file and on the disk (fdatasync): before a planned stop's index.
  void sync();
  // The file stays when this is destroyed: a planned stop kept sessions in it.
  void keep() { keep_ = true; }
  // `bytes` of the file at `off` to device memory at dst, queued on st (the read itself is done when
  // this returns; its buffer is read again only after the copy).
  void read(void* dst, size_t bytes, size_t off, cudaStream_t st);

  // The checksum slot of the block at file offset `off`: as it went out, or as it came back.
  uint64_t* sum(size_t off, bool back) { return sums_ + (back ? slots_ : 0) + off / kPoolGrain; }

  const std::string& path() const { return path_; }
  size_t size() const { return size_; }
  bool direct() const { return direct_; }

 private:
  struct Job {
    uint64_t seq;
    size_t at, off, n;  // staging offset, file offset, bytes
    cudaEvent_t ev;
    bool last;  // the last job waiting for ev (it destroys it, and logs the flush)
    cudaEvent_t pre = nullptr, start = nullptr;  // the last job's: begin() and the first copy
    size_t batch = 0;                            // the last job's: the flush's bytes
  };
  void writer();

  std::string path_;
  int fd_ = -1;
  size_t size_ = 0;
  bool direct_ = true;
  bool keep_ = false;
  uint8_t* stage_ = nullptr;  // pinned
  Pool stage_pool_;           // offsets into stage_ (base null)
  uint8_t* rbuf_[2] = {};     // pinned read buffers of kReadBuf
  cudaEvent_t rev_[2] = {};   // their last copies
  int rnext_ = 0;
  uint64_t* sums_ = nullptr;  // pinned: 2 x slots_, out then back
  size_t slots_ = 0;
  cudaEvent_t pre_ = nullptr, start_ = nullptr;  // the pending flush's (the caller's thread only)
  size_t batch_ = 0;
  std::vector<Job> pending_;  // written to staging, not yet flushed (the caller's thread only)
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> q_;  // flushed, not yet written
  uint64_t seq_ = 0, done_ = 0;
  bool stop_ = false;
  std::thread th_;
};

}  // namespace model::detail
