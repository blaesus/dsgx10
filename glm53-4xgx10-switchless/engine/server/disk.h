// The session store's disk tier on one rank (docs/serving.md): base.spill, a file of `pages` pages of page_bytes
// (one pool page of the paged cache, Model::cache_get's layout), written and read through one pinned staging page
// (O_DIRECT when the file system takes it). Rank 0 decides which page holds what; every rank keeps a hash of each
// page it wrote and checks it when the page comes back (a mismatch stops the process).
// Kept sessions: keep(gen) syncs the file and writes base.index (the generation, the run id, the geometry, the
// pages' hashes); the next start's kept_check reads it back, and a start whose ranks all found the same generation
// opens the file as it is. The index is removed either way, so that a crash from then on starts clean.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace srv {

class Disk {
 public:
  // The index of an earlier planned stop beside base: its generation when it is whole and of this run id and
  // geometry, at most max_age_s old, with its file beside it (pages: its size), else 0 with *why ("": no index).
  static uint64_t kept_check(const std::string& base, uint64_t run_id, size_t page_bytes, double max_age_s,
                             int* pages, std::string* why);
  // Opens (reuse: the kept file as it is, its hashes from the index) or creates base.spill; removes the index.
  Disk(const std::string& base, int pages, size_t page_bytes, bool reuse, uint64_t run_id);
  ~Disk();
  Disk(const Disk&) = delete;
  Disk& operator=(const Disk&) = delete;

  int pages() const { return pages_; }
  size_t page_bytes() const { return page_bytes_; }
  void* staging() const { return stage_; }  // page_bytes of pinned host memory
  const std::string& path() const { return path_; }
  void write_page(int p);  // the staging page to page p
  void read_page(int p);   // page p to the staging page, its hash checked
  void keep(uint64_t gen);  // sync, then base.index

 private:
  std::string base_, path_;
  int fd_ = -1;
  bool direct_ = false;
  int pages_ = 0;
  size_t page_bytes_ = 0;
  uint64_t run_id_ = 0;
  void* stage_ = nullptr;
  std::vector<uint64_t> hash_;  // per page: what was written (0: nothing)
};

uint64_t hash64(const void* p, size_t n);  // a fast 64-bit hash of n bytes (n a multiple of 32)

}  // namespace srv
