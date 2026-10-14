#include "weights/loader.h"

#include <cuda_runtime.h>

#include <atomic>
#include <thread>

#include "weights/io.h"

namespace glw {

WeightFile::WeightFile(const std::string& path, int rank, int world) : path_(path) {
  fd_ = open_or_die(path, O_RDONLY | O_DIRECT);
  fsize_ = file_size_of(fd_);
  AlignedBuf hb(sizeof(Header));
  pread_all(fd_, hb.p, sizeof(Header), 0);
  std::memcpy(&header_, hb.p, sizeof(Header));
  const Header& h = header_;
  if (std::memcmp(h.magic, kMagic, 8) != 0) die("%s: not an altar-1 .glw file", path.c_str());
  if (h.version != kVersion || h.header_bytes != sizeof(Header) || h.toc_entry_bytes != sizeof(TocEntry) ||
      h.alignment != kAlign || h.file_bytes != fsize_ || h.data_offset + h.data_bytes != fsize_ ||
      h.data_offset % kAlign || h.n_tensors * sizeof(TocEntry) > h.data_offset - h.toc_offset)
    die("%s: bad header (version %u, %llu bytes)", path.c_str(), h.version, (unsigned long long)fsize_);
  if (rank >= 0) {
    const std::string why = rank_mismatch(h, uint32_t(rank), uint32_t(world));
    if (!why.empty()) die("%s: %s (a weight file serves only the rank it was cut for)", path.c_str(), why.c_str());
  }
  const uint64_t toc_padded = h.data_offset - h.toc_offset;
  AlignedBuf tb(toc_padded);
  pread_all(fd_, tb.p, toc_padded, h.toc_offset);
  if (crc32c(tb.p, h.n_tensors * sizeof(TocEntry)) != h.toc_crc32c) die("%s: TOC CRC mismatch", path.c_str());
  auto* toc = reinterpret_cast<const TocEntry*>(tb.p);
  tensors_.resize(h.n_tensors);
  for (uint32_t i = 0; i < h.n_tensors; i++) {
    const TocEntry& e = toc[i];
    DeviceTensor& t = tensors_[i];
    t.name.assign(e.name, strnlen(e.name, sizeof e.name));
    t.dtype = DType(e.dtype);
    if (e.layout > uint32_t(Layout::Bf16Tile)) die("%s: bad layout %u", t.name.c_str(), e.layout);
    if (e.kind < uint32_t(SliceKind::Replicated) || e.kind > uint32_t(SliceKind::Reserved))
      die("%s: bad kind %u", t.name.c_str(), e.kind);
    t.layout = Layout(e.layout);
    t.kind = SliceKind(e.kind);
    t.ndim = int(e.ndim);
    for (uint32_t d = 0; d < e.ndim && d < 4; d++) t.shape[d] = int64_t(e.shape[d]);
    t.layer = e.layer;
    t.blk_rows = e.blk_rows;
    t.blk_k = e.blk_k;
    t.flags = e.flags;
    t.fmt = e.fmt;
    t.rel_err = e.rel_err;
    t.bytes = e.bytes;
    t.reserve_bytes = e.reserve_bytes;
    t.file_offset = e.offset;
    t.offset = t.reserved() ? 0 : e.offset - h.data_offset;
    t.crc32c = e.crc32c;
    if (t.reserved()) {
      if (e.bytes || e.offset) die("%s: a reserved entry with data", t.name.c_str());
    } else if (e.offset < h.data_offset || e.offset % kAlign || e.offset + e.bytes > fsize_) {
      die("%s: bad extent", t.name.c_str());
    }
    if (!index_.emplace(t.name, i).second) die("%s: tensor %s twice", path.c_str(), t.name.c_str());
  }
}

WeightFile::~WeightFile() {
  if (arena_) cudaFree(arena_);
  if (fd_ >= 0) ::close(fd_);
}

const DeviceTensor* WeightFile::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

const DeviceTensor& WeightFile::operator[](const std::string& name) const {
  const DeviceTensor* t = find(name);
  if (!t) die("%s: no tensor %s", path_.c_str(), name.c_str());
  if (!t->data) die("%s: tensor %s is not loaded", path_.c_str(), name.c_str());
  return *t;
}

std::vector<uint8_t> WeightFile::read_host(const DeviceTensor& t) const {
  std::vector<uint8_t> out(t.bytes);
  if (!t.bytes) return out;
  AlignedBuf scratch(64ull << 20);
  direct_read(fd_, fsize_, t.file_offset, t.bytes, out.data(), scratch);
  if (crc32c(out.data(), out.size()) != t.crc32c) die("%s: %s CRC mismatch", path_.c_str(), t.name.c_str());
  return out;
}

LoadStats WeightFile::load(const LoadOptions& opt) {
  if (arena_) die("%s: already loaded", path_.c_str());
  const uint64_t chunk = align_up(opt.chunk);

  // The chosen tensors, packed in file order (with no selection the arena is the file's data region byte for
  // byte); runs of tensors adjacent in the file are read as one extent.
  struct Job {
    uint64_t file, arena, len;
  };
  std::vector<Job> ext, jobs;
  std::vector<size_t> chosen;
  uint64_t total = 0;
  for (size_t i = 0; i < tensors_.size(); i++) {
    DeviceTensor& t = tensors_[i];
    if (t.reserved() || (opt.select && !opt.select(t))) continue;
    const uint64_t padded = align_up(t.bytes);
    if (opt.limit && total + padded > opt.limit) break;
    t.offset = total;
    chosen.push_back(i);
    if (!ext.empty() && ext.back().file + ext.back().len == t.file_offset) ext.back().len += padded;
    else ext.push_back({t.file_offset, total, padded});
    total += padded;
  }
  if (!total) die("%s: no tensor selected", path_.c_str());
  for (auto& x : ext)
    for (uint64_t o = 0; o < x.len; o += chunk) jobs.push_back({x.file + o, x.arena + o, std::min(chunk, x.len - o)});
  CUDA_CHECK(cudaMalloc(&arena_, total));
  arena_bytes_ = total;

  double t0 = now_s();
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    void* buf[2];
    cudaEvent_t ev[2];
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (int b = 0; b < 2; b++) {
      CUDA_CHECK(cudaHostAlloc(&buf[b], chunk, cudaHostAllocDefault));
      CUDA_CHECK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming));
    }
    bool pending[2] = {false, false};
    int cur = 0;
    for (size_t k; (k = next.fetch_add(1)) < jobs.size();) {
      const Job& j = jobs[k];
      if (pending[cur]) CUDA_CHECK(cudaEventSynchronize(ev[cur]));
      pread_all(fd_, buf[cur], j.len, j.file);
      CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(arena_) + j.arena, buf[cur], j.len, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaEventRecord(ev[cur], st));
      pending[cur] = true;
      cur ^= 1;
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    for (int b = 0; b < 2; b++) {
      CUDA_CHECK(cudaFreeHost(buf[b]));
      CUDA_CHECK(cudaEventDestroy(ev[b]));
    }
    CUDA_CHECK(cudaStreamDestroy(st));
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, opt.threads); t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();

  LoadStats s;
  s.bytes = total;
  s.seconds = now_s() - t0;
  s.gbps = total / s.seconds / 1e9;
  for (size_t i : chosen) tensors_[i].data = static_cast<uint8_t*>(arena_) + tensors_[i].offset;

  if (opt.verify_crc) {
    double t1 = now_s();
    void* host;
    CUDA_CHECK(cudaHostAlloc(&host, chunk, cudaHostAllocDefault));
    for (auto& t : tensors_) {
      if (!t.data) continue;
      uint32_t crc = 0;
      for (uint64_t o = 0; o < t.bytes; o += chunk) {
        uint64_t n = std::min(chunk, t.bytes - o);
        CUDA_CHECK(cudaMemcpy(host, static_cast<uint8_t*>(t.data) + o, n, cudaMemcpyDeviceToHost));
        crc = crc32c(host, n, crc);
      }
      if (crc != t.crc32c) die("%s: %s CRC mismatch after load", path_.c_str(), t.name.c_str());
    }
    CUDA_CHECK(cudaFreeHost(host));
    s.verify_seconds = now_s() - t1;
  }
  return s;
}

}  // namespace glw
