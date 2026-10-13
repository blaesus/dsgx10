// dsw_convert: cut one rank's weight file out of the safetensors checkpoint.
//
//   dsw_convert --ckpt DIR --rank R --out FILE [--world 4] [--threads 8] [--mem-gib 6]
//   dsw_convert --ckpt DIR --rank R --plan         print the partition and sizes only
//
// Reads with O_DIRECT (no page-cache build-up), one output tensor per task, bounded memory, and
// writes FILE.tmp with O_DIRECT, renaming it to FILE once the TOC and header are in place. Tensors
// with a tiled layout are tiled on the GPU by the kernels' own dsk::tile_fp8 / dsk::tile_fp4, so the
// file holds exactly what those kernels were tested on.
#include <cuda_runtime.h>
#include <sys/statvfs.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <map>
#include <mutex>
#include <thread>

#include "kernels/dsk.h"
#include "weights/io.h"
#include "weights/layout.h"
#include "weights/plan.h"

using namespace dsw;

namespace {

// gemv_woa widens wo_a's FP8 to bf16 exactly only while every block scale is <= 2^7, i.e. its ue8m0
// byte <= 127 + 7 (engine/kernels/README.md).
constexpr uint8_t kWoaMaxScale = 134;

struct Budget {  // bytes of output buffers in flight
  std::mutex m;
  std::condition_variable cv;
  uint64_t left;
  void take(uint64_t n) {
    std::unique_lock<std::mutex> l(m);
    cv.wait(l, [&] { return left >= n; });
    left -= n;
  }
  void give(uint64_t n) {
    {
      std::lock_guard<std::mutex> l(m);
      left += n;
    }
    cv.notify_all();
  }
};

TileGeom geom_of(const OutTensor& o) {
  TileGeom g;
  std::string why;
  if (!tile_geom(o.layout, o.dtype, uint32_t(o.shape.size()), o.shape.data(), g, &why)) die("%s: %s", o.name.c_str(), why.c_str());
  return g;
}

// Tiles tensors in place, one at a time, through fixed device buffers: big ones for L1/L2 weights and
// L3 values, small ones for L3 scales. tile_fp4 always takes values and scales together; a tensor of
// one kind runs with whatever the other kind's buffers hold and that half of the output is dropped.
class GpuTiler {
 public:
  void init(const std::vector<OutTensor>& ts) {
    uint64_t big = 0, small = 0;
    for (auto& o : ts) {
      if (o.layout == Layout::Natural) continue;
      TileGeom g = geom_of(o);
      if (o.layout != Layout::L3) big = std::max(big, o.bytes);
      else big = std::max(big, g.rows * g.k / 2), small = std::max(small, g.rows * g.k / 32);
    }
    if (!big) return;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&a_, big));
    CUDA_CHECK(cudaMalloc(&b_, big));
    if (small) {
      CUDA_CHECK(cudaMalloc(&sa_, small));
      CUDA_CHECK(cudaMalloc(&sb_, small));
      CUDA_CHECK(cudaMemset(sa_, 0, small));
    }
    CUDA_CHECK(cudaMemset(a_, 0, big));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("GPU tiling buffers: 2 x %.0f MiB + 2 x %.0f MiB\n", big / 1048576.0, small / 1048576.0);
  }
  void run(const OutTensor& o, uint8_t* host) {
    const TileGeom g = geom_of(o);
    std::lock_guard<std::mutex> l(m_);
    const bool scales = o.layout == Layout::L3 && o.dtype == DType::F8_E8M0;
    uint8_t* in = scales ? sa_ : a_;
    uint8_t* out = scales ? sb_ : b_;
    CUDA_CHECK(cudaMemcpyAsync(in, host, o.bytes, cudaMemcpyHostToDevice, st_));
    if (o.layout == Layout::L3) dsk::tile_fp4(a_, sa_, b_, sb_, int(g.rows), int(g.k), st_);
    else dsk::tile_fp8(a_, b_, int(g.rows), int(g.k), st_);
    CUDA_CHECK(cudaMemcpyAsync(host, out, o.bytes, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
  }
  ~GpuTiler() {
    for (void* p : {a_, b_, sa_, sb_})
      if (p) cudaFree(p);
  }

 private:
  std::mutex m_;
  cudaStream_t st_ = nullptr;
  uint8_t *a_ = nullptr, *b_ = nullptr, *sa_ = nullptr, *sb_ = nullptr;
};

bool ends_with(const std::string& s, const char* suffix) {
  size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

uint64_t row_bytes(const SrcTensor& t) {
  uint64_t rb = st_dtype_bytes(t.dtype);
  for (size_t i = 1; i < t.shape.size(); i++) rb *= t.shape[i];
  return rb;
}

// Reads the aligned superset of [off, off+n) into scratch and returns a pointer to byte `off`.
const uint8_t* read_span(int fd, uint64_t fsize, uint64_t off, uint64_t n, AlignedBuf& scratch) {
  uint64_t a0 = off / kAlign * kAlign, a1 = align_up(off + n);
  scratch.reserve(a1 - a0);
  uint64_t want = std::min(a1, align_up(fsize)) - a0, got = 0;
  while (got < want) {
    ssize_t r = ::pread(fd, scratch.p + got, want - got, off_t(a0 + got));
    if (r < 0 && errno == EINTR) continue;
    if (r < 0) die("pread: %s", std::strerror(errno));
    if (r == 0) break;
    got += r;
  }
  if (got < off + n - a0) die("short read at %llu", (unsigned long long)off);
  return scratch.p + (off - a0);
}

void copy_piece(const Piece& p, int fd, uint64_t fsize, uint8_t* out, AlignedBuf& scratch) {
  const uint64_t rb = row_bytes(*p.src);
  if (p.col0 == 0 && p.ncols == rb) {
    direct_read(fd, fsize, p.src->offset + p.row0 * rb, p.nrows * rb, out + p.dst, scratch);
    return;
  }
  const uint64_t rows_per = std::max<uint64_t>(1, (64ull << 20) / rb);
  for (uint64_t r = 0; r < p.nrows; r += rows_per) {
    uint64_t n = std::min(rows_per, p.nrows - r);
    const uint8_t* src = read_span(fd, fsize, p.src->offset + (p.row0 + r) * rb, n * rb, scratch);
    for (uint64_t i = 0; i < n; i++) std::memcpy(out + p.dst + (r + i) * p.ncols, src + i * rb + p.col0, p.ncols);
  }
}

void print_plan(const Plan& plan) {
  std::map<std::string, std::pair<uint64_t, int>> comp, lay;
  for (auto& o : plan.tensors) {
    comp[o.component].first += o.bytes;
    comp[o.component].second++;
    lay[layout_name(o.layout)].first += o.bytes;
    lay[layout_name(o.layout)].second++;
  }
  std::printf("rank %d/%d: %zu tensors, file %.2f GiB, %zu checkpoint tensors left out (vision, Engram tables)\n",
              plan.rank, plan.world, plan.tensors.size(), plan.file_bytes / 1073741824.0, plan.skipped.size());
  for (auto& [c, v] : comp) std::printf("  %-26s %9.3f GiB  %5d tensors\n", c.c_str(), v.first / 1073741824.0, v.second);
  std::printf("layouts:");
  for (auto& [c, v] : lay) std::printf("  %s %.3f GiB (%d)", c.c_str(), v.first / 1073741824.0, v.second);
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt, out, only;
  int rank = -1, world = 4, threads = 8;
  double mem_gib = 6;
  bool plan_only = false, coverage = true;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--out") out = next();
    else if (a == "--rank") rank = std::stoi(next());
    else if (a == "--world") world = std::stoi(next());
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--mem-gib") mem_gib = std::stod(next());
    else if (a == "--plan") plan_only = true;
    else if (a == "--no-coverage") coverage = false;
    else if (a == "--only") only = next();
    else die("unknown argument %s", a.c_str());
  }
  if (ckpt.empty() || rank < 0 || (!plan_only && out.empty()))
    die("usage: dsw_convert --ckpt DIR --rank R (--out FILE | --plan) [--world 4] [--threads 8] [--mem-gib 6]");

  double t0 = now_s();
  Checkpoint ck = Checkpoint::open(ckpt);
  std::printf("checkpoint: %zu shards, %zu tensors (%.1f s)\n", ck.shards.size(), ck.tensors.size(), now_s() - t0);
  if (coverage) {
    check_coverage(ck, world);
    std::printf("coverage: every tensor is whole on every rank or cut into %d disjoint, complete pieces\n", world);
  }
  Plan plan = make_plan(ck, rank, world);
  if (!only.empty()) {  // a test file holding just the tensors whose names contain `only`
    std::erase_if(plan.tensors, [&](const OutTensor& o) { return o.name.find(only) == std::string::npos; });
    plan.data_offset = align_up(plan.toc_offset + plan.tensors.size() * sizeof(TocEntry));
    uint64_t off = plan.data_offset;
    for (auto& o : plan.tensors) o.offset = off, off += align_up(o.bytes);
    plan.file_bytes = off;
  }
  print_plan(plan);
  if (plan_only) return 0;

  // CUDA first: it can fail to start once the page cache has filled memory (docs/findings.md)
  GpuTiler tiler;
  tiler.init(plan.tensors);

  std::string dir = out.substr(0, out.find_last_of('/') + 1);
  struct statvfs sv;
  if (::statvfs(dir.empty() ? "." : dir.c_str(), &sv) != 0) die("statvfs %s: %s", dir.c_str(), std::strerror(errno));
  uint64_t free_bytes = uint64_t(sv.f_bavail) * sv.f_frsize;
  if (free_bytes < plan.file_bytes + (2ull << 30))
    die("%s: %.1f GiB free, need %.1f GiB + 2", dir.c_str(), free_bytes / 1073741824.0, plan.file_bytes / 1073741824.0);

  std::string tmp = out + ".tmp";
  int ofd = open_or_die(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT);
  if (int e = posix_fallocate(ofd, 0, off_t(plan.file_bytes))) die("fallocate %s: %s", tmp.c_str(), std::strerror(e));

  std::vector<int> sfd(ck.shards.size());
  for (size_t i = 0; i < sfd.size(); i++) sfd[i] = open_or_die(ck.shards[i], O_RDONLY | O_DIRECT);

  uint64_t largest = 0;
  for (auto& o : plan.tensors) largest = std::max(largest, align_up(o.bytes));
  Budget budget{.left = std::max<uint64_t>(uint64_t(mem_gib * 1073741824.0), largest)};
  std::vector<uint32_t> crc(plan.tensors.size());
  std::atomic<size_t> next{0}, done{0}, woa_scales{0};
  std::atomic<uint64_t> written{0}, tiled{0};
  std::atomic<int> woa_max{0};
  std::mutex log_m;
  double t1 = now_s();

  auto worker = [&]() {
    AlignedBuf scratch(64ull << 20), buf;
    for (size_t i; (i = next.fetch_add(1)) < plan.tensors.size();) {
      const OutTensor& o = plan.tensors[i];
      uint64_t padded = align_up(o.bytes);
      budget.take(padded);
      {
        AlignedBuf b(padded);
        std::memset(b.p + o.bytes, 0, padded - o.bytes);
        for (auto& p : o.pieces) copy_piece(p, sfd[p.src->shard], ck.shard_bytes[p.src->shard], b.p, scratch);
        if (ends_with(o.name, ".attn.wo_a.scale")) {
          const uint8_t mx = *std::max_element(b.p, b.p + o.bytes);
          if (mx > kWoaMaxScale)
            die("%s: a scale byte is %u > %u, so gemv_woa's exact widening to bf16 does not hold", o.name.c_str(), mx,
                kWoaMaxScale);
          for (int cur = woa_max.load(); mx > cur && !woa_max.compare_exchange_weak(cur, mx);) {
          }
          woa_scales++;
        }
        if (o.layout != Layout::Natural) {
          tiler.run(o, b.p);
          tiled += o.bytes;
        }
        crc[i] = crc32c(b.p, o.bytes);
        pwrite_all(ofd, b.p, padded, o.offset);
      }
      budget.give(padded);
      uint64_t w = written.fetch_add(padded) + padded;
      size_t d = done.fetch_add(1) + 1;
      if (d % 100 == 0 || d == plan.tensors.size()) {
        std::lock_guard<std::mutex> l(log_m);
        double dt = now_s() - t1;
        std::printf("  %zu/%zu tensors, %.1f GiB written, %.0f s, %.2f GB/s\n", d, plan.tensors.size(),
                    w / 1073741824.0, dt, w / dt / 1e9);
        std::fflush(stdout);
      }
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (woa_scales)
    std::printf("wo_a: %zu scale tensors, largest scale byte %d (limit %u)\n", size_t(woa_scales), woa_max.load(),
                kWoaMaxScale);
  std::printf("tiled on the GPU: %.2f GiB\n", tiled / 1073741824.0);

  // TOC, then the header last: until it is written the file has no magic
  std::vector<TocEntry> toc(plan.tensors.size());
  for (size_t i = 0; i < toc.size(); i++) {
    fill_entry(plan.tensors[i], toc[i]);
    toc[i].crc32c = crc[i];
  }
  uint64_t toc_padded = plan.data_offset - plan.toc_offset;
  AlignedBuf tb(toc_padded);
  std::memset(tb.p, 0, toc_padded);
  std::memcpy(tb.p, toc.data(), toc.size() * sizeof(TocEntry));
  pwrite_all(ofd, tb.p, toc_padded, plan.toc_offset);

  AlignedBuf hb(sizeof(Header));
  auto* h = reinterpret_cast<Header*>(hb.p);
  std::memset(h, 0, sizeof(Header));
  std::memcpy(h->magic, kMagic, 8);
  h->version = kVersion;
  h->header_bytes = sizeof(Header);
  h->rank = uint32_t(rank);
  h->world = uint32_t(world);
  h->n_tensors = uint32_t(toc.size());
  h->toc_entry_bytes = sizeof(TocEntry);
  h->toc_offset = plan.toc_offset;
  h->data_offset = plan.data_offset;
  h->data_bytes = plan.file_bytes - plan.data_offset;
  h->file_bytes = plan.file_bytes;
  h->alignment = kAlign;
  h->toc_crc32c = crc32c(toc.data(), toc.size() * sizeof(TocEntry));
  std::string model = ckpt.substr(ckpt.find_last_of('/', ckpt.size() - 2) + 1);
  if (!model.empty() && model.back() == '/') model.pop_back();
  std::snprintf(h->model, sizeof h->model, "%s", model.c_str());
  std::snprintf(h->source, sizeof h->source, "%s", ckpt.c_str());
  std::time_t now = std::time(nullptr);
  std::strftime(h->created, sizeof h->created, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  pwrite_all(ofd, hb.p, sizeof(Header), 0);
  if (::fsync(ofd) != 0) die("fsync: %s", std::strerror(errno));
  ::close(ofd);
  for (int fd : sfd) ::close(fd);
  if (::rename(tmp.c_str(), out.c_str()) != 0) die("rename: %s", std::strerror(errno));
  double dt = now_s() - t1;
  std::printf("wrote %s: %.2f GiB in %.0f s (%.2f GB/s)\n", out.c_str(), plan.file_bytes / 1073741824.0, dt,
              plan.file_bytes / dt / 1e9);
  return 0;
}
