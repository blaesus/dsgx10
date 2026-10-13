// Packs a rank's Engram shards from the checkpoint: the files the row store reads (store.cpp,
// docs/engram.md). For each Engram layer, rank r gets its quarter of the layer's table, the rows
// [rows*r/4, rows*(r+1)/4), as 264-byte records (the row's 256 e4m3 weight bytes, then its 8 ue8m0
// scale bytes) behind a 4096-byte header: six little-endian u64 (the magic "DSV41EN1", the layer, lo,
// hi (exclusive), the table's rows, 264), then zeros. That is the MiaAI kit's shard layout (its
// scripts/pack_engram.py with --tp 4), and the output is byte for byte the kit's shards.
//
//   engram_pack <checkpoint-dir> <out-dir> <rank> [--layer L] [--rows A:B]
//
// writes <out-dir>/engram-l<L>-r<rank>of4.bin for each Engram layer (1 and 14; 23.6 GiB each, 25.3 GB),
// through a .part file renamed once complete. It reads only the rank's rows, about 51 GB of the
// checkpoint, and keeps little of either side in the page cache. --layer packs one layer. --rows A:B
// writes the header and only the records [A, B) of the rank's range, to
// engram-l<L>-r<rank>of4.rows<A>-<B>.bin: the full shard's bytes [0, 4096) and [4096 + 264 A,
// 4096 + 264 B), to compare with a shard made elsewhere (cmp -n 4096, then
// cmp -i 4096:$((4096 + 264 * A)) -n $((264 * (B - A)))).
#include <fcntl.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common/check.h"
#include "engram/engram.h"
#include "weights/safetensors.h"

using namespace engram;

namespace {

constexpr uint64_t kShardMagic = 0x31344E4531565344ull;  // "DSV41EN1", as store.cpp checks it
constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kScaleBytes = kRowBytes - kWeightBytes;
constexpr uint64_t kChunkRows = uint64_t(1) << 18;  // rows per step: 64 MiB of weights, 66 MiB of records

void read_at(int fd, void* p, uint64_t n, uint64_t off, const std::string& path) {
  auto* b = static_cast<uint8_t*>(p);
  while (n > 0) {
    const ssize_t got = pread(fd, b, n, off_t(off));
    if (got <= 0)
      die("engram_pack: %s: reading at byte %llu: %s", path.c_str(), (unsigned long long)off,
          got < 0 ? strerror(errno) : "the file ends there");
    b += got, n -= uint64_t(got), off += uint64_t(got);
  }
}

void write_at(int fd, const void* p, uint64_t n, uint64_t off, const std::string& path) {
  auto* b = static_cast<const uint8_t*>(p);
  while (n > 0) {
    const ssize_t put = pwrite(fd, b, n, off_t(off));
    if (put <= 0)
      die("engram_pack: %s: writing at byte %llu: %s", path.c_str(), (unsigned long long)off,
          put < 0 ? strerror(errno) : "nothing written");
    b += put, n -= uint64_t(put), off += uint64_t(put);
  }
}

uint64_t number(const char* s, const char* what) {
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (errno || end == s || *end || *s == '-') die("engram_pack: %s %s is not a number", what, s);
  return v;
}

// One Engram layer's table in the checkpoint: its weight and scale tensors.
struct Table {
  uint64_t layer = 0, rows = 0;
  const dsw::SrcTensor* weight = nullptr;
  const dsw::SrcTensor* scale = nullptr;
};

std::map<uint64_t, Table> tables(const dsw::Checkpoint& ck) {
  std::map<uint64_t, Table> out;
  for (const auto& [name, t] : ck.tensors) {
    unsigned layer = 0;
    char rest[32] = {};
    if (std::sscanf(name.c_str(), "layers.%u.engram.embed.%31s", &layer, rest) != 2 ||
        name != "layers." + std::to_string(layer) + ".engram.embed." + rest)
      continue;
    Table& tb = out[layer];
    tb.layer = layer;
    if (std::strcmp(rest, "weight") == 0) tb.weight = &t;
    if (std::strcmp(rest, "scale") == 0) tb.scale = &t;
  }
  for (auto& [layer, tb] : out) {
    const dsw::SrcTensor *w = tb.weight, *s = tb.scale;
    if (!w || !s) die("engram_pack: layer %llu's Engram table lacks its weight or its scale", (unsigned long long)layer);
    if (w->dtype != "F8_E4M3" || w->shape.size() != 2 || w->shape[1] != uint64_t(kWeightBytes) ||
        s->dtype != "F8_E8M0" || s->shape.size() != 2 || s->shape[1] != kScaleBytes || s->shape[0] != w->shape[0] ||
        w->bytes != w->shape[0] * kWeightBytes || s->bytes != s->shape[0] * kScaleBytes)
      die("engram_pack: layer %llu: %s and %s are not [rows, %d] F8_E4M3 and [rows, %llu] F8_E8M0",
          (unsigned long long)layer, w->name.c_str(), s->name.c_str(), kWeightBytes, (unsigned long long)kScaleBytes);
    tb.rows = w->shape[0];
  }
  if (out.empty()) die("engram_pack: %s has no layers.<L>.engram.embed tensors", ck.dir.c_str());
  return out;
}

// Records [first, last) of the rank's range [lo, hi) of one table, into out (the header first).
void pack(const dsw::Checkpoint& ck, const Table& tb, uint64_t lo, uint64_t hi, uint64_t first, uint64_t last,
          const std::string& out) {
  const std::string wpath = ck.shards[tb.weight->shard], spath = ck.shards[tb.scale->shard];
  const uint64_t bytes = kHeaderBytes + (last - first) * kRowBytes;
  std::printf("layer %llu: rows [%llu, %llu) of %llu, records [%llu, %llu) of %llu: %s, %llu bytes\n",
              (unsigned long long)tb.layer, (unsigned long long)lo, (unsigned long long)hi,
              (unsigned long long)tb.rows, (unsigned long long)first, (unsigned long long)last,
              (unsigned long long)(hi - lo), out.c_str(), (unsigned long long)bytes);
  std::fflush(stdout);
  const int wfd = open(wpath.c_str(), O_RDONLY), sfd = open(spath.c_str(), O_RDONLY);
  if (wfd < 0 || sfd < 0) die("engram_pack: open %s: %s", (wfd < 0 ? wpath : spath).c_str(), strerror(errno));
  const std::string part = out + ".part";
  const int ofd = open(part.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (ofd < 0) die("engram_pack: create %s: %s", part.c_str(), strerror(errno));

  uint64_t h[kHeaderBytes / 8] = {};
  h[0] = kShardMagic, h[1] = tb.layer, h[2] = lo, h[3] = hi, h[4] = tb.rows, h[5] = kRowBytes;
  write_at(ofd, h, kHeaderBytes, 0, part);

  std::vector<uint8_t> w(kChunkRows * kWeightBytes), s(kChunkRows * kScaleBytes), rec(kChunkRows * kRowBytes);
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t prev_off = 0, prev_len = 0, shown = 0;
  for (uint64_t i = first; i < last;) {
    const uint64_t n = std::min(kChunkRows, last - i), row = lo + i;
    read_at(wfd, w.data(), n * kWeightBytes, tb.weight->offset + row * kWeightBytes, wpath);
    read_at(sfd, s.data(), n * kScaleBytes, tb.scale->offset + row * kScaleBytes, spath);
    for (uint64_t k = 0; k < n; k++) {
      std::memcpy(&rec[k * kRowBytes], &w[k * kWeightBytes], kWeightBytes);
      std::memcpy(&rec[k * kRowBytes + kWeightBytes], &s[k * kScaleBytes], kScaleBytes);
    }
    const uint64_t off = kHeaderBytes + (i - first) * kRowBytes, len = n * kRowBytes;
    write_at(ofd, rec.data(), len, off, part);
    // Neither side is read again: drop the checkpoint's pages, and write the records back as they go
    // and drop them, so the page cache (the GPU's memory on a GB10) holds at most two steps of them.
    posix_fadvise(wfd, off_t(tb.weight->offset + row * kWeightBytes), off_t(n * kWeightBytes), POSIX_FADV_DONTNEED);
    posix_fadvise(sfd, off_t(tb.scale->offset + row * kScaleBytes), off_t(n * kScaleBytes), POSIX_FADV_DONTNEED);
    sync_file_range(ofd, off_t(off), off_t(len), SYNC_FILE_RANGE_WRITE);
    if (prev_len) {
      sync_file_range(ofd, off_t(prev_off), off_t(prev_len),
                      SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
      posix_fadvise(ofd, off_t(prev_off), off_t(prev_len), POSIX_FADV_DONTNEED);
    }
    prev_off = off, prev_len = len;
    i += n;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if ((i - first) * kRowBytes >= shown + (uint64_t(4) << 30) || i == last) {
      shown = (i - first) * kRowBytes;
      std::printf("  %.1f of %.1f GiB, %.0f s, %.2f GB/s written\n", shown / 1073741824.0,
                  (last - first) * double(kRowBytes) / 1073741824.0, sec, sec > 0 ? shown / sec / 1e9 : 0.0);
      std::fflush(stdout);
    }
  }
  if (fdatasync(ofd) != 0) die("engram_pack: %s: fdatasync: %s", part.c_str(), strerror(errno));
  if (prev_len) posix_fadvise(ofd, off_t(prev_off), off_t(prev_len), POSIX_FADV_DONTNEED);
  close(ofd), close(wfd), close(sfd);
  if (std::rename(part.c_str(), out.c_str()) != 0) die("engram_pack: rename %s: %s", part.c_str(), strerror(errno));
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  long long only_layer = -1;
  bool window = false;
  uint64_t wa = 0, wb = 0;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if ((a == "--layer" || a == "--rows") && i + 1 >= argc) die("engram_pack: %s needs a value", a.c_str());
    if (a == "--layer") {
      only_layer = (long long)number(argv[++i], "--layer");
    } else if (a == "--rows") {
      const std::string v = argv[++i];
      const size_t colon = v.find(':');
      if (colon == std::string::npos) die("engram_pack: --rows %s is not A:B", v.c_str());
      wa = number(v.substr(0, colon).c_str(), "--rows"), wb = number(v.substr(colon + 1).c_str(), "--rows");
      if (wa >= wb) die("engram_pack: --rows %s is empty", v.c_str());
      window = true;
    } else if (a.size() > 1 && a[0] == '-') {
      die("engram_pack: unknown option %s", a.c_str());
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() != 3)
    die("usage: engram_pack <checkpoint-dir> <out-dir> <rank 0-%d> [--layer L] [--rows A:B]\n"
        "  writes <out-dir>/engram-l<L>-r<rank>of%d.bin for each Engram layer (or, with --rows, the header and\n"
        "  the records [A, B) of the rank's range, to engram-l<L>-r<rank>of%d.rows<A>-<B>.bin)",
        kWorld - 1, kWorld, kWorld);
  const std::string out_dir = pos[1];
  const uint64_t rank = number(pos[2].c_str(), "the rank");
  if (rank >= uint64_t(kWorld)) die("engram_pack: rank %llu: the ranks are 0-%d", (unsigned long long)rank, kWorld - 1);

  const dsw::Checkpoint ck = dsw::Checkpoint::open(pos[0]);
  std::map<uint64_t, Table> all = tables(ck);
  if (only_layer >= 0 && !all.count(uint64_t(only_layer)))
    die("engram_pack: the checkpoint has no Engram table at layer %lld", only_layer);

  struct Job {
    const Table* tb;
    uint64_t lo, hi, first, last;
    std::string out;
  };
  std::vector<Job> jobs;
  uint64_t need = 0;
  for (const auto& [layer, tb] : all) {
    if (only_layer >= 0 && layer != uint64_t(only_layer)) continue;
    const uint64_t lo = tb.rows * rank / kWorld, hi = tb.rows * (rank + 1) / kWorld;
    std::string name = out_dir + "/engram-l" + std::to_string(layer) + "-r" + std::to_string(rank) + "of" +
                       std::to_string(kWorld);
    uint64_t first = 0, last = hi - lo;
    if (window) {
      if (wb > hi - lo)
        die("engram_pack: --rows %llu:%llu: rank %llu holds %llu records of layer %llu", (unsigned long long)wa,
            (unsigned long long)wb, (unsigned long long)rank, (unsigned long long)(hi - lo), (unsigned long long)layer);
      first = wa, last = wb;
      name += ".rows" + std::to_string(wa) + "-" + std::to_string(wb);
    }
    jobs.push_back({&tb, lo, hi, first, last, name + ".bin"});
    need += kHeaderBytes + (last - first) * kRowBytes;
  }
  struct statvfs fs;
  if (statvfs(out_dir.c_str(), &fs) != 0) die("engram_pack: %s: %s", out_dir.c_str(), strerror(errno));
  const uint64_t avail = uint64_t(fs.f_bavail) * fs.f_frsize;
  if (avail < need)
    die("engram_pack: %s has %.1f GB free, and the shards need %.1f GB", out_dir.c_str(), avail / 1e9, need / 1e9);
  const auto t0 = std::chrono::steady_clock::now();
  for (const Job& j : jobs) pack(ck, *j.tb, j.lo, j.hi, j.first, j.last, j.out);
  std::printf("done: %zu file%s, %llu bytes, in %.0f s\n", jobs.size(), jobs.size() == 1 ? "" : "s",
              (unsigned long long)need,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return 0;
}
