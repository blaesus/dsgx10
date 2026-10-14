// glw_check: the C++ structural check of a .glw file (no Python, no device): header geometry, the TOC's CRC,
// offsets in order and aligned, shape against size, reserved entries without data, every tensor's CRC32C and
// zero padding. verify.py --crc adds the manifest's SHA-256s; --fresh and --indep compare every byte with the
// master.
//
//   glw_check FILE [FILE ...] [--rank R]      (--rank: each file must be rank R's; default: the file's own)
#include <atomic>
#include <cstdio>
#include <set>
#include <thread>
#include <vector>

#include "weights/io.h"

using namespace glw;

static int check(const std::string& file, int want_rank) {
  const double t0 = now_s();
  int fd = open_or_die(file, O_RDONLY | O_DIRECT);
  const uint64_t fsize = file_size_of(fd);
  AlignedBuf hb(sizeof(Header));
  pread_all(fd, hb.p, sizeof(Header), 0);
  Header h;
  std::memcpy(&h, hb.p, sizeof h);
  if (std::memcmp(h.magic, kMagic, 8) != 0) die("%s: not an altar-1 .glw file", file.c_str());
  if (h.version != kVersion || h.header_bytes != sizeof(Header) || h.toc_entry_bytes != sizeof(TocEntry) ||
      h.alignment != kAlign || h.file_bytes != fsize || h.data_offset + h.data_bytes != fsize || h.data_offset % kAlign ||
      h.n_tensors * sizeof(TocEntry) > h.data_offset - h.toc_offset)
    die("%s: bad header geometry", file.c_str());
  if (want_rank >= 0 && !rank_mismatch(h, uint32_t(want_rank), 4).empty())
    die("%s: %s", file.c_str(), rank_mismatch(h, uint32_t(want_rank), 4).c_str());
  const uint64_t toc_padded = h.data_offset - h.toc_offset;
  AlignedBuf tb(toc_padded);
  pread_all(fd, tb.p, toc_padded, h.toc_offset);
  std::vector<TocEntry> toc(h.n_tensors);
  std::memcpy(toc.data(), tb.p, toc.size() * sizeof(TocEntry));
  if (crc32c(toc.data(), toc.size() * sizeof(TocEntry)) != h.toc_crc32c) die("%s: TOC CRC mismatch", file.c_str());
  for (uint64_t k = toc.size() * sizeof(TocEntry); k < toc_padded; k++)
    if (tb.p[k]) die("%s: TOC padding not zero", file.c_str());

  int bad = 0;
  uint64_t expect = h.data_offset, reserved = 0, reserve_bytes = 0;
  std::set<std::string> names;
  std::vector<size_t> data_entries;
  for (size_t i = 0; i < toc.size(); i++) {
    const TocEntry& e = toc[i];
    const std::string name(e.name, strnlen(e.name, sizeof e.name));
    if (!names.insert(name).second) std::printf("BAD %s: twice\n", name.c_str()), bad++;
    if (e.kind == uint32_t(SliceKind::Reserved)) {
      if (e.offset || e.bytes || !(e.flags & kReservedFlag)) std::printf("BAD %s: reserved with data\n", name.c_str()), bad++;
      reserved++, reserve_bytes += e.reserve_bytes;
      continue;
    }
    uint64_t elems = 1;
    for (uint32_t d = 0; d < e.ndim; d++) elems *= e.shape[d];
    if (e.ndim < 1 || e.ndim > 4 || elems * dtype_bytes(DType(e.dtype)) != e.bytes)
      std::printf("BAD %s: shape and size disagree\n", name.c_str()), bad++;
    if (e.offset != expect) std::printf("BAD %s: offset %llu, expected %llu\n", name.c_str(),
                                        (unsigned long long)e.offset, (unsigned long long)expect), bad++;
    if (e.layout == uint32_t(Layout::T64) && (e.ndim != 2 || e.shape[0] % 8 || e.shape[1] % 64))
      std::printf("BAD %s: T64 needs [N%%8][K%%64]\n", name.c_str()), bad++;
    if (e.layout == uint32_t(Layout::T128) && (e.ndim != 2 || e.shape[0] % 8 || e.shape[1] % 128))
      std::printf("BAD %s: T128 needs [N%%8][K%%128]\n", name.c_str()), bad++;
    if (e.layout == uint32_t(Layout::Int4Tile) &&
        (e.ndim != 3 || e.dtype != uint32_t(DType::U8) || e.shape[2] != 2368 || e.blk_rows != 1 || e.blk_k != 32))
      std::printf("BAD %s: an INT4 tile is U8 [N/8][K/512][2368], blk 1 x 32\n", name.c_str()), bad++;
    if (e.layout > uint32_t(Layout::Int4Tile)) std::printf("BAD %s: layout %u\n", name.c_str(), e.layout), bad++;
    expect += align_up(e.bytes);
    data_entries.push_back(i);
  }
  if (expect != fsize) std::printf("BAD: tensors end at %llu, the file at %llu\n", (unsigned long long)expect,
                                   (unsigned long long)fsize), bad++;

  std::atomic<size_t> next{0};
  std::atomic<int> crc_bad{0};
  auto worker = [&]() {
    AlignedBuf buf;
    for (size_t j; (j = next.fetch_add(1)) < data_entries.size();) {
      const TocEntry& e = toc[data_entries[j]];
      const uint64_t padded = align_up(e.bytes);
      buf.reserve(padded);
      pread_all(fd, buf.p, padded, e.offset);
      bool ok = crc32c(buf.p, e.bytes) == e.crc32c;
      for (uint64_t k = e.bytes; ok && k < padded; k++) ok = buf.p[k] == 0;
      if (!ok) std::printf("BAD %.96s: CRC or padding\n", e.name), crc_bad++;
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < 4; t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  bad += crc_bad;
  ::close(fd);
  std::printf("%s: rank %u/%u, %u entries (%llu reserved, %.2f GiB to fill), %.3f GiB, created %.32s, tool %.64s: "
              "%s (%.1f s)\n", file.c_str(), h.rank, h.world, h.n_tensors, (unsigned long long)reserved,
              reserve_bytes / 1073741824.0, fsize / 1073741824.0, h.created, h.tool, bad ? "BAD" : "CRCs, padding, geometry OK",
              now_s() - t0);
  return bad;
}

int main(int argc, char** argv) {
  std::vector<std::string> files;
  int rank = -1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--rank" && i + 1 < argc) rank = std::atoi(argv[++i]);
    else files.push_back(a);
  }
  if (files.empty()) die("usage: glw_check FILE [FILE ...] [--rank R]");
  int bad = 0;
  for (auto& f : files) bad += check(f, rank);
  return bad ? 1 : 0;
}
