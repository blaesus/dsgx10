// dsw_verify: check a rank's weight file against the checkpoint, byte for byte.
//
//   dsw_verify --ckpt DIR --file FILE [--threads 8] [--crc-only]
//
// It does not reuse the converter's plan: each TOC entry's declarative description (kind, rank,
// world, source shape, stacking, segments) is turned back into source rows and columns here, so a
// slicing bug in the converter or a TOC that misdescribes its data both show up as mismatches.
// Tiled tensors are un-tiled first with layout.h, the host implementation of the README's formulas
// (the converter tiles with the kernels' own GPU functions instead). It also checks every CRC, that
// padding is zero, that every checkpoint tensor of the text model is referenced exactly once, and
// that every wo_a scale byte is <= 134.
#include <atomic>
#include <cstdio>
#include <mutex>
#include <set>
#include <thread>

#include "weights/io.h"
#include "weights/layout.h"
#include "weights/safetensors.h"

using namespace dsw;

namespace {

bool left_out(const std::string& n) {  // what the file deliberately omits
  return n.rfind("vision.", 0) == 0 || n.rfind("aligner.", 0) == 0 || n.rfind("image_", 0) == 0 ||
         n.find(".engram.embed.") != std::string::npos;
}

std::string expand(const char* tmpl, int e) {
  std::string s = tmpl;
  size_t p = s.find("{e}");
  if (p != std::string::npos) s.replace(p, 3, std::to_string(e));
  return s;
}

// Expected bytes of one stacked element (or the whole tensor) of entry `te`, read from the checkpoint.
void expected(const TocEntry& te, const SrcTensor& src, int rank, int world, int fd, uint64_t fsize,
              std::vector<uint8_t>& out, AlignedBuf& scratch) {
  if (src.shape.size() != te.src_ndim) die("%s: source rank differs from the TOC", src.name.c_str());
  for (uint32_t i = 0; i < te.src_ndim; i++)
    if (src.shape[i] != te.src_shape[i]) die("%s: source shape differs from the TOC", src.name.c_str());
  uint64_t eb = st_dtype_bytes(src.dtype), R = src.shape.empty() ? 1 : src.shape[0], rb = eb;
  for (size_t i = 1; i < src.shape.size(); i++) rb *= src.shape[i];
  std::vector<uint8_t> whole(R * rb);
  direct_read(fd, fsize, src.offset, whole.size(), whole.data(), scratch);
  out.clear();
  switch (SliceKind(te.kind)) {
    case SliceKind::Replicated:
      out = std::move(whole);
      break;
    case SliceKind::Rows: {
      uint64_t n = R / world;
      out.assign(whole.begin() + n * rank * rb, whole.begin() + (n * rank + n) * rb);
      break;
    }
    case SliceKind::Cols: {
      uint64_t c = src.shape[1] / world * eb;
      for (uint64_t r = 0; r < R; r++)
        out.insert(out.end(), whole.begin() + r * rb + c * rank, whole.begin() + r * rb + c * rank + c);
      break;
    }
    case SliceKind::RowsSeg: {
      uint64_t seg = R / te.nseg, n = seg / world;
      for (uint32_t s = 0; s < te.nseg; s++)
        out.insert(out.end(), whole.begin() + (s * seg + n * rank) * rb, whole.begin() + (s * seg + n * rank + n) * rb);
      break;
    }
    default: die("%s: unknown slice kind %u", te.name, te.kind);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt, file;
  int threads = 8;
  bool crc_only = false, subset = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--ckpt") ckpt = next();
    else if (a == "--file") file = next();
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--crc-only") crc_only = true;
    else if (a == "--subset") subset = true;  // a test file from dsw_convert --only
    else die("unknown argument %s", a.c_str());
  }
  if (file.empty() || (ckpt.empty() && !crc_only)) die("usage: dsw_verify --ckpt DIR --file FILE [--threads 8] [--crc-only]");

  double t0 = now_s();
  int fd = open_or_die(file, O_RDONLY | O_DIRECT);
  uint64_t fsize = file_size_of(fd);
  AlignedBuf hb(sizeof(Header));
  pread_all(fd, hb.p, sizeof(Header), 0);
  Header h;
  std::memcpy(&h, hb.p, sizeof h);
  if (std::memcmp(h.magic, kMagic, 8) != 0) die("%s: not a dsw file (bad magic)", file.c_str());
  if (h.version < kMinVersion || h.version > kVersion || h.header_bytes != sizeof(Header) ||
      h.toc_entry_bytes != sizeof(TocEntry) || h.alignment != kAlign)
    die("%s: unexpected version or geometry", file.c_str());
  if (h.file_bytes != fsize || h.data_offset + h.data_bytes != fsize || h.data_offset % kAlign)
    die("%s: sizes in the header do not match the file (%llu vs %llu)", file.c_str(), (unsigned long long)h.file_bytes,
        (unsigned long long)fsize);
  uint64_t toc_padded = h.data_offset - h.toc_offset;
  if (h.n_tensors * sizeof(TocEntry) > toc_padded) die("TOC does not fit before the data");
  AlignedBuf tb(toc_padded);
  pread_all(fd, tb.p, toc_padded, h.toc_offset);
  std::vector<TocEntry> toc(h.n_tensors);
  std::memcpy(toc.data(), tb.p, toc.size() * sizeof(TocEntry));
  if (crc32c(toc.data(), toc.size() * sizeof(TocEntry)) != h.toc_crc32c) die("TOC CRC mismatch");
  std::printf("%s: version %u, rank %u/%u, %u tensors, %.2f GiB, %s, from %s\n", file.c_str(), h.version, h.rank,
              h.world, h.n_tensors, fsize / 1073741824.0, h.created, h.source);

  // geometry: aligned, in order, non-overlapping, inside the data region
  uint64_t expect_off = h.data_offset;
  for (auto& e : toc) {
    if (e.offset != expect_off) die("%s: offset %llu, expected %llu", e.name, (unsigned long long)e.offset,
                                    (unsigned long long)expect_off);
    uint64_t elems = 1;
    for (uint32_t i = 0; i < e.ndim; i++) elems *= e.shape[i];
    if (elems * dtype_bytes(DType(e.dtype)) != e.bytes) die("%s: shape does not match size", e.name);
    expect_off += align_up(e.bytes);
    TileGeom g;
    std::string why;
    if (e.layout > uint32_t(Layout::L3) || (h.version < 2 && e.layout != 0)) die("%s: layout %u", e.name, e.layout);
    if (Layout(e.layout) != Layout::Natural && !tile_geom(Layout(e.layout), DType(e.dtype), e.ndim, e.shape, g, &why))
      die("%s: layout %s does not apply: %s", e.name, layout_name(Layout(e.layout)), why.c_str());
  }
  if (expect_off != fsize) die("tensors end at %llu, file at %llu", (unsigned long long)expect_off, (unsigned long long)fsize);

  Checkpoint ck;
  std::vector<int> sfd;
  if (!crc_only) {
    ck = Checkpoint::open(ckpt);
    for (auto& s : ck.shards) sfd.push_back(open_or_die(s, O_RDONLY | O_DIRECT));
    // every text-model tensor is referenced exactly once
    std::multiset<std::string> used;
    for (auto& e : toc)
      for (uint32_t k = 0; k < std::max(1u, e.nstack); k++) used.insert(expand(e.source, int(k)));
    size_t expected_n = 0;
    for (auto& [name, t] : ck.tensors) {
      if (left_out(name) || (subset && !used.count(name))) continue;
      expected_n++;
      if (used.count(name) != 1) die("%s is referenced %zu times", name.c_str(), used.count(name));
    }
    if (used.size() != expected_n) die("TOC references %zu sources, the text model has %zu", used.size(), expected_n);
    std::printf("sources: %s %zu text-model tensors referenced once (%.1f s)\n", subset ? "the" : "all", expected_n,
                now_s() - t0);
  }

  std::atomic<size_t> next{0}, bad{0}, tiled{0}, woa{0};
  std::atomic<uint64_t> checked{0};
  std::atomic<int> woa_max{0};
  std::mutex log_m;
  auto worker = [&]() {
    AlignedBuf data, scratch(64ull << 20);
    std::vector<uint8_t> exp, nat;
    for (size_t i; (i = next.fetch_add(1)) < toc.size();) {
      const TocEntry& e = toc[i];
      uint64_t padded = align_up(e.bytes);
      data.reserve(padded);
      pread_all(fd, data.p, padded, e.offset);
      bool ok = crc32c(data.p, e.bytes) == e.crc32c;
      for (uint64_t k = e.bytes; ok && k < padded; k++) ok = data.p[k] == 0;
      std::string why = ok ? "" : "CRC or padding";
      if (ok && std::string(e.name).ends_with(".attn.wo_a.scale")) {
        const int mx = *std::max_element(data.p, data.p + e.bytes);
        for (int cur = woa_max.load(); mx > cur && !woa_max.compare_exchange_weak(cur, mx);) {
        }
        woa++;
        if (mx > 134) ok = false, why = "a wo_a scale byte is above 134";
      }
      if (ok && !crc_only) {
        const uint8_t* got = data.p;  // the natural bytes
        if (Layout(e.layout) != Layout::Natural) {
          TileGeom g;
          tile_geom(Layout(e.layout), DType(e.dtype), e.ndim, e.shape, g, nullptr);
          nat.resize(e.bytes);
          retile(Layout(e.layout), DType(e.dtype), g, data.p, nat.data(), /*untile=*/true);
          got = nat.data();
          tiled++;
        }
        uint32_t ns = std::max(1u, e.nstack);
        uint64_t per = e.bytes / ns;
        for (uint32_t k = 0; k < ns && ok; k++) {
          const SrcTensor& src = ck.at(expand(e.source, int(k)));
          expected(e, src, int(h.rank), int(h.world), sfd[src.shard], ck.shard_bytes[src.shard], exp, scratch);
          if (exp.size() != per) {
            ok = false;
            why = "size";
          } else if (std::memcmp(exp.data(), got + k * per, per) != 0) {
            ok = false;
            why = "bytes differ from " + src.name;
          }
        }
      }
      uint64_t c = checked.fetch_add(e.bytes) + e.bytes;
      std::lock_guard<std::mutex> l(log_m);
      if (!ok) {
        bad++;
        std::printf("MISMATCH %s: %s\n", e.name, why.c_str());
      }
      if ((i + 1) % 200 == 0) {
        std::printf("  %zu/%zu, %.1f GiB checked, %.0f s\n", i + 1, toc.size(), c / 1073741824.0, now_s() - t0);
        std::fflush(stdout);
      }
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; t++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (woa) std::printf("wo_a: %zu scale tensors, largest scale byte %d (limit 134)\n", size_t(woa), woa_max.load());
  std::printf("%s: %zu tensors (%zu un-tiled), %zu bad, %s, %.0f s\n", file.c_str(), toc.size(), size_t(tiled),
              size_t(bad), crc_only ? "CRC and padding only" : "every byte compared with the checkpoint", now_s() - t0);
  return bad ? 1 : 0;
}
