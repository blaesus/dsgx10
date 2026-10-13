// Real tensors out of a .dsw rank file for the kernel benches (bench_kernels, bench_chain), read with their
// pages dropped: a rank file is ~78 GB and the box's page cache is shared.
#pragma once
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "common/check.h"
#include "weights/dsw.h"

inline void pread_drop(int fd, void* buf, size_t n, uint64_t off) {
  for (size_t got = 0; got < n;) {
    const ssize_t r = pread(fd, static_cast<char*>(buf) + got, n - got, (off_t)(off + got));
    if (r <= 0) die("short read");
    got += (size_t)r;
  }
  posix_fadvise(fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
}

// The first E experts' w1, w3 and w2 scales (L3 ue8m0, [E][N][KB]) of layer L in a .dsw file.
inline void read_expert_scales(const char* file, int L, int E, std::vector<uint8_t>& h1, std::vector<uint8_t>& h3,
                               std::vector<uint8_t>& h2) {
  const int kD = 5120, kI = 576;
  const int fd = open(file, O_RDONLY);
  if (fd < 0) die("scales: cannot open %s", file);
  dsw::Header hd;
  pread_drop(fd, &hd, sizeof hd, 0);
  if (memcmp(hd.magic, dsw::kMagic, 8)) die("scales: %s is not a .dsw file", file);
  std::vector<dsw::TocEntry> toc(hd.n_tensors);
  pread_drop(fd, toc.data(), toc.size() * sizeof(dsw::TocEntry), hd.toc_offset);
  const std::string pre = "layers." + std::to_string(L) + ".ffn.experts.";
  auto read = [&](const char* m, uint64_t N, uint64_t KB, std::vector<uint8_t>& h) {
    const std::string name = pre + m;
    for (const auto& t : toc)
      if (name == t.name) {
        if (t.dtype != (uint32_t)dsw::DType::F8_E8M0 || t.layout != (uint32_t)dsw::Layout::L3 || t.ndim != 3 ||
            t.shape[0] < (uint64_t)E || t.shape[1] != N || t.shape[2] != KB)
          die("scales: %s is not L3 ue8m0 [>= %d][%llu][%llu]", name.c_str(), E, (unsigned long long)N,
              (unsigned long long)KB);
        h.resize((size_t)E * N * KB);
        pread_drop(fd, h.data(), h.size(), t.offset);
        return;
      }
    die("scales: no %s in %s", name.c_str(), file);
  };
  read("w1.scale", kI, kD / 32, h1), read("w3.scale", kI, kD / 32, h3), read("w2.scale", kD, kI / 32, h2);
  close(fd);
}
