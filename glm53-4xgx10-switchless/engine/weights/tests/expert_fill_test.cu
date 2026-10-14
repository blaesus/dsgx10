// glw_expert_fill_test: a filled routed-expert pool against the canonical NVFP4 or S3 (S3x2) tensors it came from, with the
// expert kernels' own host readers (engine/experts/layout.h: read_q4 / read_s3 on the packed blocks, canon_q4 / canon_s3 on the
// canonical tensors). Host only, no device.
//
//   glw_expert_fill_test FILE RANK DUMP_DIR LAYER E [E ...]
// Each expert's matrices are read through its MatDesc (offsets relative to expert 0's entry, the pool base).
// DUMP_DIR holds experts_fill.py --dump-canon's raw slices L<layer>.r<rank>.e<E>.{gate,up,down}.{w,s}.
#include <cstdio>
#include <string>
#include <vector>

#include "common/check.h"
#include "experts/layout.h"
#include "weights/loader.h"

using namespace glw;

static std::vector<uint8_t> slurp(const std::string& p) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (!f) die("%s: cannot open", p.c_str());
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) v.insert(v.end(), buf, buf + n);
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 6) die("usage: glw_expert_fill_test FILE RANK DUMP_DIR LAYER E [E ...]");
  const int rank = std::atoi(argv[2]), L = std::atoi(argv[4]);
  const std::string dir = argv[3], p = "layers." + std::to_string(L) + ".experts.";
  WeightFile wf(argv[1], rank, 4);
  const DeviceTensor* dt = wf.find(p + "desc");
  const DeviceTensor* e0 = wf.find(p + "0");
  if (!dt || !e0 || dt->reserved() || e0->reserved()) die("%s: layer %d's experts are not filled", argv[1], L);
  const std::vector<uint8_t> dsc = wf.read_host(*dt);
  const auto* desc = reinterpret_cast<const gx::MatDesc*>(dsc.data());
  const uint64_t base = e0->file_offset;  // the layer's pool starts at expert 0's entry
  long bad = 0, checked = 0;
  for (int a = 5; a < argc; a++) {
    const int e = std::atoi(argv[a]);
    const DeviceTensor* et = wf.find(p + std::to_string(e));
    if (!et || et->reserved()) die("expert %d: not filled", e);
    const std::vector<uint8_t> ent = wf.read_host(*et);
    const uint64_t eoff = et->file_offset - base;  // the entry's offset in the pool
    const char* proj[3] = {"gate", "up", "down"};
    for (int m = 0; m < 3; m++) {
      const int N = m < 2 ? gx::kI : gx::kH, K = m < 2 ? gx::kH : gx::kI;
      const std::string base = dir + "/L" + std::to_string(L) + ".r" + std::to_string(rank) + ".e" + std::to_string(e) + "." + proj[m];
      const std::vector<uint8_t> w = slurp(base + ".w"), s = slurp(base + ".s");
      const gx::MatDesc& d = desc[3 * e + m];
      // NVFP4: w [N][K/2] nibbles; S3 (S3x2, s3x2_pack.py): w the codes [N][K/16][6], the subsets from the descriptor
      const bool s3 = d.fmt == gx::kS3;
      float mags[2][4];
      if (w.size() != (s3 ? size_t(N) * K / 16 * 6 : size_t(N) * K / 2) || s.size() != size_t(N) * K / 16)
        die("%s: canonical sizes", base.c_str());
      if ((d.fmt != gx::kNVFP4 && !s3) || expert_matrix_fmt(et->fmt, m) != d.fmt || d.off % 256 || d.off < eoff ||
          d.off - eoff + gx::mat_bytes(d.fmt, N, K) > ent.size() || !(d.s2 > 0) || !std::isfinite(d.s2) ||
          (s3 && !gx::s3_desc_check(d, mags)))
        die("expert %d %s: bad descriptor (off %llu, fmt %u, s2 %g)", e, proj[m], (unsigned long long)d.off, d.fmt, d.s2);
      long b = 0;
      const uint8_t* pm = ent.data() + (d.off - eoff);
      for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
          const float got = s3 ? gx::read_s3(pm, mags, K, n, k) : gx::read_q4(pm, K, n, k);
          const float want = s3 ? gx::canon_s3(w.data(), s.data(), mags, K, n, k) : gx::canon_q4(w.data(), s.data(), K, n, k);
          b += std::memcmp(&got, &want, 4) != 0;
        }
      std::printf("layer %d expert %3d %-4s [%d][%d] off %10llu s2 %.6g: %ld of %d values differ\n", L, e, proj[m], N,
                  K, (unsigned long long)d.off, d.s2, b, N * K);
      bad += b;
      checked += long(N) * K;
    }
  }
  std::printf("%s: %ld values checked, %ld differ\n", bad ? "FAIL" : "PASS", checked, bad);
  return bad ? 1 : 0;
}
