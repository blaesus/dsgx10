// glw_expert_fill_test: a filled routed-expert pool against the canonical INT4 or BF16 tensors it came from, with the
// expert kernels' own host readers (engine/experts/layout.h: read_i4 / read_b16 on the packed blocks, canon_i4 or
// the BF16 bits on the canonical slices). Host only, no device.
//
//   glw_expert_fill_test FILE RANK DUMP_DIR LAYER E [E ...]
// Each expert's matrices are read through its MatDesc (offsets relative to expert 0's entry, the pool base).
// DUMP_DIR holds engine/weights/int4_pack.py dump's canonical rank slices l<layer>e<E>r<rank>m<m>.{wp,s,zp} (INT4)
// or .w (BF16), m = 0 gate, 1 up, 2 down.
#include <cmath>
#include <cstdio>
#include <cstring>
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
      const std::string cb = dir + "/l" + std::to_string(L) + "e" + std::to_string(e) + "r" + std::to_string(rank) + "m" + std::to_string(m);
      const gx::MatDesc& d = desc[3 * e + m];
      if ((d.fmt != gx::kINT4 && d.fmt != gx::kBF16) || expert_matrix_fmt(et->fmt, m) != d.fmt || !gx::desc_ok(d, d.fmt) ||
          d.off < eoff || d.off - eoff + gx::mat_bytes(d.fmt, N, K) > ent.size())
        die("expert %d %s: bad descriptor (off %llu, fmt %u, s2 %g)", e, proj[m], (unsigned long long)d.off, d.fmt, d.s2);
      std::vector<uint8_t> w, s, z;
      if (d.fmt == gx::kINT4) {
        w = slurp(cb + ".wp"), s = slurp(cb + ".s"), z = slurp(cb + ".zp");
        if (w.size() != size_t(N) * K / 2 || s.size() != size_t(N) * K / 32 * 2 || z.size() != size_t(N) / 8 * K / 32 * 4)
          die("%s: canonical INT4 sizes", cb.c_str());
      } else {
        w = slurp(cb + ".w");
        if (w.size() != size_t(N) * K * 2) die("%s: canonical BF16 size", cb.c_str());
      }
      long b = 0;
      const uint8_t* pm = ent.data() + (d.off - eoff);
      for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
          uint16_t got, want;
          if (d.fmt == gx::kINT4) {
            got = gx::read_i4(pm, K, n, k);
            want = gx::canon_i4(w.data(), reinterpret_cast<const uint16_t*>(s.data()), reinterpret_cast<const uint32_t*>(z.data()), K, n, k);
          } else {
            got = gx::read_b16(pm, K, n, k);
            std::memcpy(&want, w.data() + 2 * (size_t(n) * K + k), 2);
          }
          b += got != want;
        }
      std::printf("layer %d expert %3d %-4s %s [%d][%d] off %10llu: %ld of %d values differ\n", L, e, proj[m],
                  gx::fmt_name(d.fmt), N, K, (unsigned long long)d.off, b, N * K);
      bad += b;
      checked += long(N) * K;
    }
  }
  std::printf("%s: %ld values checked, %ld differ\n", bad ? "FAIL" : "PASS", checked, bad);
  return bad ? 1 : 0;
}
