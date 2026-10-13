// Host-side reference of the tiled layouts (dsw.h Layout, engine/kernels/README.md "Weight layouts"),
// written from the README's formulas and independent of the kernels' dsk::tile_fp8 / dsk::tile_fp4,
// which dsw_convert uses. dsw_verify un-tiles with these; dsw_kcheck checks them against the kernels.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>

#include "weights/dsw.h"

namespace dsw {

// A tiled tensor as the tiling functions see it: `rows` rows over the whole stack, each matrix of the
// stack `mat_rows` of them; `k` is the row length in bytes for L1/L2 and the logical K (elements) for
// L3, whose values have K/2 bytes per row and whose scales K/32.
struct TileGeom {
  uint64_t rows = 0, k = 0, mat_rows = 0;
};

// The geometry of a tensor of dtype `dt` and natural shape `shape` in layout `l`, or false (with the
// reason in `why`) when the layout does not apply or the shape does not fit its tiles.
inline bool tile_geom(Layout l, DType dt, uint32_t ndim, const uint64_t* shape, TileGeom& g, std::string* why) {
  auto fail = [&](const char* m) {
    if (why) *why = m;
    return false;
  };
  if (ndim < 2 || ndim > 4) return fail("needs 2 to 4 dims");
  uint64_t lead = 1;
  for (uint32_t i = 0; i + 1 < ndim; i++) lead *= shape[i];
  const uint64_t last = shape[ndim - 1];
  switch (l) {
    case Layout::L1:
      if (dt != DType::F8_E4M3 || ndim != 2) return fail("L1 is for 2-d e4m3 weights");
      g = {shape[0], last, shape[0]};
      if (g.rows % 8 || g.k % 64) return fail("L1 needs N % 8 == 0 and K % 64 == 0");
      return true;
    case Layout::L2:
      if (dt != DType::BF16 || ndim != 2) return fail("L2 is for 2-d bf16 weights");
      g = {shape[0], 2 * last, shape[0]};
      if (g.rows % 8 || g.k % 64) return fail("L2 needs N % 8 == 0 and 2K % 64 == 0");
      return true;
    case Layout::L3:
      if (dt == DType::E2M1X2) g = {lead, 2 * last, shape[ndim - 2]};
      else if (dt == DType::F8_E8M0) g = {lead, 32 * last, shape[ndim - 2]};
      else return fail("L3 is for packed e2m1 values and their ue8m0 scales");
      if (g.mat_rows % 16 || g.k % 64) return fail("L3 needs N % 16 == 0 and K % 64 == 0");
      return true;
    case Layout::Natural: return fail("natural is not tiled");
  }
  return fail("unknown layout");
}

// L1 (L2 is L1 on bf16 rows as bytes): natural [N][K] bytes <-> [N/8][K/64][8 rows][64 B].
inline uint64_t l1_offset(uint64_t n, uint64_t k, uint64_t K) {
  return ((n / 8) * (K / 64) + k / 64) * 512 + (n % 8) * 64 + k % 64;
}
inline void l1_tile(const uint8_t* nat, uint8_t* til, uint64_t N, uint64_t K) {
  for (uint64_t n = 0; n < N; n++)
    for (uint64_t k = 0; k < K; k += 64) std::memcpy(til + l1_offset(n, k, K), nat + n * K + k, 64);
}
inline void l1_untile(const uint8_t* til, uint8_t* nat, uint64_t N, uint64_t K) {
  for (uint64_t n = 0; n < N; n++)
    for (uint64_t k = 0; k < K; k += 64) std::memcpy(nat + n * K + k, til + l1_offset(n, k, K), 64);
}

// L3 values of one matrix: natural [N][K/2] bytes. Row n = 16*n16 + 8*h + g; the 4 bytes at k-byte
// 32p + 16l + 4t of row n sit at ((n16*P + p)*32 + 4g + t)*16 + 4j with j = 2h + l.
inline uint64_t l3_value_offset(uint64_t n, uint64_t p, uint64_t l, uint64_t t, uint64_t P) {
  const uint64_t n16 = n / 16, h = n % 16 / 8, g = n % 8;
  return ((n16 * P + p) * 32 + 4 * g + t) * 16 + 4 * (2 * h + l);
}
// L3 scales of one matrix: natural [N][K/32]; byte [n][2p + l] sits at ((n16*P + p)*8 + g)*4 + j.
inline uint64_t l3_scale_offset(uint64_t n, uint64_t p, uint64_t l, uint64_t P) {
  const uint64_t n16 = n / 16, h = n % 16 / 8, g = n % 8;
  return ((n16 * P + p) * 8 + g) * 4 + 2 * h + l;
}

// One L3 matrix (values: scales = false), natural <-> tiled; `untile` picks the direction.
inline void l3_matrix(const uint8_t* src, uint8_t* dst, uint64_t N, uint64_t K, bool scales, bool untile) {
  const uint64_t P = K / 64;
  if (scales) {
    const uint64_t rb = K / 32;
    for (uint64_t n = 0; n < N; n++)
      for (uint64_t p = 0; p < P; p++)
        for (uint64_t l = 0; l < 2; l++) {
          const uint64_t nat = n * rb + 2 * p + l, til = l3_scale_offset(n, p, l, P);
          if (untile) dst[nat] = src[til];
          else dst[til] = src[nat];
        }
    return;
  }
  const uint64_t rb = K / 2;
  for (uint64_t n = 0; n < N; n++)
    for (uint64_t p = 0; p < P; p++)
      for (uint64_t l = 0; l < 2; l++)
        for (uint64_t t = 0; t < 4; t++) {
          const uint64_t nat = n * rb + 32 * p + 16 * l + 4 * t, til = l3_value_offset(n, p, l, t, P);
          if (untile) std::memcpy(dst + nat, src + til, 4);
          else std::memcpy(dst + til, src + nat, 4);
        }
}

// Natural <-> tiled for a whole tensor of geometry g (a stack is tiled matrix by matrix).
inline void retile(Layout l, DType dt, const TileGeom& g, const uint8_t* src, uint8_t* dst, bool untile) {
  if (l == Layout::L1 || l == Layout::L2) {
    if (untile) l1_untile(src, dst, g.rows, g.k);
    else l1_tile(src, dst, g.rows, g.k);
    return;
  }
  const bool scales = dt == DType::F8_E8M0;
  const uint64_t per = g.mat_rows * (scales ? g.k / 32 : g.k / 2);
  for (uint64_t e = 0; e < g.rows / g.mat_rows; e++)
    l3_matrix(src + e * per, dst + e * per, g.mat_rows, g.k, scales, untile);
}

}  // namespace dsw
