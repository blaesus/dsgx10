// Host-side: the canonical expert tensors (as the checkpoint and the quantiser write them) and the engine's
// block layout (experts.h). The packers here are the specification the converter follows (docs/experts.md,
// "The layout the converter emits"); the readers decode a canonical tensor independently of the kernels.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

#include "experts/formats.cuh"

namespace gx {

// ---- canonical tensors --------------------------------------------------------------------------------------
// NVFP4 (model.md 5.2): w [N][K/2] (low nibble = even k), s [N][K/16] e4m3, s2 f32.
//   W(n, k) = e2m1(nibble(n, k)) * e4m3(s[n][k / 16]) * s2
// S3 (quant/formats.py levelset3): codes [N][K/16][6] (each group of 16 a little-endian 48-bit word, value j
// at bits 3j..3j+2), s [N][K/16] (bit 7: the group's subset, bits 6:0 its e4m3 scale), s2 f32, subsets
// [1 or 2][4] ascending magnitudes.  W(n, k) = (-1)^(c >> 2) * mags[s >> 7][c & 3] * e4m3(s & 0x7f) * s2
// S3x2 (the quantiser's format, docs/quantiser.md Q1; docs/experts.md section 13) is S3 with mags =
// kS3x2Mags = {0.5, 2, 4, 6}, {0.5, 1.5, 3, 6}: per group of 16 the quantiser picks the subset and the scale
// (bit 7 and bits 6:0 of the group's byte) by least squared error; s2 = f32(amax) / f32(2688) over the tensor.
// The canonical tensors are levelset3's output as is; engine/weights/s3x2_pack.py is the converter's port of
// pack_s3 and make_desc_s3x2 (tested byte for byte against these, tests/test_experts.cu "real").
inline int q4_nibble(const uint8_t* w, int K, int n, int k) { return (w[(size_t)n * (K / 2) + k / 2] >> (4 * (k & 1))) & 15; }
inline int s3_code(const uint8_t* c6, int K, int n, int k) {
  const uint8_t* p = c6 + ((size_t)n * (K / 16) + k / 16) * 6;
  uint64_t word = 0;
  for (int i = 0; i < 6; ++i) word |= (uint64_t)p[i] << (8 * i);
  return (int)((word >> (3 * (k % 16))) & 7);
}
// the value without s2 (exact in f32 and in bf16)
inline float canon_q4(const uint8_t* w, const uint8_t* s, int K, int n, int k) {
  return e2m1_to_f(q4_nibble(w, K, n, k)) * e4m3_to_f(s[(size_t)n * (K / 16) + k / 16]);
}
inline float canon_s3(const uint8_t* c6, const uint8_t* s, const float mags[2][4], int K, int n, int k) {
  const uint8_t sc = s[(size_t)n * (K / 16) + k / 16];
  const int c = s3_code(c6, K, n, k);
  return (c & 4 ? -1.f : 1.f) * mags[sc >> 7][c & 3] * e4m3_to_f(sc & 0x7F);
}

// ---- the engine layout --------------------------------------------------------------------------------------
// Block (n8, b) of [N][K] at byte ((n8 * (K / 512)) + b) * block size. Lane l = 4 g + t holds row n = 8 n8 + g;
// unit jj (0..3) of the block is k = 512 b + 128 jj .. + 127; in it the lane's 32 values are
// v[16 q + i] = (row n, k = 512 b + 128 jj + 64 q + 16 t + i), q = 0, 1, i = 0..15: two whole scale groups.
//   NVFP4: unit jj's values at byte 512 jj + 16 l: byte 8 q + i / 2, low nibble = even i (the canonical nibble);
//   S3: plane P0 at 0, P1 at 512, Q at 1024, word jj of the lane at plane + 16 l + 4 jj (s3_pack_lane of v);
//   scales at 2048 (NVFP4) / 1536 (S3): byte 8 l + 2 jj + q = the canonical scale byte of the lane's group q
//   of unit jj, i.e. s[n][(512 b + 128 jj + 64 q + 16 t) / 16].
inline void pack_nvfp4(const uint8_t* w, const uint8_t* s, int N, int K, uint8_t* out) {
  const int B = K / kBlkK;
  for (int n8 = 0; n8 < N / 8; ++n8)
    for (int b = 0; b < B; ++b) {
      uint8_t* blk = out + ((size_t)n8 * B + b) * kQ4Blk;
      for (int l = 0; l < 32; ++l) {
        const int g = l >> 2, t = l & 3, n = 8 * n8 + g;
        for (int jj = 0; jj < 4; ++jj)
          for (int q = 0; q < 2; ++q) {
            const int k0 = 512 * b + 128 * jj + 64 * q + 16 * t;
            for (int i = 0; i < 16; i += 2) {
              const int lo = q4_nibble(w, K, n, k0 + i), hi = q4_nibble(w, K, n, k0 + i + 1);
              blk[512 * jj + 16 * l + 8 * q + i / 2] = (uint8_t)(lo | hi << 4);
            }
            blk[2048 + 8 * l + 2 * jj + q] = s[(size_t)n * (K / 16) + k0 / 16];
          }
      }
    }
}
inline void pack_s3(const uint8_t* c6, const uint8_t* s, int N, int K, uint8_t* out) {
  const int B = K / kBlkK;
  for (int n8 = 0; n8 < N / 8; ++n8)
    for (int b = 0; b < B; ++b) {
      uint8_t* blk = out + ((size_t)n8 * B + b) * kS3Blk;
      for (int l = 0; l < 32; ++l) {
        const int g = l >> 2, t = l & 3, n = 8 * n8 + g;
        for (int jj = 0; jj < 4; ++jj) {
          uint8_t v[32];
          for (int q = 0; q < 2; ++q) {
            const int k0 = 512 * b + 128 * jj + 64 * q + 16 * t;
            for (int i = 0; i < 16; ++i) v[16 * q + i] = (uint8_t)s3_code(c6, K, n, k0 + i);
            blk[1536 + 8 * l + 2 * jj + q] = s[(size_t)n * (K / 16) + k0 / 16];
          }
          uint32_t P0, P1, Q;
          s3_pack_lane(v, P0, P1, Q);
          memcpy(blk + 0 + 16 * l + 4 * jj, &P0, 4);
          memcpy(blk + 512 + 16 * l + 4 * jj, &P1, 4);
          memcpy(blk + 1024 + 16 * l + 4 * jj, &Q, 4);
        }
      }
    }
}
// FP8 (shared expert): canonical e4m3 [N][K] -> [N/8][K/128][32 lanes][32 B]
inline void pack_fp8(const uint8_t* w, int N, int K, uint8_t* out) {
  const int J = K / 128;
  for (int n8 = 0; n8 < N / 8; ++n8)
    for (int j = 0; j < J; ++j)
      for (int l = 0; l < 32; ++l) {
        const int g = l >> 2, t = l & 3, n = 8 * n8 + g;
        for (int q = 0; q < 2; ++q)
          for (int i = 0; i < 16; ++i)
            out[(((size_t)n8 * J + j) * 32 + l) * 32 + 16 * q + i] = w[(size_t)n * K + 128 * j + 64 * q + 16 * t + i];
      }
}

// The engine-layout readers: value (without s2) of (n, k) read back from the packed blocks.
inline float read_q4(const uint8_t* m, int K, int n, int k) {
  const int B = K / kBlkK, n8 = n / 8, g = n % 8, b = k / 512, jj = (k / 128) % 4, q = (k / 64) % 2, t = (k / 16) % 4,
            i = k % 16, l = 4 * g + t;
  const uint8_t* blk = m + ((size_t)n8 * B + b) * kQ4Blk;
  const int nib = (blk[512 * jj + 16 * l + 8 * q + i / 2] >> (4 * (i & 1))) & 15;
  return e2m1_to_f(nib) * e4m3_to_f(blk[2048 + 8 * l + 2 * jj + q]);
}
inline float read_s3(const uint8_t* m, const float mags[2][4], int K, int n, int k) {
  const int B = K / kBlkK, n8 = n / 8, g = n % 8, b = k / 512, jj = (k / 128) % 4, q = (k / 64) % 2, t = (k / 16) % 4,
            i = k % 16, l = 4 * g + t;
  const uint8_t* blk = m + ((size_t)n8 * B + b) * kS3Blk;
  uint32_t P0, P1, Q;
  memcpy(&P0, blk + 16 * l + 4 * jj, 4);
  memcpy(&P1, blk + 512 + 16 * l + 4 * jj, 4);
  memcpy(&Q, blk + 1024 + 16 * l + 4 * jj, 4);
  const int idx = 16 * q + i, mm = idx % 8, hf = (idx / 8) % 2;
  const int low2 = ((idx < 16 ? P0 : P1) >> (4 * mm + 2 * hf)) & 3, hb = (Q >> (4 * mm + idx / 8)) & 1;
  const int c = hb << 2 | low2;
  const uint8_t sc = blk[1536 + 8 * l + 2 * jj + q];
  return (c & 4 ? -1.f : 1.f) * mags[sc >> 7][c & 3] * e4m3_to_f(sc & 0x7F);
}
inline float read_fp8(const uint8_t* m, int K, int n, int k) {
  const int J = K / 128, n8 = n / 8, g = n % 8, j = k / 128, q = (k / 64) % 2, t = (k / 16) % 4, i = k % 16;
  return e4m3_to_f(m[(((size_t)n8 * J + j) * 32 + 4 * g + t) * 32 + 16 * q + i]);
}

// A MatDesc for a packed matrix.
inline MatDesc make_desc(uint64_t off, float s2, uint32_t fmt, int nsub = 0, const float mags[2][4] = nullptr) {
  MatDesc d;
  memset(&d, 0, sizeof d);
  d.off = off;
  d.s2 = s2;
  d.fmt = fmt;
  if (fmt == kS3)
    for (int s = 0; s < 2; ++s) {
      const S3Tab t = s3_table(mags[s < nsub ? s : 0]);
      d.tab[4 * s] = t.tl0, d.tab[4 * s + 1] = t.tl1, d.tab[4 * s + 2] = t.th0, d.tab[4 * s + 3] = t.th1;
      d.nib[s] = s3_nibble_table(mags[s < nsub ? s : 0]);
    }
  return d;
}
// S3x2 (experts.h kS3x2Mags): the quantiser's two subsets, A for scale bytes with bit 7 clear, B with it set.
inline MatDesc make_desc_s3x2(uint64_t off, float s2) { return make_desc(off, s2, kS3, 2, kS3x2Mags); }

// The loader's check of an S3 descriptor (glw_kcheck): reads the two subsets back from the prmt tables and
// accepts the descriptor only if each is four ascending E2M1 magnitudes and tab and nib are exactly what
// make_desc builds from them (so the bf16 path and the FP4 path decode the same values). Returns the
// subsets in mags; *pair_is_s3x2 tells whether they are kS3x2Mags.
inline bool s3_desc_check(const MatDesc& d, float mags[2][4], bool* pair_is_s3x2 = nullptr) {
  if (d.fmt != kS3) return false;
  static const float grid[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int s = 0; s < 2; ++s)
    for (int j = 0; j < 4; ++j) {
      const uint32_t lo = (d.tab[4 * s] >> 8 * j) & 0xFF, hi = (d.tab[4 * s + 2] >> 8 * j) & 0xFF;
      mags[s][j] = bf16_bits_to_f((uint16_t)(hi << 8 | lo));
      bool on_grid = false;
      for (float g : grid) on_grid |= mags[s][j] == g;
      if (!on_grid || (j && !(mags[s][j] > mags[s][j - 1]))) return false;
    }
  const MatDesc r = make_desc(d.off, d.s2, kS3, 2, mags);
  if (memcmp(r.tab, d.tab, sizeof d.tab) || memcmp(r.nib, d.nib, sizeof d.nib)) return false;
  if (pair_is_s3x2) *pair_is_s3x2 = !memcmp(mags, kS3x2Mags, sizeof(float) * 8);
  return true;
}
// A matrix's scale bytes as stored in its blocks (blocks of size bs, scales from byte so): none may be the e4m3
// NaN (low 7 bits 0x7F). NVFP4's must be positive (the FP4 tensor-core path reads them as ue4m3); in S3, bit 7
// is the subset, so a matrix with one subset (tab halves equal) may still set it. Returns the number of bad bytes,
// and counts the groups of subset B in *nb (S3).
inline size_t scale_bytes_check(const uint8_t* m, uint32_t fmt, int N, int K, size_t* nb = nullptr) {
  const int bs = fmt == kNVFP4 ? kQ4Blk : kS3Blk, so = fmt == kNVFP4 ? 2048 : 1536;
  const size_t blocks = (size_t)(N / 8) * (K / kBlkK);
  size_t bad = 0, b = 0;
  for (size_t i = 0; i < blocks; ++i)
    for (int j = 0; j < 256; ++j) {
      const uint8_t s = m[i * bs + so + j];
      bad += (s & 0x7F) == 0x7F || (fmt == kNVFP4 && (s & 0x80));
      b += s >> 7;
    }
  if (nb) *nb = fmt == kS3 ? b : 0;
  return bad;
}

}  // namespace gx
