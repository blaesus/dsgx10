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

}  // namespace gx
