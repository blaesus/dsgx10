// Host-side: the canonical expert tensors (as the checkpoint stores them) and the engine's block layout
// (experts.h). The packers here are the specification the converter follows (docs/experts.md section 2;
// engine/weights/int4_pack.py is its Python port, tested byte for byte against these); the readers decode a
// packed matrix independently of the kernels, and canon_* read the checkpoint's own tensors.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "experts/formats.cuh"

namespace gx {

[[noreturn]] inline void layout_die(const char* what, uint32_t fmt) {
  fprintf(stderr, "experts/layout.h: %s: expert format %u is not NVFP4 (0), INT4 (2) or BF16 (3)\n", what, fmt);
  abort();
}

// ---- canonical tensors --------------------------------------------------------------------------------------
// NVFP4 (model.md 5.2): w [N][K/2] (low nibble = even k), s [N][K/16] e4m3, s2 f32.
//   W(n, k) = e2m1(nibble(n, k)) * e4m3(s[n][k / 16]) * s2
// INT4 (compressed-tensors pack-quantized, group 32 along K, asymmetric; altar-1's layers 4-76):
//   wp = weight_packed I32 [N][K/8] read as bytes [N][K/2] (little-endian: value k = 8 j + i at bits 4 i of word
//        j, so byte k / 2, low nibble = even k: NVFP4's byte order), the stored nibble u = q + 8;
//   s  = weight_scale BF16 [N][K/32];
//   zp = weight_zero_point I32 [N/8][K/32]: the zero point of row n, group c at bits 4 (n % 8) of word
//        (n / 8, c), the stored nibble u_z = z + 8;
//   W(n, k) = bf16((u - u_z) * s[n][k / 32]), the product rounded once to nearest even (i4_value_bits): what
//   compressed-tensors' dequantize computes (docs/findings.md A0).
// BF16 (layers 3, 77 and the MTP layer 78): w [N][K] bf16 bits, W(n, k) = w[n][k].
inline int q4_nibble(const uint8_t* w, int K, int n, int k) { return (w[(size_t)n * (K / 2) + k / 2] >> (4 * (k & 1))) & 15; }
inline int i4_zp(const uint32_t* zp, int K, int n, int c) { return (zp[(size_t)(n / 8) * (K / 32) + c] >> (4 * (n % 8))) & 15; }
// the value without s2 (exact in f32 and in bf16)
inline float canon_q4(const uint8_t* w, const uint8_t* s, int K, int n, int k) {
  return e2m1_to_f(q4_nibble(w, K, n, k)) * e4m3_to_f(s[(size_t)n * (K / 16) + k / 16]);
}
// the contract's weight as bf16 bits
inline uint16_t canon_i4(const uint8_t* wp, const uint16_t* s, const uint32_t* zp, int K, int n, int k) {
  return i4_value_bits(q4_nibble(wp, K, n, k), i4_zp(zp, K, n, k / 32), s[(size_t)n * (K / 32) + k / 32]);
}

// ---- the engine layout --------------------------------------------------------------------------------------
// Block (n8, b) of [N][K] at byte ((n8 * (K / kBK)) + b) * block size. Lane l = 4 g + t holds row n = 8 n8 + g;
// unit jj of the block is k = kBK b + 128 jj .. + 127; in it the lane's 32 values are
// v[16 q + i] = (row n, k = kBK b + 128 jj + 64 q + 16 t + i), q = 0, 1, i = 0..15.
//   NVFP4 (kBK 512, 2304 B): unit jj's values at byte 512 jj + 16 l: byte 8 q + i / 2, low nibble = even i (the
//     canonical nibble); scales at 2048 + 8 l + 2 jj + q = s[n][(512 b + 128 jj + 64 q + 16 t) / 16].
//   INT4 (kBK 512, 2368 B): the stored value nibbles exactly as NVFP4's (the checkpoint's bytes of the lane's 16
//     values: byte 8 q + i / 2 of the lane's 16 B of unit jj); the lane's 16 values lie in group
//     c = 16 b + 4 jj + 2 q + (t >> 1), whose
//       scale (bf16, 2 B) is at 2048 + 32 g + 16 (t >> 1) + 2 (2 jj + q)       [8 rows x 2 halves x 8 groups]
//       zero point (nibble) is nibble 2 jj + q of the u32 at 2304 + 8 g + 4 (t >> 1), low nibble first
//                                                                              [8 rows x 2 halves x 8 nibbles]
//     so a block's 16 groups of a row c' = 0..15 (k 32 c' ..) sit at h = c' & 1, (2 jj + q) = c' >> 1.
//   BF16 (kBK 128, 2048 B): the lane's 16 B at 512 p + 16 l, p = 0..3, are its values 8 p .. 8 p + 7 (bf16, the
//     checkpoint's bits): k = 128 b + 64 (p >> 1) + 16 t + 8 (p & 1) + 0..7.
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
inline void pack_int4(const uint8_t* wp, const uint16_t* s, const uint32_t* zp, int N, int K, uint8_t* out) {
  const int B = K / kBlkK;
  for (int n8 = 0; n8 < N / 8; ++n8)
    for (int b = 0; b < B; ++b) {
      uint8_t* blk = out + ((size_t)n8 * B + b) * kI4Blk;
      memset(blk + 2304, 0, 64);
      for (int l = 0; l < 32; ++l) {
        const int g = l >> 2, t = l & 3, h = t >> 1, n = 8 * n8 + g;
        for (int jj = 0; jj < 4; ++jj)
          for (int q = 0; q < 2; ++q) {
            const int k0 = 512 * b + 128 * jj + 64 * q + 16 * t, c = k0 / 32, j = 2 * jj + q;
            memcpy(blk + 512 * jj + 16 * l + 8 * q, wp + (size_t)n * (K / 2) + k0 / 2, 8);  // the stored bytes
            memcpy(blk + 2048 + 32 * g + 16 * h + 2 * j, s + (size_t)n * (K / 32) + c, 2);
            if (t == 2 * h) blk[2304 + 8 * g + 4 * h + j / 2] |= (uint8_t)(i4_zp(zp, K, n, c) << (4 * (j & 1)));
          }
      }
    }
}
inline void pack_bf16(const uint16_t* w, int N, int K, uint8_t* out) {
  const int B = K / kBlkKBF16;
  for (int n8 = 0; n8 < N / 8; ++n8)
    for (int b = 0; b < B; ++b) {
      uint8_t* blk = out + ((size_t)n8 * B + b) * kB16Blk;
      for (int l = 0; l < 32; ++l) {
        const int g = l >> 2, t = l & 3, n = 8 * n8 + g;
        for (int p = 0; p < 4; ++p)
          memcpy(blk + 512 * p + 16 * l, w + (size_t)n * K + 128 * b + 64 * (p >> 1) + 16 * t + 8 * (p & 1), 16);
      }
    }
}
// FP8 (a fused shared expert): canonical e4m3 [N][K] -> [N/8][K/128][32 lanes][32 B]
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

// The engine-layout readers: the value of (n, k) read back from the packed blocks by the formulas above.
inline float read_q4(const uint8_t* m, int K, int n, int k) {  // without s2
  const int B = K / kBlkK, n8 = n / 8, g = n % 8, b = k / 512, jj = (k / 128) % 4, q = (k / 64) % 2, t = (k / 16) % 4,
            i = k % 16, l = 4 * g + t;
  const uint8_t* blk = m + ((size_t)n8 * B + b) * kQ4Blk;
  const int nib = (blk[512 * jj + 16 * l + 8 * q + i / 2] >> (4 * (i & 1))) & 15;
  return e2m1_to_f(nib) * e4m3_to_f(blk[2048 + 8 * l + 2 * jj + q]);
}
struct I4Parts { int u, uz; uint16_t s; };  // the stored nibble, zero-point nibble and scale of (n, k)
inline I4Parts read_i4_parts(const uint8_t* m, int K, int n, int k) {
  const int B = K / kBlkK, n8 = n / 8, g = n % 8, b = k / 512, jj = (k / 128) % 4, q = (k / 64) % 2, t = (k / 16) % 4,
            i = k % 16, l = 4 * g + t, h = t >> 1, j = 2 * jj + q;
  const uint8_t* blk = m + ((size_t)n8 * B + b) * kI4Blk;
  I4Parts p;
  p.u = (blk[512 * jj + 16 * l + 8 * q + i / 2] >> (4 * (i & 1))) & 15;
  memcpy(&p.s, blk + 2048 + 32 * g + 16 * h + 2 * j, 2);
  p.uz = (blk[2304 + 8 * g + 4 * h + j / 2] >> (4 * (j & 1))) & 15;
  return p;
}
inline uint16_t read_i4(const uint8_t* m, int K, int n, int k) {  // bf16 bits of the contract's weight
  const I4Parts p = read_i4_parts(m, K, n, k);
  return i4_value_bits(p.u, p.uz, p.s);
}
inline uint16_t read_b16(const uint8_t* m, int K, int n, int k) {
  const int B = K / kBlkKBF16, n8 = n / 8, g = n % 8, b = k / 128, kk = k % 128, q = kk / 64, t = (kk % 64) / 16,
            r = kk % 16, p = 2 * q + r / 8, l = 4 * g + t;
  uint16_t v;
  memcpy(&v, m + ((size_t)n8 * B + b) * kB16Blk + 512 * p + 16 * l + 2 * (r % 8), 2);
  return v;
}
inline float read_fp8(const uint8_t* m, int K, int n, int k) {
  const int J = K / 128, n8 = n / 8, g = n % 8, j = k / 128, q = (k / 64) % 2, t = (k / 16) % 4, i = k % 16;
  return e4m3_to_f(m[(((size_t)n8 * J + j) * 32 + 4 * g + t) * 32 + 16 * q + i]);
}
// The value of (n, k) of a packed matrix of any format as f32, without s2 (exact: every format's value is a bf16)
inline float read_value(const uint8_t* m, uint32_t fmt, int K, int n, int k) {
  switch (fmt) {
    case kNVFP4: return read_q4(m, K, n, k);
    case kINT4: return bf16_bits_to_f(read_i4(m, K, n, k));
    case kBF16: return bf16_bits_to_f(read_b16(m, K, n, k));
  }
  layout_die("read_value", fmt);
}

// A MatDesc for a packed matrix (s2: NVFP4's weight_scale_2; INT4 and BF16 have none, 1.0).
inline MatDesc make_desc(uint64_t off, float s2, uint32_t fmt) {
  if (!fmt_known(fmt)) layout_die("make_desc", fmt);
  MatDesc d;
  memset(&d, 0, sizeof d);
  d.off = off;
  d.s2 = fmt == kNVFP4 ? s2 : 1.f;
  d.fmt = fmt;
  return d;
}
// The loader's check of a descriptor of a layer of format fmt (glw_kcheck): the format, the offset's alignment,
// s2 (positive and finite; exactly 1 for INT4 and BF16), the reserved words zero.
inline bool desc_ok(const MatDesc& d, uint32_t fmt) {
  if (d.fmt != fmt || !fmt_known(fmt) || d.off % 256) return false;
  if (!(d.s2 > 0.f) || !std::isfinite(d.s2) || (fmt != kNVFP4 && d.s2 != 1.f)) return false;
  for (uint32_t r : d.rsv)
    if (r) return false;
  return true;
}
// A matrix's scale fields as stored in its blocks: NVFP4's e4m3 bytes may not be the NaN (low 7 bits 0x7F) and must
// be positive (the FP4 tensor-core path reads them as ue4m3); INT4's bf16 scales must be finite and positive (the
// checkpoint's all are: docs/findings.md A0). BF16 has none. Returns the number of bad fields.
inline size_t scale_bytes_check(const uint8_t* m, uint32_t fmt, int N, int K) {
  if (!fmt_known(fmt)) layout_die("scale_bytes_check", fmt);
  if (fmt == kBF16) return 0;
  const size_t blocks = (size_t)(N / 8) * (K / kBlkK);
  size_t bad = 0;
  for (size_t i = 0; i < blocks; ++i) {
    if (fmt == kNVFP4) {
      for (int j = 0; j < 256; ++j) {
        const uint8_t s = m[i * kQ4Blk + 2048 + j];
        bad += (s & 0x7F) == 0x7F || (s & 0x80);
      }
    } else {
      for (int j = 0; j < 128; ++j) {
        uint16_t s;
        memcpy(&s, m + i * kI4Blk + 2048 + 2 * j, 2);
        const float f = bf16_bits_to_f(s);
        bad += !(f > 0.f) || !std::isfinite(f);
      }
    }
  }
  return bad;
}

}  // namespace gx
