// The per-rank weight file (".dsw"): one rank's slice of every tensor the text model uses, in the
// checkpoint's own numeric formats, with the GEMV weights pre-tiled for the decode kernels, laid out
// so a loader can stream it into one device arena with large O_DIRECT reads. Layout and partition:
// docs/layout.md.
//
//   [0, 4096)                 Header
//   [toc_offset, ...)         n_tensors TocEntry records (512 B each)
//   [data_offset, file_bytes) tensors, each starting on a 4 KiB boundary, zero-padded to 4 KiB
//
// Device address of a tensor = arena + (entry.offset - header.data_offset).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace dsw {

constexpr char kMagic[8] = {'D', 'S', 'G', 'B', '1', '0', 'W', '1'};
constexpr uint32_t kVersion = 2;     // written by dsw_convert
constexpr uint32_t kMinVersion = 1;  // oldest the loader and verifier read (version 1 is all Natural)
constexpr uint64_t kAlign = 4096;

inline uint64_t align_up(uint64_t x, uint64_t a = kAlign) { return (x + a - 1) / a * a; }

// Values are the checkpoint's; nothing is converted.
enum class DType : uint32_t {
  F32 = 1,
  BF16 = 2,
  F8_E4M3 = 3,  // e4m3fn
  F8_E8M0 = 4,  // ue8m0 scale: 2^(byte - 127)
  E2M1X2 = 5,   // packed fp4 e2m1, two per byte, low nibble first; the last dim of `shape` counts bytes
};

inline uint32_t dtype_bytes(DType t) { return t == DType::F32 ? 4 : t == DType::BF16 ? 2 : 1; }
inline const char* dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "F32";
    case DType::BF16: return "BF16";
    case DType::F8_E4M3: return "F8_E4M3";
    case DType::F8_E8M0: return "F8_E8M0";
    case DType::E2M1X2: return "E2M1X2";
  }
  return "?";
}

// How a rank's tensor was cut from its source tensor(s), each viewed as [rows, row_bytes].
enum class SliceKind : uint32_t {
  Replicated = 1,  // the whole tensor
  Rows = 2,        // rows [R/world * rank, +R/world)
  Cols = 3,        // columns [C/world * rank, +C/world)
  RowsSeg = 4,     // dim 0 is `nseg` equal segments; each segment's rows are split as in Rows
};

inline const char* kind_name(SliceKind k) {
  switch (k) {
    case SliceKind::Replicated: return "replicated";
    case SliceKind::Rows: return "rows";
    case SliceKind::Cols: return "cols";
    case SliceKind::RowsSeg: return "rows-seg";
  }
  return "?";
}

// How a tensor's bytes are arranged (TocEntry::layout). `shape` always stays the natural, logical
// shape. The tiled layouts are the ones engine/kernels reads (engine/kernels/README.md, "Weight
// layouts"); dsw_convert produces them with the kernels' own dsk::tile_fp8 / dsk::tile_fp4.
enum class Layout : uint32_t {
  Natural = 0,  // row-major, as in the checkpoint
  // FP8 e4m3 [N][K], N % 8 == 0, K % 64 == 0: [N/8][K/64][8 rows][64 B], i.e. byte (n, k) at
  // ((n/8)*(K/64) + k/64)*512 + (n%8)*64 + k%64. Its ue8m0 [N/32][K/32] scales stay Natural.
  L1 = 1,
  // bf16 [N][K] viewed as bytes [N][2K], then L1 with K' = 2K.
  L2 = 2,
  // MXFP4 (tile_fp4), applied to each matrix of a stack ([E][N][..]; matrices are tiled one after
  // another, so expert e starts at e times its size). N % 16 == 0, K % 64 == 0, P = K/64,
  // n = 16*n16 + 8*h + g, k = 64*p + 32*l + 8*t + i, j = 2*h + l:
  //   values  E2M1X2 [N][K/2]: the 4 bytes [n][32p + 16l + 4t, +4) go to ((n16*P + p)*32 + 4g + t)*16 + 4j;
  //   scales F8_E8M0 [N][K/32]: byte [n][2p + l] goes to ((n16*P + p)*8 + g)*4 + j.
  L3 = 3,
};

inline const char* layout_name(Layout l) {
  switch (l) {
    case Layout::Natural: return "natural";
    case Layout::L1: return "L1";
    case Layout::L2: return "L2";
    case Layout::L3: return "L3";
  }
  return "?";
}

struct Header {
  char magic[8];
  uint32_t version;
  uint32_t header_bytes;  // 4096
  uint32_t rank;
  uint32_t world;
  uint32_t n_tensors;
  uint32_t toc_entry_bytes;  // sizeof(TocEntry)
  uint64_t toc_offset;
  uint64_t data_offset;
  uint64_t data_bytes;
  uint64_t file_bytes;
  uint32_t alignment;
  uint32_t toc_crc32c;  // CRC32C of the whole TOC
  char model[64];
  char source[256];  // checkpoint directory the file was cut from
  char created[32];  // UTC, ISO 8601
  uint8_t reserved[4096 - 8 - 6 * 4 - 4 * 8 - 2 * 4 - 64 - 256 - 32];
};
static_assert(sizeof(Header) == 4096);

struct TocEntry {
  char name[128];    // stacked experts drop the expert index: layers.7.ffn.experts.w1.weight
  char source[128];  // source tensor; "{e}" stands for the expert index when stacked
  char slice[96];    // human-readable description of the cut
  uint32_t dtype;    // DType
  uint32_t ndim;
  uint64_t shape[4];
  uint64_t offset;  // absolute file offset, kAlign-aligned
  uint64_t bytes;   // unpadded
  uint32_t kind;    // SliceKind
  uint32_t nstack;  // number of stacked source tensors (experts), 0 if not stacked
  uint32_t nseg;    // RowsSeg segments, else 0
  uint32_t crc32c;  // of the tensor's unpadded bytes
  uint32_t src_ndim;
  uint32_t layout;  // Layout; 0 (Natural) in version 1
  uint64_t src_shape[4];  // shape of one source tensor
  uint8_t reserved[512 - 128 - 128 - 96 - 4 * 2 - 8 * 4 - 8 * 2 - 4 * 6 - 8 * 4];
};
static_assert(sizeof(TocEntry) == 512);

// "" if a file's header says it was cut for `rank` of `world`, else what it says and what was wanted: a
// rank's file holds that rank's slices, so another rank's would load and compute garbage.
inline std::string rank_mismatch(const Header& h, uint32_t rank, uint32_t world) {
  if (h.rank == rank && h.world == world) return "";
  char s[128];
  std::snprintf(s, sizeof s, "the file was cut for rank %u of %u, but this process is rank %u of %u", h.rank,
                h.world, rank, world);
  return s;
}

}  // namespace dsw
