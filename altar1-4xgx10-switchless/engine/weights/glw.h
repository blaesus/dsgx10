// The per-rank weight file (".glw"): one rank's slice of every tensor GLM-5.3's engine uses, in the formats and
// tile layouts the kernels read, laid out so a loader can stream it into one device arena with large O_DIRECT
// reads. The DeepSeek engine's .dsw container (ds41f-4xgx10-switchless/engine/weights/dsw.h), this engine's
// own copy: a new magic, dtypes, layouts and TOC fields. Format, plan and tools: docs/layout.md.
//
//   [0, 4096)                 Header
//   [toc_offset, ...)         n_tensors TocEntry records (512 B each), padded to 4 KiB
//   [data_offset, file_bytes) tensors, each starting on a 4 KiB boundary, zero-padded to 4 KiB
//
// Device address of a tensor = arena + (entry.offset - header.data_offset). Reserved entries (the routed experts
// until they are filled) have no data: offset 0, bytes 0, flag kReserved, and reserve_bytes says what the fill
// will add. They are the last entries, so filling them appends to the file and leaves every other byte in place.
// glw.py writes this format (the Python side mirrors these structs field by field).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace glw {

constexpr char kMagic[8] = {'G', 'L', 'M', '5', '3', 'W', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint64_t kAlign = 4096;
constexpr uint32_t kGlobal = 0xFFFFFFFFu;  // TocEntry::layer of a tensor that belongs to no layer

inline uint64_t align_up(uint64_t x, uint64_t a = kAlign) { return (x + a - 1) / a * a; }

enum class DType : uint32_t {
  F32 = 1,
  BF16 = 2,
  F8_E4M3 = 3,  // e4m3fn
  U8 = 4,       // raw bytes (expert pools, their descriptor tables)
  U32 = 5,      // the RoPE table: bf16 cos low, bf16 sin high
};

inline uint32_t dtype_bytes(DType t) {
  return t == DType::F32 || t == DType::U32 ? 4 : t == DType::BF16 ? 2 : 1;
}
inline const char* dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "F32";
    case DType::BF16: return "BF16";
    case DType::F8_E4M3: return "F8_E4M3";
    case DType::U8: return "U8";
    case DType::U32: return "U32";
  }
  return "?";
}

// How a rank's tensor relates to its source tensor(s).
enum class SliceKind : uint32_t {
  Replicated = 1,  // the whole source
  Rows = 2,        // a contiguous run of rows of one source
  Cols = 3,        // a contiguous run of columns of one source
  Recut = 4,       // pieces of several sources, or rows transposed / interleaved (docs/layout.md, the plan table)
  Generated = 5,   // computed, not read (the RoPE table)
  Reserved = 6,    // no data yet (routed experts)
};
inline const char* kind_name(SliceKind k) {
  switch (k) {
    case SliceKind::Replicated: return "replicated";
    case SliceKind::Rows: return "rows";
    case SliceKind::Cols: return "cols";
    case SliceKind::Recut: return "recut";
    case SliceKind::Generated: return "generated";
    case SliceKind::Reserved: return "reserved";
  }
  return "?";
}

// How a tensor's bytes are arranged. `shape` stays the natural (logical) shape.
enum class Layout : uint32_t {
  Natural = 0,  // row-major
  // FP8 e4m3 [N][K], N % 8 == 0, K % 64 == 0, the attention GEMV's tile (docs/attention.md § 2,
  // engine/attn/tools/pack.py tiles()): [N/8][K/64][32 lanes][16 B]; lane 4g + t holds row 8i + g,
  // k = 64j + 16t + b at byte b: byte (n, k) at ((n/8)*(K/64) + k/64)*512 + (4*(n%8) + (k%64)/16)*16 + k%16.
  T64 = 1,
  // FP8 e4m3 [N][K], N % 8 == 0, K % 128 == 0, the expert kernels' shared-expert tile (experts/layout.h
  // pack_fp8): [N/8][K/128][32 lanes][32 B]; lane 4g + t holds row 8i + g, k = 128j + 64q + 16t + b at byte
  // 16q + b: byte (n, k) at (((n/8)*(K/128) + k/128)*32 + 4*(n%8) + (k%64)/16)*32 + 16*((k%128)/64) + k%16.
  T128 = 2,
  // Routed experts: one entry an expert, its gate, up and down matrices back to back, each in the expert kernels'
  // 8-row x 512-k blocks of its own format (experts/layout.h pack_nvfp4 / pack_s3). A layer's 256 expert entries
  // are contiguous (every matrix is a multiple of 4 KiB), so they form the layer's pool from expert 0's entry on;
  // the layer's desc entry (gx::MatDesc [256][3]) holds offsets relative to it.
  ExpertPool = 3,
};
inline const char* layout_name(Layout l) {
  switch (l) {
    case Layout::Natural: return "natural";
    case Layout::T64: return "T64";
    case Layout::T128: return "T128";
    case Layout::ExpertPool: return "expert-pool";
  }
  return "?";
}

// The routed experts' formats (experts/experts.h Fmt). An expert entry's (and its layer's desc entry's) `fmt` is
// one nibble a matrix: gate | up << 4 | down << 8 (a layer is single-format: altar-1 stores INT4 = 0x222 on layers
// 4-76 and BF16 = 0x333 on 3, 77, 78; NVFP4 = 0 is kept for the MTP layer's option); every other entry has None.
// 1 was GLM-5.3's S3, retired: readers refuse it as any unknown nibble.
enum class ExpertFmt : uint32_t { NVFP4 = 0, INT4 = 2, BF16 = 3, None = 0xFFFFFFFFu };
inline uint32_t expert_matrix_fmt(uint32_t tag, int m) { return (tag >> (4 * m)) & 15; }  // m: 0 gate, 1 up, 2 down

enum Flags : uint32_t {
  kLossy = 1,        // converted from the BF16 master (FP8 by round-to-nearest); rel_err says how far
  kReservedFlag = 2, // no data yet
  kScale = 4,        // an FP8 weight's f32 scale table: [N/blk_rows][K/blk_k], or [N/2/blk_rows][2][K/blk_k]
                     // for an interleaved gate/up pair
  kInterleaved = 8,  // gate/up rows interleaved (2i gate, 2i+1 up): the SwiGLU GEMV's input
};

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
  uint32_t toc_crc32c;  // CRC32C of the whole TOC (n_tensors entries)
  char model[64];
  char source[256];  // the master checkpoint (directory name and revision)
  char created[32];  // UTC, ISO 8601
  char tool[64];     // the converter's version: SHA-256 prefix of its sources (and the commit when given)
  uint8_t reserved[4096 - 8 - 6 * 4 - 4 * 8 - 2 * 4 - 64 - 256 - 32 - 64];
};
static_assert(sizeof(Header) == 4096);

struct TocEntry {
  char name[96];      // engine name, e.g. layers.10.attn.xproj.weight
  char source[192];   // source tensors (checkpoint names without "model.", comma-separated) or the generator
  char slice[96];     // human-readable description of the cut
  uint32_t dtype;     // DType
  uint32_t ndim;
  uint64_t shape[4];  // natural shape
  uint64_t offset;    // absolute file offset, kAlign-aligned (0 for Reserved)
  uint64_t bytes;     // unpadded (0 for Reserved)
  uint32_t kind;      // SliceKind
  uint32_t layout;    // Layout
  uint32_t crc32c;    // of the unpadded bytes
  uint32_t layer;     // 0..78, or kGlobal
  uint32_t blk_rows;  // an FP8 weight and its scale table: the scale block (rows, k); else 0
  uint32_t blk_k;
  uint32_t fmt;       // expert entries: ExpertFmt a matrix (gate | up << 4 | down << 8), else None
  uint32_t flags;     // Flags
  float rel_err;      // kLossy: relative RMS error of the FP8 source(s) against the master (the largest)
  uint32_t pad0;
  uint64_t reserve_bytes;  // Reserved: bytes the fill will write
  uint8_t reserved[512 - 96 - 192 - 96 - 4 * 2 - 8 * 4 - 8 * 2 - 4 * 10 - 8];
};
static_assert(sizeof(TocEntry) == 512);

// "" if a file's header says it was cut for `rank` of `world`, else what it says and what was wanted: a rank's
// file holds that rank's slices, so another rank's would load and compute garbage.
inline std::string rank_mismatch(const Header& h, uint32_t rank, uint32_t world) {
  if (h.rank == rank && h.world == world) return "";
  char s[128];
  std::snprintf(s, sizeof s, "the file was cut for rank %u of %u, but this process is rank %u of %u", h.rank,
                h.world, rank, world);
  return s;
}

// Byte offset of natural element (n, k) in the tiled layouts (the verifier's and kcheck's independent copy of the
// formulas above).
inline uint64_t t64_offset(uint64_t n, uint64_t k, uint64_t K) {
  return ((n / 8) * (K / 64) + k / 64) * 512 + (4 * (n % 8) + (k % 64) / 16) * 16 + k % 16;
}
inline uint64_t t128_offset(uint64_t n, uint64_t k, uint64_t K) {
  return (((n / 8) * (K / 128) + k / 128) * 32 + 4 * (n % 8) + (k % 64) / 16) * 32 + 16 * ((k % 128) / 64) + k % 16;
}

}  // namespace glw
