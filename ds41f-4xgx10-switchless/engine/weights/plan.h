// Which slice of which checkpoint tensor each rank gets (DESIGN.md decision 1, docs/layout.md).
// The partition is a table keyed by the normalised tensor name (layers.N.attn.wq_b.weight), so a
// split can change by editing one line in plan.cpp and re-running the converter.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "weights/dsw.h"
#include "weights/safetensors.h"

namespace dsw {

// One rectangle of a source tensor, viewed as [rows, row_bytes], copied to a contiguous run of the
// output tensor (rows land back to back, ncols bytes each).
struct Piece {
  const SrcTensor* src = nullptr;
  uint64_t row0 = 0, nrows = 0;
  uint64_t col0 = 0, ncols = 0;  // bytes within a source row
  uint64_t dst = 0;              // byte offset in the output tensor
};

struct OutTensor {
  std::string name;    // output name (stacked experts lose the expert index)
  std::string source;  // source name, "{e}" for the expert index when stacked
  std::string slice;   // human-readable
  std::string component;
  DType dtype = DType::F32;
  SliceKind kind = SliceKind::Replicated;
  Layout layout = Layout::Natural;
  uint32_t nstack = 0, nseg = 0;
  std::vector<uint64_t> shape, src_shape;
  uint64_t bytes = 0;
  uint64_t offset = 0;  // absolute file offset once laid out
  std::vector<Piece> pieces;
};

struct Plan {
  int rank = 0, world = 4;
  std::vector<OutTensor> tensors;  // file order
  std::vector<std::string> skipped;
  uint64_t toc_offset = 0, data_offset = 0, file_bytes = 0;
};

// Normalised form: every dot-separated all-digit component becomes N.
std::string normalize(const std::string& name);

// Builds rank `rank`'s plan and lays it out (offsets). Dies on a tensor the table does not know, a
// dimension that does not divide, a slice that cuts through a scale block, or a shape that does not
// fit its tiled layout.
Plan make_plan(const Checkpoint& ck, int rank, int world);

// Cross-rank check: for every source tensor, replicated ones are whole on every rank, split ones
// are cut into disjoint pieces that together cover it exactly.
void check_coverage(const Checkpoint& ck, int world);

// The TOC record for a planned tensor (crc32c left 0 for the caller).
void fill_entry(const OutTensor& o, TocEntry& e);

}  // namespace dsw
