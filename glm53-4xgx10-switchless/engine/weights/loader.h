// Loading a rank's .glw file into one device arena: O_DIRECT reads by several threads into pinned staging buffers
// (two per thread), each copied to the device while the next is read (the DeepSeek engine's loader). Reserved
// entries (routed experts not yet filled) take no arena space and have data == nullptr.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "weights/glw.h"

namespace glw {

struct DeviceTensor {
  std::string name;
  DType dtype = DType::F32;
  Layout layout = Layout::Natural;
  SliceKind kind = SliceKind::Replicated;
  int ndim = 0;
  int64_t shape[4] = {0, 0, 0, 0};  // natural (logical) shape, whatever the layout
  uint32_t layer = kGlobal;
  uint32_t blk_rows = 0, blk_k = 0;  // FP8 weights and their scale tables: the scale block
  uint32_t flags = 0, fmt = 0;
  float rel_err = 0;
  uint64_t bytes = 0;
  uint64_t reserve_bytes = 0;
  uint64_t file_offset = 0;  // absolute, in the file
  uint64_t offset = 0;       // in the arena, once loaded
  uint32_t crc32c = 0;
  void* data = nullptr;  // device pointer, nullptr until loaded (or not selected, or reserved)
  bool reserved() const { return kind == SliceKind::Reserved; }
};

struct LoadOptions {
  int threads = 4;               // concurrent O_DIRECT reads
  uint64_t chunk = 64ull << 20;  // bytes per read
  uint64_t limit = 0;            // stop once the arena would exceed this many bytes (tests); 0 = no limit
  bool verify_crc = false;       // afterwards copy every loaded tensor back and check its CRC
  // load only the tensors this accepts (in file order, packed into a smaller arena); empty = all
  std::function<bool(const DeviceTensor&)> select;
};

struct LoadStats {
  uint64_t bytes = 0;
  double seconds = 0, gbps = 0;  // streaming file -> device
  double verify_seconds = 0;     // when verify_crc
};

class WeightFile {
 public:
  // Reads and validates the header and TOC (no device work); with rank >= 0, also that the file is rank `rank` of
  // `world`'s (rank_mismatch), dying with the path and both ranks if not.
  explicit WeightFile(const std::string& path, int rank = -1, int world = 0);
  ~WeightFile();
  WeightFile(const WeightFile&) = delete;
  WeightFile& operator=(const WeightFile&) = delete;

  LoadStats load(const LoadOptions& opt = {});  // allocates the arena and streams the data in

  // Host copy of one tensor's bytes (no device): for checks and the bind step's host-side validation.
  std::vector<uint8_t> read_host(const DeviceTensor& t) const;

  const DeviceTensor& operator[](const std::string& name) const;  // dies if missing or not loaded
  const DeviceTensor* find(const std::string& name) const;        // nullptr if missing
  const std::vector<DeviceTensor>& tensors() const { return tensors_; }
  const Header& header() const { return header_; }
  const std::string& path() const { return path_; }
  void* arena() const { return arena_; }
  uint64_t arena_bytes() const { return arena_bytes_; }

 private:
  std::string path_;
  int fd_ = -1;
  uint64_t fsize_ = 0;
  Header header_{};
  std::vector<DeviceTensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
  void* arena_ = nullptr;
  uint64_t arena_bytes_ = 0;
};

}  // namespace glw
