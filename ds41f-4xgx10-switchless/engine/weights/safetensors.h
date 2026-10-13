// Reading a safetensors checkpoint without any library: a minimal JSON parser for the shard headers
// and the index, and a name -> (shard, dtype, shape, offset) table.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsw {

// Just enough JSON for safetensors headers and model.safetensors.index.json.
struct Json {
  enum class T { Null, Bool, Int, Num, Str, Arr, Obj } t = T::Null;
  bool b = false;
  int64_t i = 0;
  double d = 0;
  std::string s;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  const Json* get(const std::string& key) const;  // object member or nullptr
  static Json parse(const std::string& text);     // dies on malformed input
};

struct SrcTensor {
  std::string name;
  int shard = -1;     // index into Checkpoint::shards
  std::string dtype;  // safetensors dtype string, e.g. "F8_E4M3"
  std::vector<uint64_t> shape;
  uint64_t offset = 0;  // absolute file offset of the first byte
  uint64_t bytes = 0;
};

uint32_t st_dtype_bytes(const std::string& dtype);  // dies on unknown

struct Checkpoint {
  std::string dir;
  std::vector<std::string> shards;  // absolute paths, sorted
  std::vector<uint64_t> shard_bytes;
  std::unordered_map<std::string, SrcTensor> tensors;

  // Parses every *.safetensors header in `dir` and cross-checks model.safetensors.index.json.
  static Checkpoint open(const std::string& dir);
  const SrcTensor& at(const std::string& name) const;  // dies if missing
};

}  // namespace dsw
