#pragma once
#include "common/frontend_protocol.h"
#include <cstdint>
#include <string>
#include <vector>

namespace srv {
// Whole images intersecting this encoder chunk. Complete layout is sent separately:
// Engram's q-3 history mask never depends on these feature bytes.
inline bool vision_window(const std::vector<fe::ImageSpan>& images, int begin, int count,
                          uint64_t max_bytes, std::vector<size_t>& indices, uint64_t& bytes,
                          std::string& error) {
  indices.clear(); bytes = 0;
  if (begin < 0 || count < 1 || uint64_t(begin) + count > uint64_t(INT32_MAX)) {
    error = "invalid image feature prefill window"; return false;
  }
  const uint64_t end = uint64_t(begin) + count;
  for (size_t i = 0; i < images.size(); ++i) {
    const auto& s = images[i];
    if (uint64_t(s.start) + s.rows <= uint64_t(begin) || s.start >= end) continue;
    const uint64_t add = uint64_t(s.rows) * 5120 * 2;
    if (add > max_bytes || bytes > max_bytes - add) {
      error = "active image feature window exceeds 512 MiB GPU budget"; return false;
    }
    indices.push_back(i); bytes += add;
  }
  return true;
}
}  // namespace srv
