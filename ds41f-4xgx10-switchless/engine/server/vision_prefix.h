#pragma once
#include <algorithm>
#include <climits>
#include <vector>
#include "common/frontend_protocol.h"
namespace srv {
// Engram n-grams also inspect the three positions preceding the restart.
inline bool vision_needs_replay(const fe::ImageSpan& image, int q) {
  return image.start + image.rows > uint32_t(std::max(0, q - 3));
}
// Pure token LCP is insufficient because different images share token 129264.
// Text CED and image exact-prefill states never cross this policy boundary.
inline int vision_prefix_limit(const std::vector<fe::ImageSpan>& a,
                               const std::vector<fe::ImageSpan>& b, bool full_prefix = false) {
  if (!full_prefix && a.empty() != b.empty()) return 0;
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
  return std::min(i < a.size() ? int(a[i].start) : INT_MAX,
                  i < b.size() ? int(b[i].start) : INT_MAX);
}
}
