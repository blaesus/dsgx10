#pragma once
#include "model/config.h"
#include <algorithm>
#include <vector>

namespace model::detail {
// Host-only contracts shared by the runtime and CPU tests. Full-layout offsets are dummy zero:
// only the mask-only kernel sees them; chunk offsets address the bounded BF16 feature array.
template<class Span>
bool valid_vision_layout(const std::vector<Span>& spans, int max_ctx) {
  if (max_ctx < 1 || spans.size() > 4096 || spans.size() > size_t(max_ctx)) return false;
  int end = 0;
  for (const auto& span : spans) {
    if (span.start < end || span.rows < 1 || span.rows > 1024 || span.offset != 0 ||
        span.rows > max_ctx || span.start > max_ctx - span.rows) return false;
    end = span.start + span.rows;
  }
  return true;
}
template<class Span>
bool valid_vision_chunk(const std::vector<Span>& spans, int rows, int max_ctx) {
  if (rows < 0 || rows > kMaxVisionRows || rows > max_ctx || spans.size() > size_t(rows)) return false;
  int total = 0, end = 0;
  for (const auto& span : spans) {
    if (span.start < end || span.rows < 1 || span.rows > 1024 || span.offset != total ||
        span.rows > max_ctx || span.start > max_ctx - span.rows ||
        total > rows - span.rows) return false;
    total += span.rows;
    end = span.start + span.rows;
  }
  return total == rows;
}
template<class Span>
void vision_live_layout(std::vector<uint8_t>& live, const std::vector<Span>& spans, int max_ctx) {
  if (spans.empty()) { live.clear(); return; }
  live.assign(max_ctx, 1);
  for (const auto& span : spans) std::fill(live.begin() + span.start, live.begin() + span.start + span.rows, 0);
}
}  // namespace model::detail
