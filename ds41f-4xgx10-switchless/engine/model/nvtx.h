// NVTX ranges for profiling builds (nsys): configure with -DDSGB10_NVTX=ON (model/CMakeLists.txt defines
// DSGB10_NVTX for dsmodel and its users). Other builds compile NVTX_RANGE to nothing. A range covers the
// host's enqueueing; nsys projects it onto the GPU work launched inside it (nvtx_gpu_proj_trace).
#pragma once
#ifdef DSGB10_NVTX
#include <nvtx3/nvToolsExt.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace model {
struct NvtxRange {
  explicit NvtxRange(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
    char b[128];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    nvtxRangePushA(b);
  }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};
}  // namespace model
#define DSGB10_NVTX_CAT2(a, b) a##b
#define DSGB10_NVTX_CAT(a, b) DSGB10_NVTX_CAT2(a, b)
#define NVTX_RANGE(...) ::model::NvtxRange DSGB10_NVTX_CAT(nvtx_range_, __LINE__)(__VA_ARGS__)
#else
#define NVTX_RANGE(...) ((void)0)
#endif
