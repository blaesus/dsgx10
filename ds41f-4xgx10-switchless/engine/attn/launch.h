// Host side of the fused decode path's launches: cudaLaunchKernelEx with programmatic stream
// serialization (PDL) when asked, so a kernel's prologue overlaps the previous kernel in the stream.
#pragma once
#include <cuda_runtime.h>

#include "common/check.h"

namespace attn {

template <class... KArgs, class... Args>
inline void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, bool pdl,
                   Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

}  // namespace attn
