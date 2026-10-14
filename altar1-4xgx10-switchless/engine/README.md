# The engine

C++20 + CUDA for GB10 (sm_121a) only, as the DeepSeek engine's. One subdirectory per part, picked up by
CMake when it has a CMakeLists.txt:

- `common/`: error checks (copied from the DeepSeek engine).
- `experts/`: the routed-expert kernels: the E2M1-subset/NVFP4 mixed format, the decode GEMV at 1–16 rows over
  the union of the rows' experts, the prefill GEMM, and the tile layout the converter writes.
- `attn/`: the attention side: FP8 projections, absorbed sparse MLA attention, the indexer, rope, norms, the
  shared expert and dense MLPs, the head.
- `weights/`, `comm/`, `model/`, `server/`: later milestones (G2, G4, G6); `comm/` is the DeepSeek ring.

Build: `cmake -S engine -B engine/build && cmake --build engine/build -j4` (never more jobs on a box that
serves production). The fleet file is the DeepSeek configuration's until this one has its own.
