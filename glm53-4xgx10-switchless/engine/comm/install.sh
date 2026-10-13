#!/bin/bash
# Build the ring tests (CPU only, -j4) and install them with the sources' hash for run.sh.
#   install.sh [bin dir]     (default ~/dsgb10-data/glm53-comm/bin)
# Builds through an out-of-tree wrapper project holding the engine's top-level flags, comm/ and attn/ (coll_tests
# checks the fused norm against attn's rmsnorm), so a half-written sibling directory cannot stop the build.
set -eu
cd "$(dirname "$0")"
ENGINE=$(cd .. && pwd)
TOP=$HOME/dsgb10-data/glm53-comm
BIN=${1:-$TOP/bin}
mkdir -p "$TOP/wrap" "$BIN"
sed -n '/^set(CMAKE_CUDA_FLAGS/p' "$ENGINE/CMakeLists.txt" > "$TOP/wrap/flags.cmake"
cat > "$TOP/wrap/CMakeLists.txt" << 'W'
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(glm53_comm LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
include(${CMAKE_CURRENT_SOURCE_DIR}/flags.cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
include_directories(${GLM_ENGINE})
add_subdirectory(${GLM_ENGINE}/comm comm)
add_subdirectory(${GLM_ENGINE}/attn attn)
W
cmake -S "$TOP/wrap" -B "$TOP/build" -DGLM_ENGINE="$ENGINE" > /dev/null
cmake --build "$TOP/build" -j4 --target ring_test coll_tests > /dev/null
cp "$TOP/build/comm/ring_test" "$TOP/build/comm/coll_tests" "$BIN/"
h=$(cat ./*.h ./*.cuh ./*.cpp ./*.cu tests/*.cu ../attn/attn.h ../attn/prep.cu | sha256sum | cut -c1-16)
echo "sources sha256 $h, built $(date '+%F %T %z')" > "$BIN/SOURCES"
cat "$BIN/SOURCES"
