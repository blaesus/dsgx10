#!/bin/bash
# Build the model's driver glm_run (CPU only, -j4) and install it with the sources' hash for run.sh.
#   install.sh [bin dir]     (default ~/dsgb10-data/altar1-model/bin)
# Builds through an out-of-tree wrapper project holding the engine's top-level flags and the directories the model
# links (attn, experts, comm, weights), so a half-written sibling directory cannot stop the build.
set -eu
cd "$(dirname "$0")"
ENGINE=$(cd .. && pwd)
TOP=$HOME/dsgb10-data/${DATA_NAME:-altar1-model}  # DATA_NAME: another tree (a tuning branch beside the main one)
BIN=${1:-$TOP/bin}
mkdir -p "$TOP/wrap" "$BIN"
sed -n '/^set(CMAKE_CUDA_FLAGS/p' "$ENGINE/CMakeLists.txt" > "$TOP/wrap/flags.cmake"
cat > "$TOP/wrap/CMakeLists.txt" << 'W'
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(altar1_model LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
include(${CMAKE_CURRENT_SOURCE_DIR}/flags.cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
include_directories(${GLM_ENGINE})
foreach(d comm attn experts weights model)
  add_subdirectory(${GLM_ENGINE}/${d} ${d})
endforeach()
W
cmake -S "$TOP/wrap" -B "$TOP/build" -DGLM_ENGINE="$ENGINE" > /dev/null
cmake --build "$TOP/build" -j4 --target glm_run glw_load glw_kcheck > /dev/null
cp "$TOP/build/model/glm_run" "$TOP/build/weights/glw_load" "$TOP/build/weights/glw_kcheck" "$BIN/"
h=$(cat ./*.h ./*.cu tools/*.cu tools/*.h tools/*.py ../attn/*.h ../attn/*.cu ../experts/*.h ../experts/*.cu ../comm/*.h ../comm/*.cu ../comm/*.cpp \
  ../weights/*.h ../weights/*.cpp | sha256sum | cut -c1-16)
echo "sources sha256 $h, built $(date '+%F %T %z')" > "$BIN/SOURCES"
cat "$BIN/SOURCES"
