#!/bin/bash
# Build altar1_rank (CPU only, -j4) out of tree and install it with the sources' hash (docs/serving.md).
#   install.sh [bin dir]     (default ~/dsgb10-data/altar1-serve/bin)
# A wrapper project holds the engine's top-level flags and the directories the server links (comm, attn, experts,
# weights, model, server), as engine/model/install.sh does.
set -eu
cd "$(dirname "$0")"
ENGINE=$(cd .. && pwd)
TOP=${SERVER_TOP:-$HOME/dsgb10-data/altar1-serve}  # SERVER_TOP: another build tree (a branch beside the main one)
BIN=${1:-$TOP/bin}
mkdir -p "$TOP/wrap" "$BIN"
sed -n '/^set(CMAKE_CUDA_FLAGS/p' "$ENGINE/CMakeLists.txt" > "$TOP/wrap/flags.cmake"
cat > "$TOP/wrap/CMakeLists.txt" << 'W'
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(altar1_server LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
include(${CMAKE_CURRENT_SOURCE_DIR}/flags.cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
include_directories(${GLM_ENGINE})
foreach(d comm attn experts weights model server)
  add_subdirectory(${GLM_ENGINE}/${d} ${d})
endforeach()
W
cmake -S "$TOP/wrap" -B "$TOP/build" -DGLM_ENGINE="$ENGINE" > /dev/null
cmake --build "$TOP/build" -j4 --target altar1_rank > /dev/null
cp "$TOP/build/server/altar1_rank" "$BIN/altar1_rank.new" && mv -f "$BIN/altar1_rank.new" "$BIN/altar1_rank"
h=$(cat ./*.h ./*.cpp ../model/*.h ../model/*.cu ../attn/*.h ../attn/*.cu ../experts/*.h ../experts/*.cu ../comm/*.h \
  ../comm/*.cu ../comm/*.cpp ../weights/*.h ../weights/*.cpp | sha256sum | cut -c1-16)
c=$(git rev-parse --short HEAD 2> /dev/null)$(git status --porcelain .. 2> /dev/null | grep -q . && echo +dirty || true)
echo "sources sha256 $h, commit $c, built $(date '+%F %T %z')" > "$BIN/SOURCES"
cat "$BIN/SOURCES"
