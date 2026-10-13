#!/bin/bash
# Build the glm53 micro benchmarks for GB10 (sm_121a: block-scaled MMA needs the arch-specific target).
#   ./build.sh [outdir]      (default: ~/dsgb10-data/glm53-micro/bin)
set -eu
cd "$(dirname "$0")"
OUT=${1:-$HOME/dsgb10-data/glm53-micro/bin}
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
FLAGS="-O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -lineinfo"
mkdir -p "$OUT"
$NVCC $FLAGS -Xcompiler -fopenmp -o "$OUT/gemv" gemv.cu -lgomp
$NVCC $FLAGS -o "$OUT/mma" mma.cu
echo "built in $OUT with: $NVCC $FLAGS"
