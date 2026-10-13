# The engine: C++20 + CUDA for GB10 (sm_121a) only.
# Each line of work owns one subdirectory with its own CMakeLists.txt; it is picked up when present.
cmake_minimum_required(VERSION 3.28)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
endif()
# Block-scaled MMA needs the arch-specific target: compute_121a/sm_121a (see docs/findings.md#tensor-cores).
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(glm53 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(dir common weights comm experts attn model server tests)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()

# This project's fleet file beside the binaries that read one (dsgb10_rank, ring_test, coll_tests), as
# fleet.conf, where they look by default; run4.sh and service.sh install their FLEET beside the copies
# they run, and --fleet FILE reads another.
foreach(dir server comm model)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../../ds41f-4xgx10-switchless/fleet/homelab.conf ${CMAKE_BINARY_DIR}/${dir}/fleet.conf COPYONLY)
endforeach()
