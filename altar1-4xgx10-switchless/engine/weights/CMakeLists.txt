# Per-rank weight files (.glw): the loader, the C++ checks and the kernel cross-check. The converter and the
# byte-for-byte verifier are Python (convert.py, verify.py; docs/layout.md); libglw_crc.so is their CRC32C.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# CRC32C for glw.py (ctypes); the project has no C language enabled, so the C file builds as C++ (extern "C")
add_library(glw_crc SHARED crc32c.c)
set_source_files_properties(crc32c.c PROPERTIES LANGUAGE CXX)

add_library(glw_loader STATIC loader.cpp)
target_include_directories(glw_loader PUBLIC ${CMAKE_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(glw_loader PUBLIC CUDA::cudart_static Threads::Threads)

add_executable(glw_load load_main.cpp)
target_link_libraries(glw_load PRIVATE glw_loader)

# structure, TOC and per-tensor CRCs, padding (host only)
add_executable(glw_check check.cpp)
target_link_libraries(glw_check PRIVATE glw_loader)

# the loader refuses another rank's file (host only)
add_executable(glw_rank_test tests/rank_test.cpp)
target_link_libraries(glw_rank_test PRIVATE glw_loader)

# the file against attn.h / experts.h (host), and the attention GEMVs on the file's tiles (--gpu)
add_executable(glw_kcheck kcheck.cu)
target_link_libraries(glw_kcheck PRIVATE glw_loader glmattn)

# a filled expert pool against the canonical NVFP4 slices, with layout.h's own readers (host only)
add_executable(glw_expert_fill_test tests/expert_fill_test.cu)
target_link_libraries(glw_expert_fill_test PRIVATE glw_loader)
