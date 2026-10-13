# Per-rank weight files: partition plan, converter (tiles on the GPU with engine/kernels' own
# functions), verifier (host only), the device loader and the kernel cross-check.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(dsw_host STATIC safetensors.cpp plan.cpp)
target_include_directories(dsw_host PUBLIC ${CMAKE_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(dsw_host PUBLIC Threads::Threads)

add_executable(dsw_convert convert.cpp)
target_link_libraries(dsw_convert PRIVATE dsw_host dsk_kernels CUDA::cudart_static)

add_executable(dsw_verify verify.cpp)
target_link_libraries(dsw_verify PRIVATE dsw_host)

add_library(dsw_loader STATIC loader.cpp)
target_link_libraries(dsw_loader PUBLIC dsw_host CUDA::cudart_static)

add_executable(dsw_load load_main.cpp)
target_link_libraries(dsw_load PRIVATE dsw_loader)

add_executable(dsw_kcheck kcheck.cu)
target_link_libraries(dsw_kcheck PRIVATE dsw_loader dsk_kernels)

# the loader refuses another rank's file (host only: tests/rank_test.cpp)
add_executable(dsw_rank_test tests/rank_test.cpp)
target_link_libraries(dsw_rank_test PRIVATE dsw_loader)
