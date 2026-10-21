add_library(dsmodel STATIC model.cu ops.cu fused.cu sample.cu prefill.cu state.cu spec.cu wave_probe.cu spill.cpp)
target_include_directories(dsmodel PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(dsmodel PUBLIC dsattn dsk_kernels dsgb10_comm engram dsw_loader dspark)

add_executable(coll_tests tests/coll_tests.cu)
target_include_directories(coll_tests PRIVATE ${CMAKE_SOURCE_DIR}/attn/tests)
target_link_libraries(coll_tests dsmodel)

# B3: the batch spec step's layout and verdicts (spec_batch.h) against a host reference; host only, no GPU
add_executable(spec_batch_test tests/spec_batch_test.cu)
target_include_directories(spec_batch_test PRIVATE ${CMAKE_SOURCE_DIR})

# Profiling builds (-DDSGB10_NVTX=ON): NVTX ranges per turn, chunk, layer and part for nsys (model/nvtx.h),
# and DSGB10_ROUTE_DUMP (model/prefill.cu). The ranges cost nothing without a profiler attached.
option(DSGB10_NVTX "NVTX ranges in the model and the server, for nsys" OFF)
if(DSGB10_NVTX)
  target_compile_definitions(dsmodel PUBLIC DSGB10_NVTX)
endif()

# Explicit CUDA test binary; never launched by CTest.
option(DSGB10_BUILD_VISION_GPU_TEST "Build the opt-in vision CUDA test" OFF)
if(DSGB10_BUILD_VISION_GPU_TEST)
  add_executable(vision_test tests/vision_test.cu)
  target_link_libraries(vision_test dsmodel)
  target_compile_options(vision_test PRIVATE -UNDEBUG)
endif()
