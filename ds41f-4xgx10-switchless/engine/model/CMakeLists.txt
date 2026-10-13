add_library(dsmodel STATIC model.cu ops.cu fused.cu sample.cu prefill.cu state.cu spec.cu wave_probe.cu spill.cpp)
target_include_directories(dsmodel PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(dsmodel PUBLIC dsattn dsk_kernels dsgb10_comm engram dsw_loader dspark)

add_executable(coll_tests tests/coll_tests.cu)
target_include_directories(coll_tests PRIVATE ${CMAKE_SOURCE_DIR}/attn/tests)
target_link_libraries(coll_tests dsmodel)

# Profiling builds (-DDSGB10_NVTX=ON): NVTX ranges per turn, chunk, layer and part for nsys (model/nvtx.h),
# and DSGB10_ROUTE_DUMP (model/prefill.cu). The ranges cost nothing without a profiler attached.
option(DSGB10_NVTX "NVTX ranges in the model and the server, for nsys" OFF)
if(DSGB10_NVTX)
  target_compile_definitions(dsmodel PUBLIC DSGB10_NVTX)
endif()
