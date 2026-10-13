find_package(CUDAToolkit REQUIRED)

add_library(dsgb10_comm STATIC ring.cpp fleet.cpp collectives.cu)
target_link_libraries(dsgb10_comm PUBLIC ibverbs CUDA::cuda_driver pthread)

# ring_test's copy, with the test hooks (collectives.cu: a stalled block for the race test); the engine links
# dsgb10_comm, which has none
add_library(dsgb10_comm_test STATIC ring.cpp fleet.cpp collectives.cu)
target_compile_definitions(dsgb10_comm_test PUBLIC DSGB10_COMM_TEST_HOOKS)
target_link_libraries(dsgb10_comm_test PUBLIC ibverbs CUDA::cuda_driver pthread)

add_executable(ring_test tests/ring_test.cu)
target_link_libraries(ring_test PRIVATE dsgb10_comm_test)
