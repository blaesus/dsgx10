# The ring and its collectives: the DeepSeek engine's comm library as this engine's own (docs/ring.md), plus
# GLM-5.3's collectives (glmcoll.cu).
find_package(CUDAToolkit REQUIRED)

set(GLM_COMM_SRC ring.cpp fleet.cpp collectives.cu glmcoll.cu)
add_library(glm53_comm STATIC ${GLM_COMM_SRC})
target_include_directories(glm53_comm PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(glm53_comm PUBLIC ibverbs CUDA::cuda_driver pthread)

# the tests' copy, with the test hooks (a stalled block for the race test); the engine links glm53_comm, which
# has none
add_library(glm53_comm_test STATIC ${GLM_COMM_SRC})
target_include_directories(glm53_comm_test PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(glm53_comm_test PUBLIC GLM_COMM_TEST_HOOKS)
target_link_libraries(glm53_comm_test PUBLIC ibverbs CUDA::cuda_driver pthread)

add_executable(ring_test tests/ring_test.cu)
target_link_libraries(ring_test PRIVATE glm53_comm_test)

# GLM's collectives on the four ranks; the fused all-reduce's norm is checked against attn's rmsnorm kernel.
# Host references in exact f32: no contraction into FMAs.
add_executable(coll_tests tests/coll_tests.cu)
target_compile_options(coll_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
target_link_libraries(coll_tests PRIVATE glm53_comm_test glmattn)
