# Routed experts (A3e): routing, the decode expert step, the prefill expert GEMMs; tests and bench. The bench
# also times altar-1's BF16 shared expert through attn's BF16 GEMV (glmattn).
add_library(glm_experts STATIC route.cu decode.cu prefill.cu)
target_include_directories(glm_experts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(glm_experts PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

find_package(OpenMP REQUIRED)
add_executable(test_experts tests/test_experts.cu)
target_link_libraries(test_experts glm_experts OpenMP::OpenMP_CXX)
target_compile_options(test_experts PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

add_executable(bench_experts tests/bench_experts.cu)
target_link_libraries(bench_experts glm_experts glmattn)
