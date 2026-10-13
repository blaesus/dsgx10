add_library(dsk_kernels STATIC
  act_quant.cu
  gemv_fp8.cu
  moe.cu
  gemv16.cu
  gemv_fp4.cu
  l2pf.cu
  norm_router.cu
  prefill.cu
  prefill_attn.cu
  prefill_index.cu
  prefill_comp.cu
  prefill_mhc.cu
)
target_include_directories(dsk_kernels PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels dsk_kernels)

add_executable(bench_kernels tests/bench_kernels.cu)
target_link_libraries(bench_kernels dsk_kernels)

# the prefill attention side (M5, part 2), checked against engine/attn's kernels and reference cases
add_executable(test_pattn tests/test_pattn.cu)
target_link_libraries(test_pattn dsk_kernels dsattn)
add_executable(bench_pattn tests/bench_pattn.cu)
target_link_libraries(bench_pattn dsk_kernels dsattn)
add_executable(bench_chain tests/bench_chain.cu)
target_link_libraries(bench_chain dsk_kernels dsattn)
