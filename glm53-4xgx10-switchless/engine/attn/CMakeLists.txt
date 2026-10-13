add_library(glmattn STATIC gemv.cu gemm.cu prep.cu indexer.cu sparse_attn.cu)
target_include_directories(glmattn PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(attn_tests tests/attn_tests.cu)
target_include_directories(attn_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(attn_tests glmattn)
add_executable(bench_attn tests/bench_attn.cu)
target_include_directories(bench_attn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(bench_attn glmattn)
