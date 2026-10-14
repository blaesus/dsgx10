# The serving layer (G6, docs/serving.md): altar1_rank --mode serve.
add_executable(altar1_rank main.cpp cmd.cpp serve.cpp disk.cpp)
target_link_libraries(altar1_rank PRIVATE glm_model)
