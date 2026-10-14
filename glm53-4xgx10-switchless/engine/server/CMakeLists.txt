# The serving layer (G6, docs/serving.md): glm53_rank --mode serve.
add_executable(glm53_rank main.cpp cmd.cpp serve.cpp disk.cpp)
target_link_libraries(glm53_rank PRIVATE glm_model)
