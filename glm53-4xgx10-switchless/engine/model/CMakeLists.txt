# The layer runner and the TP=4 decode step (G4), prefill, the MTP passes; the driver glm_run (docs/engine.md).
find_package(CUDAToolkit REQUIRED)
add_library(glm_model STATIC bind.cu model.cu ops.cu sample.cu prefill.cu mtp.cu l2pf.cu)
target_include_directories(glm_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(glm_model PUBLIC glmattn glm_experts glm53_comm glw_loader)

add_executable(glm_run tools/glm_run.cu)
target_link_libraries(glm_run PRIVATE glm_model)
