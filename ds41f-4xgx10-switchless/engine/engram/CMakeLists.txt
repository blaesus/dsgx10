# Engram: hashing, the NVMe row store (io_uring, raw syscalls) and its tools. Host only.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(engram STATIC hash.cpp store.cpp)
target_include_directories(engram PUBLIC ${CMAKE_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(engram PUBLIC dsw_host Threads::Threads)

foreach(tool hashcheck iobench rowcheck storecheck gathercheck cachesim e2e pack)
  add_executable(engram_${tool} tools/${tool}.cpp)
  target_link_libraries(engram_${tool} PRIVATE engram)
endforeach()
