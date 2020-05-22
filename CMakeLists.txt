cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test test.cc)
target_link_libraries(test "${TORCH_LIBRARIES}")
message("${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)

