cmake_minimum_required(VERSION 3.17)
project(libtorch-example 
  VERSION 1.0 
  DESCRIPTION "Simple libtorch example"
  LANGUAGES CXX)

find_package(Torch)

add_executable(test)
target_sources(test PRIVATE "test.cc")
target_link_libraries(test PRIVATE ${TORCH_LIBRARIES})
target_compile_features(test PRIVATE cxx_std_14)
