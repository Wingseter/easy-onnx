cmake_minimum_required(VERSION 3.16)
project(debugging)

set(CMAKE_CXX_STANDARD 11)

# Include directories for ONNX Runtime
include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/include)
link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/lib)

# Add executable
add_executable(debugging main.cpp)

# Link ONNX Runtime library
target_link_libraries(debugging onnxruntime)