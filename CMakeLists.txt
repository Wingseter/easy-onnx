cmake_minimum_required(VERSION 3.16)
project(easy-onnx)

set(CMAKE_CXX_STANDARD 14)

# Build CppLibrary (aiRunner) first
add_subdirectory(CppLibrary)

# Build CppApplication (depends on aiRunner)
add_subdirectory(CppApplication)

# Optional: debugging project
# add_subdirectory(debugging)
