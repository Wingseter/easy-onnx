cmake_minimum_required(VERSION 3.16)
project(easy-onnx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apple Silicon only — refuse Intel/Universal builds for now
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS architecture" FORCE)
endif()

# Expose version for subdirectories and packaging scripts
set(EASYONNX_VERSION ${PROJECT_VERSION})

# Build CppLibrary (aiRunner) first
add_subdirectory(CppLibrary)

# Build CppApplication (depends on aiRunner)
add_subdirectory(CppApplication)

# Optional: debugging project
# add_subdirectory(debugging)
