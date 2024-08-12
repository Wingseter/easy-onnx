cmake_minimum_required(VERSION 3.29)
project(CppApplication)

set(CMAKE_CXX_STANDARD 14)

# 플랫폼별 Include 및 Link 디렉토리 설정
if(WIN32)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/include)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/windows/include)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/lib)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/windows/lib)
elseif(APPLE)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/include)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/mac/include)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/lib)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/mac/lib)
else()
    include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/linux/include)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/linux/include)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/linux/lib)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/linux/lib)
endif()

# Add executable
add_executable(CppApplication main.cpp)

# Link the aiRunner library
target_link_libraries(CppApplication aiRunner)
target_link_libraries(CppApplication onnxruntime)