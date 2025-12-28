cmake_minimum_required(VERSION 3.16)
project(CppApplication)

set(CMAKE_CXX_STANDARD 14)

# Get the root directory (works both standalone and as subdirectory)
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# 플랫폼별 Include 및 Link 디렉토리 설정
if(WIN32)
    include_directories(${ROOT_DIR}/Library/ONNXRuntime/windows/include)
    include_directories(${ROOT_DIR}/CppLibrary/include)
    link_directories(${ROOT_DIR}/Library/ONNXRuntime/windows/lib)
elseif(APPLE)
    include_directories(${ROOT_DIR}/Library/ONNXRuntime/mac/include)
    include_directories(${ROOT_DIR}/CppLibrary/include)
    link_directories(${ROOT_DIR}/Library/ONNXRuntime/mac/lib)
else()
    include_directories(${ROOT_DIR}/Library/ONNXRuntime/linux/include)
    include_directories(${ROOT_DIR}/CppLibrary/include)
    link_directories(${ROOT_DIR}/Library/ONNXRuntime/linux/lib)
endif()

# Add executable
add_executable(CppApplication main.cpp)

# Link the aiRunner library (target if building together, otherwise search in lib dir)
if(TARGET aiRunner)
    target_link_libraries(CppApplication aiRunner)
else()
    if(WIN32)
        link_directories(${ROOT_DIR}/Library/aiRunner/windows/lib)
    elseif(APPLE)
        link_directories(${ROOT_DIR}/Library/aiRunner/mac/lib)
    else()
        link_directories(${ROOT_DIR}/Library/aiRunner/linux/lib)
    endif()
    target_link_libraries(CppApplication aiRunner)
endif()

target_link_libraries(CppApplication onnxruntime)