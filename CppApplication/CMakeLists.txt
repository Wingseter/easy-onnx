cmake_minimum_required(VERSION 3.16)
project(CppApplication)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get the root directory (works both standalone and as subdirectory)
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Platform-specific paths
if(WIN32)
    set(ONNXRUNTIME_INCLUDE_DIR ${ROOT_DIR}/Library/ONNXRuntime/windows/include)
    set(ONNXRUNTIME_LIB_DIR ${ROOT_DIR}/Library/ONNXRuntime/windows/lib)
    set(AIRUNNER_LIB_DIR ${ROOT_DIR}/Library/aiRunner/windows/lib)
elseif(APPLE)
    set(ONNXRUNTIME_INCLUDE_DIR ${ROOT_DIR}/Library/ONNXRuntime/mac/include)
    set(ONNXRUNTIME_LIB_DIR ${ROOT_DIR}/Library/ONNXRuntime/mac/lib)
    set(AIRUNNER_LIB_DIR ${ROOT_DIR}/Library/aiRunner/mac/lib)
else()
    set(ONNXRUNTIME_INCLUDE_DIR ${ROOT_DIR}/Library/ONNXRuntime/linux/include)
    set(ONNXRUNTIME_LIB_DIR ${ROOT_DIR}/Library/ONNXRuntime/linux/lib)
    set(AIRUNNER_LIB_DIR ${ROOT_DIR}/Library/aiRunner/linux/lib)
endif()

# Add executable
add_executable(CppApplication main.cpp)

# Include directories (target-specific)
target_include_directories(CppApplication
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${ROOT_DIR}/CppLibrary/include
)

# Link the aiRunner library (target if building together, otherwise search in lib dir)
if(TARGET aiRunner)
    target_link_libraries(CppApplication PRIVATE aiRunner)
else()
    target_link_directories(CppApplication PRIVATE ${AIRUNNER_LIB_DIR})
    target_link_libraries(CppApplication PRIVATE aiRunner)
endif()

target_link_directories(CppApplication PRIVATE ${ONNXRUNTIME_LIB_DIR})
target_link_libraries(CppApplication PRIVATE onnxruntime)
