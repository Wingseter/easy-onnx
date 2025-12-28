cmake_minimum_required(VERSION 3.16)
project(aiRunner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get the root directory (works both standalone and as subdirectory)
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Platform-specific paths
if(WIN32)
    set(ONNXRUNTIME_INCLUDE_DIR ${ROOT_DIR}/Library/ONNXRuntime/windows/include)
    set(ONNXRUNTIME_LIB_DIR ${ROOT_DIR}/Library/ONNXRuntime/windows/lib)
    set(AIRUNNER_OUTPUT_DIR ${ROOT_DIR}/Library/aiRunner/windows)
elseif(APPLE)
    set(ONNXRUNTIME_INCLUDE_DIR ${ROOT_DIR}/Library/ONNXRuntime/mac/include)
    set(ONNXRUNTIME_LIB_DIR ${ROOT_DIR}/Library/ONNXRuntime/mac/lib)
    set(AIRUNNER_OUTPUT_DIR ${ROOT_DIR}/Library/aiRunner/mac)
else()
    set(ONNXRUNTIME_INCLUDE_DIR ${ROOT_DIR}/Library/ONNXRuntime/linux/include)
    set(ONNXRUNTIME_LIB_DIR ${ROOT_DIR}/Library/ONNXRuntime/linux/lib)
    set(AIRUNNER_OUTPUT_DIR ${ROOT_DIR}/Library/aiRunner/linux)
endif()

# Add library
add_library(aiRunner SHARED
    src/aiRunner.cpp
    src/Workflow.cpp
    src/Model.cpp
    src/DataLoader.cpp
    src/Utils.cpp
    src/ModelManager.cpp
    Utils/pch.cpp
)

# Include directories (target-specific)
target_include_directories(aiRunner
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link directories (target-specific)
target_link_directories(aiRunner PRIVATE ${ONNXRUNTIME_LIB_DIR})

# Precompiled header
target_precompile_headers(aiRunner PRIVATE Utils/pch.h)

# Set properties based on the platform
if(WIN32)
    set_target_properties(aiRunner PROPERTIES SUFFIX ".dll" PREFIX "lib")
elseif(APPLE)
    set_target_properties(aiRunner PROPERTIES SUFFIX ".dylib" PREFIX "lib")
else()
    set_target_properties(aiRunner PROPERTIES SUFFIX ".so" PREFIX "lib")
endif()

# Link ONNX Runtime library
target_link_libraries(aiRunner PRIVATE onnxruntime)

# Copy the built library to the Final Output
add_custom_command(TARGET aiRunner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:aiRunner> ${AIRUNNER_OUTPUT_DIR}/lib/$<TARGET_FILE_NAME:aiRunner>
    COMMENT "Copying aiRunner library to output folder"
)

# Copy the include directory to the Final Output
add_custom_command(TARGET aiRunner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${AIRUNNER_OUTPUT_DIR}/include
    COMMENT "Copying aiRunner include files to output folder"
)
