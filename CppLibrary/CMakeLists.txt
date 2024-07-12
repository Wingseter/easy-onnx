cmake_minimum_required(VERSION 3.16)
project(aiRunner)

set(CMAKE_CXX_STANDARD 11)

# 플랫폼별 Include 및 Link 디렉토리 설정
if(WIN32)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/include)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/lib)
elseif(APPLE)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/include)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/lib)
else()
    # 다른 플랫폼을 위한 설정 (필요시 추가)
endif()

# Add library
add_library(aiRunner SHARED aiRunner.cpp
        Core/Workflow.cpp
        Core/Workflow.h
        Core/Model.cpp
        Core/Model.h
        Utils/pch.cpp
        Core/DataLoader.cpp
        Core/DataLoader.h)

# 미리 컴파일된 헤더 설정
target_precompile_headers(aiRunner PRIVATE Utils/pch.h)

# Set properties based on the platform
if(WIN32)
    set_target_properties(aiRunner PROPERTIES SUFFIX ".dll")
elseif(APPLE)
    set_target_properties(aiRunner PROPERTIES SUFFIX ".dylib")
else()
    set_target_properties(aiRunner PROPERTIES SUFFIX ".so")
endif()

# SetExport Name aiRunner
set_target_properties(aiRunner PROPERTIES PREFIX "")

# Link ONNX Runtime library
target_link_libraries(aiRunner onnxruntime)

# Copy the built library to the TestProject Plugins folder
add_custom_command(TARGET aiRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:aiRunner> ${CMAKE_SOURCE_DIR}/../TestProject/Assets/Plugins/$<TARGET_FILE_NAME:aiRunner>
        COMMENT "Copying aiRunner library to TestProject Plugins folder")

# 플랫폼별로 include 및 lib 디렉토리를 TestProject에 복사
if(WIN32)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/include ${CMAKE_SOURCE_DIR}/../TestProject/Library/OnnxRuntime/windows/include
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/lib ${CMAKE_SOURCE_DIR}/../TestProject/Library/OnnxRuntime/windows/lib
            COMMENT "Copying Windows include and lib directories to TestProject")
elseif(APPLE)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/include ${CMAKE_SOURCE_DIR}/../TestProject/Library/OnnxRuntime/mac/include
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/lib ${CMAKE_SOURCE_DIR}/../TestProject/Library/OnnxRuntime/mac/lib
            COMMENT "Copying macOS include and lib directories to TestProject")
else()
    # 다른 플랫폼을 위한 설정 (필요시 추가)
endif()