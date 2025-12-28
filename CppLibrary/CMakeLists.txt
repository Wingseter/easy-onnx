cmake_minimum_required(VERSION 3.16)
project(aiRunner)

set(CMAKE_CXX_STANDARD 14)

# Get the root directory (works both standalone and as subdirectory)
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# 플랫폼별 Include 및 Link 디렉토리 설정
if(WIN32)
    include_directories(${ROOT_DIR}/Library/ONNXRuntime/windows/include)
    link_directories(${ROOT_DIR}/Library/ONNXRuntime/windows/lib)
elseif(APPLE)
    include_directories(${ROOT_DIR}/Library/ONNXRuntime/mac/include)
    link_directories(${ROOT_DIR}/Library/ONNXRuntime/mac/lib)
else()
    include_directories(${ROOT_DIR}/Library/ONNXRuntime/linux/include)
    link_directories(${ROOT_DIR}/Library/ONNXRuntime/linux/lib)
endif()

# Add library
add_library(aiRunner SHARED src/aiRunner.cpp
        src/Workflow.cpp
        include/Workflow.h
        src/Model.cpp
        include/Model.h
        Utils/pch.cpp
        src/DataLoader.cpp
        include/DataLoader.h
        src/Utils.cpp
        include/Utils.h)

# 미리 컴파일된 헤더 설정
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
target_link_libraries(aiRunner onnxruntime)

# Copy the built library to the Final Output based on the platform
if(WIN32)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:aiRunner> ${ROOT_DIR}/Library/aiRunner/windows/lib/$<TARGET_FILE_NAME:aiRunner>
            COMMENT "Copying aiRunner library to output folder (Windows)")
elseif(APPLE)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:aiRunner> ${ROOT_DIR}/Library/aiRunner/mac/lib/$<TARGET_FILE_NAME:aiRunner>
            COMMENT "Copying aiRunner library to output folder (macOS)")
else()
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:aiRunner> ${ROOT_DIR}/Library/aiRunner/other/lib/$<TARGET_FILE_NAME:aiRunner>
            COMMENT "Copying aiRunner library to output folder (Other)")
endif()

# Copy the include directory to the Final Output
if(WIN32)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${ROOT_DIR}/Library/aiRunner/windows/include
            COMMENT "Copying aiRunner include files to output folder")
elseif(APPLE)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${ROOT_DIR}/Library/aiRunner/mac/include
            COMMENT "Copying aiRunner include files to output folder")
else()
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${ROOT_DIR}/Library/aiRunner/other/include
            COMMENT "Copying aiRunner include files to output folder")
endif()