cmake_minimum_required(VERSION 3.10)
project(aiRunner)

set(CMAKE_CXX_STANDARD 11)

# 플랫폼별 Include 및 Link 디렉토리 설정
if(WIN32)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/windows/include)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/windows/lib)
elseif(APPLE)
    include_directories(${CMAKE_SOURCE_DIR}/../Library/mac/include)
    link_directories(${CMAKE_SOURCE_DIR}/../Library/mac/lib)
else()
    # 다른 플랫폼을 위한 설정 (필요시 추가)
endif()

# Add library
add_library(aiRunner SHARED aiRunner.cpp
        Base/Workflow.cpp
        Base/Workflow.h)

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

# Link libraries
target_link_libraries(aiRunner <your_library>)

# Copy the built library to the TestProject Plugins folder
add_custom_command(TARGET aiRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:aiRunner>
        ${CMAKE_SOURCE_DIR}/../TestProject/Assets/Plugins/$<TARGET_FILE_NAME:aiRunner>)

# 플랫폼별로 include 및 lib 디렉토리를 TestProject에 복사
if(WIN32)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/../Library/windows/include
            ${CMAKE_SOURCE_DIR}/../TestProject/Library/windows/include)

    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/../Library/windows/lib
            ${CMAKE_SOURCE_DIR}/../TestProject/Library/windows/lib)
elseif(APPLE)
    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/../Library/mac/include
            ${CMAKE_SOURCE_DIR}/../TestProject/Library/mac/include)

    add_custom_command(TARGET aiRunner POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/../Library/mac/lib
            ${CMAKE_SOURCE_DIR}/../TestProject/Library/mac/lib)
else()
    # 다른 플랫폼을 위한 설정 (필요시 추가)
endif()