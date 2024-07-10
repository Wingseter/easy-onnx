cmake_minimum_required(VERSION 3.10)
project(aiRunner)

set(CMAKE_CXX_STANDARD 11)

add_library(aiRunner SHARED aiRunner.cpp)

if(WIN32)
    set_target_properties(aiRunner PROPERTIES SUFFIX ".dll")
elseif(APPLE)
    set_target_properties(aiRunner PROPERTIES SUFFIX ".dylib")
else()
    set_target_properties(aiRunner PROPERTIES SUFFIX ".so")
endif()

# SetExport Name aiRunner
set_target_properties(aiRunner PROPERTIES PREFIX "")

# 라이브러리를 TestProject Plugins 폴더에 복사하는 스크립트 추가
add_custom_command(TARGET aiRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:aiRunner>
        ${CMAKE_SOURCE_DIR}/../TestProject/Assets/Plugins/$<TARGET_FILE_NAME:aiRunner>)