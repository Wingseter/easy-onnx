 cmake_minimum_required(VERSION 3.29)
 project(CppApplication)

 set(CMAKE_CXX_STANDARD 11)

 # 플랫폼별 Include 및 Link 디렉토리 설정
 if(WIN32)
     include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/include)
     link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/windows/lib)
     include_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/include)
     link_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/windows/lib)
 elseif(APPLE)
     include_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/include)
     link_directories(${CMAKE_SOURCE_DIR}/../Library/OnnxRuntime/mac/lib)
     include_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/include)
     link_directories(${CMAKE_SOURCE_DIR}/../Library/aiRunner/mac/lib)
 else()
 endif()

 # Add executable
 add_executable(CppApplication main.cpp)

 # Link the aiRunner library
 target_link_libraries(CppApplication aiRunner)