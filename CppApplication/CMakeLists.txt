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

# rpath: find dylibs next to the executable
if(APPLE)
    set_target_properties(CppApplication PROPERTIES
        MACOSX_RPATH ON
        BUILD_RPATH "@executable_path"
        INSTALL_RPATH "@executable_path"
    )
elseif(UNIX)
    set_target_properties(CppApplication PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

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

# Place aiRunner + onnxruntime shared libs next to the executable so rpath resolves.
if(APPLE)
    add_custom_command(TARGET CppApplication POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:aiRunner>
            $<TARGET_FILE_DIR:CppApplication>/$<TARGET_FILE_NAME:aiRunner>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.1.18.1.dylib
            $<TARGET_FILE_DIR:CppApplication>/libonnxruntime.1.18.1.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libonnxruntime.1.18.1.dylib
            $<TARGET_FILE_DIR:CppApplication>/libonnxruntime.dylib
        COMMENT "Staging dylibs next to CppApplication"
    )
elseif(WIN32)
    add_custom_command(TARGET CppApplication POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:aiRunner>
            $<TARGET_FILE_DIR:CppApplication>/$<TARGET_FILE_NAME:aiRunner>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll
            $<TARGET_FILE_DIR:CppApplication>/onnxruntime.dll
        COMMENT "Staging DLLs next to CppApplication"
    )
endif()
