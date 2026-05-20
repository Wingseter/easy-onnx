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
    src/Logger.cpp
    src/ExecutionProviderConfig.cpp
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
    set_target_properties(aiRunner PROPERTIES
        SUFFIX ".dylib"
        PREFIX "lib"
        MACOSX_RPATH ON
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path"
    )
else()
    set_target_properties(aiRunner PROPERTIES
        SUFFIX ".so"
        PREFIX "lib"
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# Link ONNX Runtime library
target_link_libraries(aiRunner PRIVATE onnxruntime)

# Copy the built library to the Final Output (Library/aiRunner/<os>/lib/)
add_custom_command(TARGET aiRunner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${AIRUNNER_OUTPUT_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:aiRunner> ${AIRUNNER_OUTPUT_DIR}/lib/$<TARGET_FILE_NAME:aiRunner>
    COMMENT "Copying aiRunner library to output folder"
)

# Copy the include directory to the Final Output
add_custom_command(TARGET aiRunner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${AIRUNNER_OUTPUT_DIR}/include
    COMMENT "Copying aiRunner include files to output folder"
)

# Copy the ONNXRuntime shared library next to the produced aiRunner library so
# rpath @loader_path can resolve it at runtime.
if(APPLE)
    add_custom_command(TARGET aiRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.1.18.1.dylib
            $<TARGET_FILE_DIR:aiRunner>/libonnxruntime.1.18.1.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libonnxruntime.1.18.1.dylib
            $<TARGET_FILE_DIR:aiRunner>/libonnxruntime.dylib
        COMMAND ${CMAKE_COMMAND} -E copy
            ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.1.18.1.dylib
            ${AIRUNNER_OUTPUT_DIR}/lib/libonnxruntime.1.18.1.dylib
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libonnxruntime.1.18.1.dylib
            ${AIRUNNER_OUTPUT_DIR}/lib/libonnxruntime.dylib
        COMMENT "Copying ONNXRuntime dylib next to aiRunner"
    )
elseif(WIN32)
    add_custom_command(TARGET aiRunner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll
            $<TARGET_FILE_DIR:aiRunner>/onnxruntime.dll
        COMMAND ${CMAKE_COMMAND} -E copy
            ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll
            ${AIRUNNER_OUTPUT_DIR}/lib/onnxruntime.dll
        COMMENT "Copying onnxruntime.dll next to aiRunner"
    )
endif()

# ----------------------------------------------------------------------------
# Install rules (consumed by scripts/build_release.{sh,ps1} via cmake --install)
# ----------------------------------------------------------------------------
install(TARGETS aiRunner
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

if(APPLE)
    install(FILES
        ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.1.18.1.dylib
        DESTINATION lib
    )
    # Recreate the unversioned symlink at install time.
    install(CODE "
        execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
            libonnxruntime.1.18.1.dylib
            \${CMAKE_INSTALL_PREFIX}/lib/libonnxruntime.dylib)
    ")
elseif(WIN32)
    install(FILES ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll DESTINATION lib)
endif()

if(EXISTS ${ROOT_DIR}/README.md)
    install(FILES ${ROOT_DIR}/README.md DESTINATION .)
endif()
if(EXISTS ${ROOT_DIR}/LICENSE)
    install(FILES ${ROOT_DIR}/LICENSE DESTINATION .)
endif()
