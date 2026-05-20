# C++ Integration

This guide describes how to consume the `easy-onnx` release zip from your own
C++ project.

## 1. Get the binaries

Download a release zip (e.g. `easyonnx-mac-arm64-v0.1.0.zip`) and extract it.
The layout is:

```
easyonnx-mac-arm64/
├── lib/
│   ├── libaiRunner.dylib
│   ├── libonnxruntime.1.18.1.dylib
│   └── libonnxruntime.dylib -> libonnxruntime.1.18.1.dylib
├── include/
│   ├── aiRunner.h
│   ├── Workflow.h
│   ├── Model.h
│   └── ...
└── README.md
```

Windows: `easyonnx-windows-x64-v0.1.0.zip` contains `lib/aiRunner.dll`,
`lib/onnxruntime.dll` and the same headers.

## 2. macOS: Gatekeeper quarantine

The dylibs are **not code-signed**. The first time you copy the zip to a new
mac, macOS sets a quarantine attribute that blocks loading. Strip it:

```sh
xattr -dr com.apple.quarantine /path/to/easyonnx-mac-arm64
```

Long-term: code signing and notarization is on the roadmap but out of scope
for v0.1.x.

## 3. CMake consumer example

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_app CXX)
set(CMAKE_CXX_STANDARD 17)

set(EASYONNX_DIR "/path/to/easyonnx-mac-arm64")

add_executable(my_app main.cpp)

target_include_directories(my_app PRIVATE ${EASYONNX_DIR}/include)
target_link_directories(my_app    PRIVATE ${EASYONNX_DIR}/lib)
target_link_libraries(my_app      PRIVATE aiRunner onnxruntime)

# Resolve dylibs next to (or relative to) the executable at runtime.
if(APPLE)
    set_target_properties(my_app PROPERTIES
        BUILD_RPATH   "${EASYONNX_DIR}/lib"
        INSTALL_RPATH "@executable_path/../lib"
    )
endif()
```

## 4. Minimal call site

```cpp
#include "aiRunner.h"

int main() {
    // cpu_use=false enables the platform accelerator
    // (CoreML on macOS, DirectML on Windows). If the model is not
    // accelerator-compatible, the library logs a warning and uses CPU.
    if (!InitModel("model.onnx", /*cpu_use=*/false)) return 1;

    int n = GetElementsPerSample();
    std::vector<float> input(n, 0.0f);
    RunModelFloat(input.data(), n);

    int out_size = 0;
    const float* out = GetFlattenedOutput(&out_size);
    // ... use out[0..out_size)
}
```

## 5. Execution provider behavior

| `cpu_use` | macOS (arm64)          | Windows (x64)            | Linux            |
|-----------|------------------------|--------------------------|------------------|
| `true`    | CPU (multi-threaded)   | CPU (multi-threaded)     | CPU              |
| `false`   | CoreML EP (ANE/GPU/CPU)| DirectML EP (device 0)   | CPU (no EP)      |

If EP append fails for any reason (unsupported ops, missing DirectML.dll, etc.)
the library logs an error and continues with CPU — calls do **not** raise.

## 6. Windows: DirectML.dll note

The release zip does **not** ship `DirectML.dll`. Windows 10 1903+ provides a
system copy at `%SystemRoot%\System32\DirectML.dll`, which is usually sufficient
for onnxruntime 1.18.1. If you hit a version mismatch at runtime, install the
[DirectML redistributable](https://learn.microsoft.com/windows/ai/directml/dml-version-history)
or drop a matching `DirectML.dll` next to your executable.
