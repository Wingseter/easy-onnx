# EasyONNX

A small C++ library that wraps [ONNX Runtime](https://onnxruntime.ai) behind a
flat C ABI so you can load and run ONNX models from C++, Unity, or any other
runtime that can P/Invoke a shared library.

## Platform support

| Platform        | Shared library            | Accelerated EP (`cpu_use=false`) |
|-----------------|---------------------------|----------------------------------|
| macOS (arm64)   | `libaiRunner.dylib`       | CoreML                           |
| Windows (x64)   | `aiRunner.dll`            | DirectML                         |

`cpu_use=true` runs on a multi-threaded CPU EP on both platforms. If an
accelerator is unavailable or the model is incompatible, the library logs a
warning and silently falls back to CPU — calls never throw across the C ABI.

## Compatibility

> **v0.2.0 is a breaking release.** The global single-model API (`InitModel`,
> `RunModelFloat`, `GetFlattenedOutput`, …) was **removed**. The public C ABI is
> now a single handle-based surface: every call takes an `instance_id` from
> `CreateModelInstance()`. See
> [docs/migration-v0.2.0.md](docs/migration-v0.2.0.md) for the full old→new
> function mapping.

This release also bundles ONNX Runtime **1.26.0** (was 1.18.1 in v0.1.x).

## Quick start (consume a release zip)

1. Download a release zip from the Releases page.
2. Extract it. Layout:
   ```
   easyonnx-mac-arm64/
     lib/      libaiRunner.dylib, libonnxruntime.*.dylib
     include/  aiRunner.h, ...
   ```
3. (macOS only) Strip Gatekeeper quarantine:
   ```sh
   xattr -dr com.apple.quarantine /path/to/easyonnx-mac-arm64
   ```
4. Link against `aiRunner` and `onnxruntime`. See [docs/cpp-integration.md](docs/cpp-integration.md)
   for a complete CMake consumer example. For Unity, see
   [docs/unity-integration.md](docs/unity-integration.md).

## Public API (highlights)

Every model lives behind an `int` handle. Create it, load a model, run, read,
destroy:

```cpp
extern "C" int  CreateModelInstance();
extern "C" bool InitModelInstance(int instance_id, const char* modelPath, bool cpu_use);
extern "C" bool RunModelInstanceFloat(int instance_id, float* data, int num_elements);
extern "C" const float* GetInstanceFlattenedOutput(int instance_id, int* size);
extern "C" int  GetInstanceElementsPerSample(int instance_id);
extern "C" bool DestroyModelInstance(int instance_id);

// Batch, async (future + callback), int/double variants, and a process-global
// logging API also exist — see include/aiRunner.h in the release zip for the
// full surface.
```

```cpp
#include "aiRunner.h"

int h = CreateModelInstance();
InitModelInstance(h, "model.onnx", /*cpu_use=*/false);

int n = GetInstanceElementsPerSample(h);
std::vector<float> input(n, 0.0f);
RunModelInstanceFloat(h, input.data(), n);

int out_size = 0;
const float* out = GetInstanceFlattenedOutput(h, &out_size);
// ... use out[0..out_size)

DestroyModelInstance(h);
```

## Build from source

```sh
git clone <repo>
cd easy-onnx
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Outputs:
- `build/CppLibrary/libaiRunner.dylib` (or `.dll`)
- `build/CppApplication/CppApplication` — a smoke-test CLI:
  ```sh
  ./build/CppApplication/CppApplication path/to/model.onnx          # CPU
  ./build/CppApplication/CppApplication path/to/model.onnx --gpu    # CoreML/DirectML
  ```

## Cut a release zip

On a mac (Apple Silicon):
```sh
./scripts/build_release.sh
# → easyonnx-mac-arm64-v0.2.0.zip
```

On Windows:
```powershell
.\scripts\build_release.ps1
# → easyonnx-windows-x64-v0.2.0.zip
```

Each script does a clean Release build, runs `cmake --install` into a staging
directory, and zips the result. The two zips are produced independently on
their respective hosts — there is no CI yet.

## Repository layout

```
easy-onnx/
├── CppLibrary/          # The aiRunner shared library (source of the public ABI)
├── CppApplication/      # CLI smoke-test app
├── Library/
│   ├── ONNXRuntime/     # Vendored ORT 1.26.0 (mac arm64, windows x64)
│   └── aiRunner/        # Build output staging
├── docs/                # Integration guides
├── scripts/             # build_release.{sh,ps1}
└── debugging/           # Standalone ORT scratch project (disabled by default)
```

## Out of scope (for now)

- CI / automated releases (each platform's zip is built manually).
- Code signing & notarization (mac zips trigger Gatekeeper on first use —
  see the quarantine workaround above).
- Linux / Intel mac / Universal builds.
- DirectML.dll bundling — Windows users rely on the system copy
  (`%SystemRoot%\System32\DirectML.dll`).
