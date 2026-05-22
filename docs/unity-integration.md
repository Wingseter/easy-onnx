# Unity Integration

`easy-onnx` exposes a C ABI (`extern "C"`) that Unity can P/Invoke directly.
This guide covers plugin placement and a minimal C# wrapper.

> **v0.2.0 is a breaking release.** The global single-model API was removed; the
> ABI is now handle-based (every call takes an `instance_id`). If you are
> upgrading a v0.1.x binding, see [migration-v0.2.0.md](migration-v0.2.0.md).

## 1. Plugin placement

After extracting the release zip, copy the libraries into your Unity project:

### macOS (Apple Silicon)
```
Assets/Plugins/macOS/
├── libaiRunner.dylib
├── libonnxruntime.1.26.0.dylib
├── libonnxruntime.1.dylib    # symlink to .1.26.0.dylib
└── libonnxruntime.dylib      # symlink to .1.26.0.dylib
```

`libaiRunner.dylib` resolves its ONNX Runtime dependency as
`@rpath/libonnxruntime.1.dylib`, so the `.1.dylib` entry must be present —
keep all three. Extract the release zip with symlinks preserved (`unzip`
keeps them; the bundled `build_release.sh` zips with `--symlinks`).

In the Inspector for each dylib:
- **Select platforms for plugin**: macOS only
- **CPU**: ARM64 (Apple silicon)

If Unity refuses to load the plugin with a quarantine error, strip the attribute
on the whole folder:

```sh
xattr -dr com.apple.quarantine Assets/Plugins/macOS
```

### Windows (x64)
```
Assets/Plugins/x86_64/
├── aiRunner.dll
└── onnxruntime.dll
```

In the Inspector:
- **Select platforms for plugin**: Windows only
- **CPU**: x86_64

## 2. C# binding

```csharp
using System;
using System.Runtime.InteropServices;

public static class EasyONNX
{
    private const string Lib = "aiRunner"; // Unity strips lib prefix on macOS

    // --- Instance lifecycle ---
    [DllImport(Lib)]
    public static extern int CreateModelInstance();

    [DllImport(Lib)]
    public static extern bool DestroyModelInstance(int instanceId);

    [DllImport(Lib)]
    public static extern bool InitModelInstance(int instanceId, string modelPath, bool cpuUse);

    // --- Inference ---
    [DllImport(Lib)]
    public static extern bool RunModelInstanceFloat(int instanceId, float[] data, int numElements);

    [DllImport(Lib)]
    public static extern IntPtr GetInstanceFlattenedOutput(int instanceId, out int size);

    [DllImport(Lib)]
    public static extern int GetInstanceElementsPerSample(int instanceId);

    // --- Logging (process-global, optional) ---
    [DllImport(Lib)] public static extern void SetLogLevel(int level);
    [DllImport(Lib)] public static extern void EnableLogConsole(bool enable);
}
```

## 3. Usage in a MonoBehaviour

```csharp
using UnityEngine;

public class OnnxDemo : MonoBehaviour
{
    void Start()
    {
        int h = EasyONNX.CreateModelInstance();

        // cpuUse=false enables CoreML on macOS / DirectML on Windows.
        if (!EasyONNX.InitModelInstance(h, Application.streamingAssetsPath + "/model.onnx", false))
        {
            Debug.LogError("Failed to load ONNX model");
            return;
        }

        int n = EasyONNX.GetInstanceElementsPerSample(h);
        var input = new float[n];
        EasyONNX.RunModelInstanceFloat(h, input, n);

        IntPtr ptr = EasyONNX.GetInstanceFlattenedOutput(h, out int outSize);
        var output = new float[outSize];
        Marshal.Copy(ptr, output, 0, outSize);

        Debug.Log($"output[0] = {output[0]}");

        EasyONNX.DestroyModelInstance(h);
    }
}
```

Keep the handle (`h`) in a field if you run inference repeatedly — create it
once (e.g. in `Awake`) and `DestroyModelInstance` it in `OnDestroy`, rather than
per-frame.

## 4. IL2CPP build notes

When building the Standalone player with IL2CPP:
- Confirm Unity automatically copies plugin files into
  `<Build>.app/Contents/Plugins/` (macOS) or `<Build>_Data/Plugins/x86_64/`
  (Windows). If a dylib is missing at runtime, re-check the platform/CPU
  settings in the Inspector.
- macOS distribution to other developers will trigger Gatekeeper warnings
  unless the build is code-signed and notarized. That's out of scope for
  v0.2.x of easy-onnx.

## 5. Async / batch / multi-model

Everything is handle-based, so multiple models loaded at once is just multiple
handles. The C ABI also exposes:

- **Async inference**: `RunModelInstanceFloatAsync`,
  `IsInstanceInferenceRunning`, `WaitForInstanceInference`,
  `GetInstanceAsyncResult`, and a callback form `RunModelInstanceFloatAsyncCallback`.
- **Batch inference**: `RunModelInstanceBatchFloat` and its int/double/async
  variants.

Bind them the same way (handle first, `bool` return) — see `include/aiRunner.h`
in the release zip for the full surface.
