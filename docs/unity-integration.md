# Unity Integration

`easy-onnx` exposes a C ABI (`extern "C"`) that Unity can P/Invoke directly.
This guide covers plugin placement and a minimal C# wrapper.

## 1. Plugin placement

After extracting the release zip, copy the libraries into your Unity project:

### macOS (Apple Silicon)
```
Assets/Plugins/macOS/
├── libaiRunner.dylib
├── libonnxruntime.1.18.1.dylib
└── libonnxruntime.dylib    # symlink to .1.18.1.dylib
```

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

    [DllImport(Lib)]
    public static extern bool InitModel(string modelPath, bool cpu_use);

    [DllImport(Lib)]
    public static extern void RunModelFloat(float[] data, int num_elements);

    [DllImport(Lib)]
    public static extern IntPtr GetFlattenedOutput(out int size);

    [DllImport(Lib)]
    public static extern int GetElementsPerSample();

    // --- Logging (optional) ---
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
        // cpu_use=false enables CoreML on macOS / DirectML on Windows.
        if (!EasyONNX.InitModel(Application.streamingAssetsPath + "/model.onnx", false))
        {
            Debug.LogError("Failed to load ONNX model");
            return;
        }

        int n = EasyONNX.GetElementsPerSample();
        var input = new float[n];
        EasyONNX.RunModelFloat(input, n);

        IntPtr ptr = EasyONNX.GetFlattenedOutput(out int outSize);
        var output = new float[outSize];
        Marshal.Copy(ptr, output, 0, outSize);

        Debug.Log($"output[0] = {output[0]}");
    }
}
```

## 4. IL2CPP build notes

When building the Standalone player with IL2CPP:
- Confirm Unity automatically copies plugin files into
  `<Build>.app/Contents/Plugins/` (macOS) or `<Build>_Data/Plugins/x86_64/`
  (Windows). If a dylib is missing at runtime, re-check the platform/CPU
  settings in the Inspector.
- macOS distribution to other developers will trigger Gatekeeper warnings
  unless the build is code-signed and notarized. That's out of scope for
  v0.1.x of easy-onnx.

## 5. Async / multi-model API

The C ABI also exposes async inference (`RunModelFloatAsync`,
`WaitForInference`, …) and per-instance variants (`CreateModelInstance`,
`InitModelInstance`, …). Bind them the same way — see `include/aiRunner.h` in
the release zip for the full surface.
