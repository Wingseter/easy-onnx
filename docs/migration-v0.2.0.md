# Migration guide — v0.1.x → v0.2.0

v0.2.0 is a **breaking release**. The public C ABI was unified into a single
handle-based surface and the global single-model API was removed.

If you only ever used the multi-model API (`CreateModelInstance`,
`RunModelInstanceFloat`, …), you have **almost nothing to change** — those names
are unchanged. If you used the global API (`InitModel`, `RunModelFloat`, …), see
the mapping below.

## Why it changed

v0.1.x shipped two parallel APIs for the same job:

- a **global-state** single-model API (`InitModel` + `RunModelFloat` + …) that
  kept the model in a hidden process-global, returned `void`, and could not be
  used for more than one model at a time;
- a **handle-based** multi-model API (`CreateModelInstance` + `InitModelInstance`
  + …) that returned `bool` and supported any number of models.

Maintaining both meant duplicated state and an unclear "which one do I use?".
v0.2.0 keeps only the handle-based form.

## Core concept change: handles and lifetime

Every model now lives behind an `int` handle:

1. `int h = CreateModelInstance();` — allocate a handle.
2. `InitModelInstance(h, "model.onnx", cpu_use);` — load a model into it.
3. Run inference with the `RunModelInstance*` calls, passing `h` first.
4. Read results with `GetInstance*` accessors, passing `h` first.
5. `DestroyModelInstance(h);` — free it when done.

Rules:

- Every inference/accessor call takes the handle as its **first argument**.
- Action functions return `bool` (`true` = success). An unknown handle makes
  them fail fast and return `false` instead of crashing.
- Output pointers from `GetInstanceFlattenedOutput` / `GetInstanceOriginalShape`
  stay valid until the next run on that handle or until it is destroyed.
- You must call `DestroyModelInstance` yourself — there is no implicit cleanup
  of the old global slot anymore.

## Function mapping (global API → handle API)

Assume `int h` is a handle from `CreateModelInstance()`.

| v0.1.x (removed)                              | v0.2.0                                                       |
|-----------------------------------------------|--------------------------------------------------------------|
| `InitModel(path, cpu_use)`                    | `h = CreateModelInstance(); InitModelInstance(h, path, cpu_use)` |
| `RunModelInt(data, n)`                        | `RunModelInstanceInt(h, data, n)`                            |
| `RunModelFloat(data, n)`                      | `RunModelInstanceFloat(h, data, n)`                          |
| `RunModelDouble(data, n)`                     | `RunModelInstanceDouble(h, data, n)`                         |
| `RunModelBatchInt(data, b, e)`                | `RunModelInstanceBatchInt(h, data, b, e)`                    |
| `RunModelBatchFloat(data, b, e)`              | `RunModelInstanceBatchFloat(h, data, b, e)`                  |
| `RunModelBatchDouble(data, b, e)`             | `RunModelInstanceBatchDouble(h, data, b, e)`                 |
| `RunModelIntAsync(data, n)`                   | `RunModelInstanceIntAsync(h, data, n)`                       |
| `RunModelFloatAsync(data, n)`                 | `RunModelInstanceFloatAsync(h, data, n)`                     |
| `RunModelDoubleAsync(data, n)`                | `RunModelInstanceDoubleAsync(h, data, n)`                    |
| `RunModelBatchFloatAsync(data, b, e)`         | `RunModelInstanceBatchFloatAsync(h, data, b, e)`             |
| `RunModelFloatAsyncCallback(data, n, cb, ud)` | `RunModelInstanceFloatAsyncCallback(h, data, n, cb, ud)`     |
| `IsInferenceRunning()`                        | `IsInstanceInferenceRunning(h)`                              |
| `WaitForInference()`                          | `WaitForInstanceInference(h)`                                |
| `GetAsyncResult()`                            | `GetInstanceAsyncResult(h)`                                  |
| `GetFlattenedOutput(&size)`                   | `GetInstanceFlattenedOutput(h, &size)`                       |
| `GetOriginalShape(&size)`                     | `GetInstanceOriginalShape(h, &size)`                         |
| `GetElementsPerSample()`                      | `GetInstanceElementsPerSample(h)`                            |
| `allCheck(...)`                               | removed — was a dev-only scratch helper, no replacement      |
| —                                             | `DestroyModelInstance(h)` — **new, you must call it**        |

New in v0.2.0 (no v0.1.x equivalent): `RunModelInstanceBatchIntAsync`,
`RunModelInstanceBatchDoubleAsync`.

## Unchanged

These multi-model functions keep the exact same name and signature — code using
them needs no change:

`CreateModelInstance`, `DestroyModelInstance`, `InitModelInstance`,
`RunModelInstanceInt/Float/Double`, `RunModelInstanceBatchFloat`,
`GetInstanceFlattenedOutput`, `GetInstanceOriginalShape`,
`GetInstanceElementsPerSample`, `IsInstanceInitialized`, `GetModelInstanceCount`.

The entire **logging API** (`SetLogLevel`, `EnableLogConsole`,
`EnableLogTimestamp`, `SetLogFile`, `CloseLogFile`, `SetLogCallback`,
`ClearLogCallback`, `LogDebug/Info/Warn/Error`) is process-global and unchanged.

## Behavior changes to note

- **Return types**: the single-model run calls used to return `void`; the
  handle calls return `bool`. Old code that ignored the return value still
  compiles — you can now check it.
- **Async rejects concurrency**: starting an async run while one is already in
  flight on the same handle now returns `false` instead of overwriting the
  in-flight task (which could stall). Wait for the previous run to finish first.
- **Async result**: read the success flag with `GetInstanceAsyncResult(h)` after
  `WaitForInstanceInference(h)` or after the `IsInstanceInferenceRunning(h)` loop
  reports completion.

## Before / after

```cpp
// ---- v0.1.x ----
InitModel("model.onnx", /*cpu_use=*/false);
int n = GetElementsPerSample();
std::vector<float> input(n, 0.0f);
RunModelFloat(input.data(), n);
int out_size = 0;
const float* out = GetFlattenedOutput(&out_size);
// ... use out

// ---- v0.2.0 ----
int h = CreateModelInstance();
InitModelInstance(h, "model.onnx", /*cpu_use=*/false);
int n = GetInstanceElementsPerSample(h);
std::vector<float> input(n, 0.0f);
RunModelInstanceFloat(h, input.data(), n);
int out_size = 0;
const float* out = GetInstanceFlattenedOutput(h, &out_size);
// ... use out
DestroyModelInstance(h);
```

## C# / Unity

The same mapping applies to P/Invoke declarations. Update each `[DllImport]`
signature to the v0.2.0 name, add the leading `int instance_id` parameter, and
change the single-model run declarations from `void` to `bool` if you want to
check success. See [unity-integration.md](unity-integration.md) for a full
updated binding.
