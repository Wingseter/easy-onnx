//
// Created by 권용훈 on 7/10/24.
//
// easy-onnx public C ABI — v0.2.0
//
// Every inference call is handle-based: create an instance, initialize it with
// a model, run inference, read the output, then destroy it. There is no global
// / single-model API anymore (removed in v0.2.0 — see docs/migration-v0.2.0.md).
//
// Conventions:
//   - The first argument of every instance call is the `instance_id` handle
//     returned by CreateModelInstance().
//   - Functions that can fail return `bool` (true = success). An unknown
//     instance_id makes them fail fast and return false.
//   - Output accessors (GetInstance*) return a pointer/value; the pointer stays
//     valid until the next run on that instance or until it is destroyed.
//
#pragma once

#include <cstdint>

// ============================================
// Instance lifecycle
// ============================================

// Create a new model instance. Returns its instance id (always >= 1).
extern "C" int CreateModelInstance();

// Destroy a model instance. Returns false if the id is unknown.
extern "C" bool DestroyModelInstance(int instance_id);

// Load a model into an instance. `cpu_use` true selects the CPU EP; false
// selects the platform accelerator (CoreML on macOS, DirectML on Windows) and
// transparently falls back to CPU when it is unavailable.
extern "C" bool InitModelInstance(int instance_id, const char* modelPath, bool cpu_use);

// ============================================
// Synchronous inference
// ============================================

extern "C" bool RunModelInstanceInt(int instance_id, int* data, int num_elements);
extern "C" bool RunModelInstanceFloat(int instance_id, float* data, int num_elements);
extern "C" bool RunModelInstanceDouble(int instance_id, double* data, int num_elements);

extern "C" bool RunModelInstanceBatchInt(int instance_id, int* data, int batch_size, int elements_per_sample);
extern "C" bool RunModelInstanceBatchFloat(int instance_id, float* data, int batch_size, int elements_per_sample);
extern "C" bool RunModelInstanceBatchDouble(int instance_id, double* data, int batch_size, int elements_per_sample);

// ============================================
// Asynchronous inference
// ============================================

// Start a background inference. Returns true if it was started, false if the
// instance is unknown or an async run is already in flight on it.
extern "C" bool RunModelInstanceIntAsync(int instance_id, int* data, int num_elements);
extern "C" bool RunModelInstanceFloatAsync(int instance_id, float* data, int num_elements);
extern "C" bool RunModelInstanceDoubleAsync(int instance_id, double* data, int num_elements);

extern "C" bool RunModelInstanceBatchIntAsync(int instance_id, int* data, int batch_size, int elements_per_sample);
extern "C" bool RunModelInstanceBatchFloatAsync(int instance_id, float* data, int batch_size, int elements_per_sample);
extern "C" bool RunModelInstanceBatchDoubleAsync(int instance_id, double* data, int batch_size, int elements_per_sample);

// True while a background inference is running on the instance.
extern "C" bool IsInstanceInferenceRunning(int instance_id);

// Block until the background inference on the instance finishes. Returns false
// if the instance is unknown.
extern "C" bool WaitForInstanceInference(int instance_id);

// Result of the most recent async run on the instance (true = succeeded).
extern "C" bool GetInstanceAsyncResult(int instance_id);

// Async inference with a completion callback. The callback runs on the worker
// thread; `output` is valid only for the duration of the call.
typedef void (*InferenceCallbackC)(bool success, const float* output, int output_size, void* user_data);
extern "C" bool RunModelInstanceFloatAsyncCallback(int instance_id, float* data, int num_elements,
                                                   InferenceCallbackC callback, void* user_data);

// ============================================
// Output retrieval
// ============================================

extern "C" const float* GetInstanceFlattenedOutput(int instance_id, int* size);
extern "C" const int64_t* GetInstanceOriginalShape(int instance_id, int* size);

// ============================================
// Instance info
// ============================================

extern "C" int GetInstanceElementsPerSample(int instance_id);
extern "C" bool IsInstanceInitialized(int instance_id);
extern "C" int GetModelInstanceCount();

// ============================================
// Logging API (process-global)
// ============================================

// Log levels: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NONE
extern "C" void SetLogLevel(int level);
extern "C" int GetLogLevel();

// Console output control
extern "C" void EnableLogConsole(bool enable);
extern "C" void EnableLogTimestamp(bool enable);

// File logging
extern "C" bool SetLogFile(const char* filepath);
extern "C" void CloseLogFile();

// Custom callback for log messages
typedef void (*LogCallbackC)(int level, const char* message, void* user_data);
extern "C" void SetLogCallback(LogCallbackC callback, void* user_data);
extern "C" void ClearLogCallback();

// Manual logging
extern "C" void LogDebug(const char* message);
extern "C" void LogInfo(const char* message);
extern "C" void LogWarn(const char* message);
extern "C" void LogError(const char* message);
