//
// Created by 권용훈 on 7/10/24.
//
#pragma once

#include <vector>
#include <cstdint>

// Model initialization
extern "C" bool InitModel(const char* modelPath, bool cpu_use);

// Single sample inference (synchronous)
extern "C" void RunModelInt(int* data, int num_elements);
extern "C" void RunModelFloat(float* data, int num_elements);
extern "C" void RunModelDouble(double* data, int num_elements);

// Batch inference (synchronous)
extern "C" void RunModelBatchInt(int* data, int batch_size, int elements_per_sample);
extern "C" void RunModelBatchFloat(float* data, int batch_size, int elements_per_sample);
extern "C" void RunModelBatchDouble(double* data, int batch_size, int elements_per_sample);

// Async inference - starts inference in background
extern "C" void RunModelFloatAsync(float* data, int num_elements);
extern "C" void RunModelIntAsync(int* data, int num_elements);
extern "C" void RunModelDoubleAsync(double* data, int num_elements);

// Async batch inference
extern "C" void RunModelBatchFloatAsync(float* data, int batch_size, int elements_per_sample);

// Async inference with callback
typedef void (*InferenceCallbackC)(bool success, const float* output, int output_size, void* user_data);
extern "C" void RunModelFloatAsyncCallback(float* data, int num_elements, InferenceCallbackC callback, void* user_data);

// Async status and control
extern "C" bool IsInferenceRunning();
extern "C" void WaitForInference();
extern "C" bool GetAsyncResult();  // Returns true if last async inference succeeded

// Output retrieval
extern "C" const float* GetFlattenedOutput(int* size);
extern "C" const int64_t* GetOriginalShape(int* size);

// Utility
extern "C" int GetElementsPerSample();
extern "C" bool allCheck(const char* modelPath, bool cpu_use, float* data, int num_elements);

// ============================================
// Multi-Model Management API
// ============================================

// Create/Destroy model instances
extern "C" int CreateModelInstance();                    // Returns instance ID
extern "C" bool DestroyModelInstance(int instance_id);

// Initialize model for specific instance
extern "C" bool InitModelInstance(int instance_id, const char* modelPath, bool cpu_use);

// Run inference on specific instance
extern "C" bool RunModelInstanceFloat(int instance_id, float* data, int num_elements);
extern "C" bool RunModelInstanceInt(int instance_id, int* data, int num_elements);
extern "C" bool RunModelInstanceDouble(int instance_id, double* data, int num_elements);

// Run batch inference on specific instance
extern "C" bool RunModelInstanceBatchFloat(int instance_id, float* data, int batch_size, int elements_per_sample);

// Get output from specific instance
extern "C" const float* GetInstanceFlattenedOutput(int instance_id, int* size);
extern "C" const int64_t* GetInstanceOriginalShape(int instance_id, int* size);

// Instance info
extern "C" int GetInstanceElementsPerSample(int instance_id);
extern "C" bool IsInstanceInitialized(int instance_id);
extern "C" int GetModelInstanceCount();