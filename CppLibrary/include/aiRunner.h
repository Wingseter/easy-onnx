//
// Created by 권용훈 on 7/10/24.
//
#pragma once

#include <vector>
#include <cstdint>

// Model initialization
extern "C" bool InitModel(const char* modelPath, bool cpu_use);

// Single sample inference
extern "C" void RunModelInt(int* data, int num_elements);
extern "C" void RunModelFloat(float* data, int num_elements);
extern "C" void RunModelDouble(double* data, int num_elements);

// Batch inference
extern "C" void RunModelBatchInt(int* data, int batch_size, int elements_per_sample);
extern "C" void RunModelBatchFloat(float* data, int batch_size, int elements_per_sample);
extern "C" void RunModelBatchDouble(double* data, int batch_size, int elements_per_sample);

// Output retrieval
extern "C" const float* GetFlattenedOutput(int* size);
extern "C" const int64_t* GetOriginalShape(int* size);

// Utility
extern "C" int GetElementsPerSample();
extern "C" bool allCheck(const char* modelPath, bool cpu_use, float* data, int num_elements);