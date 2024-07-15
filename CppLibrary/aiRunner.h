//
// Created by 권용훈 on 7/10/24.
//
#pragma once

#include <vector>

extern "C" bool InitModel(const char* modelPath, bool cpu_use);
extern "C" void RunModelInt(int* data, size_t num_elements);
extern "C" void RunModelFloat(float* data, size_t num_elements);
extern "C" void RunModelDouble(double* data, size_t num_elements);
extern "C" const float* GetFlattenedOutput(size_t* size);
extern "C" const int64_t* GetOriginalShape(size_t* size);