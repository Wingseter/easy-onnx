//
// Created by 권용훈 on 7/10/24.
//
#pragma once

#include <vector>

extern "C" bool allCheck();
extern "C" bool InitModel(const char* modelPath, bool cpu_use);
extern "C" void RunModelInt(int* data, int num_elements);
extern "C" void RunModelFloat(float* data, int  num_elements);
extern "C" void RunModelDouble(double* data, int num_elements);
extern "C" const float* GetFlattenedOutput(int* size);
extern "C" const int64_t* GetOriginalShape(int* size);