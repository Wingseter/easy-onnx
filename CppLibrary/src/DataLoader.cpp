//
// Created by 권용훈 on 7/11/24.
//
#include "../Utils/pch.h"
#include "../include/DataLoader.h"
#include "../include/Model.h"

DataLoader::DataLoader(const std::vector<int64_t>& input_dims, ONNXTensorElementDataType input_type)
        : input_dims_(input_dims), input_type_(input_type) {}

Ort::Value DataLoader::float_load_data(float *data, int num_elements) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    return Ort::Value::CreateTensor<float>(memory_info, data, num_elements, input_dims_.data(), input_dims_.size());
}

// Template definition
Ort::Value DataLoader::load_data_float(float* data, int num_elements) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    return Ort::Value::CreateTensor<float>(memory_info, data, num_elements, input_dims_.data(), input_dims_.size());
}

