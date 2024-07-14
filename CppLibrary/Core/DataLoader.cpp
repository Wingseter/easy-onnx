//
// Created by 권용훈 on 7/11/24.
//
#include "../Utils/pch.h"
#include "DataLoader.h"
#include "Model.h"

DataLoader::DataLoader(const std::vector<int64_t>& input_dims, ONNXTensorElementDataType input_type)
        : input_dims_(input_dims), input_type_(input_type) {}


template <typename T>
Ort::Value DataLoader::load_data(T* data, size_t num_elements) {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    switch (input_type_) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(data), num_elements, input_dims_.data(), input_dims_.size());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return Ort::Value::CreateTensor<double>(memory_info, reinterpret_cast<double*>(data), num_elements, input_dims_.data(), input_dims_.size());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return Ort::Value::CreateTensor<int32_t>(memory_info, reinterpret_cast<int32_t*>(data), num_elements, input_dims_.data(), input_dims_.size());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return Ort::Value::CreateTensor<int64_t>(memory_info, reinterpret_cast<int64_t*>(data), num_elements, input_dims_.data(), input_dims_.size());
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

// Template Instantianion
template Ort::Value DataLoader::load_data<int>(int*, size_t);
template Ort::Value DataLoader::load_data<float>(float*, size_t);
template Ort::Value DataLoader::load_data<double>(double*, size_t);