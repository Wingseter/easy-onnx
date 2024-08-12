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
template <typename T>
Ort::Value DataLoader::load_data(T* data, int num_elements) {
    std::cout << "Starting load_data function" << std::endl;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::cout << "Input type: " << input_type_ << std::endl;
    std::cout << "Number of elements: " << num_elements << std::endl;
    std::cout << "Input dimensions: ";
    for (auto dim : input_dims_) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    switch (input_type_) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            std::cout << "Creating tensor of type float" << std::endl;
            return Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float*>(data), num_elements, input_dims_.data(), input_dims_.size());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            std::cout << "Creating tensor of type double" << std::endl;
            return Ort::Value::CreateTensor<double>(memory_info, reinterpret_cast<double*>(data), num_elements, input_dims_.data(), input_dims_.size());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            std::cout << "Creating tensor of type int32" << std::endl;
            return Ort::Value::CreateTensor<int32_t>(memory_info, reinterpret_cast<int32_t*>(data), num_elements, input_dims_.data(), input_dims_.size());
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            std::cout << "Creating tensor of type int64" << std::endl;
            return Ort::Value::CreateTensor<int64_t>(memory_info, reinterpret_cast<int64_t*>(data), num_elements, input_dims_.data(), input_dims_.size());
        default:
            std::cerr << "Unsupported data type" << std::endl;
            throw std::invalid_argument("Unsupported data type");
    }
}

// Template Instantiation
template Ort::Value DataLoader::load_data<int>(int*, int);
template Ort::Value DataLoader::load_data<float>(float*, int);
template Ort::Value DataLoader::load_data<double>(double*, int);
