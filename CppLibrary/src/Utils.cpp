//
// Created by 권용훈 on 8/2/24.
//

#include "../include/Utils.h"
#include "../Utils/pch.h"

std::string Utils::onnxTypeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "FLOAT";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "UINT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "INT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "UINT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "INT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "INT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "INT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "STRING";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "BOOL";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "FLOAT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "DOUBLE";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "UINT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "UINT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "COMPLEX64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "COMPLEX128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "BFLOAT16";
        default: return "UNKNOWN";
    }
}