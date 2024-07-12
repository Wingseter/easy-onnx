//
// Created by 권용훈 on 7/11/24.
//

#include "Model.h"
#include "../Utils/pch.h"


void Model::setSessionOption(bool cpu_use) {
    // Get CPU Thread count
    int n = static_cast<int>(std::thread::hardware_concurrency());

    // Session Option setting
    session_options.SetIntraOpNumThreads(n / 2);
    session_options.SetInterOpNumThreads(n / 2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // basically use CPU
    if (cpu_use) {
        // if force use cpu
        return;
    }

    // 플랫폼에 따른 실행 프로바이더 설정
#ifdef _WIN32
    // Windows: DirectML
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
#elif defined(__APPLE__)
    // macOS: CoreML
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0));
#else
    // cuda for coincident situation (Linux ?)
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif
}

void Model::setModel(const char * model_path) {
    // ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "airunner");

    session = make_shared<Ort::Session>(env, model_path, session_options);

    setModelInOutput();
    setModelInputTypeDim();
}

void Model::setModelInOutput() {
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input name
    auto input_name_allocated = session->GetInputNameAllocated(0, allocator);
    input_name = std::string(input_name_allocated.get());

    // Get output name
    auto output_name_allocated = session->GetOutputNameAllocated(0, allocator);
    output_name = std::string(output_name_allocated.get());
}


void Model::setModelInputTypeDim() {
    auto input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

    input_dims = input_tensor_info.GetShape();
    input_type = input_tensor_info.GetElementType();
}

