//
// Created by 권용훈 on 7/11/24.
//

#include "../include/Model.h"
#include "../Utils/pch.h"


void Model::setSessionOption(bool cpu_use) {
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


void Model::setModelInOutputTypeDim() {
    auto input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

    input_dims = input_tensor_info.GetShape();
    input_dims[0] = 1;
    input_type = input_tensor_info.GetElementType();

    auto output_type_info = session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

    output_dims = input_tensor_info.GetShape();
    output_type = input_tensor_info.GetElementType();

}

bool Model::runInference(Ort::Value input_tensor) {
    // Check if the session is valid
    if (!session) {
        std::cerr << "Session is not initialized!" << std::endl;
        return false;
    }

    std::cout << "Session is valid. Starting inference..." << std::endl;

    // Prepare input and output names as const char* arrays
    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    std::cout << "Input name: " << input_names[0] << std::endl;
    std::cout << "Output name: " << output_names[0] << std::endl;

    // Validate input tensor
    std::cout << "Validating input tensor..." << std::endl;
    auto input_tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    auto input_shape = input_tensor_info.GetShape();
    std::cout << "Input tensor shape: ";
    for (const auto& dim : input_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Run inference
    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 1);

    // Check if we have any output tensors
    if (output_tensors.empty()) {
        std::cerr << "Inference failed: no output tensors" << std::endl;
        return false;
    }

    std::cout << "Output tensors received." << std::endl;

    Ort::Value& output_tensor = output_tensors.front();
    auto output_data = output_tensor.GetTensorMutableData<float>();
    original_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    std::cout << "Output tensor shape: ";
    for (const auto& dim : original_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    std::cout << "Output tensor size: " << output_size << std::endl;

    flattened_output = std::vector<float>(output_data, output_data + output_size);

    std::cout << "Flattened output size: " << flattened_output.size() << std::endl;

    return true;
}

std::vector<float> Model::getFlattenedOutput() const {
    return flattened_output;
}

std::vector<int64_t> Model::getOriginalShape() const {
    return original_shape;
}