//
// Created by 권용훈 on 7/11/24.
//

#include "../include/Model.h"
#include "../Utils/pch.h"

void Model::SetSessionOption(bool cpu_use) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2; // Fallback to 2 if hardware_concurrency returns 0 (which is implementation-defined)
    }

    // Basically Use CPU
    if (cpu_use) {
        // On other platforms, set session options for using CPU
        session_options.SetIntraOpNumThreads((int) num_threads / 2);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    } else {
#ifdef __APPLE__
        // On macOS, use Core ML if available and specified
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0));
#else
        // Set options for using GPU or other accelerators
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
#endif
    }
}


void Model::setModel(const char * modelPath) {
    // ONNX Runtime environment
    auto envLocal = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "airunner");
    env = std::move(envLocal);
    auto sessionLocal = std::make_unique<Ort::Session>(*env, modelPath, session_options);
    session = std::move(sessionLocal);
}

void Model::setModelInOutput() {
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input, output name
    auto input_name_allocated = session->GetInputNameAllocated(0, allocator);
    auto output_name_allocated = session->GetOutputNameAllocated(0, allocator);

    input_names = { input_name_allocated.get() };
    output_names = { output_name_allocated.get() };
}


void Model::setModelInOutputTypeDim() {
    auto input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

    // get input dimension and type
    input_dims = input_tensor_info.GetShape();
    input_dims[0] = 1;
    input_type = input_tensor_info.GetElementType();

    auto output_type_info = session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

    // get output dimension and type
    output_dims = input_tensor_info.GetShape();
    output_type = input_tensor_info.GetElementType();

}

bool Model::runInference(Ort::Value input_tensor) {
    // Check if the session is valid
    if (!session) {
        std::cerr << "Session is not initialized!" << std::endl;
        return false;
    }
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

    std::cout << "Running inference..." << std::endl;

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                  input_names.data(), &input_tensor, 1,
                                  output_names.data(), 1);

    // Check if we have any output tensors
    if (output_tensors.empty()) {
        std::cout << "Inference failed: no output tensors" << std::endl;
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