//
// Created by 권용훈 on 7/10/24.
//

#include "../Utils/pch.h"
#include "../include/Workflow.h"
#include "../include/Model.h"
#include "../include/DataLoader.h"
#include "../include/Utils.h"

void Workflow::init_model(const char* modelPath, bool cpu_use) {
    std::cout << "Initializing model..." << std::endl;

    model_ = new Model();
    std::cout << "Model object created." << std::endl;

    model_->setSessionOption(cpu_use);
    std::cout << "Session options set. CPU use: " << (cpu_use ? "enabled" : "disabled") << std::endl;

    model_->setModel(modelPath);
    std::cout << "Model loaded from path: " << modelPath << std::endl;

    model_->setModelInOutput();
    std::cout << "Model input and output set." << std::endl;

    model_->setModelInOutputTypeDim();
    std::cout << "Model input type and dimensions set." << std::endl;

    std::vector<int64_t> input_dims = model_->getInputDims();
    ONNXTensorElementDataType input_type = model_->getInputType();
    std::cout << "Model input dimensions and type retrieved." << std::endl;

    // input_dims 출력
    std::cout << "Input dimensions: [";
    for (size_t i = 0; i < input_dims.size(); ++i) {
        std::cout << input_dims[i];
        if (i < input_dims.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    // input_type 출력
    std::cout << "Input type: " << Utils::onnxTypeToString(input_type) << std::endl;

    data_loader_ = new DataLoader(input_dims, input_type);
    std::cout << "Data loader initialized with input dimensions and type." << std::endl;

    std::cout << "Model initialization complete." << std::endl;
}

void Workflow::run_model(int* data, int num_elements) {
    run_inference(data, num_elements);
}

void Workflow::run_model(float* data, int num_elements) {
    run_inference(data, num_elements);
}

void Workflow::run_model(double* data, int num_elements) {
    run_inference(data, num_elements);
}

void Workflow::run_test(const char* modelPath, bool cpu_use, float* data, int num_elements) {
    std::cout << "Hello This is ai Running Tester" << std::endl;
    std::vector<int64_t> dimensions = {1, 4, 128, 128, 80};

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ai_running_tester");

    // Create session options and set the graph optimization level
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Create session
    Ort::Session session(env, modelPath, session_options);

    // Get input and output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    std::cout << "Input Name: " << input_name.get() << std::endl;
    std::cout << "Output Name: " << output_name.get() << std::endl;

    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, data, num_elements, dimensions.data(), dimensions.size());

    // Run inference
    std::vector<const char*> input_names = { input_name.get() };
    std::vector<const char*> output_names = { output_name.get() };

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Get output tensor
    Ort::Value& output_tensor = output_tensors.front();
    float* output_data = output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    std::cout << "Output tensor size: " << output_size << std::endl;

}

template <typename T>
void Workflow::run_inference(T* data, int num_elements) {
    Ort::Value input_tensor = data_loader_->load_data(data, num_elements);

    auto output_tensors = model_->runInference(std::move(input_tensor));
}

std::vector<float> Workflow::getFlattenedOutput() const {
    return model_->getFlattenedOutput();
}

std::vector<int64_t> Workflow::getOriginalShape() const {
    return model_->getOriginalShape();
}

// template instanciation
template void Workflow::run_inference<int>(int*, int);
template void Workflow::run_inference<float>(float*, int);
template void Workflow::run_inference<double>(double*, int);