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

    model_ = std::make_shared<Model>();
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

    data_loader_ = std::make_shared<DataLoader>(input_dims, input_type);
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

template <typename T>
void Workflow::run_inference(T* data, size_t num_elements) {
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
template void Workflow::run_inference<int>(int*, size_t);
template void Workflow::run_inference<float>(float*, size_t);
template void Workflow::run_inference<double>(double*, size_t);