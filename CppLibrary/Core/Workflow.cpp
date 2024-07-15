//
// Created by 권용훈 on 7/10/24.
//

#include "../Utils/pch.h"
#include "Workflow.h"
#include "Model.h"
#include "DataLoader.h"

void Workflow::init_model(const char* modelPath, bool cpu_use) {
    model_ = std::make_shared<Model>();
    model_->setSessionOption(cpu_use);
    model_->setModel(modelPath);
    model_->setModelInOutput();
    model_->setModelInputTypeDim();

    std::vector<int64_t> input_dims = model_->getInputDims();
    ONNXTensorElementDataType input_type = model_->getInputType();
    data_loader_ = std::make_shared<DataLoader>(input_dims, input_type);
}

void Workflow::run_model(int* data, size_t num_elements) {
    run_inference(data, num_elements);
}

void Workflow::run_model(float* data, size_t num_elements) {
    run_inference(data, num_elements);
}

void Workflow::run_model(double* data, size_t num_elements) {
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