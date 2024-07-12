//
// Created by 권용훈 on 7/10/24.
//

#include "../Utils/pch.h"
#include "Workflow.h"
#include "Model.h"
#include "DataLoader.h"

void Workflow::init_model(const char* modelPath) {
    model_ = std::make_shared<Model>();
    model_->setModel(modelPath);
    model_->setModelInOutput();
    model_->setModelInputTypeDim();

    data_loader_ = std::make_shared<DataLoader>(model_);
}

void Workflow::run_model(int* data, size_t num_elements, bool cpu_use) {
    run_inference(data, num_elements, cpu_use);
}

void Workflow::run_model(float* data, size_t num_elements, bool cpu_use) {
    run_inference(data, num_elements, cpu_use);
}

void Workflow::run_model(double* data, size_t num_elements, bool cpu_use) {
    run_inference(data, num_elements, cpu_use);
}

template <typename T>
void Workflow::run_inference(T* data, size_t num_elements, bool cpu_use) {
    model_->setSessionOption(cpu_use);

    Ort::Value input_tensor = data_loader_->load_data(data, num_elements);

    const char* input_names[] = {model_->getInputName().c_str()};
    const char* output_names[] = {model_->getOutputName().c_str()};

    auto output_tensors = model_->run(input_tensor, input_names, output_names);
    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    std::cout << "Output: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;
}

// 템플릿 명시적 인스턴스화
template void Workflow::run_inference<int>(int*, size_t, bool);
template void Workflow::run_inference<float>(float*, size_t, bool);
template void Workflow::run_inference<double>(double*, size_t, bool);