//
// Created by 권용훈 on 7/10/24.
//

#include "../Utils/pch.h"
#include "../include/Workflow.h"
#include "../include/Model.h"
#include "../include/DataLoader.h"
#include "../include/Utils.h"
#include "../include/Logger.h"
#include <thread>
#include <sstream>

void Workflow::init_model(const char* modelPath, bool cpu_use) {
    Logger::getInstance().info("Initializing model...");

    // if new model is same as previous one pass model Initialize
    if (std::strcmp(previous.c_str(), modelPath) == 0){
        Logger::getInstance().info("Model already initialized; skipping re-init.");
        return;
    }

    previous = string(modelPath);

    if(model_ == nullptr)
        model_ = std::make_shared<Model>();
    if(data_loader_ != nullptr)
        data_loader_.reset();

    if(model_->isInitialized())
        model_->resetModel();

    model_->setSessionOption(cpu_use);
    model_->setModel(modelPath);
    Logger::getInstance().info(std::string("Model loaded from path: ") + modelPath);

    model_->setModelInOutput();
    model_->setModelInOutputTypeDim();

    std::vector<int64_t> input_dims = model_->getInputDims();
    ONNXTensorElementDataType input_type = model_->getInputType();

    {
        std::ostringstream oss;
        oss << "Input dimensions: [";
        for (size_t i = 0; i < input_dims.size(); ++i) {
            if (i) oss << ", ";
            oss << input_dims[i];
        }
        oss << "] type=" << Utils::onnxTypeToString(input_type);
        Logger::getInstance().info(oss.str());
    }

    data_loader_ = std::make_shared<DataLoader>(model_->getInputDims(), model_->getInputType());
    Logger::getInstance().info("Model initialization complete.");
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

void Workflow::run_model_batch(int* data, int batch_size, int elements_per_sample) {
    run_batch_inference(data, batch_size, elements_per_sample);
}

void Workflow::run_model_batch(float* data, int batch_size, int elements_per_sample) {
    run_batch_inference(data, batch_size, elements_per_sample);
}

void Workflow::run_model_batch(double* data, int batch_size, int elements_per_sample) {
    run_batch_inference(data, batch_size, elements_per_sample);
}

int Workflow::getElementsPerSample() const {
    std::vector<int64_t> dims = model_->getInputDims();
    int elements = 1;
    for (size_t i = 1; i < dims.size(); ++i) {  // Skip batch dimension
        elements *= static_cast<int>(dims[i]);
    }
    return elements;
}

void Workflow::run_test(const char* modelPath, bool cpu_use, float* data, int num_elements) {
    Logger::getInstance().info("run_test: dev path");
    (void)cpu_use;
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

    Logger::getInstance().info(std::string("run_test input='") + input_name.get() +
                               "' output='" + output_name.get() + "'");

    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, data, num_elements, dimensions.data(), dimensions.size());

    // Run inference
    std::vector<const char*> input_names = { input_name.get() };
    std::vector<const char*> output_names = { output_name.get() };

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Get output tensor
    Ort::Value& output_tensor = output_tensors.front();
    (void)output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    Logger::getInstance().info(std::string("run_test output size=") + std::to_string(output_size));
}

template <typename T>
void Workflow::run_inference(T* data, int num_elements) {
    Ort::Value input_tensor = data_loader_->load_data(data, num_elements);
    model_->runInference(std::move(input_tensor));
}

template <typename T>
void Workflow::run_batch_inference(T* data, int batch_size, int elements_per_sample) {
    Ort::Value input_tensor = data_loader_->load_batch_data(data, batch_size, elements_per_sample);
    model_->runInference(std::move(input_tensor));
}

const std::vector<float>& Workflow::getFlattenedOutput() const {
    return model_->getFlattenedOutput();
}

const std::vector<int64_t>& Workflow::getOriginalShape() const {
    return model_->getOriginalShape();
}

// Async inference implementations
template <typename T>
std::future<bool> Workflow::run_model_async(T* data, int num_elements) {
    std::lock_guard<std::mutex> lock(async_mutex_);
    is_running_.store(true);

    return std::async(std::launch::async, [this, data, num_elements]() {
        try {
            run_inference(data, num_elements);
            is_running_.store(false);
            return true;
        } catch (const std::exception& e) {
            Logger::getInstance().error(std::string("Async inference error: ") + e.what());
            is_running_.store(false);
            return false;
        }
    });
}

template <typename T>
std::future<bool> Workflow::run_model_batch_async(T* data, int batch_size, int elements_per_sample) {
    std::lock_guard<std::mutex> lock(async_mutex_);
    is_running_.store(true);

    return std::async(std::launch::async, [this, data, batch_size, elements_per_sample]() {
        try {
            run_batch_inference(data, batch_size, elements_per_sample);
            is_running_.store(false);
            return true;
        } catch (const std::exception& e) {
            Logger::getInstance().error(std::string("Async batch inference error: ") + e.what());
            is_running_.store(false);
            return false;
        }
    });
}

template <typename T>
void Workflow::run_model_async_callback(T* data, int num_elements, InferenceCallback callback) {
    std::lock_guard<std::mutex> lock(async_mutex_);
    is_running_.store(true);

    std::thread([this, data, num_elements, callback]() {
        try {
            run_inference(data, num_elements);
            is_running_.store(false);
            if (callback) {
                callback(true, model_->getFlattenedOutput());
            }
        } catch (const std::exception& e) {
            Logger::getInstance().error(std::string("Async inference error: ") + e.what());
            is_running_.store(false);
            if (callback) {
                callback(false, {});
            }
        }
    }).detach();
}

void Workflow::waitForInference() {
    while (is_running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Explicit template instantiations for async methods
template std::future<bool> Workflow::run_model_async<int>(int*, int);
template std::future<bool> Workflow::run_model_async<float>(float*, int);
template std::future<bool> Workflow::run_model_async<double>(double*, int);

template std::future<bool> Workflow::run_model_batch_async<int>(int*, int, int);
template std::future<bool> Workflow::run_model_batch_async<float>(float*, int, int);
template std::future<bool> Workflow::run_model_batch_async<double>(double*, int, int);

template void Workflow::run_model_async_callback<int>(int*, int, InferenceCallback);
template void Workflow::run_model_async_callback<float>(float*, int, InferenceCallback);
template void Workflow::run_model_async_callback<double>(double*, int, InferenceCallback);
