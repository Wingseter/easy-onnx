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

    if(model_->is_initialized())
        model_->reset_model();

    model_->set_session_option(cpu_use);
    model_->set_model(modelPath);
    Logger::getInstance().info(std::string("Model loaded from path: ") + modelPath);

    model_->set_model_in_output();
    model_->set_model_in_output_type_dim();

    std::vector<int64_t> input_dims = model_->get_input_dims();
    ONNXTensorElementDataType input_type = model_->get_input_type();

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

    data_loader_ = std::make_shared<DataLoader>(model_->get_input_dims(), model_->get_input_type());
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

int Workflow::get_elements_per_sample() const {
    std::vector<int64_t> dims = model_->get_input_dims();
    int elements = 1;
    for (size_t i = 1; i < dims.size(); ++i) {  // Skip batch dimension
        elements *= static_cast<int>(dims[i]);
    }
    return elements;
}

template <typename T>
void Workflow::run_inference(T* data, int num_elements) {
    Ort::Value input_tensor = data_loader_->load_data(data, num_elements);
    model_->run_inference(std::move(input_tensor));
}

template <typename T>
void Workflow::run_batch_inference(T* data, int batch_size, int elements_per_sample) {
    Ort::Value input_tensor = data_loader_->load_batch_data(data, batch_size, elements_per_sample);
    model_->run_inference(std::move(input_tensor));
}

const std::vector<float>& Workflow::get_flattened_output() const {
    return model_->get_flattened_output();
}

const std::vector<int64_t>& Workflow::get_original_shape() const {
    return model_->get_original_shape();
}

// Async inference implementations
template <typename T>
bool Workflow::run_model_async(T* data, int num_elements) {
    std::lock_guard<std::mutex> lock(async_mutex_);
    if (is_running_.load()) {
        Logger::getInstance().warn("Async inference already running; request rejected.");
        return false;
    }
    is_running_.store(true);

    current_future_ = std::async(std::launch::async, [this, data, num_elements]() {
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
    return true;
}

template <typename T>
bool Workflow::run_model_batch_async(T* data, int batch_size, int elements_per_sample) {
    std::lock_guard<std::mutex> lock(async_mutex_);
    if (is_running_.load()) {
        Logger::getInstance().warn("Async inference already running; request rejected.");
        return false;
    }
    is_running_.store(true);

    current_future_ = std::async(std::launch::async, [this, data, batch_size, elements_per_sample]() {
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
    return true;
}

template <typename T>
void Workflow::run_model_async_callback(T* data, int num_elements, InferenceCallback callback) {
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        is_running_.store(true);
    }

    std::thread([this, data, num_elements, callback]() {
        try {
            run_inference(data, num_elements);
            is_running_.store(false);
            if (callback) {
                callback(true, model_->get_flattened_output());
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

void Workflow::wait_for_inference() {
    while (is_running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::lock_guard<std::mutex> lock(async_mutex_);
    if (current_future_.valid()) {
        last_result_.store(current_future_.get());
    }
}

bool Workflow::get_async_result() {
    std::lock_guard<std::mutex> lock(async_mutex_);
    if (current_future_.valid()) {
        last_result_.store(current_future_.get());
    }
    return last_result_.load();
}

// Explicit template instantiations for async methods
template bool Workflow::run_model_async<int>(int*, int);
template bool Workflow::run_model_async<float>(float*, int);
template bool Workflow::run_model_async<double>(double*, int);

template bool Workflow::run_model_batch_async<int>(int*, int, int);
template bool Workflow::run_model_batch_async<float>(float*, int, int);
template bool Workflow::run_model_batch_async<double>(double*, int, int);

template void Workflow::run_model_async_callback<int>(int*, int, InferenceCallback);
template void Workflow::run_model_async_callback<float>(float*, int, InferenceCallback);
template void Workflow::run_model_async_callback<double>(double*, int, InferenceCallback);
