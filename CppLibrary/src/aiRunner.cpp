//
// Created by 권용훈 on 7/10/24.
//

#include "../Utils/pch.h"
#include "../include/aiRunner.h"
#include "../include/Workflow.h"
#include "../include/ModelManager.h"
#include <string.h>
#include <future>
#include <atomic>

static std::shared_ptr<Workflow> workflow = nullptr;
static std::vector<float> flattened_output;
static std::vector<int64_t> original_shape;

// Async state
static std::future<bool> async_future;
static std::atomic<bool> async_result{false};

extern "C" bool allCheck(const char* modelPath, bool cpu_use, float* data, int num_elements) {
    if (!workflow) {
        std::cerr << "Workflow not initialized. Call InitModel first." << std::endl;
        return false;
    }

    try {
        workflow->run_test(modelPath, cpu_use, data, num_elements);
    } catch (const std::exception& e) {
        std::cerr << "Error running test: " << e.what() << std::endl;
        return false;
    }

    return true;
}

extern "C" bool InitModel(const char* modelPath, bool cpu_use) {
    try {
        workflow = std::make_shared<Workflow>();
        workflow->init_model(modelPath, cpu_use);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing model: " << e.what() << std::endl;
        workflow = nullptr;
        return false;
    }
}

extern "C" void RunModelInt(int* data, int num_elements) {
    if (workflow) {
        workflow->run_model(data, num_elements);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" void RunModelFloat(float* data, int num_elements) {
    if (workflow) {
        workflow->run_model(data, num_elements);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" void RunModelDouble(double* data, int num_elements) {
    if (workflow) {
        workflow->run_model(data, num_elements);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" const float* GetFlattenedOutput(int* size) {
    if (size) {
        *size = static_cast<int>(flattened_output.size());
    }
    return flattened_output.data();
}

extern "C" const int64_t* GetOriginalShape(int* size) {
    if (size) {
        *size = static_cast<int>(original_shape.size());
    }
    return original_shape.data();
}

// Batch inference APIs
extern "C" void RunModelBatchInt(int* data, int batch_size, int elements_per_sample) {
    if (workflow) {
        workflow->run_model_batch(data, batch_size, elements_per_sample);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" void RunModelBatchFloat(float* data, int batch_size, int elements_per_sample) {
    if (workflow) {
        workflow->run_model_batch(data, batch_size, elements_per_sample);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" void RunModelBatchDouble(double* data, int batch_size, int elements_per_sample) {
    if (workflow) {
        workflow->run_model_batch(data, batch_size, elements_per_sample);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" int GetElementsPerSample() {
    if (workflow) {
        return workflow->getElementsPerSample();
    }
    return 0;
}

// Async inference implementations
extern "C" void RunModelFloatAsync(float* data, int num_elements) {
    if (workflow) {
        async_future = workflow->run_model_async(data, num_elements);
    }
}

extern "C" void RunModelIntAsync(int* data, int num_elements) {
    if (workflow) {
        async_future = workflow->run_model_async(data, num_elements);
    }
}

extern "C" void RunModelDoubleAsync(double* data, int num_elements) {
    if (workflow) {
        async_future = workflow->run_model_async(data, num_elements);
    }
}

extern "C" void RunModelBatchFloatAsync(float* data, int batch_size, int elements_per_sample) {
    if (workflow) {
        async_future = workflow->run_model_batch_async(data, batch_size, elements_per_sample);
    }
}

extern "C" void RunModelFloatAsyncCallback(float* data, int num_elements, InferenceCallbackC callback, void* user_data) {
    if (workflow) {
        workflow->run_model_async_callback(data, num_elements,
            [callback, user_data](bool success, const std::vector<float>& output) {
                if (callback) {
                    // Update static vectors for GetFlattenedOutput
                    flattened_output = output;
                    callback(success, output.data(), static_cast<int>(output.size()), user_data);
                }
            });
    }
}

extern "C" bool IsInferenceRunning() {
    if (workflow) {
        return workflow->isInferenceRunning();
    }
    return false;
}

extern "C" void WaitForInference() {
    if (workflow) {
        workflow->waitForInference();
        // Update output after waiting
        if (async_future.valid()) {
            async_result.store(async_future.get());
            flattened_output = workflow->getFlattenedOutput();
            original_shape = workflow->getOriginalShape();
        }
    }
}

extern "C" bool GetAsyncResult() {
    // If future is still valid, wait for it and get result
    if (async_future.valid()) {
        async_result.store(async_future.get());
        if (workflow) {
            flattened_output = workflow->getFlattenedOutput();
            original_shape = workflow->getOriginalShape();
        }
    }
    return async_result.load();
}

// ============================================
// Multi-Model Management API Implementation
// ============================================

// Static storage for instance outputs (to maintain pointer validity)
static std::map<int, std::vector<float>> instance_outputs;
static std::map<int, std::vector<int64_t>> instance_shapes;

extern "C" int CreateModelInstance() {
    return ModelManager::getInstance().createInstance();
}

extern "C" bool DestroyModelInstance(int instance_id) {
    // Clean up cached outputs
    instance_outputs.erase(instance_id);
    instance_shapes.erase(instance_id);
    return ModelManager::getInstance().destroyInstance(instance_id);
}

extern "C" bool InitModelInstance(int instance_id, const char* modelPath, bool cpu_use) {
    return ModelManager::getInstance().initModel(instance_id, modelPath, cpu_use);
}

extern "C" bool RunModelInstanceFloat(int instance_id, float* data, int num_elements) {
    bool result = ModelManager::getInstance().runModel(instance_id, data, num_elements);
    if (result) {
        instance_outputs[instance_id] = ModelManager::getInstance().getFlattenedOutput(instance_id);
        instance_shapes[instance_id] = ModelManager::getInstance().getOriginalShape(instance_id);
    }
    return result;
}

extern "C" bool RunModelInstanceInt(int instance_id, int* data, int num_elements) {
    bool result = ModelManager::getInstance().runModel(instance_id, data, num_elements);
    if (result) {
        instance_outputs[instance_id] = ModelManager::getInstance().getFlattenedOutput(instance_id);
        instance_shapes[instance_id] = ModelManager::getInstance().getOriginalShape(instance_id);
    }
    return result;
}

extern "C" bool RunModelInstanceDouble(int instance_id, double* data, int num_elements) {
    bool result = ModelManager::getInstance().runModel(instance_id, data, num_elements);
    if (result) {
        instance_outputs[instance_id] = ModelManager::getInstance().getFlattenedOutput(instance_id);
        instance_shapes[instance_id] = ModelManager::getInstance().getOriginalShape(instance_id);
    }
    return result;
}

extern "C" bool RunModelInstanceBatchFloat(int instance_id, float* data, int batch_size, int elements_per_sample) {
    bool result = ModelManager::getInstance().runModelBatch(instance_id, data, batch_size, elements_per_sample);
    if (result) {
        instance_outputs[instance_id] = ModelManager::getInstance().getFlattenedOutput(instance_id);
        instance_shapes[instance_id] = ModelManager::getInstance().getOriginalShape(instance_id);
    }
    return result;
}

extern "C" const float* GetInstanceFlattenedOutput(int instance_id, int* size) {
    auto it = instance_outputs.find(instance_id);
    if (it != instance_outputs.end()) {
        if (size) {
            *size = static_cast<int>(it->second.size());
        }
        return it->second.data();
    }
    if (size) *size = 0;
    return nullptr;
}

extern "C" const int64_t* GetInstanceOriginalShape(int instance_id, int* size) {
    auto it = instance_shapes.find(instance_id);
    if (it != instance_shapes.end()) {
        if (size) {
            *size = static_cast<int>(it->second.size());
        }
        return it->second.data();
    }
    if (size) *size = 0;
    return nullptr;
}

extern "C" int GetInstanceElementsPerSample(int instance_id) {
    return ModelManager::getInstance().getElementsPerSample(instance_id);
}

extern "C" bool IsInstanceInitialized(int instance_id) {
    return ModelManager::getInstance().isInitialized(instance_id);
}

extern "C" int GetModelInstanceCount() {
    return ModelManager::getInstance().getInstanceCount();
}