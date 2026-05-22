//
// Created by 권용훈 on 7/10/24.
//
// Thin C ABI adapters. All state lives in ModelManager (model instances,
// default-instance bookkeeping, async futures) or in Logger (log callback
// bridge). No file-scope statics here.

#include "../Utils/pch.h"
#include "../include/aiRunner.h"
#include "../include/ModelManager.h"
#include "../include/Logger.h"

namespace {
inline ModelManager& mgr() { return ModelManager::getInstance(); }
}

// =====================================================================
// Single-model (legacy) API — routed to ModelManager's default instance.
// =====================================================================

extern "C" bool allCheck(const char* modelPath, bool cpu_use, float* data, int num_elements) {
    return mgr().runDefaultTest(modelPath, cpu_use, data, num_elements);
}

extern "C" bool InitModel(const char* modelPath, bool cpu_use) {
    mgr().resetDefault();
    return mgr().initDefault(modelPath, cpu_use);
}

extern "C" void RunModelInt(int* data, int num_elements) {
    mgr().runDefault(data, num_elements);
}

extern "C" void RunModelFloat(float* data, int num_elements) {
    mgr().runDefault(data, num_elements);
}

extern "C" void RunModelDouble(double* data, int num_elements) {
    mgr().runDefault(data, num_elements);
}

extern "C" const float* GetFlattenedOutput(int* size) {
    const auto& out = mgr().getDefaultFlattenedOutput();
    if (size) *size = static_cast<int>(out.size());
    return out.data();
}

extern "C" const int64_t* GetOriginalShape(int* size) {
    const auto& shape = mgr().getDefaultOriginalShape();
    if (size) *size = static_cast<int>(shape.size());
    return shape.data();
}

// Batch inference APIs
extern "C" void RunModelBatchInt(int* data, int batch_size, int elements_per_sample) {
    mgr().runDefaultBatch(data, batch_size, elements_per_sample);
}

extern "C" void RunModelBatchFloat(float* data, int batch_size, int elements_per_sample) {
    mgr().runDefaultBatch(data, batch_size, elements_per_sample);
}

extern "C" void RunModelBatchDouble(double* data, int batch_size, int elements_per_sample) {
    mgr().runDefaultBatch(data, batch_size, elements_per_sample);
}

extern "C" int GetElementsPerSample() {
    return mgr().getDefaultElementsPerSample();
}

// Async inference
extern "C" void RunModelFloatAsync(float* data, int num_elements) {
    mgr().runDefaultAsync(data, num_elements);
}

extern "C" void RunModelIntAsync(int* data, int num_elements) {
    mgr().runDefaultAsync(data, num_elements);
}

extern "C" void RunModelDoubleAsync(double* data, int num_elements) {
    mgr().runDefaultAsync(data, num_elements);
}

extern "C" void RunModelBatchFloatAsync(float* data, int batch_size, int elements_per_sample) {
    mgr().runDefaultBatchAsync(data, batch_size, elements_per_sample);
}

extern "C" void RunModelFloatAsyncCallback(float* data, int num_elements, InferenceCallbackC callback, void* user_data) {
    mgr().runDefaultAsyncCallback(data, num_elements,
        [callback, user_data](bool success, const std::vector<float>& output) {
            if (callback) {
                callback(success, output.data(), static_cast<int>(output.size()), user_data);
            }
        });
}

extern "C" bool IsInferenceRunning() {
    return mgr().isDefaultRunning();
}

extern "C" void WaitForInference() {
    mgr().waitForDefault();
}

extern "C" bool GetAsyncResult() {
    return mgr().getDefaultAsyncResult();
}

// =====================================================================
// Multi-Model Management API
// =====================================================================

extern "C" int CreateModelInstance() {
    return mgr().createInstance();
}

extern "C" bool DestroyModelInstance(int instance_id) {
    return mgr().destroyInstance(instance_id);
}

extern "C" bool InitModelInstance(int instance_id, const char* modelPath, bool cpu_use) {
    return mgr().initModel(instance_id, modelPath, cpu_use);
}

extern "C" bool RunModelInstanceFloat(int instance_id, float* data, int num_elements) {
    return mgr().runModel(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceInt(int instance_id, int* data, int num_elements) {
    return mgr().runModel(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceDouble(int instance_id, double* data, int num_elements) {
    return mgr().runModel(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceBatchFloat(int instance_id, float* data, int batch_size, int elements_per_sample) {
    return mgr().runModelBatch(instance_id, data, batch_size, elements_per_sample);
}

extern "C" const float* GetInstanceFlattenedOutput(int instance_id, int* size) {
    const auto& out = mgr().getFlattenedOutput(instance_id);
    if (size) *size = static_cast<int>(out.size());
    return out.data();
}

extern "C" const int64_t* GetInstanceOriginalShape(int instance_id, int* size) {
    const auto& shape = mgr().getOriginalShape(instance_id);
    if (size) *size = static_cast<int>(shape.size());
    return shape.data();
}

extern "C" int GetInstanceElementsPerSample(int instance_id) {
    return mgr().getElementsPerSample(instance_id);
}

extern "C" bool IsInstanceInitialized(int instance_id) {
    return mgr().isInitialized(instance_id);
}

extern "C" int GetModelInstanceCount() {
    return mgr().getInstanceCount();
}

// =====================================================================
// Logging API
// =====================================================================

extern "C" void SetLogLevel(int level) {
    if (level >= 0 && level <= 4) {
        Logger::getInstance().setLogLevel(static_cast<LogLevel>(level));
    }
}

extern "C" int GetLogLevel() {
    return static_cast<int>(Logger::getInstance().getLogLevel());
}

extern "C" void EnableLogConsole(bool enable) {
    Logger::getInstance().enableConsole(enable);
}

extern "C" void EnableLogTimestamp(bool enable) {
    Logger::getInstance().enableTimestamp(enable);
}

extern "C" bool SetLogFile(const char* filepath) {
    if (filepath) {
        return Logger::getInstance().setLogFile(filepath);
    }
    return false;
}

extern "C" void CloseLogFile() {
    Logger::getInstance().closeLogFile();
}

extern "C" void SetLogCallback(LogCallbackC callback, void* user_data) {
    if (callback) {
        Logger::getInstance().setCCallback(callback, user_data);
    } else {
        Logger::getInstance().clearCCallback();
    }
}

extern "C" void ClearLogCallback() {
    Logger::getInstance().clearCCallback();
}

extern "C" void LogDebug(const char* message) {
    if (message) Logger::getInstance().debug(message);
}

extern "C" void LogInfo(const char* message) {
    if (message) Logger::getInstance().info(message);
}

extern "C" void LogWarn(const char* message) {
    if (message) Logger::getInstance().warn(message);
}

extern "C" void LogError(const char* message) {
    if (message) Logger::getInstance().error(message);
}
