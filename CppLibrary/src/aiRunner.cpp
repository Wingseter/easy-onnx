//
// Created by 권용훈 on 7/10/24.
//
// Thin C ABI adapters. All inference state lives in ModelManager (model
// instances, async futures) or in Logger (log callback bridge). No file-scope
// statics here. As of v0.2.0 the surface is purely handle-based — see
// aiRunner.h and docs/migration-v0.2.0.md.

#include "../Utils/pch.h"
#include "../include/aiRunner.h"
#include "../include/ModelManager.h"
#include "../include/Logger.h"

namespace {
inline ModelManager& mgr() { return ModelManager::getInstance(); }
}

// =====================================================================
// Instance lifecycle
// =====================================================================

extern "C" int CreateModelInstance() {
    return mgr().create_instance();
}

extern "C" bool DestroyModelInstance(int instance_id) {
    return mgr().destroy_instance(instance_id);
}

extern "C" bool InitModelInstance(int instance_id, const char* modelPath, bool cpu_use) {
    return mgr().init_model(instance_id, modelPath, cpu_use);
}

// =====================================================================
// Synchronous inference
// =====================================================================

extern "C" bool RunModelInstanceInt(int instance_id, int* data, int num_elements) {
    return mgr().run_model(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceFloat(int instance_id, float* data, int num_elements) {
    return mgr().run_model(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceDouble(int instance_id, double* data, int num_elements) {
    return mgr().run_model(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceBatchInt(int instance_id, int* data, int batch_size, int elements_per_sample) {
    return mgr().run_model_batch(instance_id, data, batch_size, elements_per_sample);
}

extern "C" bool RunModelInstanceBatchFloat(int instance_id, float* data, int batch_size, int elements_per_sample) {
    return mgr().run_model_batch(instance_id, data, batch_size, elements_per_sample);
}

extern "C" bool RunModelInstanceBatchDouble(int instance_id, double* data, int batch_size, int elements_per_sample) {
    return mgr().run_model_batch(instance_id, data, batch_size, elements_per_sample);
}

// =====================================================================
// Asynchronous inference
// =====================================================================

extern "C" bool RunModelInstanceIntAsync(int instance_id, int* data, int num_elements) {
    return mgr().run_model_async(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceFloatAsync(int instance_id, float* data, int num_elements) {
    return mgr().run_model_async(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceDoubleAsync(int instance_id, double* data, int num_elements) {
    return mgr().run_model_async(instance_id, data, num_elements);
}

extern "C" bool RunModelInstanceBatchIntAsync(int instance_id, int* data, int batch_size, int elements_per_sample) {
    return mgr().run_model_batch_async(instance_id, data, batch_size, elements_per_sample);
}

extern "C" bool RunModelInstanceBatchFloatAsync(int instance_id, float* data, int batch_size, int elements_per_sample) {
    return mgr().run_model_batch_async(instance_id, data, batch_size, elements_per_sample);
}

extern "C" bool RunModelInstanceBatchDoubleAsync(int instance_id, double* data, int batch_size, int elements_per_sample) {
    return mgr().run_model_batch_async(instance_id, data, batch_size, elements_per_sample);
}

extern "C" bool IsInstanceInferenceRunning(int instance_id) {
    return mgr().is_inference_running(instance_id);
}

extern "C" bool WaitForInstanceInference(int instance_id) {
    return mgr().wait_for_inference(instance_id);
}

extern "C" bool GetInstanceAsyncResult(int instance_id) {
    return mgr().get_async_result(instance_id);
}

extern "C" bool RunModelInstanceFloatAsyncCallback(int instance_id, float* data, int num_elements,
                                                   InferenceCallbackC callback, void* user_data) {
    return mgr().run_model_async_callback(instance_id, data, num_elements,
        [callback, user_data](bool success, const std::vector<float>& output) {
            if (callback) {
                callback(success, output.data(), static_cast<int>(output.size()), user_data);
            }
        });
}

// =====================================================================
// Output retrieval
// =====================================================================

extern "C" const float* GetInstanceFlattenedOutput(int instance_id, int* size) {
    const auto& out = mgr().get_flattened_output(instance_id);
    if (size) *size = static_cast<int>(out.size());
    return out.data();
}

extern "C" const int64_t* GetInstanceOriginalShape(int instance_id, int* size) {
    const auto& shape = mgr().get_original_shape(instance_id);
    if (size) *size = static_cast<int>(shape.size());
    return shape.data();
}

// =====================================================================
// Instance info
// =====================================================================

extern "C" int GetInstanceElementsPerSample(int instance_id) {
    return mgr().get_elements_per_sample(instance_id);
}

extern "C" bool IsInstanceInitialized(int instance_id) {
    return mgr().is_initialized(instance_id);
}

extern "C" int GetModelInstanceCount() {
    return mgr().get_instance_count();
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
