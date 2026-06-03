//
// Created for multi-model management
//

#ifndef AIRUNNER_MODELMANAGER_H
#define AIRUNNER_MODELMANAGER_H

#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <vector>
#include <cstdint>

#include "Workflow.h"  // brings in Workflow + InferenceCallback

// Owns every model instance. Each instance is addressed by an integer handle
// returned from create_instance(). As of v0.2.0 there is no default/global
// instance — the legacy single-model C ABI was removed.
class ModelManager {
public:
    static ModelManager& getInstance();

    // ------------------------------------------------------------------
    // Instance lifecycle
    // ------------------------------------------------------------------
    int  create_instance();                 // returns a new instance id (>= 1)
    bool destroy_instance(int instance_id);
    bool init_model(int instance_id, const char* modelPath, bool cpu_use);

    // ------------------------------------------------------------------
    // Synchronous inference
    // ------------------------------------------------------------------
    bool run_model(int instance_id, float* data, int num_elements);
    bool run_model(int instance_id, int* data, int num_elements);
    bool run_model(int instance_id, double* data, int num_elements);

    bool run_model_batch(int instance_id, float* data, int batch_size, int elements_per_sample);
    bool run_model_batch(int instance_id, int* data, int batch_size, int elements_per_sample);
    bool run_model_batch(int instance_id, double* data, int batch_size, int elements_per_sample);

    // ------------------------------------------------------------------
    // Asynchronous inference — the future/result live in the target Workflow.
    // ------------------------------------------------------------------
    bool run_model_async(int instance_id, float* data, int num_elements);
    bool run_model_async(int instance_id, int* data, int num_elements);
    bool run_model_async(int instance_id, double* data, int num_elements);

    bool run_model_batch_async(int instance_id, float* data, int batch_size, int elements_per_sample);
    bool run_model_batch_async(int instance_id, int* data, int batch_size, int elements_per_sample);
    bool run_model_batch_async(int instance_id, double* data, int batch_size, int elements_per_sample);

    bool run_model_async_callback(int instance_id, float* data, int num_elements,
                                  InferenceCallback callback);

    bool is_inference_running(int instance_id) const;
    bool wait_for_inference(int instance_id);
    bool get_async_result(int instance_id);

    // ------------------------------------------------------------------
    // Output / info
    // ------------------------------------------------------------------
    const std::vector<float>& get_flattened_output(int instance_id) const;
    const std::vector<int64_t>& get_original_shape(int instance_id) const;

    int  get_elements_per_sample(int instance_id) const;
    bool is_initialized(int instance_id) const;

    std::vector<int> get_active_instances() const;
    int get_instance_count() const;

private:
    ModelManager() = default;
    ~ModelManager() = default;
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    std::shared_ptr<Workflow> get_workflow(int instance_id) const;

    std::map<int, std::shared_ptr<Workflow>> instances_;
    mutable std::mutex instances_mutex_;
    std::atomic<int> next_id_{1};

    // Empty vectors returned for unknown instance ids.
    static const std::vector<float> empty_float_vector_;
    static const std::vector<int64_t> empty_int64_vector_;
};

#endif //AIRUNNER_MODELMANAGER_H
