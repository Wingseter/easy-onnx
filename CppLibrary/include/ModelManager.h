//
// Created for multi-model management
//

#ifndef AIRUNNER_MODELMANAGER_H
#define AIRUNNER_MODELMANAGER_H

#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>

class Workflow;

class ModelManager {
public:
    static ModelManager& getInstance();

    // Create a new model instance, returns instance ID
    int createInstance();

    // Destroy a model instance
    bool destroyInstance(int instance_id);

    // Initialize model for a specific instance
    bool initModel(int instance_id, const char* modelPath, bool cpu_use);

    // Run inference on a specific instance
    bool runModel(int instance_id, float* data, int num_elements);
    bool runModel(int instance_id, int* data, int num_elements);
    bool runModel(int instance_id, double* data, int num_elements);

    // Run batch inference on a specific instance
    bool runModelBatch(int instance_id, float* data, int batch_size, int elements_per_sample);

    // Get output from a specific instance
    const std::vector<float>& getFlattenedOutput(int instance_id) const;
    const std::vector<int64_t>& getOriginalShape(int instance_id) const;

    // Get info from a specific instance
    int getElementsPerSample(int instance_id) const;
    bool isInitialized(int instance_id) const;

    // Get all active instance IDs
    std::vector<int> getActiveInstances() const;

    // Get instance count
    int getInstanceCount() const;

private:
    ModelManager() = default;
    ~ModelManager() = default;
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    std::shared_ptr<Workflow> getWorkflow(int instance_id) const;

    std::map<int, std::shared_ptr<Workflow>> instances_;
    mutable std::mutex mutex_;
    std::atomic<int> next_id_{1};

    // Empty vectors for error cases
    static const std::vector<float> empty_float_vector_;
    static const std::vector<int64_t> empty_int64_vector_;
};

#endif //AIRUNNER_MODELMANAGER_H
