//
// Created for multi-model management
//

#ifndef AIRUNNER_MODELMANAGER_H
#define AIRUNNER_MODELMANAGER_H

#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <future>
#include <functional>
#include <string>
#include <vector>

class Workflow;

// Async callback used by the default-instance C ABI bridge (float-typed only,
// matching the public C surface). Internal multi-instance API does not expose
// async yet — that lands with Step 3.
using DefaultAsyncCallback =
    std::function<void(bool success, const std::vector<float>& output)>;

class ModelManager {
public:
    static ModelManager& getInstance();

    // ------------------------------------------------------------------
    // Multi-instance API — caller chooses the instance id explicitly.
    // ------------------------------------------------------------------

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
    bool runModelBatch(int instance_id, int* data, int batch_size, int elements_per_sample);
    bool runModelBatch(int instance_id, double* data, int batch_size, int elements_per_sample);

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

    // ------------------------------------------------------------------
    // Default-instance API — backs the legacy single-model C ABI.
    //
    // The default instance lives in the same `instances_` map as user-created
    // ones, but its id is tracked here so the C ABI does not need any
    // file-scope state. Calling initDefault() destroys any prior default.
    // ------------------------------------------------------------------

    bool initDefault(const char* modelPath, bool cpu_use);
    void resetDefault();

    bool runDefault(float* data, int num_elements);
    bool runDefault(int* data, int num_elements);
    bool runDefault(double* data, int num_elements);

    bool runDefaultBatch(float* data, int batch_size, int elements_per_sample);
    bool runDefaultBatch(int* data, int batch_size, int elements_per_sample);
    bool runDefaultBatch(double* data, int batch_size, int elements_per_sample);

    bool runDefaultAsync(float* data, int num_elements);
    bool runDefaultAsync(int* data, int num_elements);
    bool runDefaultAsync(double* data, int num_elements);

    bool runDefaultBatchAsync(float* data, int batch_size, int elements_per_sample);

    bool runDefaultAsyncCallback(float* data, int num_elements,
                                 DefaultAsyncCallback callback);

    bool isDefaultRunning() const;
    void waitForDefault();
    bool getDefaultAsyncResult();

    const std::vector<float>& getDefaultFlattenedOutput() const;
    const std::vector<int64_t>& getDefaultOriginalShape() const;
    int getDefaultElementsPerSample() const;
    bool isDefaultInitialized() const;

    bool runDefaultTest(const char* modelPath, bool cpu_use, float* data, int num_elements);

private:
    ModelManager() = default;
    ~ModelManager() = default;
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    std::shared_ptr<Workflow> getWorkflow(int instance_id) const;
    std::shared_ptr<Workflow> getDefaultWorkflow() const;

    // Ensures a default-instance workflow exists; returns its id. Caller
    // must NOT hold instances_mutex_ when calling.
    int ensureDefaultInstance();

    std::map<int, std::shared_ptr<Workflow>> instances_;
    mutable std::mutex instances_mutex_;
    std::atomic<int> next_id_{1};

    // ---- default-instance state -------------------------------------------
    // default_id_ is read under default_mutex_; future + last result are also
    // guarded by default_mutex_ because RunDefaultAsync may be invoked from
    // one thread while GetDefaultAsyncResult / WaitForDefault are called from
    // another (legacy C ABI did not document a threading contract).
    mutable std::mutex default_mutex_;
    int default_id_ = -1;
    std::future<bool> default_async_future_;
    std::atomic<bool> default_async_result_{false};

    // Empty vectors for error cases
    static const std::vector<float> empty_float_vector_;
    static const std::vector<int64_t> empty_int64_vector_;
};

#endif //AIRUNNER_MODELMANAGER_H
