#ifndef AIRUNNER_WORKFLOW_H
#define AIRUNNER_WORKFLOW_H

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <future>
#include <functional>
#include <atomic>

class Model;
class DataLoader;

// Callback type for async inference
using InferenceCallback = std::function<void(bool success, const std::vector<float>& output)>;

class Workflow {
public:
    void init_model(const char* modelPath, bool cpu_use);

    // Single sample inference
    void run_model(int* data, int num_elements);
    void run_model(float* data, int num_elements);
    void run_model(double* data, int num_elements);

    // Batch inference
    void run_model_batch(int* data, int batch_size, int elements_per_sample);
    void run_model_batch(float* data, int batch_size, int elements_per_sample);
    void run_model_batch(double* data, int batch_size, int elements_per_sample);

    const std::vector<float>& getFlattenedOutput() const;
    const std::vector<int64_t>& getOriginalShape() const;

    // Get elements per sample (for batch processing)
    int getElementsPerSample() const;

    // Async inference - Future based
    template <typename T>
    std::future<bool> run_model_async(T* data, int num_elements);

    template <typename T>
    std::future<bool> run_model_batch_async(T* data, int batch_size, int elements_per_sample);

    // Async inference - Callback based
    template <typename T>
    void run_model_async_callback(T* data, int num_elements, InferenceCallback callback);

    // Async status
    bool isInferenceRunning() const { return is_running_.load(); }
    void waitForInference();

    void run_test(const char* modelPath, bool cpu_use, float* data, int num_elements);

private:
    template <typename T>
    void run_inference(T* data, int num_elements);

    template <typename T>
    void run_batch_inference(T* data, int batch_size, int elements_per_sample);

    std::string previous = "";
    std::shared_ptr<Model> model_ = nullptr;
    std::shared_ptr<DataLoader> data_loader_ = nullptr;

    // Async state
    std::atomic<bool> is_running_{false};
    std::future<bool> current_future_;
    std::mutex async_mutex_;
};

#endif //AIRUNNER_WORKFLOW_H
