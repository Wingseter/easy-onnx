#ifndef AIRUNNER_WORKFLOW_H
#define AIRUNNER_WORKFLOW_H

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <future>
#include <functional>
#include <atomic>
#include <mutex>

class Model;
class DataLoader;

// Callback type for callback-based async inference. Invoked on the worker
// thread once inference finishes.
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

    const std::vector<float>& get_flattened_output() const;
    const std::vector<int64_t>& get_original_shape() const;

    // Elements per sample (input element count with the batch dimension dropped).
    int get_elements_per_sample() const;

    // ---- Async inference -------------------------------------------------
    // run_model_async / run_model_batch_async start a background task and
    // return true if it was started, false if one is already in flight. The
    // future and the last result are owned by this Workflow; observe them with
    // is_inference_running(), wait_for_inference() and get_async_result().
    template <typename T>
    bool run_model_async(T* data, int num_elements);

    template <typename T>
    bool run_model_batch_async(T* data, int batch_size, int elements_per_sample);

    // Callback-based async: the result is delivered to `callback` on the
    // worker thread instead of through get_async_result().
    template <typename T>
    void run_model_async_callback(T* data, int num_elements, InferenceCallback callback);

    bool is_inference_running() const { return is_running_.load(); }

    // Block until the in-flight future-based async task finishes (no-op if none).
    void wait_for_inference();

    // Result of the most recent future-based async run. Joins the future if it
    // has not been collected yet.
    bool get_async_result();

private:
    template <typename T>
    void run_inference(T* data, int num_elements);

    template <typename T>
    void run_batch_inference(T* data, int batch_size, int elements_per_sample);

    std::string previous = "";
    std::shared_ptr<Model> model_ = nullptr;
    std::shared_ptr<DataLoader> data_loader_ = nullptr;

    // Async state. async_mutex_ guards current_future_ and the check-and-set of
    // is_running_; is_running_ / last_result_ stay atomic for lock-free reads.
    std::atomic<bool> is_running_{false};
    std::atomic<bool> last_result_{false};
    std::future<bool> current_future_;
    std::mutex async_mutex_;
};

#endif //AIRUNNER_WORKFLOW_H
