//
// Created for multi-model management
//

#include "../include/ModelManager.h"
#include "../include/Workflow.h"
#include "../include/Logger.h"

// Static empty vectors for error cases
const std::vector<float> ModelManager::empty_float_vector_;
const std::vector<int64_t> ModelManager::empty_int64_vector_;

namespace {
// Logs "Instance <id> not found" once, returns false. Keeps the not-found
// branch of every delegating method down to a single line.
bool instance_not_found(int instance_id) {
    Logger::getInstance().error(
        std::string("Instance ") + std::to_string(instance_id) + " not found");
    return false;
}
}  // namespace

ModelManager& ModelManager::getInstance() {
    // Immortal singleton: leaked at process exit on purpose. Owning
    // Ort::Session/Env shared_ptrs whose destruction order is fragile under
    // ORT 1.26 (it grabs internal mutexes during exit); deferring destruction
    // to the OS sidesteps the race entirely.
    static ModelManager* instance = new ModelManager();
    return *instance;
}

// =====================================================================
// Instance lifecycle
// =====================================================================

int ModelManager::create_instance() {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    int id = next_id_.fetch_add(1);
    instances_[id] = std::make_shared<Workflow>();
    return id;
}

bool ModelManager::destroy_instance(int instance_id) {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    auto it = instances_.find(instance_id);
    if (it != instances_.end()) {
        instances_.erase(it);
        return true;
    }
    return false;
}

bool ModelManager::init_model(int instance_id, const char* modelPath, bool cpu_use) {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return instance_not_found(instance_id);

    try {
        workflow->init_model(modelPath, cpu_use);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(
            std::string("Error initializing model for instance ") +
            std::to_string(instance_id) + ": " + e.what());
        return false;
    }
}

// =====================================================================
// Synchronous inference
// =====================================================================

bool ModelManager::run_model(int instance_id, float* data, int num_elements) {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return instance_not_found(instance_id);

    try {
        workflow->run_model(data, num_elements);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(
            std::string("Error running model for instance ") +
            std::to_string(instance_id) + ": " + e.what());
        return false;
    }
}

bool ModelManager::run_model(int instance_id, int* data, int num_elements) {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return instance_not_found(instance_id);

    try {
        workflow->run_model(data, num_elements);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(
            std::string("Error running model for instance ") +
            std::to_string(instance_id) + ": " + e.what());
        return false;
    }
}

bool ModelManager::run_model(int instance_id, double* data, int num_elements) {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return instance_not_found(instance_id);

    try {
        workflow->run_model(data, num_elements);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(
            std::string("Error running model for instance ") +
            std::to_string(instance_id) + ": " + e.what());
        return false;
    }
}

bool ModelManager::run_model_batch(int instance_id, float* data, int batch_size, int elements_per_sample) {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return instance_not_found(instance_id);

    try {
        workflow->run_model_batch(data, batch_size, elements_per_sample);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(
            std::string("Error running batch model for instance ") +
            std::to_string(instance_id) + ": " + e.what());
        return false;
    }
}

bool ModelManager::run_model_batch(int instance_id, int* data, int batch_size, int elements_per_sample) {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return instance_not_found(instance_id);

    try {
        workflow->run_model_batch(data, batch_size, elements_per_sample);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(
            std::string("Error running batch model for instance ") +
            std::to_string(instance_id) + ": " + e.what());
        return false;
    }
}

bool ModelManager::run_model_batch(int instance_id, double* data, int batch_size, int elements_per_sample) {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return instance_not_found(instance_id);

    try {
        workflow->run_model_batch(data, batch_size, elements_per_sample);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(
            std::string("Error running batch model for instance ") +
            std::to_string(instance_id) + ": " + e.what());
        return false;
    }
}

// =====================================================================
// Asynchronous inference — delegated to the target Workflow, which owns
// its own future and last-result state.
// =====================================================================

bool ModelManager::run_model_async(int instance_id, float* data, int num_elements) {
    auto wf = get_workflow(instance_id);
    if (!wf) return instance_not_found(instance_id);
    return wf->run_model_async(data, num_elements);
}

bool ModelManager::run_model_async(int instance_id, int* data, int num_elements) {
    auto wf = get_workflow(instance_id);
    if (!wf) return instance_not_found(instance_id);
    return wf->run_model_async(data, num_elements);
}

bool ModelManager::run_model_async(int instance_id, double* data, int num_elements) {
    auto wf = get_workflow(instance_id);
    if (!wf) return instance_not_found(instance_id);
    return wf->run_model_async(data, num_elements);
}

bool ModelManager::run_model_batch_async(int instance_id, float* data, int batch_size, int elements_per_sample) {
    auto wf = get_workflow(instance_id);
    if (!wf) return instance_not_found(instance_id);
    return wf->run_model_batch_async(data, batch_size, elements_per_sample);
}

bool ModelManager::run_model_batch_async(int instance_id, int* data, int batch_size, int elements_per_sample) {
    auto wf = get_workflow(instance_id);
    if (!wf) return instance_not_found(instance_id);
    return wf->run_model_batch_async(data, batch_size, elements_per_sample);
}

bool ModelManager::run_model_batch_async(int instance_id, double* data, int batch_size, int elements_per_sample) {
    auto wf = get_workflow(instance_id);
    if (!wf) return instance_not_found(instance_id);
    return wf->run_model_batch_async(data, batch_size, elements_per_sample);
}

bool ModelManager::run_model_async_callback(int instance_id, float* data, int num_elements,
                                            InferenceCallback callback) {
    auto wf = get_workflow(instance_id);
    if (!wf) return instance_not_found(instance_id);
    wf->run_model_async_callback(data, num_elements, std::move(callback));
    return true;
}

bool ModelManager::is_inference_running(int instance_id) const {
    auto wf = get_workflow(instance_id);
    if (!wf) return false;
    return wf->is_inference_running();
}

bool ModelManager::wait_for_inference(int instance_id) {
    auto wf = get_workflow(instance_id);
    if (!wf) return false;
    wf->wait_for_inference();
    return true;
}

bool ModelManager::get_async_result(int instance_id) {
    auto wf = get_workflow(instance_id);
    if (!wf) return false;
    return wf->get_async_result();
}

// =====================================================================
// Output / info
// =====================================================================

const std::vector<float>& ModelManager::get_flattened_output(int instance_id) const {
    auto workflow = get_workflow(instance_id);
    if (!workflow) {
        return empty_float_vector_;
    }
    return workflow->get_flattened_output();
}

const std::vector<int64_t>& ModelManager::get_original_shape(int instance_id) const {
    auto workflow = get_workflow(instance_id);
    if (!workflow) {
        return empty_int64_vector_;
    }
    return workflow->get_original_shape();
}

int ModelManager::get_elements_per_sample(int instance_id) const {
    auto workflow = get_workflow(instance_id);
    if (!workflow) return 0;
    return workflow->get_elements_per_sample();
}

bool ModelManager::is_initialized(int instance_id) const {
    auto workflow = get_workflow(instance_id);
    return workflow != nullptr;
}

std::vector<int> ModelManager::get_active_instances() const {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    std::vector<int> ids;
    ids.reserve(instances_.size());
    for (const auto& pair : instances_) {
        ids.push_back(pair.first);
    }
    return ids;
}

int ModelManager::get_instance_count() const {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    return static_cast<int>(instances_.size());
}

std::shared_ptr<Workflow> ModelManager::get_workflow(int instance_id) const {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    auto it = instances_.find(instance_id);
    if (it != instances_.end()) {
        return it->second;
    }
    return nullptr;
}
