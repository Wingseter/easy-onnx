//
// Created for multi-model management
//

#include "../include/ModelManager.h"
#include "../include/Workflow.h"
#include "../include/Logger.h"

// Static empty vectors for error cases
const std::vector<float> ModelManager::empty_float_vector_;
const std::vector<int64_t> ModelManager::empty_int64_vector_;

ModelManager& ModelManager::getInstance() {
    // Immortal singleton: leaked at process exit on purpose. Owning
    // Ort::Session/Env shared_ptrs whose destruction order is fragile under
    // ORT 1.26 (it grabs internal mutexes during exit); deferring destruction
    // to the OS sidesteps the race entirely.
    static ModelManager* instance = new ModelManager();
    return *instance;
}

int ModelManager::createInstance() {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    int id = next_id_.fetch_add(1);
    instances_[id] = std::make_shared<Workflow>();
    return id;
}

bool ModelManager::destroyInstance(int instance_id) {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    auto it = instances_.find(instance_id);
    if (it != instances_.end()) {
        instances_.erase(it);
        return true;
    }
    return false;
}

bool ModelManager::initModel(int instance_id, const char* modelPath, bool cpu_use) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) {
        Logger::getInstance().error(
            std::string("Instance ") + std::to_string(instance_id) + " not found");
        return false;
    }

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

bool ModelManager::runModel(int instance_id, float* data, int num_elements) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) {
        Logger::getInstance().error(
            std::string("Instance ") + std::to_string(instance_id) + " not found");
        return false;
    }

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

bool ModelManager::runModel(int instance_id, int* data, int num_elements) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) return false;

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

bool ModelManager::runModel(int instance_id, double* data, int num_elements) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) return false;

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

bool ModelManager::runModelBatch(int instance_id, float* data, int batch_size, int elements_per_sample) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) return false;

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

bool ModelManager::runModelBatch(int instance_id, int* data, int batch_size, int elements_per_sample) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) return false;

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

bool ModelManager::runModelBatch(int instance_id, double* data, int batch_size, int elements_per_sample) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) return false;

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

const std::vector<float>& ModelManager::getFlattenedOutput(int instance_id) const {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) {
        return empty_float_vector_;
    }
    return workflow->getFlattenedOutput();
}

const std::vector<int64_t>& ModelManager::getOriginalShape(int instance_id) const {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) {
        return empty_int64_vector_;
    }
    return workflow->getOriginalShape();
}

int ModelManager::getElementsPerSample(int instance_id) const {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) return 0;
    return workflow->getElementsPerSample();
}

bool ModelManager::isInitialized(int instance_id) const {
    auto workflow = getWorkflow(instance_id);
    return workflow != nullptr;
}

std::vector<int> ModelManager::getActiveInstances() const {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    std::vector<int> ids;
    ids.reserve(instances_.size());
    for (const auto& pair : instances_) {
        ids.push_back(pair.first);
    }
    return ids;
}

int ModelManager::getInstanceCount() const {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    return static_cast<int>(instances_.size());
}

std::shared_ptr<Workflow> ModelManager::getWorkflow(int instance_id) const {
    std::lock_guard<std::mutex> lock(instances_mutex_);
    auto it = instances_.find(instance_id);
    if (it != instances_.end()) {
        return it->second;
    }
    return nullptr;
}

// =====================================================================
// Default-instance API
// =====================================================================

std::shared_ptr<Workflow> ModelManager::getDefaultWorkflow() const {
    int id;
    {
        std::lock_guard<std::mutex> lock(default_mutex_);
        id = default_id_;
    }
    if (id < 0) return nullptr;
    return getWorkflow(id);
}

int ModelManager::ensureDefaultInstance() {
    // Take default_mutex_ to publish default_id_ atomically with any prior
    // future cleanup; createInstance() takes instances_mutex_ separately
    // (lock order: default_mutex_ -> instances_mutex_, never the reverse).
    std::lock_guard<std::mutex> lock(default_mutex_);
    if (default_id_ < 0 || !getWorkflow(default_id_)) {
        int id = next_id_.fetch_add(1);
        {
            std::lock_guard<std::mutex> ilock(instances_mutex_);
            instances_[id] = std::make_shared<Workflow>();
        }
        default_id_ = id;
    }
    return default_id_;
}

void ModelManager::resetDefault() {
    int prior;
    {
        std::lock_guard<std::mutex> lock(default_mutex_);
        prior = default_id_;
        default_id_ = -1;
        default_async_result_.store(false);
        // Future is left as-is; if a prior async run is still in flight the
        // caller is responsible for waitForDefault() before re-init. We drop
        // it here defensively.
        if (default_async_future_.valid()) {
            // detach: it will run to completion against the dying workflow.
            // Better than blocking destroyInstance() unpredictably.
            default_async_future_ = std::future<bool>();
        }
    }
    if (prior >= 0) {
        std::lock_guard<std::mutex> lock(instances_mutex_);
        instances_.erase(prior);
    }
}

bool ModelManager::initDefault(const char* modelPath, bool cpu_use) {
    int id = ensureDefaultInstance();
    return initModel(id, modelPath, cpu_use);
}

bool ModelManager::runDefault(float* data, int num_elements) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    try {
        wf->run_model(data, num_elements);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Default run error: ") + e.what());
        return false;
    }
}
bool ModelManager::runDefault(int* data, int num_elements) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    try {
        wf->run_model(data, num_elements);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Default run error: ") + e.what());
        return false;
    }
}
bool ModelManager::runDefault(double* data, int num_elements) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    try {
        wf->run_model(data, num_elements);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Default run error: ") + e.what());
        return false;
    }
}

bool ModelManager::runDefaultBatch(float* data, int batch_size, int elements_per_sample) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    try {
        wf->run_model_batch(data, batch_size, elements_per_sample);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Default batch run error: ") + e.what());
        return false;
    }
}
bool ModelManager::runDefaultBatch(int* data, int batch_size, int elements_per_sample) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    try {
        wf->run_model_batch(data, batch_size, elements_per_sample);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Default batch run error: ") + e.what());
        return false;
    }
}
bool ModelManager::runDefaultBatch(double* data, int batch_size, int elements_per_sample) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    try {
        wf->run_model_batch(data, batch_size, elements_per_sample);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Default batch run error: ") + e.what());
        return false;
    }
}

bool ModelManager::runDefaultAsync(float* data, int num_elements) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    std::lock_guard<std::mutex> lock(default_mutex_);
    default_async_future_ = wf->run_model_async(data, num_elements);
    return true;
}
bool ModelManager::runDefaultAsync(int* data, int num_elements) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    std::lock_guard<std::mutex> lock(default_mutex_);
    default_async_future_ = wf->run_model_async(data, num_elements);
    return true;
}
bool ModelManager::runDefaultAsync(double* data, int num_elements) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    std::lock_guard<std::mutex> lock(default_mutex_);
    default_async_future_ = wf->run_model_async(data, num_elements);
    return true;
}

bool ModelManager::runDefaultBatchAsync(float* data, int batch_size, int elements_per_sample) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    std::lock_guard<std::mutex> lock(default_mutex_);
    default_async_future_ = wf->run_model_batch_async(data, batch_size, elements_per_sample);
    return true;
}

bool ModelManager::runDefaultAsyncCallback(float* data, int num_elements,
                                           DefaultAsyncCallback callback) {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    wf->run_model_async_callback(data, num_elements, std::move(callback));
    return true;
}

bool ModelManager::isDefaultRunning() const {
    auto wf = getDefaultWorkflow();
    if (!wf) return false;
    return wf->isInferenceRunning();
}

void ModelManager::waitForDefault() {
    auto wf = getDefaultWorkflow();
    if (!wf) return;
    wf->waitForInference();
    std::lock_guard<std::mutex> lock(default_mutex_);
    if (default_async_future_.valid()) {
        default_async_result_.store(default_async_future_.get());
    }
}

bool ModelManager::getDefaultAsyncResult() {
    std::lock_guard<std::mutex> lock(default_mutex_);
    if (default_async_future_.valid()) {
        default_async_result_.store(default_async_future_.get());
    }
    return default_async_result_.load();
}

const std::vector<float>& ModelManager::getDefaultFlattenedOutput() const {
    auto wf = getDefaultWorkflow();
    if (!wf) return empty_float_vector_;
    return wf->getFlattenedOutput();
}

const std::vector<int64_t>& ModelManager::getDefaultOriginalShape() const {
    auto wf = getDefaultWorkflow();
    if (!wf) return empty_int64_vector_;
    return wf->getOriginalShape();
}

int ModelManager::getDefaultElementsPerSample() const {
    auto wf = getDefaultWorkflow();
    if (!wf) return 0;
    return wf->getElementsPerSample();
}

bool ModelManager::isDefaultInitialized() const {
    return getDefaultWorkflow() != nullptr;
}

bool ModelManager::runDefaultTest(const char* modelPath, bool cpu_use, float* data, int num_elements) {
    auto wf = getDefaultWorkflow();
    if (!wf) {
        Logger::getInstance().error("Default workflow not initialized; call InitModel first.");
        return false;
    }
    try {
        wf->run_test(modelPath, cpu_use, data, num_elements);
        return true;
    } catch (const std::exception& e) {
        Logger::getInstance().error(std::string("Error running test: ") + e.what());
        return false;
    }
}
