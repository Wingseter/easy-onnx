//
// Created for multi-model management
//

#include "../include/ModelManager.h"
#include "../include/Workflow.h"
#include <iostream>

// Static empty vectors for error cases
const std::vector<float> ModelManager::empty_float_vector_;
const std::vector<int64_t> ModelManager::empty_int64_vector_;

ModelManager& ModelManager::getInstance() {
    static ModelManager instance;
    return instance;
}

int ModelManager::createInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    int id = next_id_.fetch_add(1);
    instances_[id] = std::make_shared<Workflow>();
    return id;
}

bool ModelManager::destroyInstance(int instance_id) {
    std::lock_guard<std::mutex> lock(mutex_);
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
        std::cerr << "Instance " << instance_id << " not found" << std::endl;
        return false;
    }

    try {
        workflow->init_model(modelPath, cpu_use);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing model for instance " << instance_id << ": " << e.what() << std::endl;
        return false;
    }
}

bool ModelManager::runModel(int instance_id, float* data, int num_elements) {
    auto workflow = getWorkflow(instance_id);
    if (!workflow) {
        std::cerr << "Instance " << instance_id << " not found" << std::endl;
        return false;
    }

    try {
        workflow->run_model(data, num_elements);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error running model for instance " << instance_id << ": " << e.what() << std::endl;
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
        std::cerr << "Error running model for instance " << instance_id << ": " << e.what() << std::endl;
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
        std::cerr << "Error running model for instance " << instance_id << ": " << e.what() << std::endl;
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
        std::cerr << "Error running batch model for instance " << instance_id << ": " << e.what() << std::endl;
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
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> ids;
    ids.reserve(instances_.size());
    for (const auto& pair : instances_) {
        ids.push_back(pair.first);
    }
    return ids;
}

int ModelManager::getInstanceCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(instances_.size());
}

std::shared_ptr<Workflow> ModelManager::getWorkflow(int instance_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(instance_id);
    if (it != instances_.end()) {
        return it->second;
    }
    return nullptr;
}
