//
// Created by 권용훈 on 7/10/24.
//

#include "../Utils/pch.h"
#include "../include/aiRunner.h"
#include "../include/Workflow.h"
#include <string.h>

static std::shared_ptr<Workflow> workflow = nullptr;
static std::vector<float> flattened_output;
static std::vector<int64_t> original_shape;

extern "C" bool InitModel(const char* modelPath, bool cpu_use) {
    workflow = std::make_shared<Workflow>();
    workflow->init_model(modelPath, cpu_use);
    return true;
}

extern "C" void RunModelInt(int* data, size_t num_elements) {
    if (workflow) {
        workflow->run_model(data, num_elements);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" void RunModelFloat(float* data, size_t num_elements) {
    if (workflow) {
        workflow->run_model(data, num_elements);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" void RunModelDouble(double* data, size_t num_elements) {
    if (workflow) {
        workflow->run_model(data, num_elements);
        flattened_output = workflow->getFlattenedOutput();
        original_shape = workflow->getOriginalShape();
    }
}

extern "C" const float* GetFlattenedOutput(size_t* size) {
    if (size) {
        *size = flattened_output.size();
    }
    return flattened_output.data();
}

extern "C" const int64_t* GetOriginalShape(size_t* size) {
    if (size) {
        *size = original_shape.size();
    }
    return original_shape.data();
}