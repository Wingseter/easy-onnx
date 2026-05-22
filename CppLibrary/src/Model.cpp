//
// Created by 권용훈 on 7/11/24.
//

#include "../include/Model.h"
#include "../include/ExecutionProviderConfig.h"
#include "../include/Logger.h"
#include "../Utils/pch.h"

#include <sstream>

void Model::set_session_option(bool cpu_use) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 2; // Fallback to 2 if hardware_concurrency returns 0 (which is implementation-defined)
    }

    if (cpu_use) {
        session_options.SetIntraOpNumThreads((int) num_threads / 2);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    } else {
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        // ORT 1.26 requires the default logger (owned by Ort::Env) to be live
        // before any EP factory is invoked, so create the env up-front here.
        if (!env) {
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "airunner");
        }
        TryAppendAcceleratedEP(session_options);
    }
}


void Model::set_model(const char * modelPath) {
    initialized = true;
    if (!env) {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "airunner");
    }
    auto sessionLocal = std::make_unique<Ort::Session>(*env, modelPath, session_options);
    session = std::move(sessionLocal);
}

void Model::set_model_in_output() {
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input, output name
    auto input_name_allocated = session->GetInputNameAllocated(0, allocator);
    auto output_name_allocated = session->GetOutputNameAllocated(0, allocator);

    input_name = std::string(input_name_allocated.get());
    output_name = std::string(output_name_allocated.get());

    Logger::getInstance().info(std::string("Input Name: ") + input_name);
    Logger::getInstance().info(std::string("Output Name: ") + output_name);
}


void Model::set_model_in_output_type_dim() {
    auto input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

    // get input dimension and type
    input_dims = input_tensor_info.GetShape();
    input_dims[0] = 1;
    input_type = input_tensor_info.GetElementType();

    auto output_type_info = session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

    // get output dimension and type
    output_dims = output_tensor_info.GetShape();
    output_type = output_tensor_info.GetElementType();

}

bool Model::run_inference(Ort::Value input_tensor) {
    if (!session) {
        Logger::getInstance().error("Session is not initialized!");
        return false;
    }

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    auto input_tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    auto input_shape = input_tensor_info.GetShape();

    if (Logger::getInstance().getLogLevel() <= LogLevel::DEBUG) {
        std::ostringstream oss;
        oss << "Inference: in='" << input_name << "' out='" << output_name << "' shape=[";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i) oss << ",";
            oss << input_shape[i];
        }
        oss << "]";
        Logger::getInstance().debug(oss.str());
    }

    auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 1);

    if (output_tensors.empty()) {
        Logger::getInstance().error("Inference failed: no output tensors");
        return false;
    }

    Ort::Value& output_tensor = output_tensors.front();
    auto output_data = output_tensor.GetTensorMutableData<float>();
    original_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    if (Logger::getInstance().getLogLevel() <= LogLevel::DEBUG) {
        std::ostringstream oss;
        oss << "Output shape=[";
        for (size_t i = 0; i < original_shape.size(); ++i) {
            if (i) oss << ",";
            oss << original_shape[i];
        }
        oss << "] size=" << output_size;
        Logger::getInstance().debug(oss.str());
    }

    flattened_output = std::vector<float>(output_data, output_data + output_size);

    return true;
}

const std::vector<float>& Model::get_flattened_output() const {
    return flattened_output;
}

const std::vector<int64_t>& Model::get_original_shape() const {
    return original_shape;
}

void Model::reset_model() {
    if (session) {
        session.reset();
        env.reset();
    }
}
