//
// Created by 권용훈 on 7/11/24.
//

#ifndef AIRUNNER_MODEL_H
#define AIRUNNER_MODEL_H

#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

class Model {
public:
    void set_session_option(bool cpu_use);
    void set_model(const char* model_path);
    void set_model_in_output();
    void set_model_in_output_type_dim();
    void reset_model();

    const std::vector<float>& get_flattened_output() const;
    const std::vector<int64_t>& get_original_shape() const;

    bool run_inference(Ort::Value input_tensor);
    bool is_initialized() const { return initialized; }

    std::string get_input_name() const { return input_name; }
    std::string get_output_name() const { return output_name; }
    std::vector<int64_t> get_input_dims() const { return input_dims; }
    ONNXTensorElementDataType get_input_type() const { return input_type; }

private:
    Ort::SessionOptions session_options;
    bool initialized = false;

    // ONNX Runtime Environment and Session variable
    // ENV Must be use as pointer if not it makes runtime error
    std::unique_ptr<Ort::Session> session= nullptr;
    std::unique_ptr<Ort::Env> env = nullptr;

    // Input Variables
    std::string input_name;
    std::vector<int64_t> input_dims;
    ONNXTensorElementDataType input_type;

    // Output Variables
    std::string output_name;
    std::vector<int64_t> output_dims;
    ONNXTensorElementDataType output_type;

    // Inference Result
    std::vector<float> flattened_output;
    std::vector<int64_t> original_shape;
};


#endif //AIRUNNER_MODEL_H
