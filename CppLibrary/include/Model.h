//
// Created by 권용훈 on 7/11/24.
//

#ifndef AIRUNNER_MODEL_H
#define AIRUNNER_MODEL_H


class Model {
public:
    void SetSessionOption(bool cpu_use);
    void setModel(const char* model_path);
    void setModelInOutput();
    void setModelInOutputTypeDim();

    std::vector<float> getFlattenedOutput() const;
    std::vector<int64_t> getOriginalShape() const;
public:
    bool runInference(Ort::Value input_tensor);

public:
    std::string getInputName() {return input_name;}
    std::string getOutputName(){ return output_name;}
    std::vector<int64_t> getInputDims() const { return input_dims; }
    ONNXTensorElementDataType getInputType() const { return input_type; }

private:
    Ort::SessionOptions session_options;

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
