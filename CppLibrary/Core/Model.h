//
// Created by 권용훈 on 7/11/24.
//

#ifndef AIRUNNER_MODEL_H
#define AIRUNNER_MODEL_H


class Model {
public:
    void setSessionOption(bool cpu_use);
    void setModel(const char* model_path);
    void setModelInOutput();
    void setModelInputTypeDim();

public:
    bool runInference(Ort::Value input_tensor);

public:
    std::string getInputName() {return input_name;}
    std::string getOutputName(){ return output_name;}
    std::vector<int64_t> getInputDims() const { return input_dims; }
    ONNXTensorElementDataType getInputType() const { return input_type; }

private:
    Ort::SessionOptions session_options;
    shared_ptr<Ort::Session> session = nullptr;

    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_dims;
    ONNXTensorElementDataType input_type;
};


#endif //AIRUNNER_MODEL_H
