//
// Created by 권용훈 on 7/11/24.
//

#ifndef AIRUNNER_DATALOADER_H
#define AIRUNNER_DATALOADER_H


class Model;

class DataLoader {
public:
    DataLoader(const std::vector<int64_t>& input_dims, ONNXTensorElementDataType input_type);

    Ort::Value load_data(float* data, int num_elements);
    Ort::Value float_load_data(float * data, int num_elements);

private:
    std::vector<int64_t> input_dims_;
    ONNXTensorElementDataType input_type_;
};

#endif //AIRUNNER_DATALOADER_H
