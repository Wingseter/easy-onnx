//
// Created by 권용훈 on 7/11/24.
//

#ifndef AIRUNNER_DATALOADER_H
#define AIRUNNER_DATALOADER_H


class Model;

class DataLoader {
public:
    explicit DataLoader(std::shared_ptr<Model> model);

    template <typename T>
    Ort::Value load_data(T* data, size_t num_elements);

private:
    std::vector<int64_t> input_dims_;
    ONNXTensorElementDataType input_type_;
};

#endif //AIRUNNER_DATALOADER_H
