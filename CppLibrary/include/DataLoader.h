//
// Created by 권용훈 on 7/11/24.
//

#ifndef AIRUNNER_DATALOADER_H
#define AIRUNNER_DATALOADER_H

#include <vector>
#include <cstdint>
#include <onnxruntime_cxx_api.h>

class DataLoader {
public:
    DataLoader(const std::vector<int64_t>& input_dims, ONNXTensorElementDataType input_type);

    template <typename T>
    Ort::Value load_data(T* data, int num_elements);
    Ort::Value float_load_data(float* data, int num_elements);

    // Batch loading: loads multiple samples into a single tensor
    template <typename T>
    Ort::Value load_batch_data(T* data, int batch_size, int elements_per_sample);

    // Get dimensions with custom batch size
    std::vector<int64_t> getBatchDims(int batch_size) const;

private:
    std::vector<int64_t> input_dims_;
    ONNXTensorElementDataType input_type_;
};

#endif //AIRUNNER_DATALOADER_H
