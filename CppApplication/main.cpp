#include <iostream>
#include "aiRunner.h"

int main() {
    std::cout << "Hello This is ai Running Tester" << std::endl;
    bool cpu_use = true;
    const char* modelPath = "/Users/kwon/Workspace/C/ai-run-library/SampleModel/best_metric_model_0.7747.onnx";


    int dimensions[] = {1, 4, 128, 128, 80};
    int total_elements = 1;

    // Calculate the total number of elements
    for (int i = 0; i < 5; ++i) {
        total_elements *= dimensions[i];
    }

    // Create a vector to hold the flattened data
    std::vector<float> data(total_elements);
//
//
    // Fill the data vector with random values
    for (int i = 0; i < total_elements; ++i) {
        data[i] = i % 100;  // Random values between 0 and 99
    }

//    allCheck(modelPath, cpu_use, data.data(), total_elements);

    // model initialize
    if (!InitModel(modelPath, cpu_use)) {
        std::cerr << "Failed to initialize model." << std::endl;
        return -1;
    }

    RunModelFloat(data.data(), total_elements);

//    // Get Final Result
//    int size;
//    const float* output = GetFlattenedOutput(&size);
//
//    std::cout << "final output size is " << size << std::endl;
//    std::cout << "first element is " << output[0] << std::endl;

    return 0;
}
