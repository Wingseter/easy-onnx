#include <iostream>
#include "aiRunner.h"

int main() {
    std::cout << "Hello This is ai Running Tester" << std::endl;
    bool cpu_use = true;
    const char* modelPath = "/Users/kwon/Workspace/C/ai-run-library/SampleModel/best_metric_model_0.7747.onnx";

    // model initialize
    if (!InitModel(modelPath, cpu_use)) {
        std::cerr << "Failed to initialize model." << std::endl;
        return -1;
    }

    int dimensions[] = {1, 4, 128, 128, 80};
    int total_elements = 1;

    // Calculate the total number of elements
    for (int i = 0; i < 5; ++i) {
        total_elements *= dimensions[i];
    }

    // Create a vector to hold the flattened data
    std::vector<int> data(total_elements);

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Fill the data vector with random values
    for (int i = 0; i < total_elements; ++i) {
        data[i] = std::rand() % 100;  // Random values between 0 and 99
    }

    RunModelInt(data, total_elements);

    return 0;
}
