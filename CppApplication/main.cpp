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


    return 0;
}
