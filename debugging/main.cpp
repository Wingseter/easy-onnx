#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "Hello This is ai Running Tester" << std::endl;
    bool cpu_use = true;
    const char* modelPath = "/Users/kwon/Workspace/C/ai-run-library/SampleModel/best_metric_model_0.7747.onnx";

    std::vector<int64_t> dimensions = {1, 4, 128, 128, 80};
    int total_elements = 1;

    // Calculate the total number of elements
    for (size_t i = 0; i < dimensions.size(); ++i) {
        total_elements *= dimensions[i];
    }

    // Create a vector to hold the flattened data
    std::vector<float> data(total_elements);

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Fill the data vector with random values
    for (int i = 0; i < total_elements; ++i) {
        data[i] = static_cast<float>(std::rand() % 100);  // Random values between 0 and 99
    }

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ai_running_tester");

    // Create session options and set the graph optimization level
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Create session
    Ort::Session session(env, modelPath, session_options);

    // Get input and output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);

    std::cout << "Input Name: " << input_name.get() << std::endl;
    std::cout << "Output Name: " << output_name.get() << std::endl;

    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), dimensions.data(), dimensions.size());

    // Run inference
    std::vector<const char*> input_names = { input_name.get() };
    std::vector<const char*> output_names = { output_name.get() };

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Get output tensor
    Ort::Value& output_tensor = output_tensors.front();
    float* output_data = output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    std::cout << "Output tensor size: " << output_size << std::endl;

    return 0;
}