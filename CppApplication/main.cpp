#include <iostream>
#include <chrono>
#include "aiRunner.h"

int main() {
    std::cout << "===== AI Runner Test =====" << std::endl;
    bool cpu_use = true;
    const char* modelPath = "/Users/kwon/Workspace/C/ai-run-library/SampleModel/best_metric_model_0.7747.onnx";

    // Model initialization
    if (!InitModel(modelPath, cpu_use)) {
        std::cerr << "Failed to initialize model." << std::endl;
        return -1;
    }

    // Get elements per sample
    int elements_per_sample = GetElementsPerSample();
    std::cout << "Elements per sample: " << elements_per_sample << std::endl;

    // ===== Single Sample Inference Test =====
    std::cout << "\n--- Single Sample Inference ---" << std::endl;
    std::vector<float> single_data(elements_per_sample);
    for (int i = 0; i < elements_per_sample; ++i) {
        single_data[i] = static_cast<float>(i % 100);
    }

    auto start = std::chrono::high_resolution_clock::now();
    RunModelFloat(single_data.data(), elements_per_sample);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int size;
    const float* output = GetFlattenedOutput(&size);
    std::cout << "Output size: " << size << std::endl;
    std::cout << "First element: " << output[0] << std::endl;
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    // ===== Batch Inference Test =====
    std::cout << "\n--- Batch Inference (batch_size=2) ---" << std::endl;
    int batch_size = 2;
    std::vector<float> batch_data(batch_size * elements_per_sample);
    for (int i = 0; i < batch_size * elements_per_sample; ++i) {
        batch_data[i] = static_cast<float>(i % 100);
    }

    start = std::chrono::high_resolution_clock::now();
    RunModelBatchFloat(batch_data.data(), batch_size, elements_per_sample);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    output = GetFlattenedOutput(&size);
    int shape_size;
    const int64_t* shape = GetOriginalShape(&shape_size);

    std::cout << "Output size: " << size << std::endl;
    std::cout << "Output shape: [";
    for (int i = 0; i < shape_size; ++i) {
        std::cout << shape[i];
        if (i < shape_size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "First element: " << output[0] << std::endl;
    std::cout << "Batch inference time: " << duration.count() << " ms" << std::endl;

    // ===== Performance Comparison =====
    std::cout << "\n--- Performance Comparison ---" << std::endl;

    // Single sample x2
    start = std::chrono::high_resolution_clock::now();
    RunModelFloat(single_data.data(), elements_per_sample);
    RunModelFloat(single_data.data(), elements_per_sample);
    end = std::chrono::high_resolution_clock::now();
    auto single_x2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "2x Single inference: " << single_x2_duration.count() << " ms" << std::endl;

    // Batch size 2
    start = std::chrono::high_resolution_clock::now();
    RunModelBatchFloat(batch_data.data(), 2, elements_per_sample);
    end = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Batch(2) inference: " << batch_duration.count() << " ms" << std::endl;

    if (batch_duration.count() > 0) {
        double speedup = static_cast<double>(single_x2_duration.count()) / batch_duration.count();
        std::cout << "Speedup: " << speedup << "x" << std::endl;
    }

    std::cout << "\n===== Test Complete =====" << std::endl;
    return 0;
}
