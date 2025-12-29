#include <iostream>
#include <chrono>
#include <thread>
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

    // ===== Async Inference Test =====
    std::cout << "\n--- Async Inference Test ---" << std::endl;

    // Start async inference
    std::cout << "Starting async inference..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    RunModelFloatAsync(single_data.data(), elements_per_sample);

    // Do other work while inference is running
    std::cout << "Inference running in background..." << std::endl;
    int work_count = 0;
    while (IsInferenceRunning()) {
        // Simulate doing other work
        work_count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "  Working... (" << work_count * 50 << " ms elapsed)" << std::endl;
    }

    // Get result
    bool async_success = GetAsyncResult();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    output = GetFlattenedOutput(&size);
    std::cout << "Async inference completed: " << (async_success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Output size: " << size << std::endl;
    std::cout << "First element: " << output[0] << std::endl;
    std::cout << "Total time (including background work): " << duration.count() << " ms" << std::endl;

    // ===== Async Batch Inference Test =====
    std::cout << "\n--- Async Batch Inference Test ---" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    RunModelBatchFloatAsync(batch_data.data(), 2, elements_per_sample);

    std::cout << "Batch inference running in background..." << std::endl;
    WaitForInference();  // Wait for completion

    async_success = GetAsyncResult();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    output = GetFlattenedOutput(&size);
    shape = GetOriginalShape(&shape_size);
    std::cout << "Async batch inference: " << (async_success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Output shape: [";
    for (int i = 0; i < shape_size; ++i) {
        std::cout << shape[i];
        if (i < shape_size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    // ===== Multi-Model Management Test =====
    std::cout << "\n--- Multi-Model Management Test ---" << std::endl;

    // Create two model instances
    int instance1 = CreateModelInstance();
    int instance2 = CreateModelInstance();
    std::cout << "Created instances: " << instance1 << ", " << instance2 << std::endl;
    std::cout << "Active instance count: " << GetModelInstanceCount() << std::endl;

    // Initialize both instances with the same model (could be different models)
    std::cout << "\nInitializing instance " << instance1 << "..." << std::endl;
    if (!InitModelInstance(instance1, modelPath, cpu_use)) {
        std::cerr << "Failed to initialize instance " << instance1 << std::endl;
    }

    std::cout << "Initializing instance " << instance2 << "..." << std::endl;
    if (!InitModelInstance(instance2, modelPath, cpu_use)) {
        std::cerr << "Failed to initialize instance " << instance2 << std::endl;
    }

    // Prepare different data for each instance
    int elem1 = GetInstanceElementsPerSample(instance1);
    int elem2 = GetInstanceElementsPerSample(instance2);
    std::cout << "Instance " << instance1 << " elements per sample: " << elem1 << std::endl;
    std::cout << "Instance " << instance2 << " elements per sample: " << elem2 << std::endl;

    std::vector<float> data1(elem1, 1.0f);  // Fill with 1.0
    std::vector<float> data2(elem2, 2.0f);  // Fill with 2.0

    // Run inference on both instances
    std::cout << "\nRunning inference on instance " << instance1 << "..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    RunModelInstanceFloat(instance1, data1.data(), elem1);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Instance " << instance1 << " inference time: " << duration.count() << " ms" << std::endl;

    std::cout << "\nRunning inference on instance " << instance2 << "..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    RunModelInstanceFloat(instance2, data2.data(), elem2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Instance " << instance2 << " inference time: " << duration.count() << " ms" << std::endl;

    // Get outputs from each instance
    int size1, size2;
    const float* out1 = GetInstanceFlattenedOutput(instance1, &size1);
    const float* out2 = GetInstanceFlattenedOutput(instance2, &size2);

    std::cout << "\nInstance " << instance1 << " output: size=" << size1 << ", first=" << out1[0] << std::endl;
    std::cout << "Instance " << instance2 << " output: size=" << size2 << ", first=" << out2[0] << std::endl;

    // Verify outputs are different (different input data)
    bool outputs_different = (out1[0] != out2[0]);
    std::cout << "Outputs are different: " << (outputs_different ? "YES" : "NO") << std::endl;

    // Destroy instances
    std::cout << "\nDestroying instances..." << std::endl;
    DestroyModelInstance(instance1);
    DestroyModelInstance(instance2);
    std::cout << "Remaining instance count: " << GetModelInstanceCount() << std::endl;

    // ===== Logging System Test =====
    std::cout << "\n--- Logging System Test ---" << std::endl;

    // Test different log levels
    std::cout << "\n1. Testing log levels (current: INFO)" << std::endl;
    LogDebug("This DEBUG message should NOT appear (level too low)");
    LogInfo("This INFO message should appear");
    LogWarn("This WARN message should appear");
    LogError("This ERROR message should appear");

    // Change log level to DEBUG
    std::cout << "\n2. Setting log level to DEBUG" << std::endl;
    SetLogLevel(0);  // DEBUG
    LogDebug("This DEBUG message should NOW appear");
    LogInfo("INFO after level change");

    // Test file logging
    std::cout << "\n3. Testing file logging" << std::endl;
    if (SetLogFile("airunner_test.log")) {
        LogInfo("This message goes to both console and file");
        LogWarn("Warning message in file");
        LogError("Error message in file");
        std::cout << "Log file created: airunner_test.log" << std::endl;
    }

    // Test disabling console
    std::cout << "\n4. Testing console disable (next messages only in file)" << std::endl;
    EnableLogConsole(false);
    LogInfo("This message only goes to file, not console");
    LogWarn("This warning only in file");
    EnableLogConsole(true);
    LogInfo("Console re-enabled - this appears on console");

    // Test timestamp toggle
    std::cout << "\n5. Testing timestamp disable" << std::endl;
    EnableLogTimestamp(false);
    LogInfo("Message without timestamp");
    EnableLogTimestamp(true);
    LogInfo("Message with timestamp restored");

    // Test log level NONE (disable all)
    std::cout << "\n6. Testing NONE level (all logging disabled)" << std::endl;
    SetLogLevel(4);  // NONE
    LogError("This ERROR should NOT appear (logging disabled)");
    SetLogLevel(1);  // Back to INFO
    LogInfo("Logging re-enabled");

    // Close log file
    CloseLogFile();
    std::cout << "\nLog file closed. Check airunner_test.log for file output." << std::endl;

    std::cout << "\n===== Test Complete =====" << std::endl;
    return 0;
}
