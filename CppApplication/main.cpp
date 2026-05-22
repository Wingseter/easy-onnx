#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include "aiRunner.h"

int main(int argc, char** argv) {
    std::cout << "===== AI Runner Test =====" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << (argc > 0 ? argv[0] : "CppApplication")
                  << " <path/to/model.onnx> [--gpu]" << std::endl;
        std::cerr << "  --gpu : use platform accelerator (CoreML on macOS, DirectML on Windows)" << std::endl;
        return 1;
    }

    const char* modelPath = argv[1];
    bool cpu_use = true;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--gpu") {
            cpu_use = false;
        }
    }
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Mode: " << (cpu_use ? "CPU" : "GPU/accelerator (EP fallback if unavailable)") << std::endl;

    using clock = std::chrono::high_resolution_clock;
    auto ms = [](clock::duration d) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };

    // Primary instance used by scenarios 1-5.
    int model = CreateModelInstance();
    std::cout << "Created model instance: " << model << std::endl;

    if (!InitModelInstance(model, modelPath, cpu_use)) {
        std::cerr << "Failed to initialize model instance." << std::endl;
        return -1;
    }

    int elements_per_sample = GetInstanceElementsPerSample(model);
    std::cout << "Elements per sample: " << elements_per_sample << std::endl;

    // ===== Single Sample Inference Test =====
    std::cout << "\n--- Single Sample Inference ---" << std::endl;
    std::vector<float> single_data(elements_per_sample);
    for (int i = 0; i < elements_per_sample; ++i) {
        single_data[i] = static_cast<float>(i % 100);
    }

    auto start = clock::now();
    bool ok = RunModelInstanceFloat(model, single_data.data(), elements_per_sample);
    auto end = clock::now();

    int size = 0;
    const float* output = GetInstanceFlattenedOutput(model, &size);
    std::cout << "Inference: " << (ok ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Output size: " << size << std::endl;
    if (size > 0) std::cout << "First element: " << output[0] << std::endl;
    std::cout << "Inference time: " << ms(end - start) << " ms" << std::endl;

    // ===== Batch Inference Test =====
    std::cout << "\n--- Batch Inference (batch_size=2) ---" << std::endl;
    int batch_size = 2;
    std::vector<float> batch_data(batch_size * elements_per_sample);
    for (int i = 0; i < batch_size * elements_per_sample; ++i) {
        batch_data[i] = static_cast<float>(i % 100);
    }

    start = clock::now();
    ok = RunModelInstanceBatchFloat(model, batch_data.data(), batch_size, elements_per_sample);
    end = clock::now();

    std::cout << "Batch inference: "
              << (ok ? "SUCCESS" : "FAILED (model may require a fixed batch size)") << std::endl;
    if (ok) {
        output = GetInstanceFlattenedOutput(model, &size);
        int shape_size = 0;
        const int64_t* shape = GetInstanceOriginalShape(model, &shape_size);
        std::cout << "Output size: " << size << std::endl;
        std::cout << "Output shape: [";
        for (int i = 0; i < shape_size; ++i) {
            std::cout << shape[i];
            if (i < shape_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        if (size > 0) std::cout << "First element: " << output[0] << std::endl;
    }
    std::cout << "Batch inference time: " << ms(end - start) << " ms" << std::endl;

    // ===== Performance Comparison =====
    std::cout << "\n--- Performance Comparison ---" << std::endl;

    start = clock::now();
    RunModelInstanceFloat(model, single_data.data(), elements_per_sample);
    RunModelInstanceFloat(model, single_data.data(), elements_per_sample);
    end = clock::now();
    auto single_x2 = ms(end - start);
    std::cout << "2x Single inference: " << single_x2 << " ms" << std::endl;

    start = clock::now();
    RunModelInstanceBatchFloat(model, batch_data.data(), 2, elements_per_sample);
    end = clock::now();
    auto batch_ms = ms(end - start);
    std::cout << "Batch(2) inference: " << batch_ms << " ms" << std::endl;

    if (batch_ms > 0) {
        std::cout << "Speedup: " << (static_cast<double>(single_x2) / batch_ms) << "x" << std::endl;
    }

    // ===== Async Inference Test =====
    std::cout << "\n--- Async Inference Test ---" << std::endl;

    std::cout << "Starting async inference..." << std::endl;
    start = clock::now();
    if (!RunModelInstanceFloatAsync(model, single_data.data(), elements_per_sample)) {
        std::cerr << "Failed to start async inference." << std::endl;
    }

    std::cout << "Inference running in background..." << std::endl;
    int work_count = 0;
    while (IsInstanceInferenceRunning(model)) {
        work_count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "  Working... (" << work_count * 50 << " ms elapsed)" << std::endl;
    }

    bool async_success = GetInstanceAsyncResult(model);
    end = clock::now();

    output = GetInstanceFlattenedOutput(model, &size);
    std::cout << "Async inference completed: " << (async_success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Output size: " << size << std::endl;
    if (size > 0) std::cout << "First element: " << output[0] << std::endl;
    std::cout << "Total time (including background work): " << ms(end - start) << " ms" << std::endl;

    // ===== Async Batch Inference Test =====
    std::cout << "\n--- Async Batch Inference Test ---" << std::endl;

    start = clock::now();
    if (!RunModelInstanceBatchFloatAsync(model, batch_data.data(), 2, elements_per_sample)) {
        std::cerr << "Failed to start async batch inference." << std::endl;
    }

    std::cout << "Batch inference running in background..." << std::endl;
    WaitForInstanceInference(model);

    async_success = GetInstanceAsyncResult(model);
    end = clock::now();

    std::cout << "Async batch inference: "
              << (async_success ? "SUCCESS" : "FAILED (model may require a fixed batch size)") << std::endl;
    if (async_success) {
        int shape_size = 0;
        const int64_t* shape = GetInstanceOriginalShape(model, &shape_size);
        std::cout << "Output shape: [";
        for (int i = 0; i < shape_size; ++i) {
            std::cout << shape[i];
            if (i < shape_size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "Inference time: " << ms(end - start) << " ms" << std::endl;

    // The primary instance is done. Free it before the multi-model scenario so
    // the counts printed there reflect only what that scenario itself creates.
    DestroyModelInstance(model);

    // ===== Multi-Model Management Test =====
    std::cout << "\n--- Multi-Model Management Test ---" << std::endl;

    int instance1 = CreateModelInstance();
    int instance2 = CreateModelInstance();
    std::cout << "Created instances: " << instance1 << ", " << instance2 << std::endl;
    std::cout << "Active instance count: " << GetModelInstanceCount() << std::endl;

    std::cout << "\nInitializing instance " << instance1 << "..." << std::endl;
    if (!InitModelInstance(instance1, modelPath, cpu_use)) {
        std::cerr << "Failed to initialize instance " << instance1 << std::endl;
    }

    std::cout << "Initializing instance " << instance2 << "..." << std::endl;
    if (!InitModelInstance(instance2, modelPath, cpu_use)) {
        std::cerr << "Failed to initialize instance " << instance2 << std::endl;
    }

    int elem1 = GetInstanceElementsPerSample(instance1);
    int elem2 = GetInstanceElementsPerSample(instance2);
    std::cout << "Instance " << instance1 << " elements per sample: " << elem1 << std::endl;
    std::cout << "Instance " << instance2 << " elements per sample: " << elem2 << std::endl;

    std::vector<float> data1(elem1, 1.0f);  // Fill with 1.0
    std::vector<float> data2(elem2, 2.0f);  // Fill with 2.0

    std::cout << "\nRunning inference on instance " << instance1 << "..." << std::endl;
    start = clock::now();
    RunModelInstanceFloat(instance1, data1.data(), elem1);
    end = clock::now();
    std::cout << "Instance " << instance1 << " inference time: " << ms(end - start) << " ms" << std::endl;

    std::cout << "\nRunning inference on instance " << instance2 << "..." << std::endl;
    start = clock::now();
    RunModelInstanceFloat(instance2, data2.data(), elem2);
    end = clock::now();
    std::cout << "Instance " << instance2 << " inference time: " << ms(end - start) << " ms" << std::endl;

    int size1 = 0, size2 = 0;
    const float* out1 = GetInstanceFlattenedOutput(instance1, &size1);
    const float* out2 = GetInstanceFlattenedOutput(instance2, &size2);

    std::cout << "\nInstance " << instance1 << " output: size=" << size1
              << ", first=" << (size1 > 0 ? out1[0] : 0.0f) << std::endl;
    std::cout << "Instance " << instance2 << " output: size=" << size2
              << ", first=" << (size2 > 0 ? out2[0] : 0.0f) << std::endl;

    bool outputs_different = (size1 > 0 && size2 > 0 && out1[0] != out2[0]);
    std::cout << "Outputs are different: " << (outputs_different ? "YES" : "NO") << std::endl;

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
