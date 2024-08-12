
# EasyONNX

## Overview
**EasyONNX** is a easy-to-use C++ library designed to simplify the integration of ONNX models into various applications, including those built in C++ and Unity (C#). It abstracts the complexity of model loading, data preprocessing, and inference execution, making it straightforward to apply machine learning models in your projects.

## Key Features

1. **Platform Support**:  
   - **Windows**: (Uses DirectML for accelerated execution.)
   - **macOS**

2. **Flexible Data Handling**:  
   - Supports multiple data types (`int`, `float`, `double`), automatically converting inputs to the appropriate type for model inference.
   - Handles data in a flattened format for easy integration with other data processing workflows.

3. **Simple API**:
   - **Model Initialization**: Load models with minimal configuration.
   - **Inference Execution**: Run models with a single function call.
   - **Result Retrieval**: Get flattened outputs for easy consumption by your application.

4. **Easy Integration with Unity**:
   - **C# Support**: Provides examples and helper functions to integrate with Unity applications.

5. **Extensible**:  
   - The library is written in C++, making it easily extendable and adaptable for various languages and environments.

## API Functions

### 1. Load the Model
```cpp
bool InitModel(const char* modelPath, bool cpu_use);
```
- **modelPath**: Path to the ONNX model file.
- **cpu_use**: Boolean flag to determine if the CPU should be used for inference (true for CPU, false for GPU if supported).

### 2. Inference Execution
```cpp
void RunModelInt(int* data, int num_elements);
void RunModelFloat(float* data, int num_elements);
void RunModelDouble(double* data, int num_elements);
```
- **data**: Pointer to the input data array.
- **num_elements**: Number of elements in the input data array.

### 3. Get Results
```cpp
const float* GetFlattenedOutput(int* size);
```
- **size**: Pointer to an integer that will be set to the size of the flattened output array.
- **Returns**: Pointer to the flattened output data.

## Example Usage

### C++ Example:
```cpp
#include "EasyONNX.h"

int main() {
    // Initialize model
    bool modelLoaded = InitModel("model.onnx", true);
    if (!modelLoaded) {
        std::cerr << "Model loading failed!" << std::endl;
        return -1;
    }

    // Input data
    float inputData[] = { /* your input data */ };
    int numElements = sizeof(inputData) / sizeof(inputData[0]);

    // Run inference
    RunModelFloat(inputData, numElements);

    // Retrieve and print the output
    int outputSize;
    const float* outputData = GetFlattenedOutput(&outputSize);

    for (int i = 0; i < outputSize; ++i) {
        std::cout << outputData[i] << " ";
    }

    return 0;
}
```

### Unity (C#) Example:
```csharp
using System;
using System.Runtime.InteropServices;

public class EasyONNXExample {
    [DllImport("EasyONNX")]
    private static extern bool InitModel(string modelPath, bool cpu_use);

    [DllImport("EasyONNX")]
    private static extern void RunModelFloat(float[] data, int num_elements);

    [DllImport("EasyONNX")]
    private static extern IntPtr GetFlattenedOutput(ref int size);

    public void RunInference() {
        // Initialize model
        if (!InitModel("model.onnx", true)) {
            Console.WriteLine("Model loading failed!");
            return;
        }

        // Input data
        float[] inputData = { /* your input data */ };

        // Run inference
        RunModelFloat(inputData, inputData.Length);

        // Retrieve and process output
        int outputSize = 0;
        IntPtr outputPtr = GetFlattenedOutput(ref outputSize);

        float[] outputData = new float[outputSize];
        Marshal.Copy(outputPtr, outputData, 0, outputSize);

        // Output results
        foreach (float result in outputData) {
            Console.WriteLine(result);
        }
    }
}
```

## Usage

1. **If you want to build yourself**:
   - Navigate to the `CppLibrary` directory.
   - Run `cmake .` to configure the build.
   - Run `make` (or equivalent) to build the library.

2. **Download and Use**:
   - Download the zip file from the release section.
   - Extract it to your target applcation folder

## Documentation and Support
- **Detailed Documentation**: Comprehensive documentation will be provided, including API references, usage examples, and troubleshooting guides.
- **Community and Contributions**: Open for community contributions, with guidelines provided for contributing to the project.

## Conclusion
**EasyONNX** simplifies the process of integrating ONNX models into your applications, allowing you to focus on what matters most—building great software. Whether you're developing in C++ or Unity, this library provides the tools you need to get started quickly and efficiently.
