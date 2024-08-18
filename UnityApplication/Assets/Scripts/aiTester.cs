using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class AiTester : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello, this is AI Running Tester");

        bool cpuUse = true;
        string modelPath = "/Users/kwon/Workspace/C/ai-run-library/SampleModel/best_metric_model_0.7747.onnx";

        int[] dimensions = { 1, 4, 128, 128, 80 };
        int totalElements = 1;

        // Calculate the total number of elements
        foreach (var dim in dimensions)
        {
            totalElements *= dim;
        }

        // Create an array to hold the flattened data
        float[] data = new float[totalElements];

        // Fill the data array with values
        for (int i = 0; i < totalElements; ++i)
        {
            data[i] = i % 100;  // Values between 0 and 99
        }

        // Initialize the model
        if (!AiRunner.InitModel(modelPath, cpuUse))
        {
            Debug.Log("Failed to initialize model.");
            return;
        }

        // Run the model
        AiRunner.RunModelFloat(data, totalElements);

        // Get the final result
        int size = 0;
        IntPtr outputPtr = AiRunner.GetFlattenedOutput(ref size);

        // Convert the output to a float array
        float[] output = new float[size];
        Marshal.Copy(outputPtr, output, 0, size);

        Debug.Log("Final output size is " + size);
        Debug.Log("First element is " + output[0]);
    }

    // Update is called once per frame
    void Update()
    {
        // Optional: Add any update logic here
    }
}
