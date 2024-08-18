using System;
using System.Runtime.InteropServices;

class AiRunner
{
    // Importing the functions from the library.
    
    [DllImport("libaiRunner.dylib", EntryPoint = "allCheck", CallingConvention = CallingConvention.Cdecl)]
    public static extern bool AllCheck(string modelPath, bool cpu_use, float[] data, int num_elements);

    [DllImport("libaiRunner.dylib", EntryPoint = "InitModel", CallingConvention = CallingConvention.Cdecl)]
    public static extern bool InitModel(string modelPath, bool cpu_use);

    [DllImport("libaiRunner.dylib", EntryPoint = "RunModelInt", CallingConvention = CallingConvention.Cdecl)]
    public static extern void RunModelInt(int[] data, int num_elements);

    [DllImport("libaiRunner.dylib", EntryPoint = "RunModelFloat", CallingConvention = CallingConvention.Cdecl)]
    public static extern void RunModelFloat(float[] data, int num_elements);

    [DllImport("libaiRunner.dylib", EntryPoint = "RunModelDouble", CallingConvention = CallingConvention.Cdecl)]
    public static extern void RunModelDouble(double[] data, int num_elements);

    [DllImport("libaiRunner.dylib", EntryPoint = "GetFlattenedOutput", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr GetFlattenedOutput(ref int size);

    [DllImport("libaiRunner.dylib", EntryPoint = "GetOriginalShape", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr GetOriginalShape(ref int size);
}