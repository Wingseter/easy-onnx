using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class HelloWorld : MonoBehaviour
{
    [DllImport("aiRunner")]
    private static extern IntPtr GetResponse(string input);

    void Start()
    {
        IntPtr responsePtr = GetResponse("hello");
        string response = Marshal.PtrToStringAnsi(responsePtr);
        Debug.Log("Response from DLL: " + response);
    }
}