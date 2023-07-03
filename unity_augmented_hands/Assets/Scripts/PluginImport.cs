using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class PluginImport : MonoBehaviour
{
    //Lets make our calls from the Plugin
    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern int PrintANumber();

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr PrintHello();

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AddTwoIntegers(int i1, int i2);

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern float AddTwoFloats(float f1, float f2);

    void Start()
    {
        Debug.Log(PrintANumber());
        Debug.Log(Marshal.PtrToStringAnsi(PrintHello()));
        Debug.Log(AddTwoIntegers(2, 2));
        Debug.Log(AddTwoFloats(2.5F, 4F));
    }
}
