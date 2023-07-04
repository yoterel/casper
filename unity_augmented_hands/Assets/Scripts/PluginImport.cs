using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class PluginImport : MonoBehaviour
{
    //Lets make our calls from the Plugin
    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr createUnityPlugin();

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern void freeUnityPlugin(IntPtr instance);

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern int debug(IntPtr instance, int debug_value);

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern bool initialize_projector(IntPtr instance);

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern void projector_show_white(IntPtr instance, int iterations);

    // [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    // private static unsafe extern void buffer_to_image(IntPtr instance, char* buffer, int width, int height);

    void Start()
    {
        IntPtr plugin = createUnityPlugin();
        Debug.Log(plugin);
        Debug.Log(debug(plugin, 1));
        bool initialized = initialize_projector(plugin);
        if (initialized)
        {
            Debug.Log("Projector initialized");
            projector_show_white(plugin, 1000);
        }
        else
        {
            Debug.Log("Projector not initialized");
        }
        freeUnityPlugin(plugin);
    }
}
