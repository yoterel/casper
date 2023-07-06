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

    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern void buffer_to_image(IntPtr instance, ref Color32[] buffer, int width, int height);
    
    private float update;
    private IntPtr plugin;
    private Texture2D destinationTexture;

    static public void GetRTPixels(RenderTexture rt, Texture2D destinationTexture2D)
    {
        // Remember currently active render texture
        RenderTexture currentActiveRT = RenderTexture.active;
        // Set the supplied RenderTexture as the active one
        RenderTexture.active = rt;
        HorizontallyFlipRenderTexture(rt);

        // Create a new Texture2D and read the RenderTexture image into it
        // Texture2D tex = new Texture2D(rt.width, rt.height);
        destinationTexture2D.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        // tex.Apply();
        // byte[] bytes;
        // bytes = tex.EncodeToPNG();
        
        // string path = "bla.png";
        // System.IO.File.WriteAllBytes(path, bytes);
        // Restore previously active render texture
        RenderTexture.active = currentActiveRT;
    }

    public static void VerticallyFlipRenderTexture(RenderTexture target)
    {
        var temp = RenderTexture.GetTemporary(target.descriptor);
        Graphics.Blit(target, temp, new Vector2(1, -1), new Vector2(0, 1));
        Graphics.Blit(temp, target);
        RenderTexture.ReleaseTemporary(temp);
    }

    public static void HorizontallyFlipRenderTexture(RenderTexture target)
    {
        var temp = RenderTexture.GetTemporary(target.descriptor);
        Graphics.Blit(target, temp, new Vector2(-1, 1), new Vector2(1, 0));
        Graphics.Blit(temp, target);
        RenderTexture.ReleaseTemporary(temp);
    }

    void Start()
    {
        print(SystemInfo.graphicsDeviceName);
        plugin = createUnityPlugin();
        RenderTexture rt = Camera.main.targetTexture;
        destinationTexture = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        bool initialized = initialize_projector(plugin);
        if (initialized)
        {
            Debug.Log("Projector initialized");
        }
        else
        {
            Debug.Log("Projector not initialized");
        }
        
        // Graphics.CopyTexture(RenderTexture.active, destinationTexture2D);
    }

    void Update()
    {
        RenderTexture rt = Camera.main.targetTexture;
        GetRTPixels(rt, destinationTexture);
        var buffer = destinationTexture.GetPixels32();
        buffer_to_image(plugin, ref buffer, destinationTexture.width, destinationTexture.height);
        // projector_show_white(plugin, 1);
    }

    void OnApplicationQuit()
    {
        Debug.Log("Application ending after " + Time.time + " seconds");
        freeUnityPlugin(plugin);
    }
}
