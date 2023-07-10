using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Profiling;
using ThreadSupport;

// public class CircularObject
// {
//     public Color32[] buffer = null;
//     public CircularObject() { buffer = null; }
//     public CircularObject(Color32[] inbuffer) { buffer = inbuffer; }
// }
public class PluginImport : MonoBehaviour
{
    // static readonly ProfilerMarker s_readrtMarker = new ProfilerMarker("MySystem.Prepare");
    //Lets make our calls from the Plugin
    [DllImport("AHUnityPlugin", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr createUnityPlugin(int width, int height);

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
    private int proj_width = 1024;
    private int proj_height = 768;
    // public ThreadSupport.CircularObjectBuffer<CircularObject> circular_buffer;

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
        plugin = createUnityPlugin(proj_width, proj_height);
        RenderTexture rt = Camera.main.targetTexture;
        destinationTexture = new Texture2D(proj_width, proj_height, TextureFormat.RGB24, false);
        bool initialized = initialize_projector(plugin);
        if (initialized)
        {
            Debug.Log("Projector initialized");
        }
        else
        {
            Debug.Log("Projector not initialized");
        }
        // circular_buffer = new ThreadSupport.CircularObjectBuffer<CircularObject>(10);
        // Graphics.CopyTexture(RenderTexture.active, destinationTexture2D);
    }

    void Update()
    {
        RenderTexture rt = Camera.main.targetTexture;
        // using (s_readrtMarker.Auto())
        // {
        GetRTPixels(rt, destinationTexture);
        // }
        //Color32[] buffer = new Color32[destinationTexture.width * destinationTexture.height * 4];
        var buffer = destinationTexture.GetPixels32();
        // var buffer2 = destinationTexture.GetPixelData<Color24>(1);
        Task.Run(() => {buffer_to_image(plugin, ref buffer, proj_width, proj_height); });
        // Debug.Log(buffer.Length);
        // for (int i = 0; i < buffer.Length; i++)
        // {
        //     if (buffer[i].r > 0 || buffer[i].g > 0 || buffer[i].b > 0)
        //     {
        //         Debug.Log(i);
        //     }
        // }
        // CircularObject circular_object = new CircularObject(buffer);
        // circular_buffer.Put(ref circular_object);
        // Task.Run(() => {
        //     CircularObject circular_object = new CircularObject();
        //     circular_buffer.Get(out circular_object);
        //     buffer_to_image(plugin, ref circular_object.buffer, proj_width, proj_height); }
        //     );
        
        // projector_show_white(plugin, 1);
    }

    void OnApplicationQuit()
    {
        Debug.Log("Application ending after " + Time.time + " seconds");
        freeUnityPlugin(plugin);
    }
}
