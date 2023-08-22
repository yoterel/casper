#ifndef CANVAS_H
#define CANVAS_H

#include "shader.h"
#include "timer.h"
#include "opencv2/opencv.hpp"
// GL includes
#include <glad/glad.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "quad.h"
#include "fbo.h"
// #define USE_TEXSUBIMAGE2D
// #define USE_TEXTURE_RGBA8UI
extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                                   cudaArray *g_data, unsigned int *g_odata,
                                   int imgw, int imgh, int tilew, int radius,
                                   float threshold, float highlight);

extern cudaTextureObject_t inTexObject;
class Canvas
{
public:
    Canvas(unsigned int m_srcWidth, unsigned int m_srcHeight,
           unsigned int m_dstWidth, unsigned int m_dstHeight,
           bool use_cuda);
    ~Canvas() { Clear(); };
    void Render(Shader &shader, uint8_t *buffer);
    void Render(Shader &jfaInit, Shader &jfa, Shader &fast_tracker,
                unsigned int texture, uint8_t *buffer, bool use_pbo = true);
    void RenderBuffer(Shader &shader, uint8_t *buffer, Quad &quad, bool use_pbo = true);
    void RenderTexture(Shader &shader, unsigned int texture, Quad &quad);
    void RenderTexture(Shader &shader, unsigned int texture);
    void getTimerValues(double &time0, double &time1, double &time2);
    void resetTimers();

private:
    void initGLBuffers();
    // void CreateTexture();
    void Clear();
    void ProcesssWithCuda();
    void ProcesssWithGL();
    bool m_use_cuda;
    Quad m_quad;
    unsigned int m_depth_buffer[2] = {0};
    unsigned int m_FBO[2] = {0};
    unsigned int m_pingpong_textures[2] = {0};
    // unsigned int m_texture_dst;  // create texture that will receive the result of CUDA
    unsigned int m_texture_src; // create texture for blitting onto the screen
    float bg_thresh = 0.05f;
    unsigned int m_srcWidth, m_srcHeight;
    unsigned int m_dstWidth, m_dstHeight;
    // #ifdef USE_TEXSUBIMAGE2D
    unsigned int m_PBO = 0;
    struct cudaGraphicsResource *m_PBO_CUDA = NULL;
    // #else
    unsigned int *m_cuda_dest_resource = NULL;
    struct cudaGraphicsResource *m_cuda_tex_result_resource = NULL;
    // #endif
    struct cudaGraphicsResource *m_cuda_tex_screen_resource = NULL;
    unsigned int m_size_tex_data;
    unsigned int m_num_texels;
    unsigned int m_num_values;
    Timer t0, t1, t2;
    // struct cudaResourceDesc m_resourceDesc;
    // cudaGraphicsResource *m_cudaGraphicsResource = NULL;
    // cudaArray            *m_cudaArray = NULL;
    // /** reference to exture to read data through*/
    // // cudaTextureObject_t m_cudaTexture;
    // /** reference to surface to write data to*/
    // cudaSurfaceObject_t m_surface;
};
#endif // CANVAS_H