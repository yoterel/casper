#ifndef CANVAS_H
#define CANVAS_H

#include "shader.h"
#include "opencv2/opencv.hpp"
//GL includes
#include <glad/glad.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define USE_TEXSUBIMAGE2D

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
    ~Canvas(){ Clear(); };
    void Render(Shader& shader, uint8_t* buffer);
    
private:
    void PopulateBuffers();
    // void CreateTexture();
    void Clear();
    void ProcesssWithCuda();
    void checkCudaErrors(int result);
    bool m_use_cuda;
    unsigned int m_VAO = 0;
    unsigned int m_VBO = 0;
    unsigned int m_EBO = 0;
    unsigned int m_PBO = 0;
    unsigned int m_FBO = 0;
    unsigned int m_depth_buffer = 0;
    unsigned int m_texture_dst;  // create texture that will receive the result of CUDA
    unsigned int m_texture_src;  // create texture for blitting onto the screen
    float bg_thresh = 0.05f;
    unsigned int m_srcWidth, m_srcHeight;
    unsigned int m_dstWidth, m_dstHeight;
    struct cudaGraphicsResource *m_cuda_pbo_dest_resource = NULL;
    struct cudaGraphicsResource *m_cuda_tex_result_resource = NULL;
    struct cudaGraphicsResource *m_cuda_tex_screen_resource = NULL; 
    // struct cudaResourceDesc m_resourceDesc;
    // cudaGraphicsResource *m_cudaGraphicsResource = NULL;
    // cudaArray            *m_cudaArray = NULL;
    // /** reference to exture to read data through*/
    // // cudaTextureObject_t m_cudaTexture;
    // /** reference to surface to write data to*/
    // cudaSurfaceObject_t m_surface;
};
#endif