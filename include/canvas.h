#ifndef CANVAS_H
#define CANVAS_H

#include "shader.h"
//GL includes
#include <glad/glad.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "opencv2/opencv.hpp"

#define USE_TEXSUBIMAGE2D

extern "C" void launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                                   unsigned int *g_odata, int imgw);

class Canvas
{
public:
    Canvas(unsigned int m_srcWidth, unsigned int m_srcHeight,
           unsigned int m_dstWidth, unsigned int m_dstHeight);
    ~Canvas(){ Clear(); };
    void Render(Shader& shader, uint8_t* buffer);
    
private:
    void PopulateBuffers();
    // void CreateTexture();
    void Clear();
    void ProcesssWithCuda();
    unsigned int m_VAO = 0;
    unsigned int m_VBO = 0;
    unsigned int m_EBO = 0;
    unsigned int m_PBO = 0;
    unsigned int m_srcWidth;
    unsigned int m_srcHeight;
    unsigned int m_dstWidth, m_dstHeight;
    unsigned int m_texture;
    float bg_thresh = 0.05f;
};
#endif