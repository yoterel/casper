#include "canvas.h"
#include "utils.h"
#include "image_process.h"

Canvas::Canvas(unsigned int srcWidth, unsigned int srcHeight, unsigned int dstWidth, unsigned int dstHeight, bool use_cuda) : m_srcWidth(srcWidth),
                                                                                                                              m_srcHeight(srcHeight),
                                                                                                                              m_dstWidth(dstWidth),
                                                                                                                              m_dstHeight(dstHeight),
                                                                                                                              m_use_cuda(use_cuda),
                                                                                                                              m_quad(0.0f),
                                                                                                                              m_fbo(dstWidth, dstHeight)
{
    m_num_texels = m_srcWidth * m_srcHeight;
    m_num_values = m_num_texels * 4;
    m_size_tex_data = sizeof(GLubyte) * m_num_values;
    initGLBuffers();
    // initCUDABuffers();
}

void Canvas::clear()
{
    if (m_PBO != 0)
    {
        if (m_use_cuda)
        {
            cudaGraphicsUnregisterResource(m_PBO_CUDA);
        }
        glDeleteBuffers(1, &m_PBO);
        m_PBO = 0;
    }
    // unregister this buffer object with CUDA
    // checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_tex_screen_resource));
    // #ifdef USE_TEXSUBIMAGE2D
    // checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_pbo_dest_resource));
    // #else
    // cudaFree(m_cuda_dest_resource);
    // #endif
    if (m_texture_src != 0)
    {
        glDeleteTextures(1, &m_texture_src);
        m_texture_src = 0;
    }
    if (m_depth_buffer[0] != 0)
    {
        glDeleteRenderbuffers(2, m_depth_buffer);
        // m_depth_buffer = 0;
    }
    if (m_FBO[0] != 0)
    {
        glDeleteFramebuffers(2, m_FBO);
        // m_FBO = {0};
    }
    if (m_pingpong_textures[0] != 0)
    {
        glDeleteTextures(2, m_pingpong_textures);
        // m_texture_dst = 0;
    }
}

void Canvas::renderBuffer(Shader &shader, Quad &quad)
{
    glBindTexture(GL_TEXTURE_2D, m_texture_src);
    shader.setInt("src", 0);
    quad.render();
    glBindTexture(GL_TEXTURE_2D, 0);
}

unsigned int Canvas::renderToFBO(Shader &shader, Quad &quad)
{
    m_fbo.bind();
    quad.render();
    m_fbo.unbind();
    return m_fbo.getTexture();
}

unsigned int Canvas::renderToFBO(unsigned int tex, Shader &shader, Quad &quad)
{
    // saveImage("test.png", tex, m_dstWidth, m_dstHeight, shader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    m_fbo.bind();
    shader.use();
    shader.setInt("src", 0);
    quad.render();
    m_fbo.unbind();
    glBindTexture(GL_TEXTURE_2D, 0);
    // m_fbo.saveColorToFile("test.png");
    return m_fbo.getTexture();
}

void Canvas::renderTexture(unsigned int tex, Shader &shader, Quad &quad)
{
    // saveImage("test.png", tex, m_dstWidth, m_dstHeight, shader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    shader.use();
    shader.setInt("src", 0);
    shader.setBool("flipVer", false);
    quad.render();
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Canvas::renderTexture(unsigned int tex, Shader &shader)
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    m_quad.render();
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Canvas::render(Shader &jfaInit, Shader &jfa, Shader &fast_tracker, unsigned int texture, unsigned int cam_texture, bool use_pbo)
{
    // if (use_pbo)
    // {
    //     t0.start();
    //     // transfer data from memory to GPU texture (using PBO)
    //     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
    //     glBufferData(GL_PIXEL_UNPACK_BUFFER, m_size_tex_data, 0, GL_STREAM_DRAW);
    //     GLubyte *ptr = (GLubyte *)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    //     if (ptr)
    //     {
    //         memcpy(ptr, buffer, m_size_tex_data);
    //         glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
    //     }
    //     t0.stop();
    //     t1.start();
    //     glBindTexture(GL_TEXTURE_2D, m_texture_src);
    //     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    //     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    //     t1.stop();
    // }
    // else
    // {
    //     t1.start();
    //     glBindTexture(GL_TEXTURE_2D, m_texture_src);
    //     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_BGRA, GL_UNSIGNED_BYTE, buffer);
    //     t1.stop();
    // }
    t2.start();
    // init jfa seeds
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[0]);
    jfaInit.use();
    jfaInit.setInt("src", 0);
    jfaInit.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    glViewport(0, 0, m_dstWidth, m_dstHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_quad.render();
    glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[0]);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[1]);
    glViewport(0, 0, m_dstWidth, m_dstHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* sanity (render to screen)*/
    // debug.use();
    // debug.setInt("jfa", 0);
    // debug.setInt("src", 1);
    // debug.setBool("flipVer", false);
    // debug.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // glActiveTexture(GL_TEXTURE0);
    // glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[0]);
    // glActiveTexture(GL_TEXTURE1);
    // glBindTexture(GL_TEXTURE_2D, texture);
    // glViewport(0, 0, m_dstWidth, m_dstHeight);
    // glDisable(GL_DEPTH_TEST);
    // glDisable(GL_CULL_FACE);
    // glDisable(GL_BLEND);
    // glBindVertexArray(m_VAO);
    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    // glEnable(GL_CULL_FACE);
    // glEnable(GL_BLEND);
    // glEnable(GL_DEPTH_TEST);
    /* sanity */

    // jump flood
    int numPasses = 11;
    for (int i = 0; i < numPasses; i++)
    {
        jfa.use();
        jfa.setInt("src", 0);
        jfa.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
        jfa.setInt("pass", i);
        jfa.setInt("numPasses", numPasses);
        m_quad.render();
        glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[(i + 1) % 2]);
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[i % 2]);
        // glViewport(0, 0, m_srcWidth, m_srcHeight);
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    // finally render to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, m_dstWidth, m_dstHeight);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[1]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, cam_texture);
    fast_tracker.use();
    fast_tracker.setInt("jfa", 0);
    fast_tracker.setInt("src", 1);
    fast_tracker.setInt("cam_image", 2);
    fast_tracker.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    fast_tracker.setBool("flipVer", false);
    m_quad.render();
    t2.stop();
}
void Canvas::render(Shader &shader, uint8_t *buffer)
{
    if (m_use_cuda)
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, m_size_tex_data, 0, GL_STREAM_DRAW);
        GLubyte *ptr = (GLubyte *)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        if (ptr)
        {
            memcpy(ptr, buffer, m_size_tex_data);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        processsWithCuda();
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
        glBindTexture(GL_TEXTURE_2D, m_texture_src);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, m_size_tex_data, 0, GL_STREAM_DRAW);
        GLubyte *ptr = (GLubyte *)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        if (ptr)
        {
            memcpy(ptr, buffer, m_size_tex_data);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
        }
        glBindTexture(GL_TEXTURE_2D, m_texture_src);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // glBindTexture(GL_TEXTURE_2D, m_texture_dst);
        // glCheckError();
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_srcWidth, m_srcHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, buffer);
        // glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_BGRA, GL_UNSIGNED_BYTE, buffer);
        // glCheckError();
        // uint8_t* colorBuffer = new uint8_t[m_srcWidth*m_srcHeight*3];
        // glReadBuffer(GL_FRONT);
        // //glCheckError();
        // glReadPixels(0, 0, m_srcWidth, m_srcHeight, GL_BGR, GL_UNSIGNED_BYTE, colorBuffer);
        // //glCheckError();
        // cv::Mat img(m_srcHeight, m_srcWidth, CV_8UC3, colorBuffer);
        // cv::imwrite("test.png", img);
        // //glCheckError();
        // glBindTexture(GL_TEXTURE_2D, m_texture_dst);
    }
    // second pass: render the texture onto quad
    shader.use();
    shader.setInt("src", 0);
    shader.setFloat("threshold", bg_thresh);
    shader.setBool("flipVer", true);
    // draw texture on quad
    m_quad.render();
}

void Canvas::processsWithCuda()
{
    uint8_t *out_data;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &m_PBO_CUDA, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&out_data,
                                         &num_bytes,
                                         m_PBO_CUDA);
    // do work
    // NPP_wrapper::process(out_data, m_srcWidth, m_srcHeight);
    NPP_wrapper::distanceTransform(out_data, m_srcWidth, m_srcHeight);
    cudaGraphicsUnmapResources(1, &m_PBO_CUDA, 0);
}

void Canvas::initGLBuffers()
{
    // set up vertex data parameter
    void *data = malloc(m_size_tex_data);

    // create buffer object
    glGenBuffers(1, &m_PBO);
    // glCheckError();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
    // glCheckError();
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_size_tex_data, data, GL_STREAM_DRAW);
    // glCheckError();
    free(data);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    if (m_use_cuda)
    {
        cudaGraphicsGLRegisterBuffer(&m_PBO_CUDA,
                                     m_PBO,
                                     cudaGraphicsMapFlagsWriteDiscard);
    }

    // #ifdef USE_TEXSUBIMAGE2D
    // //glCheckError();

    // // register this buffer object with CUDA
    // checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_pbo_dest_resource, m_PBO,
    //                                            cudaGraphicsMapFlagsNone));
    // #endif
    // texture dst from cuda processing
    // glCheckError();
    // #else
    // if (m_use_cuda)
    // {
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, m_srcWidth, m_srcHeight, 0,
    //                 GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
    //     //glCheckError();
    //     // register this texture with CUDA
    //     checkCudaErrors(cudaGraphicsGLRegisterImage(
    //         &m_cuda_tex_result_resource, m_texture_dst, GL_TEXTURE_2D,
    //         cudaGraphicsMapFlagsWriteDiscard));
    // }
    // else
    // {
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_srcWidth, m_srcHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    //     //glCheckError();
    // }
    // #endif
    // if (proccess_with_cuda)
    // {
    // // resource description for surface
    //     checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    //     memset(&m_resourceDesc, 0, sizeof(m_resourceDesc));
    //     m_resourceDesc.resType = cudaResourceTypeArray;
    // }
    // texture to receive cuda processing
    // ---------
    glGenTextures(1, &m_texture_src);
    // glCheckError();
    glBindTexture(GL_TEXTURE_2D, m_texture_src);
    // glCheckError();
    //  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA, m_srcWidth, m_srcHeight);
    //  set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER); // GL_REPEAT, GL_CLAMP_TO_BORDER
    // glCheckError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    // glCheckError();
    //  set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glCheckError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glCheckError();
    //  buffer data
    //  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_srcWidth, m_srcHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    //  #ifndef USE_TEXTURE_RGBA8UI
    //  printf("Creating a Texture render target GL_RGBA16F\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_srcWidth, m_srcHeight, 0, GL_BGRA,
                 GL_UNSIGNED_BYTE, NULL);
    // glCheckError();
    //  glGenerateMipmap(GL_TEXTURE_2D);
    // glCheckError();
    //  #else
    //  printf("Creating a Texture render target GL_RGBA8UI\n");
    //  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, m_srcWidth, m_srcHeight, 0,
    //              GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
    //  #endif
    //  register this texture with CUDA
    //  if (m_use_cuda)
    //  {
    //  checkCudaErrors(cudaGraphicsGLRegisterImage(
    //  &m_cuda_tex_screen_resource, m_texture_src, GL_TEXTURE_2D,
    //  cudaGraphicsMapFlagsReadOnly));
    // }

    // create fbos
    glGenTextures(2, m_pingpong_textures);
    glGenRenderbuffers(2, m_depth_buffer);
    glGenFramebuffers(2, m_FBO);
    for (int i = 0; i < 2; i++)
    {
        // glCheckError();
        glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[i]);
        // glCheckError();
        //  set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
        // glCheckError();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        // glCheckError();
        //  set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        // glCheckError();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // glCheckError();
        //  #ifdef USE_TEXSUBIMAGE2D
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_dstWidth, m_dstHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        // glCheckError();
        glBindRenderbuffer(GL_RENDERBUFFER, m_depth_buffer[i]);
        // glCheckError();
        //  allocate storage
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_dstWidth, m_dstHeight);
        // glCheckError();
        //  clean up
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        // glCheckError();
        // glCheckError();
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[i]);
        // glCheckError();
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_pingpong_textures[i], 0);
        // glCheckError();
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_buffer[i]);
        // glCheckError();
        GLenum fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cout << "ERROR: Incomplete framebuffer status." << std::endl;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    // glCheckError();
}
void Canvas::getTimerValues(double &time0, double &time1, double &time2)
{
    time0 = t0.averageLapInMilliSec();
    time1 = t1.averageLapInMilliSec();
    time2 = t2.averageLapInMilliSec();
}
void Canvas::resetTimers()
{
    t0.reset();
    t1.reset();
    t2.reset();
}
// #ifndef USE_TEXSUBIMAGE2D
// void Canvas::initCUDABuffers() {
// set up vertex data parameter
//   checkCudaErrors(cudaMalloc((void **)&m_cuda_dest_resource, m_size_tex_data));
// checkCudaErrors(cudaHostAlloc((void**)&cuda_dest_resource, size_tex_data,
// ));
// }
// #endif

// void Canvas::ProcesssWithCuda()
// {
//     cudaArray *in_array;
//     unsigned int *out_data;

//     #ifdef USE_TEXSUBIMAGE2D
//     checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_pbo_dest_resource, 0));
//     size_t num_bytes;
//     checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
//       (void **)&out_data, &num_bytes, m_cuda_pbo_dest_resource));
//     // printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n",
//     // num_bytes, size_tex_data);
//     #else
//     out_data = m_cuda_dest_resource;
//     #endif
//     // map buffer objects to get CUDA device pointers
//     checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_tex_screen_resource, 0));
//     // printf("Mapping tex_in\n");
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
//         &in_array, m_cuda_tex_screen_resource, 0, 0));

//     // calculate grid size
//     dim3 block(16, 16, 1);
//     // dim3 block(16, 16, 1);
//     int radius = 5;
//     dim3 grid(m_srcWidth / block.x, m_srcHeight / block.y, 1);
//     int sbytes = (block.x + (2 * radius)) * (block.y + (2 * radius)) *
//                 sizeof(unsigned int);

//     // execute CUDA kernel
//     launch_cudaProcess(grid, block, sbytes, in_array, out_data, m_srcWidth, m_srcHeight,
//                         block.x + (2 * radius), radius, 0.8f, 4.0f);

//     checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_tex_screen_resource, 0));
//     #ifdef USE_TEXSUBIMAGE2D
//     checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_pbo_dest_resource, 0));
//     #endif
//     checkCudaErrors(cudaDestroyTextureObject(inTexObject));
//     #ifdef USE_TEXSUBIMAGE2D
//     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
//     glBindTexture(GL_TEXTURE_2D, m_texture_dst);
//     //glCheckError();
//     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_RGBA,
//                     GL_UNSIGNED_BYTE, NULL);
//     //glCheckError();
//     glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
//     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//     #else
//     // We want to copy cuda_dest_resource data to the texture
//     // map buffer objects to get CUDA device pointers
//     cudaArray *texture_ptr;
//     checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_tex_result_resource, 0));
//     checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
//         &texture_ptr, m_cuda_tex_result_resource, 0, 0));

//     int num_texels = m_srcWidth * m_srcHeight;
//     int num_values = num_texels * 4;
//     int size_tex_data = sizeof(GLubyte) * num_values;
//     checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, m_cuda_dest_resource,
//                                         size_tex_data, cudaMemcpyDeviceToDevice));

//     checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_tex_result_resource, 0));
//     #endif
//     // cudaArray_t cudaArray;
//     // unsigned int *out_data;
//     // map the CUDA graphics resource to a CUDA device pointer
//     // checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0));
//     // checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArray, m_cudaGraphicsResource, 0, 0));
//     // m_resourceDesc.res.array.array = m_cudaArray;
//     // checkCudaErrors(cudaCreateSurfaceObject(&m_surface, &m_resourceDesc));
//     // checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaTexture, 0));
//     // size_t num_bytes;
//     // checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
//     //     (void **)&out_data,
//     //     &num_bytes,
//     //     m_cudaTexture));
//     // dim3 block(16, 16, 1);
//     // dim3 block(16, 16, 1);
//     // dim3 grid(m_srcWidth / block.x, m_srcHeight / block.y, 1);
//     // execute CUDA kernel
//     // dim3 blockDim(16, 16);
//     // dim3 gridDim((m_srcWidth + blockDim.x - 1) / blockDim.x, (m_srcHeight + blockDim.y - 1) / blockDim.y);
//     // cudaSurfaceObject_t surface;

//     // checkCudaErrors(cudaGraphicsResourceGetMappedSurface(&surface, m_cudaTexture, 0, 0));
//     // launch_cudaProcess(gridDim, blockDim, m_surface, m_srcWidth, m_srcHeight);
//     // Unmap the CUDA graphics resource
//     // cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
//     // checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_pbo_dest_resource, 0));
// }
