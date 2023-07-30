#include "canvas.h"

Canvas::Canvas(unsigned int srcWidth, unsigned int srcHeight, unsigned int dstWidth, unsigned int dstHeight, bool use_cuda) :
    m_srcWidth(srcWidth),
    m_srcHeight(srcHeight),
    m_dstWidth(dstWidth),
    m_dstHeight(dstHeight),
    m_use_cuda(use_cuda)
{
    PopulateBuffers();
}

void Canvas::Clear()
{
    if (m_PBO != 0) {
        glDeleteBuffers(1, &m_PBO);
        m_PBO = 0;
    }
    if (m_VBO != 0) {
        glDeleteBuffers(1, &m_VBO);
        m_PBO = 0;
    }
    if (m_EBO != 0) {
        glDeleteBuffers(1, &m_EBO);
        m_EBO = 0;
    }
    if (m_VAO != 0) {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
    if (m_texture_src != 0) {
        glDeleteTextures(1, &m_texture_src);
        m_texture_src = 0;
    }
    if (m_texture_dst != 0) {
        glDeleteTextures(1, &m_texture_dst);
        m_texture_dst = 0;
    }
    if (m_depth_buffer != 0) {
        glDeleteRenderbuffers(1, &m_depth_buffer);
        m_depth_buffer = 0;
    }
    if (m_FBO != 0) {
        glDeleteFramebuffers(1, &m_FBO);
        m_FBO = 0;
    }
}

void Canvas::checkCudaErrors(int result)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \n", __FILE__, __LINE__,
                static_cast<unsigned int>(result), cudaGetErrorName((cudaError_t)result));
        exit(EXIT_FAILURE);
    }
}

void Canvas::Render(Shader& shader, uint8_t* buffer)
{

    if (m_use_cuda)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_srcWidth, m_srcHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_RGB, GL_UNSIGNED_BYTE, buffer);
        this->ProcesssWithCuda();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, m_texture_dst);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, m_texture_dst);
        // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_srcWidth, m_srcHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_RGB, GL_UNSIGNED_BYTE, buffer);
        // glBindTexture(GL_TEXTURE_2D, m_texture_dst);
    }
    shader.use();
    shader.setInt("camera_texture", 0);
    shader.setFloat("threshold", bg_thresh);
    // draw texture on quad
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}
void Canvas::ProcesssWithCuda()
{
    cudaArray *in_array;
    unsigned int *out_data;

    #ifdef USE_TEXSUBIMAGE2D
    this->checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    this->checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
      (void **)&out_data, &num_bytes, m_cuda_pbo_dest_resource));
    // printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n",
    // num_bytes, size_tex_data);
    #else
    out_data = cuda_dest_resource;
    #endif
    // map buffer objects to get CUDA device pointers
    checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_tex_screen_resource, 0));
    // printf("Mapping tex_in\n");
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &in_array, m_cuda_tex_screen_resource, 0, 0));

    // calculate grid size
    dim3 block(16, 16, 1);
    // dim3 block(16, 16, 1);
    int radius = 5;
    dim3 grid(m_srcWidth / block.x, m_srcHeight / block.y, 1);
    int sbytes = (block.x + (2 * radius)) * (block.y + (2 * radius)) *
                sizeof(unsigned int);

    // execute CUDA kernel
    launch_cudaProcess(grid, block, sbytes, in_array, out_data, m_srcWidth, m_srcHeight,
                        block.x + (2 * radius), radius, 0.8f, 4.0f);

    this->checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_tex_screen_resource, 0));
    #ifdef USE_TEXSUBIMAGE2D
    this->checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_pbo_dest_resource, 0));
    #endif
    this->checkCudaErrors(cudaDestroyTextureObject(inTexObject));
    #ifdef USE_TEXSUBIMAGE2D
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);

    glBindTexture(GL_TEXTURE_2D, m_texture_dst);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_srcWidth, m_srcHeight, GL_RGBA,
                    GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    #else
    // We want to copy cuda_dest_resource data to the texture
    // map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(
        &texture_ptr, cuda_tex_result_resource, 0, 0));

    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource,
                                        size_tex_data, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
    #endif
    // cudaArray_t cudaArray;
    // unsigned int *out_data;
    // map the CUDA graphics resource to a CUDA device pointer
    // this->checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0));
    // this->checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&m_cudaArray, m_cudaGraphicsResource, 0, 0));
    // m_resourceDesc.res.array.array = m_cudaArray;
    // this->checkCudaErrors(cudaCreateSurfaceObject(&m_surface, &m_resourceDesc));
    // this->checkCudaErrors(cudaGraphicsMapResources(1, &m_cudaTexture, 0));
    // size_t num_bytes;
    // this->checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
    //     (void **)&out_data,
    //     &num_bytes,
    //     m_cudaTexture));
    // dim3 block(16, 16, 1);
    // dim3 block(16, 16, 1);
    // dim3 grid(m_srcWidth / block.x, m_srcHeight / block.y, 1);
    // execute CUDA kernel
    // dim3 blockDim(16, 16);
    // dim3 gridDim((m_srcWidth + blockDim.x - 1) / blockDim.x, (m_srcHeight + blockDim.y - 1) / blockDim.y);
    // cudaSurfaceObject_t surface;
    
    // this->checkCudaErrors(cudaGraphicsResourceGetMappedSurface(&surface, m_cudaTexture, 0, 0));
    // launch_cudaProcess(gridDim, blockDim, m_surface, m_srcWidth, m_srcHeight);
    // Unmap the CUDA graphics resource
    // cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0);
    // this->checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_pbo_dest_resource, 0));
}
void Canvas::PopulateBuffers()
{
    #ifdef USE_TEXSUBIMAGE2D
    // set up vertex data parameter
    unsigned int num_texels = m_srcWidth * m_srcHeight;
    unsigned int num_values = num_texels * 4;
    unsigned int size_tex_data = sizeof(GLubyte) * num_values;
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, &m_PBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_PBO);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    this->checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_pbo_dest_resource, m_PBO,
                                               cudaGraphicsMapFlagsNone));
    #endif
    // texture dst from cuda processing
    glGenTextures(1, &m_texture_dst);
    glBindTexture(GL_TEXTURE_2D, m_texture_dst); 
     // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    #ifdef USE_TEXSUBIMAGE2D
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_srcWidth, m_srcHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    #else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, m_srcWidth, m_srcHeight, 0,
                GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    // register this texture with CUDA
    this->checkCudaErrors(cudaGraphicsGLRegisterImage(
        &m_cuda_tex_result_resource, m_texture_dst, GL_TEXTURE_2D,
        cudaGraphicsMapFlagsWriteDiscard));
    #endif
    // if (proccess_with_cuda)
    // {
    // // resource description for surface
    //     this->checkCudaErrors(cudaGraphicsGLRegisterImage(&m_cudaGraphicsResource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    //     memset(&m_resourceDesc, 0, sizeof(m_resourceDesc));
    //     m_resourceDesc.resType = cudaResourceTypeArray;
    // }
    // texture to receive cuda processing
    // ---------
    glGenTextures(1, &m_texture_src);
    glBindTexture(GL_TEXTURE_2D, m_texture_src); 
     // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // buffer data
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_srcWidth, m_srcHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    #ifndef USE_TEXTURE_RGBA8UI
    printf("Creating a Texture render target GL_RGBA16F\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_srcWidth, m_srcHeight, 0, GL_RGB/* GL_RGBA*/,
                GL_UNSIGNED_BYTE, NULL);
    #else
    printf("Creating a Texture render target GL_RGBA8UI_EXT\n");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, m_srcWidth, m_srcHeight, 0,
                GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    #endif
    // register this texture with CUDA
    this->checkCudaErrors(cudaGraphicsGLRegisterImage(
        &m_cuda_tex_screen_resource, m_texture_src, GL_TEXTURE_2D,
        cudaGraphicsMapFlagsReadOnly));


    // create a renderbuffer
    glGenRenderbuffers(1, &m_depth_buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_depth_buffer);

    // allocate storage
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_srcWidth, m_srcHeight);

    // clean up
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture_src, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_buffer);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    
    // unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);
    glBindVertexArray(m_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}