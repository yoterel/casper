#include "canvas.h"

Canvas::Canvas(unsigned int srcWidth, unsigned int srcHeight, unsigned int dstWidth, unsigned int dstHeight) :
    m_srcWidth(srcWidth),
    m_srcHeight(srcHeight),
    m_dstWidth(dstWidth),
    m_dstHeight(dstHeight)
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
}


void Canvas::Render(Shader& shader, uint8_t* buffer)
{
    // this->ProcesssWithCuda();
    // bind textures on corresponding texture units
    // cv::Mat myimage = cv::Mat(m_dstHeight, m_dstWidth, CV_8UC3, (uint8_t*) buffer);
    // cv::imwrite("test1.png", myimage);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    // load texture to GPU (large overhead)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_dstWidth, m_dstHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, buffer);
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
    // unsigned int *out_data;
    // checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    // size_t num_bytes;
    // checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
    //     (void **)&out_data,
    //     &num_bytes,
    //     cuda_pbo_dest_resource));
    // dim3 block(16, 16, 1);
    // // dim3 block(16, 16, 1);
    // dim3 grid(m_width / block.x, m_height / block.y, 1);
    // // execute CUDA kernel
    // launch_cudaProcess(grid, block, 0, out_data, m_width);
}
void Canvas::PopulateBuffers()
{
    // set up vertex data parameter
    // unsigned int num_texels = m_width * m_height;
    // unsigned int num_values = num_texels * 4;
    // unsigned int size_tex_data = sizeof(GLubyte) * num_values;
    // void *data = malloc(size_tex_data);

    // // create buffer object
    // glGenBuffers(1, m_PBO);
    // glBindBuffer(GL_ARRAY_BUFFER, *m_PBO);
    // glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    // free(data);

    // glBindBuffer(GL_ARRAY_BUFFER, 0);

    // // register this buffer object with CUDA
    // checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *m_PBO,
    //                                             cudaGraphicsMapFlagsNone));

    // SDK_CHECK_ERROR_GL();

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
    // glGenBuffers(1, &PBO);

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

    // texture
    // ---------
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture); 
     // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}