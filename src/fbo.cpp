#include "fbo.h"
#include <glad/glad.h>
#include <iostream>
#include <vector>
#include "stb_image_write.h"

FBO::FBO(unsigned int width, unsigned int height, unsigned int channels) : m_width(width), m_height(height), m_channels(channels), m_texture()
{
    init();
}

FBO::~FBO()
{
    if (m_depthBuffer != 0)
    {
        glDeleteRenderbuffers(1, &m_depthBuffer);
        // m_depth_buffer = 0;
    }
    if (m_FBO != 0)
    {
        glDeleteFramebuffers(1, &m_FBO);
        // m_FBO = {0};
    }
}

void FBO::init(unsigned int input_color_format, unsigned int texture_color_format)
{
    m_texture.init(m_width, m_height, m_channels, input_color_format, texture_color_format);
    // glGenTextures(1, &m_texture);
    glGenRenderbuffers(1, &m_depthBuffer);
    glGenFramebuffers(1, &m_FBO);

    // glBindTexture(GL_TEXTURE_2D, m_texture);
    // //  set the texture wrapping parameters
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    // //  set texture filtering parameters
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // if (m_channels == 4)
    // {
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    // }
    // else
    // {
    //     if (m_channels == 3)
    //     {
    //         glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    //     }
    //     else
    //     {
    //         if (m_channels == 2)
    //         {
    //             glTexImage2D(GL_TEXTURE_2D, 0, GL_RG, m_width, m_height, 0, GL_RG, GL_FLOAT, 0);
    //         }
    //         else
    //         {
    //             std::cout << "FBO ERROR: Unsupported number of channels." << std::endl;
    //             exit(1);
    //         }
    //     }
    // }

    glBindRenderbuffer(GL_RENDERBUFFER, m_depthBuffer);
    //  allocate storage
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_width, m_height);
    //  clean up
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture.getTexture(), 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthBuffer);

    GLenum fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "ERROR: Incomplete framebuffer status." << std::endl;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void FBO::bind(bool clear)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    if (clear)
    {
        glViewport(0, 0, m_width, m_height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
}

void FBO::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FBO::saveColorToFile(std::string filepath)
{
    unsigned int nrChannels = 4;
    std::vector<char> buffer(m_width * m_height * nrChannels);
    GLsizei stride = nrChannels * m_width;
    this->bind(false);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filepath.c_str(), m_width, m_height, 4, buffer.data(), stride);
    this->unbind();
}

cv::Mat FBO::toOpenCVMat()
{
    unsigned int nrChannels = 4;
    std::vector<char> buffer(m_width * m_height * nrChannels);
    GLsizei stride = nrChannels * m_width;
    this->bind(false);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
    cv::Mat fbo_image(m_height, m_width, CV_8UC4, buffer.data());
    this->unbind();
    return fbo_image;
}