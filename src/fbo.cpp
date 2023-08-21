#include "fbo.h"
#include <glad/glad.h>
#include <iostream>

FBO::FBO(unsigned int width, unsigned int height) : m_width(width), m_height(height)
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
    if (m_texture != 0)
    {
        glDeleteTextures(1, &m_texture);
        // m_texture_dst = 0;
    }
}

void FBO::init()
{
    glGenTextures(1, &m_texture);
    glGenRenderbuffers(1, &m_depthBuffer);
    glGenFramebuffers(1, &m_FBO);
    // glCheckError();
    glBindTexture(GL_TEXTURE_2D, m_texture);
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    // glCheckError();
    glBindRenderbuffer(GL_RENDERBUFFER, m_depthBuffer);
    // glCheckError();
    //  allocate storage
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_width, m_height);
    // glCheckError();
    //  clean up
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    // glCheckError();
    // glCheckError();
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    // glCheckError();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture, 0);
    // glCheckError();
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthBuffer);
    // glCheckError();
    GLenum fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "ERROR: Incomplete framebuffer status." << std::endl;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FBO::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, m_width, m_height);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void FBO::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}