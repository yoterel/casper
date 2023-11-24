#include "post_process.h"
#include "stb_image_write.h"
#include "GLMhelpers.h"
#include <opencv2/opencv.hpp>

PostProcess::PostProcess(unsigned int srcWidth, unsigned int srcHeight,
                         unsigned int dstWidth, unsigned int dstHeight) : m_srcWidth(srcWidth),
                                                                          m_srcHeight(srcHeight),
                                                                          m_dstWidth(dstWidth),
                                                                          m_dstHeight(dstHeight),
                                                                          m_quad(0.0f)
{
    m_num_texels = m_srcWidth * m_srcHeight;
    m_num_values = m_num_texels * 4;
    m_size_tex_data = sizeof(GLubyte) * m_num_values;
    initGLBuffers();
}

glm::mat4 PostProcess::findHomography(std::vector<glm::vec2> screen_verts)
{
    std::vector<glm::vec2> orig_screen_verts = {{-1.0f, 1.0f},
                                                {-1.0f, -1.0f},
                                                {1.0f, -1.0f},
                                                {1.0f, 1.0f}};
    std::vector<cv::Point2f> origpts, newpts;
    for (int i = 0; i < 4; ++i)
    {
        origpts.push_back(cv::Point2f(orig_screen_verts[i].x, orig_screen_verts[i].y));
        newpts.push_back(cv::Point2f(screen_verts[i].x, screen_verts[i].y));
    }
    cv::Mat1f hom = cv::getPerspectiveTransform(origpts, newpts);
    cv::Mat1f perspective = cv::Mat::zeros(4, 4, CV_32F);
    perspective.at<float>(0, 0) = hom.at<float>(0, 0);
    perspective.at<float>(0, 1) = hom.at<float>(0, 1);
    perspective.at<float>(0, 3) = hom.at<float>(0, 2);
    perspective.at<float>(1, 0) = hom.at<float>(1, 0);
    perspective.at<float>(1, 1) = hom.at<float>(1, 1);
    perspective.at<float>(1, 3) = hom.at<float>(1, 2);
    perspective.at<float>(3, 0) = hom.at<float>(2, 0);
    perspective.at<float>(3, 1) = hom.at<float>(2, 1);
    perspective.at<float>(3, 3) = hom.at<float>(2, 2);
    for (int i = 0; i < 4; ++i)
    {
        cv::Vec4f cord = cv::Vec4f(orig_screen_verts[i].x, orig_screen_verts[i].y, 0.0f, 1.0f);
        cv::Mat tmp = perspective * cv::Mat(cord);
        screen_verts[i].x = tmp.at<float>(0, 0) / tmp.at<float>(3, 0);
        screen_verts[i].y = tmp.at<float>(1, 0) / tmp.at<float>(3, 0);
    }
    // cv::Mat hom4x4 = cv::Mat::eye(4, 4, CV_32FC1);
    // hom.copyTo(hom4x4(cv::Rect(0, 0, 3, 3)));
    glm::mat4 projection;
    GLMHelpers::CV2GLM(perspective, &projection);
    return projection;
}

void PostProcess::jump_flood(Shader &jfaInit, Shader &jfa, Shader &NN_shader, unsigned int renderedSceneTexture, unsigned int camTexture, FBO *target_fbo)
{
    // init jump flood seeds
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderedSceneTexture);
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
        // bind texture from current iter, bind next fbo
        glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[(i + 1) % 2]);
        glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[i % 2]);
        // glViewport(0, 0, m_srcWidth, m_srcHeight);
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    if (target_fbo != NULL)
    {
        target_fbo->bind();
    }
    else
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, m_dstWidth, m_dstHeight);
    }
    // flood fill result will be contained in the second ping pong buffer texture
    // mask result using the camera texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderedSceneTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_pingpong_textures[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, camTexture);
    // glBindFramebuffer(GL_FRAMEBUFFER, m_FBO[0]);
    // unbind FBO and return viewport to default
    NN_shader.use();
    NN_shader.setInt("src", 0);
    NN_shader.setInt("jfa", 1);
    NN_shader.setInt("mask", 2);
    NN_shader.setBool("flipVer", true);
    NN_shader.setBool("flipMaskVer", true);
    NN_shader.setBool("flipMaskHor", true);
    NN_shader.setVec2("resolution", glm::vec2(m_dstWidth, m_dstHeight));
    m_quad.render();
    // result will be contained in the first ping pong buffer texture
    if (target_fbo != NULL)
    {
        target_fbo->unbind();
    }
}

void PostProcess::optical_flow(Shader &shader, Texture renderedSceneTexture, Texture camTexture)
{
}

void PostProcess::saveColorToFile(std::string filepath, unsigned int fbo_id)
{
    unsigned int nrChannels = 4;
    std::vector<char> buffer(m_num_values);
    GLsizei stride = nrChannels * m_dstWidth;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_dstWidth, m_dstHeight, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filepath.c_str(), m_dstWidth, m_dstHeight, 4, buffer.data(), stride);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PostProcess::initGLBuffers()
{
    // todo: replace with FBO class
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