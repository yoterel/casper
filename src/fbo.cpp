#include "fbo.h"
#include <glad/glad.h>
#include <iostream>
#include <vector>
#include "stb_image_write.h"

FBO::FBO(unsigned int width, unsigned int height, unsigned int channels, bool auto_init) : m_width(width),
                                                                                           m_height(height),
                                                                                           m_channels(channels),
                                                                                           m_texture()
{
    if (auto_init)
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

void FBO::initDepthOnly()
{
    // this function is used for shadow mapping, as only depth is needed
    m_depthOnly = true;
    // create underlying framebuffer
    glGenFramebuffers(1, &m_FBO);
    // and a texture as depth attachment
    // texture class doesn't support depth component yet, so we do it manually here
    glGenTextures(1, &m_depthBuffer);
    glBindTexture(GL_TEXTURE_2D, m_depthBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                 m_width, m_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // bind and attach depth buffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depthBuffer, 0);
    glDrawBuffer(GL_NONE); // signal no color buffer is used
    glReadBuffer(GL_NONE); // signal no color buffer is used
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FBO::init(unsigned int input_color_format,
               unsigned int texture_color_format,
               unsigned int texture_interpolation_mode,
               unsigned int texture_wrap_mode)
{
    // create underlying framebuffer
    glGenFramebuffers(1, &m_FBO);
    // and a texture as color attachment
    m_texture.init(m_width, m_height, m_channels,
                   input_color_format,
                   texture_color_format,
                   texture_interpolation_mode,
                   texture_wrap_mode);
    // and a RBO attachment for depth
    glGenRenderbuffers(1, &m_depthBuffer); // todo: this is fast to write into but very slow for reading out from for depth queries...
    glBindRenderbuffer(GL_RENDERBUFFER, m_depthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_width, m_height); //  allocate storage
    glBindRenderbuffer(GL_RENDERBUFFER, 0);                                          //  cleanup

    // attach the attachments to the framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texture.getTexture(), 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthBuffer);

    // check status
    GLenum fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "ERROR: Incomplete framebuffer status." << std::endl;
    }
    // cleanup
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void FBO::bind(bool clear, glm::vec4 clear_color)
{
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    if (clear)
    {
        glViewport(0, 0, m_width, m_height);
        if (m_depthOnly)
        {
            glClear(GL_DEPTH_BUFFER_BIT);
        }
        else
        {
            glClearColor(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
    }
}

void FBO::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// saves the color buffer to file
void FBO::saveColorToFile(std::string filepath, bool flip_vertically)
{
    if (m_depthOnly)
    {
        std::cout << "FBO ERROR: Cannot save color buffer of depth-only FBO." << std::endl;
        return;
    }
    std::vector<unsigned char> buffer(m_width * m_height * 4);
    GLsizei stride = 4 * m_width;
    this->bind(false);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, buffer.data());
    if (flip_vertically)
        stbi_flip_vertically_on_write(true);
    else
        stbi_flip_vertically_on_write(false);
    stbi_write_png(filepath.c_str(), m_width, m_height, 4, buffer.data(), stride);
    this->unbind();
}

// saves the entire depth buffer to file
// near and far are used to linearize the depth buffer
// depth is converted to 8bit and saved as a grayscale image (visualization purposes)
void FBO::saveDepthToFile(std::string filepath, bool flip_vertically, float near, float far)
{
    std::vector<float> raw_depth_buffer(m_width * m_height);
    GLsizei stride = m_width;
    this->bind(false);
    glReadBuffer(GL_DEPTH_ATTACHMENT);
    glReadPixels(0, 0, m_width, m_height, GL_DEPTH_COMPONENT, GL_FLOAT, raw_depth_buffer.data());
    if (flip_vertically)
        stbi_flip_vertically_on_write(true);
    else
        stbi_flip_vertically_on_write(false);
    std::vector<uint8_t> buffer_uint8(m_width * m_height);
    for (int i = 0; i < raw_depth_buffer.size(); i++)
    {
        float ndc = raw_depth_buffer[i] * 2.0 - 1.0;
        float linearDepth = (2.0 * near * far) / (far + near - ndc * (far - near));
        float normalizedLinearDepth = (linearDepth - near) / (far - near);
        buffer_uint8[i] = static_cast<uint8_t>((1 - normalizedLinearDepth) * 255);
    }
    stbi_write_png(filepath.c_str(), m_width, m_height, 1, buffer_uint8.data(), stride);
    this->unbind();
}

// samples the depth buffer at the given locations
std::vector<float> FBO::sampleDepthBuffer(std::vector<glm::vec2> sample_locations)
{
    std::vector<float> depthBufferSamples;
    this->bind(false);
    glReadBuffer(GL_DEPTH_ATTACHMENT);
    for (int i = 0; i < sample_locations.size(); i++)
    {
        float depth;
        int point_x = static_cast<int>(sample_locations[i].x);
        int point_y = static_cast<int>(sample_locations[i].y);
        glReadPixels(point_x, point_y, // Cast 2D coordinates to GLint
                     1, 1,             // Reading one pixel
                     GL_DEPTH_COMPONENT, GL_FLOAT,
                     &depth);
        depthBufferSamples.push_back(depth);
    }
    this->unbind();
    return depthBufferSamples;
}
// Return the rendered texture as a vector of floats
// n_channels is the number of channels in output.
// width and height are determined by framebuffer size.
std::vector<uint8_t> FBO::getBuffer(int n_channels)
{
    if (m_depthOnly)
    {
        std::cout << "FBO ERROR: Cannot get color buffer of depth-only FBO." << std::endl;
        exit(1);
    }
    int texFormat;
    switch (n_channels)
    {
    case 4:
        texFormat = GL_RGBA;
        break;
    case 3:
        texFormat = GL_RGB;
        break;
    case 2:
        texFormat = GL_RG;
        break;
    case 1:
        texFormat = GL_RED;
        break;
    default:
        std::cout << "FBO ERROR: Unsupported texture format." << std::endl;
        exit(1);
    }
    std::vector<uint8_t> buffer(m_width * m_height * n_channels);
    this->bind(false);
    // glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_width, m_height, texFormat, GL_UNSIGNED_BYTE, buffer.data());
    this->unbind();
    return buffer;
}

unsigned int FBO::getDepthTexture()
{
    if (m_depthOnly)
    {
        return m_depthBuffer;
    }
    else
    {
        std::cout << "FBO ERROR: Cannot get depth as a texture, as RBO is being used (you should init with initDepthOnly())." << std::endl;
        exit(1);
    };
}