#ifndef FBO_H
#define FBO_H
#include <string>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include "texture.h"

class FBO
{
public:
    FBO(unsigned int width, unsigned int height, unsigned int channels = 4, bool auto_init = true);
    ~FBO();
    FBO(const FBO &) = delete;
    FBO &operator=(const FBO &) = delete;
    void init(unsigned int input_color_format = GL_BGRA,
              unsigned int texture_color_format = GL_RGBA,
              unsigned int texture_interpolation_mode = GL_LINEAR,
              unsigned int texture_wrap_mode = GL_CLAMP_TO_BORDER);
    void bind(bool clear = true);
    void unbind();
    void saveColorToFile(std::string filepath, bool flip_vertically = true);
    void saveDepthToFile(std::string filepath, bool flip_vertically = true);
    std::vector<float> sampleDepthBuffer(std::vector<glm::vec2> sample_locations);
    std::vector<uchar> getBuffer(int n_channels);
    Texture *getTexture() { return &m_texture; };

private:
    unsigned int m_width, m_height, m_channels;
    // unsigned int m_texture = 0; // todo: change to texture class
    Texture m_texture;
    unsigned int m_depthBuffer = 0;
    unsigned int m_FBO = 0;
};
#endif