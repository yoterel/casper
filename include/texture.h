#ifndef TEXTURE_H
#define TEXTURE_H

#include <string>
#include <iostream>
#include <glad/glad.h>
#include <vector>

class Texture
{
public:
    Texture(const std::string &FileName, GLenum TextureTarget = GL_TEXTURE_2D);

    Texture(GLenum TextureTarget = GL_TEXTURE_2D);

    ~Texture();
    // Should be called once to load the texture
    bool init_from_file(unsigned int texture_interpolation_mode = GL_LINEAR,
                        unsigned int texture_wrap_mode = GL_CLAMP_TO_BORDER,
                        bool flip_vertically = true);

    bool init_from_file(const std::string &Filename,
                        unsigned int texture_interpolation_mode = GL_LINEAR,
                        unsigned int texture_wrap_mode = GL_CLAMP_TO_BORDER,
                        bool flip_vertically = true);

    void init(int width, int height, int bpp = 4,
              unsigned int input_color_format = GL_BGRA,
              unsigned int texture_color_format = GL_RGBA,
              unsigned int texture_interpolation_mode = GL_LINEAR,
              unsigned int texture_wrap_mode = GL_CLAMP_TO_BORDER);

    void init(uint8_t *buffer, int width, int height, int bpp = 4);

    // Must be called at least once for the specific texture unit
    void bind(GLenum TextureUnit = GL_TEXTURE0);

    void load(uint8_t *buffer, bool use_pbo = true, unsigned int buffer_color_format = GL_BGRA);

    void load(std::vector<uint8_t> buffer, bool use_pbo, unsigned int buffer_color_format = GL_BGRA);

    void getImageSize(int &ImageWidth, int &ImageHeight)
    {
        ImageWidth = m_imageWidth;
        ImageHeight = m_imageHeight;
    }

    GLuint getTexture() const { return m_textureObj; }
    GLuint getActualTextureFormat() const { return m_actualTextureFormat; }

private:
    void initInternal(void *image_data,
                      unsigned int input_color_format = GL_BGRA,
                      unsigned int texture_color_format = GL_RGBA,
                      unsigned int texture_interpolation_mode = GL_LINEAR,
                      unsigned int texture_wrap_mode = GL_CLAMP_TO_BORDER);

    std::string m_fileName = "";
    GLenum m_textureTarget;
    GLuint m_textureObj;
    GLuint m_PBO;
    int m_imageWidth = 0;
    int m_imageHeight = 0;
    int m_imageBPP = 0;
    int m_sizeTexData = 0;
    GLuint m_actualTextureFormat = GL_RGBA;
};

#endif /* TEXTURE_H */
