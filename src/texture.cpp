
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "texture.h"

Texture::Texture(const std::string &FileName, GLenum TextureTarget)
{
    m_textureTarget = TextureTarget;
    m_fileName = FileName;
}

Texture::Texture(GLenum TextureTarget)
{
    m_textureTarget = TextureTarget;
}

Texture::~Texture()
{
    if (m_textureObj != 0)
    {
        glDeleteTextures(1, &m_textureObj);
        m_textureObj = 0;
    }
    if (m_PBO != 0)
    {
        glDeleteBuffers(1, &m_PBO);
        m_PBO = 0;
    }
}

bool Texture::init()
{
    stbi_set_flip_vertically_on_load(1);
    unsigned char *image_data = stbi_load(m_fileName.c_str(), &m_imageWidth, &m_imageHeight, &m_imageBPP, 0);
    if (!image_data)
    {
        std::cout << "Can't load texture from '" << m_fileName << "' - " << stbi_failure_reason() << std::endl;
        exit(0);
    }
    m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
    std::cout << "Width " << m_imageWidth << ", height " << m_imageHeight << ", bpp " << m_imageBPP << std::endl;
    initInternal(image_data, GL_RGBA);
    load(image_data, false);
    return true;
}

bool Texture::init(const std::string &Filename)
{
    m_fileName = Filename;
    return init();
}

bool Texture::init(void *pData, uint32_t bufferSize)
{
    uint8_t *image_data = stbi_load_from_memory((const stbi_uc *)pData, bufferSize, &m_imageWidth, &m_imageHeight, &m_imageBPP, 0);
    m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
    initInternal(image_data, GL_RGBA);
    load(image_data, false);
    stbi_image_free(image_data);
    return true;
}

void Texture::init(int width, int height, int bpp, unsigned int input_color_format, unsigned int texture_color_format)
{
    m_imageWidth = width;
    m_imageHeight = height;
    m_imageBPP = bpp;
    m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
    initInternal(NULL, input_color_format, texture_color_format);
}

void Texture::initRaw(unsigned char *pData, int width, int height, int bpp)
{
    m_imageWidth = width;
    m_imageHeight = height;
    m_imageBPP = bpp;
    m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
    initInternal(pData, GL_RGBA);
    load(pData, false);
}

void Texture::initInternal(void *image_data, unsigned int input_color_format, unsigned int texture_color_format)
{
    // pbo
    void *data = malloc(m_sizeTexData);
    glGenBuffers(1, &m_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_sizeTexData, data, GL_STREAM_DRAW);
    free(data);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // texture
    glGenTextures(1, &m_textureObj);
    glBindTexture(m_textureTarget, m_textureObj);

    glTexParameteri(m_textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(m_textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glTexParameterf(m_textureTarget, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(m_textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER); // GL_REPEAT
    glTexParameteri(m_textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    // glGenerateMipmap(m_textureTarget);

    if (m_textureTarget == GL_TEXTURE_2D)
    {
        switch (m_imageBPP)
        {
        case 1:
            glTexImage2D(m_textureTarget, 0, GL_RED, m_imageWidth, m_imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, image_data);
            break;
        case 2:
            glTexImage2D(m_textureTarget, 0, GL_RG32F, m_imageWidth, m_imageHeight, 0, GL_RG, GL_FLOAT, image_data);
            break;
        case 3:
            glTexImage2D(m_textureTarget, 0, GL_RGB, m_imageWidth, m_imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data);
            break;
        case 4:
            glTexImage2D(m_textureTarget, 0, texture_color_format, m_imageWidth, m_imageHeight, 0, input_color_format, GL_UNSIGNED_BYTE, image_data);
            break;

        default:
            std::cout << "not implemented" << std::endl;
            exit(1);
        }
    }
    else
    {
        std::cout << "support for texture target " << m_textureTarget << " is not implemented" << std::endl;
        exit(1);
    }

    glBindTexture(m_textureTarget, 0);
}

void Texture::load(uint8_t *buffer, bool use_pbo)
{
    if (use_pbo)
    {
        // transfer data from memory to GPU texture (using PBO)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, m_sizeTexData, 0, GL_STREAM_DRAW);
        GLubyte *ptr = (GLubyte *)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        if (ptr)
        {
            memcpy(ptr, buffer, m_sizeTexData);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
        }
        glBindTexture(m_textureTarget, m_textureObj);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_imageWidth, m_imageHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, m_textureObj);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_imageWidth, m_imageHeight, GL_BGRA, GL_UNSIGNED_BYTE, buffer);
    }
}

void Texture::bind(GLenum TextureUnit)
{
    glActiveTexture(TextureUnit);
    glBindTexture(m_textureTarget, m_textureObj);
}
