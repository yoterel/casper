#include "stb_image.h"
#include "stb_image_write.h"
#include "texture.h"
#include <filesystem>

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

bool Texture::init_from_file(unsigned int texture_interpolation_mode, unsigned int texture_wrap_mode, bool flip_vertically)
{
    if (m_fileName == "")
    {
        std::cout << "Texture file name is empty" << std::endl;
        exit(1);
    }
    if (flip_vertically)
        stbi_set_flip_vertically_on_load(1);
    else
        stbi_set_flip_vertically_on_load(0);
    std::filesystem::path p = m_fileName;
    std::cout << "Loading texture: " << std::filesystem::absolute(p) << std::endl;
    unsigned char *image_data = stbi_load(m_fileName.c_str(), &m_imageWidth, &m_imageHeight, &m_imageBPP, 0);
    if (!image_data)
    {
        std::cout << "Can't load texture from '" << m_fileName << "' - " << stbi_failure_reason() << std::endl;
        exit(0);
    }
    m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
    std::cout << "Width " << m_imageWidth << ", height " << m_imageHeight << ", bpp " << m_imageBPP << std::endl;
    if (m_imageBPP == 1)
    {
        initInternal(image_data, GL_RED, GL_RED, texture_interpolation_mode, texture_wrap_mode);
        load(image_data, false, GL_RED);
    }
    else if (m_imageBPP == 2)
    {
        initInternal(image_data, GL_RG, GL_RG32F, texture_interpolation_mode, texture_wrap_mode);
        load(image_data, false, GL_RG);
    }
    else if (m_imageBPP == 3)
    {
        initInternal(image_data, GL_RGB, GL_RGB, texture_interpolation_mode, texture_wrap_mode);
        load(image_data, false, GL_RGB);
    }
    else
    {
        initInternal(image_data, GL_RGBA, GL_RGBA, texture_interpolation_mode, texture_wrap_mode);
        load(image_data, false, GL_RGBA);
    }
    return true;
}

bool Texture::init_from_file(const std::string &Filename, unsigned int texture_interpolation_mode, unsigned int texture_wrap_mode, bool flip_vertically)
{
    m_fileName = Filename;
    return init_from_file(texture_interpolation_mode, texture_wrap_mode, flip_vertically);
}

// bool Texture::init(void *pData, uint32_t bufferSize)
// {
//     uint8_t *image_data = stbi_load_from_memory((const stbi_uc *)pData, bufferSize, &m_imageWidth, &m_imageHeight, &m_imageBPP, 0);
//     m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
//     initInternal(image_data, GL_RGBA);
//     load(image_data, false);
//     stbi_image_free(image_data);
//     return true;
// }

void Texture::init(int width, int height, int bpp,
                   unsigned int input_color_format,
                   unsigned int texture_color_format,
                   unsigned int texture_interpolation_mode,
                   unsigned int texture_wrap_mode)
{
    m_imageWidth = width;
    m_imageHeight = height;
    m_imageBPP = bpp;
    m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
    initInternal(NULL, input_color_format, texture_color_format, texture_interpolation_mode, texture_wrap_mode);
}

void Texture::init(uint8_t *buffer, int width, int height, int bpp)
{
    m_imageWidth = width;
    m_imageHeight = height;
    m_imageBPP = bpp;
    m_sizeTexData = sizeof(GLubyte) * m_imageWidth * m_imageHeight * m_imageBPP;
    initInternal(buffer, GL_RGBA);
    // load(pData, false);
}

void Texture::initInternal(void *image_data, unsigned int input_color_format,
                           unsigned int texture_color_format,
                           unsigned int texture_interpolation_mode,
                           unsigned int texture_wrap_mode)
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

    glTexParameteri(m_textureTarget, GL_TEXTURE_MIN_FILTER, texture_interpolation_mode); // GL_LINEAR, GL_NEAREST
    glTexParameteri(m_textureTarget, GL_TEXTURE_MAG_FILTER, texture_interpolation_mode); // GL_LINEAR, GL_NEAREST
    // glTexParameterf(m_textureTarget, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(m_textureTarget, GL_TEXTURE_WRAP_S, texture_wrap_mode); // GL_REPEAT, GL_MIRRORED_REPEAT, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE
    glTexParameteri(m_textureTarget, GL_TEXTURE_WRAP_T, texture_wrap_mode);

    // glGenerateMipmap(m_textureTarget);

    if (m_textureTarget == GL_TEXTURE_2D)
    {
        switch (m_imageBPP)
        {
        case 1:
            m_actualTextureFormat = GL_RED;
            glTexImage2D(m_textureTarget, 0, m_actualTextureFormat, m_imageWidth, m_imageHeight, 0, GL_RED, GL_UNSIGNED_BYTE, image_data);
            break;
        case 2:
            m_actualTextureFormat = GL_RG32F;
            glTexImage2D(m_textureTarget, 0, m_actualTextureFormat, m_imageWidth, m_imageHeight, 0, GL_RG, GL_FLOAT, image_data);
            break;
        case 3:
            m_actualTextureFormat = GL_RGB;
            glTexImage2D(m_textureTarget, 0, m_actualTextureFormat, m_imageWidth, m_imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data);
            break;
        case 4:
            m_actualTextureFormat = texture_color_format;
            glTexImage2D(m_textureTarget, 0, m_actualTextureFormat, m_imageWidth, m_imageHeight, 0, input_color_format, GL_UNSIGNED_BYTE, image_data);
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

void Texture::load(uint8_t *buffer, bool use_pbo, unsigned int buffer_color_format)
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
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_imageWidth, m_imageHeight, buffer_color_format, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, m_textureObj);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_imageWidth, m_imageHeight, buffer_color_format, GL_UNSIGNED_BYTE, buffer);
    }
}

void Texture::load(std::vector<uint8_t> buffer, bool use_pbo, unsigned int buffer_color_format)
{
    if (buffer.size() != m_sizeTexData)
    {
        std::cout << "buffer size " << buffer.size() << " does not match texture size " << m_sizeTexData << std::endl;
        exit(1);
    }
    if (use_pbo)
    {
        // transfer data from memory to GPU texture (using PBO)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, m_sizeTexData, 0, GL_STREAM_DRAW);
        GLubyte *ptr = (GLubyte *)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        if (ptr)
        {
            std::copy(buffer.begin(), buffer.end(), ptr);
            // memcpy(ptr, buffer, m_sizeTexData);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
        }
        glBindTexture(m_textureTarget, m_textureObj);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_imageWidth, m_imageHeight, buffer_color_format, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, m_textureObj);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_imageWidth, m_imageHeight, buffer_color_format, GL_UNSIGNED_BYTE, buffer.data());
    }
}

void Texture::bind(GLenum TextureUnit)
{
    glActiveTexture(TextureUnit);
    glBindTexture(m_textureTarget, m_textureObj);
}
