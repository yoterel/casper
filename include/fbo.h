#ifndef FBO_H
#define FBO_H
#include <string>

class FBO
{
public:
    FBO(unsigned int width, unsigned int height, unsigned int channels = 4);
    ~FBO();
    FBO(const FBO &) = delete;
    FBO &operator=(const FBO &) = delete;
    void init();
    void bind(bool clear = true);
    void unbind();
    void saveColorToFile(std::string filepath);
    unsigned int getTexture() { return m_texture; };

private:
    unsigned int m_width, m_height, m_channels;
    unsigned int m_texture = 0;
    unsigned int m_depthBuffer = 0;
    unsigned int m_FBO = 0;
};
#endif