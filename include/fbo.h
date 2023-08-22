#ifndef FBO_H
#define FBO_H
#include <string>

class FBO
{
public:
    FBO(unsigned int width, unsigned int height);
    ~FBO();
    void init();
    void bind(bool clear = true);
    void unbind();
    void saveColorToFile(std::string filepath);
    unsigned int getTexture() { return m_texture; };

private:
    unsigned int m_width, m_height;
    unsigned int m_texture;
    unsigned int m_depthBuffer;
    unsigned int m_FBO;
};
#endif