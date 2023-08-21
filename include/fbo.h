#ifndef FBO_H
#define FBO_H

class FBO
{
public:
    FBO(){};
    FBO(unsigned int width, unsigned int height);
    ~FBO();
    FBO(FBO &s) // copy constructor
    {
        m_width = s.m_width;
        m_height = s.m_height;
        m_texture = s.m_texture;
        m_depthBuffer = s.m_depthBuffer;
        m_FBO = s.m_FBO;
    }
    void init();
    void bind();
    void unbind();
    unsigned int getTexture() { return m_texture; };

private:
    unsigned int m_width, m_height;
    unsigned int m_texture;
    unsigned int m_depthBuffer;
    unsigned int m_FBO;
};
#endif