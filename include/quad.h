#ifndef QUAD_H
#define QUAD_H

class Quad
{
public:
    Quad();
    ~Quad();
    Quad(Quad &s) // copy constructor
    {
        m_VBO = s.m_VBO;
        m_VAO = s.m_VAO;
        m_EBO = s.m_EBO;
    }
    void render();

private:
    unsigned int m_VBO;
    unsigned int m_VAO;
    unsigned int m_EBO;
};
#endif