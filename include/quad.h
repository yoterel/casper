#ifndef QUAD_H
#define QUAD_H
#include <vector>
#include <glm/glm.hpp>

class Quad
{
public:
    Quad(float depth = 0.0f);
    Quad(std::vector<glm::vec3> &vertices);
    ~Quad();
    Quad(Quad &s) // copy constructor
    {
        m_VBO = s.m_VBO;
        m_VAO = s.m_VAO;
    }
    void render();

private:
    void init(std::vector<float> &verts);
    unsigned int m_VBO;
    unsigned int m_VAO;
};
#endif