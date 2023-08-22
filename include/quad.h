#ifndef QUAD_H
#define QUAD_H
#include <vector>
#include <glm/glm.hpp>

class Quad
{
public:
    Quad(float depth);
    Quad(std::vector<glm::vec3> &vertices);
    ~Quad();
    void render();
    Quad(const Quad &) = delete;
    Quad &operator=(const Quad &) = delete;

private:
    void init(std::vector<float> &verts);
    unsigned int m_VBO = 0;
    unsigned int m_VAO = 0;
};
#endif