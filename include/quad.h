#ifndef QUAD_H
#define QUAD_H
#include <vector>
#include <glm/glm.hpp>
#include <string>
#define EPISILON 0.0001f
class Quad
{
public:
    Quad(float depth, bool autoinit = true);
    Quad(std::string location, float depth, bool autoinit = true);
    Quad(std::vector<glm::vec3> &vertices, bool autoinit = true);
    Quad(std::vector<glm::vec2> &vertices, bool autoinit = true);
    ~Quad();
    void render(bool wireFrame = false, bool points = false, bool alphaBlend = false);
    Quad(const Quad &) = delete;
    Quad &operator=(const Quad &) = delete;
    void init();

private:
    unsigned int m_VBO = 0;
    unsigned int m_VAO = 0;
    std::vector<float> m_verts;
    glm::vec3 m_normal;
};
#endif