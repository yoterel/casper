#ifndef QUAD_H
#define QUAD_H
#include <vector>
#include <glm/glm.hpp>
#include <string>
#define EPISILON 0.0001f
class Quad
{
public:
    Quad(float depth);
    Quad(std::string location, float depth);
    Quad(std::vector<glm::vec3> &vertices);
    Quad(std::vector<glm::vec2> &vertices);
    ~Quad();
    void render(bool wireFrame = false, bool points = false, bool alphaBlend = false);
    Quad(const Quad &) = delete;
    Quad &operator=(const Quad &) = delete;

private:
    void init(std::vector<float> &verts);
    unsigned int m_VBO = 0;
    unsigned int m_VAO = 0;
    glm::vec3 m_normal;
};
#endif