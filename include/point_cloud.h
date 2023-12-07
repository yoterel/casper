#ifndef POINTCLOUD_H
#define POINTCLOUD_H
#include <vector>
#include <glm/glm.hpp>

class PointCloud
{
public:
    PointCloud(std::vector<glm::vec3> &vertices, std::vector<glm::vec3> &colors);
    PointCloud(std::vector<glm::vec2> &vertices, std::vector<glm::vec3> &colors);
    ~PointCloud();
    void render(float pointSize = 10.0f);
    PointCloud(const PointCloud &) = delete;
    PointCloud &operator=(const PointCloud &) = delete;

private:
    void init(std::vector<float> &verts);
    unsigned int m_VBO = 0;
    unsigned int m_VAO = 0;
    unsigned int m_verts = 0;
};
#endif