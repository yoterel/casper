#include "point_cloud.h"
#include <glad/glad.h>
#include <iostream>

PointCloud::PointCloud(std::vector<glm::vec3> &vertices, std::vector<glm::vec3> &colors)
{
    if (colors.size() != vertices.size())
    {
        std::cout << "Error: colors.size() != vertices.size()" << std::endl;
        exit(1);
    }
    std::vector<float> verts;
    for (int i = 0; i < vertices.size(); i++)
    {
        verts.push_back(vertices[i].x);
        verts.push_back(vertices[i].y);
        verts.push_back(vertices[i].z);
        verts.push_back(colors[i].x);
        verts.push_back(colors[i].y);
        verts.push_back(colors[i].z);
    }
    this->init(verts);
}
PointCloud::PointCloud(std::vector<glm::vec2> &vertices, std::vector<glm::vec3> &colors)
{
    if (colors.size() != vertices.size())
    {
        std::cout << "Error: colors.size() != vertices.size()" << std::endl;
        exit(1);
    }
    std::vector<float> verts;
    for (int i = 0; i < vertices.size(); i++)
    {
        verts.push_back(vertices[i].x);
        verts.push_back(vertices[i].y);
        verts.push_back(0.0f);
        verts.push_back(colors[i].x);
        verts.push_back(colors[i].y);
        verts.push_back(colors[i].z);
    }
    this->init(verts);
}

PointCloud::~PointCloud()
{
    if (m_VBO != 0)
    {
        glDeleteBuffers(1, &m_VBO);
        m_VBO = 0;
    }
    if (m_VAO != 0)
    {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
}

void PointCloud::init(std::vector<float> &verts)
{
    // unsigned int VBO, VAO, EBO;

    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * verts.size(), verts.data(), GL_STATIC_DRAW);
    // position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);

    //  texture coord attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void PointCloud::render()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_POINTS, 0, 6);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(0);
}