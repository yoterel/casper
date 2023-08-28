#include "quad.h"
#include <glad/glad.h>
#include <iostream>

Quad::Quad(std::vector<glm::vec3> &vertices)
{
    std::vector<float> verts{
        // positions          // colors           // texture coords
        vertices[0].x, vertices[0].y, vertices[0].z, 0.0f, 1.0f, // top left
        vertices[1].x, vertices[1].y, vertices[1].z, 0.0f, 0.0f, // bottom left
        vertices[2].x, vertices[2].y, vertices[2].z, 1.0f, 0.0f, // bottom right

        vertices[0].x, vertices[0].y, vertices[0].z, 0.0f, 1.0f, // top left
        vertices[2].x, vertices[2].y, vertices[2].z, 1.0f, 0.0f, // bottom right
        vertices[3].x, vertices[3].y, vertices[3].z, 1.0f, 1.0f  // top right
    };
    this->init(verts);
}
Quad::Quad(float depth)
{
    std::vector<float> verts{
        -1.0f, 1.0f, depth, 0.0f, 1.0f,  // top left
        -1.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
        1.0f, -1.0f, depth, 1.0f, 0.0f,  // bottom right

        -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
        1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
        1.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    this->init(verts);
}

Quad::~Quad()
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

void Quad::init(std::vector<float> &verts)
{
    // unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * verts.size(), verts.data(), GL_STATIC_DRAW);
    // position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);

    //  texture coord attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}
void Quad::render(bool wireFrame)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    if (wireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    if (wireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(0);
}