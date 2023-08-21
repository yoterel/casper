#include "quad.h"
#include <glad/glad.h>
#include <iostream>

Quad::Quad(std::vector<glm::vec3> &vertices)
{
    float verts[] = {
        // positions          // colors           // texture coords
        vertices[0].x, vertices[0].y, vertices[0].z, 1.0f, 1.0f, // top right
        vertices[1].x, vertices[1].y, vertices[1].z, 1.0f, 0.0f, // bottom right
        vertices[1].x, vertices[2].y, vertices[2].z, 0.0f, 0.0f, // bottom left
        vertices[3].x, vertices[3].y, vertices[3].z, 0.0f, 1.0f  // top left
    };
    this->init(verts);
}
Quad::Quad(float depth)
{
    float verts[] = {
        // positions          // colors           // texture coords
        1.0f, 1.0f, depth, 1.0f, 1.0f,   // top right
        1.0f, -1.0f, depth, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
        -1.0f, 1.0f, depth, 0.0f, 1.0f   // top left
    };
    this->init(verts);
}

Quad::~Quad()
{
    if (m_VBO != 0)
    {
        glDeleteBuffers(1, &m_VBO);
        m_VBO = 0;
    }
    if (m_EBO != 0)
    {
        glDeleteBuffers(1, &m_EBO);
        m_EBO = 0;
    }
    if (m_VAO != 0)
    {
        glDeleteVertexArrays(1, &m_VAO);
        m_VAO = 0;
    }
}

void Quad::init(float vertices[])
{
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    // unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &m_VAO);
    // glCheckError();
    glGenBuffers(1, &m_VBO);
    // glCheckError();
    glGenBuffers(1, &m_EBO);
    // glCheckError();

    glBindVertexArray(m_VAO);
    // glCheckError();

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    // glCheckError();
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // glCheckError();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    // glCheckError();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    // glCheckError();

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    // glCheckError();
    glEnableVertexAttribArray(0);
    // glCheckError();
    //  texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    // glCheckError();
    glEnableVertexAttribArray(1);
    // glCheckError();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glCheckError();
}
void Quad::render()
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glBindVertexArray(m_VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}