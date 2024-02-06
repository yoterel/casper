#include "quad.h"
#include <glad/glad.h>
#include <iostream>
#include <glm/gtx/normal.hpp>
#include <glm/gtx/norm.hpp>

Quad::Quad(std::vector<glm::vec3> &vertices, bool autoinit)
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
    glm::vec3 normal1 = glm::triangleNormal(vertices[0], vertices[1], vertices[2]);
    glm::vec3 normal2 = glm::triangleNormal(vertices[0], vertices[2], vertices[3]);
    float len = glm::length2(normal1 - normal2);
    if (!(len < EPISILON))
    {
        std::cout << "Warning: Quad is not planar!" << std::endl;
        exit(1);
    }
    m_normal = normal1;
    m_verts = verts;
    if (autoinit)
        this->init();
}
Quad::Quad(std::vector<glm::vec2> &vertices, bool autoinit)
{
    std::vector<float> verts{
        // positions          // colors           // texture coords
        vertices[0].x, vertices[0].y, 0.0f, 0.0f, 1.0f, // top left
        vertices[1].x, vertices[1].y, 0.0f, 0.0f, 0.0f, // bottom left
        vertices[2].x, vertices[2].y, 0.0f, 1.0f, 0.0f, // bottom right

        vertices[0].x, vertices[0].y, 0.0f, 0.0f, 1.0f, // top left
        vertices[2].x, vertices[2].y, 0.0f, 1.0f, 0.0f, // bottom right
        vertices[3].x, vertices[3].y, 0.0f, 1.0f, 1.0f  // top right
    };
    glm::vec3 normal1 = glm::triangleNormal(glm::vec3(vertices[0], 0.0f), glm::vec3(vertices[1], 0.0f), glm::vec3(vertices[2], 0.0f));
    glm::vec3 normal2 = glm::triangleNormal(glm::vec3(vertices[0], 0.0f), glm::vec3(vertices[2], 0.0f), glm::vec3(vertices[3], 0.0f));
    float len = glm::length2(normal1 - normal2);
    if (!(len < EPISILON))
    {
        std::cout << "Warning: Quad is not planar!" << std::endl;
        exit(1);
    }
    m_normal = normal1;
    m_verts = verts;
    if (autoinit)
        this->init();
}
Quad::Quad(float depth, bool autoinit)
{
    std::vector<float> verts{
        -1.0f, 1.0f, depth, 0.0f, 1.0f,  // top left
        -1.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
        1.0f, -1.0f, depth, 1.0f, 0.0f,  // bottom right

        -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
        1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
        1.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    m_verts = verts;
    if (autoinit)
        this->init();
}

Quad::Quad(std::string location, float depth, bool autoinit)
{
    std::vector<float> verts;
    if (location == "top_half")
    {
        verts = {
            -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
            -1.0f, 0.0f, depth, 0.0f, 0.0f, // bottom left
            1.0f, 0.0f, depth, 1.0f, 0.0f,  // bottom right

            -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
            1.0f, 0.0f, depth, 1.0f, 0.0f,  // bottom right
            1.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "bottom_half")
    {
        verts = {
            -1.0f, 0.0f, depth, 0.0f, 1.0f,  // top left
            -1.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
            1.0f, -1.0f, depth, 1.0f, 0.0f,  // bottom right

            -1.0f, 0.0f, depth, 0.0f, 1.0f, // top left
            1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
            1.0f, 0.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "left_half")
    {
        verts = {
            -1.0f, 1.0f, depth, 0.0f, 1.0f,  // top left
            -1.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
            0.0f, -1.0f, depth, 1.0f, 0.0f,  // bottom right

            -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
            0.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
            0.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "right_half")
    {
        verts = {
            0.0f, 1.0f, depth, 0.0f, 1.0f,  // top left
            0.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
            1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right

            0.0f, 1.0f, depth, 0.0f, 1.0f,  // top left
            1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
            1.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "top_left")
    {
        verts = {
            -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
            -1.0f, 0.0f, depth, 0.0f, 0.0f, // bottom left
            0.0f, 0.0f, depth, 1.0f, 0.0f,  // bottom right

            -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
            0.0f, 0.0f, depth, 1.0f, 0.0f,  // bottom right
            0.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "top_right")
    {
        verts = {
            0.0f, 1.0f, depth, 0.0f, 1.0f, // top left
            0.0f, 0.0f, depth, 0.0f, 0.0f, // bottom left
            1.0f, 0.0f, depth, 1.0f, 0.0f, // bottom right

            0.0f, 1.0f, depth, 0.0f, 1.0f,  // top left
            1.0f, 0.0f, depth, 1.0f, 0.0f,  // bottom right
            1.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "bottom_left")
    {
        verts = {
            -1.0f, 0.0f, depth, 0.0f, 1.0f,  // top left
            -1.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
            0.0f, -1.0f, depth, 1.0f, 0.0f,  // bottom right

            -1.0f, 0.0f, depth, 0.0f, 1.0f, // top left
            0.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
            0.0f, 0.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "bottom_right")
    {
        verts = {
            0.0f, 0.0f, depth, 0.0f, 1.0f,  // top left
            0.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
            1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right

            0.0f, 0.0f, depth, 0.0f, 1.0f,  // top left
            1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
            1.0f, 0.0f, depth, 1.0f, 1.0f}; // top right
    }
    else if (location == "tiny_top_right")
    {
        verts = {
            0.5f, 1.0f, depth, 0.0f, 1.0f, // top left
            0.5f, 0.5f, depth, 0.0f, 0.0f, // bottom left
            1.0f, 0.5f, depth, 1.0f, 0.0f, // bottom right

            0.5f, 1.0f, depth, 0.0f, 1.0f,  // top left
            1.0f, 0.5f, depth, 1.0f, 0.0f,  // bottom right
            1.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    }
    else
    {
        verts = {
            -1.0f, 1.0f, depth, 0.0f, 1.0f,  // top left
            -1.0f, -1.0f, depth, 0.0f, 0.0f, // bottom left
            1.0f, -1.0f, depth, 1.0f, 0.0f,  // bottom right

            -1.0f, 1.0f, depth, 0.0f, 1.0f, // top left
            1.0f, -1.0f, depth, 1.0f, 0.0f, // bottom right
            1.0f, 1.0f, depth, 1.0f, 1.0f}; // top right
    }
    m_verts = verts;
    if (autoinit)
        this->init();
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

void Quad::init()
{
    // unsigned int VBO, VAO, EBO;
    if (m_verts.size() != 30) // 6 verts * 5 (xyz + uv)
    {
        std::cout << "Error: Quad missing vertices!" << std::endl;
        exit(1);
    }
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * m_verts.size(), m_verts.data(), GL_STATIC_DRAW);
    // position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);

    //  texture coord attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));

    glBindVertexArray(0);
}
void Quad::render(bool wireFrame, bool points, bool alphaBlend)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    if (!alphaBlend)
        glDisable(GL_BLEND);
    if (wireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBindVertexArray(m_VAO);
    if (points)
        glDrawArrays(GL_POINTS, 0, 6);
    else
        glDrawArrays(GL_TRIANGLES, 0, 6);
    if (wireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (!alphaBlend)
        glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(0);
}