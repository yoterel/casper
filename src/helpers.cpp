#include "helpers.h"
#include "fbo.h"
#include "quad.h"
#include "shader.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <numeric>

std::vector<glm::vec2> Helpers::vec3to2(std::vector<glm::vec3> vec)
{
    std::vector<glm::vec2> vec2;
    for (int i = 0; i < vec.size(); i++)
    {
        vec2.push_back(glm::vec2(vec[i].x, vec[i].y));
    }
    return vec2;
}

glm::vec2 Helpers::ScreenToNDC(const glm::vec2 &pixel, int width, int height, bool flip_y)
{
    glm::vec2 uv;
    uv.x = (2.0f * pixel.x / width) - 1.0f;
    uv.y = (2.0f * pixel.y / height) - 1.0f;
    if (flip_y)
    {
        uv.y *= -1.0f;
    }
    return uv;
}

std::vector<glm::vec2> Helpers::ScreenToNDC(const std::vector<glm::vec2> &pixels, int width, int height, bool flip_y)
{
    std::vector<glm::vec2> uv;
    for (int i = 0; i < pixels.size(); i++)
    {
        uv.push_back(ScreenToNDC(pixels[i], width, height, flip_y));
    }
    return uv;
}

glm::vec2 Helpers::NDCtoScreen(const glm::vec2 &NDC, int width, int height, bool flip_y)
{
    glm::vec2 pixel;
    float multiplier = flip_y ? -1.0f : 1.0f;
    pixel.x = ((width - 1.0f) * (NDC.x + 1.0f) * 0.5f);
    pixel.y = ((height - 1.0f) * (multiplier * NDC.y + 1.0f) * 0.5f);
    return pixel;
}

std::vector<glm::vec2> Helpers::NDCtoScreen(const std::vector<glm::vec2> &NDCs, int width, int height, bool flip_y)
{
    std::vector<glm::vec2> pixels;
    for (int i = 0; i < NDCs.size(); i++)
    {
        pixels.push_back(NDCtoScreen(NDCs[i], width, height, flip_y));
    }
    return pixels;
}

void Helpers::UV2NDC(std::vector<glm::vec2> &uv)
{
    for (int i = 0; i < uv.size(); i++)
    {
        uv[i] = glm::vec2(uv[i].x * 2.0f - 1.0f, uv[i].y * 2.0f - 1.0f);
    }
}

void Helpers::saveTexture(std::string filepath,
                          unsigned int texture,
                          unsigned int width,
                          unsigned int height,
                          bool flipVertically,
                          bool threshold)
{
    unsigned int nrChannels = 4;
    GLsizei stride = nrChannels * width;
    FBO fbo(width, height);
    Quad quad(0.0f);
    Shader textureShader("../../src/shaders/color_by_texture.vs", "../../src/shaders/color_by_texture.fs");
    textureShader.use();
    textureShader.setInt("src", 0);
    textureShader.setMat4("view", glm::mat4(1.0f));
    textureShader.setMat4("projection", glm::mat4(1.0f));
    textureShader.setMat4("model", glm::mat4(1.0f));
    textureShader.setBool("flipHor", false);
    textureShader.setBool("flipVer", flipVertically);
    textureShader.setBool("binary", false);
    textureShader.setBool("isGray", true);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    fbo.bind();
    quad.render();
    fbo.unbind();
    fbo.saveColorToFile(filepath);
}

std::vector<float> Helpers::flatten_glm(std::vector<glm::vec2> vec)
{
    std::vector<float> flat_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        flat_vec.push_back(vec[i].x);
        flat_vec.push_back(vec[i].y);
    }
    return flat_vec;
}

std::vector<float> Helpers::flatten_glm(std::vector<glm::vec3> vec)
{
    std::vector<float> flat_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        flat_vec.push_back(vec[i].x);
        flat_vec.push_back(vec[i].y);
        flat_vec.push_back(vec[i].z);
    }
    return flat_vec;
}

std::vector<double> Helpers::flatten_cv(std::vector<cv::Point> vec)
{
    std::vector<double> flat_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        flat_vec.push_back(static_cast<double>(vec[i].x));
        flat_vec.push_back(static_cast<double>(vec[i].y));
    }
    return flat_vec;
}

std::vector<glm::vec2> Helpers::flatten_2dgrid(cv::Mat grid)
{
    std::vector<glm::vec2> flat_vec;
    for (int i = 0; i < grid.cols; i++)
    {
        flat_vec.push_back(glm::vec2(grid.at<float>(0, i), grid.at<float>(1, i)));
    }
    return flat_vec;
}

std::vector<glm::vec2> Helpers::cv2glm(std::vector<cv::Point2f> vec)
{
    std::vector<glm::vec2> glm_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        glm_vec.push_back(glm::vec2(vec[i].x, vec[i].y));
    }
    return glm_vec;
}

std::vector<glm::vec2> Helpers::cv2glm(std::vector<cv::Point> vec)
{
    std::vector<glm::vec2> glm_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        glm_vec.push_back(glm::vec2(vec[i].x, vec[i].y));
    }
    return glm_vec;
}

std::vector<glm::vec3> Helpers::cv2glm(std::vector<cv::Point3f> vec)
{
    std::vector<glm::vec3> glm_vec;
    for (int i = 0; i < vec.size(); i++)
    {
        glm_vec.push_back(glm::vec3(vec[i].x, vec[i].y, vec[i].z));
    }
    return glm_vec;
}

std::vector<cv::Point2f> Helpers::glm2cv(std::vector<glm::vec2> glm_vec)
{
    std::vector<cv::Point2f> cv_vec;
    for (int i = 0; i < glm_vec.size(); i++)
    {
        cv_vec.push_back(cv::Point2f(glm_vec[i].x, glm_vec[i].y));
    }
    return cv_vec;
}

cv::Point2f Helpers::glm2cv(glm::vec2 glm_vec)
{
    return cv::Point2f(glm_vec.x, glm_vec.y);
}

glm::vec2 Helpers::project_point(glm::vec3 point, glm::mat4 mvp)
{
    glm::vec4 point4 = glm::vec4(point, 1.0f);
    point4 = mvp * point4;
    point4 /= point4.w;
    return glm::vec2(point4.x, point4.y);
}

glm::vec2 Helpers::project_point(glm::vec3 point, glm::mat4 model, glm::mat4 view, glm::mat4 projection)
{
    glm::vec4 point4 = glm::vec4(point, 1.0f);
    point4 = projection * view * model * point4;
    point4 /= point4.w;
    return glm::vec2(point4.x, point4.y);
}

std::vector<glm::vec2> Helpers::project_points(std::vector<glm::vec3> points, glm::mat4 model, glm::mat4 view, glm::mat4 projection)
{
    // project 3d points to NDC space
    std::vector<glm::vec2> projected_points;
    for (int i = 0; i < points.size(); i++)
    {
        glm::vec4 point = glm::vec4(points[i], 1.0f);
        point = projection * view * model * point;
        point /= point.w;
        projected_points.push_back(glm::vec2(point.x, point.y));
    }
    return projected_points;
}

std::vector<glm::vec3> Helpers::project_points_w_depth(std::vector<glm::vec3> points, glm::mat4 model, glm::mat4 view, glm::mat4 projection)
{
    // project 3d points to NDC space
    std::vector<glm::vec3> projected_points;
    for (int i = 0; i < points.size(); i++)
    {
        glm::vec4 point = glm::vec4(points[i], 1.0f);
        point = view * model * point;
        float point_depth = -point.z;
        point = projection * point;
        point /= point.w;
        projected_points.push_back(glm::vec3(point.x, point.y, point_depth));
    }
    return projected_points;
}

void Helpers::setupGizmoBuffers(unsigned int &VAO, unsigned int &VBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions         // colors
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // X
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // X
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y
        0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // Y
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, // Z
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // Z
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void Helpers::setupFrustrumBuffers(unsigned int &VAO, unsigned int &VBO)
{
    float vertices[] = {
        // frame near
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        // frame far
        -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        // connect frames
        -1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        // hat
        -1.0f, 1.0f, -1.0f,
        0.0f, 1.5f, -1.0f,
        0.0f, 1.5f, -1.0f,
        1.0f, 1.0f, -1.0f};
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

void Helpers::setupCubeTexturedBuffers(unsigned int &VAO, unsigned int &VBO1, unsigned int &VBO2)
{
    const GLfloat vertices[] = {
        -1.0f, -1.0f, -1.0f, // triangle 1 : begin
        -1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f, // triangle 1 : end
        1.0f, 1.0f, -1.0f, // triangle 2 : begin
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f, // triangle 2 : end
        1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f};
    const GLfloat uvCoord[] = {
        0.000059f, 1.0f - 0.000004f,
        0.000103f, 1.0f - 0.336048f,
        0.335973f, 1.0f - 0.335903f,
        1.000023f, 1.0f - 0.000013f,
        0.667979f, 1.0f - 0.335851f,
        0.999958f, 1.0f - 0.336064f,
        0.667979f, 1.0f - 0.335851f,
        0.336024f, 1.0f - 0.671877f,
        0.667969f, 1.0f - 0.671889f,
        1.000023f, 1.0f - 0.000013f,
        0.668104f, 1.0f - 0.000013f,
        0.667979f, 1.0f - 0.335851f,
        0.000059f, 1.0f - 0.000004f,
        0.335973f, 1.0f - 0.335903f,
        0.336098f, 1.0f - 0.000071f,
        0.667979f, 1.0f - 0.335851f,
        0.335973f, 1.0f - 0.335903f,
        0.336024f, 1.0f - 0.671877f,
        1.000004f, 1.0f - 0.671847f,
        0.999958f, 1.0f - 0.336064f,
        0.667979f, 1.0f - 0.335851f,
        0.668104f, 1.0f - 0.000013f,
        0.335973f, 1.0f - 0.335903f,
        0.667979f, 1.0f - 0.335851f,
        0.335973f, 1.0f - 0.335903f,
        0.668104f, 1.0f - 0.000013f,
        0.336098f, 1.0f - 0.000071f,
        0.000103f, 1.0f - 0.336048f,
        0.000004f, 1.0f - 0.671870f,
        0.336024f, 1.0f - 0.671877f,
        0.000103f, 1.0f - 0.336048f,
        0.336024f, 1.0f - 0.671877f,
        0.335973f, 1.0f - 0.335903f,
        0.667969f, 1.0f - 0.671889f,
        1.000004f, 1.0f - 0.671847f,
        0.667979f, 1.0f - 0.335851f};
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO1);
    glGenBuffers(1, &VBO2);
    glBindVertexArray(VAO);

    // position attribute
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    // uv coord attribute
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvCoord), uvCoord, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    // to draw: glDrawArrays(GL_TRIANGLES, 0, 36);
}
void Helpers::setupCubeBuffers(unsigned int &VAO, unsigned int &VBO, unsigned int &EBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // unit cube
    // A cube has 6 sides and each side has 4 vertices, therefore, the total number
    // of vertices is 24 (6 sides * 4 verts), and 72 floats in the vertex array
    // since each vertex has 3 components (x,y,z) (= 24 * 3)
    //    v6----- v5
    //   /|      /|
    //  v1------v0|
    //  | |     | |
    //  | v7----|-v4
    //  |/      |/
    //  v2------v3

    // vertex position array
    GLfloat vertices[] = {
        .5f, .5f, .5f, -.5f, .5f, .5f, -.5f, -.5f, .5f, .5f, -.5f, .5f,     // v0,v1,v2,v3 (front)
        .5f, .5f, .5f, .5f, -.5f, .5f, .5f, -.5f, -.5f, .5f, .5f, -.5f,     // v0,v3,v4,v5 (right)
        .5f, .5f, .5f, .5f, .5f, -.5f, -.5f, .5f, -.5f, -.5f, .5f, .5f,     // v0,v5,v6,v1 (top)
        -.5f, .5f, .5f, -.5f, .5f, -.5f, -.5f, -.5f, -.5f, -.5f, -.5f, .5f, // v1,v6,v7,v2 (left)
        -.5f, -.5f, -.5f, .5f, -.5f, -.5f, .5f, -.5f, .5f, -.5f, -.5f, .5f, // v7,v4,v3,v2 (bottom)
        .5f, -.5f, -.5f, -.5f, -.5f, -.5f, -.5f, .5f, -.5f, .5f, .5f, -.5f  // v4,v7,v6,v5 (back)
    };

    // normal array
    GLfloat normals[] = {
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,     // v0,v1,v2,v3 (front)
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,     // v0,v3,v4,v5 (right)
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,     // v0,v5,v6,v1 (top)
        -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, // v1,v6,v7,v2 (left)
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, // v7,v4,v3,v2 (bottom)
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1  // v4,v7,v6,v5 (back)
    };

    // colour array
    GLfloat colors[] = {
        1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, // v0,v1,v2,v3 (front)
        1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, // v0,v3,v4,v5 (right)
        1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, // v0,v5,v6,v1 (top)
        1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, // v1,v6,v7,v2 (left)
        0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, // v7,v4,v3,v2 (bottom)
        0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1  // v4,v7,v6,v5 (back)
    };

    // texture coord array
    GLfloat texCoords[] = {
        1, 0, 0, 0, 0, 1, 1, 1, // v0,v1,v2,v3 (front)
        0, 0, 0, 1, 1, 1, 1, 0, // v0,v3,v4,v5 (right)
        1, 1, 1, 0, 0, 0, 0, 1, // v0,v5,v6,v1 (top)
        1, 0, 0, 0, 0, 1, 1, 1, // v1,v6,v7,v2 (left)
        0, 1, 1, 1, 1, 0, 0, 0, // v7,v4,v3,v2 (bottom)
        0, 1, 1, 1, 1, 0, 0, 0  // v4,v7,v6,v5 (back)
    };

    // index array for glDrawElements()
    // A cube requires 36 indices = 6 sides * 2 tris * 3 verts
    GLuint indices[] = {
        0, 1, 2, 2, 3, 0,       // v0-v1-v2, v2-v3-v0 (front)
        4, 5, 6, 6, 7, 4,       // v0-v3-v4, v4-v5-v0 (right)
        8, 9, 10, 10, 11, 8,    // v0-v5-v6, v6-v1-v0 (top)
        12, 13, 14, 14, 15, 12, // v1-v6-v7, v7-v2-v1 (left)
        16, 17, 18, 18, 19, 16, // v7-v4-v3, v3-v2-v7 (bottom)
        20, 21, 22, 22, 23, 20  // v4-v7-v6, v6-v5-v4 (back)
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices) + sizeof(colors), 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(vertices), sizeof(colors), colors);

    // store index data to VBO
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // enable vertex array attributes for bound VAO
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    // store vertex array pointers to bound VAO
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)sizeof(vertices));

    glBindVertexArray(0);
}

void Helpers::setupSkeletonBuffers(unsigned int &VAO, unsigned int &VBO)
{
    // set up vertex data (and buffer(s)) and configure vertex attributes for hand skeleton (to be uploaded later)
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions         // colors
        0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom right
        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // bottom left
        0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f    // top
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

float Helpers::MSE(const std::vector<glm::vec2> &a, const std::vector<glm::vec2> &b)
{
    float avg_error = 0.0f;
    if (a.size() != b.size())
    {
        std::cout << "ERROR: vectors must be of same size." << std::endl;
        return avg_error;
    }
    for (int i = 0; i < a.size(); i++)
    {
        glm::vec2 diff = a[i] - b[i];
        float error = sqrt(diff.x * diff.x + diff.y * diff.y);
        // mse.push_back(error);
        avg_error += error;
    }
    avg_error /= a.size();
    return avg_error;
}

std::vector<glm::vec2> Helpers::accumulate(const std::vector<std::vector<glm::vec2>> &a, bool normalize)
{
    if (a.size() == 0)
    {
        std::cout << "ERROR: vector of vectors must have at least one vector." << std::endl;
        return std::vector<glm::vec2>();
    }
    unsigned long long reduce_size = a[0].size();
    std::vector<glm::vec2> accumulator(reduce_size, glm::vec2(0.0f, 0.0f));
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < reduce_size; j++)
        {
            accumulator[j] += a[i][j];
        }
    }
    if (normalize)
    {
        for (int i = 0; i < accumulator.size(); i++)
        {
            accumulator[i] /= a.size();
        }
    }
    return accumulator;
}

std::vector<glm::vec3> Helpers::accumulate(const std::vector<std::vector<glm::vec3>> &a, bool normalize)
{
    if (a.size() == 0)
    {
        // std::cout << "ERROR: vector of vectors must have at least one vector." << std::endl;
        return std::vector<glm::vec3>();
    }
    unsigned long long reduce_size = a[0].size();
    std::vector<glm::vec3> accumulator(reduce_size, glm::vec3(0.0f, 0.0f, 0.0f));
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < reduce_size; j++)
        {
            accumulator[j] += a[i][j];
        }
    }
    if (normalize)
    {
        for (int i = 0; i < accumulator.size(); i++)
        {
            accumulator[i] /= a.size();
        }
    }
    return accumulator;
}

std::vector<glm::mat4> Helpers::accumulate(const std::vector<std::vector<glm::mat4>> &a, bool normalize)
{
    if (a.size() == 0)
    {
        // std::cout << "ERROR: vector of vectors must have at least one vector." << std::endl;
        return std::vector<glm::mat4>();
    }
    unsigned long long reduce_size = a[0].size();
    std::vector<glm::mat4> accumulator(reduce_size, glm::mat4(0.0f));
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < reduce_size; j++)
        {
            accumulator[j] += a[i][j];
        }
    }
    if (normalize)
    {
        for (int i = 0; i < accumulator.size(); i++)
        {
            accumulator[i] /= a.size();
        }
    }
    return accumulator;
}

float Helpers::average(std::vector<float> &v)
{
    if (v.empty())
    {
        return 0;
    }

    auto const count = static_cast<float>(v.size());
    return std::reduce(v.begin(), v.end()) / count;
}

glm::mat4 Helpers::interpolate(const glm::mat4 &_mat1, const glm::mat4 &_mat2, float _time, bool prescale, bool isRightHand)
{
    // if you can't join'm, slerp'm.
    glm::mat4 mat1, mat2;
    glm::mat4 scalar = glm::scale(glm::mat4(1.0f), glm::vec3(10.f));
    glm::mat4 inv_scalar = glm::inverse(scalar);
    // first undo the scale, which is the first matrix in the bone transform (slerp requires pure rotation)
    if (prescale)
    {
        mat1 = _mat1 * inv_scalar;
        mat2 = _mat2 * inv_scalar;
    }
    else
    {
        mat1 = _mat1;
        mat2 = _mat2;
    }
    // also undo chirality for right hand, we require det=1
    glm::mat4 flip_x = glm::mat4(1.0f);
    flip_x[0][0] = -1.0f;
    if (isRightHand)
    {
        // formally should have undone the other transforms up to chirality, but they all commute
        mat1 = mat1 * glm::inverse(flip_x);
        mat2 = mat2 * glm::inverse(flip_x);
    }
    // finally slerp the quaternion rep. of the rotation matrix
    glm::quat rot1 = glm::quat_cast(mat1);
    glm::quat rot2 = glm::quat_cast(mat2);
    glm::quat finalRot = glm::slerp(rot1, rot2, _time);
    glm::mat4 finalMat = glm::mat4_cast(finalRot);
    // redo transforms in inverse order
    if (isRightHand)
        finalMat = finalMat * flip_x;
    if (prescale)
        finalMat = finalMat * scalar;
    // regular interpolation for translation component (lerp)
    finalMat[3] = _mat1[3] * (1 - _time) + _mat2[3] * _time; // lerp them for translation though
    return finalMat;
}

bool Helpers::isPalmFacingCamera(glm::mat4 palm_bone, glm::mat4 cam_view_transform)
{
    glm::vec3 palm_normal = glm::vec3(palm_bone * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));
    glm::vec3 cam_normal = glm::vec3(cam_view_transform * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
    float dot_product = glm::dot(palm_normal, cam_normal);
    // std::cout << dot_product << std::endl;
    return dot_product > 0;
}
// void setup_circle_buffers(unsigned int& VAO, unsigned int& VBO)
// {
//     std::vector<glm::vec3> vertices;
//     // std::vector<glm::vec3> colors;
//     float radius = 0.5f;
//     glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);
//     int n_vertices = 50;
//     vertices.push_back(center);
//     vertices.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
//     for (int i = 0; i <= n_vertices; i++)   {
//         float twicePI = 2*glm::pi<float>();
//         vertices.push_back(glm::vec3(center.x + (radius * cos(i * twicePI / n_vertices)),
//                                      center.y,
//                                      center.z + (radius * sin(i * twicePI / n_vertices))));
//         vertices.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
//     }
//     glGenVertexArrays(1, &VAO);
//     glGenBuffers(1, &VBO);
//     glBindVertexArray(VAO);

//     auto test = vertices.data();
//     glBindBuffer(GL_ARRAY_BUFFER, VBO);
//     glBufferData(GL_ARRAY_BUFFER, 6*(n_vertices+2) * sizeof(float), vertices.data(), GL_STATIC_DRAW);

//     // position attribute
//     glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
//     glEnableVertexAttribArray(0);
//     // color attribute
//     glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
//     glEnableVertexAttribArray(1);
// }