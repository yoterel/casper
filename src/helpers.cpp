#include "helpers.h"
#include "fbo.h"
#include "quad.h"
#include "shader.h"

glm::vec2 Helpers::ScreenToNDC(const glm::vec2 &pixel, int width, int height, bool flip_y)
{
    glm::vec2 uv;
    uv.x = (2.0f * pixel.x / width) - 1.0f;
    uv.y = ((2.0f * pixel.y / height) - 1.0f);
    if (flip_y)
    {
        uv.y *= -1.0f;
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
