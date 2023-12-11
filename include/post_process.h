
#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "shader.h"
#include "quad.h"
#include "fbo.h"
#include "skinned_model.h"
#include <glad/glad.h>

class PostProcess
{
public:
    PostProcess(unsigned int srcWidth, unsigned int srcHeight,
                unsigned int dstWidth, unsigned int dstHeight);
    void mask(Shader &mask_shader, unsigned int renderedSceneTexture, unsigned int camTexture, FBO *target_fbo, const float threshold = 0.01f);
    void jump_flood(Shader &jfaInit, Shader &jfa, Shader &NN_shader, unsigned int renderedSceneTexture, unsigned int camTexture, FBO *target_fbo = NULL, const float threshold = 0.01f);
    static glm::mat4 findHomography(std::vector<glm::vec2> screen_verts);
    void optical_flow(Shader &shader, Texture renderedSceneTexture, Texture camTexture);
    void bake(Shader &uvShader, unsigned int textureToBake, unsigned int TextureUV, const std::string &filepath);
    void saveColorToFile(std::string filepath, unsigned int fbo_id);

private:
    void initGLBuffers();
    Quad m_quad;
    unsigned int m_srcWidth, m_srcHeight;
    unsigned int m_dstWidth, m_dstHeight;
    unsigned int m_size_tex_data;
    unsigned int m_num_texels;
    unsigned int m_num_values;
    unsigned int m_depth_buffer[2] = {0};
    unsigned int m_FBO[2] = {0};
    unsigned int m_pingpong_textures[2] = {0};
};
#endif POSTPROCESS