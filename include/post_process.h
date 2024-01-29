
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
    void mask(Shader *mask_shader, unsigned int renderedSceneTexture, unsigned int camTexture, FBO *target_fbo, const float threshold = 0.01f);
    void jump_flood(Shader &jfaInit, Shader &jfa, Shader &NN_shader,
                    unsigned int renderedSceneTexture, unsigned int camTexture, FBO *target_fbo = NULL,
                    const float threshold = 0.01f, const float distance_threshold = 50.0f);
    void jump_flood_uv(Shader &jfaInit, Shader &jfa, Shader &uv_NN_shader,
                       unsigned int uvTexture, unsigned int uvUnwrappedTexture, unsigned int camTexture, FBO *target_fbo,
                       const float threshold, const float distance_threshold = 50.0f, const float seam_threshold = 0.1f);
    static glm::mat4 findHomography(std::vector<glm::vec2> screen_verts);
    void bake(Shader &uvShader, unsigned int textureToBake, unsigned int TextureUV, const std::string &filepath);
    void saveColorToFile(std::string filepath, unsigned int fbo_id);
    cv::Mat icp(cv::Mat render, cv::Mat gray, float threshold, glm::mat4 &transform);
    cv::Mat findFingers(cv::Mat gray, float threshold,
                        std::vector<cv::Point> &fingers,
                        std::vector<cv::Point> &valleys);
    double findPointsDistance(cv::Point a, cv::Point b);
    double findPointsDistanceOnX(cv::Point a, cv::Point b);
    std::vector<cv::Point> findClosestOnX(std::vector<cv::Point> points, cv::Point pivot);
    std::vector<cv::Point> compactOnNeighborhoodMedian(std::vector<cv::Point> points, double max_neighbor_distance);
    bool isFinger(cv::Point a, cv::Point b, cv::Point c, double limit_angle_inf, double limit_angle_sup, cv::Point palm_center, double min_distance_from_palm);
    double findAngle(cv::Point a, cv::Point b, cv::Point c);
    void drawVectorPoints(cv::Mat image, std::vector<cv::Point> points, cv::Scalar color, bool with_numbers);
    void initGLBuffers();

private:
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