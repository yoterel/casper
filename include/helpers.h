#ifndef HELPERS_H
#define HELPERS_H
#include <vector>
#include <glm/glm.hpp>
#include <string>
#include <opencv2/opencv.hpp>

class Helpers
{
public:
    static glm::vec2 ScreenToNDC(const glm::vec2 &pixel, int width, int height, bool flip_y = false);
    static std::vector<glm::vec2> ScreenToNDC(const std::vector<glm::vec2> &pixels, int width, int height, bool flip_y);
    static glm::vec2 NDCtoScreen(const glm::vec2 &NDC, int width, int height, bool flip_y = false);
    static std::vector<glm::vec2> NDCtoScreen(const std::vector<glm::vec2> &NDCs, int width, int height, bool flip_y);
    static void UV2NDC(std::vector<glm::vec2> &uv);
    static void saveTexture(std::string filepath,
                            unsigned int texture,
                            unsigned int width,
                            unsigned int height,
                            bool flipVertically = false,
                            bool threshold = false);
    static std::vector<float> flatten_glm(std::vector<glm::vec2> vec);
    static std::vector<float> flatten_glm(std::vector<glm::vec3> vec);
    static std::vector<glm::vec2> opencv2glm(std::vector<cv::Point2f> vec);
    static void setupGizmoBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupFrustrumBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupCubeBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupCubeTexturedBuffers(unsigned int &VAO, unsigned int &VBO1, unsigned int &VBO2);
    static void setupSkeletonBuffers(unsigned int &VAO, unsigned int &VBO);
    static float MSE(const std::vector<glm::vec2> &a, const std::vector<glm::vec2> &b /*, std::vector<float> &mse*/);

private:
    Helpers();
};

#endif HELPERS_H