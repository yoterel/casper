#ifndef HELPERS_H
#define HELPERS_H
#include <vector>
#include <glm/glm.hpp>
#include <string>

class Helpers
{
public:
    static glm::vec2 ScreenToNDC(const glm::vec2 &pixel, int width, int height, bool flip_y = false);
    static glm::vec2 NDCtoScreen(const glm::vec2 &NDC, int width, int height, bool flip_y = false);
    static void saveTexture(std::string filepath,
                            unsigned int texture,
                            unsigned int width,
                            unsigned int height,
                            bool flipVertically = false,
                            bool threshold = false);
    static std::vector<float> flatten_glm(std::vector<glm::vec2> vec);
    static std::vector<float> flatten_glm(std::vector<glm::vec3> vec);
    static void setupGizmoBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupFrustrumBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupCubeBuffers(unsigned int &VAO, unsigned int &VBO);
    static void setupCubeTexturedBuffers(unsigned int &VAO, unsigned int &VBO1, unsigned int &VBO2);
    static void setupSkeletonBuffers(unsigned int &VAO, unsigned int &VBO);

private:
    Helpers();
};

#endif HELPERS_H