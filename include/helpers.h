#ifndef HELPERS_H
#define HELPERS_H
#include <vector>
#include <glm/glm.hpp>

class Helpers
{
public:
    static glm::vec2 ScreenToNDC(const glm::vec2 &pixel, int width, int height, bool flip_y = false);
    static glm::vec2 NDCtoScreen(const glm::vec2 &NDC, int width, int height, bool flip_y = false);

private:
    Helpers();
};

#endif HELPERS_H