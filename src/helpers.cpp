#include "helpers.h"

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