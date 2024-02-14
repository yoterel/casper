#ifndef TEXT_H
#define TEXT_H

#include <glad/glad.h>
#include <iostream>
#include <ft2build.h>
#include <glm/glm.hpp>
#include <map>
#include <vector>
#include "shader.h"
#include FT_FREETYPE_H

struct Character {
    unsigned int TextureID; // ID handle of the glyph texture
    glm::ivec2   Size;      // Size of glyph
    glm::ivec2   Bearing;   // Offset from baseline to left/top of glyph
    unsigned int Advance;   // Horizontal offset to advance to next glyph
};

class TextModel
{
public:
    TextModel(const std::string& Filename);
    void Render(Shader &shader, std::string text, float x, float y, float scale, glm::vec3 color);
private:
    std::map<GLchar, Character> Characters;
    unsigned int VAO, VBO;
};

#endif