#ifndef MATERIAL_H
#define MATERIAL_H

#include "texture.h"
#include <glm/glm.hpp>

struct PBRMaterial
{
    float Roughness = 0.0f;
    bool IsMetal = false;
    glm::vec3 Color = glm::vec3(0.0f, 0.0f, 0.0f);
};


class Material {

 public:
    glm::vec3 AmbientColor = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 DiffuseColor = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 SpecularColor = glm::vec3(0.0f, 0.0f, 0.0f);

    PBRMaterial PBRmaterial;

    // TODO: need to deallocate these
    Texture* pDiffuse = NULL; // base color of the material
    Texture* pSpecularExponent = NULL;
};


#endif
