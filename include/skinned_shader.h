#ifndef SKINNED_SHADER_H
#define SKINNED_SHADER_H

#include "shader.h"
#include "material.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define MAX_BONES (50)
#define INVALID_UNIFORM_LOCATION 0xffffffff

class BaseLight
{
public:
    BaseLight(glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f),
              float ambientIntensity = 0.0f,
              float diffuseIntensity = 0.0f) : Color(color),
                                               AmbientIntensity(ambientIntensity),
                                               DiffuseIntensity(diffuseIntensity) {}
    void setColor(const glm::vec3 &color) { Color = color; }
    void setAmbientIntensity(float ambientIntensity) { AmbientIntensity = ambientIntensity; }
    void setDiffuseIntensity(float diffuseIntensity) { DiffuseIntensity = diffuseIntensity; }
    glm::vec3 Color;
    float AmbientIntensity;
    float DiffuseIntensity;
};

class DirectionalLight : public BaseLight
{
public:
    DirectionalLight(glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f),
                     float ambientIntensity = 0.0f,
                     float diffuseIntensity = 0.0f,
                     glm::vec3 worldDir = glm::vec3(0.0f, 0.0f, 0.0f)) : BaseLight(color, ambientIntensity, diffuseIntensity)
    {
        worldDirection = glm::normalize(worldDir);
        localDirection = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    void calcLocalDirection(const glm::mat4 &localToWorld);
    void setWorldDirection(const glm::vec3 worldDir) { worldDirection = glm::normalize(worldDir); }
    const glm::vec3 &getWorldDirection() const { return worldDirection; }
    const glm::vec3 &getLocalDirection() const { return localDirection; }

private:
    glm::vec3 worldDirection;
    glm::vec3 localDirection;
};

struct LightAttenuation
{
    float Constant = 1.0f;
    float Linear = 0.0f;
    float Exp = 0.0f;
};

class PointLight : public BaseLight
{
public:
    glm::vec3 WorldPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    LightAttenuation Attenuation;

    void CalcLocalPosition(const glm::mat4 &worldTransform)
    {
        localPosition = glm::vec3(glm::inverse(worldTransform) * glm::vec4(WorldPosition, 1.0f));
    };

    const glm::vec3 &GetLocalPosition() const { return localPosition; }

private:
    glm::vec3 localPosition = glm::vec3(0.0f, 0.0f, 0.0f);
};

class SpotLight : public PointLight
{
public:
    glm::vec3 WorldDirection = glm::vec3(0.0f, 0.0f, 0.0f);
    float Cutoff = 0.0f;

    void calcLocalDirectionAndPosition(const glm::mat4 &worldTransform);

    const glm::vec3 &GetLocalDirection() const { return LocalDirection; }

private:
    glm::vec3 LocalDirection = glm::vec3(0.0f, 0.0f, 0.0f);
};

class SkinningShader : public Shader
{
public:
    static const unsigned int MAX_POINT_LIGHTS = 2;
    static const unsigned int MAX_SPOT_LIGHTS = 2;

    SkinningShader(const std::string &vertexPath, const std::string &fragmentPath, const std::string &geometryPath = "");

    void SetWorldTransform(const glm::mat4 &worldTransform);
    void SetProjectorTransform(const glm::mat4 &worldTransform);
    void SetTextureUnit(unsigned int TextureUnit);
    void SetSpecularExponentTextureUnit(unsigned int TextureUnit);
    void SetDirectionalLight(const DirectionalLight &Light);
    void SetPointLights(unsigned int NumLights, const PointLight *pLights);
    void SetSpotLights(unsigned int NumLights, const SpotLight *pLights);
    void SetCameraLocalPos(const glm::vec3 &CameraLocalPos);
    void SetMaterial(const Material &material);
    void SetBoneTransform(unsigned int Index, const glm::mat4 &Transform);
    GLint GetUniformLocation(const char *pUniformName);

private:
    GLuint worldTransformLoc;
    GLuint projectorTransformLoc;
    GLuint samplerLoc;
    GLuint samplerSpecularExponentLoc;
    GLuint CameraLocalPosLoc;
    GLuint NumPointLightsLocation;
    GLuint NumSpotLightsLocation;
    // GLuint displayBoneIndexLocation;

    struct
    {
        GLuint AmbientColor;
        GLuint DiffuseColor;
        GLuint SpecularColor;
    } materialLoc;

    struct
    {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint Direction;
        GLuint DiffuseIntensity;
    } dirLightLoc;

    struct
    {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint Position;
        GLuint DiffuseIntensity;

        struct
        {
            GLuint Constant;
            GLuint Linear;
            GLuint Exp;
        } Atten;
    } PointLightsLocation[MAX_POINT_LIGHTS];

    struct
    {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint DiffuseIntensity;
        GLuint Position;
        GLuint Direction;
        GLuint Cutoff;
        struct
        {
            GLuint Constant;
            GLuint Linear;
            GLuint Exp;
        } Atten;
    } SpotLightsLocation[MAX_SPOT_LIGHTS];
    GLuint m_boneLocation[MAX_BONES];
};

#endif /* SKINNED_SHADER_H */
