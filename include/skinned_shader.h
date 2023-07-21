#ifndef SKINNING_TECHNIQUE_H
#define SKINNING_TECHNIQUE_H

#include "shader.h"
#include "material.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define MAX_BONES (200)
#define INVALID_UNIFORM_LOCATION 0xffffffff

class BaseLight
{
public:
    glm::vec3 Color = glm::vec3(1.0f, 1.0f, 1.0f);
    float AmbientIntensity = 0.0f;
    float DiffuseIntensity = 0.0f;
};


class DirectionalLight : public BaseLight
{
public:
    glm::vec3 WorldDirection = glm::vec3(0.0f, 0.0f, 0.0f);

    void CalcLocalDirection(const glm::mat4& worldTransform);

    const glm::vec3& GetLocalDirection() const { return LocalDirection; }

private:
    glm::vec3 LocalDirection = glm::vec3(0.0f, 0.0f, 0.0f);
};

struct LightAttenuation
{
    float Constant = 1.0f;
    float Linear = 0.0f;
    float Exp = 0.0f;
};


class PointLight: public BaseLight
{
public:
    glm::vec3 WorldPosition = glm::vec3(0.0f, 0.0f, 0.0f);
    LightAttenuation Attenuation;

    void CalcLocalPosition(const glm::mat4& worldTransform);

    const glm::vec3& GetLocalPosition() const { return LocalPosition; }

private:
    glm::vec3 LocalPosition = glm::vec3(0.0f, 0.0f, 0.0f);
};

class SpotLight : public PointLight
{
public:
    glm::vec3 WorldDirection = glm::vec3(0.0f, 0.0f, 0.0f);
    float Cutoff = 0.0f;

    void CalcLocalDirectionAndPosition(const glm::mat4& worldTransform);

    const glm::vec3& GetLocalDirection() const { return LocalDirection; }

private:
    glm::vec3 LocalDirection = glm::vec3(0.0f, 0.0f, 0.0f);


};


class SkinningShader : public Shader
{
public:

    static const unsigned int MAX_POINT_LIGHTS = 2;
    static const unsigned int MAX_SPOT_LIGHTS = 2;

    SkinningShader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);

    void SetWorldTransform(const glm::mat4& worldTransform);
    void SetTextureUnit(unsigned int TextureUnit);
    void SetSpecularExponentTextureUnit(unsigned int TextureUnit);
    void SetDirectionalLight(const DirectionalLight& Light);
    void SetPointLights(unsigned int NumLights, const PointLight* pLights);
    void SetSpotLights(unsigned int NumLights, const SpotLight* pLights);
    void SetCameraLocalPos(const glm::vec3& CameraLocalPos);
    void SetMaterial(const Material& material);
    void SetBoneTransform(unsigned int Index, const glm::mat4& Transform);
    GLint GetUniformLocation(const char* pUniformName);
    // void SetDisplayBoneIndex(uint DisplayBoneIndex);
    
private:

    GLuint worldTransformLoc;
    GLuint samplerLoc;
    GLuint samplerSpecularExponentLoc;
    GLuint CameraLocalPosLoc;
    GLuint NumPointLightsLocation;
    GLuint NumSpotLightsLocation;

    struct {
        GLuint AmbientColor;
        GLuint DiffuseColor;
        GLuint SpecularColor;
    } materialLoc;

    struct {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint Direction;
        GLuint DiffuseIntensity;
    } dirLightLoc;

    struct {
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

struct {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint DiffuseIntensity;
        GLuint Position;
        GLuint Direction;
        GLuint Cutoff;
        struct {
            GLuint Constant;
            GLuint Linear;
            GLuint Exp;
        } Atten;
    } SpotLightsLocation[MAX_SPOT_LIGHTS];
    GLuint m_boneLocation[MAX_BONES];
    // GLuint displayBoneIndexLocation;
};


#endif  /* SKINNING_TECHNIQUE_H */
