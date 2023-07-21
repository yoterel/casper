#include "skinned_shader.h"

SkinningShader::SkinningShader(const char* vertexPath, const char* fragmentPath, const char* geometryPath) : 
    Shader(vertexPath, fragmentPath, geometryPath)
    {
        worldTransformLoc = GetUniformLocation("gTransform");
        samplerLoc = GetUniformLocation("gSampler");
        samplerSpecularExponentLoc = GetUniformLocation("gSamplerSpecularExponent");
        materialLoc.AmbientColor = GetUniformLocation("gMaterial.AmbientColor");
        materialLoc.DiffuseColor = GetUniformLocation("gMaterial.DiffuseColor");
        materialLoc.SpecularColor = GetUniformLocation("gMaterial.SpecularColor");
        dirLightLoc.Color = GetUniformLocation("gDirectionalLight.Base.Color");
        dirLightLoc.AmbientIntensity = GetUniformLocation("gDirectionalLight.Base.AmbientIntensity");
        dirLightLoc.Direction = GetUniformLocation("gDirectionalLight.Direction");
        dirLightLoc.DiffuseIntensity = GetUniformLocation("gDirectionalLight.Base.DiffuseIntensity");
        CameraLocalPosLoc = GetUniformLocation("gCameraLocalPos");
        NumPointLightsLocation = GetUniformLocation("gNumPointLights");
        NumSpotLightsLocation = GetUniformLocation("gNumSpotLights");
        for (unsigned int i = 0 ; i < sizeof(PointLightsLocation)/sizeof(PointLightsLocation[0]) ; i++) {
        char Name[128];
        memset(Name, 0, sizeof(Name));
        _snprintf_s(Name, sizeof(Name), "gPointLights[%d].Base.Color", i);
        PointLightsLocation[i].Color = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gPointLights[%d].Base.AmbientIntensity", i);
        PointLightsLocation[i].AmbientIntensity = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gPointLights[%d].LocalPos", i);
        PointLightsLocation[i].Position = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gPointLights[%d].Base.DiffuseIntensity", i);
        PointLightsLocation[i].DiffuseIntensity = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gPointLights[%d].Atten.Constant", i);
        PointLightsLocation[i].Atten.Constant = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gPointLights[%d].Atten.Linear", i);
        PointLightsLocation[i].Atten.Linear = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gPointLights[%d].Atten.Exp", i);
        PointLightsLocation[i].Atten.Exp = GetUniformLocation(Name);

        if (PointLightsLocation[i].Color == INVALID_UNIFORM_LOCATION ||
            PointLightsLocation[i].AmbientIntensity == INVALID_UNIFORM_LOCATION ||
            PointLightsLocation[i].Position == INVALID_UNIFORM_LOCATION ||
            PointLightsLocation[i].DiffuseIntensity == INVALID_UNIFORM_LOCATION ||
            PointLightsLocation[i].Atten.Constant == INVALID_UNIFORM_LOCATION ||
            PointLightsLocation[i].Atten.Linear == INVALID_UNIFORM_LOCATION ||
            PointLightsLocation[i].Atten.Exp == INVALID_UNIFORM_LOCATION) {
            std::cout << "Error: PointLightsLocation[i] is invalid" << std::endl;
            exit(0);
        }
    }

    for (unsigned int i = 0 ; i < sizeof(SpotLightsLocation)/sizeof(SpotLightsLocation[0]) ; i++) {
        char Name[128];
        memset(Name, 0, sizeof(Name));
        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Base.Base.Color", i);
        SpotLightsLocation[i].Color = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Base.Base.AmbientIntensity", i);
        SpotLightsLocation[i].AmbientIntensity = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Base.LocalPos", i);
        SpotLightsLocation[i].Position = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Direction", i);
        SpotLightsLocation[i].Direction = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Cutoff", i);
        SpotLightsLocation[i].Cutoff = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Base.Base.DiffuseIntensity", i);
        SpotLightsLocation[i].DiffuseIntensity = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Constant", i);
        SpotLightsLocation[i].Atten.Constant = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Linear", i);
        SpotLightsLocation[i].Atten.Linear = GetUniformLocation(Name);

        _snprintf_s(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Exp", i);
        SpotLightsLocation[i].Atten.Exp = GetUniformLocation(Name);

        if (SpotLightsLocation[i].Color == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].AmbientIntensity == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].Position == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].Direction == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].Cutoff == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].DiffuseIntensity == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].Atten.Constant == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].Atten.Linear == INVALID_UNIFORM_LOCATION ||
            SpotLightsLocation[i].Atten.Exp == INVALID_UNIFORM_LOCATION) {
            std::cout << "Error: SpotLightsLocation[i] is invalid" << std::endl;
            exit(0);
        }
    }

    for (unsigned int i = 0 ; i < sizeof(m_boneLocation)/sizeof(m_boneLocation[0]) ; i++) {
        char Name[128];
        memset(Name, 0, sizeof(Name));
        _snprintf_s(Name, sizeof(Name), "gBones[%d]", i);
        m_boneLocation[i] = GetUniformLocation(Name);
    }
    };

GLint SkinningShader::GetUniformLocation(const char* pUniformName)
{
    GLuint Location = glGetUniformLocation(ID, pUniformName);

    if (Location == INVALID_UNIFORM_LOCATION) {
        fprintf(stderr, "Warning! Unable to get the location of uniform '%s'\n", pUniformName);
    }

    return Location;
}

void SkinningShader::SetWorldTransform(const glm::mat4& worldTransform)
{
    glUniformMatrix4fv(worldTransformLoc, 1, GL_FALSE, glm::value_ptr(worldTransform));
}


void SkinningShader::SetTextureUnit(unsigned int TextureUnit)
{
    glUniform1i(samplerLoc, TextureUnit);
}

void SkinningShader::SetSpecularExponentTextureUnit(unsigned int TextureUnit)
{
    glUniform1i(samplerSpecularExponentLoc, TextureUnit);
}


void SkinningShader::SetDirectionalLight(const DirectionalLight& Light)
{
    glUniform3f(dirLightLoc.Color, Light.Color.x, Light.Color.y, Light.Color.z);
    glUniform1f(dirLightLoc.AmbientIntensity, Light.AmbientIntensity);
    glm::vec3 LocalDirection = Light.GetLocalDirection();
    glUniform3f(dirLightLoc.Direction, LocalDirection.x, LocalDirection.y, LocalDirection.z);
    glUniform1f(dirLightLoc.DiffuseIntensity, Light.DiffuseIntensity);
}


void SkinningShader::SetCameraLocalPos(const glm::vec3& CameraLocalPos)
{
    glUniform3f(CameraLocalPosLoc, CameraLocalPos.x, CameraLocalPos.y, CameraLocalPos.z);
}


void SkinningShader::SetMaterial(const Material& material)
{
    glUniform3f(materialLoc.AmbientColor, material.AmbientColor.r, material.AmbientColor.g, material.AmbientColor.b);
    glUniform3f(materialLoc.DiffuseColor, material.DiffuseColor.r, material.DiffuseColor.g, material.DiffuseColor.b);
    glUniform3f(materialLoc.SpecularColor, material.SpecularColor.r, material.SpecularColor.g, material.SpecularColor.b);
}

void SkinningShader::SetPointLights(unsigned int NumLights, const PointLight* pLights)
{
    glUniform1i(NumPointLightsLocation, NumLights);

    for (unsigned int i = 0 ; i < NumLights ; i++) {
        glUniform3f(PointLightsLocation[i].Color, pLights[i].Color.x, pLights[i].Color.y, pLights[i].Color.z);
        glUniform1f(PointLightsLocation[i].AmbientIntensity, pLights[i].AmbientIntensity);
        glUniform1f(PointLightsLocation[i].DiffuseIntensity, pLights[i].DiffuseIntensity);
        const glm::vec3& LocalPos = pLights[i].GetLocalPosition();
        //LocalPos.Print();printf("\n");
        glUniform3f(PointLightsLocation[i].Position, LocalPos.x, LocalPos.y, LocalPos.z);
        glUniform1f(PointLightsLocation[i].Atten.Constant, pLights[i].Attenuation.Constant);
        glUniform1f(PointLightsLocation[i].Atten.Linear, pLights[i].Attenuation.Linear);
        glUniform1f(PointLightsLocation[i].Atten.Exp, pLights[i].Attenuation.Exp);
    }
}

void SkinningShader::SetSpotLights(unsigned int NumLights, const SpotLight* pLights)
{
    glUniform1i(NumSpotLightsLocation, NumLights);

    for (unsigned int i = 0 ; i < NumLights ; i++) {
        glUniform3f(SpotLightsLocation[i].Color, pLights[i].Color.x, pLights[i].Color.y, pLights[i].Color.z);
        glUniform1f(SpotLightsLocation[i].AmbientIntensity, pLights[i].AmbientIntensity);
        glUniform1f(SpotLightsLocation[i].DiffuseIntensity, pLights[i].DiffuseIntensity);
        const glm::vec3& LocalPos = pLights[i].GetLocalPosition();
        glUniform3f(SpotLightsLocation[i].Position, LocalPos.x, LocalPos.y, LocalPos.z);
        glm::vec3 Direction = pLights[i].GetLocalDirection();
        Direction = glm::normalize(Direction);
        // Direction.Normalize();
        glUniform3f(SpotLightsLocation[i].Direction, Direction.x, Direction.y, Direction.z);
        glUniform1f(SpotLightsLocation[i].Cutoff, cosf(glm::radians(pLights[i].Cutoff)));
        glUniform1f(SpotLightsLocation[i].Atten.Constant, pLights[i].Attenuation.Constant);
        glUniform1f(SpotLightsLocation[i].Atten.Linear,   pLights[i].Attenuation.Linear);
        glUniform1f(SpotLightsLocation[i].Atten.Exp,      pLights[i].Attenuation.Exp);
    }
}


// void SkinningShader::SetDisplayBoneIndex(uint DisplayBoneIndex)
// {
//     glUniform1i(displayBoneIndexLocation, DisplayBoneIndex);
// }

void SkinningShader::SetBoneTransform(unsigned int Index, const glm::mat4& Transform)
{
    //assert(Index < MAX_BONES);
    if (Index >= MAX_BONES) {
        return;
    }
    //Transform.Print();
    glUniformMatrix4fv(m_boneLocation[Index], 1, GL_TRUE, (const GLfloat*)glm::value_ptr(Transform));
}