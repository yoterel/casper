#version 330

const int MAX_POINT_LIGHTS = 2;
const int MAX_SPOT_LIGHTS = 2;

in vec2 TexCoord0;
in vec3 ProjTexCoord;
in vec3 Normal0;
in vec3 LocalPos0;
in vec3 metricColor;
in vec4 LightPos0;
in vec3 Tangent0;
// flat in ivec4 BoneIDs00;
// flat in ivec2 BoneIDs11;
// in vec4 Weights00;
// in vec2 Weights11;
// in vec3 ourColor;
out vec4 FragColor;

struct BaseLight
{
    vec3 Color;
    float AmbientIntensity;
    float DiffuseIntensity;
};

struct DirectionalLight
{
    BaseLight Base;
    vec3 Direction;
};

struct Attenuation
{
    float Constant;
    float Linear;
    float Exp;
};

struct PointLight
{
    BaseLight Base;
    vec3 LocalPos;
    Attenuation Atten;
};

struct SpotLight
{
    PointLight Base;
    vec3 Direction;
    float Cutoff;
};

struct Material
{
    vec3 AmbientColor;
    vec3 DiffuseColor;
    vec3 SpecularColor;
};

uniform DirectionalLight gDirectionalLight;
uniform int gNumPointLights = 0;
uniform PointLight gPointLights[MAX_POINT_LIGHTS];
uniform int gNumSpotLights = 0;
uniform SpotLight gSpotLights[MAX_SPOT_LIGHTS];
uniform Material gMaterial;
uniform sampler2D gSamplerSpecularExponent;
uniform sampler2D normalMap;
uniform sampler2D armMap;
// uniform sampler2D dispMap;
uniform vec3 gCameraLocalPos;
uniform sampler2D src;
uniform sampler2D projector;
uniform bool useProjector = false;
uniform bool projectorOnly = true;
uniform bool projectorIsSingleChannel = false;
uniform bool flipTexVertically = false;
uniform bool flipTexHorizontally = false;
uniform bool useGGX = false;
uniform bool renderUV = false;
uniform bool useMetric = false;
uniform float ambientCoeff = 0.0;
uniform sampler2D shadowMap;
uniform bool useShadow = false;
uniform float shadowBias = 0.005;
uniform bool useNormalMap = false;
uniform bool useArmMap = false;
// uniform bool useDispMap = false;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    float shadow = currentDepth - shadowBias > closestDepth  ? 1.0 : 0.0;
    // float shadow = currentDepth > closestDepth  ? 1.0 : 0.0;
    return shadow;
}  

vec3 CalcBumpedNormal()
{
    vec3 Normal = normalize(Normal0);
    vec3 Tangent = normalize(Tangent0);
    Tangent = normalize(Tangent - dot(Tangent, Normal) * Normal);
    vec3 Bitangent = cross(Tangent, Normal);
    vec3 BumpMapNormal = texture(normalMap, TexCoord0).xyz;
    BumpMapNormal = 2.0 * BumpMapNormal - vec3(1.0, 1.0, 1.0);
    vec3 NewNormal;
    mat3 TBN = mat3(Tangent, Bitangent, Normal);
    NewNormal = TBN * BumpMapNormal;
    NewNormal = normalize(NewNormal);
    return NewNormal;
}

vec4 CalcLightInternal(BaseLight Light, vec3 LightDirection, vec3 Normal, vec4 projectiveColor)
{
    float armAmbient = 1.0f;
    float armRoughness = 1.0f;
    float armMetalic = 1.0f;
    if (useArmMap)
    {
        vec3 armColor = texture(armMap, TexCoord0).rgb;
        armAmbient = armColor.r;
        armRoughness = armColor.g;
        armMetalic = armColor.b;
    }
    vec4 AmbientColor = vec4(Light.Color, 1.0f) *
                        Light.AmbientIntensity *
                        vec4(gMaterial.AmbientColor, 1.0f) *
                        armAmbient;

    float DiffuseFactor = dot(Normal, -LightDirection);

    vec4 DiffuseColor = vec4(0, 0, 0, 0);
    vec4 SpecularColor = vec4(0, 0, 0, 0);

    if (DiffuseFactor > 0) {
        DiffuseColor = vec4(Light.Color, 1.0f) *
                       Light.DiffuseIntensity *
                       vec4(gMaterial.DiffuseColor, 1.0f) *
                       DiffuseFactor * projectiveColor;

        vec3 PixelToCamera = normalize(gCameraLocalPos - LocalPos0);
        vec3 LightReflect = normalize(reflect(LightDirection, Normal));
        float SpecularFactor = dot(PixelToCamera, LightReflect);
        if (SpecularFactor > 0) {
            // float SpecularExponent = texture2D(gSamplerSpecularExponent, TexCoord0).r * 255.0;
            float SpecularExponent = 1.0;
            SpecularFactor = pow(SpecularFactor, SpecularExponent);
            SpecularColor = vec4(Light.Color, 1.0f) *
                            Light.DiffuseIntensity * // using the diffuse intensity for diffuse/specular
                            vec4(gMaterial.SpecularColor, 1.0f) *
                            SpecularFactor * 
                            (1-armRoughness);
        }
    }

    return (AmbientColor + DiffuseColor + SpecularColor);
}


vec4 CalcDirectionalLight(vec3 Normal, vec4 projectiveColor)
{
    return CalcLightInternal(gDirectionalLight.Base, gDirectionalLight.Direction, Normal, projectiveColor);
}

vec4 CalcPointLight(PointLight l, vec3 Normal)
{
    vec3 LightDirection = LocalPos0 - l.LocalPos;
    float Distance = length(LightDirection);
    LightDirection = normalize(LightDirection);

    vec4 Color = CalcLightInternal(l.Base, LightDirection, Normal, vec4(1.0, 1.0, 1.0, 1.0));
    float Attenuation =  l.Atten.Constant +
                         l.Atten.Linear * Distance +
                         l.Atten.Exp * Distance * Distance;

    return Color / Attenuation;
}

vec4 CalcSpotLight(SpotLight l, vec3 Normal)
{
    vec3 LightToPixel = normalize(LocalPos0 - l.Base.LocalPos);
    float SpotFactor = dot(LightToPixel, l.Direction);

    if (SpotFactor > l.Cutoff) {
        vec4 Color = CalcPointLight(l.Base, Normal);
        float SpotLightIntensity = (1.0 - (1.0 - SpotFactor)/(1.0 - l.Cutoff));
        return Color * SpotLightIntensity;
    }
    else {
        return vec4(0,0,0,0);
    }
}

void main()
{
    if (useMetric)
    {
        FragColor = vec4(metricColor, 1.0);
    }
    else
    {
        if (useGGX)
        {
            vec4 projColor = vec4(1.0, 1.0, 1.0, 1.0);
            if (useProjector)
            {
                float u = (ProjTexCoord.x / ProjTexCoord.z + 1.0) * 0.5;
                float v = (ProjTexCoord.y / ProjTexCoord.z + 1.0) * 0.5;
                if (flipTexVertically)
                {
                    v = 1.0 - v;
                }
                if (flipTexHorizontally)
                {
                    u = 1.0 - u;
                }
                
                if (projectorIsSingleChannel)
                {
                    projColor = texture(projector, vec2(u, v)).rrrr;
                }
                else
                {
                    projColor = texture(projector, vec2(u, v)).rgba;
                }
            }
            vec3 Normal;
            if (useNormalMap)
                Normal = CalcBumpedNormal();
            else
                Normal = normalize(Normal0);
            vec4 TotalLight = CalcDirectionalLight(Normal, projColor);

            for (int i = 0 ;i < gNumPointLights ;i++) {
                TotalLight += CalcPointLight(gPointLights[i], Normal);
            }

            for (int i = 0 ;i < gNumSpotLights ;i++) {
                TotalLight += CalcSpotLight(gSpotLights[i], Normal);
            }
            FragColor = texture2D(src, TexCoord0.xy) * TotalLight;
        }
        else
        {
            if (useProjector)
            {
                float u = (ProjTexCoord.x / ProjTexCoord.z + 1.0) * 0.5;
                float v = (ProjTexCoord.y / ProjTexCoord.z + 1.0) * 0.5;
                if (flipTexVertically)
                {
                    v = 1.0 - v;
                }
                if (flipTexHorizontally)
                {
                    u = 1.0 - u;
                }
                vec3 projColor;
                if (projectorIsSingleChannel)
                {
                    projColor = texture(projector, vec2(u, v)).rrr;
                }
                else
                {
                    projColor = texture(projector, vec2(u, v)).rgb;
                }
                if (projectorOnly)
                {
                    FragColor = vec4(projColor, 1.0);
                }
                else
                {
                    vec3 diffuse_color = texture(src, TexCoord0).rgb;
                    vec3 ambient_color = diffuse_color * ambientCoeff;
                    FragColor = vec4(ambient_color + (diffuse_color * projColor), 1.0);
                }
            }
            else
            {
                if (renderUV)
                {
                    FragColor = vec4(TexCoord0.xy, 0.0, 1.0);
                }
                else
                {
                    vec4 diffuse_color = texture(src, TexCoord0);
                    if (useShadow)
                    {
                        float shadow = ShadowCalculation(LightPos0);
                        FragColor = vec4(vec3(diffuse_color * (1.0 - shadow)), diffuse_color.a);
                    }
                    else
                    {
                        vec4 diffuse_color = texture(src, TexCoord0);
                        FragColor = diffuse_color;
                    }
                    
                    
                    
                }
            }
        }
    }
}
