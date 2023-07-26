#version 330

// const int MAX_POINT_LIGHTS = 2;
// const int MAX_SPOT_LIGHTS = 2;

in vec2 TexCoord0;
// in vec3 Normal0;
// in vec3 LocalPos0;
flat in ivec4 BoneIDs00;
flat in ivec2 BoneIDs11;
in vec4 Weights00;
in vec2 Weights11;
in vec3 ourColor;

out vec4 FragColor;
struct Material
{
    vec3 AmbientColor;
    vec3 DiffuseColor;
    vec3 SpecularColor;
};
uniform Material gMaterial;
uniform sampler2D gSampler;
// uniform int gDisplayBoneIndex;

void main()
{
    // FragColor = vec4(ourColor, 0.9); // boneweight debug
    FragColor = texture(gSampler, TexCoord0);  // diffuse texture
}
