#version 330

// const int MAX_POINT_LIGHTS = 2;
// const int MAX_SPOT_LIGHTS = 2;

in vec2 TexCoord0;
in vec2 ProjTexCoord0;
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

uniform sampler2D projTexture;
uniform bool binary;
// uniform int gDisplayBoneIndex;

void main()
{
    float u = (ProjTexCoord0.x + 1.0) * 0.5;
    float v = (ProjTexCoord0.y + 1.0) * 0.5;
    vec4 proj_col = vec4(texture(projTexture, vec2(u, 1-v)).rgb, 1.0);
    if (binary)
    {
        float avg = (proj_col.r + proj_col.g + proj_col.b) * 0.333333;
        if (avg > 0.0) {
            proj_col = vec4(1.0, 1.0, 1.0, 1.0);
        } else {
            proj_col = vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
    // finalColor = vec4(ourColor, 0.9); // boneweight debug
    vec4 diffuse_color = texture(gSampler, TexCoord0);  // diffuse texture
    // finalColor.w = 1.0;
    FragColor = diffuse_color * proj_col;  // * diffuse_color * vec4(1.0, 1.0, 1.0, proj_col.w);
}
