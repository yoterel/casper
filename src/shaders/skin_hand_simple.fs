#version 330

in vec2 TexCoord0;
in vec3 ProjTexCoord;
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
uniform Material material;
uniform sampler2D src;
uniform bool flipVer = false;
uniform bool useProjector = false;

void main()
{
    if (useProjector)
    {
        float u = (ProjTexCoord.x / ProjTexCoord.z + 1.0) * 0.5;
        float v = (ProjTexCoord.y / ProjTexCoord.z + 1.0) * 0.5;
        if (flipVer)
        {
            v = 1.0 - v;
        }
        vec3 projColor = texture(src, vec2(u, v)).rgb;
        FragColor = vec4(projColor, 1.0);
    }
    else
    {
        vec4 diffuse_color = texture(src, TexCoord0);
        FragColor = diffuse_color;
    }
}
