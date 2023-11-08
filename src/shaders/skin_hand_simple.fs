#version 330

in vec2 TexCoord0;
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

void main()
{
    vec4 diffuse_color = texture(src, TexCoord0);
    FragColor = diffuse_color;
}
