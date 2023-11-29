#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D toBake;
uniform sampler2D uvTexture;

void main()
{
    vec2 uv_coords = texture(uvTexture, TexCoord).xy;
    FragColor = texture(toBake, uv_coords.yx);
}