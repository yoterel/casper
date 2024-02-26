#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;
uniform float threshold = 0.5;

void main()
{
    float value = texture(text, TexCoords).r;
    if(value >= threshold)
    {
        vec4 sampled = vec4(value, value, value, 1.0);
        color = vec4(textColor, 1.0) * sampled;
    }
    else
    {
        color = vec4(0.0, 0.0, 0.0, 0.0);
    }
}