#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D objectTexture;
uniform sampler2D projectiveTexture;
uniform float mixRatio = 0.7;
void main()
{
    vec4 outputColor;
    vec4 col1 = texture(objectTexture, vec2(1-TexCoord.x, 1-TexCoord.y));
    vec4 col2 = texture(projectiveTexture, TexCoord);
    if ((col1.w != 0.0) && (col2.w != 0.0))
        outputColor = vec4(mix(vec3(col1), vec3(col2), mixRatio), 1.0);
    else
    {
        outputColor = col1;
    }
    FragColor = outputColor;
}