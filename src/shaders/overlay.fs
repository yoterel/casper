#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D objectTexture;
uniform sampler2D projectiveTexture;
uniform float mixRatio = 0.7;
uniform float skinBrightness = 1.0;
uniform bool gammaCorrect = false;
void main()
{
    vec4 outputColor;
    vec4 col1 = texture(objectTexture, vec2(1-TexCoord.x, 1-TexCoord.y));
    vec4 col2 = texture(projectiveTexture, TexCoord);
    vec3 skinColor = vec3(col1.r*skinBrightness, col1.g*skinBrightness, col1.b*skinBrightness);
    if ((col1.w != 0.0) && (col2.w != 0.0))
    {
        // outputColor = vec4(mix(skinColor, vec3(col2), mixRatio), 1.0);
        outputColor = vec4(vec3(col1)*vec3(col2), 1.0);
    }
    else
    {
        outputColor = vec4(skinColor, col1.a);
    }
    if (gammaCorrect)
    {
        float gamma = 2.2;
        outputColor = pow(outputColor, vec4(gamma));
    }
    FragColor = outputColor;
}