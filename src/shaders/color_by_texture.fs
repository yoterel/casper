#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D src;
uniform vec3 bgColor;
uniform bool binary = false;
uniform float threshold = 0.01;
uniform bool isGray = false;
uniform bool justGreen = false;
uniform bool onlyGreen = false;
uniform bool gammaCorrection = false;
void main()
{
    vec4 col;
    if (isGray)
        col = vec4(texture(src, TexCoord).rrr, 1.0);
    else
        col = texture(src, TexCoord);
    if (justGreen)
        col = vec4(0.0, col.r, 0.0, 1.0);
    if (onlyGreen)
        col = vec4(0.0, 1.0, 0.0, 1.0);
    if (col.a != 1.0)
        col = vec4(bgColor, 1.0);
    if (binary)
    {
        float avg = (col.r + col.g + col.b) * 0.333333;
        if (avg >= threshold)
		    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	    else
		    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
        if (gammaCorrection)
        {
            float gamma = 2.2;
            col = pow(col, vec4(gamma, gamma, gamma, 1.0));
        }
	    FragColor = col;
    }
}