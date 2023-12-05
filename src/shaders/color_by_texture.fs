#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D src;
uniform bool binary = false;
uniform bool isGray = false;
uniform bool allGreen = false;
void main()
{
    vec4 col;
    if (isGray)
        col = vec4(texture(src, TexCoord).rrr, 1.0);
    else
        col = texture(src, TexCoord);
    if (allGreen)
        col = vec4(0.0, col.r, 0.0, 1.0);
    if (binary)
    {
        float avg = (col.r + col.g + col.b) * 0.333333;
        if (avg > 0.0)
		    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	    else
		    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
	    FragColor = col;
    }
}