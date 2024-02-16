#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D src;
uniform bool binary = false;
uniform float threshold = 0.5;
uniform bool isGray = false;
void main()
{
    vec4 col;
    if (isGray)
        col = vec4(texture(src, TexCoord).rrr, 1.0);
    else
        col = texture(src, TexCoord);
    if (binary)
    {
        if (col.a >= threshold)
		    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	    else
		    FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
    else
    {
	    FragColor = col;
    }
}