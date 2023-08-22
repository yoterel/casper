#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D src;
uniform bool binary;

void main()
{
    if (binary)
    {
        vec3 col = texture(src, TexCoord).rgb;
        float avg = (col.r + col.g + col.b) * 0.333333;
        if (avg > 0.0)
		    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	    else
		    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
	    FragColor = texture(src, TexCoord);
    }
}