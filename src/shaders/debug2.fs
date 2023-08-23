#version 330
out vec4 FragColor;

in vec2 TexCoord;
in vec2 ProjTexCoord;

uniform sampler2D src;
uniform bool binary;

void main()
{
    vec3 col = texture(src, ProjTexCoord).rgb;
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
        FragColor = vec4(col, 1.0);
    }
    
}
