#version 330
out vec4 FragColor;

in vec2 ProjTexCoord;

uniform sampler2D src;
uniform bool binary;

void main()
{
    float u = (ProjTexCoord.x + 1) * 0.5;
    float v = (ProjTexCoord.y + 1) * 0.5;
    vec3 col = texture(src, vec2(u, v)).rgb;
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
