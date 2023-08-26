#version 330 core
out vec4 FragColor;

//in vec3 ourColor;
in vec2 TexCoord;

// texture samplers
uniform sampler2D src;
uniform bool binary;
// uniform float threshold;

void main()
{
	vec3 col = texture(src, TexCoord).rgb;
	// float avg = (col.r + col.g + col.b) * 0.333333;
	// float bin = mix(0.0, 1.0, step(threshold, avg));
	// FragColor = vec4(bin, bin, bin, 1.0);
	FragColor = vec4(col, 1.0);
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