#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
// uniform sampler2D jfa;
uniform sampler2D src;
// uniform vec2 resolution;

void main()
{
	// vec3 col = texture(jfa, TexCoord).rgb;
	// FragColor = vec4(col, 1.0);
    // FragColor = texture(src, col.xy / resolution);
	// FragColor = vec4(col.x / resolution.x, col.y / resolution.y, col.z, 1.0);
	vec3 col = texture(src, TexCoord).rgb;
	float avg = (col.r + col.g + col.b) * 0.333333;
	if (avg > 0.0)
		FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	else
		FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	// FragColor = texture(src, TexCoord);
	// FragColor = vec4(0.0, 1.0, 0.0, 1.0);
}