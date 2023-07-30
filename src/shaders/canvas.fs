#version 330 core
out vec4 FragColor;

//in vec3 ourColor;
in vec2 TexCoord;

// texture samplers
uniform sampler2D camera_texture;
uniform float threshold;

void main()
{
	vec3 col = texture(camera_texture, TexCoord).rgb;
	float avg = (col.r + col.g + col.b) * 0.333333;
	float bin = mix(0.0, 1.0, step(threshold, avg));
	FragColor = vec4(0.0, bin, 0.0, 1.0);
	// FragColor = vec4(col, 1.0);
}