#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture samplers
uniform sampler2D texture1;
uniform float threshold;
//uniform sampler2D texture2;

void main()
{
	//FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);
	// vec2 flippedTexCoord = vec2(1.0 - TexCoord.x, 1.0 - TexCoord.y);
	vec3 col = texture(texture1, TexCoord).rgb;
	float avg = (col.r + col.g + col.b) * 0.333333;
	float bin = mix(0.0, 1.0, step(threshold, avg));
	FragColor = vec4(0.0, bin, 0.0, 1.0);
}