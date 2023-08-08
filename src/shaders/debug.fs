#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D jfa;
uniform sampler2D src;
uniform vec2 resolution;

void main()
{
	vec3 col = texture(jfa, TexCoord).rgb;
	// FragColor = vec4(col, 1.0);
    // FragColor = texture(src, col.xy / resolution);
	FragColor = vec4(col.x / resolution.x, col.y / resolution.y, col.z, 1.0);
	// FragColor = texture(src, TexCoord);
}