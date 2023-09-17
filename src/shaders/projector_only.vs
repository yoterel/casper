#version 330 core
layout (location = 0) in vec3 aPos;
//layout (location = 1) in vec3 aVertColor;
layout (location = 2) in vec2 aTexCoord;

out vec3 ProjTexCoord;
//out vec2 TexCoord;
uniform mat4 camTransform; // camera w2c
uniform mat4 projTransform; // projector w2c

void main()
{
	vec4 pos = camTransform * vec4(aPos, 1.0);
    vec4 proj_pos = projTransform * vec4(aPos, 1.0);
	gl_Position = pos;
	ProjTexCoord = vec3(proj_pos.x, proj_pos.y, proj_pos.z);
	//TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}