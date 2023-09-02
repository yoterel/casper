#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 ProjTexCoord;
uniform mat4 camTransform; // camera w2c
uniform mat4 projTransform; // projector w2c
uniform bool flipVer;  // should projector output flip input?

void main()
{
	vec4 pos = camTransform * vec4(aPos, 1.0);
    vec4 proj_pos = projTransform * vec4(aPos, 1.0);
	gl_Position = pos;
	if (flipVer)
		ProjTexCoord = vec2(proj_pos.x / proj_pos.w, 1 - (proj_pos.y / proj_pos.w));
	else
		ProjTexCoord = vec2(proj_pos.x / proj_pos.w, proj_pos.y / proj_pos.w);  // proj_pos.w
    
}