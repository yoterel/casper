#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;
uniform bool flipVer = false;
uniform bool flipHor = false;

void main()
{
	float u, v;
	gl_Position = vec4(aPos, 1.0);
	if (flipHor)
		u = 1.0 - aTexCoord.x;
	else
		u = aTexCoord.x;
	if (flipVer)
		v = 1.0 - aTexCoord.y;
	else
		v = aTexCoord.y;
	TexCoord = vec2(u, v);
}