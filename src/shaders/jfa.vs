#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

//out vec2 TexCoord;
//uniform bool flipVer;

void main()
{
	gl_Position = vec4(aPos, 1.0);
	//if (flipVer)
	//	TexCoord = vec2(aTexCoord.x, 1-aTexCoord.y);
	//else
	//	TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}