#version 330 core
layout (location = 0) in vec3 aPos;
//layout (location = 1) in vec3 aColor;
layout (location = 1) in vec2 aTexCoord;

//out vec3 ourColor;
out vec2 TexCoord;
//uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;
uniform bool flipVer;

void main()
{
	// gl_Position = projection * view * model * vec4(aPos, 1.0);
	gl_Position = vec4(aPos, 1.0);
	//ourColor = aColor;
	if (flipVer)
		TexCoord = vec2(aTexCoord.x, 1-aTexCoord.y);
	else
		TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}