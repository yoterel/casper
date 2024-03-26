#version 330 core

out vec4 FragColor;
in vec3 ourColor;

uniform bool allWhite = false;
void main()
{
	if (allWhite)
		FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	else
		FragColor = vec4(ourColor, 1.0);
}