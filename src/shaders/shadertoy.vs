#version 330 core
layout (location = 0) in vec3 position;            
layout (location = 1) in vec2 inTexCoord;

out vec2 texCoord;
void main()
{
    texCoord = inTexCoord;
    gl_Position = vec4(position.x, position.y, 0.0f, 1.0f);
}