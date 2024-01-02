#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aTexCoord;

out vec3 ourColor;
out vec2 TexCoord;
uniform bool flipVer = true;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor;
    if (flipVer)
        TexCoord = vec2(aTexCoord.x,1.0-aTexCoord.y);
    else
        TexCoord = vec2(aTexCoord.x,aTexCoord.y);
    
}