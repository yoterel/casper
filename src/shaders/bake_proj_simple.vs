#version 330 core

layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoord;

out vec3 ProjTexCoord;
out vec4 pos;
out vec2 TexCoord0;
uniform mat4 camTransform; // camera w2c
uniform mat4 projTransform; // projector w2c

void main()
{
    pos = camTransform * vec4(Position, 1.0);
    gl_Position = vec4((TexCoord.x*2) - 1, (TexCoord.y*2) - 1, 0.0, 1.0);
    vec4 proj_pos = projTransform * vec4(Position, 1.0);
    ProjTexCoord = vec3(proj_pos.x, proj_pos.y, proj_pos.z);
    TexCoord0 = TexCoord;
}
