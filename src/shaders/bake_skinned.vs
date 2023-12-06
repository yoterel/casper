#version 330 core

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 VertColor;
layout (location = 2) in vec2 TexCoord;
layout (location = 3) in vec3 Normal;
layout (location = 4) in ivec4 BoneIDs0;
layout (location = 5) in ivec2 BoneIDs1;
layout (location = 6) in vec4 Weights0;
layout (location = 7) in vec2 Weights1;

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
