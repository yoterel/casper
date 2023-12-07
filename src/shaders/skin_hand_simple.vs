#version 330

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 VertColor;
layout (location = 2) in vec2 TexCoord;
layout (location = 3) in vec3 Normal;
layout (location = 4) in ivec4 BoneIDs0;
layout (location = 5) in ivec2 BoneIDs1;
layout (location = 6) in vec4 Weights0;
layout (location = 7) in vec2 Weights1;

out vec2 TexCoord0;
out vec3 ProjTexCoord;
flat out ivec4 BoneIDs00;
flat out ivec2 BoneIDs11;
out vec4 Weights00;
out vec2 Weights11;
out vec3 ourColor;
const int MAX_BONES = 50;

uniform mat4 gTransform;
uniform mat4 projTransform;
uniform mat4 gBones[MAX_BONES];
uniform int gDisplayBoneIndex;
uniform bool bake = false;
void main()
{
    mat4 BoneTransform = gBones[BoneIDs0[0]] * Weights0[0];
    BoneTransform     += gBones[BoneIDs0[1]] * Weights0[1];
    BoneTransform     += gBones[BoneIDs0[2]] * Weights0[2];
    BoneTransform     += gBones[BoneIDs0[3]] * Weights0[3];
    BoneTransform     += gBones[BoneIDs1[0]] * Weights1[0];
    BoneTransform     += gBones[BoneIDs1[1]] * Weights1[1];
    
    vec4 pos = BoneTransform * vec4(Position, 1.0);
    if (bake)
    {
        gl_Position = vec4((TexCoord.x*2) - 1, (TexCoord.y*2) - 1, 0.0, 1.0);
    }
    else
    {
        gl_Position = gTransform * pos;
    }
    vec4 proj_pos = projTransform * pos;
    ProjTexCoord = vec3(proj_pos.x, proj_pos.y, proj_pos.z);
    TexCoord0 = TexCoord;
    BoneIDs00 = BoneIDs0;
    BoneIDs11 = BoneIDs1;
    Weights00 = Weights0;
    Weights11 = Weights1;
    ourColor = vec3(0.0, 0.0, 0.0);
    for (int i = 0 ; i < 6 ; i++) {
        if (i < 4)
        {
            if (BoneIDs00[i] == gDisplayBoneIndex)
            {
                ourColor = vec3(0.0, 0.0, 1.0) * Weights0[i];
                break;
            }
        }
        else
        {
            if (BoneIDs11[i-4] == gDisplayBoneIndex)
            {
                ourColor = vec3(0.0, 0.0, 1.0) * Weights1[i-4];
                break;
            }
        }
    }
}
