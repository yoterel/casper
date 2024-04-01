#version 330

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 VertColor;
layout (location = 2) in vec2 TexCoord;
layout (location = 3) in vec3 Normal;
layout (location = 4) in ivec4 BoneIDs0;
layout (location = 5) in ivec2 BoneIDs1;
layout (location = 6) in vec4 Weights0;
layout (location = 7) in vec2 Weights1;
layout (location = 8) in vec3 Tangent;

out vec2 TexCoord0;
out vec3 ProjTexCoord;
out vec3 Normal0;
out vec3 LocalPos0;
out vec4 LightPos0;
out vec3 Tangent0;
out vec3 metricColor;

const int MAX_BONES = 50;

uniform mat4 mvp;
uniform mat4 projTransform;
uniform mat4 lightTransform;
uniform mat4 gBones[MAX_BONES];
uniform float gBoneMetric[MAX_BONES];
uniform bool bake = false;
uniform bool useMetric = false;

void main()
{
    // compute the world position of the vertex, using all bones that effect it (using their respected weights)
    mat4 BoneTransform = gBones[BoneIDs0[0]] * Weights0[0];
    BoneTransform     += gBones[BoneIDs0[1]] * Weights0[1];
    BoneTransform     += gBones[BoneIDs0[2]] * Weights0[2];
    BoneTransform     += gBones[BoneIDs0[3]] * Weights0[3];
    BoneTransform     += gBones[BoneIDs1[0]] * Weights1[0];
    BoneTransform     += gBones[BoneIDs1[1]] * Weights1[1];

    // compute the world position of the vertex (pos = bone2world * vertices (which are already in bone space))
    vec4 pos = BoneTransform * vec4(Position, 1.0);

    // project to clip space, unless we are baking (and then we simply use the texture coordinates as the position)
    if (bake)
    {
        gl_Position = vec4((TexCoord.x*2) - 1, (TexCoord.y*2) - 1, 0.0, 1.0);
    }
    else
    {
        gl_Position = mvp * pos;  // project to screen space
    }

    // compute clip space of vertices in light space for projective lighting
    vec4 proj_pos = projTransform * pos;
    ProjTexCoord = vec3(proj_pos.x, proj_pos.y, proj_pos.z);

    // compute clip space of vertices in light space for shadow mapping
    vec4 light_pos = lightTransform * pos;  // project to light space

    // transform normals & tangents to world space (don't use translation)
    Normal0 = vec4(BoneTransform * vec4(Normal, 0.0)).xyz; 
    Tangent0 = vec4(BoneTransform * vec4(Tangent, 0.0)).xyz;

    // transfer all other quantities to fragment shader as is for interpolation
    LocalPos0 = vec3(pos);  // This is world position, not local...todo change name
    TexCoord0 = TexCoord;
    LightPos0 = light_pos;

    // this is by far the ugliest piece of code in this project to find which bone is most influencing the vertex
    if (useMetric)  
    {
        int selectedBoneID = BoneIDs0[0];
        int maxIndex = 0;
        if (Weights0[1] > Weights0[maxIndex])
        {
            selectedBoneID = BoneIDs0[1];
            maxIndex = 1;
        }
        if (Weights0[2] > Weights0[maxIndex])
        {
            selectedBoneID = BoneIDs0[2];
            maxIndex = 2;
        }
        if (Weights0[3] > Weights0[maxIndex])
        {
            selectedBoneID = BoneIDs0[3];
            maxIndex = 3;
        }
        if (Weights1[0] > Weights0[maxIndex])
        {
            selectedBoneID = BoneIDs1[0];
            maxIndex = 4;
        }
        if (maxIndex == 4)
        {
            if (Weights1[1] > Weights1[0])
            {
                selectedBoneID = BoneIDs1[1];
            }
        }
        else
        {
            if (Weights1[1] > Weights0[maxIndex])
            {
                selectedBoneID = BoneIDs1[1];
            }
        }
        metricColor = vec3(1.0 - gBoneMetric[selectedBoneID], gBoneMetric[selectedBoneID], 0.0);        
    }
}
