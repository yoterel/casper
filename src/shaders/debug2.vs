#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;
out vec2 ProjTexCoord;
uniform mat4 vcamTransform;
uniform mat4 vprojTransform;
uniform bool flipVer;

void main()
{
	vec4 pos = vcamTransform * vec4(aPos, 1.0);
	// vec4 pos = vec4(aPos, 1.0);
	gl_Position = pos;
    vec4 proj_pos = vprojTransform * vec4(aPos, 1.0);
    ProjTexCoord = (vec2(proj_pos.x / proj_pos.w, proj_pos.y / proj_pos.w) + vec2(1.0, 1.0)) * vec2(0.5, 0.5);
    ProjTexCoord = vec2(ProjTexCoord.x, 1-ProjTexCoord.y);
	if (flipVer)
		TexCoord = vec2(aTexCoord.x, 1-aTexCoord.y);
	else
		TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}