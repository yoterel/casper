#version 330
out vec4 FragColor;

in vec3 VertColor;
in vec2 ProjTexCoord;
//in vec2 TexCoord;
uniform sampler2D src;
uniform sampler2D projTexture;
uniform bool binary;

void main()
{
    // vec3 vertColor = VertColor;
    //vec3 texColor = texture(src, TexCoord).rgb;
    float u = (ProjTexCoord.x + 1) * 0.5;
    float v = (ProjTexCoord.y + 1) * 0.5;
    vec3 projColor = texture(projTexture, vec2(u, 1-v)).rgb;
    if (binary)
    {
        float avg = (projColor.r + projColor.g + projColor.b) * 0.333333;
        if (avg > 0.0)
		    projColor = vec3(1.0, 1.0, 1.0);
	    else
		    projColor = vec3(0.0, 0.0, 0.0);
    }
    FragColor = vec4(projColor, 1.0) * vec4(VertColor, 1.0);
}
