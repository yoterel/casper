#version 330 core

in vec3 ProjTexCoord;
in vec4 pos;
in vec2 TexCoord0;
out vec4 FragColor;

uniform sampler2D src;
// uniform vec2 resolution;

void main()
{
    float u = (ProjTexCoord.x / ProjTexCoord.z + 1.0) * 0.5;
    float v = (ProjTexCoord.y / ProjTexCoord.z + 1.0) * 0.5;
    vec3 projColor = texture(src, vec2(u, 1-v)).rgb;
    FragColor = vec4(projColor, 1.0);
    // vec3 col;
    // col = texture(src, TexCoord0).rgb;
    // col = vec3(1.0, 1.0, 1.0);
    // FragColor = texture(src, gl_FragCoord.xy / resolution.xy);
    // FragColor = vec4(col, 1.0);
    // FragColor = vec4(TexCoord0, 0.0, 1.0);
}
