#version 330 core

in vec2 TexCoord0;
out vec4 FragColor;

uniform sampler2D src;
uniform vec2 resolution;

void main()
{
    vec3 col;
    col = texture(src, TexCoord0).rgb;
    // FragColor = texture(src, gl_FragCoord.xy / resolution.xy);
    FragColor = vec4(col, 1.0);
    // FragColor = vec4(TexCoord0, 0.0, 1.0);
}
