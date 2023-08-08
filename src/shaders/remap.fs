#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D jfa;
uniform sampler2D src;
uniform vec2 resolution;

void main()
{
    vec3 loc = texture(jfa, TexCoord).rgb; 
    // FragColor = vec4(loc.xy , 0.0, 1.0);
    // FragColor = vec4(loc.xy / resolution , 0.0, 1.0);
    FragColor = texture(src, loc.xy / resolution);
}