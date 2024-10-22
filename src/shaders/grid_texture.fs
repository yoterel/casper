#version 430 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;
uniform sampler2D src;
uniform float threshold = 1.0;

void main()
{
    vec4 color = texture(src, TexCoord);
    // solve black fringe artifact due to alpha interpolation
    // note: hack. the best solution is alpha premultiplication
    if(color.a < threshold)
        discard;
    FragColor = color;
}