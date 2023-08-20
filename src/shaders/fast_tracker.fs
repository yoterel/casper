#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D jfa;
uniform sampler2D src;
uniform sampler2D cam_image;
uniform vec2 resolution;

void main()
{
    vec3 loc = texture(jfa, TexCoord).rgb; 
    // FragColor = vec4(loc.xy , 0.0, 1.0);
    // FragColor = vec4(loc.xy / resolution , 0.0, 1.0);
    vec2 FlippedTexCoord = vec2(TexCoord.x, 1-TexCoord.y);
    vec3 col = texture(cam_image, FlippedTexCoord).rgb;
    float avg = (col.r + col.g + col.b) * 0.333333;
    // FragColor = texture(src, loc.xy / resolution);  // full frame jump flood
    if (avg > 0.0) { // only pixels that are not black jumpy flood
        FragColor = texture(src, loc.xy / resolution);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}