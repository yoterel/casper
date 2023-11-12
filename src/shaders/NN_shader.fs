#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D jfa;
uniform sampler2D src;
uniform sampler2D mask;
uniform vec2 resolution;
uniform bool flipMaskVer;

void main()
{
    vec3 loc = texture(jfa, TexCoord).rgb; // get location of nearest seed, result is in pixel units
    vec4 maskCol;
    if (flipMaskVer)
        maskCol = texture(mask, vec2(TexCoord.x, 1.0 - TexCoord.y));
    else
        maskCol = texture(mask, TexCoord);
    float avgMask = (maskCol.r + maskCol.g + maskCol.b) * 0.333333;
    // FragColor = texture(src, loc.xy / resolution);  // full frame jump flood
    if (avgMask > 0.0) { // select only pixels that are not black from the jump flood image
        FragColor = texture(src, loc.xy / resolution); // sample texture using the locations of the nearest seeds
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}