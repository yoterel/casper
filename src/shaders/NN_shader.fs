#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D jfa;
uniform sampler2D src;
uniform sampler2D mask;
uniform vec2 resolution;
uniform bool flipMaskVer;
uniform bool flipMaskHor = true;

void main()
{
    vec3 loc = texture(jfa, TexCoord).rgb; // get location of nearest seed, result is in pixel units
    vec2 mask_uv = TexCoord;
    if (flipMaskHor)
        mask_uv.x = 1.0 - mask_uv.x;
    if (flipMaskVer)
        mask_uv.y = 1.0 - mask_uv.y;
    vec4 maskCol = texture(mask, mask_uv);
    float avgMask = (maskCol.r + maskCol.g + maskCol.b) * 0.333333;
    // FragColor = texture(src, loc.xy / resolution);  // full frame jump flood
    if (avgMask > 0.01) { // select only pixels that are not black from the jump flood image
        FragColor = texture(src, loc.xy / resolution); // sample texture using the locations of the nearest seeds
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}