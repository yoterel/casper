#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D src;
uniform sampler2D mask;
uniform bool flipMaskVer = true;
uniform bool flipMaskHor = true;
uniform float threshold = 0.01;

void main()
{
    vec2 mask_uv = TexCoord;
    if (flipMaskHor)
        mask_uv.x = 1.0 - mask_uv.x;
    if (flipMaskVer)
        mask_uv.y = 1.0 - mask_uv.y;
    vec4 maskCol = texture(mask, mask_uv);
    float avgMask = (maskCol.r + maskCol.g + maskCol.b) * 0.333333;
    // FragColor = texture(src, loc.xy / resolution);  // full frame jump flood
    if (avgMask >= threshold) { // select only pixels that are not black from the jump flood image
        FragColor = texture(src, TexCoord); // sample texture using the locations of the nearest seeds
        if (FragColor.w == 0.0) {
            FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}