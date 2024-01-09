#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D jfa;
uniform sampler2D src;
uniform sampler2D mask;
uniform vec2 resolution;
uniform bool flipMaskVer = true;
uniform bool flipMaskHor = true;
uniform bool maskIsGray = false;
uniform float threshold = 0.01;

void main()
{
    vec3 loc = texture(jfa, TexCoord).rgb; // get location of nearest seed, result is in pixel units
    vec2 mask_uv = TexCoord;
    if (flipMaskHor)
        mask_uv.x = 1.0 - mask_uv.x;
    if (flipMaskVer)
        mask_uv.y = 1.0 - mask_uv.y;
    vec4 maskCol;
    if (maskIsGray)
        maskCol = vec4(texture(mask, mask_uv).rrr, 1.0);
    else
        maskCol = texture(mask, mask_uv);
    float avgMask = (maskCol.r + maskCol.g + maskCol.b) * 0.333333;
    // FragColor = texture(src, loc.xy / resolution);  // full frame jump flood
    if (avgMask >= threshold) { // select only pixels that are not black from the jump flood image
        float dist = distance(loc.xy, gl_FragCoord.xy); // compute distance to nearest seed
        if (dist >= 50.0)
            discard;
        FragColor = texture(src, loc.xy / resolution); // sample texture using the locations of the nearest seeds
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}