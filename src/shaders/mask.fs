#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D src;
uniform sampler2D mask;
uniform bool flipMaskVer = true;
uniform bool flipMaskHor = true;
uniform bool maskIsGray = false;
uniform bool fgSingleColor = false;
uniform bool missingColorIsCamera = false;
uniform float threshold = 0.01;
uniform vec3 missingInfoColor = vec3(1.0, 0.0, 0.0);
uniform vec3 unusedInfoColor = vec3(1.0, 0.0, 1.0);
uniform vec3 bgColor = vec3(0.0, 0.0, 0.0);
uniform vec3 fgColor = vec3(0.0, 0.0, 0.0);
uniform float alpha = 1.0;
void main()
{
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
    FragColor = maskCol*texture(src, TexCoord); // sample texture using the locations of the nearest seeds
    if (avgMask >= threshold) // if cam pixel is "on"
    { 
        if (FragColor.w == 0.0) // but render doesn't have info
            if (missingColorIsCamera)
                FragColor = maskCol;  // color from mask
            else
                FragColor = vec4(missingInfoColor, 1.0);  // color indicating missing info
        else
            if (fgSingleColor) // flood fg with color if fgSingleColor is true
                FragColor = vec4(fgColor, 1.0);
    } 
    else // if cam pixel is "off"
    { 
        if (FragColor.w == 1.0)  // but render has info
            FragColor = vec4(unusedInfoColor, 1.0);  // color indicating unused info
        else
            FragColor = vec4(bgColor, 1.0);  // otherwise color black
    }
}