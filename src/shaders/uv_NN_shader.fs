#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D uv;
uniform sampler2D jfa;
uniform sampler2D mask;
uniform sampler2D unwrapped;
uniform vec2 resolution;
uniform bool flipMaskVer = true;
uniform bool flipMaskHor = true;
uniform bool maskIsGray = false;
uniform float threshold = 0.01;
uniform float distThreshold = 50.0;
uniform float seamThreshold = 0.1;
uniform vec3 bgColor;

void main()
{
    vec2 loc = texture(jfa, TexCoord).rg; // get location of nearest seed, result is in pixel units
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

    if (avgMask >= threshold) { // select only pixels that are not black from the jump flood image (foreground)
        float dist = distance(loc.xy, gl_FragCoord.xy); // compute distance to nearest seed in pixel space
        if (dist >= distThreshold)
        {
            // discard;
            FragColor = vec4(bgColor, 1.0);
        }
        else
        {
            // step1: calculate location of reflection about nearest seed
            vec2 reflection = 2*loc.xy - gl_FragCoord.xy; // computation of reflection point in pixel space
            // step2: calculate new uv based on reflection
            vec2 reflection_uv = texture(uv, reflection / resolution).xy; // get uv of reflection point (0:1)
            vec2 seed_uv = texture(uv, loc.xy / resolution).xy; // get uv of nearest seed point (0:1)
            vec2 new_uv;
            if (distance(reflection_uv, seed_uv) > seamThreshold) // if reflection point is too far from seed, use seed uv, we dont want jumps across seams
            {
                new_uv = seed_uv;
                // FragColor = vec4(0.0, 1.0, 1.0, 1.0);   
            }
            else
            {
                new_uv = 2*seed_uv - reflection_uv; // compute new uv based on reflection
                // FragColor = vec4(1-distance(reflection_uv, seed_uv), 0.0, 0.0, 1.0);
            }
            // step3: sample unwrapped texture using the new uv
            FragColor = texture(unwrapped, new_uv); // sample unwrapped texture using new uv
        }
    } else {  // background
        FragColor = vec4(bgColor, 1.0);
    }
}