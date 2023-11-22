#version 330 core
in vec2 TexCoord;
uniform sampler2D src;
uniform sampler2D distortion_map;

void main()
{
    vec2 distortionIndex = texture2D(distortion_map, TexCoord.xy).xy;
    float hIndex = distortionIndex.x;
    float vIndex = distortionIndex.y;

    if(vIndex > 0.0 && vIndex < 1.0 && hIndex > 0.0 && hIndex < 1.0)
    {
        gl_FragColor = vec4(texture2D(src, distortionIndex).xxx, 1.0);
    }
    else
    {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
    // gl_FragColor = vec4(texture2D(src, TexCoord).rrr, 1.0);
}