#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D src;
uniform vec2 resolution;

void main()
{
    vec2 offsets[8] = vec2[8](vec2(-1,0), vec2(1,0), vec2(0,1), vec2(0,-1), vec2(-1,1), vec2(1,1), vec2(1,-1), vec2(-1,-1));
    vec2 texelsize =  1 / resolution;
    float mindist = 10000000;
    vec2 UV = vec2(gl_FragCoord.xy) * texelsize;
    vec4 sample = texture(src, UV).rgba;
    vec4 curminsample = sample;
    if (sample.w == 0)
    {
        int MaxSteps = 100;
        int i = 0;
        while(i < MaxSteps)
        { 
            i++;
            int j = 0;
            while (j < 8)
            {
                vec2 curUV = UV + offsets[j] * texelsize * i;
                vec4 offsetsample = texture(src, curUV).rgba;

                if(offsetsample.w != 0)
                {
                    float curdist = length(UV - curUV);
                    if (curdist < mindist)
                    {
                        vec2 projectUV = curUV + offsets[j] * texelsize * i * 0.25;
                        vec4 direction = texture(src, projectUV).rgba;
                        mindist = curdist;
                        if(direction.w != 0)
                        {
                            vec4 delta = offsetsample - direction;
                            curminsample = offsetsample + delta * 4;
                        }
                        else
                        {
                            curminsample = offsetsample;
                        }
                    }
                }
                j++;
            }
        }
    }
    FragColor = curminsample;
}