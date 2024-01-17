#version 330 core
out vec4 FragColor;
uniform sampler2D src;
uniform vec2 resolution;
uniform int pass;
uniform int numPasses;

vec4 empty = vec4(0., 0., 0., 1.);

void main(){
    vec2 uv = vec2(gl_FragCoord.xy); // (0:W-1, 0:H-1)
    int stepsize = int(exp2(numPasses - pass - 1));
    float minDist = 9999.;
    vec4 outCol = empty;
    for(int y = -1; y <= 1; y++){
        for(int x = -1; x <= 1; x++){
            vec2 duv = vec2(x, y) * stepsize;
            vec4 col = texture(src, (uv+duv)/resolution);
            if(col.x != 0.0 && col.y != 0.0){
                float d = length(col.xy-uv);
                if(d < minDist){
                    minDist = d;
                    outCol = vec4(col.xy, d, 1.);
                }
            }
        }
    }
    // FragColor = vec4(1.0, 0.0,0.,1.);
    FragColor = outCol;
}