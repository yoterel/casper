#version 330 core
out vec4 FragColor;
uniform sampler2D src;
uniform vec2 resolution;

vec4 empty = vec4(0.,0.,0.,1.);

void main(){
    // vec2 fragCoords = gl_FragCoord.xy;
    vec2 uv = gl_FragCoord.xy / resolution;  // (0:W-1, 0:H-1) --> (0:1, 0:1)
    vec4 col = texture(src, uv);  // color at uv
    // float avg = (col.r + col.g + col.b) * 0.333333;
    vec4 outcol;
    //CHECK IF SEED
    if(col.w > 0){
        //store coordinates if not empty
        outcol = vec4(gl_FragCoord.x, gl_FragCoord.y, 0., 1.);
        // outcol = vec4(0.0, 0.0, 1.0, 1.);
    } else {
        outcol = empty;
    }
    //outcol = vec4(0.0, 0.0, 1.0, 1.);
    FragColor = outcol;
}