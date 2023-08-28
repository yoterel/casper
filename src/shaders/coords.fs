#version 330 core
out vec4 FragColor;
uniform vec2 resolution;

vec4 empty = vec4(0.,0.,0.,1.);

void main(){
    FragColor = vec4(gl_FragCoord.x, gl_FragCoord.y, 0., 1.);
}