#version 430 core

uniform vec4 uniColor;
out vec4 color;

void main()
{
    color = vec4(1.0, 1.0, 1.0, 1.0);
    // Extra - Slowly change color over time
    // color = uniColor;
}