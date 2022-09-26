#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) uniform mat4 transf;
out vec4 fragColor;

void main()
{
    gl_Position = transf * vec4(position, 1.0f);
    fragColor = color;
}