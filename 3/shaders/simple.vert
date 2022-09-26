#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec4 normal;
layout(location = 3) uniform mat4 transf;
out vec4 fragColor;
out vec4 fragNormal;

void main()
{
    gl_Position = transf * vec4(position, 1.0f);
    fragColor = color;
    fragNormal = normal;
}