#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec4 normal;

uniform mat4 view_proj;
uniform mat4 transf;

out vec4 fragColor;
out vec4 fragNormal;

void main()
{
    gl_Position = view_proj * vec4(position, 1.0f);
    fragColor = color;
    fragNormal = normal;
}