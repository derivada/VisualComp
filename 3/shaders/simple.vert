#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normal;

layout(location = 3) uniform mat4 mvp_matrix;
layout(location = 4) uniform mat4 normal_matrix;

out vec4 fragColor;
out vec3 fragNormal;

void main()
{
    gl_Position = mvp_matrix * vec4(position, 1.0f);
    fragColor = color;
    fragNormal = normalize(mat3(normal_matrix) * normal);
}