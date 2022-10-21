#version 430 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

out vec4 fragColor;

void main()
{
    // Initialize 4x4 identity matrix
    mat4x4 matrix = mat4(1);
    // Basic alterations on the identity matrix
    //matrix[0][0] = 2;
    //matrix[1][0] = 2;
    //matrix[0][1] = 2;
    //matrix[1][1] = 2;
    //matrix[3][0] = 0.6;
    //matrix[3][1] = 0.6;

    gl_Position = matrix * vec4(position, 1.0f);
    fragColor = color;
}