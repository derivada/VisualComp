#version 430 core

in vec3 position;

void main()
{
    gl_Position = vec4(position, 1.0f);
    // Mirroring effect
    /*
    vec3 outputPosition;
    outputPosition.x = -position.x;
    outputPosition.y = -position.y;
    outputPosition.z = position.z;
    gl_Position = vec4(outputPosition, 1.0f);
    */
}