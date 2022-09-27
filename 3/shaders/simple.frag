#version 430 core

in vec4 fragColor;
in vec4 fragNormal;
out vec4 outColor;

void main()
{
    // outColor = fragNormal;
    // outColor = fragColor;
    vec4 lightDirection = normalize(vec4(0.8, -0.5, 0.6, 0.0));
    outColor.x = fragColor.x * max(0, dot(fragNormal, -lightDirection));
    outColor.y = fragColor.y * max(0, dot(fragNormal, -lightDirection));
    outColor.z = fragColor.z * max(0, dot(fragNormal, -lightDirection));
    outColor.w = fragColor.w;
}