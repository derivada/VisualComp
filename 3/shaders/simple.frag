#version 430 core

in vec4 fragColor;
in vec3 fragNormal;
out vec4 outColor;

void main()
{
    // outColor = vec4(fragNormal, 1.0f); // Normal vector color
	// Basic lightning 
    vec3 lightDirection = normalize(vec3(0.8, -0.5, 0.6));
    outColor.x = fragColor.x * max(0, dot(fragNormal, -lightDirection));
    outColor.y = fragColor.y * max(0, dot(fragNormal, -lightDirection));
    outColor.z = fragColor.z * max(0, dot(fragNormal, -lightDirection));
    outColor.w = fragColor.w;
}