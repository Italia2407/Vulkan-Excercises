#version 450

layout(location = 0) in vec3 v2fColour;

layout(location = 0) out vec4 oColour;

void main()
{
    oColour = vec4(v2fColour, 1.0f);
}
