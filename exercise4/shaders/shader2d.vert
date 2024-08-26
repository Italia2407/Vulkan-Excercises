#version 450

layout(location = 0) in vec2 iPosition;
layout(location = 1) in vec3 iColour;

layout(location = 0) out vec3 v2fColour;

void main()
{
    gl_Position = vec4(iPosition, 0.5f, 1.0f);
    v2fColour = iColour;
}
