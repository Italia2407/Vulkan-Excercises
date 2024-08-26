#version 450

layout(location = 0) in vec3 vertexColour;
layout(location = 0) out vec3 colour;


void main()
{
    colour = vertexColour;
}
