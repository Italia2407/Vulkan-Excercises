#version 450

layout(location = 0) in vec2 v2fTextureCoord;

layout(set = 1, binding = 0) uniform sampler2D uTextureColour;

layout(location = 0) out vec4 oColour;

void main()
{
    oColour = texture(uTextureColour, v2fTextureCoord).rgba;
}