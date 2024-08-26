#version 450

layout(location = 0) in vec2 v2fTextureCoord;

layout(set = 1, binding = 0) uniform sampler2D uTextureColour;

layout(location = 0) out vec4 oColour;

void main()
{
    oColour = vec4(texture(uTextureColour, v2fTextureCoord).rgb, 1.0f);
}
