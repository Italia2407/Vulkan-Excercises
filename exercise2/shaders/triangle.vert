#version 450

layout(location = 0) out vec3 vertexColour;

const vec2 kVertexPositions[6] = vec2[6](
    vec2( 0.0f, -0.8f),
    vec2(-0.7f,  0.8f),
    vec2( 0.7f,  0.8f),
    vec2( 0.1f, -0.9f),
    vec2( 0.5f, -0.1f),
    vec2( 0.9f,  0.0f)
);
const vec3 kVertexColours[6] = vec3[6](
    vec3(1.0f, 0.0f, 0.0f),
    vec3(0.0f, 1.0f, 0.0f),
    vec3(0.0f, 0.0f, 1.0f),
    vec3(0.0f, 1.0f, 1.0f),
    vec3(1.0f, 0.0f, 1.0f),
    vec3(1.0f, 1.0f, 1.0f)
);


void main()
{
    const vec2 xy = kVertexPositions[gl_VertexIndex];
    gl_Position = vec4(xy, 0.5f, 1.0f);

    vertexColour = kVertexColours[gl_VertexIndex];
}
