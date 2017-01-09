#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec3 vertexPosition;

void main()
{

    gl_Position = projectionMatrix*modelViewMatrix*vec4(vertexPosition,1.0);

}
