#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec3 vertexPosition;
layout (location = 1) in vec3 vertexCanonicalPosition;
layout (location = 2) in vec2 vertexTexCoord;
layout (location = 3) in vec3 vertexNormal;

out vec3 fragPosition;
out vec3 fragCanonicalPosition;
out vec2 fragTexCoord;
out vec3 fragNormal;

void main()
{

    fragPosition = vertexPosition;
    fragCanonicalPosition = vertexCanonicalPosition;
    fragTexCoord = vertexTexCoord;
    fragNormal = vertexNormal;

    vec4 vertexPositionCam = modelViewMatrix*vec4(vertexPosition,1.0);

    gl_Position = projectionMatrix*vertexPositionCam;

}
