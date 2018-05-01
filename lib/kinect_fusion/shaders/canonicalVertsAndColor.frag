#version 330

uniform mat4 modelViewMatrix;
uniform sampler2D materialTex;

in vec3 fragCanonicalPosition;
in vec2 fragTexCoord;

layout (location = 0) out vec3 fragOutput0;
layout (location = 1) out vec3 fragOutput1;
layout (location = 2) out vec3 fragOutput2;


void main()
{

    if (isnan(fragCanonicalPosition.x)) {
        discard;
    }

    fragOutput0 = fragCanonicalPosition;

    vec4 surfaceColor = texture(materialTex, fragTexCoord);
    fragOutput1 = vec3(surfaceColor);

    fragOutput2 = vec3(gl_FragCoord.z);

}
