#version 330

in vec3 fragCanonicalPosition;

out vec3 fragOutput;

void main()
{

    if (isnan(fragCanonicalPosition.x)) {
        discard;
    }

    fragOutput = fragCanonicalPosition;

}
