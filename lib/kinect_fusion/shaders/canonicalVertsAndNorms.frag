#version 330

in vec3 fragCanonicalPosition;
in vec3 fragCanonicalNormal;

out vec3 [2] fragOutput;

void main()
{

    if (isnan(fragCanonicalPosition.x) || isnan(fragCanonicalNormal.x)) {
        discard;
    }

    fragOutput[0] = fragCanonicalPosition;

    fragOutput[1] = fragCanonicalNormal;

}
