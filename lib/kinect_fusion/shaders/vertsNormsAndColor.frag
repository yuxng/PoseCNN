#version 330

in vec3 fragPosition;
in vec3 fragNormal;
in vec3 fragColor;

out vec3 [3] fragOutput;

void main()
{

    fragOutput[0] = fragPosition;
    fragOutput[1] = fragNormal;
    fragOutput[2] = fragColor;

}
