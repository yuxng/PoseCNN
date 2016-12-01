#version 330

in vec3 fragPosition;
in vec3 fragNormal;

out vec3 [2] fragOutput;

void main()
{

    fragOutput[0] = fragPosition;
    fragOutput[1] = fragNormal;

}
