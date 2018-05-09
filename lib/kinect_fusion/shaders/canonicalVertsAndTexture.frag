#version 330

uniform mat4 modelViewMatrix;
uniform sampler2D materialTex;
uniform float materialShininess;
uniform vec3 materialSpecularColor;

#define MAX_LIGHTS 10
uniform int numLights;
uniform struct Light {
   vec4 position;
   vec3 intensities; //a.k.a the color of the light
   float attenuation;
   float ambientCoefficient;
} allLights[MAX_LIGHTS];

in vec3 fragPosition;
in vec3 fragCanonicalPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;

layout (location = 0) out vec3 fragOutput0;
layout (location = 1) out vec4 fragOutput1;
layout (location = 2) out vec3 fragOutput2;


vec3 ApplyLight(Light light, vec3 surfaceColor, vec3 normal, vec3 surfacePos, vec3 surfaceToCamera) 
{
    vec3 surfaceToLight;
    float attenuation = 1.0;
    if(light.position.w == 0.0) {
        //directional light
        surfaceToLight = normalize(light.position.xyz);
        attenuation = 1.0; //no attenuation for directional lights
    } else {
        //point light
        surfaceToLight = normalize(light.position.xyz - surfacePos);
        float distanceToLight = length(light.position.xyz - surfacePos);
        attenuation = 1.0 / (1.0 + light.attenuation * pow(distanceToLight, 2));
    }

    //ambient
    vec3 ambient = light.ambientCoefficient * surfaceColor.rgb * light.intensities;

    //diffuse
    float diffuseCoefficient = max(0.0, dot(normal, surfaceToLight));
    vec3 diffuse = diffuseCoefficient * surfaceColor.rgb * light.intensities;
    
    //specular
    float specularCoefficient = 0.0;
    if(diffuseCoefficient > 0.0)
        specularCoefficient = pow(max(0.0, dot(surfaceToCamera, reflect(-surfaceToLight, normal))), materialShininess);
    vec3 specular = specularCoefficient * materialSpecularColor * light.intensities;

    //linear color (color before gamma correction)
    return ambient + attenuation*(diffuse + specular);
}


void main()
{

    if (isnan(fragCanonicalPosition.x)) {
        discard;
    }

    // vertex position
    fragOutput0 = fragCanonicalPosition;

    // color with lighting
    vec3 normal = normalize(transpose(inverse(mat3(modelViewMatrix))) * fragNormal);
    vec3 surfacePos = vec3(modelViewMatrix * vec4(fragPosition, 1));
    vec4 surfaceColor = texture(materialTex, fragTexCoord);
    vec3 surfaceToCamera = normalize(-surfacePos);

    //combine color from all the lights
    vec3 linearColor = vec3(0);
    for(int i = 0; i < numLights; ++i){
        linearColor += ApplyLight(allLights[i], surfaceColor.rgb, normal, surfacePos, surfaceToCamera);
    }
    
    //final color
    fragOutput1 = vec4(linearColor, surfaceColor.a);

    // depth
    fragOutput2 = vec3(gl_FragCoord.z);

}
