#version 450

layout (location = 0) in vec3 f_world_pos;
layout (location = 1) in vec3 f_world_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct MaterialProperties {
    vec3 albedo;
    float shininess;
    vec3 specular_color;
};

struct PointLight {
    vec3 position;
    float _pad0;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
};

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 ambient_color;
    int point_light_count;
    vec3 camera_pos;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    mat4 normal_matrix;
    MaterialProperties material;
};

layout (binding = 2, std430) readonly buffer PointLightsBuffer {
    PointLight point_lights[];
};

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 light_dir = normalize(light.position - frag_pos);
    // Diffuse
    float diff = max(dot(normal, light_dir), 0.0);
    // Specular (Blinn-Phong)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);

    // Attenuation
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));

    vec3 diffuse = light.color * diff * material.albedo;
    vec3 specular = light.color * spec * material.specular_color;

    return (diffuse + specular) * attenuation;
}

void main() {
    vec3 norm = normalize(f_world_normal);
    vec3 view_dir = normalize(camera_pos - f_world_pos);

    // Ambient
    vec3 result = ambient_color * material.albedo;

    // Point Lights
    for (int i = 0; i < point_light_count; i++) {
        result += CalcPointLight(point_lights[i], norm, f_world_pos, view_dir);
    }

    final_color = vec4(result, 1.0f);
}
