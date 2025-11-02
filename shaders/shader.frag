#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;


struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad0;
};

struct SpotLight {
    vec3 position;
    float radius;
    vec3 direction;
    float angle;
    vec3 color; float _pad0;
};

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_pos;
    float _pad0;  // Добавьте паддинг
    
    vec3 ambient_color;
    float _pad1;  // Добавьте паддинг
    vec3 ambient_light_intensity;
    float _pad2;  // Добавьте паддинг
    
    vec3 sun_light_direction;
    float _pad3;  // Добавьте паддинг
    vec3 sun_light_color;
    float _pad4;  // Добавьте паддинг

    uint point_light_count;
    uint spot_light_count;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    vec3 specular_color;
    float shininess;	
};

layout (binding = 2, std430) readonly buffer PointLightsBuffer {
    PointLight point_lights[];
};

layout (binding = 3, std430) readonly buffer SpotLightsBuffer {
    SpotLight spot_lights[];
};

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_pos - f_position);
    
    // Начинаем с ambient освещения
    vec3 color = ambient_light_intensity * albedo_color;
    
    // Солнечное освещение
    vec3 light_dir = -sun_light_direction;  // Инвертируйте направление!
    float sun_shade = max(0.0, dot(light_dir, normal));
    
    if (sun_shade > 0.0) {
        vec3 half_vector = normalize(view_dir + light_dir);
        vec3 sun_diffuse = albedo_color * sun_light_color * sun_shade;
        vec3 sun_specular = specular_color * sun_light_color * 
                           pow(max(0.0, dot(normal, half_vector)), shininess);
        color += sun_diffuse + sun_specular;
    }
    
    // Точечные источники света
    for (uint i = 0; i < point_light_count; ++i) {
        PointLight light = point_lights[i];
        
        vec3 light_dir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);
        
        // Закон обратных квадратов: I = 1 / d²
        float light_falloff = light.intensity / (distance * distance);
        
        // Диффузное освещение
        float light_shade = max(0.0, dot(normal, light_dir));
        vec3 light_diffuse = albedo_color * light.color * light_shade;
        
        // Бликовое освещение (Блинн-Фонг)
        vec3 half_vec = normalize(light_dir + view_dir);
        float spec_angle = max(0.0, dot(normal, half_vec));
        vec3 light_spec = specular_color * light.color * 
                        pow(spec_angle, shininess);
        
        color += light_falloff * (light_diffuse + light_spec);
    }
    
    for (uint i = 0; i < spot_light_count; ++i) {
        SpotLight light = spot_lights[i];
        
        // Вектор от фрагмента к источнику света
        vec3 light_dir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);
        
        // Проверка, находится ли фрагмент в радиусе действия прожектора
        if (distance > light.radius) {
            continue; // Пропускаем, если фрагмент за пределами радиуса
        }
        
        // Вектор направления прожектора (нормализованный)
        vec3 spot_dir = normalize(light.direction);
        
        // Угол между направлением прожектора и направлением к фрагменту
        // Используем отрицательное направление луча света (-light_dir)
        float theta = dot(-light_dir, spot_dir);
        
        // Косинус cutoff угла (angle — это полный угол конуса в радианах)
        float cutoff_cos = cos(light.angle);
        
        // Проверяем, находится ли фрагмент внутри конуса прожектора
        if (theta > cutoff_cos) {
            // Затухание по расстоянию (линейное затухание от 1.0 до 0.0)
            float distance_attenuation = clamp(1.0 - (distance / light.radius), 0.0, 1.0);
            
            // Плавное затухание по углу конуса (smooth falloff)
            // Создаем "мягкие края" прожектора
            float outer_cutoff = cos(light.angle);
            float inner_cutoff = cos(light.angle * 0.8); // Внутренний угол = 80% от внешнего
            float epsilon = inner_cutoff - outer_cutoff;
            float spot_intensity = clamp((theta - outer_cutoff) / epsilon, 0.0, 1.0);
            
            // Общее затухание = расстояние * угол
            float total_attenuation = distance_attenuation * spot_intensity;
            
            // Диффузное освещение (Блинн-Фонг)
            float light_shade = max(0.0, dot(normal, light_dir));
            vec3 light_diffuse = albedo_color * light.color * light_shade;
            
            // Бликовое освещение (Блинн-Фонг)
            vec3 half_vec = normalize(light_dir + view_dir);
            float spec_angle = max(0.0, dot(normal, half_vec));
            vec3 light_spec = specular_color * light.color * 
                            pow(spec_angle, shininess);
            
            color += total_attenuation * (light_diffuse + light_spec);
        }
    }


    final_color = vec4(color, 1.0);
}