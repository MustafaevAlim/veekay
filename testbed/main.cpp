#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>

#define _USE_MATH_DEFINES
#include <math.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

size_t aligned_sizeof;

constexpr uint32_t max_models = 1024;
constexpr uint32_t MAX_POINT_LIGHTS = 8; // Увеличим для гибкости

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

struct MaterialProperties {
    veekay::vec3 albedo{1.0f, 1.0f, 1.0f};
    float shininess = 32.0f;
    veekay::vec3 specular_color{1.0f, 1.0f, 1.0f};
    float _pad0;
};

struct PointLight {
    veekay::vec3 position{0.0f, 0.0f, 0.0f};
    float _pad0;
    veekay::vec3 color{1.0f, 1.0f, 1.0f};
    float constant{1.0f};
    float linear{0.09f};
    float quadratic{0.032f};
    float _pad1;
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    veekay::vec3 ambient_color{0.1f, 0.1f, 0.1f};
    int point_light_count = 0;
    veekay::vec3 camera_pos;
    float _pad1;
};

struct ModelUniforms {
    veekay::mat4 model;
    veekay::mat4 normal_matrix;
    MaterialProperties material;
};

struct Mesh {
    veekay::graphics::Buffer* vertex_buffer;
    veekay::graphics::Buffer* index_buffer;
    uint32_t indices;
};

struct Transform {
    veekay::vec3 position = {};
    veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
    veekay::vec3 rotation = {};
    veekay::mat4 matrix() const;
};

struct Model {
    Mesh mesh;
    Transform transform;
    MaterialProperties material;
};

struct Camera {
    constexpr static float default_fov = 60.0f;
    constexpr static float default_near_plane = 0.01f;
    constexpr static float default_far_plane = 100.0f;

    veekay::vec3 position = {};
    veekay::vec3 rotation = {};

    float fov = default_fov;
    float near_plane = default_near_plane;
    float far_plane = default_far_plane;

    veekay::mat4 view() const;
    veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
inline namespace {
    Camera camera{
        .position = {-3.85f, -0.5f, 0.25f}
    };

    std::vector<Model> models;

    // Глобальные источники света
    veekay::vec3 ambient_color{0.1f, 0.1f, 0.1f};
    std::vector<PointLight> point_lights; // Изначально пустой вектор
}

// NOTE: Vulkan objects
inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;

    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;
    veekay::graphics::Buffer* point_lights_buffer;

    Mesh plane_mesh;
    Mesh cube_mesh;

    veekay::graphics::Texture* missing_texture;
    VkSampler missing_texture_sampler;
}

float toRadians(float degrees) {
    return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
    veekay::mat4 scale_mat = veekay::mat4::scaling(scale);
    veekay::mat4 yaw_rot = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
    veekay::mat4 pitch_rot = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
    veekay::mat4 roll_rot = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
    veekay::mat4 rotation_mat = yaw_rot * pitch_rot * roll_rot;
    veekay::mat4 translation_mat = veekay::mat4::translation(position);
    return translation_mat * rotation_mat * scale_mat;
}

veekay::mat4 look_at_matrix(const veekay::vec3& eye, const veekay::vec3& target, const veekay::vec3& world_up) {
    veekay::vec3 forward = veekay::vec3::normalized(target - eye);
    veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(world_up, forward));
    veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(forward, right));

    veekay::mat4 rotation;
    rotation[0][0] = right.x;   rotation[0][1] = right.y;   rotation[0][2] = right.z;   rotation[0][3] = 0.0f;
    rotation[1][0] = up.x;      rotation[1][1] = up.y;      rotation[1][2] = up.z;      rotation[1][3] = 0.0f;
    rotation[2][0] = forward.x; rotation[2][1] = forward.y; rotation[2][2] = forward.z; rotation[2][3] = 0.0f;
    rotation[3][0] = 0.0f;      rotation[3][1] = 0.0f;      rotation[3][2] = 0.0f;      rotation[3][3] = 1.0f;

    veekay::mat4 translation = veekay::mat4::identity();
    translation[3][0] = veekay::vec3::dot(right, eye);
    translation[3][1] = -veekay::vec3::dot(up, eye);
    translation[3][2] = -veekay::vec3::dot(forward, eye);

    return rotation * translation;
}

veekay::mat4 Camera::view() const {
    float yaw_rad = rotation.y;
    float pitch_rad = rotation.x;
    
    float cos_yaw = cosf(yaw_rad);
    float sin_yaw = sinf(yaw_rad);
    float cos_pitch = cosf(pitch_rad);
    float sin_pitch = sinf(pitch_rad);
    
    veekay::vec3 forward = {
        cos_yaw * cos_pitch,
        sin_pitch,
        sin_yaw * cos_pitch
    };
    forward = veekay::vec3::normalized(forward);

    veekay::vec3 target = position + forward;
    veekay::vec3 up = {0.0f, 1.0f, 0.0f};

    return look_at_matrix(position, target, up);
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
    auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
    return view() * projection;
}

VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return nullptr;
    size_t size = file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = buffer.data(),
    };

    VkShaderModule result;
    if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
        return nullptr;
    }
    return result;
}

void initialize(VkCommandBuffer cmd) {
    VkDevice& device = veekay::app.vk_device;
    VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);
    uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
    aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

    {
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");

        VkPipelineShaderStageCreateInfo stage_infos[2] = {
            { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader_module, .pName = "main" },
            { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader_module, .pName = "main" }
        };

        VkVertexInputBindingDescription buffer_binding{ .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription attributes[] = {
            { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
            { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
            { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(Vertex, uv) },
        };

        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]), .pVertexAttributeDescriptions = attributes
        };

        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
        VkPipelineRasterizationStateCreateInfo raster_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_BACK_BIT, .frontFace = VK_FRONT_FACE_CLOCKWISE, .lineWidth = 1.0f };
        VkPipelineMultisampleStateCreateInfo sample_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
        VkViewport viewport{ .width = (float)veekay::app.window_width, .height = (float)veekay::app.window_height, .minDepth = 0.0f, .maxDepth = 1.0f };
        VkRect2D scissor{ .extent = {veekay::app.window_width, veekay::app.window_height} };
        VkPipelineViewportStateCreateInfo viewport_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor };
        VkPipelineDepthStencilStateCreateInfo depth_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL };
        VkPipelineColorBlendAttachmentState attachment_info{ .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT };
        VkPipelineColorBlendStateCreateInfo blend_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &attachment_info };

        {
            VkDescriptorPoolSize pools[] = {
                { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 8 },
                { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 8 },
                { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8 },
            };
            VkDescriptorPoolCreateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = sizeof(pools)/sizeof(pools[0]), .pPoolSizes = pools };
            vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
        }

        {
            VkDescriptorSetLayoutBinding bindings[] = {
                { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
            };
            VkDescriptorSetLayoutCreateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = sizeof(bindings)/sizeof(bindings[0]), .pBindings = bindings };
            vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout);
        }

        {
            VkDescriptorSetAllocateInfo info{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &descriptor_set_layout };
            vkAllocateDescriptorSets(device, &info, &descriptor_set);
        }

        VkPipelineLayoutCreateInfo layout_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &descriptor_set_layout };
        vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
        
        VkGraphicsPipelineCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stage_infos,
            .pVertexInputState = &input_state_info, .pInputAssemblyState = &assembly_state_info, .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info, .pMultisampleState = &sample_info, .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info, .layout = pipeline_layout, .renderPass = veekay::app.vk_render_pass,
        };
        vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline);
    }

    scene_uniforms_buffer = new veekay::graphics::Buffer(sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    model_uniforms_buffer = new veekay::graphics::Buffer(max_models * aligned_sizeof, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    point_lights_buffer = new veekay::graphics::Buffer(MAX_POINT_LIGHTS * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    {
        VkSamplerCreateInfo info{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
        vkCreateSampler(device, &info, nullptr, &missing_texture_sampler);
        uint32_t pixels[] = { 0xff000000, 0xffff00ff, 0xffff00ff, 0xff000000 };
        missing_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
    }

    {
        VkDescriptorBufferInfo buffer_infos[] = {
            { .buffer = scene_uniforms_buffer->buffer, .range = sizeof(SceneUniforms) },
            { .buffer = model_uniforms_buffer->buffer, .range = sizeof(ModelUniforms) },
            { .buffer = point_lights_buffer->buffer, .range = MAX_POINT_LIGHTS * sizeof(PointLight) },
        };

        VkWriteDescriptorSet write_infos[] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer_infos[0] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &buffer_infos[1] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buffer_infos[2] },
        };
        vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]), write_infos, 0, nullptr);
    }

    {
        std::vector<Vertex> vertices = {
            {{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        };
        std::vector<uint32_t> indices = { 0, 1, 2, 2, 3, 0 };
        plane_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        plane_mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        plane_mesh.indices = (uint32_t)indices.size();
    }

    {
        std::vector<Vertex> vertices = {
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}}, {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}}, {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}}, {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}},
            {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}}, {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}}, {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}}, {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}}, {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}}, {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}}, {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}},
            {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}}, {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}}, {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}},
            {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}}, {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}}, {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}}, {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}},
            {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}}, {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}}, {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}},
        };
        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12, 16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
        };
        cube_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        cube_mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        cube_mesh.indices = (uint32_t)indices.size();
    }

    models.emplace_back(Model{
        .mesh = plane_mesh,
        .transform = {.position = {0.0f, 0.25f, 0.0f}},
        .material = { .albedo = {0.5f, 0.5f, 0.5f}, .shininess = 16.0f }
    });
    models.emplace_back(Model{
        .mesh = cube_mesh, .transform = {.position = {1.5f, -0.5f, -0.5f}, .scale = {0.5f, 0.5f, 0.5f}},
        .material = { .albedo = {0.0f, 1.0f, 0.0f}, .shininess = 64.0f, .specular_color = {0.0f, 1.0f, 0.0f}}
    });
    models.emplace_back(Model{
        .mesh = cube_mesh, .transform = { .position = {-2.0f, -0.5f, -1.5f} },
        .material = { .albedo = {1.0f, 0.0f, 0.0f}, .shininess = 32.0f }
    });
    models.emplace_back(Model{
        .mesh = cube_mesh, .transform = { .position = {0.0f, -0.5f, 1.0f} },
        .material = { .albedo = {0.0f, 0.5f, 0.8f}, .shininess = 256.0f }
    });

    // Источники света теперь не инициализируются здесь
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    vkDestroySampler(device, missing_texture_sampler, nullptr);
    delete missing_texture;

    delete cube_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;
    delete plane_mesh.index_buffer;
    delete plane_mesh.vertex_buffer;

    delete point_lights_buffer;
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;

    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    ImGui::Begin("Controls:");
    ImGui::ColorEdit3("Ambient Color", &ambient_color.x);

    if (ImGui::Button("Add Point Light") && point_lights.size() < MAX_POINT_LIGHTS) {
        point_lights.push_back(PointLight{ .position = camera.position });
    }

    if (ImGui::CollapsingHeader("Point Lights")) {
        // Используем временный вектор для отслеживания индексов для удаления
        std::vector<int> lights_to_remove;
        for (int i = 0; i < point_lights.size(); ++i) {
            std::string label = "Point Light " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
                ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                ImGui::DragFloat("Linear", &point_lights[i].linear, 0.001f, 0.0f, 1.0f);
                ImGui::DragFloat("Quadratic", &point_lights[i].quadratic, 0.001f, 0.0f, 1.0f);
                if (ImGui::Button("Remove")) {
                    lights_to_remove.push_back(i);
                }
                ImGui::TreePop();
            }
        }
        // Удаляем источники в обратном порядке, чтобы не нарушать индексы
        for (int i = lights_to_remove.size() - 1; i >= 0; --i) {
            point_lights.erase(point_lights.begin() + lights_to_remove[i]);
        }
    }
    ImGui::End();

    if (!ImGui::IsWindowHovered()) {
        using namespace veekay::input;
        if (mouse::isButtonDown(mouse::Button::right)) {
            auto move_delta = mouse::cursorDelta();
            float sensitivity = 0.002f;
            camera.rotation.y += move_delta.x * sensitivity;
            camera.rotation.x -= move_delta.y * sensitivity;
            camera.rotation.x = std::max(-1.57f, std::min(1.57f, camera.rotation.x));
        }

        auto view = camera.view();
        veekay::vec3 right = {view[0][0], view[1][0], view[2][0]};
        veekay::vec3 up = {view[0][1], view[1][1], view[2][1]};
        veekay::vec3 forward = {-view[0][2], -view[1][2], -view[2][2]};

        float speed = 0.05f;
        if (keyboard::isKeyDown(keyboard::Key::w)) camera.position += forward * speed;
        if (keyboard::isKeyDown(keyboard::Key::s)) camera.position -= forward * speed;
        if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * speed;
        if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * speed;
        if (keyboard::isKeyDown(keyboard::Key::q)) camera.position += up * speed;
        if (keyboard::isKeyDown(keyboard::Key::z)) camera.position -= up * speed;
    }

    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .ambient_color = ambient_color,
        .point_light_count = (int)point_lights.size(),
        .camera_pos = camera.position,
    };
    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

    if (!point_lights.empty()) {
        std::memcpy(point_lights_buffer->mapped_region, point_lights.data(), point_lights.size() * sizeof(PointLight));
    }

    uint8_t* base = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms uniforms;
        uniforms.model = model.transform.matrix();
        uniforms.normal_matrix = veekay::mat4::transpose(veekay::mat4::inverse(uniforms.model));
        uniforms.material = model.material;
        std::memcpy(base + i * aligned_sizeof, &uniforms, sizeof(ModelUniforms));
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(cmd, 0);

    {
        VkCommandBufferBeginInfo info{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        vkBeginCommandBuffer(cmd, &info);
    }

    {
        VkClearValue clear_values[] = { {.color = {{0.1f, 0.1f, 0.1f, 1.0f}}}, {.depthStencil = {1.0f, 0}} };
        VkRenderPassBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = veekay::app.vk_render_pass, .framebuffer = framebuffer,
            .renderArea = { .extent = { veekay::app.window_width, veekay::app.window_height }},
            .clearValueCount = 2, .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize zero_offset = 0;

    VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
    VkBuffer current_index_buffer = VK_NULL_HANDLE;

    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        const Mesh& mesh = model.mesh;

        if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
            current_vertex_buffer = mesh.vertex_buffer->buffer;
            vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
        }

        if (current_index_buffer != mesh.index_buffer->buffer) {
            current_index_buffer = mesh.index_buffer->buffer;
            vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
        }

        uint32_t offset = uint32_t(i * aligned_sizeof);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &offset);
        vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
    std::srand(time(nullptr));  
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}
