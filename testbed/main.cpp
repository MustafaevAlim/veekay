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
#include <lodepng.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

size_t aligned_sizeof;

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 8;
constexpr uint32_t max_spot_lights = 8;

struct Material {
    veekay::graphics::Texture* texture = nullptr;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

veekay::graphics::Texture* missing_texture = nullptr;
VkSampler missing_texture_sampler = VK_NULL_HANDLE;

struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

struct PointLight {
    veekay::vec3 position{0.0f, 0.0f, 0.0f};
    float intensity{1.0f};
    veekay::vec3 color{1.0f, 1.0f, 1.0f};
    float _pad0;
};

struct SpotLight {
    veekay::vec3 position;
    float radius;
    veekay::vec3 direction;
    float angle;
    veekay::vec3 color;
    float _pad0;
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    veekay::vec3 camera_pos;
    float _pad0;

    veekay::vec3 ambient_color{0.1f, 0.1f, 0.1f};
    float _pad1;
    veekay::vec3 ambient_lights_intensity;
    float _pad2;

    veekay::vec3 sun_light_direction;
    float _pad3;
    veekay::vec3 sun_light_color;
    float _pad4;

    uint32_t point_light_count = 0;
    uint32_t spot_light_count = 0;
};

struct ModelUniforms {
    veekay::mat4 model;
    veekay::vec3 albedo_color;
    float _pad0;
    veekay::vec3 specular_color;
    float _pad1;
    float shininess;
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
    veekay::vec3 albedo_color;
    veekay::vec3 specular_color;
    float shininess;

    Material* material = nullptr; 
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

// Scene objects
inline namespace {
    Camera camera{
        .position = {2.0f, -0.5f, 0.0f}
    };

    std::vector<Model> models;

    veekay::vec3 ambient_lights_intensity{0.2f, 0.2f, 0.2f};


    veekay::vec3 sun_light_direction{3.0f, 2.0f, 3.0f};
    veekay::vec3 sun_light_color{1.0f, 1.0f, 1.0f};
    veekay::vec3 ambient_color{1.0f, 1.0f, 1.0f};
    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;
}

// Vulkan objects
inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout_ubo;   
    VkDescriptorSetLayout descriptor_set_layout_tex;   
    VkDescriptorSet descriptor_set_ubo;                

    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;
    veekay::graphics::Buffer* point_lights_buffer;
    veekay::graphics::Buffer* spot_lights_buffer;

    Mesh plane_mesh;
    Mesh cube_mesh;
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
    return scale_mat * rotation_mat * translation_mat;
}

veekay::mat4 look_at_matrix(const veekay::vec3& eye, const veekay::vec3& target, const veekay::vec3& world_up) {
    veekay::vec3 forward = veekay::vec3::normalized(eye - target);
    veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(world_up, forward));
    veekay::vec3 up = veekay::vec3::cross(forward, right);

    veekay::mat4 result;
    result[0][0] = right.x;   result[0][1] = up.x;   result[0][2] = forward.x;   result[0][3] = 0.0f;
    result[1][0] = right.y;   result[1][1] = up.y;   result[1][2] = forward.y;   result[1][3] = 0.0f;
    result[2][0] = right.z;   result[2][1] = up.z;   result[2][2] = forward.z;   result[2][3] = 0.0f;
    result[3][0] = -veekay::vec3::dot(right, eye);
    result[3][1] = -veekay::vec3::dot(up, eye);
    result[3][2] = -veekay::vec3::dot(forward, eye);
    result[3][3] = 1.0f;

    return result;
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

bool loadPNG(const std::string& filename, std::vector<uint8_t>& pixels, uint32_t& width, uint32_t& height) {
    unsigned error = lodepng::decode(pixels, width, height, filename);
    if (error) {
        std::cerr << "Ошибка загрузки PNG: " << filename << " : " << lodepng_error_text(error) << std::endl;
        return false;
    }
    return true;
}

Material* createMaterialFromFile(VkCommandBuffer cmd, const std::string& filename, VkDescriptorPool pool) {
    auto* mat = new Material();

    std::vector<uint8_t> pixels;
    uint32_t width = 0, height = 0;
    bool ok = loadPNG(filename, pixels, width, height);

    if (!ok) {
        mat->texture = missing_texture;
        mat->sampler = missing_texture_sampler;
    } else {
        mat->texture = new veekay::graphics::Texture(
            cmd,
            width,
            height,
            VK_FORMAT_R8G8B8A8_UNORM,
            pixels.data()
        );

        VkSamplerCreateInfo sampler_info{};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sampler_info.anisotropyEnable = VK_TRUE;
        sampler_info.maxAnisotropy = 16.0f;
        sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sampler_info.unnormalizedCoordinates = VK_FALSE;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler_info.minLod = 0.0f;
        sampler_info.maxLod = 0.0f;
        sampler_info.mipLodBias = 0.0f;

        if (vkCreateSampler(veekay::app.vk_device, &sampler_info, nullptr, &mat->sampler) != VK_SUCCESS) {
            std::cerr << "Не удалось создать VkSampler для " << filename << std::endl;
            mat->sampler = missing_texture_sampler;
            mat->texture = missing_texture;
        }
    }

    VkDescriptorSetLayout layouts[] = { descriptor_set_layout_tex };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(veekay::app.vk_device, &allocInfo, &mat->set) != VK_SUCCESS) {
        std::cerr << "Не удалось выделить descriptor set для материала " << filename << std::endl;
        mat->set = VK_NULL_HANDLE;
        return mat;
    }

    VkDescriptorImageInfo img{};
    img.imageView = mat->texture->view;
    img.sampler = mat->sampler;
    img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = mat->set;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &img;

    vkUpdateDescriptorSets(veekay::app.vk_device, 1, &write, 0, nullptr);

    return mat;
}

VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return nullptr;
    size_t size = (size_t)file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = buffer.data();

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

    vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
    fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");

    VkVertexInputBindingDescription buffer_binding{};
    buffer_binding.binding = 0;
    buffer_binding.stride = sizeof(Vertex);
    buffer_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(Vertex, uv) },
    };

    VkPipelineVertexInputStateCreateInfo input_state_info{};
    input_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    input_state_info.vertexBindingDescriptionCount = 1;
    input_state_info.pVertexBindingDescriptions = &buffer_binding;
    input_state_info.vertexAttributeDescriptionCount = (uint32_t)(sizeof(attributes) / sizeof(attributes[0]));
    input_state_info.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo assembly_state_info{};
    assembly_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly_state_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineRasterizationStateCreateInfo raster_info{};
    raster_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
    raster_info.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo sample_info{};
    sample_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    sample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkViewport viewport{};
    viewport.width = (float)veekay::app.window_width;
    viewport.height = (float)veekay::app.window_height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = {veekay::app.window_width, veekay::app.window_height};

    VkPipelineViewportStateCreateInfo viewport_info{};
    viewport_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = &viewport;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = &scissor;

    VkPipelineDepthStencilStateCreateInfo depth_info{};
    depth_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState attachment_info{};
    attachment_info.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend_info{};
    blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.attachmentCount = 1;
    blend_info.pAttachments = &attachment_info;

    {
        VkDescriptorPoolSize pools[] = {
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,           1 },      // Scene
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,   1 },      // Models
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,           2 },      // point+spot
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,   16 },     // до 16 материалов
        };
        VkDescriptorPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        info.maxSets = 1 + 16;
        info.poolSizeCount = (uint32_t)(sizeof(pools) / sizeof(pools[0]));
        info.pPoolSizes = pools;
        vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
    }

    {
        VkDescriptorSetLayoutBinding bindings[] = {
            { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
            { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
            { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
            { .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
        };
        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = (uint32_t)(sizeof(bindings) / sizeof(bindings[0]));
        info.pBindings = bindings;
        vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_ubo);
    }

    // Set 1: только combined image sampler
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = 1;
        info.pBindings = &binding;
        vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_tex);
    }

    {
        VkDescriptorSetLayout layouts[] = { descriptor_set_layout_ubo, descriptor_set_layout_tex };
        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
    }

    {
        VkPipelineShaderStageCreateInfo stages[2]{};

        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertex_shader_module;
        stages[0].pName = "main";

        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragment_shader_module;
        stages[1].pName = "main";

        VkGraphicsPipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &input_state_info;
        info.pInputAssemblyState = &assembly_state_info;
        info.pViewportState = &viewport_info;
        info.pRasterizationState = &raster_info;
        info.pMultisampleState = &sample_info;
        info.pDepthStencilState = &depth_info;
        info.pColorBlendState = &blend_info;
        info.layout = pipeline_layout;
        info.renderPass = veekay::app.vk_render_pass;

        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    }

    scene_uniforms_buffer = new veekay::graphics::Buffer(sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    model_uniforms_buffer = new veekay::graphics::Buffer(max_models * aligned_sizeof, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    point_lights_buffer = new veekay::graphics::Buffer(max_point_lights * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    spot_lights_buffer = new veekay::graphics::Buffer(max_spot_lights * sizeof(SpotLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Missing texture (клеточка)
    {
        VkSamplerCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter = VK_FILTER_NEAREST;
        info.minFilter = VK_FILTER_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device, &info, nullptr, &missing_texture_sampler);

        uint32_t pixels[] = {
            0xff000000, 0xffff00ff,
            0xffff00ff, 0xff000000
        };
        missing_texture = new veekay::graphics::Texture(
            cmd,
            2,
            2,
            VK_FORMAT_B8G8R8A8_UNORM,
            pixels
        );
    }

    // Descriptor set для UBO/SSBO (set 0)
    {
        VkDescriptorSetLayout layouts[] = { descriptor_set_layout_ubo };
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptor_pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = layouts;

        vkAllocateDescriptorSets(device, &allocInfo, &descriptor_set_ubo);

        VkDescriptorBufferInfo buffer_infos[] = {
            { scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms) },
            { model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms) },
            { point_lights_buffer->buffer,   0, max_point_lights * sizeof(PointLight) },
            { spot_lights_buffer->buffer,    0, max_spot_lights   * sizeof(SpotLight) },
        };

        VkWriteDescriptorSet writes[4]{};

        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptor_set_ubo;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &buffer_infos[0];

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptor_set_ubo;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &buffer_infos[1];

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptor_set_ubo;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &buffer_infos[2];

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = descriptor_set_ubo;
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].descriptorCount = 1;
        writes[3].pBufferInfo = &buffer_infos[3];

        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }

    {
        std::vector<Vertex> vertices = {
            {{-5.0f, 0.0f,  5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{ 5.0f, 0.0f,  5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{ 5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
        };
        std::vector<uint32_t> idx = {0, 1, 2, 2, 3, 0};

        plane_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        plane_mesh.index_buffer  = new veekay::graphics::Buffer(idx.size() * sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        plane_mesh.indices = (uint32_t)idx.size();
    }

    {
        std::vector<Vertex> vertices = {
            // front
            {{-0.5f, -0.5f, -0.5f}, {0,0,-1}, {0,0}},
            {{+0.5f, -0.5f, -0.5f}, {0,0,-1}, {1,0}},
            {{+0.5f, +0.5f, -0.5f}, {0,0,-1}, {1,1}},
            {{-0.5f, +0.5f, -0.5f}, {0,0,-1}, {0,1}},
            // right
            {{+0.5f, -0.5f, -0.5f}, {1,0,0}, {0,0}},
            {{+0.5f, -0.5f, +0.5f}, {1,0,0}, {1,0}},
            {{+0.5f, +0.5f, +0.5f}, {1,0,0}, {1,1}},
            {{+0.5f, +0.5f, -0.5f}, {1,0,0}, {0,1}},
            // back
            {{+0.5f, -0.5f, +0.5f}, {0,0,1}, {0,0}},
            {{-0.5f, -0.5f, +0.5f}, {0,0,1}, {1,0}},
            {{-0.5f, +0.5f, +0.5f}, {0,0,1}, {1,1}},
            {{+0.5f, +0.5f, +0.5f}, {0,0,1}, {0,1}},
            // left
            {{-0.5f, -0.5f, +0.5f}, {-1,0,0}, {0,0}},
            {{-0.5f, -0.5f, -0.5f}, {-1,0,0}, {1,0}},
            {{-0.5f, +0.5f, -0.5f}, {-1,0,0}, {1,1}},
            {{-0.5f, +0.5f, +0.5f}, {-1,0,0}, {0,1}},
            // bottom
            {{-0.5f, -0.5f, +0.5f}, {0,-1,0}, {0,0}},
            {{+0.5f, -0.5f, +0.5f}, {0,-1,0}, {1,0}},
            {{+0.5f, -0.5f, -0.5f}, {0,-1,0}, {1,1}},
            {{-0.5f, -0.5f, -0.5f}, {0,-1,0}, {0,1}},
            // top
            {{-0.5f, +0.5f, -0.5f}, {0,1,0}, {0,0}},
            {{+0.5f, +0.5f, -0.5f}, {0,1,0}, {1,0}},
            {{+0.5f, +0.5f, +0.5f}, {0,1,0}, {1,1}},
            {{-0.5f, +0.5f, +0.5f}, {0,1,0}, {0,1}},
        };

        std::vector<uint32_t> idx = {
            0,1,2,2,3,0,
            4,5,6,6,7,4,
            8,9,10,10,11,8,
            12,13,14,14,15,12,
            16,17,18,18,19,16,
            20,21,22,22,23,20
        };

        cube_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        cube_mesh.index_buffer  = new veekay::graphics::Buffer(idx.size() * sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        cube_mesh.indices = (uint32_t)idx.size();
    }

    Material* mat_ground = createMaterialFromFile(cmd, "/home/traktor/Graphics/veekay/grass-clipart-lg_imresizer.png", descriptor_pool);
    Material* mat_red    = createMaterialFromFile(cmd, "/home/traktor/Graphics/veekay/woodrat-md_imresizer.png", descriptor_pool);
    Material* mat_green  = createMaterialFromFile(cmd, "/home/traktor/Graphics/veekay/woodrat-md_imresizer.png", descriptor_pool);
    Material* mat_blue   = createMaterialFromFile(cmd, "/home/traktor/Graphics/veekay/Bricks102_4K-PNG_Color.png", descriptor_pool);

    models.emplace_back(Model{
        .mesh = plane_mesh,
        .transform = Transform{},
        .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
        .specular_color = veekay::vec3{1.0f, 1.0f, 1.0f},
        .shininess = 64.0f,
        .material = mat_ground,
    });

    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {-2.0f, -0.5f, -1.5f},
            .scale = {2.0f, 2.0f, 2.0f},
            .rotation = {0.0f, 1.0f, 0.0f},
        },
        .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
        .specular_color = {1.0f, 1.0f, 1.0f},
        .shininess = 56.0f,
        .material = mat_red,
    });

    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {1.5f, -0.5f, -0.5f},
        },
        .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
        .specular_color = {1.0f, 1.0f, 1.0f},
        .shininess = 128.0f,
        .material = mat_green,
    });

    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {0.0f, -0.5f, 1.0f},
        },
        .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
        .specular_color = {1.0f, 0.5f, 0.6f},
        .shininess = 16.0f,
        .material = mat_blue,
    });
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    for (Model& m : models) {
        if (!m.material) continue;
        if (m.material->texture && m.material->texture != missing_texture) {
            delete m.material->texture;
        }
        if (m.material->sampler && m.material->sampler != missing_texture_sampler) {
            vkDestroySampler(device, m.material->sampler, nullptr);
        }
    }

    if (missing_texture_sampler) {
        vkDestroySampler(device, missing_texture_sampler, nullptr);
    }
    if (missing_texture) {
        delete missing_texture;
    }

    delete cube_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;
    delete plane_mesh.index_buffer;
    delete plane_mesh.vertex_buffer;

    delete point_lights_buffer;
    delete spot_lights_buffer;
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;

    vkDestroyDescriptorSetLayout(device, descriptor_set_layout_tex, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout_ubo, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    ImGui::Begin("Controls:");
    ImGui::ColorEdit3("Ambient Color", &ambient_color.x);
    ImGui::DragFloat3("Ambient intensity", &ambient_lights_intensity.x);
    ImGui::ColorEdit3("Sun Color", &sun_light_color.x);
    ImGui::DragFloat3("Sun direction", &sun_light_direction.x);

    if (ImGui::Button("Add Point Light") && point_lights.size() < max_point_lights) {
        point_lights.push_back(PointLight{ .position = camera.position });
    }

    if (ImGui::CollapsingHeader("Point Lights")) {
        std::vector<int> lights_to_remove;
        for (int i = 0; i < (int)point_lights.size(); ++i) {
            std::string label = "Point Light " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
                ImGui::DragFloat("Intensity", &point_lights[i].intensity, 0.1f, 0.0f);
                ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                if (ImGui::Button("Remove")) {
                    lights_to_remove.push_back(i);
                }
                ImGui::TreePop();
            }
        }
        for (int j = (int)lights_to_remove.size() - 1; j >= 0; --j) {
            point_lights.erase(point_lights.begin() + lights_to_remove[j]);
        }
    }

    if (ImGui::Button("Add Spot Light") && spot_lights.size() < max_spot_lights) {
        spot_lights.push_back(SpotLight{
            .position = camera.position,
            .radius = 3.0f,
            .direction = {camera.position.x, -camera.position.y, camera.position.z},
            .angle = toRadians(180.0f),
            .color = {1.0f, 1.0f, 1.0f},
        });
    }

    if (ImGui::CollapsingHeader("Spot Lights")) {
        std::vector<int> lights_to_remove;
        for (int i = 0; i < (int)spot_lights.size(); ++i) {
            std::string label = "Spot Light " + std::to_string(i);
            if (ImGui::TreeNode(label.c_str())) {
                ImGui::DragFloat3("Position", &spot_lights[i].position.x, 0.1f);
                ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.1f);
                ImGui::DragFloat("Radius", &spot_lights[i].radius, 0.1f, 0.0f);
                ImGui::DragFloat("angle", &spot_lights[i].angle, 0.01f);
                ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
                if (ImGui::Button("Remove")) {
                    lights_to_remove.push_back(i);
                }
                ImGui::TreePop();
            }
        }
        for (int j = (int)lights_to_remove.size() - 1; j >= 0; --j) {
            spot_lights.erase(spot_lights.begin() + lights_to_remove[j]);
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
        veekay::vec3 forward = {view[0][2], view[1][2], view[2][2]};

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
        .camera_pos = camera.position,
        .ambient_color = ambient_color,
        .ambient_lights_intensity = ambient_lights_intensity,
        .sun_light_direction = veekay::vec3::normalized(sun_light_direction),
        .sun_light_color = sun_light_color,
        .point_light_count = (uint32_t)point_lights.size(),
        .spot_light_count = (uint32_t)spot_lights.size(),
    };
    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

    if (!point_lights.empty()) {
        std::memcpy(point_lights_buffer->mapped_region, point_lights.data(), point_lights.size() * sizeof(PointLight));
    }
    if (!spot_lights.empty()) {
        std::memcpy(spot_lights_buffer->mapped_region, spot_lights.data(), spot_lights.size() * sizeof(SpotLight));
    }

    uint8_t* base = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms uniforms{};
        uniforms.model = model.transform.matrix();
        uniforms.albedo_color = model.albedo_color;
        uniforms.specular_color = model.specular_color;
        uniforms.shininess = model.shininess;
        std::memcpy(base + i * aligned_sizeof, &uniforms, sizeof(ModelUniforms));
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    VkClearValue clear_values[2];
    clear_values[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp_info{};
    rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_info.renderPass = veekay::app.vk_render_pass;
    rp_info.framebuffer = framebuffer;
    rp_info.renderArea.offset = {0,0};
    rp_info.renderArea.extent = {veekay::app.window_width, veekay::app.window_height};
    rp_info.clearValueCount = 2;
    rp_info.pClearValues = clear_values;

    vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDeviceSize zero_offset = 0;

    VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
    VkBuffer current_index_buffer  = VK_NULL_HANDLE;

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

        vkCmdBindDescriptorSets(
            cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,                
            1,                
            &descriptor_set_ubo,
            1,               
            &offset           
        );

        if (model.material && model.material->set != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(
                cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout,
                1,                
                1,                
                &model.material->set,
                0,
                nullptr
            );
        }

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
