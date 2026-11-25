// The wind field round the camera: the air that people and cars stir up, for the grass.
//
// the reference game keeps a small fluid simulation on a 3D grid centred on the camera (its
// "wind fluid": interaction shader, read from the compiled shaders; this is our own code for it,
// the passes and formulas are the game's - shaders/wind_fluid.comp says what each does). Its
// foliage reads the result within 64 m of the camera: where the air moves, the wind there
// turns that way and grows up to full strength (tuft_cd.glsl).
//
//   48 x 48 x 48 cells of 2 m (a 96 m cube), RGBA16F, velocity in metres per 1/30 s.
//   Every frame (wind_fluid_update, before the world is drawn), in the game's order:
//     advect -> forces -> curl -> confinement -> pressure x 8 -> gradient -> history
//   and the history - what the plants read - is at scene binding 45.
//   The grid's size, its cell and the number of pressure steps are not in the game's shaders
//   (its CPU sets them): chosen here.
//
// The forces (the game's list is filled by its CPU, with whatever the game wants to blow):
// everything that moves on the ground near by drags the air along - the player on foot, the
// players' cars, the world's people and their cars; twenty at most, the nearest.
//
// VIBE_NO_WIND_FLUID=1: no forces (the field stays still, the plants see the global wind only).
// VIBE_TEST_WIND=<power>: a vortex of that power 6 m in front of the camera (test shots).
#include "engine/vk_engine/internal.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "npc/npc.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vibe {

namespace {
constexpr uint32_t kN = 48;                 // cells a side (shaders/tuft_cd.glsl kWindN: keep in step)
constexpr float    kCell = 2.0f;            // metres a cell (kWindCell there)
constexpr int      kPressureSteps = 8;      // (even: the result ends in the first pressure picture)
constexpr int      kForces = 20;
constexpr VkFormat kFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
// The pictures: 0, 1 velocity; 2, 3 pressure; 4 curl; 5 history (the plants'); 6 last history.
constexpr int kImages = 7;

struct Forces { glm::vec4 pos_power[kForces]; glm::vec4 dir_radius[kForces]; glm::vec4 type[kForces]; };

struct WindFluid {
    VkImage        image[kImages] = {};
    VmaAllocation  alloc[kImages] = {};
    VkImageView    view[kImages] = {};
    VkSampler      sampler = VK_NULL_HANDLE;          // trilinear, still air beyond the edge
    VkBuffer       forces = VK_NULL_HANDLE;
    VmaAllocation  forces_alloc = nullptr;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkDescriptorSet       set = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkShaderModule        module = VK_NULL_HANDLE;
    glm::ivec3     cell{0};                           // the camera's cell last frame
    bool           have_cell = false;
    float          time = 0.0f;
    std::vector<glm::vec3> cars_prev;
} g_wf;
}  // namespace

void VulkanEngine::init_wind_fluid() {
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    vk_check(vkCreateSampler(device_, &si, nullptr, &g_wf.sampler), "wind fluid sampler");
    for (int l = 0; l < kImages; ++l) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_3D;
        ici.format = kFormat;
        ici.extent = { kN, kN, kN };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &g_wf.image[l], &g_wf.alloc[l], nullptr), "wind fluid image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = g_wf.image[l];
        vci.viewType = VK_IMAGE_VIEW_TYPE_3D;
        vci.format = kFormat;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &g_wf.view[l]), "wind fluid view");
        // (still air, and in the one layout they stay in: read and written by compute, read by the plants)
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, g_wf.image[l], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            VkClearColorValue zero{};
            VkImageSubresourceRange all{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, g_wf.image[l], VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        });
    }
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(Forces);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &g_wf.forces, &g_wf.forces_alloc, nullptr), "wind fluid forces");
    }
    // The plants' picture: scene binding 45.
    {
        VkDescriptorImageInfo ii{ g_wf.sampler, g_wf.view[5], VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 45; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    // The simulation's own set: 0..6 the pictures to read, 7..13 the same to write, 14 the forces.
    VkDescriptorSetLayoutBinding b[15]{};
    for (uint32_t i = 0; i < 15; ++i) {
        b[i].binding = i;
        b[i].descriptorType = i < 7 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : (i < 14 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 15; lci.pBindings = b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &g_wf.set_layout), "wind fluid set layout");
    VkDescriptorPoolSize ps[3] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 3; pci.pPoolSizes = ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &g_wf.pool), "wind fluid pool");
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = g_wf.pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &g_wf.set_layout;
    vk_check(vkAllocateDescriptorSets(device_, &dai, &g_wf.set), "wind fluid set");
    VkDescriptorImageInfo ii[14]{};
    VkWriteDescriptorSet w[15]{};
    for (uint32_t i = 0; i < 14; ++i) {
        ii[i] = { i < 7 ? g_wf.sampler : VK_NULL_HANDLE, g_wf.view[i % 7], VK_IMAGE_LAYOUT_GENERAL };
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = g_wf.set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = i < 7 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &ii[i];
    }
    VkDescriptorBufferInfo bi{ g_wf.forces, 0, VK_WHOLE_SIZE };
    w[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[14].dstSet = g_wf.set; w[14].dstBinding = 14; w[14].descriptorCount = 1;
    w[14].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[14].pBufferInfo = &bi;
    vkUpdateDescriptorSets(device_, 15, w, 0, nullptr);

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::vec4) * 3 };
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &g_wf.set_layout;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, &g_wf.layout), "wind fluid layout");
    const std::string sd = VIBE_SHADER_DIR;
    g_wf.module = vkpipe::load_shader_module(device_, sd + "/wind_fluid.comp.spv");
    if (g_wf.module) {
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = g_wf.module;
        cci.stage.pName = "main";
        cci.layout = g_wf.layout;
        vk_check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &g_wf.pipeline), "wind fluid pipeline");
    }
    if (!g_wf.pipeline) log::warnf("[wind] wind_fluid shader missing - the plants see the global wind only");
    else log::infof("wind field built (%u^3 cells of %.0f m round the camera)", kN, double(kCell));
}

void VulkanEngine::destroy_wind_fluid() {
    if (g_wf.pipeline) vkDestroyPipeline(device_, g_wf.pipeline, nullptr);
    if (g_wf.layout) vkDestroyPipelineLayout(device_, g_wf.layout, nullptr);
    if (g_wf.module) vkDestroyShaderModule(device_, g_wf.module, nullptr);
    if (g_wf.pool) vkDestroyDescriptorPool(device_, g_wf.pool, nullptr);
    if (g_wf.set_layout) vkDestroyDescriptorSetLayout(device_, g_wf.set_layout, nullptr);
    if (g_wf.forces) vmaDestroyBuffer(allocator_, g_wf.forces, g_wf.forces_alloc);
    if (g_wf.sampler) vkDestroySampler(device_, g_wf.sampler, nullptr);
    for (int l = 0; l < kImages; ++l) {
        if (g_wf.view[l]) vkDestroyImageView(device_, g_wf.view[l], nullptr);
        if (g_wf.image[l]) vmaDestroyImage(allocator_, g_wf.image[l], g_wf.alloc[l]);
    }
    g_wf = WindFluid{};
}

// Once a frame, before anything that reads the wind field is drawn.
void VulkanEngine::wind_fluid_update(VkCommandBuffer cmd) {
    if (!g_wf.pipeline) return;
    static const bool off = std::getenv("VIBE_NO_WIND_FLUID") != nullptr;
    const float dt = std::clamp(anim_time_ - g_wf.time, 0.0f, 0.1f);
    g_wf.time = anim_time_;
    const glm::vec3 eye = player_.eye_position();
    const glm::ivec3 cell = glm::ivec3(glm::floor(eye / kCell));
    glm::ivec3 shift = g_wf.have_cell ? cell - g_wf.cell : glm::ivec3(0);
    if (glm::any(glm::greaterThan(glm::abs(shift), glm::ivec3(int(kN))))) shift = glm::ivec3(int(kN));      // (moved elsewhere: all still air)
    g_wf.cell = cell; g_wf.have_cell = true;

    // ---- The forces: what moves near by drags the air along (the nearest twenty) ----
    struct F { glm::vec3 pos; glm::vec3 dir; float power, radius, type, d2; };
    std::vector<F> all;
    const float reach = 0.5f * float(kN) * kCell;
    auto mover = [&](glm::vec3 p, glm::vec3 vel, float radius, float lift) {
        const float sp = glm::length(vel);
        const glm::vec3 d = p - eye;
        if (sp < 0.3f || glm::dot(d, d) > reach * reach) return;
        // (velocity is in metres per 1/30 s; the air comes up to the body's speed in about a
        // sixth of a second under it)
        all.push_back({ p + glm::vec3(0.0f, lift, 0.0f), vel / sp, std::min(sp / 30.0f, 2.0f) * dt * 6.0f, radius, 0.0f, glm::dot(d, d) });
    };
    if (!off && dt > 0.0f && globe::current().home) {
        if (!body_seated_) mover(eye - glm::vec3(0.0f, 0.9f, 0.0f), player_.velocity, 1.5f, 0.0f);
        std::vector<glm::vec3> cars;
        jeep_positions(cars);
        for (size_t i = 0; i < cars.size(); ++i)
            if (g_wf.cars_prev.size() == cars.size()) {
                const glm::vec3 vel = (cars[i] - g_wf.cars_prev[i]) / dt;
                if (glm::dot(vel, vel) < 80.0f * 80.0f) mover(cars[i], vel, 8.0f, 0.0f);
            }
        g_wf.cars_prev = cars;
        for (const npc::View& v : npc_views_) {
            if (v.hidden) continue;
            const bool car = v.kind == npc::Kind::Car;
            mover(v.pos, glm::vec3(std::sin(v.yaw), 0.0f, -std::cos(v.yaw)) * v.speed, car ? 8.0f : 1.5f, car ? 0.8f : 0.9f);
        }
        // (VIBE_TEST_WIND=<power>: a vortex about the vertical 6 m in front of the camera)
        static const float test = std::getenv("VIBE_TEST_WIND") ? float(std::atof(std::getenv("VIBE_TEST_WIND"))) : 0.0f;
        if (test != 0.0f) {
            const glm::mat4& vm = current_frame_view_.view;
            const glm::vec3 fw = glm::normalize(glm::vec3(-vm[0][2], 0.0f, -vm[2][2]) + glm::vec3(1e-4f, 0.0f, 0.0f));
            all.push_back({ eye + fw * 6.0f - glm::vec3(0.0f, 1.2f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), test * dt, 12.0f, 2.0f, 0.0f });
        }
    }
    std::sort(all.begin(), all.end(), [](const F& a, const F& b) { return a.d2 < b.d2; });
    Forces fb{};
    const int nf = int(std::min<size_t>(all.size(), kForces));
    for (int i = 0; i < nf; ++i) {
        fb.pos_power[i]  = glm::vec4(all[size_t(i)].pos, all[size_t(i)].power);
        fb.dir_radius[i] = glm::vec4(all[size_t(i)].dir, all[size_t(i)].radius);
        fb.type[i]       = glm::vec4(all[size_t(i)].type, 0.0f, 0.0f, 0.0f);
    }
    vkCmdUpdateBuffer(cmd, g_wf.forces, 0, sizeof(fb), &fb);

    // ---- The passes ----
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    struct { glm::vec4 cam, grid; glm::ivec4 shift; } pc{};
    pc.cam = glm::vec4(eye, dt);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_wf.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_wf.layout, 0, 1, &g_wf.set, 0, nullptr);
    auto pass = [&](int mode, int pressure_src) {
        vkCmdPipelineBarrier2(cmd, &dep);
        pc.grid = glm::vec4(float(kN), kCell, float(nf), float(mode));
        pc.shift = glm::ivec4(shift, pressure_src);
        vkCmdPushConstants(cmd, g_wf.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, kN / 8, kN / 8, kN / 8);
    };
    pass(0, 0);                                        // advect       v0 -> v1
    pass(1, 0);                                        // forces       v1 -> v0
    pass(2, 0);                                        // curl         v0 -> curl
    pass(3, 0);                                        // confinement  v0, curl -> v1
    for (int k = 0; k < kPressureSteps; ++k) pass(4, k & 1);     // pressure     p0 -> p1 -> p0 ...
    pass(5, 0);                                        // gradient     v1, p0 -> v0
    pass(6, 0);                                        // history      v0, last history -> history
    vkCmdPipelineBarrier2(cmd, &dep);
    // Keep the history as the next frame's "last".
    VkImageCopy r{};
    r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    r.extent = { kN, kN, kN };
    vkCmdCopyImage(cmd, g_wf.image[5], VK_IMAGE_LAYOUT_GENERAL, g_wf.image[6], VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    vkCmdPipelineBarrier2(cmd, &dep);
}

}  // namespace vibe
