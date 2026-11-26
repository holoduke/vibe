// The local shallow-water solver round the camera, after the reference game's (its numerics and
// constants; the code is ours; shaders/water_solver.comp has each pass, docs/ARCHITECTURE.md
// "The local shallow-water solver" the choices):
//   * a 512 x 512 height field of 0.2 m cells (102 m) centred on the eye, scrolling in whole cells;
//     per cell the surface height, velocity, two foam channels and whether it is the solver's
//     (the bed and the authored top are worked out afresh every frame: mode 0);
//   * fixed steps of 0.01 s, as many as the frame needs, at most 4 (sw::take_steps);
//   * the bed is the home tile's ground with the boulders, rocks and piers stamped in
//     (water_fx.cpp water_bed_obstacles); the authored top is the lake's / river's level or the
//     stream's sloped surface; the open sea is not the solver's;
//   * Vibe's river is level - the reference's gravity-only current would be zero there - so the
//     velocity is pulled toward the course's current (water_course_flow, a 32 x 32 grid of 3.2 m
//     handed over in the parameters) with a 2 s time constant; the solver bends it round the stones;
//   * moving bodies (the player, cars, the world's people) push bow waves and drag the water;
//     splashes (bullets, bodies, blasts) push craters out;
//   * foam particles (up to 262 144): spawned by the foam rule's tests, carried on the surface,
//     falling where it drops away, drawn as lit points at the end of render_fx;
//   * the CPU read-back (f32 height + 2 x f16 velocity per cell, every other frame, read two frames
//     later): water_flow_at - the current that carries the swimmer and the wader, splashes and rings.
// Switches: VIBE_NO_SOLVER=1 (the analytic water alone), VIBE_NO_FOAM_PARTICLES=1. Hooks:
// VIBE_SW_STEPS=<n> (exactly n steps every frame: pictures that do not depend on the frame rate),
// VIBE_LOG_SOLVER=1 (once a second: steps, particles, the read-back at the player).
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include "engine/globe.h"
#include "engine/water_solver_cpu.h"
#include "npc/npc.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vibe {

namespace {
constexpr uint32_t kN = uint32_t(sw::kN);
constexpr float    kCell = sw::kCell;
constexpr int      kObs = 128;
constexpr int      kBodies = 32;
constexpr int      kFlowN = 32;
constexpr float    kFlowCell = 3.2f;
constexpr uint32_t kMaxPart = 262144;
constexpr uint32_t kSlots = 3;
constexpr int      kWarmSteps = 150;     // after a reset: 1.5 s of water before the first picture

struct Params {
    glm::vec4 grid, misc, flow_org, part;
    glm::vec4 obs[kObs];
    glm::vec4 body_a[kBodies], body_b[kBodies];
    glm::vec4 flow[kFlowN * kFlowN];
};
struct Push { glm::mat4 vp; glm::ivec4 m; glm::ivec4 s; glm::vec4 f; };
static_assert(sizeof(Push) == 112, "push");

struct Img { VkImage image = VK_NULL_HANDLE; VmaAllocation alloc = nullptr; VkImageView view = VK_NULL_HANDLE; };
struct Buf { VkBuffer buf = VK_NULL_HANDLE; VmaAllocation alloc = nullptr; void* ptr = nullptr; };

struct Solver {
    Img s0, s1, f0, f1, in, r;
    VkSampler samp = VK_NULL_HANDLE;
    Buf params, parts, counters, pack, rb[kSlots];
    glm::ivec2 rb_corner[kSlots];
    uint64_t rb_frame[kSlots] = {};
    bool rb_valid[kSlots] = {};
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkImageView set_hview = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE, draw = VK_NULL_HANDLE;
    bool ok = false;
    // the plan of this frame (water_solver_fill) and the state's window
    bool have_state = false, reset_all = false;
    glm::ivec2 corner{ 0 }, shift{ 0 };
    int steps = 0, warm = 0;
    double clock = -1.0, acc = 0.0;
    int cur = 0;                     // the particle list that holds the live ones
    bool parts_drawn = false;
    // the course's current on a 32 x 32 grid of 3.2 m (world-aligned; refreshed a part a frame)
    glm::ivec2 flow_corner{ INT32_MIN };
    std::vector<glm::vec4> flow;
    std::vector<uint8_t> flow_fresh;
    int flow_cursor = 0;
    // read-back
    sw::Readback cpu;
    // bodies
    std::vector<glm::vec3> cars_prev;
    float log_t = 0.0f;
    int last_bodies = 0, last_obs = 0;
    uint32_t stat[8] = {};       // VIBE_LOG_SOLVER: what the foam rule saw (water_solver.comp C.stat)
} g_sw;

bool env_on(const char* k) { const char* v = std::getenv(k); return v && *v && *v != '0'; }

void make_img(VkDevice dev, VmaAllocator al, Img& im, VkFormat fmt, const char* name) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { kN, kN, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(al, &ici, &aci, &im.image, &im.alloc, nullptr), name);
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = im.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(dev, &vci, nullptr, &im.view), name);
}
void free_img(VkDevice dev, VmaAllocator al, Img& im) {
    if (im.view) vkDestroyImageView(dev, im.view, nullptr);
    if (im.image) vmaDestroyImage(al, im.image, im.alloc);
    im = Img{};
}
void make_buf(VmaAllocator al, Buf& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host, const char* name) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    VmaAllocationCreateInfo aci{};
    aci.usage = host ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (host) aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    vk_check(vmaCreateBuffer(al, &bci, &aci, &b.buf, &b.alloc, &info), name);
    b.ptr = host ? info.pMappedData : nullptr;
}
void free_buf(VmaAllocator al, Buf& b) { if (b.buf) vmaDestroyBuffer(al, b.buf, b.alloc); b = Buf{}; }
void barrier_all(VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT |
                       VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    mb.dstStageMask = mb.srcStageMask;
    mb.dstAccessMask = mb.srcAccessMask;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}
}  // namespace

void VulkanEngine::init_water_solver() {
    Solver& W = g_sw;
    make_img(device_, allocator_, W.s0, VK_FORMAT_R32G32B32A32_SFLOAT, "solver state 0");
    make_img(device_, allocator_, W.s1, VK_FORMAT_R32G32B32A32_SFLOAT, "solver state 1");
    make_img(device_, allocator_, W.f0, VK_FORMAT_R16G16B16A16_SFLOAT, "solver foam 0");
    make_img(device_, allocator_, W.f1, VK_FORMAT_R16G16B16A16_SFLOAT, "solver foam 1");
    make_img(device_, allocator_, W.in, VK_FORMAT_R32G32B32A32_UINT, "solver inputs");
    make_img(device_, allocator_, W.r,  VK_FORMAT_R16G16B16A16_SFLOAT, "solver render");
    make_buf(allocator_, W.params, sizeof(Params), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, "solver params");
    make_buf(allocator_, W.parts, VkDeviceSize(kMaxPart) * 2 * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, "solver particles");
    make_buf(allocator_, W.counters, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, "solver counters");
    make_buf(allocator_, W.pack, VkDeviceSize(kN) * kN * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false, "solver pack");
    for (uint32_t s = 0; s < kSlots; ++s) make_buf(allocator_, W.rb[s], VkDeviceSize(kN) * kN * 8 + 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, "solver read-back");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (Img* im : { &W.s0, &W.s1, &W.f0, &W.f1, &W.in, &W.r }) {
            VkImageMemoryBarrier2 b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = im->image;
            b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VkDependencyInfo d{}; d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            d.imageMemoryBarrierCount = 1; d.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cb, &d);
            VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            if (im == &W.in) { VkClearColorValue z{}; vkCmdClearColorImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
            else if (im == &W.r) { VkClearColorValue z{}; z.float32[3] = -1.0f; vkCmdClearColorImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
            else { VkClearColorValue z{}; vkCmdClearColorImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
        }
        vkCmdFillBuffer(cb, W.counters.buf, 0, VK_WHOLE_SIZE, 0u);
    });
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &W.samp), "solver sampler");
    }
    // The set: 0 the scene block, 1 the parameters, 2..7 the pictures, 8 the home tile's ground,
    // 9 the particles, 10 their counters and draw arguments, 11 the read-back's packing.
    {
        VkDescriptorSetLayoutBinding b[12]{};
        for (uint32_t i = 0; i < 12; ++i) {
            b[i].binding = i; b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            b[i].descriptorType = i <= 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                : i <= 7 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                : i == 8 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 12; lci.pBindings = b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &W.dsl), "solver dsl");
        VkDescriptorPoolSize ps[4] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 },
                                       { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 } };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 4; pci.pPoolSizes = ps;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &W.pool), "solver pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = W.pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &W.dsl;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &W.set), "solver set");
        VkDescriptorBufferInfo bu{ scene_ubo_buffer_, 0, VK_WHOLE_SIZE }, bp{ W.params.buf, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo bs[3] = { { W.parts.buf, 0, VK_WHOLE_SIZE }, { W.counters.buf, 0, VK_WHOLE_SIZE }, { W.pack.buf, 0, VK_WHOLE_SIZE } };
        VkDescriptorImageInfo ii[6] = { { VK_NULL_HANDLE, W.s0.view, VK_IMAGE_LAYOUT_GENERAL }, { VK_NULL_HANDLE, W.s1.view, VK_IMAGE_LAYOUT_GENERAL },
                                        { VK_NULL_HANDLE, W.f0.view, VK_IMAGE_LAYOUT_GENERAL }, { VK_NULL_HANDLE, W.f1.view, VK_IMAGE_LAYOUT_GENERAL },
                                        { VK_NULL_HANDLE, W.in.view, VK_IMAGE_LAYOUT_GENERAL }, { VK_NULL_HANDLE, W.r.view, VK_IMAGE_LAYOUT_GENERAL } };
        VkWriteDescriptorSet w[11]{};
        for (int i = 0; i < 11; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = W.set; w[i].descriptorCount = 1; }
        w[0].dstBinding = 0; w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &bu;
        w[1].dstBinding = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[1].pBufferInfo = &bp;
        for (int i = 0; i < 6; ++i) { w[2 + i].dstBinding = uint32_t(2 + i); w[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[2 + i].pImageInfo = &ii[i]; }
        for (int i = 0; i < 3; ++i) { w[8 + i].dstBinding = uint32_t(9 + i); w[8 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[8 + i].pBufferInfo = &bs[i]; }
        vkUpdateDescriptorSets(device_, 11, w, 0, nullptr);
        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Push) };
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1; pli.pSetLayouts = &W.dsl;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &pli, nullptr, &W.layout), "solver layout");
        VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/water_solver.comp.spv");
        if (m) {
            VkComputePipelineCreateInfo cci{};
            cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cci.stage.module = m; cci.stage.pName = "main";
            cci.layout = W.layout;
            vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &W.pipe), "solver pipeline");
            vkDestroyShaderModule(device_, m, nullptr);
        }
        VkShaderModule vm = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/water_particles.vert.spv");
        VkShaderModule fm = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/water_particles.frag.spv");
        if (vm && fm) {
            vkpipe::GraphicsPipelineConfig cfg{};
            cfg.vert = vm; cfg.frag = fm;
            cfg.layout = W.layout;
            cfg.color_format = scene_color_format_;
            cfg.color_attachment_count = 1;
            cfg.depth_format = depth_format_;
            cfg.cull = VK_CULL_MODE_NONE;
            cfg.depth_test = true;
            cfg.depth_write = false;
            cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
            cfg.premultiplied_blend = true;
            W.draw = vkpipe::build_graphics_pipeline(device_, cfg);
        }
        if (vm) vkDestroyShaderModule(device_, vm, nullptr);
        if (fm) vkDestroyShaderModule(device_, fm, nullptr);
    }
    // The scene set: 54 the render picture (the surface over the authored top, the current, the foam).
    {
        VkDescriptorImageInfo ii{ W.samp, W.r.view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = scene_desc_set_;
        w.dstBinding = 54; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    W.ok = W.pipe != VK_NULL_HANDLE;
    if (!W.ok) log::warnf("[solver] water_solver shader missing - no shallow-water solver");
    else log::infof("[solver] shallow-water solver ready: %u^2 cells of %.2f m (%.1f m), steps of %.3f s (at most %d a frame), %u foam particles at most%s",
                    kN, double(kCell), double(kN * kCell), double(sw::kStep), sw::kMaxSteps, kMaxPart, W.draw ? "" : " (their draw failed)");
}

void VulkanEngine::destroy_water_solver() {
    Solver& W = g_sw;
    if (W.pipe) vkDestroyPipeline(device_, W.pipe, nullptr);
    if (W.draw) vkDestroyPipeline(device_, W.draw, nullptr);
    if (W.layout) vkDestroyPipelineLayout(device_, W.layout, nullptr);
    if (W.pool) vkDestroyDescriptorPool(device_, W.pool, nullptr);
    if (W.dsl) vkDestroyDescriptorSetLayout(device_, W.dsl, nullptr);
    if (W.samp) vkDestroySampler(device_, W.samp, nullptr);
    for (Buf* b : { &W.params, &W.parts, &W.counters, &W.pack }) free_buf(allocator_, *b);
    for (uint32_t s = 0; s < kSlots; ++s) free_buf(allocator_, W.rb[s]);
    for (Img* im : { &W.s0, &W.s1, &W.f0, &W.f1, &W.in, &W.r }) free_img(device_, allocator_, *im);
    g_sw = Solver{};
}

// The plan of this frame (update_scene_ubo, before the passes are recorded): the window and the
// steps, so the scene block names the window the render picture will be in.
void VulkanEngine::water_solver_fill(SceneUBO& d) {
    Solver& W = g_sw;
    static const bool off = env_on("VIBE_NO_SOLVER"), no_parts = env_on("VIBE_NO_FOAM_PARTICLES");
    static const int fixed = std::getenv("VIBE_SW_STEPS") ? std::clamp(std::atoi(std::getenv("VIBE_SW_STEPS")), 0, 8) : -1;
    const bool on = W.ok && !off && rt_.water_enabled && globe::current().home && terrain_height_full_view_ != VK_NULL_HANDLE;
    W.steps = 0;
    if (on) {
        const glm::vec3 eye = current_frame_view_.eye_pos;
        const glm::ivec2 corner = sw::corner_cells(glm::vec2(eye.x, eye.z));
        const double now = glfwGetTime();
        const double dt = W.clock < 0.0 ? 0.0 : now - W.clock;
        W.clock = now;
        int steps = fixed >= 0 ? fixed : sw::take_steps(W.acc, dt);
        const glm::ivec2 mv = corner - W.corner;
        if (!W.have_state || std::abs(mv.x) > int(kN) / 4 || std::abs(mv.y) > int(kN) / 4) {
            // (the first frame, or a jump: everything new, and a second and a half of water first)
            W.reset_all = true;
            W.warm = kWarmSteps;
            steps = std::max(steps, 1);
        }
        if (steps == 0 && mv != glm::ivec2(0)) { steps = 1; W.acc -= sw::kStep; }     // (the window moves only with a step)
        if (steps > 0) {
            W.shift = W.have_state ? mv : glm::ivec2(0);
            W.corner = corner;
            W.have_state = true;
        }
        W.steps = steps;
    } else if (W.have_state) {
        W.have_state = false;           // (switched off or away: a fresh start when it comes back)
    }
    d.water_sw = on && W.have_state ? glm::vec4(glm::vec2(W.corner) * kCell, float(kN) * kCell, 1.0f) : glm::vec4(0.0f);
    static const bool dbg = env_on("VIBE_DEBUG_SOLVER");
    d.water_sw2 = glm::vec4(on && !no_parts && W.draw ? 1.0f : 0.0f, dbg ? 1.0f : 0.0f, 0.0f, 0.0f);
}

glm::vec2 VulkanEngine::water_flow_at(float x, float z, float ring) const {
    const Solver& W = g_sw;
    const glm::vec2 base = water_river_flow(x, z);
    if (!W.cpu.valid || !W.have_state) return base;
    float h = 0.0f, fade = 0.0f;
    glm::vec2 v(0.0f);
    if (ring <= 0.0f) {
        if (!W.cpu.sample(x, z, h, v, &fade)) return base;
        return glm::mix(base, v, fade);
    }
    // (the mean round a ring: the free stream round a body, not the water it drags along itself)
    glm::vec2 sum(0.0f);
    float fs = 0.0f;
    int n = 0;
    for (int k = 0; k < 8; ++k) {
        const float a = float(k) * 0.785398f;
        glm::vec2 vk(0.0f);
        float fk = 0.0f;
        if (!W.cpu.sample(x + ring * std::cos(a), z + ring * std::sin(a), h, vk, &fk)) continue;
        sum += vk; fs += fk; ++n;
    }
    if (n < 3) return base;
    return glm::mix(base, sum / float(n), fs / float(n));
}

bool VulkanEngine::water_solver_drawing() const { return g_sw.parts_drawn; }

void VulkanEngine::water_solver_update(VkCommandBuffer cmd) {
    Solver& W = g_sw;
    if (!W.ok) return;
    static const bool no_parts = env_on("VIBE_NO_FOAM_PARTICLES"), log_on = env_on("VIBE_LOG_SOLVER");
    // ---- the CPU's copy of the read-back two frames old (written every other frame)
    {
        int best = -1;
        for (uint32_t s = 0; s < kSlots; ++s)
            if (W.rb_valid[s] && frame_number_ >= W.rb_frame[s] + 2 && (best < 0 || W.rb_frame[s] > W.rb_frame[size_t(best)])) best = int(s);
        if (best >= 0 && (W.cpu.data.empty() || W.rb_frame[size_t(best)] != 0)) {
            const uint32_t s = uint32_t(best);
            vmaInvalidateAllocation(allocator_, W.rb[s].alloc, 0, VK_WHOLE_SIZE);
            const uint32_t* src = static_cast<const uint32_t*>(W.rb[s].ptr);
            W.cpu.data.assign(src, src + size_t(kN) * kN * 2);
            std::memcpy(W.stat, src + size_t(kN) * kN * 2 + 6, sizeof(W.stat));
            W.cpu.corner = W.rb_corner[s];
            W.cpu.valid = W.have_state;
            W.rb_valid[s] = false;          // (taken)
        }
        if (!W.have_state) W.cpu.valid = false;
    }
    W.parts_drawn = false;
    if (W.steps <= 0 || !W.have_state) return;
    if (W.set_hview != terrain_height_full_view_) {
        VkDescriptorImageInfo hi{ terrain_height_full_sampler_, terrain_height_full_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = W.set; w.dstBinding = 8; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &hi;
        // (the set is used by this frame's commands only after this point; the previous frame's are done
        //  with it when the view changes - the ground is baked again between frames)
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
        W.set_hview = terrain_height_full_view_;
    }
    const glm::vec3 eye = current_frame_view_.eye_pos;
    const float L = float(kN) * kCell;
    const glm::vec2 org = glm::vec2(W.corner) * kCell;
    static Params P;                 // (20 KB: not on the stack)
    std::memset(&P, 0, sizeof(P));
    P.grid = glm::vec4(org, kCell, float(kN));

    // ---- the stones, rocks and piers in the window (the nearest first)
    {
        static std::vector<glm::vec4> obs;
        obs.clear();
        water_bed_obstacles(org + glm::vec2(0.5f * L), 0.5f * L + 2.0f, obs);
        const glm::vec2 e2(eye.x, eye.z);
        if (obs.size() > size_t(kObs))
            std::partial_sort(obs.begin(), obs.begin() + kObs, obs.end(), [&](const glm::vec4& a, const glm::vec4& b) {
                return glm::distance(glm::vec2(a), e2) < glm::distance(glm::vec2(b), e2); });
        const int no = int(std::min<size_t>(obs.size(), size_t(kObs)));
        for (int i = 0; i < no; ++i) P.obs[i] = obs[size_t(i)];
        P.misc.x = float(no);
        W.last_obs = no;
    }
    // ---- bodies: the player, cars, the world's people; new splashes
    {
        const float fdt = std::max(1e-3f, float(W.steps) * sw::kStep);
        int nb = 0;
        auto body = [&](glm::vec3 p, glm::vec3 vel, float radius, float bottom, float top) {
            if (nb >= kBodies) return;
            if (p.x < org.x || p.z < org.y || p.x > org.x + L || p.z > org.y + L) return;
            if (glm::dot(vel, vel) < 0.01f) return;
            P.body_a[nb] = glm::vec4(p.x, p.z, radius, bottom);
            P.body_b[nb] = glm::vec4(vel, top);
            ++nb;
        };
        if (!body_seated_) body(player_.position, player_.velocity, 0.35f, player_.position.y - 0.9f, player_.position.y + 0.9f);
        std::vector<glm::vec3> cars;
        jeep_positions(cars);
        for (size_t i = 0; i < cars.size(); ++i)
            if (W.cars_prev.size() == cars.size()) {
                const glm::vec3 vel = (cars[i] - W.cars_prev[i]) / std::max(1e-3f, float(W.steps) * sw::kStep);
                if (glm::dot(vel, vel) < 80.0f * 80.0f) body(cars[i], vel, 1.6f, cars[i].y - 0.4f, cars[i].y + 1.2f);
            }
        W.cars_prev = cars;
        for (const npc::View& v : npc_views_) {
            if (v.hidden || nb >= kBodies) continue;
            const bool car = v.kind == npc::Kind::Car;
            body(v.pos, glm::vec3(std::sin(v.yaw), 0.0f, -std::cos(v.yaw)) * v.speed, car ? 1.6f : 0.35f, v.pos.y - (car ? 0.4f : 0.9f), v.pos.y + (car ? 1.2f : 0.9f));
        }
        // VIBE_TEST_SW_WADER=1: a made-up wader walking a 2 m circle 6 m in front of the camera at 1.2 m/s
        static const bool test_wader = env_on("VIBE_TEST_SW_WADER");
        if (test_wader) {
            const glm::mat4& vm = current_frame_view_.view;
            glm::vec2 fw(-vm[0][2], -vm[2][2]);
            fw = glm::dot(fw, fw) > 1e-6f ? glm::normalize(fw) : glm::vec2(0.0f, -1.0f);
            const glm::vec2 c = glm::vec2(eye.x, eye.z) + fw * 6.0f;
            const float a = float(glfwGetTime()) * 0.6f;
            const glm::vec2 p = c + 2.0f * glm::vec2(std::cos(a), std::sin(a));
            const float ws = water_surface_y(p.x, p.y);
            body(glm::vec3(p.x, ws + 0.3f, p.y), 1.2f * glm::vec3(-std::sin(a), 0.0f, std::cos(a)), 0.35f, ws - 0.6f, ws + 1.2f);
        }
        // a new splash ring (bullets, bodies, droplets): a crater pushed out once
        for (const auto& r : water_ripples_)
            if (r.age <= fdt + 0.02f && nb < kBodies) {
                const float s = glm::clamp(r.amp * 28.5f, 1.0f, 6.0f);
                P.body_a[nb] = glm::vec4(r.xz, 0.15f * std::sqrt(s) + 0.1f, 0.0f);
                P.body_b[nb] = glm::vec4(0.0f, 0.0f, 0.0f, -std::min(0.6f * r.amp, 0.06f));
                ++nb;
            }
        P.misc.y = float(nb);
        W.last_bodies = nb;
    }
    // ---- the course's current on the 32 x 32 grid (world-aligned 3.2 m; a quarter refreshed a frame,
    //      the new cells first when the grid moves), and the sea flag
    {
        const glm::ivec2 fc = glm::ivec2(glm::floor((org + glm::vec2(0.5f * L)) / kFlowCell)) - glm::ivec2(kFlowN / 2);
        const size_t nn = size_t(kFlowN) * kFlowN;
        if (W.flow.size() != nn) { W.flow.assign(nn, glm::vec4(0.0f)); W.flow_fresh.assign(nn, 0); W.flow_corner = glm::ivec2(INT32_MIN); }
        if (fc != W.flow_corner) {
            std::vector<glm::vec4> nf(nn, glm::vec4(0.0f));
            std::vector<uint8_t> nfr(nn, 0);
            if (W.flow_corner.x != INT32_MIN)
                for (int j = 0; j < kFlowN; ++j)
                    for (int i = 0; i < kFlowN; ++i) {
                        const glm::ivec2 o = glm::ivec2(i, j) + fc - W.flow_corner;
                        if (o.x < 0 || o.y < 0 || o.x >= kFlowN || o.y >= kFlowN) continue;
                        nf[size_t(j * kFlowN + i)] = W.flow[size_t(o.y * kFlowN + o.x)];
                        nfr[size_t(j * kFlowN + i)] = W.flow_fresh[size_t(o.y * kFlowN + o.x)];
                    }
            W.flow.swap(nf); W.flow_fresh.swap(nfr);
            W.flow_corner = fc;
        }
        auto eval = [&](size_t k) {
            const int i = int(k % size_t(kFlowN)), j = int(k / size_t(kFlowN));
            const glm::vec2 p = (glm::vec2(W.flow_corner + glm::ivec2(i, j)) + 0.5f) * kFlowCell;
            const glm::vec2 f = water_course_flow(p.x, p.y);
            W.flow[k] = glm::vec4(f, water_sea_weight(p.x, p.y) > 0.5f ? 1.0f : 0.0f, 0.0f);
            W.flow_fresh[k] = 1;
        };
        int budget = W.reset_all ? int(nn) : 256;
        for (size_t k = 0; k < nn && budget > 0; ++k) if (!W.flow_fresh[k]) { eval(k); --budget; }
        for (int b = 0; b < 64; ++b) { eval(size_t(W.flow_cursor)); W.flow_cursor = (W.flow_cursor + 1) % int(nn); }
        std::memcpy(P.flow, W.flow.data(), nn * sizeof(glm::vec4));
        P.flow_org = glm::vec4(glm::vec2(W.flow_corner) * kFlowCell, kFlowCell, float(kFlowN));
    }
    const float now = float(std::fmod(glfwGetTime(), 4096.0));
    const bool parts_on = !no_parts && W.draw != VK_NULL_HANDLE;
    P.misc.z = now;
    P.misc.w = parts_on ? 1.0f : 0.0f;
    { const float r = std::sqrt(std::abs(now)) * 3734.4f; P.part = glm::vec4(float(kMaxPart), r - std::floor(r), W.warm > 0 ? 0.0f : 1.0f, log_on ? 1.0f : 0.0f); }

    // ---- the passes
    barrier_all(cmd);
    vkCmdUpdateBuffer(cmd, W.params.buf, 0, sizeof(P), &P);
    if (log_on) vkCmdFillBuffer(cmd, W.counters.buf, 24, 32, 0u);
    barrier_all(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, W.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, W.layout, 0, 1, &W.set, 0, nullptr);
    auto run = [&](int mode, int step, int flags, glm::ivec2 sh, uint32_t gx, uint32_t gy) {
        Push p{};
        p.m = glm::ivec4(mode, step, W.cur, flags);
        p.s = glm::ivec4(sh, 0, 0);
        vkCmdPushConstants(cmd, W.layout, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, gx, gy, 1);
        barrier_all(cmd);
    };
    const uint32_t g = kN / 16;
    run(0, 0, 0, glm::ivec2(0), g, g);                              // the inputs
    const int total = W.steps + W.warm;
    for (int k = 0; k < total; ++k) {
        // (the list the particles go to this step starts empty)
        vkCmdFillBuffer(cmd, W.counters.buf, VkDeviceSize(1 - W.cur) * 4, 4, 0u);
        barrier_all(cmd);
        const bool first = k == 0;
        run(1, first ? 0 : 1, first && W.reset_all ? 1 : 0, first ? W.shift : glm::ivec2(0), g, g);     // height
        run(2, first ? 0 : 1, 0, glm::ivec2(0), g, g);                                                     // pressure, foam, bodies
        if (parts_on) run(3, 0, 0, glm::ivec2(0), kMaxPart / 256, 1);                                      // the particles
        W.cur = 1 - W.cur;
    }
    W.warm = 0;
    W.reset_all = false;
    W.shift = glm::ivec2(0);
    if (parts_on) { run(5, 0, 0, glm::ivec2(0), 1, 1); W.parts_drawn = true; }
    // ---- the read-back (every other frame)
    if ((frame_number_ & 1) == 0) {
        const uint32_t slot = uint32_t((frame_number_ / 2) % kSlots);
        run(4, 0, 0, glm::ivec2(0), g, g);
        VkBufferCopy bc{ 0, 0, VkDeviceSize(kN) * kN * 8 };
        vkCmdCopyBuffer(cmd, W.pack.buf, W.rb[slot].buf, 1, &bc);
        VkBufferCopy cc{ 0, VkDeviceSize(kN) * kN * 8, 64 };
        vkCmdCopyBuffer(cmd, W.counters.buf, W.rb[slot].buf, 1, &cc);
        barrier_all(cmd);
        W.rb_frame[slot] = frame_number_;
        W.rb_corner[slot] = W.corner;
        W.rb_valid[slot] = true;
    }

    if (log_on && (W.log_t -= float(W.steps) * sw::kStep) <= 0.0f) {
        W.log_t = 1.0f;
        float h = 0.0f, fade = 0.0f;
        glm::vec2 v(0.0f);
        const glm::vec3 pp = player_.position;
        const bool got = W.cpu.sample(pp.x, pp.z, h, v, &fade);
        const glm::vec2 an = water_river_flow(pp.x, pp.z);
        log::infof("[solver] window (%.1f, %.1f), %d steps this frame, %d obstacles, %d bodies; at the player: %s h %.3f (authored %.3f), current (%.2f, %.2f) vs the analytic (%.2f, %.2f)",
                   double(org.x), double(org.y), W.steps, W.last_obs, W.last_bodies, got ? "water" : "no water/no read-back",
                   double(h), double(water_surface_y(pp.x, pp.z)), double(v.x), double(v.y), double(an.x), double(an.y));
        log::infof("[solver] foam rule: max O %.4f, max foam %.3f, %u cells with O > 0.004; max approach %.2f m/s, max rough x rise %.3f, max depth %.2f m",
                   double(W.stat[0]) * 1e-4, double(W.stat[1]) * 1e-4, W.stat[2], double(W.stat[3]) * 0.01, double(W.stat[4]) * 1e-4, double(W.stat[5]) * 0.01);
    }
}

// The foam particles, at the end of render_fx (the lit picture, its depth read-only).
void VulkanEngine::water_solver_draw(VkCommandBuffer cmd) {
    Solver& W = g_sw;
    if (!W.parts_drawn || !W.draw) return;
    Push p{};
    p.vp = current_frame_view_.vp;
    p.m = glm::ivec4(0, 0, W.cur, 0);
    p.f = glm::vec4(1.678f / std::max(1.0f, float(render_extent_.height)), float(kMaxPart), float(kN) * kCell, 0.0f);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, W.draw);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, W.layout, 0, 1, &W.set, 0, nullptr);
    vkCmdPushConstants(cmd, W.layout, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(p), &p);
    vkCmdDrawIndirect(cmd, W.counters.buf, 8, 1, sizeof(VkDrawIndirectCommand));
}

}  // namespace vibe
