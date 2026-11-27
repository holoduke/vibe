// Post effects after the reference game (docs/ARCHITECTURE.md "Post effects"):
//
//   - histogram exposure metering (exposure.comp; its pipeline lives in post.cpp) with the
//     reference's dark-scene curve and an indoor term from upward physics rays;
//   - the AgX look (day saturation, shadow gain, night look) - tonemap.glsl / post_fx.glsl;
//   - local tone mapping by exposure fusion (ltm_*.comp) -> a guided fit compose applies;
//   - glare sprites on very bright pixels, motion blur and depth of field, all into the post
//     colour P that compose then reads instead of the history (its own copies of compose's sets);
//   - the colour-blind view (compose.frag, a matrix at the end).
//
// One parameter block (PostFxGpu / post_fx.glsl PostFxData) carries every setting; its first
// vec4 is written by exposure.comp. All of this file's images stay in VK_IMAGE_LAYOUT_GENERAL.

#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/post_fx.h"
#include "engine/vk_engine/state/compose.h"
#include "engine/vk_engine/state/bloom.h"
#include "engine/vk_engine/state/fog.h"
#include "engine/vk_engine/state/atmosphere.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_pipelines.h"
#include "engine/physics_world.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace vibe {

namespace {

// Vibe's luminance -> the reference's metering units. Measured (VIBE_LOG_EXPOSURE=1) at the
// castle by day (26.6 4.0 123.8, 15:30, clear): metered avg 0.197, where the owner's look has
// exposure 1. Vibe's day / night range is small (the lamp-lit station at 23:00 meters 0.061,
// a third of the day), so the reference's units are placed with that day at the knee of its
// dark-scene curve (avg_r 0.01: below it the curve takes up to 3 stops off), and a key
// (kKey) brings the day back to exposure 1. Mapped 1 : 1 the nights were lifted 5x (grey).
constexpr float kDayAvg = 0.20f;
constexpr float kRefAtDay = 0.01f;
constexpr float kKey = kRefAtDay / 0.104f;
// The AgX look's scene luminance L: about 5 at that same view (the reference's day look runs
// over L 3 .. 10, its night look below 0.6).
constexpr float kLookLAtDay = 5.0f;
// The glare sprites' brightness per unit of glare_strength (0.4 until 2026-11-27: the owner found
// the glow round the station's and the cars' lights too much).
constexpr float kGlareGain = 0.28f;

// How far the sights are raised (0 .. 1): the depth of field follows it. VIBE_TEST_AIM=1 holds
// it at 1 for a capture.
#define pfx_aim() pfx_aim_of(aim_zoom_)
float pfx_aim_of(float z) {
    static const bool test = std::getenv("VIBE_TEST_AIM") != nullptr;
    return test ? 1.0f : glm::smoothstep(0.0f, 1.0f, z);
}

struct Res {
    VkDescriptorType type;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
};

}  // namespace

// ---------------------------------------------------------------------
// Settings: VIBE_PFX_<KEY> overrides (the persisted keys are in settings.cpp) and ranges.
// ---------------------------------------------------------------------
void VulkanEngine::post_fx_settings_env() {
    auto envf = [](const char* k, float& v) { if (const char* e = std::getenv(k)) v = float(std::atof(e)); };
    auto envi = [](const char* k, int& v) { if (const char* e = std::getenv(k)) v = std::atoi(e); };
    auto envb = [](const char* k, bool& v) { if (const char* e = std::getenv(k)) v = std::atoi(e) != 0; };
    envi("VIBE_PFX_AGX_LOOK", rt_.agx_look);
    envf("VIBE_PFX_EXPOSURE_COMP_EV", rt_.exposure_comp_ev);
    envf("VIBE_PFX_EXPOSURE_INDOOR", rt_.exposure_indoor);
    envf("VIBE_PFX_EXPOSURE_SPEED_UP", rt_.exposure_speed_up);
    envf("VIBE_PFX_EXPOSURE_SPEED_DOWN", rt_.exposure_speed_down);
    envf("VIBE_PFX_AUTO_EXPOSURE", rt_.auto_exposure_strength);
    envi("VIBE_PFX_TONEMAP", rt_.tonemap_mode);
    envf("VIBE_PFX_LTM_STRENGTH", rt_.ltm_strength);
    envf("VIBE_PFX_GLARE_STRENGTH", rt_.glare_strength);
    envf("VIBE_PFX_MOTION_BLUR", rt_.motion_blur);
    envb("VIBE_PFX_MOTION_BLUR_CAMERA", rt_.motion_blur_camera);
    envf("VIBE_PFX_DOF_STRENGTH", rt_.dof_strength);
    envi("VIBE_PFX_COLORBLIND_MODE", rt_.colorblind_mode);
    envb("VIBE_PFX_PARTICLE_FOG_LIGHT", rt_.particle_fog_light);
    envb("VIBE_PFX_SNOW_TRACKS", rt_.snow_tracks);
    rt_.agx_look = std::clamp(rt_.agx_look, 0, 1);
    rt_.exposure_comp_ev = std::clamp(rt_.exposure_comp_ev, -4.0f, 4.0f);
    rt_.exposure_indoor = std::clamp(rt_.exposure_indoor, 0.0f, 1.0f);
    rt_.exposure_speed_up = std::clamp(rt_.exposure_speed_up, 0.05f, 20.0f);
    rt_.exposure_speed_down = std::clamp(rt_.exposure_speed_down, 0.05f, 20.0f);
    rt_.ltm_strength = std::clamp(rt_.ltm_strength, 0.0f, 1.0f);
    rt_.glare_strength = std::clamp(rt_.glare_strength, 0.0f, 4.0f);
    rt_.motion_blur = std::clamp(rt_.motion_blur, 0.0f, 2.0f);
    rt_.dof_strength = std::clamp(rt_.dof_strength, 0.0f, 2.0f);
    rt_.colorblind_mode = std::clamp(rt_.colorblind_mode, 0, 3);
    if (const char* e = std::getenv("VIBE_LOG_EXPOSURE")) pfx_->log_every = std::max(1, std::atoi(e) * 30);
}

// ---------------------------------------------------------------------
// Init / teardown
// ---------------------------------------------------------------------
void VulkanEngine::init_post_fx() {
    PostFxState& P = *pfx_;
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(PostFxGpu);
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &P.buf, &P.buf_alloc, nullptr), "post_fx buffer");
        // (a small host copy of the GPU-written header: VIBE_LOG_EXPOSURE)
        VkBufferCreateInfo rci = bci;
        rci.size = 48;
        rci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo rai{};
        rai.usage = VMA_MEMORY_USAGE_AUTO;
        rai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ri{};
        vk_check(vmaCreateBuffer(allocator_, &rci, &rai, &P.rb_buf, &P.rb_alloc, &ri), "post_fx readback");
        P.rb_ptr = ri.pMappedData;
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkCmdFillBuffer(cb, P.buf, 0, VK_WHOLE_SIZE, 0u);      // valid = 0: the first frame snaps
        });
    }
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 16.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &P.sampler), "post_fx sampler");
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        vk_check(vkCreateSampler(device_, &si, nullptr, &P.point), "post_fx point sampler");
    }
    {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          128 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         96 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 96; pci.poolSizeCount = 3; pci.pPoolSizes = sizes;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &P.pool), "post_fx pool");
    }
    pfx_create_passes();
    pfx_create_targets();
    // The tracks and this block in the scene set (55, 59) right away: the ground's shaders read
    // them (foot_depth) and may run before the first frame (bakes at load). 56..58: the first frame.
    if (scene_desc_set_) {
        VkDescriptorBufferInfo bi{ P.buf, 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo ti{ P.sampler, P.trk_c.view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        for (auto& x : w) { x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; x.dstSet = scene_desc_set_; x.descriptorCount = 1; }
        w[0].dstBinding = 55; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &ti;
        w[1].dstBinding = 59; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &bi;
        if (P.trk_c.view) vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
    } else {
        log::warn("[post_fx] the scene set does not exist yet: tracks bound on the first frame");
    }
    log::info("[post_fx] histogram exposure, AgX look, local tone mapping, glare, motion blur, depth of field ready");
}

static void pfx_destroy_image(VkDevice device_, VmaAllocator allocator_, PfxImage& im) {
    for (auto& v : im.mip) { if (v) vkDestroyImageView(device_, v, nullptr); v = VK_NULL_HANDLE; }
    if (im.view) vkDestroyImageView(device_, im.view, nullptr);
    if (im.image) vmaDestroyImage(allocator_, im.image, im.alloc);
    im = PfxImage{};
}

static void pfx_make_image(VkDevice device_, VmaAllocator allocator_, PfxImage& im, VkExtent2D ext, uint32_t mips, VkFormat fmt, const char* name) {
    pfx_destroy_image(device_, allocator_, im);
    im.ext = { std::max(1u, ext.width), std::max(1u, ext.height) };
    im.mips = std::max(1u, std::min(mips, 8u));
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { im.ext.width, im.ext.height, 1 };
    ici.mipLevels = im.mips; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &im.image, &im.alloc, nullptr), name);
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = im.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, im.mips, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &im.view), name);
    for (uint32_t m = 0; m < im.mips; ++m) {
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &im.mip[m]), name);
    }
}

static PfxPass pfx_make_pass(VkDevice device_, VkPipelineCache pipeline_cache_, const char* spv, std::initializer_list<VkDescriptorType> types, uint32_t pc_size) {
    PfxPass p{};
    std::vector<VkDescriptorSetLayoutBinding> b(types.size());
    uint32_t i = 0;
    for (VkDescriptorType t : types) {
        b[i] = {};
        b[i].binding = i; b[i].descriptorType = t; b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        ++i;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = uint32_t(b.size()); lci.pBindings = b.data();
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &p.dsl), spv);
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, std::max(pc_size, 16u) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &p.dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &p.layout), spv);
    VkShaderModule m = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/" + spv);
    if (!m) { log::warnf("[post_fx] %s missing", spv); return p; }
    VkComputePipelineCreateInfo cci{};
    cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cci.stage.module = m;
    cci.stage.pName = "main";
    cci.layout = p.layout;
    vk_check(vkCreateComputePipelines(device_, pipeline_cache_, 1, &cci, nullptr, &p.pipe), spv);
    vkDestroyShaderModule(device_, m, nullptr);
    return p;
}

static void pfx_destroy_pass(VkDevice device_, PfxPass& p) {
    if (p.pipe) vkDestroyPipeline(device_, p.pipe, nullptr);
    if (p.layout) vkDestroyPipelineLayout(device_, p.layout, nullptr);
    if (p.dsl) vkDestroyDescriptorSetLayout(device_, p.dsl, nullptr);
    p = PfxPass{};
}

void VulkanEngine::pfx_create_passes() {
    PostFxState& P = *pfx_;
    const VkDescriptorType S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                           B = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    P.p_ltm_lum   = pfx_make_pass(device_, pipeline_cache_, "ltm_lum.comp.spv",   { S, S, B, I, I }, 16);
    P.p_ltm_down  = pfx_make_pass(device_, pipeline_cache_, "ltm_down.comp.spv",  { S, S, I, I }, 16);
    P.p_ltm_fuse  = pfx_make_pass(device_, pipeline_cache_, "ltm_fuse.comp.spv",  { S, S, S, I }, 16);
    P.p_ltm_guide = pfx_make_pass(device_, pipeline_cache_, "ltm_guide.comp.spv", { S, S, I }, 16);
    P.p_copy       = pfx_make_pass(device_, pipeline_cache_, "post_copy.comp.spv",   { S, I }, 16);
    P.p_glare_find = pfx_make_pass(device_, pipeline_cache_, "glare_find.comp.spv",  { S, S, B, B, I, S, S }, 16);
    P.p_mb_tile    = pfx_make_pass(device_, pipeline_cache_, "mblur_tile.comp.spv",  { S, S, B, I }, 16);
    P.p_mb_neigh   = pfx_make_pass(device_, pipeline_cache_, "mblur_neigh.comp.spv", { S, I }, 16);
    P.p_mb_gather  = pfx_make_pass(device_, pipeline_cache_, "mblur.comp.spv",       { S, S, S, S, B, I }, 16);
    P.p_dof_pre    = pfx_make_pass(device_, pipeline_cache_, "dof_pre.comp.spv",     { S, S, B, I }, 16);
    P.p_dof_gather = pfx_make_pass(device_, pipeline_cache_, "dof_gather.comp.spv",  { S, I, I }, 16);
    P.p_dof_comb   = pfx_make_pass(device_, pipeline_cache_, "dof_comb.comp.spv",    { S, S, S, I }, 16);
    // Tracks in snow and sand: two passes, the stamps' buffer, the two pictures (fixed size).
    P.p_trk_stamp = pfx_make_pass(device_, pipeline_cache_, "track_stamp.comp.spv", { S, I, B, B }, 16);
    P.p_trk_rim   = pfx_make_pass(device_, pipeline_cache_, "track_rim.comp.spv",   { S, I }, 16);
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 48 + 64 * 32;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &P.trk_buf, &P.trk_alloc, nullptr), "track stamps");
        const VkExtent2D te{ PostFxState::kTrkN, PostFxState::kTrkN };
        pfx_make_image(device_, allocator_, P.trk_c, te, 1, VK_FORMAT_R16G16_SFLOAT, "tracks");
        pfx_make_image(device_, allocator_, P.trk_tmp, te, 1, VK_FORMAT_R16_SFLOAT, "tracks depth");
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, P.trk_c.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            vkinit::transition_image(cb, P.trk_tmp.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            VkClearColorValue z{};
            VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, P.trk_c.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
            vkCmdClearColorImage(cb, P.trk_tmp.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
        });
    }
    // Glare sprites: the list (indirect draw arguments first, then up to 1000 sprites) and the
    // additive quad pipeline into the post colour.
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 16 + 1000 * 32;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &P.sprite_buf, &P.sprite_alloc, nullptr), "glare sprites");
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = B; b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 1; lci.pBindings = &b;
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &P.glare_draw_dsl), "glare dsl");
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, 16 };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &P.glare_draw_dsl;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &P.glare_draw_layout), "glare layout");
        const std::string sd = VIBE_SHADER_DIR;
        P.glare_vert = vkpipe::load_shader_module(device_, sd + "/glare.vert.spv");
        P.glare_frag = vkpipe::load_shader_module(device_, sd + "/glare.frag.spv");
        if (P.glare_vert && P.glare_frag) {
            vkpipe::GraphicsPipelineConfig g{};
            g.vert = P.glare_vert; g.frag = P.glare_frag; g.layout = P.glare_draw_layout;
            g.color_format = VK_FORMAT_R16G16B16A16_SFLOAT;
            g.depth_format = VK_FORMAT_UNDEFINED;
            g.cull = VK_CULL_MODE_NONE;
            g.depth_test = false; g.depth_write = false;
            g.premultiplied_blend = true;      // (alpha 0 out: additive, the colour's alpha kept)
            P.glare_draw_pipe = vkpipe::build_graphics_pipeline(device_, g);
        }
    }
}

// The targets that depend on the output size. Called at init and when the size changes.
void VulkanEngine::pfx_create_targets() {
    PostFxState& P = *pfx_;
    P.out_ext = swapchain_extent_;
    // Local tone mapping fit (compose binding 11); 1/8 of the output.
    VkExtent2D l0{ std::max(1u, (P.out_ext.width + 7) / 8), std::max(1u, (P.out_ext.height + 7) / 8) };
    pfx_make_image(device_, allocator_, P.ltm_c, l0, 1, VK_FORMAT_R16G16_SFLOAT, "ltm fit");
    pfx_make_image(device_, allocator_, P.ltm_a, l0, PostFxState::kLtmLevels, VK_FORMAT_R16G16B16A16_SFLOAT, "ltm lumas");
    pfx_make_image(device_, allocator_, P.ltm_w, l0, PostFxState::kLtmLevels, VK_FORMAT_R16G16B16A16_SFLOAT, "ltm weights");
    pfx_make_image(device_, allocator_, P.ltm_f, l0, PostFxState::kLtmLevels, VK_FORMAT_R16_SFLOAT, "ltm fused");
    // The post colour (ping-pong), the glare hold per 8x8 block, the motion blur tiles (16 px),
    // the depth of field at half size.
    const VkFormat F4 = VK_FORMAT_R16G16B16A16_SFLOAT;
    pfx_make_image(device_, allocator_, P.post[0], P.out_ext, 1, F4, "post colour 0");
    pfx_make_image(device_, allocator_, P.post[1], P.out_ext, 1, F4, "post colour 1");
    pfx_make_image(device_, allocator_, P.glare_hold, l0, 1, VK_FORMAT_R16_SFLOAT, "glare hold");
    const VkExtent2D tiles{ (P.out_ext.width + 15) / 16, (P.out_ext.height + 15) / 16 };
    pfx_make_image(device_, allocator_, P.mb_tile, tiles, 1, VK_FORMAT_R16G16_SFLOAT, "motion blur tiles");
    pfx_make_image(device_, allocator_, P.mb_neigh, tiles, 1, VK_FORMAT_R16G16_SFLOAT, "motion blur neighbours");
    const VkExtent2D half{ std::max(1u, P.out_ext.width / 2), std::max(1u, P.out_ext.height / 2) };
    pfx_make_image(device_, allocator_, P.dof_half, half, 1, F4, "dof half");
    pfx_make_image(device_, allocator_, P.dof_blur, half, 1, F4, "dof far");
    pfx_make_image(device_, allocator_, P.dof_near, half, 1, F4, "dof near");
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        for (PfxImage* im : { &P.ltm_c, &P.ltm_a, &P.ltm_w, &P.ltm_f, &P.post[0], &P.post[1], &P.glare_hold,
                              &P.mb_tile, &P.mb_neigh, &P.dof_half, &P.dof_blur, &P.dof_near })
            if (im->image) {
                vkinit::transition_image(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            }
        VkClearColorValue ident{};
        ident.float32[0] = 1.0f;   // fused = 1 * luma + 0: gain 1
        VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, P.ltm_c.image, VK_IMAGE_LAYOUT_GENERAL, &ident, 1, &rr);
        VkClearColorValue z{};
        vkCmdClearColorImage(cb, P.glare_hold.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
    });
}

void VulkanEngine::pfx_destroy_targets() {
    PostFxState& P = *pfx_;
    for (PfxImage* im : { &P.ltm_a, &P.ltm_w, &P.ltm_f, &P.ltm_c, &P.post[0], &P.post[1], &P.glare_hold,
                          &P.mb_tile, &P.mb_neigh, &P.dof_half, &P.dof_blur, &P.dof_near })
        pfx_destroy_image(device_, allocator_, *im);
}

void VulkanEngine::destroy_post_fx() {
    if (!pfx_ || !pfx_->buf) return;
    PostFxState& P = *pfx_;
    pfx_destroy_targets();
    for (PfxPass* p : { &P.p_ltm_lum, &P.p_ltm_down, &P.p_ltm_fuse, &P.p_ltm_guide, &P.p_copy, &P.p_glare_find,
                        &P.p_mb_tile, &P.p_mb_neigh, &P.p_mb_gather, &P.p_dof_pre, &P.p_dof_gather, &P.p_dof_comb, &P.p_trk_stamp, &P.p_trk_rim })
        pfx_destroy_pass(device_, *p);
    if (P.glare_draw_pipe) vkDestroyPipeline(device_, P.glare_draw_pipe, nullptr);
    if (P.glare_draw_layout) vkDestroyPipelineLayout(device_, P.glare_draw_layout, nullptr);
    if (P.glare_draw_dsl) vkDestroyDescriptorSetLayout(device_, P.glare_draw_dsl, nullptr);
    if (P.sprite_buf) vmaDestroyBuffer(allocator_, P.sprite_buf, P.sprite_alloc);
    if (P.trk_buf) vmaDestroyBuffer(allocator_, P.trk_buf, P.trk_alloc);
    pfx_destroy_image(device_, allocator_, P.trk_c);
    pfx_destroy_image(device_, allocator_, P.trk_tmp);
    if (P.glare_vert) vkDestroyShaderModule(device_, P.glare_vert, nullptr);
    if (P.glare_frag) vkDestroyShaderModule(device_, P.glare_frag, nullptr);
    if (P.pool) vkDestroyDescriptorPool(device_, P.pool, nullptr);
    if (P.sampler) vkDestroySampler(device_, P.sampler, nullptr);
    if (P.point) vkDestroySampler(device_, P.point, nullptr);
    if (P.rb_buf) vmaDestroyBuffer(allocator_, P.rb_buf, P.rb_alloc);
    vmaDestroyBuffer(allocator_, P.buf, P.buf_alloc);
    *pfx_ = PostFxState{};
}

// ---------------------------------------------------------------------
// Descriptor sets
// ---------------------------------------------------------------------
static VkDescriptorSet pfx_alloc_set(VkDevice device_, VkDescriptorPool pool, VkDescriptorSetLayout dsl) {
    VkDescriptorSet s = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    vk_check(vkAllocateDescriptorSets(device_, &dai, &s), "post_fx set");
    return s;
}

// Writes the compose / exposure bindings this file owns, and (re)builds this file's own sets.
void VulkanEngine::post_fx_bind_compose(int slot, VkImageView hist_src) {
    PostFxState& P = *pfx_;
    if (!P.buf) return;
    P.hist_src[slot] = hist_src;
    VkDescriptorBufferInfo bi{ P.buf, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo ci{ P.sampler, P.ltm_c.view, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w[3]{};
    for (auto& x : w) { x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; x.descriptorCount = 1; }
    w[0].dstSet = compose_->desc_sets[slot]; w[0].dstBinding = 10;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi;
    w[1].dstSet = compose_->desc_sets[slot]; w[1].dstBinding = 11;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &ci;
    w[2].dstSet = exposure_sets_[slot]; w[2].dstBinding = 3;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[2].pBufferInfo = &bi;
    vkUpdateDescriptorSets(device_, exposure_sets_[slot] ? 3 : 2, w, 0, nullptr);
    if (slot == kHistorySlots - 1) pfx_write_sets();   // (both compose sets are complete now)
}

VkDescriptorSet VulkanEngine::post_fx_compose_set(int slot) const {
    if (pfx_ && pfx_->post_used && pfx_->compose_sets[slot]) return pfx_->compose_sets[slot];
    return compose_->desc_sets[slot];
}

// ---------------------------------------------------------------------
// Per frame
// ---------------------------------------------------------------------
void VulkanEngine::post_fx_prepare() {
    PostFxState& P = *pfx_;
    if (!P.buf) return;
    const double now = glfwGetTime();
    const float dt = P.last_t > 0.0 ? std::min(float(now - P.last_t), 0.25f) : 0.0f;
    // How enclosed the camera is: thirteen rays up into the hemisphere against the physics
    // world (buildings, rock, terrain - not moving bodies: the car's own roof or a crate does
    // not make it "indoors"). Weighted toward the zenith; within 30 m.
    if (physics_ && rt_.exposure_indoor > 0.0f) {
        const glm::vec3 eye = current_frame_view_.eye_pos;
        float hit_w = 0.0f, tot_w = 0.0f;
        auto ray = [&](glm::vec3 d, float w) {
            PhysicsWorld::RayHit h = physics_->raycast(eye + d * 0.3f, d, 30.0f);
            tot_w += w;
            if (h.hit && !h.dynamic) hit_w += w;
        };
        ray(glm::vec3(0.0f, 1.0f, 0.0f), 2.0f);
        for (int i = 0; i < 6; ++i) {
            const float a = float(i) * 1.0471976f;
            const float a2 = a + 0.5235988f;
            ray(glm::vec3(std::cos(a) * 0.5f, 0.8660254f, std::sin(a) * 0.5f), 1.0f);              // 60 deg up
            ray(glm::vec3(std::cos(a2) * 0.8191520f, 0.5735764f, std::sin(a2) * 0.8191520f), 0.5f); // 35 deg up
        }
        float raw = tot_w > 0.0f ? hit_w / tot_w : 0.0f;
        // (inside a registered roofed box - a house, the pavilion's hall, the mart: indoors,
        //  whatever the rays found; scene_sheltered's boxes)
        for (const auto* list : { &shelters_, &site_shelters_ })
            for (const Shelter& s : *list)
                if (eye.x > s.lo.x && eye.x < s.hi.x && eye.z > s.lo.y && eye.z < s.hi.y && eye.y < s.top) raw = 1.0f;
        P.enclosed += (raw - P.enclosed) * (1.0f - std::exp(-dt * 2.0f));
    } else {
        P.enclosed = 0.0f;
    }
    P.last_t = now;
    // The fog volumes and this block for the FX shaders (scene set 56..59, fx_fog.glsl): once,
    // as soon as the fog's images exist (with the aerial volume standing in where they do not).
    if (!P.fog_bound_any && scene_desc_set_ && atmo_ && atmo_->ap.view) {
        wait_gpu_frames_idle();   // (the scene set may be in use by a frame in flight)
        const VkImageView fb = atmo_->ap.view;
        VkDescriptorImageInfo ii[3] = {
            { P.sampler, fog_->integ.view ? fog_->integ.view : fb, VK_IMAGE_LAYOUT_GENERAL },
            { P.sampler, fog_->inject.view ? fog_->inject.view : fb, VK_IMAGE_LAYOUT_GENERAL },
            { P.sampler, fog_->inject_b.view ? fog_->inject_b.view : fb, VK_IMAGE_LAYOUT_GENERAL },
        };
        VkDescriptorBufferInfo bi{ P.buf, 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo ti{ P.sampler, P.trk_c.view, VK_IMAGE_LAYOUT_GENERAL };   // 55: the tracks
        VkWriteDescriptorSet w[5]{};
        for (int i = 0; i < 5; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = scene_desc_set_; w[i].dstBinding = uint32_t(56 + i); w[i].descriptorCount = 1;
            w[i].descriptorType = (i == 3) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            if (i < 3) w[i].pImageInfo = &ii[i]; else w[i].pBufferInfo = &bi;
        }
        w[4].dstBinding = 55; w[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[4].pImageInfo = &ti; w[4].pBufferInfo = nullptr;
        vkUpdateDescriptorSets(device_, 5, w, 0, nullptr);
        P.fog_bound = fog_->integ.view && fog_->inject.view && fog_->inject_b.view;
        if (!P.fog_bound) log::warn("[post_fx] no fog volume: the FX keep their own lighting");
        P.fog_bound_any = true;
    }
    // The output size changed: remake the targets and every set that points at them.
    if (P.out_ext.width != swapchain_extent_.width || P.out_ext.height != swapchain_extent_.height) {
        wait_gpu_frames_idle();
        pfx_destroy_targets();
        pfx_create_targets();
        rewrite_compose_image_bindings();
    }
}

void VulkanEngine::post_fx_upload(VkCommandBuffer cmd) {
    PostFxState& P = *pfx_;
    if (!P.buf) return;
    // (the GPU-written header of the frame before, for the log)
    if (P.log_every > 0 && P.rb_ptr && (frame_number_ % uint64_t(P.log_every)) == 0) {
        float h[12];
        std::memcpy(h, P.rb_ptr, sizeof(h));
        uint32_t sprites = 0;
        std::memcpy(&sprites, static_cast<const char*>(P.rb_ptr) + 36, 4);
        log::infof("[exposure] avg %.5f (ref units %.5f)  target %.3f  exposure %.3f  enclosed %.2f  look L %.2f  focus %.3f m  glare sprites %u",
                   h[1], h[1] * (kRefAtDay / kDayAvg), h[2], h[0], P.enclosed, h[1] * (kLookLAtDay / kDayAvg), h[4], sprites);
    }
    PostFxGpu g{};
    const float dt = 1.0f / 60.0f;
    const float hour = rt_.time_of_day;
    {
        // The reference's night factor from the hour: 0.5 by day, 1 at night.
        const float q = std::pow(std::abs(hour / 8.57f - 1.4f), 8.0f);
        const float n = 1.0f / (1.0f + std::exp(-q));
        g.look = glm::vec4(rt_.tonemap_mode == 1 && rt_.agx_look ? 1.0f : 0.0f, n, kLookLAtDay / kDayAvg,
                           float(rt_.colorblind_mode));
    }
    const float indoor = glm::smoothstep(0.30f, 0.70f, P.enclosed) * rt_.exposure_indoor;
    g.expo = glm::vec4(rt_.auto_exposure_strength, std::exp2(rt_.exposure_comp_ev), indoor, kRefAtDay / kDayAvg);
    g.expo2 = glm::vec4(0.40f, 0.95f, rt_.exposure_speed_up, rt_.exposure_speed_down);
    {
        const glm::vec3 sky = rt_.sky_color;
        float sky_lum = 0.2126f * sky.r + 0.7152f * sky.g + 0.0722f * sky.b;
        if (skybox_.hdr && skybox_.avg_lum > 0.0f) sky_lum = skybox_.avg_lum;
        sky_lum = std::max(0.05f, sky_lum);
        const bool sky_in = (compose_->uses_dlss && deferred_lighting_active_) || (compose_->uses_fsr3 && sky_filled_this_frame_);
        const double now = glfwGetTime();
        static double prev = now;
        const float fdt = std::clamp(float(now - prev), 0.0f, 0.25f);
        prev = now;
        g.expo3 = glm::vec4(fdt > 0.0f ? fdt : dt, sky_lum, sky_in ? 1.0f : 0.0f, kKey);
    }
    // Local tone mapping: shadows +1.5 stops, highlights -1.5 stops, Gaussian sigma 25 (Mertens' 0.2).
    g.ltm = glm::vec4(rt_.ltm_strength, std::exp2(1.5f), std::exp2(-1.5f), 25.0f);
    g.ltm2 = glm::vec4(float(rt_.tonemap_mode), 0.0f, 0.0f, 1e-3f);
    {
        const float hk = float(swapchain_extent_.height) / 1080.0f;
        g.glare = glm::vec4(rt_.glare_strength * kGlareGain, 1.0f, 120.0f, float(frame_number_ % 4096u));
        g.glare2 = glm::vec4(float(swapchain_extent_.width), float(swapchain_extent_.height),
                             1.0f / float(swapchain_extent_.width), 1.0f / float(swapchain_extent_.height));
        // Motion blur: x strength (shutter share of a frame's motion), y longest blur (px),
        // z camera-turn blur, w the reference's 4-pixel dead zone (at 1080 lines).
        g.mb = glm::vec4(rt_.motion_blur, 48.0f * hk, rt_.motion_blur_camera ? 1.0f : 0.0f, 4.0f * hk);
        // Depth of field, while aiming down the sights: x amount, y largest radius (half-size
        // pixels; 3.5 at 1080 lines - subtle: 6 blurred the castle away), w the aperture term A of the reference's coc.
        const float aim = pfx_aim();
        g.dof = glm::vec4(aim * rt_.dof_strength, 3.5f * hk, 0.0f, 1.0f);
        const glm::mat4& pj = current_frame_view_.proj;
        glm::mat4 pu = pj;
        pu[2][0] -= current_frame_view_.jitter.x;     // (the motion vectors are unjittered)
        pu[2][1] -= current_frame_view_.jitter.y;
        const glm::mat4 rot_vp = pu * glm::mat4(glm::mat3(current_frame_view_.view));
        g.cur_rot_ivp = glm::inverse(rot_vp);
        g.prev_rot_vp = P.prev_valid ? P.prev_rot_vp : rot_vp;
        P.prev_rot_vp = rot_vp;
        P.prev_valid = true;
        g.proj = glm::vec4(pj[3][2], pj[2][2], 1.0f / float(render_extent_.width), 1.0f / float(render_extent_.height));
        // Smoke from the fog volume (fx_fog.glsl): what dispatch_fog injects with this frame -
        // the next frame's FX read this frame's volume.
        const bool fog_on = rt_.particle_fog_light && rt_.fog_volumetric && P.fog_bound;
        g.fog0 = glm::vec4(last_sun_rgb_ * rt_.sun_intensity * 3.14159265f, glm::clamp(rt_.fog_anisotropy, -0.9f, 0.95f));
        g.fog1 = glm::vec4(last_sky_rgb_ * rt_.ambient_strength * 0.8f, std::max(20.0f, rt_.fog_max_dist));
        g.fog2 = glm::vec4(current_frame_view_.eye_pos, fog_on ? 1.0f : 0.0f);
        g.fog3 = glm::vec4(glm::normalize(-glm::vec3(current_frame_view_.view[0][2], current_frame_view_.view[1][2],
                                                     current_frame_view_.view[2][2])), 0.0f);
        g.fog_vp = current_frame_view_.vp;
        {
            // (as frame.cpp makes compose's sun direction; at night it is the moon's: no disc sprite then)
            const float sp = glm::radians(rt_.sun_pitch_deg), sy = glm::radians(rt_.sun_yaw_deg);
            const glm::vec3 sd(std::cos(sp) * std::sin(sy), std::sin(sp), std::cos(sp) * std::cos(sy));
            const bool day = sd.y > 0.0f && !(env_ && env_->night > 0.5f);
            g.sun_d = glm::vec4(glm::normalize(sd), day ? 1.0f : 0.0f);
        }
    }
    // (the frame before read it in its compose / exposure passes: write after those)
    {
        VkMemoryBarrier2 wb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT };
        VkDependencyInfo wd{}; wd.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        wd.memoryBarrierCount = 1; wd.pMemoryBarriers = &wb;
        vkCmdPipelineBarrier2(cmd, &wd);
    }
    const size_t off = 32;   // (the GPU-written header stays)
    vkCmdUpdateBuffer(cmd, P.buf, off, sizeof(PostFxGpu) - off, reinterpret_cast<const char*>(&g) + off);
    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
    VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

// Points this file's own sets at the current targets / sources (after a resize or a rebind).
// The pool is reset and every set made again: they are few, and this runs only on a rebind.
void VulkanEngine::pfx_write_sets() {
    PostFxState& P = *pfx_;
    if (!P.pool || !P.ltm_a.image) return;
    vkResetDescriptorPool(device_, P.pool, 0);
    std::vector<VkWriteDescriptorSet> w;
    std::vector<VkDescriptorImageInfo> ii;
    std::vector<VkDescriptorBufferInfo> bi;
    ii.reserve(512); bi.reserve(64); w.reserve(512);   // (no reallocation: the writes point into these)
    auto tex = [&](VkDescriptorSet s, uint32_t b, VkImageView v, VkImageLayout lay = VK_IMAGE_LAYOUT_GENERAL, VkSampler smp = VK_NULL_HANDLE) {
        ii.push_back({ smp ? smp : P.sampler, v, lay });
        VkWriteDescriptorSet x{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        x.dstSet = s; x.dstBinding = b; x.descriptorCount = 1;
        x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; x.pImageInfo = &ii.back();
        w.push_back(x);
    };
    auto img = [&](VkDescriptorSet s, uint32_t b, VkImageView v) {
        ii.push_back({ VK_NULL_HANDLE, v, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet x{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        x.dstSet = s; x.dstBinding = b; x.descriptorCount = 1;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; x.pImageInfo = &ii.back();
        w.push_back(x);
    };
    auto buf = [&](VkDescriptorSet s, uint32_t b, VkBuffer bf) {
        bi.push_back({ bf, 0, VK_WHOLE_SIZE });
        VkWriteDescriptorSet x{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        x.dstSet = s; x.dstBinding = b; x.descriptorCount = 1;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; x.pBufferInfo = &bi.back();
        w.push_back(x);
    };
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const int L = PostFxState::kLtmLevels;
    // Local tone mapping.
    for (int s = 0; s < kHistorySlots; ++s) {
        if (!P.hist_src[s]) continue;
        P.s_ltm_lum[s] = pfx_alloc_set(device_, P.pool, P.p_ltm_lum.dsl);
        tex(P.s_ltm_lum[s], 0, P.hist_src[s], RO);
        tex(P.s_ltm_lum[s], 1, depth_view_, RO);
        buf(P.s_ltm_lum[s], 2, P.buf);
        img(P.s_ltm_lum[s], 3, P.ltm_a.mip[0]);
        img(P.s_ltm_lum[s], 4, P.ltm_w.mip[0]);
    }
    for (int l = 0; l + 1 < L; ++l) {
        VkDescriptorSet s = P.s_ltm_down[l] = pfx_alloc_set(device_, P.pool, P.p_ltm_down.dsl);
        tex(s, 0, P.ltm_a.view); tex(s, 1, P.ltm_w.view);
        img(s, 2, P.ltm_a.mip[l + 1]); img(s, 3, P.ltm_w.mip[l + 1]);
    }
    for (int l = 0; l < L; ++l) {
        VkDescriptorSet s = P.s_ltm_fuse[l] = pfx_alloc_set(device_, P.pool, P.p_ltm_fuse.dsl);
        tex(s, 0, P.ltm_a.view); tex(s, 1, P.ltm_w.view); tex(s, 2, P.ltm_f.view);
        img(s, 3, P.ltm_f.mip[l]);
    }
    P.s_ltm_guide = pfx_alloc_set(device_, P.pool, P.p_ltm_guide.dsl);
    tex(P.s_ltm_guide, 0, P.ltm_a.view, VK_IMAGE_LAYOUT_GENERAL, P.point);
    tex(P.s_ltm_guide, 1, P.ltm_f.view, VK_IMAGE_LAYOUT_GENERAL, P.point);
    img(P.s_ltm_guide, 2, P.ltm_c.mip[0]);
    // Compose's own sets, copied, with binding 0 (the history) swapped for the post colour.
    // (copies first: one vkUpdateDescriptorSets call does its writes before its copies)
    {
        std::vector<VkCopyDescriptorSet> cp;
        for (int s = 0; s < kHistorySlots; ++s) {
            if (!compose_->desc_sets[s]) continue;
            P.compose_sets[s] = pfx_alloc_set(device_, P.pool, compose_->desc_set_layout);
            for (uint32_t b = 0; b < 15; ++b) {      // (12 - 14: lens_fx.cpp - the lens water, the shimmer, its block)
                VkCopyDescriptorSet c{ VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET };
                c.srcSet = compose_->desc_sets[s]; c.srcBinding = b;
                c.dstSet = P.compose_sets[s]; c.dstBinding = b; c.descriptorCount = 1;
                cp.push_back(c);
            }
        }
        vkUpdateDescriptorSets(device_, 0, nullptr, uint32_t(cp.size()), cp.data());
        for (int s = 0; s < kHistorySlots; ++s)
            if (P.compose_sets[s]) tex(P.compose_sets[s], 0, P.post[0].view, VK_IMAGE_LAYOUT_GENERAL, linear_sampler_);
    }
    // The post colour's passes (point reads for motion vectors / depth / tiles).
    for (int s = 0; s < kHistorySlots; ++s) {
        if (!P.hist_src[s]) continue;
        VkDescriptorSet c = P.s_copy[s] = pfx_alloc_set(device_, P.pool, P.p_copy.dsl);
        tex(c, 0, P.hist_src[s], RO); img(c, 1, P.post[0].mip[0]);
        VkDescriptorSet g = P.s_glare_find[s] = pfx_alloc_set(device_, P.pool, P.p_glare_find.dsl);
        tex(g, 0, P.hist_src[s], RO); tex(g, 1, depth_view_, RO, P.point); buf(g, 2, P.buf); buf(g, 3, P.sprite_buf);
        tex(g, 5, gbuffer0_view_, RO, P.point); tex(g, 6, gbuffer1_view_, RO, P.point);   // (what each pixel is)
        img(g, 4, P.glare_hold.mip[0]);
        VkDescriptorSet d = P.s_dof_pre[s] = pfx_alloc_set(device_, P.pool, P.p_dof_pre.dsl);
        tex(d, 0, P.hist_src[s], RO); tex(d, 1, depth_view_, RO, P.point); buf(d, 2, P.buf); img(d, 3, P.dof_half.mip[0]);
        for (int dst = 0; dst < 2; ++dst) {
            VkDescriptorSet e = P.s_dof_comb[s][dst] = pfx_alloc_set(device_, P.pool, P.p_dof_comb.dsl);
            tex(e, 0, P.hist_src[s], RO); tex(e, 1, P.dof_blur.view); tex(e, 2, P.dof_near.view); img(e, 3, P.post[dst].mip[0]);
        }
    }
    for (int k = 0; k < 3; ++k) {
        const bool from_post = (k == 2);
        if (!from_post && !P.hist_src[k]) continue;
        VkDescriptorSet m = P.s_mb_gather[k] = pfx_alloc_set(device_, P.pool, P.p_mb_gather.dsl);
        if (from_post) tex(m, 0, P.post[1].view); else tex(m, 0, P.hist_src[k], RO);
        tex(m, 1, motion_vec_view_, RO, P.point); tex(m, 2, depth_view_, RO, P.point); tex(m, 3, P.mb_neigh.view, VK_IMAGE_LAYOUT_GENERAL, P.point);
        buf(m, 4, P.buf); img(m, 5, P.post[0].mip[0]);
    }
    P.s_mb_tile = pfx_alloc_set(device_, P.pool, P.p_mb_tile.dsl);
    tex(P.s_mb_tile, 0, motion_vec_view_, RO, P.point); tex(P.s_mb_tile, 1, depth_view_, RO, P.point);
    buf(P.s_mb_tile, 2, P.buf); img(P.s_mb_tile, 3, P.mb_tile.mip[0]);
    P.s_mb_neigh = pfx_alloc_set(device_, P.pool, P.p_mb_neigh.dsl);
    tex(P.s_mb_neigh, 0, P.mb_tile.view, VK_IMAGE_LAYOUT_GENERAL, P.point); img(P.s_mb_neigh, 1, P.mb_neigh.mip[0]);
    P.s_dof_gather = pfx_alloc_set(device_, P.pool, P.p_dof_gather.dsl);
    tex(P.s_dof_gather, 0, P.dof_half.view); img(P.s_dof_gather, 1, P.dof_blur.mip[0]); img(P.s_dof_gather, 2, P.dof_near.mip[0]);
    P.s_glare_draw = pfx_alloc_set(device_, P.pool, P.glare_draw_dsl);
    buf(P.s_glare_draw, 0, P.sprite_buf);
    if (P.p_trk_stamp.dsl && P.trk_c.view) {
        P.s_trk_stamp = pfx_alloc_set(device_, P.pool, P.p_trk_stamp.dsl);
        tex(P.s_trk_stamp, 0, P.trk_c.view, VK_IMAGE_LAYOUT_GENERAL, P.point); img(P.s_trk_stamp, 1, P.trk_tmp.mip[0]);
        buf(P.s_trk_stamp, 2, P.trk_buf); buf(P.s_trk_stamp, 3, P.buf);
        P.s_trk_rim = pfx_alloc_set(device_, P.pool, P.p_trk_rim.dsl);
        tex(P.s_trk_rim, 0, P.trk_tmp.view, VK_IMAGE_LAYOUT_GENERAL, P.point); img(P.s_trk_rim, 1, P.trk_c.mip[0]);
    }
    vkUpdateDescriptorSets(device_, uint32_t(w.size()), w.data(), 0, nullptr);
}

// A compute -> compute / fragment dependency on everything this file writes (one global barrier).
static void pfx_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                             VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT };
    VkDependencyInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

static void pfx_dispatch(VkCommandBuffer cmd, const PfxPass& p, VkDescriptorSet s, glm::vec4 pc, uint32_t w, uint32_t h) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &s, 0, nullptr);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
}

void VulkanEngine::post_fx_run(VkCommandBuffer cmd) {
    PostFxState& P = *pfx_;
    if (!P.buf) return;
    // (the frame before read the targets in its compose pass, and the glare list in its draw:
    // this frame's writes wait for those reads)
    {
        VkMemoryBarrier2 pb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                             VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT };
        VkDependencyInfo pd{}; pd.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        pd.memoryBarrierCount = 1; pd.pMemoryBarriers = &pb;
        vkCmdPipelineBarrier2(cmd, &pd);
    }
    // (for the log: the header exposure.comp just wrote, and the glare list's count of the frame before)
    if (P.log_every > 0 && P.rb_buf) {
        VkMemoryBarrier2 rb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT };
        VkDependencyInfo rd{}; rd.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        rd.memoryBarrierCount = 1; rd.pMemoryBarriers = &rb;
        vkCmdPipelineBarrier2(cmd, &rd);
        VkBufferCopy bc{ 0, 0, 32 };
        vkCmdCopyBuffer(cmd, P.buf, P.rb_buf, 1, &bc);
        VkBufferCopy sc{ 0, 32, 16 };
        if (P.sprite_buf) vkCmdCopyBuffer(cmd, P.sprite_buf, P.rb_buf, 1, &sc);
        VkMemoryBarrier2 ra{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT };
        rd.pMemoryBarriers = &ra;
        vkCmdPipelineBarrier2(cmd, &rd);
    }
    P.post_used = false;
    // ---- Local tone mapping ----
    if (rt_.ltm_strength > 0.0f && P.p_ltm_lum.pipe && P.s_ltm_lum[history_write_slot_]) {
        const int L = PostFxState::kLtmLevels;
        auto ext = [&](int l) { return VkExtent2D{ std::max(1u, P.ltm_a.ext.width >> l), std::max(1u, P.ltm_a.ext.height >> l) }; };
        const VkExtent2D e0 = ext(0);
        pfx_dispatch(cmd, P.p_ltm_lum, P.s_ltm_lum[history_write_slot_], glm::vec4(float(e0.width), float(e0.height), 0, 0), e0.width, e0.height);
        for (int l = 0; l + 1 < L; ++l) {
            pfx_barrier(cmd);
            const VkExtent2D e = ext(l + 1);
            pfx_dispatch(cmd, P.p_ltm_down, P.s_ltm_down[l], glm::vec4(float(l), 0, float(e.width), float(e.height)), e.width, e.height);
        }
        for (int l = L - 1; l >= 0; --l) {
            pfx_barrier(cmd);
            const VkExtent2D e = ext(l);
            pfx_dispatch(cmd, P.p_ltm_fuse, P.s_ltm_fuse[l], glm::vec4(float(l), l == L - 1 ? 1.0f : 0.0f, float(e.width), float(e.height)), e.width, e.height);
        }
        pfx_barrier(cmd);
        pfx_dispatch(cmd, P.p_ltm_guide, P.s_ltm_guide, glm::vec4(float(e0.width), float(e0.height), 1e-3f, 0), e0.width, e0.height);
        pfx_barrier(cmd);
        prof_mark(cmd, "post_ltm");
    }
    // ---- The post colour: depth of field, motion blur, glare sprites ----
    const int slot = history_write_slot_;
    const bool dof_on = rt_.dof_strength > 0.0f && pfx_aim() > 0.01f && P.p_dof_pre.pipe && P.s_dof_pre[slot];
    const bool mb_on = rt_.motion_blur > 0.0f && P.p_mb_gather.pipe && P.s_mb_gather[slot] && P.s_mb_gather[2];
    const bool glare_on = rt_.glare_strength > 0.0f && P.glare_draw_pipe && P.s_glare_find[slot];
    if (!(dof_on || mb_on || glare_on)) return;
    const VkExtent2D oe = P.out_ext;
    const float hk = float(oe.height) / 1080.0f;
    if (dof_on) {
        const VkExtent2D he = P.dof_half.ext;
        pfx_dispatch(cmd, P.p_dof_pre, P.s_dof_pre[slot], glm::vec4(float(he.width), float(he.height), 0, 0), he.width, he.height);
        pfx_barrier(cmd);
        const float R = 3.5f * hk * std::min(rt_.dof_strength, 2.0f) * pfx_aim();
        pfx_dispatch(cmd, P.p_dof_gather, P.s_dof_gather, glm::vec4(float(he.width), float(he.height), std::max(R, 1.0f), 0), he.width, he.height);
        pfx_barrier(cmd);
        pfx_dispatch(cmd, P.p_dof_comb, P.s_dof_comb[slot][mb_on ? 1 : 0], glm::vec4(float(oe.width), float(oe.height), 0, 0), oe.width, oe.height);
        pfx_barrier(cmd);
        prof_mark(cmd, "post_dof");
    }
    if (mb_on) {
        const VkExtent2D te = P.mb_tile.ext;
        pfx_dispatch(cmd, P.p_mb_tile, P.s_mb_tile, glm::vec4(float(te.width), float(te.height), float(oe.width), float(oe.height)), te.width, te.height);
        pfx_barrier(cmd);
        pfx_dispatch(cmd, P.p_mb_neigh, P.s_mb_neigh, glm::vec4(float(te.width), float(te.height), 0, 0), te.width, te.height);
        pfx_barrier(cmd);
        pfx_dispatch(cmd, P.p_mb_gather, P.s_mb_gather[dof_on ? 2 : slot], glm::vec4(float(oe.width), float(oe.height), 0, 0), oe.width, oe.height);
        pfx_barrier(cmd);
        prof_mark(cmd, "post_mblur");
    }
    if (!dof_on && !mb_on) {
        pfx_dispatch(cmd, P.p_copy, P.s_copy[slot], glm::vec4(float(oe.width), float(oe.height), 0, 0), oe.width, oe.height);
        pfx_barrier(cmd);
    }
    if (glare_on) {
        // (the frame before drew from the list: reset its header after that)
        {
            VkMemoryBarrier2 wb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                                 VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT };
            VkDependencyInfo wd{}; wd.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            wd.memoryBarrierCount = 1; wd.pMemoryBarriers = &wb;
            vkCmdPipelineBarrier2(cmd, &wd);
            const uint32_t hdr[4] = { 6u, 0u, 0u, 0u };
            vkCmdUpdateBuffer(cmd, P.sprite_buf, 0, sizeof(hdr), hdr);
            VkMemoryBarrier2 ub{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT };
            wd.pMemoryBarriers = &ub;
            vkCmdPipelineBarrier2(cmd, &wd);
        }
        const VkExtent2D be = P.glare_hold.ext;
        // (one workgroup per 8x8 block: every pixel is looked at)
        pfx_dispatch(cmd, P.p_glare_find, P.s_glare_find[slot], glm::vec4(float(be.width), float(be.height), float(oe.width), float(oe.height)), be.width * 8, be.height * 8);
        pfx_barrier(cmd);
        VkRenderingAttachmentInfo att{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        att.imageView = P.post[0].mip[0];
        att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        ri.renderArea = { { 0, 0 }, oe };
        ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &att;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{ 0.0f, 0.0f, float(oe.width), float(oe.height), 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ { 0, 0 }, oe };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, P.glare_draw_pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, P.glare_draw_layout, 0, 1, &P.s_glare_draw, 0, nullptr);
        const glm::vec4 gpc(float(oe.width), float(oe.height), rt_.glare_strength * kGlareGain, 0.0f);
        vkCmdPushConstants(cmd, P.glare_draw_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(gpc), &gpc);
        vkCmdDrawIndirect(cmd, P.sprite_buf, 0, 1, 16);
        vkCmdEndRendering(cmd);
        pfx_barrier(cmd);
        prof_mark(cmd, "post_glare");
    }
    P.post_used = true;
}

}  // namespace vibe

namespace vibe {

// ---------------------------------------------------------------------
// Tracks in snow and sand (track_stamp.comp, track_rim.comp; read by the ground shader through
// scene binding 55 and the window corner in the GPU-written header: terrain_rm_pebbles.glsl).
// The window follows the player in whole texels; stamps from grass_trail_update.
// ---------------------------------------------------------------------
void VulkanEngine::post_fx_tracks(VkCommandBuffer cmd, const std::vector<glm::vec4>& stamps, glm::vec3 eye) {
    PostFxState& P = *pfx_;
    if (!P.buf || !P.s_trk_stamp || !P.p_trk_stamp.pipe || !P.p_trk_rim.pipe) return;
    if (!rt_.snow_tracks) {
        if (!P.trk_have) return;
        P.trk_have = false;           // (switched off: the ground stops reading them)
        const float off[3] = { 0.0f, 0.0f, 0.0f };
        vkCmdUpdateBuffer(cmd, P.buf, 20, sizeof(off), off);
        return;
    }
    const float tex = PostFxState::kTrkTexel;
    const float half = 0.5f * float(PostFxState::kTrkN) * tex;
    const glm::vec2 prev = P.trk_have ? P.trk_org : glm::vec2(1e6f);
    P.trk_org = glm::floor((glm::vec2(eye.x, eye.z) - glm::vec2(half)) / tex) * tex;
    P.trk_have = true;
    // With no snow lying they fill again over half a minute (the reference never refills; its
    // snow comes and goes with the weather here).
    const float snow = env_ ? env_->snow : 0.0f;
    const float decay = snow > 0.05f ? 1.0f : 0.9995f;
    struct { glm::vec4 head, org, flags; glm::vec4 s[128]; } blk{};
    std::vector<glm::vec4> st = stamps;
    // VIBE_TEST_TRACKS=1: a made-up car drives a circle of 4.5 m, 9 m ahead of where the camera
    // looks, and a walker a smaller one - tracks to look at from a held camera.
    if (std::getenv("VIBE_TEST_TRACKS")) {
        const float t = anim_time_;
        glm::vec2 fwd(-current_frame_view_.view[0][2], -current_frame_view_.view[2][2]);
        fwd = glm::dot(fwd, fwd) > 1e-6f ? glm::normalize(fwd) : glm::vec2(0.0f, -1.0f);
        const glm::vec2 c = glm::vec2(eye.x, eye.z) + fwd * 9.0f;      // (the camera is held in a test)
        const float a = t * 0.9f;
        const glm::vec2 pos = c + 4.5f * glm::vec2(std::cos(a), std::sin(a));
        const glm::vec2 fw(-std::sin(a), std::cos(a)), rt(-fw.y, fw.x);
        const glm::vec2 back = fw * (4.5f * 0.9f / 60.0f);
        for (int k = 0; k < 4; ++k) {
            st.push_back(glm::vec4(pos + fw * ((k < 2) ? 1.3f : -1.3f) + rt * ((k & 1) ? 0.78f : -0.78f), back));
            st.push_back(glm::vec4(0.15f, 0.10f, 0.0f, 0.0f));
        }
        const float b = -t * 0.5f;
        st.push_back(glm::vec4(c + glm::vec2(4.0f, 2.0f) + 2.5f * glm::vec2(std::cos(b), std::sin(b)), glm::vec2(0.0f)));
        st.push_back(glm::vec4(0.20f, 0.05f, 0.0f, 0.0f));
    }
    const int n = int(std::min<size_t>(st.size() / 2, 64));
    blk.head = glm::vec4(float(n), decay, tex, float(PostFxState::kTrkN));
    blk.org = glm::vec4(P.trk_org, prev);
    for (int i = 0; i < 2 * n; ++i) blk.s[i] = st[size_t(i)];
    // (the frame before's ground pass read the tracks: write after it)
    {
        VkMemoryBarrier2 wb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_READ_BIT,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT };
        VkDependencyInfo wd{}; wd.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        wd.memoryBarrierCount = 1; wd.pMemoryBarriers = &wb;
        vkCmdPipelineBarrier2(cmd, &wd);
    }
    static const char* tt = std::getenv("VIBE_TEST_TRACKS");
    blk.flags = glm::vec4(tt && std::atoi(tt) == 2 ? 2.0f : 1.0f, 0.0f, 0.0f, 0.0f);
    vkCmdUpdateBuffer(cmd, P.trk_buf, 0, 48 + size_t(n) * 32, &blk);
    {
        VkMemoryBarrier2 ub{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT };
        VkDependencyInfo ud{}; ud.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        ud.memoryBarrierCount = 1; ud.pMemoryBarriers = &ub;
        vkCmdPipelineBarrier2(cmd, &ud);
    }
    const uint32_t N = PostFxState::kTrkN;
    pfx_dispatch(cmd, P.p_trk_stamp, P.s_trk_stamp, glm::vec4(0.0f), N, N);
    pfx_barrier(cmd);
    pfx_dispatch(cmd, P.p_trk_rim, P.s_trk_rim, glm::vec4(float(N), 2.6f, 0.0f, 0.0f), N, N);
    pfx_barrier(cmd);
}

}  // namespace vibe
