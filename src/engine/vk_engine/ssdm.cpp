// Screen-space displacement mapping (SSDM), the way the reference game does it.
//
// The stonework and the ground are drawn flat; then, on the G-buffer and before the lighting,
// their pixels are moved to where the stones really stand out. This is our code for the
// reference game's "rasterizer / reconstruct" route, read from its compiled shaders
// (screen-space tessellation): every formula and constant of the shaders is the game's.
//
// WHERE it runs: in the middle of the main pass - after the static stonework and the ground
// are drawn, before trees, moving things, effects, grass and the weapon. So the lighting and
// the shadows fall on the displaced surface, and everything drawn afterwards is depth-tested
// against it.
//
// render_ssdm(), called by render_world() right after the castle meshes:
//   (the main pass is ended)
//   0. colour, motion, both G-buffer layers and depth are copied (the passes read the copies)
//   1. height feed   ssdm_height / ssdm_ground: per pixel the vertex normal and the height
//                    byte (the game has both in its G-buffer)
//   2. ssdm_raster   compute: one vertex per pixel, displaced (ssdm_displace.glsl), each cell
//                    rasterized into a 64-bit word per pixel (atomic max: nearest wins);
//                    cells too big for it go to a list
//   3. ssdm_hwraster those cells drawn by the hardware, the same word (ssdm_hwargs makes the
//                    indirect draw)
//   4. ssdm_resolve  full screen: covered pixels rebuilt from their source pixel, the rest
//                    pulled toward the camera; the behind-material mask
//   5. ssdm_dilate + ssdm_behind  the mask spread, and the displaced copy blended back
//                    toward the original where the relief exposed what lay behind
//   6. the displaced copy copied into the live attachments (depth was written in 4)
//   (the main pass is taken up again, loading what is there)
//
// WHAT feeds it: mesh parts with CastleMeshGpu::ssdm > 0 (amplitude in metres), the static
// box brushes of home with a height texture (brush_ssdm_amplitude), the raymarched ground
// (terrain_raymarch.frag writes its material's height into the G-buffer). While this pass is
// on, all of these are drawn flat - mesh parts without parallax relief (draw_castle_meshes),
// brushes with wall mode 0 (descriptors.cpp) - because the stones must not be shifted twice.
//
// The game's runtime constants are not in its shaders; ours (kSsdmReach, kSsdmAmpScale) are
// chosen here. Everything else - the fade over the last quarter of the reach, the amplitude
// growing to 5 x by 100 m, the blur, the cuts at depth jumps, the edge fade - is the game's.
//
// Switches: VIBE_NO_SSDM=1 off (the stonework is back on its parallax / box relief);
// VIBE_SSDM_NO_BRUSHES=1 mesh parts only; VIBE_SSDM_DEBUG=1 paints the covered pixels red.
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_engine/state/ssdm.h"
#include "engine/vk_engine/state/static_clusters.h"
#include "engine/vk_pipelines.h"
#include "engine/vk_initializers.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>

namespace vibe {

namespace {
// (shaders/ssdm_common.glsl)
struct SsdmPC {
    glm::mat4 inv_vp;
    glm::mat4 vp;
    glm::mat4 prev_vp;
    glm::vec4 size, depth, view, misc;
};
static_assert(sizeof(SsdmPC) == 256, "SsdmPC must match ssdm_common.glsl");
constexpr uint32_t kSsdmBindings = 16;
constexpr VkShaderStageFlags kSsdmStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
constexpr VkFormat kSsdmFeedFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

// The reach (m): the game's _terrainNormalParams.x. Full displacement to 0.75 of it, none
// beyond it. (Not in the game's shaders; ours.)
constexpr float kSsdmReach = 48.0f;
// The amplitude scale: the game's _deformBlurParam.y - height 1 stands 0.2 * this metres out
// close by (growing to 5 x by 100 m). (Not in the game's shaders; ours.)
constexpr float kSsdmAmpScale = 1.0f;
// The ground (ssdm_ground.frag): the height in metres that the terrain shader's largest value
// stands for (terrain_raymarch.frag / terrain_rm_heightfield.glsl kGndSsdmMax: keep the same).
constexpr float kGndSsdmMax = 0.20f;

// How far the stones of a box brush stand out (m), 0 = not displaced. The depth the box relief
// of cube.frag gives the same wall (deeper for bigger stones, never more than a part of the
// block's thinnest side), a little less because here it is added on top of the wall instead
// of cut into it. Floor tiles (albedo 5 / 6): a low relief on their top face.
float brush_ssdm_amplitude(const game::Brush& b) {
    const float thin = std::min({ b.size.x, b.size.y, b.size.z });
    if (b.tex_albedo == 1 || b.tex_albedo == 4)
        return 0.55f * std::min(std::clamp(0.05f / std::max(b.uv_scale, 0.01f), 0.06f, 0.3f), 0.3f * thin);     // (0.8 before: a little less, the user asked)
    if (b.tex_albedo == 5 || b.tex_albedo == 6) return std::min(0.03f, 0.3f * thin);
    return 0.0f;
}
// A material's amplitude (m) as the height factor the feed multiplies its height by (the
// game's saturate(_alphaHeightIntensity * vertex alpha)).
float height_factor(float amp_m) { return std::clamp(amp_m / (0.2f * kSsdmAmpScale), 0.0f, 1.0f); }

void mem_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkAccessFlags sa, VkPipelineStageFlags dst, VkAccessFlags da) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = sa; mb.dstAccessMask = da;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
}
}  // namespace

// Is the screen-space displacement on this frame? (The parts it displaces are then drawn flat.)
bool VulkanEngine::ssdm_active() const {
    static const bool off = std::getenv("VIBE_NO_SSDM") != nullptr;
    return !off && !ssdm_failed_ && deferred_lighting_active_;
}

// Is the raymarched ground displaced too this frame? (terrain_raymarch.frag then writes its
// material's height into the G-buffer and leaves its own parallax out: scene constant
// rm_perf bit 64, descriptors.cpp.) Only with the terrain at full resolution: the
// low-resolution terrain's compose pass carries no height. VIBE_NO_GROUND_DETAIL=1 off.
// (Before the pass is first built its ground pipeline does not exist yet: the ground still counts,
//  so that the ground alone switches it on - it used to wait for a castle wall to come in view.)
bool VulkanEngine::ground_ssdm_active() const {
    static const bool off = std::getenv("VIBE_NO_GROUND_DETAIL") != nullptr;
    return !off && ssdm_active() && ssdm_ && (!ssdm_ready_ || ssdm_->ground_pipeline) && rt_.terrain_raymarch_enabled && !tr_use_lowres() && rt_.textures_enabled;
}

void VulkanEngine::destroy_ssdm_targets() {
    for (SsdmImage* t : { &ssdm_->hn, &ssdm_->color, &ssdm_->g0, &ssdm_->g1, &ssdm_->motion, &ssdm_->scene_z,
                          &ssdm_->o_color, &ssdm_->o_g0, &ssdm_->o_g1, &ssdm_->b_mask, &ssdm_->b_tiles, &ssdm_->b_weight }) {
        if (t->view) vkDestroyImageView(device_, t->view, nullptr);
        if (t->image) vmaDestroyImage(allocator_, t->image, t->alloc);
        *t = SsdmImage{};
    }
    auto drop = [&](VkBuffer& b, VmaAllocation& a) { if (b) vmaDestroyBuffer(allocator_, b, a); b = VK_NULL_HANDLE; a = nullptr; };
    drop(ssdm_->raster, ssdm_->raster_alloc);
    drop(ssdm_->hw_list, ssdm_->hw_list_alloc);
    drop(ssdm_->hw_ctl, ssdm_->hw_ctl_alloc);
    ssdm_->extent = { 0, 0 };
}

void VulkanEngine::destroy_ssdm() {
    if (device_) vkDeviceWaitIdle(device_);
    destroy_ssdm_targets();
    for (VkPipeline* p : { &ssdm_->height_pipeline, &ssdm_->ground_pipeline, &ssdm_->raster_pipeline, &ssdm_->hwargs_pipeline,
                           &ssdm_->hwraster_pipeline, &ssdm_->resolve_pipeline, &ssdm_->dilate_pipeline, &ssdm_->behind_pipeline }) {
        if (*p) vkDestroyPipeline(device_, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
    for (VkShaderModule* m : { &ssdm_->height_vert, &ssdm_->height_frag, &ssdm_->ground_frag, &ssdm_->raster_comp, &ssdm_->hwargs_comp,
                               &ssdm_->hwraster_vert, &ssdm_->hwraster_frag, &ssdm_->resolve_frag, &ssdm_->dilate_comp, &ssdm_->behind_comp }) {
        if (*m) vkDestroyShaderModule(device_, *m, nullptr);
        *m = VK_NULL_HANDLE;
    }
    if (ssdm_->layout) vkDestroyPipelineLayout(device_, ssdm_->layout, nullptr);
    if (ssdm_->set_layout) vkDestroyDescriptorSetLayout(device_, ssdm_->set_layout, nullptr);
    if (ssdm_->pool) vkDestroyDescriptorPool(device_, ssdm_->pool, nullptr);
    if (ssdm_->nearest) vkDestroySampler(device_, ssdm_->nearest, nullptr);
    if (ssdm_->linear) vkDestroySampler(device_, ssdm_->linear, nullptr);
    ssdm_->layout = VK_NULL_HANDLE; ssdm_->set_layout = VK_NULL_HANDLE; ssdm_->pool = VK_NULL_HANDLE;
    ssdm_->nearest = VK_NULL_HANDLE; ssdm_->linear = VK_NULL_HANDLE; ssdm_->set = VK_NULL_HANDLE;
    ssdm_ready_ = false;
}

namespace {
VkDescriptorType ssdm_binding_type(uint32_t i) {
    if (i <= 5 || i == 15) return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    if (i <= 8) return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
}
}  // namespace

bool VulkanEngine::init_ssdm() {
    if (ssdm_ready_) return true;
    if (ssdm_failed_ || !taa_->vert_module || !pipeline_layout_) return false;
    const std::string sd = VIBE_SHADER_DIR;
    auto load = [&](const char* n) { return vkpipe::load_shader_module(device_, sd + "/" + n + ".spv"); };
    ssdm_->height_vert   = load("ssdm_height.vert");
    ssdm_->height_frag   = load("ssdm_height.frag");
    ssdm_->raster_comp   = load("ssdm_raster.comp");
    ssdm_->hwargs_comp   = load("ssdm_hwargs.comp");
    ssdm_->hwraster_vert = load("ssdm_hwraster.vert");
    ssdm_->hwraster_frag = load("ssdm_hwraster.frag");
    ssdm_->resolve_frag  = load("ssdm_resolve.frag");
    ssdm_->dilate_comp   = load("ssdm_dilate.comp");
    ssdm_->behind_comp   = load("ssdm_behind.comp");
    if (!ssdm_->height_vert || !ssdm_->height_frag || !ssdm_->raster_comp || !ssdm_->hwargs_comp || !ssdm_->hwraster_vert ||
        !ssdm_->hwraster_frag || !ssdm_->resolve_frag || !ssdm_->dilate_comp || !ssdm_->behind_comp) {
        log::warnf("[ssdm] shaders missing - screen-space displacement off");
        ssdm_failed_ = true;
        return false;
    }
    // Samplers: nearest + clamp (exact texels; G-buffer data must not be blended), and bilinear
    // + clamp for the behind-material weight.
    {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST; sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &sci, nullptr, &ssdm_->nearest), "ssdm sampler");
        sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        vk_check(vkCreateSampler(device_, &sci, nullptr, &ssdm_->linear), "ssdm linear sampler");
    }
    // The passes' set (ssdm_common.glsl).
    {
        std::array<VkDescriptorSetLayoutBinding, kSsdmBindings> b{};
        uint32_t n_img = 0, n_buf = 0, n_sto = 0;
        for (uint32_t i = 0; i < kSsdmBindings; ++i) {
            b[i].binding = i;
            b[i].descriptorType = ssdm_binding_type(i);
            b[i].descriptorCount = 1;
            b[i].stageFlags = kSsdmStages;
            if (b[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ++n_img;
            else if (b[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ++n_buf;
            else ++n_sto;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = kSsdmBindings; lci.pBindings = b.data();
        vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &ssdm_->set_layout), "ssdm set layout");
        const VkDescriptorPoolSize ps[3] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, n_img },
                                             { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, n_buf },
                                             { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, n_sto } };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 3; pci.pPoolSizes = ps;
        vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &ssdm_->pool), "ssdm pool");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = ssdm_->pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &ssdm_->set_layout;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &ssdm_->set), "ssdm set");
        VkPushConstantRange pcr{ kSsdmStages, 0, sizeof(SsdmPC) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &ssdm_->set_layout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &ssdm_->layout), "ssdm layout");
    }
    // Height feed: the world's vertex layout, push constants and scene set; tested against the
    // scene's depth, which it does not write.
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = ssdm_->height_vert; cfg.frag = ssdm_->height_frag;
        cfg.layout = pipeline_layout_;
        cfg.color_formats = { kSsdmFeedFormat };
        cfg.depth_format = depth_format_;
        VkVertexInputBindingDescription vb{};
        vb.binding = 0; vb.stride = sizeof(Vertex); vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        cfg.vbindings = { vb };
        VkVertexInputAttributeDescription a0{}, a1{}, a2{};
        a0.location = 0; a0.format = VK_FORMAT_R32G32B32_SFLOAT; a0.offset = offsetof(Vertex, position);
        a1.location = 1; a1.format = VK_FORMAT_R32G32B32_SFLOAT; a1.offset = offsetof(Vertex, normal);
        a2.location = 2; a2.format = VK_FORMAT_R32G32_SFLOAT;    a2.offset = offsetof(Vertex, uv);
        cfg.vattrs = { a0, a1, a2 };
        cfg.cull = VK_CULL_MODE_NONE;           // (the depth test keeps the face that is seen)
        cfg.depth_test = true; cfg.depth_write = false;
        cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
        ssdm_->height_pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    // Ground feed (ssdm_ground.frag): full-screen triangle into the same target, after the
    // meshes, inside the same pass (which has the depth attachment; not tested, not written).
    // Optional: without it only the stonework is displaced.
    ssdm_->ground_frag = load("ssdm_ground.frag");
    if (ssdm_->ground_frag) {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = taa_->vert_module; cfg.frag = ssdm_->ground_frag;
        cfg.layout = ssdm_->layout;
        cfg.color_formats = { kSsdmFeedFormat };
        cfg.depth_format = depth_format_;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = false; cfg.depth_write = false;
        ssdm_->ground_pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    // Hardware path: no attachments - the fragments only do the atomic max.
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = ssdm_->hwraster_vert; cfg.frag = ssdm_->hwraster_frag;
        cfg.layout = ssdm_->layout;
        cfg.color_attachment_count = 0;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = false; cfg.depth_write = false;
        ssdm_->hwraster_pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    // Resolve: the displaced copy + the live motion, and the live depth (written as it is).
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = taa_->vert_module; cfg.frag = ssdm_->resolve_frag;
        cfg.layout = ssdm_->layout;
        cfg.color_formats = { VK_FORMAT_R16G16B16A16_SFLOAT, motion_vec_format_, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT };
        cfg.depth_format = depth_format_;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = true; cfg.depth_write = true;
        cfg.depth_compare = VK_COMPARE_OP_ALWAYS;
        ssdm_->resolve_pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    auto compute = [&](VkShaderModule m, VkPipeline& out, const char* what) {
        VkComputePipelineCreateInfo cci{};
        cci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cci.stage.module = m;
        cci.stage.pName = "main";
        cci.layout = ssdm_->layout;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cci, nullptr, &out) != VK_SUCCESS) {
            log::warnf("[ssdm] %s pipeline failed", what);
            out = VK_NULL_HANDLE;
        }
    };
    compute(ssdm_->raster_comp, ssdm_->raster_pipeline, "raster");
    compute(ssdm_->hwargs_comp, ssdm_->hwargs_pipeline, "hwargs");
    compute(ssdm_->dilate_comp, ssdm_->dilate_pipeline, "dilate");
    compute(ssdm_->behind_comp, ssdm_->behind_pipeline, "behind");
    if (!ssdm_->height_pipeline || !ssdm_->hwraster_pipeline || !ssdm_->resolve_pipeline || !ssdm_->raster_pipeline ||
        !ssdm_->hwargs_pipeline || !ssdm_->dilate_pipeline || !ssdm_->behind_pipeline) {
        log::warnf("[ssdm] pipeline creation failed - screen-space displacement off");
        ssdm_failed_ = true;
        return false;
    }
    ssdm_ready_ = true;
    log::infof("[ssdm] screen-space displacement mapping ready (compute rasterizer + resolve, reach %.0f m)", kSsdmReach);
    return true;
}

void VulkanEngine::create_ssdm_targets() {
    destroy_ssdm_targets();
    const uint32_t W = render_extent_.width, H = render_extent_.height;
    ssdm_->half  = { (W + 1) / 2, (H + 1) / 2 };
    ssdm_->tiles = { (ssdm_->half.width + 7) / 8 + 1, (ssdm_->half.height + 7) / 8 + 1 };
    auto make = [&](VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkExtent2D ex, SsdmImage& t) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D; ici.format = fmt;
        ici.extent = { ex.width, ex.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage; ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &t.image, &t.alloc, nullptr), "ssdm image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt;
        vci.subresourceRange = { aspect, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &t.view), "ssdm view");
    };
    auto make_buf = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& b, VmaAllocation& a) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &b, &a, nullptr), "ssdm buffer");
    };
    const VkImageUsageFlags S = VK_IMAGE_USAGE_SAMPLED_BIT, CA = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                            TD = VK_IMAGE_USAGE_TRANSFER_DST_BIT, TS = VK_IMAGE_USAGE_TRANSFER_SRC_BIT, ST = VK_IMAGE_USAGE_STORAGE_BIT;
    const VkImageAspectFlags C = VK_IMAGE_ASPECT_COLOR_BIT, D = VK_IMAGE_ASPECT_DEPTH_BIT;
    const VkExtent2D full = render_extent_;
    make(kSsdmFeedFormat, CA | S, C, full, ssdm_->hn);
    make(scene_color_format_, TD | S, C, full, ssdm_->color);
    make(gbuffer0_format_, TD | S, C, full, ssdm_->g0);
    make(gbuffer1_format_, TD | S, C, full, ssdm_->g1);
    make(motion_vec_format_, TD | S, C, full, ssdm_->motion);
    make(depth_format_, TD | S, D, full, ssdm_->scene_z);
    make(VK_FORMAT_R16G16B16A16_SFLOAT, CA | ST | TS, C, full, ssdm_->o_color);
    make(VK_FORMAT_R8G8B8A8_UNORM, CA | ST | TS, C, full, ssdm_->o_g0);
    make(VK_FORMAT_R32_UINT, CA | ST | TS, C, full, ssdm_->o_g1);
    make(VK_FORMAT_R32_SFLOAT, ST | TD, C, ssdm_->half, ssdm_->b_mask);
    make(VK_FORMAT_R32_UINT, ST | TD, C, ssdm_->tiles, ssdm_->b_tiles);
    make(VK_FORMAT_R32_SFLOAT, ST | S | TD, C, ssdm_->half, ssdm_->b_weight);
    make_buf(VkDeviceSize(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, ssdm_->raster, ssdm_->raster_alloc);
    make_buf(VkDeviceSize(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ssdm_->hw_list, ssdm_->hw_list_alloc);
    make_buf(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             ssdm_->hw_ctl, ssdm_->hw_ctl_alloc);
    ssdm_->extent = render_extent_;
    ssdm_first_use_ = true;

    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, GE = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo ii[kSsdmBindings]{};
    ii[0]  = { ssdm_->nearest, ssdm_->hn.view, RO };      ii[1] = { ssdm_->nearest, ssdm_->scene_z.view, RO };
    ii[2]  = { ssdm_->nearest, ssdm_->color.view, RO };   ii[3] = { ssdm_->nearest, ssdm_->g0.view, RO };
    ii[4]  = { ssdm_->nearest, ssdm_->g1.view, RO };      ii[5] = { ssdm_->nearest, ssdm_->motion.view, RO };
    ii[9]  = { VK_NULL_HANDLE, ssdm_->o_color.view, GE }; ii[10] = { VK_NULL_HANDLE, ssdm_->o_g0.view, GE };
    ii[11] = { VK_NULL_HANDLE, ssdm_->o_g1.view, GE };    ii[12] = { VK_NULL_HANDLE, ssdm_->b_mask.view, GE };
    ii[13] = { VK_NULL_HANDLE, ssdm_->b_tiles.view, GE }; ii[14] = { VK_NULL_HANDLE, ssdm_->b_weight.view, GE };
    ii[15] = { ssdm_->linear, ssdm_->b_weight.view, GE };
    const VkDescriptorBufferInfo bi[3] = { { ssdm_->raster, 0, VK_WHOLE_SIZE }, { ssdm_->hw_list, 0, VK_WHOLE_SIZE },
                                           { ssdm_->hw_ctl, 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet w[kSsdmBindings]{};
    for (uint32_t i = 0; i < kSsdmBindings; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ssdm_->set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = ssdm_binding_type(i);
        if (i >= 6 && i <= 8) w[i].pBufferInfo = &bi[i - 6];
        else w[i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(device_, kSsdmBindings, w, 0, nullptr);
}

// Called INSIDE the main pass (render_world, after the static stonework). Returns false and
// touches nothing when there is nothing to displace in view; otherwise it ends the pass, does
// its work, begins the pass again (loading its attachments) and returns true - the caller then
// binds its own pipeline / set / buffers again.
bool VulkanEngine::render_ssdm(VkCommandBuffer cmd) {
    if (!ssdm_active()) return false;
    const FrameView& fv = current_frame_view_;
    const float rw = float(render_extent_.width), rh = float(render_extent_.height);

    // ---- What is in view within the reach ----
    std::vector<const CastleMeshGpu*> parts;
    for (const CastleMeshGpu* gp : mesh_parts_now()) {
        const CastleMeshGpu& g = *gp;
        if (g.ssdm <= 0.0f || g.tex_albedo == -3) continue;
        const glm::vec3 lo = g.bmin - glm::vec3(g.ssdm), hi = g.bmax + glm::vec3(g.ssdm);
        if (!aabb_visible(fv.frustum, lo, hi)) continue;
        if (glm::distance(glm::clamp(fv.eye_pos, lo, hi), fv.eye_pos) > kSsdmReach) continue;
        parts.push_back(gp);
    }
    std::vector<uint32_t> brushes;
    static const bool no_brushes = std::getenv("VIBE_SSDM_NO_BRUSHES") != nullptr;
    if (globe::current().home && !no_brushes && rt_.textures_enabled)
        for (size_t i = 0; i < world_.brushes.size() && i < static_brush_models_.size(); ++i) {
            const float amp = brush_ssdm_amplitude(world_.brushes[i]);
            if (amp <= 0.0f) continue;
            const auto& a = world_.render_aabbs[i];
            const glm::vec3 lo = a.min - glm::vec3(amp), hi = a.max + glm::vec3(amp);
            if (!aabb_visible(fv.frustum, lo, hi)) continue;
            if (glm::distance(glm::clamp(fv.eye_pos, lo, hi), fv.eye_pos) > kSsdmReach) continue;
            brushes.push_back(uint32_t(i));
        }
    // The ground: not from high up (nothing of it is within the terrain shader's reach).
    const bool ground = ground_ssdm_active() && rm_air_lod_k_ < 40.0f;
    if (parts.empty() && brushes.empty() && !ground) return false;
    if (!init_ssdm()) return false;
    // (destroy_taa() drops these targets with the ones they are sized like, so a missing image
    // here means: first use, or the render size changed and nothing in flight refers to them)
    if (!ssdm_->hn.image || ssdm_->extent.width != render_extent_.width || ssdm_->extent.height != render_extent_.height)
        create_ssdm_targets();

    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, CA = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    const VkImageLayout UD = VK_IMAGE_LAYOUT_UNDEFINED, DA = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, GE = VK_IMAGE_LAYOUT_GENERAL;
    const VkImageLayout TS = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, TD = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageAspectFlags DEPTH = VK_IMAGE_ASPECT_DEPTH_BIT;
    const VkImageLayout from_c = ssdm_first_use_ ? UD : RO;
    const uint32_t W = render_extent_.width, H = render_extent_.height;
    VkViewport vp{ 0.0f, 0.0f, rw, rh, 0.0f, 1.0f };
    VkRect2D full{ {0, 0}, render_extent_ };
    auto begin = [&](const VkRenderingAttachmentInfo* cols, uint32_t n, const VkRenderingAttachmentInfo* depth) {
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = full; ri.layerCount = 1;
        ri.colorAttachmentCount = n; ri.pColorAttachments = cols; ri.pDepthAttachment = depth;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &full);
    };
    auto att = [](VkImageView v, VkImageLayout l, bool clear) {
        VkRenderingAttachmentInfo a{};
        a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        a.imageView = v; a.imageLayout = l;
        a.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        return a;
    };
    auto dont_care = [](VkImageView v, VkImageLayout l) {
        VkRenderingAttachmentInfo a{};
        a.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        a.imageView = v; a.imageLayout = l;
        a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        return a;
    };

    // ---- The main pass stops here ----
    vkCmdEndRendering(cmd);

    // ---- 0. Copies of what the passes read while the live attachments are written ----
    {
        VkImageCopy r{};
        r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.extent = { W, H, 1 };
        const VkImage src[4] = { scene_color_image_, gbuffer0_image_, gbuffer1_image_, motion_vec_image_ };
        SsdmImage* dst[4] = { &ssdm_->color, &ssdm_->g0, &ssdm_->g1, &ssdm_->motion };
        for (int i = 0; i < 4; ++i) {
            vkinit::transition_image(cmd, src[i], CA, TS);
            vkinit::transition_image(cmd, dst[i]->image, from_c, TD);
            vkCmdCopyImage(cmd, src[i], TS, dst[i]->image, TD, 1, &r);
            vkinit::transition_image(cmd, dst[i]->image, TD, RO);
            vkinit::transition_image(cmd, src[i], TS, CA);
        }
        VkImageCopy rd = r;
        rd.srcSubresource = { DEPTH, 0, 0, 1 };
        rd.dstSubresource = { DEPTH, 0, 0, 1 };
        vkinit::transition_image_aspect(cmd, depth_image_, DA, TS, DEPTH);
        vkinit::transition_image_aspect(cmd, ssdm_->scene_z.image, from_c, TD, DEPTH);
        vkCmdCopyImage(cmd, depth_image_, TS, ssdm_->scene_z.image, TD, 1, &rd);
        vkinit::transition_image_aspect(cmd, ssdm_->scene_z.image, TD, RO, DEPTH);
        vkinit::transition_image_aspect(cmd, depth_image_, TS, DA, DEPTH);
    }

    // ---- The passes' constants: camera-relative matrices (ssdm_common.glsl) ----
    SsdmPC pc{};
    {
        const glm::dmat4 to_world = glm::translate(glm::dmat4(1.0), glm::dvec3(fv.eye_pos));
        const glm::dmat4 vp_rel = glm::dmat4(fv.vp) * to_world;
        pc.vp = glm::mat4(vp_rel);
        pc.inv_vp = glm::mat4(glm::inverse(vp_rel));
        glm::dmat4 prev = vp_rel;
        if (prev_view_proj_valid_) {
            glm::dmat4 T(1.0);
            if (rt_.fsr2_enabled) {
                const glm::vec2 jd = -(fv.jitter - prev_jitter_) * 0.5f;
                T[3][0] = 2.0 * jd.x; T[3][1] = 2.0 * jd.y;
            }
            prev = T * glm::dmat4(prev_view_proj_) * to_world;
        }
        pc.prev_vp = glm::mat4(prev);
        pc.size  = glm::vec4(rw, rh, 1.0f / rw, 1.0f / rh);
        // linear depth = x / (y - z): from the projection's z row (finite or infinite far plane)
        pc.depth = glm::vec4(-fv.proj[3][2], -fv.proj[2][2], kSsdmReach, kSsdmAmpScale);
        const glm::vec3 fwd = -glm::normalize(glm::vec3(fv.inv_view[2]));
        pc.view  = glm::vec4(fwd, 1.0f / std::max(std::abs(fv.proj[0][0]), 1e-6f));
        static const bool debug = std::getenv("VIBE_SSDM_DEBUG") != nullptr;
        pc.misc  = glm::vec4(float(frame_number_ & 1023u), debug ? 1.0f : 0.0f, 0.0f, kGndSsdmMax / (0.2f * kSsdmAmpScale));
    }

    // ---- 1. The height feed ----
    vkinit::transition_image(cmd, ssdm_->hn.image, from_c, CA);
    {
        VkRenderingAttachmentInfo col = att(ssdm_->hn.view, CA, true);
        col.clearValue.color = { { 0.0f, 0.0f, 0.0f, -1.0f } };       // (w < 0: nothing fed)
        const VkRenderingAttachmentInfo dep = att(depth_view_, DA, false);
        begin(&col, 1, &dep);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_->height_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
        VkDeviceSize off0 = 0;
        for (const CastleMeshGpu* gp : parts) {
            const CastleMeshGpu& g = *gp;
            PushConstants pcm{};
            pcm.mvp = fv.vp * g.xf;
            pcm.model = g.xf;
            pcm.prev_mvp = pcm.mvp;
            pcm.tex_params = glm::vec4(float(g.tex_albedo), float(g.tex_normal), g.uv_scale, 0.0f);
            pcm.grass_params = glm::vec4(height_factor(g.ssdm), 0.0f, 0.0f, 0.0f);
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pcm);
            // (the same triangles the colour pass drew: its cluster selection - draw_castle_meshes)
            VkBuffer ib = g.mesh.index_buffer;
            const bool clustered = scl_->ready && g.cl_part >= 0 && size_t(g.cl_part) < scl_->parts.size();
            if (clustered) { const SclPart& P = scl_->parts[size_t(g.cl_part)]; ib = (P.state == 2 && P.ib_full) ? P.ib_full : P.ib_coarse; }
            vkCmdBindVertexBuffers(cmd, 0, 1, &g.mesh.vertex_buffer, &off0);
            vkCmdBindIndexBuffer(cmd, ib, 0, VK_INDEX_TYPE_UINT32);
            if (clustered)
                vkCmdDrawIndexedIndirectCount(cmd, scl_->cmd_buf, 256 + VkDeviceSize(g.cl_cmd_base) * sizeof(VkDrawIndexedIndirectCommand),
                                              scl_->cmd_buf, VkDeviceSize(g.cl_part) * 4, g.cl_count, sizeof(VkDrawIndexedIndirectCommand));
            else vkCmdDrawIndexed(cmd, g.mesh.index_count, 1, 0, 0, 0);
        }
        if (!brushes.empty()) {
            vkCmdBindVertexBuffers(cmd, 0, 1, &cube_mesh_.vertex_buffer, &off0);
            vkCmdBindIndexBuffer(cmd, cube_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            for (uint32_t i : brushes) {
                const game::Brush& b = world_.brushes[i];
                PushConstants pcm{};
                pcm.mvp = fv.vp * static_brush_models_[i];
                pcm.model = static_brush_models_[i];
                pcm.prev_mvp = pcm.mvp;
                pcm.tex_params = glm::vec4(float(b.tex_albedo), float(b.tex_normal), b.uv_scale, 0.0f);
                pcm.grass_params = glm::vec4(height_factor(brush_ssdm_amplitude(b)), 0.0f, 0.0f, 1.0f);
                vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pcm);
                vkCmdDrawIndexed(cmd, cube_mesh_.index_count, 1, 0, 0, 0);
            }
        }
        if (ground && ssdm_->ground_pipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_->ground_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_->layout, 0, 1, &ssdm_->set, 0, nullptr);
            vkCmdPushConstants(cmd, ssdm_->layout, kSsdmStages, 0, sizeof(pc), &pc);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }
        vkCmdEndRendering(cmd);
    }
    vkinit::transition_image(cmd, ssdm_->hn.image, CA, RO);

    // ---- Clears: the rasterizer's words, the hardware list's count, the behind images ----
    if (ssdm_first_use_)
        for (SsdmImage* t : { &ssdm_->o_color, &ssdm_->o_g0, &ssdm_->o_g1, &ssdm_->b_mask, &ssdm_->b_tiles, &ssdm_->b_weight })
            vkinit::transition_image(cmd, t->image, UD, GE);
    mem_barrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(cmd, ssdm_->raster, 0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, ssdm_->hw_ctl, 0, VK_WHOLE_SIZE, 0u);
    {
        const VkClearColorValue z{ { 0.0f, 0.0f, 0.0f, 0.0f } };
        const VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        for (SsdmImage* t : { &ssdm_->b_mask, &ssdm_->b_tiles, &ssdm_->b_weight })
            vkCmdClearColorImage(cmd, t->image, GE, &z, 1, &rr);
    }
    mem_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // ---- 2. The compute rasterizer ----
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssdm_->layout, 0, 1, &ssdm_->set, 0, nullptr);
    vkCmdPushConstants(cmd, ssdm_->layout, kSsdmStages, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssdm_->raster_pipeline);
    vkCmdDispatch(cmd, (W + 15) / 16, (H + 15) / 16, 1);
    mem_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssdm_->hwargs_pipeline);
    vkCmdDispatch(cmd, 1, 1, 1);
    mem_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // ---- 3. The hardware path ----
    {
        begin(nullptr, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_->hwraster_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_->layout, 0, 1, &ssdm_->set, 0, nullptr);
        vkCmdPushConstants(cmd, ssdm_->layout, kSsdmStages, 0, sizeof(pc), &pc);
        vkCmdDrawIndirect(cmd, ssdm_->hw_ctl, 4, 1, sizeof(VkDrawIndirectCommand));
        vkCmdEndRendering(cmd);
    }
    mem_barrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // ---- 4. The resolve: the displaced copy, the live motion and depth ----
    {
        const VkRenderingAttachmentInfo cols[4] = { dont_care(ssdm_->o_color.view, GE), att(motion_vec_view_, CA, false),
                                                    dont_care(ssdm_->o_g0.view, GE), dont_care(ssdm_->o_g1.view, GE) };
        const VkRenderingAttachmentInfo dep = att(depth_view_, DA, false);
        begin(cols, 4, &dep);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_->resolve_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssdm_->layout, 0, 1, &ssdm_->set, 0, nullptr);
        vkCmdPushConstants(cmd, ssdm_->layout, kSsdmStages, 0, sizeof(pc), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }
    mem_barrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // ---- 5. The behind-material chain ----
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssdm_->layout, 0, 1, &ssdm_->set, 0, nullptr);
    vkCmdPushConstants(cmd, ssdm_->layout, kSsdmStages, 0, sizeof(pc), &pc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssdm_->dilate_pipeline);
    vkCmdDispatch(cmd, (ssdm_->half.width + 7) / 8, (ssdm_->half.height + 7) / 8, 1);
    mem_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ssdm_->behind_pipeline);
    vkCmdDispatch(cmd, (W + 7) / 8, (H + 7) / 8, 1);
    mem_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    ssdm_first_use_ = false;

    // ---- 6. The displaced copy into the live attachments (raw bits: sRGB bytes, 10:10:10:2) ----
    {
        VkImageCopy r{};
        r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.extent = { W, H, 1 };
        const VkImage dst[3] = { scene_color_image_, gbuffer0_image_, gbuffer1_image_ };
        const SsdmImage* src[3] = { &ssdm_->o_color, &ssdm_->o_g0, &ssdm_->o_g1 };
        for (int i = 0; i < 3; ++i) {
            vkinit::transition_image(cmd, dst[i], CA, TD);
            vkCmdCopyImage(cmd, src[i]->image, GE, dst[i], TD, 1, &r);
            vkinit::transition_image(cmd, dst[i], TD, CA);
        }
        mem_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    // ---- The main pass again, loading what is there ----
    {
        const VkRenderingAttachmentInfo cols[4] = { att(scene_color_view_, CA, false), att(motion_vec_view_, CA, false),
                                                    att(gbuffer0_view_, CA, false), att(gbuffer1_view_, CA, false) };
        const VkRenderingAttachmentInfo dep = att(depth_view_, DA, false);
        begin(cols, 4, &dep);
    }
    return true;
}

}  // namespace vibe
