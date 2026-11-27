// The ray-marched terrain in three passes: hit, ground, water.
//
// The terrain shader used to find the surface, shade the ground AND shade the water in one draw: one
// program so large that a graphics driver needed two minutes and several GB to compile each of its
// builds. It is now three programs from the same source (shaders/terrain_raymarch.frag):
//
//   * the hit pass (terrain_hit.frag.spv, VIBE_HIT): the march (the patch terrain's depth picture,
//     the march, the far march), the water's surface (the plane, the stream, the sea's swell and
//     grid). It writes each pixel's result into a picture (scene binding 64, RGBA32F; layout in the
//     shader): no surface / the ground at distance t / the water with what its shading needs.
//     A render pass of its own with no attachments, before the pass that draws the terrain.
//   * the ground pass (terrain_raymarch.frag.spv and its summer / low-resolution builds,
//     VIBE_GROUND): the ground's material and light for the pixels the hit says are ground.
//   * the water pass (terrain_water_pass.frag.spv, VIBE_WATER_PASS): the water for the pixels the
//     hit says are water. Drawn right after the ground, in the same render pass (the two touch
//     different pixels); it writes the water's depth itself, tested against what is in front.
//
// The lite build (first start) still does all three in one, simply; the three-pass path is drawn
// only once the hit and water passes are built (terrain_water_pass_ready). Every pixel of the hit
// pass's area is written every frame, so the picture needs no clearing.
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/terrain_raymarch.h"

#include "engine/log.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <cstdlib>
#include <future>
#include <string>

namespace vibe {

namespace {
struct TerrainPasses {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent2D extent{ 0, 0 };
    VkShaderModule frag = VK_NULL_HANDLE, hit_frag = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE, pipe_lr = VK_NULL_HANDLE, hit = VK_NULL_HANDLE;
    std::future<VkPipeline> build, build_lr, build_hit;
    bool lr_asked = false;
};
TerrainPasses g_tp;

void tp_make_image(VkDevice dev, VmaAllocator al, VkExtent2D ext, VkQueue q, uint32_t qf, VkDescriptorSet set) {
    TerrainPasses& W = g_tp;
    if (W.view) vkDestroyImageView(dev, W.view, nullptr);
    if (W.image) vmaDestroyImage(al, W.image, W.alloc);
    W.view = VK_NULL_HANDLE; W.image = VK_NULL_HANDLE; W.alloc = nullptr;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ici.extent = { std::max(ext.width, 1u), std::max(ext.height, 1u), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(al, &ici, &aci, &W.image, &W.alloc, nullptr), "terrain hit picture");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = W.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(dev, &vci, nullptr, &W.view), "terrain hit picture view");
    W.extent = VkExtent2D{ ici.extent.width, ici.extent.height };
    vkinit::one_time_submit(dev, q, qf, [&](VkCommandBuffer cb) {
        vkinit::transition_image(cb, W.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkClearColorValue z{};
        VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cb, W.image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
    });
    VkDescriptorImageInfo ii{ VK_NULL_HANDLE, W.view, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = set;
    w.dstBinding = 64; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

void tp_collect() {
    TerrainPasses& W = g_tp;
    auto ready = [](std::future<VkPipeline>& f) { return f.valid() && f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready; };
    if (ready(W.build)) W.pipe = W.build.get();
    if (ready(W.build_lr)) W.pipe_lr = W.build_lr.get();
    if (ready(W.build_hit)) W.hit = W.build_hit.get();
}

// The water pass's pipeline: as the ground's (same vertex shader and layout), its own depth tested
// against what was drawn before it.
vkpipe::GraphicsPipelineConfig tp_water_config(VkShaderModule vert, VkShaderModule frag, VkPipelineLayout layout,
                                               std::vector<VkFormat> formats, VkFormat depth, bool vrs) {
    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = vert;
    cfg.frag = frag;
    cfg.layout = layout;
    cfg.color_formats = std::move(formats);
    cfg.depth_format = depth;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = true;
    cfg.depth_compare = VK_COMPARE_OP_LESS_OR_EQUAL;
    cfg.enable_vrs = vrs;
    return cfg;
}
} // namespace

void VulkanEngine::init_terrain_water_pass() {
    TerrainPasses& W = g_tp;
    tp_make_image(device_, allocator_, render_extent_, graphics_queue_, graphics_queue_family_, scene_desc_set_);
    const std::string sd = VIBE_SHADER_DIR;
    W.frag = vkpipe::load_shader_module(device_, sd + "/terrain_water_pass.frag.spv");
    W.hit_frag = vkpipe::load_shader_module(device_, sd + "/terrain_hit.frag.spv");
    if (!W.frag || !W.hit_frag || !terrain_raymarch_vert_module_) { log::warn("terrain passes: a shader is missing - the lite build stays in use"); return; }
    // (queued before the ground's builds, which are drawn only once these are there)
    {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = terrain_raymarch_vert_module_;
        cfg.frag = W.hit_frag;
        cfg.layout = pipeline_layout_;
        cfg.color_attachment_count = 0;          // (writes binding 64 only)
        cfg.depth_format = VK_FORMAT_UNDEFINED;
        cfg.depth_test = false;
        cfg.depth_write = false;
        cfg.cull = VK_CULL_MODE_NONE;
        W.build_hit = vkpipe::build_graphics_pipeline_async(device_, cfg, 0);
    }
    W.build = vkpipe::build_graphics_pipeline_async(device_,
        tp_water_config(terrain_raymarch_vert_module_, W.frag, pipeline_layout_,
                        { scene_color_format_, motion_vec_format_, gbuffer0_format_, gbuffer1_format_ }, depth_format_, vrs_supported_), 0);
}

void VulkanEngine::destroy_terrain_water_pass() {
    TerrainPasses& W = g_tp;
    if (W.build.valid()) W.pipe = W.build.get();
    if (W.build_lr.valid()) W.pipe_lr = W.build_lr.get();
    if (W.build_hit.valid()) W.hit = W.build_hit.get();
    for (VkPipeline p : { W.pipe, W.pipe_lr, W.hit }) if (p) vkDestroyPipeline(device_, p, nullptr);
    if (W.frag) vkDestroyShaderModule(device_, W.frag, nullptr);
    if (W.hit_frag) vkDestroyShaderModule(device_, W.hit_frag, nullptr);
    if (W.view) vkDestroyImageView(device_, W.view, nullptr);
    if (W.image) vmaDestroyImage(allocator_, W.image, W.alloc);
    W = TerrainPasses{};
}

void VulkanEngine::terrain_water_pass_prepare() {
    TerrainPasses& W = g_tp;
    if (!W.image) return;
    // (the full-resolution pass is the larger of the two that use it)
    if (render_extent_.width > W.extent.width || render_extent_.height > W.extent.height) {
        wait_gpu_frames_idle();
        tp_make_image(device_, allocator_, render_extent_, graphics_queue_, graphics_queue_family_, scene_desc_set_);
    }
}

bool VulkanEngine::terrain_water_pass_ready(bool lr) {
    TerrainPasses& W = g_tp;
    tp_collect();
    if (lr && !W.pipe_lr && !W.lr_asked && W.frag && terrain_raymarch_vert_module_) {
        // (the low-resolution terrain's build: only when that path is used - colour and motion, no VRS)
        W.lr_asked = true;
        W.build_lr = vkpipe::build_graphics_pipeline_async(device_,
            tp_water_config(terrain_raymarch_vert_module_, W.frag, pipeline_layout_,
                            { scene_color_format_, motion_vec_format_ }, depth_format_, false), 0);
    }
    return W.hit != VK_NULL_HANDLE && (lr ? W.pipe_lr != VK_NULL_HANDLE : W.pipe != VK_NULL_HANDLE);
}

void VulkanEngine::render_terrain_hit(VkCommandBuffer cmd, bool lr) {
    TerrainPasses& W = g_tp;
    if (!W.hit || !W.image) return;
    const VkExtent2D ext = lr ? tr_->lr_extent : render_extent_;
    // (last frame's ground and water passes read the picture: done before it is written again)
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        .renderArea = { {0, 0}, ext },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 0, .pColorAttachments = nullptr,
        .pDepthAttachment = nullptr, .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{};
    vp.width = float(ext.width); vp.height = float(ext.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0, 0}, ext };
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, W.hit);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
    terrain_rm_push(cmd, lr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanEngine::draw_terrain_water_pass(VkCommandBuffer cmd, bool lr) {
    TerrainPasses& W = g_tp;
    const VkPipeline pipe = lr ? W.pipe_lr : W.pipe;
    if (!pipe) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);      // (the ground's push constants and set stay)
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace vibe
