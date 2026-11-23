// The world's slow fields round the camera, as two textures.
//
// Beyond 1450 m from home every height the terrain shader works out passes through the
// shape of the world (shaders/world_map.glsl): land or sea (wm_continent: 21 noise look-ups,
// 29 near a coast) and where the ranges stand (wm_ranges: 2). Both change over hundreds of
// metres - and were worked out again at every step of every pixel's march and three times
// for its normal (most of the terrain pass outside the home tile, measured 2026-11-23).
//
// Here they are drawn into two RG32F pictures round the camera, by the same GLSL
// (shaders/wm_cache.frag includes the terrain shader's own functions), in the chart's
// coordinates, whichever chart that is:
//   near   2048 x 2048 at 4 m    (+-4 km: everything the normal march reaches)
//   far    1024 x 1024 at 160 m  (+-82 km: the far march from the air, which ends at 80 km)
// The terrain shader reads them with its own bilinear sum of four texels
// (terrain_rm_noise_road.glsl rm_world_vr; a filtered read moves in 1/256 steps of a texel,
// too coarse for a normal taken over 2 cm), the near one where the position lies in it. It
// no longer holds the formula at all, which also makes it a good deal smaller to compile.
// A picture is drawn again when the camera has moved on from its middle (1 km / 1.9 km), or
// the chart's mapping has changed. VIBE_LOG_WM_CACHE=1 logs each drawing.
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vibe {

namespace {
constexpr uint32_t kWmN[2]     = { 2048, 1024 };       // texels a side
constexpr float    kWmTexel[2] = { 4.0f, 160.0f };     // metres a texel
constexpr float    kWmMove[2]  = { 1024.0f, 1920.0f }; // drawn again when the camera is this far from the middle
}

void VulkanEngine::init_wm_cache() {
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &wm_cache_sampler_), "wm_cache sampler");
    for (int l = 0; l < 2; ++l) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R32G32_SFLOAT;
        ici.extent = { kWmN[l], kWmN[l], 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &wm_cache_image_[l], &wm_cache_alloc_[l], nullptr), "wm_cache image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = wm_cache_image_[l];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R32G32_SFLOAT;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &wm_cache_view_[l]), "wm_cache view");
        // (readable from the start: the binding must be valid before anything is drawn into it)
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, wm_cache_image_[l], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        VkDescriptorImageInfo ii{};
        ii.sampler = wm_cache_sampler_;
        ii.imageView = wm_cache_view_[l];
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_;
        w.dstBinding = 39 + uint32_t(l);
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    std::string sd = VIBE_SHADER_DIR;
    wm_cache_frag_ = vkpipe::load_shader_module(device_, sd + "/wm_cache.frag.spv");
    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = taa_->vert_module;            // (the full-screen triangle)
    cfg.frag = wm_cache_frag_;
    cfg.layout = pipeline_layout_;
    cfg.color_format = VK_FORMAT_R32G32_SFLOAT;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = false;
    cfg.depth_write = false;
    wm_cache_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    log::info("world-field cache built (4 m and 160 m a texel)");
}

void VulkanEngine::destroy_wm_cache() {
    if (wm_cache_pipeline_) vkDestroyPipeline(device_, wm_cache_pipeline_, nullptr);
    if (wm_cache_frag_)     vkDestroyShaderModule(device_, wm_cache_frag_, nullptr);
    if (wm_cache_sampler_)  vkDestroySampler(device_, wm_cache_sampler_, nullptr);
    for (int l = 0; l < 2; ++l) {
        if (wm_cache_view_[l])  vkDestroyImageView(device_, wm_cache_view_[l], nullptr);
        if (wm_cache_image_[l]) vmaDestroyImage(allocator_, wm_cache_image_[l], wm_cache_alloc_[l]);
        wm_cache_view_[l] = VK_NULL_HANDLE; wm_cache_image_[l] = VK_NULL_HANDLE; wm_cache_alloc_[l] = nullptr;
    }
    wm_cache_pipeline_ = VK_NULL_HANDLE; wm_cache_frag_ = VK_NULL_HANDLE; wm_cache_sampler_ = VK_NULL_HANDLE;
}

// Decides, while the frame's uniforms are filled, where the pictures lie this frame and
// which have to be drawn again (wm_cache_bake does that, before the terrain is drawn).
void VulkanEngine::wm_cache_plan(SceneUBO& data) {
    // Everything the chart's mapping is made of: another chart, other pictures.
    float key[32] = {};
    int n = 0;
    key[n++] = data.globe_info.x; key[n++] = data.globe_info.y;
    for (int i = 0; i < 4; ++i) { key[n++] = data.ring_info[i]; key[n++] = data.ring_h0[i]; key[n++] = data.ring_j[i]; }
    for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) key[n++] = data.globe_frame[c][r];
    const bool rekey = !wm_cache_valid_ || std::memcmp(key, wm_cache_key_, sizeof(key)) != 0;
    std::memcpy(wm_cache_key_, key, sizeof(key));
    wm_cache_valid_ = true;
    const glm::vec2 eye(data.camera_pos.x, data.camera_pos.z);
    for (int l = 0; l < 2; ++l) {
        const float half = 0.5f * float(kWmN[l]) * kWmTexel[l];
        const glm::vec2 mid = wm_cache_org_[l] + glm::vec2(half);
        if (rekey || std::max(std::abs(eye.x - mid.x), std::abs(eye.y - mid.y)) > kWmMove[l]) {
            // (whole texels: the same places are sampled before and after, so nothing shifts)
            wm_cache_org_[l] = glm::floor((eye - glm::vec2(half)) / kWmTexel[l]) * kWmTexel[l];
            wm_cache_dirty_[l] = true;
            static const bool log_it = std::getenv("VIBE_LOG_WM_CACHE") != nullptr;
            if (log_it) log::infof("[wm-cache] level %d drawn at %.0f %.0f (camera %.1f %.1f)", l, wm_cache_org_[l].x, wm_cache_org_[l].y, eye.x, eye.y);
        }
    }
    data.wm_cache  = glm::vec4(wm_cache_org_[0].x, wm_cache_org_[0].y, 1.0f / kWmTexel[0], float(kWmN[0]));
    data.wm_cache1 = glm::vec4(wm_cache_org_[1].x, wm_cache_org_[1].y, 1.0f / kWmTexel[1], float(kWmN[1]));
    jeep_grass_press(data.car_press);
}

void VulkanEngine::wm_cache_bake(VkCommandBuffer cmd) {
    if (wm_cache_pipeline_ == VK_NULL_HANDLE) return;
    for (int l = 0; l < 2; ++l) {
        if (!wm_cache_dirty_[l]) continue;
        wm_cache_dirty_[l] = false;
        vkinit::transition_image(cmd, wm_cache_image_[l], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkClearValue cv{};
        VkRenderingAttachmentInfo att = vkinit::color_attachment_info(wm_cache_view_[l], &cv,
                                                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea = { {0, 0}, { kWmN[l], kWmN[l] } };
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &att;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{};
        vp.width = float(kWmN[l]); vp.height = float(kWmN[l]); vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, { kWmN[l], kWmN[l] } };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wm_cache_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
        PushConstants pc{};
        pc.emissive = glm::vec4(wm_cache_org_[l].x, wm_cache_org_[l].y, kWmTexel[l], 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
        vkinit::transition_image(cmd, wm_cache_image_[l], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

}  // namespace vibe
