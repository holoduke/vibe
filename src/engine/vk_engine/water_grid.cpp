// The open sea as a camera-projected grid (after the reference game's sea surface; the code is ours).
//
// The full-screen terrain pass used to SEARCH every sea pixel's ray for the swell and the wind sea
// (terrain_raymarch.frag: 6 steps + 4 of regula falsi, heights only - no horizontal displacement).
// Here the sea surface is a grid laid out on the SCREEN: each vertex's view ray is dropped onto the
// sea plane and displaced there by the five swell waves and the four FFT cascades - height AND the
// cascades' horizontal ("choppy") displacement, so the crests are as sharp as the spectrum makes
// them - and by the foam's lift. The cells beyond the screen's sides and bottom are stretched, so a
// vertex pushed sideways never opens a gap at the frame's edge. The grid is rasterised DEPTH ONLY
// into a depth picture of its own (reversed depth from kNear, no far plane: the patch terrain's
// technique, terrain_patches.cpp), and the terrain pass reads the sea's hit there (scene binding 62)
// instead of iterating. Only the open sea: lakes, the river and the stream keep their march.
// The CPU twin (water_fx.cpp water_swell_height + water_sim.cpp water_fft_height) is unchanged: it
// already reads the same cascades back (heights; the horizontal displacement only moves where a
// crest stands, not how high it is).
//
// Per frame (frame.cpp): water_grid_prepare() before any recording (the picture's size; whether it
// is drawn this frame - the scene block's water_grid.x says so), render_water_grid() as pass 0.52b
// after the patch terrain's depth picture (gpu mark `water_grid`).
// Switch: VIBE_NO_SEA_GRID=1 (the search along the ray, as before). VIBE_LOG_SEA_GRID=1: the grid's size.
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/environment.h"

#include "engine/globe.h"
#include "engine/log.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace vibe {

void water_grid_set_active(bool on);     // (water_wet.cpp: the scene block's water_grid.x)

namespace {
constexpr float kNear = 0.05f;           // (terrain_raymarch.frag wg_hit: keep the two the same)
constexpr int   kCellPx = 8;             // grid cell, pixels
constexpr int   kBorder = 2;             // stretched cells beyond the sides and the bottom
struct Grid {
    VkImage        image = VK_NULL_HANDLE;
    VmaAllocation  alloc = nullptr;
    VkImageView    view  = VK_NULL_HANDLE;
    VkSampler      samp  = VK_NULL_HANDLE;
    VkExtent2D     extent{ 0, 0 };
    VkShaderModule vert = VK_NULL_HANDLE;
    VkPipeline     pipe = VK_NULL_HANDLE;
    bool           active = false;       // drawn this frame
    int            cols = 0, rows = 0;
};
Grid g_wg;

bool wg_env_on(const char* k) { const char* v = std::getenv(k); return v && *v && *v != '0'; }
}  // namespace

static void wg_make_image(VkDevice dev, VmaAllocator al, VkFormat fmt, VkExtent2D ext) {
    Grid& G = g_wg;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { std::max(ext.width, 1u), std::max(ext.height, 1u), 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(al, &ici, &aci, &G.image, &G.alloc, nullptr), "sea grid depth");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = G.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(dev, &vci, nullptr, &G.view), "sea grid depth view");
    G.extent = VkExtent2D{ ici.extent.width, ici.extent.height };
}

// The picture at the render size, cleared to "no sea" and readable; binding 62 pointed at it.
// (Called at start-up and from water_grid_prepare, before anything is recorded, after the GPU is idle.)
static void wg_rebuild(VkDevice dev, VmaAllocator al, VkFormat fmt, VkExtent2D ext, VkQueue q, uint32_t qf, VkDescriptorSet set) {
    Grid& G = g_wg;
    if (G.view) vkDestroyImageView(dev, G.view, nullptr);
    if (G.image) vmaDestroyImage(al, G.image, G.alloc);
    G.view = VK_NULL_HANDLE; G.image = VK_NULL_HANDLE; G.alloc = nullptr;
    wg_make_image(dev, al, fmt, ext);
    vkinit::one_time_submit(dev, q, qf, [&](VkCommandBuffer cb) {
        vkinit::transition_image_aspect(cb, G.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkClearDepthStencilValue z{ 0.0f, 0 };
        VkImageSubresourceRange rr{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        vkCmdClearDepthStencilImage(cb, G.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &rr);
        vkinit::transition_image_aspect(cb, G.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    });
    VkDescriptorImageInfo ii{ G.samp, G.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet = set;
    w.dstBinding = 62; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

void VulkanEngine::init_water_grid() {
    Grid& G = g_wg;
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &G.samp), "sea grid sampler");
    }
    wg_rebuild(device_, allocator_, depth_format_, render_extent_, graphics_queue_, graphics_queue_family_, scene_desc_set_);
    G.vert = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/water_grid.vert.spv");
    if (G.vert) {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = G.vert;
        cfg.frag = VK_NULL_HANDLE;                   // depth only
        cfg.layout = pipeline_layout_;
        cfg.color_attachment_count = 0;
        cfg.depth_format = depth_format_;
        cfg.cull = VK_CULL_MODE_NONE;                // (a folded crest turns cells over)
        cfg.depth_test = true;
        cfg.depth_write = true;
        cfg.depth_compare = VK_COMPARE_OP_GREATER;   // reversed depth
        G.pipe = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    log::infof("[sea-grid] projected grid: %s", G.pipe ? "pipeline built" : "NO pipeline (water_grid.vert missing)");
}

void VulkanEngine::destroy_water_grid() {
    Grid& G = g_wg;
    if (G.pipe) vkDestroyPipeline(device_, G.pipe, nullptr);
    if (G.vert) vkDestroyShaderModule(device_, G.vert, nullptr);
    if (G.view) vkDestroyImageView(device_, G.view, nullptr);
    if (G.image) vmaDestroyImage(allocator_, G.image, G.alloc);
    if (G.samp) vkDestroySampler(device_, G.samp, nullptr);
    G = Grid{};
}

void VulkanEngine::water_grid_prepare() {
    Grid& G = g_wg;
    static const bool off = wg_env_on("VIBE_NO_SEA_GRID");
    G.active = false;
    if (!G.pipe || off || !rt_.water_enabled || !rt_.terrain_raymarch_enabled) { water_grid_set_active(false); return; }
    if (render_extent_.width != G.extent.width || render_extent_.height != G.extent.height) {
        wait_gpu_frames_idle();
        wg_rebuild(device_, allocator_, depth_format_, render_extent_, graphics_queue_, graphics_queue_family_, scene_desc_set_);
    }
    // (the sea has height only with wind or the cascades on; from below the surface the march does it)
    const glm::vec3 eye = current_frame_view_.eye_pos;
    const bool waves = env_->water_wind.w > 1e-4f || water_fft_bound() > 0.0f;
    G.active = waves && eye.y > rt_.water_level - 0.5f && eye.y < rt_.water_level + 3000.0f;
    water_grid_set_active(G.active);
}

void VulkanEngine::render_water_grid(VkCommandBuffer cmd) {
    Grid& G = g_wg;
    if (!G.active) return;
    G.cols = (int(G.extent.width) + kCellPx - 1) / kCellPx + 2 * kBorder;
    G.rows = (int(G.extent.height) + kCellPx - 1) / kCellPx + kBorder;
    vkinit::transition_image_aspect(cmd, G.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkClearValue clear{};
    clear.depthStencil.depth = 0.0f;                 // reversed depth: 0 = no sea
    VkRenderingAttachmentInfo da{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = G.view,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clear,
    };
    VkRenderingInfo ri{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr, .flags = 0,
        .renderArea = { {0, 0}, G.extent },
        .layerCount = 1, .viewMask = 0,
        .colorAttachmentCount = 0, .pColorAttachments = nullptr,
        .pDepthAttachment = &da, .pStencilAttachment = nullptr,
    };
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{};
    vp.width = float(G.extent.width); vp.height = float(G.extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0, 0}, G.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, G.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
    const FrameView& fv = current_frame_view_;
    PushConstants pc{};
    pc.mvp = fv.vp;
    glm::mat4 proj = fv.proj;
    proj[2][2] = 0.0f;
    proj[3][2] = kNear;
    pc.model = glm::mat4(glm::dmat4(proj) * glm::dmat4(glm::dmat3(fv.view)));
    pc.prev_mvp = fv.vp;
    pc.color = glm::vec4(float(G.cols), float(G.rows), float(kBorder), float(kBorder));
    // (the height formula's constants, as the terrain's full-screen pass has them: the bed's depth)
    pc.emissive = glm::vec4(
        -std::clamp(rt_.terrain_raymarch_step_factor, 0.05f, 0.8f),
        std::max(1.0f, rt_.terrain_raymarch_lod_near_m * rm_air_lod_k_),
        std::max(rt_.terrain_raymarch_lod_near_m * rm_air_lod_k_ + 1.0f, rt_.terrain_raymarch_lod_far_m * rm_air_lod_k_),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_lod_min_octaves, 2, 8)));
    pc.tex_params = glm::vec4(
        static_cast<float>(std::clamp(rt_.terrain_raymarch_max_steps, 60, 300)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_shadow_steps, 16, 96)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_octaves, 4, 24)),
        static_cast<float>(std::clamp(rt_.terrain_raymarch_normal_octaves, 4, 32)));
    vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
    vkCmdDraw(cmd, uint32_t(G.cols) * uint32_t(G.rows) * 6u, 1, 0, 0);
    vkCmdEndRendering(cmd);
    vkinit::transition_image_aspect(cmd, G.image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    static const bool kLog = wg_env_on("VIBE_LOG_SEA_GRID");
    if (kLog && (frame_number_ % 120) == 100)
        log::infof("[sea-grid] %d x %d cells (%d px, %d stretched at the sides and bottom), %u x %u depth picture",
                   G.cols, G.rows, kCellPx, kBorder, G.extent.width, G.extent.height);
}

}  // namespace vibe
