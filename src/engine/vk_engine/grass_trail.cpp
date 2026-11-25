// The trail: where people and cars have just been, as a texture round the player.
//
// Grass that is walked or driven through leans away and comes back up by itself. For that the
// plants have to know, cheaply and for any number of walkers, where something pushed and which
// way - also a moment AFTER it has passed. This is how Crimson Desert does it (its
// interaction.hlsl GenerateTrailInteractionCS, read from the compiled shader; this is our own
// code for it): a top-down texture that follows the player, drawn again every frame from the
// last one.
//
//   256 x 256 texels of 25 cm (a 64 m square), RGBA16F:
//     .xy  the direction things were pushed in (world x, z; unit length where .w > 0)
//     .z   how hard (0 a walk .. 1 a run or a car)
//     .w   how much the plants here are down (0 = standing, 1 = fully pushed over)
//   Every frame (grass_trail_update, before the world is drawn; shaders/grass_trail.frag):
//     - the old picture is shifted by however many whole texels the window moved,
//     - .z and .w fall off with a half-life of 0.56 s (exp2(-1.7847 dt), the game's rate),
//     - every "pusher" (the player on foot, cars near by) stamps a disc: direction outward
//       from its middle, turned toward the way it moves.
//
// Who reads it: scatter.vert (the grass tufts: bent about their root away from the push).
// The window's place goes to it in the plant draws' push constants (grass_trail_window).
// Scene set bindings: 41 = the trail (linear), 42 = last frame's (read only by grass_trail.frag).
//
// VIBE_NO_GRASS_TRAIL=1: nothing is stamped (the texture stays empty: plants stand).
// VIBE_TEST_TRAIL_R=<m>: the player's disc that wide (test shots with a standing camera).
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace vibe {

namespace {
constexpr uint32_t kTrailN     = 256;       // texels a side
constexpr float    kTrailTexel = 0.25f;     // metres a texel
constexpr VkFormat kTrailFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr int      kTrailPushers = 12;      // (three matrices of the push-constant block)
}

void VulkanEngine::init_grass_trail() {
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vk_check(vkCreateSampler(device_, &si, nullptr, &grass_trail_sampler_), "grass trail sampler");
    for (int l = 0; l < 2; ++l) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = kTrailFormat;
        ici.extent = { kTrailN, kTrailN, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        vk_check(vmaCreateImage(allocator_, &ici, &aci, &grass_trail_image_[l], &grass_trail_alloc_[l], nullptr), "grass trail image");
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = grass_trail_image_[l];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = kTrailFormat;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &grass_trail_view_[l]), "grass trail view");
        // (empty and readable from the start: the bindings must be valid before the first frame)
        vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
            vkinit::transition_image(cb, grass_trail_image_[l], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue zero{};
            VkImageSubresourceRange all{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, grass_trail_image_[l], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
            vkinit::transition_image(cb, grass_trail_image_[l], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
        VkDescriptorImageInfo ii{};
        ii.sampler = grass_trail_sampler_;
        ii.imageView = grass_trail_view_[l];
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_;
        w.dstBinding = 41 + uint32_t(l);
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    std::string sd = VIBE_SHADER_DIR;
    grass_trail_frag_ = vkpipe::load_shader_module(device_, sd + "/grass_trail.frag.spv");
    if (grass_trail_frag_) {
        vkpipe::GraphicsPipelineConfig cfg{};
        cfg.vert = taa_->vert_module;            // (the full-screen triangle)
        cfg.frag = grass_trail_frag_;
        cfg.layout = pipeline_layout_;
        cfg.color_format = kTrailFormat;
        cfg.cull = VK_CULL_MODE_NONE;
        cfg.depth_test = false;
        cfg.depth_write = false;
        grass_trail_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
    }
    if (!grass_trail_pipeline_) log::warnf("[grass-trail] shader or pipeline missing - plants will not bend");
    else log::info("grass trail built (64 m round the player, 25 cm a texel)");
}

void VulkanEngine::destroy_grass_trail() {
    if (grass_trail_pipeline_) vkDestroyPipeline(device_, grass_trail_pipeline_, nullptr);
    if (grass_trail_frag_)     vkDestroyShaderModule(device_, grass_trail_frag_, nullptr);
    if (grass_trail_sampler_)  vkDestroySampler(device_, grass_trail_sampler_, nullptr);
    for (int l = 0; l < 2; ++l) {
        if (grass_trail_view_[l])  vkDestroyImageView(device_, grass_trail_view_[l], nullptr);
        if (grass_trail_image_[l]) vmaDestroyImage(allocator_, grass_trail_image_[l], grass_trail_alloc_[l]);
        grass_trail_view_[l] = VK_NULL_HANDLE; grass_trail_image_[l] = VK_NULL_HANDLE; grass_trail_alloc_[l] = nullptr;
    }
    grass_trail_pipeline_ = VK_NULL_HANDLE; grass_trail_frag_ = VK_NULL_HANDLE; grass_trail_sampler_ = VK_NULL_HANDLE;
}

// The window as the plant shaders need it: xy = the world x, z of its corner, z = 1 / its side
// in metres, w = 1 when there is a trail to read.
glm::vec4 VulkanEngine::grass_trail_window() const {
    return glm::vec4(grass_trail_org_.x, grass_trail_org_.y, 1.0f / (float(kTrailN) * kTrailTexel),
                     grass_trail_pipeline_ ? 1.0f : 0.0f);
}

// Once a frame, before anything that reads the trail is drawn.
void VulkanEngine::grass_trail_update(VkCommandBuffer cmd) {
    if (grass_trail_pipeline_ == VK_NULL_HANDLE) return;
    static const bool off = std::getenv("VIBE_NO_GRASS_TRAIL") != nullptr;
    const float dt = std::clamp(anim_time_ - grass_trail_time_, 0.0f, 0.25f);
    grass_trail_time_ = anim_time_;

    // ---- The window: centred on the player, moved in whole texels (so the old picture is
    // shifted without being blurred) ----
    const glm::vec3 eye = player_.eye_position();
    const glm::vec2 prev_org = grass_trail_org_;
    const float half = 0.5f * float(kTrailN) * kTrailTexel;
    grass_trail_org_ = glm::floor((glm::vec2(eye.x, eye.z) - glm::vec2(half)) / kTrailTexel) * kTrailTexel;

    // ---- The pushers: xz, velocity xz; radius ----
    glm::vec4 push[kTrailPushers] = {};
    float     radius[8] = {};
    int n = 0;
    auto add = [&](glm::vec2 xz, glm::vec2 vel, float r) {
        if (n >= kTrailPushers) return;
        push[n] = glm::vec4(xz, vel);
        if (n < 8) radius[n] = r;
        ++n;
    };
    if (!off && globe::current().home) {
        // The player, on foot and on the ground (not in a car's seat, not in the air).
        const float feet = eye.y - 1.7f;
        if (!body_seated_ && feet - sample_terrain_height(eye.x, eye.z) < 1.0f)
        {
            // (VIBE_TEST_TRAIL_R=<m>: a wider disc, to see the plants round a standing camera go down)
            static const float test_r = std::getenv("VIBE_TEST_TRAIL_R") ? float(std::atof(std::getenv("VIBE_TEST_TRAIL_R"))) : 0.0f;
            add(glm::vec2(eye.x, eye.z), glm::vec2(player_.velocity.x, player_.velocity.z), test_r > 0.0f ? test_r : 0.75f);
        }
        // Cars near by: a disc under the body (their speed from where they were a frame ago
        // is not kept here: they push outward from their middle).
        std::vector<glm::vec3> cars;
        jeep_positions(cars);
        for (const glm::vec3& p : cars) {
            if (std::abs(p.x - eye.x) > half || std::abs(p.z - eye.z) > half) continue;
            if (p.y - sample_terrain_height(p.x, p.z) > 2.5f) continue;
            add(glm::vec2(p.x, p.z), glm::vec2(0.0f), 2.1f);
        }
    }

    // ---- Draw it: into [0], reading last frame's [1]; then keep a copy as the next "last" ----
    const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, CA = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    const VkImageLayout TS = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, TD = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkinit::transition_image(cmd, grass_trail_image_[0], RO, CA);
    VkClearValue cv{};
    VkRenderingAttachmentInfo att = vkinit::color_attachment_info(grass_trail_view_[0], &cv, CA);
    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea = { {0, 0}, { kTrailN, kTrailN } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{};
    vp.width = float(kTrailN); vp.height = float(kTrailN); vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0, 0}, { kTrailN, kTrailN } };
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, grass_trail_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
    PushConstants pc{};          // (as grass_trail.frag reads the block)
    for (int i = 0; i < 4; ++i) { pc.mvp[i] = push[i]; pc.model[i] = push[4 + i]; pc.prev_mvp[i] = push[8 + i]; }
    pc.color = glm::vec4(grass_trail_org_, prev_org);
    pc.emissive = glm::vec4(dt, float(n), kTrailTexel, float(kTrailN));
    pc.tex_params = glm::vec4(radius[0], radius[1], radius[2], radius[3]);
    pc.grass_params = glm::vec4(radius[4], radius[5], radius[6], radius[7]);
    vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    vkinit::transition_image(cmd, grass_trail_image_[0], CA, TS);
    vkinit::transition_image(cmd, grass_trail_image_[1], RO, TD);
    VkImageCopy r{};
    r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    r.extent = { kTrailN, kTrailN, 1 };
    vkCmdCopyImage(cmd, grass_trail_image_[0], TS, grass_trail_image_[1], TD, 1, &r);
    vkinit::transition_image(cmd, grass_trail_image_[1], TD, RO);
    vkinit::transition_image(cmd, grass_trail_image_[0], TS, RO);
}

}  // namespace vibe
