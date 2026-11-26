// Small-object sun shadows: flowers, ferns and scatter rocks rendered
// (depth only) into a dedicated 2048^2 map covering 64 m around the
// camera. Too many instances for the ray-tracing TLAS, and the main sun
// shadow map also contains the terrain (a terrain receiver would need a
// large depth bias that erases 30 cm flower shadows). Sampled by
// deferred_lighting.frag (binding 14) for terrain receivers.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/clutter_shadow.h"
#include "engine/vk_engine/state/multiplayer.h"

#include <algorithm>
#include "engine/log.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

#ifndef VIBE_SHADER_DIR
#define VIBE_SHADER_DIR "shaders"
#endif

namespace vibe {

namespace {
constexpr uint32_t kClutterDim  = 2048;
constexpr float    kClutterHalf = 32.0f;   // metres (layer 0: 3.1 cm texels)
constexpr float    kClutterDepth = 60.0f;  // +- along the sun axis
// Layer 1: near cascade, same centre and depth range, 4x finer (0.78 cm
// texels over 16 m) - thin stems and petals survive the 3x3 PCF. The
// ratio is an exact power of two so the far map's texel snap (3.125 cm)
// is exactly 4 near texels: no extra snap, no shimmer, and the shader
// derives the near coordinates as far NDC * kClutterNearScale.
constexpr float    kClutterNearScale = 4.0f;
}

void VulkanEngine::init_clutter_shadow() {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_D32_SFLOAT;
    ici.extent = { kClutterDim, kClutterDim, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 5;   // + 3, 4 = lantern atlas (4 lanterns each)
    //   // + 2 = flashlight view (small plants)
    //   // 0 = 64 m, 1 = 16 m near cascade
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &clutter_->shadow_image, &clutter_->shadow_alloc, nullptr),
             "clutter shadow image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = clutter_->shadow_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_->shadow_view), "clutter shadow view");
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 1, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_->shadow_near_view), "clutter near view");
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 2, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_->shadow_flash_view), "clutter flash view");
    for (uint32_t k = 0; k < 2; ++k) {
        vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 3 + k, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_->shadow_lamp_view[k]), "clutter lamp view");
    }
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 5 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_->shadow_array_view), "clutter array view");
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &clutter_->shadow_sampler), "clutter shadow sampler");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkinit::transition_image_aspect(cb, clutter_->shadow_image, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_ASPECT_DEPTH_BIT);
    });

    VkPushConstantRange pc{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) + 3 * sizeof(glm::vec4) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &clutter_->shadow_layout), "clutter layout");
    clutter_->shadow_vert = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/clutter_shadow.vert.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = clutter_->shadow_vert;
    cfg.frag = VK_NULL_HANDLE;
    cfg.layout = clutter_->shadow_layout;
    cfg.color_attachment_count = 0;
    cfg.depth_format = VK_FORMAT_D32_SFLOAT;
    cfg.cull = VK_CULL_MODE_NONE;
    cfg.depth_test = true;
    cfg.depth_write = true;
    cfg.depth_compare = VK_COMPARE_OP_LESS;
    VkVertexInputBindingDescription b0{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputBindingDescription b1{ 1, sizeof(ScatterInstance), VK_VERTEX_INPUT_RATE_INSTANCE };
    cfg.vbindings = { b0, b1 };
    VkVertexInputAttributeDescription a0{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, position)) };
    VkVertexInputAttributeDescription a2{ 2, 0, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(Vertex, uv)) };
    VkVertexInputAttributeDescription a3{ 3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0 };
    VkVertexInputAttributeDescription a4{ 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
    cfg.vattrs = { a0, a2, a3, a4 };
    clutter_->shadow_pipeline = vkpipe::build_graphics_pipeline(device_, cfg);
    // Skinned characters: two vertex streams (rest pose; bones and weights), the scene set for
    // the bone palette.
    skin_shadow_vert_ = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/skin_shadow.vert.spv");
    if (skin_shadow_vert_ && pipeline_layout_) {
        vkpipe::GraphicsPipelineConfig sc = cfg;
        sc.vert = skin_shadow_vert_;
        sc.layout = pipeline_layout_;
        VkVertexInputBindingDescription s1{ 1, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
        sc.vbindings = { b0, s1 };
        VkVertexInputAttributeDescription k3{ 3, 1, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, position)) };
        VkVertexInputAttributeDescription k4{ 4, 1, VK_FORMAT_R32G32B32_SFLOAT, uint32_t(offsetof(Vertex, normal)) };
        VkVertexInputAttributeDescription k5{ 5, 1, VK_FORMAT_R32G32_SFLOAT, uint32_t(offsetof(Vertex, uv)) };
        sc.vattrs = { a0, k3, k4, k5 };
        skin_shadow_pipeline_ = vkpipe::build_graphics_pipeline(device_, sc);
    }
    // The grass placed on the GPU: the plants' instance layout, the colour pass's blade maths.
    clutter_->grass_vert = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/tuft_shadow.vert.spv");
    if (clutter_->grass_vert && pipeline_layout_) {
        vkpipe::GraphicsPipelineConfig gc = cfg;
        gc.vert = clutter_->grass_vert;
        gc.layout = pipeline_layout_;
        clutter_->grass_pipeline = vkpipe::build_graphics_pipeline(device_, gc);
    }
    {
        VkDescriptorImageInfo ii{ clutter_->shadow_sampler, clutter_->shadow_array_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 28; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    clutter_->shadow_vp = glm::mat4(0.0f);
    log::info("[clutter shadow] 2x 2048^2 cascades: 64 m (3.1 cm) + 16 m near (0.8 cm)");
}

void VulkanEngine::destroy_clutter_shadow() {
    if (skin_shadow_pipeline_) vkDestroyPipeline(device_, skin_shadow_pipeline_, nullptr);
    if (skin_shadow_vert_) vkDestroyShaderModule(device_, skin_shadow_vert_, nullptr);
    skin_shadow_pipeline_ = VK_NULL_HANDLE;
    skin_shadow_vert_ = VK_NULL_HANDLE;
    if (clutter_->grass_pipeline) vkDestroyPipeline(device_, clutter_->grass_pipeline, nullptr);
    if (clutter_->grass_vert) vkDestroyShaderModule(device_, clutter_->grass_vert, nullptr);
    clutter_->grass_pipeline = VK_NULL_HANDLE; clutter_->grass_vert = VK_NULL_HANDLE;
    deferred_desc_sig_ = 0;   // handles may be recreated (even at the same address)
    if (clutter_->shadow_pipeline) vkDestroyPipeline(device_, clutter_->shadow_pipeline, nullptr);
    if (clutter_->shadow_layout) vkDestroyPipelineLayout(device_, clutter_->shadow_layout, nullptr);
    if (clutter_->shadow_vert) vkDestroyShaderModule(device_, clutter_->shadow_vert, nullptr);
    if (clutter_->shadow_view) vkDestroyImageView(device_, clutter_->shadow_view, nullptr);
    if (clutter_->shadow_near_view) vkDestroyImageView(device_, clutter_->shadow_near_view, nullptr);
    if (clutter_->shadow_flash_view) vkDestroyImageView(device_, clutter_->shadow_flash_view, nullptr);
    clutter_->shadow_flash_view = VK_NULL_HANDLE;
    for (auto& v : clutter_->shadow_lamp_view) { if (v) vkDestroyImageView(device_, v, nullptr); v = VK_NULL_HANDLE; }
    if (clutter_->shadow_array_view) vkDestroyImageView(device_, clutter_->shadow_array_view, nullptr);
    clutter_->shadow_near_view = clutter_->shadow_array_view = VK_NULL_HANDLE;
    if (clutter_->shadow_sampler) vkDestroySampler(device_, clutter_->shadow_sampler, nullptr);
    clutter_->shadow_sampler = VK_NULL_HANDLE;
    if (clutter_->shadow_image) vmaDestroyImage(allocator_, clutter_->shadow_image, clutter_->shadow_alloc);
    clutter_->shadow_pipeline = VK_NULL_HANDLE; clutter_->shadow_layout = VK_NULL_HANDLE;
    clutter_->shadow_vert = VK_NULL_HANDLE; clutter_->shadow_view = VK_NULL_HANDLE;
    clutter_->shadow_image = VK_NULL_HANDLE;
}

// Computed in update_scene_ubo() (before any pass) so the deferred pass
// AND the raymarched terrain (via the scene UBO) use the same matrix the
// map is rendered with this frame.
static const bool kNoClutter = std::getenv("VIBE_NO_CLUTTER_SHADOW") != nullptr;   // A/B test hook
void VulkanEngine::update_clutter_shadow_vp() {
    // Every other frame (in turn with the sun's shadow map); every frame while the flashlight
    // is on - its shadows move with the hand.
    clutter_->shadow_skip = (frame_number_ & 1) == 0 && !flashlight_on_ && clutter_->shadow_vp[3][3] > 0.5f;
    if (clutter_->shadow_skip) return;
    if (!clutter_->shadow_pipeline || scatter_.instance_count == 0 || kNoClutter) {
        clutter_->shadow_vp = glm::mat4(0.0f);
        return;
    }
    // Light VP: ortho box around the camera along the sun, texel-snapped.
    float p_rad = glm::radians(rt_.sun_pitch_deg), y_rad = glm::radians(rt_.sun_yaw_deg);
    glm::vec3 sun = glm::normalize(glm::vec3(std::sin(y_rad) * std::cos(p_rad), std::sin(p_rad),
                                             std::cos(y_rad) * std::cos(p_rad)));
    glm::vec3 up_ref = std::abs(sun.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    glm::vec3 rx = glm::normalize(glm::cross(up_ref, sun)), ry = glm::normalize(glm::cross(sun, rx));
    glm::vec3 cam = current_frame_view_.eye_pos;
    const float texel = 2.0f * kClutterHalf / float(kClutterDim);
    float u = std::round(glm::dot(cam, rx) / texel) * texel;
    float v = std::round(glm::dot(cam, ry) / texel) * texel;
    glm::vec3 c = rx * u + ry * v + sun * glm::dot(cam, sun);
    glm::mat4 view = glm::lookAt(c + sun * kClutterDepth, c, ry);
    glm::mat4 proj = glm::ortho(-kClutterHalf, kClutterHalf, -kClutterHalf, kClutterHalf, 0.0f, 2.0f * kClutterDepth);
    proj[1][1] *= -1.0f;
    clutter_->shadow_vp = proj * view;
}

void VulkanEngine::render_clutter_shadow(VkCommandBuffer cmd) {
    if (clutter_->shadow_vp[3][3] < 0.5f || clutter_->shadow_skip) return;
    const glm::vec3 cam = current_frame_view_.eye_pos;

    vkinit::transition_image_aspect(cmd, clutter_->shadow_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    // Two cascades into the two array layers: 64 m (layer 0) and the 4x
    // finer 16 m near cascade (layer 1, same centre: VP scaled in x/y).
    glm::mat4 near_scale(1.0f);
    near_scale[0][0] = near_scale[1][1] = kClutterNearScale;
    // Layer 2: the same plants seen from the flashlight (perspective), for
    // their shadows in its beam.
    // Pass 3: one road lantern's fisheye map into its atlas quarter.
    const bool flash_pass = flash_vp_[3][3] != 0.0f || flash_vp_[2][3] != 0.0f;
    const bool lamp_pass = lamp_map_slot_ >= 0 && size_t(lamp_map_slot_) < lamp_lights_.size() && lamp_map_slot_ < 8;
    const glm::vec3 lamp_pos = lamp_pass ? lamp_lights_[size_t(lamp_map_slot_)] : glm::vec3(0.0f);
    const uint32_t kQ = kClutterDim / 2;
    const glm::vec3 view_cam = cam;
    for (int layer = 0; layer < 4; ++layer) {
        if ((layer == 2 && !flash_pass) || (layer == 3 && !lamp_pass)) continue;
        const glm::vec3 cam = layer == 3 ? lamp_pos : view_cam;      // centre of this pass
        const float half = layer == 0 ? kClutterHalf : layer == 1 ? kClutterHalf / kClutterNearScale
                         : layer == 2 ? 26.0f : 10.3f;
        const glm::mat4 cvp = layer == 0 ? clutter_->shadow_vp : layer == 1 ? near_scale * clutter_->shadow_vp : flash_vp_;
        VkRenderingAttachmentInfo da{};
        da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        da.imageView = layer == 0 ? clutter_->shadow_view : layer == 1 ? clutter_->shadow_near_view
                     : layer == 2 ? clutter_->shadow_flash_view : clutter_->shadow_lamp_view[lamp_map_slot_ >> 2];
        da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        da.clearValue.depthStencil = { 1.0f, 0 };
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        const int32_t qx0 = layer == 3 ? int32_t((lamp_map_slot_ & 1) * kQ) : 0, qy0 = layer == 3 ? int32_t(((lamp_map_slot_ >> 1) & 1) * kQ) : 0;
        const uint32_t qdim = layer == 3 ? kQ : kClutterDim;
        ri.renderArea = { { qx0, qy0 }, { qdim, qdim } };   // (the clear covers only this area)
        ri.layerCount = 1;
        ri.pDepthAttachment = &da;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{ float(qx0), float(qy0), float(qdim), float(qdim), 0.0f, 1.0f };
        VkRect2D sc{ { qx0, qy0 }, { qdim, qdim } };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, clutter_->shadow_pipeline);
        // p: .x wind time, .y plant (sway) flag, .z colour-pass draw distance -
        // the same fade / sway as scatter.vert so shadows match what is drawn.
        struct { glm::mat4 vp; glm::vec4 cam; glm::vec4 p; glm::vec4 lamp; } pc{ cvp, glm::vec4(cam, half * 1.42f),
            glm::vec4(anim_time_, 0.0f,
                      std::max(60.0f, rt_.grass_distance * 1.6f), 0.0f),
            layer == 3 ? glm::vec4(lamp_pos, 14.0f) : glm::vec4(0.0f) };
        vkCmdPushConstants(cmd, clutter_->shadow_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        VkDeviceSize offs[2] = { 0, 0 };
        // Only cells overlapping this cascade's box around the camera (XZ).
        const float box_r = half * 1.42f;
        auto in_box = [&](const ScatterCell& c) {
            float qx = std::clamp(cam.x, c.mn.x, c.mx.x), qz = std::clamp(cam.z, c.mn.z, c.mx.z);
            return (qx - cam.x) * (qx - cam.x) + (qz - cam.z) * (qz - cam.z) < box_r * box_r;
        };
        {
            VkBuffer b[2] = { scatter_.rock_mesh.vertex_buffer, scatter_.instance_buffer };
            vkCmdBindVertexBuffers(cmd, 0, 2, b, offs);
            vkCmdBindIndexBuffer(cmd, scatter_.rock_mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            scatter_draw_cells(cmd, scatter_.rock_mesh.index_count, scatter_.rock_cells, in_box);
        }
        if (scatter_.plant_buffer) {
            pc.p.y = 1.0f;
            pc.p.w = rt_.terrain_raymarch_enabled ? 1.0f : 0.0f;
            pc.p.z = std::max(45.0f, rt_.grass_distance * 0.9f);
            vkCmdPushConstants(cmd, clutter_->shadow_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
                if (!scatter_.plant_count[sp]) continue;
                if (sp == 16) continue;          // water lilies lie flat on the water: no shadow pass
                pc.p.z = plant_max_d(sp);   // same fade as the colour pass
                vkCmdPushConstants(cmd, clutter_->shadow_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
                // The same levels of detail as the colour pass (by the cell's distance from the eye).
                const Mesh* lods[3] = { &scatter_.plant_mesh[sp], &scatter_.plant_mesh_far[sp], &scatter_.plant_mesh_far2[sp] };
                if (!lods[1]->vertex_buffer) lods[1] = lods[0];
                if (!lods[2]->vertex_buffer) lods[2] = lods[1];
                const glm::vec3 eye_s = current_frame_view_.eye_pos;
                auto band = [&](const ScatterCell& c) {
                    const glm::vec3 q = glm::clamp(eye_s, c.mn, c.mx);
                    const float d2 = glm::dot(q - eye_s, q - eye_s);
                    return d2 < ScatterSet::kPlantFarLod * ScatterSet::kPlantFarLod ? 0
                         : (d2 < ScatterSet::kPlantFarLod2 * ScatterSet::kPlantFarLod2 ? 1 : 2);
                };
                for (int l = 0; l < 3; ++l) {
                    VkBuffer b[2] = { lods[l]->vertex_buffer, scatter_.plant_buffer };
                    vkCmdBindVertexBuffers(cmd, 0, 2, b, offs);
                    vkCmdBindIndexBuffer(cmd, lods[l]->index_buffer, 0, VK_INDEX_TYPE_UINT32);
                    scatter_draw_cells(cmd, lods[l]->index_count, scatter_.plant_cells[sp],
                                       [&](const ScatterCell& c) { return band(c) == l && in_box(c); });
                }
            }
        }
        // The grass placed on the GPU (long tufts, turf): from the lists the placement filled
        // this frame - what the camera sees, so grass out of view casts no shadow into it.
        // VIBE_NO_GRASS_SHADOW=1: none; VIBE_GRASS_SHADOW=1: the long tufts only, 2: the near map only.
        static const int kGrassSh = std::getenv("VIBE_NO_GRASS_SHADOW") ? 0 : std::getenv("VIBE_GRASS_SHADOW") ? std::atoi(std::getenv("VIBE_GRASS_SHADOW")) : 3;
        if (layer <= 2 && kGrassSh && clutter_->grass_pipeline && !(kGrassSh == 2 && layer != 1)) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, clutter_->grass_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
            PushConstants gpc{};
            gpc.mvp = cvp;
            gpc.model[0] = glm::vec4(cam, half * 1.42f);
            gpc.grass_params = glm::vec4(0.0f, 1.0f, anim_time_, rt_.terrain_raymarch_enabled ? 1.0f : 0.0f);
            gpc.emissive = grass_trail_window();
            gpc.color = glm::vec4(0.871f, 0.490f, glm::clamp(rt_.grass_wind * 12.5f, 0.0f, 1.0f), 0.0f);   // (as world_render.cpp)
            for (int sp : { 15, 17, 18 }) {
                if (sp != 15 && kGrassSh == 1) continue;
                if (sp != 15 && layer == 0) continue;        // (turf and meadow: in the near map only - in the 64 m map their blades are thinner than a texel and cost a millisecond)
                const float* s = sp == 15 ? ScatterSet::kTuftShape : sp == 17 ? ScatterSet::kTurfShape : ScatterSet::kMeadowShape;
                gpc.tex_params = glm::vec4(s[0], s[1], s[2], s[3]);
                vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &gpc);
                draw_plants_gpu(cmd, sp);
            }
            // The small debris (species 19..26, ground_fx): small contact shadows, in the near map only.
            if (layer == 1 && rt_.ground_debris)
                for (int sp = 19; sp <= 26; ++sp) draw_plants_gpu(cmd, sp);
        }
        // The soldiers: in the sun's two maps, and in the flashlight's (not the one holding it).
        if (layer <= 2) draw_skinned_shadows(cmd, cvp, layer != 2);
        vkCmdEndRendering(cmd);
    }
    vkinit::transition_image_aspect(cmd, clutter_->shadow_image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
}

// Every soldier with a pose this frame, depth only, into the shadow map being rendered: the
// player's own body (with the head it is otherwise drawn without: a shadow has one), the other
// players and the fallen. One level of detail below what is drawn; a shadow does not show it.
void VulkanEngine::draw_skinned_shadows(VkCommandBuffer cmd, const glm::mat4& light_vp, bool with_local) {
    static const bool kOff = std::getenv("VIBE_NO_BODY_SHADOW") != nullptr;
    if (kOff || !skin_shadow_pipeline_ || !body_lod_[0].index_count || !body_palette_map_) return;
    bool bound = false;
    auto draw = [&](const glm::mat4& model, int palette, int lod) {
        if (palette < 0) return;
        if (!bound) {
            bound = true;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skin_shadow_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &scene_desc_set_, 0, nullptr);
        }
        PushConstants pc{};
        pc.mvp = light_vp * model;
        pc.model = model;
        pc.grass_params = glm::vec4(0.0f, float(palette), 0.0f, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
        lod = glm::clamp(lod + 1, 1, kBodyLods - 1);
        for (const SkinMesh* set : { body_lod_, head_lod_ }) {
            const SkinMesh* m = nullptr;
            for (int i = lod; i >= 0 && !m; --i) if (set[i].index_count) m = &set[i];
            if (!m) continue;
            const VkBuffer streams[2] = { m->rest.vertex_buffer, m->skin.vertex_buffer };
            const VkDeviceSize offs[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, streams, offs);
            vkCmdBindIndexBuffer(cmd, m->rest.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, m->index_count, 1, 0, 0, 0);
        }
    };
    if (with_local && body_drawn_) draw(body_model_, body_palette_, 0);
    if (net_)
        for (const MpAvatar& a : mp_->avatars)
            if (a.id && a.drawn) draw(a.model, a.palette, a.mesh_lod);
    for (const Corpse& c : corpses_)
        if (c.rag && c.av.drawn) draw(c.av.model, c.av.palette, c.av.mesh_lod);
    // The NPC engine's people: by distance - to 45 m the mesh they are drawn with one level
    // down, to 90 m the coarsest, beyond that none (a shadow a few pixels long).
    {
        const glm::vec3 eye = current_frame_view_.eye_pos;
        for (const MpAvatar& a : npc_bodies_) {
            if (!a.id || !a.drawn) continue;
            const float d = glm::distance(glm::vec3(a.model[3]), eye);
            if (d > 90.0f) continue;
            draw(a.model, a.palette, d < 45.0f ? a.mesh_lod : kBodyLods - 1);
        }
    }
}

} // namespace vibe

