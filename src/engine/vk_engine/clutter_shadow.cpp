// Small-object sun shadows: flowers, ferns and scatter rocks rendered
// (depth only) into a dedicated 2048^2 map covering 64 m around the
// camera. Too many instances for the ray-tracing TLAS, and the main sun
// shadow map also contains the terrain (a terrain receiver would need a
// large depth bias that erases 30 cm flower shadows). Sampled by
// deferred_lighting.frag (binding 14) for terrain receivers.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"

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
    vk_check(vmaCreateImage(allocator_, &ici, &aci, &clutter_shadow_image_, &clutter_shadow_alloc_, nullptr),
             "clutter shadow image");
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = clutter_shadow_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_shadow_view_), "clutter shadow view");
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 1, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_shadow_near_view_), "clutter near view");
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 2, 1 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_shadow_flash_view_), "clutter flash view");
    for (uint32_t k = 0; k < 2; ++k) {
        vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 3 + k, 1 };
        vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_shadow_lamp_view_[k]), "clutter lamp view");
    }
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 5 };
    vk_check(vkCreateImageView(device_, &vci, nullptr, &clutter_shadow_array_view_), "clutter array view");
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 0.0f;
        vk_check(vkCreateSampler(device_, &si, nullptr, &clutter_shadow_sampler_), "clutter shadow sampler");
    }
    vkinit::one_time_submit(device_, graphics_queue_, graphics_queue_family_, [&](VkCommandBuffer cb) {
        vkinit::transition_image_aspect(cb, clutter_shadow_image_, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_IMAGE_ASPECT_DEPTH_BIT);
    });

    VkPushConstantRange pc{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) + 3 * sizeof(glm::vec4) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
    vk_check(vkCreatePipelineLayout(device_, &plci, nullptr, &clutter_shadow_layout_), "clutter layout");
    clutter_shadow_vert_ = vkpipe::load_shader_module(device_, std::string(VIBE_SHADER_DIR) + "/clutter_shadow.vert.spv");

    vkpipe::GraphicsPipelineConfig cfg{};
    cfg.vert = clutter_shadow_vert_;
    cfg.frag = VK_NULL_HANDLE;
    cfg.layout = clutter_shadow_layout_;
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
    clutter_shadow_pipeline_ = vkpipe::build_graphics_pipeline(device_, cfg);
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
    {
        VkDescriptorImageInfo ii{ clutter_shadow_sampler_, clutter_shadow_array_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = scene_desc_set_; w.dstBinding = 28; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }
    clutter_shadow_vp_ = glm::mat4(0.0f);
    log::info("[clutter shadow] 2x 2048^2 cascades: 64 m (3.1 cm) + 16 m near (0.8 cm)");
}

void VulkanEngine::destroy_clutter_shadow() {
    if (skin_shadow_pipeline_) vkDestroyPipeline(device_, skin_shadow_pipeline_, nullptr);
    if (skin_shadow_vert_) vkDestroyShaderModule(device_, skin_shadow_vert_, nullptr);
    skin_shadow_pipeline_ = VK_NULL_HANDLE;
    skin_shadow_vert_ = VK_NULL_HANDLE;
    deferred_desc_sig_ = 0;   // handles may be recreated (even at the same address)
    if (clutter_shadow_pipeline_) vkDestroyPipeline(device_, clutter_shadow_pipeline_, nullptr);
    if (clutter_shadow_layout_) vkDestroyPipelineLayout(device_, clutter_shadow_layout_, nullptr);
    if (clutter_shadow_vert_) vkDestroyShaderModule(device_, clutter_shadow_vert_, nullptr);
    if (clutter_shadow_view_) vkDestroyImageView(device_, clutter_shadow_view_, nullptr);
    if (clutter_shadow_near_view_) vkDestroyImageView(device_, clutter_shadow_near_view_, nullptr);
    if (clutter_shadow_flash_view_) vkDestroyImageView(device_, clutter_shadow_flash_view_, nullptr);
    clutter_shadow_flash_view_ = VK_NULL_HANDLE;
    for (auto& v : clutter_shadow_lamp_view_) { if (v) vkDestroyImageView(device_, v, nullptr); v = VK_NULL_HANDLE; }
    if (clutter_shadow_array_view_) vkDestroyImageView(device_, clutter_shadow_array_view_, nullptr);
    clutter_shadow_near_view_ = clutter_shadow_array_view_ = VK_NULL_HANDLE;
    if (clutter_shadow_sampler_) vkDestroySampler(device_, clutter_shadow_sampler_, nullptr);
    clutter_shadow_sampler_ = VK_NULL_HANDLE;
    if (clutter_shadow_image_) vmaDestroyImage(allocator_, clutter_shadow_image_, clutter_shadow_alloc_);
    clutter_shadow_pipeline_ = VK_NULL_HANDLE; clutter_shadow_layout_ = VK_NULL_HANDLE;
    clutter_shadow_vert_ = VK_NULL_HANDLE; clutter_shadow_view_ = VK_NULL_HANDLE;
    clutter_shadow_image_ = VK_NULL_HANDLE;
}

// Computed in update_scene_ubo() (before any pass) so the deferred pass
// AND the raymarched terrain (via the scene UBO) use the same matrix the
// map is rendered with this frame.
static const bool kNoClutter = std::getenv("VIBE_NO_CLUTTER_SHADOW") != nullptr;   // A/B test hook
void VulkanEngine::update_clutter_shadow_vp() {
    // Every other frame (in turn with the sun's shadow map); every frame while the flashlight
    // is on - its shadows move with the hand.
    clutter_shadow_skip_ = (frame_number_ & 1) == 0 && !flashlight_on_ && clutter_shadow_vp_[3][3] > 0.5f;
    if (clutter_shadow_skip_) return;
    if (!clutter_shadow_pipeline_ || scatter_.instance_count == 0 || kNoClutter) {
        clutter_shadow_vp_ = glm::mat4(0.0f);
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
    clutter_shadow_vp_ = proj * view;
}

void VulkanEngine::render_clutter_shadow(VkCommandBuffer cmd) {
    if (clutter_shadow_vp_[3][3] < 0.5f || clutter_shadow_skip_) return;
    const glm::vec3 cam = current_frame_view_.eye_pos;

    vkinit::transition_image_aspect(cmd, clutter_shadow_image_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
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
        const glm::mat4 cvp = layer == 0 ? clutter_shadow_vp_ : layer == 1 ? near_scale * clutter_shadow_vp_ : flash_vp_;
        VkRenderingAttachmentInfo da{};
        da.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        da.imageView = layer == 0 ? clutter_shadow_view_ : layer == 1 ? clutter_shadow_near_view_
                     : layer == 2 ? clutter_shadow_flash_view_ : clutter_shadow_lamp_view_[lamp_map_slot_ >> 2];
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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, clutter_shadow_pipeline_);
        // p: .x wind time, .y plant (sway) flag, .z colour-pass draw distance -
        // the same fade / sway as scatter.vert so shadows match what is drawn.
        struct { glm::mat4 vp; glm::vec4 cam; glm::vec4 p; glm::vec4 lamp; } pc{ cvp, glm::vec4(cam, half * 1.42f),
            glm::vec4(anim_time_, 0.0f,
                      std::max(60.0f, rt_.grass_distance * 1.6f), 0.0f),
            layer == 3 ? glm::vec4(lamp_pos, 14.0f) : glm::vec4(0.0f) };
        vkCmdPushConstants(cmd, clutter_shadow_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
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
            vkCmdPushConstants(cmd, clutter_shadow_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
                if (!scatter_.plant_count[sp]) continue;
                if (sp == 16) continue;          // water lilies lie flat on the water: no shadow pass
                pc.p.z = plant_max_d(sp);   // same fade as the colour pass
                vkCmdPushConstants(cmd, clutter_shadow_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
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
        // The soldiers: in the sun's two maps, and in the flashlight's (not the one holding it).
        if (layer <= 2) draw_skinned_shadows(cmd, cvp, layer != 2);
        vkCmdEndRendering(cmd);
    }
    vkinit::transition_image_aspect(cmd, clutter_shadow_image_, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
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
        for (const MpAvatar& a : mp_avatars_)
            if (a.id && a.drawn) draw(a.model, a.palette, a.mesh_lod);
    for (const Corpse& c : corpses_)
        if (c.rag && c.av.drawn) draw(c.av.model, c.av.palette, c.av.mesh_lod);
}

} // namespace vibe

