// SKIN: light that goes under it comes out again a little further on - red furthest, blue
// least. As the reference game does it: a separable blur of the lit picture over the skin's
// pixels only (G-buffer material 4: shaders/people.glsl), as wide as a few millimetres are on
// the screen at that depth, held to the surface by the depth (a tap off the face does not
// count) and done on the light, not the colour (the lit colour over the albedo, blurred, times
// the albedo again: pores and freckles stay sharp). Two passes after the deferred lighting
// (frame.cpp, before the effects and the temporal filter):
//   across:  the scene colour -> the staging colour (the deferred pass's own target, free here)
//   down:    the staging colour -> the scene colour
// Every other pixel is left as it is (both passes discard it). Only while somebody is near enough
// for a face to be more than a few pixels (skin_blur_wanted). VIBE_NO_SKIN_BLUR=1: never.
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/taa.h"
#include "engine/vk_engine/state/multiplayer.h"
#include "engine/vk_pipelines.h"
#include "engine/vk_initializers.h"
#include "engine/log.h"

#include <cstdlib>

namespace vibe {

struct VulkanEngine::SkinBlurState {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kFrameOverlap][2]{};
    bool failed = false;
};

namespace {
struct SkinBlurPC { glm::vec4 dir; glm::vec4 proj; };
}

bool VulkanEngine::skin_blur_wanted() const {
    static const bool off = std::getenv("VIBE_NO_SKIN_BLUR") != nullptr;
    if (off) return false;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    for (const MpAvatar& a : npc_bodies_)
        if (a.id && a.drawn && glm::distance(a.eye, eye) < 25.0f) return true;
    if (mp_) for (const MpAvatar& a : mp_->avatars)
        if (a.id && a.drawn && glm::distance(a.eye, eye) < 25.0f) return true;
    return body_front_;
}

void VulkanEngine::render_skin_blur(VkCommandBuffer cmd) {
    if (!skin_blur_wanted() || !taa_ || !taa_->vert_module) return;
    if (!skin_blur_) skin_blur_ = new SkinBlurState();
    SkinBlurState& S = *skin_blur_;
    if (S.failed) return;
    if (!S.pipeline) {
        VkDescriptorSetLayoutBinding b[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        li.bindingCount = 3; li.pBindings = b;
        VkPushConstantRange pr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkinBlurPC) };
        VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pli.setLayoutCount = 1; pli.pSetLayouts = &S.set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pr;
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 2 * kFrameOverlap };
        VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pi.maxSets = 2 * kFrameOverlap; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        const std::string sd = VIBE_SHADER_DIR;
        S.frag = vkpipe::load_shader_module(device_, sd + "/skin_blur.frag.spv");
        if (!S.frag || vkCreateDescriptorSetLayout(device_, &li, nullptr, &S.set_layout) != VK_SUCCESS ||
            vkCreatePipelineLayout(device_, &pli, nullptr, &S.layout) != VK_SUCCESS ||
            vkCreateDescriptorPool(device_, &pi, nullptr, &S.pool) != VK_SUCCESS) {
            log::warnf("[skin] the skin blur could not be set up: none");
            S.failed = true;
            return;
        }
        VkDescriptorSetLayout lays[2 * kFrameOverlap];
        for (auto& l : lays) l = S.set_layout;
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = S.pool; ai.descriptorSetCount = 2 * kFrameOverlap; ai.pSetLayouts = lays;
        vkAllocateDescriptorSets(device_, &ai, &S.sets[0][0]);
        vkpipe::GraphicsPipelineConfig c{};
        c.vert = taa_->vert_module;
        c.frag = S.frag;
        c.layout = S.layout;
        c.color_format = scene_color_format_;
        c.depth_format = VK_FORMAT_UNDEFINED;
        c.cull = VK_CULL_MODE_NONE;
        c.depth_test = false;
        c.depth_write = false;
        S.pipeline = vkpipe::build_graphics_pipeline(device_, c);
        if (!S.pipeline) { S.failed = true; return; }
        log::info("[skin] the skin blur is ready");
    }
    // (this frame's two sets, pointed at the images as they are now: a resize makes new ones)
    VkDescriptorSet* fs = S.sets[frame_number_ % kFrameOverlap];
    for (int p = 0; p < 2; ++p) {
        VkDescriptorImageInfo ii[3] = {
            { linear_sampler_, p == 0 ? scene_color_view_ : staging_color_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { linear_sampler_, gbuffer0_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            { linear_sampler_, depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL } };
        VkWriteDescriptorSet w[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = fs[p]; w[i].dstBinding = i;
            w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[i].pImageInfo = &ii[i];
        }
        vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
    }
    const glm::mat4& P = current_frame_view_.proj;
    SkinBlurPC pc{};
    // proj: x, y turn a depth into metres (P[2][2], P[3][2]); z pixels a metre covers at 1 m; w the
    // width of the scatter, metres (VIBE_SKIN_BLUR_MM, default 9)
    static const float kWidth = (std::getenv("VIBE_SKIN_BLUR_MM") ? float(std::atof(std::getenv("VIBE_SKIN_BLUR_MM"))) : 9.0f) * 0.001f;
    pc.proj = glm::vec4(P[2][2], P[3][2], std::abs(P[1][1]) * 0.5f * float(render_extent_.height), kWidth);
    VkViewport vp{ 0.0f, 0.0f, float(render_extent_.width), float(render_extent_.height), 0.0f, 1.0f };
    VkRect2D sc{ { 0, 0 }, render_extent_ };
    auto pass = [&](VkImageView target, VkDescriptorSet set, glm::vec2 dir) {
        auto at = vkinit::color_attachment_info(target, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        ri.renderArea = sc; ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &at;
        vkCmdBeginRendering(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, S.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, S.layout, 0, 1, &set, 0, nullptr);
        pc.dir = glm::vec4(dir, 0.0f, 0.0f);
        vkCmdPushConstants(cmd, S.layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    };
    // across: scene colour -> staging (which the blit left as a transfer source)
    vkinit::transition_image(cmd, staging_color_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    pass(staging_color_view_, fs[0], glm::vec2(1.0f, 0.0f));
    // down: staging -> scene colour
    vkinit::transition_image(cmd, staging_color_image_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkinit::transition_image(cmd, scene_color_image_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    pass(scene_color_view_, fs[1], glm::vec2(0.0f, 1.0f));
    vkinit::transition_image(cmd, scene_color_image_, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    vkinit::transition_image(cmd, staging_color_image_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}

void VulkanEngine::destroy_skin_blur() {
    if (!skin_blur_) return;
    SkinBlurState& S = *skin_blur_;
    if (S.pipeline) vkDestroyPipeline(device_, S.pipeline, nullptr);
    if (S.layout) vkDestroyPipelineLayout(device_, S.layout, nullptr);
    if (S.pool) vkDestroyDescriptorPool(device_, S.pool, nullptr);
    if (S.set_layout) vkDestroyDescriptorSetLayout(device_, S.set_layout, nullptr);
    if (S.frag) vkDestroyShaderModule(device_, S.frag, nullptr);
    delete skin_blur_;
    skin_blur_ = nullptr;
}

}  // namespace vibe
