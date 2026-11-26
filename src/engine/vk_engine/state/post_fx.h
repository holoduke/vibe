#pragma once
// Post-processing after the reference game (post_fx.cpp): the metered exposure, the AgX look,
// local tone mapping, glare sprites, motion blur, depth of field, the colour-blind view - their
// state. Private to the files that implement it (vk_engine.h only holds the pointer pfx_).
// Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

#include <glm/glm.hpp>

namespace vibe {

// The parameter block shared with the shaders (shaders/post_fx.glsl PostFxData; std430).
struct PostFxGpu {
    float exposure, avg, target, valid;       // GPU-written (exposure.comp)
    float focus, trk_x, trk_z, trk_on;        // GPU-written (exposure.comp: the autofocus; track_stamp.comp: the tracks' window)
    glm::vec4 look, expo, expo2, expo3;
    glm::vec4 ltm, ltm2;
    glm::vec4 glare, glare2;
    glm::vec4 mb, dof;
    glm::mat4 cur_rot_ivp, prev_rot_vp;
    glm::vec4 proj;
    glm::vec4 fog0, fog1, fog2, fog3;
    glm::mat4 fog_vp;
    glm::vec4 sun_d;
};
static_assert(sizeof(PostFxGpu) % 16 == 0, "std430 vec4 block");

struct PfxImage {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;              // all mips (sampled)
    VkImageView mip[8]{};                           // one per mip (storage)
    VkExtent2D ext{ 0, 0 };
    uint32_t mips = 1;
};

// One compute pass: its set layout, pipeline layout and pipeline.
struct PfxPass {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
};

struct VulkanEngine::PostFxState {
    VkBuffer buf = VK_NULL_HANDLE;                  // PostFxGpu
    VmaAllocation buf_alloc = nullptr;
    VkBuffer rb_buf = VK_NULL_HANDLE;               // host copy of the GPU-written header (VIBE_LOG_EXPOSURE)
    VmaAllocation rb_alloc = nullptr;
    void* rb_ptr = nullptr;
    VkSampler sampler = VK_NULL_HANDLE;             // linear, clamp
    VkSampler point = VK_NULL_HANDLE;               // nearest, clamp
    VkDescriptorPool pool = VK_NULL_HANDLE;         // every set below (reset on resize)
    VkExtent2D out_ext{ 0, 0 };                     // the size the targets were made for
    VkImageView hist_src[2]{};                      // what compose reads as binding 0, per slot

    // Local tone mapping: per level, the three exposures' lumas (A) and weights (W), and the
    // fused luma (F); the guided fit at level 0 (C, sampled by compose binding 11).
    static constexpr int kLtmLevels = 6;
    PfxImage ltm_a, ltm_w, ltm_f, ltm_c;
    PfxPass p_ltm_lum, p_ltm_down, p_ltm_fuse, p_ltm_guide;
    VkDescriptorSet s_ltm_lum[2]{}, s_ltm_down[kLtmLevels]{}, s_ltm_fuse[kLtmLevels]{}, s_ltm_guide = VK_NULL_HANDLE;

    // The post colour P (output size, rgba16f): the history after depth of field and motion
    // blur, with the glare sprites added; compose reads it through its own sets (binding 0 = P).
    PfxImage post[2];                               // ping-pong
    VkDescriptorSet compose_sets[2]{};              // copies of compose's sets with binding 0 = post[0]
    bool post_used = false;                         // this frame compose reads P
    PfxPass p_copy;
    VkDescriptorSet s_copy[2]{};

    // Glare sprites.
    VkBuffer sprite_buf = VK_NULL_HANDLE;           // uint count + 3 pad, then kMaxSprites x 2 vec4
    VmaAllocation sprite_alloc = nullptr;
    PfxPass p_glare_find;
    VkDescriptorSet s_glare_find[2]{};
    VkShaderModule glare_vert = VK_NULL_HANDLE, glare_frag = VK_NULL_HANDLE;
    VkPipelineLayout glare_draw_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout glare_draw_dsl = VK_NULL_HANDLE;
    VkDescriptorSet s_glare_draw = VK_NULL_HANDLE;
    VkPipeline glare_draw_pipe = VK_NULL_HANDLE;
    PfxImage glare_hold;                            // per 8x8 block: kept brightness (temporal hold)

    // Motion blur.
    PfxImage mb_tile, mb_neigh;
    PfxPass p_mb_tile, p_mb_neigh, p_mb_gather;
    VkDescriptorSet s_mb_tile = VK_NULL_HANDLE, s_mb_neigh = VK_NULL_HANDLE;
    VkDescriptorSet s_mb_gather[3]{};               // source: history slot 0, 1, or post[1] (after depth of field)

    // Depth of field (half size).
    PfxImage dof_half, dof_blur, dof_near;
    PfxPass p_dof_pre, p_dof_gather, p_dof_comb;
    VkDescriptorSet s_dof_pre[2]{}, s_dof_gather = VK_NULL_HANDLE;
    VkDescriptorSet s_dof_comb[2][2]{};             // [history slot][destination post image]

    // Tracks in snow and sand (track_stamp.comp, track_rim.comp; scene binding 55).
    static constexpr uint32_t kTrkN = 1024;
    static constexpr float kTrkTexel = 0.0625f;
    PfxImage trk_c, trk_tmp;
    PfxPass p_trk_stamp, p_trk_rim;
    VkDescriptorSet s_trk_stamp = VK_NULL_HANDLE, s_trk_rim = VK_NULL_HANDLE;
    VkBuffer trk_buf = VK_NULL_HANDLE;
    VmaAllocation trk_alloc = nullptr;
    glm::vec2 trk_org{ 0.0f };
    bool trk_have = false;

    // CPU side.
    float enclosed = 0.0f;                          // smoothed share of the up-rays that hit within reach
    double last_t = 0.0;
    glm::mat4 prev_rot_vp{ 1.0f };
    bool prev_valid = false;
    int log_every = 0;                              // VIBE_LOG_EXPOSURE
    bool fog_bound = false;                         // the fog volumes are in the scene set (56..59: fx_fog.glsl)
    bool fog_bound_any = false;                     // (written once, with or without the fog)
};

} // namespace vibe
