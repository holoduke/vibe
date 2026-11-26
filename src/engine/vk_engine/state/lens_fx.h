#pragma once
// Camera weather and heat (vk_engine/lens_fx.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (lens_fx_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::LensFxState {
    // Rain on the lens: water on the glass, r = thickness, g = wet trail (shaders/lens_fx.comp
    // modes 1 and 2 go A -> B -> A each frame; compose.frag reads A).
    AtmoImage lens_a, lens_b;
    uint32_t  lens_w = 1024, lens_h = 576;
    // Heat shimmer: screen offsets (uv) at half the render resolution, written when anything
    // hot is in view (mode 3), read by compose.frag.
    AtmoImage shimmer;
    uint32_t  sh_w = 0, sh_h = 0;
    bool      shimmer_on = false;      // drawn this frame (compose skips it otherwise)
    // The block shared by this pass and compose.frag (binding 12 there): the moon, the lens,
    // the hot sources. Written with vkCmdUpdateBuffer each frame.
    VkBuffer      ubo = VK_NULL_HANDLE;
    VmaAllocation ubo_alloc = nullptr;
    // x: how much open sky the lens sees (eased on the GPU).
    VkBuffer      state = VK_NULL_HANDLE;
    VmaAllocation state_alloc = nullptr;
    VkSampler     sampler = VK_NULL_HANDLE;      // linear, clamp
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkDescriptorSet       set = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkAccelerationStructureKHR tlas_bound = VK_NULL_HANDLE;
    VkImageView           depth_bound = VK_NULL_HANDLE;
    double   last_t = -1.0;
    float    drop_acc = 0.0f;          // drops owed (rain x time)
    float    wet_left = 0.0f;          // seconds the lens passes keep running after the rain
    uint32_t frame = 0;
};

} // namespace vibe
