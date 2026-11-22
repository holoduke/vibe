#pragma once
// The clutter's shadow map (clutter_shadow.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (clutter_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::ClutterShadowState {
    // Lantern small-plant shadow maps (atlas quarters in layers 3 / 4 of the
    // small-object shadow array): one lantern is re-rendered per frame.
    VkImageView shadow_lamp_view[2]{};
    VkImageView shadow_flash_view = VK_NULL_HANDLE;   // layer 2 of the small-object shadow array
    bool shadow_skip = false;                   // this frame the small plants' shadow maps keep last frame's
    VkImage        shadow_image = VK_NULL_HANDLE;
    VmaAllocation  shadow_alloc = nullptr;
    VkImageView    shadow_view = VK_NULL_HANDLE;         // layer 0 (64 m), 2D
    VkImageView    shadow_near_view = VK_NULL_HANDLE;    // layer 1 (16 m, 4x finer), 2D
    VkImageView    shadow_array_view = VK_NULL_HANDLE;   // both layers (deferred)
    VkSampler      shadow_sampler = VK_NULL_HANDLE;   // NEAREST: depth compares must not average
    VkPipelineLayout shadow_layout = VK_NULL_HANDLE;
    VkPipeline     shadow_pipeline = VK_NULL_HANDLE;
    VkShaderModule shadow_vert = VK_NULL_HANDLE;
    glm::mat4      shadow_vp{0.0f};
};

} // namespace vibe
