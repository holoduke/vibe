#pragma once
// Volumetric clouds: the low-resolution layer (atmosphere.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (cloud_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::CloudState {
    uint32_t w = 1920, h = 1080;
    double travel = 0.0, travel_t = -1.0;   // metres the wind has carried the clouds; time of the last step
    // Cloud layer as last sent to the shaders (for the CPU copy of the
    // cloud-shadow function: sun glare when a cloud passes in front).
    glm::vec4 a{0.0f}, b{0.0f};
    float shadow_k_now = 0.0f;
    // Sunlight let through by the cloud layer, as a map of the land
    // (512^2 over 6144 m; written by clouds.comp mode 3 every frame).
    AtmoImage              shadow_map;
    VkSampler              shadow_sampler = VK_NULL_HANDLE;
    bool                   shadow_bound = false;
    // Volumetric clouds (clouds.comp): 960x540 cloud layer + tileable noise.
    AtmoImage              img, shape, detail;
    VkDescriptorSetLayout  set_layout = VK_NULL_HANDLE;
    VkDescriptorPool       pool = VK_NULL_HANDLE;
    VkDescriptorSet        set = VK_NULL_HANDLE;
    VkImageView            depth_bound = VK_NULL_HANDLE;   // rewrite binding 2 only on change
    VkPipelineLayout       pipeline_layout = VK_NULL_HANDLE;
    VkPipeline             pipeline = VK_NULL_HANDLE;
    bool                   noise_ready = false;
    VkSampler              sampler = VK_NULL_HANDLE;   // linear + REPEAT (tileable noise)
};

} // namespace vibe
