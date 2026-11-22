#pragma once
// Guide images for DLSS Ray Reconstruction (rr_guides.cpp, dlss.cpp): its state.
//
// Private to the few files that implement it - vk_engine.h only knows the name of this
// struct and holds a pointer to it (rr_), so a change here rebuilds those files and not
// the whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::RrGuideState {
    VkImage                image[3]{};          // diffuse albedo, specular albedo, normal+roughness
    VmaAllocation          alloc[3]{};
    VkImageView            view[3]{};
    VkImageView            src_views[4]{};      // G-buffer / depth / scene colour views the set was written with
    VkExtent2D             extent{0, 0};
    VkSampler              sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout  set_layout = VK_NULL_HANDLE;
    VkDescriptorPool       pool = VK_NULL_HANDLE;
    VkDescriptorSet        set = VK_NULL_HANDLE;
    VkPipelineLayout       pipeline_layout = VK_NULL_HANDLE;
    VkPipeline             pipeline = VK_NULL_HANDLE;
};

} // namespace vibe
