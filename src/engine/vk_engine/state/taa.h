#pragma once
// TAA: scene colour and history (taa.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (taa_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::TaaState {
    VkDescriptorPool       desc_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout  desc_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet        desc_sets[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkBuffer               ubo_buffer = VK_NULL_HANDLE;
    VmaAllocation          ubo_alloc = nullptr;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkShaderModule   vert_module = VK_NULL_HANDLE;
    VkShaderModule   frag_module = VK_NULL_HANDLE;
};

} // namespace vibe
