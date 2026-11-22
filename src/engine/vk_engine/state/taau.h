#pragma once
// TAAU: temporal upsample (taa.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (taau_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::TaauState {
    // --- TAAU (temporal upsample). Optional pass between TAA and compose.
    // Reads LR history + LR motion + LR depth + previous-frame NATIVE
    // upscaled, outputs native upscaled with motion-reproject + variance
    // clamp. Compose then samples view instead of history_view_.
    // Sized at swapchain_extent_; ping-pong on history_write_slot_.
    VkImage          image[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VmaAllocation    alloc[kHistorySlots]{ nullptr, nullptr };
    VkImageView      view[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorPool      desc_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet       desc_sets[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkBuffer              ubo_buffer = VK_NULL_HANDLE;
    VmaAllocation         ubo_alloc = nullptr;
    VkPipelineLayout      pipeline_layout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkShaderModule        frag_module = VK_NULL_HANDLE;
    bool                  history_valid = false;
};

} // namespace vibe
