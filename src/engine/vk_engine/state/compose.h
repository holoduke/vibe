#pragma once
// The compose / tonemap pass (post.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (compose_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::ComposeState {
    // Compose desc set is rewired when this flag toggles — tracks the
    // currently-bound source so we only re-write on actual change.
    bool                  uses_taau = false;
    bool          uses_fsr2 = false;
    bool          uses_fsr3 = false;
    // --- Compose / tonemap pass ---
    VkDescriptorPool       desc_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout  desc_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet        desc_sets[kHistorySlots]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout       pipeline_layout = VK_NULL_HANDLE;
    VkPipeline             pipeline = VK_NULL_HANDLE;
    bool                   uses_dlss = false;
    VkShaderModule         frag_module = VK_NULL_HANDLE;
};

} // namespace vibe
