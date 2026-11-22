#pragma once
// The physical sky: its look-up tables and passes (atmosphere.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (atmo_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::AtmosphereState {
    AtmoImage              trans, ms, sky, ap;
    VkSampler              sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout  set_layout = VK_NULL_HANDLE;
    VkDescriptorPool       pool = VK_NULL_HANDLE;
    VkDescriptorSet        set = VK_NULL_HANDLE;
    VkPipelineLayout       pipeline_layout = VK_NULL_HANDLE;
    VkPipeline             pipeline = VK_NULL_HANDLE;
    bool                   luts_ready = false;
    bool                   dispatched_this_frame = false;
};

} // namespace vibe
