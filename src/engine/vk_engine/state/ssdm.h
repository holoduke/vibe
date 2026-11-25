#pragma once
// Screen-space displacement mapping (ssdm.cpp): its state.
//
// Private to the few files that implement it - vk_engine.h only knows the name of this
// struct and holds a pointer to it (ssdm_), so a change here rebuilds those files and not
// the whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::SsdmState {
    // hn: normal + height of the displaced stonework (ssdm_height); off: the displacement
    // vector per pixel (ssdm_solve); color / g0 / g1 / scene_z: the main pass's attachments
    // as they were before displacement (what the grid reads while it writes the live ones).
    SsdmImage hn, off, color, g0, g1, scene_z;
    VkExtent2D extent{ 0, 0 };
    VkPipeline height_pipeline = VK_NULL_HANDLE, solve_pipeline = VK_NULL_HANDLE, grid_pipeline = VK_NULL_HANDLE;
    VkShaderModule height_vert = VK_NULL_HANDLE, height_frag = VK_NULL_HANDLE, solve_frag = VK_NULL_HANDLE,
                   grid_vert = VK_NULL_HANDLE, grid_frag = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet  set = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE;
};

} // namespace vibe
