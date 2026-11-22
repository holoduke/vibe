#pragma once
// Screen-space displacement mapping (ssdm.cpp): its state.
//
// Private to the few files that implement it - vk_engine.h only knows the name of this
// struct and holds a pointer to it (ssdm_), so a change here rebuilds those files and not
// the whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::SsdmState {
    SsdmImage disp, ray, hitz, shell_z, color, resolve, scene_z, mv, g0, g1;
    VkPipeline gbuffer_pipeline = VK_NULL_HANDLE;
    VkShaderModule gbuffer_frag = VK_NULL_HANDLE;
    VkExtent2D extent{ 0, 0 };
    VkPipeline mesh_pipeline = VK_NULL_HANDLE, color_pipeline = VK_NULL_HANDLE,
               depth_pipeline = VK_NULL_HANDLE, velocity_pipeline = VK_NULL_HANDLE;
    VkShaderModule mesh_vert = VK_NULL_HANDLE, mesh_frag = VK_NULL_HANDLE, color_frag = VK_NULL_HANDLE,
                   depth_frag = VK_NULL_HANDLE, velocity_frag = VK_NULL_HANDLE;
    VkPipelineLayout fs_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet  set = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE;
};

} // namespace vibe
