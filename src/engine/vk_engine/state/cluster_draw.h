#pragma once
// Cluster drawing of the rocks (cluster_draw.cpp): its state.
//
// Private to the few files that implement it - vk_engine.h only knows the name of this
// struct and holds a pointer to it (cl_), so a change here rebuilds those files and not
// the whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::ClusterDrawState {
    uint32_t      vis_base = 0, cmd_cap = 0;
    Mesh          mesh{};
    VkBuffer      cluster_buf = VK_NULL_HANDLE, table_buf = VK_NULL_HANDLE, cmd_buf = VK_NULL_HANDLE;
    VmaAllocation cluster_alloc = nullptr, table_alloc = nullptr, cmd_alloc = nullptr;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkDescriptorSet       set = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkShaderModule        module = VK_NULL_HANDLE;
};

} // namespace vibe
