#pragma once
// Cluster LOD of the static world meshes: castle, chapel, gravestones (cluster_draw.cpp, castle_mesh.cpp): its state.
//
// Private to the few files that implement it - vk_engine.h only knows the name of this
// struct and holds a pointer to it (scl_), so a change here rebuilds those files and not
// the whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::StaticClusterState {
    uint32_t      total = 0;
    bool          hiz_bound = false, ready = false;
    std::vector<SclPart>    parts;
    std::vector<SclRetired> retired;
    VkBuffer      cluster_buf = VK_NULL_HANDLE, meta_buf = VK_NULL_HANDLE, cmd_buf = VK_NULL_HANDLE, cmd_shadow_buf = VK_NULL_HANDLE;
    VmaAllocation cluster_alloc = nullptr, meta_alloc = nullptr, cmd_alloc = nullptr, cmd_shadow_alloc = nullptr;
    VkDescriptorSet set_shadow = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkDescriptorSet       set = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkShaderModule        module = VK_NULL_HANDLE;
};

} // namespace vibe
