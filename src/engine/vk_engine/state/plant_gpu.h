#pragma once
// Small plants on the GPU path (plant_gpu.cpp, plant_cull.comp): its state.
//
// Private to the few files that implement it - vk_engine.h only knows the name of this
// struct and holds a pointer to it (pg_), so a change here rebuilds those files and not
// the whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::PlantGpuState {
    uint32_t      total = 0, packed_cap = 0;
    bool          hiz_bound = false, drawn = false;
    VkBuffer      packed_buf = VK_NULL_HANDLE, cmd_buf = VK_NULL_HANDLE, tmpl_buf = VK_NULL_HANDLE, spec_buf = VK_NULL_HANDLE;
    VmaAllocation packed_alloc = nullptr, cmd_alloc = nullptr, tmpl_alloc = nullptr, spec_alloc = nullptr;
    void*         spec_mapped = nullptr;
    // The culling shader's own copy of the plants, in GPU memory (the engine's instance buffer is
    // in host memory, for the CPU to rewrite: read from a shader it costs a trip per plant).
    VkBuffer      inst_buf = VK_NULL_HANDLE;
    VmaAllocation inst_alloc = nullptr;
    uint32_t      inst_gen = 0;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkDescriptorSet       set = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkShaderModule        module = VK_NULL_HANDLE;
};

} // namespace vibe
