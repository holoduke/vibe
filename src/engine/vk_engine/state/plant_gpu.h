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
    // The automatic grass - long tufts and turf - placed on the GPU every frame (tuft_place.comp):
    // the rules picture it reads (ScatterSet::tuft_rules, binding 5), the height map and lawn
    // mask last bound at 6 and 7, its pipeline, and how many each list of species 15 (tufts)
    // and 17 (turf) has room for.
    VkImage        tuft_img = VK_NULL_HANDLE;
    VmaAllocation  tuft_img_alloc = nullptr;
    VkImageView    tuft_view = VK_NULL_HANDLE;
    VkImageView    ground_view = VK_NULL_HANDLE, lawn_view = VK_NULL_HANDLE;
    VkSampler      tuft_sampler = VK_NULL_HANDLE;
    VkPipeline     tuft_pipeline = VK_NULL_HANDLE;
    VkShaderModule tuft_module = VK_NULL_HANDLE;
    uint32_t       tuft_room = 0, turf_room = 0, meadow_room = 0;
    VkImageView    near_view = VK_NULL_HANDLE;      // the ground beyond the home tile, last bound at 8
    VkImageView    shade_view = VK_NULL_HANDLE;     // how much vegetation stands over the ground (albedo slot 22), last bound at 9 (the debris)
    VkImageView    tops_view = VK_NULL_HANDLE;      // the rock formations' tops from above, last bound at 10 (rock_formations_gpu.cpp)
    uint32_t       debris_room = 0;                 // room in each list of the debris species 19..26 (ground_fx: VIBE_NO_GROUND_DEBRIS)
    uint32_t       tuft_gen = 0;
};

} // namespace vibe
