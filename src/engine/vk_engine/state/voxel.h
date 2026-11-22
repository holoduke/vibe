#pragma once
// Voxel buildings (voxel.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (voxel_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"
#include "engine/voxel/voxel_world.h"

namespace vibe {

struct VulkanEngine::VoxelState {
    // CPU brickmap world (owns all bricks + shape directories).
    std::unique_ptr<voxel::VoxelWorld> world;
    // GPU-side: brick atlas SSBO (all bricks for all shapes), shape
    // directory SSBO (single shape for now), and the per-frame camera
    // UBO. Pipeline + layout are fully self-contained (no overlap with
    // scene_desc_set_). One descriptor set, three bindings:
    //   0 = camera UBO, 1 = brick atlas, 2 = shape directory.
    VkBuffer              atlas_buffer      = VK_NULL_HANDLE;
    VmaAllocation         atlas_alloc       = nullptr;
    // Host-mapped pointer into the brick atlas (Session C). Carves modify
    // the CPU brick data then memcpy the changed bricks here + flush, so
    // the hole appears next frame with no command-buffer round-trip. The
    // atlas lives in device-local-host-visible (ReBAR) memory when available
    // — 4 MB fits the guaranteed 256 MB BAR window, so GPU reads stay fast.
    void*                 atlas_mapped      = nullptr;
    VkDeviceSize          atlas_bytes       = 0;
    // World-space solid boxes (player collision) + dirty-brick scratch.
    std::vector<collision::AABB> collision_aabbs;
    bool  update_pending  = false;  // a carve happened, not yet settled
    int   removed_accum   = 0;      // voxels carved since last collapse
    // Shape indices that were carved this frame (or since last collapse).
    // process_voxel_updates collapses only these — chunks split into smaller
    // chunks when hit again, and main-shape collapse stays as before.
    std::vector<int> dirty_shapes;
    float collapse_cd     = 0.0f;   // min time between collapse BFS runs
    float collision_cd    = 0.0f;   // min time between collision rebuilds
    VkBuffer              dir_buffer        = VK_NULL_HANDLE;
    VmaAllocation         dir_alloc         = nullptr;
    VkBuffer              camera_ubo        = VK_NULL_HANDLE;
    VmaAllocation         camera_alloc      = nullptr;
    void*                 camera_mapped     = nullptr;
    VkShaderModule        vert_module       = VK_NULL_HANDLE;
    VkShaderModule        frag_module       = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_set_layout   = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline          = VK_NULL_HANDLE;
    VkDescriptorPool      desc_pool         = VK_NULL_HANDLE;
    VkDescriptorSet       desc_set          = VK_NULL_HANDLE;
    std::vector<VoxelChunk> chunks;
    // Upload watermarks: how many bricks / directory entries have been
    // pushed to the GPU. After a collapse we push the newly-added range.
    uint32_t atlas_uploaded = 0;
    uint32_t dir_uploaded   = 0;
    // Index of the next NEW shape whose directory needs uploading. Lets
    // upload_voxel_growth_() skip already-pushed shapes in O(1) instead of
    // re-scanning every shape every collapse (which grew linearly with
    // session-long debris history).
    uint32_t dir_next_shape = 0;
    // Mapped pointers for live updates (atlas pointer already exists above).
    void*       dir_mapped  = nullptr;
    VkDeviceSize dir_bytes  = 0;
};

} // namespace vibe
