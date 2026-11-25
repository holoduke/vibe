#pragma once
// The patch terrain (vk_engine/terrain_patches.cpp; docs/ARCHITECTURE.md "Patch terrain"): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (tp_). Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

#include <unordered_map>
#include <vector>

namespace vibe {

struct VulkanEngine::TerrainPatchState {
    // One patch: kN x kN quads (shaders/terrain_patch.vert kPatchN - keep the two the same).
    static constexpr int      kN          = 64;
    // Vertex spacing of the finest level (m); every level up doubles it. A patch of level L
    // is kN * kSpacing0 * 2^L metres a side: 16 m .. 2048 m.
    static constexpr float    kSpacing0   = 0.25f;
    static constexpr int      kLevels     = 8;
    static constexpr uint32_t kMaxPatches = 1024;
    // The patches' own projection: reversed depth from this near plane, no far plane
    // (terrain_rm_heightfield.glsl kPatchNear - keep the two the same).
    static constexpr float    kNear       = 0.05f;
    // How far the patches reach: what the ray march reaches (its t_limit). Beyond it the far
    // march draws the land, as before.
    static constexpr float    kRange      = 1500.0f;

    struct Instance {
        glm::vec4 patch;        // corner x, corner z, vertex spacing, level
        glm::vec4 view;         // the camera's height above the ground under it; spare
    };

    // The grid's index buffer (the vertices are made from gl_VertexIndex: no vertex buffer).
    VkBuffer       index_buf   = VK_NULL_HANDLE;
    VmaAllocation  index_alloc = nullptr;
    uint32_t       index_count = 0;
    // This frame's instances: a ring of kFrameOverlap lists, written every frame.
    VkBuffer       inst_buf    = VK_NULL_HANDLE;
    VmaAllocation  inst_alloc  = nullptr;
    void*          inst_map    = nullptr;
    VkDeviceSize   inst_offset = 0;
    uint32_t       count       = 0;

    VkShaderModule vert     = VK_NULL_HANDLE;
    VkPipeline     pipeline = VK_NULL_HANDLE;      // depth only

    // This frame: the patches are drawn and the terrain pass reads their depth picture.
    bool           active   = false;
    bool           was_on   = false;               // last frame's terrain_patches_on(): a change resizes the depth picture
    float          cam_h    = 0.0f;                // the camera's height above the ground under it
    glm::mat4      rel_proj{1.0f};                 // the patches' projection x the view's rotation

    // Lowest and highest ground of a quadtree node, from the CPU's ground function, kept:
    // key = level and cell, made again when the chart changes (bounds_serial).
    struct Bounds { float lo, hi; };
    std::unordered_map<uint64_t, Bounds> bounds;
    uint32_t       bounds_serial = ~0u;
    int            bounds_made   = 0;              // (this frame: for the log line)

    int            per_level[kLevels] = {};
    std::vector<Instance> list;
    std::vector<float>    list_d;                  // each one's distance, for the order
    std::vector<uint32_t> order;
};

} // namespace vibe
