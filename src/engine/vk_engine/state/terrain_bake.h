#pragma once
// The baked ground texture (vk_engine/terrain_bake.cpp; shaders/terrain_rm_bake.glsl): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (tb_). Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

#include <future>
#include <vector>

namespace vibe {

struct VulkanEngine::TerrainBakeState {
    // (shaders/terrain_rm_bake.glsl kBakeLevels / kBakeSize / kBakeTexel0 - keep them the same)
    static constexpr int   kLevels = 7;
    static constexpr int   kSize   = 2048;
    static constexpr float kTexel0 = 0.0075f;
    static constexpr int   kSnap   = 16;        // a level's window moves in steps of this many texels
    // Texels drawn in one frame while a level is being filled (a quarter of a level).
    static constexpr int   kFillTexels = (kSize / 2) * (kSize / 2);

    // Three pictures (terrain_rm_bake.glsl), each an array of kLevels layers.
    VkImage        image[3]{};
    VmaAllocation  alloc[3]{};
    VkImageView    array_view[3]{};
    VkImageView    layer_view[3][kLevels]{};
    VkFormat       format[3]{};
    VkSampler      sampler = VK_NULL_HANDLE;    // linear, repeating: a texel's place is its world index modulo kSize
    VkShaderModule frag = VK_NULL_HANDLE;
    VkPipeline     pipeline = VK_NULL_HANDLE;
    std::future<VkPipeline> build;              // the bake shader is as large as the terrain's: built on a thread

    struct Level {
        glm::ivec2 c{0};        // the window's centre, a world texel index (the window is c - kSize/2 .. c + kSize/2)
        bool       valid = false;
        int        tiles = 0;   // while filling: quarters done
    };
    Level    level[kLevels];
    // What the material depends on, hashed (terrain_bake_signature): the pictures are made
    // again when it has changed and then stood still for a moment.
    uint64_t sig = 0, sig_baked = ~0ull, sig_seen = 0;
    int      sig_still = 0;
    bool     ready = false;     // every level is whole: the terrain pass may read them
    // Craters: the list as last seen, the ones that last changed (drawn again when the height
    // texture follows a frame or two later), and the ground having been changed at all.
    std::vector<glm::vec4> craters_seen, crater_last;
    int      crater_cool = 0;
    bool     height_dirty = false;
    float    t0 = 0.0f;         // the distance level 0 is made for
    int      lmin = 0, want_min = 0;    // the first level read this frame / wanted at this speed
    glm::vec3 prev_eye{0.0f};
    float    speed = 0.0f;      // the camera's, smoothed (m/s)
    uint32_t texels_frame = 0;  // drawn this frame (for the log)
};

} // namespace vibe
