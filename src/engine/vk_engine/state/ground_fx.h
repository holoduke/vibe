#pragma once
// The ground's object effects (vk_engine/ground_fx.cpp; shaders/ground_fx.glsl): the paste of
// meshes into the ground, the far ground's colour of placed objects (the top-down capture),
// the deferred decals. Private to the files that implement it - vk_engine.h only knows the
// name of this struct and holds a pointer to it (gfx_). Created and freed in private_state.cpp.

#include "engine/vk_engine.h"

#include <cstdint>
#include <vector>

namespace vibe {

// (shaders/ground_fx.glsl GroundFxB - keep the two layouts the same)
struct GfxDecalGpu { glm::vec4 c, ax, ay, az; };
struct GroundFxGpuHead {
    glm::vec4 paste;      // x on, y bake t0, z bake first level, w the zone's near height (m)
    glm::vec4 lsd;        // x on, yz window corner (world x, z), w window size (m)
    glm::vec4 lsd_fade;   // x fade in from, y full by (m), z VIBE_TEST_DEBRIS, w VIBE_TEST_LSD
    glm::vec4 dec;        // x decals, y tiles across, z tiles down, w on
};

struct VulkanEngine::GroundFxState {
    static constexpr int kMaxDecals = 256;
    static constexpr int kTile      = 32;
    static constexpr int kMaxTiles  = 8192;     // (3440 x 1440 at full scale: 108 x 45 tiles)

    VkBuffer      buf = VK_NULL_HANDLE;
    VmaAllocation buf_alloc = nullptr;
    VkDeviceSize  buf_size = 0;

    // ---- the top-down capture of placed objects ("large scale detail") ----
    static constexpr int   kLsdSize  = 1024;        // texels a side
    static constexpr float kLsdTexel = 4.0f;        // metres a texel: a 4 km window round the camera
    static constexpr int   kLsdSnap  = 32;          // the window moves in steps of this many texels
    VkImage        lsd_image = VK_NULL_HANDLE;      // RGBA8: rgb the objects' colour from above, a coverage
    VmaAllocation  lsd_alloc = nullptr;
    VkImageView    lsd_view = VK_NULL_HANDLE;
    VkImage        lsd_depth = VK_NULL_HANDLE;      // D32: the highest surface wins
    VmaAllocation  lsd_depth_alloc = nullptr;
    VkImageView    lsd_depth_view = VK_NULL_HANDLE;
    VkSampler      lsd_sampler = VK_NULL_HANDLE;    // linear, repeating: a texel's place is its world index modulo kLsdSize
    VkPipeline     lsd_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout lsd_layout = VK_NULL_HANDLE;
    bool           lsd_layout_ready = false;        // image in SHADER_READ_ONLY (after its first clear)
    bool           lsd_depth_ready = false;         // depth in DEPTH_ATTACHMENT (after its first use)
    glm::ivec2     lsd_c{0};                         // the window's centre, a world texel index
    bool           lsd_valid = false;
    uint64_t       lsd_parts_sig = 0;                // what was drawn (the parts' set)
    int            lsd_cool = 0;                     // frames until a change of the set may redraw everything
    std::vector<glm::vec4> lsd_redo;                 // world boxes (x0, z0, x1, z1) to draw again
    uint32_t       lsd_drawn = 0;                    // parts drawn this frame (log)

    // ---- the decals ----
    struct Decal { glm::vec3 c; int kind; glm::vec3 ax, ay, az; float strength, seed, ratio; };
    std::vector<Decal> placed;                       // the authored set near the camera (by rule; streamed)
    glm::ivec2         placed_cell{INT32_MAX, INT32_MAX};
    std::vector<GfxDecalGpu> frame_list;             // this frame's visible ones
    std::vector<uint32_t>    tiles;                  // 8 words a tile
    int tiles_x = 0, tiles_y = 0;
    bool log = false;
};

} // namespace vibe
