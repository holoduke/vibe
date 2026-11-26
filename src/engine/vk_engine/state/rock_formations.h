#pragma once
// Rock formations (vk_engine/rock_formations_gpu.cpp): what the engine holds of them.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct and
// holds a pointer to it (rockf_). Created on first use and freed by rockf_destroy().

#include "engine/vk_engine.h"
#include "engine/vk_engine/rock_formations.h"

#include <array>
#include <unordered_map>
#include <vector>

namespace vibe {

struct VulkanEngine::RockfState {
    bool tried = false, ok = false;
    // The kit on the GPU: one mesh and one ray-tracing structure per piece and level, made
    // when a formation first needs it, given back when none has for a while.
    struct Level { CastleMeshGpu part; bool made = false; uint32_t last_used = 0; uint32_t tris = 0; };
    std::vector<std::array<Level, rockf::kLevels>> kit;
    std::vector<int8_t> level;                  // per formation: the level it is drawn at now, -1 = not drawn
    // What is drawn and traced this frame: a copy of the level's part per formation, with its
    // place (the copies share the kit's buffers and own nothing).
    std::vector<CastleMeshGpu> draw;
    uint32_t tris = 0;                          // triangles in `draw`
    std::vector<std::pair<int, std::vector<CastleMeshGpu>>> retired;      // kit levels given back, waiting out the frames in flight
    // Collision: the boxes of the formations near the player.
    std::vector<collision::AABB> cols;
    std::unordered_map<int, std::vector<collision::AABB>> boxes;          // per formation, made when the player comes near
    glm::vec3 cols_at{1e9f};
    uint32_t  cols_gen = 0;
};

} // namespace vibe
