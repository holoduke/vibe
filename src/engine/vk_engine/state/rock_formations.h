#pragma once
// Rock formations (vk_engine/rock_formations_gpu.cpp): what the engine holds of them.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct and
// holds a pointer to it (rockf_). Created on first use and freed by rockf_destroy().

#include "engine/vk_engine.h"
#include "engine/vk_engine/rock_formations.h"

#include <array>
#include <memory>
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
    // Collision is the rock's own surface (rockf::collision_triangles). The physics world holds
    // one mesh per piece (built once on a worker) and a static body per formation near the
    // player - the car, crates and debris touch the rock itself; the walker has the world-space
    // triangles of the formations very near (game/tri_collision.h).
    bool phys_requested = false;
    std::unordered_map<int, uint32_t> bodies;                             // formation -> PhysicsWorld::add_mesh_instance
    std::unordered_map<int, std::unique_ptr<tricol::Mesh>> walk;          // formation -> its triangles in the world
    std::vector<const tricol::Mesh*> walk_near;                           // those near the player now
    glm::vec3 cols_at{1e9f};
};

} // namespace vibe
