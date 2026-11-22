#pragma once
// Enemy aircraft (combat.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (spacejet_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::SpacejetState {
    std::vector<SpacejetMesh>   meshes;
    std::vector<SpacejetFlight> flights;
    glm::vec3 centre_offset{0.0f};      // gltf bbox-centre, applied per draw
    float     base_scale      = 1.0f;   // gltf-bbox-derived scale
    std::vector<JetChunk> chunks;
    float     spawn_timer     = 1.0f;   // first wave after 2s
    uint32_t  rng_state       = 0xC0DEFEEDu;
    uint32_t  next_id         = 0;
};

} // namespace vibe
