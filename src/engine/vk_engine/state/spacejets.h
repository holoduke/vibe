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
    // Where the aircraft patrol (x, z, radius m): the castle, the launch site (the base with the
    // pad, the pavilion and the station) and the open sea beyond the coast east of it. Made once
    // (props.cpp jet_zones), [0] is the castle.
    std::vector<glm::vec3> zones;
    glm::vec2 zone_mid{0.0f};
    float     zone_reach = 1350.0f;   // waves come in from this far beyond the middle
};

} // namespace vibe
