#pragma once
// The surroundings as the ear has them (vk_engine/sound_env.cpp): how much echo there is where
// the listener stands, and what stands near that sound does not go through.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (sound_env_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/sound_space.h"
#include "engine/vk_engine.h"

#include <vector>

namespace vibe {

struct VulkanEngine::SoundEnvState {
    float wait = 0.0f;                        // until the surroundings are looked at again (s)
    bool  cabin = false;                      // as it was at the last look
    bool  room = false;                       // the ear is in a room (its walls are in every line: Space::outside_* is what they do)
    bool  probe_set = false;                  // the audio engine has been given sound_occlusion
    int   looks = 0;
    sound_space::Space space;                 // the last look's result
    std::vector<collision::AABB> walls;       // the big boxes near the ear: walls, buildings (not trunks, kerbs, parapets)
    glm::vec3 walls_at{1.0e9f};               // where they were gathered
};

} // namespace vibe
