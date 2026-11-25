#pragma once
// Cars among cars (jeep.cpp, "CARS STRIKE CARS" and "THE MILITARY POLICE"): what is kept from
// frame to frame about cars that were struck, about the police cars' light bars and about their
// sirens. Private to jeep.cpp; vk_engine.h holds a pointer (private_state.cpp makes and frees it).
#include "engine/vk_engine.h"

#include <unordered_map>

namespace vibe {

struct VulkanEngine::CarTrafficState {
    // A players' car that nobody simulates and that was struck: it slides and turns, less and
    // less (cars_collide). `cool`: not struck again by the same blow. `claim`: this machine
    // took it over a moment ago - what another machine still says of it meanwhile is old.
    struct Shove { glm::vec2 v{0.0f}; float w = 0.0f, cool = 0.0f, claim = 0.0f; };
    std::unordered_map<const void*, Shove> shove;        // by the car (a JeepState*)
    float npc_cool[100] = {};                            // an NPC car: not pushed again by the same blow
    int   strikes = 0;                                   // (tests: how many blows this machine has dealt)

    // The police cars' light bars: three drawn parts each (the base, the left lamp, the right
    // lamp), from `first` in jeep_draw_. `agent`: the NPC car it sits on this frame, -1 none.
    struct Bar {
        size_t    first = 0;
        int       agent = -1;
        bool      siren = false;
        glm::vec3 lamp[2]{};                             // where the two lamps are (world): their glow (jeep_fx)
        bool      lit[2] = { false, false };
        // its siren's voice (siren_audio_tick)
        uint32_t  voice = 0;
        float     vol = 0.0f;
        float     rate = 1.0f;                           // its pitch (kept while it fades out)
        glm::vec3 last{0.0f}, vel{0.0f};
        bool      has_last = false;
    };
    std::vector<Bar> bars;
    float flash_t = 0.0f;
    int   siren_clip = 0;                                // 0 not asked for yet, 1 there, -1 none
    uint8_t style_was[100] = {};                         // each NPC car's View::style last frame (VIBE_LOG_TRAFFIC: what changed)
};

}  // namespace vibe
