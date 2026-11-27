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
    };
    std::vector<Bar> bars;
    float flash_t = 0.0f;
    // The sirens that sound (siren_audio_tick): one source of the spatialiser
    // (AudioEngine::source_*) a police car whose siren is on, BY THE NPC ENGINE'S AGENT - not
    // by the bar, not by whether the car is drawn.
    struct Siren {
        uint32_t  src = 0;
        glm::vec3 pos{0.0f};                             // where it was last frame (an agent's number given to another car: not the same siren)
        float     log_t = 0.0f;                          // VIBE_LOG_TRAFFIC: a line every half second
    };
    std::unordered_map<int, Siren> sirens;
    int   siren_clip = 0;                                // 0 not asked for yet, 1 there, -1 none
    bool  siren_test_placed = false;                     // VIBE_TEST_SIREN=stay: the eye was put by the road
    // The driven car's tyres squealing (jeep.cpp car_tyre_audio): its source, the clip (0 not asked for,
    // 1 there, -1 none), the level it sounds at now.
    uint32_t squeal_src = 0;
    int      squeal_clip = 0;
    float    squeal_level = 0.0f;
    uint8_t style_was[100] = {};                         // each NPC car's View::style last frame (VIBE_LOG_TRAFFIC: what changed)
};

}  // namespace vibe
