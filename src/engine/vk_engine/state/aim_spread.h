#pragma once
// The local player's weapon spread (game/aim_spread.h has the model and its numbers): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (spread_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"
#include "game/aim_spread.h"

namespace vibe {

struct VulkanEngine::AimSpreadState {
    game::AimSpread model;
    float now_deg = 0.0f;      // the cone's half-angle this frame (what the next shot gets)
    bool  logged_demo = false; // VIBE_LOG_SPREAD: the simulated bursts were written
};

} // namespace vibe
