#pragma once
// LAN multiplayer (multiplayer.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (mp_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::MultiplayerState {
    std::array<MpAvatar, net::kMaxPlayers> avatars{};
    std::vector<MpFeed> feed;
    const MpAvatar* draw = nullptr;                // set while a remote player's body + gun are drawn
    bool      remote_shot = false;                 // fire_projectile is replaying someone else's shot
    glm::vec3 spawn{0.0f}, last_sent_pos{0.0f};
    float     health = 100.0f, hurt = 0.0f, hitmark = 0.0f, teleport_hold = 0.0f;
    int       kills = 0;
    net::WorldState world_last{};                  // what was last sent / applied (change detection)
    bool      world_init = false;
    float     world_timer = 0.0f;
    bool      host_tgt_valid = false;              // client: the host decides the weather that changes by itself
    Wx    host_tgt{};             // multiplayer client: the weather target the host decided
};

} // namespace vibe
