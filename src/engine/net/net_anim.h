// From a replicated player state to what that player's animator is told. One function, used by
// the game (vk_engine/multiplayer.cpp) and by the tests, so what is tested is what runs.
//
// Everything here comes from the network state of THAT player: nothing of the local player's
// camera or input is involved, and nothing is sent back.
#pragma once

#include "anim/character_animator.h"
#include "engine/net/net_state.h"
#include "game/player.h"

#include <glm/glm.hpp>

namespace qlike::net {

struct RemoteFrame {
    glm::mat4 model{1.0f};        // the body's model matrix (model space: +X right, +Y up, -Z forward)
    glm::mat4 view_inv{1.0f};     // their camera: what they carry and their head are drawn from it
    glm::vec3 eye{0.0f};
};

// How far the eye sits in front of the neck (the body is stood back by this much so a first
// person view is not inside the chest; the same rule places every player's body).
inline float eye_back_offset(float pitch, float crouch) {
    return 0.11f + 0.17f * glm::smoothstep(0.15f, 1.25f, -pitch) + 0.25f * glm::clamp(crouch, 0.0f, 1.0f);
}

// Fills position, motion, aim, stance, what is held, the shot counter and the movement the
// player wants. The caller adds dt, level of detail, reload sequence and the ground probe.
inline RemoteFrame remote_animation_input(const PlayerState& s, anim::CharacterAnimationInput& in) {
    RemoteFrame f;
    const float yaw = s.yaw, pitch = s.pitch, crouch = glm::clamp(s.crouch, 0.0f, 1.0f);
    const glm::vec3 fwd(std::sin(yaw), 0.0f, -std::cos(yaw)), right(std::cos(yaw), 0.0f, std::sin(yaw));
    const float eye_off = game::Player::kEyeOffsetStand + (game::Player::kEyeOffsetCrawl - game::Player::kEyeOffsetStand) * crouch;
    const float eye_h = game::Player::kHalfExtents.y + eye_off;
    f.eye = glm::vec3(s.px, s.py, s.pz) + glm::vec3(0.0f, eye_off, 0.0f);
    const float back = eye_back_offset(pitch, crouch);
    const glm::vec3 feet = f.eye - glm::vec3(0.0f, eye_h, 0.0f) - fwd * back;
    f.model = glm::mat4(glm::vec4(right, 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f), glm::vec4(-fwd, 0.0f), glm::vec4(feet, 1.0f));
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const glm::vec3 look(fwd.x * cp, sp, fwd.z * cp), up = glm::cross(right, look);
    f.view_inv = glm::mat4(glm::vec4(right, 0.0f), glm::vec4(up, 0.0f), glm::vec4(-look, 0.0f), glm::vec4(f.eye, 1.0f));

    in.position = feet;
    in.velocity = glm::vec3(s.vx, s.vy, s.vz);
    in.facing_yaw = yaw;
    in.aim_pitch = pitch;
    in.grounded = (s.flags & kFlagGround) != 0 && !(s.flags & kFlagSwim);
    in.swimming = (s.flags & kFlagSwim) != 0;
    in.wants_jump = (s.move & kMoveJump) != 0;
    in.crouch = crouch;
    in.eye_local = glm::vec3(0.0f, eye_h, -back);
    in.stand_eye_height = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
    in.first_person = false;
    glm::vec3 wish(0.0f);
    if (s.move & kMoveFwd) wish += fwd;
    if (s.move & kMoveBack) wish -= fwd;
    if (s.move & kMoveRight) wish += right;
    if (s.move & kMoveLeft) wish -= right;
    in.desired_velocity = glm::vec3(0.0f);
    if (glm::dot(wish, wish) > 1e-6f)
        in.desired_velocity = glm::normalize(wish) * game::kPlayerMove.max_walk_speed * ((s.move & kMoveSprint) ? 1.5f : (s.move & kMoveCrawl) ? 0.4f : 1.0f);
    // (a torch is not drawn in other players' hands: they are shown with what they carry besides it)
    in.held = s.weapon == 1 ? anim::HeldItem::Tablet : anim::HeldItem::Rifle;
    in.shot_seq = s.shot_seq;
    in.reloading = (s.flags & kFlagReload) != 0;
    in.aiming = (s.flags & kFlagAim) != 0;
    return f;
}

// The local player's state as it is sent (the same fields a remote animator will be given).
inline uint8_t move_bits(const game::PlayerInput& p) {
    return uint8_t((p.fwd ? kMoveFwd : 0) | (p.back ? kMoveBack : 0) | (p.left ? kMoveLeft : 0) | (p.right ? kMoveRight : 0) |
                   (p.sprint ? kMoveSprint : 0) | (p.jump ? kMoveJump : 0) | (p.crawl ? kMoveCrawl : 0));
}

}  // namespace qlike::net
