#pragma once

#include "game/collision.h"
#include "game/level.h"
#include "game/physics.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <span>

namespace vibe::game {

struct PlayerInput {
    bool fwd = false, back = false, left = false, right = false;
    // The same as an amount, from a controller's stick (x right, y forward, length 0..1).
    // Zero when the keys are used: then the four above count.
    float ax = 0.0f, ay = 0.0f;
    glm::vec2 move_axis() const {
        if (ax != 0.0f || ay != 0.0f) return glm::vec2(ax, ay);
        return glm::vec2(float(right) - float(left), float(fwd) - float(back));
    }
    bool jump = false;
    bool sprint = false;    // Shift held — wishspeed × 1.5 on the ground
    bool aiming = false;    // sights up: a careful walk (x 0.55), no sprint
    bool crawl  = false;    // Ctrl held — wishspeed × 0.4 on the ground
    float mouse_dx = 0.0f;  // pixels of horizontal motion since last update
    float mouse_dy = 0.0f;
    // Override of the kPlayerMove.gravity constant. Engine sets this from
    // RtSettings::gravity each tick so the menu slider can drive fall rate
    // without rebuilding the constexpr MoveParams.
    float gravity = 25.0f;
    // Water. `swim`: the body is deep enough in open water to float -
    // movement follows the view direction (Space up, Ctrl down), with
    // water drag and buoyancy instead of ground friction / gravity.
    // `water_y` is the surface height. `wade` (0..1) slows walking
    // through shallows.
    bool  swim = false;
    float water_y = 0.0f;
    float wade = 0.0f;
};

struct Player {
    // AABB center (NOT feet). The eye sits near the top.
    // Spawn slightly above floor (feet at y≈0.6) so first-frame gravity has room
    // to settle without depenetration kicking in.
    glm::vec3 position{ 0.0f, 1.5f, 12.0f };
    glm::vec3 velocity{ 0.0f };
    float yaw   = 0.0f;   // radians; 0 looks down -Z
    float pitch = 0.0f;   // radians; clamped to (-pi/2, +pi/2)
    bool on_ground = false;
    // 0 = standing, 1 = crawling. Lerped toward in.crawl per tick. Drives the
    // eye-height drop so the camera physically lowers when Ctrl is held.
    // Hitbox is intentionally NOT scaled — avoids the "uncrouch into a low
    // ceiling" trap that needs explicit clearance checks to handle safely.
    float crouch_factor = 0.0f;
    // Smoothing offset for stair step-ups. When slide_move snaps the player
    // up by a step's height, the physics position jumps instantly but the
    // eye is rendered at `position.y + eye_offset_y - step_smooth_offset`
    // — so it keeps its old height for a frame and rises smoothly back as
    // step_smooth_offset decays toward zero.
    float step_smooth_offset = 0.0f;
    // Double-jump state. air_jumps_left starts at kAirJumps when the
    // player leaves the ground; each in-air space press consumes one.
    // jump_edge tracks the previous tick's `in.jump` so we only fire
    // on the leading edge (held-down doesn't burn both jumps in one
    // frame).
    int  air_jumps_left = 0;
    bool jump_edge      = false;
    static constexpr int kAirJumps = 1;        // 1 = double-jump total

    // 0.6 m wide, 1.8 m tall — narrower than the previous 0.8 m so the player
    // can fit between closer obstacles (settled boxes, lantern posts, etc.).
    static constexpr glm::vec3 kHalfExtents{ 0.3f, 0.9f, 0.3f };
    static constexpr float kEyeOffsetStand = 0.7f;    // from AABB center, standing
    static constexpr float kEyeOffsetCrawl = -0.5f;   // crawling: eye near floor
    static constexpr float kMouseSensitivity = 0.0035f;

    collision::AABB shape() const {
        return { -kHalfExtents, kHalfExtents };
    }

    float eye_offset_y() const {
        return kEyeOffsetStand +
               (kEyeOffsetCrawl - kEyeOffsetStand) * crouch_factor;
    }

    glm::vec3 eye_position() const {
        return position + glm::vec3(0.0f,
                                     eye_offset_y() - step_smooth_offset,
                                     0.0f);
    }

    glm::vec3 forward() const;  // unit vector in look direction
    glm::vec3 right() const;    // unit vector to right of look (horizontal-only)
    glm::mat4 view_matrix() const;
};

// Quake-feel parameters scaled for a 1-unit ≈ 1-meter world.
inline constexpr physics::MoveParams kPlayerMove{
    .ground_friction = 6.0f,
    .ground_accel    = 10.0f,
    .air_accel       = 10.0f,
    .air_wish_cap    = 0.75f,
    .stop_speed      = 2.5f,
    .max_walk_speed  = 8.0f,
    .gravity         = 25.0f,
    // ~1.45 m vertical clearance: jump_speed^2 / (2 * gravity).
    .jump_speed      = 8.5f,
};

// `static_count` is the number of leading entries in world_aabbs that are
// STATIC level geometry — collision::slide_move uses it so the auto step-up
// only fires on stairs/static brushes, not on pushable dynamic crates.
// Default npos treats every entry as static (matches the convenience
// overload below).
void update_player(Player& p, const PlayerInput& in,
                   std::span<const collision::AABB> world_aabbs, float dt,
                   size_t static_count = static_cast<size_t>(-1));

// Convenience overload: collision against the static level only.
void update_player(Player& p, const PlayerInput& in, const Level& world, float dt);

} // namespace vibe::game
