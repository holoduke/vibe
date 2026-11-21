#pragma once
// Where the character will be over the next second, predicted with the character
// controller's own movement model (game/physics.h: Quake ground friction + acceleration, air
// acceleration with its wish-speed cap). The prediction reads the controller; it never feeds
// back into it, and gameplay input is never smoothed to make it look better.
#include "anim/motion_db.h"
#include "game/physics.h"

namespace qlike::anim {

struct TrajectoryInput {
    glm::vec2 velocity{0.0f};           // world, horizontal (x, z), m/s
    glm::vec2 desired_velocity{0.0f};   // world, horizontal: wish direction x wish speed (zero = no input)
    float     facing_yaw = 0.0f;        // the player's yaw convention: 0 looks down -Z, positive turns right
    float     yaw_rate = 0.0f;          // rad/s
    bool      grounded = true;
};

struct Trajectory {
    // In the character's frame now (x right, z back), per horizon of the database.
    glm::vec2 pos[MotionSchema::kHorizons] = {};
    glm::vec2 dir[MotionSchema::kHorizons] = {};     // facing, unit
    glm::vec2 vel_end{0.0f};                         // velocity at the last horizon (same frame)
};

struct TrajectoryParams {
    float step = 1.0f / 60.0f;          // integration step, seconds
    float turn_decay = 0.25f;           // seconds over which a turn in progress is assumed to die out
    // While a key is held the wish direction turns with the view: the player steers with the mouse.
    bool  desired_follows_view = true;
};

// world (x, z) <-> character frame (x right, z back) for a given yaw
glm::vec2 world_to_facing(glm::vec2 world_xz, float yaw);
glm::vec2 facing_to_world(glm::vec2 local_xz, float yaw);

Trajectory predict_trajectory(const TrajectoryInput& in, const physics::MoveParams& move, const float* horizons,
                              const TrajectoryParams& p = {});

}  // namespace qlike::anim
