#include "anim/trajectory.h"

#include <algorithm>
#include <cmath>

namespace qlike::anim {

glm::vec2 world_to_facing(glm::vec2 w, float yaw) {
    const float s = std::sin(yaw), c = std::cos(yaw);
    // right = (cos, sin), forward = (sin, -cos); local z points back
    return glm::vec2(w.x * c + w.y * s, -(w.x * s - w.y * c));
}

glm::vec2 facing_to_world(glm::vec2 l, float yaw) {
    const float s = std::sin(yaw), c = std::cos(yaw);
    return glm::vec2(l.x * c - l.y * s, l.x * s + l.y * c);
}

Trajectory predict_trajectory(const TrajectoryInput& in, const physics::MoveParams& move, const float* horizons,
                              const TrajectoryParams& p) {
    Trajectory out;
    float max_h = 0.0f;
    for (int i = 0; i < MotionSchema::kHorizons; ++i) max_h = std::max(max_h, horizons[i]);
    const float h = std::clamp(p.step, 1.0f / 240.0f, 0.05f);
    const int steps = int(std::ceil(max_h / h));
    glm::vec3 vel(in.velocity.x, 0.0f, in.velocity.y);
    glm::vec2 pos(0.0f);
    const float wish_speed = glm::length(in.desired_velocity);
    const glm::vec2 wish0 = wish_speed > 1e-4f ? in.desired_velocity / wish_speed : glm::vec2(0.0f);
    const float rate = std::clamp(in.yaw_rate, -12.0f, 12.0f);
    int next = 0;
    float t = 0.0f;
    // horizons are sorted ascending in the database; handle any order anyway
    bool done[MotionSchema::kHorizons] = {};
    for (int k = 1; k <= steps + 1 && next < MotionSchema::kHorizons; ++k) {
        t += h;
        const float turned = rate * p.turn_decay * (1.0f - std::exp(-t / p.turn_decay));
        glm::vec2 wd = wish0;
        if (p.desired_follows_view && wish_speed > 1e-4f) {
            const float s = std::sin(turned), c = std::cos(turned);
            wd = glm::vec2(wish0.x * c - wish0.y * s, wish0.x * s + wish0.y * c);
        }
        const glm::vec3 wishdir(wd.x, 0.0f, wd.y);
        if (in.grounded) {
            vel = physics::apply_friction(vel, move.ground_friction, move.stop_speed, h, 0.05f);
            vel = physics::accelerate(vel, wishdir, wish_speed, move.ground_accel, h);
        } else {
            vel = physics::air_accelerate(vel, wishdir, wish_speed, move.air_accel, move.air_wish_cap, h);
        }
        pos += glm::vec2(vel.x, vel.z) * h;
        for (int i = 0; i < MotionSchema::kHorizons; ++i) {
            if (done[i] || t + 1e-5f < horizons[i]) continue;
            done[i] = true;
            ++next;
            out.pos[i] = world_to_facing(pos, in.facing_yaw);
            out.dir[i] = glm::vec2(std::sin(turned), -std::cos(turned));
            if (horizons[i] >= max_h - 1e-5f) out.vel_end = world_to_facing(glm::vec2(vel.x, vel.z), in.facing_yaw);
        }
    }
    return out;
}

}  // namespace qlike::anim
