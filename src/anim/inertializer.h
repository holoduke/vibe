#pragma once
// Inertialisation (Bollo, GDC 2018; the spring form of Holden): when playback jumps from one
// motion to another, the difference between what was shown and the new motion - in pose AND
// in rate of change - is carried on as an offset that decays like a critically damped spring.
// Nothing blends two clips; the new motion is followed exactly once the offset has died.
//
// Rotations are handled as rotation vectors of the difference quaternion taken the short way
// round, so the q / -q double cover cannot cause a spin. Translations (the hips) likewise as
// plain vector offsets. The character's root is not touched: the controller owns it.
#include "anim/types.h"

namespace qlike::anim {

class Inertializer {
public:
    void reset(int joints);
    // Playback is about to continue from `dst` instead of what apply() last produced.
    // `dst_next` is the new motion `dst_dt` seconds later (its rate of change).
    void transition(std::span<const Transform> dst, std::span<const Transform> dst_next, float dst_dt);
    // out = the source pose plus what is left of the offsets, which decay by dt.
    void apply(std::span<const Transform> src, float dt, float halflife, std::span<Transform> out);
    bool  active() const { return energy_ > 1e-5f; }
    float energy() const { return energy_; }           // sum of offset magnitudes (rad + m), for debug views

private:
    std::vector<glm::vec3> rx_, rv_, tx_, tv_;         // rotation / translation offsets and their rates
    std::vector<Transform> last_, prev_;               // what apply() produced last and the time before
    float last_dt_ = 0.0f;
    int   have_ = 0;                                   // how many outputs exist (0, 1, 2+)
    float energy_ = 0.0f;
};

}  // namespace qlike::anim
