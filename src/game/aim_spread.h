#pragma once
// Weapon spread: how far from the middle of the crosshair a shot may go.
//
// A shot leaves inside a cone round the view direction. The cone's half-angle is
//   a base (from the hip; none when aiming)
// + the "heat" of the burst (every shot adds, it cools quickly once the trigger rests - so
//   short bursts stay accurate and a held trigger does not), up to a maximum
// + a little for moving and for being in the air.
// The crosshair shows the same angle: its four arms stand that far from the middle.
//
// Pure arithmetic, no engine types: the engine ticks it (combat.cpp spread_tick), asks it for
// each shot's direction (spread_shot_dir) and draws it (ui_hud.cpp). All the numbers are in the
// two tables below.

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace vibe::game {

// One weapon's numbers. Angles: half-angle of the cone, in degrees.
struct SpreadDef {
    float hip_base;        // from the hip, standing still, first shot
    float hip_per_shot;    // each shot adds this ...
    float hip_bloom_max;   // ... up to this much on top of the base
    float ads_base;        // aiming, first shot (0: the shot goes where the sights are)
    float ads_per_shot;    // aiming, each shot of a held trigger adds this ...
    float ads_bloom_max;   // ... up to this much
    float move_add;        // from the hip at full walking speed (in proportion below that)
    float air_add;         // from the hip, off the ground (jumping, falling)
    float ads_move_scale;  // aiming: this fraction of the two above
    float crouch_scale;    // crouched: the hip base times this
    float recover_delay;   // seconds after a shot before the heat starts to go (longer than the gap between two shots of a held trigger)
    float recover_rate;    // then: shots' worth of heat lost per second
};

// The rifle: automatic, 11 rounds a second. From the hip 1.1 degrees, 4.1 after ten rounds;
// aimed 0, 0.9 after ten rounds. A burst of three, aimed, stays within 0.2 degrees, and
// the heat of it is gone a quarter of a second after the last round.
inline constexpr SpreadDef kRifleSpread{
    .hip_base = 1.1f, .hip_per_shot = 0.30f, .hip_bloom_max = 3.0f,
    .ads_base = 0.0f, .ads_per_shot = 0.09f, .ads_bloom_max = 0.9f,
    .move_add = 0.7f, .air_add = 1.6f, .ads_move_scale = 0.3f, .crouch_scale = 0.75f,
    .recover_delay = 0.12f, .recover_rate = 22.0f,
};
// The laser gun: a little tighter than the rifle all round.
inline constexpr SpreadDef kLaserSpread{
    .hip_base = 0.8f, .hip_per_shot = 0.22f, .hip_bloom_max = 2.2f,
    .ads_base = 0.0f, .ads_per_shot = 0.06f, .ads_bloom_max = 0.6f,
    .move_add = 0.5f, .air_add = 1.2f, .ads_move_scale = 0.3f, .crouch_scale = 0.75f,
    .recover_delay = 0.12f, .recover_rate = 22.0f,
};

// What the shooter is doing this moment.
struct SpreadInput {
    float aim = 0.0f;          // 0 = from the hip, 1 = aiming (eased in between)
    float move = 0.0f;         // horizontal speed / walking speed, 0 .. 1
    bool  airborne = false;
    float crouch = 0.0f;       // 0 = standing, 1 = crouched
};

struct AimSpread {
    float heat = 0.0f;         // in shots
    float since_shot = 1e3f;   // seconds
    float shown_deg = 0.0f;    // what the crosshair shows: the angle, eased (a shot does not make it jump)

    // The cone's half-angle now, degrees.
    static float angle_deg(const SpreadDef& d, float heat, const SpreadInput& in) {
        const float moving = d.move_add * std::clamp(in.move, 0.0f, 1.0f) + (in.airborne ? d.air_add : 0.0f);
        const float hip = d.hip_base * (1.0f + (d.crouch_scale - 1.0f) * std::clamp(in.crouch, 0.0f, 1.0f)) +
                          std::min(heat * d.hip_per_shot, d.hip_bloom_max) + moving;
        const float ads = d.ads_base + std::min(heat * d.ads_per_shot, d.ads_bloom_max) + moving * d.ads_move_scale;
        const float a = std::clamp(in.aim, 0.0f, 1.0f);
        return hip + (ads - hip) * a;
    }
    float angle_deg(const SpreadDef& d, const SpreadInput& in) const { return angle_deg(d, heat, in); }

    // A shot was fired (ask for its direction first: the shot that starts a burst is the accurate one).
    void on_shot(const SpreadDef& d) {
        const float cap = std::max(d.hip_bloom_max / std::max(d.hip_per_shot, 1e-4f),
                                   d.ads_bloom_max / std::max(d.ads_per_shot, 1e-4f));
        heat = std::min(heat + 1.0f, cap);
        since_shot = 0.0f;
    }

    // Time passes. `target_deg`: the angle the crosshair should show (0 with no gun in the hand).
    void tick(const SpreadDef& d, float dt, float target_deg) {
        since_shot += dt;
        if (since_shot > d.recover_delay) heat = std::max(0.0f, heat - d.recover_rate * dt);
        shown_deg += (target_deg - shown_deg) * (1.0f - std::exp(-dt / 0.045f));
    }

    // A direction inside the cone of `deg` round `forward` (unit), every part of the disc as
    // likely as any other. u, v: two random numbers in 0 .. 1.
    static glm::vec3 perturb(glm::vec3 forward, float deg, float u, float v) {
        if (deg <= 1e-4f) return forward;
        const glm::vec3 right = std::abs(forward.y) < 0.99f ? glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)))
                                                           : glm::vec3(1.0f, 0.0f, 0.0f);
        const glm::vec3 up = glm::cross(right, forward);
        const float r = std::sqrt(std::clamp(u, 0.0f, 1.0f)) * std::tan(glm::radians(deg));
        const float th = v * 6.2831853f;
        return glm::normalize(forward + right * (r * std::cos(th)) + up * (r * std::sin(th)));
    }
};

} // namespace vibe::game
