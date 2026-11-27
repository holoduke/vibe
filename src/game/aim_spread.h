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

// VIEW RECOIL: what a held trigger does to the view itself. Every shot kicks the muzzle up and a
// little to one side; the side wanders (a few shots one way, then the other, leaning one way on
// the whole - a right-handed shooter's rifle pulls up and to the right). The kick is given over
// a few hundredths of a second (no jump). What the player pulls back against the climb is
// theirs: it is taken off what is owed, so when firing stops the view only goes back by what
// the player did NOT counter, eased, after a short pause. Together with the spread above: the
// cone (where the round goes round the crosshair) and this (where the crosshair itself goes).
// Angles in degrees.
struct RecoilDef {
    float pitch_per_shot;   // up, each shot (the first shots a little less: first_scale)
    float first_scale;      // the first shot of a burst kicks this share; it grows to 1 over ramp_shots
    float ramp_shots;
    float yaw_per_shot;     // the sideways step's size
    float yaw_bias;         // -1 .. 1: which way the side drifts on the whole (+ right)
    float aim_scale;        // aimed: this share of all of it (braced in the shoulder)
    float crouch_scale;     // crouched: this share
    float kick_tau;         // s: how fast a kick is given
    float recover_delay;    // s after the last shot before the view goes back
    float recover_tau;      // s: how fast it goes back (time constant)
    float recover_share;    // share of what is still owed that is given back (the rest stays: the view drifted)
};
// The rifle at 11 rounds a second: ten rounds from the hip climb about 4.5 degrees and drift a
// degree or so; aimed about 2.9 and 0.6. Back within half a second of letting go.
inline constexpr RecoilDef kRifleRecoil{
    .pitch_per_shot = 0.48f, .first_scale = 0.55f, .ramp_shots = 3.0f,
    .yaw_per_shot = 0.16f, .yaw_bias = 0.35f, .aim_scale = 0.65f, .crouch_scale = 0.8f,
    .kick_tau = 0.035f, .recover_delay = 0.10f, .recover_tau = 0.16f, .recover_share = 0.85f,
};
// The laser gun: hardly any (a light weapon that throws no brass).
inline constexpr RecoilDef kLaserRecoil{
    .pitch_per_shot = 0.14f, .first_scale = 0.6f, .ramp_shots = 2.0f,
    .yaw_per_shot = 0.06f, .yaw_bias = 0.0f, .aim_scale = 0.7f, .crouch_scale = 0.85f,
    .kick_tau = 0.03f, .recover_delay = 0.08f, .recover_tau = 0.14f, .recover_share = 0.9f,
};

struct ViewRecoil {
    float owe_pitch = 0.0f, owe_yaw = 0.0f;     // degrees the view is away from where the player put it (still to be given back)
    float due_pitch = 0.0f, due_yaw = 0.0f;     // degrees of kick not yet given to the view
    float since_shot = 1e3f;
    float shots = 0.0f;                         // in this burst
    float drift = 0.0f;                         // the side's wander, -1 .. 1

    // A shot. u: a random number 0 .. 1. stance: aim (0..1), crouch (0..1).
    void on_shot(const RecoilDef& d, float u, float aim, float crouch) {
        if (since_shot > d.recover_delay + 0.15f) shots = 0.0f;
        const float ramp = d.first_scale + (1.0f - d.first_scale) * std::clamp(shots / std::max(d.ramp_shots, 1e-3f), 0.0f, 1.0f);
        const float k = ramp * (1.0f + (d.aim_scale - 1.0f) * std::clamp(aim, 0.0f, 1.0f)) *
                        (1.0f + (d.crouch_scale - 1.0f) * std::clamp(crouch, 0.0f, 1.0f));
        drift = std::clamp(drift * 0.8f + (u * 2.0f - 1.0f) * 0.9f, -1.0f, 1.0f);
        due_pitch += d.pitch_per_shot * k;
        due_yaw += d.yaw_per_shot * k * std::clamp(drift + d.yaw_bias, -1.0f, 1.0f);
        shots += 1.0f;
        since_shot = 0.0f;
    }
    // Time passes. `user_pitch`: degrees the PLAYER turned the view (up +) since the last call,
    // yaw likewise. Returns, in out_pitch / out_yaw, the degrees to add to the view now.
    void tick(const RecoilDef& d, float dt, float user_pitch, float user_yaw, float& out_pitch, float& out_yaw) {
        since_shot += dt;
        // (what the player pulled against the climb is no longer owed)
        if (owe_pitch > 0.0f && user_pitch < 0.0f) owe_pitch = std::max(0.0f, owe_pitch + user_pitch);
        if (owe_yaw * user_yaw < 0.0f) owe_yaw = std::abs(user_yaw) >= std::abs(owe_yaw) ? 0.0f : owe_yaw + user_yaw;
        const float g = 1.0f - std::exp(-dt / std::max(d.kick_tau, 1e-3f));
        float p = due_pitch * g, y = due_yaw * g;
        due_pitch -= p; due_yaw -= y;
        owe_pitch += p; owe_yaw += y;
        if (since_shot > d.recover_delay) {
            const float r = 1.0f - std::exp(-dt / std::max(d.recover_tau, 1e-3f));
            const float bp = owe_pitch * r * d.recover_share, by = owe_yaw * r * d.recover_share;
            p -= bp; y -= by;
            owe_pitch -= owe_pitch * r; owe_yaw -= owe_yaw * r;          // (the share not given back is let go)
        }
        out_pitch = p; out_yaw = y;
    }
};

} // namespace vibe::game
