// Water effects: what happens when something hits the water plane.
//   * Splashes - a bullet (or a bomb / wreck) throws up a central jet
//     (the Worthington jet that rebounds out of the cavity), a crown of
//     droplets and a puff of mist. Droplets are CPU ballistic particles
//     (gravity + air drag) drawn as motion-blurred streaks by the FX pass;
//     the mist is closed-form in the splash age.
//   * Ripples - each impact starts a ring wave packet. The list goes to the
//     scene UBO (ripples[16]); the water shader adds the packet's slope to
//     its wave normal, so the rings bend reflections and glints.
//   * Droplets falling back start small secondary rings.
// Swimming lives in game/player.cpp; this file only decides WHEN (how deep
// the body is in open water).

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/audio.h"

#include <algorithm>
#include <cmath>

namespace qlike {

namespace {
constexpr float kG = 9.81f;
constexpr size_t kMaxRipples = 16;
constexpr size_t kMaxDrops = 1400;
}

// Open water at (x, z): the water plane is above the ground there.
bool VulkanEngine::water_at(float x, float z) const {
    return rt_.water_enabled && sample_terrain_height(x, z) < rt_.water_level - 0.02f;
}

void VulkanEngine::add_water_ripple(float x, float z, float amp) {
    WaterRipple r{ glm::vec2(x, z), 0.0f, amp };
    // How visible a ring still is: its amplitude times the fade the water
    // shader applies (fade^1.5 over its life).
    auto strength = [](const WaterRipple& w) {
        const float s = glm::clamp(w.amp * 28.5f, 1.0f, 6.0f);
        const float fade = glm::clamp(1.0f - w.age / (2.5f + 1.5f * s), 0.0f, 1.0f);
        return w.amp * fade * std::sqrt(fade);
    };
    // Small rings (falling droplets, wading) share a few slots between
    // themselves. A splash throws dozens of droplets; without this cap
    // their rings filled the list within a second and pushed out the big
    // ring of the impact itself, which then vanished mid-spread.
    constexpr float kSmallAmp = 0.015f;
    constexpr size_t kMaxSmall = 6;
    const bool is_small = amp < kSmallAmp;
    size_t n_small = 0, weakest = SIZE_MAX, weakest_small = SIZE_MAX;
    float weakest_k = 1.0e9f, weakest_small_k = 1.0e9f;
    for (size_t i = 0; i < water_ripples_.size(); ++i) {
        const float k = strength(water_ripples_[i]);
        if (k < weakest_k) { weakest_k = k; weakest = i; }
        if (water_ripples_[i].amp < kSmallAmp) {
            ++n_small;
            if (k < weakest_small_k) { weakest_small_k = k; weakest_small = i; }
        }
    }
    if (is_small && n_small >= kMaxSmall) { water_ripples_[weakest_small] = r; return; }
    if (water_ripples_.size() < kMaxRipples) { water_ripples_.push_back(r); return; }
    // Full: the new ring replaces the faintest one - never a stronger one.
    // and only one that has nearly faded, so a visible ring is not cut off
    // (under sustained fire some impacts then get a splash but no ring).
    if (weakest_k < 0.35f * amp) water_ripples_[weakest] = r;
}

// scale 1 = rifle bullet (jet ~1.3 m), ~3 = a body dropping in, 8+ = bomb.
void VulkanEngine::spawn_water_splash(glm::vec3 pos, float scale) {
    pos.y = rt_.water_level;
    scale = glm::clamp(scale, 0.3f, 16.0f);
    const float vs = std::sqrt(scale);            // ballistic height ~ v^2
    uint32_t& rng = spacejet_rng_state_;

    WaterSplash s{ pos, 0.0f, scale, frand_range(rng, 0.0f, 100.0f) };
    water_splashes_.push_back(s);
    if (water_splashes_.size() > 24) water_splashes_.erase(water_splashes_.begin());

    add_water_ripple(pos.x, pos.z, 0.035f * vs);

    auto drop = [&](glm::vec3 v, float size, float delay) {
        if (water_drops_.size() >= kMaxDrops) return;
        WaterDrop d{};
        d.pos = pos + glm::vec3(v.x, 0.0f, v.z) * 0.02f;
        d.vel = v;
        d.t = -delay;                               // negative = not yet released
        d.life = 4.0f + 2.0f * vs;
        d.size = size;
        water_drops_.push_back(d);
    };
    const float far_k = glm::clamp(1.0f - glm::distance(pos, player_.eye_position()) / 350.0f, 0.25f, 1.0f);
    // Crown: a ring of droplets thrown outward at 50-75 degrees as the
    // cavity opens.
    const int n_crown = int((26.0f + 10.0f * vs) * far_k);
    for (int i = 0; i < n_crown; ++i) {
        const float a = frand(rng) * 6.2831853f;
        const float el = frand_range(rng, 0.85f, 1.32f);
        const float sp = frand_range(rng, 2.2f, 4.6f) * vs;
        drop(glm::vec3(std::cos(a) * std::cos(el), std::sin(el), std::sin(a) * std::cos(el)) * sp,
             frand_range(rng, 0.010f, 0.022f) * vs, frand_range(rng, 0.0f, 0.04f));
    }
    // Central jet: the cavity collapses and squirts a thin column straight
    // up a moment later; it breaks into beads, fastest at the tip.
    const int n_jet = int((14.0f + 6.0f * vs) * far_k);
    for (int i = 0; i < n_jet; ++i) {
        const float k = float(i) / float(std::max(1, n_jet - 1));
        const float a = frand(rng) * 6.2831853f;
        const float side = frand_range(rng, 0.0f, 0.35f) * vs;
        const float up = glm::mix(2.6f, 5.6f, k) * vs * frand_range(rng, 0.92f, 1.08f);
        drop(glm::vec3(std::cos(a) * side, up, std::sin(a) * side),
             glm::mix(0.026f, 0.012f, k) * vs, 0.07f + 0.05f * k);
    }
    // Fine spray: many tiny fast droplets that stall quickly in the air.
    const int n_spray = int((18.0f + 8.0f * vs) * far_k);
    for (int i = 0; i < n_spray; ++i) {
        const float a = frand(rng) * 6.2831853f;
        const float el = frand_range(rng, 0.35f, 1.45f);
        const float sp = frand_range(rng, 3.0f, 8.5f) * vs;
        drop(glm::vec3(std::cos(a) * std::cos(el), std::sin(el), std::sin(a) * std::cos(el)) * sp,
             frand_range(rng, 0.004f, 0.008f) * vs, 0.0f);
    }

    if (audio_) {
        const float d = glm::distance(pos, player_.eye_position());
        if (d < 260.0f)
            audio_->play_at(ClipID::Splash, pos, glm::clamp(0.45f + 0.18f * scale, 0.3f, 1.6f),
                            0.12f, 0.08f);
    }
}

void VulkanEngine::tick_water_fx(float dt) {
    if (dt <= 0.0f) return;
    dt = std::min(dt, 0.05f);
    for (auto& r : water_ripples_) r.age += dt;
    water_ripples_.erase(std::remove_if(water_ripples_.begin(), water_ripples_.end(),
                                        [](const WaterRipple& r) {
                                            const float s = glm::clamp(r.amp * 28.5f, 1.0f, 6.0f);
                                            return r.age > 2.5f + 1.5f * s;
                                        }),
                         water_ripples_.end());
    for (auto& s : water_splashes_) s.t += dt;
    water_splashes_.erase(std::remove_if(water_splashes_.begin(), water_splashes_.end(),
                                         [](const WaterSplash& s) { return s.t > 1.2f + 0.9f * std::sqrt(s.scale); }),
                          water_splashes_.end());

    const float wl = rt_.water_level;
    int rings = 0;
    for (size_t i = 0; i < water_drops_.size();) {
        WaterDrop& d = water_drops_[i];
        d.t += dt;
        if (d.t < 0.0f) { ++i; continue; }
        // Air drag: small droplets stall fast (terminal ~ 9 m/s at 3 mm,
        // far less for mist-sized ones) - drag ~ 1 / size.
        const float v = glm::length(d.vel);
        const float kd = 0.0035f / std::max(d.size, 0.002f);
        d.vel.y -= kG * dt;
        d.vel /= (1.0f + kd * v * dt);
        d.pos += d.vel * dt;
        bool dead = d.t > d.life;
        if (!dead && d.vel.y < 0.0f && d.pos.y <= wl) {
            dead = true;
            // The bigger beads ring the surface where they land.
            if (d.size > 0.016f && rings < 2 && water_at(d.pos.x, d.pos.z)) {
                add_water_ripple(d.pos.x, d.pos.z, 0.010f);
                ++rings;
            }
        } else if (!dead && d.vel.y < 0.0f && d.pos.y <= sample_terrain_height(d.pos.x, d.pos.z)) {
            dead = true;
        }
        if (dead) { water_drops_[i] = water_drops_.back(); water_drops_.pop_back(); }
        else ++i;
    }
}

}  // namespace qlike
