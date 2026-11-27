#pragma once
// The rifle's sound as a machine gun, made here: no recordings (as engine/glass_synth.h).
//
//   report(variant)  0.42 s, one round heard by whoever fires it: the muzzle blast (a very short,
//                    very bright burst of noise with a low thump under it - the pressure wave),
//                    the supersonic crack of the bullet leaving, the gun's own action a few
//                    hundredths later (the bolt slamming home: a dry metallic clack, a couple of
//                    ringing partials), and the first, close reflections. Three variants so a
//                    burst at 11 rounds a second is not one sample hammered.
//   tail()           1.6 s, the same shot from the surroundings: no attack, a low rolling
//                    rumble that dies away with three slap-back echoes (a wall, a slope, the
//                    woods). Played through the spatialiser as a source of its own at the gun,
//                    so it carries, comes from where the gun is and is darker far away.
//
// write_wav() (glass_synth.h) writes them as 16-bit mono WAV at 44.1 kHz. combat.cpp writes
// assets/sounds/mg_shot_v1_*.wav and mg_tail_v1.wav the first time they are needed and are not
// there. A change of the sound: count the v1 in the names up, so the files are made again.
#include "engine/glass_synth.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace vibe::gun_synth {

using glass_synth::kRate;
using glass_synth::Rng;

// Noise from t0 dying with tau, through a one-pole low-pass at `hz` (a rumble, a thump's body).
inline void dull(std::vector<float>& out, Rng& r, float t0, float tau, float amp, float hz, float attack_s = 0.002f) {
    const int i0 = std::max(0, int(t0 * float(kRate)));
    const int n = int(tau * 7.0f * float(kRate));
    const float a = 1.0f - std::exp(-6.2831853f * hz / float(kRate));
    float y = 0.0f, y2 = 0.0f;
    for (int i = 0; i < n && size_t(i0 + i) < out.size(); ++i) {
        const float t = float(i) / float(kRate);
        y += a * (r.pm() - y);
        y2 += a * (y - y2);                                          // (two poles: no hiss left on it)
        const float env = std::min(1.0f, t / std::max(attack_s, 1e-4f)) * std::exp(-t / tau);
        out[size_t(i0 + i)] += amp * env * y2 * 3.0f;
    }
}

// A low sine that drops in pitch as it dies (the blast's thump: a pressure wave, not a tone).
inline void thump(std::vector<float>& out, float t0, float hz0, float hz1, float tau, float amp) {
    const int i0 = std::max(0, int(t0 * float(kRate)));
    const int n = int(tau * 7.0f * float(kRate));
    float ph = 0.0f;
    for (int i = 0; i < n && size_t(i0 + i) < out.size(); ++i) {
        const float t = float(i) / float(kRate);
        const float hz = hz1 + (hz0 - hz1) * std::exp(-t / (tau * 0.6f));
        ph += 6.2831853f * hz / float(kRate);
        const float attack = std::min(1.0f, float(i) / 20.0f);
        out[size_t(i0 + i)] += amp * attack * std::exp(-t / tau) * std::sin(ph);
    }
}

inline std::vector<int16_t> report(int variant) {
    std::vector<float> f(size_t(kRate * 0.42f), 0.0f);
    Rng r{ 0x6A11u + uint32_t(variant) * 0x9E3779B9u };
    const float v = r.range(0.92f, 1.08f);                               // (each variant a hair different)
    // the blast
    glass_synth::burst(f, r, 0.0f, 0.0018f, 1.0f, 0.55f);
    glass_synth::burst(f, r, 0.0f, 0.009f, 0.55f, 0.35f);
    thump(f, 0.0f, 150.0f * v, 55.0f, 0.030f, 0.95f);
    dull(f, r, 0.0005f, 0.028f, 0.55f, 900.0f * v, 0.0008f);
    // the crack (just after: the bullet's own shock wave, brighter and thinner)
    glass_synth::burst(f, r, 0.0016f, 0.0012f, 0.45f, 0.95f);
    // the action: the bolt back and home, a dry clack with a short metallic ring
    const float t_act = r.range(0.030f, 0.038f);
    glass_synth::burst(f, r, t_act, 0.0016f, 0.16f, 0.85f);
    glass_synth::ring(f, t_act, r.range(2100.0f, 2600.0f), 0.010f, 0.07f, r.range(0.0f, 6.28f));
    glass_synth::ring(f, t_act, r.range(3900.0f, 4700.0f), 0.006f, 0.05f, r.range(0.0f, 6.28f));
    const float t_home = t_act + r.range(0.018f, 0.024f);
    glass_synth::burst(f, r, t_home, 0.0012f, 0.20f, 0.8f);
    glass_synth::ring(f, t_home, r.range(1500.0f, 1900.0f), 0.012f, 0.08f);
    glass_synth::ring(f, t_home, r.range(5200.0f, 6400.0f), 0.004f, 0.04f);
    // the body of the shot and the close reflections (the ground, the shooter's own surroundings)
    dull(f, r, 0.002f, 0.070f, 0.30f, 420.0f, 0.004f);
    dull(f, r, 0.011f, 0.050f, 0.16f, 1300.0f, 0.002f);
    dull(f, r, 0.024f, 0.090f, 0.12f, 700.0f, 0.006f);
    return glass_synth::to_pcm(f, 0.95f);
}

inline std::vector<int16_t> tail() {
    std::vector<float> f(size_t(kRate * 1.6f), 0.0f);
    Rng r{ 0x7A11u };
    dull(f, r, 0.0f, 0.22f, 0.55f, 260.0f, 0.020f);                      // the roll
    dull(f, r, 0.0f, 0.10f, 0.30f, 900.0f, 0.006f);
    thump(f, 0.0f, 90.0f, 45.0f, 0.12f, 0.35f);
    const float slaps[3] = { 0.085f, 0.21f, 0.47f };                     // three surfaces: near, mid, far
    const float level[3] = { 0.42f, 0.30f, 0.18f };
    for (int k = 0; k < 3; ++k) {
        dull(f, r, slaps[k], 0.035f + 0.02f * float(k), level[k], 1600.0f - 450.0f * float(k), 0.002f);
        dull(f, r, slaps[k], 0.16f + 0.08f * float(k), level[k] * 0.5f, 300.0f, 0.015f);
    }
    return glass_synth::to_pcm(f, 0.85f);
}

}  // namespace vibe::gun_synth
