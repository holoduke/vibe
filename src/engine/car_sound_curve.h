#pragma once
// A car's engine heard from outside: how loud and how dull it is against the listener's
// distance (jeep.cpp car_audio_tick; docs/ARCHITECTURE.md "Car sound, fourth pass"). Set by
// numbers - the table in the doc is this file's output (VIBE_LOG_CAR_SOUND_CURVE=1).
//
//   gain(d)      = 1 / (1 + (d / 12)^2)  x  (1 - smoothstep(90, 180, d))
//                  half at 12 m, then the square of the distance (-12 dB a doubling), and a
//                  window that takes it to exactly nothing at kHear - no step where it ends.
//   cutoff_hz(d) = 4500 / (1 + d / 12), not under 300
//                  the low-pass (one biquad a voice, 12 dB an octave): beside the car its top
//                  end is already gone (what is heard outside is exhaust and body, not the
//                  intake and valve gear the driver hears), and the further the duller, down
//                  to a drone.
//
// The car the player DRIVES does not use these (its sound is the cabin's or the chase view's,
// car_audio_tick).
#include <algorithm>
#include <cmath>

namespace vibe::car_sound {

constexpr float kHear = 180.0f;       // beyond it a car is not heard and has no voice (m)

inline float gain(float d) {
    d = (std::max)(d, 0.0f);
    const float t = std::clamp((d - 90.0f) / (kHear - 90.0f), 0.0f, 1.0f);
    const float window = 1.0f - t * t * (3.0f - 2.0f * t);
    const float q = d / 12.0f;
    return window / (1.0f + q * q);
}

inline float cutoff_hz(float d) {
    return std::clamp(4500.0f / (1.0f + (std::max)(d, 0.0f) / 12.0f), 300.0f, 4500.0f);
}

// As it was before 2026-11-25's fourth pass (VIBE_CAR_SOUND_OLD=1: to compare the mixer's cost).
constexpr float kHearOld = 240.0f;
inline float gain_old(float d) {
    const float t = std::clamp((d - 100.0f) / (kHearOld - 100.0f), 0.0f, 1.0f);
    return (1.0f - t * t * (3.0f - 2.0f * t)) / (1.0f + std::pow((std::max)(d, 0.0f) / 14.0f, 1.5f));
}
inline float cutoff_hz_old(float d) {
    const float t = std::clamp((d - 3.0f) / 117.0f, 0.0f, 1.0f);
    return 7000.0f * std::pow(350.0f / 7000.0f, t * t * (3.0f - 2.0f * t));
}

}  // namespace vibe::car_sound
