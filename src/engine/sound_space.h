#pragma once
// SOUND IN SPACE: what a source that moves (a car, a siren) sounds like from where the listener
// is. The curves, as plain functions of numbers - no audio library, no engine: audio.cpp uses
// them on the game thread once a frame for every source, tests/test_sound_space.cpp checks
// that none of them steps, and docs/ARCHITECTURE.md "Sound in space" has their table (it is
// this file's output: VIBE_LOG_SOUND_CURVE=1).
//
// Set by numbers. The constants an ear may want to turn are all in this file, each with what
// it does; the reverb's own are in sound_dsp.h (kTail*, kEarly*).
//
//   LOUDNESS   gain(d) = ref / sqrt(ref^2 + d^2)            the inverse of the distance (-6 dB a
//                                                           doubling) from a reference distance,
//                        x 10^(-kExcessDbPerM d / 20)       what ground and air take on top of it,
//                        x (1 - smoothstep(kFadeFrom, kHear, d))   and a window to exactly nothing.
//              One curve from beside the source to silence: no step anywhere, and at kHear the
//              gain is so far down (-60 dB before the window) that the end is not heard.
//   TONE       cutoff(d) = top / (1 + d / kAirHalfM)        the air takes the highs first: a
//              low-pass (two one-pole stages, 12 dB an octave, soft knee) that comes down with
//              the distance. `top` is the source's own brightness heard from beside it.
//   DOPPLER    ratio = (c - v_listener) / (c - v_source)    both speeds along the line between
//              them; clamped, and nothing within a metre (the line swings round there).
//   DIRECTION  equal-power panning by the source's bearing, the far ear later (kEarDelayS) and
//              duller (kShadow), a source behind duller (kRearDull).
//   BLOCKED    occlusion 0..1 (a hill or a building in the line): quieter and much duller.
//   SPACE      what is sent to the reverb falls slower with distance than the direct sound
//              (kSendLevel gain^kSendPower): far off, a source is heard more by what comes back from the
//              surroundings than by itself.
#include <algorithm>
#include <cmath>

namespace vibe::sound_space {

constexpr float kSpeedOfSound = 343.0f;     // m/s

// --- loudness ---
constexpr float kExcessDbPerM = 0.06f;      // ground and air, on top of the distance law (dB a metre): more = quieter far off
constexpr float kFadeFrom     = 300.0f;     // the window to nothing begins here (m)
constexpr float kHear         = 450.0f;     // ... and ends here: beyond it a source has no voice (m)
constexpr float kFloor        = 0.00025f;   // -72 dB: under it a voice is not mixed at all
// --- tone ---
constexpr float kAirHalfM     = 30.0f;      // the cut-off is halved at this distance (m): less = duller far off
constexpr float kMinCutoffHz  = 300.0f;
// --- Doppler ---
constexpr float kDopplerMin   = 0.75f, kDopplerMax = 1.35f;
constexpr float kDopplerAmount = 1.0f;      // 1 = the physical shift; less = milder
// --- direction ---
constexpr float kPanWidth     = 0.85f;      // 1 = a source at the side is in one ear only
constexpr float kEarDelayS    = 0.00066f;   // the far ear hears it this much later, at the most (s)
constexpr float kShadow       = 0.70f;      // how much of the far ear's highs the head takes, at the most
constexpr float kRearDull     = 0.30f;      // a source straight behind: cut-off x (1 - this)
// --- blocked ---
constexpr float kOccGain      = 0.40f;      // fully blocked: loudness x this (-8 dB)
constexpr float kOccCutoff    = 0.28f;      // ... and cut-off x this
// --- space ---
constexpr float kSendLevel    = 0.50f;      // how loud the reverb is beside the source, against the source itself (in a place that is all walls)
constexpr float kSendPower    = 0.80f;      // 1 = the reverb falls as the direct sound does; less = a far source is more echo than itself

// What kind of source: how far it carries and how bright it is from beside it.
// The last three are for a source that is made to carry (a siren, a horn): it is heard further
// than an engine, loses less to ground and air a metre, and keeps its highs longer. The window
// to nothing begins at two thirds of hear_m (as kFadeFrom is of kHear).
struct Shape {
    float ref_m  = 6.0f;        // full loudness within about this distance, then the inverse of the distance
    float top_hz = 4500.0f;     // its low-pass heard from beside it (the -3 dB point)
    float hear_m = kHear;       // beyond this it has no voice (m)
    float excess_db_m = kExcessDbPerM;      // ground and air, on top of the distance law (dB a metre)
    float air_half_m  = kAirHalfM;          // the cut-off is halved at this distance (m)
};
// A SIREN: made to be heard. Beside it as loud as it is; at 300 m clearly there (-30 dB), at
// 600 m faint (-42 dB), at 800 m almost gone (-52 dB), at 1 km nothing. Its tone (650 - 1500 Hz
// and overtones) comes through the air's low-pass to the end: 2.6 kHz at 300 m, 1 kHz at 1 km.
constexpr Shape kSirenShape{ 20.0f, 9000.0f, 1000.0f, 0.02f, 120.0f };

inline float smooth01(float t) { t = std::clamp(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }

inline float gain(float d, const Shape& s) {
    d = (std::max)(d, 0.0f);
    const float from = s.hear_m * (kFadeFrom / kHear);
    const float window = 1.0f - smooth01((d - from) / (std::max)(s.hear_m - from, 1.0f));
    return s.ref_m / std::sqrt(s.ref_m * s.ref_m + d * d) * std::pow(10.0f, -s.excess_db_m * d / 20.0f) * window;
}
inline float gain(float d, float ref_m) { Shape s; s.ref_m = ref_m; return gain(d, s); }

inline float cutoff_hz(float d, float top_hz, float air_half_m = kAirHalfM) {
    return (std::max)(top_hz / (1.0f + (std::max)(d, 0.0f) / air_half_m), kMinCutoffHz);
}

// to_listener: the unit vector from the source to the listener. Speeds in m/s along it:
// v_src_toward > 0 when the source comes nearer, v_lis_away > 0 when the listener flees.
inline float doppler(float v_src_toward, float v_lis_away, float d) {
    const float lim = 0.5f * kSpeedOfSound;
    const float r = (kSpeedOfSound - std::clamp(v_lis_away, -lim, lim)) / (kSpeedOfSound - std::clamp(v_src_toward, -lim, lim));
    const float apart = smooth01((d - 1.0f) / 4.0f);     // (beside the source the line to it swings round: no shift there)
    return std::clamp(1.0f + (r - 1.0f) * apart * kDopplerAmount, kDopplerMin, kDopplerMax);
}

// Where it is heard. side: -1 full left .. +1 full right (the source's direction along the
// listener's right hand); front: +1 ahead .. -1 behind; d: metres.
struct Ears {
    float gain_l = 0.7071f, gain_r = 0.7071f;     // equal power: l^2 + r^2 = 1
    float delay_l_s = 0.0f, delay_r_s = 0.0f;     // the far ear's lateness
    float shadow_l = 0.0f, shadow_r = 0.0f;       // the far ear's loss of highs, 0..kShadow
    float rear = 0.0f;                            // 0 ahead or beside .. 1 straight behind
    float pan = 0.0f;
};
inline Ears ears(float side, float front, float d) {
    Ears e;
    // (a source at the listener's feet has no side: within a couple of metres it comes to the middle)
    e.pan = std::clamp(side, -1.0f, 1.0f) * (d / (d + 1.5f));
    const float th = (e.pan * kPanWidth + 1.0f) * 0.78539816f;
    e.gain_l = std::cos(th); e.gain_r = std::sin(th);
    const float right = (std::max)(e.pan, 0.0f), left = (std::max)(-e.pan, 0.0f);
    e.delay_l_s = kEarDelayS * right; e.delay_r_s = kEarDelayS * left;
    e.shadow_l = kShadow * right;     e.shadow_r = kShadow * left;
    e.rear = smooth01(-front) * (d / (d + 1.5f));
    return e;
}

inline float occluded_gain(float occ)   { return 1.0f + (kOccGain - 1.0f) * std::clamp(occ, 0.0f, 1.0f); }
inline float occluded_cutoff(float occ) { return 1.0f + (kOccCutoff - 1.0f) * std::clamp(occ, 0.0f, 1.0f); }

// The share of a source that goes to the reverb (before the surroundings' own amount).
inline float send(float dist_gain) { return kSendLevel * std::pow((std::max)(dist_gain, 0.0f), kSendPower); }

// THE SURROUNDINGS, as the game finds them a few times a second (vk_engine/sound_env.cpp) and
// the audio engine follows (AudioEngine::set_space). All 0..1 unless said.
struct Space {
    float amount     = 0.04f;     // how much comes back: an open field almost nothing, between walls / under a bridge / in a valley a lot
    float decay_s    = 0.5f;      // how long the tail rings (to -60 dB)
    float damp_hz    = 3500.0f;   // the tail's own loss of highs (woods: low)
    float early      = 0.3f;      // the first reflections' share (hard surfaces near: high)
    float slap_s     = 0.0f;      // a single wall or cliff: its echo comes back after this long ...
    float slap       = 0.0f;      // ... this loud (0: none) ...
    float slap_pan   = 0.0f;      // ... from this side (-1 left .. +1 right)
    float outside_gain = 1.0f;    // the listener sits in a cabin or stands in a building: what is outside is this much quieter ...
    float outside_hz   = 20000.0f;// ... and through this low-pass at the most
};

}  // namespace vibe::sound_space
