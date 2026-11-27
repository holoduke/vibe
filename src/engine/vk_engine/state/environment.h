#pragma once
// Environment: time of day and weather (environment.cpp): its state.
//
// Private to the files that implement it - vk_engine.h only knows the name of this struct
// and holds a pointer to it (env_), so a change here rebuilds those files and not the
// whole engine. Created and freed in vk_engine/private_state.cpp.

#include "engine/vk_engine.h"

namespace vibe {

struct VulkanEngine::EnvState {
    EnvBase base{};              // the user's values while the driver overrides rt_
    bool  applied = false;
    float rain = 0.0f;           // rain intensity 0..1
    float snow = 0.0f;           // snow lying on the ground 0..1
    float snowfall = 0.0f;       // snow falling 0..1
    float wet = 0.0f;            // ground wetness 0..1
    // Dusk / dawn (environment.cpp): warm-tint amount, the real sun's
    // elevation (-1000 = no clock), and the sky's own sun direction while
    // the sun is under the horizon.
    float golden = 1.0f, sun_elev = -1000.0f;
    bool  sky_real = false, sky_init = false;
    float sky_pitch = 0.0f, sky_yaw = 0.0f, sky_ease = 1.0f;
    bool  moon_light = false;     // the night sky model in use: moon (true) or the glow of the sun under the horizon
    float amb_k = 1.0f;          // night fill-light factor (1 by day and on a clear full-moon night)
    float moon_vis = 1.0f;       // moon not hidden by terrain from the player (eased 0..1)
    float moon = 1.0f;           // moonlight in the sky right now, 0..1 (phase, height, setting)
    // Where the moon and the real sun stand (unit vectors toward them; the moon is drawn as a
    // sphere lit by the sun - compose.frag atmo_moon). light_is_sun: the scene's directional
    // light is the sun itself (not the moon, not the glow under the horizon).
    glm::vec3 moon_dir{0.0f, -1.0f, 0.0f}, sun_dir{0.0f, 1.0f, 0.0f};
    bool  light_is_sun = true;
    float night = -1.0f;         // lantern factor 0..1 (< 0: clock off)
    float lightning = 0.0f;      // current flash brightness
    float wind_scale = 1.0f;
    // The wind on the water (apply_environment; scene_ubo.glsl water_wind): xy = the way it
    // blows (unit), z = speed m/s, w = the swell's height scale pow(saturate(z / 30), 2.2).
    glm::vec4 water_wind{ 0.944f, 0.330f, 7.0f, 0.00718f };
    // The wind the OPEN SEA's waves are made by (m/s; environment.cpp water_wind_now): the weather's
    // wind, but never under a sea breeze that grows with it - the wind over a hundred kilometres of
    // open water, not the still air in a valley. It sets the FFT cascades (water_sim.cpp) and the
    // swell's scale (water_wind.w); scene water_sun.z for the sea's whitecaps and glint.
    float sea_wind = 7.0f;
    // The way the wind direction has carried the water's drifting patterns: the sum of the
    // direction over time (m per m/s of drift; scene_ubo.glsl water_swell.yz). Summed, not
    // direction x clock - that swung the patterns round whenever the wind veered.
    glm::vec2 water_drift{ 0.0f, 0.0f };
    float cloud_k = 1.0f, cloud_direct = 1.0f;   // cloud light: sky level, direct factor
    float cloud_shadow_k = 0.72f;                     // cloud-shadow contrast on the ground
};

} // namespace vibe
