// Environment driver: day / night clock + weather.
//
// The renderer already has every knob a sky needs (sun direction, physical
// atmosphere, volumetric clouds, froxel fog, wind). This file does not add a
// second lighting path - once per frame it DRIVES those inputs:
//
//   clock   -> sun position on a real diurnal arc (latitude 47 N, late
//              spring: sunrise ~04:50, noon elevation ~59 deg, sunset ~19:10).
//              Below the horizon the physical atmosphere produces twilight by
//              itself; just after sunset the light source becomes
//              the moon (opposite the sun, dim and blue), stars come out
//              (compose.frag draws them wherever the sky is dark) and the
//              road lanterns light up.
//   weather -> one scalar 0 (clear) .. 1 (thunderstorm) that wanders between
//              randomly chosen targets. It raises cloud cover and density,
//              thickens the fog, dims and greys the sunlight, strengthens the
//              wind and, past ~0.55, makes it rain: streaks around the camera
//              (fx.cpp), rings on the water, wet (darker) ground, rain sound,
//              lightning + thunder in storms.
//
// The user's own values for the driven settings are kept in env_->base and
// swapped back for saving, so the cfg always holds what the user chose.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/clouds.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_engine/state/multiplayer.h"
#include "engine/vk_engine/state/spacejets.h"
#include "engine/globe.h"
#include "engine/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace vibe {

// rt_ <-> env_->base for every driven field (its own inverse).
void VulkanEngine::env_swap_base() {
    std::swap(rt_.sun_pitch_deg, env_->base.sun_pitch_deg);
    std::swap(rt_.sun_yaw_deg, env_->base.sun_yaw_deg);
    std::swap(rt_.sun_intensity, env_->base.sun_intensity);
    std::swap(rt_.sun_color, env_->base.sun_color);
    std::swap(rt_.sky_color, env_->base.sky_color);
    std::swap(rt_.atmosphere_intensity, env_->base.atmosphere_intensity);
    std::swap(rt_.cloud_coverage, env_->base.cloud_coverage);
    std::swap(rt_.cloud_density, env_->base.cloud_density);
    std::swap(rt_.cloud_speed, env_->base.cloud_speed);
    std::swap(rt_.fog_density, env_->base.fog_density);
    std::swap(rt_.grass_wind, env_->base.grass_wind);
    std::swap(rt_.auto_golden_hour, env_->base.auto_golden_hour);
}

// CPU copy of scene_ubo.glsl cloud_shadow() (keep in sync): how much direct
// sun the clouds let through at a world point.
float VulkanEngine::cloud_shadow_at(glm::vec3 wp) const {
    const float k = cloud_->shadow_k_now;
    if (k <= 0.001f) return 1.0f;
    auto hash = [](float x, float y, float z) {
        const uint32_t qx = uint32_t(int32_t(x)) * 1597334673u, qy = uint32_t(int32_t(y)) * 3812015801u,
                       qz = uint32_t(int32_t(z)) * 2798796415u;
        return float((qx ^ qy ^ qz) * 1597334673u) * (1.0f / 4294967295.0f);
    };
    auto hvalue = [&](glm::vec2 p) {
        const glm::vec2 i = glm::floor(p);
        glm::vec2 f = p - i;
        f = f * f * (3.0f - 2.0f * f);
        const float a = hash(i.x, i.y, 7.0f), b = hash(i.x + 1.0f, i.y, 7.0f);
        const float c = hash(i.x, i.y + 1.0f, 7.0f), d = hash(i.x + 1.0f, i.y + 1.0f, 7.0f);
        return glm::mix(glm::mix(a, b, f.x), glm::mix(c, d, f.x), f.y);
    };
    const float pr = glm::radians(rt_.sun_pitch_deg), yr = glm::radians(rt_.sun_yaw_deg);
    const glm::vec3 L(std::cos(pr) * std::sin(yr), std::sin(pr), std::cos(pr) * std::cos(yr));
    const float size = std::max(cloud_->a.y, 0.2f);
    const float alt = cloud_->a.z + 0.30f * cloud_->a.w;
    const glm::vec2 p = glm::vec2(wp.x, wp.z) + glm::vec2(L.x, L.z) / std::max(L.y, 0.12f) * std::max(alt - wp.y, 0.0f);
    const glm::vec2 q = (p + glm::vec2(cloud_->b)) / (6500.0f * size);
    const float wn = hvalue(q) * 0.7f + hvalue(q * 2.17f + 13.0f) * 0.3f;
    const float cov = glm::clamp(cloud_->a.x, 0.0f, 1.0f);
    const float map_cov = glm::smoothstep(0.2f, 0.8f, glm::clamp(glm::smoothstep(0.2f, 0.8f, wn) + (cov - 0.5f) * 1.5f, 0.0f, 1.0f));
    const glm::vec2 qs = (p + glm::vec2(cloud_->b) * 1.6f) / (1900.0f * size);
    const float lump = hvalue(qs) * 0.6f + hvalue(qs * 2.3f + 5.0f) * 0.4f;
    const float opac = glm::smoothstep(0.30f, 0.78f, map_cov * (0.62f + 0.75f * lump)) * glm::clamp(cloud_->b.z, 0.3f, 1.5f);
    return 1.0f - k * glm::clamp(opac, 0.0f, 1.0f);
}

// The wind on the water (scene_ubo.glsl water_wind; terrain_rm_water.glsl, water_fx.cpp):
// xy = the way it blows, z = speed (m/s), w = the swell's height scale. The speed follows the
// weather's wind amount (clear 0.1 -> 2.8 m/s, cloudy 0.25 -> 6, rain 0.45 -> 11, a storm 1.0
// -> 28); without the weather a moderate 7 m/s. The direction is the clouds' drift
// (1, 0.35), veering slowly by up to about 25 degrees.
// VIBE_TEST_WATER_WIND=<m/s>[,<deg>]: that speed (and direction, degrees from +x toward +z).
// (Not VIBE_TEST_WIND: that one stirs a vortex into the wind field, wind_fluid.cpp.)
static glm::vec4 water_wind_now(bool weather_on, float wind_k, float t, float* sea_out) {
    float speed = weather_on ? 1.5f + 26.5f * std::pow(glm::clamp(wind_k, 0.0f, 1.0f), 1.3f) : 7.0f;
    // THE OPEN SEA'S WIND (2026-11-27, the owner's "the ocean has no waves and no elevation"): clear
    // weather's 2.8 m/s made a fully developed sea of Hs 0.17 m - a mirror. Over open water the wind
    // is never that still: a sea breeze of 6.5 m/s + 14 x the weather's wind amount (clear 7.9 m/s,
    // Hs 1.3 m; cloudy 10, 2.1 m; rain 12.8, 3.5 m; a storm 28) is the floor for the sea's waves.
    // Lakes, the river and the stream keep the local wind (calm water in still air).
    // VIBE_NO_SEA_BREEZE=1: the sea takes the local wind as before.
    static const bool no_breeze = [] { const char* e = std::getenv("VIBE_NO_SEA_BREEZE"); return e && *e && *e != '0'; }();
    float sea = (weather_on && !no_breeze) ? std::max(speed, 6.5f + 14.0f * glm::clamp(wind_k, 0.0f, 1.0f)) : speed;
    float ang = 0.337f + 0.32f * std::sin(t * 0.013f) + 0.12f * std::sin(t * 0.051f + 1.1f);
    static const char* test = std::getenv("VIBE_TEST_WATER_WIND");
    if (test) {
        float s = -1.0f, deg = -1000.0f;
        if (std::sscanf(test, "%f,%f", &s, &deg) >= 1 && s >= 0.0f) speed = sea = s;
        if (deg > -999.0f) ang = glm::radians(deg);
    }
    // The swell's scale: its five waves (A 13 / 8 / 7 / 2.5 / 2 m, crest 1.3 exp(sin p - 1) - 0.3) have a
    // standard deviation of 6.72 S m, i.e. Hs = 26.9 S. A fully developed sea has Hs = 0.21 U^2 / g
    // (Pierson-Moskowitz: 0.8 m at 6 m/s, 7 m at 18); the swell carries 60 % of it (was 90 %: nearly
    // all of the height went into waves 120 - 400 m long, which read as a flat sea), the FFT
    // cascades 80 % (sqrt(1 - 0.6^2); water_sim.cpp) - waves of 10 - 40 m that are seen to move.
    // (The reference's pow(U / 30, 2.2) gave 8.7 m at 18 m/s and 27 m at 30.)
    const float S = std::min(0.6f * 0.21f * sea * sea / 9.81f / 26.9f, 1.0f);
    if (sea_out) *sea_out = sea;
    return glm::vec4(std::cos(ang), std::sin(ang), speed, S);
}

// Weather types: cover, overcast, rain, thunder, wind, fog, snow on the
// ground, snowfall, darkness; and the hour the type implies (-1 = any).
const VulkanEngine::WxPreset VulkanEngine::kWxPresets[VulkanEngine::kWxPresetCount] = {
    { "Clear",         { 0.05f, 0.00f, 0.00f, 0.0f, 0.10f, 0.00f, 0.0f, 0.0f, 0.0f }, -1.0f },
    { "Sunny, cloudy", { 0.45f, 0.12f, 0.00f, 0.0f, 0.25f, 0.00f, 0.0f, 0.0f, 0.0f }, 13.0f },
    { "Sunny, windy",  { 0.22f, 0.05f, 0.00f, 0.0f, 1.00f, 0.00f, 0.0f, 0.0f, 0.0f }, 13.0f },
    { "Overcast",      { 0.90f, 0.85f, 0.00f, 0.0f, 0.35f, 0.15f, 0.0f, 0.0f, 0.0f }, -1.0f },
    { "Fog",           { 0.70f, 0.60f, 0.00f, 0.0f, 0.05f, 1.00f, 0.0f, 0.0f, 0.0f }, -1.0f },
    { "Rain",          { 0.97f, 1.00f, 0.65f, 0.0f, 0.45f, 0.35f, 0.0f, 0.0f, 0.0f }, -1.0f },
    { "Stormy",        { 1.00f, 1.00f, 1.00f, 1.0f, 1.00f, 0.50f, 0.0f, 0.0f, 0.0f }, -1.0f },
    { "Dry thunder",   { 0.97f, 1.00f, 0.00f, 1.0f, 0.80f, 0.10f, 0.0f, 0.0f, 0.0f }, -1.0f },
    { "Night thunder", { 1.00f, 1.00f, 0.80f, 1.0f, 0.90f, 0.30f, 0.0f, 0.0f, 0.0f }, 23.5f },
    { "Pitch black",   { 1.00f, 1.00f, 0.00f, 0.0f, 0.20f, 0.20f, 0.0f, 0.0f, 1.0f },  1.0f },
    { "Snow",          { 0.92f, 0.80f, 0.00f, 0.0f, 0.25f, 0.35f, 0.0f, 1.0f, 0.0f }, -1.0f },
};

const char* VulkanEngine::weather_name() const {
    const Wx& c = wx_cur_;
    if (c.snowfall > 0.3f) return "snow";
    if (c.dark > 0.5f) return "pitch black";
    if (c.thunder > 0.5f) return c.rain > 0.2f ? "thunderstorm" : "dry thunderstorm";
    if (c.rain > 0.2f) return "rain";
    if (c.fog > 0.5f) return "fog";
    if (c.overcast > 0.5f) return "overcast";
    if (c.wind > 0.7f) return "windy";
    if (c.cover > 0.3f) return "partly cloudy";
    return "clear";
}

void VulkanEngine::apply_environment(float dt) {
    const bool want = rt_.day_night_enabled || rt_.weather_enabled;
    // Back to the user's own values first; everything below derives from them.
    if (env_->applied) { env_swap_base(); env_->applied = false; }
    env_->water_wind = water_wind_now(rt_.weather_enabled, wx_cur_.wind, anim_time_, &env_->sea_wind);
    // (held with the animation clock by VIBE_TEST_ANIM_TIME: with the clock held and the drift running
    //  on, the water's patterns still moved and two test frames could not be compared)
    static const bool held_clock = std::getenv("VIBE_TEST_ANIM_TIME") != nullptr;
    if (!held_clock) env_->water_drift += glm::vec2(env_->water_wind) * glm::clamp(dt, 0.0f, 0.1f);
    if (!want) {
        env_->night = -1.0f; env_->rain = 0.0f; env_->snow = 0.0f; env_->snowfall = 0.0f; env_->wet = std::max(0.0f, env_->wet - dt / 60.0f);
        env_->golden = 1.0f; env_->sun_elev = -1000.0f; env_->sky_real = false;
        env_->lightning = 0.0f; env_->cloud_k = 1.0f; env_->cloud_direct = 1.0f; env_->cloud_shadow_k = 0.72f;
        tick_rain_audio(0.0f);
        return;
    }
    env_->base.sun_pitch_deg = rt_.sun_pitch_deg; env_->base.sun_yaw_deg = rt_.sun_yaw_deg;
    env_->base.sun_intensity = rt_.sun_intensity; env_->base.sun_color = rt_.sun_color;
    env_->base.sky_color = rt_.sky_color; env_->base.atmosphere_intensity = rt_.atmosphere_intensity;
    env_->base.cloud_coverage = rt_.cloud_coverage; env_->base.cloud_density = rt_.cloud_density;
    env_->base.cloud_speed = rt_.cloud_speed; env_->base.fog_density = rt_.fog_density;
    env_->base.grass_wind = rt_.grass_wind; env_->base.auto_golden_hour = rt_.auto_golden_hour;
    env_->applied = true;
    const EnvBase base = env_->base;
    dt = glm::clamp(dt, 0.0f, 0.1f);
    uint32_t& rng = spacejet_->rng_state;

    // ---------------- weather state ----------------
    // The weather is a handful of independent amounts (cloud cover, how grey
    // the deck is, rain, thunder, wind, fog, snow, darkness). A weather TYPE
    // (kWxPresets: the buttons in the menu, or a random pick when the weather
    // changes by itself) is a target for all of them; the current values
    // glide there, so any type turns into any other.
    if (rt_.weather_enabled) {
        float swing = glm::clamp(rt_.weather_transition_s, 1.0f, 300.0f);   // seconds for a picked type to move in (a setting)
        if (rt_.weather_preset >= 0) {
            const int pi = std::min(rt_.weather_preset, kWxPresetCount - 1);
            wx_tgt_ = kWxPresets[pi].wx;
            if (pi != wx_last_preset_) {
                wx_last_preset_ = pi;
                // Types that mean a time of day (night thunder, pitch black, sunny ...): turn the clock there.
                if (kWxPresets[pi].hour >= 0.0f) {
                    wx_clock_target_ = kWxPresets[pi].hour; rt_.day_night_enabled = true;
                    // (the clock gets there in the transition time too)
                    const float dist = std::abs(std::fmod(wx_clock_target_ - rt_.time_of_day + 36.0f, 24.0f) - 12.0f);
                    wx_clock_rate_ = std::max(dist / swing, 0.02f);
                }
                else wx_clock_target_ = -1.0f;
                log::infof("[weather] type: %s", kWxPresets[pi].name);
            }
        } else if (rt_.weather_manual >= 0.0f) {
            // The old single slider: 0 clear .. 1 thunderstorm.
            const float m = glm::clamp(rt_.weather_manual, 0.0f, 1.0f);
            wx_tgt_ = Wx{};
            wx_tgt_.cover = glm::smoothstep(0.0f, 0.72f, m);
            wx_tgt_.overcast = glm::smoothstep(0.25f, 0.70f, m);
            wx_tgt_.rain = glm::smoothstep(0.55f, 0.85f, m);
            wx_tgt_.thunder = glm::smoothstep(0.85f, 1.0f, m);
            wx_tgt_.wind = m;
            wx_last_preset_ = -1;
        } else {
            wx_last_preset_ = -1;
            weather_timer_ -= dt;
            if (mp_->host_tgt_valid) {
                wx_tgt_ = mp_->host_tgt;            // multiplayer client: the host picks the weather
                weather_timer_ = 5.0f;
            } else if (weather_timer_ <= 0.0f) {
                const float bad = glm::clamp(rt_.weather_bad_chance, 0.0f, 1.0f);
                const float r = frand(rng);
                static const int kBad[4] = { 5, 5, 6, 7 }, kMid[2] = { 3, 4 }, kFair[4] = { 0, 0, 1, 2 };
                int pick;
                if (r < bad)                              pick = kBad[int(frand(rng) * 3.99f)];     // rain, storm, dry thunder
                else if (r < bad + (1.0f - bad) * 0.30f)  pick = kMid[frand(rng) < 0.75f ? 0 : 1];  // overcast, fog
                else                                      pick = kFair[int(frand(rng) * 3.99f)];    // clear, some cloud, windy
                wx_tgt_ = kWxPresets[pick].wx;
                weather_timer_ = std::max(0.2f, rt_.weather_change_minutes) * 60.0f * frand_range(rng, 0.6f, 1.4f);
                log::infof("[weather] next: %s (for %.0f s)", kWxPresets[pick].name, weather_timer_);
            }
            // A front takes about a quarter of the change interval to move in.
            swing = std::max(0.2f, rt_.weather_change_minutes) * 60.0f * 0.25f;
        }
        // Rain needs a thick dark deck, whatever was asked for.
        const float wet_tgt = std::max(wx_tgt_.rain, wx_tgt_.snowfall * 0.8f);
        wx_tgt_.cover = std::max(wx_tgt_.cover, 0.97f * glm::smoothstep(0.0f, 0.2f, wet_tgt));
        wx_tgt_.overcast = std::max(wx_tgt_.overcast, glm::smoothstep(0.0f, 0.2f, wx_tgt_.rain));
        const float step = weather_init_ ? dt / swing : 1.0f;
        auto ease = [&](float& c, float t, float rate) { c += glm::clamp(t - c, -step * rate, step * rate); };
        // Clouds come first and leave last: rain only starts under a closed
        // deck, and the deck only opens once the rain has stopped.
        const float deck = glm::smoothstep(0.75f, 0.93f, wx_cur_.cover) * glm::smoothstep(0.6f, 0.9f, wx_cur_.overcast);
        const bool raining = wx_cur_.rain > 0.03f;
        ease(wx_cur_.cover, raining ? std::max(wx_tgt_.cover, 0.97f) : wx_tgt_.cover, 1.6f);
        ease(wx_cur_.overcast, raining ? std::max(wx_tgt_.overcast, 1.0f) : wx_tgt_.overcast, 1.6f);
        ease(wx_cur_.rain, wx_tgt_.rain * (weather_init_ ? deck : 1.0f), 1.3f);
        ease(wx_cur_.thunder, wx_tgt_.thunder * (weather_init_ ? glm::smoothstep(0.5f, 0.9f, wx_cur_.overcast) : 1.0f), 1.0f);
        ease(wx_cur_.wind, wx_tgt_.wind, 1.0f);
        ease(wx_cur_.fog, wx_tgt_.fog, 1.0f);
        // Snow on the ground is what the snowfall leaves: it builds up while it
        // snows and stays after - for good in winter, some minutes in spring
        // and autumn, briefly in summer.
        {
            const float melt_s = 30.0f * season_w_[1] + 150.0f * (season_w_[0] + season_w_[2]) + 1.0e6f * season_w_[3];
            if (wx_cur_.snowfall > 0.05f) wx_cur_.snow += dt / std::max(swing * 1.5f, 6.0f) * wx_cur_.snowfall;
            else                          wx_cur_.snow -= dt / std::max(melt_s, 1.0f);
            wx_cur_.snow = glm::clamp(wx_cur_.snow, 0.0f, 1.0f);
        }
        ease(wx_cur_.snowfall, wx_tgt_.snowfall, 1.0f);
        ease(wx_cur_.dark, wx_tgt_.dark, 1.0f);
        weather_init_ = true;
        // Turn the clock toward the hour the type asked for (2 game hours a second, the short way round).
        if (wx_clock_target_ >= 0.0f && rt_.day_night_enabled) {
            float d = std::fmod(wx_clock_target_ - rt_.time_of_day + 36.0f, 24.0f) - 12.0f;
            const float mv = glm::clamp(d, -wx_clock_rate_ * dt, wx_clock_rate_ * dt);
            rt_.time_of_day += mv;
            if (std::abs(d) < 0.03f) wx_clock_target_ = -1.0f;
        }
    } else {
        wx_cur_ = Wx{}; wx_tgt_ = Wx{}; weather_init_ = false; weather_timer_ = 0.0f; wx_last_preset_ = -1; wx_clock_target_ = -1.0f;
    }
    // ---------------- season ----------------
    // The look of the land, apart from the weather: spring (fresh green),
    // summer, autumn (the leaves turn) and winter (snow lies, the lake is
    // frozen, and what falls is snow). A change cross-fades over the
    // transition time.
    {
        const int want = glm::clamp(rt_.season, 0, 3);
        {
            static int logged = -1;
            static const char* kName[4] = { "spring", "summer", "autumn", "winter" };
            if (want != logged) { logged = want; log::infof("[weather] season: %s (changes take %.0f s)", kName[want], rt_.weather_transition_s); }
        }
        const float st = season_init_ ? dt / glm::clamp(rt_.weather_transition_s, 1.0f, 300.0f) : 1.0f;
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float tgt = i == want ? 1.0f : 0.0f;
            season_w_[i] += glm::clamp(tgt - season_w_[i], -st, st);
            sum += season_w_[i];
        }
        if (sum > 1e-4f) for (float& w : season_w_) w /= sum;
        season_init_ = true;
    }
    const float winter = season_w_[3];
    const float overcast = wx_cur_.overcast;
    // (in winter the rain comes down as snow)
    const float rain     = wx_cur_.rain * (1.0f - winter);
    const float storm    = wx_cur_.thunder;
    const float wind_k   = wx_cur_.wind;
    // (the single 0..1 "how bad" number some displays still use)
    weather_state_ = glm::clamp(std::max({ 0.5f * overcast, rain > 0.02f ? 0.6f + 0.25f * rain : 0.0f, storm > 0.3f ? 0.9f + 0.1f * storm : 0.0f }), 0.0f, 1.0f);
    env_->rain = rain;
    env_->snow = std::max(wx_cur_.snow, winter);
    env_->snowfall = std::max(wx_cur_.snowfall, wx_cur_.rain * winter);
    // Ground gets wet in ~20 s of rain and takes minutes to dry.
    env_->wet += (rain > 0.02f ? (rain - env_->wet) * dt / 20.0f * (0.5f + rain) : -env_->wet * dt / 150.0f);
    env_->wet = glm::clamp(env_->wet, 0.0f, 1.0f);
    // (VIBE_TEST_WET=<0..1>: the ground's wetness held - the puddles without waiting for the rain)
    static const char* test_wet = std::getenv("VIBE_TEST_WET");
    if (test_wet) env_->wet = glm::clamp(float(std::atof(test_wet)), 0.0f, 1.0f);

    // ---------------- clock -> sun / moon ----------------
    float sun_k = 1.0f;       // direct sunlight factor (horizon dimming, night)
    float sky_k = 1.0f;       // sky brightness factor on top of the atmosphere LUT
    float cloud_direct = 1.0f; // direct light on the clouds
    float amb_k = 1.0f;        // ambient (sky fill) factor: < 1 on nights with little or no moon
    glm::vec3 light_col = base.sun_color;
    float pitch = base.sun_pitch_deg, yaw = base.sun_yaw_deg;
    bool golden = base.auto_golden_hour;
    env_->night = -1.0f;
    float golden_k = 1.0f;         // how much of the warm low-sun tint applies
    bool  sky_real = false;        // the sky is lit by the real sun (below the horizon) while the land light is the glow
    bool  sky_model_night = false; // the night branch ran this frame (it steers env_->sky_ease)
    float sky_e = 0.0f, sky_yaw = 0.0f;
    env_->sun_elev = -1000.0f;
    if (rt_.day_night_enabled) {
        // (VIBE_SHOT_WAIT_FULL: a test copy's clock stands while it waits for the full terrain
        //  shader - the wait is 30 s .. minutes, at a 60 min day that moved the sun by up to an hour
        //  and two runs with the same settings came out lit differently)
        static const bool shot_wait = std::getenv("VIBE_SHOT_WAIT_FULL") != nullptr;
        const bool waiting = shot_wait && terrain_rm_build_[0].valid();
        if (rt_.day_length_minutes > 0.01f && !waiting)
            rt_.time_of_day += dt * 24.0f / (rt_.day_length_minutes * 60.0f);
        rt_.time_of_day = std::fmod(std::fmod(rt_.time_of_day, 24.0f) + 24.0f, 24.0f);
        const float H = (rt_.time_of_day - 12.0f) / 24.0f * 6.2831853f;
        const float lat = glm::radians(47.0f), dec = glm::radians(16.0f);
        float up    = std::sin(lat) * std::sin(dec) + std::cos(lat) * std::cos(dec) * std::cos(H);
        float east  = -std::cos(dec) * std::sin(H);
        float north = std::cos(lat) * std::sin(dec) - std::sin(lat) * std::cos(dec) * std::cos(H);
        // The clock is home's. The sun stands where it stands over the planet (engine/globe.h):
        // elsewhere it is seen from that place's own east / up / north - so it is another time
        // of day there, and on the far side it is night.
        const glm::dmat3 gfp = globe::frame_at(globe::current(), double(player_.position.x), double(player_.position.z));
        auto from_here = [&](float& u, float& e_, float& n) {
            const glm::dvec3 v(double(e_), double(u), -double(n));          // (home's frame is the planet's)
            u = float(glm::dot(v, gfp[1])); e_ = float(glm::dot(v, gfp[0])); n = float(-glm::dot(v, gfp[2]));
        };
        from_here(up, east, north);
        const float e = glm::degrees(std::asin(glm::clamp(up, -1.0f, 1.0f)));       // sun elevation
        const float az = glm::degrees(std::atan2(east, -north));                     // 0 = due south at noon
        // The moon: a body of its own. Its phase (0 new .. 0.5 full .. 1 new)
        // is how far it trails the sun around the sky, so a full moon rises at
        // sunset and a new moon is up by day only; the lit fraction follows
        // from the same angle. The phase is a setting like the time of day,
        // and (optionally) advances through a cycle of 8 days.
        if (rt_.moon_phases && rt_.day_length_minutes > 0.01f)
            rt_.moon_phase += dt / (rt_.day_length_minutes * 60.0f) / 8.0f;
        rt_.moon_phase = rt_.moon_phase - std::floor(rt_.moon_phase);
        const float mph = 6.2831853f * rt_.moon_phase;
        const float Hm = H - mph;
        const float decm = dec * std::cos(mph);
        float upm    = std::sin(lat) * std::sin(decm) + std::cos(lat) * std::cos(decm) * std::cos(Hm);
        float eastm  = -std::cos(decm) * std::sin(Hm);
        float northm = std::cos(lat) * std::sin(decm) - std::sin(lat) * std::cos(decm) * std::cos(Hm);
        from_here(upm, eastm, northm);
        const float em  = glm::degrees(std::asin(glm::clamp(upm, -1.0f, 1.0f)));   // moon elevation
        const float azm = glm::degrees(std::atan2(eastm, -northm));
        const float illum = 0.5f - 0.5f * std::cos(mph);                           // lit fraction of the disc
        // Moonlight reaching the sky: a half moon gives about a tenth of a
        // full one; nothing while it is below the horizon.
        const float moon_sky = glm::clamp(rt_.moon_brightness, 0.0f, 1.0f) * std::pow(illum, 1.8f) *
                               glm::smoothstep(-1.0f, 10.0f, em);
        env_->moon = moon_sky;
        env_->sun_elev = e;
        {   // (the same yaw convention as the light's below: base yaw + azimuth)
            auto dir_of = [](float pitch_deg, float yaw_deg) {
                const float p_ = glm::radians(pitch_deg), y_ = glm::radians(yaw_deg);
                return glm::vec3(std::sin(y_) * std::cos(p_), std::sin(p_), std::cos(y_) * std::cos(p_));
            };
            env_->sun_dir = dir_of(e, base.sun_yaw_deg + az);
            env_->moon_dir = dir_of(em, base.sun_yaw_deg + azm);
        }
        env_->light_is_sun = e > 0.6f;
        if (e > 0.6f) {
            // Day and twilight: the real sun, also below the horizon - the
            // atmosphere LUT then renders dusk on its own. Direct light dies
            // at the horizon; the sky fades out toward the end of twilight.
            pitch = e; yaw = base.sun_yaw_deg + az;
            // The sun dims the way it does through a thickening atmosphere:
            // gently from mid afternoon, half at ~8 degrees, a quarter at 4,
            // gone at the horizon. (Was full strength down to 4 degrees and
            // then off within minutes: the golden hour arrived as a jump.)
            sun_k = std::min(1.0f, std::exp(-0.11f / std::max(std::sin(glm::radians(e)), 0.012f)) / 0.88f);
            // The LUT sky is physically ~100x darker at sunset than at noon;
            // the eye (and a camera) adapts, so lift it toward dusk.
            sky_k = glm::mix(0.35f, 1.0f, glm::smoothstep(0.6f, 3.0f, e)) *
                    glm::mix(4.0f, 1.0f, glm::smoothstep(-6.0f, 12.0f, e));
            cloud_direct = glm::smoothstep(-6.0f, 1.0f, e);     // cloud tops stay lit after sunset
            env_->night = 1.0f - glm::smoothstep(0.6f, 5.0f, e);
        } else {
            // Night. What lights the land, as outdoors: the last twilight glow,
            // then the moon if it is up (by its phase), and starlight. Clouds
            // take most of it away: an overcast, moonless night is near black.
            const float twi = glm::smoothstep(-9.0f, 0.6f, e);            // twilight left (1 at sunset)
            const float cloud_tr = glm::mix(1.0f, 0.10f, overcast);        // what a cloud deck lets through
            // A moon hidden behind the mountains (as seen from where the player
            // stands) lights the sky above them but hardly the land: the far-
            // mountain shadow map, which follows the light direction, says so.
            // Eased over a second so the fill light does not pop.
            // (the moon takes over as the light only once dusk is well under way:
            // until then the sky keeps its sunset)
            // What lights the sky - the sun under the horizon (dusk / dawn glow)
            // or the moon / night - are two different sky models, and going
            // from one to the other used to happen within a frame: the sky
            // dropped to near black and came back in another colour. Now the
            // sky is dimmed down first (4 s), the model changes while it is
            // dark, and it is brought up again (6 s).
            const bool want_moon = moon_sky > 0.02f && moon_sky >= 0.5f * 0.8f * twi && twi < 0.45f;
            if (!env_->sky_init) { env_->moon_light = want_moon; env_->sky_ease = 1.0f; }
            if (want_moon != env_->moon_light) {
                env_->sky_ease -= dt / 4.0f;
                if (env_->sky_ease <= 0.0f) { env_->sky_ease = 0.0f; env_->moon_light = want_moon; }
            } else {
                env_->sky_ease = std::min(1.0f, env_->sky_ease + dt / 6.0f);
            }
            sky_model_night = true;
            const bool moon_is_light = env_->moon_light;
            const float vis_now = moon_is_light ? sun_horizon_visibility(player_.eye_position()) : 1.0f;
            env_->moon_vis += (vis_now - env_->moon_vis) * glm::clamp(dt * 1.5f, 0.0f, 1.0f);
            const float moon_gnd = moon_sky * cloud_tr * glm::mix(0.10f, 1.0f, env_->moon_vis);
            { static int dbg = 0; if (std::getenv("VIBE_TEST_MOONLOG") && (++dbg % 120) == 0) log::infof("[moon] elev %.1f deg, in sky %.2f, is light %d, visible from player %.2f (now %.2f)", em, moon_sky, moon_is_light ? 1 : 0, env_->moon_vis, vis_now); }
            const float star_gnd = 0.10f * glm::clamp(rt_.star_brightness, 0.0f, 1.5f) *
                                   (1.0f - overcast) * (1.0f - overcast);
            // The directional light: the moon at its real height (so mountains
            // can hide it), or - while it is the stronger - the glow where the
            // sun went down.
            const float glow = 0.8f * twi;
            if (moon_is_light) {
                pitch = std::max(em, 3.0f);
                yaw = base.sun_yaw_deg + azm;
                sun_k = 0.045f * moon_sky * (1.0f + 1.6f * twi);
            } else {
                pitch = 16.0f;
                yaw = base.sun_yaw_deg + az;
                // (fades in below the horizon: no step where the sun's own light ends)
                sun_k = 0.045f * glow * glm::smoothstep(0.6f, -2.5f, e);
                // Dusk and dawn: the sky itself is still lit by the real sun,
                // under the horizon - the atmosphere renders the afterglow.
                if (twi > 0.0f) { sky_real = true; sky_e = e; sky_yaw = yaw; }
            }
            sun_k *= glm::mix(1.0f, 0.45f, overcast);                      // on top of the day-time cloud dimming
            // Sky dome and clouds: twilight, moonlit haze, airglow; a cloud deck
            // seen from below at night is almost black.
            // No constant floor: with no moon and no stars the sky - and with it
            // the haze in front of far mountains and every sky reflection - goes
            // dark instead of glowing at a third of full-moon level.
            const float night_sky = glm::clamp(moon_sky + 0.10f * glm::clamp(rt_.star_brightness, 0.0f, 1.5f) + 0.012f, 0.0f, 1.0f);
            sky_k = 0.045f * glm::mix(night_sky, 2.6f * glm::mix(0.5f, 1.0f, moon_sky), twi) *
                    glm::mix(1.0f, 0.05f, overcast * (1.0f - twi));
            // (with the real sun in the sky: the same curve as by day)
            if (sky_real) sky_k = glm::mix(0.35f, 1.0f, glm::smoothstep(0.6f, 3.0f, e)) *
                                  glm::mix(4.0f, 1.0f, glm::smoothstep(-6.0f, 12.0f, e));
            // Fill light on the land (1 = a clear full-moon night).
            const float night_amb = std::max(0.90f * moon_gnd + star_gnd, 0.003f);
            amb_k = glm::mix(night_amb, 1.0f, twi);
            light_col = glm::mix(glm::vec3(0.60f, 0.73f, 1.0f), base.sun_color, twi * twi);
            golden_k = twi;                    // the warm tint leaves with the twilight, not at the horizon
            env_->night = 1.0f;
            cloud_direct = 0.6f;
        }
        while (yaw > 180.0f) yaw -= 360.0f;
        while (yaw < -180.0f) yaw += 360.0f;
    }

    // A change of what lights the sky (sun under the horizon <-> moon / night
    // glow) swaps the whole sky model: fade the new one in over a few seconds.
    env_->sky_real = sky_real;
    env_->sky_init = true;
    if (!sky_model_night) env_->sky_ease = std::min(1.0f, env_->sky_ease + dt / 6.0f);     // by day: nothing to hide
    sky_k *= glm::mix(0.15f, 1.0f, glm::smoothstep(0.0f, 1.0f, env_->sky_ease));
    env_->sky_pitch = std::max(sky_e, -14.0f);
    env_->sky_yaw = sky_yaw;
    env_->golden = golden_k;

    // ---------------- lightning ----------------
    // VIBE_TEST_LIGHTNING=1: a strike every ~0.7 s, 500 m straight ahead (captures).
    static const bool test_lightning = std::getenv("VIBE_TEST_LIGHTNING") != nullptr;
    if (test_lightning) lightning_timer_ = std::min(lightning_timer_, 0.7f);
    if (storm > 0.05f || test_lightning) {
        lightning_timer_ -= dt;
        if (lightning_timer_ <= 0.0f) {
            lightning_timer_ = frand_range(rng, 4.0f, 14.0f) / (0.35f + storm);
            lightning_t_ = 0.0f;
            lightning_seed_ = frand(rng);
            const float a = frand(rng) * 6.2831853f, dist = frand_range(rng, 250.0f, 1400.0f);
            lightning_pos_ = player_.eye_position() + glm::vec3(std::cos(a), 0.0f, std::sin(a)) * dist;
            if (test_lightning) {
                glm::vec3 fw = player_.forward(); fw.y = 0.0f;
                lightning_pos_ = player_.eye_position() + glm::normalize(fw + glm::vec3(1e-4f)) * 500.0f;
            }
            lightning_pos_.y = std::max(sample_terrain_height(lightning_pos_.x, lightning_pos_.z), rt_.water_level);
            if (audio_) {
                const glm::vec3 p = player_.eye_position() + glm::vec3(std::cos(a), 0.4f, std::sin(a)) * 60.0f;
                // Real thunder: a clap or crackle for a near strike, the long roll for a far one;
                // it arrives late and dull with distance.
                const ClipID clip = dist > 700.0f ? ClipID::Thunder2
                                                  : (frand(rng) < 0.5f ? ClipID::Thunder1 : ClipID::Thunder3);
                audio_->play_effect(clip, p, glm::clamp(1.5f - dist / 1300.0f, 0.45f, 1.4f),
                                    frand_range(rng, 0.88f, 1.06f), dist / 343.0f,
                                    glm::mix(12000.0f, 900.0f, glm::clamp(dist / 1400.0f, 0.0f, 1.0f)));
            }
        }
    }
    env_->lightning = 0.0f;
    if (lightning_t_ >= 0.0f) {
        lightning_t_ += dt;
        // Two or three flickers within a third of a second.
        const float t = lightning_t_;
        float k = std::exp(-t * 9.0f);
        k += 0.8f * std::exp(-std::abs(t - 0.11f - 0.05f * lightning_seed_) * 40.0f);
        k += 0.5f * std::exp(-std::abs(t - 0.24f) * 50.0f) * (lightning_seed_ > 0.4f ? 1.0f : 0.0f);
        env_->lightning = glm::clamp(k, 0.0f, 1.6f) * (t < 0.6f ? 1.0f : 0.0f);
        if (t > 0.6f) lightning_t_ = -1.0f;
    }

    // ---------------- write the driven settings ----------------
    // Clouds hide the sun: direct light drops to a tenth under full overcast
    // and loses its colour; the sky dome's light turns grey and dimmer.
    // ("pitch black" takes nearly all light away)
    const float dark_k = 1.0f - 0.985f * wx_cur_.dark;
    sun_k *= dark_k; sky_k *= dark_k; amb_k *= dark_k;
    const float cloud_sun = glm::mix(1.0f, 0.22f, overcast * overcast);
    const float direct = std::max(sun_k * cloud_sun, 0.0003f);
    const float lum = glm::dot(light_col, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    rt_.sun_pitch_deg = pitch;
    rt_.sun_yaw_deg = yaw;
    rt_.sun_color = glm::mix(light_col, glm::vec3(lum), overcast * 0.7f);
    rt_.sun_intensity = base.sun_intensity * direct;
    rt_.auto_golden_hour = golden;
    // The sky's brightness is LUT x sun intensity x atmosphere intensity:
    // undo the direct-light dimming there and apply the sky's own factor.
    // (a rain deck is darker still)
    const float sky_dim = glm::mix(1.0f, 0.62f, overcast) * glm::mix(1.0f, 0.72f, rain) * (1.0f + 5.0f * env_->lightning);
    // The cloud pass takes its light level directly (not times the sun intensity).
    env_->cloud_k = sky_k * sky_dim;
    // Under a closed deck the light is uniformly dim: no shadow pattern.
    env_->cloud_shadow_k = 0.72f * (1.0f - overcast);
    env_->cloud_direct = cloud_direct * glm::mix(1.0f, 0.45f, overcast);

    rt_.atmosphere_intensity = base.atmosphere_intensity * sky_k * sky_dim / direct;
    // VIBE_LOG_ENV=1: what the light is made of, every 60 frames (to find what differs between runs).
    { static int n = 0; if (std::getenv("VIBE_LOG_ENV") && (n++ % 60) == 0)
        log::infof("[env] tod %.3f elev %.2f pitch %.2f yaw %.2f sun_k %.3f sky_k %.3f overcast %.3f rain %.3f dark %.3f direct %.3f sun_I %.3f atm %.3f cloud_sh %.3f night %.2f",
                   rt_.time_of_day, env_->sun_elev, pitch, yaw, sun_k, sky_k, overcast, rain, wx_cur_.dark, direct, rt_.sun_intensity, rt_.atmosphere_intensity, env_->cloud_shadow_k, env_->night); }
    const float sky_lum = glm::dot(base.sky_color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    rt_.sky_color = glm::mix(base.sky_color, glm::vec3(0.92f, 0.96f, 1.04f) * sky_lum, overcast * 0.85f) * amb_k;
    env_->amb_k = amb_k;
    // Lightning: for the length of the flash the whole landscape is lit
    // blue-white from the sky (strongest at night, a flicker by day).
    if (env_->lightning > 0.01f) {
        const float fl = std::min(env_->lightning, 1.4f);
        rt_.sky_color += glm::vec3(0.78f, 0.86f, 1.0f) * (2.4f * fl * std::max(sky_lum, 0.35f));
        env_->amb_k = std::max(env_->amb_k, std::min(1.0f, 0.9f * fl));
    }
    rt_.cloud_coverage = glm::mix(std::min(base.cloud_coverage, 0.30f), 0.97f, wx_cur_.cover);
    if (!rt_.weather_enabled) rt_.cloud_coverage = base.cloud_coverage;
    // Rain clouds are thicker than a dry overcast.
    rt_.cloud_density = base.cloud_density * glm::mix(1.0f, 2.4f, overcast) * (1.0f + 0.6f * rain);
    rt_.cloud_speed = base.cloud_speed * glm::mix(1.0f, 3.2f, wind_k);
    rt_.fog_density = base.fog_density * (1.0f + 0.6f * overcast + 2.0f * rain + 6.0f * wx_cur_.fog + 1.5f * wx_cur_.snowfall);
    rt_.grass_wind = base.grass_wind * glm::mix(1.0f, 2.6f, wind_k);
    env_->wind_scale = glm::mix(1.0f, 2.4f, wind_k);

    tick_rain_audio(rain);
    update_footprints(dt);
}

// Footprints in the snow. Every stride of a player walking on snowy ground
// leaves a print (left, right, ...); the terrain shader presses them into
// the snow (terrain_raymarch.frag). Falling snow fills them in again.
void VulkanEngine::add_footprint(glm::vec3 pos, float yaw, bool left) {
    const glm::vec2 right(std::cos(yaw), std::sin(yaw));
    const glm::vec2 p = glm::vec2(pos.x, pos.z) + right * (left ? -0.11f : 0.11f);
    footprints_[foot_next_] = glm::vec4(p, yaw, 1.0f);
    foot_next_ = (foot_next_ + 1) % 32;
}

void VulkanEngine::update_footprints(float dt) {
    // Prints are left in snow anywhere, and (any season) on the road, where sand lies in
    // drifts: the shader presses a print in only where there is snow or sand to press.
    static bool had_snow = false;
    const bool snow_now = env_->snow >= 0.05f;
    if (had_snow != snow_now) {                        // thawed / first snow: the old prints are gone
        for (auto& f : footprints_) f.w = 0.0f;
        foot_last_ = glm::vec3(1e9f);
    }
    had_snow = snow_now;
    if (env_->snowfall > 0.05f)
        for (auto& f : footprints_) f.w = std::max(0.0f, f.w - dt * env_->snowfall / 240.0f);
    const bool snowy = env_->snow >= 0.3f;
    auto prints_at = [&](const glm::vec3& p) {
        return snowy || (!castle_road_.empty() && castle_road_.distance(p.x, p.z) < castle_road_.half_width + 0.9f);
    };
    // VIBE_TEST_FOOTPRINTS=1: a trail of prints straight ahead of the player (captures).
    if (static bool done = false; !done && std::getenv("VIBE_TEST_FOOTPRINTS")) {
        done = true;
        const glm::vec3 f(std::sin(player_.yaw), 0.0f, -std::cos(player_.yaw));
        for (int i = 0; i < 12; ++i) add_footprint(player_.position + f * (1.2f + 0.7f * float(i)), player_.yaw, (i & 1) != 0);
    }
    auto step = [&](glm::vec3 pos, glm::vec3 vel, float yaw, bool grounded, glm::vec3& last, bool& left) {
        if (!grounded) return;
        if (!prints_at(pos)) { last = pos; return; }
        const glm::vec2 d(pos.x - last.x, pos.z - last.z);
        const float speed = glm::length(glm::vec2(vel.x, vel.z));
        const float stride = glm::clamp(0.45f + 0.09f * speed, 0.55f, 1.25f);
        if (glm::dot(d, d) < stride * stride) return;
        if (glm::dot(d, d) > 25.0f) { last = pos; return; }      // teleport / first time: no print
        // Heading = the way this stride went (from the last print to here),
        // not where the head looks and not the momentary velocity.
        (void)yaw;
        const float heading = std::atan2(d.x, -d.y);
        add_footprint(pos, heading, left);
        left = !left;
        last = pos;
    };
    step(player_.position, player_.velocity, player_.yaw, player_.on_ground && !player_swimming_, foot_last_, foot_left_);
    for (auto& a : mp_->avatars) {
        if (!a.id) continue;
        step(glm::vec3(a.st.px, a.st.py, a.st.pz), glm::vec3(a.st.vx, a.st.vy, a.st.vz), a.st.yaw,
             (a.st.flags & 1) && !(a.st.flags & 2), a.foot_last, a.foot_left);
    }
}

// Rain bed: an 8 s recording with a Hann envelope baked in, played by two
// voices half a clip apart - their sum is constant, so it loops without a
// seam and without a loop API.
void VulkanEngine::tick_rain_audio(float rain) {
    if (!audio_) return;
    const bool under = rt_.water_enabled && current_frame_view_.eye_pos.y < water_surface_y(current_frame_view_.eye_pos.x, current_frame_view_.eye_pos.z);
    const float vol = under ? 0.0f : glm::clamp(rain, 0.0f, 1.0f) * 0.9f;
    const glm::vec3 pos = player_.eye_position() + glm::vec3(0.0f, 1.0f, 0.0f);
    if (vol < 0.01f) {
        for (auto& h : rain_voice_) if (h) { audio_->stop_voice(h, 0.5f); h = 0; }
        return;
    }
    for (int i = 0; i < 2; ++i) {
        if (!rain_voice_[i] || !audio_->voice_playing(rain_voice_[i])) {
            const bool first = (rain_voice_[0] == 0 && rain_voice_[1] == 0);
            rain_voice_[i] = audio_->play_tracked(ClipID::Rain, pos, vol, 1.0f,
                                                  (first && i == 1) || (i == 1 && rain_voice_[1] == 0) ? 4.0f : 0.0f);
        } else {
            audio_->set_voice(rain_voice_[i], pos, vol);
        }
    }
}

}  // namespace vibe
