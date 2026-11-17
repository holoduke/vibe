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
// The user's own values for the driven settings are kept in env_base_ and
// swapped back for saving, so the cfg always holds what the user chose.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace qlike {

// rt_ <-> env_base_ for every driven field (its own inverse).
void VulkanEngine::env_swap_base() {
    std::swap(rt_.sun_pitch_deg, env_base_.sun_pitch_deg);
    std::swap(rt_.sun_yaw_deg, env_base_.sun_yaw_deg);
    std::swap(rt_.sun_intensity, env_base_.sun_intensity);
    std::swap(rt_.sun_color, env_base_.sun_color);
    std::swap(rt_.sky_color, env_base_.sky_color);
    std::swap(rt_.atmosphere_intensity, env_base_.atmosphere_intensity);
    std::swap(rt_.cloud_coverage, env_base_.cloud_coverage);
    std::swap(rt_.cloud_density, env_base_.cloud_density);
    std::swap(rt_.cloud_speed, env_base_.cloud_speed);
    std::swap(rt_.fog_density, env_base_.fog_density);
    std::swap(rt_.grass_wind, env_base_.grass_wind);
    std::swap(rt_.auto_golden_hour, env_base_.auto_golden_hour);
}

// CPU copy of scene_ubo.glsl cloud_shadow() (keep in sync): how much direct
// sun the clouds let through at a world point.
float VulkanEngine::cloud_shadow_at(glm::vec3 wp) const {
    const float k = cloud_shadow_k_now_;
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
    const float size = std::max(cloud_a_.y, 0.2f);
    const float alt = cloud_a_.z + 0.30f * cloud_a_.w;
    const glm::vec2 p = glm::vec2(wp.x, wp.z) + glm::vec2(L.x, L.z) / std::max(L.y, 0.12f) * std::max(alt - wp.y, 0.0f);
    const glm::vec2 q = (p + glm::vec2(cloud_b_)) / (6500.0f * size);
    const float wn = hvalue(q) * 0.7f + hvalue(q * 2.17f + 13.0f) * 0.3f;
    const float cov = glm::clamp(cloud_a_.x, 0.0f, 1.0f);
    const float map_cov = glm::smoothstep(0.2f, 0.8f, glm::clamp(glm::smoothstep(0.2f, 0.8f, wn) + (cov - 0.5f) * 1.5f, 0.0f, 1.0f));
    const glm::vec2 qs = (p + glm::vec2(cloud_b_) * 1.6f) / (1900.0f * size);
    const float lump = hvalue(qs) * 0.6f + hvalue(qs * 2.3f + 5.0f) * 0.4f;
    const float opac = glm::smoothstep(0.30f, 0.78f, map_cov * (0.62f + 0.75f * lump)) * glm::clamp(cloud_b_.z, 0.3f, 1.5f);
    return 1.0f - k * glm::clamp(opac, 0.0f, 1.0f);
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
    { "Winter white",  { 0.88f, 0.70f, 0.00f, 0.0f, 0.25f, 0.35f, 1.0f, 1.0f, 0.0f }, -1.0f },
};

const char* VulkanEngine::weather_name() const {
    const Wx& c = wx_cur_;
    if (c.snow > 0.5f || c.snowfall > 0.5f) return "winter";
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
    if (env_applied_) { env_swap_base(); env_applied_ = false; }
    if (!want) {
        env_night_ = -1.0f; env_rain_ = 0.0f; env_snow_ = 0.0f; env_snowfall_ = 0.0f; env_wet_ = std::max(0.0f, env_wet_ - dt / 60.0f);
        env_golden_ = 1.0f; env_sun_elev_ = -1000.0f; env_sky_real_ = false;
        env_lightning_ = 0.0f; env_cloud_k_ = 1.0f; env_cloud_direct_ = 1.0f; env_cloud_shadow_k_ = 0.72f;
        tick_rain_audio(0.0f);
        return;
    }
    env_base_.sun_pitch_deg = rt_.sun_pitch_deg; env_base_.sun_yaw_deg = rt_.sun_yaw_deg;
    env_base_.sun_intensity = rt_.sun_intensity; env_base_.sun_color = rt_.sun_color;
    env_base_.sky_color = rt_.sky_color; env_base_.atmosphere_intensity = rt_.atmosphere_intensity;
    env_base_.cloud_coverage = rt_.cloud_coverage; env_base_.cloud_density = rt_.cloud_density;
    env_base_.cloud_speed = rt_.cloud_speed; env_base_.fog_density = rt_.fog_density;
    env_base_.grass_wind = rt_.grass_wind; env_base_.auto_golden_hour = rt_.auto_golden_hour;
    env_applied_ = true;
    const EnvBase base = env_base_;
    dt = glm::clamp(dt, 0.0f, 0.1f);
    uint32_t& rng = spacejet_rng_state_;

    // ---------------- weather state ----------------
    // The weather is a handful of independent amounts (cloud cover, how grey
    // the deck is, rain, thunder, wind, fog, snow, darkness). A weather TYPE
    // (kWxPresets: the buttons in the menu, or a random pick when the weather
    // changes by itself) is a target for all of them; the current values
    // glide there, so any type turns into any other.
    if (rt_.weather_enabled) {
        float swing = 22.0f;                      // seconds for a picked type to move in
        if (rt_.weather_preset >= 0) {
            const int pi = std::min(rt_.weather_preset, kWxPresetCount - 1);
            wx_tgt_ = kWxPresets[pi].wx;
            if (pi != wx_last_preset_) {
                wx_last_preset_ = pi;
                // Types that mean a time of day (night thunder, pitch black, sunny ...): turn the clock there.
                if (kWxPresets[pi].hour >= 0.0f) { wx_clock_target_ = kWxPresets[pi].hour; rt_.day_night_enabled = true; }
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
            if (weather_timer_ <= 0.0f) {
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
        ease(wx_cur_.snow, wx_tgt_.snow, 0.6f);          // the ground whitens (and thaws) slowly
        ease(wx_cur_.snowfall, wx_tgt_.snowfall, 1.0f);
        ease(wx_cur_.dark, wx_tgt_.dark, 1.0f);
        weather_init_ = true;
        // Turn the clock toward the hour the type asked for (2 game hours a second, the short way round).
        if (wx_clock_target_ >= 0.0f && rt_.day_night_enabled) {
            float d = std::fmod(wx_clock_target_ - rt_.time_of_day + 36.0f, 24.0f) - 12.0f;
            const float mv = glm::clamp(d, -2.0f * dt, 2.0f * dt);
            rt_.time_of_day += mv;
            if (std::abs(d) < 0.03f) wx_clock_target_ = -1.0f;
        }
    } else {
        wx_cur_ = Wx{}; wx_tgt_ = Wx{}; weather_init_ = false; weather_timer_ = 0.0f; wx_last_preset_ = -1; wx_clock_target_ = -1.0f;
    }
    const float overcast = wx_cur_.overcast;
    const float rain     = wx_cur_.rain;
    const float storm    = wx_cur_.thunder;
    const float wind_k   = wx_cur_.wind;
    // (the single 0..1 "how bad" number some displays still use)
    weather_state_ = glm::clamp(std::max({ 0.5f * overcast, rain > 0.02f ? 0.6f + 0.25f * rain : 0.0f, storm > 0.3f ? 0.9f + 0.1f * storm : 0.0f }), 0.0f, 1.0f);
    env_rain_ = rain;
    env_snow_ = wx_cur_.snow;
    env_snowfall_ = wx_cur_.snowfall;
    // Ground gets wet in ~20 s of rain and takes minutes to dry.
    env_wet_ += (rain > 0.02f ? (rain - env_wet_) * dt / 20.0f * (0.5f + rain) : -env_wet_ * dt / 150.0f);
    env_wet_ = glm::clamp(env_wet_, 0.0f, 1.0f);

    // ---------------- clock -> sun / moon ----------------
    float sun_k = 1.0f;       // direct sunlight factor (horizon dimming, night)
    float sky_k = 1.0f;       // sky brightness factor on top of the atmosphere LUT
    float cloud_direct = 1.0f; // direct light on the clouds
    float amb_k = 1.0f;        // ambient (sky fill) factor: < 1 on nights with little or no moon
    glm::vec3 light_col = base.sun_color;
    float pitch = base.sun_pitch_deg, yaw = base.sun_yaw_deg;
    bool golden = base.auto_golden_hour;
    env_night_ = -1.0f;
    float golden_k = 1.0f;         // how much of the warm low-sun tint applies
    bool  sky_real = false;        // the sky is lit by the real sun (below the horizon) while the land light is the glow
    float sky_e = 0.0f, sky_yaw = 0.0f;
    env_sun_elev_ = -1000.0f;
    if (rt_.day_night_enabled) {
        if (rt_.day_length_minutes > 0.01f)
            rt_.time_of_day += dt * 24.0f / (rt_.day_length_minutes * 60.0f);
        rt_.time_of_day = std::fmod(std::fmod(rt_.time_of_day, 24.0f) + 24.0f, 24.0f);
        const float H = (rt_.time_of_day - 12.0f) / 24.0f * 6.2831853f;
        const float lat = glm::radians(47.0f), dec = glm::radians(16.0f);
        const float up    = std::sin(lat) * std::sin(dec) + std::cos(lat) * std::cos(dec) * std::cos(H);
        const float east  = -std::cos(dec) * std::sin(H);
        const float north = std::cos(lat) * std::sin(dec) - std::sin(lat) * std::cos(dec) * std::cos(H);
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
        const float upm    = std::sin(lat) * std::sin(decm) + std::cos(lat) * std::cos(decm) * std::cos(Hm);
        const float eastm  = -std::cos(decm) * std::sin(Hm);
        const float northm = std::cos(lat) * std::sin(decm) - std::sin(lat) * std::cos(decm) * std::cos(Hm);
        const float em  = glm::degrees(std::asin(glm::clamp(upm, -1.0f, 1.0f)));   // moon elevation
        const float azm = glm::degrees(std::atan2(eastm, -northm));
        const float illum = 0.5f - 0.5f * std::cos(mph);                           // lit fraction of the disc
        // Moonlight reaching the sky: a half moon gives about a tenth of a
        // full one; nothing while it is below the horizon.
        const float moon_sky = glm::clamp(rt_.moon_brightness, 0.0f, 1.0f) * std::pow(illum, 1.8f) *
                               glm::smoothstep(-1.0f, 10.0f, em);
        env_moon_ = moon_sky;
        env_sun_elev_ = e;
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
            env_night_ = 1.0f - glm::smoothstep(0.6f, 5.0f, e);
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
            const bool moon_is_light = moon_sky > 0.02f && moon_sky >= 0.5f * 0.8f * twi && twi < 0.45f;
            const float vis_now = moon_is_light ? sun_horizon_visibility(player_.eye_position()) : 1.0f;
            env_moon_vis_ += (vis_now - env_moon_vis_) * glm::clamp(dt * 1.5f, 0.0f, 1.0f);
            const float moon_gnd = moon_sky * cloud_tr * glm::mix(0.10f, 1.0f, env_moon_vis_);
            { static int dbg = 0; if (std::getenv("QLIKE_TEST_MOONLOG") && (++dbg % 120) == 0) log::infof("[moon] elev %.1f deg, in sky %.2f, is light %d, visible from player %.2f (now %.2f)", em, moon_sky, moon_is_light ? 1 : 0, env_moon_vis_, vis_now); }
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
            env_night_ = 1.0f;
            cloud_direct = 0.6f;
        }
        while (yaw > 180.0f) yaw -= 360.0f;
        while (yaw < -180.0f) yaw += 360.0f;
    }

    // A change of what lights the sky (sun under the horizon <-> moon / night
    // glow) swaps the whole sky model: fade the new one in over a few seconds.
    if (sky_real != env_sky_real_) { env_sky_real_ = sky_real; env_sky_ease_ = env_sky_init_ ? 0.0f : 1.0f; }
    env_sky_init_ = true;
    env_sky_ease_ = std::min(1.0f, env_sky_ease_ + dt / 5.0f);
    sky_k *= glm::mix(0.15f, 1.0f, glm::smoothstep(0.0f, 1.0f, env_sky_ease_));
    env_sky_pitch_ = std::max(sky_e, -14.0f);
    env_sky_yaw_ = sky_yaw;
    env_golden_ = golden_k;

    // ---------------- lightning ----------------
    // QLIKE_TEST_LIGHTNING=1: a strike every ~0.7 s, 500 m straight ahead (captures).
    static const bool test_lightning = std::getenv("QLIKE_TEST_LIGHTNING") != nullptr;
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
                const int clip = int(ClipID::ExplFar1) + int(frand(rng) * 4.99f);
                audio_->play_effect(ClipID(clip), p, glm::clamp(1.3f - dist / 1600.0f, 0.35f, 1.1f),
                                    frand_range(rng, 0.45f, 0.62f), dist / 343.0f,
                                    glm::mix(1400.0f, 420.0f, dist / 1400.0f));
            }
        }
    }
    env_lightning_ = 0.0f;
    if (lightning_t_ >= 0.0f) {
        lightning_t_ += dt;
        // Two or three flickers within a third of a second.
        const float t = lightning_t_;
        float k = std::exp(-t * 9.0f);
        k += 0.8f * std::exp(-std::abs(t - 0.11f - 0.05f * lightning_seed_) * 40.0f);
        k += 0.5f * std::exp(-std::abs(t - 0.24f) * 50.0f) * (lightning_seed_ > 0.4f ? 1.0f : 0.0f);
        env_lightning_ = glm::clamp(k, 0.0f, 1.6f) * (t < 0.6f ? 1.0f : 0.0f);
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
    const float sky_dim = glm::mix(1.0f, 0.62f, overcast) * glm::mix(1.0f, 0.72f, rain) * (1.0f + 5.0f * env_lightning_);
    // The cloud pass takes its light level directly (not times the sun intensity).
    env_cloud_k_ = sky_k * sky_dim;
    // Under a closed deck the light is uniformly dim: no shadow pattern.
    env_cloud_shadow_k_ = 0.72f * (1.0f - overcast);
    env_cloud_direct_ = cloud_direct * glm::mix(1.0f, 0.45f, overcast);

    rt_.atmosphere_intensity = base.atmosphere_intensity * sky_k * sky_dim / direct;
    const float sky_lum = glm::dot(base.sky_color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    rt_.sky_color = glm::mix(base.sky_color, glm::vec3(0.92f, 0.96f, 1.04f) * sky_lum, overcast * 0.85f) * amb_k;
    env_amb_k_ = amb_k;
    // Lightning: for the length of the flash the whole landscape is lit
    // blue-white from the sky (strongest at night, a flicker by day).
    if (env_lightning_ > 0.01f) {
        const float fl = std::min(env_lightning_, 1.4f);
        rt_.sky_color += glm::vec3(0.78f, 0.86f, 1.0f) * (2.4f * fl * std::max(sky_lum, 0.35f));
        env_amb_k_ = std::max(env_amb_k_, std::min(1.0f, 0.9f * fl));
    }
    rt_.cloud_coverage = glm::mix(std::min(base.cloud_coverage, 0.30f), 0.97f, wx_cur_.cover);
    if (!rt_.weather_enabled) rt_.cloud_coverage = base.cloud_coverage;
    // Rain clouds are thicker than a dry overcast.
    rt_.cloud_density = base.cloud_density * glm::mix(1.0f, 2.4f, overcast) * (1.0f + 0.6f * rain);
    rt_.cloud_speed = base.cloud_speed * glm::mix(1.0f, 3.2f, wind_k);
    rt_.fog_density = base.fog_density * (1.0f + 0.6f * overcast + 2.0f * rain + 6.0f * wx_cur_.fog + 1.5f * wx_cur_.snowfall);
    rt_.grass_wind = base.grass_wind * glm::mix(1.0f, 2.6f, wind_k);
    env_wind_scale_ = glm::mix(1.0f, 2.4f, wind_k);

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
    if (env_snow_ < 0.05f) {                           // thawed: gone
        for (auto& f : footprints_) f.w = 0.0f;
        foot_last_ = glm::vec3(1e9f);
        return;
    }
    if (env_snowfall_ > 0.05f)
        for (auto& f : footprints_) f.w = std::max(0.0f, f.w - dt * env_snowfall_ / 240.0f);
    if (env_snow_ < 0.3f) return;
    // QLIKE_TEST_FOOTPRINTS=1: a trail of prints straight ahead of the player (captures).
    if (static bool done = false; !done && std::getenv("QLIKE_TEST_FOOTPRINTS")) {
        done = true;
        const glm::vec3 f(std::sin(player_.yaw), 0.0f, -std::cos(player_.yaw));
        for (int i = 0; i < 12; ++i) add_footprint(player_.position + f * (1.2f + 0.7f * float(i)), player_.yaw, (i & 1) != 0);
    }
    auto step = [&](glm::vec3 pos, glm::vec3 vel, float yaw, bool grounded, glm::vec3& last, bool& left) {
        if (!grounded) return;
        const glm::vec2 d(pos.x - last.x, pos.z - last.z);
        const float speed = glm::length(glm::vec2(vel.x, vel.z));
        const float stride = glm::clamp(0.45f + 0.09f * speed, 0.55f, 1.25f);
        if (glm::dot(d, d) < stride * stride) return;
        if (glm::dot(d, d) > 25.0f) { last = pos; return; }      // teleport / first time: no print
        // Heading = the way the feet travel, not where the head looks.
        const float heading = speed > 0.3f ? std::atan2(vel.x, -vel.z) : yaw;
        add_footprint(pos, heading, left);
        left = !left;
        last = pos;
    };
    step(player_.position, player_.velocity, player_.yaw, player_.on_ground && !player_swimming_, foot_last_, foot_left_);
    for (auto& a : mp_avatars_) {
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
    const bool under = rt_.water_enabled && current_frame_view_.eye_pos.y < rt_.water_level;
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

}  // namespace qlike
