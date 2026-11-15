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

const char* VulkanEngine::weather_name() const {
    const float w = weather_state_;
    if (w < 0.18f) return "clear";
    if (w < 0.40f) return "fair";
    if (w < 0.58f) return "overcast";
    if (w < 0.72f) return "light rain";
    if (w < 0.87f) return "rain";
    return "thunderstorm";
}

void VulkanEngine::apply_environment(float dt) {
    const bool want = rt_.day_night_enabled || rt_.weather_enabled;
    // Back to the user's own values first; everything below derives from them.
    if (env_applied_) { env_swap_base(); env_applied_ = false; }
    if (!want) {
        env_night_ = -1.0f; env_rain_ = 0.0f; env_wet_ = std::max(0.0f, env_wet_ - dt / 60.0f);
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
    float w = 0.0f;
    if (rt_.weather_enabled) {
        if (rt_.weather_manual >= 0.0f) {
            weather_target_ = glm::clamp(rt_.weather_manual, 0.0f, 1.0f);
        } else {
            weather_timer_ -= dt;
            if (weather_timer_ <= 0.0f) {
                const float bad = glm::clamp(rt_.weather_bad_chance, 0.0f, 1.0f);
                const float r = frand(rng);
                if (r < bad)                              weather_target_ = frand_range(rng, 0.60f, 1.0f);   // rain .. storm
                else if (r < bad + (1.0f - bad) * 0.35f)  weather_target_ = frand_range(rng, 0.30f, 0.52f);  // cloudy
                else                                      weather_target_ = frand_range(rng, 0.0f, 0.22f);   // clear
                weather_timer_ = std::max(0.2f, rt_.weather_change_minutes) * 60.0f * frand_range(rng, 0.6f, 1.4f);
                log::infof("[weather] next: %.2f (in %.0f s until the change after)", weather_target_, weather_timer_);
            }
        }
        // A front takes about a quarter of the change interval to move in
        // (a forced value moves in over ~20 s).
        const float swing = rt_.weather_manual >= 0.0f
            ? 20.0f : std::max(0.2f, rt_.weather_change_minutes) * 60.0f * 0.25f;
        const float step = dt / swing;
        weather_state_ += glm::clamp(weather_target_ - weather_state_, -step, step);
        if (!weather_init_) { weather_state_ = weather_target_; weather_init_ = true; }
        w = weather_state_;
    } else {
        weather_state_ = 0.0f; weather_init_ = false; weather_timer_ = 0.0f;
    }
    const float overcast = glm::smoothstep(0.25f, 0.70f, w);
    const float rain     = glm::smoothstep(0.55f, 0.85f, w);
    const float storm    = glm::smoothstep(0.85f, 1.00f, w);
    env_rain_ = rain;
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
        if (e > 0.6f) {
            // Day and twilight: the real sun, also below the horizon - the
            // atmosphere LUT then renders dusk on its own. Direct light dies
            // at the horizon; the sky fades out toward the end of twilight.
            pitch = e; yaw = base.sun_yaw_deg + az;
            sun_k = glm::smoothstep(0.6f, 4.0f, e);
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
            const bool moon_is_light = moon_sky > 0.02f && moon_sky >= 0.5f * 0.8f * twi;
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
            if (moon_sky > 0.02f && moon_sky >= 0.5f * glow) {
                pitch = std::max(em, 3.0f);
                yaw = base.sun_yaw_deg + azm;
                sun_k = 0.045f * moon_sky * (1.0f + 1.6f * twi);
            } else {
                pitch = 16.0f;
                yaw = base.sun_yaw_deg + az;
                sun_k = 0.045f * glow;
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
            // Fill light on the land (1 = a clear full-moon night).
            const float night_amb = std::max(0.90f * moon_gnd + star_gnd, 0.003f);
            amb_k = glm::mix(night_amb, 1.0f, twi);
            light_col = glm::vec3(0.60f, 0.73f, 1.0f);
            golden = false;
            env_night_ = 1.0f;
            cloud_direct = 0.6f;
        }
        while (yaw > 180.0f) yaw -= 360.0f;
        while (yaw < -180.0f) yaw += 360.0f;
    }

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
    const float sky_dim = glm::mix(1.0f, 0.62f, overcast) * (1.0f + 5.0f * env_lightning_);
    // The cloud pass takes its light level directly (not times the sun intensity).
    env_cloud_k_ = sky_k * sky_dim;
    // Under a closed deck the light is uniformly dim: no shadow pattern.
    env_cloud_shadow_k_ = 0.72f * (1.0f - overcast);
    env_cloud_direct_ = cloud_direct * glm::mix(1.0f, 0.45f, overcast);

    rt_.atmosphere_intensity = base.atmosphere_intensity * sky_k * sky_dim / direct;
    const float sky_lum = glm::dot(base.sky_color, glm::vec3(0.2126f, 0.7152f, 0.0722f));
    rt_.sky_color = glm::mix(base.sky_color, glm::vec3(0.92f, 0.96f, 1.04f) * sky_lum, overcast * 0.85f) * amb_k;
    env_amb_k_ = amb_k;
    rt_.cloud_coverage = glm::mix(std::min(base.cloud_coverage, 0.30f), 0.97f, glm::smoothstep(0.0f, 0.72f, w));
    if (!rt_.weather_enabled) rt_.cloud_coverage = base.cloud_coverage;
    rt_.cloud_density = base.cloud_density * glm::mix(1.0f, 2.4f, overcast);
    rt_.cloud_speed = base.cloud_speed * glm::mix(1.0f, 3.2f, w);
    rt_.fog_density = base.fog_density * (1.0f + 0.6f * overcast + 2.0f * rain);
    rt_.grass_wind = base.grass_wind * glm::mix(1.0f, 2.6f, w);
    env_wind_scale_ = glm::mix(1.0f, 2.4f, w);

    tick_rain_audio(rain);
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
