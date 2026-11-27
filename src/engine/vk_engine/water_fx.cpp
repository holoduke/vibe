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
#include "engine/vk_engine/state/spacejets.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/audio.h"
#include "engine/globe.h"
#include "engine/terrain.h"
#include "engine/water_course.h"
#include "engine/river_synth.h"
#include "engine/sound_space.h"
#include <filesystem>
#include <cstdlib>

#include <algorithm>
#include <cmath>
#include <functional>
#include <mutex>
#include <vector>

namespace vibe {

namespace {
constexpr float kG = 9.81f;
constexpr size_t kMaxRipples = 16;
constexpr size_t kMaxDrops = 1400;
}

namespace { float stream_relief(glm::vec2 p, const wc::At& A); }
float VulkanEngine::water_surface_y(float x, float z) const {
    float s = wc::stream_surface(glm::vec2(x, z), rt_.water_level);
    if (s > -1e8f) {
        const wc::At A = wc::at(wc::stream(rt_.water_level), glm::vec2(x, z));
        if (A.on && A.dist < wc::wet_reach(A)) s += stream_relief(glm::vec2(x, z), A);
    }
    return std::max(rt_.water_level + water_level_off(x, z) + water_swell_height(x, z), s);
}

// Open water at (x, z): the water plane is above the ground there.
bool VulkanEngine::water_at(float x, float z) const {
    if (!rt_.water_enabled) return false;
    const float g = sample_terrain_height(x, z);
    if (g < rt_.water_level + water_level_off(x, z) - 0.02f) return true;
    // (the stream: water above the sea - engine/water_course.h)
    return g < wc::stream_surface(glm::vec2(x, z), rt_.water_level) - 0.02f;
}

// The sea's swell - the CPU twin of terrain_rm_water.glsl wt_swell / wt_sea_k / wt_swell_shore
// (same constants, same clock: anim_time_ = scene.water_params.w; keep the two in step).
// Five long waves after the reference game's sea: lengths 400 / 180 / 150 / 130 / 120 m, the
// first along the wind and the others at fixed angles to it, crest shape 1.3 exp(sin p - 1) - 0.3
// (its mean taken off, so the sea's mean level stays the water level), each phase pushed along
// its direction by a slow noise. Height x pow(saturate(wind / 30), 2.2) x the shore factor.
namespace {
// The open sea's weight (terrain_rm_water.glsl wt_sea_k, in any chart: rm_home_dist and
// rm_continent there; it was 1 everywhere off home's chart - a swimmer on an inland lake of a
// near chart bobbed on a swell the lake does not draw).
float sea_weight(float x, float z) {
    float k = glm::smoothstep(1350.0f, 1750.0f, world_home_distance(x, z));
    if (k > 0.0f) k *= 1.0f - glm::smoothstep(0.0f, 0.02f, world_continent_at(x, z));
    return k;
}
constexpr float kSwK[5]    = { 0.015708f, 0.034907f, 0.041888f, 0.048332f, 0.052360f };
constexpr float kSwF[5]    = { 0.0937143f, 0.2142082f, 0.3162719f, 0.3835667f, 0.4790747f };
constexpr float kSwA[5]    = { 13.0f, 8.0f, 7.0f, 2.5f, 2.0f };
constexpr float kSwWarp[5] = { 50.0f, 10.0f, 100.0f, 200.0f, 180.0f };
constexpr float kSwCos[5]  = { 1.0f, 0.7660444f, 0.8910065f, 0.9612617f, 0.9781476f };
constexpr float kSwSin[5]  = { 0.0f, 0.6427876f, -0.4539905f, 0.2756374f, -0.2079117f };
constexpr float kSwMean    = 0.30547f;       // the mean of 1.3 exp(sin p - 1) - 0.3

float swell_noise(glm::vec2 x, glm::vec2 drift) {
    glm::vec2 q = x * 0.005f - drift * 0.125f;          // (drift: the wind's direction summed over time, EnvState::water_drift)
    auto rot = [](glm::vec2 r) { return glm::vec2(0.8f * r.x - 0.6f * r.y, 0.6f * r.x + 0.8f * r.y); };
    float s = 0.5f * std::sin(1.5f * q.x) * std::sin(1.5f * q.y);
    glm::vec2 r = rot(q);
    s += 0.25f * std::sin(3.03f * r.x) * std::sin(3.03f * r.y);
    r = 2.02f * rot(r); s += 0.125f * std::sin(3.045f * r.x) * std::sin(3.045f * r.y);
    r = 2.03f * rot(r); s += 0.0625f * std::sin(3.015f * r.x) * std::sin(3.015f * r.y);
    return std::abs(s * 1.032258f);
}
}  // namespace

// THE WATER'S LEVEL under the datum (rt_.water_level) at (x, z) - the CPU twin of
// shaders/water_level.glsl wl_off (keep in step): the open sea 1 m under it, the river falling from
// the datum (down to below the stone bridge's pool) to the sea's level down its course, evenly with
// three short rapids; lakes away from the river at the datum. VIBE_NO_SLOPED_RIVER=1: 0 everywhere.
// The river's level under the datum `along` metres down its course (water_level.glsl wl_profile).
static float river_level_profile(float along) {
    const float g = std::clamp((along - 350.0f) / 2050.0f, 0.0f, 1.0f);
    const float r = glm::smoothstep(696.0f, 704.0f, along) + glm::smoothstep(1246.0f, 1254.0f, along) + glm::smoothstep(1796.0f, 1804.0f, along);
    return -(0.31f * g + 0.23f * r);
}
float water_level_off(float x, float z) {
    static const bool off = [] { const char* e = std::getenv("VIBE_NO_SLOPED_RIVER"); return e && *e && *e != '0'; }();
    if (off) return 0.0f;
    const float sk = sea_weight(x, z);
    if (sk >= 0.999f) return -1.0f;
    float riv = 0.0f;
    if (globe::current().home && x > -200.0f && x < 2250.0f && z > -250.0f && z < 900.0f) {
        const glm::vec2* rp = nullptr;
        const int rn = world_river_course(rp);
        const glm::vec2 p(x, z);
        float d2 = 1e12f, along = 0.0f, cum = 0.0f;
        for (int i = 0; i + 1 < rn; ++i) {
            const glm::vec2 a = rp[i], ab = rp[i + 1] - a;
            const float len = glm::length(ab);
            const float u = std::clamp(glm::dot(p - a, ab) / glm::dot(ab, ab), 0.0f, 1.0f);
            const glm::vec2 dv = p - (a + ab * u);
            const float e2 = glm::dot(dv, dv);
            if (e2 < d2) { d2 = e2; along = cum + u * len; }
            cum += len;
        }
        const float rk = 1.0f - glm::smoothstep(90.0f, 170.0f, std::sqrt(d2));
        if (rk > 0.0f) riv = rk * river_level_profile(along);
    }
    return glm::mix(riv, -1.0f, sk);
}

float VulkanEngine::water_swell_height(float x, float z) const {
    const glm::vec4 W = env_->water_wind;
    static const bool no_fft = std::getenv("VIBE_NO_FFT") != nullptr, no_sdf = std::getenv("VIBE_NO_SHORE_SDF") != nullptr;
    const float fft_b = no_fft ? 0.0f : water_fft_bound();
    if (!rt_.water_enabled || (W.w < 1e-5f && fft_b <= 0.0f)) return 0.0f;
    // The open sea only: beyond home's lakes, not on land-locked water.
    const float sea_k = sea_weight(x, z);
    if (sea_k <= 0.0f) return 0.0f;
    const float g0 = sample_terrain_height(x, z);
    const float depth = rt_.water_level + water_level_off(x, z) - g0;
    // The swell and the wind sea's cascades fade toward the coast by its distance (the shader's
    // ww_shore_dist: depth / the bed's slope, two more taps; within 60 m of depth only) and by the
    // depth (terrain_raymarch.frag, the sea's hit - keep in step).
    float dk = 1.0f;
    if (!no_sdf && depth < 60.0f) {
        const float gx = sample_terrain_height(x + 2.0f, z), gz = sample_terrain_height(x, z + 2.0f);
        const float dist = std::max(depth, 0.0f) / std::max(glm::length(glm::vec2(gx - g0, gz - g0) * 0.5f), 0.004f);
        dk = glm::smoothstep(0.0f, 1.0f, dist / 150.0f);
    }
    const float k = W.w * sea_k * std::min(dk, glm::smoothstep(0.0f, 1.0f, depth / (2.0f + 39.0f * W.w)));
    const float kf = fft_b > 0.0f ? sea_k * std::min(dk, glm::smoothstep(0.0f, 1.0f, depth / (0.5f + 3.0f * fft_b))) : 0.0f;
    const float hf = kf > 0.0f ? kf * water_fft_height(x, z) : 0.0f;
    if (k <= 0.0f) return hf;
    const glm::vec2 wd(W.x, W.y), p(x, z);
    const float t = anim_time_;
    const float n = swell_noise(p, env_->water_drift);
    float h = 0.0f;
    for (int i = 0; i < 5; ++i) {
        // (the reference's direction: minus the wind turned by the wave's angle; the crests travel with the wind)
        const glm::vec2 d(-(wd.x * kSwCos[i] - wd.y * kSwSin[i]), -(wd.x * kSwSin[i] + wd.y * kSwCos[i]));
        const float ft = kSwF[i] * t;
        const float ph = 6.2831853f * (ft - std::floor(ft)) + kSwK[i] * (glm::dot(d, p) + kSwWarp[i] * n);
        h += kSwA[i] * (1.3f * std::exp(std::sin(ph) - 1.0f) - 0.3f - kSwMean);
    }
    return k * h + hf;
}

// ---- The river's current on the CPU: the twin of terrain_rm_water.glsl wt_river and the
// current block of terrain_rm_main_water.glsl (same course, bend, pool, speed; keep in step).
// The reference game reads its flow back from the GPU so that what is drawn and what carries
// things agree; here both sides work it out from the same formula.
namespace {
struct RiverAt { glm::vec2 dir{ 1.0f, 0.0f }; float dist = -1.0f, half_w = 7.0f, along = 0.0f, pool = 0.0f; };
RiverAt river_at(glm::vec2 wp) {
    RiverAt r;
    if (!globe::current().home || wp.x < -85.0f || wp.x > 2095.0f || wp.y < -110.0f || wp.y > 780.0f) return r;
    const glm::vec2* rp = nullptr;
    const int rn = world_river_course(rp);
    glm::vec2 q = wp + world_river_wander(wp);
    r.pool = 1.0f - glm::smoothstep(22.0f, 55.0f, glm::distance(wp, glm::vec2(43.4f, 297.4f)));
    q -= r.pool * 6.0f * glm::vec2(0.719f, 0.695f);
    float d2 = 1e12f, cum = 0.0f;
    glm::vec2 tan(1e-4f, 0.0f);
    for (int i = 0; i + 1 < rn; ++i) {
        const glm::vec2 a = rp[i], ab = rp[i + 1] - a;
        const float len = glm::length(ab);
        const float u = glm::clamp(glm::dot(q - a, ab) / glm::dot(ab, ab), 0.0f, 1.0f);
        const glm::vec2 dv = q - (a + ab * u);
        const float e2 = glm::dot(dv, dv);
        if (e2 < d2) { d2 = e2; r.along = cum + u * len; }
        const float wq = 1.0f / (1.0f + e2 / 400.0f);
        tan += (ab / len) * (wq * wq);
        cum += len;
    }
    if (d2 > 60.0f * 60.0f) return r;
    const float ja = 0.516f * std::cos(wp.y * 0.043f + 1.3f) + 0.565f * std::cos(wp.y * 0.113f);
    const float jb = 0.444f * std::cos(wp.x * 0.037f + 0.4f) + 0.485f * std::cos(wp.x * 0.097f + 2.0f);
    const glm::vec2 tq = glm::normalize(tan);
    glm::vec2 t(tq.x - ja * tq.y, tq.y - jb * tq.x);
    if (1.0f - ja * jb < 0.0f) t = -t;
    r.dir = glm::dot(t, t) > 1e-12f ? glm::normalize(t) : tq;       // (where the bend's Jacobian is singular: the polyline's way)
    r.dist = std::sqrt(d2);
    r.half_w = 7.0f + 9.0f * glm::smoothstep(1450.0f, 1900.0f, wp.x) + 4.0f * r.pool;
    return r;
}
// The piers standing in the river (river_pier_add from bridges.cpp and world.cpp).
std::vector<glm::vec4> g_river_piers;
}  // namespace

// The boulders standing in the water (scatter.cpp): a box each for a walker or swimmer (they
// had none - one waded through a stone a metre high). Inside the stone's round body: 0.7 r
// either way, from its buried base to 0.6 r above the ground. Small ones (r < 0.4) are stepped
// over and get none.
namespace {
std::vector<collision::AABB> g_water_boulders;
// The bed the shallow-water solver stamps (water_solver.cpp): piers and rocks as walls, the
// boulders as the round stones they are (scatter.cpp draws them centred 0.35 r under the ground).
// (scatter may build on a worker: the list is locked)
std::mutex g_bed_mx;
std::vector<glm::vec4> g_bed_obs;
uint64_t g_bed_sig = 1;
void bed_obs_put(glm::vec4 o) {
    std::lock_guard<std::mutex> lk(g_bed_mx);
    for (glm::vec4& q : g_bed_obs)
        if (std::abs(q.x - o.x) < 0.5f && std::abs(q.y - o.y) < 0.5f) {
            // (a boulder's round shape replaces the wall its river_pier_add put there; a wall does not
            //  replace a stone)
            if (o.w < 5e4f || q.w >= 5e4f) { if (q != o) { q = o; ++g_bed_sig; } }
            return;
        }
    g_bed_obs.push_back(o);
    ++g_bed_sig;
}
}  // namespace
uint64_t water_bed_obstacles(glm::vec2 c, float reach, std::vector<glm::vec4>& out) {
    std::lock_guard<std::mutex> lk(g_bed_mx);
    for (const glm::vec4& o : g_bed_obs)
        if (std::abs(o.x - c.x) < reach + o.z && std::abs(o.y - c.y) < reach + o.z) out.push_back(o);
    return g_bed_sig;
}
float water_sea_weight(float x, float z) { return sea_weight(x, z); }
void water_boulder_add(glm::vec2 p, float ground, float r) {
    bed_obs_put(glm::vec4(p, r, ground - 0.35f * r));
    for (glm::vec4& q : g_river_piers) if (glm::distance(glm::vec2(q), p) < 0.5f) q.w = ground + 0.65f * r;     // (its top: river_piers w)
    if (r < 0.4f) return;
    const collision::AABB b{ glm::vec3(p.x - 0.7f * r, ground - 0.35f * r, p.y - 0.7f * r), glm::vec3(p.x + 0.7f * r, ground + 0.6f * r, p.y + 0.7f * r) };
    for (collision::AABB& q : g_water_boulders)
        if (std::abs(0.5f * (q.min.x + q.max.x) - p.x) < 0.5f && std::abs(0.5f * (q.min.z + q.max.z) - p.y) < 0.5f) { q = b; return; }     // (the scatter built again)
    g_water_boulders.push_back(b);
}
void water_boulder_colliders(glm::vec3 c, float reach, std::vector<collision::AABB>& out) {
    if (!globe::current().home) return;
    for (const collision::AABB& a : g_water_boulders)
        if (a.max.x > c.x - reach && a.min.x < c.x + reach && a.max.z > c.z - reach && a.min.z < c.z + reach) out.push_back(a);
}

void river_piers_clear() { g_river_piers.clear(); }
void river_pier_add(glm::vec2 p, float half_width, float top) {
    // (a stone whose top is known (under 1e4): a round stone in the solver's bed with that top;
    //  else a wall. river_piers w = the top: the stream's surface humps over one that lies under
    //  the water - water_course.glsl wc_relief)
    bed_obs_put(top < 1e4f ? glm::vec4(p, half_width, top - half_width) : glm::vec4(p, half_width, 1e5f));
    for (glm::vec4& q : g_river_piers) if (glm::distance(glm::vec2(q), p) < 0.5f) { q.w = top; return; }      // (the world built again: the same piers)
    g_river_piers.push_back(glm::vec4(p, half_width, top));
}
// The stone bridge's two piers (game/level.cpp append_stone_bridge: the same layout - keep in step).
void river_piers_stone_bridge(glm::vec3 a, glm::vec3 b, const std::function<float(float, float)>& ground, float water_level) {
    const glm::vec2 a2(a.x, a.z), b2(b.x, b.z);
    const float L = glm::length(b2 - a2);
    if (L < 20.0f) return;
    const glm::vec2 d = (b2 - a2) / L;
    const float R = std::min(7.0f, (L - 26.0f) / 6.0f), pier = 4.0f;
    const float open = 6.0f * R + 2.0f * pier, s_first = std::floor((L - open) * 0.42f * 2.0f) * 0.5f;
    for (int i = 0; i < 2; ++i) {
        const float s = s_first + R + float(i) * (2.0f * R + pier) + R + 0.5f * pier;
        const glm::vec2 p = a2 + d * s;
        if (ground(p.x, p.y) < water_level - 0.2f) river_pier_add(p, 0.5f * pier);
    }
}
// The (up to) 24 nearest the eye, for the scene block (scene_ubo.glsl river_piers); returns how many.
int river_piers_fill(glm::vec4* out, glm::vec3 eye) {
    std::vector<glm::vec4> v = g_river_piers;
    std::sort(v.begin(), v.end(), [&](const glm::vec4& A, const glm::vec4& B) {
        return glm::distance(glm::vec2(A), glm::vec2(eye.x, eye.z)) < glm::distance(glm::vec2(B), glm::vec2(eye.x, eye.z)); });
    const int n = int(std::min<size_t>(v.size(), 24));
    for (int i = 0; i < 24; ++i) out[i] = i < n ? v[size_t(i)] : glm::vec4(0.0f);
    return n;
}

// The stream's surface in relief - the CPU twin of shaders/water_relief.glsl wr_relief (keep the two
// in step; the solver's part is the GPU's only): humps over the stones under the surface with a dip
// and standing waves behind, the pillow against a stone standing out and the drawdown in its lee,
// a rounded pillow where stones squeeze the flow. All stones within reach (the GPU has the 24
// nearest the eye: the same ones near the player). water_wet bit 4096 (VIBE_NO_STREAM_RELIEF=1).
namespace {
float stream_relief(glm::vec2 p, const wc::At& A) {
    static const bool off = [] { const char* e = std::getenv("VIBE_NO_STREAM_RELIEF"); return e && *e && *e != '0'; }();
    if (off || !A.on) return 0.0f;
    const float sl = std::max(A.sslope, A.drop > 0.01f ? A.slope : 0.0f);
    const float sn = sl / std::sqrt(1.0f + sl * sl);
    const float U = std::min(0.75f + 3.0f * sn, 4.5f);
    const float hv = U * U / 19.62f;
    const float lam = std::clamp(6.2832f * U * U / 9.81f, 0.6f, 4.0f);
    const glm::vec2 fd = A.dir, ac(-fd.y, fd.x);
    const float S0 = A.surf;
    float h = 0.0f, gl = 1e9f, gr = 1e9f;
    for (const glm::vec4& P : g_river_piers) {
        const glm::vec2 d = p - glm::vec2(P);
        const float a = std::max(P.z, 0.12f);
        const float reach = 2.0f * a + 3.0f * lam + 1.0f;
        const float r2 = glm::dot(d, d);
        if (r2 > reach * reach) continue;
        const float down = glm::dot(d, fd), lat = glm::dot(d, ac), r = std::sqrt(r2);
        const float rf = 1.0f - glm::smoothstep(0.75f * reach, reach, r);
        const float S0c = S0 + sl * down;
        const float bsub = glm::smoothstep(-0.03f, 0.07f, S0c - P.w);
        {
            const float s = std::max(S0c - P.w, 0.0f);
            const float hc = std::min(0.9f * hv * a / (a + s), 0.45f);
            const glm::vec2 dd = d - fd * (0.35f * a);
            const float R = 1.4f * a + 0.3f, q = glm::dot(dd, dd) / (R * R);
            if (q < 1.0f) h += bsub * hc * (1.0f - q) * (1.0f - q);
            const float xi = down - (1.5f * a + 0.25f);
            const float wd = 1.1f * a + 0.2f + 0.15f * std::max(down, 0.0f);
            const float wl = std::exp(-lat * lat / (wd * wd));
            h += 0.55f * hc * -std::cos(6.2832f * xi / lam) * std::exp(-std::max(xi, 0.0f) / (1.5f * lam)) * glm::smoothstep(-0.6f * a - 0.2f, 0.0f, xi) * wl * rf * bsub;
        }
        {
            const float ring = std::max(r - a, 0.0f);
            const float inside = glm::smoothstep(0.8f * a, a, r);
            h += 0.8f * hv * std::exp(-ring / (0.3f + 0.2f * a)) * glm::smoothstep(-0.3f * a, 0.6f * a, -down) * inside * rf * (1.0f - bsub);
            h -= 0.5f * hv * std::exp(-lat * lat / (a * a)) * std::exp(-std::max(down, 0.0f) / (1.2f * a + 0.5f)) * glm::smoothstep(0.0f, 0.4f * a, down) * inside * rf * (1.0f - bsub);
            const float e = std::max(std::abs(lat) - a, 0.0f) + 0.05f + 20.0f * std::max(glm::smoothstep(a + 0.3f, a + 0.8f, std::abs(down)), bsub);
            if (lat > 0.0f) gr = std::min(gr, e); else gl = std::min(gl, e);
        }
    }
    {
        const float hwb = A.half_w;
        gl = std::min(gl, std::max(hwb - A.lat, 0.05f));
        gr = std::min(gr, std::max(hwb + A.lat, 0.05f));
        const float g = gl + gr;
        const float sq = 1.0f - glm::smoothstep(0.35f, 0.75f, g / std::max(2.0f * hwb, 0.1f));
        h += 0.6f * hv * sq * std::pow(std::sin(3.1416f * std::clamp(gl / g, 0.0f, 1.0f)), 0.7f);
    }
    h *= 1.0f - glm::smoothstep(0.3f, 0.8f, A.slope) * (A.drop >= 0.01f ? 1.0f : 0.0f);
    h *= 1.0f - glm::smoothstep(0.65f, 1.0f, A.dist / std::max(wc::wet_reach(A), 0.1f));
    h = std::max(h, -0.33f * A.depth * (1.0f - glm::smoothstep(0.3f, 0.9f, A.dist / std::max(A.half_w, 0.1f))));      // (a dip never bares the bed)
    return std::clamp(h, -0.3f, 0.5f);
}
}  // namespace

// The current parts round the piers and rocks standing in it: potential flow round a cylinder
// of the obstacle's radius (the water shader does the same, and adds the eddies behind it).
static glm::vec2 obstacle_bend(glm::vec2 p, glm::vec2 f) {
    const float U = glm::length(f);
    if (U < 1e-4f) return f;
    const glm::vec2 d = f / U, ac(-d.y, d.x);
    glm::vec2 out = f;
    for (const glm::vec4& o : g_river_piers) {
        const glm::vec2 e = p - glm::vec2(o);
        const float r2 = glm::dot(e, e), a = o.z;
        if (r2 > 36.0f * a * a + 4.0f) continue;
        if (r2 < a * a) return glm::vec2(0.0f);                  // (inside the rock)
        const float down = glm::dot(e, d), lat = glm::dot(e, ac), k = a * a / r2;
        const float c2 = (down * down - lat * lat) / r2, s2 = 2.0f * down * lat / r2;
        out += (d * (-k * c2) + ac * (-k * s2)) * U;
    }
    return out;
}

glm::vec2 VulkanEngine::water_river_flow(float x, float z) const {
    const glm::vec2 f = water_course_flow(x, z);
    return (f.x != 0.0f || f.y != 0.0f) ? obstacle_bend(glm::vec2(x, z), f) : f;
}

// The course's own current, before the obstacles bend it: what the shallow-water solver's
// velocity is pulled toward (water_solver.cpp - the solver bends it round them by itself).
glm::vec2 VulkanEngine::water_course_flow(float x, float z) const {
    if (!rt_.water_enabled) return glm::vec2(0.0f);
    // The stream first (engine/water_course.h: the reference game's speed rule on its slope) -
    // where its surface stands above the river's: its last points lie in the river at the river's
    // level, and there the shader draws the river (the plane is met first), so the river's current
    // is what carries things there too.
    if (wc::stream_surface(glm::vec2(x, z), rt_.water_level) > rt_.water_level + water_level_off(x, z) + 0.005f) {
        const glm::vec2 sf = wc::stream_flow(glm::vec2(x, z), rt_.water_level);
        if (sf.x != 0.0f || sf.y != 0.0f) return sf;
    }
    const RiverAt rv = river_at(glm::vec2(x, z));
    if (rv.dist < 0.0f) return glm::vec2(0.0f);
    float k = (1.0f - glm::smoothstep(rv.half_w + 4.0f, rv.half_w + 20.0f, rv.dist)) *
              glm::smoothstep(30.0f, 80.0f, glm::distance(glm::vec2(x, z), glm::vec2(62.0f, 36.0f))) *
              (1.0f - glm::smoothstep(1880.0f, 1990.0f, x));
    if (k <= 0.0f) return glm::vec2(0.0f);
    k *= 1.0f - sea_weight(x, z);
    const float depth = std::max(0.0f, rt_.water_level + water_level_off(x, z) - sample_terrain_height(x, z));
    if (depth <= 0.0f) return glm::vec2(0.0f);
    float speed = (0.30f + 1.10f * glm::smoothstep(0.3f, 2.2f, depth)) * (1.0f - 0.6f * rv.pool);
    // (the rapids where the river's level falls: terrain_rm_main_water.glsl, the same)
    static const bool flat_river = [] { const char* e = std::getenv("VIBE_NO_SLOPED_RIVER"); return e && *e && *e != '0'; }();
    if (!flat_river) speed += 3.0f * std::clamp((river_level_profile(rv.along - 6.0f) - river_level_profile(rv.along + 6.0f)) / 12.0f, 0.0f, 0.03f) * 25.0f;
    const glm::vec4 W = env_->water_wind;
    return k * (rv.dir * speed + glm::vec2(W.x, W.y) * (0.012f * W.z));
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
    pos.y = water_surface_y(pos.x, pos.z);       // (on the swell, where there is one)
    if (std::getenv("VIBE_LOG_WATER")) log::infof("[water-probe] splash at (%.2f, %.3f, %.2f), size %.2f (stream surface there %.3f)", double(pos.x), double(pos.y), double(pos.z), double(scale), double(wc::stream_surface(glm::vec2(pos.x, pos.z), rt_.water_level)));
    scale = glm::clamp(scale, 0.3f, 16.0f);
    const float vs = std::sqrt(scale);            // ballistic height ~ v^2
    uint32_t& rng = spacejet_->rng_state;

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
    tick_river_sound(dt);
    // VIBE_LOG_WATER=1: once a second, the CPU's water where the player stands, and the CPU's
    // answer to what the screen's centre shows - the nearest of the four taps the autofocus reads
    // (exposure.comp: +-0.02 of the picture round the centre), marched against the ground and the
    // CPU's water surface, as a linear depth. Compare with the "focus" of VIBE_LOG_EXPOSURE (the
    // GPU's depth there): the difference x `dy/dlin` is how far the drawn surface lies from the
    // CPU's (swimmers, splashes and rings use the CPU's).
    static const bool log_water = std::getenv("VIBE_LOG_WATER") != nullptr;
    static float log_water_t = 0.0f;
    if (log_water && (log_water_t -= dt) <= 0.0f) {
        log_water_t = 1.0f;
        const glm::vec3 pp = player_.position;
        const glm::vec2 pf = water_river_flow(pp.x, pp.z);
        log::infof("[water-probe] player (%.2f, %.2f, %.2f): surface %.3f (swell %.3f, stream %.3f), water %d, ground %.3f, current (%.2f, %.2f)",
                   double(pp.x), double(pp.y), double(pp.z), double(water_surface_y(pp.x, pp.z)), double(water_swell_height(pp.x, pp.z)),
                   double(wc::stream_surface(glm::vec2(pp.x, pp.z), rt_.water_level)), int(water_at(pp.x, pp.z)), double(sample_terrain_height(pp.x, pp.z)), double(pf.x), double(pf.y));
        {
            std::vector<collision::AABB> nb; water_boulder_colliders(pp, 3.0f, nb);
            { std::vector<collision::AABB> wide; water_boulder_colliders(pp, 60.0f, wide); float bd = 1e9f; glm::vec3 bc(0.0f);
              for (const collision::AABB& a : wide) { const glm::vec3 c = 0.5f * (a.min + a.max); const float d = glm::distance(glm::vec2(c.x, c.z), glm::vec2(pp.x, pp.z)); if (d < bd) { bd = d; bc = c; } }
              if (bd < 1e8f) log::infof("[water-probe] nearest boulder box at (%.2f, %.2f, %.2f), %.1f m away, %zu within 60 m", double(bc.x), double(bc.y), double(bc.z), double(bd), wide.size()); }
            log::infof("[water-probe] moon dir (%.2f, %.2f, %.2f), sun dir (%.2f, %.2f, %.2f); boulder boxes within 3 m: %zu", double(env_->moon_dir.x), double(env_->moon_dir.y), double(env_->moon_dir.z),
                       double(env_->sun_dir.x), double(env_->sun_dir.y), double(env_->sun_dir.z), nb.size());
        }
        const FrameView& fv = current_frame_view_;
        auto unproj = [&](float nx, float ny, float nz) { const glm::vec4 h = fv.inv_vp * glm::vec4(nx, ny, nz, 1.0f); return glm::vec3(h) / h.w; };
        const glm::vec3 fwd = glm::normalize(unproj(0.0f, 0.0f, 0.8f) - unproj(0.0f, 0.0f, 0.2f));
        auto surf = [&](float x, float z, bool& wet) { wet = water_at(x, z); return wet ? water_surface_y(x, z) : sample_terrain_height(x, z); };
        float best = 1e9f; glm::vec3 bp(0.0f); bool bwet = false; float bdy = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const float nx = (k & 1) ? 0.04f : -0.04f, ny = (k & 2) ? 0.04f : -0.04f;     // (uv +-0.02)
            const glm::vec3 d = glm::normalize(unproj(nx, ny, 0.8f) - unproj(nx, ny, 0.2f));
            const glm::vec3 o = fv.eye_pos;
            bool wet = false;
            float t0 = 0.0f, t = 0.0f;
            bool hit = false;
            while (t < 300.0f) {
                const glm::vec3 p = o + d * t;
                if (p.y <= surf(p.x, p.z, wet)) { hit = true; break; }
                t0 = t; t += 0.02f + t * 0.002f;
            }
            if (!hit) continue;
            for (int i = 0; i < 24; ++i) { const float tm = 0.5f * (t0 + t); const glm::vec3 p = o + d * tm; if (p.y <= surf(p.x, p.z, wet)) t = tm; else t0 = tm; }
            const glm::vec3 p = o + d * t;
            const float lin = t * glm::dot(d, fwd);
            if (lin > 1.2f && lin < best) { best = lin; bp = p; surf(p.x, p.z, bwet); bdy = std::abs(d.y) / std::max(glm::dot(d, fwd), 1e-3f); }
        }
        if (best < 1e8f)
            log::infof("[water-probe] centre taps: CPU depth %.3f m at (%.2f, %.3f, %.2f), %s (surface %.3f, ground %.3f), dy/dlin %.3f",
                       double(best), double(bp.x), double(bp.y), double(bp.z), bwet ? "water" : "ground", double(water_surface_y(bp.x, bp.z)),
                       double(sample_terrain_height(bp.x, bp.z)), double(bdy));
    }
    // Rings and splashes ride the current (the river's and the stream's: water_river_flow) -
    // on running water they slide downstream instead of standing still. (The reference game's
    // ripples do not drift.)
    // (inside the shallow-water solver's window its current, read back from the GPU: water_flow_at)
    for (auto& r : water_ripples_) { r.age += dt; r.xz += water_flow_at(r.xz.x, r.xz.y) * dt; }
    water_ripples_.erase(std::remove_if(water_ripples_.begin(), water_ripples_.end(),
                                        [](const WaterRipple& r) {
                                            const float s = glm::clamp(r.amp * 28.5f, 1.0f, 6.0f);
                                            return r.age > 2.5f + 1.5f * s;
                                        }),
                         water_ripples_.end());
    for (auto& s : water_splashes_) { s.t += dt; const glm::vec2 fl = water_flow_at(s.pos.x, s.pos.z); s.pos += glm::vec3(fl.x, 0.0f, fl.y) * dt; }
    tick_water_spray(dt);      // (spray and mist at the falls, the rocks, the sea's whitecaps: water_spray.cpp)
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
        // (the swell lifts and drops the surface: only worked out close above the level)
        if (!dead && d.vel.y < 0.0f && d.pos.y <= water_surface_y(d.pos.x, d.pos.z)) {
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

}  // namespace vibe

namespace vibe {

// The river's sound (and the stream's): ONE source of the spatialiser, on whichever of the two
// courses is the louder from where the listener is - at the point of that course nearest the
// listener, a little above its water - so it follows the listener along the bank. Louder where
// the water runs fast, and much louder by the stream's steps. The clip is made here the first
// time (engine/river_synth.h -> assets/sounds/river_loop.wav).
namespace {
struct RiverSound { uint32_t src = 0; int clip = 0; glm::vec3 pos{ 0.0f }; float log_t = 0.0f; };
RiverSound g_rs;
}  // namespace

void VulkanEngine::tick_river_sound(float dt) {
    // VIBE_LOG_STREAM=1: the stream's points against the ground the game stands on (once).
    static bool logged = false;
    if (!logged && std::getenv("VIBE_LOG_STREAM")) {
        logged = true;
        const wc::Course& st = wc::stream(rt_.water_level);
        for (size_t i = 0; i < st.pts.size(); i += 4) {
            const glm::vec4 p = st.pts[i];
            log::infof("[stream] %2zu (%.0f, %.0f): water %.2f, ground here %.2f (formula %.2f, natural %.2f), half width %.1f", i, double(p.x), double(p.y), double(p.z),
                       double(sample_terrain_height(p.x, p.y)), double(world_ground_height(p.x, p.y, rt_.water_level, 9)), double(world_ground_height_natural(p.x, p.y, rt_.water_level, 9)), double(p.w));
        }
        // Where the water's reach ends (wc::wet_reach) the ground must stand above the surface, or
        // the water stops there in a cut edge: the lowest ground there less the surface, along
        // the whole course (both sides, every 2 m).
        if (st.ok) {
            float worst = 1e9f; glm::vec2 wp(0.0f); int under = 0, n = 0;
            for (size_t k = 0; k + 1 < st.pts.size(); ++k) {
                const glm::vec2 a(st.pts[k]), b(st.pts[k + 1]);
                const float L = glm::distance(a, b);
                const glm::vec2 d = (b - a) / std::max(L, 1e-3f), nrm(-d.y, d.x);
                for (float u = 0.0f; u < L; u += 2.0f)
                    for (float sd : { -1.0f, 1.0f }) {
                        const glm::vec2 c = a + d * u;
                        const wc::At A0 = wc::at(st, c + nrm * sd);
                        const glm::vec2 q = c + nrm * (sd * wc::wet_reach(A0));
                        const wc::At A = wc::at(st, q);
                        if (!A.on || A.dist + 0.5f < glm::distance(c, q)) continue;       // (nearer another reach: a bend's inside)
                        if (A.surf <= rt_.water_level + 0.01f) continue;                   // (in the river: its water goes on)
                        const float m = sample_terrain_height(q.x, q.y) - A.surf;
                        ++n; if (m < 0.0f) ++under;
                        if (m < worst) { worst = m; wp = q; }
                    }
            }
            log::infof("[stream] at the water's reach: %d of %d places have ground under the surface; the lowest %.2f m at (%.0f, %.0f)", under, n, double(worst), double(wp.x), double(wp.y));
        }
    }
    if (!audio_) return;
    const sound_space::Shape shape{ 9.0f, 7000.0f, 260.0f, 0.04f, 90.0f };
    const glm::vec3 ear = player_.eye_position();
    bool want = rt_.water_enabled && globe::current().home;
    glm::vec3 best_p(0.0f);
    float best_k = 0.0f, best_level = 0.0f;
    if (want) {
        // The river: the nearest point of its course (in the bent coordinate the channel is cut
        // along, taken back to the world by the same offset - near enough for a sound).
        {
            const glm::vec2* rp = nullptr;
            const int rn = world_river_course(rp);
            const glm::vec2 L(ear.x, ear.z), q = L + world_river_wander(L);
            float d2 = 1e18f; glm::vec2 nq = q;
            for (int i = 0; i + 1 < rn; ++i) {
                const glm::vec2 a = rp[i], ab = rp[i + 1] - a;
                const glm::vec2 c = a + ab * std::clamp(glm::dot(q - a, ab) / glm::dot(ab, ab), 0.0f, 1.0f);
                const float e2 = glm::dot(q - c, q - c);
                if (e2 < d2) { d2 = e2; nq = c; }
            }
            const glm::vec2 p = L + (nq - q);
            const float spd = glm::length(water_river_flow(p.x, p.y));
            if (spd > 0.05f) {
                const float lvl = 0.30f + 0.35f * spd;
                const float d = glm::distance(glm::vec3(p.x, rt_.water_level, p.y), ear);
                const float k = lvl / (1.0f + d / shape.ref_m);
                if (k > best_k) { best_k = k; best_level = lvl; best_p = glm::vec3(p.x, rt_.water_level + 0.3f, p.y); }
            }
        }
        // The stream: the nearest point of its centre line; its steps roar.
        {
            const wc::Course& st = wc::stream(rt_.water_level);
            if (st.ok) {
                const glm::vec2 L(ear.x, ear.z);
                float d2 = 1e18f; glm::vec2 np = L;
                for (size_t i = 0; i + 1 < st.pts.size(); ++i) {
                    const glm::vec2 a(st.pts[i]), ab = glm::vec2(st.pts[i + 1]) - a;
                    const glm::vec2 c = a + ab * std::clamp(glm::dot(L - a, ab) / std::max(glm::dot(ab, ab), 1e-6f), 0.0f, 1.0f);
                    const float e2 = glm::dot(L - c, L - c);
                    if (e2 < d2) { d2 = e2; np = c; }
                }
                const wc::At A = wc::at(st, np);
                if (A.on) {
                    // (the steepest water within 40 m upstream and down: a fall is heard from round the bend)
                    // (a step counts by its height: its face is only a metre or two long, and the samples 10 m
                    //  apart stepped over it - by the stream's highest fall the level was the quiet run's)
                    auto roar = [](const wc::At& B) { return std::max(B.slope, B.drop * 0.5f); };
                    float steep = roar(A);
                    for (float o : { -40.0f, -30.0f, -20.0f, -15.0f, -10.0f, -5.0f, 5.0f, 10.0f, 15.0f, 20.0f, 30.0f, 40.0f }) { const wc::At B = wc::at(st, np + A.dir * o); if (B.on) steep = std::max(steep, roar(B) * (1.0f - std::abs(o) / 60.0f)); }
                    const float lvl = 0.45f + 1.8f * glm::smoothstep(0.1f, 0.8f, steep);
                    const glm::vec3 p(np.x, A.surf + 0.3f, np.y);
                    const float k = lvl / (1.0f + glm::distance(p, ear) / shape.ref_m);
                    if (k > best_k) { best_k = k; best_level = lvl; best_p = p; }
                }
            }
        }
        want = best_k > 0.0f && glm::distance(best_p, ear) < shape.hear_m;
    }
    if (!want) {
        if (g_rs.src) { audio_->source_stop(g_rs.src); g_rs.src = 0; }
        return;
    }
    if (g_rs.clip == 0) {
        const char* path = "assets/sounds/river_loop.wav";
        std::error_code ec;
        const bool made = !std::filesystem::exists(path, ec) && river_synth::write_wav(path);
        g_rs.clip = std::filesystem::exists(path, ec) ? 1 : -1;
        if (g_rs.clip > 0) audio_->load_clip(ClipID::River, path);
        log::infof("[river] its sound: %s%s", g_rs.clip > 0 ? path : "could not be written - none", made ? " (made now: 8 s, engine/river_synth.h)" : "");
    }
    if (g_rs.clip <= 0) return;
    if (!g_rs.src) { g_rs.src = audio_->source_start(ClipID::River, shape, true, 0.0f); g_rs.pos = best_p; }
    if (!g_rs.src) return;
    // (the point slides along the bank with the listener; a jump to the other course is eased)
    g_rs.pos = glm::mix(g_rs.pos, best_p, std::min(1.0f, dt * 2.0f));
    audio_->source_update(g_rs.src, g_rs.pos, glm::vec3(0.0f), std::min(best_level, 2.2f), 1.0f);
    static const bool lg = std::getenv("VIBE_LOG_RIVER") != nullptr;
    g_rs.log_t -= dt;
    if (lg && g_rs.log_t <= 0.0f) {
        g_rs.log_t = 1.0f;
        log::infof("[river] sound at (%.0f, %.1f, %.0f), %.0f m away, level %.2f", g_rs.pos.x, g_rs.pos.y, g_rs.pos.z, glm::distance(g_rs.pos, ear), best_level);
    }
}

}  // namespace vibe
