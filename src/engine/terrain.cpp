#include "engine/terrain.h"
#include "engine/highway.h"
#include "engine/globe.h"

#include "engine/log.h"

#include <glm/glm.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#ifdef max
#  undef max
#endif
#ifdef min
#  undef min
#endif

namespace vibe {

float Heightmap::sample_world(float x, float z) const {
    float fx = (x - origin_x) / cell;
    float fz = (z - origin_z) / cell;
    if (fx < 0.0f || fz < 0.0f) return 0.0f;
    int ix = static_cast<int>(std::floor(fx));
    int iz = static_cast<int>(std::floor(fz));
    if (ix >= dim || iz >= dim) return 0.0f;
    float tx = fx - static_cast<float>(ix);
    float tz = fz - static_cast<float>(iz);
    float h00 = at(ix,     iz);
    float h10 = at(ix + 1, iz);
    float h01 = at(ix,     iz + 1);
    float h11 = at(ix + 1, iz + 1);
    float h0 = h00 * (1.0f - tx) + h10 * tx;
    float h1 = h01 * (1.0f - tx) + h11 * tx;
    return h0 * (1.0f - tz) + h1 * tz;
}

// === FBM heightmap matching terrain_raymarch.frag ============================
//
// The fragment shader's terrainM uses 2D value noise with analytical
// derivatives + derivative-erosion suppression. We mirror it here so the
// CPU heightmap (driving physics / grass / mesh / BLAS) matches what the
// shader draws. Constants are bit-identical to the shader; small float
// drift is OK because the visible raymarch is what the user sees, and a
// few cm of grass / physics offset is invisible.
namespace {

inline glm::vec3 rm_fract(glm::vec3 v) { return v - glm::floor(v); }
inline float     rm_fract1(float v)    { return v - std::floor(v); }

// hash21 — exactly mirrors the shader's hash function.
inline float rm_hash21(glm::vec2 p) {
    glm::vec3 p3 = rm_fract(glm::vec3(p.x, p.y, p.x) * 0.1031f);
    p3 += glm::dot(p3, glm::vec3(p3.y, p3.z, p3.x) + 19.19f);
    return rm_fract1((p3.x + p3.y) * p3.z);
}

inline float rm_noised(glm::vec2 p, glm::vec2& deriv) {
    glm::vec2 i = glm::floor(p);
    glm::vec2 f = p - i;
    // Hermite cubic — MUST stay byte-identical in meaning to
    // terrain_raymarch.frag / grass_raymarch.frag noised(). This is
    // the CPU FBM that bakes the physics heightfield + grass baseline;
    // if it diverges from the shader the collision surface no longer
    // matches the visible terrain. (Reverted quintic experiment.)
    glm::vec2 u  = f * f * (3.0f - 2.0f * f);
    glm::vec2 du = 6.0f * f * (1.0f - f);
    float a = rm_hash21(i + glm::vec2(0, 0));
    float b = rm_hash21(i + glm::vec2(1, 0));
    float c = rm_hash21(i + glm::vec2(0, 1));
    float d = rm_hash21(i + glm::vec2(1, 1));
    float val = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    deriv = du * (glm::vec2(b - a, c - a) +
                  (a - b - c + d) * glm::vec2(u.y, u.x));
    return val;
}

constexpr float kRmTerrainScale  = 0.003f;
constexpr float kRmTerrainHeight = 120.0f;
constexpr float kRmPlateauBlend  = 20.0f;   // matches shader/world plateau_blend

// Mirror of shader m2 = mat2(0.8, -0.6, 0.6, 0.8). GLSL is column-major;
// applying m * p = (col0*p.x + col1*p.y) gives:
//   x' =  0.8*p.x + 0.6*p.y
//   y' = -0.6*p.x + 0.8*p.y
inline glm::vec2 rm_rotate(glm::vec2 p) {
    return glm::vec2(0.8f * p.x + 0.6f * p.y,
                     -0.6f * p.x + 0.8f * p.y);
}

// ---- The world map: the CPU's copy of shaders/world_map.glsl -----------------------------
// Keep in step with that file line for line (the collision surface out there is this).
namespace wm {
inline float noise(glm::vec2 p) { glm::vec2 d; return rm_noised(p, d); }
inline float sstep(float a, float b, float x) {
    float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
inline float fbm(glm::vec2 p) {
    float s = 0.0f, a = 0.5f;
    for (int i = 0; i < 5; ++i) {
        s += a * noise(p);
        p = rm_rotate(p) * 2.03f + glm::vec2(17.1f);
        a *= 0.5f;
    }
    return s / 0.96875f;
}
inline float fbm3(glm::vec2 p) {
    return (0.5f * noise(p) + 0.25f * noise(rm_rotate(p) * 2.03f + glm::vec2(17.1f)) +
            0.125f * noise(p * 4.1f - glm::vec2(9.4f))) / 0.875f;
}
inline float continent(glm::vec2 mk) {
    glm::vec2 q = mk * 0.00042f;
    glm::vec2 bend = glm::vec2(noise(q * 2.3f + glm::vec2(5.2f)), noise(q * 2.3f + glm::vec2(91.7f))) - glm::vec2(0.5f);
    float v = fbm(q + bend * 0.55f + glm::vec2(21.3f)) - 0.5f;
    glm::vec2 wq = mk + (glm::vec2(fbm3(mk * 0.013f + glm::vec2(3.3f)), fbm3(mk * 0.013f + glm::vec2(47.9f))) - glm::vec2(0.5f)) *
                            glm::mix(3.0f, 60.0f, sstep(4.0f, 90.0f, glm::length(mk)));
    v += 0.20f * (1.0f - sstep(250.0f, 1600.0f, glm::length(wq - glm::vec2(-600.0f, 0.0f))));
    float dh = glm::length(mk);
    glm::vec2 wq2 = wq + (glm::vec2(fbm3(mk * 0.0035f + glm::vec2(13.1f)), fbm3(mk * 0.0035f + glm::vec2(71.4f))) - glm::vec2(0.5f)) *
                             170.0f * sstep(15.0f, 250.0f, dh);
    v = glm::mix(v, std::max(v, 0.16f), 1.0f - sstep(5.0f, 25.0f, dh));
    wq2.x += ((noise(mk * 0.9f + glm::vec2(31.7f)) - 0.5f) * 0.9f + (noise(mk * 2.7f + glm::vec2(5.1f)) - 0.5f) * 0.35f) * (1.0f - sstep(8.0f, 40.0f, dh));
    float rim = glm::mix(1.5f, 45.0f, sstep(10.0f, 300.0f, dh));
    v -= 0.46f * (1.0f - sstep(327.0f - rim, 327.0f + rim, glm::length(wq2 - glm::vec2(329.2f, 0.0f))));
    if (std::abs(v) > 0.37f) return v;
    v += (fbm(mk * 0.011f + glm::vec2(8.9f)) - 0.5f) * 0.16f;
    v += (fbm3(mk * 0.35f + glm::vec2(61.0f)) - 0.5f) * 0.05f;
    return v;
}
inline float ranges(glm::vec2 mk) {
    return sstep(0.60f, 0.78f, noise(mk * 0.08f + glm::vec2(7.3f)) * 0.65f + noise(mk * 0.21f + glm::vec2(3.1f)) * 0.35f);
}
// The river (shaders/world_map.glsl wm_river: keep in step).
const glm::vec2 kRiv[26] = { {62.0f, 36.0f}, {86.0f, 70.0f}, {100.0f, 112.0f}, {96.0f, 170.0f}, {74.0f, 228.0f}, {56.0f, 290.0f}, {22.0f, 350.0f}, {48.0f, 406.0f}, {150.0f, 412.0f}, {248.0f, 470.0f}, {344.0f, 536.0f}, {410.0f, 628.0f}, {504.0f, 634.0f}, {600.0f, 570.0f}, {696.0f, 536.0f}, {792.0f, 474.0f}, {888.0f, 456.0f}, {984.0f, 456.0f}, {1080.0f, 410.0f}, {1176.0f, 410.0f}, {1272.0f, 486.0f}, {1368.0f, 552.0f}, {1464.0f, 584.0f}, {1600.0f, 604.0f}, {1752.0f, 616.0f}, {1950.0f, 620.0f} };
const float kRivBox[5][4] = { {-69.0f, -89.0f, 225.0f, 415.0f}, {-103.0f, 165.0f, 469.0f, 661.0f}, {219.0f, 349.0f, 917.0f, 759.0f}, {667.0f, 285.0f, 1397.0f, 611.0f}, {1147.0f, 361.0f, 2075.0f, 745.0f} };
inline float river(float h, glm::vec2 wp, float wl) {
    if (wp.x < -85.0f || wp.x > 2095.0f || wp.y < -110.0f || wp.y > 780.0f) return h;
    const float bed = wl - 2.4f;
    if (h <= bed - 4.0f) return h;
    glm::vec2 q = wp + 12.0f * glm::vec2(std::sin(wp.y * 0.043f + 1.3f), std::sin(wp.x * 0.037f + 0.4f)) +
                       5.0f * glm::vec2(std::sin(wp.y * 0.113f), std::sin(wp.x * 0.097f + 2.0f));
    // (the pool under the stone bridge - as the shader)
    const float pool = 1.0f - sstep(22.0f, 55.0f, glm::distance(wp, glm::vec2(43.4f, 297.4f)));
    q -= pool * 6.0f * glm::vec2(0.719f, 0.695f);
    float d2 = 1e12f;
    for (int c = 0; c < 5; ++c) {
        const float* b = kRivBox[c];
        if (q.x < b[0] || q.y < b[1] || q.x > b[2] || q.y > b[3]) continue;
        for (int i = c * 5; i < c * 5 + 5; ++i) {
            glm::vec2 a = kRiv[i], ab = kRiv[i + 1] - a;
            glm::vec2 dv = q - (a + ab * std::clamp(glm::dot(q - a, ab) / glm::dot(ab, ab), 0.0f, 1.0f));
            d2 = std::min(d2, glm::dot(dv, dv));
        }
    }
    if (d2 > 120.0f * 120.0f) return h;
    const float e = std::max(std::sqrt(d2) - (7.0f + 9.0f * sstep(1450.0f, 1900.0f, wp.x) + 4.0f * pool), 0.0f);
    const float cap = bed + glm::mix(e * 0.45f + std::max(e - 8.0f, 0.0f) * 0.35f + std::max(e - 40.0f, 0.0f) * 0.5f,
                                     std::min(e, 6.0f) * 0.45f + std::max(e - 6.0f, 0.0f) * 1.3f, pool);
    const float t = std::clamp(0.5f + 0.125f * (cap - h), 0.0f, 1.0f);
    return glm::mix(cap, h, t) - 4.0f * t * (1.0f - t);
}
// Distance (m) from the edge of the river's channel at wp (negative inside it), as river()
// above measures it - the same wandering course, the same widening toward the sea. 1e9 when
// the river is more than 120 m away.
inline float river_edge(glm::vec2 wp) {
    if (wp.x < -85.0f || wp.x > 2095.0f || wp.y < -110.0f || wp.y > 780.0f) return 1e9f;
    glm::vec2 q = wp + 12.0f * glm::vec2(std::sin(wp.y * 0.043f + 1.3f), std::sin(wp.x * 0.037f + 0.4f)) +
                       5.0f * glm::vec2(std::sin(wp.y * 0.113f), std::sin(wp.x * 0.097f + 2.0f));
    const float pool = 1.0f - sstep(22.0f, 55.0f, glm::distance(wp, glm::vec2(43.4f, 297.4f)));
    q -= pool * 6.0f * glm::vec2(0.719f, 0.695f);
    float d2 = 1e12f;
    for (int c = 0; c < 5; ++c) {
        const float* b = kRivBox[c];
        if (q.x < b[0] || q.y < b[1] || q.x > b[2] || q.y > b[3]) continue;
        for (int i = c * 5; i < c * 5 + 5; ++i) {
            glm::vec2 a = kRiv[i], ab = kRiv[i + 1] - a;
            glm::vec2 dv = q - (a + ab * std::clamp(glm::dot(q - a, ab) / glm::dot(ab, ab), 0.0f, 1.0f));
            d2 = std::min(d2, glm::dot(dv, dv));
        }
    }
    if (d2 > 120.0f * 120.0f) return 1e9f;
    return std::sqrt(d2) - (7.0f + 9.0f * sstep(1450.0f, 1900.0f, wp.x) + 4.0f * pool);
}
inline float height_core(float h, float a, float w, float rng, float v, float wl) {
    float hl = 30.0f + (h - 30.0f) * glm::mix(0.42f, 1.12f, rng);
    hl = glm::mix(wl + 0.7f + std::max(hl - 24.0f, 0.0f) * 0.5f, hl, sstep(0.0f, 0.03f, v));
    hl += 9.0f * sstep(0.0f, 0.05f, v) * (1.0f - sstep(0.05f, 0.25f, v));
    // (GLSL smoothstep with edge0 > edge1 runs the other way: written out here)
    float bed = wl - 1.5f - 7.0f * a - 14.0f * (1.0f - sstep(-0.012f, 0.0f, v)) - 45.0f * (1.0f - sstep(-0.09f, -0.01f, v));
    return glm::mix(h, glm::mix(bed, hl, sstep(-0.012f, 0.012f, v)), w);
}
inline float height_land(float a, glm::vec2 wp, float wl) {
    float h = a * 120.0f;
    float w = sstep(1450.0f, 1900.0f, glm::length(wp));
    if (w <= 0.0f) return h;
    const float rng = ranges(wp * 0.001f);
    return height_core(h, a, w, rng, continent(wp * 0.001f), wl);
}
// The globe part (shaders/world_map.glsl wm_*_globe: keep in step). hp: home coordinates (m),
// G: the point on the sphere.
inline float continent_globe(glm::vec2 mk, const glm::dvec3& G) {
    const float wC = sstep(1800.0f, 3200.0f, glm::length(mk));
    float v = wC < 1.0f ? continent(mk) : 0.0f;
    if (wC > 0.0f) v = glm::mix(v, globe::continent3(G), wC);
    return v;
}
inline float ranges_globe(glm::vec2 mk, const glm::dvec3& G) {
    const float wC = sstep(1800.0f, 3200.0f, glm::length(mk));
    float v = wC < 1.0f ? ranges(mk) : 0.0f;
    if (wC > 0.0f) v = glm::mix(v, globe::ranges3(G), wC);
    return v;
}
inline float height_globe(float a, glm::vec2 hp, const glm::dvec3& G, float wl) {
    const glm::vec2 mk = hp * 0.001f;
    const float rng = ranges_globe(mk, G);
    return height_core(a * 120.0f, a, 1.0f, rng, continent_globe(mk, G), wl);
}
// The launch site's terrace (shaders/world_map.glsl wm_pad: keep in step).
inline float pad(float h, glm::vec2 wp) {
    const glm::vec2 d = wp - glm::vec2(kLaunchSiteX, kLaunchSiteZ);
    const float r2 = glm::dot(d, d);
    if (r2 > 220.0f * 220.0f) return h;
    return glm::mix(kLaunchSiteH, h, sstep(100.0f, 220.0f, std::sqrt(r2)));
}
inline float height(float a, glm::vec2 wp, float wl) { return river(pad(height_land(a, wp, wl), wp), wp, wl); }
// (shaders/world_map.glsl wm_forest: keep in step)
inline float forest(glm::vec2 wp, float h, float ny, float wl) {
    const float alt  = h - wl;
    const float band = sstep(2.0f, 6.0f, alt) * (1.0f - sstep(40.0f, 54.0f, alt)) * sstep(0.70f, 0.84f, ny);
    if (band <= 0.0f) return 0.0f;
    float m = noise(wp * 0.0011f + glm::vec2(3.7f)) * 0.55f + noise(wp * 0.0043f + glm::vec2(17.0f)) * 0.30f + noise(wp * 0.017f + glm::vec2(5.0f)) * 0.15f;
    m += 0.10f * (1.0f - sstep(4.0f, 25.0f, alt));
    m += (noise(wp * 0.00013f + glm::vec2(41.0f)) - 0.5f) * 0.35f;
    return band * sstep(0.37f, 0.53f, m) * sstep(150.0f, 230.0f, glm::distance(wp, glm::vec2(kLaunchSiteX, kLaunchSiteZ)));
}
inline float forest_globe(glm::vec2 hp, const glm::dvec3& G, float h, float ny, float wl) {
    const float wT = sstep(float(globe::kBlend0), float(globe::kBlend1), glm::length(hp));
    const float f2 = wT < 1.0f ? forest(hp, h, ny, wl) : 0.0f;
    if (wT <= 0.0f) return f2;
    const float alt  = h - wl;
    const float band = sstep(2.0f, 6.0f, alt) * (1.0f - sstep(40.0f, 54.0f, alt)) * sstep(0.70f, 0.84f, ny);
    float f3 = 0.0f;
    if (band > 0.0f) {
        const glm::dvec3 P = G * globe::kR;
        float m = globe::noise3(P * 0.0011 + glm::dvec3(3.7)) * 0.55f + globe::noise3(P * 0.0043 + glm::dvec3(17.0)) * 0.30f + globe::noise3(P * 0.017 + glm::dvec3(5.0)) * 0.15f;
        m += 0.10f * (1.0f - sstep(4.0f, 25.0f, alt));
        m += (globe::noise3(P * 0.00013 + glm::dvec3(41.0)) - 0.5f) * 0.35f;
        f3 = band * sstep(0.37f, 0.53f, m);
    }
    return glm::mix(f2, f3, wT);
}
}  // namespace wm

// A site's ground (engine/globe.h: Site): level round its centre, then sloping to the natural
// ground. `h`: the natural height at the planet point G. (shaders: rm_site_ground in
// terrain_rm_noise_road.glsl - keep in step.)
static float site_ground(float h, const glm::dvec3& G, float water_level) {
    for (const globe::SiteShape& s : globe::sites_nearby()) {       // (the few near the player, of thousands)
        const glm::dvec3 d = G - s.G;
        const double r2 = glm::dot(d, d) * globe::kR * globe::kR;           // (the chord: as good as the arc here)
        if (r2 > double(s.slope) * double(s.slope)) continue;
        const float w = 1.0f - wm::sstep(s.flat, s.slope, float(std::sqrt(r2)));
        return glm::mix(h, water_level + s.rise, w);
    }
    return h;
}
static bool site_clearing(const glm::dvec3& G) {
    for (const globe::SiteShape& s : globe::sites_nearby()) {
        const glm::dvec3 d = G - s.G;
        if (glm::dot(d, d) * globe::kR * globe::kR < double(s.clearing) * double(s.clearing)) return true;
    }
    return false;
}

float natural_height_impl(const glm::dvec3& G, float water_level, int octaves) {
    return wm::height_globe(globe::shape3(G * globe::kR, float(octaves)), glm::vec2(globe::home_az(G)), G, water_level);
}

// The flat formula's FBM at home coordinates given in doubles: per octave the lattice cell and
// the position in it, so that it is as exact 90 km out as at home (the shader does the same
// from a chart that is not home's: terrain_rm_noise_road.glsl rm_noised_i). Cells are wrapped
// to +-2^22 as there.
static float flat_fbm_d(glm::dvec2 hp, int octaves) {
    glm::dvec2 q = hp * double(kRmTerrainScale);
    float a = 0.0f, b = 1.0f;
    glm::vec2 d(0.0f);
    for (int i = 0; i < octaves; i++) {
        const glm::dvec2 cd = glm::floor(q);
        const glm::vec2 f = glm::vec2(q - cd);
        const glm::vec2 c = glm::vec2(cd - 8388608.0 * glm::floor(cd / 8388608.0 + 0.5));
        const glm::vec2 u  = f * f * (3.0f - 2.0f * f);
        const glm::vec2 du = 6.0f * f * (1.0f - f);
        const float h00 = rm_hash21(c), h10 = rm_hash21(c + glm::vec2(1, 0)), h01 = rm_hash21(c + glm::vec2(0, 1)), h11 = rm_hash21(c + glm::vec2(1, 1));
        const float n = h00 + (h10 - h00) * u.x + (h01 - h00) * u.y + (h00 - h10 - h01 + h11) * u.x * u.y;
        d += du * (glm::vec2(h10 - h00, h01 - h00) + (h00 - h10 - h01 + h11) * glm::vec2(u.y, u.x));
        a += b * n / (1.0f + glm::dot(d, d));
        b *= 0.5f;
        q = glm::dvec2(0.8 * q.x + 0.6 * q.y, -0.6 * q.x + 0.8 * q.y) * 2.0;
    }
    return a;
}
// A place given in the current chart, in home coordinates - when the chart is home's, or lies
// in the region where home's own rules hold (globe::Ring). False otherwise.
static bool home_coords(float x, float z, glm::dvec2& hp) {
    const globe::Chart& C = globe::current();
    if (C.home) { hp = glm::dvec2(x, z); return true; }
    if (!globe::ring_now().on) return false;
    hp = globe::home_az(globe::to_planet(C, x, z));
    return true;
}

float world_ground_height_impl(float x, float z, float water_level, int octaves) {
    if (!globe::current().home) {
        // Another chart (shaders: rm_shape / rm_world_h in terrain_rm_noise_road.glsl).
        const globe::Chart& C = globe::current();
        const glm::dvec3 G = globe::to_planet(C, x, z);
        const glm::dvec2 hp = globe::home_az(G);
        const double r = glm::length(hp);
        float a;
        if (globe::ring_now().on && r < globe::kBlend1) {
            a = flat_fbm_d(hp, octaves);                   // within reach of home's flat formula
            if (r > globe::kBlend0) a = glm::mix(a, globe::shape3(G * globe::kR, float(octaves)), wm::sstep(float(globe::kBlend0), float(globe::kBlend1), float(r)));
        } else {
            a = globe::shape3(G * globe::kR, float(octaves));
        }
        return site_ground(wm::height_globe(a, glm::vec2(hp), G, water_level), G, water_level);
    }
    glm::vec2 q = glm::vec2(x, z) * kRmTerrainScale;
    float a = 0.0f, b = 1.0f;
    glm::vec2 d(0.0f);
    for (int i = 0; i < octaves; i++) {
        glm::vec2 deriv;
        float n = rm_noised(q, deriv);
        d += deriv;
        a += b * n / (1.0f + glm::dot(d, d));
        b *= 0.5f;
        q = rm_rotate(q) * 2.0f;
    }
    // On the globe (shaders: rm_shape / rm_world_h in terrain_rm_noise_road.glsl).
    const globe::Chart& C = globe::current();
    if (C.home) {
        const float r = std::hypot(x, z);
        if (r > float(globe::kBlend0)) {
            const glm::dvec3 G = globe::to_planet(C, x, z);
            a = glm::mix(a, globe::shape3(G * globe::kR, float(octaves)), wm::sstep(float(globe::kBlend0), float(globe::kBlend1), r));
        }
        return wm::height(a, glm::vec2(x, z), water_level);
    }
    return 0.0f;        // (not reached: the other charts are handled at the top)
}

float rm_terrain(glm::vec2 worldXZ, const HeightmapParams& p, int octaves = 9) {

    glm::vec2 wp = worldXZ;
    glm::vec2 q  = worldXZ * kRmTerrainScale;
    float a = 0.0f, b = 1.0f;
    glm::vec2 d(0.0f);
    for (int i = 0; i < octaves; i++) {
        glm::vec2 deriv;
        float n = rm_noised(q, deriv);
        d += deriv;
        a += b * n / (1.0f + glm::dot(d, d));
        b *= 0.5f;
        q = rm_rotate(q) * 2.0f;
    }
    float h = wm::river(a * kRmTerrainHeight, wp, p.water_level);     // (as the shader: the home formula, with the river cut into it)

    // Soft plateau blend (mirrors shader's plateauWeight + mix).
    glm::vec2 ext = p.plateau_extent;
    glm::vec2 ab  = glm::abs(wp - p.plateau_center) - ext;
    float dout = std::max(std::max(ab.x, 0.0f), std::max(ab.y, 0.0f));
    float t = dout / kRmPlateauBlend;
    t = std::clamp(t, 0.0f, 1.0f);
    float w = 1.0f - (t * t * (3.0f - 2.0f * t));   // smoothstep
    return h * (1.0f - w) + p.plateau_height * w;
}

} // namespace

float world_forest(float x, float z, float h, float ny, float water_level) {
    const globe::Chart& C = globe::current();
    glm::dvec2 hpd;
    const bool at_home = home_coords(x, z, hpd);
    if (at_home && highway::ready() && highway::distance(float(hpd.x), float(hpd.y)) < highway::kVerge + 6.0f) return 0.0f;      // (no woods on a highway)
    if (at_home && glm::dot(hpd, hpd) < globe::kBlend0 * globe::kBlend0) {
        if (!C.home && site_clearing(globe::to_planet(C, x, z))) return 0.0f;
        return wm::forest(glm::vec2(hpd), h, ny, water_level);
    }
    const glm::dvec3 G = globe::to_planet(C, x, z);
    if (site_clearing(G)) return 0.0f;        // (a site's own ground: what grows there is placed by hand)
    return wm::forest_globe(C.home ? glm::vec2(x, z) : glm::vec2(globe::home_az(G)), G, h, ny, water_level);
}

float world_forest_flat(float hx, float hz, float h, float ny, float water_level) { return wm::forest(glm::vec2(hx, hz), h, ny, water_level); }

float world_highway_distance(float x, float z) {
    glm::dvec2 hp;
    return (highway::ready() && home_coords(x, z, hp)) ? highway::distance(float(hp.x), float(hp.y)) : 1e9f;
}

float world_river_cut(float h, float x, float z, float water_level) {
    return globe::current().home ? wm::river(h, glm::vec2(x, z), water_level) : h;
}

float world_river_distance(float x, float z) {
    return globe::current().home ? wm::river_edge(glm::vec2(x, z)) : 1e9f;
}

float world_natural_height(const glm::dvec3& G, float water_level, int octaves) {
    return natural_height_impl(G, water_level, octaves);
}

float world_ground_height_natural(float x, float z, float water_level, int octaves) {
    return world_ground_height_impl(x, z, water_level, octaves);
}
float world_continent_home(float x, float z) { return wm::continent(glm::vec2(x, z) * 0.001f); }

float world_ground_height(float x, float z, float water_level, int octaves) {
    const float h = world_ground_height_impl(x, z, water_level, octaves);
    // A highway's shelf (its lines are in home coordinates; shaders: rm_highway).
    glm::dvec2 hp;
    return (highway::ready() && home_coords(x, z, hp)) ? highway::ground(h, float(hp.x), float(hp.y)) : h;
}

float raymarch_fine_correction(const Heightmap& hm, const HeightmapParams& p,
                               float x, float z, int octaves) {
    // Exact high-octave FBM minus the bilinear 9-octave grid the CPU
    // heightmap stores (sculpt deltas cancel out of the difference).
    float fx = (x - hm.origin_x) / hm.cell, fz = (z - hm.origin_z) / hm.cell;
    float ix = std::floor(fx), iz = std::floor(fz);
    float tx = fx - ix, tz = fz - iz;
    auto g = [&](float gx, float gz) {
        return rm_terrain(glm::vec2(hm.origin_x + gx * hm.cell, hm.origin_z + gz * hm.cell), p, 9);
    };
    float bil = (g(ix, iz) * (1.0f - tx) + g(ix + 1.0f, iz) * tx) * (1.0f - tz) +
                (g(ix, iz + 1.0f) * (1.0f - tx) + g(ix + 1.0f, iz + 1.0f) * tx) * tz;
    return rm_terrain(glm::vec2(x, z), p, octaves) - bil;
}

Heightmap generate_heightmap_raymarch(const HeightmapParams& p) {
    Heightmap hm;
    hm.dim       = p.dim;
    hm.cell      = p.cell_size;
    hm.origin_x  = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    hm.origin_z  = -0.5f * static_cast<float>(p.dim) * p.cell_size;
    const int W = hm.width();
    const int H = hm.height();
    hm.heights.resize(static_cast<size_t>(W) * static_cast<size_t>(H));

    // Multi-thread row-strip generation. 9 octaves × 2048² в‰€ 38 M FBM
    // evals — single-threaded would be ~10 s, parallel ~2-3 s on 8+
    // cores.
    const int hw = std::max<int>(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    workers.reserve(hw);
    int rows = (H + hw - 1) / hw;
    for (int t = 0; t < hw; ++t) {
        int z0 = t * rows;
        int z1 = std::min(H, z0 + rows);
        if (z0 >= z1) break;
        workers.emplace_back([&, z0, z1] {
            for (int iz = z0; iz < z1; ++iz) {
                float wz = hm.origin_z + static_cast<float>(iz) * hm.cell;
                for (int ix = 0; ix < W; ++ix) {
                    float wx = hm.origin_x + static_cast<float>(ix) * hm.cell;
                    hm.heights[static_cast<size_t>(iz) *
                               static_cast<size_t>(W) +
                               static_cast<size_t>(ix)] =
                        rm_terrain(glm::vec2(wx, wz), p);
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    log::infof("[terrain] heightmap_raymarch %dx%d cells, %.0fm side, plateau h=%.1f",
               p.dim, p.dim, hm.side(), p.plateau_height);
    return hm;
}

namespace {

// Generate one chunk's vertex grid + smoothed per-vertex normals from
// `hm` covering [origin_ix, origin_ix + sample_dim) × [origin_iz, ...).
// Per-vertex normals come from the global heightmap gradient (central
// differences) so adjacent chunks agree on shared-edge normals — face-
// average accumulation only saw triangles WITHIN a chunk and produced
// faint shading seams at chunk boundaries.
// Skirt depth in metres. Each chunk-edge vertex has a "twin" vertex at the
// same XZ but lowered Y by this amount; the skirt strip connects them so
// that LOD-mismatch cracks at chunk boundaries are hidden by a vertical
// wall rather than producing a see-through gap. 8 m is enough to cover
// worst-case adjacent-cell height jumps on the steepest mountains.
constexpr float kTerrainSkirtDepth = 8.0f;

// densify >= 1: sub-divide each source heightmap cell within the chunk
// into `densify` quads per axis. sample_dim therefore corresponds to
// chunk_cells * densify + 1 vertices per side. The vertex at densified
// integer coords (ix, iz) maps to source heightmap fractional coords
// (origin_ix + ix/densify, origin_iz + iz/densify) and the height is
// bilinearly sampled. At densify=1 the math collapses to integer sample
// lookups (the original code path), and at densify=N the vertex grid is
// N x N times finer in BOTH axes than the source heightmap.
void gen_chunk_verts(const Heightmap& hm, int origin_ix, int origin_iz,
                     int sample_dim,
                     std::vector<Vertex>& verts,
                     glm::vec3& aabb_min, glm::vec3& aabb_max,
                     int densify = 1) {
    if (densify < 1) densify = 1;
    verts.clear();
    verts.reserve(static_cast<size_t>(sample_dim) * static_cast<size_t>(sample_dim) +
                   static_cast<size_t>(4) * static_cast<size_t>(sample_dim));
    aabb_min = glm::vec3( 1e9f);
    aabb_max = glm::vec3(-1e9f);
    // 1-cell normal gradient — gives per-vertex normals that match the
    // 1m triangle face at LOD 0 for crisp near detail. The earlier
    // 4-cell smoothed setting was a band-aid for LOD-mismatched far
    // chunks producing patchy slope-driven rock blend; that's now
    // handled by fading slope-rock at distance and by the cull-mask
    // trick keeping RT shadow rays off the terrain BLAS, so we can
    // get sharp 1m detail back without the artifacts returning.
    constexpr int kNormalRadius = 1;
    auto sample_normal = [&](int gx, int gz) {
        int xm = std::max(0, gx - kNormalRadius);
        int xp = std::min(hm.width()  - 1, gx + kNormalRadius);
        int zm = std::max(0, gz - kNormalRadius);
        int zp = std::min(hm.height() - 1, gz + kNormalRadius);
        float inv_2dx = 1.0f / (static_cast<float>(xp - xm) * hm.cell);
        float inv_2dz = 1.0f / (static_cast<float>(zp - zm) * hm.cell);
        float dhdx = (hm.at(xp, gz) - hm.at(xm, gz)) * inv_2dx;
        float dhdz = (hm.at(gx, zp) - hm.at(gx, zm)) * inv_2dz;
        return glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));
    };
    // Bilinear-sample the source heightmap at fractional source-cell
    // coords. At densify=1 fx,fz are always integers and the math
    // reduces to a clamped hm.at() lookup (no precision loss vs the
    // original code).
    const float inv_d = 1.0f / static_cast<float>(densify);
    auto sample_h = [&](int ix, int iz) -> float {
        float fx = static_cast<float>(origin_ix) + static_cast<float>(ix) * inv_d;
        float fz = static_cast<float>(origin_iz) + static_cast<float>(iz) * inv_d;
        int ax = static_cast<int>(std::floor(fx));
        int az = static_cast<int>(std::floor(fz));
        float tx = fx - static_cast<float>(ax);
        float tz = fz - static_cast<float>(az);
        int axc = std::clamp(ax,     0, hm.width()  - 1);
        int axn = std::clamp(ax + 1, 0, hm.width()  - 1);
        int azc = std::clamp(az,     0, hm.height() - 1);
        int azn = std::clamp(az + 1, 0, hm.height() - 1);
        float h00 = hm.at(axc, azc);
        float h10 = hm.at(axn, azc);
        float h01 = hm.at(axc, azn);
        float h11 = hm.at(axn, azn);
        float h0 = h00 * (1.0f - tx) + h10 * tx;
        float h1 = h01 * (1.0f - tx) + h11 * tx;
        return h0 * (1.0f - tz) + h1 * tz;
    };
    // Source-aligned integer (gx, gz) used for the normal sample. For
    // sub-cell densified verts we round to the nearest source cell;
    // gives the same heightmap-gradient normal as the surrounding
    // source verts which avoids visible faceted shading on the
    // upsampled grid (the surface itself is the bilinear interpolant,
    // so its slope is already smooth between source verts).
    auto nearest_src = [&](int ix, int iz, int& gx, int& gz) {
        float fx = static_cast<float>(origin_ix) + static_cast<float>(ix) * inv_d;
        float fz = static_cast<float>(origin_iz) + static_cast<float>(iz) * inv_d;
        gx = std::clamp(static_cast<int>(std::floor(fx + 0.5f)),
                         0, hm.width()  - 1);
        gz = std::clamp(static_cast<int>(std::floor(fz + 0.5f)),
                         0, hm.height() - 1);
    };
    // Interior (sample_dim × sample_dim) at densified resolution.
    for (int iz = 0; iz < sample_dim; ++iz) {
        for (int ix = 0; ix < sample_dim; ++ix) {
            float wx = hm.origin_x + (static_cast<float>(origin_ix) +
                                       static_cast<float>(ix) * inv_d) * hm.cell;
            float wz = hm.origin_z + (static_cast<float>(origin_iz) +
                                       static_cast<float>(iz) * inv_d) * hm.cell;
            float hy = sample_h(ix, iz);
            int gx, gz; nearest_src(ix, iz, gx, gz);
            glm::vec3 p(wx, hy, wz);
            glm::vec2 uv(p.x, p.z);
            verts.push_back({p, sample_normal(gx, gz), uv});
            aabb_min = glm::min(aabb_min, p);
            aabb_max = glm::max(aabb_max, p);
        }
    }
    // Skirt twins: 4 edges × sample_dim verts (the densified perimeter).
    // Each one is the interior edge vertex with Y dropped by
    // kTerrainSkirtDepth. Normal inherits from the corresponding
    // interior vertex (heightmap-gradient normal, mostly upward) so the
    // skirt shades like the terrain above it rather than reading as a
    // dark vertical wall — distant chunks at low LOD would otherwise
    // show their outward-facing skirts as visible dark faces against
    // the sky.
    auto add_skirt_edge = [&](int ix0, int iz0, int dx, int dz) {
        for (int k = 0; k < sample_dim; ++k) {
            int ix = ix0 + dx * k;
            int iz = iz0 + dz * k;
            float wx = hm.origin_x + (static_cast<float>(origin_ix) +
                                       static_cast<float>(ix) * inv_d) * hm.cell;
            float wz = hm.origin_z + (static_cast<float>(origin_iz) +
                                       static_cast<float>(iz) * inv_d) * hm.cell;
            float hy = sample_h(ix, iz);
            int gx, gz; nearest_src(ix, iz, gx, gz);
            glm::vec3 p(wx, hy - kTerrainSkirtDepth, wz);
            verts.push_back({p, sample_normal(gx, gz), glm::vec2(p.x, p.z)});
        }
    };
    add_skirt_edge(0,             0,             1, 0);  // top  (z=0)
    add_skirt_edge(sample_dim - 1, 0,             0, 1);  // right(x=N-1)
    add_skirt_edge(0,             sample_dim - 1, 1, 0);  // bot  (z=N-1)
    add_skirt_edge(0,             0,             0, 1);  // left (x=0)

    // Expand the chunk AABB to cover (a) the skirt twins (dropped by
    // kTerrainSkirtDepth) and (b) the worst-case tessellated vertex
    // displacement applied in terrain_tess.tese. The .tese is bumps-
    // up-only. Worst-case stack at the current UI sliders:
    //   - per-type ampM peaks at ~1.5 m on rock pixels when the
    //     "Rock relief" slider hits its max (1.0) -- see the
    //     `ampM += kRockReliefV() * (0.40 + wRockN * 1.10)` line.
    //   - the rocky-grass height-map contribution adds up to
    //     kRockyAmp == `terrain_disp_amp` (slider max 1.0 m) on top of
    //     that on grass-band pixels.
    // Total peak ~2.5 m up. 4 m of headroom keeps a comfortable margin
    // so a chunk near the frustum edge whose interior verts barely
    // fall outside the frustum still has its displaced verts inside
    // the AABB -- otherwise we frustum-cull it and drop a chunk-sized
    // patch of terrain at the screen edge. Was 3 m before -- tight at
    // max slider settings; bumped to 4 m for safety headroom.
    aabb_min.y -= kTerrainSkirtDepth;
    aabb_max.y += 4.0f;
}



// Full-LOD index buffer for a chunk of `sample_dim` × `sample_dim` verts.
void gen_full_indices(int sample_dim, std::vector<uint32_t>& out) {
    const int cells = sample_dim - 1;
    out.clear();
    out.reserve(static_cast<size_t>(cells) * static_cast<size_t>(cells) * 6u);
    for (int iz = 0; iz < cells; ++iz) {
        for (int ix = 0; ix < cells; ++ix) {
            uint32_t i00 = static_cast<uint32_t>(iz * sample_dim + ix);
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + static_cast<uint32_t>(sample_dim);
            uint32_t i11 = i01 + 1;
            out.push_back(i00); out.push_back(i11); out.push_back(i10);
            out.push_back(i00); out.push_back(i01); out.push_back(i11);
        }
    }
}


} // namespace

namespace {
// Inner ray-march for a single rectangular region. `out` points to the
// top-left of a w×h sub-image with the given row stride (in bytes).
// RG8 interleaved, 2 bytes per texel:
//   R: 0 = lit, 255 = in shadow
//   G: blocker distance along the sun ray, t / 400 m normalised to 0-255
//      (0 where lit). The shaders run PCSS on this — penumbra width
//      scales with blocker distance, so a tower top 60 m up casts a
//      wide soft shadow while a low wall stays tight and dark.
// The full-map and tile bake share this loop. `stride` is in TEXELS.
void bake_shadow_region(const Heightmap& hm, glm::vec3 L,
                        int ix0, int iz0, int w, int h,
                        uint8_t* out, int stride) {
    const float max_t = 400.0f;
    if (L.y < 0.05f) {
        for (int dz = 0; dz < h; ++dz) {
            uint8_t* row = out + static_cast<size_t>(dz) *
                                  static_cast<size_t>(stride) * 2;
            for (int dx = 0; dx < w; ++dx) {
                row[2 * dx + 0] = 255;
                row[2 * dx + 1] = 255;
            }
        }
        return;
    }
    const float step = hm.cell * 0.5f;
    const int   max_steps = static_cast<int>(max_t / step);
    const int W = hm.width();
    const int H = hm.height();
    for (int dz = 0; dz < h; ++dz) {
        int iz = iz0 + dz;
        uint8_t* row = out + static_cast<size_t>(dz) *
                              static_cast<size_t>(stride) * 2;
        if (iz < 0 || iz >= H) {
            std::fill_n(row, static_cast<size_t>(w) * 2, uint8_t(0));
            continue;
        }
        for (int dx = 0; dx < w; ++dx) {
            int ix = ix0 + dx;
            if (ix < 0 || ix >= W) { row[2 * dx] = 0; row[2 * dx + 1] = 0; continue; }
            float h0 = hm.at(ix, iz);
            glm::vec3 p0(hm.origin_x + static_cast<float>(ix) * hm.cell,
                         h0 + 0.20f,
                         hm.origin_z + static_cast<float>(iz) * hm.cell);
            bool shadowed = false;
            float t_hit = 0.0f;
            for (int s = 1; s <= max_steps; ++s) {
                float t = static_cast<float>(s) * step;
                glm::vec3 p = p0 + L * t;
                float h_at_p = hm.sample_world(p.x, p.z);
                if (p.y < h_at_p) { shadowed = true; t_hit = t; break; }
                if (p.y - h0 > 100.0f) break;
            }
            row[2 * dx + 0] = shadowed ? 255 : 0;
            row[2 * dx + 1] = shadowed
                ? static_cast<uint8_t>(std::min(t_hit * (255.0f / max_t), 255.0f))
                : 0;
        }
    }
}
} // namespace

void bake_heightmap_shadow_tile(const Heightmap& hm, glm::vec3 sun_dir,
                                int ix0, int iz0, int w, int h,
                                uint8_t* out_tile) {
    bake_shadow_region(hm, glm::normalize(sun_dir), ix0, iz0, w, h, out_tile, w);
}

void bake_heightmap_shadow_tile_ss(const Heightmap& hm, glm::vec3 sun_dir,
                                    int tx0, int tz0, int w, int h, int ss,
                                    uint8_t* out_tile) {
    if (ss < 1) ss = 1;
    glm::vec3 L = glm::normalize(sun_dir);
    const float max_t = 400.0f;
    if (L.y < 0.05f) {
        std::fill_n(out_tile,
                     static_cast<size_t>(w) * static_cast<size_t>(h) * 2,
                     uint8_t(255));
        return;
    }
    const float sub_cell = hm.cell / static_cast<float>(ss);
    const float step = sub_cell * 0.5f;
    const int   max_steps = static_cast<int>(max_t / step);
    for (int dz = 0; dz < h; ++dz) {
        uint8_t* row = out_tile + static_cast<size_t>(dz) *
                                   static_cast<size_t>(w) * 2;
        for (int dx = 0; dx < w; ++dx) {
            float wx = hm.origin_x + (static_cast<float>(tx0 + dx) + 0.5f) * sub_cell;
            float wz = hm.origin_z + (static_cast<float>(tz0 + dz) + 0.5f) * sub_cell;
            float h0 = hm.sample_world(wx, wz);
            glm::vec3 p0(wx, h0 + 0.20f, wz);
            bool shadowed = false;
            float t_hit = 0.0f;
            for (int s = 1; s <= max_steps; ++s) {
                float t = static_cast<float>(s) * step;
                glm::vec3 p = p0 + L * t;
                float h_at_p = hm.sample_world(p.x, p.z);
                if (p.y < h_at_p) { shadowed = true; t_hit = t; break; }
                if (p.y - h0 > 100.0f) break;
            }
            row[2 * dx + 0] = shadowed ? 255 : 0;
            row[2 * dx + 1] = shadowed
                ? static_cast<uint8_t>(std::min(t_hit * (255.0f / max_t), 255.0f))
                : 0;
        }
    }
}

std::vector<uint8_t> bake_heightmap_shadow(const Heightmap& hm,
                                           glm::vec3 sun_dir) {
    const int W = hm.width();
    const int H = hm.height();
    std::vector<uint8_t> out(static_cast<size_t>(W) * static_cast<size_t>(H) * 2, 0);
    glm::vec3 L = glm::normalize(sun_dir);

    // Multi-threaded — split rows across hardware threads. ~1024² cells
    // × ~80 marching steps drops from ~1s single-threaded to ~100-150 ms
    // with 8-12 worker threads. Used at level load; sun-change re-bakes
    // go through the progressive tile path instead.
    unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    int rows_per = (H + static_cast<int>(n_threads) - 1) /
                   static_cast<int>(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
        int z_lo = static_cast<int>(t) * rows_per;
        int z_hi = std::min(H, z_lo + rows_per);
        if (z_lo >= z_hi) break;
        workers.emplace_back([&, z_lo, z_hi]() {
            uint8_t* row0 = out.data() +
                static_cast<size_t>(z_lo) * static_cast<size_t>(W) * 2;
            bake_shadow_region(hm, L, 0, z_lo, W, z_hi - z_lo, row0, W);
        });
    }
    for (auto& th : workers) th.join();

    log::infof("[terrain] heightmap shadow baked: %dx%d (%u threads)",
               W, H, n_threads);
    return out;
}

Mesh build_terrain_mesh(VkDevice device, VmaAllocator alloc, VkQueue queue,
                        uint32_t queue_family, const Heightmap& hm) {
    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;
    glm::vec3 amin, amax;
    gen_chunk_verts(hm, 0, 0, hm.dim + 1, verts, amin, amax);
    gen_full_indices(hm.dim + 1, indices);
    Mesh m = create_mesh_from_data(device, alloc, queue, queue_family,
                                   verts.data(), static_cast<uint32_t>(verts.size()),
                                   indices.data(), static_cast<uint32_t>(indices.size()));
    log::infof("[terrain] mesh: %zu verts, %zu indices",
               verts.size(), indices.size());
    return m;
}

void rebuild_chunk_vertices(VkDevice device, VmaAllocator alloc, VkQueue queue,
                            uint32_t queue_family,
                            const Heightmap& hm, TerrainChunk& c) {
    std::vector<Vertex> verts;
    // Thread c.densify through so the brush rebuild matches the chunk's
    // CURRENT density. Without this, the VBO is regenerated at densify=1
    // even when the chunk's sample_dim was sized for densify>1; the
    // resulting VBO has the wrong number of vertices and gen_chunk_verts
    // bilinear-samples on the wrong source-grid spacing. The next densify
    // dirty-handler tick would eventually reupload at the correct density,
    // but during the brush stroke the chunk briefly renders with broken
    // topology. fake chunks built from default initialisation get
    // densify=1 from the struct default so the previous behaviour is
    // preserved for refresh_terrain_blas's whole-mesh "chunk".
    gen_chunk_verts(hm, c.origin_ix, c.origin_iz, c.sample_dim,
                    verts, c.aabb_min, c.aabb_max, c.densify);
    // Re-upload via stage-then-copy. We could keep a persistent host-
    // mapped staging buffer for sculpt strokes if this becomes a hot
    // path, but a single chunk is ~40KB so the one-shot path is fine.
    const VkDeviceSize size = sizeof(Vertex) *
                              static_cast<VkDeviceSize>(verts.size());
    VkBufferCreateInfo bci_stage{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr, .flags = 0, .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0, .pQueueFamilyIndices = nullptr,
    };
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    VkBuffer stage = VK_NULL_HANDLE;
    VmaAllocation stage_alloc = nullptr;
    vmaCreateBuffer(alloc, &bci_stage, &aci, &stage, &stage_alloc, nullptr);
    void* mapped = nullptr;
    vmaMapMemory(alloc, stage_alloc, &mapped);
    std::memcpy(mapped, verts.data(), static_cast<size_t>(size));
    vmaUnmapMemory(alloc, stage_alloc);

    VkCommandPoolCreateInfo pci{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = queue_family,
    };
    VkCommandPool pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &ai, &cb);
    VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    vkBeginCommandBuffer(cb, &bi);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cb, stage, c.mesh.vertex_buffer, 1, &region);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = nullptr,
        .waitSemaphoreCount = 0, .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1, .pCommandBuffers = &cb,
        .signalSemaphoreCount = 0, .pSignalSemaphores = nullptr,
    };
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkDestroyCommandPool(device, pool, nullptr);
    vmaDestroyBuffer(alloc, stage, stage_alloc);
}

} // namespace vibe

namespace vibe {
float road_edge_jitter(float x, float z) {
    // = terrain_raymarch.frag road weights: jit (keep in sync).
    glm::vec2 d;
    const glm::vec2 q(x, z);
    return (rm_noised(q * 1.7f, d) - 0.5f) * 0.45f + (rm_noised(q * 0.31f + glm::vec2(17.0f), d) - 0.5f) * 0.6f;
}
float raymarch_ground_relief(float x, float z, float h, float water_level) {
    float hw_keep = 1.0f;                // (a highway: none on the asphalt, growing over the verge - as the shader)
    glm::dvec2 hw_hp;
    if (highway::ready() && home_coords(x, z, hw_hp)) {
        const float t = std::clamp((highway::distance(float(hw_hp.x), float(hw_hp.y)) - 5.4f) / 3.0f, 0.0f, 1.0f);
        hw_keep = t * t * (3.0f - 2.0f * t);
        if (hw_keep <= 0.0f) return 0.0f;
    }
    // Mirror of terrain_raymarch.frag rm_ground_detail() (keep in sync).
    auto noise2 = [](glm::vec2 p) { glm::vec2 d; return rm_noised(p, d); };
    auto sstep = [](float a, float b, float v) { float t = std::clamp((v - a) / (b - a), 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); };
    const glm::vec2 wp(x, z);
    const float wn = noise2(wp * 0.21f);
    const glm::vec2 q = wp * 0.55f + glm::vec2(wn, 1.0f - wn) * 1.7f;
    const float g1 = 1.0f - sstep(0.0f, 0.22f, std::abs(noise2(q) * 2.0f - 1.0f));
    const float g2 = 1.0f - sstep(0.0f, 0.25f, std::abs(noise2(q * 2.7f + glm::vec2(3.1f)) * 2.0f - 1.0f));
    float d = 0.11f * (1.0f - g1) + 0.045f * (1.0f - g2) + 0.09f * 0.5f;
    const float sand = 1.0f - sstep(water_level + 0.4f, water_level + 2.5f, h);
    d += sand * 0.03f * (0.5f + 0.5f * std::sin(glm::dot(wp, glm::vec2(9.0f, 3.5f)) + 6.0f * wn));
    return d * hw_keep;
}
}  // namespace vibe
