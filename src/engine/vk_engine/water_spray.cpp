// Spray and mist where water is thrown into the air (after the reference game's spray: short-lived
// droplets and a soft sheet of mist lit by the sun, glowing when the sun is behind it):
//   * at the foot of the stream's falls - by the fall's height and the width pouring over it;
//   * where the current runs hard into a stone, a rock or a pier (faster than 1 m/s; the current is
//     the shallow-water solver's inside its window - water_flow_at);
//   * off the open sea's whitecaps in a strong wind (from 12 m/s), blown down the wind.
// CPU particles near the eye (within 90 m), drawn by render_fx: droplets as streaks (kind 4), the
// mist as soft billboards (kind 8, fx_ribbon.frag). Rates: engine/water_solver_cpu.h.
// Switch: VIBE_NO_SPRAY=1. Hook: VIBE_LOG_SPRAY=1 (once a second: emitters, droplets, mist puffs).
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/globe.h"
#include "engine/water_course.h"
#include "engine/water_solver_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace vibe {

namespace {
constexpr size_t kMaxDrops = 2500;
constexpr size_t kMaxMist = 120;
constexpr float  kReach = 90.0f;

struct Drop { glm::vec3 p, v; float t, life, size; };
struct Mist { glm::vec3 p, v; float t, life, r, a, seed; };
struct Spray {
    std::vector<Drop> drops;
    std::vector<Mist> mist;
    uint32_t rng = 0x9E3779B9u;
    float log_t = 0.0f;
    int emitters = 0;
    std::vector<glm::vec4> obs;
} g_sp;

float rnd() { uint32_t& s = g_sp.rng; s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float(s >> 8) * (1.0f / 16777216.0f); }
float rr(float a, float b) { return a + (b - a) * rnd(); }
// How many to emit this tick for a rate (/s): the whole part, and the rest by chance.
int count(float rate, float dt) { const float n = rate * dt; const int k = int(n); return k + (rnd() < n - float(k) ? 1 : 0); }
}  // namespace

void VulkanEngine::tick_water_spray(float dt) {
    static const bool off = std::getenv("VIBE_NO_SPRAY") != nullptr, lg = std::getenv("VIBE_LOG_SPRAY") != nullptr;
    Spray& S = g_sp;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    S.emitters = 0;
    if (!off && rt_.water_enabled && globe::current().home) {
        auto drop = [&](glm::vec3 p, glm::vec3 v, float size, float life) {
            if (S.drops.size() < kMaxDrops) S.drops.push_back(Drop{ p, v, 0.0f, life, size });
        };
        auto puff = [&](glm::vec3 p, glm::vec3 v, float r, float a, float life) {
            if (S.mist.size() < kMaxMist) S.mist.push_back(Mist{ p, v, 0.0f, life, r, a, rr(0.0f, 100.0f) });
        };
        // ---- the stream's falls: at the foot of each step within reach
        const wc::Course& st = wc::stream(rt_.water_level);
        static bool listed = false;
        if (lg && !listed && st.ok) {
            listed = true;
            for (size_t k = 0; k + 1 < st.pts.size(); ++k)
                if (st.pts[k + 1].w < 0.0f && st.pts[k].z - st.pts[k + 1].z >= 0.25f)
                    log::infof("[spray] the stream's step %zu: %.2f m at (%.1f, %.2f, %.1f)", k, double(st.pts[k].z - st.pts[k + 1].z), double(st.pts[k + 1].x), double(st.pts[k + 1].z), double(st.pts[k + 1].y));
        }
        for (size_t k = 0; st.ok && k + 1 < st.pts.size(); ++k) {
            const glm::vec4 A = st.pts[k], B = st.pts[k + 1];
            if (B.w >= 0.0f) continue;                       // (only a segment that ends in a step)
            const float fall = A.z - B.z;
            const glm::vec2 a(A), b(B);
            if (fall < 0.25f || glm::distance(b, glm::vec2(eye.x, eye.z)) > kReach) continue;
            const glm::vec2 d = glm::normalize(b - a), n(-d.y, d.x);
            const float hw = std::abs(B.w);
            const float rate = sw::spray_rate_fall(fall, 1.6f * hw);
            ++S.emitters;
            if (lg && S.log_t <= 0.0f) log::infof("[spray] a fall of %.2f m at (%.1f, %.2f, %.1f), flowing (%.2f, %.2f)", double(fall), double(b.x), double(B.z), double(b.y), double(d.x), double(d.y));
            for (int i = count(rate, dt); i > 0; --i) {
                const glm::vec2 q = b + n * rr(-0.8f, 0.8f) * hw + d * rr(-0.2f, 0.6f);
                const float y = water_surface_y(q.x, q.y);
                const float up = rr(1.2f, 3.4f) * std::sqrt(std::min(fall, 3.0f));
                const glm::vec2 h = d * rr(0.2f, 1.6f) + n * rr(-0.8f, 0.8f);
                drop(glm::vec3(q.x, y + 0.05f, q.y), glm::vec3(h.x, up, h.y), rr(0.0025f, 0.006f), rr(0.8f, 1.6f));
            }
            // (the mist: a soft sheet hanging over the foot, rising and drifting down the stream)
            // (2026-11-26, the owner's "big white fog" by the rapid: the puffs were 0.6 - 2 m, 10 - 20 %
            //  opaque, lived 3.5 s and grew 2.4x - a bank of fog over every step. Now only a real fall
            //  (>= 0.5 m) mists, the puffs are small, faint and gone in a second or two.)
            const int nm = fall >= 0.5f ? count(0.6f * std::min(fall, 3.0f) * std::clamp(hw, 0.5f, 2.0f), dt) : 0;
            for (int i = nm; i > 0; --i) {
                const glm::vec2 q = b + n * rr(-0.6f, 0.6f) * hw + d * rr(0.0f, 0.8f);
                puff(glm::vec3(q.x, water_surface_y(q.x, q.y) + rr(0.05f, 0.3f) * std::sqrt(fall), q.y),
                     glm::vec3(d.x * 0.4f, rr(0.10f, 0.30f), d.y * 0.4f), rr(0.3f, 0.6f) * std::sqrt(std::min(fall, 3.0f)), rr(0.04f, 0.08f), rr(1.0f, 1.8f));
            }
            // (2026-11-27, a WATERFALL - a fall over 1.2 m, its face near sheer: water_course.cpp - also
            //  breathes a veil of mist up its foot: puffs rising along the face, larger and a little longer
            //  lived, still faint)
            const int nw = fall >= 1.2f ? count(0.8f * std::min(fall, 4.0f) * std::clamp(hw, 0.5f, 2.0f), dt) : 0;
            for (int i = nw; i > 0; --i) {
                const glm::vec2 q = b + n * rr(-0.7f, 0.7f) * hw + d * rr(-0.3f, 1.2f);
                puff(glm::vec3(q.x, water_surface_y(q.x, q.y) + rr(0.1f, 0.6f) * fall, q.y),
                     glm::vec3(d.x * 0.5f, rr(0.25f, 0.6f), d.y * 0.5f), rr(0.5f, 1.0f) * std::sqrt(std::min(fall, 4.0f)), rr(0.03f, 0.06f), rr(1.4f, 2.4f));
            }
        }
        // ---- stones, rocks and piers the current runs into
        S.obs.clear();
        water_bed_obstacles(glm::vec2(eye.x, eye.z), 60.0f, S.obs);
        for (const glm::vec4& o : S.obs) {
            const float r = o.z;
            const glm::vec2 c(o.x, o.y);
            if (glm::distance(c, glm::vec2(eye.x, eye.z)) > 60.0f) continue;
            const glm::vec2 fl0 = water_course_flow(c.x, c.y);
            const float sp0 = glm::length(fl0);
            if (sp0 < 0.5f) continue;
            const glm::vec2 fd = fl0 / sp0;
            const glm::vec2 up = c - fd * (r + 0.35f);       // (just upstream of its face)
            if (!water_at(up.x, up.y)) continue;
            const float ys = water_surface_y(up.x, up.y);
            if (o.w < 5e4f && o.w + r < ys + 0.05f) continue;      // (a stone under the water throws no spray)
            const float sp = glm::length(water_flow_at(up.x - fd.x * 0.6f, up.y - fd.y * 0.6f));
            const float rate = sw::spray_rate_rock(sp, r);
            if (rate <= 0.0f) continue;
            ++S.emitters;
            const glm::vec2 n(-fd.y, fd.x);
            for (int i = count(rate, dt); i > 0; --i) {
                const glm::vec2 q = up + n * rr(-0.9f, 0.9f) * r;
                const glm::vec2 h = -fd * rr(0.0f, 0.6f) + n * rr(-0.7f, 0.7f);
                drop(glm::vec3(q.x, ys + 0.03f, q.y), glm::vec3(h.x, rr(0.8f, 2.2f) * std::sqrt(sp - 0.9f), h.y), rr(0.002f, 0.005f), rr(0.5f, 1.1f));
            }
            // (mist only where the water hits hard - over 2 m/s -, small, faint and brief)
            for (int i = sp > 2.0f ? count(0.15f * (sp - 2.0f) * std::clamp(r, 0.3f, 1.5f), dt) : 0; i > 0; --i)
                puff(glm::vec3(up.x, ys + 0.15f, up.y), glm::vec3(fd.x * 0.3f, 0.2f, fd.y * 0.3f), rr(0.2f, 0.4f) * std::clamp(r, 0.5f, 1.5f), rr(0.03f, 0.06f), rr(0.7f, 1.3f));
        }
        // ---- the open sea's whitecaps in a strong wind
        const glm::vec4 W = env_->water_wind;
        const float rate = sw::spray_rate_sea(W.z);
        if (rate > 0.0f) {
            const glm::vec2 wd(W.x, W.y);
            for (int i = count(rate, dt); i > 0; --i) {
                const float a = rr(0.0f, 6.2831853f), dd = std::sqrt(rnd()) * 70.0f + 6.0f;
                const glm::vec2 q(eye.x + std::cos(a) * dd, eye.z + std::sin(a) * dd);
                if (water_sea_weight(q.x, q.y) < 0.5f || !water_at(q.x, q.y)) continue;
                ++S.emitters;
                const float y = water_surface_y(q.x, q.y);
                for (int j = 0; j < 10; ++j) {
                    const glm::vec2 h = wd * (W.z * rr(0.25f, 0.5f)) + glm::vec2(rr(-1.0f, 1.0f), rr(-1.0f, 1.0f));
                    drop(glm::vec3(q.x + rr(-0.8f, 0.8f), y + 0.1f, q.y + rr(-0.8f, 0.8f)), glm::vec3(h.x, rr(1.0f, 3.0f), h.y), rr(0.002f, 0.005f), rr(0.6f, 1.4f));
                }
                puff(glm::vec3(q.x, y + 0.4f, q.y), glm::vec3(wd.x * W.z * 0.35f, 0.2f, wd.y * W.z * 0.35f), rr(1.2f, 2.6f), rr(0.08f, 0.16f), rr(1.5f, 2.8f));
            }
        }
    }
    // ---- the droplets fly and fall (air drag as the splashes' droplets: water_fx.cpp)
    for (size_t i = 0; i < S.drops.size();) {
        Drop& d = S.drops[i];
        d.t += dt;
        const float v = glm::length(d.v);
        d.v.y -= 9.81f * dt;
        d.v /= (1.0f + (0.0035f / std::max(d.size, 0.002f)) * v * dt);
        d.p += d.v * dt;
        bool dead = d.t > d.life || glm::distance(d.p, eye) > kReach + 20.0f;
        if (!dead && d.v.y < 0.0f && (d.p.y <= water_surface_y(d.p.x, d.p.z) || d.p.y <= sample_terrain_height(d.p.x, d.p.z))) dead = true;
        if (dead) { S.drops[i] = S.drops.back(); S.drops.pop_back(); } else ++i;
    }
    for (size_t i = 0; i < S.mist.size();) {
        Mist& m = S.mist[i];
        m.t += dt;
        m.p += m.v * dt;
        m.v *= 1.0f - 0.6f * dt;
        if (m.t > m.life || glm::distance(m.p, eye) > kReach + 30.0f) { S.mist[i] = S.mist.back(); S.mist.pop_back(); } else ++i;
    }
    if (lg && (S.log_t -= dt) <= 0.0f) {
        S.log_t = 1.0f;
        log::infof("[spray] %d emitters, %zu droplets, %zu mist puffs", S.emitters, S.drops.size(), S.mist.size());
    }
}

// The vertices for render_fx: droplets as short streaks (kind 4), mist as soft sheets (kind 8:
// tint = alpha, brightness, seed).
void VulkanEngine::water_spray_fx(std::vector<Vertex>& drops, std::vector<Vertex>& mist, glm::vec3 eye, glm::vec3 R, glm::vec3 U) {
    const Spray& S = g_sp;
    const float px_ang = 1.678f / std::max(1.0f, float(render_extent_.height));
    for (const Drop& d : S.drops) {
        const float dist = glm::distance(d.p, eye);
        const float sp = glm::length(d.v);
        const glm::vec3 dir = sp > 1e-3f ? d.v / sp : glm::vec3(0.0f, -1.0f, 0.0f);
        const float len = std::max(d.size * 2.2f, sp * 0.016f);
        const float w = std::max(d.size, dist * px_ang * 0.75f);
        float a = glm::clamp(d.size / w, 0.12f, 1.0f) * glm::clamp((d.life - d.t) * 3.0f, 0.0f, 1.0f) * std::min(1.0f, d.t * 20.0f + 0.3f);
        const glm::vec3 p0 = d.p - dir * len, p1 = d.p;
        glm::vec3 s0 = glm::cross(p1 - p0, eye - p0);
        const float l = glm::length(s0);
        s0 = l > 1e-6f ? s0 / l : U;
        const glm::vec3 a0 = s0 * (w * 0.6f), a1 = s0 * w;
        const Vertex q0{ p0 - a0, glm::vec3(1.0f), { a * 0.35f, -1.0f } }, q1{ p0 + a0, glm::vec3(1.0f), { a * 0.35f, 1.0f } };
        const Vertex q2{ p1 + a1, glm::vec3(1.0f), { a, 1.0f } }, q3{ p1 - a1, glm::vec3(1.0f), { a, -1.0f } };
        drops.insert(drops.end(), { q0, q1, q2, q0, q2, q3 });
    }
    for (const Mist& m : S.mist) {
        const float k = m.t / m.life;
        const float r = m.r * (1.0f + 0.8f * k);
        const float a = m.a * glm::smoothstep(0.0f, 0.15f, k) * (1.0f - glm::smoothstep(0.3f, 1.0f, k));
        const glm::vec3 Rr = R * r, Uu = U * r;
        const glm::vec3 tint(a, 1.0f, m.seed);
        const Vertex q0{ m.p - Rr - Uu, tint, { -1.0f, -1.0f } }, q1{ m.p + Rr - Uu, tint, { 1.0f, -1.0f } };
        const Vertex q2{ m.p + Rr + Uu, tint, { 1.0f, 1.0f } }, q3{ m.p - Rr + Uu, tint, { -1.0f, 1.0f } };
        mist.insert(mist.end(), { q0, q1, q2, q0, q2, q3 });
    }
}

}  // namespace vibe
