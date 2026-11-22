// Bridges: where the highway crosses water it used to run on fill - a dam across the river's
// mouth. Those stretches (highway::bridge: the line over ground that is under water) are now
// left open - the shelf is not made there, on the CPU or in the shaders - and a bridge stands
// in its place: a concrete box girder on piers, the road's asphalt and markings on its deck,
// a walkway and a steel railing on either side, lamp posts.
//
// Built once when the world is made (few, small, in home's chart: the whole course lies in
// it), as static meshes with the launch site's. Solid for a walker (boxes a metre long: the
// deck and both parapets) and for a vehicle (highway::deck in its ground patch, jeep.cpp).
#include "engine/vk_engine/internal.h"
#include "engine/highway.h"
#include "engine/terrain.h"
#include "game/level.h"

#include <algorithm>
#include <cmath>

namespace vibe {

void VulkanEngine::place_bridges() {
    const int nb = highway::bridge_count();
    if (nb <= 0) return;
    auto start = [](int tex, glm::vec3 tint, glm::vec4 emissive = glm::vec4(0.0f)) {
        game::LevelMesh m;
        m.tex_albedo = m.tex_normal = tex;
        m.tint = tint; m.fallback = tex >= 0 ? tint * 0.25f : tint;
        m.relief = 0.0f;
        m.emissive = glm::vec3(emissive); m.full_emissive = emissive.a > 0.5f;
        return m;
    };
    game::LevelMesh asphalt = start(kTexAsphalt, glm::vec3(0.60f, 0.62f, 0.66f));      // (the tone the road in the ground has)
    game::LevelMesh concrete = start(24, glm::vec3(1.04f, 1.03f, 1.0f));
    game::LevelMesh paint = start(-1, glm::vec3(0.80f, 0.80f, 0.77f));
    game::LevelMesh steel = start(-5, glm::vec3(0.34f, 0.35f, 0.37f));
    game::LevelMesh lamp = start(-1, glm::vec3(1.0f, 0.96f, 0.86f), glm::vec4(2.6f, 2.4f, 2.0f, 1.0f));

    // A quad a-b-c-d whose face looks the way `out` points. uv: (ua, va) .. (ub, vb) over it.
    auto quad = [](game::LevelMesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 out, glm::vec2 uv0, glm::vec2 uv1) {
        glm::vec3 n = glm::cross(b - a, d - a);
        if (glm::dot(n, n) < 1e-10f) return;
        n = glm::normalize(n);
        glm::vec2 ta(uv0.x, uv0.y), tb(uv0.x, uv1.y), tc(uv1.x, uv1.y), td(uv1.x, uv0.y);
        if (glm::dot(n, out) < 0.0f) { std::swap(b, d); std::swap(tb, td); n = -n; }
        const uint32_t i0 = uint32_t(m.verts.size());
        m.verts.push_back({ a, n, ta.x, ta.y }); m.verts.push_back({ b, n, tb.x, tb.y });
        m.verts.push_back({ c, n, tc.x, tc.y }); m.verts.push_back({ d, n, td.x, td.y });
        m.idx.insert(m.idx.end(), { i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3 });
    };
    // A box given by its centre and three half-axes.
    auto obox = [&](game::LevelMesh& m, glm::vec3 c, glm::vec3 ax, glm::vec3 ay, glm::vec3 az, float tile) {
        const glm::vec2 u0(0.0f), u1(glm::length(ax) * 2.0f / tile, glm::length(ay) * 2.0f / tile), u2(glm::length(az) * 2.0f / tile, glm::length(ay) * 2.0f / tile);
        quad(m, c + az - ax - ay, c + az - ax + ay, c + az + ax + ay, c + az + ax - ay, az, u0, glm::vec2(u1.y, u1.x));
        quad(m, c - az - ax - ay, c - az - ax + ay, c - az + ax + ay, c - az + ax - ay, -az, u0, glm::vec2(u1.y, u1.x));
        quad(m, c + ax - az - ay, c + ax - az + ay, c + ax + az + ay, c + ax + az - ay, ax, u0, glm::vec2(u2.y, u2.x));
        quad(m, c - ax - az - ay, c - ax - az + ay, c - ax + az + ay, c - ax + az - ay, -ax, u0, glm::vec2(u2.y, u2.x));
        quad(m, c + ay - ax - az, c + ay - ax + az, c + ay + ax + az, c + ay + ax - az, ay, u0, glm::vec2(u2.x, u1.x));
        quad(m, c - ay - ax - az, c - ay - ax + az, c - ay + ax + az, c - ay + ax - az, -ay, u0, glm::vec2(u2.x, u1.x));
    };
    struct St { glm::vec3 p; glm::vec3 t, r; float s; };
    const float cf = highway::cross_fall(highway::kPaved);            // the asphalt's edge lies this far under the crown
    const float kHalf = 6.6f;                                         // the deck's half width: asphalt 5.5, a walkway 1.1
    for (int b = 0; b < nb; ++b) {
        float s0 = 0.0f, s1 = 0.0f;
        highway::bridge(b, s0, s1);
        std::vector<St> st;
        // (The deck runs 6 m onto the land at either end, a hair above the road there: the
        //  road in the ground fades out over the last metres before the span, and the
        //  two never met - a gap of grass and a dark line across the lanes.)
        const float kOn = 6.0f;
        for (float s = s0 - kOn; s <= s1 + kOn + 0.01f; s += 2.0f) {
            glm::vec3 p; glm::vec2 d;
            if (!highway::point_along(s, p, d)) continue;
            p.y += 0.03f * glm::clamp(std::min(s - (s0 - kOn), (s1 + kOn) - s) / 2.0f, 0.35f, 1.0f);
            st.push_back({ p, glm::vec3(d.x, 0.0f, d.y), glm::vec3(-d.y, 0.0f, d.x), s });
        }
        if (st.size() < 3) continue;
        const glm::vec3 up(0.0f, 1.0f, 0.0f);
        // -- the deck, station to station: each profile segment a strip --
        struct Seg { float o0, y0, o1, y1; int mat; glm::vec2 out; float tile; };       // (offset across, height from the crown)
        std::vector<Seg> prof;
        for (float sd : { -1.0f, 1.0f }) {
            prof.push_back({ 0.0f, 0.0f, sd * highway::kPaved, -cf, 0, { 0.0f, 1.0f }, 3.0f });                       // asphalt
            prof.push_back({ sd * highway::kPaved, -cf, sd * highway::kPaved, -cf + 0.15f, 1, { -sd, 0.0f }, 4.0f }); // kerb
            prof.push_back({ sd * highway::kPaved, -cf + 0.15f, sd * kHalf, -cf + 0.15f, 1, { 0.0f, 1.0f }, 4.0f });  // walkway
            prof.push_back({ sd * kHalf, -cf + 0.15f, sd * kHalf, -0.62f, 1, { sd, 0.0f }, 4.0f });                   // fascia
            prof.push_back({ sd * kHalf, -0.62f, sd * 3.3f, -0.86f, 1, { 0.0f, -1.0f }, 4.0f });                      // cantilever soffit
            prof.push_back({ sd * 3.3f, -0.86f, sd * 2.4f, -2.25f, 1, { sd, -0.2f }, 4.0f });                         // the girder's web
            prof.push_back({ sd * 2.4f, -2.25f, 0.0f, -2.25f, 1, { 0.0f, -1.0f }, 4.0f });                            // its bottom
        }
        for (size_t i = 0; i + 1 < st.size(); ++i) {
            const St& A = st[i]; const St& B = st[i + 1];
            for (const Seg& g : prof) {
                game::LevelMesh& m = g.mat == 0 ? asphalt : concrete;
                const glm::vec3 a0 = A.p + A.r * g.o0 + up * g.y0, a1 = A.p + A.r * g.o1 + up * g.y1;
                const glm::vec3 b0 = B.p + B.r * g.o0 + up * g.y0, b1 = B.p + B.r * g.o1 + up * g.y1;
                const float w0 = (std::abs(g.o0) + std::abs(g.y0)) / g.tile, w1 = w0 + glm::distance(a0, a1) / g.tile;
                quad(m, a0, b0, b1, a1, A.r * g.out.x + up * g.out.y, glm::vec2(w0, A.s / g.tile), glm::vec2(w1, B.s / g.tile));
            }
            // markings: the edge lines, and the centre line's dashes (4 m in every 12)
            for (float sd : { -1.0f, 1.0f }) {
                const float o0 = sd * 5.02f, o1 = sd * 5.17f, y = -highway::cross_fall(5.1f) + 0.012f;
                quad(paint, A.p + A.r * o0 + up * y, B.p + B.r * o0 + up * y, B.p + B.r * o1 + up * y, A.p + A.r * o1 + up * y, up, glm::vec2(0.0f), glm::vec2(1.0f));
            }
            if (std::fmod(std::abs(A.s) + 6000.0f, 12.0f) < 4.0f)
                quad(paint, A.p - A.r * 0.075f + up * 0.012f, B.p - B.r * 0.075f + up * 0.012f, B.p + B.r * 0.075f + up * 0.012f, A.p + A.r * 0.075f + up * 0.012f, up, glm::vec2(0.0f), glm::vec2(1.0f));
            // the railing: a post every 2 m, a hand rail and two below it
            for (float sd : { -1.0f, 1.0f }) {
                const glm::vec3 a = A.p + A.r * (sd * (kHalf - 0.16f)) + up * (-cf + 0.15f), bb = B.p + B.r * (sd * (kHalf - 0.16f)) + up * (-cf + 0.15f);
                obox(steel, a + up * 0.55f, A.r * 0.035f, up * 0.55f, A.t * 0.035f, 1.0f);
                const glm::vec3 mid = 0.5f * (a + bb), half = 0.5f * (bb - a);
                const glm::vec3 across = glm::normalize(glm::cross(up, half));
                obox(steel, mid + up * 1.10f, across * 0.04f, up * 0.04f, half, 1.0f);
                obox(steel, mid + up * 0.72f, across * 0.02f, up * 0.02f, half, 1.0f);
                obox(steel, mid + up * 0.36f, across * 0.02f, up * 0.02f, half, 1.0f);
            }
        }
        // -- the ends of the girder, the abutments, the piers --
        for (const St* e : { &st[std::min<size_t>(3, st.size() - 1)], &st[st.size() - 1 - std::min<size_t>(3, st.size() - 1)] }) {
            const float gn = world_ground_height(e->p.x, e->p.z, rt_.water_level, 9);
            const float top = e->p.y - 0.62f, bot = std::min(gn, rt_.water_level) - 2.5f;
            obox(concrete, e->p + up * (0.5f * (top + bot) - e->p.y), e->r * (kHalf + 0.5f), up * (0.5f * (top - bot)), e->t * 0.9f, 4.0f);
        }
        const float len = st.back().s - st.front().s;
        const int   n_piers = std::max(0, int(std::round(len / 30.0f)) - 1);
        for (int k = 1; k <= n_piers; ++k) {
            const St& P = st[std::min(st.size() - 1, size_t(std::round(float(k) * float(st.size() - 1) / float(n_piers + 1))))];
            const float gn = world_ground_height(P.p.x, P.p.z, rt_.water_level, 9);
            const float top = P.p.y - 2.25f, bot = std::min(gn, rt_.water_level) - 3.0f;
            if (top - bot < 0.3f) continue;
            obox(concrete, P.p + up * (0.5f * (top + bot) - P.p.y), P.r * 2.0f, up * (0.5f * (top - bot)), P.t * 0.55f, 4.0f);
            obox(concrete, P.p + up * (top - 0.25f - P.p.y), P.r * 2.6f, up * 0.25f, P.t * 0.75f, 4.0f);             // its head
        }
        // -- lamp posts: every 28 m, on alternate sides, their arm over the road --
        int ln = 0;
        for (size_t i = 3; i + 3 < st.size(); i += 14, ++ln) {
            const St& P = st[i];
            const float sd = (ln & 1) ? 1.0f : -1.0f;
            const glm::vec3 foot = P.p + P.r * (sd * (kHalf - 0.45f)) + up * (-cf + 0.15f);
            obox(steel, foot + up * 4.0f, P.r * 0.07f, up * 4.0f, P.t * 0.07f, 1.0f);
            obox(steel, foot + up * 8.0f - P.r * (sd * 1.1f), P.r * 1.15f, up * 0.05f, P.t * 0.06f, 1.0f);
            obox(lamp, foot + up * 7.92f - P.r * (sd * 1.9f), P.r * 0.35f, up * 0.035f, P.t * 0.12f, 1.0f);
            lamp_lights_.push_back(foot + up * 7.6f - P.r * (sd * 1.9f));
        }
        // -- solid for a walker: the deck and the parapets, a metre at a time --
        for (float s = s0; s <= s1; s += 1.0f) {
            glm::vec3 p; glm::vec2 d;
            if (!highway::point_along(s, p, d)) continue;
            const glm::vec3 r(-d.y, 0.0f, d.x), t(d.x, 0.0f, d.y);
            glm::vec3 lo(1e9f), hi(-1e9f);
            for (float a : { -kHalf, kHalf }) for (float c : { -0.5f, 0.5f }) { const glm::vec3 q = p + r * a + t * c; lo = glm::min(lo, q); hi = glm::max(hi, q); }
            world_.aabbs.push_back({ glm::vec3(lo.x, p.y - 0.9f, lo.z), glm::vec3(hi.x, p.y - cf * 0.5f, hi.z) });
            for (float sd : { -1.0f, 1.0f }) {
                const glm::vec3 q = p + r * (sd * (kHalf - 0.16f));
                world_.aabbs.push_back({ glm::vec3(q.x - 0.5f, p.y, q.z - 0.5f), glm::vec3(q.x + 0.5f, p.y + 1.15f, q.z + 0.5f) });
            }
        }
        log::infof("[bridge] %d: %.0f m long, from (%.0f, %.1f, %.0f) to (%.0f, %.1f, %.0f), %d piers", b, double(len),
                   double(st.front().p.x), double(st.front().p.y), double(st.front().p.z), double(st.back().p.x), double(st.back().p.y), double(st.back().p.z), n_piers);
    }
    for (game::LevelMesh* m : { &asphalt, &concrete, &paint, &steel, &lamp }) {
        if (m->idx.empty()) continue;
        m->bmin = glm::vec3(1e9f); m->bmax = glm::vec3(-1e9f);
        for (const auto& v : m->verts) { m->bmin = glm::min(m->bmin, v.p); m->bmax = glm::max(m->bmax, v.p); }
        world_.meshes.push_back(std::move(*m));
    }
}

}  // namespace vibe
