// What stands beside a highway (engine/highway.h): a steel guard rail along the seaward side
// and reflector posts along the landward one - streamed.
//
// The road is cut into 300 m stretches. A stretch's objects are made from the line of the road
// when the camera comes within kLoad of it (meshes, their ray-tracing structures), at most one
// stretch a frame, and freed when the camera is kFree away - so what exists at any time is the
// kilometre or two around the player, however long the roads are. When the engine re-centres
// by whole tiles (engine/globe.h near_chart) the stretches are not made again: each part's
// transform takes the shift (CastleMeshGpu::xf). Like a site's content
// (sites.cpp) the freed parts wait a dozen frames before they are destroyed (frames in flight
// may still draw them) and nothing new is made while any wait. Home's chart only.
//
// To add another kind of roadside object: make it in build_stretch(). No colliders yet.
#include "engine/vk_engine/internal.h"
#include "engine/highway.h"
#include "engine/terrain.h"
#include "engine/globe.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>

namespace vibe {

namespace {

constexpr float  kLoad = 1500.0f, kFree = 1900.0f;      // from the camera to a stretch's middle (m)
constexpr size_t kStretch = 75;                          // stations a stretch (300 m)

void push(game::LevelMesh& m, glm::vec3 p, glm::vec3 n, float u, float v) {
    m.verts.push_back({ p, n, u, v });
    m.bmin = glm::min(m.bmin, p);
    m.bmax = glm::max(m.bmax, p);
}
// (front = cross(b - a, c - a) pointing along the normals: the winding the engine culls by)
void quad(game::LevelMesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 na, glm::vec3 nb, glm::vec3 nc, glm::vec3 nd) {
    const glm::vec3 g = glm::cross(b - a, c - a);
    const bool flip = glm::dot(g, na + nb + nc + nd) < 0.0f;
    const uint32_t base = uint32_t(m.verts.size());
    push(m, a, na, 0.0f, 0.0f); push(m, b, nb, 1.0f, 0.0f); push(m, c, nc, 1.0f, 1.0f); push(m, d, nd, 0.0f, 1.0f);
    if (flip) m.idx.insert(m.idx.end(), { base, base + 2, base + 1, base, base + 3, base + 2 });
    else      m.idx.insert(m.idx.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
}
void quad(game::LevelMesh& m, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 n) { quad(m, a, b, c, d, n, n, n, n); }
// An upright box: `c` on the ground under its middle, half-size hx along ax and hz along az
// (unit, level), from y0 to y1 above c.
void post(game::LevelMesh& m, glm::vec3 c, glm::vec3 ax, glm::vec3 az, float hx, float hz, float y0, float y1) {
    const glm::vec3 X = ax * hx, Z = az * hz, lo(0.0f, y0, 0.0f), hi(0.0f, y1, 0.0f);
    quad(m, c - X - Z + lo, c + X - Z + lo, c + X - Z + hi, c - X - Z + hi, -az);
    quad(m, c - X + Z + lo, c + X + Z + lo, c + X + Z + hi, c - X + Z + hi,  az);
    quad(m, c - X - Z + lo, c - X + Z + lo, c - X + Z + hi, c - X - Z + hi, -ax);
    quad(m, c + X - Z + lo, c + X + Z + lo, c + X + Z + hi, c + X - Z + hi,  ax);
    quad(m, c - X - Z + hi, c + X - Z + hi, c + X + Z + hi, c - X + Z + hi, glm::vec3(0.0f, 1.0f, 0.0f));
}
void finish(game::LevelMesh& m, int tex, glm::vec3 tint) {
    m.tex_albedo = m.tex_normal = tex;
    m.tint = tint; m.fallback = tint;
    m.relief = 0.0f;
}

// Stations 4 m apart along the coast road: where a post stands, which way the road runs and
// on which side the sea is. A few numbers a station, made once (a 26 km road: 6,700 of them);
// the objects themselves are made from these, a stretch at a time.
struct St { glm::vec3 p; glm::vec3 dir; glm::vec3 sea; };
std::vector<St> g_st;
bool g_st_made = false;

void make_stations() {
    g_st_made = true;
    g_st.clear();
    if (highway::line_count() == 0) return;
    const std::vector<glm::vec3>& P = highway::line_points(0);
    if (P.size() < 4) return;
    float next = 0.0f, walked = 0.0f;                               // metres along: the next station, the start of this segment
    for (size_t i = 0; i + 1 < P.size(); ++i) {
        const glm::vec3 a = P[i], b = P[i + 1];
        const float len = glm::distance(glm::vec2(a.x, a.z), glm::vec2(b.x, b.z));
        if (len < 1e-3f) continue;
        const glm::vec3 d = glm::normalize(glm::vec3(b.x - a.x, 0.0f, b.z - a.z));
        while (next < walked + len) {
            St s; s.p = glm::mix(a, b, (next - walked) / len); s.dir = d;
            const glm::vec3 n(-d.z, 0.0f, d.x);
            const glm::vec3 q0 = s.p + n * 40.0f, q1 = s.p - n * 40.0f;
            s.sea = world_continent_home(q0.x, q0.z) < world_continent_home(q1.x, q1.z) ? n : -n;
            g_st.push_back(s);
            next += 4.0f;
        }
        walked += len;
    }
    // The direction at a station: from the stations either side of it (a smooth turn through a bend).
    for (size_t i = 1; i + 1 < g_st.size(); ++i) {
        const glm::vec3 d(g_st[i + 1].p.x - g_st[i - 1].p.x, 0.0f, g_st[i + 1].p.z - g_st[i - 1].p.z);
        if (glm::dot(d, d) > 1e-6f) g_st[i].dir = glm::normalize(d);
    }
    // The side the sea is on must not flip from one post to the next (where a headland is cut
    // through): each station takes the side most of its neighbours have.
    std::vector<glm::vec3> sea(g_st.size());
    for (size_t i = 0; i < g_st.size(); ++i) {
        const glm::vec3 n(-g_st[i].dir.z, 0.0f, g_st[i].dir.x);
        float vote = 0.0f;
        for (int k = -40; k <= 40; ++k) {
            const size_t j = size_t(std::clamp(int(i) + k, 0, int(g_st.size()) - 1));
            vote += glm::dot(g_st[j].sea, n);
        }
        sea[i] = vote >= 0.0f ? n : -n;
    }
    for (size_t i = 0; i < g_st.size(); ++i) g_st[i].sea = sea[i];
}

size_t stretch_count() { return g_st.size() < 3 ? 0 : (g_st.size() - 2) / kStretch + 1; }
glm::vec3 stretch_middle(size_t c) { return g_st[std::min(c * kStretch + kStretch / 2, g_st.size() - 1)].p; }

// The objects of stretch c.
void build_stretch(size_t c, std::vector<game::LevelMesh>& out, const globe::Ring& rg) {
    // (in a chart that is not home's the stations are carried into it: engine/globe.h Ring)
    std::vector<St> carried;
    if (rg.on) {
        const size_t s0 = c * kStretch, s1 = std::min(s0 + kStretch + 3, g_st.size());
        carried = g_st;                                    // (indices as they are; only this stretch's are used)
        const glm::dmat2 Ji = glm::inverse(rg.J);
        for (size_t i = s0; i < s1; ++i) {
            St& s = carried[i];
            const glm::dvec2 p = rg.wp0 + Ji * (glm::dvec2(s.p.x, s.p.z) - rg.h0);
            const glm::dvec2 d = Ji * glm::dvec2(s.dir.x, s.dir.z), w = Ji * glm::dvec2(s.sea.x, s.sea.z);
            s.p = glm::vec3(float(p.x), s.p.y, float(p.y));
            s.dir = glm::normalize(glm::vec3(float(d.x), 0.0f, float(d.y)));
            s.sea = glm::normalize(glm::vec3(float(w.x), 0.0f, float(w.y)));
        }
    }
    const std::vector<St>& st = rg.on ? carried : g_st;
    const size_t c0 = c * kStretch, c1 = std::min(c0 + kStretch, st.size() - 1);       // (its last station is the next one's first)
    const float kRail = 6.35f;                       // the face of the rail, from the centre line
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    // The beam in section (toward the road, up): a W.
    const float prof[5][2] = { { 0.0f, 0.40f }, { 0.045f, 0.47f }, { 0.0f, 0.555f }, { 0.045f, 0.64f }, { 0.0f, 0.71f } };
    game::LevelMesh steel, white, black, amber;
    for (size_t i = c0; i < c1; ++i) {
        const St& s = st[i];
        // -- the post of the guard rail: a steel channel behind the beam --
        post(steel, s.p + s.sea * (kRail + 0.09f), s.dir, s.sea, 0.045f, 0.06f, -0.25f, 0.72f);
        // -- the beam, from every second station to the one 8 m on, seen from both sides --
        if (i % 2 == 0 && i + 2 < st.size()) {
            const St& e = st[i + 2];
            for (int k = 0; k < 4; ++k) {
                auto at = [&](const St& q, int j) { return q.p + q.sea * (kRail - prof[j][0]) + up * prof[j][1]; };
                auto nr = [&](const St& q) {                      // the normal of this strip, toward the road
                    const float dy = prof[k + 1][1] - prof[k][1], dx = prof[k + 1][0] - prof[k][0];
                    return glm::normalize(-q.sea * dy - up * dx);
                };
                const glm::vec3 a = at(s, k), b = at(e, k), cc = at(e, k + 1), d = at(s, k + 1);
                const glm::vec3 n0 = nr(s), n1 = nr(e);
                quad(steel, a, b, cc, d, n0, n1, n1, n0);
                quad(steel, a, b, cc, d, -n0, -n1, -n1, -n0);
            }
        }
        // -- a reflector post on the landward side, every 48 m --
        if (i % 12 == 6) {
            const glm::vec3 b = s.p - s.sea * 6.7f;
            post(white, b, s.dir, s.sea, 0.06f, 0.022f, -0.2f, 1.05f);
            post(black, b, s.dir, s.sea, 0.063f, 0.025f, 0.74f, 0.96f);
            const glm::vec3 r = b + up * 0.85f;
            for (float sg : { -1.0f, 1.0f }) {
                const glm::vec3 o = r + s.sea * (sg * 0.027f);
                quad(amber, o - s.dir * 0.035f - up * 0.07f, o + s.dir * 0.035f - up * 0.07f, o + s.dir * 0.035f + up * 0.07f, o - s.dir * 0.035f + up * 0.07f, s.sea * sg);
            }
        }
    }
    if (!steel.idx.empty()) { finish(steel, -5, glm::vec3(0.56f, 0.58f, 0.60f)); out.push_back(std::move(steel)); }
    if (!white.idx.empty()) { finish(white, -1, glm::vec3(0.80f, 0.80f, 0.78f)); out.push_back(std::move(white)); }
    if (!black.idx.empty()) { finish(black, -1, glm::vec3(0.02f, 0.02f, 0.022f)); out.push_back(std::move(black)); }
    if (!amber.idx.empty()) { finish(amber, -1, glm::vec3(0.95f, 0.50f, 0.06f)); out.push_back(std::move(amber)); }
}

}  // namespace

namespace { uint32_t g_serial = 0; }       // the chart the stretches are laid out in (globe::chart_serial)

// The engine's coordinates have just shifted by whole tiles (old = new + s): every part's
// transform takes the shift, now - a frame later the rail would be drawn 6 km off for a frame.
void VulkanEngine::hw_objects_shift(glm::dvec2 s) {
    const glm::vec3 sh(-float(s.x), 0.0f, -float(s.y));
    if (s != glm::dvec2(0.0)) {
        for (auto& ch : hw_chunks_)
            for (CastleMeshGpu& g : ch.second) {
                g.xf = glm::translate(glm::mat4(1.0f), sh) * g.xf;
                g.xf_prev = g.xf;
                g.bmin += sh; g.bmax += sh;
            }
        ++hw_gen_;                             // (the ray-tracing structure takes the new places)
    }
    g_serial = globe::chart_serial();
}

// Once a frame (game_loop.cpp): frees the stretches left behind, makes the nearest one that is
// wanted and missing.
void VulkanEngine::hw_objects_poll() {
    // What was freed waits out the frames in flight.
    for (auto it = hw_retired_.begin(); it != hw_retired_.end();) {
        if (--it->first > 0) { ++it; continue; }
        destroy_mesh_parts(it->second);
        it = hw_retired_.erase(it);
    }
    if (!highway::ready() || !scene_desc_set_) return;
    if (!g_st_made) make_stations();
    // VIBE_TEST_HIGHWAY_RIDE=<m/s>: carries the player along the coast road (tests the streaming).
    {
        static const char* ride = std::getenv("VIBE_TEST_HIGHWAY_RIDE");
        if (ride) {
            // ("<m/s>[,<start, metres along>]")
            static float along = [] { const char* c = std::strchr(std::getenv("VIBE_TEST_HIGHWAY_RIDE"), ','); return c ? float(std::atof(c + 1)) : -3000.0f; }();
            along += float(std::atof(ride)) / 60.0f;
            glm::vec3 rp; glm::vec2 rd;
            if (highway::point_along(along, rp, rd)) {
                // (the road's points are home coordinates: into the chart the engine is in)
                const globe::Chart& rc = globe::current();
                if (!rc.home) { static const globe::Chart hc; const glm::dvec2 q = globe::to_chart(rc, globe::to_planet(hc, double(rp.x), double(rp.z))); rp.x = float(q.x); rp.z = float(q.y); }
                player_.position = rp + glm::vec3(0.0f, 1.0f, 0.0f);
            }
            if (frame_number_ % 120 == 0) log::infof("highway ride: %.0f m along, %zu stretches loaded, %zu waiting to be freed", double(along), hw_chunks_.size(), hw_retired_.size());
        }
    }
    const globe::Ring rg = globe::ring_now();
    const bool here = (globe::current().home || rg.on) && !g_st.empty();
    glm::vec3 eye = player_.eye_position();
    if (rg.on) { const glm::dvec2 h = rg.to_home(glm::dvec2(eye.x, eye.z)); eye.x = float(h.x); eye.z = float(h.y); }      // (the stations are in home coordinates)
    // Another chart: other coordinates. A shift by whole tiles has been dealt with already, in
    // the step it happened in (hw_objects_shift, from globe_enter): the parts were moved. Any
    // other change (a site's chart, a high one): what was made is freed and made again.
    const bool rechart = g_serial != globe::chart_serial();
    g_serial = globe::chart_serial();
    // Free: everything when home is left, otherwise what is beyond kFree.
    for (size_t k = 0; k < hw_chunks_.size();) {
        const glm::vec3 m = stretch_middle(size_t(hw_chunks_[k].first));
        if (here && !rechart && glm::distance(glm::vec2(m.x, m.z), glm::vec2(eye.x, eye.z)) < kFree && eye.y - m.y < 2500.0f) { ++k; continue; }
        hw_retired_.emplace_back(12, std::move(hw_chunks_[k].second));
        hw_chunks_.erase(hw_chunks_.begin() + ptrdiff_t(k));
        ++hw_gen_;
    }
    if (!here || !hw_retired_.empty()) return;
    // Make: the nearest missing stretch within kLoad, one a frame.
    int want = -1; float best = kLoad;
    const size_t n = stretch_count();
    for (size_t c = 0; c < n; ++c) {
        const glm::vec3 m = stretch_middle(c);
        if (eye.y - m.y > 2000.0f) break;
        const float d = glm::distance(glm::vec2(m.x, m.z), glm::vec2(eye.x, eye.z));
        if (d >= best) continue;
        bool have = false;
        for (const auto& ch : hw_chunks_) if (ch.first == int(c)) { have = true; break; }
        if (!have) { want = int(c); best = d; }
    }
    if (want < 0) return;
    std::vector<game::LevelMesh> lms;
    build_stretch(size_t(want), lms, rg);
    // A stretch's vertices are kept relative to a point of its own (small numbers: exact to a
    // hundredth of a millimetre) and placed by the part's transform - always, so that a part
    // is drawn and traced the same way whether or not the engine has re-centred since.
    glm::vec3 org(0.0f);
    if (!lms.empty() && !lms[0].verts.empty()) org = glm::round(lms[0].verts[0].p);
    std::vector<CastleMeshGpu> parts;
    for (game::LevelMesh& lm : lms) {
        for (auto& v : lm.verts) v.p -= org;
        lm.bmin -= org; lm.bmax -= org;
        CastleMeshGpu g = make_mesh_part(lm);
        g.xf = glm::translate(glm::mat4(1.0f), org);
        g.xf_prev = g.xf;
        g.bmin += org; g.bmax += org;
        parts.push_back(std::move(g));
    }
    build_mesh_part_blas(parts);
    hw_chunks_.emplace_back(want, std::move(parts));
    ++hw_gen_;
    static const bool verbose = std::getenv("VIBE_LOG_HIGHWAY") != nullptr;
    if (verbose) log::infof("highway: stretch %d made (%zu loaded, %zu waiting to be freed)", want, hw_chunks_.size(), hw_retired_.size());
}

// The guard rail within `reach` of a place, as boxes for the physics (home's chart): one per
// 4 m of rail, turned along the road.
void VulkanEngine::hw_rail_boxes(glm::vec3 at, float reach, std::vector<PhysicsWorld::OrientedBox>& out) const {
    if (!g_st_made || g_st.size() < 2 || !globe::current().home) return;
    for (size_t i = 0; i + 1 < g_st.size(); ++i) {
        const St& s = g_st[i];
        if (std::abs(s.p.x - at.x) > reach || std::abs(s.p.z - at.z) > reach) continue;
        const St& e = g_st[i + 1];
        const glm::vec3 a = s.p + s.sea * 6.35f, b = e.p + e.sea * 6.35f;
        const glm::vec3 d = b - a;
        const float len = glm::length(glm::vec2(d.x, d.z));
        if (len < 0.5f || len > 6.0f) continue;
        // (a box's own z axis along the rail: turned about the vertical by atan2(dx, dz))
        out.push_back({ 0.5f * (a + b) + glm::vec3(0.0f, 0.40f, 0.0f), glm::vec3(0.06f, 0.36f, 0.5f * len + 0.05f), std::atan2(d.x, d.z) });
    }
}

void VulkanEngine::hw_objects_destroy() {

    for (auto& r : hw_retired_) destroy_mesh_parts(r.second);
    hw_retired_.clear();
    for (auto& c : hw_chunks_) destroy_mesh_parts(c.second);
    hw_chunks_.clear();
}

}  // namespace vibe
