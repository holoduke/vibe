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
#include "engine/vk_engine/lane_edge.h"
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
// A narrow road's dry-stone wall (lane_edge.h) is streamed with the rest: its stretches have the
// numbers kLaneId + c, and come and go nearer by (stones are small).
constexpr int    kLaneId = 1 << 20;
constexpr float  kLaneLoad = 520.0f, kLaneFree = 700.0f;

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
// Street lamps: along the road for 2 km either way from the launch site ("starbase"), one every
// 40 m on the landward side. `g_lamp[i]`: station i has one; `g_lamp_light`: where their lights
// are (home coordinates) - the scene's lamp lights take the nearest of them (hw_lamps_near).
std::vector<uint8_t> g_lamp;
std::vector<glm::vec3> g_lamp_light;
const float kLampSide = 7.5f, kLampHigh = 8.0f, kLampArm = 2.4f;
// The junction with the castle lane (highway.h Junction). `g_mouth[i]`: station i lies across
// the lane's mouth - 1: the lane leaves on the landward side (no lamp, no reflector post
// there), 2: on the seaward side (no guard rail there). The stop sign: on the lane's
// right-hand side for whoever drives up to the highway, just before the stop line.
std::vector<uint8_t> g_mouth;
int       g_jn_station = -1;                 // the station nearest the junction (its stretch makes the sign)
bool      g_sign_on = false;
glm::vec3 g_sign_base(0.0f);                 // the foot of the sign's post, home coordinates
const float kSignU = 7.5f, kSignSide = 3.3f, kSignHigh = 2.2f;      // metres from J along the lane; from the lane's axis; the sign's middle above the road

// A flat polygon (a fan about its first point), seen from the side n points to.
void fan(game::LevelMesh& m, const glm::vec3* p, int n_pts, glm::vec3 n) {
    const uint32_t base = uint32_t(m.verts.size());
    for (int i = 0; i < n_pts; ++i) push(m, p[i], n, 0.0f, 0.0f);
    const bool flip = glm::dot(glm::cross(p[1] - p[0], p[2] - p[0]), n) < 0.0f;
    for (int i = 1; i + 1 < n_pts; ++i) {
        if (flip) m.idx.insert(m.idx.end(), { base, base + uint32_t(i + 1), base + uint32_t(i) });
        else      m.idx.insert(m.idx.end(), { base, base + uint32_t(i), base + uint32_t(i + 1) });
    }
}
// A stop sign: `foot` on the ground, `face` the way its front looks (unit, level), `right` the
// reader's right (unit, level). An octagon 75 cm across - a white rim, a red field, the word in
// white (letters built from bars, five by seven) - on a galvanised post, its middle kSignHigh up.
void stop_sign(game::LevelMesh& steel, game::LevelMesh& red, game::LevelMesh& white, glm::vec3 foot, glm::vec3 face, glm::vec3 right) {
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    post(steel, foot, right, face, 0.03f, 0.03f, -0.4f, kSignHigh + 0.30f);
    const glm::vec3 c = foot + up * kSignHigh;
    const float R = 0.375f / 0.92388f;                          // (to a corner)
    glm::vec3 o[8], in[8], b[8];
    for (int k = 0; k < 8; ++k) {
        const float th = glm::radians(22.5f + 45.0f * float(k));
        const glm::vec3 r = right * std::cos(th) + up * std::sin(th);
        o[k] = c + face * 0.036f + r * R;
        in[k] = c + face * 0.036f + r * (R * 0.92f);
        b[k] = c + face * 0.031f + r * R;
    }
    for (int k = 0; k < 8; ++k) quad(white, o[k], o[(k + 1) % 8], in[(k + 1) % 8], in[k], face);       // the rim
    fan(red, in, 8, face);
    fan(steel, b, 8, -face);                                    // its back: bare sheet
    static const uint8_t kFont[4][7] = {
        { 0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E },           // S
        { 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04 },           // T
        { 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E },           // O
        { 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 },           // P
    };
    const float cw = 0.026f, ch = 0.25f / 7.0f;                 // a bar's cell: the letters are a third of the sign high
    const glm::vec3 lc = c + face * 0.041f;
    for (int l = 0; l < 4; ++l)
        for (int row = 0; row < 7; ++row)
            for (int col = 0; col < 5;) {
                if (!(kFont[l][row] & (0x10 >> col))) { ++col; continue; }
                int e = col; while (e < 5 && (kFont[l][row] & (0x10 >> e))) ++e;
                const float x0 = (-11.5f + 6.0f * float(l) + float(col)) * cw, x1 = (-11.5f + 6.0f * float(l) + float(e)) * cw;
                const float y1 = (3.5f - float(row)) * ch, y0 = y1 - ch;
                quad(white, lc + right * x0 + up * y0, lc + right * x1 + up * y0, lc + right * x1 + up * y1, lc + right * x0 + up * y1, face);
                col = e;
            }
}

void make_stations(glm::vec3 base) {
    g_st_made = true;
    g_st.clear();
    g_lamp.clear(); g_lamp_light.clear();
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
    // The junction: the stations across the lane's mouth (the lane and its two rounded corners,
    // and a little more), and the stop sign.
    g_mouth.assign(g_st.size(), 0);
    g_jn_station = -1; g_sign_on = false;
    if (const highway::Junction& jn = highway::junction(); jn.on) {
        const glm::vec2 side(-jn.w.y, jn.w.x);
        float bd = 1e30f;
        for (size_t i = 0; i < g_st.size(); ++i) {
            const glm::vec2 r = glm::vec2(g_st[i].p.x, g_st[i].p.z) - jn.j;
            const float d2 = glm::dot(r, r);
            if (d2 < bd) { bd = d2; g_jn_station = int(i); }
            if (std::abs(glm::dot(r, jn.w)) < 12.0f && std::abs(glm::dot(r, side)) < highway::kPaved * jn.k + highway::kJnFlare + 2.5f)
                g_mouth[i] = glm::dot(glm::vec2(g_st[i].sea.x, g_st[i].sea.z), jn.w) > 0.0f ? 2 : 1;
        }
        const glm::vec2 right(jn.w.y, -jn.w.x);                  // (of a driver coming up the lane to the highway)
        const glm::vec2 f = jn.j + jn.w * kSignU + right * kSignSide;
        g_sign_base = glm::vec3(f.x, highway::junction_lane_h(kSignU) - 0.05f, f.y);
        g_sign_on = g_jn_station >= 0;
    }
    // The lamps: the station nearest the launch site, and 500 stations (2 km) either way along
    // the road (the course is a ring: counted round it).
    g_lamp.assign(g_st.size(), 0);
    if (g_st.size() > 20) {
        size_t k0 = 0; float bd = 1e30f;
        for (size_t i = 0; i < g_st.size(); ++i) {
            const float d = glm::distance(glm::vec2(g_st[i].p.x, g_st[i].p.z), glm::vec2(base.x, base.z));
            if (d < bd) { bd = d; k0 = i; }
        }
        const size_t n = g_st.size();
        for (size_t i0 = 5; i0 < n; i0 += 10) {
            size_t i = i0;
            while (i + 1 < n && i < i0 + 5 && g_mouth[i] == 1) ++i;      // (not in the mouth of the castle lane: beside it)
            const size_t d = i > k0 ? i - k0 : k0 - i;
            if (std::min(d, n - d) > 500) continue;
            // (not in the mouth of the service station's way in: terrain.h kStation*)
            if (std::abs(g_st[i].p.z - kStationZ) < 19.0f && std::abs(g_st[i].p.x - (kStationX + 42.7f)) < 14.0f) continue;
            if (g_mouth[i] == 1) continue;
            g_lamp[i] = 1;
            g_lamp_light.push_back(g_st[i].p - g_st[i].sea * (kLampSide - kLampArm) + glm::vec3(0.0f, kLampHigh - 0.25f, 0.0f));
            // (VIBE_LOG_LAMPS=1: where they stand and which way the road runs there - for test views)
            if (static const bool lg = std::getenv("VIBE_LOG_LAMPS") != nullptr; lg && g_lamp_light.size() % 10 == 1)
                log::infof("[lamps] street lamp %zu: road at (%.1f, %.1f, %.1f), running (%.2f, %.2f)", g_lamp_light.size() - 1,
                           double(g_st[i].p.x), double(g_st[i].p.y), double(g_st[i].p.z), double(g_st[i].dir.x), double(g_st[i].dir.z));
        }
    }
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
    game::LevelMesh steel, white, black, amber, pole, lamp, red;
    auto mouth = [&](size_t i) { return i < g_mouth.size() ? g_mouth[i] : uint8_t(0); };
    // -- the stop sign at the castle lane's mouth (made with the stretch its junction lies in) --
    if (g_sign_on && size_t(g_jn_station) >= c0 && size_t(g_jn_station) < c1) {
        const highway::Junction& jn = highway::junction();
        glm::vec3 foot = g_sign_base, face(jn.w.x, 0.0f, jn.w.y), right(jn.w.y, 0.0f, -jn.w.x);
        if (rg.on) {
            const glm::dmat2 Ji = glm::inverse(rg.J);
            const glm::dvec2 p = rg.wp0 + Ji * (glm::dvec2(foot.x, foot.z) - rg.h0);
            const glm::dvec2 a = Ji * glm::dvec2(face.x, face.z), b = Ji * glm::dvec2(right.x, right.z);
            foot = glm::vec3(float(p.x), foot.y, float(p.y));
            face = glm::normalize(glm::vec3(float(a.x), 0.0f, float(a.y)));
            right = glm::normalize(glm::vec3(float(b.x), 0.0f, float(b.y)));
        }
        stop_sign(steel, red, white, foot, face, right);
    }
    for (size_t i = c0; i < c1; ++i) {
        const St& s = st[i];
        // -- the post of the guard rail: a steel channel behind the beam (not across a side road's mouth) --
        if (mouth(i) != 2) post(steel, s.p + s.sea * (kRail + 0.09f), s.dir, s.sea, 0.045f, 0.06f, -0.25f, 0.72f);
        // -- the beam, from every second station to the one 8 m on, seen from both sides --
        if (i % 2 == 0 && i + 2 < st.size() && mouth(i) != 2 && mouth(i + 1) != 2 && mouth(i + 2) != 2) {
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
        // -- a street lamp on the landward side (near the launch site only: g_lamp) --
        if (i < g_lamp.size() && g_lamp[i]) {
            const glm::vec3 b = s.p - s.sea * kLampSide;
            post(pole, b, s.dir, s.sea, 0.07f, 0.07f, -0.3f, kLampHigh);
            // the arm, out over the road: a flat bar, seen from above and below
            const glm::vec3 a0 = b + up * kLampHigh, a1 = a0 + s.sea * kLampArm;
            for (float sg : { 1.0f, -1.0f })
                quad(pole, a0 - s.dir * 0.05f + up * (0.04f * sg), a1 - s.dir * 0.05f + up * (0.04f * sg), a1 + s.dir * 0.05f + up * (0.04f * sg), a0 + s.dir * 0.05f + up * (0.04f * sg), up * sg);
            for (float sg : { 1.0f, -1.0f })
                quad(pole, a0 + s.dir * (0.05f * sg) - up * 0.04f, a1 + s.dir * (0.05f * sg) - up * 0.04f, a1 + s.dir * (0.05f * sg) + up * 0.04f, a0 + s.dir * (0.05f * sg) + up * 0.04f, s.dir * sg);
            // the lamp's head at the end of it: a lit panel facing down, a housing over it
            const glm::vec3 h = a1 - up * 0.06f;
            quad(lamp, h - s.sea * 0.55f - s.dir * 0.16f, h - s.sea * 0.55f + s.dir * 0.16f, h + s.sea * 0.10f + s.dir * 0.16f, h + s.sea * 0.10f - s.dir * 0.16f, -up);
            quad(pole, h - s.sea * 0.58f - s.dir * 0.19f + up * 0.10f, h + s.sea * 0.13f - s.dir * 0.19f + up * 0.10f, h + s.sea * 0.13f + s.dir * 0.19f + up * 0.10f, h - s.sea * 0.58f + s.dir * 0.19f + up * 0.10f, up);
        }
        // -- a reflector post on the landward side, every 48 m --
        if (i % 12 == 6 && mouth(i) != 1) {
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
    if (!pole.idx.empty()) { finish(pole, -5, glm::vec3(0.34f, 0.36f, 0.38f)); out.push_back(std::move(pole)); }
    if (!red.idx.empty()) { finish(red, -1, glm::vec3(0.62f, 0.035f, 0.03f)); out.push_back(std::move(red)); }
    if (!lamp.idx.empty()) {
        finish(lamp, -1, glm::vec3(1.0f, 0.96f, 0.86f));
        lamp.emissive = glm::vec3(3.4f, 3.2f, 2.7f); lamp.full_emissive = true;
        out.push_back(std::move(lamp));
    }
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
    if (!g_st_made) {
        if (!launch_site_placed_) return;            // (the lamps stand by the launch site: it has to be there)
        make_stations(launch_pad_);
    }
    // The ground as it is drawn: the home tile's own heights, the formula with the road's shelf beyond it.
    const lane_edge::Ground lane_ground = [this](float x, float z) {
        return (!terrain_data_.heights.empty() && std::abs(x) < 1020.0f && std::abs(z) < 1020.0f) ? sample_terrain_height(x, z) : world_ground_height(x, z, rt_.water_level, 9);
    };
    if (!lane_edge::made()) {
        // (no wall across the castle car park's entrance: terrain.h kCastlePark)
        lane_edge::make(lane_ground, [](float x, float z) { return x > kCastlePark[0] - 4.0f && x < kCastlePark[2] + 4.0f && z > kCastlePark[1] - 4.0f && z < kCastlePark[3] + 4.0f; });
        lane_rock_cols_ = lane_edge::walker_boxes();          // (solid for a walker: props.cpp)
    }
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
    const bool lane_here = globe::current().home && lane_edge::stretch_count() > 0;      // (the walls: home's chart only)
    glm::vec3 eye = player_.eye_position();
    if (rg.on) { const glm::dvec2 h = rg.to_home(glm::dvec2(eye.x, eye.z)); eye.x = float(h.x); eye.z = float(h.y); }      // (the stations are in home coordinates)
    // Another chart: other coordinates. A shift by whole tiles has been dealt with already, in
    // the step it happened in (hw_objects_shift, from globe_enter): the parts were moved. Any
    // other change (a site's chart, a high one): what was made is freed and made again.
    const bool rechart = g_serial != globe::chart_serial();
    g_serial = globe::chart_serial();
    // Free: everything when home is left, otherwise what is beyond kFree.
    for (size_t k = 0; k < hw_chunks_.size();) {
        const int id = hw_chunks_[k].first;
        const bool lane = id >= kLaneId;
        const glm::vec3 m = lane ? lane_edge::stretch_middle(size_t(id - kLaneId)) : stretch_middle(size_t(id));
        if ((lane ? lane_here : here) && !rechart && glm::distance(glm::vec2(m.x, m.z), glm::vec2(eye.x, eye.z)) < (lane ? kLaneFree : kFree) && eye.y - m.y < 2500.0f) { ++k; continue; }
        hw_retired_.emplace_back(12, std::move(hw_chunks_[k].second));
        hw_chunks_.erase(hw_chunks_.begin() + ptrdiff_t(k));
        ++hw_gen_;
    }
    if ((!here && !lane_here) || !hw_retired_.empty()) return;
    // Make: the nearest missing stretch within kLoad, one a frame.
    int want = -1; float best = kLoad;
    const size_t n = here ? stretch_count() : 0;
    // (a wall's stretch first: it is only wanted when the player is near it)
    for (size_t c = 0; lane_here && c < lane_edge::stretch_count(); ++c) {
        if (!lane_edge::stretch_has_wall(c)) continue;
        const glm::vec3 m = lane_edge::stretch_middle(c);
        const float d = glm::distance(glm::vec2(m.x, m.z), glm::vec2(eye.x, eye.z));
        if (d >= std::min(best, kLaneLoad) || eye.y - m.y > 600.0f) continue;
        bool have = false;
        for (const auto& ch : hw_chunks_) if (ch.first == kLaneId + int(c)) { have = true; break; }
        if (!have) { want = kLaneId + int(c); best = d; }
    }
    for (size_t c = 0; want < 0 && c < n; ++c) {
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
    if (want >= kLaneId) lane_edge::build_stretch(size_t(want - kLaneId), lane_ground, lms);
    else build_stretch(size_t(want), lms, rg);
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
    if (!globe::current().home) return;
    lane_edge::car_boxes(at, reach, out);          // (a narrow road's dry-stone wall: lane_edge.cpp)
    if (!g_st_made || g_st.size() < 2) return;
    // (the stop sign's post: thicker than it is, as the rail - a fast car's corner must not pass it between two steps)
    if (g_sign_on && std::abs(g_sign_base.x - at.x) < reach && std::abs(g_sign_base.z - at.z) < reach)
        out.push_back({ g_sign_base + glm::vec3(0.0f, 1.25f, 0.0f), glm::vec3(0.09f, 1.30f, 0.09f), 0.0f });
    for (size_t i = 0; i + 1 < g_st.size(); ++i) {
        const St& s = g_st[i];
        if (std::abs(s.p.x - at.x) > reach || std::abs(s.p.z - at.z) > reach) continue;
        if (g_mouth[i] == 2 || g_mouth[i + 1] == 2) continue;      // (no rail across a side road's mouth)
        const St& e = g_st[i + 1];
        const glm::vec3 a = s.p + s.sea * 6.35f, b = e.p + e.sea * 6.35f;
        const glm::vec3 d = b - a;
        const float len = glm::length(glm::vec2(d.x, d.z));
        if (len < 0.5f || len > 6.0f) continue;
        // (a box's own z axis along the rail: turned about the vertical by atan2(dx, dz).
        //  For a car it is 40 cm thick, grown away from the road - the face toward the road is
        //  where the rail's is: a sheet of 12 cm let a fast car's corner through between two
        //  physics steps, and what got through was held. And a little higher than the rail, so
        //  a body leaning over it meets a face, not the top edge. Each box runs a hand's width
        //  into the next: no gap at a bend.)
        const glm::vec3 out_dir = glm::normalize(s.sea + e.sea);
        out.push_back({ 0.5f * (a + b) + glm::vec3(0.0f, 0.50f, 0.0f) + out_dir * 0.14f, glm::vec3(0.20f, 0.46f, 0.5f * len + 0.10f), std::atan2(d.x, d.z) });
    }
}

// The street lamps' poles near `at`, as boxes (for bolts: combat.cpp).
void VulkanEngine::hw_pole_boxes(glm::vec3 at, float reach, std::vector<PhysicsWorld::OrientedBox>& out) const {
    if (!globe::current().home) return;
    if (g_sign_on && std::abs(g_sign_base.x - at.x) < reach && std::abs(g_sign_base.z - at.z) < reach)       // (the stop sign's)
        out.push_back({ g_sign_base + glm::vec3(0.0f, 1.25f, 0.0f), glm::vec3(0.05f, 1.30f, 0.05f), 0.0f });
    for (size_t i = 0; i < g_lamp.size() && i < g_st.size(); ++i) {
        if (!g_lamp[i]) continue;
        const glm::vec3 b = g_st[i].p - g_st[i].sea * kLampSide;
        if (std::abs(b.x - at.x) > reach || std::abs(b.z - at.z) > reach) continue;
        out.push_back({ b + glm::vec3(0.0f, 0.5f * kLampHigh, 0.0f), glm::vec3(0.09f, 0.5f * kLampHigh, 0.09f), 0.0f });
    }
}

// The road's street lamps within `reach` of the eye, for the scene's lamp lights (descriptors.cpp).
void VulkanEngine::hw_lamps_near(glm::vec3 eye, float reach, std::vector<glm::vec3>& out) const {
    if (!globe::current().home) return;
    for (const glm::vec3& l : g_lamp_light)
        if (std::abs(l.x - eye.x) < reach && std::abs(l.z - eye.z) < reach) out.push_back(l);
}

void VulkanEngine::hw_objects_destroy() {

    for (auto& r : hw_retired_) destroy_mesh_parts(r.second);
    hw_retired_.clear();
    for (auto& c : hw_chunks_) destroy_mesh_parts(c.second);
    hw_chunks_.clear();
}

}  // namespace vibe
