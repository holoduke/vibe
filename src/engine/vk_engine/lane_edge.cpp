// The edge of a narrow road at a drop: a low dry-stone wall (see lane_edge.h).
//
// WHERE. A station every 2 m of the line. On each side: the fall of the drawn ground from the
// road to the lowest point within 4 m beyond the verge. More than kDrop of fall = a drop, and a
// wall is wanted there; nowhere else. Not on a bridge (it has its parapets), not where another
// road joins or crosses, not where the caller says no (the car park's entrance). A gap of up to
// two stations between two walls is closed, a wall shorter than four stations is none.
//
// WHAT. Stones, each a rounded block of its own size, laid so that every one runs into its
// neighbours: a lower course of two or three to the 2 m, an upper course of smaller ones set
// half a stone along (no joint over a joint), now and then a boulder the height of both; at
// the wall's two ends the upper course is left off. Its foot is level with the road's edge and
// sunk a hand's width; where the ground outside lies lower, a footing course of large blocks
// goes from that ground up into the wall (so no bare bank of fill shows under it). Three shades of the same rock, so three meshes a stretch.
//
// Every stone is a function of its station, side and place in the course (a hash): a stretch
// can be made, freed and made again and is the same, and two stretches meet without a seam.
#include "engine/vk_engine/lane_edge.h"
#include "engine/highway.h"
#include "engine/log.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace vibe::lane_edge {

namespace {

constexpr float  kDrop = 1.5f;            // m of fall ...
constexpr float  kWithin = 4.0f;          // ... within this far beyond the verge
constexpr float  kOut = 0.50f;            // the wall's middle, beyond the verge's edge
constexpr size_t kStretch = 60;           // stations a stretch (120 m)
constexpr float  kCarHalf = 0.20f;        // a car's box: half its thickness, and its height
constexpr float  kTop = 0.78f;            // the wall's height over its foot (colliders)

struct St {
    glm::vec3 p; glm::vec2 t; float run; int line;
    uint8_t   want[2] = { 0, 0 };         // a wall on the left (0) / right (1) of the line's direction
    uint8_t   end[2] = { 0, 0 };          // ... and this is the first or last station of one
    glm::vec3 base[2];                    // the wall's foot there (middle of its thickness), at the height of the road's edge
    float     dig[2] = { 0.0f, 0.0f };    // how far the drawn ground lies under that (the lower course goes down to it)
    float     out_y[2] = { 0.0f, 0.0f };  // the ground just outside the wall, on the side of the drop
};
std::vector<St> g_st;
std::vector<collision::AABB> g_walk;
bool g_made = false;

uint32_t hash_u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed | 1u) {}
    float operator()() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float(s >> 8) / 16777216.0f; }
    float in(float a, float b) { return a + (b - a) * (*this)(); }
};

bool seg(size_t i, int sd) {              // a piece of wall from station i to i + 1 on side sd
    return i + 1 < g_st.size() && g_st[i].line == g_st[i + 1].line && g_st[i].want[sd] && g_st[i + 1].want[sd];
}
// A point of the wall's foot line: station i, a share f of the way to the next (f may pass 1
// into the next piece, when there is one).
glm::vec3 foot(size_t i, int sd, float f) {
    while (f > 1.0f && seg(i + 1, sd)) { f -= 1.0f; ++i; }
    return glm::mix(g_st[i].base[sd], g_st[i + 1].base[sd], std::min(f, 1.0f));
}

// One stone: a rounded block (a superquadric, 3 x 3 quads a face) with half-sizes `half` along
// the unit axes ax (its length), ay (up) and az; `seed` gives its roundness, taper and lumps.
void stone(game::LevelMesh& m, glm::vec3 c, glm::vec3 ax, glm::vec3 ay, glm::vec3 az, glm::vec3 half, uint32_t seed) {
    Rng r(seed);
    const float e = r.in(2.6f, 4.6f);                       // 2 = an egg, large = a brick
    const float taper = r.in(0.02f, 0.16f), lean = r.in(-0.10f, 0.10f);
    const uint32_t ls = hash_u(seed, 77u, 5u);
    constexpr int N = 3;
    auto at = [&](glm::vec3 q, glm::vec3& pos, glm::vec3& nrm) {
        const glm::vec3 a = glm::abs(q);
        const float k = std::pow(std::pow(a.x, e) + std::pow(a.y, e) + std::pow(a.z, e), -1.0f / e);
        glm::vec3 p = q * k;                                 // on the unit shape
        // (lumps: by the grid point, so the faces' shared edges get the same)
        const glm::ivec3 g = glm::ivec3(glm::round((q + 1.0f) * (0.5f * float(N))));
        const float lump = float(hash_u(ls, uint32_t(g.x + 4 * g.y + 16 * g.z), 3u) >> 8) / 16777216.0f;
        const glm::vec3 ap = glm::abs(p);
        glm::vec3 gn = glm::sign(p) * glm::vec3(std::pow(ap.x, e - 1.0f), std::pow(ap.y, e - 1.0f), std::pow(ap.z, e - 1.0f)) / half;
        p *= 1.0f + 0.13f * (lump - 0.5f);
        const float w = 1.0f - taper * p.y;                  // narrower toward the top
        glm::vec3 l(p.x * half.x * w + lean * p.y * half.y * 0.3f, p.y * half.y, p.z * half.z * w);
        pos = c + ax * l.x + ay * l.y + az * l.z;
        nrm = glm::normalize(ax * gn.x + ay * gn.y + az * gn.z);
    };
    for (int face = 0; face < 6; ++face) {
        const int ua = (face / 2 + 1) % 3, va = (face / 2 + 2) % 3, wa = face / 2;
        const float ws = face % 2 ? 1.0f : -1.0f;
        const uint32_t b0 = uint32_t(m.verts.size());
        for (int j = 0; j <= N; ++j)
            for (int i = 0; i <= N; ++i) {
                glm::vec3 q(0.0f);
                q[ua] = -1.0f + 2.0f * float(i) / float(N); q[va] = -1.0f + 2.0f * float(j) / float(N); q[wa] = ws;
                glm::vec3 p, n; at(q, p, n);
                m.verts.push_back({ p, n, 0.0f, 0.0f });
                m.bmin = glm::min(m.bmin, p); m.bmax = glm::max(m.bmax, p);
            }
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const uint32_t a = b0 + uint32_t(j * (N + 1) + i), b = a + 1, d = a + uint32_t(N + 1), cc = d + 1;
                // (front = cross(b - a, c - a) pointing outward: the winding the engine culls by)
                const glm::vec3 g = glm::cross(m.verts[b].p - m.verts[a].p, m.verts[cc].p - m.verts[a].p);
                if (glm::dot(g, m.verts[a].p + m.verts[cc].p - 2.0f * c) >= 0.0f) m.idx.insert(m.idx.end(), { a, b, cc, a, cc, d });
                else m.idx.insert(m.idx.end(), { a, cc, b, a, d, cc });
            }
    }
}

}  // namespace

bool made() { return g_made; }
const std::vector<collision::AABB>& walker_boxes() { return g_walk; }

void make(const Ground& ground, const std::function<bool(float, float)>& no_wall) {
    g_made = true;
    g_st.clear(); g_walk.clear();
    if (std::getenv("VIBE_NO_LANE_ROCKS")) return;
    for (int li = 1; li < highway::line_count(); ++li) {
        const std::vector<glm::vec3>& P = highway::line_points(li);
        const float verge = highway::kVerge * highway::line_width(li);
        const size_t first = g_st.size();
        float run = 0.0f, since = 1e9f;
        for (size_t i = 1; i + 1 < P.size(); ++i) {
            const float dl = glm::distance(glm::vec2(P[i].x, P[i].z), glm::vec2(P[i - 1].x, P[i - 1].z));
            run += dl; since += dl;
            if (since < 2.0f) continue;
            since = 0.0f;
            St s; s.p = P[i]; s.run = run; s.line = li;
            s.t = glm::normalize(glm::vec2(P[i + 1].x - P[i - 1].x, P[i + 1].z - P[i - 1].z) + glm::vec2(1e-6f, 0.0f));
            g_st.push_back(s);
        }
        const size_t last = g_st.size();
        for (int sd = 0; sd < 2; ++sd) {
            const float sg = sd ? 1.0f : -1.0f;
            for (size_t i = first; i < last; ++i) {
                St& s = g_st[i];
                const glm::vec2 c(s.p.x, s.p.z), nrm(-s.t.y * sg, s.t.x * sg);
                const glm::vec2 b = c + nrm * (verge + kOut);
                // the wall's foot: level with the road's edge (the shelf); where the drawn ground has
                // already begun to fall there, the lower course goes down to it
                s.base[sd] = glm::vec3(b.x, s.p.y - 0.06f, b.y);
                s.dig[sd] = std::clamp(s.base[sd].y - ground(b.x, b.y), 0.0f, 0.6f);
                { const glm::vec2 q = b + nrm * 0.75f; s.out_y[sd] = ground(q.x, q.y); }
                // THE RULE: how far the ground falls within kWithin beyond the verge
                float low = 1e9f;
                for (float o = 1.0f; o <= kWithin + 0.01f; o += 1.0f) { const glm::vec2 q = c + nrm * (verge + o); low = std::min(low, ground(q.x, q.y)); }
                if (s.p.y - low < kDrop) continue;
                float d, h, al; int ln = 0;
                if (!highway::nearest(c.x, c.y, 4.0f, d, h, al, ln) || ln != li) continue;       // (where this road meets a wider one)
                float dk; if (highway::deck(c.x, c.y, dk) || highway::deck(b.x, b.y, dk)) continue;   // (on a bridge)
                if (highway::distance(b.x, b.y) < highway::kVerge + 0.4f) continue;             // (on a road: another, or this one again round a hairpin)
                if (no_wall && (no_wall(b.x, b.y) || no_wall(c.x, c.y))) continue;
                s.want[sd] = 1;
            }
            auto W = [&](size_t i) -> uint8_t& { return g_st[i].want[sd]; };
            // (a gap of one or two stations is closed - unless it is one that must stay open)
            auto open = [&](size_t i) {
                const St& s = g_st[i]; float dk;
                return highway::deck(s.p.x, s.p.z, dk) || (no_wall && (no_wall(s.base[sd].x, s.base[sd].z) || no_wall(s.p.x, s.p.z)));
            };
            for (size_t i = first + 1; i + 1 < last; ++i) {
                if (W(i) || !W(i - 1) || open(i)) continue;
                if (W(i + 1)) W(i) = 1;
                else if (i + 2 < last && W(i + 2) && !open(i + 1)) { W(i) = 1; W(i + 1) = 1; }
            }
            // (a wall under four stations long is none)
            for (size_t i = first; i < last;) {
                if (!W(i)) { ++i; continue; }
                size_t j = i; while (j < last && W(j)) ++j;
                if (j - i < 4) for (size_t k = i; k < j; ++k) W(k) = 0;
                else { g_st[i].end[sd] = 1; g_st[j - 1].end[sd] = 1; }
                i = j;
            }
        }
    }
    // Solid for a walker: upright boxes half a metre along the wall.
    int pieces = 0; float metres = 0.0f;
    for (int sd = 0; sd < 2; ++sd)
        for (size_t i = 0; i + 1 < g_st.size(); ++i) {
            if (!seg(i, sd)) continue;
            ++pieces; metres += glm::distance(g_st[i].base[sd], g_st[i + 1].base[sd]);
            for (int k = 0; k < 4; ++k) {
                const glm::vec3 a = foot(i, sd, float(k) * 0.25f), b = foot(i, sd, float(k + 1) * 0.25f);
                const glm::vec3 lo = glm::min(a, b) - glm::vec3(0.24f, 0.3f, 0.24f), hi = glm::max(a, b) + glm::vec3(0.24f, kTop, 0.24f);
                g_walk.push_back({ lo, hi });
            }
        }
    log::infof("[lane] dry-stone wall where the ground falls more than %.1f m within %.0f m of the verge: %.0f m of it in %d pieces (%zu stations)",
               double(kDrop), double(kWithin), double(metres), pieces, g_st.size());
    if (std::getenv("VIBE_LOG_LANE")) {
        for (int sd = 0; sd < 2; ++sd)
            for (size_t i = 0; i < g_st.size();) {
                if (!g_st[i].want[sd]) { ++i; continue; }
                size_t j = i; while (j + 1 < g_st.size() && seg(j, sd)) ++j;
                log::infof("[lane] wall, %s side: %.0f m, from (%.1f, %.1f, %.1f) to (%.1f, %.1f, %.1f)", sd ? "right" : "left", double(g_st[j].run - g_st[i].run),
                           double(g_st[i].base[sd].x), double(g_st[i].base[sd].y), double(g_st[i].base[sd].z), double(g_st[j].base[sd].x), double(g_st[j].base[sd].y), double(g_st[j].base[sd].z));
                i = j + 1;
            }
    }
}

size_t stretch_count() { return g_st.size() < 2 ? 0 : (g_st.size() - 1) / kStretch + 1; }
glm::vec3 stretch_middle(size_t c) { return g_st[std::min(c * kStretch + kStretch / 2, g_st.size() - 1)].p; }
bool stretch_has_wall(size_t c) {
    for (size_t i = c * kStretch; i < std::min((c + 1) * kStretch, g_st.size()); ++i) if (seg(i, 0) || seg(i, 1)) return true;
    return false;
}

void build_stretch(size_t c, const Ground&, std::vector<game::LevelMesh>& out) {
    game::LevelMesh shade[3];
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    int stones = 0;
    for (int sd = 0; sd < 2; ++sd) {
        const float sg = sd ? 1.0f : -1.0f;
        for (size_t i = c * kStretch; i < std::min((c + 1) * kStretch, g_st.size()); ++i) {
            if (!seg(i, sd)) continue;
            const St& s = g_st[i];
            const glm::vec3 a = s.base[sd], b = g_st[i + 1].base[sd];
            const float len = glm::distance(a, b);
            if (len < 0.2f) continue;
            const glm::vec3 nout(-s.t.y * sg, 0.0f, s.t.x * sg);               // away from the road
            const bool low = s.end[sd] || g_st[i + 1].end[sd];                 // the wall's first and last 2 m: one course
            // A stone between the shares f0 and f1 of this piece (f1 may reach into the next).
            auto lay = [&](float f0, float f1, float y0, float hgt, float dep, float side, uint32_t seed) {
                Rng r(seed);
                const glm::vec3 p0 = foot(i, sd, f0), p1 = foot(i, sd, f1);
                glm::vec3 ax = p1 - p0;
                const float l = glm::length(ax);
                if (l < 0.12f) return;
                ax /= l;
                // (turned a little out of line and tipped a little: laid by hand)
                const float yaw = r.in(-0.07f, 0.07f), tip = r.in(-0.05f, 0.05f);
                glm::vec3 az = glm::normalize(glm::cross(ax, up));
                ax = glm::normalize(ax * std::cos(yaw) + az * std::sin(yaw));
                az = glm::normalize(glm::cross(ax, up));
                const glm::vec3 ay = glm::normalize(up + az * tip);
                const glm::vec3 half(0.5f * l * 1.10f + 0.03f, 0.5f * hgt, dep);       // (a tenth longer than its place: into its neighbours)
                const glm::vec3 ctr = 0.5f * (p0 + p1) + up * (y0 + 0.5f * hgt) + nout * side;
                stone(shade[hash_u(seed, 9u, 1u) % 3u], ctr, ax, ay, az, half, seed);
                ++stones;
            };
            const uint32_t key = uint32_t(i) * 2u + uint32_t(sd);
            const float dig = std::max(s.dig[sd], g_st[i + 1].dig[sd]);
            // -- the lower course: two or three stones to the piece, joints where a hash puts them --
            {
                Rng r(hash_u(key, 1u, 0u));
                const int n = r() < 0.45f ? 2 : 3;
                float f = 0.0f;
                for (int k = 0; k < n; ++k) {
                    const float f1 = k + 1 == n ? 1.0f : (float(k + 1) + r.in(-0.2f, 0.2f)) / float(n);
                    lay(f, f1, -0.10f - dig, r.in(0.44f, 0.56f) + dig, r.in(0.23f, 0.30f), r.in(-0.03f, 0.03f), hash_u(key, 11u, uint32_t(k)));
                    f = f1;
                }
            }
            // -- the upper course: smaller, half a stone along (the last one lies over the joint
            //    with the next piece), its top uneven --
            if (!low) {
                Rng r(hash_u(key, 2u, 0u));
                const int n = r() < 0.5f ? 3 : 4;
                const bool on = seg(i + 1, sd) && !(g_st[i + 1].end[sd] || (i + 2 < g_st.size() && g_st[i + 2].end[sd]));
                // (where the piece before has an upper course, its last stone reaches to this piece's first joint)
                const bool before = i > 0 && seg(i - 1, sd) && !(g_st[i - 1].end[sd] || s.end[sd]);
                float f = before ? 0.5f / float(n) + Rng(hash_u(key, 2u, 7u)).in(-0.06f, 0.06f) : 0.0f;
                for (int k = 0; k < n; ++k) {
                    float f1 = (float(k) + 1.5f + r.in(-0.15f, 0.15f)) / float(n);
                    if (k + 1 == n) {
                        if (on) { const uint32_t nk = key + 2u; const int nn = Rng(hash_u(nk, 2u, 0u))() < 0.5f ? 3 : 4; f1 = 1.0f + 0.5f / float(nn) + Rng(hash_u(nk, 2u, 7u)).in(-0.06f, 0.06f); }
                        else f1 = 1.0f;
                    }
                    lay(f, f1, 0.36f, r.in(0.40f, 0.54f), r.in(0.18f, 0.25f), r.in(-0.04f, 0.04f), hash_u(key, 12u, uint32_t(k)));
                    f = f1;
                }
            }
            // -- now and then a boulder as high as both courses, standing in the wall --
            {
                Rng r(hash_u(key, 3u, 0u));
                if (r() < (low ? 0.9f : 0.24f)) {
                    const float f0 = r.in(0.05f, 0.50f), w = r.in(0.75f, 1.10f) * (low ? 0.8f : 1.0f);      // (w: its length, m)
                    lay(f0, f0 + w / len, -0.16f, low ? r.in(0.55f, 0.70f) : r.in(0.92f, 1.08f), r.in(0.29f, 0.37f), r.in(-0.03f, 0.05f), hash_u(key, 13u, 0u));
                }
            }
            // -- under it on the side of the drop: a footing course of large blocks from the
            //    ground up into the wall, where the ground there lies lower than the wall's foot
            //    (the road's fill would show as a bare bank). Two to the piece, each into the next. --
            {
                Rng r(hash_u(key, 4u, 0u));
                const float j = r.in(0.38f, 0.62f);
                for (int k = 0; k < 2; ++k) {
                    const float f0 = k ? j : 0.0f, f1 = k ? 1.0f : j, fm = 0.5f * (f0 + f1);
                    const float top = glm::mix(a.y, b.y, fm) + 0.14f;
                    const float bot = glm::mix(s.out_y[sd], g_st[i + 1].out_y[sd], fm) - r.in(0.22f, 0.34f);
                    const float hgt = std::min(top - bot, 1.7f);
                    const float dep = r.in(0.36f, 0.48f), side = r.in(0.34f, 0.44f);
                    if (hgt < 0.42f) continue;
                    lay(f0, f1, top - hgt - glm::mix(a.y, b.y, fm), hgt, dep, side, hash_u(key, 14u, uint32_t(k)));
                }
            }
        }
    }
    // (2026-11-27, the owner: "the stones are too white, a bit more like sand colour": sandstone tones)
    const glm::vec3 tints[3] = { glm::vec3(1.30f, 1.10f, 0.82f), glm::vec3(1.04f, 0.92f, 0.74f), glm::vec3(1.42f, 1.19f, 0.86f) };
    for (int k = 0; k < 3; ++k) {
        game::LevelMesh& m = shade[k];
        if (m.idx.empty()) continue;
        m.triplanar = true;                     // (world-space rock, as the landscape's boulders: world_props_meshes.inl)
        m.tex_albedo = 7; m.tex_normal = 7; m.uv_scale = 0.45f;
        m.tint = tints[k]; m.fallback = glm::vec3(0.45f) * tints[k];
        out.push_back(std::move(m));
    }
    if (std::getenv("VIBE_LOG_LANE")) log::infof("[lane] stretch %zu: %d stones", c, stones);
}

void car_boxes(glm::vec3 at, float reach, std::vector<PhysicsWorld::OrientedBox>& out) {
    for (int sd = 0; sd < 2; ++sd) {
        const float sg = sd ? 1.0f : -1.0f;
        for (size_t i = 0; i + 1 < g_st.size(); ++i) {
            const St& s = g_st[i];
            if (std::abs(s.p.x - at.x) > reach || std::abs(s.p.z - at.z) > reach || !seg(i, sd)) continue;
            const glm::vec3 a = s.base[sd], b = g_st[i + 1].base[sd];
            const glm::vec3 d = b - a;
            const float len = glm::length(glm::vec2(d.x, d.z));
            if (len < 0.5f || len > 6.0f) continue;
            // (as the highway's rail - highway_furniture.cpp hw_rail_boxes: 40 cm thick, grown
            //  away from the road from the wall's face, each box a hand's width into the next)
            const glm::vec3 nout(-s.t.y * sg, 0.0f, s.t.x * sg);
            out.push_back({ 0.5f * (a + b) + glm::vec3(0.0f, 0.5f * kTop, 0.0f) + nout * (kCarHalf - 0.24f), glm::vec3(kCarHalf, 0.5f * kTop + 0.1f, 0.5f * len + 0.10f), std::atan2(d.x, d.z) });
        }
    }
}

}  // namespace vibe::lane_edge
