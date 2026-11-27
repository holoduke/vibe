// Rock formations: the kit's file, and the rule that stands the pieces in the landscape
// (see rock_formations.h).
//
// THE RULE. Everything is found from the ground as it is drawn and from the roads' lines, with
// hashes of where it is - no state, the same list on every PC.
//
// Along a narrow road (every line but the coast highway), a station every 5 m, each side:
//   * a CUT - the ground stands more than 1.4 m over the road 5 m beyond the verge and more
//     than 3 m at 12 m: the cut face is dressed with a buttress (small), a cleft wall, an
//     overhang, a stack or an outcrop, its front 1.6 m beyond the verge, its back in the bank;
//   * the FOOT OF A BANK - the ground lies more than 2.2 m under the road 8 m out: a boulder
//     pile, an outcrop or a small stack where the fall eases (it may stand with its feet in
//     the river's edge);
//   * otherwise now and then an outcrop or a small pile 5 - 14 m from the verge;
//   * two GATES: the arch a third of the way along, its passage along the road, and the
//     leaning pillars two thirds along - on the most level ground beside the road near those
//     places (a metre of unevenness counts as 60 m of road), beside the road, never over it.
// On the home tile's CLIFFS: the ground's slope is read every 16 m, summed over 96 m cells;
// the three steepest cells that lie 240 m apart (not the castle's own rock) are dressed -
// buttresses standing at the cliff's foot, sunk into the face to over half their depth (as tall as the cliff
// there, at most 32 m), overhangs, cleft walls and stacks on the face itself, piles and
// outcrops at its foot.
// Every piece stands upright (its beds level), turned to face down the slope or toward the
// road, and is let into the ground: its ground line lies a little under the LOWEST ground
// anywhere beneath it, so no edge hangs in the air and the contact is everywhere inside the
// blend (cube.frag, rock_form.glsl).
// Nothing stands on a road or within 1.3 m of a verge, on a bridge, in the water, where the
// caller says no (buildings, the car park, the castle), or through another formation.
#include "engine/vk_engine/rock_formations.h"
#include "engine/highway.h"
#include "engine/water_course.h"
#include "engine/log.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// (engine/terrain.h: the river's course - declared here, that header brings in the Windows
//  macros min and max that the code below does not expect)
namespace vibe { int world_river_course(const glm::vec2*& pts); glm::vec2 world_river_wander(glm::vec2 wp); }

namespace vibe::rockf {

namespace {

std::vector<Piece> g_kit;
std::vector<Inst>  g_inst;
bool g_placed = false;
int  g_why[11] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
int  g_shore_base = -1;                 // the shore kit's first piece (load_kit append), -1 none

uint32_t hash_u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed | 1u) { (*this)(); (*this)(); }
    float operator()() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float(s >> 8) / 16777216.0f; }
    float in(float a, float b) { return a + (b - a) * (*this)(); }
};

int find(const char* name) {
    for (size_t i = 0; i < g_kit.size(); ++i) if (g_kit[i].name == name) return int(i);
    return -1;
}

float slope_at(const Ground& g, float x, float z, glm::vec2* down = nullptr) {
    const glm::vec2 d(g(x + 4.0f, z) - g(x - 4.0f, z), g(x, z + 4.0f) - g(x, z - 4.0f));
    const float l = glm::length(d);
    if (down) *down = l > 1e-5f ? -d / l : glm::vec2(1.0f, 0.0f);
    return l / 8.0f;
}

float radius_of(const Piece& P, float s) { return 0.5f * s * std::max(P.bmax.x - P.bmin.x, P.bmax.z - P.bmin.z); }

// One formation, if it may stand there. c: its origin on the plan; `sink`: how far its ground
// line goes under the lowest ground beneath it (a share of its size, plus a hand's width).
bool try_add(const Env& env, int piece, glm::vec2 c, float yaw, float scale, int why, float sink, bool checks = true, float wade = 0.0f) {
    if (piece < 0) return false;
    const Piece& P = g_kit[size_t(piece)];
    const glm::vec2 ax(std::cos(yaw), -std::sin(yaw)), az(std::sin(yaw), std::cos(yaw));
    const float r = radius_of(P, scale);
    if (checks) {
        if (env.blocked && env.blocked(c.x, c.y, r)) return false;
        for (const Inst& o : g_inst) {
            const float ro = radius_of(g_kit[size_t(o.piece)], o.scale);
            if (glm::distance(glm::vec2(o.pos.x, o.pos.z), c) < 0.62f * (r + ro)) return false;
        }
    }
    float low = 1e9f;
    for (int j = 0; j < 5; ++j)
        for (int i = 0; i < 5; ++i) {
            const float u = scale * glm::mix(P.bmin.x, P.bmax.x, float(i) * 0.25f), v = scale * glm::mix(P.bmin.z, P.bmax.z, float(j) * 0.25f);
            const glm::vec2 q = c + ax * u + az * v;
            if (checks) {
                float d, h, s; int ln = 0;
                if (highway::nearest(q.x, q.y, 60.0f, d, h, s, ln)) {
                    const float k = highway::line_width(ln);
                    if (std::abs(d) * k < highway::kVerge * k + 1.3f) return false;
                }
                float dk; if (highway::deck(q.x, q.y, dk)) return false;
            }
            low = std::min(low, env.ground(q.x, q.y));
        }
    // (not in the water - but a pile at a bank's foot may stand with its feet in it: `wade`)
    // (the water here: the river falls toward the sea, so its level is a field - water_level_off)
    const float wl = env.level ? env.level(c.x, c.y) : env.water;
    if (checks && (low < wl + 0.3f - wade || env.ground(c.x, c.y) < wl + 0.3f - 0.4f * wade)) return false;
    Inst in;
    in.piece = piece; in.yaw = yaw; in.scale = scale; in.why = why;
    // (the test row stands on the ground under each piece's middle: there it is the piece that is looked at)
    in.pos = why == 5 ? glm::vec3(c.x, env.ground(c.x, c.y) - 0.25f, c.y) : glm::vec3(c.x, low - sink * scale - 0.10f, c.y);
    const glm::mat4 M = transform(in);
    in.lo = glm::vec3(1e9f); in.hi = glm::vec3(-1e9f);
    for (int k = 0; k < 8; ++k) {
        const glm::vec3 l((k & 1) ? P.bmax.x : P.bmin.x, (k & 2) ? P.bmax.y : P.bmin.y, (k & 4) ? P.bmax.z : P.bmin.z);
        const glm::vec3 w = glm::vec3(M * glm::vec4(l, 1.0f));
        in.lo = glm::min(in.lo, w); in.hi = glm::max(in.hi, w);
    }
    in.reach = glm::clamp((in.hi.y - in.lo.y) / 18.0f, 1.0f, 3.0f);      // (what is tall is seen from far)
    g_inst.push_back(in);
    ++g_why[why];
    return true;
}

float yaw_of(glm::vec2 front) { return std::atan2(front.x, front.y); }

void place_lanes(const Env& env) {
    const int arch = find("arch"), cleft = find("cleft"), overhang = find("overhang"), stack = find("stack"),
              pillars = find("pillars"), pile = find("pile"), buttress = find("buttress"), outcrop = find("outcrop");
    for (int li = 1; li < highway::line_count(); ++li) {
        const std::vector<glm::vec3>& P = highway::line_points(li);
        if (P.size() < 8) continue;
        const float verge = highway::kVerge * highway::line_width(li);
        struct St { glm::vec3 p; glm::vec2 t; float run; };
        std::vector<St> st;
        float run = 0.0f, since = 1e9f;
        for (size_t i = 1; i + 1 < P.size(); ++i) {
            const float dl = glm::distance(glm::vec2(P[i].x, P[i].z), glm::vec2(P[i - 1].x, P[i - 1].z));
            run += dl; since += dl;
            if (since < 5.0f) continue;
            since = 0.0f;
            st.push_back({ P[i], glm::normalize(glm::vec2(P[i + 1].x - P[i - 1].x, P[i + 1].z - P[i - 1].z) + glm::vec2(1e-6f, 0.0f)), run });
        }
        const float len = run;
        auto rise = [&](const St& s, glm::vec2 nrm, float o) {
            const glm::vec2 q = glm::vec2(s.p.x, s.p.z) + nrm * (verge + o);
            return env.ground(q.x, q.y) - s.p.y;
        };
        auto usable = [&](const St& s) {
            float dk;
            return s.run > 70.0f && s.run < len - 30.0f && !highway::deck(s.p.x, s.p.z, dk);
        };
        // -- the gates first (they need the most room) --
        for (int gate = 0; gate < 2; ++gate) {
            const int piece = gate ? pillars : arch;
            if (piece < 0) continue;
            const Piece& G = g_kit[size_t(piece)];
            const float want = len * (gate ? 0.72f : 0.36f), sc = gate ? 0.85f : 0.80f;
            std::vector<std::pair<float, uint32_t>> cand;
            for (size_t i = 0; i < st.size(); ++i)
                for (uint32_t sd = 0; sd < 2; ++sd) {
                    if (!usable(st[i])) continue;
                    const glm::vec2 nrm = glm::vec2(-st[i].t.y, st[i].t.x) * (sd ? 1.0f : -1.0f);
                    // (the most level ground beside the road within reach of the wanted place:
                    //  a metre of unevenness counts as 60 m of road)
                    const float a = std::abs(rise(st[i], nrm, 4.0f)), b = std::abs(rise(st[i], nrm, 10.0f)), c = std::abs(rise(st[i], nrm, 18.0f));
                    const float uneven = std::max(a, std::max(b * 0.8f, c * 0.6f));
                    if (uneven > 4.0f || rise(st[i], nrm, 10.0f) + st[i].p.y < env.water + 0.5f) continue;
                    cand.emplace_back(std::abs(st[i].run - want) + 60.0f * uneven, uint32_t(i) * 2u + sd);
                }
            std::sort(cand.begin(), cand.end());
            for (const auto& cd : cand) {
                const St& s = st[cd.second / 2u];
                const glm::vec2 nrm = glm::vec2(-s.t.y, s.t.x) * ((cd.second & 1u) ? 1.0f : -1.0f);
                const float half = sc * (gate ? 0.5f * (G.bmax.z - G.bmin.z) + 1.0f : std::max(-G.bmin.x, G.bmax.x));
                // (the arch's passage runs along the road; the pillars face it)
                const float yaw = gate ? yaw_of(-nrm) : yaw_of(s.t);
                if (try_add(env, piece, glm::vec2(s.p.x, s.p.z) + nrm * (verge + 2.6f + half), yaw, sc, 3, 0.10f)) break;
            }
        }
        // -- cuts, feet of banks, and a rock here and there --
        for (uint32_t sd = 0; sd < 2; ++sd) {
            float next_cut = 0.0f, next_foot = 0.0f, next_side = 40.0f;
            for (size_t i = 0; i < st.size(); ++i) {
                const St& s = st[i];
                if (!usable(s)) continue;
                const glm::vec2 c(s.p.x, s.p.z), nrm = glm::vec2(-s.t.y, s.t.x) * (sd ? 1.0f : -1.0f);
                Rng r(hash_u(uint32_t(li), uint32_t(i) * 2u + sd, 0x51u));
                const float r5 = rise(s, nrm, 5.0f), r8 = rise(s, nrm, 8.0f), r12 = rise(s, nrm, 12.0f);
                const float pick = r();
                if (r5 > 1.4f && r12 > 3.0f) {
                    if (s.run < next_cut) continue;
                    int piece; float sc;
                    if (pick < 0.30f)      { piece = buttress; sc = glm::clamp((r12 + 5.0f) / 32.0f, 0.26f, 0.45f) * r.in(0.9f, 1.15f); }
                    else if (pick < 0.50f) { piece = cleft;    sc = r.in(0.45f, 0.72f); }
                    else if (pick < 0.68f) { piece = overhang; sc = r.in(0.45f, 0.66f); }
                    else if (pick < 0.85f) { piece = stack;    sc = r.in(0.70f, 1.05f); }
                    else                   { piece = outcrop;  sc = r.in(1.3f, 2.3f); }
                    if (piece < 0) continue;
                    const Piece& K = g_kit[size_t(piece)];
                    const glm::vec2 at = c + nrm * (verge + 1.6f + sc * K.bmax.z);
                    if (try_add(env, piece, at, yaw_of(-nrm) + r.in(-0.18f, 0.18f), sc, 0, 0.12f))
                        next_cut = s.run + sc * (K.bmax.x - K.bmin.x) * 0.85f + r.in(3.0f, 14.0f);
                } else if (r8 < -2.2f) {
                    if (s.run < next_foot) continue;
                    float o = 12.0f;
                    for (float q = 6.0f; q <= 16.0f; q += 2.0f) if (rise(s, nrm, q + 2.0f) - rise(s, nrm, q) > -0.5f) { o = q + 1.0f; break; }
                    int piece; float sc;
                    if (pick < 0.5f)       { piece = pile;    sc = r.in(0.55f, 1.0f); }
                    else if (pick < 0.85f) { piece = outcrop; sc = r.in(1.2f, 2.4f); }
                    else                   { piece = stack;   sc = r.in(0.5f, 0.8f); }
                    if (piece < 0) continue;
                    o = std::max(o, 2.0f + radius_of(g_kit[size_t(piece)], sc));
                    if (try_add(env, piece, c + nrm * (verge + o), r.in(-3.14f, 3.14f), sc, 1, 0.08f, true, 0.9f)) next_foot = s.run + r.in(22.0f, 55.0f);
                } else if (s.run >= next_side) {
                    const bool big = pick < 0.55f;
                    const int piece = big ? outcrop : pile;
                    const float sc = big ? r.in(1.0f, 2.2f) : r.in(0.45f, 0.8f);
                    if (piece < 0) continue;
                    const float o = r.in(5.0f, 14.0f) + radius_of(g_kit[size_t(piece)], sc);
                    if (try_add(env, piece, c + nrm * (verge + o), r.in(-3.14f, 3.14f), sc, 2, 0.08f)) next_side = s.run + r.in(35.0f, 90.0f);
                }
            }
        }
    }
}

// ---- CLIFF RUNS: the face pieces side by side ------------------------------------------------
// A face piece is 30 m wide at size 1 and its ends match its neighbours' (make_formations.py,
// "the cliff-face family") - when they stand at ONE size, at ONE height, each 96 % of its
// width from the last. So a run is laid out as a whole and its pieces are not let into the
// ground one by one.
constexpr float kFaceW = 30.0f, kFaceH = 45.0f;

// A formation at a given height (a piece of a run, a sea stack).
bool add_fixed(const Env& env, int piece, glm::vec2 c, float y, float yaw, float scale, int why, int look, bool keep_clear = true) {
    if (piece < 0) return false;
    const Piece& P = g_kit[size_t(piece)];
    const glm::vec2 ax(std::cos(yaw), -std::sin(yaw)), az(std::sin(yaw), std::cos(yaw));
    if (keep_clear) {
        if (env.blocked && env.blocked(c.x, c.y, radius_of(P, scale))) return false;
        for (int j = 0; j < 5; ++j)
            for (int i = 0; i < 5; ++i) {
                const glm::vec2 q = c + ax * (scale * glm::mix(P.bmin.x, P.bmax.x, float(i) * 0.25f)) + az * (scale * glm::mix(P.bmin.z, P.bmax.z, float(j) * 0.25f));
                float d, h, s; int ln = 0;
                if (highway::nearest(q.x, q.y, 60.0f, d, h, s, ln)) {
                    const float k = highway::line_width(ln);
                    if (std::abs(d) * k < highway::kVerge * k + 4.0f) return false;
                }
                float dk; if (highway::deck(q.x, q.y, dk)) return false;
            }
    }
    Inst in;
    in.piece = piece; in.yaw = yaw; in.scale = scale; in.why = why; in.look = look;
    in.pos = glm::vec3(c.x, y, c.y);
    const glm::mat4 M = transform(in);
    in.lo = glm::vec3(1e9f); in.hi = glm::vec3(-1e9f);
    for (int k = 0; k < 8; ++k) {
        const glm::vec3 l((k & 1) ? P.bmax.x : P.bmin.x, (k & 2) ? P.bmax.y : P.bmin.y, (k & 4) ? P.bmax.z : P.bmin.z);
        const glm::vec3 w = glm::vec3(M * glm::vec4(l, 1.0f));
        in.lo = glm::min(in.lo, w); in.hi = glm::max(in.hi, w);
    }
    // (what is tall is seen from far: its levels of detail are held further out)
    in.reach = glm::clamp((in.hi.y - in.lo.y) / 18.0f, 1.0f, 3.0f);
    g_inst.push_back(in);
    ++g_why[why];
    return true;
}

// THE SEA CLIFF. The shore near home is searched for its steepest stretch - where the land
// stands highest over the sea 30 m in from the waterline, averaged over 340 m of shore - away
// from the highways (30 m from a verge), their bridges, the launch site and the station. From
// the middle of that stretch the waterline is followed both ways, a piece's width at a time,
// for as long as the land stays at least 45 % as high (at most 9 pieces each way). The pieces
// stand with their feet in the water, 3 m (x size) in from the waterline, facing the sea, in
// the WHITE look (chalk: rock_form.glsl); their size is the land's height there (16 - 60 m of
// cliff). The run's last piece on the side its right end points to is the headland; one or
// two stacks stand 45 - 75 m out in the sea.
// The terrain is NOT changed for it: where the land rises more gently than the cliff stands
// (everywhere a real sea cliff would need a sheer drop), there is lower ground behind the top.
void place_coast(const Env& env) {
    const int faces[4] = { find("face_a"), find("face_b"), find("face_c"), find("face_cave") };
    const int head = find("headland"), stack = find("seastack");
    if (faces[0] < 0 || faces[1] < 0 || faces[2] < 0 || faces[3] < 0) return;
    const float wl = env.water;
    // the waterline near a point, and which way is inland there
    auto shore = [&](glm::vec2 guess, glm::vec2& q, glm::vec2& in) {
        const glm::vec2 g(env.ground(guess.x + 10.0f, guess.y) - env.ground(guess.x - 10.0f, guess.y), env.ground(guess.x, guess.y + 10.0f) - env.ground(guess.x, guess.y - 10.0f));
        if (glm::length(g) < 1e-4f) return false;
        in = glm::normalize(g);
        float a = -48.0f, fa = env.ground(guess.x + in.x * a, guess.y + in.y * a) - wl;
        for (float b = -44.0f; b <= 48.0f; b += 4.0f) {
            const float fb = env.ground(guess.x + in.x * b, guess.y + in.y * b) - wl;
            if (fa < 0.0f && fb >= 0.0f) {
                float lo = a, hi = b;
                for (int k = 0; k < 12; ++k) { const float m = 0.5f * (lo + hi); (env.ground(guess.x + in.x * m, guess.y + in.y * m) - wl < 0.0f ? lo : hi) = m; }
                q = guess + in * (0.5f * (lo + hi));
                return true;
            }
            a = b; fa = fb;
        }
        return false;
    };
    auto rise_at = [&](glm::vec2 q, glm::vec2 in) { return env.ground(q.x + in.x * 30.0f, q.y + in.y * 30.0f) - wl; };
    auto clear = [&](glm::vec2 q, glm::vec2 in) {
        for (float o = -10.0f; o <= 50.0f; o += 15.0f) {
            const glm::vec2 p = q + in * o;
            float d, h, s; int ln = 0;
            if (highway::nearest(p.x, p.y, 64.0f, d, h, s, ln)) { const float k = highway::line_width(ln); if (std::abs(d) * k < highway::kVerge * k + 30.0f) return false; }
            float dk; if (highway::deck(p.x, p.y, dk)) return false;
        }
        return !(env.blocked && env.blocked(q.x, q.y, 60.0f));
    };
    // 1. the shore's points
    struct Sh { glm::vec2 q, in; float rise; };
    std::vector<Sh> sh;
    constexpr float kX0 = 900.0f, kX1 = 3700.0f, kZ0 = -2600.0f, kZ1 = 3400.0f, kStep = 28.0f;
    const int nx = int((kX1 - kX0) / kStep), nz = int((kZ1 - kZ0) / kStep);
    std::vector<float> hh(size_t(nx * nz));
    for (int j = 0; j < nz; ++j) for (int i = 0; i < nx; ++i) hh[size_t(j * nx + i)] = env.ground(kX0 + float(i) * kStep, kZ0 + float(j) * kStep);
    for (int j = 1; j + 1 < nz; ++j)
        for (int i = 1; i + 1 < nx; ++i) {
            const float h = hh[size_t(j * nx + i)];
            if (h < wl + 0.3f) continue;
            if (!(hh[size_t(j * nx + i - 1)] < wl - 0.2f || hh[size_t(j * nx + i + 1)] < wl - 0.2f || hh[size_t((j - 1) * nx + i)] < wl - 0.2f || hh[size_t((j + 1) * nx + i)] < wl - 0.2f)) continue;
            Sh s;
            if (!shore(glm::vec2(kX0 + float(i) * kStep, kZ0 + float(j) * kStep), s.q, s.in)) continue;
            // (the open sea, not the river or a pond: water still 120 m out)
            if (env.ground(s.q.x - s.in.x * 120.0f, s.q.y - s.in.y * 120.0f) > wl - 1.0f) continue;
            if (!clear(s.q, s.in)) continue;
            s.rise = rise_at(s.q, s.in);
            sh.push_back(s);
        }
    if (sh.size() < 8) { log::infof("[rocks] sea cliff: no open shore found near home (%zu points)", sh.size()); return; }
    // 2. its steepest stretch
    size_t best = 0; float best_rise = -1.0f, all = 0.0f;
    for (size_t i = 0; i < sh.size(); ++i) {
        float sum = 0.0f; int n = 0;
        for (const Sh& o : sh) if (glm::distance(o.q, sh[i].q) < 170.0f) { sum += o.rise; ++n; }
        all += sh[i].rise;
        if (n >= 6 && sum / float(n) > best_rise) { best_rise = sum / float(n); best = i; }
    }
    if (best_rise < 0.0f) return;
    const float cliff = glm::clamp(best_rise, 16.0f, 60.0f), sc = cliff / kFaceH, step = kFaceW * sc * 0.96f;
    // 3. along the waterline from there, both ways
    std::vector<Sh> chain;
    for (int dir = -1; dir <= 1; dir += 2) {
        std::vector<Sh> part;
        Sh cur = sh[best];
        for (int k = 0; k < 9; ++k) {
            const glm::vec2 t = glm::vec2(-cur.in.y, cur.in.x) * float(dir);
            Sh nxt;
            if (!shore(cur.q + t * step, nxt.q, nxt.in)) break;
            if (glm::dot(nxt.in, cur.in) < 0.80f) break;                              // (the shore turns a corner)
            nxt.rise = rise_at(nxt.q, nxt.in);
            if (nxt.rise < 0.45f * best_rise || !clear(nxt.q, nxt.in)) break;
            part.push_back(nxt);
            cur = nxt;
        }
        if (dir < 0) { std::reverse(part.begin(), part.end()); chain = part; chain.push_back(sh[best]); }
        else chain.insert(chain.end(), part.begin(), part.end());
    }
    log::infof("[rocks] sea cliff: %zu shore points near home, the land 30 m in stands %.1f m over the sea on average; steepest stretch round (%.0f, %.0f): %.1f m - a cliff of %.0f m, %zu pieces of %.0f m",
               sh.size(), double(all / float(sh.size())), double(sh[best].q.x), double(sh[best].q.y), double(best_rise), double(cliff), chain.size() > 0 ? chain.size() - 1 : 0, double(step));
    if (chain.size() < 3) return;
    const float y = wl - 0.35f * sc;
    int last_kind = -1;
    for (size_t i = 0; i + 1 < chain.size(); ++i) {
        const glm::vec2 a = chain[i].q, b = chain[i + 1].q, d = glm::normalize(b - a);
        glm::vec2 in(-d.y, d.x);
        if (glm::dot(in, chain[i].in + chain[i + 1].in) < 0.0f) in = -in;
        const float yaw = yaw_of(-in);
        const glm::vec2 ax(std::cos(yaw), -std::sin(yaw));
        // (the headland: the end piece its right end points out of the run from)
        const bool right_is_onward = glm::dot(ax, d) > 0.0f;
        const bool end_piece = right_is_onward ? i + 2 == chain.size() : i == 0;
        int kind = int(hash_u(uint32_t(i), 0x5EAu, 7u) % 4u);
        if (kind == last_kind) kind = (kind + 1) % 4;
        last_kind = kind;
        add_fixed(env, end_piece && head >= 0 ? head : faces[kind], 0.5f * (a + b) + in * (3.0f * sc), y, yaw, sc, 6, 1);
    }
    // 4. stacks out in the sea
    Rng r(0x57ACu);
    for (int k = 1; k <= 2; ++k) {
        const Sh& s = chain[chain.size() * size_t(k) / 3];
        const glm::vec2 t(-s.in.y, s.in.x);
        const glm::vec2 p = s.q - s.in * (r.in(45.0f, 75.0f) * std::max(sc, 0.7f)) + t * r.in(-12.0f, 12.0f);
        if (env.ground(p.x, p.y) > wl - 0.3f) continue;
        add_fixed(env, stack, p, wl - 0.5f * sc, r.in(-3.14f, 3.14f), sc * r.in(0.75f, 1.0f), 6, 1);
    }
}

// A CRAG BAND on an inland cliff: three face pieces in the dark rock along the contour at the
// foot of the area's steepest ground, as tall as the slope rises behind them (at most 40 m),
// their backs in the slope.
void place_band(const Env& env, glm::vec2 at) {
    const int faces[4] = { find("face_a"), find("face_b"), find("face_c"), find("face_cave") };
    if (faces[0] < 0 || faces[1] < 0 || faces[2] < 0) return;
    glm::vec2 down;
    slope_at(env.ground, at.x, at.y, &down);
    glm::vec2 foot = at;
    for (int k = 0; k < 16; ++k) { foot += down * 3.0f; if (slope_at(env.ground, foot.x, foot.y) < 0.5f) break; }
    const float tall = env.ground(foot.x - down.x * 30.0f, foot.y - down.y * 30.0f) - env.ground(foot.x, foot.y);
    if (tall < 12.0f) return;
    const float sc = glm::clamp(tall / kFaceH, 0.35f, 0.9f), step = kFaceW * sc * 0.96f;
    const glm::vec2 t(-down.y, down.x), base = foot - down * (4.0f * sc);
    float low = 1e9f;
    for (int k = -1; k <= 1; ++k)
        for (float o = -0.5f; o <= 0.5f; o += 0.25f) { const glm::vec2 q = base + t * ((float(k) + o) * step) + down * (5.0f * sc); low = std::min(low, env.ground(q.x, q.y)); }
    for (int k = -1; k <= 1; ++k)
        add_fixed(env, faces[size_t(hash_u(uint32_t(int(at.x) + 4096), uint32_t(k + 2), 3u) % 3u + (k == 0 ? 1u : 0u)) % 4u], base + t * (float(k) * step), low - 0.8f * sc, yaw_of(down), sc, 7, 0);
}

void place_cliffs(const Env& env) {
    const int cleft = find("cleft"), overhang = find("overhang"), stack = find("stack"),
              pile = find("pile"), buttress = find("buttress"), outcrop = find("outcrop");
    constexpr float kR = 960.0f, kCell = 96.0f;
    constexpr int kN = int(2.0f * kR / kCell);
    std::vector<float> score(size_t(kN * kN), 0.0f);
    for (float z = -kR + 8.0f; z < kR; z += 16.0f)
        for (float x = -kR + 8.0f; x < kR; x += 16.0f) {
            if (env.ground(x, z) < env.water + 1.0f) continue;
            const float sl = slope_at(env.ground, x, z);
            if (sl > 0.8f) score[size_t(int((z + kR) / kCell) * kN + int((x + kR) / kCell))] += sl - 0.8f;
        }
    std::vector<std::pair<float, int>> order;
    for (int i = 0; i < kN * kN; ++i) if (score[size_t(i)] > 0.0f) order.emplace_back(-score[size_t(i)], i);
    std::sort(order.begin(), order.end());
    std::vector<glm::vec2> areas;
    for (const auto& o : order) {
        const glm::vec2 c(-kR + (float(o.second % kN) + 0.5f) * kCell, -kR + (float(o.second / kN) + 0.5f) * kCell);
        if (glm::length(c) < 170.0f) continue;                       // (the castle's own rock)
        bool far_enough = true;
        for (const glm::vec2& a : areas) if (glm::distance(a, c) < 240.0f) far_enough = false;
        if (!far_enough) continue;
        areas.push_back(c);
        log::infof("[rocks] cliff area %zu round (%.0f, %.0f): steepness %.1f", areas.size(), double(c.x), double(c.y), double(-o.first));
        if (areas.size() >= 3) break;
    }
    for (size_t ai = 0; ai < areas.size(); ++ai) {
        const glm::vec2 ac = areas[ai];
        struct Cand { uint32_t key; glm::vec2 p; };
        std::vector<Cand> cand;
        float steepest = 0.0f;
        for (float z = ac.y - 62.0f; z <= ac.y + 62.0f; z += 5.0f)
            for (float x = ac.x - 62.0f; x <= ac.x + 62.0f; x += 5.0f) steepest = std::max(steepest, slope_at(env.ground, x, z));
        const float thr = std::max(0.80f, 0.68f * steepest);
        {   // (a crag band where the area is steepest - before the single pieces, which then keep off it)
            glm::vec2 top = ac; float ts = 0.0f;
            for (float z = ac.y - 62.0f; z <= ac.y + 62.0f; z += 5.0f)
                for (float x = ac.x - 62.0f; x <= ac.x + 62.0f; x += 5.0f) { const float sl = slope_at(env.ground, x, z); if (sl > ts) { ts = sl; top = glm::vec2(x, z); } }
            if (!std::getenv("VIBE_NO_CRAG_BANDS")) place_band(env, top);
        }
        for (float z = ac.y - 62.0f; z <= ac.y + 62.0f; z += 5.0f)
            for (float x = ac.x - 62.0f; x <= ac.x + 62.0f; x += 5.0f)
                if (slope_at(env.ground, x, z) > thr) cand.push_back({ hash_u(uint32_t(int(x) + 4096), uint32_t(int(z) + 4096), 0xC1u), glm::vec2(x, z) });
        std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.key < b.key; });
        int made = 0, feet = 0;
        for (const Cand& cd : cand) {
            if (made >= 13) break;
            Rng r(cd.key);
            glm::vec2 down;
            slope_at(env.ground, cd.p.x, cd.p.y, &down);
            const float pick = r();
            // the foot of the face under this point
            glm::vec2 foot = cd.p;
            for (int k = 0; k < 16; ++k) { foot += down * 3.0f; if (slope_at(env.ground, foot.x, foot.y) < 0.5f) break; }
            const float tall = env.ground(foot.x - down.x * 28.0f, foot.y - down.y * 28.0f) - env.ground(foot.x, foot.y);
            const float yaw = yaw_of(down) + r.in(-0.25f, 0.25f);
            bool ok = false;
            if (pick < 0.34f && tall > 8.0f && buttress >= 0) {
                const float sc = glm::clamp(tall * 1.15f / 32.0f, 0.32f, 1.0f);
                ok = try_add(env, buttress, foot - down * (sc * g_kit[size_t(buttress)].bmax.z * 0.55f), yaw, sc, 4, 0.10f);
            } else if (pick < 0.52f) ok = try_add(env, overhang, cd.p, yaw, r.in(0.8f, 1.3f), 4, 0.10f);
            else if (pick < 0.68f)   ok = try_add(env, cleft, cd.p, yaw, r.in(0.7f, 1.1f), 4, 0.10f);
            else if (pick < 0.86f)   ok = try_add(env, stack, cd.p, yaw, r.in(0.9f, 1.5f), 4, 0.10f);
            else                     ok = try_add(env, outcrop, cd.p, yaw, r.in(1.6f, 2.8f), 4, 0.10f);
            if (ok) ++made;
            if (feet < 5 && r() < 0.5f) {
                const glm::vec2 side(-down.y, down.x);
                const bool big = r() < 0.5f;
                if (try_add(env, big ? pile : outcrop, foot + down * r.in(2.0f, 9.0f) + side * r.in(-25.0f, 25.0f), r.in(-3.14f, 3.14f), big ? r.in(0.6f, 1.1f) : r.in(1.2f, 2.4f), 4, 0.08f)) ++feet;
            }
        }
    }
}

}  // namespace

const std::vector<Piece>& kit() { return g_kit; }
bool placed() { return g_placed; }
const std::vector<Inst>& instances() { return g_inst; }

int shore_base() { return g_shore_base; }
bool no_collider(const Inst& in) {
    return in.why == 10 && std::max(in.hi.x - in.lo.x, in.hi.z - in.lo.z) < 0.6f;
}

bool load_kit(const char* path, bool append, int* base) {
    const size_t keep = append ? g_kit.size() : 0;
    if (!append) { g_kit.clear(); g_shore_base = -1; }
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    const bool wood_ok = append;          // (the shore kit's driftwood: wood, not the rock scan)
    char magic[4]; uint32_t count = 0;
    bool ok = std::fread(magic, 1, 4, fp) == 4 && std::memcmp(magic, "RKF2", 4) == 0 && std::fread(&count, 4, 1, fp) == 1 && count < 64;
    for (uint32_t pi = 0; ok && pi < count; ++pi) {
        Piece P;
        char name[25] = {}; uint32_t levels = 0;
        ok = std::fread(name, 1, 24, fp) == 24 && std::fread(&levels, 4, 1, fp) == 1 && levels == uint32_t(kLevels);
        P.name = name;
        for (int l = 0; ok && l < kLevels; ++l) {
            uint32_t nv = 0, ni = 0; float bb[6];
            ok = std::fread(&nv, 4, 1, fp) == 1 && std::fread(&ni, 4, 1, fp) == 1 && std::fread(bb, 4, 6, fp) == 6 && nv < 4000000u && ni < 12000000u;
            if (!ok) break;
            game::LevelMesh& m = P.lod[l];
            static_assert(sizeof(game::LevelVertex) == 32, "the file's vertex is the engine's");
            m.verts.resize(nv); m.idx.resize(ni);
            ok = std::fread(m.verts.data(), sizeof(game::LevelVertex), nv, fp) == nv && std::fread(m.idx.data(), 4, ni, fp) == ni;
            m.bmin = glm::vec3(bb[0], bb[1], bb[2]); m.bmax = glm::vec3(bb[3], bb[4], bb[5]);
            // (the rock material: the landscape's rock scan laid on by world position - cube.frag,
            //  rock_form.glsl; normal index -9 says "a rock formation")
            m.tex_albedo = 7; m.tex_normal = -9; m.triplanar = true; m.uv_scale = 0.30f;
            m.fallback = glm::vec3(0.34f, 0.32f, 0.29f);
            m.ssdm = 0.075f;                                   // the scan's stones and ribs stand out this far (m)
            if (wood_ok && P.name.rfind("drift", 0) == 0) {
                // (bleached driftwood: the weathered timber laid on by world position, pale grey)
                m.tex_albedo = 17; m.tex_normal = -1; m.uv_scale = 0.6f;
                m.fallback = glm::vec3(0.62f, 0.59f, 0.54f);
                m.ssdm = 0.0f;
            }
            if (l == 0) { P.bmin = m.bmin; P.bmax = m.bmax; }
        }
        float g[4]; uint32_t n[3];
        ok = ok && std::fread(g, 4, 4, fp) == 4 && std::fread(n, 4, 3, fp) == 3 && n[0] * n[1] * n[2] < 4000000u;
        if (ok) {
            P.org = glm::vec3(g[0], g[1], g[2]); P.cell = g[3]; P.n = glm::ivec3(int(n[0]), int(n[1]), int(n[2]));
            P.solid.resize(size_t(n[0]) * n[1] * n[2]);
            ok = std::fread(P.solid.data(), 1, P.solid.size(), fp) == P.solid.size();
        }
        if (ok) g_kit.push_back(std::move(P));
    }
    std::fclose(fp);
    if (!ok) { g_kit.resize(keep); return false; }
    if (append && g_kit.size() > keep) { if (base) *base = int(keep); else g_shore_base = int(keep); }
    return !g_kit.empty();
}

glm::mat4 transform(const Inst& in) {
    glm::mat4 M = glm::translate(glm::mat4(1.0f), in.pos);
    M = glm::rotate(M, in.yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    if (in.tilt != 0.0f) M = glm::rotate(M, in.tilt, glm::vec3(1.0f, 0.0f, 0.0f));      // (the cliff walls: rock_cliffs.cpp)
    return glm::scale(M, glm::vec3(in.scale));
}

// ROCKS IN THE WATER: boulders and small piles standing in the stream's rapids and below its
// steps, and some in the river near its banks (never in the middle: a swimmer always has a
// way). By rule along the courses (engine/water_course.h; the river's polyline and its bend),
// the same on every PC. The water shader parts the current round each (water_fx.cpp
// river_piers: pillow, wake, eddies). Sized so that they stand 0.3 - 1.2 m out of the water.
void place_water_rocks(const Env& env) {
    const int pile = find("pile"), outcrop = find("outcrop");
    if (pile < 0 && outcrop < 0) return;
    auto add = [&](int piece, glm::vec2 c, float water_y, float out_of_water, float max_r, float yaw) {
        if (piece < 0) return false;
        const Piece& P = g_kit[size_t(piece)];
        const float bed = env.ground(c.x, c.y);
        const float depth = water_y - bed;
        if (depth < 0.15f) return false;
        const float sink = 0.25f;
        float sc = (depth + out_of_water + sink) / std::max(P.bmax.y, 0.1f);
        sc = std::min(sc, max_r / std::max(radius_of(P, 1.0f), 0.1f));
        if (sc < 0.08f) return false;
        if (sc * P.bmax.y - sink - 0.1f - depth < 0.12f) return false;         // (it must stand out of the water: too deep here for its size)
        // (stood on the lowest ground under it, it may end under water: then a size larger, as
        //  far as it may be, and then not here)
        const float sc_max = max_r / std::max(radius_of(P, 1.0f), 0.1f);
        for (int tries = 0; tries < 3; ++tries) {
            if (!try_add(env, piece, c, yaw, sc, 8, sink / sc, true, 1000.0f)) return false;
            const float top = g_inst.back().pos.y + sc * P.bmax.y;
            if (top >= water_y + 0.15f) return true;
            g_inst.pop_back(); --g_why[8];
            const float sc2 = sc + (water_y + 0.15f + out_of_water - top) / std::max(P.bmax.y, 0.1f);
            if (sc2 > sc_max) return false;
            sc = sc2;
        }
        return false;
    };
    // The same, tried from `c` outward along `side` (toward a bank) until the water is shallow
    // enough for a rock of that size to stand out of it.
    auto add_toward_bank = [&](int piece, glm::vec2 c, glm::vec2 side, float span, float water_y, float out_of_water, float max_r, float yaw) {
        for (int k = 0; k <= 8; ++k)
            if (add(piece, c + side * (span * float(k) / 8.0f), water_y, out_of_water, max_r, yaw)) return true;
        return false;
    };
    // The stream: below every step a boulder or two, and now and then one in a run.
    const wc::Course& st = wc::stream(env.water);
    if (st.ok) {
        for (size_t i = 1; i + 1 < st.pts.size(); ++i) {
            const glm::vec4 A = st.pts[i - 1], B = st.pts[i], C = st.pts[i + 1];
            const glm::vec2 d = glm::normalize(glm::vec2(C) - glm::vec2(B)), nrm(-d.y, d.x);
            Rng r(hash_u(uint32_t(i), 0x57AE11u, 9u));
            const float drop_in = A.z - B.z, run = (B.z - C.z) / std::max(glm::distance(glm::vec2(B), glm::vec2(C)), 1.0f);
            const float hw = std::abs(B.w);
            if (B.w < 0.0f && drop_in > 0.05f) {      // (a step ends at B: water_course.h)
                const int n = 1 + (r() < 0.5f ? 1 : 0);
                for (int k = 0; k < n; ++k) {
                    const float sd = r() < 0.5f ? -1.0f : 1.0f;
                    const glm::vec2 c = glm::vec2(B) + d * r.in(1.2f, 3.5f) + nrm * (sd * hw * r.in(0.0f, 0.5f));
                    add_toward_bank(r() < 0.6f ? outcrop : pile, c, nrm * sd, hw * 1.3f, B.z, r.in(0.2f, 0.7f), std::min(0.6f * hw + 0.8f, 2.6f), r.in(-3.14f, 3.14f));
                }
            } else if (run > 0.01f && r() < 0.5f) {
                const float sd = r() < 0.5f ? -1.0f : 1.0f;
                const glm::vec2 c = glm::vec2(B) + d * r.in(0.0f, 4.0f) + nrm * (sd * hw * r.in(0.3f, 0.6f));
                add_toward_bank(outcrop, c, nrm * sd, hw * 1.0f, B.z, r.in(0.15f, 0.5f), std::min(0.55f * hw + 0.6f, 2.2f), r.in(-3.14f, 3.14f));
            }
        }
    }
    // The river: every ~50 m a chance of one near a bank (the outer third of its channel), more
    // where it is narrow; none within 35 m of a bridge or the castle's lake.
    {
        const glm::vec2* rp = nullptr;
        const int rn = world_river_course(rp);
        auto world_of = [](glm::vec2 q) { glm::vec2 w = q; for (int k = 0; k < 6; ++k) w = q - world_river_wander(w); return w; };
        float s = 0.0f;
        int idx = 0;
        for (int i = 0; i + 1 < rn; ++i) {
            const glm::vec2 a = rp[i], b = rp[i + 1];
            const float len = glm::distance(a, b);
            for (float u = std::fmod(50.0f - std::fmod(s, 50.0f), 50.0f); u < len; u += 50.0f, ++idx) {
                const float along = s + u;
                if (along < 140.0f || along > 2050.0f) continue;              // (not on home's lake; not in the tidal mouth)
                Rng r(hash_u(uint32_t(idx), 0x41BE2u, 17u));
                if (r() > 0.55f) continue;
                const glm::vec2 qd = (b - a) / len;
                const glm::vec2 q = a + qd * u;
                const glm::vec2 w = world_of(q);
                if (glm::distance(w, glm::vec2(43.4f, 297.4f)) < 45.0f) continue;      // (the stone bridge and its pool)
                bool near_bridge = false;
                for (float ox = -35.0f; ox <= 35.0f && !near_bridge; ox += 7.0f)
                    for (float oz = -35.0f; oz <= 35.0f && !near_bridge; oz += 7.0f) { float dk; if (highway::deck(w.x + ox, w.y + oz, dk)) near_bridge = true; }
                if (near_bridge) continue;
                const float hw = 7.0f + 9.0f * glm::smoothstep(1450.0f, 1900.0f, w.x);
                const glm::vec2 nrm(-qd.y, qd.x);
                // (in the bent coordinate: across the channel, then back to the world)
                const float sd = r() < 0.5f ? -1.0f : 1.0f;
                const glm::vec2 c = world_of(q + nrm * (sd * hw * r.in(0.85f, 1.0f)));
                const glm::vec2 out = glm::normalize(world_of(q + nrm * (sd * (hw + 4.0f))) - world_of(q + nrm * (sd * hw)));
                const int n = r() < 0.35f ? 2 : 1;
                for (int k = 0; k < n; ++k)
                    add_toward_bank(r() < 0.5f ? outcrop : pile, c + glm::vec2(r.in(-2.5f, 2.5f), r.in(-2.5f, 2.5f)) * float(k), out, 9.0f, env.water, r.in(0.3f, 1.0f), r.in(1.8f, 3.0f), r.in(-3.14f, 3.14f));
            }
            s += len;
        }
    }
}

// BOULDERS ALONG THE RIVER: small, medium and large water-worn stones (kit pieces boulder_s*,
// _m*, _l*) on both banks, by formula - the same on every PC. Every 7 m of the river's course
// and on each side:
//   * a cluster, or nothing: a slow noise along the river (a few hundred metres per swell)
//     decides how stony that stretch is (0.15 .. 0.8 chance), so there are stony reaches and
//     clean ones, not an even sprinkle;
//   * a cluster is one stone plus up to three smaller ones round it within 2.5 m;
//   * its size by chance: 12 % large, 33 % medium, the rest small - fewer large ones where the
//     channel is narrow; each stone 0.8 .. 1.25 x its piece, turned any way, a quarter of it in
//     the ground;
//   * where: across the bank from the water's edge (85 % of the half-width) to 5 m up the bank
//     - the large ones keep to the edge (their feet may stand in the water), the small ones
//     scatter further up;
//   * not on home's lake, not in the tidal mouth, not near a bridge, not on a road or a building
//     (try_add's checks).
void place_river_boulders(const Env& env) {
    std::vector<int> S, M, L;
    for (int k = 1; k <= 4; ++k) {
        int i;
        if ((i = find(("boulder_s" + std::to_string(k)).c_str())) >= 0) S.push_back(i);
        if ((i = find(("boulder_m" + std::to_string(k)).c_str())) >= 0) M.push_back(i);
        if ((i = find(("boulder_l" + std::to_string(k)).c_str())) >= 0) L.push_back(i);
    }
    log::infof("[rocks] river boulders: %zu small, %zu medium, %zu large pieces in the kit", S.size(), M.size(), L.size());
    if (S.empty() && M.empty() && L.empty()) return;
    auto pick = [&](Rng& r, float big_ok) -> int {
        const float u = r();
        const std::vector<int>* set = (u < 0.12f * big_ok && !L.empty()) ? &L : (u < 0.45f && !M.empty()) ? &M : (!S.empty() ? &S : (!M.empty() ? &M : &L));
        return (*set)[size_t(r() * float(set->size())) % set->size()];
    };
    const glm::vec2* rp = nullptr;
    const int rn = world_river_course(rp);
    auto world_of = [](glm::vec2 q) { glm::vec2 w = q; for (int k = 0; k < 6; ++k) w = q - world_river_wander(w); return w; };
    float s = 0.0f;
    int idx = 0;
    for (int i = 0; i + 1 < rn; ++i) {
        const glm::vec2 a = rp[i], b = rp[i + 1];
        const float len = glm::distance(a, b);
        if (len < 1e-3f) continue;
        const glm::vec2 qd = (b - a) / len, nrm(-qd.y, qd.x);
        for (float u = std::fmod(7.0f - std::fmod(s, 7.0f), 7.0f); u < len; u += 7.0f, ++idx) {
            const float along = s + u;
            if (along < 140.0f || along > 2050.0f) continue;                      // (not on home's lake; not in the tidal mouth)
            const glm::vec2 q = a + qd * u;
            const glm::vec2 w0 = world_of(q);
            if (glm::distance(w0, glm::vec2(43.4f, 297.4f)) < 40.0f) continue;     // (the stone bridge and its pool)
            {
                bool near_bridge = false; float dk;
                for (float ox = -25.0f; ox <= 25.0f && !near_bridge; ox += 5.0f)
                    for (float oz = -25.0f; oz <= 25.0f && !near_bridge; oz += 5.0f) if (highway::deck(w0.x + ox, w0.y + oz, dk)) near_bridge = true;
                if (near_bridge) continue;
            }
            const float hw = 7.0f + 9.0f * glm::smoothstep(1450.0f, 1900.0f, w0.x);  // (the channel's half-width, as place_water_rocks)
            // how stony this reach is: a slow swell along the river
            const float sw = 0.5f + 0.5f * std::sin(along * 0.011f + 1.3f) * std::cos(along * 0.0047f + 0.4f);
            const float chance = 0.15f + 0.65f * sw * sw;
            for (int side = 0; side < 2; ++side) {
                Rng r(hash_u(uint32_t(idx), 0xB0D1E5u, uint32_t(side) + 3u));
                if (r() > chance) continue;
                const float sd = side == 0 ? -1.0f : 1.0f;
                // the water's edge on this side: out from the river's line until the ground stands
                // above the water (the channel is wider than its nominal half-width in places)
                const glm::vec2 qa = q + qd * r.in(-3.0f, 3.0f);
                float edge = -1.0f;
                for (float e = 0.5f * hw; e < 70.0f; e += 0.5f) {
                    const glm::vec2 wq = world_of(qa + nrm * (sd * e));
                    if (env.ground(wq.x, wq.y) > (env.level ? env.level(wq.x, wq.y) : env.water) + 0.12f) { edge = e; break; }
                }
                if (edge < 0.0f) continue;
                // the size: fewer large where the channel is narrow (its real width: the edge found)
                const int piece = pick(r, glm::smoothstep(5.0f, 11.0f, edge));
                const Piece& P = g_kit[size_t(piece)];
                const bool large = std::find(L.begin(), L.end(), piece) != L.end();
                const bool small = std::find(S.begin(), S.end(), piece) != S.end();
                // across the bank from there: the large at the edge (feet in the water), the small further up
                const float up = large ? r.in(-0.6f, 1.2f) : (small ? r.in(0.4f, 5.0f) : r.in(0.0f, 3.0f));
                const glm::vec2 c = world_of(qa + nrm * (sd * (edge + up)));
                const float sc = r.in(0.8f, 1.25f);
                const float wade = large ? 1.2f : (small ? 0.0f : 0.4f);        // (how far below the water the footprint may reach: try_add)
                if (!try_add(env, piece, c, r.in(-3.14f, 3.14f), sc, 9, 0.25f * P.bmax.y, true, wade)) {
                    if (std::getenv("VIBE_LOG_ROCKS")) log::infof("[rocks] boulder refused at (%.1f, %.1f): ground %.2f, water %.2f, wade %.1f, blocked %d", double(c.x), double(c.y), double(env.ground(c.x, c.y)), double(env.level ? env.level(c.x, c.y) : env.water), double(wade), env.blocked ? int(env.blocked(c.x, c.y, radius_of(P, sc))) : -1);
                    continue;
                }
                // the smaller ones round it
                const int kids = int(r() * 3.999f);
                for (int k = 0; k < kids; ++k) {
                    const int pk = S.empty() ? piece : S[size_t(r() * float(S.size())) % S.size()];
                    const float ang = r.in(-3.14f, 3.14f), dist = r.in(0.9f, 2.5f) * (large ? 1.6f : 1.0f);
                    const glm::vec2 ck = c + glm::vec2(std::cos(ang), std::sin(ang)) * dist;
                    try_add(env, pk, ck, r.in(-3.14f, 3.14f), r.in(0.7f, 1.2f), 9, 0.25f * g_kit[size_t(pk)].bmax.y, true, 0.2f);
                }
            }
        }
        s += len;
    }
}

// BOULDERS IN THE STREAM (the owner: "more boulders, small and medium, in the small river"): along
// the whole course (engine/water_course.h), by formula - the same on every PC. A station every
// 1.6 m of the course; at each a chance (higher in fast water: runs and sheets 0.55, pools 0.3,
// the boiling foot of a step 0.7) of one stone, now and then with a smaller one beside it:
//   * SIZE (across): 70 % small 0.3 - 0.8 m, 30 % medium 0.8 - 1.5 m (fewer medium where the
//     channel is narrow); the piece from the river boulders (boulder_s* / _m*) or the shore kit's
//     large cobbles (cobble_l*);
//   * WHERE across: anywhere in the channel (to 0.95 of its half width), most toward the banks;
//   * HOW HIGH: its top from 0.25 m UNDER the surface (a quarter of them: the water humps over
//     them - the stream's surface relief, water_course.glsl wc_relief) to half its size out of it;
//     always let 10 % into the bed;
//   * not on a fall's face (its lip and the chute right below are kept clear), not within 0.62 x
//     the two radii of another formation (try_add), not past the mouth (the river's level).
// why 8 (the rocks in the water): each is an obstacle for the current and the solver's bed
// (rock_formations_gpu.cpp -> water_fx.cpp river_pier_add with its top: a submerged one is a
// round stone the water runs over, not a wall). VIBE_NO_STREAM_BOULDERS=1: none.
void place_stream_boulders(const Env& env) {
    const wc::Course& st = wc::stream(env.water);
    if (!st.ok || st.pts.size() < 3) return;
    std::vector<int> S, M;
    for (int k = 1; k <= 4; ++k) {
        int i;
        if ((i = find(("boulder_s" + std::to_string(k)).c_str())) >= 0) S.push_back(i);
        if ((i = find(("boulder_m" + std::to_string(k)).c_str())) >= 0) M.push_back(i);
    }
    if (g_shore_base >= 0)
        for (int i = g_shore_base; i < int(g_kit.size()); ++i)
            if (g_kit[size_t(i)].name.rfind("cobble_l", 0) == 0) { S.push_back(i); M.push_back(i); }
    if (S.empty() && M.empty()) return;
    if (S.empty()) S = M;
    if (M.empty()) M = S;
    int placed = 0, under = 0, tried = 0;
    float s_along = 0.0f;
    const size_t n = st.pts.size();
    for (size_t k = 0; k + 2 < n; ++k) {           // (the last segment runs into the river)
        const glm::vec4 A = st.pts[k], B = st.pts[k + 1];
        const float L = glm::distance(glm::vec2(A), glm::vec2(B));
        if (L < 0.5f) continue;
        for (float u = std::fmod(1.6f - std::fmod(s_along, 1.6f), 1.6f); u < L; u += 1.6f) {
            const glm::vec2 c0 = glm::mix(glm::vec2(A), glm::vec2(B), u / L);
            Rng r(hash_u(uint32_t((s_along + u) * 4.0f), 0x5B01D3u, 21u));
            const wc::At W = wc::at(st, c0);
            if (!W.on) continue;
            if (W.surf < (env.level ? env.level(c0.x, c0.y) : env.water) + 0.3f) continue;                                    // (the mouth)
            if (W.drop > 0.05f && u > L - std::min(0.5f * W.drop, 2.0f) - 1.8f) continue; // (a fall's lip and face)
            const float sp = wc::flow_speed(W);
            const float chance = W.foot > 0.2f ? 0.7f : glm::mix(0.3f, 0.55f, glm::smoothstep(0.8f, 1.6f, sp));
            if (r() > chance) continue;
            const int stones = r() < 0.3f ? 2 : 1;
            for (int j = 0; j < stones; ++j) {
                ++tried;
                const bool medium = j == 0 && r() < 0.30f * glm::smoothstep(1.2f, 2.5f, W.half_w);
                const std::vector<int>& set = medium ? M : S;
                const int piece = set[size_t(r() * float(set.size())) % set.size()];
                const Piece& P = g_kit[size_t(piece)];
                // across: toward the banks more often (|x|^0.6), never quite at the bank's foot
                const float ax = r.in(-1.0f, 1.0f);
                const float lat = (ax < 0.0f ? -1.0f : 1.0f) * std::pow(std::abs(ax), 0.6f) * 0.95f * W.half_w;
                const glm::vec2 c = c0 + glm::vec2(-W.dir.y, W.dir.x) * lat + W.dir * (j > 0 ? r.in(-1.2f, 1.2f) : 0.0f) +
                                    glm::vec2(-W.dir.y, W.dir.x) * (j > 0 ? r.in(-0.9f, 0.9f) : 0.0f);
                const wc::At Wc = wc::at(st, c);
                if (!Wc.on || Wc.dist > Wc.half_w) continue;
                const float bed = env.ground(c.x, c.y);
                const float depth = Wc.surf - bed;
                if (depth < 0.08f) continue;                                            // (dry here)
                // its size: a quarter of them lie under the surface (the water humps over them); the rest
                // stand out of it - in a deep pool a small stone could not, so it is made as large as it
                // must be (up to the medium's 1.5 m; deeper than that it lies under too)
                const float aspect = std::max(P.bmax.y, 0.05f) / std::max(2.0f * radius_of(P, 1.0f), 0.05f);   // (height / across)
                bool sub = r() < 0.25f;
                float D = medium ? r.in(0.8f, 1.5f) : r.in(0.3f, 0.8f) * (j > 0 ? 0.7f : 1.0f);
                const float out = r.in(0.05f, 0.35f);
                if (!sub) {
                    const float need = (depth + out + 0.12f) / 0.9f / aspect;          // (the across that stands `out` over the water, a tenth buried)
                    if (need > 1.5f) sub = true; else D = std::max(D, need);
                }
                const float sc = D / std::max(2.0f * radius_of(P, 1.0f), 0.05f);
                const float H = sc * P.bmax.y;                                          // (its height over its ground line)
                const float top = Wc.surf + (sub ? -r.in(0.05f, 0.25f) : out);
                // pos.y = low - sink * sc - 0.10 (try_add), top = pos.y + H: the sink that puts the top
                // there, at least 10 % of the piece in the bed
                const float sink = std::max((bed - 0.10f - (top - H)) / sc, 0.10f * P.bmax.y);
                if (try_add(env, piece, c, r.in(-3.14f, 3.14f), sc, 8, sink, true, 1000.0f)) {
                    // (try_add stands it on the LOWEST ground under its footprint - in the channel that
                    //  lies well under the bed at its middle, and nearly every stone ended under the
                    //  water: lifted to the top wanted, as long as a tenth of it stays in the ground)
                    Inst& in = g_inst.back();
                    // (the limit is the bed under its middle, 15 % of it below: at its edge, over the channel's
                    //  curve, it may stand a little clear of the lowest ground - under the water)
                    const float dz = std::min(top - in.hi.y, bed - 0.15f * H - in.pos.y);
                    if (dz > 0.0f) { in.pos.y += dz; in.lo.y += dz; in.hi.y += dz; }
                    ++placed;
                    if (g_inst.back().hi.y < Wc.surf) ++under;
                }
            }
        }
        s_along += L;
    }
    log::infof("[rocks] stream boulders: %d placed of %d tried (%d under the surface), %zu small / %zu medium pieces", placed, tried, under, S.size(), M.size());
}

// THE SHORE KIT along the banks (assets/shore/shore.rockf, a second kit: load_kit append; the
// reference game places its bank layer's stones and wood by its layer's placement groups - here
// one rule over every bank of the home tile, the same on every PC):
//   * the banks are found on a 1.5 m grid of the home tile: a dry cell (the ground over the water
//     there - the level, or the stream's surface) beside a wet one is a waterline cell;
//   * the current there: the stream's own speed (water_course.h flow_speed); the river's by its
//     bend (inside slow, outside fast, a straight reach between); a lake (no river within 35 m)
//     still water;
//   * COBBLES in clumps of 1 - 4 within 1.2 m, from 0.3 m into the water to 1.5 m up the bank:
//     a clump in up to 55 % of the cells where the water runs fast, 12 % where it is still,
//     fewer on steep banks; the sizes sorted - large ones (35 - 70 cm) by fast water at the
//     line, medium ones there and a little up, small ones everywhere and furthest up;
//   * DRIFTWOOD on slow banks and lake shores (1.2 % of the cells), 0.5 - 2.5 m up the bank and
//     over the water by 15 cm at least, along the bank: a log, a limb or sticks;
//   * SHINGLE SLABS at the edge of fast reaches (3 % of the cells where it runs over 1.2 m/s);
//   * not on roads, buildings, bridges (env.blocked, the highways' decks), not through a rock
//     formation or another shore piece, not under 0.3 m of water; let into the ground 2 cm under
//     its lowest point (the pieces carry their own roots).
// Reeds where the bank is slow: the river plants (river_plants.h) stand there already.
// VIBE_NO_SHORE_KIT=1: none (and the kit is not loaded). Its pieces have why 10.
void place_shore(const Env& env) {
    if (g_shore_base < 0 || std::getenv("VIBE_NO_SHORE_KIT")) return;
    std::vector<int> CS, CM, CL, SH;
    int log_p = -1, limb = -1, st1 = -1, st2 = -1;
    for (int i = g_shore_base; i < int(g_kit.size()); ++i) {
        const std::string& n = g_kit[size_t(i)].name;
        if (n.rfind("cobble_s", 0) == 0) CS.push_back(i);
        else if (n.rfind("cobble_m", 0) == 0) CM.push_back(i);
        else if (n.rfind("cobble_l", 0) == 0) CL.push_back(i);
        else if (n.rfind("shingle", 0) == 0) SH.push_back(i);
        else if (n == "drift_log") log_p = i;
        else if (n == "drift_limb") limb = i;
        else if (n == "drift_stick1") st1 = i;
        else if (n == "drift_stick2") st2 = i;
    }
    if (CS.empty() && CM.empty() && CL.empty()) return;
    // the river's middle every 7 m within the home tile, with its bend (river_plants.h's stations)
    struct Mid { glm::vec2 p; float bend; glm::vec2 dir; };
    std::vector<Mid> mids;
    {
        const glm::vec2* rp = nullptr;
        const int rn = world_river_course(rp);
        std::vector<glm::vec2> m;
        for (int i = 0; i + 1 < rn; ++i) {
            const float len = glm::distance(rp[i], rp[i + 1]);
            for (float u = 0.0f; u < len; u += 7.0f) { const glm::vec2 c = glm::mix(rp[i], rp[i + 1], u / std::max(len, 1e-3f)); m.push_back(c - world_river_wander(c)); }
        }
        for (size_t i = 2; i + 2 < m.size(); ++i) {
            if (std::abs(m[i].x) > 1060.0f || std::abs(m[i].y) > 1060.0f) continue;
            const glm::vec2 d0 = m[i] - m[i - 2], d1 = m[i + 2] - m[i];
            const float l0 = glm::length(d0), l1 = glm::length(d1);
            const float bend = (l0 > 1.0f && l1 > 1.0f) ? (d0.x * d1.y - d0.y * d1.x) / (l0 * l1) * 30.0f / (0.5f * (l0 + l1)) : 0.0f;
            mids.push_back({ m[i], bend, glm::normalize(m[i + 2] - m[i - 2] + glm::vec2(1e-5f, 0.0f)) });
        }
    }
    const wc::Course& stc = wc::stream(env.water);
    glm::vec4 sbox(1e9f, 1e9f, -1e9f, -1e9f);
    for (const glm::vec4& B : stc.box) { sbox.x = std::min(sbox.x, B.x); sbox.y = std::min(sbox.y, B.y); sbox.z = std::max(sbox.z, B.z); sbox.w = std::max(sbox.w, B.w); }
    // the ground and the water surface on the grid
    constexpr float kC = 1.5f, kHalf = 1015.0f;
    const int N = int(2.0f * kHalf / kC);
    std::vector<float> G(size_t(N) * N), S(size_t(N) * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float x = -kHalf + (float(i) + 0.5f) * kC, z = -kHalf + (float(j) + 0.5f) * kC;
            const float g = env.ground(x, z);
            G[size_t(j) * N + i] = g;
            float s = (env.level && g < env.water + 2.0f) ? env.level(x, z) : env.water;      // (the sea and the lower river stand under the datum: water_fx.cpp water_level_off)
            if (stc.ok && x > sbox.x && z > sbox.y && x < sbox.z && z < sbox.w && g > env.water - 0.5f) s = std::max(s, wc::stream_surface(glm::vec2(x, z), env.water));
            S[size_t(j) * N + i] = s;
        }
    // what stands already (the formations), on an 8 m grid
    std::vector<std::vector<int>> big(size_t(256) * 256);
    auto bcell = [](float v) { return std::clamp(int((v + 1024.0f) / 8.0f), 0, 255); };
    for (size_t k = 0; k < g_inst.size(); ++k) {
        const Inst& o = g_inst[k];
        for (int bz = bcell(o.lo.z - 1.0f); bz <= bcell(o.hi.z + 1.0f); ++bz)
            for (int bx = bcell(o.lo.x - 1.0f); bx <= bcell(o.hi.x + 1.0f); ++bx) big[size_t(bz) * 256 + bx].push_back(int(k));
    }
    // the shore pieces placed so far, on a 1 m grid (hashed)
    std::vector<std::vector<int>> own(size_t(1) << 16);
    auto okey = [](int x, int z) { return (uint32_t(x) * 73856093u ^ uint32_t(z) * 19349663u) & 0xFFFFu; };
    auto free_at = [&](glm::vec2 c, float r) {
        for (int k : big[size_t(bcell(c.y)) * 256 + bcell(c.x)]) {
            const Inst& o = g_inst[size_t(k)];
            if (c.x > o.lo.x - r && c.x < o.hi.x + r && c.y > o.lo.z - r && c.y < o.hi.z + r) return false;
        }
        const int cx = int(std::floor(c.x)), cz = int(std::floor(c.y)), reach = int(std::ceil(r + 1.7f));
        for (int dz = -reach; dz <= reach; ++dz)
            for (int dx = -reach; dx <= reach; ++dx)
                for (int k : own[okey(cx + dx, cz + dz)]) {
                    const Inst& o = g_inst[size_t(k)];
                    const float ro = radius_of(g_kit[size_t(o.piece)], o.scale);
                    if (glm::distance(glm::vec2(o.pos.x, o.pos.z), c) < 0.8f * (r + ro)) return false;
                }
        return true;
    };
    auto add = [&](int piece, glm::vec2 c, float yaw, float sc, float surf, bool dry) {
        if (piece < 0) return false;
        const Piece& P = g_kit[size_t(piece)];
        const float r = radius_of(P, sc);
        if (!free_at(c, r)) return false;
        if (env.blocked && env.blocked(c.x, c.y, r)) return false;
        float dk; if (highway::deck(c.x, c.y, dk)) return false;
        { float d, h, s; int ln = 0;
          if (highway::nearest(c.x, c.y, 30.0f, d, h, s, ln) && std::abs(d) < highway::kVerge * highway::line_width(ln) + 0.8f + r) return false; }
        const glm::vec2 ax(std::cos(yaw), -std::sin(yaw)), az(std::sin(yaw), std::cos(yaw));
        float low = 1e9f;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                const glm::vec2 q = c + ax * (sc * glm::mix(P.bmin.x, P.bmax.x, float(i) * 0.5f)) + az * (sc * glm::mix(P.bmin.z, P.bmax.z, float(j) * 0.5f));
                low = std::min(low, env.ground(q.x, q.y));
            }
        if (low < surf - 0.3f) return false;                                  // (under 0.3 m of water: the bed's own stones)
        if (dry && low < surf + 0.15f) return false;
        Inst in;
        in.piece = piece; in.yaw = yaw; in.scale = sc; in.why = 10;
        in.pos = glm::vec3(c.x, low - 0.02f, c.y);
        const glm::mat4 M = transform(in);
        in.lo = glm::vec3(1e9f); in.hi = glm::vec3(-1e9f);
        for (int k = 0; k < 8; ++k) {
            const glm::vec3 l((k & 1) ? P.bmax.x : P.bmin.x, (k & 2) ? P.bmax.y : P.bmin.y, (k & 4) ? P.bmax.z : P.bmin.z);
            const glm::vec3 w = glm::vec3(M * glm::vec4(l, 1.0f));
            in.lo = glm::min(in.lo, w); in.hi = glm::max(in.hi, w);
        }
        in.reach = 1.0f;
        own[okey(int(std::floor(c.x)), int(std::floor(c.y)))].push_back(int(g_inst.size()));
        g_inst.push_back(in);
        ++g_why[10];
        return true;
    };
    auto pick = [](Rng& r, const std::vector<int>& v) { return v.empty() ? -1 : v[size_t(r() * float(v.size())) % v.size()]; };
    int cells = 0, n_cob = 0, n_wood = 0, n_sh = 0;
    for (int j = 1; j + 1 < N; ++j)
        for (int i = 1; i + 1 < N; ++i) {
            const size_t at = size_t(j) * N + i;
            if (G[at] < S[at]) continue;                                     // (a wet cell)
            const float s = S[at];
            if (G[at] > s + 0.6f) continue;
            bool edge = false;
            for (int k = 0; k < 4 && !edge; ++k) {
                const size_t nb = size_t(j + (k == 2) - (k == 3)) * N + size_t(i + (k == 0) - (k == 1));
                edge = G[nb] < S[nb];
            }
            if (!edge) continue;
            ++cells;
            const glm::vec2 c(-kHalf + (float(i) + 0.5f) * kC, -kHalf + (float(j) + 0.5f) * kC);
            Rng r(hash_u(uint32_t(i), uint32_t(j), 0x5E0EEu));
            // up the bank, and how steep
            glm::vec2 up(G[at + 1] - G[at - 1], G[at + size_t(N)] - G[at - size_t(N)]);
            const float slope = glm::length(up) / (2.0f * kC);
            up = glm::length(up) > 1e-4f ? glm::normalize(up) : glm::vec2(1.0f, 0.0f);
            const glm::vec2 along(-up.y, up.x);
            // the current
            float speed;
            const bool in_stream = s > (env.level ? env.level(c.x, c.y) : env.water) + 0.02f;
            if (in_stream) {
                speed = wc::flow_speed(wc::at(stc, c));
            } else {
                float bd = 1e9f; const Mid* bm = nullptr;
                for (const Mid& m : mids) { const float d = glm::distance(m.p, c); if (d < bd) { bd = d; bm = &m; } }
                if (!bm || bd > 35.0f) speed = 0.15f;                       // (a lake)
                else {
                    // which side of the bend: the course turns toward +nrm when bend > 0 - that
                    // side is the inside (slow), the other the outside (fast)
                    const glm::vec2 nrm(-bm->dir.y, bm->dir.x);
                    const float sd = glm::dot(c - bm->p, nrm) >= 0.0f ? 1.0f : -1.0f;
                    const float turn = std::clamp(bm->bend * sd, -1.0f, 1.0f);
                    speed = 0.8f - 0.5f * glm::smoothstep(0.15f, 0.6f, turn) + 0.6f * glm::smoothstep(0.15f, 0.6f, -turn);
                }
            }
            const float fast = glm::smoothstep(0.4f, 1.5f, speed);
            const float steep = glm::smoothstep(0.6f, 1.1f, slope);
            // -- cobbles --
            if (r() < glm::mix(0.12f, 0.55f, fast) * (1.0f - 0.8f * steep)) {
                const int n = 1 + int(r() * 3.99f);
                const float o0 = r.in(-0.3f, 1.2f);
                for (int k = 0; k < n; ++k) {
                    const float o = o0 + r.in(-0.3f, 0.5f) * float(k > 0);
                    const glm::vec2 q = c + up * o + along * r.in(-1.2f, 1.2f) * float(k > 0) + along * r.in(-0.6f, 0.6f);
                    const float u = r();
                    const bool at_line = o < 0.4f;
                    int p;
                    if (u < 0.40f * fast * float(at_line) && !CL.empty()) p = pick(r, CL);
                    else if (u < 0.35f + 0.25f * fast * float(o < 0.8f) && !CM.empty()) p = pick(r, CM);
                    else p = pick(r, CS.empty() ? CM : CS);
                    if (add(p, q, r.in(-3.14f, 3.14f), r.in(0.8f, 1.2f), s, false)) ++n_cob;
                }
            }
            // -- driftwood (slow banks, lake shores) --
            if (speed < 0.6f && steep < 0.5f && r() < 0.012f) {
                const float u = r();
                const int p = u < 0.15f ? log_p : (u < 0.45f ? limb : (u < 0.75f ? st1 : st2));
                const glm::vec2 q = c + up * r.in(0.5f, 2.5f);
                const float yaw = std::atan2(along.x, along.y) + r.in(-0.5f, 0.5f) + (r() < 0.5f ? 3.14159f : 0.0f);
                if (add(p, q, yaw, r.in(0.85f, 1.15f), s, true)) ++n_wood;
            }
            // -- shingle slabs (the edge of fast reaches) --
            if (speed > 1.2f && !SH.empty() && r() < 0.03f) {
                if (add(pick(r, SH), c + up * r.in(-0.2f, 0.5f), r.in(-3.14f, 3.14f), r.in(0.8f, 1.2f), s, false)) ++n_sh;
            }
        }
    log::infof("[rocks] shore kit: %d waterline cells, %d cobbles, %d driftwood, %d shingle slabs", cells, n_cob, n_wood, n_sh);
}

void place(const Env& env) {
    g_placed = true;
    g_inst.clear();
    for (int& w : g_why) w = 0;
    if (g_kit.empty()) return;
    // VIBE_TEST_ROCKS=x,z: the whole kit in a row from there, along the hillside, fronts down the slope.
    if (const char* e = std::getenv("VIBE_TEST_ROCKS")) {
        float x = 0.0f, z = 0.0f;
        if (std::sscanf(e, "%f,%f", &x, &z) == 2)
            for (size_t i = 0; i < g_kit.size(); ++i) {
                // (along the contour, so that all of them stand at about one height on a hillside;
                //  each faces down the slope)
                const Piece& P = g_kit[i];
                glm::vec2 down;
                slope_at(env.ground, x, z, &down);
                const glm::vec2 along(-down.y, down.x);
                x += along.x * -P.bmin.x; z += along.y * -P.bmin.x;
                try_add(env, int(i), glm::vec2(x, z), yaw_of(down), 1.0f, 5, 0.0f, false);
                x += along.x * (P.bmax.x + 5.0f); z += along.y * (P.bmax.x + 5.0f);
            }
    }
    if (!std::getenv("VIBE_NO_LANE_FORMATIONS")) place_lanes(env);
    if (!std::getenv("VIBE_NO_CLIFF_FORMATIONS")) place_cliffs(env);
    if (!std::getenv("VIBE_NO_SEA_CLIFF")) place_coast(env);
    if (!std::getenv("VIBE_NO_WATER_ROCKS")) place_water_rocks(env);
    if (!std::getenv("VIBE_NO_RIVER_BOULDERS")) place_river_boulders(env);
    if (!std::getenv("VIBE_NO_STREAM_BOULDERS")) place_stream_boulders(env);
    place_shore(env);
    log::infof("[rocks] %zu formations: %d at cuts beside a road, %d at the foot of a bank, %d beside a road, %d gates, %d on cliffs, %d in crag bands, %d of the sea cliff, %d in the test row",
               g_inst.size(), g_why[0], g_why[1], g_why[2], g_why[3], g_why[4], g_why[7], g_why[6], g_why[5]);
    log::infof("[rocks] %d in the water (the stream's steps and rapids, the river's banks)", g_why[8]);
    log::infof("[rocks] %d boulders along the river's banks", g_why[9]);
    if (std::getenv("VIBE_LOG_ROCKS")) {
        static const char* why[11] = { "cut", "foot", "roadside", "gate", "cliff", "test", "seacliff", "band", "water", "boulder", "shore" };
        for (const Inst& in : g_inst)
            log::infof("[rocks] %-8s %-8s at (%.1f, %.1f, %.1f) yaw %.2f size %.2f: %.1f x %.1f m, %.1f m high", why[in.why], g_kit[size_t(in.piece)].name.c_str(),
                       double(in.pos.x), double(in.pos.y), double(in.pos.z), double(in.yaw), double(in.scale),
                       double(in.hi.x - in.lo.x), double(in.hi.z - in.lo.z), double(in.hi.y - in.pos.y));
    }
}

// The triangles a piece collides with: its middle level of detail (a few centimetres from the
// drawn one near by), in the piece's own axes, three corners each.
void collision_triangles(int piece, std::vector<glm::vec3>& out) {
    out.clear();
    if (piece < 0 || size_t(piece) >= g_kit.size()) return;
    const game::LevelMesh& m = g_kit[size_t(piece)].lod[kCollisionLevel];
    out.reserve(m.idx.size());
    for (size_t k = 0; k + 2 < m.idx.size(); k += 3)
        for (int c = 0; c < 3; ++c) out.push_back(m.verts[m.idx[k + size_t(c)]].p);
}

void collision_mesh(const Inst& in, tricol::Mesh& out) {
    out = tricol::Mesh{};
    if (in.piece < 0 || size_t(in.piece) >= g_kit.size()) return;
    const game::LevelMesh& m = g_kit[size_t(in.piece)].lod[kCollisionLevel];
    const glm::mat4 M = transform(in);
    std::vector<glm::vec3> w(m.verts.size());
    for (size_t i = 0; i < m.verts.size(); ++i) w[i] = glm::vec3(M * glm::vec4(m.verts[i].p, 1.0f));
    out.v.reserve(m.idx.size()); out.n.reserve(m.idx.size() / 3);
    for (size_t k = 0; k + 2 < m.idx.size(); k += 3) out.add(w[m.idx[k]], w[m.idx[k + 1]], w[m.idx[k + 2]]);
    out.build(2.0f);
}

}  // namespace vibe::rockf
