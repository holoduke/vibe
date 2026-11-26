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
#include "engine/log.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vibe::rockf {

namespace {

std::vector<Piece> g_kit;
std::vector<Inst>  g_inst;
bool g_placed = false;
int  g_why[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

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
    if (checks && (low < env.water + 0.3f - wade || env.ground(c.x, c.y) < env.water + 0.3f - 0.4f * wade)) return false;
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

bool load_kit(const char* path) {
    g_kit.clear();
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    char magic[4]; uint32_t count = 0;
    bool ok = std::fread(magic, 1, 4, fp) == 4 && std::memcmp(magic, "RKF1", 4) == 0 && std::fread(&count, 4, 1, fp) == 1 && count < 64;
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
    if (!ok) g_kit.clear();
    return ok && !g_kit.empty();
}

glm::mat4 transform(const Inst& in) {
    glm::mat4 M = glm::translate(glm::mat4(1.0f), in.pos);
    M = glm::rotate(M, in.yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    return glm::scale(M, glm::vec3(in.scale));
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
    log::infof("[rocks] %zu formations: %d at cuts beside a road, %d at the foot of a bank, %d beside a road, %d gates, %d on cliffs, %d in crag bands, %d of the sea cliff, %d in the test row",
               g_inst.size(), g_why[0], g_why[1], g_why[2], g_why[3], g_why[4], g_why[7], g_why[6], g_why[5]);
    if (std::getenv("VIBE_LOG_ROCKS")) {
        static const char* why[8] = { "cut", "foot", "roadside", "gate", "cliff", "test", "seacliff", "band" };
        for (const Inst& in : g_inst)
            log::infof("[rocks] %-8s %-8s at (%.1f, %.1f, %.1f) yaw %.2f size %.2f: %.1f x %.1f m, %.1f m high", why[in.why], g_kit[size_t(in.piece)].name.c_str(),
                       double(in.pos.x), double(in.pos.y), double(in.pos.z), double(in.yaw), double(in.scale),
                       double(in.hi.x - in.lo.x), double(in.hi.z - in.lo.z), double(in.hi.y - in.pos.y));
    }
}

void boxes(const Inst& in, const Ground& ground, std::vector<collision::AABB>& out) {
    const Piece& P = g_kit[size_t(in.piece)];
    if (P.solid.empty()) return;
    const float cw = std::max(P.cell * in.scale, 0.35f);
    const float cy = std::cos(in.yaw), sy = std::sin(in.yaw);
    const glm::vec3 lo = in.lo;
    const int nx = int(std::ceil((in.hi.x - lo.x) / cw)), ny = int(std::ceil((in.hi.y - lo.y) / cw)), nz = int(std::ceil((in.hi.z - lo.z) / cw));
    auto solid = [&](int ix, int iy, int iz) {
        const glm::vec3 w = lo + (glm::vec3(float(ix), float(iy), float(iz)) + 0.5f) * cw - in.pos;
        // (the turn undone: local = R^T w / size)
        const glm::vec3 l = glm::vec3(cy * w.x - sy * w.z, w.y, sy * w.x + cy * w.z) / in.scale;
        const glm::ivec3 c = glm::ivec3(glm::floor((l - P.org) / P.cell));
        if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= P.n.x || c.y >= P.n.y || c.z >= P.n.z) return false;
        return P.solid[size_t(c.x) + size_t(P.n.x) * (size_t(c.y) + size_t(P.n.y) * size_t(c.z))] != 0;
    };
    struct Run { int x0, x1; size_t box; };
    std::vector<Run> prev, cur;
    for (int iz = 0; iz < nz; ++iz) {
        prev.clear();
        for (int iy = 0; iy < ny; ++iy) {
            cur.clear();
            const float y0 = lo.y + float(iy) * cw;
            for (int ix = 0; ix < nx;) {
                auto here = [&](int x) {
                    if (!solid(x, iy, iz)) return false;
                    return y0 + cw > ground(lo.x + (float(x) + 0.5f) * cw, lo.z + (float(iz) + 0.5f) * cw) - 0.2f;      // (not what lies under the ground)
                };
                if (!here(ix)) { ++ix; continue; }
                int jx = ix; while (jx < nx && here(jx)) ++jx;
                // (the same run one level down: that box grows upward)
                size_t box = SIZE_MAX;
                for (const Run& p : prev) if (p.x0 == ix && p.x1 == jx) { box = p.box; break; }
                if (box != SIZE_MAX) out[box].max.y = y0 + cw;
                else {
                    box = out.size();
                    out.push_back({ glm::vec3(lo.x + float(ix) * cw, y0, lo.z + float(iz) * cw), glm::vec3(lo.x + float(jx) * cw, y0 + cw, lo.z + float(iz + 1) * cw) });
                }
                cur.push_back({ ix, jx, box });
                ix = jx;
            }
            prev.swap(cur);
        }
    }
}

}  // namespace vibe::rockf
