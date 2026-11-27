// Water above the sea: courses with their own surface height (water_course.h; GPU twin
// shaders/water_course.glsl - keep the two in step).
#include "engine/water_course.h"
#include "engine/globe.h"
#include "engine/log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>

namespace vibe {
float world_river_distance(float x, float z);       // (engine/terrain.h - not included: it pulls in Vulkan, and the unit tests link this file alone)
}

namespace vibe::wc {

namespace {
thread_local bool t_building = false;       // (laying out the stream reads the ground formula, which carves the stream)

inline float sstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

}  // namespace

// 1D value noise from an integer hash (water_course.glsl wc_n1: the same bits).
float n1(float x) {
    const float fl = std::floor(x), f = x - fl;
    auto h = [](int32_t i) { uint32_t v = uint32_t(i) * 0x9E3779B1u; v ^= v >> 15; v *= 0x85EBCA6Bu; v ^= v >> 13; return float(v >> 8) * (1.0f / 16777216.0f); };
    const int32_t i = int32_t(fl);
    const float u = f * f * (3.0f - 2.0f * f);
    return h(i) + (h(i + 1) - h(i)) * u;
}

// The surface along segment k at u (0..1) of its length L, `lat` metres across from its centre
// line, and its fall per metre there. A STEP (the point at the segment's end has w < 0): level
// above it - the pool -, the drop over fl metres at the lower end, the lip not straight: it lies
// up to 0.9 m further up or down from place to place across the stream (n1 of `lat`). Else a
// run: the surface falls evenly.
float seg_surface(const Course& c, int k, float u, float L, float lat, float& slope) {
    const float s0 = c.pts[size_t(k)].z, s1 = c.pts[size_t(k) + 1].z, drop = s0 - s1;
    if (c.pts[size_t(k) + 1].w < 0.0f && drop > 0.05f) {
        // (a real fall - over 1.2 m - drops nearly sheer: its face 0.3 x its height, at most 1 m; water_course.glsl the same)
        const float fl = (std::min)(drop > 1.2f ? std::clamp(0.3f * drop, 0.45f, 1.0f) : std::clamp(drop * 0.5f, 0.4f, 2.0f), 0.6f * L);
        const float shift = 0.9f * (n1(lat * 0.9f + float(k) * 3.17f) - 0.5f) * 2.0f;
        const float a = (std::max)(L - fl - 0.9f + shift, 0.5f);
        const float t = std::clamp((u * L - a) / fl, 0.0f, 1.0f);
        slope = drop * 6.0f * t * (1.0f - t) / fl;
        return s0 - drop * t * t * (3.0f - 2.0f * t);
    }
    slope = drop / (std::max)(L, 1e-3f);
    return s0 - drop * u;
}

// The channel's depth under the surface at point k: deep in pools and runs (0.45 m at the
// spring .. 1.3 at the mouth), a thin sheet of 0.22 m where the water runs down steep ground
// (water_course.glsl wc_depth_pt: the same).
static float seg_slope(const Course& c, int j) {
    const int n = int(c.pts.size());
    if (j < 0 || j >= n - 1) return 0.0f;
    if (c.pts[size_t(j) + 1].w < 0.0f) return 0.0f;                    // (a step: its pool is deep)
    return (c.pts[size_t(j)].z - c.pts[size_t(j) + 1].z) / (std::max)(glm::distance(glm::vec2(c.pts[size_t(j)]), glm::vec2(c.pts[size_t(j) + 1])), 1e-3f);
}
float depth_pt(const Course& c, int k) {
    const int n = int(c.pts.size());
    const float sheet = sstep(0.06f, 0.12f, (std::max)(seg_slope(c, k - 1), seg_slope(c, k)));
    return glm::mix(glm::mix(c.depth0, c.depth1, float(k) / float((std::max)(n - 1, 1))), 0.22f, sheet);
}

static At seg_at(const Course& c, glm::vec2 wp, int bk, float bu, float best);
At at(const Course& c, glm::vec2 wp) {
    At r;
    if (!c.ok) return r;
    const int n = int(c.pts.size());
    float best = 1e18f, bkey = 1e18f;
    int bk = -1; float bu = 0.0f;
    float bkey2 = 1e18f, bu2 = 0.0f; int bk2 = -1;      // (the runner-up: blended with at the seam)
    // The segment that holds wp is the one wp lies DEEPEST inside the reach of (its distance less an
    // estimate of its reach), not merely the nearest (water_course.glsl wc_stream_at, keep in step):
    // where two segments' nearest regions met (a bend, the course doubling back) the nearest one's
    // reach could end while the other's water stood there, and the water was cut along the straight
    // line between the regions. VIBE_NO_STREAM_SEAM_FIX=1: the nearest, as before.
    static const bool reach_pick = [] { const char* e = std::getenv("VIBE_NO_STREAM_SEAM_FIX"); return !(e && *e && *e != '0'); }();
    const float inv_n = 1.0f / float((std::max)(n - 1, 1));
    for (int b = 0; b < int(c.box.size()); ++b) {
        const glm::vec4 B = c.box[size_t(b)];
        if (wp.x < B.x || wp.y < B.y || wp.x > B.z || wp.y > B.w) continue;
        for (int k = b * kSegsPerBox; k < (std::min)((b + 1) * kSegsPerBox, n - 1); ++k) {
            const glm::vec2 a(c.pts[size_t(k)].x, c.pts[size_t(k)].y), ab = glm::vec2(c.pts[size_t(k) + 1].x, c.pts[size_t(k) + 1].y) - a;
            const float u = std::clamp(glm::dot(wp - a, ab) / (std::max)(glm::dot(ab, ab), 1e-6f), 0.0f, 1.0f);
            const glm::vec2 dv = wp - (a + ab * u);
            const float d2 = glm::dot(dv, dv);
            float key = d2;
            if (reach_pick) {
                const float ra = glm::mix(std::abs(c.pts[size_t(k)].w), std::abs(c.pts[size_t(k) + 1].w), u) +
                                 1.18f * glm::mix(c.depth0, c.depth1, (float(k) + u) * inv_n) + 0.35f;
                key = std::sqrt(d2) - ra;
            }
            if (key < bkey) { bkey2 = bkey; bk2 = bk; bu2 = bu; bkey = key; best = d2; bk = k; bu = u; }
            else if (key < bkey2) { bkey2 = key; bk2 = k; bu2 = u; }
        }
    }
    if (bk < 0 || best > kBoxPad * kBoxPad) return r;
    r = seg_at(c, wp, bk, bu, best);
    // (the seam: within 1.2 m of where the next segment's region begins the two surfaces are blended,
    //  half and half on the seam - one continuous surface; water_course.glsl wc_stream_at, the same)
    if (reach_pick && bk2 >= 0 && bkey2 - bkey < 1.2f) {
        const glm::vec2 a2(c.pts[size_t(bk2)].x, c.pts[size_t(bk2)].y), b2(c.pts[size_t(bk2) + 1].x, c.pts[size_t(bk2) + 1].y);
        const float L2 = glm::distance(a2, b2);
        const glm::vec2 d2v = (b2 - a2) / (std::max)(L2, 1e-4f);
        float sl2 = 0.0f;
        const float s2 = seg_surface(c, bk2, bu2, L2, glm::dot(wp - a2, glm::vec2(-d2v.y, d2v.x)), sl2);
        r.surf = glm::mix(r.surf, s2, 0.5f - 0.5f * (bkey2 - bkey) / 1.2f);
    }
    return r;
}

// The stream as segment bk alone sees it at wp (bu: wp's projection on it, best: the squared
// distance; water_course.glsl wc_seg_at). at() takes the nearest segment.
static At seg_at(const Course& c, glm::vec2 wp, int bk, float bu, float best) {
    At r;
    float cum = 0.0f, bcum = 0.0f;
    for (int k = 0; k < bk; ++k) cum += glm::distance(glm::vec2(c.pts[size_t(k)].x, c.pts[size_t(k)].y), glm::vec2(c.pts[size_t(k) + 1].x, c.pts[size_t(k) + 1].y));
    bcum = cum;
    const glm::vec2 a(c.pts[size_t(bk)].x, c.pts[size_t(bk)].y), b(c.pts[size_t(bk) + 1].x, c.pts[size_t(bk) + 1].y);
    const float L = glm::distance(a, b);
    r.on = true;
    r.dist = std::sqrt(best);
    const glm::vec2 dirv = (b - a) / (std::max)(L, 1e-4f);
    r.lat = glm::dot(wp - a, glm::vec2(-dirv.y, dirv.x));
    r.seg = float(bk) + bu;
    r.surf = seg_surface(c, bk, bu, L, r.lat, r.slope);
    {
        const float drop = c.pts[size_t(bk)].z - c.pts[size_t(bk) + 1].z;
        r.drop = (c.pts[size_t(bk) + 1].w < 0.0f && drop > 0.05f) ? drop : 0.0f;
    }
    r.sslope = glm::mix((std::max)(seg_slope(c, bk - 1), seg_slope(c, bk)), (std::max)(seg_slope(c, bk), seg_slope(c, bk + 1)), bu);
    r.half_w = glm::mix(std::abs(c.pts[size_t(bk)].w), std::abs(c.pts[size_t(bk) + 1].w), bu);
    // (the boiling pool at a step's foot: the first metres of the segment after a step)
    if (bk > 0 && c.pts[size_t(bk)].w < 0.0f) r.foot = std::exp(-bu * L / 4.0f) * sstep(0.0f, 1.2f, bu * L) * (std::min)((c.pts[size_t(bk) - 1].z - c.pts[size_t(bk)].z) / 1.2f, 1.0f);
    r.depth = glm::mix(depth_pt(c, bk), depth_pt(c, bk + 1), bu);
    r.along = bcum + bu * L;
    r.dir = (b - a) / (std::max)(L, 1e-4f);
    return r;
}

// The channel's half width at the bed on this side (the banks wander in and out by up to a third
// of the width, each side its own way: no straight-walled trench).
static float bank_hw(const At& A) {
    const float side = A.lat >= 0.0f ? 37.0f : 0.0f;
    return A.half_w * (0.85f + 0.4f * n1(A.seg * 0.7f + side));
}
// How far from the centre line the water can reach: the bank's foot, plus the bank (rising 0.55 a
// metre from 0.65 x depth under the surface) up to the surface, plus a margin. Beyond it nothing
// is water; within it the ground decides (it stands above the surface where the bank has risen
// out of the water: carve).
float wet_reach(const At& A) {
    return bank_hw(A) + 0.65f * A.depth / 0.55f + 0.35f;
}

// The channel: a bed `depth` under the surface across the channel's width, banks rising 0.55 m a
// metre and 1 m a metre from 2.5 m out, met by the land in a rounding (a smooth minimum - the
// carve only ever lowers the ground). (As wm_river's valley, smaller.)
// The rounding is at most 4 m but never deeper than half the height of the meeting point above
// the water: a fixed 4 m rounding dug the ground up to 1 m below BOTH the bank and the land, i.e.
// under the stream's surface (a sheet is only 0.22 m deep) - the water then reached out to where
// its footprint stopped and ended there in a straight cut wall, a slab floating over its banks.
// With this rule the ground is never under the surface where both the bank and the land are above
// it, so the water's edge is the bank's.
// THE BANK THAT HOLDS THE WATER (2026-11-26, the owner's "hard straight edges where the stream's
// water stops"): where the land beside the channel lies LOWER than the stream's surface (a stream
// running across a slope: its downhill side; a sheet on steep ground), the carve - which only
// lowers - left that land under the surface, and the water reached out over it to its footprint's
// end (wet_reach) and stopped there in a straight line parallel to the course. Now the channel's
// own bank profile (cap) is built up there to 0.15 m over the surface at the reach and falls off
// 0.5 m a metre beyond it into the land: the water's edge is the bank's, wandering with it. Not at
// the mouth (within 0.2 - 0.6 m of the last point's surface): no dam across the confluence.
// (water_course.glsl wc_bank_lift: keep the two in step.)
static float bank_lift(const Course& c, const At& A, float cap) {
    const float k = sstep(0.2f, 0.6f, A.surf - c.pts.back().z);
    if (k <= 0.0f) return -1e9f;
    const float top = A.surf + 0.15f - 0.5f * (std::max)(A.dist - wet_reach(A) - 1.0f, 0.0f);      // (flat for 1 m past the reach: the ground grid is up to 1 m - bilinear between the bank and the land the edge sank under the water)
    return (std::min)(cap, top) - (1.0f - k) * 1.0f;      // (fading: sunk a metre - under the land)
}
// THE CARVE RUNS OUT INTO THE LAND (2026-11-27, the owner's "a clear cut in the mountain"): the
// valley's side rose about 1 m a metre to some 37 m over the water at the carve's reach (kBoxPad,
// 40 m), and there the carve simply stopped - wherever the land stood higher than that within
// 40 m it was cut down to the side's height and stood up again, sheer, along a line 40 m from the
// course (a 20 m wall west of the old spring, where the course crossed a spur). Now the side
// climbs ever faster from 24 m out (as wm_river's rim), and over the last 6 m the carve fades to
// the land itself: the ground is continuous at the reach whatever the land does there.
// (water_course.glsl wc_stream_carve: the same.)
static float rim_rise(float e) { const float r = (std::max)(e - 24.0f, 0.0f); return 0.25f * r * r; }
static float reach_fade(float dist) { return sstep(kBoxPad - 6.0f, kBoxPad, dist); }
float carve(const Course& c, float h, glm::vec2 wp) {
    const At A = at(c, wp);
    if (!A.on) return h;
    const float hw = bank_hw(A);
    const float e = (std::max)(A.dist - hw, 0.0f);
    const float bed = A.surf - A.depth * (1.0f - 0.35f * sstep(0.5f, 1.0f, A.dist / (std::max)(hw, 0.1f)));
    const float cap = bed + e * 0.55f + (std::max)(e - 2.5f, 0.0f) * 0.45f + rim_rise(e);
    const float kr = std::clamp(2.0f * ((std::min)(cap, h) - A.surf), 0.0f, 4.0f);
    float r = (std::min)(cap, h);
    if (kr >= 1e-4f) {
        const float t = std::clamp(0.5f + 0.5f * (cap - h) / kr, 0.0f, 1.0f);
        r = glm::mix(cap, h, t) - kr * t * (1.0f - t);
    }
    return glm::mix((std::max)(r, bank_lift(c, A, cap)), h, reach_fade(A.dist));
}

// ---- The stream ------------------------------------------------------------------------------
// From where the owner stood on the hillside east of the castle (542, 158; about 93 m) it runs
// south-east down a side valley, through a low saddle (cut a few metres into it) and down the
// steep last slope to the river at about (850, 460) - laid along the lowest ground of a height
// plan (VIBE_DUMP_MAP), with a slow sideways wander for its bends. No road crosses it (the
// castle lane runs on the river's far bank there).
namespace { Course g_stream; }

const Course& stream(float) { return g_stream; }

void build_stream(float wl0, const std::function<float(float, float)>& ground_fn, const std::function<float(float, float)>& level) {
    float wl = wl0;      // (the level where the stream meets the river: set once its mouth is known, below)
    Course& c = g_stream;
    c = Course{};
    {
        t_building = true;
        // (2026-11-27, the owner's "make the river longer": it now rises ~150 m further up, in the
        //  gentle valley that comes down from the north-east (675, 88; about 105 m) to where it began)
        // (2026-11-27, the owner's "a clear cut in the mountain": from (590, 134) the course went on
        //  WEST, up over the spur to the old spring at (540, 148) and down its far side - the water,
        //  which only falls, ran level at the hollow's 79.6 m through ground up to 98 m, an 18 m
        //  trench whose 40 m carve sliced the 140 m hillside beside it into a 20 m wall. Now it
        //  follows the hollow's own valley straight south, the lowest ground of the height plan
        //  (VIBE_LOG_CUT grid): 80 m at (580, 160), 72 at (568, 184), 66 at (592, 208), into the old
        //  course at (586, 238).)
        const glm::vec2 C[] = { {676, 86}, {662, 103}, {641, 117}, {616, 127}, {594, 136}, {581, 153}, {577, 170}, {569, 186}, {574, 200}, {588, 214}, {586, 238}, {610, 247}, {632, 253}, {649, 270},
                                {670, 281}, {700, 288}, {722, 302}, {745, 318}, {765, 335}, {777, 356}, {778, 378}, {793, 396},
                                {812, 413}, {830, 432}, {844, 450} };
        const int nc = int(sizeof(C) / sizeof(C[0]));
        // Catmull-Rom through the control points, as close as the scene block's points allow (about
        // every 5 - 6 m: the bends below are tighter than 8 m points could follow).
        float guide_len = 0.0f;
        for (int i = 0; i + 1 < nc; ++i) guide_len += glm::distance(C[i], C[i + 1]);
        const float spacing = (std::max)(4.0f, guide_len / float(kMaxPts - 8));
        std::vector<glm::vec2> P;
        for (int i = 0; i + 1 < nc; ++i) {
            const glm::vec2 p0 = C[(std::max)(i - 1, 0)], p1 = C[i], p2 = C[i + 1], p3 = C[(std::min)(i + 2, nc - 1)];
            const int steps = (std::max)(1, int(std::round(glm::distance(p1, p2) / spacing)));
            for (int s = 0; s < steps; ++s) {
                const float t = float(s) / float(steps), t2 = t * t, t3 = t2 * t;
                P.push_back(0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3));
            }
        }
        P.push_back(C[nc - 1]);
        // Its bends: along the guide line the stream takes the hollow of the ground - at every
        // point the lowest place within 14 m to either side, held to the previous point's choice
        // (momentum) and pulled toward a gentle S that grows where the slope is steep (a stream
        // does not run straight down the fall line: it swings across it). Smoothed twice.
        {
            const size_t n0 = P.size();
            std::vector<float> off(n0, 0.0f);
            float s = 0.0f, prev_o = 0.0f;
            for (size_t i = 1; i + 1 < n0; ++i) {
                s += glm::distance(P[i - 1], P[i]);
                const glm::vec2 d = glm::normalize(P[i + 1] - P[i - 1]), nrm(-d.y, d.x);
                const float steep = std::abs(ground_fn(P[i - 1].x, P[i - 1].y) - ground_fn(P[i + 1].x, P[i + 1].y)) / (std::max)(glm::distance(P[i - 1], P[i + 1]), 1.0f);
                // (2026-11-27, the owner's "make the river longer with some more bends": the S is tighter and
                //  wider - a 66 m wave with a 29 m one on it, 6.5 - 10 m either way - sought within 20 m and
                //  pulled to harder; it was a 163 m wave of 4 - 11 m within 14 m: the stream ran nearly straight)
                const float S = (6.5f + 3.5f * sstep(0.06f, 0.25f, steep)) * (std::sin(s / 10.5f) + 0.3f * std::sin(s / 4.6f + 1.0f)) * sstep(0.0f, 25.0f, s) * (1.0f - sstep(guide_len - 30.0f, guide_len - 8.0f, s));      // (straight into the confluence)
                float best = 1e9f, bo = 0.0f;
                for (float o = -20.0f; o <= 20.0f; o += 1.0f) {
                    const glm::vec2 q = P[i] + nrm * o;
                    const float score = ground_fn(q.x, q.y) + 0.05f * (o - S) * (o - S) + 0.015f * (o - prev_o) * (o - prev_o);
                    if (score < best) { best = score; bo = o; }
                }
                off[i] = bo; prev_o = bo;
            }
            for (int pass = 0; pass < 1; ++pass) {
                std::vector<float> o2 = off;
                for (size_t i = 1; i + 1 < n0; ++i) o2[i] = 0.25f * off[i - 1] + 0.5f * off[i] + 0.25f * off[i + 1];
                off = o2;
            }
            std::vector<glm::vec2> Q = P;
            for (size_t i = 1; i + 1 < n0; ++i) {
                const glm::vec2 d = glm::normalize(P[i + 1] - P[i - 1]), nrm(-d.y, d.x);
                Q[i] = P[i] + nrm * off[i];
            }
            P = Q;
        }
        // On into the river: on along the last direction until well inside the river's channel.
        {
            const glm::vec2 d = glm::normalize(P.back() - P[P.size() - 2]);
            glm::vec2 q = P.back();
            for (int i = 0; i < 12 && world_river_distance(q.x, q.y) > -4.0f; ++i) { q += d * 4.0f; }
            P.push_back(q);
        }
        if (int(P.size()) > kMaxPts) P.resize(size_t(kMaxPts));
        if (level) wl = level(P.back().x, P.back().y);
        const int n = int(P.size());
        auto ground = [&](glm::vec2 p) { return ground_fn(p.x, p.y); };
        // The profile. Where the ground is steep (> ~8 %) the water FOLLOWS ITS BED - one sloped
        // sheet a few tens of cm deep (the reference game's rule: a rapid or a fall is the same
        // water tilted, running faster); its depth is in water_course.h depth_at. Where the
        // ground is gentle: runs that fall up to 6 %, pools, and small steps (0.3 - 1 m) where a
        // run is not enough. At most two real falls (2 - 3 m), only at a natural ledge (the ground
        // dropping more than 2 m within one segment between gentler stretches). The width
        // wanders (0.75 - 1.25 x), wider in the pool above a step, narrower below it. w < 0 marks
        // a point that ends a step.
        float prev = 1e9f;
        int falls = 0, big_falls = 0; float fall_max = 0.0f;
        std::vector<bool> stepped(size_t(n), false);
        std::vector<float> S(size_t(n), wl), HW(size_t(n), 1.0f), G(size_t(n), 0.0f);
        for (int i = 0; i < n; ++i) {
            const float f = float(i) / float(n - 1);
            // (2026-11-26: 1.5 x wider along its whole course, the owner's "a little wider" - it was
            //  4 - 5 m across at the rapid; everything else - the carve, the reach, the boulders, the
            //  solver's bed, the grass's clearance - follows the half widths)
            HW[size_t(i)] = glm::mix(1.35f, 3.9f, f) * (0.75f + 0.5f * n1(float(i) * 0.43f + 11.0f));
            const float hw = HW[size_t(i)];
            const glm::vec2 d = glm::normalize(P[(std::min)(i + 1, n - 1)] - P[(std::max)(i - 1, 0)]), nrm(-d.y, d.x);
            float gmin = 1e9f;
            // (across the banks and 4 m up and down the course: between two points the ground beside a
            //  bend can lie lower than at either - the water stood above it there, a floating sheet)
            for (float a : { -4.0f, 0.0f, 4.0f })
                for (float o : { -(hw + 9.0f), -(hw + 6.0f), -(hw + 3.0f), -(hw + 1.0f), -hw * 0.5f, 0.0f, hw * 0.5f, hw + 1.0f, hw + 3.0f, hw + 6.0f, hw + 9.0f })
                    gmin = (std::min)(gmin, ground(P[size_t(i)] + d * a + nrm * o));
            G[size_t(i)] = gmin;
        }
        for (int i = 1; i + 1 < n; ++i) G[size_t(i)] = (std::min)(G[size_t(i)], 0.25f * G[size_t(i - 1)] + 0.5f * G[size_t(i)] + 0.25f * G[size_t(i + 1)]);
        for (int i = 0; i < n; ++i) {
            const float g = G[size_t(i)];
            float s;
            if (i == 0) s = g - 0.6f;
            else {
                const float L = (std::max)(glm::distance(P[size_t(i - 1)], P[size_t(i)]), 1.0f);
                const float gs = (G[size_t(i - 1)] - g) / L;                          // the ground's fall here
                const float gs_next = i + 1 < n ? (g - G[size_t(i + 1)]) / (std::max)(glm::distance(P[size_t(i)], P[size_t(i + 1)]), 1.0f) : 0.0f;
                if (gs > 0.25f && G[size_t(i - 1)] - g > 2.0f && gs_next < 0.12f && big_falls < 2 && prev - (g - 0.15f) > 1.8f) {
                    // a ledge: a real fall
                    s = (std::max)(prev - (std::min)(prev - (g - 0.15f), 3.0f), wl);
                    stepped[size_t(i)] = true; ++big_falls;
                } else if (gs > 0.08f) {
                    s = (std::min)(prev, g - 0.12f);                                    // the sheet: the bed lifted
                } else {
                    const float req = g - 0.6f;
                    s = prev - (std::min)(0.06f * L, (std::max)(prev - req, 0.0f));      // the run
                    if (s > req + 0.02f) {                                                // not enough: a small step
                        const uint32_t hh = uint32_t(i) * 2654435761u ^ 0x5EA11u;
                        const float r2 = float((hh >> 4) & 0xFFF) / 4095.0f;
                        s = prev - (std::min)((std::max)(0.3f + 0.7f * r2, prev - req), 1.0f);
                        stepped[size_t(i)] = prev - s > 0.05f;
                    }
                }
                s = (std::max)(s, wl);
            }
            if (i >= n - 2) { s = wl; stepped[size_t(i)] = false; }          // (the confluence: the river's level, no step)
            if (stepped[size_t(i)]) {
                ++falls; fall_max = (std::max)(fall_max, prev - s);
                if (prev - s > 1.0f) {      // (where the waterfalls are: test views, VIBE_TEST_AT_STREAM)
                    float al = 0.0f; for (int j = 1; j <= i; ++j) al += glm::distance(P[size_t(j - 1)], P[size_t(j)]);
                    log::infof("[stream] a fall of %.1f m at point %d, %.0f m down the course (%.0f, %.0f)", double(prev - s), i, double(al), double(P[size_t(i)].x), double(P[size_t(i)].y));
                }
            }
            prev = s;
            S[size_t(i)] = s;
        }
        if (std::getenv("VIBE_LOG_STREAM"))
            for (int i = 0; i < n; ++i)
                log::infof("[stream] point %3d (%.0f, %.0f): lowest ground round it %.2f, centre %.2f, water %.2f", i, double(P[size_t(i)].x), double(P[size_t(i)].y), double(G[size_t(i)]), double(ground(P[size_t(i)])), double(S[size_t(i)]));
        for (int i = 0; i < n; ++i) {
            float hw = HW[size_t(i)];
            if (i + 1 < n && stepped[size_t(i + 1)]) hw *= 1.3f;              // (the pool above a step)
            if (stepped[size_t(i)]) hw *= 0.8f;                               // (the chute below it)
            c.pts.push_back(glm::vec4(P[size_t(i)], S[size_t(i)], stepped[size_t(i)] ? -hw : hw));
        }
        for (int b = 0; b * kSegsPerBox < n - 1; ++b) {
            glm::vec2 lo(1e9f), hi(-1e9f);
            for (int k = b * kSegsPerBox; k <= (std::min)((b + 1) * kSegsPerBox, n - 1); ++k) { lo = (glm::min)(lo, glm::vec2(c.pts[size_t(k)])); hi = (glm::max)(hi, glm::vec2(c.pts[size_t(k)])); }
            c.box.push_back(glm::vec4(lo - glm::vec2(kBoxPad), hi + glm::vec2(kBoxPad)));
        }
        c.ok = n >= 2;
        t_building = false;
        float len = 0.0f;
        for (int i = 1; i < n; ++i) len += glm::distance(glm::vec2(c.pts[size_t(i - 1)]), glm::vec2(c.pts[size_t(i)]));
        log::infof("[stream] %d points, %.0f m, from (%.0f, %.1f, %.0f) to (%.0f, %.1f, %.0f), %d steps (highest %.1f m)", n, double(len),
                   double(c.pts.front().x), double(c.pts.front().z), double(c.pts.front().y), double(c.pts.back().x), double(c.pts.back().z), double(c.pts.back().y), falls, double(fall_max));
    }
}

float carve_stream(float h, glm::vec2 wp, float wl) {
    if (t_building) return h;
    if (!globe::current().home) return h;
    return carve(stream(wl), h, wp);
}

float stream_surface(glm::vec2 wp, float wl) {
    if (t_building || !globe::current().home) return -1e9f;
    const At A = at(stream(wl), wp);
    // Past the reach the surface sinks 1 m per metre for 1.5 m (water_course.glsl wc_stream_surface,
    // keep in step): the water ends where it meets the ground, through the 8 cm depth fade, not in
    // a straight cut along the reach over land at about the surface height.
    const float over = A.on ? A.dist - wet_reach(A) : 1e9f;
    if (over > 1.5f) return -1e9f;
    return A.surf - (std::max)(over, 0.0f);
}

// The current's speed (m/s) at A - the water shader's (terrain_rm_main_water.glsl, the stream's
// speed): the reference game's rule, 0.75 m/s on the level + 3 x the sine of the surface's fall -
// a step's own slope on its face, else the runs' slope smoothed across the points -, slower
// toward the banks.
float flow_speed(const At& A) {
    const float sl = A.drop > 0.01f ? A.slope : A.sslope;
    const float sn = sl / std::sqrt(1.0f + sl * sl);
    return (std::min)(0.75f + 3.0f * sn, 4.5f) * (1.0f - 0.5f * sstep(0.4f, 1.0f, A.dist / (std::max)(A.half_w, 0.1f)));
}

glm::vec2 stream_flow(glm::vec2 wp, float wl) {
    if (t_building || !globe::current().home) return glm::vec2(0.0f);
    const At A = at(stream(wl), wp);
    if (!A.on || A.dist > wet_reach(A)) return glm::vec2(0.0f);
    return A.dir * flow_speed(A);
}

float stream_distance(glm::vec2 wp, float wl) {
    if (t_building || !globe::current().home) return 1e9f;
    const At A = at(stream(wl), wp);
    return A.on ? A.dist - (wet_reach(A) + 0.1f) : 1e9f;
}

}  // namespace vibe::wc

namespace vibe::wc {
// The point `along` metres down the course and `side` metres across it (left of the flow > 0),
// and the water's surface there (the test hook VIBE_TEST_AT_STREAM, game_loop.cpp).
bool point_along(const Course& c, float along, float side, glm::vec2& p, float& surf) {
    if (!c.ok || c.pts.size() < 2) return false;
    float cum = 0.0f;
    for (size_t k = 0; k + 1 < c.pts.size(); ++k) {
        const glm::vec2 a(c.pts[k].x, c.pts[k].y), b(c.pts[k + 1].x, c.pts[k + 1].y);
        const float L = glm::distance(a, b);
        if (cum + L >= along || k + 2 == c.pts.size()) {
            const float u = std::clamp((along - cum) / (std::max)(L, 1e-3f), 0.0f, 1.0f);
            const glm::vec2 d = (b - a) / (std::max)(L, 1e-3f);
            p = glm::mix(a, b, u) + glm::vec2(-d.y, d.x) * side;
            float sl = 0.0f;
            surf = seg_surface(c, int(k), u, L, side, sl);
            return true;
        }
        cum += L;
    }
    return false;
}
}  // namespace vibe::wc
