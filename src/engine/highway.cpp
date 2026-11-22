#include "engine/highway.h"

#include "engine/terrain.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <unordered_map>

#undef min
#undef max

namespace vibe::highway {

namespace {

struct Line {
    std::vector<glm::vec3> p;      // centre line, y = surface height
    std::vector<float>     s;      // metres along, per point
};

std::vector<Line> g_lines;
// Segments by 64 m cell: every cell a segment comes within kGridReach of. Value: line << 20 | segment.
constexpr float kCell = 64.0f, kGridReach = 66.0f;
std::unordered_map<uint64_t, std::vector<uint32_t>> g_grid;
std::atomic<bool> g_ready{false};
std::mutex g_build_mutex;
int   g_junction = -1;             // index into g_lines[0] where the spur leaves
glm::vec4 g_tightest(0.0f);        // the tightest bend of the coast road: where (xyz), its radius (w)

uint64_t cell_key(int cx, int cz) { return (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz); }

void index_line(uint32_t li) {
    const Line& L = g_lines[li];
    for (size_t i = 0; i + 1 < L.p.size(); ++i) {
        const glm::vec2 a(L.p[i].x, L.p[i].z), b(L.p[i + 1].x, L.p[i + 1].z);
        const int x0 = int(std::floor((std::min(a.x, b.x) - kGridReach) / kCell)), x1 = int(std::floor((std::max(a.x, b.x) + kGridReach) / kCell));
        const int z0 = int(std::floor((std::min(a.y, b.y) - kGridReach) / kCell)), z1 = int(std::floor((std::max(a.y, b.y) + kGridReach) / kCell));
        for (int cz = z0; cz <= z1; ++cz)
            for (int cx = x0; cx <= x1; ++cx) g_grid[cell_key(cx, cz)].push_back((li << 20) | uint32_t(i));
    }
}

float sstep(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// One segment's answer for q: squared distance, and where on it.
struct Hit { float d2 = 1e30f; float t = 0.0f; uint32_t li = 0, i = 0; };
void test_segment(const glm::vec2 q, uint32_t li, uint32_t i, Hit& best) {
    const Line& L = g_lines[li];
    const glm::vec2 a(L.p[i].x, L.p[i].z), ab = glm::vec2(L.p[i + 1].x, L.p[i + 1].z) - a;
    const float t = std::clamp(glm::dot(q - a, ab) / std::max(glm::dot(ab, ab), 1e-6f), 0.0f, 1.0f);
    const glm::vec2 dv = q - (a + ab * t);
    const float d2 = glm::dot(dv, dv);
    if (d2 < best.d2) { best.d2 = d2; best.t = t; best.li = li; best.i = i; }
}
void resolve(const std::vector<Line>& lines, const glm::vec2 q, const Hit& b, float& d, float& h, float& s) {
    const Line& L = lines[b.li];
    const glm::vec3 pa = L.p[b.i], pb = L.p[b.i + 1];
    const glm::vec2 ab(pb.x - pa.x, pb.z - pa.z), rel(q.x - pa.x, q.y - pa.z);
    const float side = ab.x * rel.y - ab.y * rel.x;
    d = std::sqrt(b.d2) * (side < 0.0f ? -1.0f : 1.0f);
    h = glm::mix(pa.y, pb.y, b.t);
    s = glm::mix(L.s[b.i], L.s[b.i + 1], b.t);
}

void finish_line(Line& L, float s0) {
    L.s.resize(L.p.size());
    float s = s0;
    for (size_t i = 0; i < L.p.size(); ++i) {
        if (i) s += glm::distance(glm::vec2(L.p[i].x, L.p[i].z), glm::vec2(L.p[i - 1].x, L.p[i - 1].z));
        L.s[i] = s;
    }
}

// A running mean over +-r points, `passes` times (the ends keep their place).
template <typename T>
void smooth(std::vector<T>& v, int r, int passes) {
    const int n = int(v.size());
    if (n < 3) return;
    std::vector<T> tmp(v.size());
    for (int p = 0; p < passes; ++p) {
        for (int i = 0; i < n; ++i) {
            const int rr = std::min({ r, i, n - 1 - i });          // (the window shrinks toward the ends)
            T acc = v[size_t(i)];
            for (int k = 1; k <= rr; ++k) acc += v[size_t(i - k)] + v[size_t(i + k)];
            tmp[size_t(i)] = acc / float(2 * rr + 1);
        }
        v.swap(tmp);
    }
}

// Points `step` apart along a polyline.
std::vector<glm::vec2> resample(const std::vector<glm::vec2>& in, float step) {
    std::vector<glm::vec2> out;
    if (in.size() < 2) return out;
    out.push_back(in[0]);
    float carry = 0.0f;
    for (size_t i = 0; i + 1 < in.size(); ++i) {
        const glm::vec2 a = in[i], b = in[i + 1];
        const float len = glm::distance(a, b);
        float t = step - carry;
        while (t <= len) { out.push_back(glm::mix(a, b, t / std::max(len, 1e-6f))); t += step; }
        carry = len - (t - step);
    }
    return out;
}

// The shoreline's contour, traced from `start` one way (dir = +1 / -1) in 25 m steps.
std::vector<glm::vec2> trace(glm::vec2 start, float v0, float dir, int max_steps, float max_range) {
    auto V = [](glm::vec2 p) { return world_continent_home(p.x, p.y); };
    auto grad = [&](glm::vec2 p) {
        const float e = 8.0f;
        return glm::vec2(V(p + glm::vec2(e, 0.0f)) - V(p - glm::vec2(e, 0.0f)), V(p + glm::vec2(0.0f, e)) - V(p - glm::vec2(0.0f, e))) / (2.0f * e);
    };
    std::vector<glm::vec2> out;
    glm::vec2 p = start;
    for (int n = 0; n < max_steps; ++n) {
        glm::vec2 g = grad(p);
        float g2 = glm::dot(g, g);
        if (g2 < 1e-16f) break;
        p += dir * glm::vec2(-g.y, g.x) / std::sqrt(g2) * 25.0f;
        for (int k = 0; k < 2; ++k) {                              // back onto the contour
            g = grad(p); g2 = glm::dot(g, g);
            if (g2 < 1e-16f) break;
            const glm::vec2 corr = (V(p) - v0) * g / g2;
            const float cl = glm::length(corr);
            p -= cl > 20.0f ? corr * (20.0f / cl) : corr;
        }
        if (glm::distance(p, start) > max_range) break;
        if (std::max(std::abs(p.x), std::abs(p.y)) < 1150.0f) break;          // (not into the home tile: its ground is a bake)
        out.push_back(p);
    }
    return out;
}

}  // namespace

bool ready() { return g_ready.load(std::memory_order_acquire); }

void add_line(const std::vector<glm::vec3>& pts, float s0) {
    if (pts.size() < 2) return;
    Line L;
    L.p = pts;
    finish_line(L, s0);
    g_lines.push_back(std::move(L));
    index_line(uint32_t(g_lines.size() - 1));
}

void build(float water_level) {
    std::lock_guard<std::mutex> lk(g_build_mutex);
    if (g_ready.load()) return;
    // ---- the coast road ----
    // The landscape formula's land / sea value is 0 at the waterline and rises inland. This
    // coast is steep - the hills stand 40 to 90 m over the water within a hundred metres of it -
    // so the road is a corniche: it follows the contour just above the waterline, benched into
    // the foot of the slope, 8 to 15 m over the sea. Found by walking east from the launch pad.
    const float v0 = 0.005f;
    const glm::vec2 pad(kLaunchSiteX, kLaunchSiteZ);
    auto V = [](glm::vec2 p) { return world_continent_home(p.x, p.y); };
    glm::vec2 a = pad, b = pad;
    bool found = false;
    for (int i = 1; i <= 400 && !found; ++i) {
        b = pad + glm::vec2(20.0f * float(i), 0.0f);
        if (V(b) < v0) found = true; else a = b;
    }
    if (found) {
        for (int i = 0; i < 24; ++i) { const glm::vec2 m = 0.5f * (a + b); if (V(m) < v0) b = m; else a = m; }
        const glm::vec2 start = 0.5f * (a + b);
        std::vector<glm::vec2> north = trace(start, v0, +1.0f, 640, 13000.0f), south = trace(start, v0, -1.0f, 640, 13000.0f);
        std::vector<glm::vec2> raw(north.rbegin(), north.rend());
        raw.push_back(start);
        raw.insert(raw.end(), south.begin(), south.end());
        if (raw.size() > 40) {
            // A highway's curves, not a shoreline's: the coves and headlands shorter than a few
            // hundred metres are cut across (fill over the water, a cutting through the land).
            smooth(raw, 9, 4);
            std::vector<glm::vec2> pts = resample(raw, 20.0f);
            smooth(pts, 3, 2);
            // No bend tighter than kMinRadius: where the line turns harder than that, that part of
            // it is relaxed (each point toward the middle of its neighbours) until it does not.
            const float kMinRadius = 160.0f;
            auto radius_at = [&](size_t i) {
                const glm::vec2 a = pts[i - 1], b = pts[i], c = pts[i + 1];
                const float area2 = std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
                return area2 < 1e-6f ? 1e9f : glm::distance(a, b) * glm::distance(b, c) * glm::distance(c, a) / (2.0f * area2);
            };
            for (int it = 0; it < 400; ++it) {
                bool tight = false;
                std::vector<glm::vec2> nx = pts;
                for (size_t i = 1; i + 1 < pts.size(); ++i) {
                    if (radius_at(i) >= kMinRadius) continue;
                    tight = true;
                    for (size_t k = (i > 6 ? i - 6 : 1); k <= i + 6 && k + 1 < pts.size(); ++k)
                        nx[k] = glm::mix(pts[k], 0.5f * (pts[k - 1] + pts[k + 1]), 0.5f);
                }
                if (!tight) break;
                pts.swap(nx);
            }
            pts = resample(pts, 20.0f);
            // The surface follows the land: the natural ground under the line, with only the bumps
            // shorter than a car smoothed off, never lower than 3 m over the water (a cove is
            // crossed on fill) - and no steeper than a road can be (kGrade). Where the land is
            // steeper than that along the line, the road climbs at its limit: the top of the
            // rise is cut into and the foot filled, by equal measure.
            const float kGrade = 0.10f;                           // 10 %
            std::vector<float> hs(pts.size());
            for (size_t i = 0; i < pts.size(); ++i)
                hs[i] = std::max(world_ground_height_natural(pts[i].x, pts[i].y, water_level, 9), water_level + 3.0f);
            smooth(hs, 2, 2);
            {
                const float g = kGrade * 20.0f;                   // (points are 20 m apart)
                std::vector<float> up = hs, dn = hs;
                // `up`: the lowest profile at or above the land that keeps the grade; `dn`: the
                // highest one at or below it. Their mean keeps the grade too.
                for (size_t i = 1; i < up.size(); ++i) up[i] = std::max(up[i], up[i - 1] - g);
                for (size_t i = up.size() - 1; i-- > 0;) up[i] = std::max(up[i], up[i + 1] - g);
                for (size_t i = 1; i < dn.size(); ++i) dn[i] = std::min(dn[i], dn[i - 1] + g);
                for (size_t i = dn.size() - 1; i-- > 0;) dn[i] = std::min(dn[i], dn[i + 1] + g);
                for (size_t i = 0; i < hs.size(); ++i) hs[i] = std::max(0.5f * (up[i] + dn[i]), water_level + 3.0f);
            }
            smooth(hs, 3, 2);                                     // (crests and dips rounded: a vertical curve, not a kink)
            // Past the launch pad it runs at the pad's own level (it passes along its seaward edge).
            for (size_t i = 0; i < pts.size(); ++i)
                hs[i] = glm::mix(kLaunchSiteH, hs[i], sstep(130.0f, 420.0f, glm::distance(pts[i], pad)));
            // The line: a smooth curve through these points (Catmull-Rom), in 4 m pieces - with 20 m
            // pieces a bend was a polygon: the lane lines and the rail had a corner every 20 m.
            Line L;
            {
                if (hs.size() > pts.size()) hs.resize(pts.size());
                const size_t n = pts.size();
                auto P3 = [&](size_t i) { return glm::vec3(pts[i].x, hs[i], pts[i].y); };
                for (size_t i = 0; i + 1 < n; ++i) {
                    const glm::vec3 p0 = P3(i ? i - 1 : 0), p1 = P3(i), p2 = P3(i + 1), p3 = P3(i + 2 < n ? i + 2 : n - 1);
                    for (int k = 0; k < 5; ++k) {
                        const float t = float(k) / 5.0f, t2 = t * t, t3 = t2 * t;
                        L.p.push_back(0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3));
                    }
                }
                L.p.push_back(P3(n - 1));
            }
            finish_line(L, 0.0f);
            {   // (for the log: how tight the tightest bend is, and where)
                float rmin = 1e9f; size_t at = 0;
                for (size_t i = 5; i + 5 < L.p.size(); i += 5) {
                    const glm::vec2 a(L.p[i - 5].x, L.p[i - 5].z), b(L.p[i].x, L.p[i].z), c(L.p[i + 5].x, L.p[i + 5].z);
                    const float area2 = std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
                    const float r = area2 < 1e-6f ? 1e9f : glm::distance(a, b) * glm::distance(b, c) * glm::distance(c, a) / (2.0f * area2);
                    if (r < rmin) { rmin = r; at = i; }
                }
                g_tightest = glm::vec4(L.p[at], rmin);
            }
            g_lines.push_back(std::move(L));
            // ---- the spur to the launch pad ----
            const Line& H = g_lines[0];
            size_t jn = 0; float best = 1e30f;
            for (size_t i = 0; i < H.p.size(); ++i) {
                const float d2 = glm::dot(glm::vec2(H.p[i].x, H.p[i].z) - pad, glm::vec2(H.p[i].x, H.p[i].z) - pad);
                if (d2 < best) { best = d2; jn = i; }
            }
            g_junction = int(jn);
            const glm::vec2 J(H.p[jn].x, H.p[jn].z);
            const float len = glm::distance(J, pad);
            if (len > 160.0f) {
                const glm::vec2 u = (pad - J) / len;
                const float run = len - 96.0f;                 // (to the edge of the pad's level ground)
                Line S;
                const int n = std::max(2, int(std::ceil(run / 20.0f)));
                for (int i = 0; i <= n; ++i) {
                    const float t = float(i) / float(n);
                    const glm::vec2 q = J + u * (run * t);
                    S.p.push_back(glm::vec3(q.x, glm::mix(H.p[jn].y, kLaunchSiteH, sstep(0.0f, 1.0f, t)), q.y));
                }
                finish_line(S, H.s[jn]);
                g_lines.push_back(std::move(S));
            }
        }
    }
    for (uint32_t li = 0; li < g_lines.size(); ++li) index_line(li);
    g_ready.store(true, std::memory_order_release);
}

bool nearest(float x, float z, float reach, float& d, float& h, float& s) {
    if (!ready() || g_lines.empty()) return false;
    const auto it = g_grid.find(cell_key(int(std::floor(x / kCell)), int(std::floor(z / kCell))));
    if (it == g_grid.end()) return false;
    const glm::vec2 q(x, z);
    Hit best;
    for (uint32_t key : it->second) test_segment(q, key >> 20, key & 0xFFFFFu, best);
    if (best.d2 > reach * reach) return false;
    resolve(g_lines, q, best, d, h, s);
    return true;
}

float distance(float x, float z) {
    float d, h, s;
    return nearest(x, z, kFieldMaxD, d, h, s) ? std::abs(d) : 1e9f;
}

float ground(float h_nat, float x, float z) {
    float d, h, s;
    if (!nearest(x, z, kBlend, d, h, s)) return h_nat;
    return glm::mix(h_nat, h, 1.0f - sstep(kVerge, kBlend, std::abs(d)));
}

bool bake(glm::vec2 center, std::vector<uint8_t>& A, std::vector<uint8_t>& B, glm::vec4& bbox, const globe::Ring& ring) {
    const int N = kFieldN;
    A.assign(size_t(N) * N * 4, 0);
    B.assign(size_t(N) * N * 4, 255);
    bbox = glm::vec4(0.0f);
    if (!ready() || g_lines.empty()) return false;
    // The lines in the coordinates the field is laid out in.
    std::vector<Line> carried;
    if (ring.on) {
        carried = g_lines;
        const glm::dmat2 Ji = glm::inverse(ring.J);
        for (Line& L : carried)
            for (glm::vec3& p : L.p) {
                const glm::dvec2 c = ring.wp0 + Ji * (glm::dvec2(p.x, p.z) - ring.h0);
                p.x = float(c.x); p.z = float(c.y);
            }
    }
    const std::vector<Line>& lines = ring.on ? carried : g_lines;
    const glm::vec2 org = center - glm::vec2(0.5f * kFieldSide);          // (texel i's centre: org + (i + 0.5) * cell)
    std::vector<float> D2(size_t(N) * N, 1e30f);
    std::vector<uint32_t> W(size_t(N) * N, 0);                            // the winning segment (line << 20 | i) + 1
    glm::vec2 lo(1e30f), hi(-1e30f);
    float s_near = 0.0f, s_best = 1e30f;
    bool any = false;
    for (uint32_t li = 0; li < lines.size(); ++li) {
        const Line& L = lines[li];
        for (uint32_t i = 0; i + 1 < L.p.size(); ++i) {
            const glm::vec2 a(L.p[i].x, L.p[i].z), b(L.p[i + 1].x, L.p[i + 1].z);
            const glm::vec2 mn = glm::min(a, b) - glm::vec2(kFieldMaxD), mx = glm::max(a, b) + glm::vec2(kFieldMaxD);
            if (mx.x < org.x || mx.y < org.y || mn.x > org.x + kFieldSide || mn.y > org.y + kFieldSide) continue;
            any = true;
            lo = glm::min(lo, glm::min(a, b)); hi = glm::max(hi, glm::max(a, b));
            const float dc = glm::dot(a - center, a - center);
            if (li == 0 && dc < s_best) { s_best = dc; s_near = L.s[i]; }
            const int i0 = std::max(0, int(std::floor((mn.x - org.x) / kFieldCell))), i1 = std::min(N - 1, int(std::floor((mx.x - org.x) / kFieldCell)));
            const int j0 = std::max(0, int(std::floor((mn.y - org.y) / kFieldCell))), j1 = std::min(N - 1, int(std::floor((mx.y - org.y) / kFieldCell)));
            const glm::vec2 ab = b - a;
            const float inv = 1.0f / std::max(glm::dot(ab, ab), 1e-6f);
            for (int j = j0; j <= j1; ++j)
                for (int ii = i0; ii <= i1; ++ii) {
                    const glm::vec2 q = org + (glm::vec2(float(ii), float(j)) + 0.5f) * kFieldCell;
                    const float t = std::clamp(glm::dot(q - a, ab) * inv, 0.0f, 1.0f);
                    const glm::vec2 dv = q - (a + ab * t);
                    const float d2 = glm::dot(dv, dv);
                    const size_t k = size_t(j) * N + size_t(ii);
                    if (d2 < D2[k]) { D2[k] = d2; W[k] = ((li << 20) | i) + 1u; }
                }
        }
    }
    if (!any) return false;
    const float s_ref = std::round(s_near / 12.0f) * 12.0f;               // (a whole number of dash periods: the pattern keeps its place)
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const size_t k = size_t(j) * N + size_t(i);
            float d = kFieldMaxD, h = 0.0f, s = s_ref;
            if (W[k]) {
                Hit hit; hit.d2 = D2[k]; hit.li = (W[k] - 1u) >> 20; hit.i = (W[k] - 1u) & 0xFFFFFu;
                const glm::vec2 q = org + (glm::vec2(float(i), float(j)) + 0.5f) * kFieldCell;
                const Line& L = lines[hit.li];
                const glm::vec2 a(L.p[hit.i].x, L.p[hit.i].z), ab = glm::vec2(L.p[hit.i + 1].x, L.p[hit.i + 1].z) - a;
                hit.t = std::clamp(glm::dot(q - a, ab) / std::max(glm::dot(ab, ab), 1e-6f), 0.0f, 1.0f);
                resolve(lines, q, hit, d, h, s);
                d = std::clamp(d, -kFieldMaxD, kFieldMaxD);
            }
            const uint32_t dv = uint32_t(std::clamp((d + kFieldMaxD) / (2.0f * kFieldMaxD), 0.0f, 1.0f) * 65535.0f + 0.5f);
            const uint32_t hv = uint32_t(std::clamp((h - kFieldHLo) / kFieldHSpan, 0.0f, 1.0f) * 65535.0f + 0.5f);
            const uint32_t sv = uint32_t(std::clamp((s - s_ref) * 1000.0f + 8388608.0f, 0.0f, 16777215.0f));
            uint8_t* pa = &A[k * 4]; uint8_t* pb = &B[k * 4];
            pa[0] = uint8_t(dv >> 8); pa[1] = uint8_t(dv); pa[2] = uint8_t(hv >> 8); pa[3] = uint8_t(hv);
            pb[0] = uint8_t(sv >> 16); pb[1] = uint8_t(sv >> 8); pb[2] = uint8_t(sv); pb[3] = 255;
        }
    // Where the shader reads the field: round the lines, inside the window (a margin keeps the
    // bilinear read off the window's edge).
    const float m = 3.0f * kFieldCell;
    lo = glm::max(lo - glm::vec2(kBlend + 2.0f), org + glm::vec2(m));
    hi = glm::min(hi + glm::vec2(kBlend + 2.0f), org + glm::vec2(kFieldSide - m));
    if (hi.x <= lo.x || hi.y <= lo.y) return false;
    bbox = glm::vec4(lo, hi);
    return true;
}

int line_count() { return ready() ? int(g_lines.size()) : 0; }
const std::vector<glm::vec3>& line_points(int line) { return g_lines[size_t(line)].p; }

glm::vec4 tightest_bend() { return g_tightest; }

bool point_along(float metres, glm::vec3& pos, glm::vec2& dir) {
    if (!ready() || g_lines.empty() || g_junction < 0) return false;
    const Line& H = g_lines[0];
    const float target = H.s[size_t(g_junction)] + metres;
    size_t i = 0;
    while (i + 2 < H.p.size() && H.s[i + 1] < target) ++i;
    const float t = std::clamp((target - H.s[i]) / std::max(H.s[i + 1] - H.s[i], 1e-3f), 0.0f, 1.0f);
    pos = glm::mix(H.p[i], H.p[i + 1], t);
    dir = glm::normalize(glm::vec2(H.p[i + 1].x - H.p[i].x, H.p[i + 1].z - H.p[i].z));
    return true;
}

}  // namespace vibe::highway
