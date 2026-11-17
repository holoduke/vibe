#include "engine/road.h"

#include "engine/terrain.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace qlike {

namespace {

float smooth01(float e0, float e1, float v) {
    float t = std::clamp((v - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

glm::vec2 catmull(const glm::vec2& p0, const glm::vec2& p1,
                  const glm::vec2& p2, const glm::vec2& p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5f * ((2.0f * p1) + (-p0 + p2) * t +
                   (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                   (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

} // namespace

float Road::distance(float x, float z, float* out_h) const {
    float best = (std::numeric_limits<float>::max)();
    float best_h = 0.0f;
    const glm::vec2 p(x, z);
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        glm::vec2 a(pts[i].x, pts[i].z), b(pts[i + 1].x, pts[i + 1].z);
        glm::vec2 ab = b - a;
        float l2 = glm::dot(ab, ab);
        float t = l2 > 1e-8f ? std::clamp(glm::dot(p - a, ab) / l2, 0.0f, 1.0f) : 0.0f;
        glm::vec2 q = a + ab * t;
        glm::vec2 d = p - q;
        float dd = glm::dot(d, d);
        if (dd < best) {
            best = dd;
            best_h = pts[i].y + (pts[i + 1].y - pts[i].y) * t;
        }
    }
    if (out_h) *out_h = best_h;
    return std::sqrt(best);
}

Road make_castle_road(const Heightmap& hm, float water_level, float gate_h, const glm::vec2* spur_to) {
    Road r;
    // Hand-placed through the terrain: out of the north gate (+Z), west
    // along the land strip between the two lakes, then south-east down
    // the valley. Picked from a top-down height render; anything that
    // dips under the lake is filled into a causeway by the levelling.
    const glm::vec2 ctrl[] = {
        { 0.0f, 13.1f}, { 0.0f, 18.0f}, { -8.0f, 24.0f}, {-22.0f, 29.0f},
        {-36.0f, 36.0f}, {-45.0f, 50.0f}, {-47.0f, 68.0f}, {-40.0f, 86.0f},
        {-22.0f, 102.0f}, { 2.0f, 120.0f}, {20.0f, 148.0f}, {28.0f, 182.0f},
        {26.0f, 225.0f}, {18.0f, 270.0f},
    };
    const int n = static_cast<int>(sizeof(ctrl) / sizeof(ctrl[0]));

    // Densify (Catmull-Rom, ~1.5 m spacing).
    std::vector<glm::vec2> xz;
    for (int i = 0; i + 1 < n; ++i) {
        glm::vec2 p0 = ctrl[(std::max)(0, i - 1)], p1 = ctrl[i];
        glm::vec2 p2 = ctrl[i + 1], p3 = ctrl[(std::min)(n - 1, i + 2)];
        int steps = (std::max)(2, static_cast<int>(glm::length(p2 - p1) / 1.5f));
        for (int s = 0; s < steps; ++s) {
            xz.push_back(catmull(p0, p1, p2, p3, float(s) / float(steps)));
        }
    }
    xz.push_back(ctrl[n - 1]);

    // Level the surface: terrain under the centreline, Gaussian-smoothed
    // over ~+-12 m, kept above the lake, grade-limited to 9 %, and pinned
    // to the castle floor height for the first few metres.
    const size_t m = xz.size();
    std::vector<float> s(m, 0.0f), raw(m), h(m);
    for (size_t i = 1; i < m; ++i) s[i] = s[i - 1] + glm::length(xz[i] - xz[i - 1]);
    for (size_t i = 0; i < m; ++i) raw[i] = hm.sample_world(xz[i].x, xz[i].y);
    const float sigma = 6.0f;
    for (size_t i = 0; i < m; ++i) {
        float acc = 0.0f, wsum = 0.0f;
        for (size_t j = 0; j < m; ++j) {
            float d = s[j] - s[i];
            if (std::abs(d) > 3.0f * sigma) continue;
            float w = std::exp(-0.5f * d * d / (sigma * sigma));
            acc += raw[j] * w;
            wsum += w;
        }
        h[i] = acc / (std::max)(wsum, 1e-6f);
        h[i] = (std::max)(h[i], water_level + 0.7f);
    }
    for (size_t i = 0; i < m; ++i) {
        float pin = 1.0f - smooth01(4.0f, 14.0f, s[i]);
        h[i] = h[i] + (gate_h - 0.03f - h[i]) * pin;
    }
    const float kGrade = 0.09f;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 1; i < m; ++i) {
            float ds = s[i] - s[i - 1];
            h[i] = std::clamp(h[i], h[i - 1] - kGrade * ds, h[i - 1] + kGrade * ds);
        }
        for (size_t i = m - 1; i > 0; --i) {
            float ds = s[i] - s[i - 1];
            if (i - 1 == 0) break;   // keep the gate pinned
            h[i - 1] = std::clamp(h[i - 1], h[i] - kGrade * ds, h[i] + kGrade * ds);
        }
    }

    r.pts.resize(m);
    glm::vec2 lo(1e9f), hi(-1e9f);
    for (size_t i = 0; i < m; ++i) {
        r.pts[i] = glm::vec3(xz[i].x, h[i], xz[i].y);
        lo = glm::min(lo, xz[i]);
        hi = glm::max(hi, xz[i]);
    }
    // Side road: out from the nearest centreline point to `spur_to` and back
    // along the same points (one polyline keeps every consumer - carving,
    // grass, plants, the shader paving - unchanged). The main road is
    // levelled on its own above; the side road starts at exactly the
    // junction's height and climbs or falls from there at no more than 10 %.
    if (spur_to) {
        size_t jn = 0; float best = 1e18f;
        for (size_t i = 0; i < m; ++i) {
            const float d = glm::length(xz[i] - *spur_to);
            if (d < best) { best = d; jn = i; }
        }
        const glm::vec2 a = xz[jn], b = *spur_to;
        const int steps = (std::max)(2, static_cast<int>(glm::length(b - a) / 1.5f));
        const float ds = glm::length(b - a) / float(steps);
        std::vector<glm::vec2> sx(static_cast<size_t>(steps));
        std::vector<float> sr(static_cast<size_t>(steps)), sh(static_cast<size_t>(steps));
        for (int k = 0; k < steps; ++k) {
            sx[size_t(k)] = a + (b - a) * (float(k + 1) / float(steps));
            sr[size_t(k)] = hm.sample_world(sx[size_t(k)].x, sx[size_t(k)].y);
        }
        for (int k = 0; k < steps; ++k) {              // smooth the ground under it (+-6 m)
            float acc = 0.0f; int cnt = 0;
            for (int j = (std::max)(0, k - 4); j <= (std::min)(steps - 1, k + 4); ++j) { acc += sr[size_t(j)]; ++cnt; }
            sh[size_t(k)] = acc / float(cnt);
        }
        float prev = h[jn];
        for (int k = 0; k < steps; ++k) {
            sh[size_t(k)] = std::clamp(sh[size_t(k)], prev - 0.10f * ds, prev + 0.10f * ds);
            prev = sh[size_t(k)];
        }
        std::vector<glm::vec3> ins;
        for (int k = 0; k < steps; ++k) ins.emplace_back(sx[size_t(k)].x, sh[size_t(k)], sx[size_t(k)].y);
        for (int k = steps - 2; k >= 0; --k) ins.emplace_back(sx[size_t(k)].x, sh[size_t(k)], sx[size_t(k)].y);
        ins.emplace_back(a.x, h[jn], a.y);
        r.pts.insert(r.pts.begin() + static_cast<long long>(jn) + 1, ins.begin(), ins.end());
        r.spur_a = static_cast<int>(jn);
        r.spur_tip = static_cast<int>(jn) + steps;
        r.spur_b = static_cast<int>(jn) + 2 * steps;
        lo = glm::min(lo, b); hi = glm::max(hi, b);
    }
    const float margin = r.half_width + r.shoulder + 27.0f;   // max carve blend
    r.bb_min = lo - glm::vec2(margin);
    r.bb_max = hi + glm::vec2(margin);
    return r;
}

void carve_heights(const Road& road, Heightmap& hm) {
    if (road.empty() || hm.heights.empty()) return;
    const int W = hm.width();
    auto to_ix = [&](float w, float o) {
        return std::clamp(static_cast<int>((w - o) / hm.cell), 0, W - 1);
    };
    int x0 = to_ix(road.bb_min.x, hm.origin_x), x1 = to_ix(road.bb_max.x, hm.origin_x);
    int z0 = to_ix(road.bb_min.y, hm.origin_z), z1 = to_ix(road.bb_max.y, hm.origin_z);
    const float flat = road.half_width + road.shoulder;
    const float kMaxBlend = 26.0f;
    for (int iz = z0; iz <= z1; ++iz) {
        float wz = hm.origin_z + iz * hm.cell;
        for (int ix = x0; ix <= x1; ++ix) {
            float wx = hm.origin_x + ix * hm.cell;
            float rh = 0.0f;
            float d = road.distance(wx, wz, &rh);
            if (d > flat + kMaxBlend) continue;
            float& hv = hm.heights[size_t(iz) * W + ix];
            // Falloff widens with the cut/fill depth so embankments stay
            // ~30 deg instead of sheer trench walls through hillsides.
            float blend = std::clamp(std::abs(hv - rh) * 1.8f, road.blend, kMaxBlend);
            if (d > flat + blend) continue;
            float w = 1.0f - smooth01(flat, flat + blend, d);
            // Paving sits 3 cm under the levelled line so the cobble
            // relief reads as set INTO the ground.
            hv = hv + (rh - 0.03f - hv) * w;
        }
    }
}

void clear_grass_mask(const Road& road, std::vector<uint8_t>& mask, int dim,
                      float world_side) {
    if (road.empty() || mask.empty()) return;
    const float texel = world_side / dim;
    auto to_t = [&](float w) {
        return std::clamp(static_cast<int>((w + world_side * 0.5f) / texel), 0, dim - 1);
    };
    int x0 = to_t(road.bb_min.x), x1 = to_t(road.bb_max.x);
    int z0 = to_t(road.bb_min.y), z1 = to_t(road.bb_max.y);
    const float inner = road.half_width + 0.3f;
    const float outer = road.half_width + road.shoulder + 0.8f;
    for (int z = z0; z <= z1; ++z) {
        float wz = (z + 0.5f) * texel - world_side * 0.5f;
        for (int x = x0; x <= x1; ++x) {
            float wx = (x + 0.5f) * texel - world_side * 0.5f;
            float d = road.distance(wx, wz);
            if (d > outer + 1.0f) continue;
            d += 1.6f * road_edge_jitter(wx, wz);       // follow the drawn verge edge
            float keep = smooth01(inner, outer, d);
            uint8_t& r = mask[(size_t(z) * dim + x) * 2];
            r = static_cast<uint8_t>(r * keep);
        }
    }
}

std::vector<glm::vec4> road_gpu_points(const Road& road, int max_pts) {
    std::vector<glm::vec4> out;
    if (road.empty() || max_pts < 2) return out;
    if (road.spur_a >= 0 && road.spur_b < static_cast<int>(road.pts.size())) {
        // Main road before / after the junction re-sampled evenly; the side
        // road with the SAME points out and back, so the two passes lie
        // exactly on each other.
        auto piece = [&](int i0, int i1, int count, bool skip_first) {
            std::vector<float> s(size_t(i1 - i0 + 1), 0.0f);
            for (int i = i0 + 1; i <= i1; ++i)
                s[size_t(i - i0)] = s[size_t(i - i0 - 1)] + glm::length(glm::vec2(road.pts[size_t(i)].x - road.pts[size_t(i - 1)].x,
                                                                                    road.pts[size_t(i)].z - road.pts[size_t(i - 1)].z));
            size_t j = 0;
            for (int k = skip_first ? 1 : 0; k < count; ++k) {
                const float target = s.back() * float(k) / float(count - 1);
                while (j + 2 < s.size() && s[j + 1] < target) ++j;
                const float seg = (std::max)(s[j + 1] - s[j], 1e-6f);
                const float t = std::clamp((target - s[j]) / seg, 0.0f, 1.0f);
                out.emplace_back(road.pts[size_t(i0) + j] + (road.pts[size_t(i0) + j + 1] - road.pts[size_t(i0) + j]) * t, 0.0f);
            }
        };
        const int last = static_cast<int>(road.pts.size()) - 1;
        const int spur_n = 6;                                   // points along the side road (excl. the junction)
        const int main_n = max_pts - 2 * spur_n;
        const float la = float(road.spur_a), lb = float(last - road.spur_b);
        int na = (std::max)(2, static_cast<int>(std::round(float(main_n) * la / (std::max)(la + lb, 1.0f))));
        na = (std::min)(na, main_n - 2);
        piece(0, road.spur_a, na, false);                       // ends exactly on the junction
        std::vector<glm::vec4> sp;
        for (int k = 1; k <= spur_n; ++k) {
            const float fi = float(road.spur_a) + float(road.spur_tip - road.spur_a) * float(k) / float(spur_n);
            const int i = static_cast<int>(fi); const float t = fi - float(i);
            const glm::vec3 p = road.pts[size_t(i)] + (road.pts[size_t((std::min)(i + 1, road.spur_tip))] - road.pts[size_t(i)]) * t;
            sp.emplace_back(p, 0.0f);
        }
        for (const auto& p : sp) out.push_back(p);
        {   // The turn at the end is a small hook, not an exact reversal: the
            // shader builds the paving direction from the bisector of two
            // neighbouring segments, which is zero (-> NaN, a black patch)
            // when they point exactly opposite ways.
            const glm::vec3 d = glm::normalize(glm::vec3(sp[size_t(spur_n - 1)] - sp[size_t(spur_n - 2)]));
            out.emplace_back(glm::vec3(sp[size_t(spur_n - 1)]) + glm::vec3(-d.z, 0.0f, d.x) * 0.6f, 0.0f);
        }
        for (int k = spur_n - 2; k >= 0; --k) out.push_back(sp[size_t(k)]);
        out.emplace_back(road.pts[size_t(road.spur_b)], 0.0f);  // the junction again
        piece(road.spur_b, last, main_n - na, true);
        float acc = 0.0f;
        for (size_t k = 0; k < out.size(); ++k) {
            if (k > 0) acc += glm::length(glm::vec2(out[k].x - out[k - 1].x, out[k].z - out[k - 1].z));
            out[k].w = acc;
        }
        return out;
    }
    const size_t m = road.pts.size();
    if (static_cast<int>(m) <= max_pts) {

        float acc = 0.0f;
        for (size_t k = 0; k < m; ++k) {
            if (k > 0) acc += glm::length(glm::vec2(road.pts[k].x - road.pts[k - 1].x,
                                                    road.pts[k].z - road.pts[k - 1].z));
            out.emplace_back(road.pts[k], acc);
        }
        return out;
    }
    std::vector<float> s(m, 0.0f);
    for (size_t i = 1; i < m; ++i)
        s[i] = s[i - 1] + glm::length(glm::vec2(road.pts[i].x - road.pts[i - 1].x,
                                                road.pts[i].z - road.pts[i - 1].z));
    size_t j = 0;
    for (int k = 0; k < max_pts; ++k) {
        float target = s[m - 1] * float(k) / float(max_pts - 1);
        while (j + 2 < m && s[j + 1] < target) ++j;
        float seg = (std::max)(s[j + 1] - s[j], 1e-6f);
        float t = std::clamp((target - s[j]) / seg, 0.0f, 1.0f);
        out.emplace_back(road.pts[j] + (road.pts[j + 1] - road.pts[j]) * t, 0.0f);
    }
    // .w = cumulative XZ arc length (road-space V coordinate for the shader).
    float acc = 0.0f;
    for (size_t k = 0; k < out.size(); ++k) {
        if (k > 0) acc += glm::length(glm::vec2(out[k].x - out[k - 1].x, out[k].z - out[k - 1].z));
        out[k].w = acc;
    }
    return out;
}

} // namespace qlike
