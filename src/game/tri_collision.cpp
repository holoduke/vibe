#include "game/tri_collision.h"

#include <algorithm>
#include <cmath>

namespace vibe::tricol {

void Mesh::add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    v.push_back(a); v.push_back(b); v.push_back(c);
    const glm::vec3 cr = glm::cross(b - a, c - a);
    const float l = glm::length(cr);
    n.push_back(l > 1e-12f ? cr / l : glm::vec3(0.0f));
    lo = glm::min(lo, glm::min(a, glm::min(b, c)));
    hi = glm::max(hi, glm::max(a, glm::max(b, c)));
}

void Mesh::build(float cell_size) {
    cell = cell_size;
    start.clear(); items.clear();
    if (n.empty()) { dim = glm::ivec2(0); return; }
    dim = glm::ivec2(std::max(1, int(std::ceil((hi.x - lo.x) / cell)) + 1), std::max(1, int(std::ceil((hi.z - lo.z) / cell)) + 1));
    auto range = [&](size_t t, glm::ivec2& c0, glm::ivec2& c1) {
        const glm::vec3 &a = v[t * 3], &b = v[t * 3 + 1], &c = v[t * 3 + 2];
        const float x0 = std::min({ a.x, b.x, c.x }), x1 = std::max({ a.x, b.x, c.x });
        const float z0 = std::min({ a.z, b.z, c.z }), z1 = std::max({ a.z, b.z, c.z });
        c0 = glm::clamp(glm::ivec2(int((x0 - lo.x) / cell), int((z0 - lo.z) / cell)), glm::ivec2(0), dim - 1);
        c1 = glm::clamp(glm::ivec2(int((x1 - lo.x) / cell), int((z1 - lo.z) / cell)), glm::ivec2(0), dim - 1);
    };
    std::vector<uint32_t> count(size_t(dim.x) * size_t(dim.y) + 1, 0u);
    for (size_t t = 0; t < n.size(); ++t) {
        glm::ivec2 c0, c1; range(t, c0, c1);
        for (int z = c0.y; z <= c1.y; ++z) for (int x = c0.x; x <= c1.x; ++x) ++count[size_t(z) * size_t(dim.x) + size_t(x)];
    }
    start.assign(count.size(), 0u);
    for (size_t i = 1; i < count.size(); ++i) start[i] = start[i - 1] + count[i - 1];
    items.resize(start.back());
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    for (size_t t = 0; t < n.size(); ++t) {
        glm::ivec2 c0, c1; range(t, c0, c1);
        for (int z = c0.y; z <= c1.y; ++z) for (int x = c0.x; x <= c1.x; ++x) items[fill[size_t(z) * size_t(dim.x) + size_t(x)]++] = uint32_t(t);
    }
}

glm::vec3 closest_on_triangle(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    // (Ericson, Real-Time Collision Detection, 5.1.5)
    const glm::vec3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const glm::vec3 bp = p - b;
    const float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return a + ab * (d1 / (d1 - d3));
    const glm::vec3 cp = p - c;
    const float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return a + ac * (d2 / (d2 - d6));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    const float den = 1.0f / (va + vb + vc);
    return a + ab * (vb * den) + ac * (vc * den);
}

namespace {
// The nearest points of segments p1..q1 and p2..q2 (Ericson 5.1.9). Returns the squared distance.
float segment_segment(const glm::vec3& p1, const glm::vec3& q1, const glm::vec3& p2, const glm::vec3& q2, glm::vec3& c1, glm::vec3& c2) {
    const glm::vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const float a = glm::dot(d1, d1), e = glm::dot(d2, d2), f = glm::dot(d2, r);
    float s, t;
    if (a <= 1e-12f && e <= 1e-12f) { c1 = p1; c2 = p2; return glm::dot(c1 - c2, c1 - c2); }
    if (a <= 1e-12f) { s = 0.0f; t = glm::clamp(f / e, 0.0f, 1.0f); }
    else {
        const float c = glm::dot(d1, r);
        if (e <= 1e-12f) { t = 0.0f; s = glm::clamp(-c / a, 0.0f, 1.0f); }
        else {
            const float b = glm::dot(d1, d2), den = a * e - b * b;
            s = den != 0.0f ? glm::clamp((b * f - c * e) / den, 0.0f, 1.0f) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0.0f) { t = 0.0f; s = glm::clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1.0f) { t = 1.0f; s = glm::clamp((b - c) / a, 0.0f, 1.0f); }
        }
    }
    c1 = p1 + d1 * s; c2 = p2 + d2 * t;
    return glm::dot(c1 - c2, c1 - c2);
}
}  // namespace

float segment_triangle(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& a, const glm::vec3& b,
                       const glm::vec3& c, glm::vec3& s, glm::vec3& t) {
    // The segment through the triangle: distance 0 at the crossing.
    const glm::vec3 nrm = glm::cross(b - a, c - a);
    const float d0 = glm::dot(nrm, p0 - a), d1 = glm::dot(nrm, p1 - a);
    if ((d0 <= 0.0f && d1 >= 0.0f) || (d0 >= 0.0f && d1 <= 0.0f)) {
        if (d0 != d1) {
            const glm::vec3 x = p0 + (p1 - p0) * (d0 / (d0 - d1));
            const glm::vec3 q = closest_on_triangle(x, a, b, c);
            if (glm::dot(q - x, q - x) < 1e-12f) { s = x; t = x; return 0.0f; }
        }
    }
    // Otherwise the nearest pair is an end of the segment against the face, or the segment
    // against one of the three edges.
    float best = 1e30f;
    auto take = [&](const glm::vec3& ps, const glm::vec3& pt) {
        const float d = glm::dot(ps - pt, ps - pt);
        if (d < best) { best = d; s = ps; t = pt; }
    };
    take(p0, closest_on_triangle(p0, a, b, c));
    take(p1, closest_on_triangle(p1, a, b, c));
    glm::vec3 cs, ct;
    segment_segment(p0, p1, a, b, cs, ct); take(cs, ct);
    segment_segment(p0, p1, b, c, cs, ct); take(cs, ct);
    segment_segment(p0, p1, c, a, cs, ct); take(cs, ct);
    return best;
}

Contact resolve_capsule(const std::vector<const Mesh*>& meshes, glm::vec3& a, glm::vec3& b, float r,
                        glm::vec3& vel, float walk_ny, int iters) {
    Contact out;
    for (int it = 0; it < iters; ++it) {
        bool any = false;
        const glm::vec3 lo = glm::min(a, b) - glm::vec3(r), hi = glm::max(a, b) + glm::vec3(r);
        for (const Mesh* m : meshes) {
            if (!m || m->dim.x == 0) continue;
            if (hi.x < m->lo.x || lo.x > m->hi.x || hi.y < m->lo.y || lo.y > m->hi.y || hi.z < m->lo.z || lo.z > m->hi.z) continue;
            const glm::ivec2 c0 = glm::clamp(glm::ivec2(int((lo.x - m->lo.x) / m->cell), int((lo.z - m->lo.z) / m->cell)), glm::ivec2(0), m->dim - 1);
            const glm::ivec2 c1 = glm::clamp(glm::ivec2(int((hi.x - m->lo.x) / m->cell), int((hi.z - m->lo.z) / m->cell)), glm::ivec2(0), m->dim - 1);
            for (int cz = c0.y; cz <= c1.y; ++cz)
                for (int cx = c0.x; cx <= c1.x; ++cx) {
                    const size_t ci = size_t(cz) * size_t(m->dim.x) + size_t(cx);
                    for (uint32_t k = m->start[ci]; k < m->start[ci + 1]; ++k) {
                        const uint32_t tri = m->items[k];
                        const glm::vec3 &ta = m->v[size_t(tri) * 3], &tb = m->v[size_t(tri) * 3 + 1], &tc = m->v[size_t(tri) * 3 + 2];
                        if (std::max({ ta.y, tb.y, tc.y }) < lo.y || std::min({ ta.y, tb.y, tc.y }) > hi.y) continue;
                        glm::vec3 s, t;
                        const float d2 = segment_triangle(a, b, ta, tb, tc, s, t);
                        if (d2 >= r * r) continue;
                        const float d = std::sqrt(d2);
                        glm::vec3 dir;
                        if (d > 1e-5f) dir = (s - t) / d;
                        else {
                            // (the segment lies in the face: out along its normal, to the side the
                            //  capsule's middle is on)
                            dir = m->n[tri];
                            if (glm::dot(dir, 0.5f * (a + b) - ta) < 0.0f) dir = -dir;
                            if (glm::dot(dir, dir) < 0.5f) continue;
                        }
                        const float pen = r - d;
                        glm::vec3 push;
                        if (dir.y >= walk_ny) {
                            // standing on it: straight up, no sliding down the slope
                            push = glm::vec3(0.0f, std::min(pen / dir.y, r), 0.0f);
                            out.ground = true;
                            out.ground_n = m->n[tri].y > 0.0f ? m->n[tri] : dir;
                            if (vel.y < 0.0f) vel.y = 0.0f;
                        } else {
                            push = dir * pen;
                            const float vn = glm::dot(vel, dir);
                            if (vn < 0.0f) vel -= dir * vn;
                        }
                        a += push; b += push;
                        out.push += push;
                        out.touched = true;
                        any = true;
                    }
                }
        }
        if (!any) break;
    }
    return out;
}

Round round_in_box(const glm::vec3& lo, const glm::vec3& hi) {
    Round o;
    o.r = 0.5f * std::min(hi.x - lo.x, hi.z - lo.z);
    const glm::vec2 c(0.5f * (lo.x + hi.x), 0.5f * (lo.z + hi.z));
    o.a = glm::vec3(c.x, lo.y, c.y);
    o.b = glm::vec3(c.x, std::max(hi.y - o.r, lo.y), c.y);
    return o;
}

void resolve_capsule_rounds(const std::vector<Round>& rounds, glm::vec3& a, glm::vec3& b, float r,
                            glm::vec3& vel, float walk_ny, Contact& out, int iters) {
    for (int it = 0; it < iters; ++it) {
        bool any = false;
        for (const Round& o : rounds) {
            const float rr = r + o.r;
            if (std::abs(a.x - o.a.x) > rr || std::abs(a.z - o.a.z) > rr) continue;
            if (std::min(a.y, b.y) > o.b.y + rr || std::max(a.y, b.y) < o.a.y - rr) continue;
            glm::vec3 s, t;
            const float d2 = segment_segment(a, b, o.a, o.b, s, t);
            if (d2 >= rr * rr) continue;
            const float d = std::sqrt(d2);
            glm::vec3 dir = d > 1e-5f ? (s - t) / d : glm::vec3(0.0f, 1.0f, 0.0f);
            const float pen = rr - d;
            glm::vec3 push;
            if (dir.y >= walk_ny) {
                push = glm::vec3(0.0f, std::min(pen / dir.y, r), 0.0f);
                out.ground = true;
                out.ground_n = dir;
                if (vel.y < 0.0f) vel.y = 0.0f;
            } else {
                // (round the side: along the horizontal only when it is a wall, so a body that
                //  brushes a stone glides round it and is not lifted or pressed down)
                if (std::abs(dir.y) < 0.35f) { dir.y = 0.0f; const float l = glm::length(dir); dir = l > 1e-5f ? dir / l : glm::vec3(1.0f, 0.0f, 0.0f); }
                push = dir * pen;
                const float vn = glm::dot(vel, dir);
                if (vn < 0.0f) vel -= dir * vn;
            }
            a += push; b += push;
            out.push += push;
            out.touched = true;
            any = true;
        }
        if (!any) break;
    }
}

bool floor_below_rounds(const std::vector<Round>& rounds, float x, float z, float y_hi, float y_lo,
                        float walk_ny, float& y, glm::vec3& normal) {
    bool found = false;
    for (const Round& o : rounds) {
        const float dx = x - o.b.x, dz = z - o.b.z, d2 = dx * dx + dz * dz;
        if (d2 >= o.r * o.r) continue;
        const float up = std::sqrt(o.r * o.r - d2);
        if (up < walk_ny * o.r) continue;                         // (too steep there: the side)
        const float h = o.b.y + up;
        if (h > y_hi || h < y_lo) continue;
        if (!found || h > y) { y = h; normal = glm::vec3(dx, up, dz) / o.r; found = true; }
    }
    return found;
}

bool floor_below(const std::vector<const Mesh*>& meshes, float x, float z, float y_hi, float y_lo,
                 float walk_ny, float& y, glm::vec3& normal) {
    bool found = false;
    for (const Mesh* m : meshes) {
        if (!m || m->dim.x == 0) continue;
        if (x < m->lo.x || x > m->hi.x || z < m->lo.z || z > m->hi.z || y_lo > m->hi.y || y_hi < m->lo.y) continue;
        const glm::ivec2 c = glm::clamp(glm::ivec2(int((x - m->lo.x) / m->cell), int((z - m->lo.z) / m->cell)), glm::ivec2(0), m->dim - 1);
        const size_t ci = size_t(c.y) * size_t(m->dim.x) + size_t(c.x);
        for (uint32_t k = m->start[ci]; k < m->start[ci + 1]; ++k) {
            const uint32_t tri = m->items[k];
            const glm::vec3& nn = m->n[tri];
            if (std::abs(nn.y) < walk_ny) continue;                  // (either winding: a face is a face)
            const glm::vec3 &ta = m->v[size_t(tri) * 3], &tb = m->v[size_t(tri) * 3 + 1], &tc = m->v[size_t(tri) * 3 + 2];
            // barycentric in x / z
            const float det = (tb.z - tc.z) * (ta.x - tc.x) + (tc.x - tb.x) * (ta.z - tc.z);
            if (std::abs(det) < 1e-12f) continue;
            const float l0 = ((tb.z - tc.z) * (x - tc.x) + (tc.x - tb.x) * (z - tc.z)) / det;
            const float l1 = ((tc.z - ta.z) * (x - tc.x) + (ta.x - tc.x) * (z - tc.z)) / det;
            const float l2 = 1.0f - l0 - l1;
            if (l0 < -1e-5f || l1 < -1e-5f || l2 < -1e-5f) continue;
            const float h = l0 * ta.y + l1 * tb.y + l2 * tc.y;
            if (h > y_hi || h < y_lo) continue;
            if (!found || h > y) { y = h; normal = nn.y > 0.0f ? nn : -nn; found = true; }
        }
    }
    return found;
}

}  // namespace vibe::tricol
