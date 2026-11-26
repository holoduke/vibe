#pragma once
// PANES: finding the sheets of glass in a mesh, and the panes a sheet is divided into.
//
// A glass mesh part is a bag of triangles (one part for all the glass of a building). What
// breaks is a PANE: one piece of glass in its frame. They are found in two steps, from the
// geometry alone - nothing about a building is written down for this:
//
//   1. SHEETS (find_sheets): triangles that lie in one plane and share an edge are one sheet.
//      A sheet seen from both sides (the same triangles wound both ways) is one sheet. A sheet
//      whose triangles fill its bounding rectangle is a rectangle and can break; anything else
//      (a sloping balustrade, a windscreen) stays whole.
//   2. PANES (Bars, split): a glazed wall is modelled as one sheet with mullions and transoms
//      standing on it. Every triangle of the building's OTHER parts that lies flat against the
//      sheet, long and narrow, is such a bar (Bars::add); the bars cut the sheet into panes.
//
// Used by engine/vk_engine/glass.cpp (lazily: when the player comes near a glass part, or a
// bolt reaches it). Tested in tests/test_glass.cpp.

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vibe::game::glass {

struct Sheet {
    glm::vec3 origin{0.0f};                 // its lower left corner
    glm::vec3 u{1.0f, 0.0f, 0.0f};          // along it (level, where the sheet is not)
    glm::vec3 v{0.0f, 1.0f, 0.0f};          // up it
    glm::vec3 n{0.0f, 0.0f, 1.0f};
    glm::vec2 size{0.0f};                   // metres along u and v
    bool      rect = false;                 // a full rectangle: it can break
    std::vector<uint32_t> tris;             // its triangles (numbers: index / 3)
};

namespace detail {
inline uint64_t key_of(glm::vec3 p) {
    // (welded to the millimetre; 21 bits a coordinate: +- 1 km)
    const auto q = [](float x) { return uint64_t(int64_t(std::llround(double(x) * 1000.0)) + (1ll << 20)) & 0x1FFFFFull; };
    return (q(p.x) << 42) | (q(p.y) << 21) | q(p.z);
}
struct Dsu {
    std::vector<uint32_t> p;
    explicit Dsu(size_t n) : p(n) { for (size_t i = 0; i < n; ++i) p[i] = uint32_t(i); }
    uint32_t find(uint32_t x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void join(uint32_t a, uint32_t b) { a = find(a); b = find(b); if (a != b) p[b] = a; }
};
}  // namespace detail

// The sheets of a triangle list. `pos(i)` gives the position of vertex i.
template <class PosFn>
std::vector<Sheet> find_sheets(PosFn pos, const uint32_t* idx, size_t n_idx) {
    const size_t nt = n_idx / 3;
    std::vector<Sheet> out;
    if (nt == 0) return out;
    // Weld the vertices by position (a quad written twice, or its two triangles with their own
    // corners, must be seen to share edges).
    std::unordered_map<uint64_t, uint32_t> weld;
    std::vector<uint32_t> wid(n_idx);
    for (size_t i = 0; i < n_idx; ++i) {
        const uint64_t k = detail::key_of(pos(idx[i]));
        auto it = weld.find(k);
        if (it == weld.end()) it = weld.emplace(k, uint32_t(weld.size())).first;
        wid[i] = it->second;
    }
    struct Tri { glm::vec3 n{0.0f}; float d = 0.0f, area = 0.0f; bool flipped = false, ok = false; };
    std::vector<Tri> tri(nt);
    for (size_t t = 0; t < nt; ++t) {
        const glm::vec3 a = pos(idx[t * 3]), b = pos(idx[t * 3 + 1]), c = pos(idx[t * 3 + 2]);
        glm::vec3 n = glm::cross(b - a, c - a);
        const float l = glm::length(n);
        if (l < 1e-8f) continue;
        n /= l;
        // (one sign for both windings: the largest component is positive)
        const glm::vec3 an = glm::abs(n);
        const float big = an.x >= an.y && an.x >= an.z ? n.x : an.y >= an.z ? n.y : n.z;
        tri[t].flipped = big < 0.0f;
        if (tri[t].flipped) n = -n;
        tri[t].n = n; tri[t].d = glm::dot(n, a); tri[t].area = 0.5f * l; tri[t].ok = true;
    }
    // Triangles that share an edge and a plane are one sheet.
    detail::Dsu dsu(nt);
    std::unordered_map<uint64_t, std::vector<uint32_t>> edge_tris;
    for (size_t t = 0; t < nt; ++t) {
        if (!tri[t].ok) continue;
        for (int e = 0; e < 3; ++e) {
            uint32_t a = wid[t * 3 + size_t(e)], b = wid[t * 3 + size_t((e + 1) % 3)];
            if (a == b) continue;
            if (a > b) std::swap(a, b);
            std::vector<uint32_t>& on = edge_tris[(uint64_t(a) << 32) | b];
            // (two sheets may meet at an edge - a glass corner: only those in this one's plane)
            for (uint32_t o : on)
                if (glm::dot(tri[o].n, tri[t].n) > 0.9995f && std::abs(tri[o].d - tri[t].d) < 0.004f) dsu.join(o, uint32_t(t));
            on.push_back(uint32_t(t));
        }
    }
    std::unordered_map<uint32_t, size_t> sheet_of;
    for (size_t t = 0; t < nt; ++t) {
        if (!tri[t].ok) continue;
        const uint32_t r = dsu.find(uint32_t(t));
        auto it = sheet_of.find(r);
        if (it == sheet_of.end()) { it = sheet_of.emplace(r, out.size()).first; out.emplace_back(); }
        out[it->second].tris.push_back(uint32_t(t));
    }
    for (Sheet& s : out) {
        glm::vec3 n(0.0f);
        float d = 0.0f, area_front = 0.0f, area_back = 0.0f;
        for (uint32_t t : s.tris) {
            n += tri[t].n; d += tri[t].d;
            (tri[t].flipped ? area_back : area_front) += tri[t].area;
        }
        n = glm::normalize(n); d /= float(s.tris.size());
        s.n = n;
        if (std::abs(n.y) < 0.9f) s.u = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), n));
        else                      s.u = glm::normalize(glm::cross(n, glm::vec3(0.0f, 0.0f, 1.0f)));
        s.v = glm::cross(n, s.u);
        glm::vec2 lo(1e30f), hi(-1e30f);
        for (uint32_t t : s.tris)
            for (int k = 0; k < 3; ++k) {
                const glm::vec3 p = pos(idx[size_t(t) * 3 + size_t(k)]);
                const glm::vec2 q(glm::dot(p, s.u), glm::dot(p, s.v));
                lo = glm::min(lo, q); hi = glm::max(hi, q);
            }
        s.origin = n * d + s.u * lo.x + s.v * lo.y;
        s.size = hi - lo;
        const float area = std::max(area_front, area_back);          // (one face of it)
        s.rect = s.size.x > 0.12f && s.size.y > 0.12f && area > 0.97f * s.size.x * s.size.y && area < 1.03f * s.size.x * s.size.y;
    }
    return out;
}

// The bars that stand on a sheet: where they cut it (metres from its origin, along u and along v).
struct Bars {
    std::vector<float> u_cuts, v_cuts;
    // A triangle of something else in the building. If it lies flat against the sheet, long and
    // narrow and not along the sheet's own edge, it is the face of a mullion or a transom.
    void add(const Sheet& s, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        const float da = glm::dot(a - s.origin, s.n);
        if (std::abs(da) > 0.15f) return;
        if (std::abs(glm::dot(b - s.origin, s.n) - da) > 0.01f || std::abs(glm::dot(c - s.origin, s.n) - da) > 0.01f) return;
        glm::vec2 lo(1e30f), hi(-1e30f);
        for (const glm::vec3& p : { a, b, c }) {
            const glm::vec2 q(glm::dot(p - s.origin, s.u), glm::dot(p - s.origin, s.v));
            lo = glm::min(lo, q); hi = glm::max(hi, q);
        }
        if (hi.x < 0.0f || hi.y < 0.0f || lo.x > s.size.x || lo.y > s.size.y) return;
        lo = glm::max(lo, glm::vec2(0.0f)); hi = glm::min(hi, s.size);
        const glm::vec2 e = hi - lo, m = 0.5f * (lo + hi);
        const float margin = 0.12f;
        if (e.x <= 0.30f && e.x > 0.005f && e.y >= 0.5f * s.size.y && m.x > margin && m.x < s.size.x - margin) put(u_cuts, m.x);
        if (e.y <= 0.30f && e.y > 0.005f && e.x >= 0.5f * s.size.x && m.y > margin && m.y < s.size.y - margin) put(v_cuts, m.y);
    }
    static void put(std::vector<float>& cuts, float x) {
        for (float c : cuts) if (std::abs(c - x) < 0.10f) return;
        cuts.push_back(x);
    }
};

// The panes of a sheet: the rectangles between its bars, (lower left, upper right) in metres
// from the sheet's origin. No bars: the sheet is one pane.
inline std::vector<std::pair<glm::vec2, glm::vec2>> split(const Sheet& s, Bars bars) {
    std::sort(bars.u_cuts.begin(), bars.u_cuts.end());
    std::sort(bars.v_cuts.begin(), bars.v_cuts.end());
    std::vector<float> xs{ 0.0f }, ys{ 0.0f };
    xs.insert(xs.end(), bars.u_cuts.begin(), bars.u_cuts.end()); xs.push_back(s.size.x);
    ys.insert(ys.end(), bars.v_cuts.begin(), bars.v_cuts.end()); ys.push_back(s.size.y);
    std::vector<std::pair<glm::vec2, glm::vec2>> out;
    for (size_t j = 0; j + 1 < ys.size(); ++j)
        for (size_t i = 0; i + 1 < xs.size(); ++i)
            out.push_back({ glm::vec2(xs[i], ys[j]), glm::vec2(xs[i + 1], ys[j + 1]) });
    return out;
}

}  // namespace vibe::game::glass
