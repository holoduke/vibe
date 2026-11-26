#include "engine/sector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_set>

namespace vibe::sector {

namespace {
constexpr double kPi = 3.14159265358979323846;
// Outward, a axis, b axis of each face (see the table in sector.h).
const glm::dvec3 kN[6] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
const glm::dvec3 kU[6] = { { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } };
const glm::dvec3 kV[6] = { { 0, 0, 1 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 } };

inline double warp(double u)   { return 0.5 + (2.0 / kPi) * std::atan(u); }          // gnomonic -> 0..1, equal angle
inline double unwarp(double a) { return std::tan((2.0 * a - 1.0) * (kPi / 4.0)); }
inline double cells(int level) { return double(uint64_t(1) << level); }
}  // namespace

Id make(int face, int level, uint32_t i, uint32_t j) {
    if (face < 0 || face > 5 || level < 0 || level > kMaxLevel) return Id{};
    const uint32_t n = uint32_t(1) << level;
    if (i >= n || j >= n) return Id{};
    return Id{ (uint64_t(face + 1) << 61) | (uint64_t(level) << 56) | (uint64_t(i) << 28) | uint64_t(j) };
}

glm::dvec2 face_ab(int face, const glm::dvec3& G) {
    const double n = glm::dot(G, kN[face]);
    return glm::dvec2(warp(glm::dot(G, kU[face]) / n), warp(glm::dot(G, kV[face]) / n));
}

FacePos face_pos(const glm::dvec3& G) {
    const double ax = std::abs(G.x), ay = std::abs(G.y), az = std::abs(G.z);
    FacePos p;
    if (ay >= ax && ay >= az) p.face = G.y >= 0.0 ? 2 : 3;
    else if (ax >= az)        p.face = G.x >= 0.0 ? 0 : 1;
    else                      p.face = G.z >= 0.0 ? 4 : 5;
    const glm::dvec2 ab = glm::clamp(face_ab(p.face, G), 0.0, 1.0);
    p.a = ab.x; p.b = ab.y;
    return p;
}

glm::dvec3 point(int face, double a, double b) {
    return glm::normalize(kN[face] + kU[face] * unwarp(a) + kV[face] * unwarp(b));
}

Id cell_at(const glm::dvec3& G, int level, glm::dvec2* frac) {
    level = std::clamp(level, 0, kMaxLevel);
    const FacePos p = face_pos(G);
    const double n = cells(level);
    const double fi = std::min(std::floor(p.a * n), n - 1.0), fj = std::min(std::floor(p.b * n), n - 1.0);
    if (frac) *frac = glm::dvec2(p.a * n - fi, p.b * n - fj);
    return make(p.face, level, uint32_t(fi), uint32_t(fj));
}

glm::dvec3 point(Id id, double fa, double fb) {
    const double n = cells(id.level());
    return point(id.face(), (double(id.i()) + fa) / n, (double(id.j()) + fb) / n);
}

Id parent(Id id) { return id.valid() && id.level() > 0 ? make(id.face(), id.level() - 1, id.i() >> 1, id.j() >> 1) : Id{}; }
Id child(Id id, int k) {
    return id.valid() && id.level() < kMaxLevel ? make(id.face(), id.level() + 1, id.i() * 2u + uint32_t(k & 1), id.j() * 2u + uint32_t((k >> 1) & 1)) : Id{};
}
Id ancestor(Id id, int level) {
    if (!id.valid() || level > id.level() || level < 0) return Id{};
    const int s = id.level() - level;
    return make(id.face(), level, id.i() >> s, id.j() >> s);
}
bool contains(Id outer, Id inner) { return outer.valid() && inner.valid() && ancestor(inner, outer.level()) == outer; }

Id neighbour(Id id, int di, int dj) {
    if (!id.valid()) return Id{};
    const int64_t n = int64_t(1) << id.level();
    const int64_t ni = int64_t(id.i()) + di, nj = int64_t(id.j()) + dj;
    const bool out_i = ni < 0 || ni >= n, out_j = nj < 0 || nj >= n;
    if (!out_i && !out_j) return make(id.face(), id.level(), uint32_t(ni), uint32_t(nj));
    if (out_i && out_j) return Id{};
    // Across the face's edge: the cell that holds the point a hair beyond the edge, level with
    // this cell's middle (cells meet edge to edge across a cube edge - the equal-angle
    // coordinate along the edge is the same on both faces).
    const double eps = 1e-12, dn = double(n);
    const double a = out_i ? (ni < 0 ? -eps : 1.0 + eps) : (double(ni) + 0.5) / dn;
    const double b = out_j ? (nj < 0 ? -eps : 1.0 + eps) : (double(nj) + 0.5) / dn;
    return cell_at(point(id.face(), a, b), id.level());
}

int neighbours(Id id, Id out[8]) {
    int k = 0;
    const auto add = [&](Id c) {
        if (!c.valid() || c == id) return;
        for (int q = 0; q < k; ++q) if (out[q] == c) return;
        out[k++] = c;
    };
    static const int e[4][2] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
    for (const auto& d : e) add(neighbour(id, d[0], d[1]));
    for (int dj = -1; dj <= 1; dj += 2)
        for (int di = -1; di <= 1; di += 2) {
            Id c = neighbour(id, di, dj);
            if (!c.valid()) {
                // (over a face's edge: first along the edge, within the face, then across it)
                const Id along_j = neighbour(id, 0, dj), along_i = neighbour(id, di, 0);
                if (along_j.valid() && along_j.face() == id.face()) c = neighbour(along_j, di, 0);
                else if (along_i.valid() && along_i.face() == id.face()) c = neighbour(along_i, 0, dj);
            }
            add(c);
        }
    return k;
}

glm::dvec2 size_m(Id id) {
    return glm::dvec2(globe::ground_distance(point(id, 0.0, 0.5), point(id, 1.0, 0.5)),
                      globe::ground_distance(point(id, 0.5, 0.0), point(id, 0.5, 1.0)));
}

double radius_m(Id id) {
    const glm::dvec3 c = centre(id);
    double r = 0.0;
    for (int k = 0; k < 4; ++k) r = std::max(r, globe::ground_distance(c, point(id, double(k & 1), double(k >> 1))));
    return r;
}

void cover(const glm::dvec3& G, double radius, int level, std::vector<Id>& out) {
    out.clear();
    const Id first = cell_at(G, level);
    if (!first.valid()) return;
    std::unordered_set<uint64_t> seen{ first.v };
    out.push_back(first);
    for (size_t at = 0; at < out.size(); ++at) {
        Id nb[8];
        const int n = neighbours(out[at], nb);
        for (int k = 0; k < n; ++k) {
            if (!seen.insert(nb[k].v).second) continue;
            if (globe::ground_distance(G, centre(nb[k])) <= radius + radius_m(nb[k])) out.push_back(nb[k]);
        }
    }
}

std::string name(Id id) {
    if (!id.valid()) return "none";
    char b[64];
    std::snprintf(b, sizeof(b), "%d-%d-%u-%u", id.face(), id.level(), id.i(), id.j());
    return b;
}

bool parse(const std::string& s, Id& out) {
    int f = -1, l = -1;
    unsigned i = 0, j = 0;
    char tail = 0;
    if (std::sscanf(s.c_str(), "%d-%d-%u-%u%c", &f, &l, &i, &j, &tail) != 4) return false;
    const Id id = make(f, l, i, j);
    if (!id.valid()) return false;
    out = id;
    return true;
}

glm::dvec2 to_chart(const globe::Chart& c, Id id, double fa, double fb) { return globe::to_chart(c, point(id, fa, fb)); }

Id from_chart(const globe::Chart& c, double x, double z, int level, glm::dvec2* frac) {
    return cell_at(globe::to_planet(c, x, z), level, frac);
}

glm::dvec2 chart_cell_exact(const globe::Chart& c, const glm::dvec2& p, int face, int level, int64_t base_i, int64_t base_j) {
    const glm::dvec2 ab = face_ab(face, globe::to_planet(c, p.x, p.y));
    const double n = cells(level);
    return glm::dvec2(ab.x * n - double(base_i), ab.y * n - double(base_j));
}

ChartMap chart_map(const globe::Chart& c, const glm::dvec2& anchor, int face, int level, int64_t base_i, int64_t base_j) {
    ChartMap m;
    m.face = face; m.level = level; m.base_i = base_i; m.base_j = base_j; m.anchor = anchor;
    const double e = 200.0;
    const auto f = [&](double dx, double dz) { return chart_cell_exact(c, anchor + glm::dvec2(dx, dz), face, level, base_i, base_j); };
    const glm::dvec2 f0 = f(0, 0), fxp = f(e, 0), fxm = f(-e, 0), fzp = f(0, e), fzm = f(0, -e);
    const glm::dvec2 fpp = f(e, e), fpm = f(e, -e), fmp = f(-e, e), fmm = f(-e, -e);
    m.c = f0;
    m.A = glm::dmat2((fxp - fxm) / (2.0 * e), (fzp - fzm) / (2.0 * e));
    const glm::dvec2 hxx = (fxp - 2.0 * f0 + fxm) / (e * e), hzz = (fzp - 2.0 * f0 + fzm) / (e * e);
    const glm::dvec2 hxz = (fpp - fpm - fmp + fmm) / (4.0 * e * e);
    m.Hi = glm::dvec3(hxx.x, hxz.x, hzz.x);
    m.Hj = glm::dvec3(hxx.y, hxz.y, hzz.y);
    return m;
}

}  // namespace vibe::sector
