#include "engine/globe.h"

#include <atomic>
#include <cmath>

namespace qlike::globe {

namespace {
// Charts are published as immutable objects (re-anchoring is rare; the old ones are left
// alive for the worker threads that may still be reading them).
std::atomic<const Chart*> g_chart{ new Chart{} };

glm::dvec3 rotate_about(const glm::dvec3& v, const glm::dvec3& axis, double ang) {
    const double c = std::cos(ang), s = std::sin(ang);
    return v * c + glm::cross(axis, v) * s + axis * (glm::dot(axis, v) * (1.0 - c));
}
double smooth(double a, double b, double x) {
    const double t = glm::clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
float hash3(int32_t x, int32_t y, int32_t z) {
    uint32_t h = uint32_t(x) * 0x8da6b343u ^ uint32_t(y) * 0xd8163841u ^ uint32_t(z) * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return float(h & 0xFFFFFFu) * (1.0f / 16777216.0f);
}
// value + gradient at lattice position p
void noised3(const glm::dvec3& p, float& v, glm::vec3& grad) {
    const glm::dvec3 fl = glm::floor(p);
    const int32_t ix = int32_t(int64_t(fl.x)), iy = int32_t(int64_t(fl.y)), iz = int32_t(int64_t(fl.z));
    const glm::vec3 f = glm::vec3(p - fl);
    const glm::vec3 u = f * f * (3.0f - 2.0f * f), du = 6.0f * f * (1.0f - f);
    const float a = hash3(ix, iy, iz), b = hash3(ix + 1, iy, iz), c = hash3(ix, iy + 1, iz), d = hash3(ix + 1, iy + 1, iz);
    const float e = hash3(ix, iy, iz + 1), g = hash3(ix + 1, iy, iz + 1), k = hash3(ix, iy + 1, iz + 1), l = hash3(ix + 1, iy + 1, iz + 1);
    v = glm::mix(glm::mix(glm::mix(a, b, u.x), glm::mix(c, d, u.x), u.y), glm::mix(glm::mix(e, g, u.x), glm::mix(k, l, u.x), u.y), u.z);
    grad = du * glm::vec3(glm::mix(glm::mix(b - a, d - c, u.y), glm::mix(g - e, l - k, u.y), u.z),
                          glm::mix(glm::mix(c - a, d - b, u.x), glm::mix(k - e, l - g, u.x), u.z),
                          glm::mix(glm::mix(e - a, g - b, u.x), glm::mix(k - c, l - d, u.x), u.y));
}
float fbm3d(glm::dvec3 p) {
    float s = 0.0f, a = 0.5f;
    for (int i = 0; i < 5; ++i) { s += a * noise3(p); p = p * 2.03 + glm::dvec3(17.1); a *= 0.5f; }
    return s / 0.96875f;
}
}  // namespace

const Chart& current() { return *g_chart.load(std::memory_order_acquire); }
void set_current(const Chart& c) { g_chart.store(new Chart(c), std::memory_order_release); }

glm::dvec3 to_planet(const Chart& c, double x, double z) {
    const double lx = x - c.ox(), lz = z;
    const double d = std::hypot(lx, lz);
    if (d < 1e-6) return c.F[1];
    const glm::dvec3 dir = (c.F[0] * lx + c.F[2] * lz) / d;
    const double th = d / kR;
    return glm::normalize(c.F[1] * std::cos(th) + dir * std::sin(th));
}

glm::dmat3 frame_at(const Chart& c, double x, double z) {
    const double lx = x - c.ox(), lz = z;
    const double d = std::hypot(lx, lz);
    if (d < 1e-6) return c.F;
    const glm::dvec3 dir = (c.F[0] * lx + c.F[2] * lz) / d;
    const glm::dvec3 axis = glm::normalize(glm::cross(c.F[1], dir));
    const double th = d / kR;
    glm::dmat3 f;
    f[0] = rotate_about(c.F[0], axis, th);
    f[1] = rotate_about(c.F[1], axis, th);
    f[2] = rotate_about(c.F[2], axis, th);
    return f;
}

glm::dvec2 home_az(const glm::dvec3& G) {
    const double sth = std::hypot(G.x, G.z);
    if (sth < 1e-12) return glm::dvec2(0.0);
    const double th = std::atan2(sth, G.y);
    return glm::dvec2(G.x, G.z) * (kR * th / sth);
}

glm::dvec2 to_chart(const Chart& c, const glm::dvec3& G) {
    const double e = glm::dot(G, c.F[0]), u = glm::dot(G, c.F[1]), s = glm::dot(G, c.F[2]);
    const double sth = std::hypot(e, s);
    if (sth < 1e-12) return glm::dvec2(c.ox(), 0.0);
    const double th = std::atan2(sth, u);
    return glm::dvec2(c.ox() + e * (kR * th / sth), s * (kR * th / sth));
}

Chart far_chart(const glm::dmat3& frame) {
    Chart c;
    c.F = frame;
    // (keep it orthonormal: frames are carried from chart to chart)
    c.F[1] = glm::normalize(c.F[1]);
    c.F[0] = glm::normalize(c.F[0] - c.F[1] * glm::dot(c.F[0], c.F[1]));
    c.F[2] = glm::cross(c.F[0], c.F[1]);
    c.home = false;
    return c;
}

void lattice(const glm::dvec3& origin, int32_t cells[kOctaves][4], float fracs[kOctaves][4]) {
    double s = 0.003;
    for (int i = 0; i < kOctaves; ++i) {
        const glm::dvec3 c = origin * s + glm::dvec3(37.2, 11.7, 53.9) * double(i);
        const glm::dvec3 fl = glm::floor(c);
        cells[i][0] = int32_t(int64_t(fl.x)); cells[i][1] = int32_t(int64_t(fl.y)); cells[i][2] = int32_t(int64_t(fl.z)); cells[i][3] = 0;
        fracs[i][0] = float(c.x - fl.x); fracs[i][1] = float(c.y - fl.y); fracs[i][2] = float(c.z - fl.z); fracs[i][3] = 0.0f;
        s *= 2.0;
    }
}

float noise3(const glm::dvec3& p) {
    float v; glm::vec3 g;
    noised3(p, v, g);
    return v;
}

float shape3(const glm::dvec3& P, float oct_f) {
    const int oct = int(std::ceil(oct_f));
    const float fr = 1.0f - (float(oct) - oct_f);
    float a = 0.0f, b = 1.0f;
    glm::vec3 d(0.0f);
    double s = 0.003;
    for (int i = 0; i < kOctaves && i < oct; ++i) {
        float v; glm::vec3 g;
        noised3(P * s + glm::dvec3(37.2, 11.7, 53.9) * double(i), v, g);
        const float w = (i == oct - 1) ? fr : 1.0f;
        d += g * (0.8f * w);
        a += b * v * w / (1.0f + glm::dot(d, d));
        b *= 0.5f; s *= 2.0;
    }
    return a;
}

float continent3(const glm::dvec3& G) {
    const glm::dvec3 q = G * 2.6712;
    const glm::dvec3 bend = glm::dvec3(noise3(q * 2.3 + glm::dvec3(5.2)), noise3(q * 2.3 + glm::dvec3(91.7)), noise3(q * 2.3 + glm::dvec3(33.1))) - glm::dvec3(0.5);
    float v = (fbm3d(q + bend * 0.55 + glm::dvec3(21.3)) - 0.5f) * 1.25f;
    if (std::abs(v) > 0.37f) return v;
    const glm::dvec3 k = G * 6360.0;
    v += (fbm3d(k * 0.011 + glm::dvec3(8.9)) - 0.5f) * 0.16f;
    v += ((0.5f * noise3(k * 0.35 + glm::dvec3(61.0)) + 0.25f * noise3(k * 0.71 + glm::dvec3(78.1)) + 0.125f * noise3(k * 1.435 - glm::dvec3(9.4))) / 0.875f - 0.5f) * 0.05f;
    return v;
}

float ranges3(const glm::dvec3& G) {
    const glm::dvec3 k = G * 6360.0;
    return float(smooth(0.56, 0.72, double(noise3(k * 0.08 + glm::dvec3(7.3)) * 0.65f + noise3(k * 0.21 + glm::dvec3(3.1)) * 0.35f)));
}

}  // namespace qlike::globe
