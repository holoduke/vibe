#include "engine/eztree.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

#undef min
#undef max

namespace vibe {
namespace {

constexpr float kPi = 3.14159265358979f;

struct Rng {
    uint32_t s;
    float u() { s = s * 1664525u + 1013904223u; return float((s >> 8) & 0xFFFFFFu) / 16777216.0f; }
    // ez-tree's rng.random(max, min).
    float random(float mx = 1.0f, float mn = 0.0f) { return mn + (mx - mn) * u(); }
};

struct Section { glm::vec3 origin; glm::quat q; float radius; };
struct Branch { glm::vec3 origin; glm::quat q; float length, radius; int level, sections, segments; float prio; };
struct SkBranch { std::vector<Section> sections; int segments; int level; float prio; };
struct Leaf { glm::vec3 origin; glm::quat q; float size; float prio; };

glm::quat axis_q(glm::vec3 axis, float ang) { return glm::angleAxis(ang, axis); }

// ---- Habit growth: helpers ----
// The shortest rotation that takes unit `a` to unit `b`.
glm::quat arc(glm::vec3 a, glm::vec3 b) {
    const float d = glm::dot(a, b);
    if (d < -0.9999f) {
        const glm::vec3 other = std::abs(a.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
        return glm::angleAxis(kPi, glm::normalize(glm::cross(a, other)));
    }
    const glm::vec3 c = glm::cross(a, b);
    return glm::normalize(glm::quat(1.0f + d, c.x, c.y, c.z));
}
// A direction leaving a parent that runs along `pd`: `ang` away from it, at `az` round it.
// az 0 / pi = the two level sides, pi/2 = the upper side, -pi/2 = underneath.
glm::vec3 side_dir(glm::vec3 pd, float az, float ang) {
    glm::vec3 u = glm::cross(pd, glm::vec3(0, 1, 0));
    u = glm::dot(u, u) < 1e-4f ? glm::vec3(1, 0, 0) : glm::normalize(u);
    const glm::vec3 v = glm::cross(u, pd);
    return glm::normalize(pd * std::cos(ang) + (u * std::cos(az) + v * std::sin(az)) * std::sin(ang));
}
// Place, direction and radius at fraction `f` of a branch's length.
void point_at(const std::vector<Section>& s, float f, glm::vec3& p, glm::vec3& d, float& r) {
    const size_t n = s.size();
    if (n < 2) { p = s[0].origin; d = glm::vec3(0, 1, 0); r = s[0].radius; return; }
    float total = 0.0f;
    for (size_t i = 0; i + 1 < n; ++i) total += glm::length(s[i + 1].origin - s[i].origin);
    float t = glm::clamp(f, 0.0f, 1.0f) * total;
    for (size_t i = 0; i + 1 < n; ++i) {
        const glm::vec3 e = s[i + 1].origin - s[i].origin;
        const float L = glm::length(e);
        if (t <= L || i + 2 == n) {
            const float a = L > 1e-6f ? glm::clamp(t / L, 0.0f, 1.0f) : 0.0f;
            p = s[i].origin + e * a;
            r = glm::mix(s[i].radius, s[i + 1].radius, a);
            d = L > 1e-6f ? e / L : glm::vec3(0, 1, 0);
            return;
        }
        t -= L;
    }
}
// What a branch is asked to be (metres, radians).
struct Axis {
    glm::vec3 origin{0.0f}, dir{0.0f, 1.0f, 0.0f};
    float length = 1.0f, radius = 0.05f;
    int level = 0, sections = 4, segments = 4;
    float prio = 0.0f;
    float sag = 0.0f;        // turned down by its weight, most near the base (in all, for a level branch)
    float lift = 0.0f;       // turned up to the light, most at the tip (negative: the tip hangs)
    float wander = 0.0f;     // random curvature a section
    float tip = 0.2f;        // radius at the tip / radius at the base
    float collar = 0.0f;     // > 0: how far along it leaves the parent's bark - it swells there
    glm::vec3 kick{0.0f};    // trunk: the lean it takes on above its first section
    bool cap = false;        // close the end (a stem that ends where it divides)
};
// The crown's outline: half an ellipsoid above its middle, another below.
struct Crown { glm::vec3 c{0.0f}; float rx = 1.0f, up = 1.0f, dn = 1.0f; };
float reach(const Crown& e, glm::vec3 p, glm::vec3 d) {
    const glm::vec3 R(e.rx, d.y >= 0.0f ? e.up : e.dn, e.rx);
    const glm::vec3 P = (p - e.c) / R, D = d / R;
    const float dd = glm::dot(D, D), pd = glm::dot(P, D);
    const float disc = pd * pd - dd * (glm::dot(P, P) - 1.0f);
    return disc > 0.0f ? std::max(0.0f, (-pd + std::sqrt(disc)) / dd) : 0.0f;
}
// The twigs of a tree: what carries its leaf cards.
struct Twig {
    float len = 0.5f, rad = 0.008f, ang = 0.8f;
    float hang = 0.0f;       // how much they leave pointing down (birch)
    float sag = 0.1f, lift = 0.2f;
    int   sections = 2;
    int   leaves = 5;        // cards along it (one more at its end)
    float leaf_ang = 0.75f, leaf_droop = 0.1f;
    float plo = 0.18f, phi = 0.55f;      // priority after its branch
};

struct Gen {
    const EzTreeOptions& o;
    Rng rng;
    std::deque<Branch> queue;
    std::vector<SkBranch> branches;
    std::vector<Leaf> leaves;

    Rng prng{ 0x51ED270Bu };          // priorities: separate stream, the shape stays that of the seed
    float cur_prio = 0.0f;            // priority of the branch being grown
    void record_leaf(glm::vec3 origin, glm::quat q) {
        const float size = o.leaf_size * (1.0f + rng.random(o.leaf_size_variance, -o.leaf_size_variance));
        leaves.push_back({ origin, q, size, prng.u() });   // uniform: the leaf count is proportional to the detail
    }
    // Children appear after their parent; thin orders come in late.
    float child_prio(int level) {
        const float lo = o.evergreen ? 0.0f : (level == 1 ? 0.0f : level == 2 ? 0.08f : 0.18f);
        const float hi = o.evergreen ? 0.85f : (level == 1 ? 0.10f : level == 2 ? 0.32f : 0.55f);
        return std::min(0.9f, cur_prio + lo + (hi - lo) * prng.u());
    }
    std::vector<int> shuffled(int count) {
        std::vector<int> a(static_cast<size_t>(count));
        for (int k = 0; k < count; ++k) a[size_t(k)] = k;
        for (int k = count - 1; k > 0; --k) std::swap(a[size_t(k)], a[size_t(std::min(k, int(rng.random() * float(k + 1))))]);
        return a;
    }
    // Position / orientation / radius at fraction `f` along a branch.
    void at(const std::vector<Section>& sec, float f, glm::vec3& origin, glm::quat& q, float& radius) {
        const int last = int(sec.size()) - 1;
        const int idx = std::clamp(int(std::floor(f * float(last))), 0, last);
        const Section& a = sec[size_t(idx)];
        const Section& b = sec[size_t(std::min(idx + 1, last))];
        const float alpha = std::clamp(f * float(last) - float(idx), 0.0f, 1.0f);
        origin = glm::mix(a.origin, b.origin, alpha);
        radius = (1.0f - alpha) * a.radius + alpha * b.radius;
        q = glm::slerp(a.q, b.q, alpha);
    }

    void grow(const Branch& br) {
        glm::quat q = br.q;
        glm::vec3 origin = br.origin;
        const float section_len = br.length / float(br.sections);
        std::vector<Section> sec;
        for (int i = 0; i <= br.sections; ++i) {
            float r = br.radius;
            const float f = float(i) / float(br.sections);
            if (i == br.sections && br.level == o.levels) r = 0.001f;
            else if (!o.evergreen) r *= 1.0f - o.taper[br.level] * f;
            else r *= 1.0f - f;                              // evergreens taper to a point
            sec.push_back({ origin, q, r });
            origin += q * glm::vec3(0.0f, section_len, 0.0f);

            // Random perturbation, stronger on thin branches.
            const float g = std::max(1.0f, 1.0f / std::sqrt(std::max(r, 1e-4f))) * o.gnarliness[br.level];
            q = q * axis_q(glm::vec3(1, 0, 0), rng.random(g, -g)) * axis_q(glm::vec3(0, 0, 1), rng.random(g, -g));
            q = q * axis_q(glm::vec3(0, 1, 0), o.twist[br.level]);
            // Growth force: rotate the growth direction toward (or away from)
            // force_dir by strength / radius, never past it.
            const glm::vec3 up = q * glm::vec3(0, 1, 0);
            const glm::vec3 target = glm::normalize(o.force_dir);
            glm::vec3 axis = glm::cross(up, target);
            const float sin_full = glm::length(axis);
            if (sin_full > 1e-6f) {
                axis /= sin_full;
                const float full = std::atan2(sin_full, glm::dot(up, target));
                const float step = o.force_strength / std::max(r, 1e-3f);
                q = axis_q(axis, std::clamp(step, -full, full)) * q;
            }
            q = glm::normalize(q);
        }
        branches.push_back({ sec, br.segments, br.level, br.prio });
        cur_prio = br.prio;

        // Deciduous: a terminal branch continues from the end of the parent.
        if (!o.evergreen) {
            const Section& lastS = sec.back();
            if (br.level < o.levels)
                queue.push_back({ lastS.origin, lastS.q, o.length[br.level + 1], lastS.radius, br.level + 1,
                                  br.sections, br.segments, child_prio(br.level + 1) });
            else
                record_leaf(lastS.origin, lastS.q);
        }
        if (br.level == o.levels) gen_leaves(sec);
        else if (br.level < o.levels) gen_children(o.children[br.level], br.level + 1, sec);
    }

    void gen_children(int count, int level, const std::vector<Section>& sec) {
        if (count <= 0) return;
        const float radial_offset = rng.random();
        const float start_min = o.start[level];
        const float step = (1.0f - start_min) / float(count);
        const std::vector<int> slots = shuffled(count);
        for (int i = 0; i < count; ++i) {
            const float st = start_min + (float(i) + rng.random()) * step;
            glm::vec3 origin; glm::quat pq; float pr;
            at(sec, st, origin, pq, pr);
            const float jitter = rng.random(0.5f, -0.5f);
            const float radial = 2.0f * kPi * (radial_offset + (float(slots[size_t(i)]) + jitter) / float(count));
            const glm::quat cq = pq * axis_q(glm::vec3(0, 1, 0), radial) *
                                 axis_q(glm::vec3(1, 0, 0), glm::radians(o.angle[level]));
            const float len = o.length[level] * (o.evergreen ? 1.0f - st : 1.0f);
            queue.push_back({ origin, cq, len, o.radius[level] * pr, level, o.sections[level], o.segments[level], child_prio(level) });
        }
    }

    void gen_leaves(const std::vector<Section>& sec) {
        const float radial_offset = rng.random();
        const int count = o.leaf_count;
        if (count <= 0) return;
        const float step = (1.0f - o.leaf_start) / float(count);
        const std::vector<int> slots = shuffled(count);
        for (int i = 0; i < count; ++i) {
            const float st = o.leaf_start + (float(i) + rng.random()) * step;
            glm::vec3 origin; glm::quat pq; float pr;
            at(sec, st, origin, pq, pr);
            const float jitter = rng.random(0.5f, -0.5f);
            const float radial = 2.0f * kPi * (radial_offset + (float(slots[size_t(i)]) + jitter) / float(count));
            record_leaf(origin, pq * axis_q(glm::vec3(0, 1, 0), radial) *
                                     axis_q(glm::vec3(1, 0, 0), glm::radians(o.leaf_angle)));
        }
    }

    void run() {
        queue.push_back({ glm::vec3(0.0f), glm::quat(1, 0, 0, 0), o.length[0], o.radius[0], 0, o.sections[0], o.segments[0], 0.0f });
        while (!queue.empty()) {
            const Branch b = queue.front();
            queue.pop_front();
            grow(b);
        }
    }

    // =================================================================================
    // Habit growth (this engine's own; eztree.h TreeHabit). Everything in metres. The
    // skeleton is laid down branch by branch, each one an Axis: it starts on its parent's
    // centre line, swells where it leaves the bark, then runs out in sections that wander,
    // sag and turn up. Leaf cards hang on the twigs only.
    // =================================================================================
    float rr(float a, float b) { return a + (b - a) * rng.u(); }
    float pr(float base, float lo, float hi) { return std::min(0.9f, base + lo + (hi - lo) * prng.u()); }
    static Axis mk(glm::vec3 p, glm::vec3 d, float len, float rad, int level, int sections, int segments, float prio) {
        Axis a;
        a.origin = p; a.dir = d; a.length = len; a.radius = rad; a.level = level;
        a.sections = sections; a.segments = segments; a.prio = prio;
        return a;
    }

    std::vector<Section> grow_axis(const Axis& a) {
        const glm::vec3 Y(0.0f, 1.0f, 0.0f);
        std::vector<Section> sec;
        glm::vec3 dir = glm::normalize(a.dir), pos = a.origin;
        glm::quat q = arc(Y, dir);
        float len = a.length, flare = 1.0f;
        if (a.collar > 0.0f) {
            // The join: two rings more, the widest where the branch comes out of the parent.
            const float e = a.collar, w = 1.8f * a.radius;
            if (len > 2.5f * (e + w)) {
                sec.push_back({ pos, q, a.radius * 1.25f });
                sec.push_back({ pos + dir * e, q, a.radius * 1.42f });
                pos += dir * (e + w); len -= e + w; flare = 1.08f;
            }
        }
        const int n = std::max(a.sections, 1);
        const float step = len / float(n);
        glm::vec3 kv(0.0f);
        float r = a.radius;
        for (int i = 0; i <= n; ++i) {
            const float f = float(i) / float(n);
            r = std::max(a.radius * glm::mix(1.0f, a.tip, std::pow(f, 0.85f)) * (i == 0 ? flare : 1.0f), 0.0015f);
            sec.push_back({ pos, q, r });
            if (i == n) break;
            pos += dir * step;
            glm::vec3 nd = dir;
            const glm::vec3 rv(rng.u() - 0.5f, rng.u() - 0.5f, rng.u() - 0.5f);
            if (i == 0 && a.level == 0) nd += a.kick;            // (the foot stands straight: the root flare is built round it)
            else {
                kv = glm::mix(kv, rv * 2.0f, 0.6f);
                nd += (kv - dir * glm::dot(kv, dir)) * a.wander;
                const glm::vec3 up = Y - dir * dir.y;            // toward the sky, square to the branch: a level branch bends most
                const float fm = (float(i) + 0.5f) / float(n);
                nd += up * ((a.lift * 3.0f * fm * fm - a.sag * 2.0f * (1.0f - fm)) / float(n));
            }
            nd = glm::normalize(nd);
            q = glm::normalize(arc(dir, nd) * q);
            dir = nd;
        }
        if (a.cap) sec.push_back({ pos + dir * (r * 0.8f), q, 0.003f });
        branches.push_back({ sec, a.segments, a.level, a.prio });
        return sec;
    }

    // A leaf card growing from `p` along `d`; `roll` turns its face about that line (0 = facing up).
    void add_leaf(glm::vec3 p, glm::vec3 d, float roll) {
        d = glm::normalize(d);
        glm::vec3 x = glm::cross(d, glm::vec3(0, 1, 0));
        x = glm::dot(x, x) < 1e-4f ? glm::vec3(1, 0, 0) : glm::normalize(x);
        glm::vec3 z = glm::cross(x, d);
        x = glm::normalize(x * std::cos(roll) + z * std::sin(roll));
        z = glm::cross(x, d);
        record_leaf(p, glm::normalize(glm::quat_cast(glm::mat3(x, d, z))));
    }
    // Cards along a twig, to both sides in turn, and one at its end.
    void bear_leaves(const std::vector<Section>& s, float f0, int n, float ang, float droop) {
        float side = rng.u() < 0.5f ? 0.0f : kPi;
        glm::vec3 p, d; float r;
        for (int k = 0; k < n; ++k) {
            point_at(s, f0 + (1.0f - f0) * (float(k) + rng.u()) / float(n), p, d, r);
            glm::vec3 ld = side_dir(d, side + rr(-0.7f, 0.7f), ang * rr(0.7f, 1.25f));
            side += kPi;
            ld.y -= droop;
            add_leaf(p, ld, rr(-1.15f, 1.15f));
        }
        point_at(s, 1.0f, p, d, r);
        d.y -= droop * 0.6f;
        add_leaf(p, d, rr(-1.15f, 1.15f));
    }
    // A tuft: cards standing round the end of a shoot (pine).
    void tuft(glm::vec3 p, glm::vec3 d, int n) {
        const float a0 = rng.u() * 6.2832f;
        for (int k = 0; k < n; ++k)
            add_leaf(p - d * (o.leaf_size * 0.18f), side_dir(d, a0 + 6.2832f * (float(k) + rr(-0.25f, 0.25f)) / float(n), rr(0.35f, 0.75f)),
                     rr(-1.5f, 1.5f));
        add_leaf(p - d * (o.leaf_size * 0.25f), d, rr(-1.5f, 1.5f));
    }
    // Twigs along a branch between f0 and f1, each with its leaves.
    void bear_twigs(const std::vector<Section>& s, float base_prio, float f0, float f1, int n, const Twig& t, int level) {
        float side = rng.u() < 0.5f ? 0.0f : kPi, gold = rng.u() * 6.2832f;
        for (int k = 0; k < n; ++k) {
            glm::vec3 p, d; float r;
            const float f = f0 + (f1 - f0) * (float(k) + rng.u()) / float(n);
            point_at(s, f, p, d, r);
            float az;
            if (std::abs(d.y) > 0.85f) { gold += 2.39996f; az = gold; }       // on an upright shoot: all round
            else { az = side + rr(-0.6f, 0.6f); side += kPi; }                // on a leaning one: to both sides, a flat spray
            glm::vec3 td = side_dir(d, az, t.ang * rr(0.75f, 1.25f));
            td.y -= t.hang;
            Axis a = mk(p, td, t.len * (1.0f - 0.4f * f) * rr(0.7f, 1.15f), std::min(t.rad, r * 0.7f), level, t.sections, 3,
                        pr(base_prio, t.plo, t.phi));
            a.sag = t.sag; a.lift = t.lift; a.wander = 0.15f; a.tip = 0.3f;
            const auto ts = grow_axis(a);
            bear_leaves(ts, 0.2f, t.leaves, t.leaf_ang, t.leaf_droop);
        }
    }

    // ---- Oak (1) and birch (2) ----
    void grow_broadleaf(float H) {
        const TreeHabit& h = o.habit;
        const glm::vec3 Y(0.0f, 1.0f, 0.0f);
        const bool  birch = h.kind == 2;
        const float age = glm::clamp(h.age, 0.0f, 1.0f);
        const bool  one_stem = birch || age < 0.2f;                  // the stem runs to the top
        const float dens = std::max(h.density, 0.3f);
        const float r0 = std::min(0.9f, H * (birch ? 0.013f : 0.019f) * (0.7f + 0.9f * age) + 0.03f);
        const float hb = H * (birch ? rr(0.26f, 0.34f) : glm::mix(rr(0.26f, 0.33f), rr(0.15f, 0.21f), age));   // clear trunk
        const float Lt = one_stem ? H * 0.96f
                                  : std::max(hb + 0.08f * H, H * glm::mix(0.50f, 0.36f, age) * rr(0.94f, 1.06f));
        const glm::vec3 lean_v = glm::vec3(std::cos(h.lean_az), 0.0f, std::sin(h.lean_az)) * h.lean;
        const glm::vec3 asym_v = glm::vec3(std::cos(h.asym_az), 0.0f, std::sin(h.asym_az)) * h.asym;

        Axis ta = mk(glm::vec3(0.0f), Y, Lt, r0, 0, one_stem ? 9 : 6, age > 0.75f ? 10 : 8, 0.0f);
        ta.kick = lean_v; ta.lift = h.lean * (one_stem ? 1.3f : 0.6f);
        ta.wander = birch ? 0.05f : 0.03f + 0.04f * age;
        ta.tip = one_stem ? 0.07f : 0.66f; ta.cap = !one_stem;
        const auto trunk = grow_axis(ta);
        const glm::vec3 top = trunk.back().origin;

        Crown env;
        env.rx = H * (birch ? 0.20f : glm::mix(0.30f, 0.42f, age)) * h.crown_w;
        const float yc = glm::mix(hb, H, birch ? 0.45f : 0.36f);
        env.c = glm::vec3(top.x * 0.7f, yc, top.z * 0.7f) + asym_v * env.rx;
        env.up = H - yc; env.dn = (yc - hb) * 1.2f;

        Twig tw;
        tw.len = H * (birch ? 0.085f : 0.070f); tw.rad = 0.003f + 0.0005f * H; tw.ang = 0.8f;
        tw.hang = birch ? 0.9f : 0.0f; tw.sag = birch ? 0.7f : 0.1f; tw.lift = birch ? 0.0f : 0.25f;
        tw.leaves = std::max(o.leaf_count, 1); tw.leaf_ang = 0.75f; tw.leaf_droop = birch ? 0.7f : 0.15f;

        // A limb off the stem: out to the crown's outline, side branches along its outer three quarters,
        // twigs on those and on its own end.
        auto limb = [&](glm::vec3 p, glm::vec3 d, float rad, float collar, int segs, float sag, float lift, float prio) {
            const float len = std::max(0.10f * H, reach(env, p, d) * rr(0.82f, 1.02f));
            Axis a = mk(p, d, len, rad, 1, std::clamp(int(len / (0.085f * H)) + 3, 4, 8), segs, prio);
            a.sag = sag; a.lift = lift; a.wander = birch ? 0.07f : 0.10f + 0.06f * age; a.tip = 0.16f; a.collar = collar;
            const auto ls = grow_axis(a);
            const int n2 = std::clamp(int(len / (0.07f * H) * dens + 0.5f), 2, 9);
            float gold = rng.u() * 6.2832f;
            for (int k = 0; k < n2; ++k) {
                glm::vec3 p2, d2; float r2;
                const float f = 0.26f + 0.66f * (float(k) + rng.u()) / float(n2);
                point_at(ls, f, p2, d2, r2);
                gold += 2.39996f;
                float az = gold;
                if (std::sin(az) < -0.35f) az = -az;                      // few branches leave a limb underneath
                const float ang = rr(0.6f, 1.0f);
                glm::vec3 sd = side_dir(d2, az, ang);
                if (birch) sd = glm::normalize(sd - glm::vec3(0.0f, 0.25f, 0.0f));
                float l2 = std::min(len * (0.50f - 0.22f * f) * rr(0.8f, 1.15f), reach(env, p2, sd) * 1.05f);
                l2 = std::max(l2, 0.05f * H);
                Axis sa = mk(p2, sd, l2, std::max(0.004f, r2 * rr(0.45f, 0.6f)), 2, l2 > 0.1f * H ? 4 : 3, 3, pr(prio, 0.10f, 0.42f));
                sa.sag = birch ? 0.2f : 0.15f; sa.lift = birch ? -0.9f : 0.30f; sa.wander = 0.12f; sa.tip = 0.22f;
                if (r2 > 0.035f) sa.collar = std::min(r2 / std::max(std::sin(ang), 0.5f), 2.0f * r2);
                const auto ss = grow_axis(sa);
                bear_twigs(ss, sa.prio, 0.22f, 1.0f, std::clamp(int(l2 / (0.036f * H) * dens) + 1, 2, 8), tw, 3);
            }
            bear_twigs(ls, prio, 0.70f, 1.0f, 3, tw, 3);
        };

        glm::vec3 p, d; float r;
        float gold = rng.u() * 6.2832f;
        if (!one_stem) {
            // The trunk divides: two to four limbs share its top ...
            const float rt = trunk[trunk.size() - 2].radius;
            const int   nt = (age > 0.6f ? 3 : 2) + int(rng.u() * 1.99f);
            const float a0 = rng.u() * 6.2832f;
            point_at(trunk, 0.97f, p, d, r);
            for (int j = 0; j < nt; ++j) {
                const glm::vec3 ld = side_dir(d, a0 + 6.2832f * (float(j) + rr(-0.2f, 0.2f)) / float(nt), rr(0.40f, 0.72f));
                limb(p, ld, rt * (nt == 2 ? 0.80f : nt == 3 ? 0.70f : 0.62f) * rr(0.9f, 1.05f), 0.0f, 6,
                     0.10f + 0.20f * age, 0.30f, pr(0.0f, 0.0f, 0.04f));
            }
            // ... and lower limbs stand out from it, the lowest the widest.
            const int ns = 2 + int(age * 3.0f + rng.u() * 2.0f);
            for (int j = 0; j < ns; ++j) {
                const float t = (float(j) + rng.u()) / float(ns);
                point_at(trunk, glm::mix(hb / Lt, 0.88f, t), p, d, r);
                gold += 2.39996f;
                const float ang = glm::mix(1.30f, 0.90f, t) + rr(-0.12f, 0.12f);
                limb(p, side_dir(d, gold, ang), r * rr(0.36f, 0.50f), std::min(r / std::sin(ang), 2.0f * r), 5,
                     0.15f + 0.35f * age, 0.35f, pr(0.0f, 0.0f, 0.10f));
            }
        } else {
            const int nl = int((birch ? 12.0f + 6.0f * age : 9.0f) * dens + rng.u() * 3.0f);
            for (int j = 0; j < nl; ++j) {
                const float t = (float(j) + rng.u()) / float(nl);
                point_at(trunk, glm::mix(hb / Lt, 0.92f, t), p, d, r);
                gold += 2.39996f;
                const float ang = birch ? rr(0.55f, 0.85f) : rr(0.8f, 1.1f);
                limb(p, side_dir(d, gold, ang), std::max(0.006f, r * rr(0.32f, 0.45f)),
                     r > 0.035f ? std::min(r / std::sin(ang), 2.0f * r) : 0.0f, 4,
                     birch ? 0.0f : 0.15f, birch ? -0.45f : 0.30f, pr(0.0f, 0.0f, 0.50f));      // (many thin limbs: they thin out with distance)
            }
            bear_twigs(trunk, 0.0f, 0.86f, 1.0f, 4, tw, 3);
        }
    }

    // ---- Spruce (3) ----
    void grow_spruce(float H) {
        const TreeHabit& h = o.habit;
        const glm::vec3 Y(0.0f, 1.0f, 0.0f);
        const float age = glm::clamp(h.age, 0.0f, 1.0f);
        const float dens = std::max(h.density, 0.3f);
        const float r0 = std::min(0.9f, H * 0.012f * (0.8f + 0.6f * age) + 0.03f);
        const glm::vec3 asym_v = glm::vec3(std::cos(h.asym_az), 0.0f, std::sin(h.asym_az)) * h.asym;
        Axis ta = mk(glm::vec3(0.0f), Y, H * 0.97f, r0, 0, 10, 8, 0.0f);
        ta.kick = glm::vec3(std::cos(h.lean_az), 0.0f, std::sin(h.lean_az)) * h.lean;
        ta.lift = h.lean * 1.3f; ta.wander = 0.012f; ta.tip = 0.04f;
        const auto trunk = grow_axis(ta);

        const float bare = glm::mix(0.07f, 0.30f, age) * rr(0.8f, 1.2f);
        const float Rmax = H * 0.27f * h.crown_w;
        const int   nwh = std::clamp(int((1.0f - bare) / 0.058f * dens), 7, 18);
        const int   leaves = std::max(o.leaf_count, 1);
        float az0 = rng.u() * 6.2832f;
        glm::vec3 p, d; float r;
        for (int w = 0; w < nwh; ++w) {
            const float t = (float(w) + 0.5f) / float(nwh);                  // 0 = the lowest whorl, 1 = the top
            const int   nb = t > 0.85f ? 3 : 4 + (rng.u() < 0.5f ? 1 : 0);
            az0 += 2.39996f;
            for (int b = 0; b < nb; ++b) {
                const float az = az0 + 6.2832f * (float(b) + rr(-0.18f, 0.18f)) / float(nb);
                point_at(trunk, bare + (0.985f - bare) * t + rr(-0.010f, 0.010f), p, d, r);
                const glm::vec3 out(std::cos(az), 0.0f, std::sin(az));
                // Longest at the bottom of the crown, a spire at the top; the lowest, shaded whorls fall short.
                const float len = Rmax * (std::pow(1.0f - t, 0.8f) * 0.96f + 0.05f) * rr(0.82f, 1.10f) * (t < 0.12f ? 0.8f : 1.0f) *
                                  (1.0f + glm::dot(out, asym_v));
                // Rising at the top, level in the middle, drooping below - and every tip turns up.
                const float ang = glm::mix(1.58f + 0.32f * age, 0.90f, glm::smoothstep(0.0f, 1.0f, t)) + rr(-0.08f, 0.08f);
                const glm::vec3 bd = glm::normalize(out * std::sin(ang) + Y * std::cos(ang));
                Axis a = mk(p, bd, len, std::max(0.006f, std::min(r * 0.34f, 0.010f + 0.022f * len)), 1, len > 0.12f * H ? 5 : 3, 4,
                            pr(0.0f, 0.02f, 0.85f));
                a.sag = glm::mix(0.45f, 0.0f, t) * (0.5f + age); a.lift = glm::mix(0.55f, 0.15f, t); a.wander = 0.04f; a.tip = 0.12f;
                const auto bs = grow_axis(a);
                // Side shoots to the left and right of the bough, hanging a little: a flat spray.
                const int nl = std::clamp(int(len / (0.06f * H) * dens + 0.5f), 1, 5);
                float side = rng.u() < 0.5f ? 0.0f : kPi;
                for (int k = 0; k < nl; ++k) {
                    glm::vec3 p2, d2; float r2;
                    const float f = 0.18f + 0.72f * (float(k) + rng.u()) / float(nl);
                    point_at(bs, f, p2, d2, r2);
                    glm::vec3 sd = side_dir(d2, side + rr(-0.25f, 0.25f), rr(0.75f, 1.05f));
                    side += kPi;
                    sd.y -= 0.18f + 0.25f * age;
                    const float l2 = len * (0.42f - 0.20f * f) * rr(0.8f, 1.15f);
                    Axis sa = mk(p2, sd, l2, std::min(0.004f + 0.0004f * H, r2 * 0.7f), 2, l2 > 0.06f * H ? 2 : 1, 3, pr(a.prio, 0.05f, 0.60f));
                    sa.sag = 0.25f; sa.wander = 0.08f; sa.tip = 0.3f;
                    const auto ss = grow_axis(sa);
                    bear_leaves(ss, 0.12f, leaves, 0.5f, 0.12f + 0.2f * age);
                }
                bear_leaves(bs, 0.35f, std::min(leaves, 3), 0.5f, 0.1f);
            }
        }
        bear_leaves(trunk, 0.93f, 3, 0.6f, 0.0f);           // the leader
    }

    // ---- Pine (4) ----
    void grow_pine(float H) {
        const TreeHabit& h = o.habit;
        const glm::vec3 Y(0.0f, 1.0f, 0.0f);
        const float age = glm::clamp(h.age, 0.0f, 1.0f);
        const bool  young = age < 0.3f;                              // still a cone with a leader
        const float dens = std::max(h.density, 0.3f);
        const float r0 = std::min(0.9f, H * 0.014f * (0.8f + 0.6f * age) + 0.03f);
        const glm::vec3 asym_v = glm::vec3(std::cos(h.asym_az), 0.0f, std::sin(h.asym_az)) * h.asym;
        const float bare = (young ? 0.22f : glm::mix(0.36f, 0.58f, age)) * rr(0.92f, 1.08f);
        const float lt = young ? 0.95f : 0.84f;                      // the stem's length / H
        Axis ta = mk(glm::vec3(0.0f), Y, H * lt, r0, 0, 10, 8, 0.0f);
        ta.kick = glm::vec3(std::cos(h.lean_az), 0.0f, std::sin(h.lean_az)) * h.lean;
        ta.lift = h.lean * 1.2f; ta.wander = 0.035f; ta.tip = young ? 0.06f : 0.30f; ta.cap = !young;
        const auto trunk = grow_axis(ta);

        const float Rmax = H * (young ? 0.26f : glm::mix(0.30f, 0.38f, age)) * h.crown_w;
        const int   cards = std::max(o.leaf_count, 1);
        glm::vec3 p, d; float r;
        // A limb: crooked, its end turned up; side branches all round but underneath, each with
        // short shoots, and a tuft of needles at every shoot's end.
        auto limb = [&](glm::vec3 lp, glm::vec3 ld, float len, float rad, float collar, float sag, float prio) {
            Axis a = mk(lp, ld, len, rad, 1, len > 0.14f * H ? 6 : 4, 5, prio);
            a.sag = sag; a.lift = rr(0.5f, 0.8f); a.wander = 0.10f; a.tip = 0.14f; a.collar = collar;
            const auto ls = grow_axis(a);
            const int n2 = std::clamp(int(len / (0.050f * H) * dens + 0.5f), 2, 6);
            const int nsh = age > 0.5f ? 3 : 2;
            float gold = rng.u() * 6.2832f;
            auto shoots = [&](const std::vector<Section>& s, float base, int n, float f0) {
                float g2 = rng.u() * 6.2832f;
                for (int k = 0; k < n; ++k) {
                    glm::vec3 p3, d3; float r3;
                    point_at(s, f0 + (0.95f - f0) * (float(k) + rng.u()) / float(n), p3, d3, r3);
                    g2 += 2.39996f;
                    float az = g2;
                    if (std::sin(az) < -0.3f) az = -az;
                    Axis t = mk(p3, side_dir(d3, az, rr(0.6f, 0.95f)), H * rr(0.030f, 0.050f), std::min(0.004f + 0.0004f * H, r3 * 0.7f),
                                3, 1, 3, pr(base, 0.18f, 0.50f));
                    t.lift = 0.5f; t.tip = 0.4f;
                    const auto ts = grow_axis(t);
                    tuft(ts.back().origin, glm::normalize(ts.back().origin - ts.front().origin), cards);
                }
                glm::vec3 pe, de; float re;
                point_at(s, 1.0f, pe, de, re);
                tuft(pe, de, cards);
            };
            for (int k = 0; k < n2; ++k) {
                glm::vec3 p2, d2; float r2;
                const float f = 0.30f + 0.65f * (float(k) + rng.u()) / float(n2);
                point_at(ls, f, p2, d2, r2);
                gold += 2.39996f;
                float az = gold;
                if (std::sin(az) < -0.3f) az = -az;
                const float l2 = len * (0.42f - 0.15f * f) * rr(0.8f, 1.15f);
                Axis sa = mk(p2, side_dir(d2, az, rr(0.6f, 0.9f)), l2, std::max(0.004f, r2 * rr(0.45f, 0.6f)), 2, 3, 3, pr(prio, 0.10f, 0.50f));
                sa.lift = 0.5f; sa.wander = 0.10f; sa.tip = 0.25f;
                const auto ss = grow_axis(sa);
                shoots(ss, sa.prio, nsh, 0.40f);
            }
            shoots(ls, prio, 2, 0.72f);
        };

        const int nwh = std::clamp(int((lt - bare) / glm::mix(0.070f, 0.050f, age) * dens), 3, young ? 7 : 9);
        float az0 = rng.u() * 6.2832f;
        for (int w = 0; w < nwh; ++w) {
            const float t = (float(w) + 0.5f) / float(nwh);
            const int   nb = (young ? 3 : 2) + int(rng.u() * (young ? 1.5f : 2.4f));
            az0 += 2.39996f;
            for (int b = 0; b < nb; ++b) {
                const float az = az0 + 6.2832f * (float(b) + rr(-0.25f, 0.25f)) / float(nb);
                point_at(trunk, (bare + (lt * 0.97f - bare) * t) / lt + rr(-0.012f, 0.012f), p, d, r);
                const glm::vec3 out(std::cos(az), 0.0f, std::sin(az));
                const float shape = young ? std::pow(1.0f - t, 0.7f) * 0.95f + 0.08f
                                          : 0.55f + 0.45f * std::sin(kPi * std::min(t * 1.15f, 1.0f));
                const float len = Rmax * shape * rr(0.75f, 1.10f) * (1.0f + glm::dot(out, asym_v));
                const float ang = glm::mix(1.45f, 0.75f, t) + rr(-0.12f, 0.12f);
                const glm::vec3 ld = glm::normalize(out * std::sin(ang) + Y * std::cos(ang));
                limb(p, ld, len, std::max(0.008f, r * rr(0.32f, 0.44f)), std::min(r / std::sin(ang), 2.0f * r),
                     0.30f * (1.0f - t) * (0.4f + age), pr(0.0f, 0.0f, 0.75f));
            }
        }
        if (!young) {
            // The top: the stem ends in a few rising limbs - the rounded head of a grown pine.
            const float rt = trunk[trunk.size() - 2].radius;
            const int   nt = 3 + int(rng.u() * 1.5f);
            const float a0 = rng.u() * 6.2832f;
            point_at(trunk, 0.97f, p, d, r);
            for (int j = 0; j < nt; ++j)
                limb(p, side_dir(d, a0 + 6.2832f * (float(j) + rr(-0.2f, 0.2f)) / float(nt), rr(0.35f, 0.80f)),
                     Rmax * rr(0.50f, 0.75f), rt * 0.62f, 0.0f, 0.0f, pr(0.0f, 0.0f, 0.06f));
            // Dead stubs on the bare stem.
            const int nd = 2 + int(age * 4.0f * rng.u() + 0.5f);
            for (int j = 0; j < nd; ++j) {
                point_at(trunk, bare / lt * rr(0.45f, 0.95f), p, d, r);
                Axis s = mk(p, side_dir(d, rng.u() * 6.2832f, rr(1.1f, 1.5f)), r + H * rr(0.02f, 0.05f), std::max(0.006f, r * 0.14f), 2, 2, 3,
                            pr(0.0f, 0.30f, 0.60f));
                s.sag = 0.3f; s.wander = 0.15f; s.tip = 0.35f;
                grow_axis(s);
            }
        } else {
            tuft(trunk.back().origin, Y, cards);
        }
    }

    void run_habit(float H) {
        switch (o.habit.kind) {
            case 3:  grow_spruce(H); break;
            case 4:  grow_pine(H); break;
            default: grow_broadleaf(H); break;
        }
    }
};

void push(game::LevelMesh& m, glm::vec3 p, glm::vec3 n, float u, float v) {
    m.verts.push_back({ p, n, u, v });
    m.bmin = glm::min(m.bmin, p);
    m.bmax = glm::max(m.bmax, p);
}

}  // namespace

EzTreeOptions ez_preset_oak() {
    // presets/oak_medium.json
    EzTreeOptions o;
    o.seed = 35729; o.evergreen = false; o.levels = 3;
    const float angle[4] = { 0, 54, 58, 32 };           const int children[4] = { 6, 4, 3, 0 };
    const float gn[4] = { 0.0f, -0.1f, -0.15f, 0.09f };  const float len[4] = { 37.24f, 11.08f, 12.39f, 7.16f };
    const float rad[4] = { 1.41f, 0.9f, 0.69f, 1.19f };  const int sec[4] = { 8, 6, 3, 1 };
    const int seg[4] = { 7, 5, 3, 3 };                   const float st[4] = { 0, 0.49f, 0.06f, 0.12f };
    const float tp[4] = { 0.73f, 0.42f, 0.69f, 0.75f };  const float tw[4] = { -0.23f, 0.42f, 0, 0 };
    for (int i = 0; i < 4; ++i) {
        o.angle[i] = angle[i]; o.children[i] = children[i]; o.gnarliness[i] = gn[i]; o.length[i] = len[i];
        o.radius[i] = rad[i]; o.sections[i] = sec[i]; o.segments[i] = seg[i]; o.start[i] = st[i];
        o.taper[i] = tp[i]; o.twist[i] = tw[i];
    }
    o.force_strength = 0.02f;
    o.leaf_angle = 42; o.leaf_count = 18; o.leaf_start = 0.16f; o.leaf_size = 2.5f; o.leaf_size_variance = 0.7f;
    o.leaf_double = true; o.leaf_kind = 0.0f;
    return o;
}

EzTreeOptions ez_preset_pine() {
    // presets/pine_medium.json
    EzTreeOptions o;
    o.seed = 13977; o.evergreen = true; o.levels = 1;
    const float angle[4] = { 0, 110, 16, 60 };          const int children[4] = { 82, 3, 5, 0 };
    const float gn[4] = { 0.05f, 0.08f, 0, 0 };          const float len[4] = { 50, 23.87f, 14.08f, 1 };
    const float rad[4] = { 1.05f, 0.36f, 0.7f, 0.7f };   const int sec[4] = { 12, 10, 8, 6 };
    const int seg[4] = { 8, 6, 4, 3 };                   const float st[4] = { 0, 0.27f, 0.14f, 0.3f };
    for (int i = 0; i < 4; ++i) {
        o.angle[i] = angle[i]; o.children[i] = children[i]; o.gnarliness[i] = gn[i]; o.length[i] = len[i];
        o.radius[i] = rad[i]; o.sections[i] = sec[i]; o.segments[i] = seg[i]; o.start[i] = st[i];
        o.taper[i] = 0.7f; o.twist[i] = 0.0f;
    }
    o.force_strength = -0.003f;
    o.leaf_angle = 39; o.leaf_count = 30; o.leaf_start = 0.09f; o.leaf_size = 1.435f; o.leaf_size_variance = 0.201f;
    o.leaf_double = true; o.leaf_kind = 1.0f;
    return o;
}

EzTreeOptions ez_preset_bush() {
    // A multi-stemmed shrub: no real trunk, stems fan out from the base
    // and fork once; leaf sprays all along the outer branches.
    EzTreeOptions o;
    o.seed = 4242; o.evergreen = false; o.levels = 2;
    const float angle[4] = { 0, 62, 48, 40 };           const int children[4] = { 9, 4, 0, 0 };
    const float gn[4] = { 0.10f, 0.18f, 0.25f, 0.0f };   const float len[4] = { 4.0f, 9.0f, 6.0f, 1.0f };
    const float rad[4] = { 0.55f, 0.55f, 0.6f, 0.6f };   const int sec[4] = { 3, 5, 3, 1 };
    const int seg[4] = { 5, 4, 3, 3 };                   const float st[4] = { 0, 0.0f, 0.25f, 0.3f };
    const float tp[4] = { 0.5f, 0.6f, 0.7f, 0.7f };
    for (int i = 0; i < 4; ++i) {
        o.angle[i] = angle[i]; o.children[i] = children[i]; o.gnarliness[i] = gn[i]; o.length[i] = len[i];
        o.radius[i] = rad[i]; o.sections[i] = sec[i]; o.segments[i] = seg[i]; o.start[i] = st[i];
        o.taper[i] = tp[i]; o.twist[i] = 0.0f;
    }
    o.force_strength = 0.012f;
    o.leaf_angle = 50; o.leaf_count = 5; o.leaf_start = 0.1f; o.leaf_size = 5.0f; o.leaf_size_variance = 0.4f;
    o.leaf_double = true; o.leaf_kind = 0.0f;
    return o;
}

EzTreeOptions ez_preset_river_shrub() {
    // The scrub of a river bank: a sallow (bush willow). Many long, whippy stems from one
    // stool, rising steeply and then arching outward under their own weight, forking once or
    // twice, with small narrow leaves all along the outer two thirds - an open, airy dome
    // taller than a man, quite unlike the dense round bushes of the forest edge.
    EzTreeOptions o;
    o.seed = 7717; o.evergreen = false; o.levels = 2;
    const float angle[4] = { 0, 34, 40, 40 };           const int children[4] = { 11, 3, 0, 0 };
    const float gn[4] = { 0.06f, 0.10f, 0.16f, 0.0f };   const float len[4] = { 2.2f, 15.0f, 8.0f, 1.0f };
    const float rad[4] = { 0.50f, 0.42f, 0.55f, 0.6f };  const int sec[4] = { 2, 7, 4, 1 };
    const int seg[4] = { 5, 4, 3, 3 };                   const float st[4] = { 0, 0.0f, 0.30f, 0.3f };
    const float tp[4] = { 0.4f, 0.72f, 0.8f, 0.7f };
    for (int i = 0; i < 4; ++i) {
        o.angle[i] = angle[i]; o.children[i] = children[i]; o.gnarliness[i] = gn[i]; o.length[i] = len[i];
        o.radius[i] = rad[i]; o.sections[i] = sec[i]; o.segments[i] = seg[i]; o.start[i] = st[i];
        o.taper[i] = tp[i]; o.twist[i] = 0.0f;
    }
    o.force_dir = glm::vec3(0.0f, -1.0f, 0.0f);          // the stems bow over
    o.force_strength = 0.0045f;
    o.leaf_angle = 58; o.leaf_count = 9; o.leaf_start = 0.22f; o.leaf_size = 3.0f; o.leaf_size_variance = 0.45f;
    o.leaf_double = true; o.leaf_kind = 2.0f;            // the small-leaf cell
    return o;
}

void build_ez_tree(
game::LevelMesh& m, const EzTreeOptions& o, float max_detail, float target_height) {
    Gen g{ o, Rng{ o.seed * 2654435761u + 977u }, {}, {}, {} };
    if (o.habit.kind != 0) g.run_habit(target_height);
    else g.run();
    float top = 1e-3f;
    for (const auto& b : g.branches) for (const auto& s : b.sections) top = std::max(top, s.origin.y);
    for (const auto& l : g.leaves) top = std::max(top, l.origin.y);
    const float sc = target_height / top;
    // Priorities are compressed to 0..0.78: veg.vert grows an element over
    // detail - prio in (0.03 + 0.25 prio), so anything above ~0.78 never
    // reached full size even at full detail.
    auto sp = [](float p) { return glm::clamp(p, 0.0f, 1.0f) * 0.78f; };
    auto qprio = [](float p) { return std::floor(glm::clamp(p, 0.0f, 1.0f) * 1023.0f); };

    // ---- Branches: tubes through the section rings ----
    // Trunk and main limbs (levels 0-1) keep their full tessellation in
    // every detail bucket: they are the big silhouettes, so losing sides or
    // rings at a bucket switch was a visible pop (and they are cheap next
    // to the leaves). Only thin twigs get coarser.
    for (const auto& b : g.branches) {
        if (sp(b.prio) > max_detail) continue;
        const bool major = b.level <= 1;
        const int stride = (major || max_detail > 0.6f) ? 1 : 2;
        const int segs = major ? b.segments
                               : std::max(3, int(std::lround(float(b.segments) * glm::mix(0.5f, 1.0f, max_detail))));
        std::vector<Section> ring;
        for (size_t i = 0; i < b.sections.size(); i += size_t(stride)) ring.push_back(b.sections[i]);
        if ((b.sections.size() - 1) % size_t(stride) != 0) ring.push_back(b.sections.back());
        if (ring.size() < 2) continue;
        const uint32_t base = uint32_t(m.verts.size());
        const float qp = qprio(sp(b.prio));
        for (const Section& s : ring) {
            const glm::vec3 c = s.origin * sc;
            const float r = glm::clamp(std::max(s.radius, 0.004f) * sc, 0.002f, 0.98f);
            for (int j = 0; j < segs; ++j) {
                const float a = 2.0f * kPi * float(j) / float(segs);
                const glm::vec3 n = glm::normalize(s.q * glm::vec3(std::cos(a), 0.0f, std::sin(a)));
                push(m, c + n * r, n, qp + r, 8.0f);
            }
        }
        for (size_t i = 0; i + 1 < ring.size(); ++i)
            for (int j = 0; j < segs; ++j) {
                const uint32_t v1 = base + uint32_t(i) * uint32_t(segs) + uint32_t(j);
                const uint32_t v2 = base + uint32_t(i) * uint32_t(segs) + uint32_t((j + 1) % segs);
                const uint32_t v3 = v1 + uint32_t(segs), v4 = v2 + uint32_t(segs);
                m.idx.insert(m.idx.end(), { v1, v3, v2, v2, v3, v4 });
            }
    }

    // ---- Roots: a flared foot round the trunk and roots running out of it into the ground ----
    // (this engine's own, not part of EZ-Tree.) The trunk used to end as a plain cylinder
    // stuck in the soil. Now its lowest part widens into buttresses, and from each buttress a
    // root arches out and dives into the ground, some of them forking. How many, how long and
    // how heavy follows `o.roots` and the tree's own random numbers (a separate stream: the
    // crown stays that of the seed). They rise well clear of the ground near the trunk (the
    // drawn ground can stand up to ~0.25 m above the tree's base close to the camera) and end
    // deep enough to stay buried on a slope. All of it is ordinary bark tube (priority near
    // 0), so it is lit, shadowed and ray traced with the trunk.
    if (o.roots > 0.01f && !g.branches.empty() && !g.branches[0].sections.empty()) {
        Rng rr{ o.seed * 747796405u + 2891336453u };
        const float R  = glm::clamp(g.branches[0].sections[0].radius * sc, 0.02f, 0.60f);     // trunk radius at the ground (m)
        const float k  = glm::clamp(o.roots, 0.0f, 1.5f);
        const int   n  = std::clamp(3 + int(k * 3.0f + rr.u() * 2.0f), 3, 8);
        struct Root { float ang, len, r0, hs; };
        std::vector<Root> roots;
        const float a0 = rr.u() * 2.0f * kPi;
        for (int j = 0; j < n; ++j) {
            Root r;
            r.ang = a0 + 2.0f * kPi * (float(j) + rr.random(0.32f, -0.32f)) / float(n);
            const float big = rr.u();                                        // some roots are much heavier than others
            r.len = R * (2.2f + 3.6f * big) * (0.55f + 0.55f * k) + 0.25f;
            r.r0  = R * (0.24f + 0.26f * big) * (0.70f + 0.30f * std::min(k, 1.0f));
            r.hs  = std::max(0.30f, R * (0.55f + 0.75f * big) * (0.6f + 0.5f * k));
            roots.push_back(r);
        }
        const bool fine = max_detail > 0.3f;

        // The flared foot: a sleeve round the trunk's first stretch, bulging toward each root.
        {
            const float sec_len = (g.branches[0].sections.size() > 1)
                ? glm::length(g.branches[0].sections[1].origin - g.branches[0].sections[0].origin) * sc : 3.0f * R;
            const float Hf = std::min(R * (2.2f + 1.2f * k), sec_len * 0.95f);
            const int segs = fine ? 16 : 8;
            const float ys[6] = { -0.55f, 0.0f, 0.14f, 0.34f, 0.64f, 1.0f };       // x Hf (the first: buried, in metres)
            // (the top ring lies just INSIDE the trunk: the sleeve then leaves the bark along a slanting
            // line where it has almost no flare left, and no ledge shows where it ends)
            const uint32_t base = uint32_t(m.verts.size());
            for (int i = 0; i < 6; ++i) {
                const float y = ys[i] < 0.0f ? ys[i] : ys[i] * Hf;
                const float f = ys[i] <= 0.0f ? 1.0f : std::exp(-y / (0.9f * R + 0.25f * Hf)) * (1.0f - ys[i]);
                for (int j = 0; j < segs; ++j) {
                    const float a = 2.0f * kPi * float(j) / float(segs);
                    float lobe = 0.0f;
                    for (const Root& r : roots) {
                        const float c = std::max(std::cos(a - r.ang), 0.0f);
                        lobe = std::max(lobe, std::pow(c, 5.0f) * (r.r0 / (0.5f * R)));
                    }
                    const float rad = std::min(R * (0.985f + f * (0.19f + 0.62f * lobe) * (0.55f + 0.45f * k)), 0.97f);
                    const glm::vec3 nrm(std::cos(a), 0.0f, std::sin(a));
                    push(m, glm::vec3(0.0f, y, 0.0f) + nrm * rad, nrm, 0.0f + rad, 8.0f);
                }
            }
            for (int i = 0; i + 1 < 6; ++i)
                for (int j = 0; j < segs; ++j) {
                    const uint32_t v1 = base + uint32_t(i * segs + j), v2 = base + uint32_t(i * segs + (j + 1) % segs);
                    const uint32_t v3 = v1 + uint32_t(segs), v4 = v2 + uint32_t(segs);
                    m.idx.insert(m.idx.end(), { v1, v3, v2, v2, v3, v4 });
                }
        }

        // One root: a tube from (d0, y0) at angle `ang`, `len` long, arching down into the ground.
        auto emit_root = [&](float ang, float d0, float y0, float len, float r0, float prio, float bend) {
            // (the random numbers are drawn whether or not this detail level keeps the root:
            // every level of one tree must grow the same roots)
            const float ph = rr.u() * 6.28f, wig = rr.random(0.30f, 0.10f);
            if (sp(prio) > max_detail) return;
            const int N = fine ? 7 : 4, segs = fine ? 6 : 4;
            const float sink = 0.40f + 0.22f * len;                               // how deep the tip ends
            std::vector<glm::vec3> c(static_cast<size_t>(N + 1));
            for (int i = 0; i <= N; ++i) {
                const float t = float(i) / float(N);
                const float a = ang + bend * t + wig * std::sin(t * 4.1f + ph) * t;
                const float d = d0 + len * t;
                const float y = y0 * (1.0f - t) * (1.0f - t) - sink * t * t + 0.05f * R * std::sin(t * 9.0f + ph) * (1.0f - t);
                c[size_t(i)] = glm::vec3(std::cos(a) * d, y, std::sin(a) * d);
            }
            const uint32_t base = uint32_t(m.verts.size());
            const float qp = qprio(sp(prio));
            for (int i = 0; i <= N; ++i) {
                const float t = float(i) / float(N);
                const glm::vec3 tan = glm::normalize(c[size_t(std::min(i + 1, N))] - c[size_t(std::max(i - 1, 0))]);
                const glm::vec3 side = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), tan));
                const glm::vec3 upv = glm::cross(tan, side);
                const float rad = glm::clamp(r0 * (1.0f - 0.80f * t), 0.006f, 0.97f);
                for (int j = 0; j < segs; ++j) {
                    const float a = 2.0f * kPi * float(j) / float(segs);
                    const glm::vec3 nrm = glm::normalize(side * std::cos(a) + upv * std::sin(a));
                    push(m, c[size_t(i)] + nrm * rad, nrm, qp + rad, 8.0f);
                }
            }
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < segs; ++j) {
                    const uint32_t v1 = base + uint32_t(i * segs + j), v2 = base + uint32_t(i * segs + (j + 1) % segs);
                    const uint32_t v3 = v1 + uint32_t(segs), v4 = v2 + uint32_t(segs);
                    m.idx.insert(m.idx.end(), { v1, v3, v2, v2, v3, v4 });
                }
        };
        for (const Root& r : roots) {
            const float bend = rr.random(0.45f, -0.45f);
            const float prio = 0.02f + 0.10f * rr.u();
            emit_root(r.ang, 0.45f * R, r.hs, r.len, r.r0, prio, bend);
            // A side root forking off the heavier ones.
            const float fk = rr.u(), fs = rr.u(), fl = rr.u();
            const float t = 0.30f + 0.25f * fs;
            const float d = 0.45f * R + r.len * t;
            const float y = r.hs * (1.0f - t) * (1.0f - t) - (0.40f + 0.22f * r.len) * t * t;
            if (k > 0.5f && fk < 0.30f + 0.25f * k)
                emit_root(r.ang + bend * t, d, y, r.len * (0.35f + 0.35f * fl), r.r0 * (1.0f - 0.80f * t) * 0.85f,
                          0.10f + 0.12f * fl, (fs < 0.5f ? 1.0f : -1.0f) * (0.7f + 0.5f * fl));
            else { rr.u(); rr.u(); }        // (keeps the stream in step with the branch taken above)
        }
    }

    // ---- Leaves: cards. A second, crossed card per leaf joins in at the
    // upper half of the detail range. ----
    m.card_index_start = uint32_t(m.idx.size());
    m.card_cell = int(o.leaf_kind + 0.5f);
    const float mat = o.leaf_kind < 0.5f ? 7.0f : 8.0f + std::floor(o.leaf_kind + 0.5f);
    for (const Leaf& l : g.leaves) {
        const float size = l.size * sc;
        const glm::vec3 origin = l.origin * sc;
        const int cards = o.leaf_double ? 2 : 1;
        for (int cIdx = 0; cIdx < cards; ++cIdx) {
            const float prio = cIdx == 0 ? l.prio : 0.5f + 0.5f * l.prio;
            if (sp(prio) > max_detail) continue;
            const float qp = qprio(sp(prio)) * 4.0f;
            const glm::quat q = l.q * axis_q(glm::vec3(0, 1, 0), cIdx == 0 ? 0.0f : kPi * 0.5f);
            const glm::vec3 x = q * glm::vec3(size * 0.5f, 0.0f, 0.0f), y = q * glm::vec3(0.0f, size, 0.0f);
            const uint32_t k = uint32_t(m.verts.size());
            // corner id = u + 2 v
            push(m, origin - x,     -x,     qp + 0.0f, mat);
            push(m, origin + x,      x,     qp + 1.0f, mat);
            push(m, origin - x + y, -x + y, qp + 2.0f, mat);
            push(m, origin + x + y,  x + y, qp + 3.0f, mat);
            m.idx.insert(m.idx.end(), { k, k + 1, k + 3, k, k + 3, k + 2 });
        }
    }
}

}  // namespace vibe
