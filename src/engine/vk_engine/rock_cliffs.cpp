// THE CLIFF WALLS: steep ground round the camera covered with rock-wall pieces (see
// rock_formations.h, "the cliff walls"). The pieces are the cliff kit
// (tools/rocks/make_formations.py --set cliffs -> assets/cliffs/cliffs.rockf, a third kit after
// the rock kit and the shore kit): 24 walls in four kinds of rock (layered sandstone, blocky
// granite, fractured slate, rounded weathered rock), three sizes, two shapes each - every one
// built round the plane of a 65-degree slope, its relief going back into that plane along all
// four edges, so pieces laid over and beside each other meet without a step.
//
// THE RULE, per 64 m cell, from the ground as it is drawn (no state; the same on every PC):
//   1. the ground's height every 2 m; a sample is STEEP where its slope is over 55 degrees
//      (VIBE_CLIFF_SLOPE=<degrees>) and it is above the water;
//   2. the steep samples in an order by hash of their place; the first one not yet covered gets
//      a piece - the largest that FITS: a large wall (about 30 m wide), else a medium (14 m),
//      else a small one (6 m), its kind of rock by the 384 m region it lies in;
//   3. a piece is laid on the slope: its front down the fall line (yaw, +-7 degrees by hash),
//      tilted from its own 65 degrees to the ground's by up to 15 degrees, the middle of its
//      plane on the ground there; then the ground is read at 5 x 5 points of its plane - it is
//      let down until no point of its edge stands more than 0.3 m out of the ground (its edges
//      lie in the plane: what stands out is its relief), and it FITS when its middle is then not
//      buried by more than half its relief (on ground that bends away - a cliff's top, a
//      gully - a smaller piece is tried);
//   4. every steep sample the piece's face lies over is covered; on to the next.
// Nothing stands where the caller says no (roads, buildings, the castle, the station: Env::
// blocked), under water, or on top of a fixed formation. At most 48 pieces a cell.
// STREAMED: cells whose middle is within 420 m of the camera are made (one a frame, nearest
// first) and given back beyond 560 m. A formation keeps its SLOT while it lives; the engine
// (rock_formations_gpu.cpp) draws, traces and collides with the slots as with the fixed list.
#include "engine/vk_engine/rock_formations.h"
#include "engine/log.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace vibe::rockf {

namespace {

constexpr float kCell = 64.0f, kGrid = 2.0f;
constexpr float kLoad = 420.0f, kFree = 560.0f;
constexpr float kLeanTan = 0.46630766f;                 // tan 25 degrees: the pieces' plane leans back this much (make_formations.py CLIFF_TL)
constexpr float kNominal = 1.1344640f;                   // 65 degrees: the slope a piece is made for
constexpr float kMaxTilt = 0.2617994f;                   // 15 degrees
constexpr size_t kMaxSlots = 1400;
constexpr int kPerCell = 48;
constexpr float kTierW[3] = { 31.0f, 14.5f, 6.4f };     // the width aimed at per size (m)

int g_base = -1;
struct Shape { int piece = -1; float W = 1.0f, H = 1.0f, D = 1.0f, rel = 1.0f; };   // D: the block behind the plane
std::vector<Shape> g_shapes[4][3];                       // kind of rock x size
std::vector<Inst> g_slots;
std::vector<uint32_t> g_gen;
std::vector<uint32_t> g_free;
std::unordered_map<uint64_t, std::vector<uint32_t>> g_cells;
size_t g_live = 0;

uint32_t hash_u(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t h = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
float h01(uint32_t h) { return float(h >> 8) / 16777216.0f; }
uint64_t cell_key(int cx, int cz) { return (uint64_t(uint32_t(cx)) << 32) | uint64_t(uint32_t(cz)); }

float steep_tan() {
    static const float t = [] {
        float deg = 55.0f;
        if (const char* e = std::getenv("VIBE_CLIFF_SLOPE")) deg = std::clamp(float(std::atof(e)), 30.0f, 85.0f);
        return std::tan(glm::radians(deg));
    }();
    return t;
}

float water_at(const Env& env, float x, float z) { return env.level ? env.level(x, z) : env.water; }

// One piece laid on the slope at `surf` (see THE RULE, 3). False when it does not fit there.
bool fit(const Env& env, const Shape& S, glm::vec3 surf, glm::vec2 down, float angle, float scale, uint32_t h, Inst& in) {
    in = Inst{};
    in.piece = S.piece; in.scale = scale; in.why = 11; in.look = 0;
    in.yaw = std::atan2(down.x, down.y) + (h01(hash_u(h, 7u, 0u)) - 0.5f) * 0.24f;
    in.tilt = std::clamp(angle - kNominal, -kMaxTilt, kMaxTilt);
    in.pos = glm::vec3(0.0f);
    glm::mat4 M = transform(in);
    in.pos = surf - glm::vec3(M * glm::vec4(0.0f, 0.5f * S.H, -0.5f * S.H * kLeanTan, 1.0f));
    M = transform(in);
    float e[5][5], edge = -1e9f;
    for (int b = 0; b < 5; ++b)
        for (int a = 0; a < 5; ++a) {
            const float u = (-0.45f + 0.225f * float(a)) * S.W, v = (0.1f + 0.2f * float(b)) * S.H;
            const glm::vec3 q = glm::vec3(M * glm::vec4(u, v, -v * kLeanTan, 1.0f));
            const float g = env.ground(q.x, q.z);
            e[b][a] = g - q.y;
            if (a == 0 || a == 4 || b == 0 || b == 4) edge = std::max(edge, -e[b][a]);
        }
    const float drop = std::max(0.0f, edge + 0.3f);
    const float rel = S.rel * scale;
    float inner[9]; int n = 0;
    for (int b = 1; b < 4; ++b) for (int a = 1; a < 4; ++a) inner[n++] = e[b][a] + drop;
    std::nth_element(inner, inner + 4, inner + 9);
    if (inner[4] > 0.5f * rel) return false;                                         // (its middle would be under the ground)
    if (*std::max_element(inner, inner + 9) > 1.1f * rel) return false;
    if (*std::min_element(inner, inner + 9) < -0.4f) return false;                  // (its middle would stand off the ground: a pillar, not a skin)
    in.pos.y -= drop;
    M = transform(in);
    // The block behind the plane must stay in the hill: on a ridge or a bulge its back and its
    // ends would come out behind (bumps on the skyline).
    for (int b = 0; b < 3; ++b)
        for (int a = 0; a < 3; ++a)
            for (int k = 0; k < 2; ++k) {
                const float u = (-0.5f + 0.5f * float(a)) * S.W, v = (0.05f + 0.45f * float(b)) * S.H;
                const glm::vec3 q = glm::vec3(M * glm::vec4(u, v, -v * kLeanTan - (k ? 0.95f : 0.4f) * S.D, 1.0f));
                if (env.ground(q.x, q.z) < q.y - 0.2f) return false;
            }
    for (int a = 0; a < 3; ++a) {                                                    // (and its top edge in the slope)
        const float u = (-0.45f + 0.45f * float(a)) * S.W, v = 0.98f * S.H;
        const glm::vec3 q = glm::vec3(M * glm::vec4(u, v, -v * kLeanTan, 1.0f));
        if (env.ground(q.x, q.z) < q.y - 0.4f) return false;
    }
    const glm::vec3 mid = glm::vec3(M * glm::vec4(0.0f, 0.5f * S.H, -0.5f * S.H * kLeanTan, 1.0f));
    if (env.ground(mid.x, mid.z) < water_at(env, mid.x, mid.z) + 0.3f) return false;
    const float r = 0.45f * S.W * scale;
    if (env.blocked && env.blocked(mid.x, mid.z, r)) return false;
    for (const Inst& o : instances()) {
        const float ro = 0.5f * std::max(o.hi.x - o.lo.x, o.hi.z - o.lo.z);
        if (glm::distance(glm::vec2(0.5f * (o.lo.x + o.hi.x), 0.5f * (o.lo.z + o.hi.z)), glm::vec2(mid.x, mid.z)) < 0.55f * (r + ro)) return false;
    }
    const Piece& P = kit()[size_t(S.piece)];
    in.lo = glm::vec3(1e9f); in.hi = glm::vec3(-1e9f);
    for (int k = 0; k < 8; ++k) {
        const glm::vec3 l((k & 1) ? P.bmax.x : P.bmin.x, (k & 2) ? P.bmax.y : P.bmin.y, (k & 4) ? P.bmax.z : P.bmin.z);
        const glm::vec3 w = glm::vec3(M * glm::vec4(l, 1.0f));
        in.lo = glm::min(in.lo, w); in.hi = glm::max(in.hi, w);
    }
    in.reach = glm::clamp((in.hi.y - in.lo.y) / 18.0f, 1.0f, 3.0f);
    return true;
}

uint32_t take_slot(const Inst& in) {
    uint32_t s;
    if (!g_free.empty()) { s = g_free.back(); g_free.pop_back(); }
    else { s = uint32_t(g_slots.size()); g_slots.emplace_back(); g_gen.push_back(0u); }
    g_slots[s] = in;
    ++g_gen[s];
    ++g_live;
    return s;
}

void make_cell(const Env& env, int cx, int cz, std::vector<uint32_t>& out) {
    const float x0 = float(cx) * kCell, z0 = float(cz) * kCell;
    constexpr int N = int(kCell / kGrid);
    constexpr int G = N + 2;
    std::vector<float> hh(size_t(G) * G);
    for (int j = 0; j < G; ++j)
        for (int i = 0; i < G; ++i) hh[size_t(j) * G + i] = env.ground(x0 + float(i - 1) * kGrid, z0 + float(j - 1) * kGrid);
    struct Smp { glm::vec3 p; glm::vec2 down; float angle; uint32_t key; bool covered; };
    std::vector<Smp> smp;
    const float st = steep_tan();
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
            const float gx = (hh[size_t(j + 1) * G + i + 2] - hh[size_t(j + 1) * G + i]) / (2.0f * kGrid);
            const float gz = (hh[size_t(j + 2) * G + i + 1] - hh[size_t(j) * G + i + 1]) / (2.0f * kGrid);
            const float sl = std::sqrt(gx * gx + gz * gz);
            if (sl < st) continue;
            const float x = x0 + float(i) * kGrid, z = z0 + float(j) * kGrid, y = hh[size_t(j + 1) * G + i + 1];
            if (y < water_at(env, x, z) + 0.5f) continue;
            smp.push_back({ glm::vec3(x, y, z), glm::vec2(-gx, -gz) / sl, std::atan(sl), hash_u(uint32_t(int(x * 0.5f) + 65536), uint32_t(int(z * 0.5f) + 65536), 0xC11Fu), false });
        }
    std::sort(smp.begin(), smp.end(), [](const Smp& a, const Smp& b) { return a.key < b.key; });
    int made = 0;
    for (Smp& s : smp) {
        if (s.covered) continue;
        if (made >= kPerCell || g_live >= kMaxSlots) break;
        s.covered = true;
        // (the kind of rock: one per 384 m region)
        const int geo = int(hash_u(uint32_t(int(std::floor(s.p.x / 384.0f)) + 4096), uint32_t(int(std::floor(s.p.z / 384.0f)) + 4096), 0x6E0u) % 4u);
        Inst in; bool ok = false;
        for (int tier = 0; tier < 3 && !ok; ++tier) {
            const std::vector<Shape>& L = !g_shapes[geo][tier].empty() ? g_shapes[geo][tier] : g_shapes[0][tier];
            if (L.empty()) continue;
            const uint32_t h = hash_u(s.key, uint32_t(tier), 0x51u);
            const Shape& S = L[h % uint32_t(L.size())];
            const float scale = std::clamp(kTierW[tier] / S.W * (0.88f + 0.27f * h01(hash_u(h, 3u, 1u))), 0.7f, 1.3f);
            ok = fit(env, S, s.p, s.down, s.angle, scale, h, in);
        }
        if (!ok) continue;
        out.push_back(take_slot(in));
        ++made;
        // what its face lies over is covered
        const glm::mat4 Mi = glm::inverse(transform(in));
        const Piece& P = kit()[size_t(in.piece)];
        const float W = P.bmax.x - P.bmin.x, H = std::max(P.bmax.y, 1.0f);
        for (Smp& o : smp) {
            if (o.covered) continue;
            const glm::vec3 l = glm::vec3(Mi * glm::vec4(o.p, 1.0f));
            if (std::abs(l.x) < 0.42f * W && l.y > 0.06f * H && l.y < 0.9f * H && std::abs(l.z + l.y * kLeanTan) < 0.35f * H) o.covered = true;
        }
    }
    static const bool verbose = std::getenv("VIBE_LOG_CLIFFS") != nullptr || std::getenv("VIBE_LOG_ROCKS") != nullptr;
    if (verbose && !smp.empty()) {
        log::infof("[cliffs] cell (%d, %d): %zu steep samples, %d walls", cx, cz, smp.size(), made);
        for (uint32_t s : out) {             // (where each stands: a place to look from)
            const Inst& in = g_slots[s];
            log::infof("[cliffs]   %s at (%.1f, %.1f, %.1f) front (%.2f, %.2f) tilt %.0f deg size %.2f", kit()[size_t(in.piece)].name.c_str(),
                       double(0.5f * (in.lo.x + in.hi.x)), double(0.5f * (in.lo.y + in.hi.y)), double(0.5f * (in.lo.z + in.hi.z)),
                       double(std::sin(in.yaw)), double(std::cos(in.yaw)), double(glm::degrees(in.tilt)), double(in.scale));
        }
    }
}

}  // namespace

bool load_cliff_kit(const char* path) {
    if (!load_kit(path, true, &g_base) || g_base < 0) { g_base = -1; return false; }
    static const char* geos[4] = { "sand", "gran", "slate", "round" };
    int n = 0;
    for (size_t p = size_t(g_base); p < kit().size(); ++p) {
        const Piece& P = kit()[p];
        if (P.name.rfind("cliff_", 0) != 0) continue;
        for (int g = 0; g < 4; ++g) {
            const std::string pre = std::string("cliff_") + geos[g] + "_";
            if (P.name.rfind(pre, 0) != 0 || P.name.size() < pre.size() + 1) continue;
            const char t = P.name[pre.size()];
            const int tier = t == 'l' ? 0 : t == 'm' ? 1 : t == 's' ? 2 : -1;
            if (tier < 0) continue;
            Shape S;
            S.piece = int(p);
            S.W = std::max(P.bmax.x - P.bmin.x, 1.0f);
            S.H = std::max(P.bmax.y, 1.0f);
            S.D = std::max(-P.bmin.z - S.H * kLeanTan, 0.5f);
            S.rel = 0.1f * std::min(S.W, S.H) + 0.6f;                      // (make_formations.py cliff_blockout: what stands out of the plane)
            g_shapes[g][tier].push_back(S);
            ++n;
        }
    }
    log::infof("[cliffs] the cliff kit: %d walls from piece %d", n, g_base);
    return n > 0;
}

int cliff_base() { return g_base; }

bool cliff_stream(const Env& env, glm::vec3 eye, int budget) {
    static const bool off = std::getenv("VIBE_NO_CLIFF_WALLS") != nullptr;
    if (off || g_base < 0 || !env.ground) return false;
    bool changed = false;
    // given back: cells beyond kFree
    for (auto it = g_cells.begin(); it != g_cells.end();) {
        const int cx = int(int32_t(uint32_t(it->first >> 32))), cz = int(int32_t(uint32_t(it->first & 0xFFFFFFFFu)));
        const glm::vec2 c((float(cx) + 0.5f) * kCell, (float(cz) + 0.5f) * kCell);
        if (glm::distance(c, glm::vec2(eye.x, eye.z)) <= kFree) { ++it; continue; }
        for (uint32_t s : it->second) { g_slots[s].piece = -1; ++g_gen[s]; g_free.push_back(s); --g_live; }
        changed = changed || !it->second.empty();
        it = g_cells.erase(it);
    }
    // made: the nearest cells within kLoad that are not there yet
    const int ex = int(std::floor(eye.x / kCell)), ez = int(std::floor(eye.z / kCell));
    const int R = int(std::ceil(kLoad / kCell)) + 1;
    for (int k = 0; k < budget; ++k) {
        float best = 1e30f; int bx = 0, bz = 0;
        for (int dz = -R; dz <= R; ++dz)
            for (int dx = -R; dx <= R; ++dx) {
                const int cx = ex + dx, cz = ez + dz;
                const float d = glm::distance(glm::vec2((float(cx) + 0.5f) * kCell, (float(cz) + 0.5f) * kCell), glm::vec2(eye.x, eye.z));
                if (d > kLoad || d >= best || g_cells.count(cell_key(cx, cz))) continue;
                best = d; bx = cx; bz = cz;
            }
        if (best > 1e29f) break;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<uint32_t>& cell = g_cells[cell_key(bx, bz)];
        make_cell(env, bx, bz, cell);
        changed = changed || !cell.empty();
        static const bool verbose = std::getenv("VIBE_LOG_CLIFFS") != nullptr;
        if (verbose && !cell.empty())
            log::infof("[cliffs] cell (%d, %d) made in %.2f ms; %zu walls stand", bx, bz,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), g_live);
    }
    return changed;
}

const std::vector<Inst>& cliff_slots() { return g_slots; }
const std::vector<uint32_t>& cliff_slot_gen() { return g_gen; }

}  // namespace vibe::rockf
