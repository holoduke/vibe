#include "engine/region_tile.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace vibe::region {

namespace {
constexpr int kSteps = kTileN - 1;                         // 512
inline float sm(float t) { t = std::clamp(t, 0.0f, 1.0f); return t * t * (3.0f - 2.0f * t); }
const int kEdgeDir[4][2] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };       // W, E, N, S
// A corner's two edges and its diagonal step: NW, NE, SW, SE.
const int kCornerEdge[4][2] = { { 0, 2 }, { 1, 2 }, { 0, 3 }, { 1, 3 } };
const int kCornerDir[4][2]  = { { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 } };

uint32_t fnv1a(const uint8_t* p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
    return h;
}
template <class T> void put(std::vector<uint8_t>& o, size_t at, T v) { std::memcpy(o.data() + at, &v, sizeof(T)); }
template <class T> T get(const uint8_t* d, size_t at) { T v; std::memcpy(&v, d + at, sizeof(T)); return v; }

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamoff n = f.tellg();
    if (n < 0) return false;
    out.resize(size_t(n));
    f.seekg(0);
    if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
    return bool(f);
}

std::atomic<std::shared_ptr<const ActiveSet>> g_active{};
std::atomic<int> g_active_count{0};

// Sample k (0..512) along edge e of a tile.
inline int16_t edge_at(const Tile& t, int e, int k) {
    switch (e) { case 0: return t.at(0, k); case 1: return t.at(kSteps, k); case 2: return t.at(k, 0); default: return t.at(k, kSteps); }
}
inline void edge_set(Tile& t, int e, int k, int16_t v) {
    switch (e) { case 0: t.at(0, k) = v; break; case 1: t.at(kSteps, k) = v; break; case 2: t.at(k, 0) = v; break; default: t.at(k, kSteps) = v; break; }
}
inline glm::dvec3 edge_point(sector::Id id, int e, double f) {
    switch (e) { case 0: return sector::point(id, 0.0, f); case 1: return sector::point(id, 1.0, f); case 2: return sector::point(id, f, 0.0); default: return sector::point(id, f, 1.0); }
}
}  // namespace

// ---- Reading ----

float edge_mask(float a, float b, uint32_t fade) {
    if ((fade & 0xFFu) == 0u) return 1.0f;
    const float w = kFade;
    float m = 1.0f;
    if (fade & kW) m *= sm(a / w);
    if (fade & kE) m *= sm((1.0f - a) / w);
    if (fade & kN) m *= sm(b / w);
    if (fade & kS) m *= sm((1.0f - b) / w);
    if (fade & kNW) m *= sm(std::sqrt(a * a + b * b) / w);
    if (fade & kNE) m *= sm(std::sqrt((1.0f - a) * (1.0f - a) + b * b) / w);
    if (fade & kSW) m *= sm(std::sqrt(a * a + (1.0f - b) * (1.0f - b)) / w);
    if (fade & kSE) m *= sm(std::sqrt((1.0f - a) * (1.0f - a) + (1.0f - b) * (1.0f - b)) / w);
    return m;
}

float sample(const Tile& t, uint32_t fade, float a, float b) {
    const float x = std::clamp(a, 0.0f, 1.0f) * float(kSteps), y = std::clamp(b, 0.0f, 1.0f) * float(kSteps);
    const int ix = std::min(int(x), kSteps - 1), iy = std::min(int(y), kSteps - 1);
    const float fx = x - float(ix), fy = y - float(iy);
    const int16_t* r0 = t.h.data() + size_t(iy) * kTileN + size_t(ix);
    const int16_t* r1 = r0 + kTileN;
    const float h0 = float(r0[0]) + (float(r0[1]) - float(r0[0])) * fx;
    const float h1 = float(r1[0]) + (float(r1[1]) - float(r1[0])) * fx;
    return (h0 + (h1 - h0) * fy) * t.unit_m * edge_mask(a, b, fade);
}

int16_t to_sample(float metres, float unit_m) {
    return int16_t(std::clamp(std::lround(double(metres) / double(unit_m)), -32767l, 32767l));
}

// ---- Files ----

std::vector<uint8_t> encode(const Tile& t) {
    const size_t n = size_t(kTileN) * kTileN;
    std::vector<uint8_t> o(32 + n * 2, 0);
    std::memcpy(o.data(), "VHT1", 4);
    put<uint16_t>(o, 4, 1);
    put<uint16_t>(o, 6, uint16_t(kTileN));
    put<uint64_t>(o, 8, t.id.v);
    put<float>(o, 16, t.unit_m);
    o[20] = t.joined;
    o[21] = 0;
    put<uint32_t>(o, 24, uint32_t(n * 2));
    if (t.h.size() == n) std::memcpy(o.data() + 32, t.h.data(), n * 2);
    put<uint32_t>(o, 28, fnv1a(o.data() + 32, n * 2));
    return o;
}

bool decode(const uint8_t* d, size_t size, Tile& out) {
    const size_t n = size_t(kTileN) * kTileN;
    if (size < 32 || std::memcmp(d, "VHT1", 4) != 0) return false;
    if (get<uint16_t>(d, 4) != 1 || get<uint16_t>(d, 6) != kTileN) return false;
    if (d[21] != 0) return false;                           // (a compression this build does not know)
    if (get<uint32_t>(d, 24) != n * 2 || size != 32 + n * 2) return false;
    if (get<uint32_t>(d, 28) != fnv1a(d + 32, n * 2)) return false;
    Tile t;
    t.id.v = get<uint64_t>(d, 8);
    t.unit_m = get<float>(d, 16);
    t.joined = d[20];
    if (!t.id.valid() || t.id.level() != kTileLevel || !(t.unit_m > 0.0f) || t.unit_m > 4.0f) return false;
    t.h.resize(n);
    std::memcpy(t.h.data(), d + 32, n * 2);
    for (int16_t& v : t.h) if (v == INT16_MIN) v = -32767;  // (a signed-normalised texture has no -32768)
    out = std::move(t);
    return true;
}

std::string Store::tile_file(sector::Id tile)    { return "h_" + sector::name(tile) + ".vht"; }
std::string Store::index_file(sector::Id sector) { return "idx_" + sector::name(sector) + ".txt"; }

bool Store::parse_index(const std::string& text, std::vector<sector::Id>& tiles) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream ls(line);
        std::string kind, what;
        if (!(ls >> kind >> what) || kind != "tile") continue;
        sector::Id id;
        if (!sector::parse(what, id) || id.level() != kTileLevel) continue;
        if (std::find(tiles.begin(), tiles.end(), id) == tiles.end()) tiles.push_back(id);
    }
    return true;
}

bool Store::load_index(sector::Id sector, std::vector<sector::Id>& tiles) const {
    bool any = false;
    for (const std::string* dir : { &local_dir, &cache_dir }) {
        std::ifstream f(*dir + "/" + index_file(sector));
        if (!f) continue;
        std::stringstream ss;
        ss << f.rdbuf();
        parse_index(ss.str(), tiles);
        any = true;
    }
    return any;
}

bool Store::load_tile(sector::Id tile, Tile& out) const {
    std::vector<uint8_t> bytes;
    // (the cache first: a tile edited or fetched again stands in for the shipped one)
    for (const std::string* dir : { &cache_dir, &local_dir })
        if (read_file(*dir + "/" + tile_file(tile), bytes) && decode(bytes.data(), bytes.size(), out) && out.id == tile) return true;
    return false;
}

bool Store::has_tile(sector::Id tile) const {
    std::error_code ec;
    return std::filesystem::exists(cache_dir + "/" + tile_file(tile), ec) || std::filesystem::exists(local_dir + "/" + tile_file(tile), ec);
}

bool Store::save_tile(const Tile& t, bool local) const {
    if (!t.id.valid() || t.h.size() != size_t(kTileN) * kTileN) return false;
    const std::string& dir = local ? local_dir : cache_dir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    {
        const std::vector<uint8_t> bytes = encode(t);
        // (written beside it and renamed: a reader never sees half a file)
        const std::string path = dir + "/" + tile_file(t.id), tmp = path + ".tmp";
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        f.close();
        if (!f) return false;
        std::filesystem::rename(tmp, path, ec);
        if (ec) { std::filesystem::remove(path, ec); std::filesystem::rename(tmp, path, ec); }
        if (ec) return false;
    }
    const std::string ipath = dir + "/" + index_file(sector::ancestor(t.id, kIndexLevel));
    std::vector<sector::Id> listed;
    {
        std::ifstream f(ipath);
        if (f) { std::stringstream ss; ss << f.rdbuf(); parse_index(ss.str(), listed); }
    }
    if (std::find(listed.begin(), listed.end(), t.id) == listed.end()) {
        std::ofstream f(ipath, std::ios::app);
        if (!f) return false;
        f << "tile " << sector::name(t.id) << "\n";
    }
    return true;
}

// ---- The loaded tiles ----

int ActiveSet::find(sector::Id tile) const {
    for (int s = 0; s < kSlots; ++s) if (slot[s].tile && slot[s].tile->id == tile) return s;
    return -1;
}

int edge_toward(sector::Id from, sector::Id to) {
    for (int e = 0; e < 4; ++e) if (sector::neighbour(from, kEdgeDir[e][0], kEdgeDir[e][1]) == to) return e;
    return -1;
}

void ActiveSet::update_fades() {
    // An edge continues when the tile across it is loaded and both declare the edge joined.
    const auto edge_ok = [&](sector::Id a, sector::Id b) {
        const int sa = find(a), sb = find(b);
        if (sa < 0 || sb < 0) return false;
        const int ea = edge_toward(a, b), eb = edge_toward(b, a);
        return ea >= 0 && eb >= 0 && (slot[sa].tile->joined & (1u << ea)) && (slot[sb].tile->joined & (1u << eb));
    };
    count = 0;
    for (int s = 0; s < kSlots; ++s) {
        if (!slot[s].tile) continue;
        ++count;
        const Tile& t = *slot[s].tile;
        uint32_t fade = 0;
        sector::Id nb[4];
        for (int e = 0; e < 4; ++e) {
            nb[e] = sector::neighbour(t.id, kEdgeDir[e][0], kEdgeDir[e][1]);
            if (!nb[e].valid() || !edge_ok(t.id, nb[e])) fade |= 1u << e;
        }
        // A corner where both edges continue: it continues too only when all four tiles round
        // the point are loaded, joined edge to edge all the way round, and say so. Otherwise
        // the tile fades toward the point (a corner whose edge already fades needs nothing more).
        for (int c = 0; c < 4; ++c) {
            const int e0 = kCornerEdge[c][0], e1 = kCornerEdge[c][1];
            if (fade & ((1u << e0) | (1u << e1))) continue;
            const sector::Id dg = sector::neighbour(t.id, kCornerDir[c][0], kCornerDir[c][1]);
            const bool ok = (t.joined & (16u << c)) && dg.valid() && edge_ok(nb[e0], dg) && edge_ok(nb[e1], dg);
            if (!ok) fade |= 16u << c;
        }
        slot[s].fade = fade;
    }
}

void ActiveSet::place_windows(const globe::Chart& chart, uint32_t serial, const glm::dvec2& cam) {
    chart_serial = serial;
    const glm::dvec3 G = globe::to_planet(chart, cam.x, cam.y);
    // (whole metres: the anchor is then the same number in the shader's floats)
    const glm::dvec2 anchor = glm::round(cam);
    const sector::Id c0 = sector::cell_at(G, kTileLevel);
    win[0].on = true;
    win[0].map = sector::chart_map(chart, anchor, c0.face(), kTileLevel, int64_t(c0.i()) - kWin / 2, int64_t(c0.j()) - kWin / 2);
    win[1].on = false;
    for (int s = 0; s < kSlots; ++s) {
        if (!slot[s].tile || slot[s].tile->id.face() == c0.face()) continue;
        // Another face within reach: its window round where the camera would be in that
        // face's coordinates (carried on past its edge).
        const int f = slot[s].tile->id.face();
        const glm::dvec2 ab = sector::face_ab(f, G) * double(uint64_t(1) << kTileLevel);
        win[1].on = true;
        win[1].map = sector::chart_map(chart, anchor, f, kTileLevel, int64_t(std::floor(ab.x)) - kWin / 2, int64_t(std::floor(ab.y)) - kWin / 2);
        break;
    }
    update_windows(chart);
}

bool ActiveSet::windows_fit(uint32_t serial, const glm::dvec2& cam) const {
    if (serial != chart_serial || !win[0].on) return false;
    if (glm::distance(cam, win[0].map.anchor) > 1500.0) return false;
    const glm::dvec2 cc = win[0].map.eval(cam);
    if (cc.x < 5.0 || cc.y < 5.0 || cc.x >= 11.0 || cc.y >= 11.0) return false;
    for (int s = 0; s < kSlots; ++s) {
        if (!slot[s].tile) continue;
        bool in = false;
        for (const Window& w : win) {
            if (!w.on || w.map.face != slot[s].tile->id.face()) continue;
            const int64_t ri = int64_t(slot[s].tile->id.i()) - w.map.base_i, rj = int64_t(slot[s].tile->id.j()) - w.map.base_j;
            in = in || (ri >= 0 && rj >= 0 && ri < kWin && rj < kWin);
        }
        if (!in) return false;
    }
    return true;
}

void ActiveSet::update_windows(const globe::Chart& chart) {
    glm::dvec2 sum(0.0);
    int n = 0;
    for (Window& w : win) {
        std::memset(w.slot, 0, sizeof(w.slot));
        if (!w.on) continue;
        const double cells = double(uint64_t(1) << w.map.level);
        w.lo = glm::dvec2(std::max(0.0, -double(w.map.base_i)), std::max(0.0, -double(w.map.base_j)));
        w.hi = glm::dvec2(std::min(double(kWin), cells - double(w.map.base_i)), std::min(double(kWin), cells - double(w.map.base_j)));
        for (int s = 0; s < kSlots; ++s) {
            if (!slot[s].tile || slot[s].tile->id.face() != w.map.face) continue;
            const int64_t ri = int64_t(slot[s].tile->id.i()) - w.map.base_i, rj = int64_t(slot[s].tile->id.j()) - w.map.base_j;
            if (ri < 0 || rj < 0 || ri >= kWin || rj >= kWin) continue;
            w.slot[rj * kWin + ri] = uint8_t(s + 1);
        }
    }
    std::vector<std::pair<glm::dvec2, double>> discs;
    for (int s = 0; s < kSlots; ++s) {
        if (!slot[s].tile) continue;
        const glm::dvec2 c = globe::to_chart(chart, sector::centre(slot[s].tile->id));
        discs.emplace_back(c, sector::radius_m(slot[s].tile->id) + 4.0);
        sum += c; ++n;
    }
    bound_c = n ? sum / double(n) : glm::dvec2(0.0);
    bound_r = 0.0;
    for (const auto& d : discs) bound_r = std::max(bound_r, glm::length(d.first - bound_c) + d.second);
}

float ActiveSet::delta_series(double x, double z) const {
    const glm::dvec2 p(x, z);
    const glm::dvec2 bd = p - bound_c;
    if (glm::dot(bd, bd) > bound_r * bound_r) return 0.0f;
    for (const Window& w : win) {
        if (!w.on) continue;
        const glm::dvec2 cc = w.map.eval(p);
        if (cc.x < w.lo.x || cc.y < w.lo.y || cc.x >= w.hi.x || cc.y >= w.hi.y) continue;
        const int ci = int(cc.x), cj = int(cc.y);
        const int s = w.slot[cj * kWin + ci];
        if (s == 0) return 0.0f;
        return sample(*slot[s - 1].tile, slot[s - 1].fade, float(cc.x - double(ci)), float(cc.y - double(cj)));
    }
    return 0.0f;
}

float ActiveSet::delta_exact(const globe::Chart& chart, double x, double z) const {
    glm::dvec2 fr;
    const sector::Id id = sector::from_chart(chart, x, z, kTileLevel, &fr);
    const int s = find(id);
    return s < 0 ? 0.0f : sample(*slot[s].tile, slot[s].fade, float(fr.x), float(fr.y));
}

std::shared_ptr<const ActiveSet> active() { return g_active.load(std::memory_order_acquire); }
void set_active(std::shared_ptr<const ActiveSet> s) {
    const int n = s ? s->count : 0;
    g_active.store(std::move(s), std::memory_order_release);
    g_active_count.store(n, std::memory_order_release);
}

float ground_delta(float x, float z) {
    if (g_active_count.load(std::memory_order_relaxed) == 0) return 0.0f;
    const std::shared_ptr<const ActiveSet> s = g_active.load(std::memory_order_acquire);
    if (!s || s->count == 0) return 0.0f;
    // (a caller between a change of chart and the set made for it: the exact way)
    if (s->chart_serial != globe::chart_serial()) return s->delta_exact(globe::current(), double(x), double(z));
    return s->delta_series(double(x), double(z));
}

void fill_gpu(const ActiveSet& s, GpuTable& o) {
    std::memset(&o, 0, sizeof(o));
    o.head[0] = float(s.count); o.head[1] = float(s.bound_c.x); o.head[2] = float(s.bound_c.y); o.head[3] = float(s.bound_r);
    for (int k = 0; k < 2; ++k) {
        const Window& w = s.win[k];
        GpuTable::Win& g = o.win[k];
        g.lim[0] = 1.0f; g.lim[1] = 1.0f; g.lim[2] = 0.0f; g.lim[3] = 0.0f;
        if (!w.on || s.count == 0) continue;
        g.c_anchor[0] = float(w.map.c.x); g.c_anchor[1] = float(w.map.c.y);
        g.c_anchor[2] = float(w.map.anchor.x); g.c_anchor[3] = float(w.map.anchor.y);
        g.A[0] = float(w.map.A[0][0]); g.A[1] = float(w.map.A[1][0]); g.A[2] = float(w.map.A[0][1]); g.A[3] = float(w.map.A[1][1]);
        g.Hi[0] = float(w.map.Hi.x); g.Hi[1] = float(w.map.Hi.y); g.Hi[2] = float(w.map.Hi.z);
        g.Hj[0] = float(w.map.Hj.x); g.Hj[1] = float(w.map.Hj.y); g.Hj[2] = float(w.map.Hj.z);
        g.lim[0] = float(w.lo.x); g.lim[1] = float(w.lo.y); g.lim[2] = float(w.hi.x); g.lim[3] = float(w.hi.y);
        for (int c = 0; c < kWin * kWin; ++c)
            o.map[k][c >> 4][(c >> 2) & 3] |= uint32_t(w.slot[c]) << ((c & 3) * 8);
    }
    for (int i = 0; i < kSlots; ++i) {
        if (!s.slot[i].tile) continue;
        o.slot[i][0] = s.slot[i].tile->unit_m;
        o.slot[i][1] = float(s.slot[i].fade);
    }
}

// ---- Making tiles ----

float shape_profile(Shape s, double dist, double radius, double height) {
    if (dist >= radius) return 0.0f;
    const auto sstep = [](double a, double b, double x) { const double t = std::clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); };
    if (s == Shape::Mesa) return float(height * (1.0 - sstep(0.55 * radius, radius, dist)));
    // A crater: a bowl `height` deep with a rim 35 % as high at 0.6 of the radius, falling away outside.
    const double r0 = 0.6 * radius;
    if (dist < r0) { const double t = dist / r0; return float(-height + 1.35 * height * t * t); }
    return float(0.35 * height * (1.0 - sstep(r0, radius, dist)));
}

bool join_tiles(Tile& a, Tile& b) {
    const int ea = edge_toward(a.id, b.id), eb = edge_toward(b.id, a.id);
    if (ea < 0 || eb < 0) return false;
    // (which way the neighbour's samples run along the shared edge: the same way on one face)
    const glm::dvec3 a0 = edge_point(a.id, ea, 0.0);
    const bool same = glm::distance(a0, edge_point(b.id, eb, 0.0)) < glm::distance(a0, edge_point(b.id, eb, 1.0));
    const float tol = 1.01f * std::max(a.unit_m, b.unit_m);
    for (int k = 0; k <= kSteps; ++k) {
        const float ha = float(edge_at(a, ea, k)) * a.unit_m, hb = float(edge_at(b, eb, same ? k : kSteps - k)) * b.unit_m;
        if (std::abs(ha - hb) > tol) return false;
    }
    bool changed = false;
    if (a.unit_m == b.unit_m)
        for (int k = 0; k <= kSteps; ++k) {
            const int kb = same ? k : kSteps - k;
            if (edge_at(b, eb, kb) != edge_at(a, ea, k)) { edge_set(b, eb, kb, edge_at(a, ea, k)); changed = true; }
        }
    const uint8_t ja = uint8_t(a.joined | (1u << ea)), jb = uint8_t(b.joined | (1u << eb));
    changed = changed || ja != a.joined || jb != b.joined;
    a.joined = ja; b.joined = jb;
    return changed;
}

void join_corners(Tile& t, const std::function<bool(sector::Id)>& has) {
    for (int c = 0; c < 4; ++c) {
        const uint32_t both = (1u << kCornerEdge[c][0]) | (1u << kCornerEdge[c][1]);
        const sector::Id dg = sector::neighbour(t.id, kCornerDir[c][0], kCornerDir[c][1]);
        if ((t.joined & both) == both && dg.valid() && has(dg)) t.joined |= uint8_t(16u << c);
        else t.joined &= uint8_t(~(16u << c));
    }
}

std::vector<sector::Id> stamp(const Store& store, const glm::dvec3& G, Shape shape, double radius, double height, bool local) {
    std::vector<sector::Id> cells, written;
    sector::cover(G, radius, kTileLevel, cells);
    std::vector<Tile> tiles;
    for (sector::Id id : cells) {
        Tile t;
        const bool had = store.load_tile(id, t);
        if (!had) t = Tile(id);
        bool touched = false;
        for (int ib = 0; ib < kTileN; ++ib)
            for (int ia = 0; ia < kTileN; ++ia) {
                // (the chord: the arc to well under a millimetre at these distances, and exact near zero)
                const double d = globe::kR * glm::distance(G, sector::point(id, double(ia) / kSteps, double(ib) / kSteps));
                if (d >= radius) continue;
                t.at(ia, ib) = to_sample(float(t.at(ia, ib)) * t.unit_m + shape_profile(shape, d, radius, height), t.unit_m);
                touched = true;
            }
        if (touched) tiles.push_back(std::move(t));
    }
    for (size_t i = 0; i < tiles.size(); ++i)
        for (size_t j = i + 1; j < tiles.size(); ++j) join_tiles(tiles[i], tiles[j]);
    const auto has = [&](sector::Id id) {
        for (const Tile& t : tiles) if (t.id == id) return true;
        return store.has_tile(id);
    };
    for (Tile& t : tiles) {
        join_corners(t, has);
        if (store.save_tile(t, local)) written.push_back(t.id);
    }
    return written;
}

}  // namespace vibe::region
