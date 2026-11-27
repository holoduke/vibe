// The mart's goods: the service station's shelves filled with products.
//
// THE KIT is made in Blender by tools/shop/make_goods.py (assets/shop/goods.kit + the atlas
// goods.png in texture slot kTexGoods): 130 invented products - boxes, cans, jars, bottles,
// cartons, bags, sweets, books, cleaners, motor oil, small packs, tubs, cases, shipping cartons
// - each at three levels of detail. WHERE they go is the station's own shelf file
// (assets/gasstation/gasstation.shelf, written by tools/gasstation/make_gasstation.py): per
// shelf its kind, its front edge, the way it faces, its depth and its clear height.
//
// THE FILL (once, deterministic: a hash per shelf): along each shelf's front edge a product is
// chosen from those its kind holds and that fit, set out in a run of 2 - 6 facings, each facing
// as many rows deep as the shelf takes (at most 3; 6 for thin things); small things stacked two
// high where they fit; now and then a facing's front product is missing or one stands turned.
// Fridges hold drinks, the end caps and pallets cases and big boxes, the wall behind the
// counter small packs and sweets, the clips on the stand bags of crisps, the back store stacks
// of shipping cartons; the chest ice-cream tubs.
//
// DRAWN as a few MERGED parts (the shared product meshes copied into one mesh per 3 m cell of
// the mart): a level of detail for the whole mart by the eye's distance from it - 0 within
// 30 m (back rows at the coarser levels), 1 within 120 m, 2 within 400 m (front rows only),
// none beyond; the next level's parts are made one a frame, then swapped in, and the old ones
// given back. Home's chart only. VIBE_NO_SHOP_GOODS=1: none; VIBE_LOG_SHOP=1: counts.
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/shop_goods.h"
#include "engine/globe.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace vibe {

namespace {
constexpr float kIn[3]  = { 30.0f, 120.0f, 400.0f };
constexpr float kOut[3] = { 36.0f, 135.0f, 440.0f };
constexpr float kBin = 3.0f;                          // metres a merged part covers

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E5ull) {}
    uint64_t next() { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
    float f() { return float(next() >> 40) / float(1ull << 24); }
    float r(float a, float b) { return a + (b - a) * f(); }
    int i(int n) { return n > 0 ? int(next() % uint64_t(n)) : 0; }
};

template <class SkuT>
bool load_kit(const char* path, std::vector<SkuT>& out) {
    FILE* fp = std::fopen(path, "rb");
    if (!fp) return false;
    char tag[4]; uint32_t n = 0, atlas = 0;
    bool ok = std::fread(tag, 1, 4, fp) == 4 && std::memcmp(tag, "GDK1", 4) == 0 &&
              std::fread(&n, 4, 1, fp) == 1 && std::fread(&atlas, 4, 1, fp) == 1 && n < 10000;
    for (uint32_t k = 0; ok && k < n; ++k) {
        SkuT s;
        char name[24], cat[16]; float sz[3]; uint32_t lv = 0;
        ok = std::fread(name, 1, 24, fp) == 24 && std::fread(cat, 1, 16, fp) == 16 && std::fread(sz, 4, 3, fp) == 3 && std::fread(&lv, 4, 1, fp) == 1 && lv == 3;
        if (!ok) break;
        s.name.assign(name, strnlen(name, 24)); s.cat.assign(cat, strnlen(cat, 16));
        s.size = glm::vec3(sz[0], sz[1], sz[2]);
        for (int l = 0; ok && l < 3; ++l) {
            uint32_t nv = 0, ni = 0;
            ok = std::fread(&nv, 4, 1, fp) == 1 && std::fread(&ni, 4, 1, fp) == 1 && nv < 1000000 && ni < 3000000;
            if (!ok) break;
            s.lod[l].verts.resize(nv); s.lod[l].idx.resize(ni);
            static_assert(sizeof(game::LevelVertex) == 32, "pos, normal, uv");
            ok = std::fread(s.lod[l].verts.data(), 32, nv, fp) == nv && std::fread(s.lod[l].idx.data(), 4, ni, fp) == ni;
        }
        out.push_back(std::move(s));
    }
    std::fclose(fp);
    return ok && !out.empty();
}

struct Shelf { std::string kind; glm::vec3 a, b; glm::vec2 o; float depth, height; };

std::vector<Shelf> load_shelves(const char* path) {
    std::vector<Shelf> out;
    FILE* fp = std::fopen(path, "r");
    if (!fp) return out;
    char line[256];
    while (std::fgets(line, sizeof(line), fp)) {
        Shelf s; char kind[32];
        if (std::sscanf(line, "shelf %31s %f %f %f %f %f %f %f %f %f %f", kind, &s.a.x, &s.a.y, &s.a.z, &s.b.x, &s.b.y, &s.b.z,
                        &s.o.x, &s.o.y, &s.depth, &s.height) == 11) { s.kind = kind; out.push_back(s); }
    }
    std::fclose(fp);
    return out;
}

// What each kind of shelf holds: categories with weights.
struct Want { const char* cat; int w; };
std::vector<Want> wants(const std::string& k) {
    if (k == "snacks")  return { { "chips", 5 }, { "snack", 3 }, { "cereal", 2 }, { "candy", 1 }, { "small", 1 } };
    if (k == "grocery") return { { "cereal", 3 }, { "tin", 3 }, { "jar", 3 }, { "oil", 2 }, { "small", 3 }, { "snack", 1 }, { "cleaner", 1 }, { "spray", 1 }, { "wine", 1 } };
    if (k == "auto")    return { { "motoroil", 5 }, { "cleaner", 2 }, { "spray", 2 }, { "small", 2 }, { "book", 1 }, { "mag", 1 } };
    if (k == "fridge")  return { { "can", 4 }, { "water", 3 }, { "soda", 3 }, { "carton", 2 } };
    if (k == "endcap")  return { { "case", 3 }, { "cereal", 2 }, { "chips", 2 }, { "snack", 1 } };
    if (k == "wall")    return { { "pack", 5 }, { "candy", 2 } };
    if (k == "rack")    return { { "candy", 1 } };
    if (k == "chest")   return { { "tub", 1 } };
    if (k == "pallet")  return { { "case", 1 } };
    if (k == "clip")    return { { "chips", 1 } };
    if (k == "stack")   return { { "ship", 1 } };
    return { { "small", 1 } };
}

bool stackable(const std::string& c) { return c == "small" || c == "candy" || c == "pack" || c == "tin" || c == "snack" || c == "case"; }
}  // namespace

void VulkanEngine::shop_poll() {
    static const bool off = std::getenv("VIBE_NO_SHOP_GOODS") != nullptr;
    static const bool verbose = std::getenv("VIBE_LOG_SHOP") != nullptr;
    if (off) return;
    if (shop_) {
        for (auto it = shop_->retired.begin(); it != shop_->retired.end();) {
            if (--it->first > 0) { ++it; continue; }
            destroy_mesh_parts(it->second);
            it = shop_->retired.erase(it);
        }
    }
    if (station_pos_ == glm::vec3(0.0f) || !scene_desc_set_) return;
    if (!shop_) shop_ = new ShopState();
    ShopState& S = *shop_;
    if (!S.tried) {
        S.tried = true;
        if (!load_kit("assets/shop/goods.kit", S.kit)) { log::warnf("[shop] assets/shop/goods.kit missing or not understood - no goods on the shelves (tools/shop/make_goods.py makes it)"); return; }
        const std::vector<Shelf> shelves = load_shelves("assets/gasstation/gasstation.shelf");
        if (shelves.empty()) { log::warnf("[shop] assets/gasstation/gasstation.shelf missing - no goods on the shelves (tools/gasstation/make_gasstation.py writes it)"); return; }
        std::map<std::string, std::vector<int>> by_cat;
        for (size_t k = 0; k < S.kit.size(); ++k) by_cat[S.kit[k].cat].push_back(int(k));
        const glm::vec3 up(0.0f, 1.0f, 0.0f);
        auto place = [&](int sku, glm::vec3 p, glm::vec2 o, float yaw, float lean, float scale, int row, bool hidden) {
            const glm::vec3 z = glm::normalize(glm::vec3(o.x, 0.0f, o.y)), x = glm::cross(up, z);
            glm::mat4 M(1.0f);
            M[0] = glm::vec4(x, 0.0f); M[1] = glm::vec4(up, 0.0f); M[2] = glm::vec4(z, 0.0f); M[3] = glm::vec4(p, 1.0f);
            if (yaw != 0.0f) M = M * glm::rotate(glm::mat4(1.0f), yaw, up);
            if (lean != 0.0f) M = M * glm::rotate(glm::mat4(1.0f), -lean, glm::vec3(1.0f, 0.0f, 0.0f));
            if (scale != 1.0f) M = M * glm::scale(glm::mat4(1.0f), glm::vec3(scale));
            ShopState::Item it; it.sku = sku; it.m = M; it.row = uint8_t(std::min(row, 255)); it.hidden = hidden;
            S.items.push_back(it);
        };
        for (size_t si = 0; si < shelves.size(); ++si) {
            const Shelf& sh = shelves[si];
            Rng R(si * 7919u + 17u);
            const std::vector<Want> W = wants(sh.kind);
            int wsum = 0;
            for (const Want& w : W) wsum += w.w;
            glm::vec3 dir = sh.b - sh.a;
            const float len = glm::length(dir);
            if (len < 0.02f) continue;
            dir /= len;
            const glm::vec3 o3(sh.o.x, 0.0f, sh.o.y);
            // a product of this shelf that fits in (w, h, d); -1: none
            auto pick = [&](float wmax, float hmax, float dmax) -> int {
                for (int tries = 0; tries < 12; ++tries) {
                    int roll = R.i(wsum);
                    const char* cat = W[0].cat;
                    for (const Want& w : W) { if (roll < w.w) { cat = w.cat; break; } roll -= w.w; }
                    auto f = by_cat.find(cat);
                    if (f == by_cat.end() || f->second.empty()) continue;
                    const int k = f->second[size_t(R.i(int(f->second.size())))];
                    const glm::vec3 s = S.kit[size_t(k)].size;
                    if (s.x <= wmax && s.y <= hmax && s.z <= dmax * 1.15f + 0.02f) return k;
                }
                return -1;
            };
            if (sh.kind == "clip") {
                // one bag on its hook, shrunk to the hook's size
                const int k = pick(1.0f, 1.0f, 1.0f);
                if (k < 0) continue;
                const glm::vec3 s = S.kit[size_t(k)].size;
                const float sc = std::min(1.0f, std::min(len / s.x, sh.height / s.y)) * 0.97f;
                place(k, 0.5f * (sh.a + sh.b) - o3 * (0.5f * s.z * sc), sh.o, R.r(-0.05f, 0.05f), 0.0f, sc, 0, false);
                continue;
            }
            if (sh.kind == "pallet" || sh.kind == "stack" || sh.kind == "chest") {
                // bulk: columns along the edge and back from it, layers up to the clear height
                float t = 0.01f;
                const int same = sh.kind == "stack" ? -1 : pick(len, sh.height, sh.depth);
                while (t < len) {
                    const int k = same >= 0 ? same : pick(len - t, sh.height, sh.depth);
                    if (k < 0) break;
                    const glm::vec3 s = S.kit[size_t(k)].size;
                    if (t + s.x > len + 0.01f) break;
                    for (float dd = 0.01f; dd + s.z <= sh.depth + 0.01f; dd += s.z + 0.004f) {
                        const int layers = std::max(1, int(sh.height / s.y));
                        const int top = sh.kind == "stack" ? 1 + R.i(layers) : layers;
                        float yl = 0.0f;
                        for (int l = 0; l < top; ++l) {
                            if (l == top - 1 && sh.kind != "chest" && R.f() < 0.25f) break;          // (the top layer not full)
                            const int kk = sh.kind == "stack" ? (l == 0 ? k : pick(s.x, sh.height - yl, s.z)) : k;
                            if (kk < 0) break;
                            const glm::vec3 s2 = S.kit[size_t(kk)].size;
                            const float yaw = sh.kind == "stack" ? (R.f() < 0.15f ? R.r(-0.35f, 0.35f) : R.r(-0.04f, 0.04f)) : R.r(-0.015f, 0.015f);
                            const glm::vec3 p = sh.a + dir * (t + 0.5f * s.x) - o3 * (dd + 0.5f * s.z) + glm::vec3(0.0f, yl, 0.0f);
                            if (yl + s2.y > sh.height + 0.01f) break;
                            place(kk, p, sh.o, yaw, 0.0f, 1.0f, 0, false);
                            yl += s2.y;
                        }
                    }
                    t += s.x + (sh.kind == "stack" ? R.r(0.02f, 0.12f) : 0.004f);
                }
                continue;
            }
            // shelves: runs of facings
            float t = 0.005f;
            int misses = 0;
            while (t < len - 0.03f && misses < 6) {
                const int k = pick(len - t, sh.height - 0.01f, sh.depth);
                if (k < 0) { ++misses; t += 0.02f; continue; }
                const ShopState::Sku& sk = S.kit[size_t(k)];
                const glm::vec3 s = sk.size;
                const bool thin = s.z < 0.03f;
                const int rows = std::max(1, std::min(int((sh.depth + 0.01f) / (s.z + 0.003f)), thin ? 6 : 3));
                const bool two_high = stackable(sk.cat) && 2.0f * s.y + 0.01f < sh.height;
                int facings = 2 + R.i(5);
                if (sh.kind == "wall" || sh.kind == "rack") facings = 3 + R.i(6);
                for (int f = 0; f < facings && t + s.x <= len; ++f) {
                    const bool gap = R.f() < 0.05f;
                    const bool turned = R.f() < 0.04f;
                    for (int r = 0; r < rows; ++r) {
                        if (r == 0 && gap) continue;
                        float yaw = (r == 0 && turned) ? R.r(-0.45f, 0.45f) : R.r(-0.02f, 0.02f);
                        const float lean = (sk.cat == "mag" || sk.cat == "book") && r == 0 ? 0.0f : 0.0f;
                        const glm::vec3 p = sh.a + dir * (t + 0.5f * s.x) - o3 * (0.006f + float(r) * (s.z + 0.003f) + 0.5f * s.z);
                        const bool hidden = r > 0 && !(r == 1 && gap);
                        place(k, p, sh.o, yaw, lean, 1.0f, r, hidden);
                        if (two_high && R.f() < 0.8f) place(k, p + glm::vec3(0.0f, s.y, 0.0f), sh.o, R.r(-0.03f, 0.03f), 0.0f, 1.0f, r, hidden);
                    }
                    t += s.x + 0.004f;
                }
                t += R.r(0.006f, 0.02f);
            }
        }
        // merged parts: one per 3 m cell of the mart
        std::map<std::pair<int, int>, int> bin_of;
        for (ShopState::Item& it : S.items) {
            const glm::vec3 p = glm::vec3(it.m[3]);
            const std::pair<int, int> key(int(std::floor(p.x / kBin)), int(std::floor(p.z / kBin)));
            auto f = bin_of.find(key);
            if (f == bin_of.end()) f = bin_of.emplace(key, int(bin_of.size())).first;
            it.bin = f->second;
            const glm::vec3 w = station_pos_ + p;
            S.lo = glm::min(S.lo, w - glm::vec3(0.6f)); S.hi = glm::max(S.hi, w + glm::vec3(0.6f));
        }
        S.bins = int(bin_of.size());
        S.ok = !S.items.empty();
        std::map<std::string, int> per_kind;
        for (const Shelf& sh : shelves) per_kind[sh.kind] += 1;
        size_t tri[3] = { 0, 0, 0 }, hid = 0;
        for (const ShopState::Item& it : S.items) {
            hid += it.hidden ? 1 : 0;
            for (int l = 0; l < 3; ++l) {
                if (l == 2 && it.hidden) continue;
                const int lod = it.hidden ? (it.row >= 2 ? 2 : std::max(l, 1)) : l;
                tri[l] += S.kit[size_t(it.sku)].lod[lod].idx.size() / 3;
            }
        }
        log::infof("[shop] goods: %zu products of %zu kinds on %zu shelves (%zu behind the front row), %d parts; triangles per level %zu / %zu / %zu",
                   S.items.size(), S.kit.size(), shelves.size(), hid, S.bins, tri[0], tri[1], tri[2]);
        if (verbose) for (const auto& kv : per_kind) log::infof("[shop]   %s: %d shelves", kv.first.c_str(), kv.second);
    }
    if (!S.ok) return;
    const bool home = globe::current().home;
    const glm::vec3 eye = player_.eye_position();
    const float d = glm::distance(eye, glm::clamp(eye, S.lo, S.hi));
    auto lv = [&](const float* lim) { return !home ? -1 : d < lim[0] ? 0 : d < lim[1] ? 1 : d < lim[2] ? 2 : -1; };
    const int a = lv(kIn), b = lv(kOut);
    const int want = (S.level == a || S.level == b) ? S.level : a;
    if (want == S.level) {
        if (S.building >= 0) { destroy_mesh_parts(S.next); S.building = -1; }
        return;
    }
    if (want < 0) {
        if (!S.parts.empty()) { S.retired.emplace_back(12, std::move(S.parts)); S.parts.clear(); ++hw_gen_; }
        if (S.building >= 0) { destroy_mesh_parts(S.next); S.building = -1; }
        S.level = -1; S.tris = 0;
        if (verbose) log::infof("[shop] goods given back (%.0f m away)", double(d));
        return;
    }
    if (S.building != want) {
        if (S.building >= 0) destroy_mesh_parts(S.next);
        S.building = want; S.next.clear(); S.next_bin = 0; S.next_tris = 0;
    }
    // one merged part a frame
    if (S.next_bin < size_t(S.bins)) {
        game::LevelMesh lm;
        lm.tex_albedo = lm.tex_normal = kTexGoods;
        lm.tint = glm::vec3(1.0f); lm.fallback = glm::vec3(0.55f, 0.45f, 0.35f);
        lm.relief = 0.0f; lm.uv_scale = 1.0f;
        lm.bmin = glm::vec3(1e9f); lm.bmax = glm::vec3(-1e9f);
        for (const ShopState::Item& it : S.items) {
            if (it.bin != int(S.next_bin)) continue;
            if (want == 2 && it.hidden) continue;
            const int lod = it.hidden ? (it.row >= 2 ? 2 : std::max(want, 1)) : want;
            const game::LevelMesh& src = S.kit[size_t(it.sku)].lod[lod];
            const glm::mat3 N = glm::mat3(it.m);
            const uint32_t base = uint32_t(lm.verts.size());
            for (const game::LevelVertex& v : src.verts) {
                game::LevelVertex o = v;
                o.p = station_pos_ + glm::vec3(it.m * glm::vec4(v.p, 1.0f));
                o.n = glm::normalize(N * v.n);
                lm.bmin = glm::min(lm.bmin, o.p); lm.bmax = glm::max(lm.bmax, o.p);
                lm.verts.push_back(o);
            }
            for (uint32_t i : src.idx) lm.idx.push_back(base + i);
        }
        ++S.next_bin;
        if (!lm.idx.empty()) {
            std::vector<CastleMeshGpu> one;
            one.push_back(make_mesh_part(lm));
            build_mesh_part_blas(one);
            S.next.push_back(one[0]);
            S.next_tris += uint32_t(lm.idx.size() / 3);
        }
        if (S.next_bin < size_t(S.bins)) return;
    }
    // all made: swap
    if (!S.parts.empty()) S.retired.emplace_back(12, std::move(S.parts));
    S.parts = std::move(S.next); S.next.clear();
    S.level = want; S.building = -1; S.tris = S.next_tris;
    ++hw_gen_;
    if (verbose) log::infof("[shop] goods at level %d (%.0f m away): %zu parts, %u triangles", want, double(d), S.parts.size(), S.tris);
}

void VulkanEngine::shop_parts(std::vector<const CastleMeshGpu*>& out) const {
    if (!shop_ || !globe::current().home) return;
    for (const CastleMeshGpu& g : shop_->parts) out.push_back(&g);
}

bool VulkanEngine::shop_owns(const CastleMeshGpu* g) const {
    return shop_ && !shop_->parts.empty() && g >= shop_->parts.data() && g < shop_->parts.data() + shop_->parts.size();
}

void VulkanEngine::shop_destroy() {
    if (!shop_) return;
    destroy_mesh_parts(shop_->parts);
    destroy_mesh_parts(shop_->next);
    for (auto& r : shop_->retired) destroy_mesh_parts(r.second);
    delete shop_;
    shop_ = nullptr;
}

}  // namespace vibe
