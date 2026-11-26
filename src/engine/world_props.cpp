#include "engine/world_props.h"
#include "engine/veg_paint.h"

#include "engine/road.h"
#include "engine/terrain.h"
#include "engine/water_course.h"
#include "engine/lsystem.h"
#include "engine/log.h"
#include "engine/eztree.h"

#include <glm/geometric.hpp>
#include <glm/common.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <map>
#include <random>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>

namespace vibe {
#include "engine/world_props_meshes.inl"   // mesh builders for trees, bushes and rocks (conifer, broadleaf, pine, shrub, slab rock)

// Boulders sculpted and baked in Blender (tools/rocks/make_rocks.py):
// four meshes per rock; vertex uv = position in the bake atlas. Stored the
// way the vegetation vertex format wants it: u = atlas u, v = 4 (the rock
// material) + 0.45 * atlas v.
static std::vector<std::array<game::LevelMesh, kVegLods>> load_baked_rocks() {
    std::vector<std::array<game::LevelMesh, kVegLods>> rocks;
    std::ifstream f;
    for (const char* p : { "assets/rocks/rocks.qrock", "../assets/rocks/rocks.qrock" }) {
        f.open(p, std::ios::binary);
        if (f.good()) break;
        f.close(); f.clear();
    }
    if (!f.good()) return rocks;
    char magic[4]{};
    uint32_t n = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    if (std::memcmp(magic, "QRK1", 4) != 0 || n == 0 || n > 64) return rocks;
    rocks.resize(n);
    for (uint32_t r = 0; r < n; ++r)
        for (int l = 0; l < kVegLods; ++l) {
            uint32_t nv = 0, ni = 0;
            f.read(reinterpret_cast<char*>(&nv), 4);
            f.read(reinterpret_cast<char*>(&ni), 4);
            if (!f.good() || nv > 400000 || ni > 2400000) { rocks.clear(); return rocks; }
            std::vector<float> v(size_t(nv) * 8);
            f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4));
            game::LevelMesh& m = rocks[r][size_t(l)];
            m.idx.resize(ni);
            f.read(reinterpret_cast<char*>(m.idx.data()), std::streamsize(size_t(ni) * 4));
            if (!f.good()) { rocks.clear(); return rocks; }
            m.bmin = glm::vec3(1e9f); m.bmax = glm::vec3(-1e9f);
            for (uint32_t i = 0; i < nv; ++i) {
                const float* q = &v[size_t(i) * 8];
                push_v(m, { glm::vec3(q[0], q[1], q[2]), glm::normalize(glm::vec3(q[3], q[4], q[5]) + glm::vec3(1e-9f)),
                            q[6], 4.0f + 0.45f * glm::clamp(q[7], 0.0f, 1.0f) });
            }
        }
    log::infof("[veg] %u baked rocks loaded (assets/rocks)", n);
    return rocks;
}

std::vector<VegArchetype> build_veg_archetypes(int seed) {
    const auto baked_rocks = load_baked_rocks();
    int baked_next = 0;
    std::vector<VegArchetype> out;
    std::mt19937 rng{ uint32_t(seed) };
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const Ico tiny = make_ico(0), lo = make_ico(1), hi = make_ico(2), ultra = make_ico(3);
    auto finish = [](VegArchetype& a) {
        float rmax = 0.0f, ymax = 0.0f;
        for (const auto& v : a.lods[0].verts) {
            rmax = std::max(rmax, glm::length(glm::vec2(v.p.x, v.p.z)));
            ymax = std::max(ymax, v.p.y);
        }
        a.height = ymax;
        a.radius = std::sqrt(rmax * rmax + ymax * ymax * 0.25f);
    };
    // Pines: 7 variants - young / mature / tall-narrow spruce / broad old
    // pine (height, tier count, base radius).
    for (int v = 0; v < 7; ++v) {
        VegArchetype a; a.species = 0;
        const float kind = float(v % 4);
        float H  = (kind == 0 ? 8.0f : kind == 1 ? 11.0f : kind == 2 ? 15.0f : 12.0f) + 3.0f * U(rng);
        int   tiers = 6 + int(U(rng) * 3.0f) + (kind == 2 ? 2 : 0);
        float r0 = (kind == 2 ? 1.9f : kind == 3 ? 3.1f : 2.4f) + 0.6f * U(rng);
        PineParams p{ H, tiers, r0, U(rng) * 6.28f, (U(rng) - 0.5f) * 0.6f };
        for (int l = 0; l < kVegLods; ++l) build_pine(a.lods[l], p, l, lo, hi);
        finish(a); out.push_back(std::move(a));
    }
    // Old pines: 3 giants (20 m+), for the old-growth stands.
    for (int v = 0; v < 5; ++v) {
        VegArchetype a; a.species = 0; a.old = true;
        PineParams p{ 20.5f + 2.0f * float(v) + 1.5f * U(rng), 9, 3.5f + 0.5f * U(rng), U(rng) * 6.28f, (U(rng) - 0.5f) * 0.9f, true };
        for (int l = 0; l < kVegLods; ++l) build_pine(a.lods[l], p, l, lo, hi);
        finish(a); out.push_back(std::move(a));
    }
    // Broadleaf: 7 variants - wide oak, tall narrow (birch / poplar-like),
    // round ash, young tree; crown shape + branch count + outer clumps.
    for (int v = 0; v < 7; ++v) {
        VegArchetype a; a.species = 1;
        const int kind = v % 4;
        const glm::vec2 crown = kind == 0 ? glm::vec2(1.25f, 0.95f)    // wide oak
                              : kind == 1 ? glm::vec2(0.70f, 1.45f)    // tall narrow
                              : kind == 2 ? glm::vec2(1.00f, 1.05f)    // round
                                          : glm::vec2(0.80f, 0.85f);   // young
        OakParams p{ (kind == 1 ? 9.5f : kind == 3 ? 5.5f : 7.0f) + 3.0f * U(rng),
                     4 + int(U(rng) * 3.5f), U(rng) * 6.28f, (1.8f + 0.9f * U(rng)) * crown.x,
                     glm::vec3((U(rng) - 0.5f) * 0.8f, 0.0f, (U(rng) - 0.5f) * 0.8f),
                     crown, 4 + int(U(rng) * 4.0f) };
        for (int l = 0; l < kVegLods; ++l) build_oak(a.lods[l], p, l, lo, hi, tiny, ultra);
        finish(a); out.push_back(std::move(a));
    }
    // Old oaks: 3 veterans - wide, heavy-limbed, for the old-growth stands.
    for (int v = 0; v < 5; ++v) {
        VegArchetype a; a.species = 1; a.old = true;
        const glm::vec2 crown(1.40f + 0.12f * float(v), 0.95f);
        OakParams p{ 11.0f + 1.0f * float(v) + 1.0f * U(rng), 6, U(rng) * 6.28f, 3.2f * crown.x,
                     glm::vec3((U(rng) - 0.5f) * 1.0f, 0.0f, (U(rng) - 0.5f) * 1.0f), crown, 7, true };
        for (int l = 0; l < kVegLods; ++l) build_oak(a.lods[l], p, l, lo, hi, tiny, ultra);
        finish(a); out.push_back(std::move(a));
    }
    // Bushes: 5 variants (low wide / tall / round clusters).
    for (int v = 0; v < 5; ++v) {
        VegArchetype a; a.species = 2;
        BushParams p{ 3 + int(U(rng) * 3.0f), U(rng) * 6.28f,
                      v % 3 == 0 ? 1.35f : v % 3 == 1 ? 0.85f : 1.0f,
                      v % 3 == 0 ? 0.75f : v % 3 == 1 ? 1.45f : 1.0f };
        for (int l = 0; l < kVegLods; ++l) build_bush(a.lods[l], p, l, lo, tiny, hi);
        finish(a); out.push_back(std::move(a));
    }
    // River-bank scrub: 6 variants - bush willows low and tall, and two young alders.
    for (int v = 0; v < 6; ++v) {
        VegArchetype a; a.species = 2; a.riverside = true;
        const float seed = U(rng) * 6.28f;
        const bool alder = v >= 4;
        const float H = alder ? 4.2f + 1.4f * U(rng) : (v % 2 == 0 ? 2.1f + 0.7f * U(rng) : 3.2f + 1.1f * U(rng));
        for (int l = 0; l < kVegLods; ++l) build_river_shrub(a.lods[l], seed, alder, H, l);
        finish(a); out.push_back(std::move(a));
    }
    // Rock LOD meshes: the surface of rock_sdf.glsl's shape function without
    // its finest detail (cracks / grain are added per pixel in cube.frag by
    // evaluating the same function at the mesh surface). Real geometry: the
    // same mesh is drawn, casts the ray-traced shadow and is collided with.
    // 20480 / 5120 / 1280 / 80 triangles.
    const Ico rock_l0 = make_ico(5), rock_l1 = make_ico(4);
    for (int v = 0; v < 8; ++v) {                      // rocks (species 3)
        VegArchetype a; a.species = 3;
        // Deterministic in the variant index: shaders/rock_sdf.glsl evaluates
        // the same function (rock_params) for the per-pixel detail.
        float sd = 3.7f + 6.1f * float(v);
        float strata = (v % 2 == 0) ? 0.55f : 0.0f;
        float amp = 0.26f + 0.1f * glm::fract(float(v) * 0.37f);
        if (v % 3 == 1) {                              // slab formation (rock_params: strata < 0)
            for (int l = 0; l < kVegLods; ++l) build_slab_rock(a.lods[l], sd);
            finish(a); out.push_back(std::move(a));
            continue;
        }
        // Boulders: the Blender-sculpted ones when they are there.
        if (baked_next < int(baked_rocks.size())) {
            for (int l = 0; l < kVegLods; ++l) a.lods[l] = baked_rocks[size_t(baked_next)][size_t(l)];
            a.baked = baked_next++;
            finish(a); out.push_back(std::move(a));
            continue;
        }
        // A boulder is what is left of a block broken along a handful of
        // planes: radius along a direction = the SOFT minimum of the
        // distances to 13 random planes (soft = worn edges) - flat faces
        // meeting in rounded arrises, like fractured rock, instead of a
        // noise-displaced ball. Star-shaped by construction, so pushing an
        // icosphere out to it always gives a valid closed mesh. On top: a
        // little low-frequency swelling, mid-scale lumps, and for the
        // layered variants (v % 3 == 2) recessed bedding joints.
        (void)strata; (void)amp;
        const int np = 13;
        glm::vec3 pn[13]; float ph[13];
        {
            std::mt19937 pr{ uint32_t(1337 + v * 977) };
            std::uniform_real_distribution<float> PU(0.0f, 1.0f);
            for (int k = 0; k < np; ++k) {
                glm::vec3 n(PU(pr) * 2.0f - 1.0f, PU(pr) * 2.0f - 1.0f, PU(pr) * 2.0f - 1.0f);
                if (k < 2) n.y = std::abs(n.y) + 0.9f;            // a top face or two
                if (k == 2) n = glm::vec3(n.x * 0.3f, -1.0f, n.z * 0.3f);   // roughly flat underside
                pn[k] = glm::normalize(n + glm::vec3(1e-4f));
                ph[k] = 0.62f + 0.36f * PU(pr);
            }
        }
        const bool layered = (v % 3) == 2;
        const glm::vec3 bedN = glm::normalize(glm::vec3(0.18f * std::sin(sd), 1.0f, 0.18f * std::cos(sd)));
        auto shape = [&](glm::vec3 d) {
            const float ks = 13.0f;
            float acc = std::exp(-ks * 1.30f);
            for (int k = 0; k < np; ++k) {
                const float c = glm::dot(d, pn[k]);
                if (c > 0.05f) acc += std::exp(-ks * (ph[k] / c));
            }
            float r = -std::log(acc) / ks;
            r *= 1.0f + 0.14f * (fbm(d * 1.4f + glm::vec3(sd), 3) - 0.5f);
            glm::vec3 q = d * r;
            q += d * ((fbm(q * 4.3f + glm::vec3(sd), 3) - 0.5f) * 0.07f);          // mid-scale lumps
            if (layered) {
                // Bedding: slightly tilted layers; the joints between them
                // are weathered back a few centimetres.
                const float bcoord = glm::dot(q, bedN) * 3.4f + 0.5f * fbm(q * 0.9f + glm::vec3(sd * 1.7f), 2);
                const float fr = bcoord - std::floor(bcoord);
                const float joint = 1.0f - glm::smoothstep(0.0f, 0.10f, fr) * (1.0f - glm::smoothstep(0.86f, 1.0f, fr));
                const glm::vec3 lateral = q - bedN * glm::dot(q, bedN);
                q -= lateral * (0.075f * joint);
            }
            return glm::vec3(0.0f, 0.35f, 0.0f) + q;
        };
        const Ico* ic[kVegLods] = { &rock_l0, &rock_l1, &ultra, &lo };
        for (int l = 0; l < kVegLods; ++l) {
            LevelMesh& m = a.lods[l];
            for (const glm::vec3& d : ic[l]->v) push_v(m, { shape(glm::normalize(d)), glm::normalize(d), 0.0f, 4.0f });
            for (uint32_t idx : ic[l]->f) m.idx.push_back(idx);
            std::vector<glm::vec3> nacc(m.verts.size(), glm::vec3(0.0f));
            for (size_t i = 0; i + 2 < m.idx.size(); i += 3) {
                const uint32_t i0 = m.idx[i], i1 = m.idx[i + 1], i2 = m.idx[i + 2];
                const glm::vec3 fn = glm::cross(m.verts[i1].p - m.verts[i0].p, m.verts[i2].p - m.verts[i0].p);
                nacc[i0] += fn; nacc[i1] += fn; nacc[i2] += fn;
            }
            m.bmin = glm::vec3(1e9f); m.bmax = glm::vec3(-1e9f);
            for (size_t i = 0; i < m.verts.size(); ++i) {
                m.verts[i].n = glm::normalize(nacc[i] + glm::vec3(1e-9f));
                m.bmin = glm::min(m.bmin, m.verts[i].p); m.bmax = glm::max(m.bmax, m.verts[i].p);
            }
        }
        finish(a); out.push_back(std::move(a));
    }
    if (std::getenv("VIBE_LOG_VEG")) {      // triangles of every tree / bush type, per detail bucket
        for (size_t i = 0; i < out.size(); ++i) {
            const VegArchetype& a = out[i];
            if (a.species > 2) continue;
            log::infof("[vegtris] %zu species %d cell %d%s H %.1f r %.1f: %zu / %zu / %zu / %zu", i, a.species, a.lods[0].card_cell,
                       a.old ? " old" : (a.riverside ? " river" : ""), a.height, a.radius,
                       a.lods[0].idx.size() / 3, a.lods[1].idx.size() / 3, a.lods[2].idx.size() / 3,
                       a.lods[3].idx.size() / 3);
        }
    }
    return out;
}

float river_shrub_density(float x, float z, float h, float ny, float wl) {
    const float e = world_river_distance(x, z);
    if (e > 30.0f) return 0.0f;
    const float up = h - wl;
    if (up < 0.10f || up > 3.2f || ny < 0.60f) return 0.0f;
    // On the bank: from the water's edge (about 5 m outside the channel) back to where the
    // ground has risen a couple of metres; thickest just above the water.
    float d = (1.0f - glm::smoothstep(14.0f, 30.0f, e)) * glm::smoothstep(0.10f, 0.45f, up) * (1.0f - glm::smoothstep(1.6f, 3.2f, up));
    // Thickets with gaps between them, a few bushes wide.
    auto hn = [](float px, float pz) {
        auto hv = [](int ix, int iz) {
            uint32_t n = uint32_t(ix) * 374761393u + uint32_t(iz) * 668265263u; n = (n ^ (n >> 13)) * 1274126177u;
            return float((n ^ (n >> 16)) & 0xFFFFu) / 65535.0f;
        };
        const float fx = std::floor(px), fz = std::floor(pz);
        float tx = px - fx, tz = pz - fz; tx = tx * tx * (3.0f - 2.0f * tx); tz = tz * tz * (3.0f - 2.0f * tz);
        const int ix = int(fx), iz = int(fz);
        return glm::mix(glm::mix(hv(ix, iz), hv(ix + 1, iz), tx), glm::mix(hv(ix, iz + 1), hv(ix + 1, iz + 1), tx), tz);
    };
    const float thick = hn(x / 17.0f, z / 17.0f) * 0.65f + hn(x / 6.0f + 9.0f, z / 6.0f) * 0.35f;
    d *= glm::smoothstep(0.27f, 0.55f, thick);          // (2026-11-25: the gaps a little narrower - about a quarter more scrub)
    // Down the river: the reeds have the banks by the castle's lake; the scrub takes over
    // toward the sea (the river runs away from the castle, so distance from it is how far
    // downstream this is).
    const float down = glm::smoothstep(150.0f, 620.0f, std::sqrt(x * x + z * z));
    return d * glm::mix(0.22f, 1.0f, down);
}

std::vector<VegInstance> place_vegetation(const Heightmap& hm, const Road* road,
                                          const WorldPropParams& P,
                                          const std::vector<VegArchetype>& arch,
                                          std::vector<collision::AABB>& cols,
                                          WorldPropStats& st) {
    std::vector<VegInstance> out;
    if (hm.heights.empty() || arch.empty()) return out;
    std::vector<int> by_species[4];
    std::vector<int> river_pool;                       // river-bank scrub: its own rule below
    std::vector<int> old_trees[2];                     // old-growth pines / broadleaf
    for (size_t i = 0; i < arch.size(); ++i) {
        if (arch[i].riverside) { river_pool.push_back(int(i)); continue; }
        if (arch[i].old && arch[i].species < 2) old_trees[arch[i].species].push_back(int(i));
        else by_species[arch[i].species].push_back(int(i));
    }
    std::mt19937 rng(uint32_t(P.seed) ^ 0x9E3779B9u);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    auto u = [&]() { return U(rng); };
    auto h_at = [&](float x, float z) { return hm.sample_world(x, z); };
    auto ny_at = [&](float x, float z) {
        float e = 1.5f;
        float dx = h_at(x + e, z) - h_at(x - e, z), dz = h_at(x, z + e) - h_at(x, z - e);
        return 2.0f * e / std::sqrt(dx * dx + dz * dz + 4.0f * e * e);
    };
    auto road_d = [&](float x, float z) {
        if (!road || road->empty()) return 1e9f;
        if (x < road->bb_min.x || x > road->bb_max.x || z < road->bb_min.y || z > road->bb_max.y) return 1e9f;
        return road->distance(x, z);
    };
    auto excluded = [&](float x, float z) {
        for (const auto& k : P.keep_out)
            if ((x - k.x) * (x - k.x) + (z - k.y) * (z - k.y) < k.z * k.z) return true;
        if (wc::stream_distance(glm::vec2(x, z), P.water_level) < 2.0f) return true;      // (not in the stream: engine/water_course.h)
        return false;
    };
    const float road_clear = road ? road->half_width + road->shoulder : 0.0f;
    // Forest density 0..1 (1 = closed forest). Large-scale noise with high
    // contrast (thick woods / open meadows / thin edges) plus hand-placed
    // zones: dense forest on the far (north-west) shore of the castle
    // lake, an open ring around the castle, and big treeless meadows.
    auto blob = [](float x0, float z0, float cx, float cz, float r) {
        float dx = (x0 - cx) / r, dz = (z0 - cz) / r;
        return std::exp(-(dx * dx + dz * dz) * 2.0f);
    };
    auto forest_exact = [&](float x, float z) {
        float n = fbm(glm::vec3(x * 0.0042f, 11.0f, z * 0.0042f), 3) * 0.8f +
                  fbm(glm::vec3(x * 0.017f, 4.0f, z * 0.017f), 2) * 0.35f;
        float d = std::clamp((n - 0.50f) * 6.0f, 0.0f, 1.0f);
        d = std::max(d, std::min(1.0f, blob(x, z, -70.0f, -200.0f, 190.0f) * 1.6f));   // far shore
        d = std::max(d, std::min(1.0f, blob(x, z, -230.0f, -60.0f, 120.0f) * 1.3f));
        float clear = 1.0f - std::clamp((std::sqrt(x * x + z * z) - 75.0f) / 40.0f, 0.0f, 1.0f);
        clear = std::max(clear, blob(x, z, 170.0f, 140.0f, 110.0f) * 1.4f);           // SE meadow
        clear = std::max(clear, blob(x, z, 40.0f, 330.0f, 120.0f) * 1.4f);
        return std::clamp(d * (1.0f - std::min(clear, 1.0f)), 0.0f, 1.0f);
    };
    // Forest density is smooth (>= ~60 m features + 100 m+ blobs), so a
    // 2 m grid baked on all cores and sampled bilinearly matches it while
    // taking the 5-octave noise out of the 3.5 M-attempt placement loops
    // (startup: ~0.9 s -> ~0.2 s).
    const float fcell = 2.0f;
    const float fext = P.veg_half_extent + 8.0f;
    const int fn = int(std::ceil(2.0f * fext / fcell)) + 2;
    std::vector<float> fgrid(size_t(fn) * size_t(fn));
    {
        const unsigned nt = (std::max)(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([&, t]() {
                for (int j = int(t); j < fn; j += int(nt))
                    for (int i = 0; i < fn; ++i)
                        fgrid[size_t(j) * size_t(fn) + size_t(i)] =
                            forest_exact(-fext + float(i) * fcell, -fext + float(j) * fcell);
            });
        for (auto& t : th) t.join();
    }
    auto forest = [&](float x, float z) {
        float fx = std::clamp((x + fext) / fcell, 0.0f, float(fn - 1) - 0.001f);
        float fz = std::clamp((z + fext) / fcell, 0.0f, float(fn - 1) - 0.001f);
        int ix = int(fx), iz = int(fz);
        float tx = fx - float(ix), tz = fz - float(iz);
        const float* r0 = &fgrid[size_t(iz) * size_t(fn) + size_t(ix)];
        const float* r1 = r0 + fn;
        return (r0[0] * (1.0f - tx) + r0[1] * tx) * (1.0f - tz) + (r1[0] * (1.0f - tx) + r1[1] * tx) * tz;
    };
    // Dry ground: the trunk AND a 3.5 m ring above water + margin, so
    // shore slopes don't put trees (or their crowns) in the lake.
    auto dry = [&](float x, float z, float margin) {
        if (h_at(x, z) < P.water_level + margin) return false;
        for (int k = 0; k < 6; ++k) {
            float a = float(k) * 1.0472f;
            if (h_at(x + std::cos(a) * 3.5f, z + std::sin(a) * 3.5f) < P.water_level + 0.6f) return false;
        }
        return true;
    };
    // 2.5 m occupancy grid keeps trunks apart.
    std::unordered_set<int64_t> occ;
    auto key2 = [](int x, int z) { return (int64_t(x) << 32) ^ int64_t(uint32_t(z)); };
    auto try_mark = [&](float x, float z, float r) {
        int r2 = int(std::ceil(r / 2.5f));
        int cx = int(std::floor(x / 2.5f)), cz = int(std::floor(z / 2.5f));
        for (int dz = -r2; dz <= r2; ++dz)
            for (int dx = -r2; dx <= r2; ++dx)
                if (occ.count(key2(cx + dx, cz + dz))) return false;
        occ.insert(key2(cx, cz));
        return true;
    };
    const float ext = P.veg_half_extent;
    // Rocks: formations (clusters of 3-7 boulders on slopes / rock band)
    // and lone erratics. Instances of the rock archetypes with a per-
    // instance vertical squash (rot_arch_phase.w).
    // `variant` < 0: any rock type. `sink`: fraction of the rock's height
    // buried (0.12 normal; pavements sit almost flush).
    auto add_rock_ex = [&](float x, float z, float s, float ysq, int variant, float sink) {
        const auto& pool = by_species[3];
        const size_t pi = variant >= 0 ? size_t(variant) % pool.size()
                                       : size_t(u() * float(pool.size())) % pool.size();
        int ai = pool[pi];
        float gy = h_at(x, z);
        const float rot = u() * 6.2831853f;
        const float y0 = gy - s * ysq * sink;
        VegInstance vi{};
        vi.pos_scale = glm::vec4(x, y0, z, s);
        vi.rot_arch_phase = glm::vec4(rot, float(ai), 0.0f, ysq);
        out.push_back(vi);
        if (pi % 3 == 1) {
            // Slab formation: one collider per bed, so the steps are steps.
            const float c = std::cos(rot), sn = std::sin(rot);
            for (const SlabBed& b : slab_beds(3.7f + 6.1f * float(pi))) {
                const float top = y0 + (0.35f + b.yc + b.th + 0.018f) * s * ysq;
                const float bot = std::min(gy, y0) - 1.0f;
                // The bed is an irregular rotated polygon; world colliders are
                // axis-aligned boxes. Cover it with a grid of boxes whose
                // centres lie inside the polygon (finer for big pavements), so
                // you stand where the rock is and nowhere else.
                const int N = (s > 3.0f && ysq < 0.45f) ? 4 : (s > 1.6f ? 3 : 2);
                const float R = 1.35f * b.r * s, cell = R / float(N);
                for (int gz = 0; gz < N * 2; ++gz)
                    for (int gx = 0; gx < N * 2; ++gx) {
                        const float wx = (float(gx) + 0.5f) * cell - R, wz = (float(gz) + 0.5f) * cell - R;
                        // world offset -> rock-local (inverse rotation), unit scale
                        const glm::vec2 l = glm::vec2(c * wx + sn * wz, -sn * wx + c * wz) / s - b.off;
                        bool in = true;
                        for (int k = 0; k < 5 && in; ++k)
                            // cell OVERLAPS the bed (centre within the polygon grown by ~half a cell)
                            in = glm::dot(l, glm::vec2(std::cos(b.ang[k]), std::sin(b.ang[k]))) <= b.rad[k] + 0.35f * cell / s;
                        if (!in) continue;
                        cols.push_back({ glm::vec3(x + wx - cell * 0.5f, bot, z + wz - cell * 0.5f),
                                         glm::vec3(x + wx + cell * 0.5f, top, z + wz + cell * 0.5f) });
                    }

            }
            return;
        }
        // Test aid: VIBE_LOG_ROCKS=1 lists the larger boulders near the castle.
        { static const bool kLog = std::getenv("VIBE_LOG_ROCKS") != nullptr; static int shown = 0;
          if (kLog && s > 1.6f && std::abs(x) < 160.0f && std::abs(z) < 160.0f && shown++ < 12)
              log::infof("[rock] baked %d at (%.1f, %.1f, %.1f) scale %.2f", arch[size_t(ai)].baked, x, gy, z, s); }
        float he = s * 0.72f;
        cols.push_back({ glm::vec3(x - he, gy - 1.0f, z - he),
                         glm::vec3(x + he, gy + s * ysq * 1.1f, z + he) });
    };
    auto add_rock = [&](float x, float z, float s, float ysq) { add_rock_ex(x, z, s, ysq, -1, 0.12f); };

    for (int a = 0; a < 60000 && st.formations < P.max_formations; ++a) {
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 6.0f) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 1.0f) continue;
        float slope = 1.0f - ny_at(x, z);
        float want = std::clamp(slope * 2.2f, 0.0f, 1.0f);
        want = std::max(want, std::clamp((h - P.rock_band_lo + 15.0f) / 30.0f, 0.0f, 1.0f));
        want *= 0.35f + 0.65f * std::clamp(fbm(glm::vec3(x * 0.01f, 3.0f, z * 0.01f), 2) * 2.0f - 0.4f, 0.0f, 1.0f);
        if (u() > want * 0.5f) continue;
        float size = 2.5f + 4.5f * u() * u();
        if (!try_mark(x, z, size * 1.6f)) continue;
        int n = 3 + int(u() * 4.0f);
        float rot0 = u() * 6.2831853f;
        for (int k = 0; k < n; ++k) {
            float ang = rot0 + float(k) * 2.4f + u() * 0.6f;
            float d = (k == 0) ? 0.0f : size * (0.6f + 0.6f * u());
            float s = size * ((k == 0) ? 1.0f : 0.45f + 0.45f * u());
            add_rock(x + std::cos(ang) * d, z + std::sin(ang) * d, s, 0.6f + 0.5f * u());
        }
        ++st.formations;
    }
    for (int a = 0; a < 40000 && st.boulders < P.max_boulders; ++a) {
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 3.0f) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 0.8f || h > P.tree_line + 30.0f) continue;
        if (ny_at(x, z) < 0.8f) continue;
        float s = 0.9f + 2.0f * u() * u();
        if (!try_mark(x, z, s * 1.2f)) continue;
        add_rock(x, z, s, 0.6f + 0.3f * u());
        ++st.boulders;
    }
    // Limestone pavements: wide, low slab formations lying almost flush
    // with gently sloping ground - worn stepped beds a few metres across.
    {
        int pavements = 0;
        for (int a = 0; a < 30000 && pavements < 140; ++a) {
            float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
            if (std::sqrt(x * x + z * z) < P.castle_clear + 6.0f) continue;
            if (excluded(x, z) || road_d(x, z) < road_clear + 7.0f) continue;
            float h = h_at(x, z);
            if (h < P.water_level + 1.2f || h > P.tree_line + 20.0f) continue;
            if (ny_at(x, z) < 0.90f) continue;
            float s = 3.5f + 4.5f * u();
            if (!try_mark(x, z, s * 1.25f)) continue;
            const int variant = 1 + 3 * int(u() * 2.99f);          // 1, 4 or 7: the slab archetypes
            add_rock_ex(x, z, s, 0.26f + 0.16f * u(), variant, 0.10f);
            if (pavements < 4) log::infof("[rock] pavement at (%.1f, %.1f, %.1f) size %.1f", x, h, z, s);
            ++pavements;
        }
        log::infof("[rock] %d limestone pavements", pavements);
    }
    // Trees. max_trees counts trees only (out already holds the rocks).

    // Test hook: VIBE_TEST_ROCKS=1 lines up one rock of every variant on the
    // open castle road (from about (-45, 52) toward the gate) for inspection.
    if (std::getenv("VIBE_TEST_ROCKS")) {
        for (int k = 0; k < 8; ++k) {
            const float x = -45.0f + 1.25f * float(k) + ((k & 1) ? 2.6f : -2.6f);
            const float z = 52.0f - 3.9f * float(k);
            add_rock_ex(x, z, k % 3 == 1 ? 2.2f : 1.3f, k % 3 == 1 ? 0.5f : 0.85f, k, 0.12f);
        }
    }
    const size_t rocks_placed = out.size();
    // Test hook: VIBE_TEST_TREES=x,z stands one tree of every type in rows of six from there (16 m
    // apart toward +x, a row every 20 m toward +z: conifers first, then the broadleaf trees, in
    // the order of the [vegtris] log lines of VIBE_LOG_VEG), and keeps the wood out of the way.
    float tt_x = 0.0f, tt_z = 0.0f;
    bool  tt_on = false;
    if (const char* e = std::getenv("VIBE_TEST_TREES")) {
        tt_on = std::sscanf(e, "%f,%f", &tt_x, &tt_z) == 2;
        int k = 0;
        for (size_t i = 0; tt_on && i < arch.size(); ++i) {
            if (arch[i].species > 1) continue;
            const float x = tt_x + 16.0f * float(k % 6), z = tt_z + 20.0f * float(k / 6);
            ++k;
            VegInstance vi{};
            vi.pos_scale = glm::vec4(x, h_at(x, z) - 0.05f, z, 1.0f);
            vi.rot_arch_phase = glm::vec4(0.0f, float(i), 0.0f, 0.0f);
            out.push_back(vi);
            log::infof("[vegpos] test tree %zu at (%.1f, %.1f, %.1f), %.1f m tall", i, x, vi.pos_scale.y, z, arch[i].height);
        }
    }
    {   // Dev aid: the rocks nearest the castle (where to look when tuning them).
        std::vector<size_t> ord(out.size());
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
        auto d2 = [&](size_t i) { return out[i].pos_scale.x * out[i].pos_scale.x + out[i].pos_scale.z * out[i].pos_scale.z; };
        std::partial_sort(ord.begin(), ord.begin() + std::min<size_t>(8, ord.size()), ord.end(), [&](size_t a, size_t b) { return d2(a) < d2(b); });
        for (size_t k = 0; k < std::min<size_t>(8, ord.size()); ++k) {
            const auto& r = out[ord[k]];
            log::infof("[rock] near castle: (%.1f, %.1f, %.1f) scale %.2f ysq %.2f arch %d", r.pos_scale.x, r.pos_scale.y, r.pos_scale.z,
                       r.pos_scale.w, r.rot_arch_phase.w, int(r.rot_arch_phase.y + 0.5f));
        }
    }
    // Every attempt draws from its own random stream (seeded by the attempt
    // number): what happens in one place - a painted area, a density change
    // - then cannot shift the trees everywhere else on the map.
    for (int a = 0; a < P.tree_attempts && int(out.size() - rocks_placed) < P.max_trees; ++a) {
        uint32_t rs = ((uint32_t(a) + 1u) * 2654435761u) ^ (uint32_t(P.seed) * 40503u);
        rs ^= rs >> 15; rs *= 0x2C1B3C6Du; rs ^= rs >> 12; rs |= 1u;
        auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 4.5f) continue;
        if (tt_on && x > tt_x - 30.0f && x < tt_x + 110.0f && z > tt_z - 40.0f && z < tt_z + 100.0f) continue;
        float h = h_at(x, z);
        if (h < std::max(P.grass_band_lo, P.water_level + 2.0f) || h > P.tree_line) continue;
        if (!dry(x, z, 2.0f)) continue;
        if (ny_at(x, z) < 0.72f) continue;
        const PaintPreset* pt = P.paint ? P.paint->at(kPaintTrees, x, z) : nullptr;
        float f = forest(x, z);
        float dens = f * f + 0.004f;                  // rare lone trees in meadows
        if (pt) dens = pt->tree_density;              // painted: the preset decides (0 = none)
        if (u() > dens * P.tree_density) continue;
        bool pine = u() < std::clamp((h - 28.0f) / 30.0f, 0.15f, 0.95f);
        if (pt) pine = u() < pt->pine_share;
        float scale = 0.75f + 0.55f * u();
        if (pt) scale *= std::clamp(pt->tree_size, 0.3f, 2.5f);
        // Old-growth stands: patches of the forest (a few hundred metres
        // across) where the veterans stand - bigger, further apart, with a
        // few ordinary trees between them.
        const float og = fbm(glm::vec3(x * 0.0048f, 57.0f, z * 0.0048f), 2);
        const bool old_tree = og > 0.515f && !old_trees[pine ? 0 : 1].empty() && u() < 0.72f;
        if (old_tree) scale = 0.95f + 0.35f * u();
        if (!try_mark(x, z, (pine ? 1.6f : 2.4f) * scale * (old_tree ? 2.1f : 1.0f))) continue;
        const auto& pool = old_tree ? old_trees[pine ? 0 : 1] : by_species[pine ? 0 : 1];
        int ai = pool[size_t(u() * float(pool.size())) % pool.size()];
        VegInstance vi{};
        vi.pos_scale = glm::vec4(x, h - 0.05f, z, scale);
        vi.rot_arch_phase = glm::vec4(u() * 6.2831853f, float(ai), u() * 6.2831853f, 0.0f);
        out.push_back(vi);
        float tr = 0.38f * scale;
        cols.push_back({ glm::vec3(x - tr, h - 0.5f, z - tr),
                         glm::vec3(x + tr, h + arch[size_t(ai)].height * scale * 0.6f, z + tr) });
        if (pine) ++st.pines; else ++st.broadleaf;
        if (old_tree) ++st.old;
    }
    // Bushes (no collision).
    int bushes = 0;
    for (int a = 0; a < P.bush_attempts && bushes < P.max_bushes; ++a) {
        uint32_t rs = ((uint32_t(a) + 1u) * 2246822519u) ^ (uint32_t(P.seed) * 69069u);
        rs ^= rs >> 15; rs *= 0x2C1B3C6Du; rs ^= rs >> 12; rs |= 1u;
        auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
        float x = (u() * 2.0f - 1.0f) * ext, z = (u() * 2.0f - 1.0f) * ext;
        if (std::sqrt(x * x + z * z) < P.castle_clear - 6.0f) continue;
        if (excluded(x, z) || road_d(x, z) < road_clear + 1.5f) continue;
        float h = h_at(x, z);
        if (h < P.water_level + 1.0f || h > P.tree_line + 8.0f) continue;
        if (ny_at(x, z) < 0.75f) continue;
        const PaintPreset* pb = P.paint ? P.paint->at(kPaintBushes, x, z) : nullptr;
        float f = forest(x, z);
        if (!dry(x, z, 1.2f)) continue;
        // Undergrowth inside forests + thickets along the edges.
        float dens = std::clamp(f * 0.45f + (1.0f - std::abs(f - 0.4f) * 3.0f) * 0.5f, 0.0f, 1.0f) * 0.7f + 0.01f;
        if (pb) dens = pb->bush_density;              // painted: the preset decides (0 = none)
        if (u() > dens * P.bush_density) continue;
        float scale = 0.6f + 0.9f * u();
        const auto& pool = by_species[2];
        int ai = pool[size_t(u() * float(pool.size())) % pool.size()];
        VegInstance vi{};
        vi.pos_scale = glm::vec4(x, h, z, scale);
        vi.rot_arch_phase = glm::vec4(u() * 6.2831853f, float(ai), u() * 6.2831853f, 0.0f);
        out.push_back(vi);
        ++bushes;
    }
    // Riverside: scrub on the banks of the river (river_shrub_density). The banks are a
    // narrow strip, so the tries are thrown at the river's own box, and - unlike everything
    // above - right down to the water's edge.
    if (!river_pool.empty()) {
        int placed = 0;
        const float bx0 = std::max(-85.0f, -ext), bx1 = std::min(2095.0f, ext), bz0 = std::max(-110.0f, -ext), bz1 = std::min(780.0f, ext);
        for (int a = 0; a < P.river_attempts && placed < P.max_river_shrubs && bx1 > bx0 && bz1 > bz0; ++a) {
            uint32_t rs = ((uint32_t(a) + 1u) * 2654435761u) ^ (uint32_t(P.seed) * 40503u) ^ 0x51F15EEDu;
            rs ^= rs >> 15; rs *= 0x2C1B3C6Du; rs ^= rs >> 12; rs |= 1u;
            auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
            const float x = bx0 + u() * (bx1 - bx0), z = bz0 + u() * (bz1 - bz0);
            const float pick = u(), size = u(), rot = u(), phase = u(), which = u();
            if (world_river_distance(x, z) > 40.0f) continue;                 // (cheap; most tries end here)
            if (std::sqrt(x * x + z * z) < P.castle_clear) continue;
            if (excluded(x, z) || road_d(x, z) < road_clear + 2.0f) continue;
            const float h = h_at(x, z);
            const float dens = river_shrub_density(x, z, h, ny_at(x, z), P.water_level);
            if (pick > dens * P.bush_density) continue;
            const int ai = river_pool[size_t(which * float(river_pool.size())) % river_pool.size()];
            VegInstance vi{};
            vi.pos_scale = glm::vec4(x, h - 0.06f, z, 0.75f + 0.55f * size);
            vi.rot_arch_phase = glm::vec4(rot * 6.2831853f, float(ai), phase * 6.2831853f, 0.0f);
            out.push_back(vi);
            ++placed;
        }
        st.river_shrubs = placed;
        if (std::getenv("VIBE_LOG_VEG")) {            // where to look: a few shrubs and trees for test captures
            int rn = 0, tn = 0, on = 0;
            for (const auto& v : out) {
                const auto& A = arch[size_t(v.rot_arch_phase.y + 0.5f)];
                if (A.riverside && v.pos_scale.x > 300.0f && rn < 4 && ++rn) std::puts(("[vegpos] river " + std::to_string(v.pos_scale.x) + " " + std::to_string(v.pos_scale.y) + " " + std::to_string(v.pos_scale.z)).c_str());
                else if (A.species <= 1 && A.old && on < 4 && ++on) std::puts(("[vegpos] old" + std::to_string(A.species) + " " + std::to_string(v.pos_scale.x) + " " + std::to_string(v.pos_scale.y) + " " + std::to_string(v.pos_scale.z)).c_str());
                else if (A.species <= 1 && !A.old && std::abs(v.pos_scale.x) < 300.0f && tn < 6 && ++tn) std::puts(("[vegpos] tree" + std::to_string(A.species) + " " + std::to_string(v.pos_scale.x) + " " + std::to_string(v.pos_scale.y) + " " + std::to_string(v.pos_scale.z)).c_str());
            }
            std::fflush(stdout);
        }
    }
    // Painted areas are filled deliberately as well: the random attempts
    // above cover the whole map, so a small painted patch caught few or
    // none of them (painting a bush often planted nothing). Every painted
    // 2 m cell gets its share of plants here, from a random stream of its
    // own (stable from stroke to stroke).
    if (P.paint && !P.paint->empty()) {
        const float cell = PaintMap::cell();
        for (int j = 0; j < kPaintDim; ++j)
            for (int i = 0; i < kPaintDim; ++i) {
                const size_t k = size_t(j) * kPaintDim + size_t(i);
                const uint8_t it = P.paint->layer[kPaintTrees][k], ib = P.paint->layer[kPaintBushes][k];
                if (!it && !ib) continue;
                uint32_t rs = (uint32_t(k) + 1u) * 2654435761u ^ 0x9E3779B9u;
                rs ^= rs >> 15; rs *= 0x2C1B3C6Du; rs ^= rs >> 12; rs |= 1u;
                auto u = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
                const float x = (float(i) + u()) * cell - 0.5f * kPaintSide;
                const float z = (float(j) + u()) * cell - 0.5f * kPaintSide;
                const float h = h_at(x, z);
                if (h < P.water_level + 0.4f || ny_at(x, z) < 0.6f) continue;
                if (it && it < P.paint->presets.size()) {
                    const PaintPreset& pt = P.paint->presets[it];
                    // ~one tree per 5 x 5 m at full density.
                    if (pt.trees && u() < pt.tree_density * 0.16f) {
                        const bool pine = u() < pt.pine_share;
                        const float scale = (0.75f + 0.55f * u()) * std::clamp(pt.tree_size, 0.3f, 2.5f);
                        if (!by_species[pine ? 0 : 1].empty() && try_mark(x, z, (pine ? 1.6f : 2.4f) * scale)) {
                            const auto& pool = by_species[pine ? 0 : 1];
                            const int ai = pool[size_t(u() * float(pool.size())) % pool.size()];
                            VegInstance vi{};
                            vi.pos_scale = glm::vec4(x, h - 0.05f, z, scale);
                            vi.rot_arch_phase = glm::vec4(u() * 6.2831853f, float(ai), u() * 6.2831853f, 0.0f);
                            out.push_back(vi);
                            const float tr = 0.38f * scale;
                            cols.push_back({ glm::vec3(x - tr, h - 0.5f, z - tr),
                                             glm::vec3(x + tr, h + arch[size_t(ai)].height * scale * 0.6f, z + tr) });
                            if (pine) ++st.pines; else ++st.broadleaf;
                        }
                    }
                }
                if (ib && ib < P.paint->presets.size()) {
                    const PaintPreset& pb = P.paint->presets[ib];
                    // ~one bush per 2.5 x 2.5 m at full density.
                    if (pb.bushes && !by_species[2].empty() && u() < pb.bush_density * 0.6f) {
                        const auto& pool = by_species[2];
                        const int ai = pool[size_t(u() * float(pool.size())) % pool.size()];
                        VegInstance vi{};
                        vi.pos_scale = glm::vec4(x, h, z, 0.6f + 0.9f * u());
                        vi.rot_arch_phase = glm::vec4(u() * 6.2831853f, float(ai), u() * 6.2831853f, 0.0f);
                        out.push_back(vi);
                        ++bushes;
                    }
                }
            }
    }
    st.bushes = bushes;
    return out;
}

} // namespace vibe
