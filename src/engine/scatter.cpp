#include "engine/scatter.h"
#include "engine/veg_paint.h"
#include "engine/jobs.h"
#include "engine/road.h"
#include "engine/terrain.h"
#include "engine/log.h"
#include "engine/lsystem.h"

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace qlike {
#include "engine/scatter_plant_meshes.inl"   // mesh builders for the small plants: ferns, flowers, reeds, clover, mushrooms, grass tufts, water lilies, the rock

ScatterSet build_scatter(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family, const Heightmap& hm,
                         const ScatterParams& params) {
    ScatterSet s{};
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    build_rock_mesh_data(verts, idx);
    s.rock_mesh = create_mesh_from_data(device, alloc, queue, queue_family,
                                        verts.data(), uint32_t(verts.size()),
                                        idx.data(), uint32_t(idx.size()));
    if (hm.heights.empty() || params.max_instances <= 0) return s;

    std::mt19937 rng(uint32_t(params.seed));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int W = hm.width();
    auto h_at = [&](float wx, float wz) { return hm.sample_world(wx, wz); };
    auto n_y_at = [&](float wx, float wz) {
        float e = 1.0f;
        float dx = h_at(wx + e, wz) - h_at(wx - e, wz);
        float dz = h_at(wx, wz + e) - h_at(wx, wz - e);
        return 2.0f * e / std::sqrt(dx * dx + dz * dz + 4.0f * e * e);
    };
    (void)W;

    std::vector<ScatterInstance> inst;
    inst.reserve(size_t(params.max_instances));
    const int attempts = params.max_instances * 4;
    for (int i = 0; i < attempts &&
                    int(inst.size()) < params.max_instances; ++i) {
        float wx = (u01(rng) * 2.0f - 1.0f) * params.half_extent;
        float wz = (u01(rng) * 2.0f - 1.0f) * params.half_extent;
        if (params.blocked(wx, wz)) continue;
        if (params.road && wx > params.road->bb_min.x && wx < params.road->bb_max.x &&
            wz > params.road->bb_min.y && wz < params.road->bb_max.y &&
            params.road->distance(wx, wz) + 1.6f * road_edge_jitter(wx, wz) <
                params.road->half_width + params.road->shoulder + 0.6f) continue;
        float h = h_at(wx, wz);
        if (h < params.water_level + 0.4f) continue;     // not in/at water
        float ny = n_y_at(wx, wz);
        float slope = 1.0f - ny;

        // Rule weights. Boulders: steep ground or the rock band, plus a
        // sparse scattering everywhere. Pebbles: gentler ground,
        // clustered by a low-frequency hash so they come in patches.
        float rock_w = std::clamp(slope * 2.4f, 0.0f, 1.0f);
        rock_w = (std::max)(rock_w,
                            std::clamp((h - params.rock_band_h) / 25.0f,
                                       0.0f, 1.0f));
        rock_w *= 1.0f - std::clamp((h - params.snow_band_h) / 20.0f,
                                    0.0f, 0.7f);
        float clump = hash31(glm::vec3(std::floor(wx / 9.0f), 7.0f,
                                       std::floor(wz / 9.0f)));
        float pebble_w = (1.0f - slope * 1.5f) * (clump > 0.45f ? 1.0f : 0.15f);

        float r = u01(rng);
        bool boulder = r < rock_w * 0.16f + 0.012f;       // sparse big rocks
        bool pebble  = !boulder && r < 0.55f * std::clamp(pebble_w, 0.0f, 1.0f);
        if (!boulder && !pebble) continue;

        ScatterInstance si{};
        float sc = boulder
            ? 0.15f + u01(rng) * u01(rng) * 0.45f          // 0.15-0.6 m
            : (u01(rng) < 0.28f ? 0.10f + u01(rng) * u01(rng) * 0.20f     // cobbles 0.10-0.30 m
                                : 0.035f + u01(rng) * 0.085f);            // pebbles 0.035-0.12 m
        // Sink proportionally so the flattened base sits in the ground.
        si.pos_scale = glm::vec4(wx, h - sc * 0.18f, wz, sc);
        float tint = 0.8f + u01(rng) * 0.4f;
        // Slight warm/cool per-rock variation around neutral.
        glm::vec3 tc(tint * (0.96f + u01(rng) * 0.08f),
                     tint,
                     tint * (0.94f + u01(rng) * 0.08f));
        // Stone families: most neutral, some dark basalt, pale limestone,
        // red sandstone, blue-grey slate.
        {
            float fam = u01(rng);
            if      (fam < 0.14f) tc *= glm::vec3(0.55f, 0.56f, 0.60f);
            else if (fam < 0.26f) tc *= glm::vec3(1.35f, 1.32f, 1.22f);
            else if (fam < 0.36f) tc *= glm::vec3(1.25f, 0.92f, 0.78f);
            else if (fam < 0.45f) tc *= glm::vec3(0.88f, 0.96f, 1.10f);
        }
        si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(),
                                tc.x, tc.y, tc.z);
        inst.push_back(si);
    }

    // ---- Small plants: ferns + three flower species ----
    {
        std::vector<Vertex> pv; std::vector<uint32_t> pi;
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
            pv.clear(); pi.clear();
            switch (sp) {
            case 0: build_fern(pv, pi); break;
            // { height, nodes, leaves, leaf_len, leaf_w, palmate, branches, branch_len,
            //   petals, head_r, cup, petal_w, raceme, raceme_len, raceme_roll, pedicel_pitch, lean, seed }
            case 1: build_herb(pv, pi, { 0.34f, 5, 3, 0.17f, 0.030f, false, 2, 0.11f, 14, 0.058f, 0.05f, 0.70f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 3u }); break;   // daisy: branched, several heads
            case 2: build_herb(pv, pi, { 0.44f, 5, 3, 0.20f, 0.040f, false, 1, 0.13f, 5, 0.075f, 0.55f, 1.05f, 0, 0.3f, 137.5f, 62.0f, 0.25f, 5u }); break;   // poppy: cup + a bud
            case 3: build_herb(pv, pi, { 0.34f, 3, 3, 0.24f, 0.022f, false, 0, 0.10f, 6, 0.036f, -0.9f, 0.95f, 6, 0.50f, 12.0f, 118.0f, 2.6f, 7u }); break;   // bluebell: one-sided nodding raceme
            case 4: build_herb(pv, pi, { 0.48f, 6, 4, 0.16f, 0.018f, false, 2, 0.14f, 16, 0.050f, 0.20f, 0.45f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 9u }); break;   // cornflower: thin rays, branched
            case 5: build_herb(pv, pi, { 0.26f, 4, 3, 0.13f, 0.045f, false, 3, 0.09f, 5, 0.034f, 0.70f, 1.10f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 11u }); break;  // buttercup: loose cyme of small cups
            case 7: build_reed(pv, pi); break;
            case 8:  build_clover(pv, pi); break;
            case 9:  build_plantain(pv, pi); break;
            case 10: build_dandelion(pv, pi); break;
            case 11: build_nettle(pv, pi); break;
            // foxglove: tall one-sided spike of drooping bells; thistle: branched, purple tufts
            case 12: build_herb(pv, pi, { 0.95f, 4, 4, 0.24f, 0.060f, false, 0, 0.10f, 6, 0.030f, -1.2f, 1.0f, 14, 0.55f, 22.0f, 105.0f, 0.35f, 17u }); break;
            case 13: build_herb(pv, pi, { 0.62f, 6, 5, 0.17f, 0.030f, false, 2, 0.12f, 30, 0.026f, 2.4f, 0.35f, 0, 0.3f, 137.5f, 62.0f, 0.0f, 19u }); break;
            case 14: build_mushrooms(pv, pi); break;
            case 15: build_grass_tuft(pv, pi); break;
            case 16: build_water_lily(pv, pi); break;
            case 6: build_herb(pv, pi, { 0.66f, 4, 4, 0.20f, 0.040f, true, 0, 0.10f, 5, 0.030f, 0.45f, 0.95f, 12, 0.48f, 137.5f, 58.0f, 0.0f, 13u }); break;  // lupin: palmate leaves, dense spike
            }
            s.plant_mesh[sp] = create_mesh_from_data(device, alloc, queue, queue_family,
                                                     pv.data(), uint32_t(pv.size()),
                                                     pi.data(), uint32_t(pi.size()));
            if (sp == 16) {
                std::vector<Vertex> fv; std::vector<uint32_t> fi;
                build_water_lily(fv, fi, true);
                s.plant_mesh_far[sp] = create_mesh_from_data(device, alloc, queue, queue_family,
                                                             fv.data(), uint32_t(fv.size()), fi.data(), uint32_t(fi.size()));
                log::infof("[scatter] water lily mesh: %u tris, far %u tris", uint32_t(pi.size() / 3), uint32_t(fi.size() / 3));
            }
        }
        std::vector<ScatterInstance> per[ScatterSet::kPlantSpecies];
        const float ext = std::min(params.half_extent * 1.3f, 280.0f);
        for (int i = 0; i < 420000; ++i) {
            float wx = (u01(rng) * 2.0f - 1.0f) * ext;
            float wz = (u01(rng) * 2.0f - 1.0f) * ext;
            if (params.blocked(wx, wz)) continue;
            if (params.road && wx > params.road->bb_min.x && wx < params.road->bb_max.x &&
                wz > params.road->bb_min.y && wz < params.road->bb_max.y &&
                params.road->distance(wx, wz) + road_edge_jitter(wx, wz) < params.road->half_width + 0.4f) continue;
            float h = h_at(wx, wz);
            if (h < params.water_level + 0.6f || h > params.rock_band_h + 5.0f) continue;
            float ny = n_y_at(wx, wz);
            if (ny < 0.8f) continue;
            // Patches: each species owns low-frequency noise regions so
            // flowers come in drifts of one colour, ferns in damp hollows.
            auto patch = [&](float f, float seed) {
                return hash31(glm::vec3(std::floor(wx * f), seed, std::floor(wz * f))) * 0.6f +
                       hash31(glm::vec3(std::floor(wx * f * 3.1f), seed + 9.0f, std::floor(wz * f * 3.1f))) * 0.4f;
            };
            float pf = patch(0.06f, 1.0f), pd = patch(0.05f, 2.0f), pp = patch(0.045f, 3.0f), pb = patch(0.055f, 4.0f);
            float pc = patch(0.05f, 5.0f), pu = patch(0.065f, 6.0f), pl = patch(0.04f, 7.0f);
            int sp = -1;
            float r = u01(rng) / std::max(params.flower_density, 0.01f);
            if (pf > 0.62f && r < 0.22f) sp = 0;
            else if (pd > 0.60f && r < 0.55f) sp = 1;
            else if (pp > 0.64f && r < 0.55f) sp = 2;
            else if (pb > 0.64f && r < 0.55f) sp = 3;
            else if (pc > 0.66f && r < 0.50f) sp = 4;
            else if (pu > 0.63f && r < 0.60f) sp = 5;
            else if (pl > 0.70f && r < 0.40f) sp = 6;
            if (sp < 0) continue;
            ScatterInstance si{};
            float sc = (sp == 0) ? 0.55f + u01(rng) * 0.6f : 1.1f + u01(rng) * 0.6f;
            si.pos_scale = glm::vec4(wx, h - 0.02f, wz, sc);
            glm::vec3 col;
            switch (sp) {
            case 0: col = glm::vec3(0.10f, 0.22f, 0.06f) * (0.8f + 0.4f * u01(rng)); break;
            case 1: col = glm::vec3(0.85f, 0.84f, 0.80f); break;
            case 2: col = glm::vec3(0.75f, 0.06f, 0.03f) * (0.85f + 0.3f * u01(rng)); break;
            case 3: col = glm::vec3(0.22f, 0.20f, 0.62f) * (0.85f + 0.3f * u01(rng)); break;
            case 4: col = glm::vec3(0.16f, 0.26f, 0.85f) * (0.85f + 0.3f * u01(rng)); break;   // cornflower
            case 5: col = glm::vec3(0.95f, 0.76f, 0.06f) * (0.9f + 0.2f * u01(rng)); break;    // buttercup
            default: {   // lupin: pink .. purple drifts
                float k = u01(rng);
                col = glm::mix(glm::vec3(0.80f, 0.30f, 0.55f), glm::vec3(0.45f, 0.22f, 0.75f), k);
                break;
            }
            }
            si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(), col.x, col.y, col.z);
            per[sp].push_back(si);
        }
        // Reed beds along the shore: dense clumps on the wet margin, from a
        // little below the waterline (standing in the shallows) to just above.
        {
            int reeds = 0;
            for (int i = 0; i < 600000 && reeds < 9000; ++i) {
                float wx = (u01(rng) * 2.0f - 1.0f) * ext;
                float wz = (u01(rng) * 2.0f - 1.0f) * ext;
                if (params.blocked(wx, wz)) continue;
                if (params.road && wx > params.road->bb_min.x && wx < params.road->bb_max.x &&
                    wz > params.road->bb_min.y && wz < params.road->bb_max.y &&
                    params.road->distance(wx, wz) < params.road->half_width + params.road->shoulder + 1.0f) continue;
                float h = h_at(wx, wz);
                if (h < params.water_level - 0.35f || h > params.water_level + 0.9f) continue;
                if (n_y_at(wx, wz) < 0.72f) continue;
                // Beds, not a uniform fringe: low-frequency patches plus density
                // falling off with height above the water.
                float bed = hash31(glm::vec3(std::floor(wx * 0.11f), 31.0f, std::floor(wz * 0.11f))) * 0.6f +
                            hash31(glm::vec3(std::floor(wx * 0.37f), 37.0f, std::floor(wz * 0.37f))) * 0.4f;
                float wet = 1.0f - glm::smoothstep(params.water_level + 0.1f, params.water_level + 0.9f, h);
                if (bed < 0.42f || u01(rng) > wet * glm::clamp(params.flower_density, 0.0f, 2.0f)) continue;
                ScatterInstance si{};
                si.pos_scale = glm::vec4(wx, h - 0.03f, wz, 0.75f + 0.55f * u01(rng));
                glm::vec3 col = glm::mix(glm::vec3(0.32f, 0.38f, 0.16f), glm::vec3(0.42f, 0.40f, 0.20f), u01(rng));
                si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(), col.x, col.y, col.z);
                per[7].push_back(si);
                ++reeds;
            }
            // Water lilies: rafts of pads on the quiet shallows, in groups -
            // a group is a patch of 20-60 rafts within a few metres, mostly
            // close to the shore (10 cm to 1.1 m of water).
            {
                int lilies = 0;
                for (int g = 0; g < 90000 && lilies < 9000; ++g) {
                    const float gx = (u01(rng) * 2.0f - 1.0f) * ext, gz = (u01(rng) * 2.0f - 1.0f) * ext;
                    const float gh = h_at(gx, gz);
                    if (gh > params.water_level - 0.12f || gh < params.water_level - 1.0f) continue;
                    const int n = 20 + int(u01(rng) * 40.0f);
                    const float gr = 2.0f + 3.5f * u01(rng);
                    const bool pink = u01(rng) < 0.3f;
                    for (int k = 0; k < n; ++k) {
                        const float a = u01(rng) * glm::two_pi<float>(), d = gr * std::sqrt(u01(rng));
                        const float wx = gx + std::cos(a) * d, wz = gz + std::sin(a) * d;
                        const float h = h_at(wx, wz);
                        if (h > params.water_level - 0.10f || h < params.water_level - 1.15f) continue;
                        // The whole raft must float: no pad may lie on the
                        // shore or poke into the bank (a raft reaches ~0.6 m x scale).
                        const float sc = 0.8f + 0.7f * u01(rng);
                        bool afloat = true;
                        for (int q = 0; q < 8 && afloat; ++q) {
                            const float qa = float(q) * 0.7853982f;
                            if (h_at(wx + std::cos(qa) * 0.72f * sc, wz + std::sin(qa) * 0.72f * sc) > params.water_level - 0.06f) afloat = false;
                        }
                        if (!afloat) continue;
                        ScatterInstance si{};
                        si.pos_scale = glm::vec4(wx, params.water_level, wz, sc);
                        // Flower colour; a raft without a flower is marked by a
                        // dark tint (the shader hides its petals).
                        glm::vec3 col = pink ? glm::vec3(0.92f, 0.55f, 0.66f) : glm::vec3(0.95f, 0.94f, 0.88f);
                        if (u01(rng) < 0.55f) col = glm::vec3(0.02f);
                        si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(), col.x, col.y, col.z);
                        per[16].push_back(si);
                        ++lilies;
                    }
                }
                log::infof("[scatter] %d water lily rafts", lilies);
            }
        }

        // ---- Ground cover. Ecology, the way open-world games layer it:
        // meadows get clover drifts, plantain and dandelions (plantain also
        // crowds trampled path edges); forest edges nettle clumps and
        // foxgloves; the shaded forest floor mushroom clusters; thistles
        // stand alone in rough grass. `cover` = how much tree crown is
        // overhead (0 open .. 1 under a canopy), from a 4 m grid. ----
        {
            const float gd = glm::clamp(params.ground_density, 0.0f, 3.0f);
            const int G = int(std::ceil(2.0f * ext / 4.0f));
            std::vector<float> cover(size_t(G) * size_t(G), 0.0f);
            if (params.trees) {
                for (const glm::vec3& t : *params.trees) {
                    const float R = t.z * 1.35f + 2.0f;
                    const int x0 = std::max(0, int((t.x - R + ext) / 4.0f)), x1 = std::min(G - 1, int((t.x + R + ext) / 4.0f));
                    const int z0 = std::max(0, int((t.y - R + ext) / 4.0f)), z1 = std::min(G - 1, int((t.y + R + ext) / 4.0f));
                    for (int gz = z0; gz <= z1; ++gz)
                        for (int gx = x0; gx <= x1; ++gx) {
                            const float cx = -ext + (float(gx) + 0.5f) * 4.0f, cz = -ext + (float(gz) + 0.5f) * 4.0f;
                            const float d = std::sqrt((cx - t.x) * (cx - t.x) + (cz - t.y) * (cz - t.y));
                            float& c = cover[size_t(gz) * size_t(G) + size_t(gx)];
                            c = std::min(1.0f, c + std::max(0.0f, 1.0f - d / R));
                        }
                }
            }
            auto cover_at = [&](float x, float z) {
                const float fx = (x + ext) / 4.0f - 0.5f, fz = (z + ext) / 4.0f - 0.5f;
                const int ix0 = std::clamp(int(std::floor(fx)), 0, G - 2), iz0 = std::clamp(int(std::floor(fz)), 0, G - 2);
                const float tx = glm::clamp(fx - float(ix0), 0.0f, 1.0f), tz = glm::clamp(fz - float(iz0), 0.0f, 1.0f);
                auto C = [&](int a, int b) { return cover[size_t(b) * size_t(G) + size_t(a)]; };
                return glm::mix(glm::mix(C(ix0, iz0), C(ix0 + 1, iz0), tx), glm::mix(C(ix0, iz0 + 1), C(ix0 + 1, iz0 + 1), tx), tz);
            };
            auto gpatch = [&](float x, float z, float f, float seed) {
                return hash31(glm::vec3(std::floor(x * f), seed, std::floor(z * f))) * 0.6f +
                       hash31(glm::vec3(std::floor(x * f * 3.1f), seed + 9.0f, std::floor(z * f * 3.1f))) * 0.4f;
            };
            const int tries = int(520000.0f * gd);
            for (int i = 0; i < tries; ++i) {
                const float wx = (u01(rng) * 2.0f - 1.0f) * ext;
                const float wz = (u01(rng) * 2.0f - 1.0f) * ext;
                if (params.blocked(wx, wz)) continue;
                float road_d = 1e9f;
                if (params.road && wx > params.road->bb_min.x - 4.0f && wx < params.road->bb_max.x + 4.0f &&
                    wz > params.road->bb_min.y - 4.0f && wz < params.road->bb_max.y + 4.0f)
                    road_d = params.road->distance(wx, wz) + road_edge_jitter(wx, wz) - params.road->half_width;
                if (road_d < 0.35f) continue;                                      // never on the paving
                const float h = h_at(wx, wz);
                if (h < params.water_level + 0.5f || h > params.rock_band_h + 3.0f) continue;
                if (n_y_at(wx, wz) < 0.80f) continue;
                const float cv = cover_at(wx, wz);
                const float edge = std::max(0.0f, 1.0f - std::abs(cv - 0.35f) / 0.35f);   // forest margin, peaks at cover 0.35
                const float verge = 1.0f - glm::smoothstep(0.4f, 3.0f, road_d);           // trampled path edge
                const float r = u01(rng);
                int sp = -1;
                // Weighted pick (probability per try; the try count carries
                // the density slider): the weights encode where each grows.
                float w[7];
                w[0] = (cv < 0.30f && gpatch(wx, wz, 0.07f, 21.0f) > 0.45f) ? 0.40f : 0.0f;                 // clover drifts
                w[1] = (gpatch(wx, wz, 0.09f, 22.0f) > 0.35f ? 0.06f : 0.015f) * (1.0f - cv) + 0.30f * verge;  // plantain (+ path edges)
                w[2] = (cv < 0.35f && gpatch(wx, wz, 0.06f, 23.0f) > 0.30f) ? 0.05f : 0.0f;               // dandelion
                w[3] = gpatch(wx, wz, 0.12f, 24.0f) > 0.45f ? 0.07f * edge : 0.0f;                        // nettle clumps
                w[4] = 0.014f * edge;                                                                   // foxglove
                w[5] = cv < 0.2f ? 0.0035f : 0.0f;                                                      // lone thistles
                w[6] = (cv > 0.45f && gpatch(wx, wz, 0.15f, 25.0f) > 0.58f) ? 0.035f * cv : 0.0f;          // fungi
                float acc = 0.0f;
                for (int k = 0; k < 7; ++k) { acc += w[k]; if (r < acc) { sp = 8 + k; break; } }
                if (sp < 0) continue;
                ScatterInstance si{};
                float sc = 0.85f + 0.4f * u01(rng);
                if (sp == 8) sc = 0.8f + 0.5f * u01(rng);
                if (sp == 11 || sp == 12) sc = 0.8f + 0.45f * u01(rng);
                si.pos_scale = glm::vec4(wx, h - (sp == 14 ? 0.01f : 0.015f), wz, sc);
                glm::vec3 col;
                switch (sp) {
                case 8:  col = u01(rng) < 0.8f ? glm::vec3(0.88f, 0.86f, 0.80f) : glm::vec3(0.85f, 0.55f, 0.65f); break;   // white / pink clover
                case 10: col = glm::vec3(0.98f, 0.74f, 0.04f) * (0.9f + 0.15f * u01(rng)); break;
                case 12: col = glm::mix(glm::vec3(0.80f, 0.32f, 0.62f), glm::vec3(0.62f, 0.25f, 0.70f), u01(rng)); break;
                case 13: col = glm::vec3(0.58f, 0.22f, 0.62f) * (0.85f + 0.3f * u01(rng)); break;
                case 14: {
                    const float k = u01(rng);
                    col = k < 0.08f ? glm::vec3(0.72f, 0.10f, 0.05f)                        // the odd red one
                        : glm::mix(glm::vec3(0.50f, 0.33f, 0.18f), glm::vec3(0.68f, 0.56f, 0.40f), u01(rng));
                    break;
                }
                default: col = glm::vec3(0.6f, 0.6f, 0.6f); break;                          // leaf-only: tint unused
                }
                si.rot_tint = glm::vec4(u01(rng) * glm::two_pi<float>(), col.x, col.y, col.z);
                per[sp].push_back(si);
            }
            // ---- Grass tufts: clumps of long grass over the open ground, in
            // soft patches (smooth low-frequency noise, like the EZ-Tree demo's
            // 'patchiness'), thinning out under the trees. ----
            {
                auto vnoise = [&](float x, float z, float seed) {
                    const float fx = std::floor(x), fz = std::floor(z);
                    float tx = x - fx, tz = z - fz;
                    tx = tx * tx * (3.0f - 2.0f * tx); tz = tz * tz * (3.0f - 2.0f * tz);
                    auto H = [&](float a, float b) { return hash31(glm::vec3(a, seed, b)); };
                    return glm::mix(glm::mix(H(fx, fz), H(fx + 1.0f, fz), tx),
                                    glm::mix(H(fx, fz + 1.0f), H(fx + 1.0f, fz + 1.0f), tx), tz);
                };
                int tufts = 0;
                const int ttries = int(1600000.0f * std::min(gd, 2.0f));
                // One tuft: fresh green to yellow-green, `dry` pulls it to straw.
                auto add_tuft = [&](float wx, float wz, float h, float size, float dry, float r0, float r1, float r2) {
                    ScatterInstance si{};
                    si.pos_scale = glm::vec4(wx, h - 0.02f, wz, size);
                    glm::vec3 col = glm::mix(glm::vec3(0.085f, 0.20f, 0.045f), glm::vec3(0.20f, 0.30f, 0.07f), r0);
                    col = glm::mix(col, glm::vec3(0.34f, 0.31f, 0.13f), glm::clamp(dry, 0.0f, 1.0f) * 0.8f) * (0.85f + 0.3f * r1);
                    si.rot_tint = glm::vec4(r2 * glm::two_pi<float>(), col.x, col.y, col.z);
                    per[15].push_back(si);
                };
                // Own random stream per attempt / per cell (not the shared rng):
                // painting grass somewhere must not move anything else.
                uint32_t trs = 1u;
                auto tu = [&trs]() { trs ^= trs << 13; trs ^= trs >> 17; trs ^= trs << 5; return float(trs >> 8) / 16777216.0f; };
                auto tseed = [&trs](uint32_t k, uint32_t salt) {
                    trs = ((k + 1u) * 2654435761u) ^ salt; trs ^= trs >> 15; trs *= 0x2C1B3C6Du; trs ^= trs >> 12; trs |= 1u;
                };
                for (int i = 0; i < ttries && tufts < 120000; ++i) {
                    tseed(uint32_t(i), 0x7F4A7C15u);
                    const float wx = (tu() * 2.0f - 1.0f) * ext;
                    const float wz = (tu() * 2.0f - 1.0f) * ext;
                    if (params.paint && params.paint->at(kPaintGrass, wx, wz)) continue;   // painted: second pass
                    if (params.blocked(wx, wz)) continue;
                    if (params.road && wx > params.road->bb_min.x - 4.0f && wx < params.road->bb_max.x + 4.0f &&
                        wz > params.road->bb_min.y - 4.0f && wz < params.road->bb_max.y + 4.0f &&
                        params.road->distance(wx, wz) + road_edge_jitter(wx, wz) - params.road->half_width < 0.7f) continue;
                    const float h = h_at(wx, wz);
                    if (h < params.water_level + 0.4f || h > params.rock_band_h) continue;
                    if (n_y_at(wx, wz) < 0.78f) continue;
                    const float n = vnoise(wx * 0.022f, wz * 0.022f, 41.0f) * 0.65f + vnoise(wx * 0.075f, wz * 0.075f, 43.0f) * 0.35f;
                    const float open = 1.0f - glm::smoothstep(0.25f, 0.65f, cover_at(wx, wz));
                    const float dens = (0.06f + 0.94f * glm::smoothstep(0.40f, 0.60f, n)) * (0.15f + 0.85f * open);
                    if (tu() > dens) continue;
                    const float size = 0.8f + 0.7f * tu();
                    // Drier, strawier clumps here and there.
                    const float dry = glm::smoothstep(0.55f, 0.8f, vnoise(wx * 0.05f, wz * 0.05f, 47.0f)) * tu() * 0.9f;
                    add_tuft(wx, wz, h, size, dry, tu(), tu(), tu());
                    ++tufts;
                }
                // Painted grass: every painted cell gets its preset's density,
                // anywhere on the map (not only near the castle).
                int painted = 0;
                if (params.paint && !params.paint->layer[kPaintGrass].empty()) {
                    const auto& lay = params.paint->layer[kPaintGrass];
                    const float cell = PaintMap::cell();
                    for (int j = 0; j < kPaintDim && painted < 260000; ++j)
                        for (int i = 0; i < kPaintDim; ++i) {
                            const uint8_t id = lay[size_t(j) * kPaintDim + size_t(i)];
                            if (id == 0 || id >= params.paint->presets.size()) continue;
                            const PaintPreset& pp = params.paint->presets[id];
                            if (pp.grass_density <= 0.0f) continue;
                            tseed(uint32_t(j * kPaintDim + i), 0x51ED270Bu);
                            const float want = pp.grass_density * 2.4f;          // tufts per 2 m cell
                            int n = int(want);
                            if (tu() < want - float(n)) ++n;
                            for (int k = 0; k < n; ++k) {
                                const float wx = (float(i) + tu()) * cell - 0.5f * kPaintSide;
                                const float wz = (float(j) + tu()) * cell - 0.5f * kPaintSide;
                                const float size = (0.8f + 0.7f * tu()) * glm::clamp(pp.grass_height, 0.3f, 3.0f);
                                const float r0 = tu(), r1 = tu(), r2 = tu();
                                if (params.road && wx > params.road->bb_min.x - 4.0f && wx < params.road->bb_max.x + 4.0f &&
                                    wz > params.road->bb_min.y - 4.0f && wz < params.road->bb_max.y + 4.0f &&
                                    params.road->distance(wx, wz) - params.road->half_width < 0.5f) continue;
                                const float h = h_at(wx, wz);
                                if (h < params.water_level + 0.15f) continue;
                                if (n_y_at(wx, wz) < 0.55f) continue;
                                add_tuft(wx, wz, h, size, pp.grass_dry * (0.75f + 0.25f * r0), r0, r1, r2);
                                ++painted;
                            }
                        }
                }
                log::infof("[scatter] %d grass tufts (+%d painted)", tufts, painted);
            }
        }

        // Lift for the raymarched ground relief (raymarch_ground_relief):
        // packed as whole turns of the Y rotation (4 mm steps), which the
        // vertex shaders read back with floor(rot / 2 pi) and fade with
        // distance exactly like the relief itself (10 - 26 m). Without it the
        // relief (~0.17 m on average) swallowed clover and plantain close by.
        // (Per-instance and independent: spread over the worker pool.)
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp)
            jobs::parallel_for(int(per[sp].size()), 2048, [&, sp](int i_begin, int i_end) {
            if (sp == 16) return;                      // water lilies float: no ground relief
            for (int ii = i_begin; ii < i_end; ++ii) {
                ScatterInstance& si = per[sp][size_t(ii)];
                const float x = si.pos_scale.x, z = si.pos_scale.z;
                float d = raymarch_ground_relief(x, z, h_at(x, z), params.water_level);
                if (params.road && x > params.road->bb_min.x - 2.0f && x < params.road->bb_max.x + 2.0f &&
                    z > params.road->bb_min.y - 2.0f && z < params.road->bb_max.y + 2.0f)
                    d *= glm::smoothstep(params.road->half_width + 0.1f, params.road->half_width + 0.9f,
                                         params.road->distance(x, z));
                // Ferns and flowers (species 0-3) stand on thin stems: the drawn
                // ground is up to ~5 cm below this estimate (the soil texture's own
                // displacement is only known on the GPU), which left a visible gap
                // under some of them. Root them that much lower - the ground hides
                // the rest of the stem.
                if (sp <= 3) d -= 0.055f;
                const float q = std::clamp(std::floor(d / 0.004f + 0.5f), 0.0f, 120.0f);
                si.rot_tint.x = std::fmod(si.rot_tint.x, glm::two_pi<float>()) + glm::two_pi<float>() * q;
            }
            });

        std::vector<ScatterInstance> all;
        for (int sp = 0; sp < ScatterSet::kPlantSpecies; ++sp) {
            s.plant_first[sp] = uint32_t(all.size());
            sort_into_cells(per[sp], uint32_t(all.size()), s.plant_cells[sp]);
            s.plant_count[sp] = uint32_t(per[sp].size());
            all.insert(all.end(), per[sp].begin(), per[sp].end());
        }
        if (params.rm_params) {
            s.plant_dy.resize(all.size());
            jobs::parallel_for(int(all.size()), 2048, [&](int i_begin, int i_end) {
                for (int i = i_begin; i < i_end; ++i)
                    s.plant_dy[size_t(i)] = raymarch_fine_correction(hm, *params.rm_params, all[size_t(i)].pos_scale.x,
                                                                     all[size_t(i)].pos_scale.z, params.rm_octaves);
            });
            for (uint32_t i = 0; i < s.plant_count[16]; ++i) s.plant_dy[size_t(s.plant_first[16] + i)] = 0.0f;   // lilies float
        }
        if (!all.empty()) {
            VkBufferCreateInfo pb{};
            pb.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            pb.size = sizeof(ScatterInstance) * all.size();
            pb.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VmaAllocationCreateInfo pa{};
            pa.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            pa.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo pai{};
            if (vmaCreateBuffer(alloc, &pb, &pa, &s.plant_buffer, &s.plant_alloc, &pai) == VK_SUCCESS) {
                std::memcpy(pai.pMappedData, all.data(), sizeof(ScatterInstance) * all.size());
                s.plant_mapped = pai.pMappedData;
                s.plant_cpu = all;
            }
        }
        log::infof("[scatter] plants: %u ferns, %u daisies, %u poppies, %u bluebells, %u cornflowers, %u buttercups, %u lupins, %u reeds",
                   s.plant_count[0], s.plant_count[1], s.plant_count[2], s.plant_count[3],
                   s.plant_count[4], s.plant_count[5], s.plant_count[6], s.plant_count[7]);
        log::infof("[scatter] ground: %u clover, %u plantain, %u dandelion, %u nettle, %u foxglove, %u thistle, %u fungi",
                   s.plant_count[8], s.plant_count[9], s.plant_count[10], s.plant_count[11],
                   s.plant_count[12], s.plant_count[13], s.plant_count[14]);
    }

    if (inst.empty()) return s;
    sort_into_cells(inst, 0, s.rock_cells);
    const VkDeviceSize bytes = sizeof(ScatterInstance) *
                               VkDeviceSize(inst.size());
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    if (vmaCreateBuffer(alloc, &bci, &aci, &s.instance_buffer,
                        &s.instance_alloc, &ai) != VK_SUCCESS) {
        log::warn("[scatter] instance buffer alloc failed");
        return s;
    }
    std::memcpy(ai.pMappedData, inst.data(), size_t(bytes));
    s.inst_mapped = ai.pMappedData;
    s.inst_cpu = inst;
    if (params.rm_params) {
        s.inst_dy.resize(inst.size());
        jobs::parallel_for(int(inst.size()), 2048, [&](int i_begin, int i_end) {
            for (int i = i_begin; i < i_end; ++i)
                s.inst_dy[size_t(i)] = raymarch_fine_correction(hm, *params.rm_params, inst[size_t(i)].pos_scale.x,
                                                                inst[size_t(i)].pos_scale.z, params.rm_octaves);
        });
    }
    s.instance_count = uint32_t(inst.size());
    log::infof("[scatter] %u clutter instances placed (rock mesh %u tris)",
               s.instance_count, s.rock_mesh.index_count / 3);
    return s;
}

void destroy_scatter(VmaAllocator alloc, ScatterSet& s) {
    destroy_mesh(alloc, s.rock_mesh);
    for (auto& m : s.plant_mesh) if (m.vertex_buffer) destroy_mesh(alloc, m);
    for (auto& m : s.plant_mesh_far) if (m.vertex_buffer) destroy_mesh(alloc, m);
    if (s.plant_buffer) {
        vmaDestroyBuffer(alloc, s.plant_buffer, s.plant_alloc);
        s.plant_buffer = VK_NULL_HANDLE;
    }
    if (s.instance_buffer) {
        vmaDestroyBuffer(alloc, s.instance_buffer, s.instance_alloc);
        s.instance_buffer = VK_NULL_HANDLE;
        s.instance_alloc = nullptr;
    }
    s.instance_count = 0;
}

} // namespace qlike

namespace qlike {
void scatter_set_raymarch_surface(ScatterSet& s, bool on) {
    if (s.rm_surface == on) return;
    const float sign = on ? 1.0f : -1.0f;
    auto apply = [&](void* mapped, const std::vector<float>& dy) {
        if (!mapped) return;
        auto* p = static_cast<ScatterInstance*>(mapped);
        for (size_t i = 0; i < dy.size(); ++i) p[i].pos_scale.y += sign * dy[i];
    };
    apply(s.inst_mapped, s.inst_dy);
    apply(s.plant_mapped, s.plant_dy);
    s.rm_surface = on;
}
} // namespace qlike
