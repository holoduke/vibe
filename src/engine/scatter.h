#pragma once

// AAA-terrain Phase 3: ground clutter scatter. Procedurally-generated
// rock meshes instanced over the heightmap by splat-style rules
// (boulders favour steep/rocky ground, pebbles favour dirt/sand/grass).
// AAA terrain reads "AAA" largely because the near field is covered in
// small meshes that break the bare-heightfield look — this is that
// layer, built on the same instancing pattern as the grass field.

#include "engine/mesh.h"
#include "engine/water_course.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstdint>
#include <cmath>
#include <vector>

namespace vibe {

struct Heightmap;
struct HeightmapParams;

// Per-instance data — layout consumed by scatter.vert at binding 1.
struct ScatterInstance {
    glm::vec4 pos_scale;   // .xyz = base world pos, .w = uniform scale (m)
    glm::vec4 rot_tint;    // .x = Y rotation (rad), .yzw = albedo tint
};
static_assert(sizeof(ScatterInstance) == 32, "instance layout is GPU ABI");

// Contiguous instance range of one 32 m ground cell (instances are sorted
// by cell at build time) - lets the colour and shadow passes cull per cell
// instead of drawing every instance in the world twice per frame.
struct ScatterCell {
    glm::vec3 mn, mx;      // world AABB of the cell's instances (+ margin)
    uint32_t  first = 0;   // first instance (absolute index in its buffer)
    uint32_t  count = 0;
};

struct ScatterSet {
    Mesh          rock_mesh{};                  // shared lumpy-rock mesh
    VkBuffer      instance_buffer = VK_NULL_HANDLE;
    VmaAllocation instance_alloc  = nullptr;
    uint32_t      instance_count  = 0;
    // Small plants (flowers / ferns): one mesh per species, instances in
    // one buffer sorted by species. Vertex uv.x = part (0 leaf / stem,
    // 1 petal, 2 flower centre), uv.y = wind sway weight.
    // fern, daisy, poppy, bluebell, cornflower, buttercup, lupin, reed,
    // clover, plantain, dandelion, nettle, foxglove, thistle, mushrooms,
    // grass tuft (a clump of long blades; uv.x = 8 +- 0.4 across the blade)
    static constexpr int kPlantSpecies = 27;   // 16 = water lilies, 17 = turf (short grass, placed on the GPU only) 18 = flowering meadow (in fields, placed on the GPU only), 19..26 = small debris (placed on the GPU only: scatter_debris_meshes.inl)
    Mesh          plant_mesh[kPlantSpecies]{};
    // Levels of detail for distant cells, made automatically (meshoptimizer
    // simplification; the water lily's first one is hand-built): about a
    // third of the triangles beyond kPlantFarLod, about a tenth beyond kPlantFarLod2.
    Mesh          plant_mesh_far[kPlantSpecies]{};
    Mesh          plant_mesh_far2[kPlantSpecies]{};
    static constexpr float kPlantFarLod  = 22.0f;
    static constexpr float kPlantFarLod2 = 48.0f;
    uint32_t      plant_first[kPlantSpecies]{};
    uint32_t      plant_count[kPlantSpecies]{};
    VkBuffer      plant_buffer = VK_NULL_HANDLE;
    VmaAllocation plant_alloc  = nullptr;
    // Persistently-mapped instance data + per-instance height offset to
    // the raymarched surface (finer FBM than the CPU heightmap). Applied
    // / removed in place by scatter_set_raymarch_surface().
    void*              inst_mapped  = nullptr;
    void*              plant_mapped = nullptr;
    uint32_t           plant_gen = 1;          // bumped whenever plant_mapped is rewritten (the GPU culling keeps a copy)
    std::vector<float> inst_dy, plant_dy;
    // CPU copies of the instances (the mapped buffers are write-combined):
    // craters edit these and write the changed ones through.
    std::vector<ScatterInstance> inst_cpu, plant_cpu;
    bool               rm_surface = false;
    std::vector<ScatterCell> rock_cells;
    std::vector<ScatterCell> plant_cells[kPlantSpecies];
    // The automatic grass - long tufts (species 15) and turf (17) - is placed on the GPU every
    // frame (shaders/tuft_place.comp, vk_engine/plant_gpu.cpp). The ground's height comes from
    // the terrain's own height map there. This is the rest of what that pass has to know about
    // the ground: a top-down picture of kTuftCapN x kTuftCapN texels over
    // [-tuft_cap_ext, +tuft_cap_ext] in x and z, 4 bytes a texel:
    //   0  how much long grass may grow (0 = none: painted grass, buildings, the road, water,
    //      bare rock; less under the trees)
    //   1  how dry it is there
    // Empty = no automatic grass (or it was placed on the CPU: ScatterParams::tufts_cpu).
    static constexpr int kTuftCapN = 1024;
    float                tuft_cap_ext = 0.0f;
    std::vector<uint8_t> tuft_rules;
    uint32_t             tuft_cap_gen = 0;       // bumped when the picture changes (a crater)
    // The two grass meshes as the shaders need them: height at size 1, blades, the radius the
    // blades' roots lie within, how deep they are (scatter_plant_meshes.inl tuft_root).
    static constexpr float kTuftShape[4] = { 0.90f, 32.0f, 0.14f, 0.12f };
    static constexpr float kTurfShape[4] = { 0.36f, 32.0f, 0.22f, 0.06f };
    static constexpr float kMeadowShape[4] = { 0.66f, 16.0f, 0.20f, 0.05f };
};

struct ScatterParams {
    float half_extent   = 220.0f;   // placement square around origin (m)
    // 24k x 240 verts ≈ 5.8M VS invocations — bounded vertex cost.
    int   max_instances = 32000;
    int   seed          = 0xB0BA;
    // Rules (heights in world Y, matching the splat bands).
    float water_level   = 20.9f;    // reject below water + margin
    float flower_density = 1.0f;    // ferns + flowers multiplier (0..2)
    float ground_density = 1.0f;    // ground cover (clover, weeds, nettles, fungi) multiplier (0..3)
    // Tree crowns (x, z, crown radius) - forest-floor vs meadow placement.
    const std::vector<glm::vec3>* trees = nullptr;
    float rock_band_h   = 60.0f;    // boulders favoured above this
    float snow_band_h   = 110.0f;   // density tapers off into snow
    // Castle keep-out (same as grass).
    glm::vec2 keep_out_xz = glm::vec2(14.0f, 14.0f);
    // More places nothing grows (buildings): x0, z0, x1, z1 in world metres.
    std::vector<glm::vec4> keep_rects;
    bool blocked(float x, float z) const {
        if (std::abs(x) < keep_out_xz.x && std::abs(z) < keep_out_xz.y) return true;
        for (const auto& r : keep_rects) if (x > r.x && x < r.z && z > r.y && z < r.w) return true;
        if (wc::stream_distance(glm::vec2(x, z), water_level) < 0.8f) return true;      // (the stream's water: engine/water_course.h)
        return false;
    }
    // Optional extra keep-out (the castle road). Null = none.
    const struct Road* road = nullptr;
    // Painted vegetation (veg_paint.h): its grass layer overrides the grass
    // tufts where painted. Null = automatic everywhere.
    const struct PaintMap* paint = nullptr;
    // When set, per-instance offsets to the raymarched surface are
    // precomputed (`rm_octaves` FBM octaves, the near-camera detail).
    const HeightmapParams* rm_params = nullptr;
    int   rm_octaves = 14;
    // true: the automatic grass tufts are placed here, as instances (the path without the
    // plants' GPU pass). false: only the ground picture for the GPU placement is made.
    bool  tufts_cpu = false;
};

// Builds the rock mesh + places instances. Boulders: 0.15-0.55 m on
// steep slopes / rock band / scattered through dirt. Pebbles:
// 0.04-0.12 m clustered on flatter ground. One mesh, one instance
// buffer, one draw.
ScatterSet build_scatter(VkDevice device, VmaAllocator alloc, VkQueue queue,
                         uint32_t queue_family, const Heightmap& hm,
                         const ScatterParams& params);

void destroy_scatter(VmaAllocator alloc, ScatterSet& s);

// Move all instances onto the raymarched surface (on) or back onto the
// CPU heightmap surface (off). Cheap: rewrites the mapped buffers.
void scatter_set_raymarch_surface(ScatterSet& s, bool on);

// Draw the visible cells of one mesh: consecutive visible ranges are merged
// into one instanced draw (cells are contiguous in the instance buffer).
template <class Visible>
inline uint32_t scatter_draw_cells(VkCommandBuffer cmd, uint32_t index_count,
                                   const std::vector<ScatterCell>& cells, Visible visible) {
    uint32_t run_first = 0, run_count = 0, drawn = 0;
    for (const ScatterCell& c : cells) {
        if (!visible(c)) continue;
        drawn += c.count;
        if (run_count && run_first + run_count == c.first) { run_count += c.count; continue; }
        if (run_count) vkCmdDrawIndexed(cmd, index_count, run_count, 0, 0, run_first);
        run_first = c.first;
        run_count = c.count;
    }
    if (run_count) vkCmdDrawIndexed(cmd, index_count, run_count, 0, 0, run_first);
    return drawn;
}

} // namespace vibe
