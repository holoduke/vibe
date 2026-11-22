#pragma once

// Procedural world props baked into static world-space meshes:
//   * rock formations - clustered, strata-squashed, noise-displaced
//     boulders on slopes / ridges / the rock band, plus lone erratics;
//   * trees           - pines (stacked jagged cones) and broadleaf trees
//     (branching trunk + lumpy canopy clusters), forest density from a
//     low-frequency noise field and the height / slope bands;
//   * bushes          - low lumpy shrubs around forest edges.
// Output goes through the castle static-mesh path (vk_engine/castle_mesh):
// BLAS + RT shadows, sun shadow map, depth prepass, deferred lighting.
// Meshes are chunked on a grid so frustum culling works per chunk.

#include "game/level.h"

#include <glm/vec4.hpp>

#include <vector>

namespace vibe {

struct Heightmap;
struct Road;

struct WorldPropParams {
    // Painted vegetation (veg_paint.h): overrides trees / bushes where a
    // preset was painted. nullptr = everything automatic.
    const struct PaintMap* paint = nullptr;
    float water_level   = 20.9f;
    float grass_band_lo = 17.0f;    // trees start above the beach
    float tree_line     = 72.0f;    // no trees above
    float rock_band_lo  = 60.0f;    // formations favour above this
    float half_extent   = 520.0f;   // placement square around origin
    float castle_clear  = 34.0f;    // keep the castle surroundings open
    int   seed          = 0x5EED;
    // GPU-instanced vegetation (whole map).
    float veg_half_extent = 1000.0f;
    int   max_trees       = 80000;
    int   tree_attempts   = 3000000;
    int   max_bushes      = 30000;
    int   max_formations  = 650;
    // Circles (xz, radius) kept free of trees / rocks (AA vehicles etc.).
    std::vector<glm::vec3> keep_out;
    int   max_boulders    = 1800;
    int   bush_attempts   = 500000;
    float tree_density    = 1.0f;   // multiplies the forest density field
    float bush_density    = 1.0f;
    int   river_attempts  = 220000; // tries at the river-bank scrub (a narrow strip: most miss)
    int   max_river_shrubs = 6000;
};

// Baked tree impostors (vk_engine/veg_impostor.cpp): every tree archetype seen from 8
// directions, as two RGBA8 atlases of w x h (colour + coverage: sRGB; normal + leaf mark).
// row[a] = the archetype's row of tiles, -1 for what has no impostor (bushes, rocks).
struct VegArchetype;
struct VegImpostorAtlas {
    int w = 0, h = 0, rows = 0;
    std::vector<unsigned char> color, normal;
    std::vector<int> row;
};
VegImpostorAtlas bake_veg_impostors(const std::vector<VegArchetype>& archs,
                                    const unsigned char* leaf_px, int leaf_w, int leaf_h);

struct WorldPropStats { int formations = 0, boulders = 0, pines = 0, broadleaf = 0, bushes = 0, old = 0, river_shrubs = 0; };

// Vegetation archetype: one species variant with a 4-level LOD chain in
// local space (base at origin, y up). Vertex uv = (wind weight, material:
// 0 bark, 1 needles, 2 leaves, 3 bush, 4 rock).
constexpr int kVegLods = 4;
constexpr int kVegRtLevels = 4;   // BLAS detail levels for ray tracing (one per mesh LOD)
struct VegArchetype {
    int species = 0;                 // 0 pine, 1 broadleaf, 2 bush, 3 rock
    bool old = false;                // old-growth tree (placed in old-growth stands only)
    bool riverside = false;          // river-bank scrub: planted along the river only (species 2)
    int  baked = -1;                 // rock sculpted + baked in Blender (tools/rocks): its number, else -1
    game::LevelMesh lods[kVegLods];
    float radius = 1.0f;             // bounding-sphere radius (scale 1)
    float height = 1.0f;
};
// River-bank scrub: how much of it grows at (x, z), 0..1 - the rule shared by the home
// tile's planting (world_props.cpp) and the streamed world beyond it (veg_stream.cpp).
// `h` ground height, `ny` the ground normal's y, `wl` the water level. The river leaves the
// lake by the castle among reeds and runs to the sea: scrub is sparse near the castle and
// takes the banks over downstream.
float river_shrub_density(float x, float z, float h, float ny, float wl);

// GPU instance (std430, 32 B): .xyz pos, .w scale | .x rotY, .y archetype,
// .z wind phase, .w vertical scale (0 = 1).
struct VegInstance {
    glm::vec4 pos_scale;
    glm::vec4 rot_arch_phase;
};

std::vector<VegArchetype> build_veg_archetypes(int seed);
std::vector<VegInstance> place_vegetation(const Heightmap& hm, const Road* road,
                                          const WorldPropParams& p,
                                          const std::vector<VegArchetype>& arch,
                                          std::vector<collision::AABB>& trunk_colliders,
                                          WorldPropStats& stats);

// Appends meshes (world space) and player colliders.
WorldPropStats build_world_props(const Heightmap& hm, const Road* road,
                                 const WorldPropParams& p,
                                 std::vector<game::LevelMesh>& out_meshes,
                                 std::vector<collision::AABB>& out_colliders);

} // namespace vibe
