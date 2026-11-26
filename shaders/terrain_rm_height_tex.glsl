// terrain_rm_height_tex.glsl - the home tile's height textures as functions: the sculpt / road delta
// (sampleHeightDelta) and the baked heights (meshHeight). Shared by terrain_raymarch.frag and the
// patch renderer's vertex shader (terrain_patch.vert); both declare u_terrain_height (binding 8)
// and u_terrain_height_full (binding 17) before including it.

// (in a far chart - another part of the planet - there is no home tile: its side is nothing)
bool g_far_chart = false;        // (set first thing in main: scene.globe_info.x)
#define kHeightmapSide (g_far_chart ? 1.0e-3 : 2048.0)
// Forward decl — terrainDetailNoise is defined after `noised` + `m2`
// (further down in the file) so we can reuse them. We use a forward
// declaration here so sampleHeightDelta can call it without moving
// the whole noise toolkit above the declarations of u_terrain_height.
float terrainDetailNoise(vec2 worldXZ);
vec3  noised(in vec2 p);   // single-octave value noise + analytical derivatives

// Stored ground anywhere on the planet (the regions' height tiles: region_tiles.glsl) is a
// delta over the formula like home's own sculpt, and is added here - so every height this
// shader makes has it.
#include "region_tiles.glsl"

float sampleHeightDelta(vec2 worldXZ) {
    float rd = region_delta(worldXZ);
    // Was: 4 compare branches before the fetch. Replaced with clamp +
    // textureLod 0 — sampler is CLAMP_TO_EDGE and the heightmap delta
    // is ~0 at the world boundary (no sculpting past the edge), so a
    // clamped read at out-of-world XZ returns the right value too.
    // Beyond the home tile there is no sculpting. The edge texels are NOT
    // all zero, and a clamped read drew them out into ruled furrows running
    // straight away from the tile (lines in the meadows east/west/north/south
    // of home): fade the delta out over the tile's last metres instead.
    vec2 aq = abs(worldXZ);
    float edge = kHeightmapSide * 0.5 - max(aq.x, aq.y);
    if (edge <= 0.0) return rd;
    vec2 uv = (worldXZ / kHeightmapSide) + vec2(0.5);
    return textureLod(u_terrain_height, uv, 0.0).r * smoothstep(0.0, 14.0, edge) + rd;
}

// === Mesh-terrain height (water-only mode) ==========================
// Bilinear lookup of the FULL baked heightmap — identical to the
// surface the rasterised CDLOD mesh draws. Used INSTEAD of the FBM
// terrainM* functions throughout the water path when the engine runs
// the water-only pass over mesh terrain, so the water never reflects
// or shows procedural noise that disagrees with the mesh.
float meshHeight(vec2 worldXZ) {
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
    return textureLod(u_terrain_height_full, uv, 0.0).r;
}
vec3 meshNormal(vec2 worldXZ) {
    const float e = 1.5;   // ≈ heightmap cell — smooth mesh-scale normal
    float hL = meshHeight(worldXZ - vec2(e, 0.0));
    float hR = meshHeight(worldXZ + vec2(e, 0.0));
    float hD = meshHeight(worldXZ - vec2(0.0, e));
    float hU = meshHeight(worldXZ + vec2(0.0, e));
    return normalize(vec3(hL - hR, 2.0 * e, hD - hU));
}
