// Materials at the water (after the reference game's deferred wetness pass; the code is ours) and
// the switches of the sea's surface follow-up. Fills the scene block's water_wet / water_grid
// (scene_ubo.glsl). The shading is in shaders/shore_wet.glsl and its callers:
//   * the wet band above every waterline (terrain, meshes, scattered stones): linear over W
//     above max(the authored top, the solver's surface - 0.1 W), a ragged top from a 100 m noise,
//     albedo a * saturate(0.5 - 0.5 luma), roughness to a mirror at the line;
//   * at the sea, the band follows the surf's run-up on the beach and a dry line below high water;
//   * puddles in the material's hollows (the - 1.5 h threshold), their height x (1 - 0.4 p);
//   * growth under water on up-facing submerged stone and bed, the bed's tint by depth, the bed's
//     relief marched along the refracted ray, no snow in the wet band.
// Switches (env, read once; 1 = off): VIBE_NO_WET_BAND, VIBE_NO_SEA_SWASH, VIBE_NO_PUDDLE_HOLLOWS,
// VIBE_NO_WATER_GROWTH, VIBE_NO_BED_TINT, VIBE_NO_BED_RELIEF, VIBE_NO_WET_SNOW_RULE;
// VIBE_WATER_GROWTH=<rate> (default 1). The sea's surface: VIBE_NO_SEA_GRID, VIBE_NO_FOAM_LIFT,
// VIBE_NO_LAMP_GLINT, VIBE_NO_WATER_FROXEL (see water_grid.cpp / the water shading).
#include "engine/vk_engine/internal.h"

#include <cstdlib>

namespace vibe {

namespace {
bool wet_env_on(const char* k) { const char* v = std::getenv(k); return v && *v && *v != '0'; }
float wet_env_f(const char* k, float def) { const char* v = std::getenv(k); return (v && *v) ? float(std::atof(v)) : def; }
}

// Whether the sea's projected grid drew its depth picture this frame (set by water_grid.cpp).
namespace { bool s_grid_active = false; }
void water_grid_set_active(bool on) { s_grid_active = on; }

void water_wet_fill(SceneUBO& d) {
    static const int bits =
        (wet_env_on("VIBE_NO_WET_BAND")       ? 0 : 1) |
        (wet_env_on("VIBE_NO_SEA_SWASH")      ? 0 : 2) |
        (wet_env_on("VIBE_NO_PUDDLE_HOLLOWS") ? 0 : 4) |
        (wet_env_on("VIBE_NO_WATER_GROWTH")   ? 0 : 8) |
        (wet_env_on("VIBE_NO_BED_TINT")       ? 0 : 16) |
        (wet_env_on("VIBE_NO_BED_RELIEF")     ? 0 : 32) |
        (wet_env_on("VIBE_NO_WET_SNOW_RULE")  ? 0 : 64) |
        (wet_env_on("VIBE_NO_BED_SCAN")       ? 0 : 128) |     // (the bed from the scanned sets: shaders/water_bed.glsl)
        (wet_env_on("VIBE_NO_STREAM_SEAM_FIX") ? 0 : 256) |    // (water_course.glsl wc_stream_at: the segment by reach, not by distance)
        (wet_env_on("VIBE_NO_BANK_STRIP")     ? 0 : 512) |     // (water_bed.glsl: the bank's shingle / gravel / mud strip)
        (wet_env_on("VIBE_NO_CALM_MARGIN")    ? 0 : 1024) |    // (terrain_rm_main_water.glsl: flow and ripples calm toward the bank)
        (wet_env_on("VIBE_DEBUG_STREAM_EDGE") ? 2048 : 0) |    // (terrain_raymarch.frag: the ground beside the stream painted by the rule that left it dry)
        (wet_env_on("VIBE_NO_STREAM_RELIEF")  ? 0 : 4096) |
        (wet_env_on("VIBE_NO_SLOPED_RIVER")   ? 0 : 8192);     // (water_level.glsl: the sea 1 m under the datum, the river falling to it; CPU water_fx.cpp water_level_off)     // (water_relief.glsl: the stream's surface in relief; CPU twin water_fx.cpp stream_relief)
    static const float growth = wet_env_f("VIBE_WATER_GROWTH", 1.0f);
    d.water_wet = glm::vec4(float(bits), growth, 0.25f, 0.15f);
    static const bool no_lift = wet_env_on("VIBE_NO_FOAM_LIFT"), no_glint = wet_env_on("VIBE_NO_LAMP_GLINT"),
                      no_frox = wet_env_on("VIBE_NO_WATER_FROXEL");
    d.water_grid = glm::vec4(s_grid_active ? 1.0f : 0.0f, no_lift ? 0.0f : 0.24f, no_glint ? 0.0f : 1.0f, no_frox ? 0.0f : 1.0f);
}

} // namespace vibe
