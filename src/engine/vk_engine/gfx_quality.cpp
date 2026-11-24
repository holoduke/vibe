// Graphics quality: the settings that decide the frame rate, each as five levels - low,
// medium, high, ultra, insane - and one overall choice that sets them all.
//
// Every row maps a level to the engine's own values (RtSettings); nothing new is rendered
// here. What a row costs was measured on the development machine (docs/ARCHITECTURE.md,
// "Graphics quality": the table of milliseconds per row), and a row that did not change the
// frame time or the picture was not kept. High is what the game is tuned for.
//
// The levels chosen are saved (gfx_<key>); the values they set are saved as always, so a
// value changed by hand under the Graphics tab stays - the row then simply shows the level
// it was last set to.
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/terrain_raymarch.h"

#include <imgui.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vibe {

namespace {

template <class T> T pick(int level, T a, T b, T c, T d, T e) {
    const T v[5] = { a, b, c, d, e };
    return v[std::clamp(level, 0, 4)];
}

}  // namespace

const std::vector<VulkanEngine::GfxRow>& VulkanEngine::gfx_rows() {
    using E = VulkanEngine;
    static const std::vector<GfxRow> rows = {
        // ---- resolution ----
        { "resolution", "Render resolution", "The share of the screen's pixels that is really drawn; the upscaler makes the rest.\nEverything costs by the pixel: Ultra is about a fifth slower than High, Insane about half.\nLow to High are the same: the least the upscaler takes (half the screen each way).",
          [](E& e, int l) { e.rt_.render_scale = pick(l, 0.50f, 0.50f, 0.50f, 0.59f, 0.72f); e.gfx_scale_changed_ = true; } },
        { "terrain_resolution", "Terrain resolution", "The ground is the costliest thing in the picture. It can be drawn with fewer pixels than the rest and enlarged:\nlow and medium at half size each way, high at three quarters, ultra and insane at full size.",
          [](E& e, int l) { const float s = pick(l, 0.5f, 0.5f, 0.75f, 1.0f, 1.0f);
                            if (std::abs(e.rt_.terrain_raymarch_scale - s) > 1e-3f) { e.rt_.terrain_raymarch_scale = s; e.gfx_terrain_scale_changed_ = true; }
                            e.rt_.terrain_rm_temporal = l == 1 || l == 2; } },
        // ---- the ground ----
        { "terrain_steps", "Terrain precision", "How finely the ground's shape is traced (steps along each ray, and their length).\nLower: faster, with small errors on ridges and far slopes.",
          [](E& e, int l) { e.rt_.terrain_raymarch_max_steps = pick(l, 110, 150, 200, 260, 320); e.rt_.terrain_raymarch_step_factor = pick(l, 0.75f, 0.65f, 0.55f, 0.45f, 0.38f); } },
        { "terrain_detail", "Terrain detail", "The layers of fine detail in the ground's shading (rock faces, small folds).",
          // (Never terrain_raymarch_octaves: those are the ground's SHAPE, which the game's physics and
          //  everything standing on it share - with fewer, sand and roads no longer lie where they are.)
          [](E& e, int l) { e.rt_.terrain_raymarch_normal_octaves = pick(l, 9, 10, 12, 15, 18); e.rt_.terrain_rm_normal_px = pick(l, 3.0f, 2.0f, 1.5f, 1.0f, 0.0f); } },
        { "terrain_distance", "Terrain detail distance", "How far from the camera the ground keeps its full detail.",
          [](E& e, int l) { e.rt_.terrain_raymarch_lod_near_m = pick(l, 40.0f, 60.0f, 80.0f, 120.0f, 200.0f); e.rt_.terrain_raymarch_lod_far_m = pick(l, 500.0f, 1000.0f, 2000.0f, 2000.0f, 2000.0f);
                            } },
        { "terrain_first_pass", "Terrain first pass", "The ground is found first in a coarse picture, then refined. Coarser: faster, a little less exact at silhouettes.",
          [](E& e, int l) { e.rt_.terrain_rm_prepass = true; e.rt_.terrain_rm_prepass_div = pick(l, 4, 3, 2, 2, 2); e.rt_.terrain_rm_converge = pick(l, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f); } },
        { "terrain_edges", "Terrain shading at speed", "Shades the ground close to the camera once per 2 x 2 pixels while it rushes past (driving, flying low);\nfar ground and every outline keep every pixel, and so does everything while you stand or walk.\nLow: also the nearest 8 m at any speed. Ultra and insane: every pixel, always.",
          [](E& e, int l) { e.rt_.terrain_vrs = pick(l, 2, 1, 1, 0, 0); } },
        { "terrain_relief", "Surface relief", "The fine depth of the ground close by: stones, ripples in sand, small bumps.\nLow and medium: in the light and shade only, not in the ground's outline (low also without the loose stones).\nFrom high: all of it.",
          [](E& e, int l) { e.rt_.terrain_rm_skip = pick(l, 128 | 32768, 128, 0, 0, 0); e.rt_.terrain_rm_relief_lite = l < 4;
                            e.rt_.spom_strength = pick(l, 0.0f, 0.6f, 1.0f, 1.0f, 1.0f); e.rt_.terrain_pom_far_m = pick(l, 80.0f, 140.0f, 220.0f, 300.0f, 400.0f); } },
        { "terrain_shadows", "Terrain shadows", "The ground's own shadows: hills on valleys, and their soft edges.",
          [](E& e, int l) { e.rt_.terrain_raymarch_shadow_steps = pick(l, 10, 16, 20, 32, 48); e.rt_.terrain_pcss_samples_cap = pick(l, 1, 3, 6, 8, 8); } },
        // ---- light ----
        { "sun_shadows", "Sun shadow softness", "Rays per pixel for the sun's shadow: more gives smoother soft edges.",
          [](E& e, int l) { e.rt_.shadow_samples = pick(l, 2, 6, 19, 28, 40); } },
        { "leaf_shadows", "Shadows through leaves", "Whether a ray through a tree tests every leaf or takes a leaf card as solid.\nLow: solid cards (fast, heavier shade). Insane: every ray tests every leaf.",
          [](E& e, int l) { e.rt_.foliage_ray_mode = pick(l, 2, 1, 1, 1, 0); e.rt_.gi_leaf_lite = l < 4; } },
        { "rr_model", "Denoiser model", "Which DLSS Ray Reconstruction model cleans up the ray-traced light.\nLow and medium: the earlier one (about half a millisecond a frame faster). From high: the latest.",
          [](E& e, int l) { e.rt_.dlss_rr_model = l < 2 ? 1 : 0; } },
        { "ambient_occlusion", "Ambient occlusion", "The soft darkening in corners and under things. Low and medium: from the picture only; from high: ray traced.",
          [](E& e, int l) { e.rt_.ao_mode = pick(l, 0, 1, 1, 2, 2); e.rt_.ao_samples = pick(l, 1, 2, 4, 4, 8); } },
        { "bounce_light", "Bounce light", "Light that reaches a place by way of other surfaces: rays per pixel and bounces.",
          [](E& e, int l) { e.rt_.gi_samples = pick(l, 4, 8, 12, 24, 64); e.rt_.gi_bounces = pick(l, 1, 2, 3, 3, 3); e.rt_.terrain_gi_samples_cap = pick(l, 1, 1, 2, 3, 4); e.rt_.terrain_gi_bounces_cap = pick(l, 1, 1, 1, 2, 2); } },
        { "reflections", "Reflections", "Mirror-like surfaces and the water: what they reflect.\nLow: the sky only. From high: the world, ray traced.",
          [](E& e, int l) { e.rt_.reflections_enabled = l >= 1; e.rt_.water_rt_reflections = l >= 2; e.rt_.water_tlas_reflections = l >= 3; e.rt_.reflection_strength = pick(l, 0.3f, 0.4f, 0.5f, 0.5f, 0.6f); } },
        { "light_in_air", "Light in the air", "Mist, shafts of light and the glow of fog in valleys.",
          [](E& e, int l) { e.rt_.fog_volumetric = l >= 1 ? 1 : 0; e.rt_.fog_max_dist = pick(l, 120.0f, 200.0f, 300.0f, 400.0f, 600.0f); e.rt_.terrain_raymarch_fog_godrays = l >= 3; } },
        // ---- what grows ----
        { "forest", "Forest density", "How many trees and bushes stand. The forest is planted again when this changes (about a second).",
          [](E& e, int l) { const float f = pick(l, 0.45f, 0.75f, 1.0f, 1.4f, 1.8f), b = pick(l, 0.35f, 0.65f, 1.0f, 1.3f, 1.6f);
                            if (e.rt_.forest_density != f || e.rt_.bush_density != b) e.veg_replant_request_ = true;
                            e.rt_.forest_density = f; e.rt_.bush_density = b; } },
        { "tree_detail", "Tree detail", "How far out a tree keeps all its branches and leaves before a simpler one stands in.",
          [](E& e, int l) { e.rt_.veg_lod_bias = pick(l, 0.45f, 0.6f, 0.8f, 1.0f, 1.5f); e.rt_.tree_impostor_px = pick(l, 64.0f, 56.0f, 44.0f, 36.0f, 24.0f); } },
        { "tree_distance", "Tree draw distance", "How far away trees are drawn at all.",
          [](E& e, int l) { e.rt_.tree_draw_distance = pick(l, 900.0f, 1600.0f, 2600.0f, 3800.0f, 3800.0f); } },
        { "plants", "Small plants", "Flowers, ferns, weeds and reeds: how many, and how far they are drawn.",
          [](E& e, int l) { e.rt_.flower_density = pick(l, 0.3f, 0.6f, 1.0f, 1.35f, 2.0f); e.rt_.ground_density = pick(l, 0.3f, 0.6f, 1.0f, 1.4f, 2.2f);
                            e.rt_.flower_draw_distance = pick(l, 22.0f, 32.0f, 44.0f, 70.0f, 140.0f); e.rt_.ground_draw_distance = pick(l, 20.0f, 30.0f, 42.0f, 60.0f, 120.0f);
                            e.rt_.reed_draw_distance = pick(l, 60.0f, 100.0f, 160.0f, 220.0f, 300.0f); } },
        { "grass_density", "Grass density", "How thick the grass stands, and the extra blades close to the camera.",
          [](E& e, int l) { e.rt_.grass_density = pick(l, 0.6f, 0.85f, 1.0f, 1.15f, 1.5f); e.rt_.grass_near_boost = pick(l, 0.2f, 0.5f, 0.8f, 1.0f, 1.5f); } },
        { "grass_distance", "Grass distance", "How far from the camera real blades of grass are drawn; beyond it the ground is coloured as grass.",
          [](E& e, int l) { e.rt_.grass_raymarch_distance = pick(l, 60.0f, 100.0f, 150.0f, 220.0f, 350.0f); e.rt_.grass_distance = pick(l, 35.0f, 50.0f, 62.0f, 90.0f, 140.0f); } },
        { "grass_shadows", "Grass shadows", "Shadows between blades of grass, and the grass's shadow on the ground.",
          [](E& e, int l) { e.rt_.grass_shadow_samples = pick(l, 0, 1, 2, 4, 6); e.rt_.grass_shadow_on_terrain = l >= 1; e.rt_.grass_shadow_on_terrain_samples = pick(l, 0, 2, 3, 5, 8); } },
        // ---- water, sky, lens ----
        { "water", "Water detail", "The fineness of waves on rivers and lakes, and whether shadows fall on the water.",
          [](E& e, int l) { e.rt_.water_river_detail = pick(l, 0.4f, 0.7f, 1.0f, 1.3f, 1.6f); e.rt_.water_shadows_enabled = l >= 1; e.rt_.water_lake_bump_dist = pick(l, 25.0f, 40.0f, 60.0f, 90.0f, 140.0f); } },
        { "clouds", "Clouds", "Low: a painted sky. From medium: clouds with depth, lit by the sun.",
          [](E& e, int l) { e.rt_.clouds_enabled = l >= 1 ? 1 : 0; } },
        { "lens", "Lens effects", "Bloom round bright lights, lens flares and the sun's shafts.",
          [](E& e, int l) { e.rt_.bloom_enabled = l >= 1; e.rt_.bloom_radius = pick(l, 12.0f, 16.0f, 24.0f, 28.0f, 32.0f); e.rt_.lens_flare_enabled = l >= 2; e.rt_.lens_flare_ghosts = pick(l, 0, 0, 3, 4, 5); } },
    };
    return rows;
}

void VulkanEngine::apply_gfx_row(int row, int level) {
    const auto& rows = gfx_rows();
    if (row < 0 || row >= int(rows.size())) return;
    level = std::clamp(level, 0, 4);
    rt_.gfx[row] = level;
    rows[size_t(row)].apply(*this, level);
    rt_.quality_preset = -1;
    rt_.foliage_quality = rt_.shadow_quality = -1;
    if (gfx_scale_changed_) { gfx_scale_changed_ = false; if (swapchain_) apply_render_scale(); }
    if (gfx_terrain_scale_changed_) { gfx_terrain_scale_changed_ = false; if (swapchain_ && tr_ && tr_->lr_color_image) recreate_terrain_raymarch_lowres(); }      // (before the targets exist - settings being loaded - they are simply made at this size)
}

void VulkanEngine::apply_gfx_all(int level) {
    const int n = int(gfx_rows().size());
    for (int i = 0; i < n; ++i) apply_gfx_row(i, level);
    rt_.gfx_all = std::clamp(level, 0, 4);
    log::infof("[settings] graphics quality: all rows to level %d", rt_.gfx_all);
}

// "Shading test" (Graphics tab, at the top; VIBE_TEST_SHADE): measuring switches, not quality
// levels. One drop-down per part of the picture - the ground's light, its stones, its close-up
// detail, the water, the objects' light, the grass, the trees, the sky and air, the denoiser -
// each from "off" or "flat" up to "full", so that a part can be taken out or made plain while
// the rest stays as it is, and the frame time read with and without it.
//
// The player's settings are put aside when the first drop-down leaves "full" and come back
// when the last one returns to it; nothing is saved while any is below "full"
// (save_settings). Every change starts again from the settings put aside, so the drop-downs
// do not depend on the order they are used in.
//
// What a level does is one of three things: a bit for the shaders (scene.rt_flags2.y,
// scene_ubo.glsl FLAT_*: the part is its own colour times one light), sections of the terrain
// shader left out (shade_rm_skip_, the RMDBG bits), or settings switched off.
const VulkanEngine::ShadeRow* VulkanEngine::shade_rows(int& count) {
    static const char* const kGround[]  = { "Flat (colour only)", "Simple (sun and sky, no detail)", "No shadows or bounce light", "Full" };
    static const char* const kStones[]  = { "Off", "Simple (in the light only, not in the shape)", "Full" };
    static const char* const kDetail[]  = { "Off (no close-up textures, no relief)", "Full" };
    static const char* const kWater[]   = { "Flat", "Simple (no ray-traced reflections, no shadows)", "Full" };
    static const char* const kObjects[] = { "Flat (colour only)", "Sun and sky only (no rays)", "Full" };
    static const char* const kGrass[]   = { "Off", "Flat (colour only)", "No shadows", "Full" };
    static const char* const kTrees[]   = { "Off", "Full" };
    static const char* const kSky[]     = { "Off (no clouds, mist, bloom, flares)", "Full" };
    static const char* const kDenoise[] = { "Off", "On" };
    static const ShadeRow rows[kShadeRows] = {
        { "ground",   "Ground: light",      kGround,  4, "How the ground is lit.\nFlat and Simple use the small first-start shader: no textures, stones or relief either.\nNo shadows or bounce light: the full ground, without its own shadow, the shadows of grass,\nplants and objects on it, and without bounce light." },
        { "stones",   "Ground: stones",     kStones,  3, "The pebbles and stones lying on the ground close by.\nSimple: they are in the light and the colour, but do not change the ground's outline." },
        { "detail",   "Ground: close-up",   kDetail,  2, "The photographed soil and rock textures close to the camera, and the fine relief\n(rills, ripples in sand, lumps)." },
        { "water",    "Water",              kWater,   3, "Flat: one colour and the sky's sheen. Simple: waves and foam, but what it reflects is the sky only." },
        { "objects",  "Objects: light",     kObjects, 3, "Castle, buildings, trees, cars, small plants.\nSun and sky only: no ray-traced shadows, occlusion, bounce light or reflections\n(this also takes the objects' shadows off the ground and the grass)." },
        { "grass",    "Grass",              kGrass,   4, "The blades of grass. Off: none are drawn (the ground stays coloured as lawn)." },
        { "trees",    "Trees",              kTrees,   2, "Off: no trees or bushes are drawn (their ray-traced shadows stay)." },
        { "sky",      "Sky and air",        kSky,     2, "Clouds, mist and light in the air, bloom and lens flares." },
        { "denoiser", "Denoiser (DLSS RR)", kDenoise, 2, "Ray Reconstruction. Off: the plain DLSS upscaler." },
    };
    count = kShadeRows;
    return rows;
}

void VulkanEngine::apply_shade_test() {
    int n = 0;
    const ShadeRow* rows = shade_rows(n);
    bool any = false;
    for (int i = 0; i < n; ++i) {
        shade_test_[i] = std::clamp(shade_test_[i], 0, rows[i].levels - 1);
        any = any || shade_test_[i] != rows[i].levels - 1;
    }
    if (any && !simple_shading_) simple_saved_rt_ = rt_;        // the player's own settings: put aside
    if (any || simple_shading_) {
        // (from what was put aside - but not what is not shading and has moved on meanwhile)
        RtSettings base = simple_saved_rt_;
        base.time_of_day = rt_.time_of_day;
        rt_ = base;
    }
    shade_rm_skip_ = 0;
    shade_flat_bits_ = 0;
    if (simple_shading_ != any) log::infof("[settings] shading test %s", any ? "ON (measuring: settings are not saved)" : "off");
    simple_shading_ = any;
    if (!any) return;

    const int ground = shade_test_[0], stones = shade_test_[1], detail = shade_test_[2], water = shade_test_[3],
              objects = shade_test_[4], grass = shade_test_[5], trees = shade_test_[6], sky = shade_test_[7], denoiser = shade_test_[8];
    // -- the ground (levels 0 and 1 draw it with the first-start shader: world_render.cpp) --
    if (ground == 0) shade_flat_bits_ |= 8;
    if (ground <= 2) { shade_rm_skip_ |= 4 | 8 | 16 | 32; rt_.terrain_gi_samples_cap = 0; rt_.grass_shadow_on_terrain = false; }
    if (stones == 0) shade_rm_skip_ |= 2 | 32768;
    if (stones == 1) shade_rm_skip_ |= 32768;
    if (detail == 0) { shade_rm_skip_ |= 1 | 128 | 1024; rt_.spom_strength = 0.0f; }
    // -- water --
    if (water == 0) shade_rm_skip_ |= 131072;
    if (water <= 1) { rt_.water_rt_reflections = false; rt_.water_tlas_reflections = false; rt_.water_shadows_enabled = false; }
    // -- objects --
    if (objects == 0) shade_flat_bits_ |= 2;
    if (objects <= 1) { rt_.shadow_enabled = false; rt_.ao_mode = 0; rt_.ao_samples = 0; rt_.gi_samples = 0; rt_.reflections_enabled = false; }
    // -- grass --
    if (grass == 0) { rt_.grass_enabled = false; rt_.grass_raymarch_enabled = false; }
    if (grass == 1) shade_flat_bits_ |= 4;
    if (grass <= 2) { rt_.grass_shadow_samples = 0; rt_.grass_shadow_on_terrain = false; rt_.grass_shadow_on_terrain_samples = 0; }
    // -- trees --
    if (trees == 0) rt_.tree_draw_distance = 0.0f;
    // -- sky and air --
    if (sky == 0) {
        rt_.fog_volumetric = 0; rt_.terrain_raymarch_fog_strength = 0.0f; rt_.terrain_raymarch_fog_godrays = false;
        rt_.clouds_enabled = 0; rt_.bloom_enabled = false; rt_.lens_flare_enabled = false;
    }
    if (denoiser == 0) rt_.dlss_rr = false;
}

// Everything at its plainest (but drawn: the grass flat, the trees there), or everything full.
void VulkanEngine::set_simple_shading(bool on) {
    int n = 0;
    const ShadeRow* rows = shade_rows(n);
    for (int i = 0; i < n; ++i) shade_test_[i] = on ? 0 : rows[i].levels - 1;
    if (on) { shade_test_[5] = 1; shade_test_[6] = 1; }
    apply_shade_test();
}

// VIBE_TEST_SHADE="ground=0,water=1,grass=0": levels by key (measurements). VIBE_TEST_SIMPLE=1: all plainest.
void VulkanEngine::shade_test_hook() {
    if (std::getenv("VIBE_TEST_SIMPLE")) set_simple_shading(true);
    const char* e = std::getenv("VIBE_TEST_SHADE");
    if (!e) return;
    int n = 0;
    const ShadeRow* rows = shade_rows(n);
    std::string s(e);
    size_t at = 0;
    while (at < s.size()) {
        size_t end = s.find(',', at);
        if (end == std::string::npos) end = s.size();
        const std::string item = s.substr(at, end - at);
        at = end + 1;
        const size_t eq = item.find('=');
        if (eq == std::string::npos) continue;
        for (int i = 0; i < n; ++i)
            if (item.substr(0, eq) == rows[i].key) shade_test_[i] = std::atoi(item.c_str() + eq + 1);
    }
    apply_shade_test();
}

// VIBE_TEST_GFX="all=2,terrain_steps=0,forest=4": levels by key, in order (measurements).
void VulkanEngine::gfx_test_hook() {
    shade_test_hook();
    const char* e = std::getenv("VIBE_TEST_GFX");
    if (!e) return;
    const auto& rows = gfx_rows();
    std::string s(e);
    size_t at = 0;
    while (at < s.size()) {
        size_t end = s.find(',', at);
        if (end == std::string::npos) end = s.size();
        const std::string item = s.substr(at, end - at);
        at = end + 1;
        const size_t eq = item.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = item.substr(0, eq);
        const int level = std::atoi(item.c_str() + eq + 1);
        if (key == "all") { apply_gfx_all(level); continue; }
        bool found = false;
        for (size_t i = 0; i < rows.size(); ++i)
            if (key == rows[i].key) { apply_gfx_row(int(i), level); found = true; }
        if (!found) log::warnf("[settings] VIBE_TEST_GFX: no row '%s'", key.c_str());
    }
}

// The panel: first in the settings window.
void VulkanEngine::gfx_quality_ui() {
    static const char* kLevel[5] = { "Low", "Medium", "High", "Ultra", "Insane" };
    const auto& rows = gfx_rows();
    ImGui::SeparatorText("Graphics quality");
    ImGui::TextUnformatted("Everything");
    for (int l = 0; l < 5; ++l) {
        ImGui::SameLine();
        if (ImGui::RadioButton((std::string(kLevel[l]) + "##gfx_all").c_str(), rt_.gfx_all == l)) apply_gfx_all(l);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sets every row below to this level. High is what the game is tuned for.");
    {
        static const int   kFps[]     = { 0, 45, 60, 75, 90, 120 };
        static const char* kFpsName[] = { "Off", "45", "60", "75", "90", "120" };
        int cur = 0;
        for (int i = 0; i < 6; ++i) if (kFps[i] == rt_.auto_fps) cur = i;
        ImGui::SetNextItemWidth(110.0f);
        if (ImGui::Combo("Hold frame rate", &cur, kFpsName, 6)) rt_.auto_fps = kFps[cur];
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rendered frames a second to hold (before frame generation doubles them).\nThe render resolution then follows the load: lower in heavy places, higher where there is room,\nin steps between 50 %% and 100 %%. Each step is a short hitch, so it waits a few seconds before it moves.");
        if (rt_.auto_fps > 0) { ImGui::SameLine(); ImGui::Text("render resolution now %.0f %%", rt_.render_scale * 100.0f); }
    }
    if (ImGui::TreeNodeEx("Each one by itself", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("##gfx_rows", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthStretch, 0.55f);
            ImGui::TableSetupColumn("level", ImGuiTableColumnFlags_WidthStretch, 0.45f);
            for (size_t i = 0; i < rows.size(); ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(rows[i].label);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", rows[i].tip);
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(-FLT_MIN);
                int cur = rt_.gfx[i];
                const char* shown = (cur >= 0 && cur < 5) ? kLevel[cur] : "(set by hand)";
                if (ImGui::BeginCombo((std::string("##gfx_") + rows[i].key).c_str(), shown)) {
                    for (int l = 0; l < 5; ++l)
                        if (ImGui::Selectable(kLevel[l], cur == l)) { apply_gfx_row(int(i), l); rt_.gfx_all = -1; }
                    ImGui::EndCombo();
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", rows[i].tip);
            }
            ImGui::EndTable();
        }
        ImGui::TreePop();
    }
}

}  // namespace vibe

namespace vibe {

// ---- Settings 'hold frame rate': the render scale follows the GPU's load ----
// The whole frame costs by the pixel (a tenth of render scale is 3 - 6 ms at 3440 x 1440), so
// the scale is the one lever that holds a frame rate everywhere. It moves in steps: every
// change rebuilds the render targets (a hitch of a tenth of a second, the upscaler's history
// starts again), so it steps down only after 1.5 s over the budget, up only after 4 s with
// room for the next step, and then waits 6 s. The load is the profiler's sum of the GPU
// passes plus frame generation (which has no mark: about 4 ms at 3440 x 1440) - not the frame
// time, which a frame-rate cap or v-sync holds up whatever the load is.
// VIBE_TEST_AUTO_FPS=<n> turns it on for a test run.
void VulkanEngine::auto_scale_tick() {
    static const int test_fps = [] { const char* e = std::getenv("VIBE_TEST_AUTO_FPS"); return e ? std::atoi(e) : 0; }();
    const int target = test_fps > 0 ? test_fps : rt_.auto_fps;
    static double over_t = 0.0, under_t = 0.0, cool_t = 0.0;
    if (target <= 0 || state_ != State::Playing || prof_ema_count_ == 0) { over_t = under_t = 0.0; return; }
    static const float kScale[] = { 0.50f, 0.55f, 0.60f, 0.67f, 0.75f, 0.85f, 1.00f };
    const int kN = 7;
    const double dt = std::clamp(double(last_frame_dt_), 0.0, 0.1);
    if (cool_t > 0.0) { cool_t -= dt; over_t = under_t = 0.0; return; }
    float gpu = 0.0f;
    for (uint32_t k = 0; k < prof_ema_count_; ++k) gpu += prof_ema_[k].ms;
    if (dlssg_tagged_this_frame_)
        gpu += 4.0f * float(swapchain_extent_.width) * float(swapchain_extent_.height) / (3440.0f * 1440.0f);
    const float budget = 1000.0f / float(target);
    int cur = 0;
    for (int i = 1; i < kN; ++i)
        if (std::abs(kScale[i] - rt_.render_scale) < std::abs(kScale[cur] - rt_.render_scale)) cur = i;
    // (what the next step up would cost: three quarters of the frame grow with the pixel count)
    float up_ms = gpu;
    if (cur + 1 < kN) { const float r = kScale[cur + 1] / kScale[cur]; up_ms = gpu * (1.0f + 0.75f * (r * r - 1.0f)); }
    over_t  = gpu > budget * 1.06f ? over_t + dt : 0.0;
    under_t = (cur + 1 < kN && up_ms < budget * 0.92f) ? under_t + dt : 0.0;
    int next = cur;
    if (over_t > 1.5 && cur > 0) {
        // (straight to the highest step that should fit, not one step at a time: far over the
        // budget that took four hitches and half a minute)
        next = 0;
        for (int i = cur - 1; i > 0; --i) {
            const float r = kScale[i] / kScale[cur];
            if (gpu * (1.0f + 0.75f * (r * r - 1.0f)) <= budget) { next = i; break; }
        }
    }
    else if (under_t > 4.0) next = cur + 1;
    if (next == cur) return;
    log::infof("[auto-scale] %.1f ms against %.1f (%d fps): render scale %.2f -> %.2f", gpu, budget, target, rt_.render_scale, kScale[next]);
    rt_.render_scale = kScale[next];
    apply_render_scale();
    cool_t = 6.0; over_t = under_t = 0.0;
}

// VIBE_PROF_SWEEP=<frames per step>: with the camera held, the game steps through a list of
// experiments - each turns one thing off or down from the settings it started with - and logs
// the GPU time of every pass for each ("[prof] <label>" then a "gpu-passes" line). Subtracting
// a line from the first one ("base") says what that thing costs in this view. One run does
// what took an hour of restarts. (Run it on a COPY of the settings: VIBE_CFG.)
void VulkanEngine::prof_sweep_tick() {
    // VIBE_PROF_ALWAYS=1: the GPU pass times once a second in every state (the menu too).
    { static const bool always = std::getenv("VIBE_PROF_ALWAYS") != nullptr; if (always && frame_number_ % 60 == 0) prof_log(); }
    // VIBE_TEST_GFX_SEQ="400:4,1000:3,1600:2": at those frames, every row to that level - what the
    // panel's "Everything" buttons do, while the game runs (resize, replant and all).
    {
        static const char* seq = std::getenv("VIBE_TEST_GFX_SEQ");
        if (seq) {
            for (const char* p = seq; *p; ) {
                int fr = 0, lv = 0;
                if (std::sscanf(p, "%d:%d", &fr, &lv) == 2 && uint64_t(fr) == uint64_t(frame_number_)) apply_gfx_all(lv);
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
    }
    static const char* on = std::getenv("VIBE_PROF_SWEEP");
    if (!on) return;
    using R = RtSettings;
    struct Exp { const char* label; int skip; void (*set)(R&); };
    static const Exp ex[] = {
        { "base", 0, [](R&) {} },
        // -- the terrain shader: sections (RMDBG bits) --
        { "terrain: march only (no shading)", 64, [](R&) {} },
        { "terrain: no relief in the march", 128, [](R&) {} },
        { "terrain: no relief at all", 1024, [](R&) {} },
        { "terrain: no stones in the march", 32768, [](R&) {} },
        { "terrain: section 1 off", 1, [](R&) {} },
        { "terrain: close-up detail off (2)", 2, [](R&) {} },
        { "terrain: own sun shadow off (4)", 4, [](R&) {} },
        { "terrain: grass blade shadows off (8)", 8, [](R&) {} },
        { "terrain: plant shadow map off (16)", 16, [](R&) {} },
        { "terrain: ray-traced object shadows off (32)", 32, [](R&) {} },
        { "terrain: screen-space normal (512)", 512, [](R&) {} },
        { "terrain: all shadows off (4+8+16+32)", 60, [](R&) {} },
        // -- the terrain shader: settings --
        { "terrain: no first pass", 0, [](R& r) { r.terrain_rm_prepass = false; } },
        { "terrain: first pass 1/3", 0, [](R& r) { r.terrain_rm_prepass_div = 3; } },
        { "terrain: first pass 1/4", 0, [](R& r) { r.terrain_rm_prepass_div = 4; } },
        { "terrain: edges coarser (vrs 1)", 0, [](R& r) { r.terrain_vrs = 1; } },
        { "terrain: edges coarsest (vrs 2)", 0, [](R& r) { r.terrain_vrs = 2; } },
        { "terrain: max steps 110", 0, [](R& r) { r.terrain_raymarch_max_steps = 110; } },
        { "terrain: step factor 0.8", 0, [](R& r) { r.terrain_raymarch_step_factor = 0.8f; } },
        { "terrain: octaves 6", 0, [](R& r) { r.terrain_raymarch_octaves = 6; } },
        { "terrain: normal octaves 8", 0, [](R& r) { r.terrain_raymarch_normal_octaves = 8; } },
        { "terrain: detail distance short", 0, [](R& r) { r.terrain_raymarch_lod_near_m = 40.0f; r.terrain_raymarch_lod_far_m = 400.0f; r.terrain_raymarch_lod_min_octaves = 4; } },
        { "terrain: normal 3 px", 0, [](R& r) { r.terrain_rm_normal_px = 3.0f; } },
        { "terrain: relief full (lite off)", 0, [](R& r) { r.terrain_rm_relief_lite = false; } },
        { "terrain: bounce light 0", 0, [](R& r) { r.terrain_gi_samples_cap = 0; } },
        { "terrain: soft shadow rays 1", 0, [](R& r) { r.terrain_pcss_samples_cap = 1; } },
        { "terrain: shadow steps 8", 0, [](R& r) { r.terrain_raymarch_shadow_steps = 8; } },
        { "terrain: resolution 75%", 0, [](R& r) { r.terrain_raymarch_scale = 0.75f; } },
        { "terrain: resolution 50%", 0, [](R& r) { r.terrain_raymarch_scale = 0.5f; } },
        { "terrain: resolution 50% + temporal", 0, [](R& r) { r.terrain_raymarch_scale = 0.5f; r.terrain_rm_temporal = true; } },
        { "terrain: pom off", 0, [](R& r) { r.terrain_pom_strength = 0.0f; r.spom_strength = 0.0f; } },
        { "terrain: fog off", 0, [](R& r) { r.terrain_fog_strength = 0.0f; r.terrain_raymarch_fog_strength = 0.0f; r.distance_fog_strength = 0.0f; } },
        // -- resolution --
        { "render scale 0.42", 0, [](R& r) { r.render_scale = 0.42f; } },
        { "render scale 0.59", 0, [](R& r) { r.render_scale = 0.59f; } },
        { "render scale 0.67", 0, [](R& r) { r.render_scale = 0.67f; } },
        // -- what grows --
        { "trees: detail near only (lod 0.45)", 0, [](R& r) { r.veg_lod_bias = 0.45f; } },
        { "trees: detail far (lod 1.5)", 0, [](R& r) { r.veg_lod_bias = 1.5f; } },
        { "trees: impostors late (10 px)", 0, [](R& r) { r.tree_impostor_px = 10.0f; } },
        { "trees: draw distance 900", 0, [](R& r) { r.tree_draw_distance = 900.0f; } },
        { "plants: draw distance 20", 0, [](R& r) { r.flower_draw_distance = 20.0f; r.ground_draw_distance = 20.0f; r.reed_draw_distance = 40.0f; } },
        { "grass: off", 0, [](R& r) { r.grass_raymarch_enabled = false; r.grass_enabled = false; } },
        { "grass: distance 60", 0, [](R& r) { r.grass_raymarch_distance = 60.0f; } },
        { "grass: distance 350", 0, [](R& r) { r.grass_raymarch_distance = 350.0f; } },
        { "grass: density 0.6", 0, [](R& r) { r.grass_density = 0.6f; r.grass_near_boost = 0.2f; } },
        { "grass: shadows off", 0, [](R& r) { r.grass_shadow_samples = 0; r.grass_shadow_on_terrain = false; } },
        // -- light --
        { "light: sun shadows off", 0, [](R& r) { r.shadow_enabled = false; } },
        { "light: sun shadow rays 2", 0, [](R& r) { r.shadow_samples = 2; } },
        { "light: bounce light off", 0, [](R& r) { r.gi_samples = 0; r.terrain_gi_samples_cap = 0; } },
        { "light: ambient occlusion off", 0, [](R& r) { r.ao_mode = 0; } },
        { "light: ambient occlusion ray traced", 0, [](R& r) { r.ao_mode = 2; } },
        { "light: leaf cards solid (mode 2)", 0, [](R& r) { r.foliage_ray_mode = 2; } },
        { "light: every leaf tested (mode 0)", 0, [](R& r) { r.foliage_ray_mode = 0; } },
        { "light: reflections off", 0, [](R& r) { r.reflections_enabled = false; r.water_rt_reflections = false; r.water_tlas_reflections = false; } },
        { "light: radiance cache off", 0, [](R& r) { r.radiance_cache = 0; } },
        // -- sky, air, lens, water --
        { "sky: clouds off", 0, [](R& r) { r.clouds_enabled = 0; } },
        { "air: volumetric fog off", 0, [](R& r) { r.fog_volumetric = 0; } },
        { "lens: bloom and flares off", 0, [](R& r) { r.bloom_enabled = false; r.lens_flare_enabled = false; } },
        { "water: off", 0, [](R& r) { r.water_enabled = false; } },
        { "base again", 0, [](R&) {} },
    };
    static const int kN = int(std::size(ex));
    static const uint64_t kSeg = uint64_t(std::max(60, std::atoi(on)));
    static const uint64_t kWarm = 700;
    static R base;
    static int cur = -1;
    static uint64_t seg_start = 0;
    static double cpu_ms = 0.0;
    static uint64_t cpu_n = 0;
    if (frame_number_ < kWarm) return;
    cpu_ms += double(last_frame_dt_) * 1000.0; ++cpu_n;
    if (cur >= 0 && frame_number_ - seg_start < kSeg) return;
    if (cur >= kN) return;
    if (cur >= 0) {
        log::infof("[prof] %s | frame %.2f ms", ex[cur].label, cpu_n ? cpu_ms / double(cpu_n) : 0.0);
        prof_log();
    } else {
        base = rt_;
    }
    ++cur;
    if (cur >= kN) { log::infof("[prof] done"); rt_ = base; prof_rm_skip_ = 0; opts_.max_frames = int(frame_number_) + 2; return; }
    R next = base;
    ex[cur].set(next);
    const bool rs = std::abs(next.render_scale - rt_.render_scale) > 1e-3f;
    const bool ts = std::abs(next.terrain_raymarch_scale - rt_.terrain_raymarch_scale) > 1e-3f;
    rt_ = next;
    prof_rm_skip_ = ex[cur].skip;
    if (rs) apply_render_scale();
    if (ts) recreate_terrain_raymarch_lowres();
    for (uint32_t k = 0; k < prof_ema_count_; ++k) prof_ema_[k].ms = 0.0f;
    cpu_ms = 0.0; cpu_n = 0;
    seg_start = frame_number_;
}

}  // namespace vibe
