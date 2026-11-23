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
        { "terrain_edges", "Terrain shading at screen edges", "Shades the ground more coarsely toward the edges of the screen, where the eye does not look.",
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

// VIBE_TEST_GFX="all=2,terrain_steps=0,forest=4": levels by key, in order (measurements).
void VulkanEngine::gfx_test_hook() {
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

// VIBE_PROF_SWEEP=<frames per step>: with the camera held, the game steps through a list of
// experiments - each turns one thing off or down from the settings it started with - and logs
// the GPU time of every pass for each ("[prof] <label>" then a "gpu-passes" line). Subtracting
// a line from the first one ("base") says what that thing costs in this view. One run does
// what took an hour of restarts. (Run it on a COPY of the settings: VIBE_CFG.)
void VulkanEngine::prof_sweep_tick() {
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
        { "terrain: far section off (512)", 512, [](R&) {} },
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
