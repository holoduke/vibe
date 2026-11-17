// Painted vegetation + the edit-mode tool panel.
//
// Workflow: open edit mode (E), pick the Paint tool, choose or make a paint
// preset in the panel (trees / bushes / grass, each optional, with its own
// sliders), then paint with it. The right mouse button erases the layers
// the active preset uses (back to the automatic world). See veg_paint.h for
// the data model.
//
// Painting only edits the paint map. The world catches up when a stroke
// ends (or on "Apply"): trees and bushes are re-planted (instances, player
// colliders, physics compounds, GPU buffers), grass tufts by rebuilding the
// small-plant scatter. Both are deterministic per location, so only the
// painted area changes.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/log.h"
#include "engine/veg_paint.h"
#include "engine/window.h"
#include "engine/world_props.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace qlike {

namespace {
const char* const kPaintFile = "assets/level1_paint.bin";
const char* const kOldZoneFile = "assets/level1_veg_zones.bin";
constexpr uint32_t kPaintMagic = 0x314E5051u;   // "QPN1"

const char* const kToolNames[] = { "Raise", "Lower", "Smooth", "Flatten", "Grass +", "Grass -",
                                   "Erode +", "Erode smooth", "Paint" };

PaintPreset make_preset(const char* name, bool trees, float td, float pine, bool bushes, float bd,
                        bool grass, float gd, float gh, float dry) {
    PaintPreset p{};
    std::snprintf(p.name, sizeof(p.name), "%s", name);
    p.trees = trees; p.tree_density = td; p.pine_share = pine;
    p.bushes = bushes; p.bush_density = bd;
    p.grass = grass; p.grass_density = gd; p.grass_height = gh; p.grass_dry = dry;
    return p;
}

std::vector<PaintPreset> default_presets() {
    std::vector<PaintPreset> v;
    v.push_back(make_preset("(automatic)", false, 0, 0, false, 0, false, 0, 1, 0));
    //                       name               trees dens  pine  bushes dens  grass dens height dry
    v.push_back(make_preset("Pine forest",      true, 1.0f, 1.0f, true, 0.35f, false, 1.0f, 1.0f, 0.0f));
    v.push_back(make_preset("Broadleaf forest", true, 1.0f, 0.0f, true, 0.35f, false, 1.0f, 1.0f, 0.0f));
    v.push_back(make_preset("Mixed forest",     true, 1.0f, 0.5f, true, 0.35f, false, 1.0f, 1.0f, 0.0f));
    v.push_back(make_preset("Meadow",           true, 0.0f, 0.5f, true, 0.03f, true,  1.0f, 1.0f, 0.1f));
    v.push_back(make_preset("Bushland",         true, 0.0f, 0.5f, true, 0.90f, false, 1.0f, 1.0f, 0.0f));
    v.push_back(make_preset("Bare (nothing)",   true, 0.0f, 0.5f, true, 0.00f, true,  0.0f, 1.0f, 0.0f));
    v.push_back(make_preset("Tall grass",       false, 1.0f, 0.5f, false, 0.4f, true, 1.5f, 1.5f, 0.0f));
    v.push_back(make_preset("Dry grass",        false, 1.0f, 0.5f, false, 0.4f, true, 1.0f, 1.0f, 0.85f));
    v.push_back(make_preset("No grass",         false, 1.0f, 0.5f, false, 0.4f, true, 0.0f, 1.0f, 0.0f));
    return v;
}

ImU32 preset_color(int id) {
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB(std::fmod(0.08f + float(id) * 0.1618f, 1.0f), 0.70f, 0.95f, r, g, b);
    return IM_COL32(int(r * 255.0f), int(g * 255.0f), int(b * 255.0f), 255);
}
}  // namespace

// ---------------------------------------------------------------------
// Paint map: load / save / paint
// ---------------------------------------------------------------------

void VulkanEngine::load_paint() {
    const size_t cells = size_t(kPaintDim) * kPaintDim;
    for (auto& l : paint_.layer) l.assign(cells, 0);
    paint_.presets = default_presets();

    std::ifstream f(kPaintFile, std::ios::binary);
    if (f) {
        uint32_t magic = 0, dim = 0, n = 0, psize = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&dim), 4);
        f.read(reinterpret_cast<char*>(&n), 4);
        f.read(reinterpret_cast<char*>(&psize), 4);
        if (f && magic == kPaintMagic && dim == uint32_t(kPaintDim) && psize == sizeof(PaintPreset) &&
            n >= 1 && n <= uint32_t(kPaintMaxPresets)) {
            std::vector<PaintPreset> ps(n);
            f.read(reinterpret_cast<char*>(ps.data()), std::streamsize(n * sizeof(PaintPreset)));
            for (auto& l : paint_.layer) f.read(reinterpret_cast<char*>(l.data()), std::streamsize(cells));
            if (f) {
                for (auto& p : ps) p.name[sizeof(p.name) - 1] = 0;
                paint_.presets = std::move(ps);
                for (auto& l : paint_.layer)
                    for (auto& c : l) if (c >= paint_.presets.size()) c = 0;
                log::infof("[paint] loaded %s (%zu presets)", kPaintFile, paint_.presets.size() - 1);
                return;
            }
        }
        log::warnf("[paint] %s is not readable - starting empty", kPaintFile);
        for (auto& l : paint_.layer) l.assign(cells, 0);
        paint_.presets = default_presets();
        return;
    }
    // First run after the single-layer zone map: zones 1..6 are the first six
    // presets; they steered trees and bushes only.
    std::ifstream old(kOldZoneFile, std::ios::binary);
    if (old) {
        std::vector<uint8_t> z(cells);
        old.read(reinterpret_cast<char*>(z.data()), std::streamsize(cells));
        if (size_t(old.gcount()) == cells) {
            for (size_t i = 0; i < cells; ++i)
                if (z[i] >= 1 && z[i] <= 6) paint_.layer[kPaintTrees][i] = paint_.layer[kPaintBushes][i] = z[i];
            log::infof("[paint] imported the old zone map %s", kOldZoneFile);
        }
    }
}

bool VulkanEngine::save_paint() {
    std::ofstream f(kPaintFile, std::ios::binary);
    if (!f) { log::errorf("[paint] cannot write %s", kPaintFile); return false; }
    const uint32_t hdr[4] = { kPaintMagic, uint32_t(kPaintDim), uint32_t(paint_.presets.size()),
                              uint32_t(sizeof(PaintPreset)) };
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(paint_.presets.data()),
            std::streamsize(paint_.presets.size() * sizeof(PaintPreset)));
    for (const auto& l : paint_.layer) f.write(reinterpret_cast<const char*>(l.data()), std::streamsize(l.size()));
    log::infof("[paint] saved %s", kPaintFile);
    paint_unsaved_ = false;
    return bool(f);
}

// Stamps the active preset (or, erasing, "automatic") into the layers the
// active preset uses, within the brush radius of the brush hit.
void VulkanEngine::paint_veg(bool erase) {
    if (!terrain_brush_has_hit_ || paint_.empty()) return;
    if (paint_active_ < 1 || paint_active_ >= int(paint_.presets.size())) return;
    const PaintPreset& pp = paint_.presets[size_t(paint_active_)];
    const bool use[kPaintLayers] = { pp.trees != 0, pp.bushes != 0, pp.grass != 0 };
    const uint8_t id = erase ? uint8_t(0) : uint8_t(paint_active_);
    const float cell = PaintMap::cell();
    const float r = terrain_brush_radius_;
    const glm::vec3 c = terrain_brush_world_pos_;
    const int i0 = std::max(0, int(std::floor((c.x - r + 0.5f * kPaintSide) / cell)));
    const int i1 = std::min(kPaintDim - 1, int(std::floor((c.x + r + 0.5f * kPaintSide) / cell)));
    const int j0 = std::max(0, int(std::floor((c.z - r + 0.5f * kPaintSide) / cell)));
    const int j1 = std::min(kPaintDim - 1, int(std::floor((c.z + r + 0.5f * kPaintSide) / cell)));
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i) {
            const float x = (float(i) + 0.5f) * cell - 0.5f * kPaintSide - c.x;
            const float z = (float(j) + 0.5f) * cell - 0.5f * kPaintSide - c.z;
            if (x * x + z * z > r * r) continue;
            const size_t k = size_t(j) * kPaintDim + size_t(i);
            for (int l = 0; l < kPaintLayers; ++l) {
                if (!use[l] || paint_.layer[l][k] == id) continue;
                paint_.layer[l][k] = id;
                paint_unsaved_ = true;
                if (l == kPaintGrass) paint_dirty_grass_ = true; else paint_dirty_veg_ = true;
            }
        }
}

// Queue whatever the painting / preset edits made stale.
void VulkanEngine::request_paint_apply(bool everything) {
    if (paint_dirty_veg_ || everything) veg_replant_request_ = true;
    if (paint_dirty_grass_ || everything) scatter_rebuild_request_ = true;
}

WorldPropParams VulkanEngine::veg_place_params() const {
    WorldPropParams wp{};
    wp.water_level  = rt_.water_level;
    wp.tree_density = glm::clamp(rt_.forest_density, 0.0f, 2.0f);
    wp.bush_density = glm::clamp(rt_.bush_density, 0.0f, 2.0f);
    wp.max_trees    = int(float(wp.max_trees) * std::max(1.0f, wp.tree_density));
    wp.max_bushes   = int(float(wp.max_bushes) * std::max(1.0f, wp.bush_density));
    wp.rock_band_lo = rt_.terrain_h_dirt_rock_start;
    for (const auto& a : aa_) wp.keep_out.push_back(glm::vec3(a.pos.x, a.pos.z, 9.0f));
    for (const auto& o : outposts_) wp.keep_out.push_back(glm::vec3(o.x, o.z, 10.0f));
    if (church_placed_)
        for (float dx : { -5.0f, 0.0f, 5.0f }) {
            wp.keep_out.push_back(glm::vec3(church_pos_.x + dx, church_pos_.z, 8.5f));
            wp.keep_out.push_back(glm::vec3(church_pos_.x + dx, church_pos_.z + 10.0f, 7.0f));    // the graveyard
        }
    for (const auto& l : lamp_lights_) wp.keep_out.push_back(glm::vec3(l.x, l.z, 2.5f));
    if (!paint_.empty()) wp.paint = &paint_;
    return wp;
}

// ---------------------------------------------------------------------
// Catch-up: trees + bushes
// ---------------------------------------------------------------------

void VulkanEngine::start_veg_replant() {
    if (veg_job_active_ || veg_arch_.empty() || terrain_data_.heights.empty() || !veg_inst_buf_) return;
    veg_job_hm_.dim = terrain_data_.dim;
    veg_job_hm_.cell = terrain_data_.cell;
    veg_job_hm_.origin_x = terrain_data_.origin_x;
    veg_job_hm_.origin_z = terrain_data_.origin_z;
    veg_job_hm_.heights = terrain_data_.heights;
    veg_job_paint_ = paint_;                       // snapshot: painting may go on meanwhile
    WorldPropParams wp = veg_place_params();
    if (wp.paint) wp.paint = &veg_job_paint_;
    paint_dirty_veg_ = false;
    veg_job_active_ = true;
    veg_job_ = std::async(std::launch::async, [this, wp]() {
        const auto t0 = std::chrono::steady_clock::now();
        VegJobResult r;
        r.inst = place_vegetation(veg_job_hm_, &castle_road_, wp, veg_arch_, r.cols, r.st);
        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    });
}

void VulkanEngine::replant_vegetation() {
    // Synchronous variant (level load, presets): start and wait.
    if (!veg_job_active_) start_veg_replant();
    if (!veg_job_active_) return;
    VegJobResult r = veg_job_.get();
    veg_job_active_ = false;
    finish_veg_replant(std::move(r));
}

void VulkanEngine::finish_veg_replant(VegJobResult&& r) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<collision::AABB> cols = std::move(r.cols);
    WorldPropStats st = r.st;
    std::vector<VegInstance> inst = std::move(r.inst);
    const double place_ms = r.ms;

    vkDeviceWaitIdle(device_);

    // Player colliders: the vegetation block is the tail of world_.aabbs.
    if (veg_col_begin_ <= world_.aabbs.size()) world_.aabbs.resize(veg_col_begin_);
    for (const auto& a : cols) world_.aabbs.push_back(a);
    veg_col_count_ = cols.size();
    near_static_src_[0] = SIZE_MAX;            // refresh the player broadphase now
    // Physics (bullets, bombs, crates).
    if (physics_) {
        physics_->clear_static_box_compounds();
        std::vector<PhysicsWorld::StaticBox> vb;
        vb.reserve(cols.size());
        for (const auto& a : cols) vb.push_back({ (a.min + a.max) * 0.5f, (a.max - a.min) * 0.5f });
        physics_->add_static_box_compounds(vb.data(), vb.size());
    }

    // GPU side: the visible lists are laid out per archetype x LOD.
    veg_inst_cpu_ = std::move(inst);
    std::vector<uint32_t> per_arch(veg_arch_.size(), 0);
    for (const auto& vi : veg_inst_cpu_) per_arch[size_t(vi.rot_arch_phase.y + 0.5f)]++;
    uint32_t vis_base = 0;
    size_t ci = 0;
    for (size_t a = 0; a < veg_arch_.size(); ++a)
        for (int l = 0; l < kVegDrawLods; ++l, ++ci) {
            veg_cmd_template_[ci].firstInstance = vis_base;
            vis_base += per_arch[a];
        }
    veg_count_ = uint32_t(veg_inst_cpu_.size());
    veg_vis_capacity_ = std::max(vis_base, 1u);

    auto make = [&](VkDeviceSize size, bool host, VmaAllocation& alloc, void** mapped) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (host) aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                              VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer b = VK_NULL_HANDLE;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &b, &alloc, &ai), "veg buffer (replant)");
        if (mapped) *mapped = ai.pMappedData;
        return b;
    };
    vmaDestroyBuffer(allocator_, veg_inst_buf_, veg_inst_alloc_);
    vmaDestroyBuffer(allocator_, veg_vis_buf_, veg_vis_alloc_);
    vmaDestroyBuffer(allocator_, veg_fade_buf_, veg_fade_alloc_);
    void* mp = nullptr;
    veg_inst_buf_ = make(sizeof(VegInstance) * std::max<size_t>(veg_count_, 1), true, veg_inst_alloc_, &mp);
    if (mp && veg_count_) std::memcpy(mp, veg_inst_cpu_.data(), sizeof(VegInstance) * veg_count_);
    veg_inst_mapped_ = mp;
    vmaFlushAllocation(allocator_, veg_inst_alloc_, 0, VK_WHOLE_SIZE);
    veg_vis_buf_ = make(sizeof(uint32_t) * veg_vis_capacity_, false, veg_vis_alloc_, nullptr);
    veg_fade_buf_ = make(sizeof(float) * veg_vis_capacity_, false, veg_fade_alloc_, nullptr);

    VkDescriptorBufferInfo bi[3] = { { veg_inst_buf_, 0, VK_WHOLE_SIZE }, { veg_vis_buf_, 0, VK_WHOLE_SIZE },
                                     { veg_fade_buf_, 0, VK_WHOLE_SIZE } };
    const uint32_t bmap[3] = { 0, 1, 5 };
    VkWriteDescriptorSet w[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = veg_set_; w[i].dstBinding = bmap[i]; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, w, 0, nullptr);
    veg_rt_valid_ = false;                      // re-pick the ray-traced tree set

    log::infof("[paint] replanted: placement %.0f ms (background) + apply %.0f ms: %u instances (%d pines, %d broadleaf, %d bushes)",
               place_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               veg_count_, st.pines, st.broadleaf, st.bushes);
}

// ---------------------------------------------------------------------
// Catch-up: small plants (grass tufts follow the paint map's grass layer)
// ---------------------------------------------------------------------

void VulkanEngine::build_scatter_from(const Heightmap& hm) {
    ScatterParams sp{};
    sp.water_level = rt_.water_level;
    sp.flower_density = glm::clamp(rt_.flower_density, 0.0f, 2.0f);
    sp.ground_density = glm::clamp(rt_.ground_density, 0.0f, 3.0f);
    // Tree crowns for forest-floor vs meadow ground cover.
    std::vector<glm::vec3> tree_crowns;
    for (const auto& vi : veg_inst_cpu_) {
        const size_t a = size_t(vi.rot_arch_phase.y + 0.5f);
        if (vi.pos_scale.w <= 0.0f || a >= veg_arch_.size() || veg_arch_[a].species > 1) continue;
        tree_crowns.push_back(glm::vec3(vi.pos_scale.x, vi.pos_scale.z, veg_arch_[a].radius * vi.pos_scale.w * 0.75f));
    }
    sp.trees = &tree_crowns;
    sp.rock_band_h = rt_.terrain_h_dirt_rock_start;
    sp.snow_band_h = rt_.terrain_h_rock_snow_start;
    sp.road = &castle_road_;
    // (also no tall tufts / flowers between the graves)
    if (church_placed_) sp.keep_rects.push_back(glm::vec4(church_pos_.x - 7.0f, church_pos_.z + 5.0f, church_pos_.x + 8.2f, church_pos_.z + 15.0f));
    if (church_placed_) sp.keep_rects.push_back(glm::vec4(church_pos_.x - 6.2f, church_pos_.z - 3.9f, church_pos_.x + 8.0f, church_pos_.z + 3.9f));
    sp.rm_params = &terrain_hp_;   // offsets onto the raymarched surface
    if (!paint_.empty()) sp.paint = &paint_;
    scatter_ = build_scatter(device_, allocator_, graphics_queue_, graphics_queue_family_, hm, sp);
    scatter_set_raymarch_surface(scatter_, rt_.terrain_raymarch_enabled);
}

void VulkanEngine::rebuild_scatter() {
    if (terrain_data_.heights.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    Heightmap hm;
    hm.dim = terrain_data_.dim;
    hm.cell = terrain_data_.cell;
    hm.origin_x = terrain_data_.origin_x;
    hm.origin_z = terrain_data_.origin_z;
    hm.heights = terrain_data_.heights;
    vkDeviceWaitIdle(device_);
    destroy_scatter(allocator_, scatter_);
    scatter_ = ScatterSet{};
    build_scatter_from(hm);
    paint_dirty_grass_ = false;
    log::infof("[paint] small plants rebuilt in %.0f ms",
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

// ---------------------------------------------------------------------
// Edit-mode panel (top right) + paint overlay around the brush.
// ---------------------------------------------------------------------

void VulkanEngine::build_edit_ui() {
    if (!terrain_edit_mode_) {
        if (edit_cursor_free_ && state_ == State::Playing) window_->set_cursor_captured(true);
        edit_cursor_free_ = false;
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    const bool paint_tool = terrain_brush_mode_ == TerrainBrushMode::ZonePaint;
    const int n_presets = int(paint_.presets.size());
    paint_active_ = std::clamp(paint_active_, 1, std::max(1, n_presets - 1));

    // M: hand the mouse to this panel (and back). While the cursor is free
    // the camera and the brush are parked. Read straight from the keyboard
    // (not through ImGui) so it works whatever has focus.
    {
        GLFWwindow* w = window_->handle();
        const bool m_down = glfwGetKey(w, GLFW_KEY_M) == GLFW_PRESS;
        const bool m_edge = m_down && !edit_m_prev_;
        edit_m_prev_ = m_down;
        if (state_ == State::Playing) {
            if (m_edge && !io.WantTextInput) {
                edit_cursor_free_ = !edit_cursor_free_;
                window_->set_cursor_captured(!edit_cursor_free_);
            }
            // Z / X step through the presets while painting.
            static bool z_prev = false, x_prev = false;
            const bool z_down = glfwGetKey(w, GLFW_KEY_Z) == GLFW_PRESS, x_down = glfwGetKey(w, GLFW_KEY_X) == GLFW_PRESS;
            if (paint_tool && !io.WantTextInput && n_presets > 2) {
                if (x_down && !x_prev) paint_active_ = 1 + (paint_active_ % (n_presets - 1));
                if (z_down && !z_prev) paint_active_ = 1 + ((paint_active_ + n_presets - 3) % (n_presets - 1));
            }
            z_prev = z_down; x_prev = x_down;
        } else {
            edit_cursor_free_ = false;
        }
    }

    // Overlay: coloured dots on painted cells around the brush (the colour
    // of the preset on the first painted layer: trees, bushes, grass).
    if (paint_tool && terrain_brush_has_hit_ && !paint_.empty()) {
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        const float cell = PaintMap::cell();
        const glm::vec3 c = terrain_brush_world_pos_;
        const float reach = std::clamp(terrain_brush_radius_ * 2.5f + 30.0f, 40.0f, 140.0f);
        const int step = reach > 90.0f ? 3 : 2;                       // 4 m or 6 m dot spacing
        const int ci = int(std::floor((c.x + 0.5f * kPaintSide) / cell));
        const int cj = int(std::floor((c.z + 0.5f * kPaintSide) / cell));
        const int n = int(reach / cell);
        for (int j = cj - n; j <= cj + n; j += step)
            for (int i = ci - n; i <= ci + n; i += step) {
                if (i < 0 || j < 0 || i >= kPaintDim || j >= kPaintDim) continue;
                const size_t k = size_t(j) * kPaintDim + size_t(i);
                uint8_t id = paint_.layer[kPaintTrees][k];
                if (!id) id = paint_.layer[kPaintBushes][k];
                if (!id) id = paint_.layer[kPaintGrass][k];
                if (!id) continue;
                const float x = (float(i) + 0.5f) * cell - 0.5f * kPaintSide;
                const float zz = (float(j) + 0.5f) * cell - 0.5f * kPaintSide;
                if ((x - c.x) * (x - c.x) + (zz - c.z) * (zz - c.z) > reach * reach) continue;
                const glm::vec3 p(x, sample_terrain_height(x, zz) + 0.3f, zz);
                const glm::vec4 clip = last_view_proj_ * glm::vec4(p, 1.0f);
                if (clip.w < 0.5f) continue;
                const glm::vec2 ndc(clip.x / clip.w, clip.y / clip.w);
                if (std::abs(ndc.x) > 1.0f || std::abs(ndc.y) > 1.0f) continue;
                const ImVec2 s((ndc.x * 0.5f + 0.5f) * io.DisplaySize.x, (ndc.y * 0.5f + 0.5f) * io.DisplaySize.y);
                const float rad = glm::clamp(float(step) * cell * 0.30f * io.DisplaySize.y / (clip.w * 1.15f), 1.5f, 9.0f);
                dl->AddCircleFilled(s, rad, (preset_color(id) & 0x00FFFFFFu) | 0x59000000u, 8);
            }
    }

    // ---- Simple mode: pick an item, draw it ----
    // A bar of things to plant along the bottom of the screen. 1-5 pick one,
    // left mouse draws it, right mouse removes it again. The full panel
    // (presets, densities, terrain tools) only appears with M.
    {
        struct Item { const char* label; const char* preset; bool t; float td, pine; bool b; float bd; bool g; float gd, gh; };
        static const Item kItems[] = {
            { "Pine trees", "Item: pine trees", true,  1.0f, 1.0f, false, 0.0f, false, 1.0f, 1.0f },
            { "Leaf trees", "Item: leaf trees", true,  1.0f, 0.0f, false, 0.0f, false, 1.0f, 1.0f },
            { "Bushes",     "Item: bushes",     false, 0.0f, 0.5f, true,  0.9f, false, 1.0f, 1.0f },
            { "Grass",      "Item: grass",      false, 0.0f, 0.5f, false, 0.0f, true,  1.6f, 1.3f },
            { "Clear all",  "Item: nothing",    true,  0.0f, 0.5f, true,  0.0f, true,  0.0f, 1.0f },
        };
        constexpr int kN = IM_ARRAYSIZE(kItems);
        int item_id[kN];
        for (int k = 0; k < kN; ++k) {
            item_id[k] = 0;
            for (int id = 1; id < int(paint_.presets.size()); ++id)
                if (std::strcmp(paint_.presets[size_t(id)].name, kItems[k].preset) == 0) { item_id[k] = id; break; }
            if (!item_id[k] && !paint_.empty() && int(paint_.presets.size()) < kPaintMaxPresets) {
                const Item& it = kItems[k];
                paint_.presets.push_back(make_preset(it.preset, it.t, it.td, it.pine, it.b, it.bd, it.g, it.gd, it.gh, 0.0f));
                item_id[k] = int(paint_.presets.size()) - 1;
            }
        }
        GLFWwindow* w = window_->handle();
        static bool num_prev[kN] = {};
        static bool first_pick = true;
        for (int k = 0; k < kN; ++k) {
            const bool d = glfwGetKey(w, GLFW_KEY_1 + k) == GLFW_PRESS;
            if (d && !num_prev[k] && item_id[k] && state_ == State::Playing && !io.WantTextInput) {
                paint_active_ = item_id[k];
                terrain_brush_mode_ = TerrainBrushMode::ZonePaint;
            }
            num_prev[k] = d;
        }
        if (first_pick && item_id[0]) {            // start on an item, with the plant tool
            first_pick = false;
            paint_active_ = item_id[0];
            terrain_brush_mode_ = TerrainBrushMode::ZonePaint;
        }
        if (!edit_cursor_free_) {
            ImDrawList* dl = ImGui::GetForegroundDrawList();
            const float bw2 = 150.0f, bh = 58.0f, gap = 10.0f;
            const float total = kN * bw2 + (kN - 1) * gap;
            const float x0 = (io.DisplaySize.x - total) * 0.5f, y0 = io.DisplaySize.y - bh - 92.0f;
            const bool planting = terrain_brush_mode_ == TerrainBrushMode::ZonePaint;
            dl->AddRectFilled(ImVec2(x0 - 14.0f, y0 - 34.0f), ImVec2(x0 + total + 14.0f, y0 + bh + 34.0f), IM_COL32(12, 14, 18, 190), 8.0f);
            dl->AddText(ImVec2(x0, y0 - 26.0f), IM_COL32(255, 170, 70, 255), "EDIT MODE  -  pick what to draw (1-5), then hold the left mouse button");
            for (int k = 0; k < kN; ++k) {
                const ImVec2 a(x0 + k * (bw2 + gap), y0), b(a.x + bw2, a.y + bh);
                const bool sel = planting && item_id[k] && paint_active_ == item_id[k];
                dl->AddRectFilled(a, b, sel ? IM_COL32(230, 110, 30, 255) : IM_COL32(48, 52, 60, 235), 6.0f);
                if (sel) dl->AddRect(a, b, IM_COL32(255, 235, 200, 255), 6.0f, 0, 2.0f);
                if (item_id[k]) dl->AddRectFilled(ImVec2(a.x + 10.0f, a.y + 32.0f), ImVec2(a.x + 26.0f, a.y + 48.0f), preset_color(item_id[k]), 3.0f);
                char num[4]; std::snprintf(num, sizeof(num), "%d", k + 1);
                dl->AddText(ImVec2(a.x + 10.0f, a.y + 8.0f), IM_COL32(255, 255, 255, 150), num);
                dl->AddText(ImVec2(a.x + 34.0f, a.y + 32.0f), IM_COL32(255, 255, 255, 255), kItems[k].label);
            }
            char hint[200];
            std::snprintf(hint, sizeof(hint), "Left mouse: draw      Right mouse: remove      [ ] brush size (%.0f m)      M: more options      E: done%s",
                          terrain_brush_radius_, paint_unsaved_ ? "      (unsaved - M, then Save)" : "");
            dl->AddText(ImVec2(x0, y0 + bh + 10.0f), IM_COL32(220, 220, 220, 255), hint);
            return;
        }
    }

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    if (!edit_cursor_free_) flags |= ImGuiWindowFlags_NoInputs;
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 12.0f, 12.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));   // top right: clear of the HUD
    // Never taller than the screen: the panel scrolls instead.
    ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f), ImVec2(io.DisplaySize.x, io.DisplaySize.y - 24.0f));
    ImGui::SetNextWindowBgAlpha(edit_cursor_free_ ? 0.94f : 0.72f);
    ImGui::PushStyleColor(ImGuiCol_TitleBg, IM_COL32(190, 90, 25, 255));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, IM_COL32(230, 110, 30, 255));
    ImGui::Begin(edit_cursor_free_ ? "EDIT MODE  -  mouse on panel   (M: back to painting)###edit"
                                   : "EDIT MODE  -  press M to use the mouse on this panel###edit", nullptr, flags);
    ImGui::PopStyleColor(2);

    const ImVec4 on(0.90f, 0.43f, 0.12f, 1.0f);
    const int cur = static_cast<int>(terrain_brush_mode_);
    const bool raymarch = rt_.terrain_raymarch_enabled;
    const float bw = 100.0f, sw = 190.0f;

    ImGui::SeparatorText("Tool   (Q / R)");
    for (int t = 0; t < IM_ARRAYSIZE(kToolNames); ++t) {
        if (t % 3 != 0) ImGui::SameLine();
        const bool height_tool = !(t == 4 || t == 5 || t == 8);
        const bool sel = (t == cur);
        if (sel) ImGui::PushStyleColor(ImGuiCol_Button, on);
        if (height_tool && raymarch) ImGui::BeginDisabled();
        if (ImGui::Button(kToolNames[t], ImVec2(bw, 0.0f))) terrain_brush_mode_ = static_cast<TerrainBrushMode>(t);
        if (height_tool && raymarch) ImGui::EndDisabled();
        if (sel) ImGui::PopStyleColor();
    }
    if (raymarch) ImGui::TextDisabled("Height tools need the mesh terrain renderer.");

    ImGui::SeparatorText("Brush   ( [  ] )");
    ImGui::SetNextItemWidth(sw);
    ImGui::SliderFloat("size (m)", &terrain_brush_radius_, 1.0f, 60.0f, "%.0f");
    if (!paint_tool) {
        ImGui::SetNextItemWidth(sw);
        ImGui::SliderFloat("strength", &terrain_brush_strength_, 0.5f, 60.0f, "%.1f");
    }
    if (terrain_brush_mode_ == TerrainBrushMode::Flatten) {
        ImGui::SetNextItemWidth(sw);
        ImGui::SliderFloat("flatten to Y", &terrain_brush_flatten_target_, 0.0f, 200.0f, "%.1f");
    }

    if (paint_tool && n_presets > 1) {
        // ---- Painter: pick the active preset ----
        ImGui::SeparatorText("Painter   (Z / X)");
        ImGui::BeginChild("##presets", ImVec2(318.0f, std::min(6, n_presets - 1) * ImGui::GetTextLineHeightWithSpacing() + 6.0f),
                          ImGuiChildFlags_Borders);
        for (int id = 1; id < n_presets; ++id) {
            const PaintPreset& p = paint_.presets[size_t(id)];
            ImGui::PushID(id);
            const ImVec2 q = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(q.x, q.y + 2.0f), ImVec2(q.x + 12.0f, q.y + 14.0f), preset_color(id), 2.0f);
            ImGui::Dummy(ImVec2(14.0f, 0.0f));
            ImGui::SameLine();
            char label[64];
            std::snprintf(label, sizeof(label), "%-20s %c%c%c", p.name, p.trees ? 'T' : '-', p.bushes ? 'B' : '-', p.grass ? 'G' : '-');
            if (ImGui::Selectable(label, id == paint_active_)) paint_active_ = id;
            ImGui::PopID();
        }
        ImGui::EndChild();
        if (ImGui::Button("New", ImVec2(70.0f, 0.0f)) && n_presets < kPaintMaxPresets) {
            paint_.presets.push_back(make_preset("New preset", true, 0.6f, 0.5f, true, 0.3f, true, 1.0f, 1.0f, 0.0f));
            paint_active_ = int(paint_.presets.size()) - 1;
            paint_unsaved_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Duplicate", ImVec2(90.0f, 0.0f)) && n_presets < kPaintMaxPresets) {
            PaintPreset c = paint_.presets[size_t(paint_active_)];
            char nm[40];
            std::snprintf(nm, sizeof(nm), "%.24s copy", c.name);
            std::snprintf(c.name, sizeof(c.name), "%s", nm);
            paint_.presets.push_back(c);
            paint_active_ = int(paint_.presets.size()) - 1;
            paint_unsaved_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Delete", ImVec2(70.0f, 0.0f)) && paint_.presets.size() > 2) {
            // Painted cells of the deleted preset go back to automatic; ids
            // above it shift down by one.
            const uint8_t dead = uint8_t(paint_active_);
            for (int l = 0; l < kPaintLayers; ++l)
                for (auto& c : paint_.layer[l]) {
                    if (c == dead) { c = 0; if (l == kPaintGrass) paint_dirty_grass_ = true; else paint_dirty_veg_ = true; }
                    else if (c > dead) --c;
                }
            paint_.presets.erase(paint_.presets.begin() + paint_active_);
            paint_active_ = std::clamp(paint_active_, 1, int(paint_.presets.size()) - 1);
            paint_unsaved_ = true;
            request_paint_apply(false);
        }

        // ---- Properties of the active preset ----
        PaintPreset& p = paint_.presets[size_t(paint_active_)];
        bool changed = false, veg_changed = false, grass_changed = false;
        ImGui::SeparatorText("Preset properties");
        ImGui::SetNextItemWidth(sw);
        changed |= ImGui::InputText("name", p.name, sizeof(p.name));
        bool tr = p.trees != 0, bu = p.bushes != 0, gr = p.grass != 0;
        if (ImGui::Checkbox("Trees", &tr)) { p.trees = tr; changed = true; }
        if (tr) {
            ImGui::SetNextItemWidth(sw); veg_changed |= ImGui::SliderFloat("tree density", &p.tree_density, 0.0f, 1.0f, p.tree_density <= 0.0f ? "none" : "%.2f");
            ImGui::SetNextItemWidth(sw); veg_changed |= ImGui::SliderFloat("pine share", &p.pine_share, 0.0f, 1.0f, "%.2f");
            ImGui::SetNextItemWidth(sw); veg_changed |= ImGui::SliderFloat("tree size", &p.tree_size, 0.5f, 1.8f, "%.2fx");
        }
        if (ImGui::Checkbox("Bushes", &bu)) { p.bushes = bu; changed = true; }
        if (bu) {
            ImGui::SetNextItemWidth(sw); veg_changed |= ImGui::SliderFloat("bush density", &p.bush_density, 0.0f, 1.0f, p.bush_density <= 0.0f ? "none" : "%.2f");
        }
        if (ImGui::Checkbox("Grass", &gr)) { p.grass = gr; changed = true; }
        if (gr) {
            ImGui::SetNextItemWidth(sw); grass_changed |= ImGui::SliderFloat("grass density", &p.grass_density, 0.0f, 2.0f, p.grass_density <= 0.0f ? "none" : "%.2f");
            ImGui::SetNextItemWidth(sw); grass_changed |= ImGui::SliderFloat("grass height", &p.grass_height, 0.4f, 2.5f, "%.2fx");
            ImGui::SetNextItemWidth(sw); grass_changed |= ImGui::SliderFloat("dryness", &p.grass_dry, 0.0f, 1.0f, "%.2f");
        }
        if (veg_changed) paint_dirty_veg_ = true;        // areas already painted with it are stale
        if (grass_changed) paint_dirty_grass_ = true;
        if (changed || veg_changed || grass_changed) paint_unsaved_ = true;
        ImGui::TextDisabled("Left mouse paints, right mouse erases\nthe ticked kinds (back to automatic).");

        ImGui::SeparatorText("Apply");
        ImGui::Checkbox("update the world when a stroke ends", &veg_auto_replant_);
        const bool stale = paint_dirty_veg_ || paint_dirty_grass_;
        if (stale) ImGui::PushStyleColor(ImGuiCol_Button, on);
        if (ImGui::Button(stale ? "Apply changes *" : "Apply changes", ImVec2(150.0f, 0.0f))) request_paint_apply(!stale);
        if (stale) ImGui::PopStyleColor();
        ImGui::SameLine();
        if (ImGui::Button(paint_unsaved_ ? "Save *" : "Save", ImVec2(80.0f, 0.0f))) save_paint();
        if (ImGui::Button("Clear all paint", ImVec2(150.0f, 0.0f))) {
            for (auto& l : paint_.layer) std::fill(l.begin(), l.end(), uint8_t(0));
            paint_unsaved_ = true;
            request_paint_apply(true);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%u plants", veg_count_);
    } else if (brush_mode_is_grass(terrain_brush_mode_)) {
        ImGui::SeparatorText("Lawn grass   left = this tool, right = opposite");
        if (ImGui::Button("Save grass mask", ImVec2(150.0f, 0.0f))) save_grass_mask();
    }

    ImGui::SeparatorText("Quick");
    ImGui::Checkbox("fly (V)", &fly_mode_);
    ImGui::SameLine();
    bool clock_run = rt_.day_length_minutes > 0.0f;
    if (ImGui::Checkbox("clock runs", &clock_run)) rt_.day_length_minutes = clock_run ? 24.0f : 0.0f;
    ImGui::SetNextItemWidth(sw);
    ImGui::SliderFloat("time of day", &rt_.time_of_day, 0.0f, 24.0f, "%.1f h");
    ImGui::TextDisabled("M: mouse on / off      E: leave edit mode");
    ImGui::End();
}

}  // namespace qlike
