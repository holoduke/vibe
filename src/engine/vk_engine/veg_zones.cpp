// Vegetation zones + the edit-mode tool panel.
//
// A zone map (1024 x 1024 bytes over the 2048 m terrain, 2 m cells) says
// what should grow where: 0 = automatic (the procedural forest), or a
// painted override - pine forest, broadleaf forest, mixed forest, meadow,
// bushland, bare ground. place_vegetation() reads it; the map is painted
// with the terrain brush (tool "Zones") and saved to
// assets/level1_veg_zones.bin.
//
// Painting does not move trees by itself: the forest (instances, colliders,
// physics compounds, GPU buffers) is re-planted when a stroke ends, or on
// demand from the panel.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/log.h"
#include "engine/window.h"
#include "engine/world_props.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace qlike {

namespace {
constexpr int   kZoneDim = 1024;
constexpr float kZoneSide = 2048.0f;
const char* const kZoneFile = "assets/level1_veg_zones.bin";

struct ZoneInfo { const char* name; ImU32 col; const char* tip; };
const ZoneInfo kZones[kVegZoneCount] = {
    { "Auto",      IM_COL32(150, 150, 150, 255), "Erase: back to the automatic forest" },
    { "Pine",      IM_COL32( 30, 110,  70, 255), "Dense pine forest" },
    { "Broadleaf", IM_COL32(110, 190,  60, 255), "Dense broadleaf (oak) forest" },
    { "Mixed",     IM_COL32( 60, 150, 110, 255), "Dense forest, pines and broadleaf mixed" },
    { "Meadow",    IM_COL32(230, 215,  90, 255), "Open grass: no trees, the odd bush" },
    { "Bushland",  IM_COL32(200, 130,  50, 255), "Thickets of bushes, no trees" },
    { "Bare",      IM_COL32(190,  80,  80, 255), "Nothing grows here (no trees, no bushes)" },
};

const char* const kToolNames[] = { "Raise", "Lower", "Smooth", "Flatten", "Grass +", "Grass -",
                                   "Erode +", "Erode smooth", "Zones" };
}  // namespace

// ---------------------------------------------------------------------
// Zone map
// ---------------------------------------------------------------------

void VulkanEngine::load_veg_zones() {
    veg_zones_.assign(size_t(kZoneDim) * kZoneDim, 0);
    std::ifstream f(kZoneFile, std::ios::binary);
    if (!f) return;
    f.read(reinterpret_cast<char*>(veg_zones_.data()), std::streamsize(veg_zones_.size()));
    if (size_t(f.gcount()) != veg_zones_.size()) {
        log::warnf("[zones] %s has the wrong size - ignored", kZoneFile);
        std::fill(veg_zones_.begin(), veg_zones_.end(), uint8_t(0));
        return;
    }
    for (auto& z : veg_zones_) if (z >= kVegZoneCount) z = 0;
    log::infof("[zones] loaded %s", kZoneFile);
}

bool VulkanEngine::save_veg_zones() {
    std::ofstream f(kZoneFile, std::ios::binary);
    if (!f) { log::errorf("[zones] cannot write %s", kZoneFile); return false; }
    f.write(reinterpret_cast<const char*>(veg_zones_.data()), std::streamsize(veg_zones_.size()));
    log::infof("[zones] saved %s", kZoneFile);
    veg_zones_unsaved_ = false;
    return true;
}

// Stamps `zone` into every cell within the brush radius of the brush hit.
void VulkanEngine::paint_veg_zone(uint8_t zone) {
    if (!terrain_brush_has_hit_ || veg_zones_.empty()) return;
    const float cell = kZoneSide / float(kZoneDim);
    const float r = terrain_brush_radius_;
    const glm::vec3 c = terrain_brush_world_pos_;
    const int i0 = std::max(0, int(std::floor((c.x - r + 0.5f * kZoneSide) / cell)));
    const int i1 = std::min(kZoneDim - 1, int(std::floor((c.x + r + 0.5f * kZoneSide) / cell)));
    const int j0 = std::max(0, int(std::floor((c.z - r + 0.5f * kZoneSide) / cell)));
    const int j1 = std::min(kZoneDim - 1, int(std::floor((c.z + r + 0.5f * kZoneSide) / cell)));
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i) {
            const float x = (float(i) + 0.5f) * cell - 0.5f * kZoneSide - c.x;
            const float z = (float(j) + 0.5f) * cell - 0.5f * kZoneSide - c.z;
            if (x * x + z * z > r * r) continue;
            uint8_t& dst = veg_zones_[size_t(j) * kZoneDim + i];
            if (dst != zone) { dst = zone; veg_zones_dirty_ = true; veg_zones_unsaved_ = true; }
        }
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
    for (const auto& l : lamp_lights_) wp.keep_out.push_back(glm::vec3(l.x, l.z, 2.5f));
    if (!veg_zones_.empty()) {
        wp.zones = veg_zones_.data();
        wp.zone_dim = kZoneDim;
        wp.zone_side = kZoneSide;
    }
    return wp;
}

// ---------------------------------------------------------------------
// Re-plant: run the placement again with the current zone map and swap in
// the result everywhere the old forest lived.
// ---------------------------------------------------------------------

void VulkanEngine::replant_vegetation() {
    if (veg_arch_.empty() || terrain_data_.heights.empty() || !veg_inst_buf_) return;
    const auto t0 = std::chrono::steady_clock::now();
    Heightmap hm;
    hm.dim = terrain_data_.dim;
    hm.cell = terrain_data_.cell;
    hm.origin_x = terrain_data_.origin_x;
    hm.origin_z = terrain_data_.origin_z;
    hm.heights = terrain_data_.heights;

    std::vector<collision::AABB> cols;
    WorldPropStats st{};
    const WorldPropParams wp = veg_place_params();
    std::vector<VegInstance> inst = place_vegetation(hm, &castle_road_, wp, veg_arch_, cols, st);

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
    veg_zones_dirty_ = false;

    log::infof("[zones] replanted in %.0f ms: %u instances (%d pines, %d broadleaf, %d bushes)",
               std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
               veg_count_, st.pines, st.broadleaf, st.bushes);
}

// ---------------------------------------------------------------------
// Edit-mode panel (left side) + zone overlay around the brush.
// ---------------------------------------------------------------------

void VulkanEngine::build_edit_ui() {
    if (!terrain_edit_mode_) {
        if (edit_cursor_free_ && state_ == State::Playing) window_->set_cursor_captured(true);
        edit_cursor_free_ = false;
        return;
    }
    ImGuiIO& io = ImGui::GetIO();

    // Tab: hand the mouse to this panel (and back). While the cursor is
    // free the camera and the brush are parked.
    if (state_ == State::Playing) {
        if (ImGui::IsKeyPressed(ImGuiKey_Tab, false)) {
            edit_cursor_free_ = !edit_cursor_free_;
            window_->set_cursor_captured(!edit_cursor_free_);
        }
        // Z / X step through the zone palette while the zone tool is active.
        if (terrain_brush_mode_ == TerrainBrushMode::ZonePaint && !edit_cursor_free_) {
            if (ImGui::IsKeyPressed(ImGuiKey_X, false)) veg_zone_brush_ = 1 + (veg_zone_brush_ % (kVegZoneCount - 1));
            if (ImGui::IsKeyPressed(ImGuiKey_Z, false)) veg_zone_brush_ = 1 + ((veg_zone_brush_ + kVegZoneCount - 3) % (kVegZoneCount - 1));
        }
    } else {
        edit_cursor_free_ = false;
    }

    // Zone overlay: coloured dots on the ground around the brush (painted
    // cells only), so the borders are visible while painting.
    if (terrain_brush_mode_ == TerrainBrushMode::ZonePaint && terrain_brush_has_hit_ && !veg_zones_.empty()) {
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        const float cell = kZoneSide / float(kZoneDim);
        const glm::vec3 c = terrain_brush_world_pos_;
        const float reach = std::clamp(terrain_brush_radius_ * 2.5f + 30.0f, 40.0f, 140.0f);
        const int step = reach > 90.0f ? 3 : 2;                       // 4 m or 6 m dot spacing
        const int ci = int(std::floor((c.x + 0.5f * kZoneSide) / cell));
        const int cj = int(std::floor((c.z + 0.5f * kZoneSide) / cell));
        const int n = int(reach / cell);
        const glm::vec3 eye = current_frame_view_.eye_pos;
        for (int j = cj - n; j <= cj + n; j += step)
            for (int i = ci - n; i <= ci + n; i += step) {
                if (i < 0 || j < 0 || i >= kZoneDim || j >= kZoneDim) continue;
                const uint8_t z = veg_zones_[size_t(j) * kZoneDim + i];
                if (z == 0) continue;
                const float x = (float(i) + 0.5f) * cell - 0.5f * kZoneSide;
                const float zz = (float(j) + 0.5f) * cell - 0.5f * kZoneSide;
                if ((x - c.x) * (x - c.x) + (zz - c.z) * (zz - c.z) > reach * reach) continue;
                const glm::vec3 p(x, sample_terrain_height(x, zz) + 0.3f, zz);
                const glm::vec4 clip = last_view_proj_ * glm::vec4(p, 1.0f);
                if (clip.w < 0.5f) continue;
                const glm::vec2 ndc(clip.x / clip.w, clip.y / clip.w);
                if (std::abs(ndc.x) > 1.0f || std::abs(ndc.y) > 1.0f) continue;
                const ImVec2 s((ndc.x * 0.5f + 0.5f) * io.DisplaySize.x, (ndc.y * 0.5f + 0.5f) * io.DisplaySize.y);
                const float rad = glm::clamp(float(step) * cell * 0.42f * io.DisplaySize.y / (clip.w * 1.15f), 1.5f, 14.0f);
                const ImU32 col = (kZones[z].col & 0x00FFFFFFu) | (glm::distance(p, eye) < 250.0f ? 0x66000000u : 0x40000000u);
                dl->AddCircleFilled(s, rad, col, 8);
            }
    }

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
    if (!edit_cursor_free_) flags |= ImGuiWindowFlags_NoInputs;
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 12.0f, 12.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));   // top right: clear of the HUD
    ImGui::SetNextWindowBgAlpha(edit_cursor_free_ ? 0.92f : 0.72f);
    ImGui::PushStyleColor(ImGuiCol_TitleBg, IM_COL32(190, 90, 25, 255));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, IM_COL32(230, 110, 30, 255));
    ImGui::Begin(edit_cursor_free_ ? "EDIT MODE  -  mouse on panel (Tab: back to brush)###edit"
                                   : "EDIT MODE  -  Tab: use mouse on this panel###edit", nullptr, flags);
    ImGui::PopStyleColor(2);

    const ImVec4 on(0.90f, 0.43f, 0.12f, 1.0f);
    const int cur = static_cast<int>(terrain_brush_mode_);
    const bool raymarch = rt_.terrain_raymarch_enabled;

    ImGui::SeparatorText("Tool   (Q / R)");
    const float bw = 96.0f;
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
    ImGui::SetNextItemWidth(200.0f);
    ImGui::SliderFloat("size (m)", &terrain_brush_radius_, 1.0f, 60.0f, "%.0f");
    if (terrain_brush_mode_ != TerrainBrushMode::ZonePaint) {
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("strength", &terrain_brush_strength_, 0.5f, 60.0f, "%.1f");
    }
    if (terrain_brush_mode_ == TerrainBrushMode::Flatten) {
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("flatten to Y", &terrain_brush_flatten_target_, 0.0f, 200.0f, "%.1f");
    }

    if (terrain_brush_mode_ == TerrainBrushMode::ZonePaint) {
        ImGui::SeparatorText("Zone   (Z / X)     left = paint, right = erase");
        for (int z = 1; z < kVegZoneCount; ++z) {
            if ((z - 1) % 3 != 0) ImGui::SameLine();
            const bool sel = (z == veg_zone_brush_);
            const ImVec4 c = ImGui::ColorConvertU32ToFloat4(kZones[z].col);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(c.x * (sel ? 1.0f : 0.45f), c.y * (sel ? 1.0f : 0.45f), c.z * (sel ? 1.0f : 0.45f), 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, sel ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1));
            if (ImGui::Button(kZones[z].name, ImVec2(bw, 0.0f))) veg_zone_brush_ = z;
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kZones[z].tip);
        }
        ImGui::TextDisabled("%s", kZones[veg_zone_brush_].tip);
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("forest density", &rt_.forest_density, 0.0f, 2.0f, "%.2f");
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("bush density", &rt_.bush_density, 0.0f, 2.0f, "%.2f");
        ImGui::Checkbox("replant when the stroke ends", &veg_auto_replant_);
        if (ImGui::Button("Replant forest now", ImVec2(150.0f, 0.0f))) veg_replant_request_ = true;
        ImGui::SameLine();
        if (ImGui::Button(veg_zones_unsaved_ ? "Save zones *" : "Save zones", ImVec2(110.0f, 0.0f))) save_veg_zones();
        if (ImGui::Button("Clear all zones", ImVec2(150.0f, 0.0f))) {
            std::fill(veg_zones_.begin(), veg_zones_.end(), uint8_t(0));
            veg_zones_unsaved_ = true;
            veg_replant_request_ = true;
        }
        ImGui::Text("%u plants", veg_count_);
    } else if (brush_mode_is_grass(terrain_brush_mode_)) {
        ImGui::SeparatorText("Grass     left = this tool, right = opposite");
        if (ImGui::Button("Save grass mask", ImVec2(150.0f, 0.0f))) save_grass_mask();
    }

    ImGui::SeparatorText("Quick");
    ImGui::Checkbox("fly (V)", &fly_mode_);
    ImGui::SameLine();
    bool clock_run = rt_.day_length_minutes > 0.0f;
    if (ImGui::Checkbox("clock runs", &clock_run)) rt_.day_length_minutes = clock_run ? 24.0f : 0.0f;
    ImGui::SetNextItemWidth(200.0f);
    ImGui::SliderFloat("time of day", &rt_.time_of_day, 0.0f, 24.0f, "%.1f h");
    if (terrain_brush_has_hit_)
        ImGui::TextDisabled("brush at %.0f, %.0f, %.0f", terrain_brush_world_pos_.x, terrain_brush_world_pos_.y,
                            terrain_brush_world_pos_.z);
    else
        ImGui::TextDisabled("aim at the ground");
    ImGui::TextDisabled("E: leave edit mode");
    ImGui::End();
}

}  // namespace qlike
