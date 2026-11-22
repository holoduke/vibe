// ui_hud.cpp - in-game HUD: stats boxes, crosshair, weapon bar, lightning bolt overlay
// (split out of vk_ui.cpp; see docs/ARCHITECTURE.md for the map.)
// VulkanEngine UI — ImGui setup, pause menu, HUD overlay, crosshair.
// Split out of vk_engine.cpp because UI is independent of render-state plumbing
// and grew large enough to crowd the main engine file.

#include "engine/vk_engine.h"

#include "engine/log.h"
#include "engine/vk_initializers.h"   // vibe::vk_check
#include "engine/window.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>

namespace vibe {


void VulkanEngine::build_hud_ui() {
    // Lightning bolt: a jagged, forked line from the cloud base down to
    // where the strike lands, drawn for the length of the flash. Hidden
    // where a hill stands between the camera and the bolt.
    if (env_lightning_ > 0.03f && state_ == State::Playing) {
        ImGuiIO& lio = ImGui::GetIO();
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        const glm::vec3 eye = player_.eye_position();
        uint32_t rs = uint32_t(lightning_seed_ * 65535.0f) * 2654435761u | 1u;
        auto rnd = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f - 0.5f; };
        const float dist = glm::distance(glm::vec2(eye.x, eye.z), glm::vec2(lightning_pos_.x, lightning_pos_.z));
        const float top = lightning_pos_.y + 520.0f;
        auto project = [&](const glm::vec3& p, ImVec2& out) {
            const glm::vec4 c = last_view_proj_ * glm::vec4(p, 1.0f);
            if (c.w < 1.0f) return false;
            out = ImVec2((c.x / c.w * 0.5f + 0.5f) * lio.DisplaySize.x, (c.y / c.w * 0.5f + 0.5f) * lio.DisplaySize.y);
            // Terrain between the eye and the point hides it.
            for (int k = 1; k < 10; ++k) {
                const glm::vec3 q = glm::mix(eye, p, float(k) / 10.0f);
                if (sample_terrain_height(q.x, q.z) > q.y + 2.0f) return false;
            }
            return true;
        };
        const float a = std::min(env_lightning_, 1.0f);
        const float w = glm::clamp(900.0f / std::max(dist, 150.0f), 1.2f, 4.5f);
        auto strand = [&](glm::vec3 from, glm::vec3 to, int segs, float wob, float width, int depth) {
            auto self = [&](auto& me, glm::vec3 f, glm::vec3 t, int n, float wb, float wd, int dp) -> void {
                glm::vec3 prev = f; ImVec2 sp, sq; bool pv = project(prev, sp);
                for (int i = 1; i <= n; ++i) {
                    glm::vec3 p = glm::mix(f, t, float(i) / float(n));
                    if (i < n) p += glm::vec3(rnd(), rnd() * 0.3f, rnd()) * wb;
                    const bool qv = project(p, sq);
                    if (pv && qv) {
                        dl->AddLine(sp, sq, IM_COL32(150, 170, 255, int(70 * a)), wd * 5.0f);
                        dl->AddLine(sp, sq, IM_COL32(200, 215, 255, int(150 * a)), wd * 2.2f);
                        dl->AddLine(sp, sq, IM_COL32(255, 255, 255, int(255 * a)), wd);
                    }
                    // Forks: shorter, thinner, heading outward and down.
                    if (dp > 0 && i > 1 && i < n - 1 && rnd() > 0.18f) {
                        const glm::vec3 d = glm::vec3(rnd() * 2.0f, -0.55f, rnd() * 2.0f) * (glm::distance(f, t) * 0.32f);
                        me(me, p, p + d, std::max(3, n / 2), wb * 0.6f, wd * 0.55f, dp - 1);
                    }
                    prev = p; sp = sq; pv = qv;
                }
            };
            self(self, from, to, segs, wob, width, depth);
        };
        const glm::vec3 foot = lightning_pos_;
        const glm::vec3 head(foot.x + rnd() * 160.0f, top, foot.z + rnd() * 160.0f);
        strand(head, foot, 16, 34.0f, w, 2);
    }
    if (state_ != State::Playing) return;

    // Push current frame samples into the rolling history (Three.js
    // stats-style ring buffer).
    {
        int& head = stats_history_head_;
        fps_history_[head] = ema_fps_;
        ms_history_[head]  = last_frame_dt_ * 1000.0f;
        head = (head + 1) % kStatsHistory;
    }

    // Helper: render one Three.js-style stats panel — big number on
    // top, mini histogram below, accent colour bar. ImDrawList for
    // the custom panel + ImGui for text inside the panel rect.
    auto draw_panel = [&](ImVec2 origin, const char* label,
                          float current, float min_v, float max_v,
                          const float* history, int history_count, int head,
                          ImU32 accent) {
        const ImVec2 size(120.0f, 48.0f);
        const ImU32 bg = IM_COL32(0, 0, 16, 220);
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        dl->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y), bg);
        // Top accent strip — colour identifies which stat panel.
        dl->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + 2.0f), accent);
        // Big current value, top-left.
        char buf[48];
        snprintf(buf, sizeof(buf), "%.0f %s", current, label);
        dl->AddText(ImVec2(origin.x + 4.0f, origin.y + 4.0f), accent, buf);
        // Min/Max line below the big number.
        snprintf(buf, sizeof(buf), "(%.0f-%.0f)", min_v, max_v);
        dl->AddText(ImVec2(origin.x + 4.0f, origin.y + 18.0f),
                    IM_COL32(200, 200, 220, 180), buf);
        // Histogram strip at the bottom of the panel.
        const float graph_y      = origin.y + 32.0f;
        const float graph_h      = 14.0f;
        const float graph_x_left = origin.x + 4.0f;
        const float graph_w      = size.x - 8.0f;
        // Span = max-min, padded so a flat history shows a clear baseline.
        const float span = std::max(1.0f, max_v - min_v);
        const float bar_w = graph_w / float(history_count);
        for (int i = 0; i < history_count; ++i) {
            // Read chronologically: oldest at head, newest at head-1.
            int idx = (head + i) % history_count;
            float v = history[idx];
            float h = std::min(1.0f, std::max(0.0f, (v - min_v) / span)) * graph_h;
            float bx0 = graph_x_left + float(i) * bar_w;
            float bx1 = bx0 + bar_w + 0.5f;
            dl->AddRectFilled(ImVec2(bx0, graph_y + graph_h - h),
                              ImVec2(bx1, graph_y + graph_h),
                              accent);
        }
    };

    (void)draw_panel;
    draw_game_hud();          // health, weapon, slots, frame rate (ui_gamehud.cpp)
    draw_rocket_hud();        // launch prompt, count, mission panel (rocket.cpp)

    // EDIT MODE banner — centered top, bright orange, impossible to
    // miss. Earlier issue: user pressed E by accident (next to D walk
    // key) and was in edit mode without realizing; left-click then
    // sculpted terrain instead of firing. With this banner the state
    // is always obvious.
    build_edit_ui();     // tool panel + zone overlay (veg_zones.cpp)

    // Last-target damage indicator — industry-standard FPS placement:
    // a single bar at TOP-CENTER showing the most-recently-hit plane's
    // remaining health. Fades out 3 s after the last hit. Bars float-
    // ing above each plane (the previous approach) cluttered the view
    // and overlapped the plane silhouette itself; a single fixed-
    // position bar at top of HUD is what Halo / Battlefield / Destiny
    // use for "you just damaged a target."
    if (last_target_hud_ttl_ > 0.0f && last_target_plane_hp_ >= 0) {
        ImGuiIO& io_pl = ImGui::GetIO();
        ImDrawList* dl_pl = ImGui::GetForegroundDrawList();
        const float W = io_pl.DisplaySize.x;
        const float fade = glm::clamp(last_target_hud_ttl_ / 0.5f, 0.0f, 1.0f);
        const float bar_w = 220.0f;
        const float bar_h = 10.0f;
        const float pad   = 2.0f;
        const float cx = W * 0.5f;
        const float cy = 56.0f;       // below the EDIT MODE banner
        const float a0 = cx - bar_w * 0.5f;
        const float b0 = cx + bar_w * 0.5f;
        const float a1 = cy - bar_h * 0.5f;
        const float b1 = cy + bar_h * 0.5f;
        const int   alpha_bg = int(200.0f * fade);
        const int   alpha_fg = int(245.0f * fade);
        dl_pl->AddRectFilled(ImVec2(a0 - pad, a1 - pad),
                              ImVec2(b0 + pad, b1 + pad),
                              IM_COL32(0, 0, 0, alpha_bg), 2.0f);
        // Defensive max(1,...) on the denominator. last_target_plane_hp_max_
        // defaults to 3 and combat.cpp always rewrites it on every hit, but
        // a future caller (or a load-from-savefile path) that zeros the max
        // would otherwise divide-by-zero and produce NaN/inf alpha here.
        const int max_hp = std::max(1, last_target_plane_hp_max_);
        const float frac = glm::clamp(float(last_target_plane_hp_) /
                                       float(max_hp), 0.0f, 1.0f);
        ImU32 col;
        if (frac > 0.66f)      col = IM_COL32( 80, 220,  80, alpha_fg);
        else if (frac > 0.33f) col = IM_COL32(240, 200,  50, alpha_fg);
        else                   col = IM_COL32(230,  70,  60, alpha_fg);
        dl_pl->AddRectFilled(ImVec2(a0, a1),
                              ImVec2(a0 + bar_w * frac, b1),
                              col, 2.0f);
        // Label centred over the bar, "JET  hp/max".
        char tlbl[24];
        snprintf(tlbl, sizeof(tlbl), "JET   %d / %d",
                 last_target_plane_hp_, last_target_plane_hp_max_);
        ImVec2 ts = ImGui::CalcTextSize(tlbl);
        dl_pl->AddText(ImVec2(cx - ts.x * 0.5f, cy - bar_h * 0.5f - 16.0f),
                       IM_COL32(255, 255, 255, alpha_fg), tlbl);
        // Decay outside the per-frame stats push (frame dt isn't trivially
        // available here; do it from the engine tick in vk_engine.cpp).
    }

    // Kill-flash fullscreen overlay (white, fades over kKillFlashDuration).
    // Painted BEFORE the crosshair so the crosshair stays readable on the
    // brightest frame.
    if (kill_flash_t_ > 0.0f) {
        ImGuiIO& io_kf = ImGui::GetIO();
        ImDrawList* dl_kf = ImGui::GetForegroundDrawList();
        const float f = glm::clamp(kill_flash_t_ / kKillFlashDuration,
                                    0.0f, 1.0f);
        // Ease the alpha curve so the spike is sharp and the fade gentle.
        const float a = f * f;
        const int   ia = int(220.0f * a);
        dl_kf->AddRectFilled(ImVec2(0, 0),
                              ImVec2(io_kf.DisplaySize.x,
                                     io_kf.DisplaySize.y),
                              IM_COL32(255, 240, 220, ia));
    }

    // Crosshair: small dot + tick marks at screen center.
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 cen(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImU32 col = IM_COL32(255, 255, 255, 220);
    const ImU32 col_shadow = IM_COL32(0, 0, 0, 180);
    dl->AddCircleFilled(ImVec2(cen.x, cen.y), 2.5f, col_shadow);
    dl->AddCircleFilled(ImVec2(cen.x, cen.y), 1.5f, col);
    dl->AddLine(ImVec2(cen.x - 9, cen.y), ImVec2(cen.x - 4, cen.y), col, 1.5f);
    dl->AddLine(ImVec2(cen.x + 4, cen.y), ImVec2(cen.x + 9, cen.y), col, 1.5f);
    dl->AddLine(ImVec2(cen.x, cen.y - 9), ImVec2(cen.x, cen.y - 4), col, 1.5f);
    dl->AddLine(ImVec2(cen.x, cen.y + 4), ImVec2(cen.x, cen.y + 9), col, 1.5f);
}


} // namespace vibe
