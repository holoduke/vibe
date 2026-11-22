// The in-game HUD: health (bottom left), weapon (bottom right), weapon
// slots (bottom centre), frame rate + clock + weather (top left).
// Drawn with ImGui draw lists in the HUD fonts loaded by init_imgui
// (ui_setup.cpp: font_hud_ / font_hud_big_, Bahnschrift from Windows).
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"

#include <imgui.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vibe {

void VulkanEngine::draw_game_hud() {
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const float W = io.DisplaySize.x, H = io.DisplaySize.y;
    const float s = glm::clamp(H / 1080.0f, 0.6f, 2.5f);
    ImFont* fs = font_hud_ ? font_hud_ : ImGui::GetFont();
    ImFont* fb = font_hud_big_ ? font_hud_big_ : ImGui::GetFont();
    const float now = static_cast<float>(glfwGetTime());

    const ImU32 kAccent = IM_COL32(255, 150, 40, 255);       // amber
    const ImU32 kPanel  = IM_COL32(10, 12, 16, 175);
    const ImU32 kDim    = IM_COL32(190, 196, 205, 220);
    const ImU32 kWhite  = IM_COL32(245, 247, 250, 255);
    auto size_of = [&](ImFont* f, float px, const char* t) { return f->CalcTextSizeA(px, 1e9f, 0.0f, t); };
    auto text = [&](ImFont* f, float px, ImVec2 p, ImU32 col, const char* t) {
        dl->AddText(f, px, ImVec2(p.x + 0.06f * px, p.y + 0.06f * px), IM_COL32(0, 0, 0, 170), t);
        dl->AddText(f, px, p, col, t);
    };
    // Letter-spaced small caps label.
    auto label = [&](ImVec2 p, ImU32 col, const char* t) {
        const float px = 17.0f * s;
        char one[2] = { 0, 0 };
        for (const char* c = t; *c; ++c) {
            one[0] = *c;
            dl->AddText(fs, px, ImVec2(p.x + 1.2f * s, p.y + 1.2f * s), IM_COL32(0, 0, 0, 190), one);
            dl->AddText(fs, px, p, col, one);
            p.x += size_of(fs, px, one).x + 2.2f * s;
        }
    };
    // A slanted dark panel with an amber edge on top. `lean` > 0: the top is shifted right.
    auto panel = [&](float x0, float y0, float x1, float y1, float lean) {
        dl->AddQuadFilled(ImVec2(x0 + lean, y0), ImVec2(x1 + lean, y0), ImVec2(x1, y1), ImVec2(x0, y1), kPanel);
        dl->AddQuadFilled(ImVec2(x0 + lean, y0), ImVec2(x1 + lean, y0), ImVec2(x1 + lean * 0.95f, y0 + 3.0f * s),
                          ImVec2(x0 + lean * 0.95f, y0 + 3.0f * s), kAccent);
    };

    const float base = H - 34.0f * s, top = H - 128.0f * s, lean = 26.0f * s;

    // ---------------- health (bottom left) ----------------
    {
        const float hp = glm::clamp(mp_health_, 0.0f, 100.0f);
        const float x0 = 34.0f * s, x1 = 372.0f * s;
        panel(x0, top, x1, base, lean);
        const bool low = hp <= 30.0f;
        const float pulse = low ? 0.65f + 0.35f * std::sin(now * 7.0f) : 1.0f;
        const ImU32 hc = low ? IM_COL32(255, 70, 55, int(255 * pulse)) : kWhite;
        // medical cross
        const float cx = x0 + 52.0f * s, cy = top + 50.0f * s, a = 15.0f * s, b = 5.5f * s;
        const ImU32 cc = low ? IM_COL32(255, 70, 55, 255) : IM_COL32(235, 60, 50, 255);
        dl->AddRectFilled(ImVec2(cx - a, cy - b), ImVec2(cx + a, cy + b), cc, 2.0f * s);
        dl->AddRectFilled(ImVec2(cx - b, cy - a), ImVec2(cx + b, cy + a), cc, 2.0f * s);
        char num[16];
        std::snprintf(num, sizeof(num), "%d", int(std::ceil(hp)));
        text(fb, 66.0f * s, ImVec2(x0 + 84.0f * s, top + 8.0f * s), hc, num);
        label(ImVec2(x0 + 224.0f * s, top + 20.0f * s), kDim, "HEALTH");
        // ten segments
        const float bx = x0 + 22.0f * s, by = base - 20.0f * s, bw = (x1 - x0 - 44.0f * s) / 10.0f;
        for (int i = 0; i < 10; ++i) {
            const float f = glm::clamp(hp / 10.0f - float(i), 0.0f, 1.0f);
            const ImVec2 p0(bx + float(i) * bw + 1.5f * s, by), p1(bx + float(i + 1) * bw - 1.5f * s, by + 8.0f * s);
            dl->AddRectFilled(p0, p1, IM_COL32(255, 255, 255, 36));
            if (f > 0.0f) dl->AddRectFilled(p0, ImVec2(p0.x + (p1.x - p0.x) * f, p1.y), low ? IM_COL32(255, 70, 55, 240) : kAccent);
        }
    }

    // ---------------- weapon (bottom right) ----------------
    {
        const float x1 = W - 34.0f * s, x0 = W - 372.0f * s;
        panel(x0, top, x1, base, -lean);
        const char* name = flashlight_on_ ? "LIGHT" : (weapon_ == 1 ? "STRIKE" : "LASER");
        const char* sub = "ENERGY";
        ImU32 sub_col = kDim;
        char subbuf[48];
        if (flashlight_on_) {
            std::snprintf(subbuf, sizeof(subbuf), "LEVEL %d / 4", std::clamp(flash_level_, 0, 3) + 1);
            sub = subbuf;
        } else if (weapon_ == 1) {
            switch (airstrike_.phase) {
            case 0:  sub = airstrike_.aim_valid ? "CLICK TO MARK" : "AIM AT GROUND"; sub_col = IM_COL32(110, 255, 160, 255); break;
            case 1:  sub = "INBOUND"; sub_col = IM_COL32(255, 190, 60, 255); break;
            case 2:  sub = "BOMB AWAY"; sub_col = IM_COL32(255, 80, 50, 255); break;
            default: sub = "STAND BY"; break;
            }
        } else if (airstrike_.phase == 1) { sub = "STRIKE INBOUND"; sub_col = IM_COL32(255, 190, 60, 255); }
        else if (airstrike_.phase == 2) { sub = "BOMB AWAY"; sub_col = IM_COL32(255, 80, 50, 255); }
        const ImVec2 ns = size_of(fb, 56.0f * s, name);
        text(fb, 56.0f * s, ImVec2(x1 - 30.0f * s - ns.x, top + 12.0f * s), kWhite, name);
        // right-aligned label
        float lw = 0.0f; { char one[2] = { 0, 0 }; for (const char* c = sub; *c; ++c) { one[0] = *c; lw += size_of(fs, 17.0f * s, one).x + 2.2f * s; } }
        label(ImVec2(x1 - 30.0f * s - lw, base - 30.0f * s), sub_col, sub);
        if (!flashlight_on_ && weapon_ == 0) {
            // unlimited charge: an infinity sign
            const float ix = x0 + 62.0f * s, iy = top + 48.0f * s, r = 11.0f * s;
            dl->AddCircle(ImVec2(ix - r, iy), r, kAccent, 24, 3.2f * s);
            dl->AddCircle(ImVec2(ix + r, iy), r, kAccent, 24, 3.2f * s);
        }
        char sc[32];
        std::snprintf(sc, sizeof(sc), "SCORE %d", score_);
        label(ImVec2(x0 + 30.0f * s, base - 30.0f * s), kDim, sc);
    }

    // ---------------- weapon bay (bottom centre) ----------------
    // Three tiles with the weapon's icon, its key in the corner and its name
    // underneath; the one in the hand is larger, framed and lit in amber.
    {
        struct Slot { const char* key; const char* name; bool on; };
        const Slot slots[3] = { { "1", "LASER", !flashlight_on_ && weapon_ == 0 }, { "2", "AIRSTRIKE", !flashlight_on_ && weapon_ == 1 },
                                { "F", "LIGHT", flashlight_on_ } };
        const float tile = 78.0f * s, big = 96.0f * s, gap = 12.0f * s;
        float total = -gap;
        for (const Slot& sl : slots) total += (sl.on ? big : tile) + gap;
        float x = (W - total) * 0.5f;
        const float bottom = base - 4.0f * s;
        const ImTextureID tex = static_cast<ImTextureID>(hud_icons_id_);
        for (int i = 0; i < 3; ++i) {
            const Slot& sl = slots[i];
            const float sz = sl.on ? big : tile;
            const ImVec2 p0(x, bottom - sz), p1(x + sz, bottom);
            const float cut = 12.0f * s;                      // cut corner, top right
            const ImVec2 pts[5] = { p0, ImVec2(p1.x - cut, p0.y), ImVec2(p1.x, p0.y + cut), p1, ImVec2(p0.x, p1.y) };
            dl->AddConvexPolyFilled(pts, 5, sl.on ? IM_COL32(40, 26, 10, 215) : kPanel);
            if (sl.on) {
                dl->AddPolyline(pts, 5, kAccent, ImDrawFlags_Closed, 2.5f * s);
                dl->AddRectFilled(ImVec2(p0.x, p1.y + 4.0f * s), ImVec2(p1.x, p1.y + 8.0f * s), kAccent);
            } else {
                dl->AddPolyline(pts, 5, IM_COL32(255, 255, 255, 40), ImDrawFlags_Closed, 1.2f * s);
            }
            const float pad = sz * 0.10f;
            if (hud_icons_id_) {
                const ImVec2 uv0(float(i) / 3.0f, 0.0f), uv1(float(i + 1) / 3.0f, 1.0f);
                dl->AddImage(tex, ImVec2(p0.x + pad + 2.0f * s, p0.y + pad + 2.0f * s), ImVec2(p1.x - pad + 2.0f * s, p1.y - pad + 2.0f * s),
                             uv0, uv1, IM_COL32(0, 0, 0, sl.on ? 150 : 90));                                    // drop shadow
                dl->AddImage(tex, ImVec2(p0.x + pad, p0.y + pad), ImVec2(p1.x - pad, p1.y - pad), uv0, uv1,
                             sl.on ? IM_COL32(255, 190, 90, 255) : IM_COL32(210, 216, 225, 150));
            } else {
                const ImVec2 ns = size_of(fs, 15.0f * s, sl.name);
                dl->AddText(fs, 15.0f * s, ImVec2(p0.x + (sz - ns.x) * 0.5f, p0.y + sz * 0.42f), sl.on ? kAccent : kDim, sl.name);
            }
            // key badge
            const float kpx = 16.0f * s;
            const ImVec2 ks = size_of(fs, kpx, sl.key);
            dl->AddRectFilled(p0, ImVec2(p0.x + ks.x + 10.0f * s, p0.y + kpx + 5.0f * s), sl.on ? kAccent : IM_COL32(255, 255, 255, 34));
            dl->AddText(fs, kpx, ImVec2(p0.x + 5.0f * s, p0.y + 1.5f * s), sl.on ? IM_COL32(20, 14, 6, 255) : kDim, sl.key);
            // name of the weapon in the hand, above its tile
            if (sl.on) {
                float lw = 0.0f; char one[2] = { 0, 0 };
                for (const char* c = sl.name; *c; ++c) { one[0] = *c; lw += size_of(fs, 17.0f * s, one).x + 2.2f * s; }
                label(ImVec2(p0.x + (sz - lw) * 0.5f, p0.y - 24.0f * s), kAccent, sl.name);
            }
            x += sz + gap;
        }
    }

    // ---------------- frame rate, clock, weather (top left) ----------------
    {
        const float x0 = 34.0f * s, y0 = 26.0f * s;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.0f", ema_fps_);
        const ImU32 fc = ema_fps_ >= 55.0f ? IM_COL32(170, 235, 90, 255) : (ema_fps_ >= 30.0f ? IM_COL32(255, 200, 70, 255) : IM_COL32(255, 80, 60, 255));
        text(fb, 40.0f * s, ImVec2(x0, y0), fc, buf);
        const float nw = size_of(fb, 40.0f * s, buf).x;
        label(ImVec2(x0 + nw + 8.0f * s, y0 + 6.0f * s), kDim, "FPS");
        std::snprintf(buf, sizeof(buf), "%.1f MS", last_frame_dt_ * 1000.0f);
        label(ImVec2(x0 + nw + 8.0f * s, y0 + 24.0f * s), IM_COL32(150, 156, 166, 210), buf);
        float y = y0 + 50.0f * s;
        if (dlss_fg_active()) {
            const float ratio = dlssg_present_ratio_ > 0.5f ? dlssg_present_ratio_ : float(dlssg_frames_sent_ + 1);
            std::snprintf(buf, sizeof(buf), "%.0f SHOWN  (FRAME GEN %.1fX)", ema_fps_ * ratio, ratio);
            label(ImVec2(x0, y), IM_COL32(120, 200, 255, 230), buf);
            y += 22.0f * s;
        }
        if (rt_.day_night_enabled || rt_.weather_enabled) {
            char w[40] = "";
            if (rt_.weather_enabled) { std::snprintf(w, sizeof(w), "%s", weather_name()); for (char* c = w; *c; ++c) *c = char(std::toupper(static_cast<unsigned char>(*c))); }
            if (rt_.day_night_enabled) {
                const int hh = int(rt_.time_of_day) % 24, mm = int((rt_.time_of_day - std::floor(rt_.time_of_day)) * 60.0f);
                std::snprintf(buf, sizeof(buf), "%02d:%02d   %s", hh, mm, w);
            } else std::snprintf(buf, sizeof(buf), "%s", w);
            label(ImVec2(x0, y), kDim, buf);
        }
    }
}

}  // namespace vibe
