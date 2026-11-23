// ui_menu.cpp - pause menu: every graphics / weather / gameplay setting
// (split out of vk_ui.cpp; see docs/ARCHITECTURE.md for the map.)
// VulkanEngine UI — ImGui setup, pause menu, HUD overlay, crosshair.
// Split out of vk_engine.cpp because UI is independent of render-state plumbing
// and grew large enough to crowd the main engine file.

#include "engine/vk_engine.h"
#include "engine/vk_engine/state/fsr3.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/vk_engine/state/dlss.h"

#include "engine/log.h"
#include "engine/vk_initializers.h"   // vibe::vk_check
#include "engine/window.h"
#include "engine/globe.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cfloat>
#include <cstdio>
#include <string>

namespace vibe {


// What an action is bound to on the device in the hands of the player: for prompts on screen.
std::string VulkanEngine::bind_label(Action a) const {
    const int i = static_cast<int>(a);
    if (in_using_pad_ && input_.pad[i] >= 0) return pad_binding_name(input_.pad[i]);
    return key_binding_name(input_.key[i]);
}

void VulkanEngine::build_menu_ui() {
    if (state_ != State::Paused) return;
    ImGuiViewport* vp = ImGui::GetMainViewport();
    // The frame rate, top left where the game shows it (its HUD is hidden behind the menu), so
    // what a setting costs is seen while it is changed.
    {
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        ImFont* fb = font_hud_big_ ? font_hud_big_ : ImGui::GetFont();
        ImFont* fs = font_hud_ ? font_hud_ : ImGui::GetFont();
        const float s = std::max(vp->Size.y / 1080.0f, 0.6f);
        const ImVec2 p0(vp->Pos.x + 34.0f * s, vp->Pos.y + 26.0f * s);
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%.0f", ema_fps_);
        const ImU32 fc = ema_fps_ >= 55.0f ? IM_COL32(170, 235, 90, 255) : (ema_fps_ >= 30.0f ? IM_COL32(255, 200, 70, 255) : IM_COL32(255, 80, 60, 255));
        const float nw = fb->CalcTextSizeA(40.0f * s, 1e9f, 0.0f, buf).x;
        dl->AddRectFilled(ImVec2(p0.x - 10.0f * s, p0.y - 6.0f * s), ImVec2(p0.x + nw + 110.0f * s, p0.y + (dlss_fg_active() ? 74.0f : 50.0f) * s),
                          IM_COL32(8, 10, 16, 170), 6.0f * s);
        dl->AddText(fb, 40.0f * s, p0, fc, buf);
        dl->AddText(fs, 17.0f * s, ImVec2(p0.x + nw + 8.0f * s, p0.y + 4.0f * s), IM_COL32(150, 156, 166, 230), "FPS");
        std::snprintf(buf, sizeof(buf), "%.1f MS", last_frame_dt_ * 1000.0f);
        dl->AddText(fs, 17.0f * s, ImVec2(p0.x + nw + 8.0f * s, p0.y + 23.0f * s), IM_COL32(150, 156, 166, 230), buf);
        if (dlss_fg_active()) {
            const float ratio = dlssg_present_ratio_ > 0.5f ? dlssg_present_ratio_ : float(dlssg_frames_sent_ + 1);
            std::snprintf(buf, sizeof(buf), "%.0f SHOWN (FRAME GEN %.1fX)", ema_fps_ * ratio, ratio);
            dl->AddText(fs, 15.0f * s, ImVec2(p0.x, p0.y + 50.0f * s), IM_COL32(120, 200, 255, 230), buf);
        }
    }
    // Cond_FirstUseEver: centred on first launch, then ImGui restores
    // whatever position / size the user left in vibe_imgui.ini.
    float max_h = vp->WorkSize.y * 0.90f;
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_FirstUseEver,
                             ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(560.0f, max_h), ImGuiCond_FirstUseEver);
    // No NoMove / NoSavedSettings — we want the user to drag the
    // menu and have ImGui persist position + open/closed state of
    // every CollapsingHeader inside it across restarts. The title
    // bar is the drag handle; "Pause Menu" is the persistent ID
    // that ImGui keys saved state against in vibe_imgui.ini.
    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_AlwaysVerticalScrollbar;
    ImGui::Begin("Pause Menu", nullptr, flags);

    // Big neon "PAUSED" banner — drawn via ImGui's foreground colour
    // override so it stands out against the indigo background.
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.20f, 0.85f, 1.0f));
    ImGui::SetWindowFontScale(1.6f);
    ImGui::TextUnformatted("// PAUSED //");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::Spacing();

    // Action buttons in a horizontal row across the top so the
    // user can always see + click them without scrolling.
    const ImVec2 btn((ImGui::GetContentRegionAvail().x - 20.0f) / 3.0f, 0.0f);
    if (ImGui::Button("Resume", btn)) {
        state_ = State::Playing;
        window_->set_cursor_captured(true);
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart", btn)) {
        reset_player();
        state_ = State::Playing;
        window_->set_cursor_captured(true);
    }
    ImGui::SameLine();
    if (ImGui::Button("Quit", btn)) {
        window_->request_close();
    }
    ImGui::Spacing();

    // ===== Tabs ============================================================
    if (!ImGui::BeginTabBar("##settings_tabs",
                              ImGuiTabBarFlags_FittingPolicyScroll)) {
        ImGui::End();
        return;
    }

    // ----- Game tab --------------------------------------------------------
    if (ImGui::BeginTabItem("Game")) {
        // First of all: what decides the frame rate (gfx_quality.cpp).
        gfx_quality_ui();
        ImGui::SeparatorText("Action");
        ImGui::Checkbox("Aircraft", &game_.aircraft_enabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enemy jets fly sorties over the map. Off: the sky is empty. Your own airstrike aircraft still comes when called.");
        if (!game_.aircraft_enabled) ImGui::BeginDisabled();
        ImGui::Checkbox("Aircraft drop bombs", &game_.aircraft_bombs);
        if (!game_.aircraft_enabled) ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Off: the jets only fly over.");
        ImGui::Checkbox("Anti-aircraft fire", &game_.aa_enabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The gun vehicles and missile launchers engage aircraft. Off: they stay parked and hold fire.");
        ImGui::Checkbox("Visible body", &game_.player_body);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("You see your own body: arms holding what you carry, chest and legs when you look down.");
        ImGui::Checkbox("Player weapons", &game_.weapons_enabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Left mouse fires the laser rifle / marks airstrikes. Off: no shooting.");
        if (ImGui::Button("Peaceful (all off)")) game_.aircraft_enabled = game_.aircraft_bombs = game_.aa_enabled = game_.weapons_enabled = false;
        ImGui::SameLine();
        if (ImGui::Button("War (all on)")) game_.aircraft_enabled = game_.aircraft_bombs = game_.aa_enabled = game_.weapons_enabled = true;
        ImGui::SeparatorText("Travel");
        // (home and the nearest places: there may be thousands)
        int travel_n = 0;
        for (int k : sites_by_distance(7)) {
            if (travel_n++ % 4 != 0) ImGui::SameLine();
            if (ImGui::Button((globe::site(k).name + "##site" + std::to_string(k)).c_str(), ImVec2(140.0f, 0.0f))) {
                travel_to_site(k);
                state_ = State::Playing;
                window_->set_cursor_captured(true);
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sets you down at that place. You can also fly there: V, then up and away.");
        ImGui::Checkbox("Show the places' names from high up", &rt_.site_labels);
        ImGui::SeparatorText("Physics and weapons");
        ImGui::SliderFloat("gravity (m/s^2)", &game_.gravity, 0.5f, 1250.0f, "%.1f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("cubes / minute", &game_.cubes_per_minute, 0, 1000);
        ImGui::SliderFloat("bullet mass (kg)",  &game_.bullet_mass, 0.5f, 200.0f,
                           "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("bullet speed (m/s)", &game_.bullet_speed, 20.0f, 3000.0f,
                           "%.0f");
        ImGui::SliderFloat("fire rate (rps)", &game_.fire_rate_rps, 0.0f, 30.0f,
                           "%.1f");
        ImGui::SliderFloat("spark scale",     &game_.spark_scale,   0.05f, 3.0f,
                           "%.2f");
        ImGui::EndTabItem();
    }

    // ----- Controls tab: view, mouse, controller, keys and buttons -----------
    // Test hook: VIBE_TEST_MENU=controls opens the pause menu on this tab (captures).
    static int test_controls_tab = [] { const char* e = std::getenv("VIBE_TEST_MENU"); return (e && e[0] == 'c') ? 3 : 0; }();
    const ImGuiTabItemFlags controls_flags = test_controls_tab > 0 ? ImGuiTabItemFlags_SetSelected : 0;
    if (test_controls_tab > 0) --test_controls_tab;
    if (ImGui::BeginTabItem("Controls", nullptr, controls_flags)) {
        ImGui::SeparatorText("View and mouse");
        ImGui::SliderFloat("Field of view", &rt_.fov_deg, 55.0f, 100.0f, "%.0f");
        ImGui::SliderFloat("Mouse sensitivity", &rt_.mouse_sens, 0.2f, 4.0f, "%.2f");
        ImGui::Checkbox("Invert mouse up / down", &input_.invert_mouse_y);

        ImGui::SeparatorText("Controller");
        {
            const char* pn = window_->gamepad_name();
            if (!input_.pad_enabled) ImGui::TextDisabled("Switched off.");
            else if (*pn) ImGui::Text("Connected: %s%s", pn, in_using_pad_ ? "  (in use)" : "  (keyboard and mouse in use)");
            else ImGui::TextDisabled("None connected. Plug one in or switch it on: it is picked up by itself.");
        }
        ImGui::Checkbox("Use a controller when there is one", &input_.pad_enabled);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Keyboard and mouse always keep working next to it.\nThe prompts on screen follow whichever you touched last.");
        ImGui::SliderFloat("Look speed (deg / s)", &input_.pad_look_speed, 40.0f, 500.0f, "%.0f");
        ImGui::SliderFloat("Up / down share", &input_.pad_look_ratio_y, 0.3f, 1.2f, "%.2f");
        ImGui::Checkbox("Invert look up / down", &input_.pad_invert_y);
        ImGui::SliderFloat("Dead zone", &input_.pad_deadzone, 0.02f, 0.5f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How far a stick must move before it counts. Raise it if the view drifts by itself.");
        ImGui::SliderFloat("Look curve (1 = straight)", &input_.pad_curve, 1.0f, 3.5f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Higher: finer aiming near the centre of the stick, the full speed only at its edge.");
        ImGui::Checkbox("Sprint with one press (until you stand still)", &input_.pad_sprint_toggle);
        ImGui::Checkbox("Swap the sticks (look left, move right)", &input_.pad_swap_sticks);
        ImGui::TextDisabled("Left stick: move.  Right stick: look.  Start: this menu.\nIn this menu: D-pad or left stick to move, A to press, B to go back.");

        ImGui::SeparatorText("Keys and buttons");
        // Waiting for a key: the first one pressed after everything was let go is taken.
        if (rebind_action_ >= 0) {
            if (ImGui::IsKeyDown(ImGuiKey_Escape)) {
                rebind_action_ = -1;
            } else {
                const int code = rebind_pad_ ? window_->any_pad_down() : window_->any_key_down();
                if (!rebind_armed_) {
                    if (code < 0) rebind_armed_ = true;
                } else if (code >= 0) {
                    int* col = rebind_pad_ ? input_.pad : input_.key;
                    // (one key, one action: whatever had it before loses it)
                    for (int i = 0; i < kActionCount; ++i) if (col[i] == code) col[i] = -1;
                    col[rebind_action_] = code;
                    rebind_action_ = -1;
                }
            }
        }
        ImGui::TextDisabled("Click one, then press the new key or button. Esc: leave as it is. Right click: none.");
        if (ImGui::BeginTable("##binds", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch, 1.4f);
            ImGui::TableSetupColumn("Keyboard / mouse", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("Controller", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableHeadersRow();
            for (int i = 0; i < kActionCount; ++i) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(action_label(static_cast<Action>(i)));
                for (int c = 0; c < 2; ++c) {
                    ImGui::TableNextColumn();
                    const bool pad = c == 1;
                    int* col = pad ? input_.pad : input_.key;
                    const bool waiting = rebind_action_ == i && rebind_pad_ == pad;
                    std::string label;
                    if (waiting) label = pad ? "press a button..." : "press a key...";
                    else if (pad) label = (col[i] < 0 && i < 4) ? "Left stick" : pad_binding_name(col[i]);
                    else label = key_binding_name(col[i]);
                    label += "##b" + std::to_string(i * 2 + c);
                    if (waiting) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.25f, 0.65f, 1.0f));
                    if (ImGui::Button(label.c_str(), ImVec2(-FLT_MIN, 0.0f))) {
                        rebind_action_ = i; rebind_pad_ = pad; rebind_armed_ = false;
                    }
                    if (waiting) ImGui::PopStyleColor();
                    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) { col[i] = -1; if (waiting) rebind_action_ = -1; }
                }
            }
            ImGui::EndTable();
        }
        if (ImGui::Button("Reset keys and buttons")) { input_.reset(); rebind_action_ = -1; }
        ImGui::EndTabItem();
    } else if (rebind_action_ >= 0) {
        rebind_action_ = -1;       // (left the tab while it was waiting)
    }

    // ----- Weather tab: time, weather, moon, sun, sky, clouds, fog ---------
    // Test hook: VIBE_TEST_MENU=weather opens the pause menu on this tab (captures).
    static int test_weather_tab = [] { const char* e = std::getenv("VIBE_TEST_MENU"); return (e && e[0] == 'w') ? 3 : 0; }();
    const ImGuiTabItemFlags weather_flags = test_weather_tab > 0 ? ImGuiTabItemFlags_SetSelected : 0;
    if (test_weather_tab > 0) --test_weather_tab;
    if (ImGui::BeginTabItem("Weather", nullptr, weather_flags)) {
        // While the clock / weather drive the world, rt_ holds the driven
        // values and env_->base the user's own: the sliders edit the latter.
        const bool driven = env_->applied;
        float&     sun_intensity = driven ? env_->base.sun_intensity : rt_.sun_intensity;
        glm::vec3& sun_color     = driven ? env_->base.sun_color : rt_.sun_color;
        glm::vec3& sky_color     = driven ? env_->base.sky_color : rt_.sky_color;
        float&     atmo_int      = driven ? env_->base.atmosphere_intensity : rt_.atmosphere_intensity;
        float&     cloud_cov     = driven ? env_->base.cloud_coverage : rt_.cloud_coverage;
        float&     cloud_dens    = driven ? env_->base.cloud_density : rt_.cloud_density;
        float&     cloud_speed   = driven ? env_->base.cloud_speed : rt_.cloud_speed;
        float&     fog_density   = driven ? env_->base.fog_density : rt_.fog_density;
        float&     sun_yaw       = driven ? env_->base.sun_yaw_deg : rt_.sun_yaw_deg;
        bool&      golden        = driven ? env_->base.auto_golden_hour : rt_.auto_golden_hour;

        if (ImGui::CollapsingHeader("Time of day", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Day / night cycle", &rt_.day_night_enabled);
            if (rt_.day_night_enabled) {
                ImGui::SliderFloat("Time of day (h)", &rt_.time_of_day, 0.0f, 24.0f, "%.2f");
                ImGui::SliderFloat("Clock speed: minutes per day", &rt_.day_length_minutes, 0.0f, 120.0f, "%.1f");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Real minutes one full 24 h day takes. 24 = one game hour per minute.\n"
                                      "Smaller is faster. 0 stops the clock at the time set above.");
                const float presets[5] = { 5.5f, 9.0f, 13.0f, 18.7f, 23.5f };
                const char* names[5] = { "Dawn", "Morning", "Noon", "Dusk", "Night" };
                for (int i = 0; i < 5; ++i) {
                    if (i) ImGui::SameLine();
                    if (ImGui::Button(names[i])) rt_.time_of_day = presets[i];
                }
            }
        }

        if (ImGui::CollapsingHeader("Weather", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Season: the look of the land. Weather type: what the sky does.
            {
                ImGui::TextUnformatted("Season");
                static const char* kSeason[4] = { "Spring", "Summer", "Autumn", "Winter" };
                const float sw = (ImGui::GetContentRegionAvail().x - 3.0f * ImGui::GetStyle().ItemSpacing.x) / 4.0f;
                for (int i = 0; i < 4; ++i) {
                    if (i) ImGui::SameLine();
                    const bool on = rt_.season == i;
                    if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                    if (ImGui::Button(kSeason[i], ImVec2(sw, 0.0f))) rt_.season = i;
                    if (on) ImGui::PopStyleColor();
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Spring: fresh green. Autumn: the leaves turn. Winter: snow lies, the lake freezes, and what falls is snow.");
                ImGui::SliderFloat("Change takes (s)", &rt_.weather_transition_s, 1.0f, 300.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("How long a change of season or weather type takes, from 1 second to 5 minutes.");
                ImGui::Spacing();
            }
            // Weather types: pick one and the sky, light, wind, rain and
            // ground move there over about twenty seconds, from whatever it
            // is now.
            {
                ImGui::TextUnformatted("Weather type");
                const float bw = (ImGui::GetContentRegionAvail().x - 2.0f * ImGui::GetStyle().ItemSpacing.x) / 3.0f;
                for (int i = 0; i < kWxPresetCount; ++i) {
                    if (i % 3) ImGui::SameLine();
                    const bool on = rt_.weather_enabled && rt_.weather_preset == i;
                    if (on) ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
                    if (ImGui::Button(kWxPresets[i].name, ImVec2(bw, 0.0f))) {
                        rt_.weather_enabled = true;
                        rt_.weather_preset = i;
                        wx_last_preset_ = -1;          // (picking the same type again turns the clock again)
                    }
                    if (on) ImGui::PopStyleColor();
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Night thunder and Pitch black turn the clock to night; the Sunny types turn it to midday. Snow leaves snow on the ground.");
                if (rt_.weather_enabled) ImGui::Text("Now: %s%s", weather_name(), (rt_.weather_preset < 0 && rt_.weather_manual < 0.0f) ? " (changing by itself)" : "");
                else ImGui::TextUnformatted("Now: weather off (always clear)");
            }
            ImGui::Separator();
            ImGui::Checkbox("Weather on", &rt_.weather_enabled);
            ImGui::SliderFloat("Mist on the water", &rt_.lake_mist, 0.0f, 3.0f, rt_.lake_mist <= 0.0f ? "off" : "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("A low layer of mist over the lake. Thickest at dawn and dusk, nearly gone at midday.");
            if (rt_.weather_enabled) {
                bool random = rt_.weather_manual < 0.0f && rt_.weather_preset < 0;
                if (ImGui::Checkbox("Changes by itself", &random)) { rt_.weather_manual = -1.0f; rt_.weather_preset = random ? -1 : 0; }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("On: the weather system picks new weather now and then. Off: the type above is held.");
                if (random) {
                    ImGui::SliderFloat("Bad weather chance", &rt_.weather_bad_chance, 0.0f, 1.0f, "%.2f");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip("How often a weather change picks rain or a storm. 0 = always fair, 1 = always bad.");
                    ImGui::SliderFloat("Weather change every (min)", &rt_.weather_change_minutes, 0.3f, 30.0f, "%.1f");
                }
            }
        }

        if (ImGui::CollapsingHeader("Moon and stars", ImGuiTreeNodeFlags_DefaultOpen)) {
            const float p = rt_.moon_phase;
            const char* nm = p < 0.03f || p > 0.97f ? "new moon" : p < 0.22f ? "waxing crescent" : p < 0.28f ? "first quarter"
                           : p < 0.47f ? "waxing gibbous" : p < 0.53f ? "full moon" : p < 0.72f ? "waning gibbous"
                           : p < 0.78f ? "last quarter" : "waning crescent";
            char fmt[48];
            snprintf(fmt, sizeof(fmt), "%%.2f  %s", nm);
            ImGui::SliderFloat("Moon phase", &rt_.moon_phase, 0.0f, 1.0f, fmt);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Like the time of day, but for the moon. 0 = new moon (dark nights), 0.5 = full moon.\n"
                                  "The phase also sets when it is up: a full moon rises at sunset,\na half moon is up for half the night.");
            ImGui::Checkbox("Moon phase advances", &rt_.moon_phases);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("One full cycle every 8 days while the clock runs.");
            ImGui::SliderFloat("Moon brightness", &rt_.moon_brightness, 0.0f, 1.0f, rt_.moon_brightness <= 0.0f ? "no moon" : "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Overall strength of the moonlight. 0 = never any moon.");
            ImGui::SliderFloat("Stars", &rt_.star_brightness, 0.0f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Star brightness (they also light the land a little on clear nights).\n"
                                  "Clouds hide them; a bright moon dims the faint ones.");
        }

        if (ImGui::CollapsingHeader("Sun", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (driven) ImGui::TextDisabled("Your own values. The clock and the weather dim and tint them through the day.");
            ImGui::SliderFloat("sun intensity", &sun_intensity, 0.0f, 8.0f);
            ImGui::ColorEdit3("sun colour", &sun_color.x);
            ImGui::Checkbox("Golden-hour tint (warm sun + sky when low)", &golden);
            ImGui::SliderFloat("sun glare / flare strength", &rt_.sun_glare_strength, 0.0f, 2.0f);
            ImGui::SliderFloat("sun shafts (god rays)", &rt_.sun_shaft_intensity, 0.0f, 2.0f);
            if (rt_.day_night_enabled) {
                ImGui::SliderFloat("sun path compass (deg)", &sun_yaw, -180.0f, 180.0f);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Turns the whole sun / moon path around the map (where east is).");
                ImGui::TextDisabled("Sun height follows the clock.");
            } else {
                ImGui::SliderFloat("sun height (deg)", &rt_.sun_pitch_deg, 5.0f, 89.0f);
                ImGui::SliderFloat("sun direction (deg)", &rt_.sun_yaw_deg, -180.0f, 180.0f);
            }
        }

        if (ImGui::CollapsingHeader("Sky and ambient colours", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("sky colour (fill light)", &sky_color.x);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Colour of the light the sky throws on everything that is not in direct sun.");
            ImGui::ColorEdit3("ground ambient", &rt_.ground_ambient.x);
            ImGui::SliderFloat("ambient strength", &rt_.ambient_strength, 0.0f, 3.0f);
            ImGui::SliderFloat("sky brightness", &atmo_int, 0.0f, 12.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Brightness of the sky dome itself.");
            ImGui::SliderFloat("distance haze reach (km)", &rt_.aerial_max_km, 0.5f, 32.0f, "%.1f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How far the blue distance haze of the atmosphere is computed.");
        }

        if (ImGui::CollapsingHeader("Clouds")) {
            bool clouds = rt_.clouds_enabled != 0;
            if (ImGui::Checkbox("Clouds", &clouds)) rt_.clouds_enabled = clouds ? 1 : 0;
            if (rt_.weather_enabled) ImGui::TextDisabled("With weather on these are the fair-weather values; bad weather adds cloud.");
            ImGui::SliderFloat("cloud cover", &cloud_cov, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("cloud thickness", &cloud_dens, 0.1f, 3.0f, "%.2f");
            ImGui::SliderFloat("cloud wind (m/s)", &cloud_speed, 0.0f, 40.0f, "%.1f");
            ImGui::SeparatorText("Shape");
            ImGui::SliderFloat("cloud size", &rt_.cloud_scale, 0.4f, 3.0f, "%.2fx");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of the cloud shapes: small puffs .. huge banks.");
            ImGui::SliderFloat("cloud base height (m)", &rt_.cloud_base_m, 500.0f, 5000.0f, "%.0f");
            ImGui::SliderFloat("tallest clouds (m)", &rt_.cloud_tower_m, 500.0f, 10000.0f, "%.0f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How far the tallest clouds tower above their base. Low = flat sheets, high = thunderheads.");
            ImGui::SliderFloat("edge softness", &rt_.cloud_softness, 0.02f, 0.6f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Low = crisp, hard-edged cumulus. High = soft, smoky edges.");
            ImGui::SliderFloat("edge detail", &rt_.cloud_erosion, 0.0f, 0.8f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much the fine cauliflower detail eats into the cloud edges.");
        }

        if (ImGui::CollapsingHeader("Fog and haze")) {
            bool vf = rt_.fog_volumetric != 0;
            if (ImGui::Checkbox("Ground fog with light shafts", &vf)) rt_.fog_volumetric = vf ? 1 : 0;
            ImGui::SliderFloat("fog thickness", &fog_density, 0.0f, 0.05f, "%.4f");
            ImGui::SliderFloat("fog thins out with height", &rt_.fog_height_falloff, 0.0f, 0.2f, "%.3f");
            ImGui::SliderFloat("fog reach (m)", &rt_.fog_max_dist, 50.0f, 1000.0f, "%.0f");
            ImGui::SeparatorText("Valley fog (on the terrain)");
            ImGui::SliderFloat("valley fog strength", &rt_.terrain_raymarch_fog_strength, 0.0f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = off, 1 = normal, 2 = thick fog lying in the valleys.");
            ImGui::SliderFloat("valley fog from height (m)", &rt_.terrain_raymarch_fog_y_start, -20.0f, 60.0f, "%.1f");
            ImGui::SliderFloat("valley fog up to height (m)", &rt_.terrain_raymarch_fog_y_top, -20.0f, 80.0f, "%.1f");
            ImGui::SliderFloat("valley fog wisps", &rt_.terrain_raymarch_fog_noise, 0.0f, 1.0f, "%.2f");
            ImGui::Checkbox("light shafts in the valley fog", &rt_.terrain_raymarch_fog_godrays);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("About 20% slower fog.");
            ImGui::SeparatorText("Distance haze");
            ImGui::SliderFloat("haze on the terrain", &rt_.terrain_fog_strength, 0.0f, 1.5f);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How strongly far hills and mountains fade into the haze.");
            ImGui::ColorEdit3("haze colour", &rt_.distance_fog_color.x);
            ImGui::SliderFloat("haze strength", &rt_.distance_fog_strength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("haze density", &rt_.distance_fog_density, 0.0f, 0.05f, "%.4f");
            ImGui::SliderFloat("haze starts at (m)", &rt_.distance_fog_start, 0.0f, 500.0f, "%.0f");
            ImGui::SliderFloat("haze height top (m)", &rt_.distance_fog_height, 0.0f, 400.0f, "%.0f");
            ImGui::SliderFloat("haze max", &rt_.distance_fog_max, 0.0f, 1.0f, "%.2f");
            ImGui::TextDisabled("strength 0 disables it. height top 0 = the same at every height.");
        }
        ImGui::EndTabItem();
    }

    // ----- Graphics tab ----------------------------------------------------
    if (ImGui::BeginTabItem("Graphics")) {
        if (ImGui::CollapsingHeader("Resolution & Quality", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Quality preset combo. Switching writes a bundle of values onto rt_;
        // tweaking any individual slider afterward marks preset = -1 (custom).
        const char* kPresetLabels[] = {
            "Low (0.65× res, 8spp shadows, GTAO, 16 GI)",
            "Medium (0.85× res, 16spp, GTAO, 32 GI)",
            "High (1.0× res, 24spp, GTAO, 64 GI)",
            "Ultra (1.25× res, 40spp, RTAO, 96 GI / 2-bounce)"
        };
        int preset = std::clamp(rt_.quality_preset, -1, 3);
        const char* current_label = (preset >= 0 && preset <= 3)
            ? kPresetLabels[preset] : "Custom (sliders edited)";
        if (ImGui::BeginCombo("preset", current_label)) {
            for (int i = 0; i < 4; ++i) {
                bool is_selected = (preset == i);
                if (ImGui::Selectable(kPresetLabels[i], is_selected)) {
                    apply_quality_preset(i);
                    apply_render_scale();
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        // Render-scale slider — 0.5..2.0. Apply re-allocates targets.
        // Drag changes only the value; you have to click Apply (or change
        // window size) for it to take effect, because resizing live render
        // targets requires a wait-idle.
        ImGui::SliderFloat("render scale", &rt_.render_scale,
                           (rt_.fsr_backend == 2 && rt_.fsr2_enabled && dlss_->supported) ? 0.5f : 0.4f, 2.0f, "%.2fx",
                           ImGuiSliderFlags_AlwaysClamp);
        ImGui::SameLine();
        if (ImGui::Button("Apply")) {
            apply_render_scale();
        }
        ImGui::TextDisabled("0.5× = half-res (huge speedup), 1.5×+ = SSAA");

        // Upscale presets — combine render_scale + TAAU + sharpening
        // into a single click. Each one mirrors the way a vendor
        // upscaler exposes Quality / Balanced / Performance modes.
        ImGui::TextUnformatted("Upscale preset:");
        ImGui::SameLine();
        auto apply_upscale = [&](float scale, bool taau_on, float sharp) {
            rt_.render_scale            = scale;
            rt_.taau_enabled            = taau_on;
            rt_.compose_sharpen_strength = sharp;
            rt_.quality_preset           = -1;
            apply_render_scale();
        };
        if (ImGui::Button("Native"))  apply_upscale(1.00f, false, 0.30f);
        ImGui::SameLine();
        if (ImGui::Button("Quality")) apply_upscale(0.77f, true,  0.80f);
        ImGui::SameLine();
        if (ImGui::Button("Balanced")) apply_upscale(0.67f, true, 1.10f);
        ImGui::SameLine();
        if (ImGui::Button("Perf"))    apply_upscale(0.50f, true,  1.40f);
        ImGui::TextDisabled("Quality/Balanced/Perf use TAAU + extra sharpen.");

        // AO mode combo. HBAO (mode 3) folds a horizon-angle formula into
        // the inline-ray loop -- each AO direction marches multiple short
        // taps and keeps the steepest occluder rather than a binary hit
        // test, which captures crevice / contact shadows the binary RTAO
        // misses without paying the full RTAO sample-count cost.
        const char* kAoLabels[] = { "off", "GTAO (screen-space, fast)",
                                    "RTAO (true ray-traced, slow)",
                                    "HBAO (horizon-based, contact AO)" };
        int ao = std::clamp(rt_.ao_mode, 0, 3);
        if (ImGui::BeginCombo("AO mode", kAoLabels[ao])) {
            for (int i = 0; i < 4; ++i) {
                bool is_selected = (ao == i);
                if (ImGui::Selectable(kAoLabels[i], is_selected)) {
                    rt_.ao_mode = i;
                    rt_.quality_preset = -1; // custom
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        }   // end CollapsingHeader Resolution & Quality

        if (ImGui::CollapsingHeader("Graphics Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SeparatorText("Materials");
        ImGui::Checkbox("textures enabled", &rt_.textures_enabled);

        ImGui::SeparatorText("Shadows (RT, contact-hardening)");
        if (ImGui::Button("Recalculate shadows")) shadow_rebake_request_ = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Re-bakes the precomputed ground shadow and ambient-occlusion maps\n"
                              "and the terrain ray-tracing geometry. Live shadows update every frame anyway.");
        ImGui::Checkbox("enabled", &rt_.shadow_enabled);
        ImGui::Checkbox("half-rate shadow pass (perf)",
                        &rt_.half_rate_shadows);
        ImGui::TextDisabled(
            "Trace the PCSS sun shadow at half res + bilateral\n"
            "upsample instead of per-pixel. Big perf win; OFF =\n"
            "the proven inline path.");
        ImGui::SliderInt("samples", &rt_.shadow_samples, 1, 128);
        ImGui::SliderFloat("near samples multiplier", &rt_.shadow_near_mult, 1.0f, 8.0f, "%.2fx");
        ImGui::SliderFloat("softness", &rt_.shadow_softness, 0.0f, 0.15f);
        ImGui::SliderFloat("strength", &rt_.shadow_strength, 0.0f, 1.0f);
        ImGui::SliderFloat("curve (linear→expo)", &rt_.shadow_curve, 0.0f, 1.0f);

        ImGui::SeparatorText("Castle materials (SPOM)");
        // Per-pixel parallax depth on castle walls (and any future
        // SPOM-eligible materials). 0 = flat, 1 = engine default
        // (~4 cm peak-to-trough at 1 m view), 2.5 = exaggerated
        // bricks. At 0 the SPOM march is short-circuited entirely so
        // there's no GPU cost for users who want flat textures.
        ImGui::SliderFloat("SPOM strength", &rt_.spom_strength, 0.0f, 2.5f, "%.2f");
        // Displacement technique on castle walls. Legacy SPOM = original
        // per-pixel march + ray-query silhouette probe (heavy, flat-ish
        // corners). SSDM = same march but writes recessed depth to
        // gl_FragDepth for a real relief silhouette with no ray query
        // (cheaper). Both honour the strength slider above.
        // 0 = Flat (no parallax at all -- bricks look painted on);
        // 1 = Legacy SPOM (ray-query silhouette resolution);
        // 2 = SSDM (depth-override silhouette + recessed depth write).
        // Flat is included so the user can A/B against either technique
        // and see how much relief either is adding.
        const char* spom_mode_labels[] = { "Flat (no parallax)",
                                            "SPOM (ray-query)",
                                            "SSDM (depth-override)",
                                            "POM (curved silhouette)" };
        ImGui::Combo("Wall displacement", &rt_.spom_mode, spom_mode_labels,
                     IM_ARRAYSIZE(spom_mode_labels));

        ImGui::SeparatorText("Sun shadow map (grass receiver)");
        // Resolution dropdown — apply triggers a vkDeviceWaitIdle + image
        // recreation in the next frame's tick. 4096 doubles VRAM, halves
        // texel size; 512 quarters fragment cost vs 1024.
        const char* res_labels[] = { "512", "1024", "2048", "4096", "8192" };
        int res_idx = 1;
        if      (rt_.shadow_map_resolution <= 512)  res_idx = 0;
        else if (rt_.shadow_map_resolution <= 1024) res_idx = 1;
        else if (rt_.shadow_map_resolution <= 2048) res_idx = 2;
        else if (rt_.shadow_map_resolution <= 4096) res_idx = 3;
        else                                         res_idx = 4;
        if (ImGui::Combo("shadow map resolution", &res_idx, res_labels, 5)) {
            const int values[] = { 512, 1024, 2048, 4096, 8192 };
            rt_.shadow_map_resolution = values[res_idx];
        }
        // Extended upper bound to 2000 m for the 1500 m visible-distance
        // setup -- at the previous 400 m cap the single ortho cascade was
        // sampling at <1 texel/m past 400 m, giving heavy aliasing on
        // distant ridge shadows. The texels-per-metre is W/(2*half), so
        // at 2000 m and a 4k shadow map the ratio is ~1 texel/m which
        // matches the terrain LOD detail past 800 m. Below 400 m the
        // slider behaves identically to before.
        ImGui::SliderFloat("shadow map world size (m)",
                            &rt_.shadow_map_world_half, 30.0f, 2000.0f);
        {
            // Texel density warning -- past ~800 m world half at a 1024
            // shadow map the per-metre resolution drops below 1 texel/m
            // and ridge shadows alias. Surface a live readout + a soft
            // hint to bump the resolution combo when the slider goes
            // wide. Number stays as a tooltip so the layout doesn't
            // shift on every slider tick.
            const float texels_per_m =
                static_cast<float>(rt_.shadow_map_resolution) /
                (2.0f * std::max(1.0f, rt_.shadow_map_world_half));
            if (rt_.shadow_map_world_half > 800.0f &&
                rt_.shadow_map_resolution < 4096) {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                    "  ~%.2f texels/m -- bump resolution to 4096+ to "
                    "avoid ridge aliasing", texels_per_m);
            } else if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Texel density: %.2f texels/m "
                    "(resolution %d / (2 x %.0f m)).\n"
                    "Below ~1 texel/m distant ridges alias -- raise "
                    "resolution or lower the world size.",
                    texels_per_m, rt_.shadow_map_resolution,
                    rt_.shadow_map_world_half);
            }
        }
        ImGui::Checkbox("debug overlay (frustum + bake bounds)",
                         &rt_.shadow_debug_overlay);

        ImGui::SeparatorText("Ambient Occlusion (RT)");
        ImGui::SliderInt("AO samples (0=off)", &rt_.ao_samples, 0, 64);
        ImGui::SliderFloat("AO radius", &rt_.ao_radius, 0.1f, 8.0f);
        ImGui::SliderFloat("AO floor (corner-pile-up cap)",
                            &rt_.ao_floor, 0.0f, 1.0f);

        ImGui::SeparatorText("Tone / Exposure / Contrast / Brightness");
        ImGui::SliderFloat("Auto-exposure (eye adaptation)",
                            &rt_.auto_exposure_strength, 0.0f, 1.5f);
        ImGui::SliderFloat("image contrast",
                            &rt_.image_contrast,   0.5f, 2.0f);
        ImGui::SliderFloat("image brightness",
                            &rt_.image_brightness, 0.3f, 2.5f);
        ImGui::SliderFloat("image gamma (curve)",
                            &rt_.image_gamma,      0.4f, 2.5f);

        ImGui::SeparatorText("Colour grading");
        ImGui::SliderFloat("white balance (cool - warm)", &rt_.grade_temperature, -1.0f, 1.0f);
        ImGui::SliderFloat("saturation", &rt_.grade_saturation, 0.0f, 2.0f);
        ImGui::SliderFloat("vibrance (muted colours)", &rt_.grade_vibrance, 0.0f, 1.0f);
        ImGui::SliderFloat("split toning (teal shadows / warm highlights)", &rt_.grade_split_tone, 0.0f, 1.0f);
        ImGui::SliderFloat("film grain", &rt_.film_grain, 0.0f, 1.0f);
        ImGui::SliderFloat("vignette", &rt_.vignette_strength, 0.0f, 1.0f);

        ImGui::SeparatorText("Path-traced GI (multi-bounce)");
        ImGui::SliderInt("GI samples (0=off)", &rt_.gi_samples, 0, 128);
        // ReSTIR-lite (session 1 of multi-session plan — see
        // docs/restir_plan.md). Toggle quarters gi_samples at runtime;
        // the existing TAA + spatial filter smooths the noise. Sessions
        // 2+ replace the / 4 hack with proper reservoir reuse.
        ImGui::Checkbox("ReSTIR-lite (samples / 4, lean on TAA)",
                          &rt_.gi_restir_enabled);
        ImGui::SliderInt("GI bounces", &rt_.gi_bounces, 1, 5);
        ImGui::SliderInt("GI shadow bounces (sun shadow on bounce hits)",
                          &rt_.gi_shadow_max_bounce, 0, 5);
        ImGui::SliderFloat("GI strength", &rt_.gi_strength, 0.0f, 3.0f);
        ImGui::SliderFloat("GI radius", &rt_.gi_radius, 1.0f, 400.0f);
        ImGui::SliderFloat("GI softener (raymarch terrain)",
                          &rt_.gi_softener, 0.0f, 1.0f);
        ImGui::Checkbox("GI debug viz (raymarch terrain)",
                          &rt_.gi_debug_viz);
        ImGui::SliderFloat("RT LOD distance (raymarch terrain, m)",
                          &rt_.terrain_rt_lod_distance, 50.0f, 1000.0f);
        ImGui::SliderFloat("AO darkness (raymarch terrain)",
                          &rt_.terrain_ao_punch, 0.5f, 3.0f);

        ImGui::SeparatorText("Raymarched terrain RT (per-pixel caps)");
        ImGui::TextDisabled("Higher = better visuals + heavier GPU. "
                            "Lower if you hit GPU TDR / device_lost.");
        ImGui::SliderInt("PCSS shadow rays cap (terrain)",
                          &rt_.terrain_pcss_samples_cap, 2, 16);
        ImGui::SliderInt("GI rays cap (terrain)",
                          &rt_.terrain_gi_samples_cap, 0, 16);
        ImGui::SliderInt("GI bounces cap (terrain)",
                          &rt_.terrain_gi_bounces_cap, 1, 3);
        ImGui::SliderFloat("AO final strength (terrain)",
                          &rt_.terrain_ao_final_strength, 0.0f, 1.0f);
        ImGui::SliderFloat("contrast (raymarch terrain)",
                          &rt_.terrain_rm_contrast, 0.5f, 2.5f);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("1 = neutral. Higher deepens the dark parts of the ground and lifts the bright ones.");

        ImGui::SeparatorText("Anti-aliasing (TAA + Halton sub-pixel jitter)");
        ImGui::Checkbox("sub-pixel jitter", &rt_.taa_jitter_enabled);
        ImGui::SliderFloat("jitter strength", &rt_.taa_jitter_strength, 0.0f, 2.0f);
        ImGui::SliderFloat("history blend", &rt_.taa_history_blend, 0.0f, 0.98f);
        ImGui::Checkbox("TAAU (temporal upsample)", &rt_.taau_enabled);
        ImGui::TextDisabled("Adds a temporal upscale pass after TAA. Best with\n"
                             "render_scale < 1.0 — renders at LR, reconstructs\n"
                             "at native via motion-vector reproject + variance\n"
                             "clamp. ~1.5x perf gain at render_scale=0.67.");
        // FSR2 — phased rollout. Phase 1 (foundation) is shipped:
        // Halton(2,3) jitter with sample-count-aware phase, FrameView
        // jitter capture. Phases 2-5 (compute passes for prepare /
        // depth-clip / reconstruct-prev-depth / lock / accumulate /
        // RCAS) are pending. Toggle currently only switches the
        // jitter sequence; the actual upscale path falls back to
        // FSR1 EASU+RCAS in compose.frag until Phase 5 lands.
        ImGui::Checkbox("FSR upscale (renders at render_scale, upscales to native)",
                        &rt_.fsr2_enabled);
        // Backend select — only meaningful when FSR upscale is on. FSR3
        // uses the new ffx-api ABI through the prebuilt vendored DLL;
        // FSR2 uses the legacy static-lib path. Frame generation (FSR3
        // sessions 3-6) is NOT yet wired — this only swaps the upscaler.
        {
            static const char* kVendor[] = { "unknown", "NVIDIA", "AMD", "Intel", "other" };
            ImGui::Text("GPU: %s (%s)", gpu_name_.c_str(), kVendor[int(gpu_vendor_)]);
            ImGui::SliderFloat("Trees: draw distance (m)", &rt_.tree_draw_distance, 200.0f, 3800.0f, "%.0f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Trees and bushes beyond this are not drawn. 3800: as far as the view distance allows.");
            ImGui::SliderFloat("Trees: card below this size (px)", &rt_.tree_impostor_px, 7.0f, 64.0f, "%.0f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Far trees are drawn as one baked picture of themselves instead of their mesh.\nHigher: more trees are cards (faster). 7: only the very far ones, as before.");
            ImGui::Checkbox("Auto-pick upscaler for this GPU (at start-up)", &rt_.upscaler_auto);
            {
                const char* fr_labels[] = { "Exact (slow in forests)", "Exact sun shadows nearby (default)", "Solid leaf cards for all rays (fastest)" };
                ImGui::Combo("Foliage rays", &rt_.foliage_ray_mode, fr_labels, 3);
                ImGui::Checkbox("Lighter bounce light on leaves", &rt_.gi_leaf_lite);
            }
            if (ImGui::SliderFloat("FSR: least render scale", &rt_.fsr_min_render_scale, 0.5f, 1.0f, "%.2f") && rt_.fsr_backend == 1)
                apply_render_scale();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("NVIDIA: DLSS + Ray Reconstruction + DLSS Frame Generation + Reflex.\nAMD / Intel: FSR 3.1 + FSR Frame Generation (+ AMD Anti-Lag 2).");
            if (anti_lag_supported_) ImGui::Checkbox("AMD Anti-Lag 2", &rt_.anti_lag);
        }
        const char* fsr_backends[] = { "FSR2 (legacy static)", "FSR3 (ffx-api DLL)", "DLSS (NVIDIA Streamline)" };
        ImGui::Combo("Upscaler backend", &rt_.fsr_backend, fsr_backends,
                     IM_ARRAYSIZE(fsr_backends));
        // DLSS needs a render scale of at least 0.50 (apply_render_scale clamps).
        if (rt_.fsr_backend == 2 && rt_.fsr2_enabled && dlss_->supported && rt_.render_scale < 0.5f) apply_render_scale();
        if (rt_.fsr_backend == 2) {
            ImGui::BeginDisabled(!dlss_->rr_supported);
            ImGui::Checkbox("Ray Reconstruction (DLSS-RR)", &rt_.dlss_rr);
            ImGui::Checkbox("RR: 1-sample ray tracing", &rt_.dlss_rr_low_samples);
            ImGui::EndDisabled();
            ImGui::TextDisabled("%s", dlss_->rr_supported
                ? "AI denoiser + upscaler for the ray-traced lighting.\nReplaces DLSS Super Resolution while on."
                : "DLSS Ray Reconstruction is not available on this GPU / driver.");
            // DLSS Frame Generation: 1-3 AI frames between rendered ones.
            {
                const char* fg_modes[] = { "Off", "2x (1 generated frame)", "3x (2 generated)", "4x (3 generated)" };
                const int max_mode = dlssg_loaded_ ? std::min(3, int(dlssg_max_frames_)) : 3;
                int m = std::clamp(rt_.dlss_fg, 0, 3);
                if (ImGui::BeginCombo("DLSS Frame Generation", fg_modes[m])) {
                    for (int i = 0; i <= 3; ++i) {
                        ImGui::BeginDisabled(dlssg_loaded_ && i > max_mode);
                        if (ImGui::Selectable(fg_modes[i], m == i)) rt_.dlss_fg = i;
                        ImGui::EndDisabled();
                    }
                    ImGui::EndCombo();
                }
                if (dlssg_loaded_) {
                    ImGui::TextDisabled("%s", dlss_fg_active()
                        ? "Frame Generation running (Reflex low latency on, VSync off)."
                        : "Loaded - turns on in gameplay with DLSS active.");
                } else if (rt_.dlss_fg > 0) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Restart the game to load Frame Generation.");
                } else {
                    ImGui::TextDisabled("Doubles+ the displayed frame rate. Needs an RTX 40/50 GPU; loads at start-up.");
                }
            }
        }
        if (fsr3_->fatal) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                "FSR3 unavailable on this device (CreateContext AVed in SDK DLL "
                "— see vibe.log). Backend auto-reverted to FSR2.");
        }
        // FSR3 Frame Generation. Only meaningful when FSR3 backend is
        // active. Session 3 dispatches FG to a private image that
        // isn't presented yet — toggle visibly costs GPU time but
        // produces no perceptible delta. Session 4 wires the present
        // chain so this becomes the FPS-doubling switch.
        if (rt_.fsr_backend == 1 && !fsr3_->fatal) {
            ImGui::Checkbox("FSR3 Frame Generation (perceived FPS ×2)",
                            &rt_.fg_enabled);
            if (fsr3_->fg_fatal) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                    "FG unavailable (init failed) — see vibe.log");
            } else if (rt_.fg_enabled && fg_runtime_disabled_) {
                ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f),
                    "FG auto-disabled (render < 50 fps base — interpolation "
                    "judder dominates below this; will re-enable above 60).");
            } else if (rt_.fg_enabled && fsr3_->swapchain_active) {
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.5f, 1.0f),
                    "FG active — SDK pacing real + generated frames");
            }
        }
        ImGui::TextDisabled("Phase 1/5 — Halton(2,3) jitter active.\n"
                             "Compute passes pending in future sessions.");
        ImGui::SliderFloat("spatial strength", &rt_.taa_spatial_strength, 0.0f, 1.0f);
        // Compose-pass unsharp mask — runs in compose.frag (5 texelFetches),
        // recovers detail the TAA spatial filter softened. 0 disables; 0.55
        // is the default; >1 looks deliberately punchy.
        ImGui::SliderFloat("post-TAA sharpening", &rt_.compose_sharpen_strength, 0.0f, 2.0f);
        // 4a-deep: enables the variance-moments temporal pass plus the
        // 3-pass a-trous spatial filter. Off by default (perf cost is
        // ~1.5 ms at 1080p); when on, TAA's own single-pass spatial is
        // suppressed (engine zeroes spatial_strength on the TAA UBO
        // when this checkbox is set) so the two filters don't
        // double-blur.
        ImGui::Checkbox("SVGF GI denoiser (4a-deep: moments + 3-pass a-trous)",
                        &rt_.svgf_enabled);

        ImGui::SeparatorText("Bloom (compose-pass spiral-tap)");
        ImGui::Checkbox("bloom enabled", &rt_.bloom_enabled);
        ImGui::SliderFloat("bloom strength", &rt_.bloom_strength, 0.0f, 2.0f);
        ImGui::SliderFloat("bloom threshold", &rt_.bloom_threshold, 0.5f, 3.0f);
        ImGui::SliderFloat("bloom radius (px)", &rt_.bloom_radius, 4.0f, 80.0f);
        ImGui::SliderFloat("spark bloom", &rt_.spark_bloom, 0.0f, 5.0f);

        ImGui::SeparatorText("Lens flare (Chapman screen-space)");
        ImGui::Checkbox("flare enabled", &rt_.lens_flare_enabled);
        ImGui::SliderFloat("flare strength",   &rt_.lens_flare_strength,   0.0f, 2.0f);
        ImGui::SliderFloat("flare threshold",  &rt_.lens_flare_threshold,  0.0f, 4.0f);
        ImGui::SliderFloat("ghost dispersal",  &rt_.lens_flare_dispersal,  0.05f, 0.6f);
        ImGui::SliderFloat("halo width",       &rt_.lens_flare_halo_width, 0.0f, 0.5f);
        ImGui::SliderInt  ("ghost count",      &rt_.lens_flare_ghosts,     1, 8);
        ImGui::SliderFloat("aberration",       &rt_.lens_flare_aberration, 0.0f, 0.02f, "%.4f");

        if (ImGui::Button("Reset to defaults")) {
            const float old_scale = rt_.render_scale;
            const float old_tr_scale = rt_.terrain_raymarch_scale;
            rt_ = RtSettings{};
            // Rebuild what the defaults resize (same as the Apply / preset paths).
            if (std::abs(rt_.render_scale - old_scale) > 1e-3f) apply_render_scale();
            if (std::abs(rt_.terrain_raymarch_scale - old_tr_scale) > 1e-3f)
                recreate_terrain_raymarch_lowres();
        }
        ImGui::SameLine();
        if (ImGui::Button("Low")) {
            rt_.shadow_samples = 1;
            rt_.shadow_softness = 0.0f;
            rt_.ao_samples = 0;
            rt_.gi_samples = 0;
            rt_.gi_bounces = 1;
        }
        ImGui::SameLine();
        if (ImGui::Button("High")) {
            rt_.shadow_samples = 4;  rt_.shadow_softness = 0.025f;
            rt_.ao_samples = 8;      rt_.ao_radius = 1.5f;
            rt_.gi_samples = 4;      rt_.gi_strength = 1.0f;
            rt_.gi_radius = 60.0f;   rt_.gi_bounces = 1;
        }
        ImGui::SameLine();
        if (ImGui::Button("Ultra")) {
            rt_.shadow_samples = 8;  rt_.shadow_softness = 0.04f;
            rt_.ao_samples = 16;     rt_.ao_radius = 2.5f;
            rt_.gi_samples = 8;      rt_.gi_strength = 1.0f;
            rt_.gi_radius = 100.0f;  rt_.gi_bounces = 2;
        }
        ImGui::SameLine();
        if (ImGui::Button("Insane")) {
            // The all-in knob — pushes shadow samples beyond the conservative
            // defaults; on an RTX 4080 it should still hold ~60 fps at 720p.
            rt_.shadow_samples = 16; rt_.shadow_softness = 0.06f;
            rt_.ao_samples = 32;     rt_.ao_radius = 4.0f;
            rt_.gi_samples = 16;     rt_.gi_strength = 1.0f;
            rt_.gi_radius = 200.0f;  rt_.gi_bounces = 4;
        }
        }   // end CollapsingHeader Graphics Settings
        ImGui::EndTabItem();
    }

    // ----- Terrain tab (terrain renderer + grass) -------------------------
    if (ImGui::BeginTabItem("Terrain")) {
        if (ImGui::CollapsingHeader("Terrain", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SeparatorText("Renderer");
        // Mode switch is always visible; everything below adapts to it
        // so you only see the settings for the selected terrain path.
        if (ImGui::Checkbox("Procedural raymarched terrain (FBM)",
                            &rt_.terrain_raymarch_enabled)) {
            // Mesh(water-only) wants full-res, FBM wants the scaled LR
            // buffer — resize when the mode flips.
            recreate_terrain_raymarch_lowres();
        }
        ImGui::TextDisabled("Switches the whole terrain renderer. Toggling\n"
                             "lazy-builds raster mesh + grass on demand.\n"
                             "Terrain BLAS (RT shadows on terrain) is built once\n"
                             "at load — toggling off mid-session means no RT\n"
                             "terrain shadow contact until restart.");

        if (rt_.terrain_raymarch_enabled) {
        ImGui::SeparatorText("FBM raymarch");
        ImGui::SliderInt("Raymarch steps",
                         &rt_.terrain_raymarch_max_steps, 60, 300);
        ImGui::SliderInt("Raymarch shadow steps",
                         &rt_.terrain_raymarch_shadow_steps, 16, 96);
        ImGui::SliderInt("FBM octaves (march)",
                         &rt_.terrain_raymarch_octaves, 4, 24);
        ImGui::SliderInt("FBM octaves (normals)",
                         &rt_.terrain_raymarch_normal_octaves, 4, 32);
        ImGui::SliderFloat("Step factor",
                           &rt_.terrain_raymarch_step_factor, 0.15f, 0.8f, "%.2f");
        // Distance-LOD ramp — drops FBM octaves with ray distance so the
        // expensive 3× terrainH() calls in calcNormal collapse to the
        // floor count past `lod_far_m`. Biggest perf win at horizon
        // views where most rays travel >500m.
        ImGui::SliderFloat("LOD near (m)",
                           &rt_.terrain_raymarch_lod_near_m, 20.0f, 400.0f, "%.0f");
        ImGui::SliderFloat("LOD far (m)",
                           &rt_.terrain_raymarch_lod_far_m, 200.0f, 2000.0f, "%.0f");
        ImGui::SliderInt("LOD min octaves",
                         &rt_.terrain_raymarch_lod_min_octaves, 2, 8);
        // Render-scale for the raymarch only — biggest single perf
        // knob. 0.5 = quarter the FBM evaluations per frame; bilinear
        // upscale + depth-aware composite hides most of the softness
        // because terrain noise is low-frequency anyway.
        const char* tr_scale_labels[] = { "100% (native)", "75%", "50%", "33%", "25%" };
        const float tr_scale_values[] = { 1.0f, 0.75f, 0.5f, 0.333f, 0.25f };
        int tr_scale_idx = 0;
        for (int i = 0; i < 5; ++i)
            if (std::abs(rt_.terrain_raymarch_scale - tr_scale_values[i]) < 0.02f) {
                tr_scale_idx = i; break;
            }
        if (ImGui::Combo("Raymarch render scale", &tr_scale_idx,
                          tr_scale_labels, 5)) {
            rt_.terrain_raymarch_scale = tr_scale_values[tr_scale_idx];
            recreate_terrain_raymarch_lowres();
        }
        ImGui::SliderFloat("Raymarch sharpen", &rt_.terrain_raymarch_sharpen,
                           0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Relaxation cone-stepping (Phase 6)",
                         &rt_.terrain_raymarch_relaxation);
        ImGui::SeparatorText("Speed (same picture, less work)");
        ImGui::Checkbox("Low-resolution march first", &rt_.terrain_rm_prepass);
        {
            const char* div_labels[] = { "2 x 2 pixels", "3 x 3 pixels", "4 x 4 pixels" };
            int div_idx = std::clamp(rt_.terrain_rm_prepass_div, 2, 4) - 2;
            if (ImGui::Combo("Block size of that march", &div_idx, div_labels, 3)) {
                rt_.terrain_rm_prepass_div = div_idx + 2;
                recreate_terrain_raymarch_lowres();
            }
        }
        ImGui::SliderFloat("Fast final approach", &rt_.terrain_rm_converge, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Normal detail limit (pixels)", &rt_.terrain_rm_normal_px, 0.0f, 4.0f, "%.1f");
        ImGui::Checkbox("Simpler relief under dense grass", &rt_.terrain_rm_relief_lite);
        ImGui::Checkbox("Sun shadows at low resolution (needs the march first)", &rt_.terrain_rm_shade_lo);
        ImGui::SeparatorText("Strategies to try");
        ImGui::Checkbox("Start each pixel at the low-resolution hit (needs the march first)", &rt_.terrain_rm_start_at_hit);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The low-resolution march goes on to the ground itself; where its blocks agree,\na pixel starts just short of that and has only the last steps to do.\nSame picture; the low-resolution pass costs more, the main pass less.");
        if (ImGui::Checkbox("Temporal reconstruction (with Raymarch scale below 100 %)", &rt_.terrain_rm_temporal))
            recreate_terrain_raymarch_lowres();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("The low-resolution terrain is offset by half a pixel from frame to frame\nand each frame is put together with the one before (checkerboard):\ntwice the detail of the scale alone when the view is still or moves slowly.");
        ImGui::TextDisabled("Full resolution: scale 100 %%.  Low resolution, sharpened: scale below 100 %%.");
        {
            const char* vrs_labels[] = { "Every pixel", "2x2 toward the screen edges", "2x2 edges, 4x4 corners" };
            if (ImGui::Combo("Ground shading rate", &rt_.terrain_vrs, vrs_labels, 3)) recreate_vrs_attachment();
        }
        ImGui::TextDisabled("%s",
            "March first: one ray per block of pixels walks to the ground,\n"
            "every pixel then starts its own march there.\n"
            "Fast approach: 0 = the old small steps close over the ground.\n"
            "Normal limit: 0 = keep detail finer than a pixel in the lighting.");
        // TextDisabled is printf-style; passing "%s" + a string is the
        // safe way to display literal `%` characters (e.g. "50%").
        ImGui::TextDisabled("%s",
            "Lower steps / fewer octaves / higher step factor = faster.\n"
            "50% render scale = 4x fewer FBM evaluations.\n"
            "Sharpen recovers detail lost to bilinear upscaling.\n"
            "Relaxation: faster march at the risk of skipping spikes.\n"
            "(Fog and haze settings are on the Weather tab.)");
        } else {
        ImGui::SeparatorText("Quadtree CLOD terrain");
        ImGui::SliderFloat("Detail range (m)",
                           &rt_.terrain_clod_range0, 20.0f, 200.0f, "%.0f");
        ImGui::TextDisabled("%s",
            "Leaf-node subdivision ring. Each LOD ring doubles this.\n"
            "Bigger = dense 0.25 m triangles reach further out\n"
            "(more triangles, still one instanced draw).");
        ImGui::Checkbox("Wireframe (show triangles / mesh density)",
                         &rt_.terrain_wireframe);
        }   // end FBM / mesh renderer split

        ImGui::SeparatorText("Ocean / water");
        ImGui::Checkbox("Water enabled", &rt_.water_enabled);
        ImGui::SliderFloat("Water level (Y)",
                           &rt_.water_level, -10.0f, 50.0f, "%.1f m");
        ImGui::SliderFloat("Wave strength",
                           &rt_.water_wave_strength, 0.0f, 0.6f, "%.3f");
        ImGui::SliderFloat("Wave scale (freq mult)",
                           &rt_.water_wave_scale, 0.1f, 5.0f, "%.2f");
        // Water style dropdown — picks which normal+foam generator.
        // Each style brings its own visual character; the engine's
        // depth / extinction / reflection / shadow stack stays
        // shared so the user keeps their colour + shore controls.
        const char* k_water_styles[] = {
            "Default (sum of sines)",
            "River (P_Malin — advected FBM)",
            "Lake (distance-faded bumps)",
        };
        ImGui::Combo("Water style", &rt_.water_style,
                     k_water_styles, IM_ARRAYSIZE(k_water_styles));
        if (rt_.water_style == 1) {
            ImGui::SliderFloat("river flow speed",
                               &rt_.water_river_speed, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("river normal strength",
                               &rt_.water_river_normal_str, 0.0f, 3.0f, "%.2f");
            ImGui::SliderFloat("river flow angle (deg)",
                               &rt_.water_river_flow_angle, -180.0f, 180.0f, "%.0f");
            ImGui::SliderFloat("river time speed",
                               &rt_.water_river_time_speed, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("river detail scale",
                               &rt_.water_river_detail, 0.1f, 4.0f, "%.2f");
            ImGui::SliderFloat("river foam amount",
                               &rt_.water_river_foam_amount, 0.0f, 3.0f, "%.2f");
            ImGui::ColorEdit3("river underwater tint",
                              &rt_.water_river_extinct_color.x);
            ImGui::SliderFloat("river underwater density",
                               &rt_.water_river_extinct_density, 0.0f, 4.0f, "%.2f");
        } else if (rt_.water_style == 2) {
            ImGui::SliderFloat("lake bump strength",
                               &rt_.water_lake_bump_strength, 0.0f, 1.5f, "%.2f");
            ImGui::SliderFloat("lake time speed",
                               &rt_.water_lake_time_speed, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("lake uv scale",
                               &rt_.water_lake_uv_scale, 0.1f, 4.0f, "%.2f");
            ImGui::SliderFloat("lake bump fade dist (m)",
                               &rt_.water_lake_bump_dist, 5.0f, 500.0f, "%.0f");
            ImGui::TextDisabled("Far surface goes mirror-flat past fade dist.");
        }
        ImGui::ColorEdit3("Deep water color", &rt_.water_color.x);
        ImGui::ColorEdit3("Shallow water color", &rt_.water_color_shallow.x);
        ImGui::SliderFloat("Shore band (m)",
                           &rt_.water_shore_blend, 0.5f, 20.0f, "%.1f");
        ImGui::SliderFloat("Shore noise",
                           &rt_.water_shore_noise, 0.0f, 1.0f, "%.2f");
        ImGui::ColorEdit3("Foam color", &rt_.water_foam_color.x);
        ImGui::SliderFloat("Foam strength",
                           &rt_.water_foam_strength, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Foam width (m)",
                           &rt_.water_foam_width, 0.05f, 5.0f, "%.2f");
        ImGui::SliderFloat("Foam opacity",
                           &rt_.water_foam_opacity, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("%s",
            "How opaque the foam is vs the see-through water under it.\n"
            "0 = foam as transparent as clear shallow water (light\n"
            "surf), 1 = foam fully opaque.");
        ImGui::Checkbox("RT terrain reflections", &rt_.water_rt_reflections);
        ImGui::Checkbox("RT cube/castle reflections (TLAS)",
                         &rt_.water_tlas_reflections);
        ImGui::Checkbox("Sun shadows on water", &rt_.water_shadows_enabled);
        ImGui::SliderFloat("Transparency",
                           &rt_.water_transparency, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled("%s",
            "Master see-through strength. 0 = always opaque,\n"
            "1 = full clarity (up to the clarity depth below).");
        ImGui::SliderFloat("Clarity depth (m)",
                           &rt_.water_clarity_depth, 0.2f, 20.0f, "%.1f");
        ImGui::SliderFloat("Underwater murkiness", &rt_.underwater_turbidity, 0.2f, 4.0f, "%.2f");
        ImGui::TextDisabled("%s",
            "How deep you can see, INDEPENDENT of strength.\n"
            "Small (~1 m) = only very shallow water is clear, deep\n"
            "water stays opaque (realistic shallows). Large = see\n"
            "the seabed far down (tropical lagoon).");
        ImGui::SliderFloat("Shoreline softness (m)",
                           &rt_.water_shore_softness, 0.05f, 4.0f, "%.2f");
        ImGui::TextDisabled("%s",
            "Metres of depth the water fades in over at the land\n"
            "edge. Larger = a longer, smoother wet-sand → water\n"
            "transition (also hides the waterline jitter).");
        // Quick presets so the user has a starting point per mood.
        if (ImGui::Button("Tropical")) {
            rt_.water_color         = glm::vec3(0.04f, 0.30f, 0.40f);
            rt_.water_color_shallow = glm::vec3(0.55f, 0.85f, 0.90f);
            rt_.water_shore_blend   = 6.0f;
            rt_.water_shore_noise   = 0.45f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Lake")) {
            rt_.water_color         = glm::vec3(0.05f, 0.16f, 0.20f);
            rt_.water_color_shallow = glm::vec3(0.30f, 0.55f, 0.55f);
            rt_.water_shore_blend   = 3.0f;
            rt_.water_shore_noise   = 0.40f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Deep ocean")) {
            rt_.water_color         = glm::vec3(0.02f, 0.10f, 0.18f);
            rt_.water_color_shallow = glm::vec3(0.10f, 0.35f, 0.50f);
            rt_.water_shore_blend   = 8.0f;
            rt_.water_shore_noise   = 0.30f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Murky")) {
            rt_.water_color         = glm::vec3(0.10f, 0.13f, 0.07f);
            rt_.water_color_shallow = glm::vec3(0.30f, 0.32f, 0.20f);
            rt_.water_shore_blend   = 2.0f;
            rt_.water_shore_noise   = 0.55f;
        }
        ImGui::TextDisabled("%s",
            "Cheap analytical-wave ocean folded into the\n"
            "raymarch shader. Rendered when ray hits y=level\n"
            "before terrain. Schlick fresnel + sky reflection\n"
            "+ specular highlight + depth-fade tint.");

        ImGui::SeparatorText("Heightmap resolution");
        const char* hres_labels[] = {
            "1x (2048 cells, 1 m)",
            "2x (4096 cells, 0.5 m, +1.6 s load, 4x terrain memory)",
            "4x (8192 cells, 0.25 m, NOT RECOMMENDED - multi-GB)"
        };
        const int hres_values[] = { 1, 2, 4 };
        int hres_idx = 0;
        for (int i = 0; i < 3; ++i)
            if (rt_.terrain_heightmap_scale == hres_values[i]) { hres_idx = i; break; }
        if (ImGui::Combo("Heightmap scale", &hres_idx, hres_labels, 3)) {
            rt_.terrain_heightmap_scale = hres_values[hres_idx];
        }
        ImGui::TextDisabled("* Restart required — heightmap generated at level load.");

        // ---- Mesh-path-only settings (hidden in FBM raymarch mode) ----
        if (!rt_.terrain_raymarch_enabled) {
        ImGui::SeparatorText("Detail (live)");
        ImGui::SliderFloat("Sand ripple scale",
                           &rt_.terrain_sand_ripple_scale, 1.0f, 40.0f, "%.1f");
        ImGui::TextDisabled(
            "Beach ripple spacing. Lines follow the shoreline (phase =\n"
            "height above water), so they curve with the coast.");
        ImGui::SliderFloat("Grass line scale",
                           &rt_.terrain_grass_line_scale, 0.0f, 40.0f, "%.1f");
        ImGui::TextDisabled(
            "Grass contour rows, same shore-following logic.\n"
            "0 = off. Independent of the sand ripple scale.");
        ImGui::SliderFloat("Rock parallax (per-pixel depth)",
                           &rt_.terrain_pom_strength, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled(
            "Per-pixel parallax of the eroded rock height — the FINE\n"
            "half. Pair with Rock relief above + Material strength > 0.\n"
            "Only on rocky pixels, ramps to 0 by Disp far m below.");
        ImGui::SliderFloat("Disp amplitude (m)",
                           &rt_.terrain_disp_amp, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled(
            "Vertex displacement amplitude in metres for the rocky-grass\n"
            "band material — applied in the CLOD vertex shader near the\n"
            "camera (replaces the old tessellation displacement).");
        ImGui::SliderFloat("Disp smooth (mip bias)",
                           &rt_.terrain_disp_smooth_mip, 0.0f, 6.0f, "%.1f");
        ImGui::TextDisabled(
            "Smoothing pass on the displacement height sample.\n"
            "0 = crisp LOD0 (max spike risk if amp high), +N = blur\n"
            "by N mip levels. Per-pixel SPOM still uses crisp LOD0.");
        ImGui::SliderFloat("POM far distance (m)",
                           &rt_.terrain_pom_far_m, 50.0f, 600.0f, "%.0f");
        ImGui::TextDisabled(
            "Distance beyond which per-pixel parallax stops being\n"
            "raymarched. Was hardcoded 220 m. Higher = more detail far,\n"
            "lower = perf.");

        ImGui::SeparatorText("Ground material (procedural splat)");
        ImGui::SliderFloat("Material strength",
                           &rt_.ground_mat_strength, 0.0f, 1.0f, "%.2f");
        ImGui::TextDisabled(
            "0 = old flat height-band colour (A/B off), 1 = full per-texel\n"
            "rock/grass/dirt/sand/snow blended by height + slope.");
        ImGui::SliderFloat("Material tile size (m)",
                           &rt_.ground_mat_tile_m, 0.5f, 12.0f, "%.2f");
        ImGui::SliderFloat("Detail normal strength",
                           &rt_.ground_mat_normal, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Anti-tile sampling",
                        &rt_.terrain_antitile);
        ImGui::TextDisabled(
            "Blends a second rotated/offset sample of each ground\n"
            "material weighted by a low-frequency noise mask. Cuts the\n"
            "visible grid repeat on terrain at the cost of ~1 extra\n"
            "texture tap per channel. Tess + SPOM UVs stay strict so\n"
            "geometry and per-pixel relief remain aligned.");
        ImGui::SliderFloat("Anti-tile blend strength",
                           &rt_.terrain_antitile_strength,
                           0.0f, 1.0f, "%.2f");

        ImGui::SeparatorText("Shading");
        ImGui::SliderFloat("light wrap",       &rt_.terrain_wrap_strength,   0.0f, 1.0f);
        ImGui::SliderFloat("detail brightness",&rt_.terrain_detail_strength, 0.0f, 3.0f);
        ImGui::SliderFloat("shadow softness x",&rt_.terrain_shadow_softness_scale, 0.05f, 1.5f);
        ImGui::SliderFloat("Terrain shading contrast",
                            &rt_.terrain_shading_contrast, 0.5f, 6.0f, "%.2f");
        {
            const char* ss_labels[] = { "1x (native)", "2x (sub-cell)", "4x (sharp far)" };
            const int ss_values[] = { 1, 2, 4 };
            int ss_idx = 0;
            for (int i = 0; i < 3; ++i) if (rt_.terrain_bake_supersample == ss_values[i]) { ss_idx = i; break; }
            if (ImGui::Combo("Shadow bake supersample", &ss_idx, ss_labels, 3)) {
                rt_.terrain_bake_supersample = ss_values[ss_idx];
            }
        }
        }   // end mesh-path-only settings

        // ---- Shared (both terrain modes) ----
        if (ImGui::TreeNode("Layer transitions (height in metres)")) {
            ImGui::SliderFloat("sand->grass start", &rt_.terrain_h_sand_grass_start,  -10.0f,  60.0f);
            ImGui::SliderFloat("sand->grass end",   &rt_.terrain_h_sand_grass_end,    -10.0f,  60.0f);
            ImGui::SliderFloat("grass->dirt start", &rt_.terrain_h_grass_dirt_start,    0.0f, 100.0f);
            ImGui::SliderFloat("grass->dirt end",   &rt_.terrain_h_grass_dirt_end,      0.0f, 100.0f);
            ImGui::SliderFloat("dirt->rock start",  &rt_.terrain_h_dirt_rock_start,    20.0f, 150.0f);
            ImGui::SliderFloat("dirt->rock end",    &rt_.terrain_h_dirt_rock_end,      20.0f, 150.0f);
            ImGui::SliderFloat("rock->snow start",  &rt_.terrain_h_rock_snow_start,    50.0f, 200.0f);
            ImGui::SliderFloat("rock->snow end",    &rt_.terrain_h_rock_snow_end,      50.0f, 200.0f);
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx("Shoreline tint", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextDisabled("Grass-area shore (under blades):");
            ImGui::ColorEdit3("grass shore colour",
                              &rt_.terrain_shore_color.x);
            ImGui::SliderFloat("grass shore strength",
                               &rt_.terrain_shore_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("grass shore distance (m)",
                               &rt_.terrain_shore_distance,
                               0.1f, 20.0f, "%.1f");
            ImGui::Separator();
            ImGui::TextDisabled("Bare-terrain shore (no grass):");
            ImGui::ColorEdit3("bare shore colour",
                              &rt_.terrain_shore_general_color.x);
            ImGui::SliderFloat("bare shore strength",
                               &rt_.terrain_shore_general_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("bare shore distance (m)",
                               &rt_.terrain_shore_general_distance,
                               0.1f, 20.0f, "%.1f");
            ImGui::Separator();
            ImGui::TextDisabled("Beach base material (slope-driven):");
            ImGui::ColorEdit3("sand colour",
                              &rt_.terrain_sand_color.x);
            ImGui::SeparatorText("Raymarched ground palette");
            ImGui::ColorEdit3("dry ground", &rt_.rm_ground_dry.x);
            ImGui::ColorEdit3("green ground", &rt_.rm_ground_green.x);
            ImGui::SliderFloat("green amount", &rt_.rm_green_amount, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = mostly dry brown earth, 1 = green everywhere it is flat enough");
            ImGui::ColorEdit3("red sand", &rt_.rm_red_sand.x);
            ImGui::SliderFloat("red sand amount", &rt_.rm_red_sand_amount, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("red sand patch size", &rt_.rm_red_sand_scale, 0.2f, 5.0f, "%.2f");
            ImGui::ColorEdit3("rock colour", &rt_.rm_rock_color.x);
            ImGui::SliderFloat("ground saturation", &rt_.rm_ground_saturation, 0.0f, 2.0f, "%.2f");
            if (ImGui::Button("Reset ground palette")) {
                const RtSettings d{};
                rt_.rm_ground_dry = d.rm_ground_dry; rt_.rm_ground_green = d.rm_ground_green;
                rt_.rm_green_amount = d.rm_green_amount; rt_.rm_red_sand = d.rm_red_sand;
                rt_.rm_red_sand_amount = d.rm_red_sand_amount; rt_.rm_red_sand_scale = d.rm_red_sand_scale;
                rt_.rm_rock_color = d.rm_rock_color; rt_.rm_ground_saturation = d.rm_ground_saturation;
            }
            ImGui::TreePop();
        }


        ImGui::SeparatorText("Debug");
        const char* terrain_debug_labels[] = {
            "off (full shading)",
            "simple Lambert (no RT/AO/GI)",
            "show vertex normal",
            "show face normal",
            "4: Lambert + RT shadow",
            "5: + height layers",
            "6: + slope-rock",
            "7: + cavity AO",
            "8: + triplanar detail",
            "9: + RTAO",
            "10: + GI bounce",
        };
        ImGui::Combo("terrain debug", &rt_.terrain_debug_mode,
                      terrain_debug_labels, 11);
        }   // end CollapsingHeader Terrain

        // Grass folded into the Terrain tab — they're conceptually
        // the same surface family for the player, and the previous
        // top-level Grass section was rarely opened.
        if (ImGui::CollapsingHeader("Grass")) {
            ImGui::Checkbox("grass enabled", &rt_.grass_enabled);
            ImGui::Checkbox("RAYMARCH grass (SDF blades, replaces rasterised)",
                             &rt_.grass_raymarch_enabled);
            ImGui::SliderFloat("raymarch distance (m)",
                                &rt_.grass_raymarch_distance, 5.0f, 400.0f);
            ImGui::SliderFloat("grass density", &rt_.grass_density, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("grass near density boost", &rt_.grass_near_boost, 0.0f, 2.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Extra blades close to the camera; fades out by about 10 m.");
            ImGui::SliderFloat("grass edge softness", &rt_.grass_edge_softness, 0.0f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How gradually grass thins out and shortens at the border of a patch.");
            ImGui::SeparatorText("Tree detail");
            ImGui::SliderFloat("tree / bush detail", &rt_.veg_lod_bias, 0.4f, 1.5f, "%.2fx");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("How far out trees and bushes keep their full branches and leaves.\n1 = default, 2 = full detail twice as far (more triangles), <1 = faster.");
            ImGui::SeparatorText("Plant draw distance");
            ImGui::SliderFloat("flowers + ferns (m)", &rt_.flower_draw_distance, 10.0f, 200.0f, "%.0f");
            ImGui::SliderFloat("ground plants (m)", &rt_.ground_draw_distance, 10.0f, 200.0f, "%.0f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clover, plantain, dandelions, nettles, foxgloves, thistles, mushrooms");
            ImGui::SliderFloat("reeds (m)", &rt_.reed_draw_distance, 10.0f, 300.0f, "%.0f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Plants thin out over the last part of the range instead of all vanishing at once. Longer = slower.");
            ImGui::SeparatorText("Foliage density (restart to apply)");
            ImGui::SliderFloat("forest density", &rt_.forest_density, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("bush density", &rt_.bush_density, 0.0f, 2.0f, "%.2f");
            if (ImGui::Button("Replant forest now")) veg_replant_request_ = true;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Applies forest / bush density without a restart (takes about a second).");
            ImGui::SliderFloat("flower density", &rt_.flower_density, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("ground plant density", &rt_.ground_density, 0.0f, 3.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clover, plantain, dandelions, nettles, foxgloves, thistles and mushrooms. 0 = none, 1 = normal, 3 = lush. Restart to apply.");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Blades per ground cell: 1 = every cell, 2 = two blades per cell (denser, a bit slower)");
            ImGui::SliderFloat("height scale",   &rt_.grass_height_scale, 0.30f, 2.0f);
            ImGui::SliderFloat("distance (m)",   &rt_.grass_distance,   10.0f, 200.0f);
            ImGui::SliderFloat("wind (m)",       &rt_.grass_wind,        0.0f, 0.30f);
            ImGui::SliderFloat("alpha cutoff",   &rt_.grass_alpha_cutoff, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("cutoff soft-dist (m)",
                               &rt_.grass_cutoff_soft_dist, 5.0f, 150.0f, "%.0f");
            ImGui::SliderFloat("slope cutoff",   &rt_.grass_slope_n_min, 0.55f, 1.0f, "%.2f");
            ImGui::SliderFloat("distance density falloff",
                               &rt_.grass_distance_density, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("altitude min (m)", &rt_.grass_alt_min, -20.0f, 200.0f);
            ImGui::SliderFloat("altitude max (m)", &rt_.grass_alt_max, -20.0f, 200.0f);
            ImGui::ColorEdit3("blade tip colour",
                              &rt_.grass_color_top.x);
            ImGui::ColorEdit3("blade base colour",
                              &rt_.grass_color_bottom.x);
            ImGui::SliderFloat("base AO floor",
                               &rt_.grass_base_ao_floor, 0.0f, 1.0f, "%.2f");
            ImGui::ColorEdit3("ground tint close",
                              &rt_.grass_color_ground.x);
            ImGui::ColorEdit3("ground tint far",
                              &rt_.grass_color_ground_far.x);
            ImGui::SliderFloat("ground tint strength",
                               &rt_.grass_ground_tint_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("ground tint far distance (m)",
                               &rt_.grass_ground_tint_far_distance,
                               50.0f, 1000.0f, "%.0f");
            ImGui::Separator();
            ImGui::TextUnformatted("Shoreline tint (drier near water)");
            ImGui::ColorEdit3("shore colour", &rt_.grass_shore_color.x);
            ImGui::SliderFloat("shore strength",
                               &rt_.grass_shore_strength, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("shore distance (m)",
                               &rt_.grass_shore_distance, 0.1f, 20.0f, "%.1f");
            ImGui::Separator();
            ImGui::TextUnformatted("Grass shadow on terrain (raymarched)");
            ImGui::SliderFloat("shadow strength",
                               &rt_.grass_shadow_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderInt  ("shadow samples",
                               &rt_.grass_shadow_samples, 0, 8);
            ImGui::SliderFloat("shadow reach (m)",
                               &rt_.grass_shadow_max_dist,
                               0.2f, 5.0f, "%.1f");

            ImGui::Separator();
            ImGui::TextUnformatted("Grass shadow on terrain (rasterised)");
            ImGui::Checkbox("enabled##gst",
                            &rt_.grass_shadow_on_terrain);
            ImGui::TextDisabled(
                "Walks the grass eligibility mask along the sun XZ for\n"
                "a few short steps and accumulates occlusion as the\n"
                "ground 'in the shade' of overhead blades. Cheap (no\n"
                "ray query), works on the standard chunked terrain.");
            ImGui::SliderFloat("strength##gst",
                               &rt_.grass_shadow_on_terrain_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::SliderInt  ("samples##gst",
                               &rt_.grass_shadow_on_terrain_samples, 1, 8);
            ImGui::SliderFloat("reach (m)##gst",
                               &rt_.grass_shadow_on_terrain_dist,
                               0.2f, 6.0f, "%.1f");

            ImGui::Separator();
            ImGui::TextUnformatted("Side-lit blade shading");
            ImGui::Checkbox("side-lit enabled",
                            &rt_.grass_side_lit_enabled);
            ImGui::SliderFloat("side-lit strength",
                               &rt_.grass_side_lit_strength,
                               0.0f, 1.0f, "%.2f");
            ImGui::TextDisabled(
                "Sun-side of each blade brightens, away-side darkens.\n"
                "Strength=0 restores the engine-default flat-tint look.");
        }

        ImGui::EndTabItem();
    }

    // ---------------- Terrain Edit ----------------
    // Sculpt brush + heightmap save/load + plateau noise + FBM erosion
    // bake. Lives in its own tab so the rendering knobs in the
    // "Terrain" tab stay focused on look (renderer / layers / debug)
    // and the destructive editing tools are one click away from the
    // tab bar instead of buried inside a CollapsingHeader.
    if (ImGui::BeginTabItem("Terrain Edit")) {
        ImGui::SeparatorText("Heightmap save / load");
        if (ImGui::Button("Save heightmap")) {
            save_terrain_heights();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("→ assets/level1_heights.bin (auto-loaded next launch)");
        if (ImGui::Button("Save grass mask")) {
            save_grass_mask();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("→ assets/level1_grass_mask.bin (auto-loaded next launch)");

        // Plateau noise — adds gentle relief on the castle pad so it
        // isn't a perfect flat. Two sliders (amplitude + frequency)
        // and an Apply button so accidental drag doesn't constantly
        // edit the heightmap.
        ImGui::SeparatorText("Plateau noise (one-shot)");
        static float plateau_amp_m = 0.6f;
        static float plateau_freq  = 0.20f;
        ImGui::SliderFloat("Plateau noise amp (m)", &plateau_amp_m,
                           0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Plateau noise freq",    &plateau_freq,
                           0.05f, 1.0f, "%.2f");
        if (ImGui::Button("Apply plateau noise")) {
            add_plateau_noise(plateau_amp_m, plateau_freq);
            log::info("[ui] plateau noise applied — Save heightmap to persist");
        }

        ImGui::SeparatorText("Sculpt brush");
        ImGui::Checkbox("Edit mode (left = brush mode, right = inverse)",
                        &terrain_edit_mode_);
        if (terrain_edit_mode_) {
            const char* modes[] = {"Raise", "Lower", "Smooth", "Flatten",
                                    "Grass +", "Grass −",
                                    "Erode +", "Erode smooth", "Zones"};
            int mode_i = static_cast<int>(terrain_brush_mode_);
            if (ImGui::Combo("brush mode (left-click)", &mode_i, modes,
                              IM_ARRAYSIZE(modes))) {
                terrain_brush_mode_ = static_cast<TerrainBrushMode>(mode_i);
            }
            ImGui::TextDisabled("right-click inverts Raise↔Lower, Grass+↔Grass−, Erode+↔Erode smooth");
            ImGui::SliderFloat("brush radius (m)", &terrain_brush_radius_, 1.0f, 60.0f);
            ImGui::SliderFloat("brush strength (m/s)", &terrain_brush_strength_,
                               0.5f, 60.0f);
            ImGui::SliderFloat("brush noise (raise/lower)",
                                &terrain_brush_noise_strength_, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("brush noise freq",
                                &terrain_brush_noise_freq_, 0.05f, 1.0f, "%.2f");
            ImGui::Checkbox("Use FBM erosion detail (matches raymarched terrain)",
                            &terrain_brush_use_fbm_erosion_);
            if (terrain_brush_use_fbm_erosion_) {
                ImGui::SliderInt("FBM octaves",
                                  &terrain_brush_fbm_octaves_, 2, 9);
            }
            if (terrain_brush_mode_ == TerrainBrushMode::Flatten) {
                ImGui::SliderFloat("flatten target Y",
                                   &terrain_brush_flatten_target_, 0.0f, 200.0f);
            }
            if (terrain_brush_has_hit_) {
                ImGui::Text("brush at: (%.1f, %.1f, %.1f)",
                            terrain_brush_world_pos_.x,
                            terrain_brush_world_pos_.y,
                            terrain_brush_world_pos_.z);
            } else {
                ImGui::TextDisabled("aim at terrain to engage brush");
            }
        }

        ImGui::SeparatorText("FBM erosion bake (one-shot, sculpted area)");
        static float fbm_amp_factor = 0.30f;
        static float fbm_freq       = 0.04f;
        ImGui::SliderFloat("FBM erosion amplitude", &fbm_amp_factor, 0.05f, 1.0f);
        ImGui::SliderFloat("FBM erosion freq",       &fbm_freq,        0.005f, 0.5f);
        if (ImGui::Button("Apply FBM erosion to sculpted area")) {
            add_fbm_erosion_to_sculpted(fbm_amp_factor, fbm_freq);
        }

        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();

    ImGui::Spacing();
    ImGui::TextDisabled("ESC to resume");
    ImGui::End();
}


} // namespace vibe
