// The parts of one turn of the main loop (VulkanEngine::run, vk_engine.cpp) that deal with
// the game rather than with drawing a frame: input, what the keys do, firing and the
// terrain brush, the simulation steps, and the per-frame upkeep. run() calls them in this
// order, then builds the UI and renders. They were one 900-line stretch of run().

#include "engine/vk_engine.h"
#include <ctime>
#include <filesystem>
#include <system_error>
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/taau.h"
#include "engine/vk_engine/state/voxel.h"
#include "engine/vk_engine/state/sun_shadow.h"
#include "engine/frame_pacer.h"
#include "engine/voxel/voxel_world.h"
#include "engine/audio.h"
#include "engine/frustum.h"
#include "engine/gltf_loader.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"
#include "engine/texture.h"
#include "engine/window.h"
#include "engine/vk_initializers.h"
#include "engine/vk_pipelines.h"
#include <cstdlib>
#include <VkBootstrap.h>
#include "engine/highway.h"
#include "engine/vk_engine/river_stream.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <stb_image.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#ifndef VIBE_SHADER_DIR
#define VIBE_SHADER_DIR "shaders"
#endif

namespace vibe {

// Input that does not come from the player: the --autodemo walk, the VIBE_TEST_* hooks that
// hold the camera, fly it or pull the trigger for captures, and the --scene teleport.
void VulkanEngine::loop_test_input(InputFrame& in, float frame_dt, bool autodemo, bool headless,
                                   std::chrono::steady_clock::time_point now,
                                   std::chrono::steady_clock::time_point demo_start) {
    if (autodemo) {
        float demo_t = std::chrono::duration<float>(now - demo_start).count();
        if (demo_t >= opts_.autodemo_seconds) {
            log::infof("[autodemo] elapsed %.2fs — requesting close",
                       demo_t);
            window_->request_close();
        }
        // Synthetic input: walk a calm counter-clockwise lap around the
        // castle on the plateau. Position is TELEPORTED each frame to
        // an orbit point (R=15 m, well inside the plateau and clear of
        // both the castle walls and the water below the plateau edge);
        // yaw points along the tangent so the camera looks straight
        // ahead in the direction of motion; pitch locked at 0 (no
        // head-bob / look-down). No fwd input — position override
        // does the moving, so physics never accelerates us off-orbit.
        // No jump, no fire — this demo is for visual + perf inspection,
        // not for crash repro (use a separate autodemo flavour for that
        // when it returns). y locked at 23 m (~1 m above plateau top).
        in = InputFrame{};
        // --scene N override: park the camera at one of the deferred-
        // renderer reference scenes for deterministic before/after
        // captures. Locks position/yaw/pitch and zeroes velocity each
        // frame. Each scene covers a specific lighting/material case
        // the migration must preserve.
        if (opts_.scene > 0) {
            struct ScenePose { glm::vec3 pos; float yaw; float pitch; };
            static const ScenePose kScenes[] = {
                // 1: castle exterior, north entrance, sun behind us (lit walls).
                // Yaw 0 faces -Z (south, toward the castle from z=+18);
                // the old kPi here faced +Z and framed the voxel tower
                // 100 m north instead of the castle.
                { glm::vec3(0.0f,  23.5f,  18.0f),  0.0f,       0.10f },
                // 2: castle exterior, south side (shadow side), looking north
                { glm::vec3(0.0f,  23.5f, -18.0f),  kPi,        0.10f },
                // 3: castle interior, looking up at the entrance lintel (PCSS)
                { glm::vec3(0.0f,  23.5f,   2.0f),  kPi,        0.55f },
                // 4: voxel tower mid-range (multi-shape, self-shadow)
                { glm::vec3(8.0f,  27.0f,  35.0f),  kPi,        0.10f },
                // 5: grass meadow far from castle (GI fed by terrain)
                { glm::vec3(60.0f, 26.0f,  60.0f), -0.75f,      0.05f },
                // 6: underwater at the bay (water blend chain)
                { glm::vec3(40.0f, 19.0f, -40.0f),  0.5f,      -0.15f },
                // 7: crate spawn zone (TAA + dyn-prop)
                { glm::vec3(0.0f,  24.0f,   0.0f),  kHalfPi,    0.0f  },
            };
            int idx = std::clamp(opts_.scene, 1, 7) - 1;
            player_.position = kScenes[idx].pos;
            player_.yaw      = kScenes[idx].yaw;
            player_.pitch    = kScenes[idx].pitch;
            player_.velocity = glm::vec3(0.0f);
        } else {
            const float R     = 15.0f;
            const float omega = 0.30f;   // rad/s — ~17°/s
            const float ang   = demo_t * omega;
            player_.position  = glm::vec3(R * std::sin(ang),
                                          23.0f,
                                          R * std::cos(ang));
            player_.yaw       = ang + kHalfPi;          // tangent (CCW)
            player_.pitch     = 0.0f;
        }
        in.mouse_dx = 0.0;
        in.mouse_dy = 0.0;
    }
    // Dev hook for headless captures of weapon visuals: VIBE_TEST_FIRE=1
    // pulls the trigger every 8th frame (pair with --screenshot).
    {
        static const bool test_fire = std::getenv("VIBE_TEST_FIRE") != nullptr;
        static int test_fire_n = 0;
        // VIBE_TEST_FIRE=<n > 1>: one single shot at frame n instead.
        static const int test_fire_at = test_fire ? std::atoi(std::getenv("VIBE_TEST_FIRE")) : 0;
        if (test_fire && (test_fire_at > 1 ? (++test_fire_n == test_fire_at) : ((++test_fire_n % 8) == 0))) {
            in.fire = true;
            in.fire_held = true;
        }
        npc_test_input(in);                // (npc_world.cpp: VIBE_TEST_NPC_SHOOT aims at somebody of the NPC world and fires)
        // VIBE_TEST_MOVE=1: real movement keys on a 9 s loop (forward with a jump, sprint,
        // strafe, back, a turning arc) - an unattended player for watching animation and
        // netcode from a second instance.
        static const bool test_move = std::getenv("VIBE_TEST_MOVE") != nullptr;
        if (test_move) {
            static const auto move_t0 = std::chrono::steady_clock::now();
            const float t = std::fmod(std::chrono::duration<float>(std::chrono::steady_clock::now() - move_t0).count(), 9.0f);
            if (t < 3.0f) { in.fwd = true; in.jump = t > 1.2f && t < 1.3f; in.sprint = t > 2.0f; }
            else if (t < 4.5f) in.right = true;
            else if (t < 6.0f) in.back = true;
            else if (t < 8.5f) { in.fwd = true; in.left = t > 7.2f; in.mouse_dx = 6.0; in.jump = t > 7.9f && t < 8.0f; }
        }
        // VIBE_LOG_LOOK=1: once, the ground point the camera looks at, the ground round it and
        // the highway beside it (for placing things "where I am looking").
        if (static const bool log_look = std::getenv("VIBE_LOG_LOOK") != nullptr; log_look && frame_number_ == 30) {
            const glm::vec3 e = player_.eye_position(), f = player_.forward();
            glm::vec3 hit = e; bool found = false;
            for (float t = 1.0f; t < 2000.0f; t += 0.5f) {
                const glm::vec3 p = e + f * t;
                if (p.y <= sample_terrain_height(p.x, p.z)) { hit = p; found = true; break; }
            }
            log::infof("[look] eye (%.1f, %.1f, %.1f) forward (%.3f, %.3f, %.3f) -> ground (%.1f, %.1f, %.1f)%s",
                       double(e.x), double(e.y), double(e.z), double(f.x), double(f.y), double(f.z),
                       double(hit.x), double(sample_terrain_height(hit.x, hit.z)), double(hit.z), found ? "" : " (none within 2 km)");
            // (VIBE_LOG_LOOK=<metres>: the spacing of the grid of ground heights; 20 m when it is just set)
            const float look_step = std::max(1.0f, float(std::atof(std::getenv("VIBE_LOG_LOOK")))) > 1.5f ? float(std::atof(std::getenv("VIBE_LOG_LOOK"))) : 20.0f;
            for (int j = -3; j <= 3; ++j) {
                char row[256]; int o = 0;
                for (int i = -3; i <= 3; ++i)
                    o += std::snprintf(row + o, sizeof(row) - size_t(o), "%7.1f", double(sample_terrain_height(hit.x + look_step * float(i), hit.z + look_step * float(j))));
                log::infof("[look] ground, z %+6.1f m, x %.0f m apart, -3..+3: %s", double(look_step * float(j)), double(look_step), row);
            }
            float d = 0.0f, h = 0.0f, s = 0.0f;
            if (highway::nearest(hit.x, hit.z, 400.0f, d, h, s)) {
                glm::vec3 hp; glm::vec2 hd;
                highway::point_along(s, hp, hd);
                log::infof("[look] highway: %.1f m away, at %.1f m along it: point (%.1f, %.1f, %.1f) direction (%.3f, %.3f)",
                           double(d), double(s), double(hp.x), double(hp.y), double(hp.z), double(hd.x), double(hd.y));
                for (float ds = -80.0f; ds <= 80.0f; ds += 40.0f) {
                    highway::point_along(s + ds, hp, hd);
                    log::infof("[look] highway at %+.0f m: (%.1f, %.1f, %.1f) direction (%.3f, %.3f)", double(ds), double(hp.x), double(hp.y), double(hp.z), double(hd.x), double(hd.y));
                }
            }
        }
        // VIBE_TEST_WALK=1: the forward key held (walk into something; with it the player's
        // place is logged once a second).
        static const bool test_walk = std::getenv("VIBE_TEST_WALK") != nullptr;
        if (test_walk) {
            in.fwd = true;
            if (frame_number_ % 60 == 0)
                log::infof("[walk] pos (%.2f, %.2f, %.2f)", double(player_.position.x), double(player_.position.y), double(player_.position.z));
        }
        // VIBE_TEST_HOLD=1: pin the camera at the restored settings
        // pose (no gravity) for aerial / overview captures.
        static const bool test_hold = std::getenv("VIBE_TEST_HOLD") != nullptr;
        static bool hold_init = false;
        static glm::vec3 hold_pos{0.0f};
        // VIBE_TEST_BOOM=<metres>: ground explosion that far ahead of
        // the camera at frame 40 (headless explosion captures).
        static const char* test_boom = std::getenv("VIBE_TEST_BOOM");
        static int boom_frame = 0;
        if (test_boom && ++boom_frame == 40) {
            glm::vec3 f = player_.forward(); f.y = 0.0f;
            f = glm::normalize(f + glm::vec3(1e-4f));
            glm::vec3 p = player_.eye_position() + f * static_cast<float>(std::atof(test_boom));
            p.y = std::max(sample_terrain_height(p.x, p.z), rt_.water_level);
            spawn_explosion(p, 1.0f, true);
        }
        if (test_hold) {
            if (!hold_init) {
                hold_pos = player_.position; hold_init = true;
                // VIBE_DUMP_HEIGHTS=<file>: the ground the CPU knows, as raw float32, every 8 m
                // over x -1400..3000, z -1400..1400 (551 x 351, rows of x) - for laying out
                // rivers / roads against the real relief.
                if (const char* dump = std::getenv("VIBE_DUMP_HEIGHTS")) {
                    if (FILE* f = std::fopen(dump, "wb")) {
                        for (int j = 0; j < 351; ++j) for (int i = 0; i < 551; ++i) {
                            const float g = sample_terrain_height(-1400.0f + 8.0f * float(i), -1400.0f + 8.0f * float(j));
                            std::fwrite(&g, sizeof(float), 1, f);
                        }
                        std::fclose(f);
                    }
                }
                // VIBE_TEST_GROUND=1: stand the held camera on the ground the CPU knows there (eye
                // 1.7 m up) - a capture then shows whether that is the ground the GPU draws.
                if (std::getenv("VIBE_TEST_GROUND")) {
                    const float g = sample_terrain_height(hold_pos.x, hold_pos.z);
                    hold_pos.y = std::max(g, rt_.water_level) + 1.7f - 0.75f;   // (position is 0.75 m below the eye)
                    log::infof("[test] ground at (%.0f, %.0f): %.2f m (water %.2f)", hold_pos.x, hold_pos.z, g, rt_.water_level);
                }
            }
            if (globe_moved_player_) { hold_pos = player_.position; globe_moved_player_ = false; }    // (re-anchored on the globe)
            // VIBE_TEST_DOLLY=<vx>,<vy>,<vz>: the held camera travels at that speed (m/s) - for
            // capturing what flickers while walking (consecutive frames of a steady move).
            if (static const char* dolly = std::getenv("VIBE_TEST_DOLLY")) {
                static glm::vec3 dv = [] { glm::vec3 v(0.0f); std::sscanf(std::getenv("VIBE_TEST_DOLLY"), "%f,%f,%f", &v.x, &v.y, &v.z); return v; }();
                (void)dolly;
                hold_pos += dv * frame_dt;
            }
            player_.position = hold_pos;
            // VIBE_TEST_FLIGHT=<y0>,<y1>,<frames>[,<yaw per frame>]: from frame 60 on the
            // held camera flies from altitude y0 to y1 (evenly in the logarithm, so every
            // part of a castle-to-space flight gets its share of frames) while turning.
            // With --frames the time step is fixed, so a given frame is the same in every run.
            static const char* test_flight = std::getenv("VIBE_TEST_FLIGHT");
            if (test_flight) {
                static float fy0 = 0.0f, fy1 = 0.0f, fn = 1.0f, fyaw = 0.0f, fyaw0 = 0.0f; static bool fi = false;
                if (!fi) { fi = true; std::sscanf(test_flight, "%f,%f,%f,%f", &fy0, &fy1, &fn, &fyaw); fyaw0 = player_.yaw; }
                const float k = glm::clamp((float(frame_number_) - 60.0f) / std::max(fn, 1.0f), 0.0f, 1.0f);
                player_.position.y = std::exp(glm::mix(std::log(std::max(fy0, 1.0f)), std::log(std::max(fy1, 1.0f)), k));
                player_.yaw = fyaw0 + fyaw * std::max(float(frame_number_) - 60.0f, 0.0f);
            }

            player_.velocity = glm::vec3(0.0f);
            // VIBE_TEST_PAN=<rad/s>: steady yaw pan (motion artefact captures).
            static const char* test_pan = std::getenv("VIBE_TEST_PAN");
            if (test_pan) player_.yaw += static_cast<float>(std::atof(test_pan)) * frame_dt;
        }
    }
    // --scene N teleport works WITHOUT --autodemo too. Pin the camera
    // each frame so headless captures (--frames N --screenshot) hit the
    // intended pose; without this the camera kept whatever position the
    // settings file had saved from the previous run, silently producing
    // 17%+ SAD against the reference shots.
    if (opts_.scene > 0 && !autodemo) {
        struct ScenePose { glm::vec3 pos; float yaw; float pitch; };
        static const ScenePose kScenes[] = {
            // Keep in lock-step with the autodemo copy above (scene 1/2
            // yaws face the castle, not the voxel tower north of it).
            { glm::vec3(0.0f,  23.5f,  18.0f),  0.0f,       0.10f },
            { glm::vec3(0.0f,  23.5f, -18.0f),  kPi,        0.10f },
            { glm::vec3(0.0f,  23.5f,   2.0f),  kPi,        0.55f },
            { glm::vec3(8.0f,  27.0f,  35.0f),  kPi,        0.10f },
            { glm::vec3(60.0f, 26.0f,  60.0f), -0.75f,      0.05f },
            { glm::vec3(40.0f, 19.0f, -40.0f),  0.5f,      -0.15f },
            { glm::vec3(0.0f,  24.0f,   0.0f),  kHalfPi,    0.0f  },
        };
        int idx = std::clamp(opts_.scene, 1, 7) - 1;
        player_.position = kScenes[idx].pos;
        player_.yaw      = kScenes[idx].yaw;
        player_.pitch    = kScenes[idx].pitch;
        player_.velocity = glm::vec3(0.0f);
        in.mouse_dx = 0.0;
        in.mouse_dy = 0.0;
    }
}

// What the keys and buttons do, once per frame: the menu, edit mode and its brushes, weapon
// choice, reload, aim, fly mode, back to the start - and the view (mouse and look stick).
void VulkanEngine::loop_handle_keys(InputFrame& in, float frame_dt) {
    bool menu_edge = in.menu_key && !prev_menu_key_;
    prev_menu_key_ = in.menu_key;
    // E toggles terrain edit mode (task #201). Edge-detected so a held
    // key doesn't oscillate. Only fires while playing — pressing E in
    // the menu shouldn't drop into edit-mode behind the menu.
    bool edit_edge = in.edit_key && !prev_edit_key_;
    prev_edit_key_ = in.edit_key;
    // (at the control house's launch button E presses the button instead)
    if (edit_edge && state_ == State::Playing && !(!terrain_edit_mode_ && (jeep_use() || rocket_press_button()))) {
        terrain_edit_mode_ = !terrain_edit_mode_;
        log::infof("[terrain] edit mode %s (E)",
                   terrain_edit_mode_ ? "ON" : "OFF");
    }
    if (in.cam_key && !prev_cam_key_ && state_ == State::Playing) rocket_toggle_camera();      // C: ride along with the rocket
    prev_cam_key_ = in.cam_key;
    // Digit 9 -> toggle terrain wireframe. Same edge-detect pattern
    // as the E key above. Only fires while playing so a 9 in the
    // pause menu / chat input wouldn't flip it.
    // (fly mode stays as it was through the pause menu)
    // Weapon select: 1 = laser gun, 2 = airstrike tablet, 3 = rifle.
    if (state_ == State::Playing && !terrain_edit_mode_) {      // (edit mode: the number keys pick what to draw)
        if (in.weapon1 && !prev_weapon_key_[0]) { weapon_ = 0; flashlight_on_ = false; }
        if (in.weapon2 && !prev_weapon_key_[1]) { weapon_ = 1; flashlight_on_ = false; }
        if (in.weapon3 && !prev_weapon_key_[2]) { weapon_ = 2; flashlight_on_ = false; }
    }
    if (static const char* tw = std::getenv("VIBE_TEST_WEAPON")) weapon_ = std::clamp(std::atoi(tw), 0, 2);   // (tests: which weapon is in the hand)
    prev_weapon_key_[0] = in.weapon1; prev_weapon_key_[1] = in.weapon2; prev_weapon_key_[2] = in.weapon3;
    // R: reload. Cosmetic (no ammunition is counted): it is shown and replicated, and firing ends it.
    if (state_ == State::Playing && !terrain_edit_mode_ && in.brush_mode_next && !prev_reload_key_ && gun_in_hand() &&
        body_reload_left_ <= 0.0f && body_animator_) {
        ++body_reload_seq_;
        body_reload_left_ = body_animator_->weapon.reload_time;
    }
    prev_reload_key_ = in.brush_mode_next;
    // Right mouse: aim. (Test hook: VIBE_TEST_AIM=1 holds it.)
    static const bool kTestAim = std::getenv("VIBE_TEST_AIM") != nullptr;
    body_aim_ = state_ == State::Playing && !terrain_edit_mode_ && gun_in_hand() && (in.alt_fire_held || kTestAim);
    if (state_ != State::Playing) last_pin_ = game::PlayerInput{};
    if (in.fly_key && !prev_fly_key_ && state_ == State::Playing) {
        fly_mode_ = !fly_mode_;
        player_.velocity = glm::vec3(0.0f);
        if (!fly_mode_) {
            // Landing: if the player ended up inside a wall / post / rock,
            // lift them clear (straight up, in 0.5 m steps) before normal
            // collision takes over again.
            rebuild_tick_aabbs();
            for (int k = 0; k < 80; ++k) {
                const auto box = player_.shape();
                bool inside = false;
                for (const auto& a : tick_aabbs_)
                    if (collision::aabb_overlap(collision::aabb_translate(box, player_.position), a)) { inside = true; break; }
                if (!inside) break;
                player_.position.y += 0.5f;
            }
        }
        log::infof("[player] fly mode %s", fly_mode_ ? "on" : "off");
    }
    prev_fly_key_ = in.fly_key;
    // H: home. Wherever the flight went (orbit included), back on the ground by the chapel,
    // facing the castle, on foot.
    if (in.home_key && !prev_home_key_ && state_ == State::Playing) {
        fly_mode_ = false;
        globe_go_home();
        player_.position = glm::vec3(30.741f, 27.9f, 118.9f);
        player_.velocity = glm::vec3(0.0f);
        player_.yaw = 0.6f;
        player_.pitch = 0.0f;
        prev_player_position_ = player_.position;
        log::info("[home] back to the home spot (H)");
    }
    prev_home_key_ = in.home_key;
    bool wire_edge = in.wireframe_key && !prev_wireframe_key_;
    prev_wireframe_key_ = in.wireframe_key;
    if (wire_edge && state_ == State::Playing) {
        rt_.terrain_wireframe = !rt_.terrain_wireframe;
        log::infof("[terrain] wireframe %s (9)",
                   rt_.terrain_wireframe ? "ON" : "OFF");
    }
    // Hold [ / ] in edit mode to shrink/grow the brush radius
    // (mirrors the common sculpt UX in Blender, Houdini, etc.).
    // 4 m/s rate covers the practical 0.5-50 m clamp in ~12 s; the
    // existing slider in the tools panel still works.
    if (terrain_edit_mode_ && state_ == State::Playing) {
        const float kBrushRate = 4.0f;
        if (in.brush_larger)  terrain_brush_radius_ += kBrushRate * frame_dt;
        if (in.brush_smaller) terrain_brush_radius_ -= kBrushRate * frame_dt;
        terrain_brush_radius_ = std::clamp(terrain_brush_radius_, 0.5f, 50.0f);
    }
    // Q / R cycle brush mode (edge-detected so a held key doesn't spin
    // through every mode each frame). Wraps around the 8-entry enum.
    bool bm_prev_edge = in.brush_mode_prev && !prev_brush_mode_prev_;
    bool bm_next_edge = in.brush_mode_next && !prev_brush_mode_next_;
    prev_brush_mode_prev_ = in.brush_mode_prev;
    prev_brush_mode_next_ = in.brush_mode_next;
    if (terrain_edit_mode_ && state_ == State::Playing &&
        (bm_prev_edge || bm_next_edge)) {
        constexpr int kModeCount = 9;   // matches TerrainBrushMode enum
        int m = static_cast<int>(terrain_brush_mode_);
        // The raymarched terrain cannot be sculpted: skip the height tools there.
        for (int tries = 0; tries < kModeCount; ++tries) {
            m = (m + (bm_next_edge ? 1 : kModeCount - 1)) % kModeCount;
            if (!rt_.terrain_raymarch_enabled || m == 4 || m == 5 || m == 8) break;
        }
        terrain_brush_mode_ = static_cast<TerrainBrushMode>(m);
        static const char* kModeNames[kModeCount] = {
            "Raise", "Lower", "Smooth", "Flatten",
            "GrassAdd", "GrassRemove", "Erode", "ErodeSmooth", "Zones"
        };
        log::infof("[terrain] brush mode -> %s (%s)",
                   kModeNames[m], bm_next_edge ? "R" : "Q");
    }
    if (menu_edge) {
        if (state_ == State::Playing) {
            state_ = State::Paused;
            window_->set_cursor_captured(false);
        } else {
            state_ = State::Playing;
            window_->set_cursor_captured(true);
        }
    }

    // Mouse-look at render rate (low latency). Only when window has focus
    // AND cursor is captured AND we're in Playing state.
    if (state_ == State::Playing && window_->cursor_captured() &&
        window_->has_focus()) {
        float dx = static_cast<float>(in.mouse_dx);
        float dy = static_cast<float>(in.mouse_dy);
        player_.yaw   += dx * game::Player::kMouseSensitivity * rt_.mouse_sens;
        player_.pitch -= dy * game::Player::kMouseSensitivity * rt_.mouse_sens;
        // kHalfPi from engine/vk_engine/internal.h.
        constexpr float kPitchCap = kHalfPi - 0.01f;
        if (player_.pitch >  kPitchCap) player_.pitch =  kPitchCap;
        if (player_.pitch < -kPitchCap) player_.pitch = -kPitchCap;
    }
    // The controller's look stick: an angular speed, so by time (the mouse is a distance).
    if (state_ == State::Playing && window_->has_focus() && (in.look_x != 0.0f || in.look_y != 0.0f)) {
        const float sp = glm::radians(input_.pad_look_speed) * frame_dt;
        player_.yaw   += in.look_x * sp;
        player_.pitch -= in.look_y * sp * input_.pad_look_ratio_y * (input_.pad_invert_y ? -1.0f : 1.0f);
        constexpr float kPitchCap = kHalfPi - 0.01f;
        player_.pitch = glm::clamp(player_.pitch, -kPitchCap, kPitchCap);
    }
}

// Firing: the weapon's timers, the network's turn, then either the terrain editor's brush
// (edit mode) or the rifle / the airstrike marker.
void VulkanEngine::loop_weapons_and_edit(InputFrame& in, float frame_dt) {
    // Auto-fire: hold LMB, fires at game_.fire_rate_rps. The cooldown
    // accumulates frame_dt so the rate is independent of frame rate.
    // fire_rate_rps == 0 falls back to one-shot-per-click.
    if (fire_cooldown_ > 0.0f) fire_cooldown_ -= frame_dt;
    if (muzzle_flash_timer_ > 0.0f) muzzle_flash_timer_ -= frame_dt;
    if (recoil_timer_       > 0.0f) recoil_timer_       -= frame_dt;
    mp_update(frame_dt);               // LAN multiplayer: network, other players' shots, hits
    if (state_ == State::Playing && (window_->cursor_captured() ||
                                       std::getenv("VIBE_TEST_FIRE") || std::getenv("VIBE_TEST_NPC_SHOOT"))) {
        // Raymarch terrain reads FBM directly in the shader — sculpting
        // the heightmap has no visual effect and only triggers wasted
        // chunk rebuilds / BLAS rebuilds / texture uploads.
        if (terrain_edit_mode_) {
            // ---- Phase 4 sculpt path ----
            // Raycast from the eye along the camera forward against
            // the heightfield (Jolt's static body) — works because
            // the heightfield is in NON_MOVING and add_static_heightfield
            // uses the standard collision layer.
            glm::vec3 origin = player_.eye_position();
            glm::vec3 dir    = player_.forward();
            if (physics_) {
                auto rh = physics_->raycast(origin, dir, 400.0f);
                if (rh.hit) {
                    terrain_brush_world_pos_ = rh.position;
                    terrain_brush_has_hit_ = true;
                } else {
                    terrain_brush_has_hit_ = false;
                }
            }
            // Height tools only work on the mesh terrain; grass and zone
            // painting work on both renderers.
            const bool zone_tool = terrain_brush_mode_ == TerrainBrushMode::ZonePaint;
            const bool tool_ok = zone_tool || brush_mode_is_grass(terrain_brush_mode_) ||
                                 !rt_.terrain_raymarch_enabled;
            bool sculpt_active = (in.fire_held || in.alt_fire_held) &&
                                  terrain_brush_has_hit_ && tool_ok;
            if (sculpt_active) {
                // Right-click swaps mode to its inverse for the
                // duration of the stroke: Raise↔Lower for the
                // height brush, GrassAdd↔GrassRemove for the grass
                // brush. Smooth / Flatten stay as-is. Restored
                // after the call so the UI shows the user's
                // selected mode on the next frame.
                TerrainBrushMode prev_mode = terrain_brush_mode_;
                if (in.alt_fire_held && !in.fire_held) {
                    if (terrain_brush_mode_ == TerrainBrushMode::Raise) {
                        terrain_brush_mode_ = TerrainBrushMode::Lower;
                    } else if (terrain_brush_mode_ == TerrainBrushMode::GrassAdd) {
                        terrain_brush_mode_ = TerrainBrushMode::GrassRemove;
                    } else if (terrain_brush_mode_ == TerrainBrushMode::Erode) {
                        terrain_brush_mode_ = TerrainBrushMode::ErodeSmooth;
                    }
                }
                if (zone_tool)      // right button erases (back to automatic)
                    paint_veg(in.alt_fire_held && !in.fire_held);
                else
                    apply_terrain_brush(frame_dt);
                terrain_brush_mode_ = prev_mode;
                // CLOD raster reads the heightmap texture, so the
                // stroke shows up via the terrain_height_dirty_
                // re-upload — no per-chunk mesh rebuild needed.
                if (!terrain_stroke_active_) terrain_stroke_active_ = true;
            } else if (terrain_stroke_active_ && !in.fire_held && !in.alt_fire_held) {
                // Mouse-up: heavy refresh (BLAS + Jolt) deferred until
                // here so the stroke itself stays cheap. Grass-only
                // strokes have no dirty bits to consume — the
                // mask reupload already happened the frame the
                // stroke modified the CPU mirror.
                if (zone_tool) {
                    if (veg_auto_replant_) request_paint_apply(false);
                } else if (!brush_mode_is_grass(terrain_brush_mode_)) {
                    refresh_terrain_blas();
                    refresh_terrain_collision();
                    // Re-bake the curvature/sky-vis aux for the new
                    // surface (cheap ~100 ms, only at stroke end).
                    vkDeviceWaitIdle(device_);
                    destroy_terrain_aux_texture();
                    init_terrain_aux_texture();
                }
                terrain_stroke_active_ = false;
            }
        } else if (flashlight_on_) {
            // Left click: next brightness (4 levels, wrapping to the lowest).
            if (in.fire) {
                flash_level_ = (flash_level_ + 1) % 4;
                if (audio_) audio_->play_local(ClipID::Jump, 0.22f, 0.02f, 1.7f + 0.15f * float(flash_level_));
            }
            airstrike_click_ = false;
        } else if (!game_.weapons_enabled) {
            airstrike_click_ = false;            // game setting: player weapons off
        } else if (weapon_ == 1) {
            // Airstrike tablet: a click marks the ground under the crosshair.
            airstrike_click_ = in.fire;
            fire_cooldown_ = 0.0f;
        } else {
            // The laser fires at the game's rate (0: one bolt per click); the rifle is automatic.
            const bool rifle = weapon_ == 2;
            float rps = rifle ? kRifleRps : std::max(0.0f, game_.fire_rate_rps);
            if (rps <= 0.0f) {
                if (in.fire) {
                    fire_projectile(player_.eye_position(), player_.forward(), rifle);
                }
            } else if (in.fire_held) {
                float interval = 1.0f / rps;
                if (fire_cooldown_ <= 0.0f) {
                    fire_projectile(player_.eye_position(), player_.forward(), rifle);
                    fire_cooldown_ += interval;
                }
            } else {
                fire_cooldown_ = 0.0f;  // reset between bursts
            }
        }
    }

    tick_airstrike(state_ == State::Playing ? frame_dt : 0.0f, airstrike_click_ && state_ == State::Playing);
    airstrike_click_ = false;
}

// The player and the physics world, in fixed steps of 1 / 120 s: movement (walking, swimming,
// flying), collisions, the rigid bodies - and once a second the statistics line of the log.
void VulkanEngine::loop_simulate(InputFrame& in, float frame_dt) {
    constexpr float kFixedDt = 1.0f / 120.0f;  // 120 Hz physics
    constexpr int   kMaxTicksPerFrame = 6;
    // Fixed-timestep physics: accumulate frame dt, drain in fixed ticks.
    last_physics_ticks_ = 0;
    if (state_ == State::Playing) {
        physics_accumulator_ += frame_dt;
        if (physics_accumulator_ > kFixedDt * (kMaxTicksPerFrame + 1)) {
            sim_debt_s_ += physics_accumulator_ - kFixedDt * kMaxTicksPerFrame;      // (time that will not be simulated)
            physics_accumulator_ = kFixedDt * kMaxTicksPerFrame;  // anti-spiral
        }
        while (physics_accumulator_ >= kFixedDt &&
               last_physics_ticks_ < kMaxTicksPerFrame) {
            prev_player_position_ = player_.position;
            game::PlayerInput pin{};
            pin.fwd = in.fwd; pin.back = in.back;
            pin.left = in.left; pin.right = in.right;
            pin.ax = in.move_x; pin.ay = in.move_y;     // (the stick: how far, not only which way)
            pin.jump = in.jump;
            pin.sprint = in.sprint;
            pin.crawl = in.crawl;
            pin.aiming = body_aim_;
            pin.gravity = game_.gravity;
            last_pin_ = pin;                 // (what the player wants: the animation's prediction reads it)
            // Water: how much of the 1.8 m body is under the surface.
            // Swimming starts once the water reaches the chest (the
            // feet no longer carry the body) and ends when it drops
            // back to the waist; in between the player wades.
            {
                const float wl = rt_.water_level;
                const float feet0 = player_.position.y - 0.9f;
                // Winter: the lake is frozen - no wading or swimming, the ice carries (below).
                const bool frozen = season_w_[3] > 0.6f;
                const bool open_w = !frozen && !fly_mode_ && water_at(player_.position.x, player_.position.z);
                const float sub = open_w ? glm::clamp((wl - feet0) / 1.8f, 0.0f, 1.0f) : 0.0f;
                const bool was_swim = player_swimming_;
                player_swimming_ = open_w && sub > (was_swim ? 0.52f : 0.64f);
                pin.swim = player_swimming_;
                pin.water_y = wl;
                pin.wade = player_swimming_ ? 0.0f : glm::smoothstep(0.08f, 0.6f, sub);
                // Entering the water: a splash sized by how hard we hit it.
                const float prev_feet = prev_player_position_.y - 0.9f;
                (void)prev_feet;
                if (open_w && player_water_prev_sub_ <= 0.0f && sub > 0.0f && player_.velocity.y < -2.5f)
                    spawn_water_splash(glm::vec3(player_.position.x, wl, player_.position.z),
                                       glm::clamp(1.2f + 0.25f * -player_.velocity.y, 1.5f, 5.0f));
                player_water_prev_sub_ = sub;
                // Moving through the surface leaves rings behind.
                const float hsp = std::sqrt(player_.velocity.x * player_.velocity.x +
                                            player_.velocity.z * player_.velocity.z);
                if (sub > 0.03f && sub < 0.97f) {
                    swim_ripple_acc_ += kFixedDt * (0.6f + hsp);
                    if (swim_ripple_acc_ > 1.6f) {
                        swim_ripple_acc_ = 0.0f;
                        add_water_ripple(player_.position.x, player_.position.z,
                                         hsp > 0.5f ? 0.022f : 0.012f);
                    }
                }
            }
            pin.mouse_dx = 0.0f;  // mouse already applied above
            pin.mouse_dy = 0.0f;
            glm::vec3 pre_vel = player_.velocity;
            bool was_on_ground = player_.on_ground;
            const int pre_air_jumps = player_.air_jumps_left;
            if (fly_mode_) {
                // Free flight: along the view direction, Space up, Ctrl
                // down, Shift for a burst. No gravity, no collision.
                const float cp = std::cos(player_.pitch), sp = std::sin(player_.pitch);
                const glm::vec3 fwd(std::sin(player_.yaw) * cp, sp, -std::cos(player_.yaw) * cp);
                const glm::vec3 rgt(std::cos(player_.yaw), 0.0f, std::sin(player_.yaw));
                glm::vec3 wish(0.0f);
                if (pin.fwd)   wish += fwd;
                if (pin.back)  wish -= fwd;
                if (pin.right) wish += rgt;
                if (pin.left)  wish -= rgt;
                if (pin.jump)  wish.y += 1.0f;
                if (pin.crawl) wish.y -= 1.0f;
                // The higher above the ground, the faster: 40 m/s low down, kilometres
                // a second on the way to orbit (so the planet can be left in a minute).
                // (height above the lowest ground, not above the hill being crossed: with the ground's
                // own height in it the speed rose and fell over every ridge - the flight surged)
                const float alt_fly = player_.position.y - rt_.water_level;
                const float speed = (pin.sprint ? 120.0f : 40.0f) * glm::clamp(alt_fly / 60.0f, 1.0f, 2000.0f);   // walking is 8 m/s
                const glm::vec3 target = glm::dot(wish, wish) > 1e-6f ? glm::normalize(wish) * speed : glm::vec3(0.0f);
                player_.velocity += (target - player_.velocity) * std::min(1.0f, 8.0f * kFixedDt);
                player_.position += player_.velocity * kFixedDt;
                // The ceiling, for now: 1000 km. Above that ordinary coordinates lose
                // too much precision (camera-relative rendering is the next step for going further).
                if (player_.position.y > 1.0e6f) { player_.position.y = 1.0e6f; player_.velocity.y = std::min(player_.velocity.y, 0.0f); }
                // Never below the ground.
                const float gmin = sample_terrain_height(player_.position.x, player_.position.z) + 0.95f;
                if (player_.position.y < gmin) { player_.position.y = gmin; player_.velocity.y = std::max(player_.velocity.y, 0.0f); }
                player_.on_ground = false;
                player_.crouch_factor = 0.0f;
                if (physics_) { CPU_SCOPE("+physics"); physics_->step(kFixedDt); }
                update_projectiles(kFixedDt);
                update_particles(kFixedDt);
                update_decals(kFixedDt);
                physics_accumulator_ -= kFixedDt;
                ++last_physics_ticks_;
                continue;
            }
            { CPU_SCOPE("tick-aabbs"); rebuild_tick_aabbs(); }
            CPU_SCOPE("tick-rest");
            // VIBE_TEST_BROADPHASE=1: replay the tick against EVERY static
            // collider and report any difference from the near-set result.
            static const bool kCheckBroad = std::getenv("VIBE_TEST_BROADPHASE") != nullptr;
            game::Player ref_player = player_;
            { CPU_SCOPE("+player");
            game::update_player(player_, pin, tick_aabbs_, kFixedDt, tick_static_count_); }
            // Frozen lake: the ice is a floor at the water level.
            if (season_w_[3] > 0.6f && !fly_mode_ && rt_.water_enabled &&
                player_.position.y - 0.9f < rt_.water_level + 0.02f &&
                water_at(player_.position.x, player_.position.z)) {
                player_.position.y = rt_.water_level + 0.92f;
                if (player_.velocity.y < 0.0f) player_.velocity.y = 0.0f;
                player_.on_ground = true;
                player_.air_jumps_left = game::Player::kAirJumps;
            }
            if (kCheckBroad) {
                static std::vector<collision::AABB> full;
                static uint64_t ticks = 0, bad = 0;
                full.assign(world_.aabbs.begin(), world_.aabbs.end());
                full.insert(full.end(), voxel_->collision_aabbs.begin(), voxel_->collision_aabbs.end());
                const size_t full_static = full.size();
                full.insert(full.end(), tick_aabbs_.begin() + static_cast<std::ptrdiff_t>(tick_static_count_), tick_aabbs_.end());
                game::update_player(ref_player, pin, full, kFixedDt, full_static);
                if (glm::distance(ref_player.position, player_.position) > 1e-4f ||
                    ref_player.on_ground != player_.on_ground) ++bad;
                if (++ticks % 300 == 0)
                    log::infof("[broadphase-check] %llu ticks, %llu mismatches, near set %zu of %zu colliders",
                               static_cast<unsigned long long>(ticks), static_cast<unsigned long long>(bad),
                               tick_static_count_, full_static);
            }

            // Terrain ground clamp. update_player collides only
            // against AABBs; the heightfield isn't an AABB so the
            // player would free-fall through it. We:
            //   1. Sample terrain at player XZ.
            //   2. Estimate slope normal from finite differences;
            //      anything steeper than ~50° is a "wall" — no
            //      ground-clamp, the player slides off.
            //   3. If feet sank below ground, snap up.
            //   4. STICKY GROUNDING: if the player WAS grounded
            //      last tick and is still moving downward (or zero)
            //      vertically and their feet sit within kStepDown
            //      above the new ground sample, snap them DOWN.
            //      Without this, walking down a slope alternates
            //      airborne/grounded each frame — gravity accumulates
            //      between snaps and the player slides faster than
            //      they intended (the user's "weird sliding").
            {
                constexpr float kHalfHeight   = 0.9f;
                constexpr float kStepDown     = 0.45f;
                constexpr float kMinGroundN   = 0.55f;  // ≈56° max walkable
                constexpr float kSampleDelta  = 1.0f;
                float gx = player_.position.x;
                float gz = player_.position.z;
                // The stone bridge's deck is a floor like the terrain (it has no collision
                // boxes: tiled boxes on a sloping, diagonal deck are a field of lips) - for a
                // player who is on or above it, not for one walking underneath.
                const float feet_now = player_.position.y - kHalfHeight;
                bool deck_hit = false;
                auto ground_at = [&](float x, float z) {
                    float g = sample_terrain_height(x, z);
                    deck_hit = false;
                    if (bridge_placed_) {
                        const float dk = game::stone_bridge_deck_height(bridge_a_, bridge_b_, 2.0f * castle_road_.half_width, x, z);
                        if (dk > g && feet_now > dk - 0.7f) { g = dk; deck_hit = true; }
                    }
                    return g;
                };
                float h0  = ground_at(gx, gz);
                const bool on_deck = deck_hit;
                // (on the deck the floor is level for the slope test: a sample a metre to the
                // side may lie beyond the parapet, 17 m down - that is not a slope)
                float hpx = on_deck ? h0 : ground_at(gx + kSampleDelta, gz);
                float hnx = on_deck ? h0 : ground_at(gx - kSampleDelta, gz);
                float hpz = on_deck ? h0 : ground_at(gx, gz + kSampleDelta);
                float hnz = on_deck ? h0 : ground_at(gx, gz - kSampleDelta);
                glm::vec3 n(-(hpx - hnx) / (2.0f * kSampleDelta),
                             1.0f,
                             -(hpz - hnz) / (2.0f * kSampleDelta));
                n = glm::normalize(n);
                bool walkable = n.y > kMinGroundN;

                float feet = player_.position.y - kHalfHeight;
                if (feet < h0) {
                    // Snapped up onto terrain.
                    player_.position.y = h0 + kHalfHeight;
                    if (player_.velocity.y < 0.0f) player_.velocity.y = 0.0f;
                    if (walkable) player_.on_ground = true;
                } else if (walkable && was_on_ground && !player_swimming_ &&
                           !player_.on_ground &&
                           player_.velocity.y <= 0.5f &&
                           (feet - h0) < kStepDown) {
                    // Sticky-ground: pull the player down onto the
                    // surface they were just on. Keeps slopes and
                    // stairs feeling "stuck to the ground" instead of
                    // hopping on every step.
                    //
                    // Skip if slide_move already grounded us on a brush
                    // (player_.on_ground TRUE this tick). Otherwise the
                    // castle floor brush sits 5 cm above the heightmap
                    // plateau and sticky-ground yanks the player DOWN
                    // through the floor every tick — depenetration
                    // pushes them back up next frame, oscillating
                    // between 22.0 and 22.05. That oscillation:
                    //   - reads as "invisible terrain" (heightmap
                    //     under the rendered castle floor),
                    //   - disrupts stair step-up (player feet keep
                    //     reverting to terrain Y between steps).
                    player_.position.y = h0 + kHalfHeight;
                    player_.velocity.y = 0.0f;
                    player_.on_ground = true;
                }
            }
            // Audio triggers from the player tick. Land sound used to
            // fire on every air→ground transition, but on_ground flickers
            // each tick while walking over uneven terrain → it doubled as
            // an unwanted footstep sound. Removed; only jump remains.
            if (audio_ && state_ == State::Playing) {
                if (was_on_ground && !player_.on_ground && pin.jump && !player_swimming_) {
                    audio_->play_local(ClipID::Jump, 0.7f, 0.06f, 0.10f);
                }
                // The second jump, in the air: it had no sound at all.
                if (player_.air_jumps_left < pre_air_jumps && !player_.on_ground && !player_swimming_)
                    audio_->play_local(ClipID::Jump, 0.5f, 0.06f, 0.10f);
                // Landing: only after a real fall (5 m/s down or more). The plain "touched the
                // ground" test flickers while walking over bumps - that is why it was removed.
                if (!was_on_ground && player_.on_ground && pre_vel.y < -5.0f && !player_swimming_)
                    audio_->play_local(ClipID::Land, glm::clamp(-pre_vel.y / 15.0f, 0.3f, 1.0f), 0.06f, 0.08f);
            }
            { CPU_SCOPE("+pushes"); apply_player_pushes(pre_vel); }
            if (physics_) { CPU_SCOPE("+physics"); physics_->step(kFixedDt); }
            { CPU_SCOPE("+projectiles"); update_projectiles(kFixedDt); }
            { CPU_SCOPE("+particles"); update_particles(kFixedDt); }
            { CPU_SCOPE("+decals"); update_decals(kFixedDt); }
            physics_accumulator_ -= kFixedDt;
            ++last_physics_ticks_;
        }

        // Continuous spawner: drop a new tilted box on a timer derived
        // from the user-configurable cubes-per-minute setting.
        float interval = 60.0f / std::max(1.0f, float(game_.cubes_per_minute));
        spawn_timer_ += frame_dt;
        // (2026-11-25: nothing falls on the castle any more - the user asked for it to stop.
        //  VIBE_BOX_RAIN=1 brings it back.)
        static const bool box_rain = std::getenv("VIBE_BOX_RAIN") != nullptr;
        while (spawn_timer_ >= interval) {
            spawn_timer_ -= interval;
            if (box_rain) spawn_random_box();
        }

        pos_log_timer_ += frame_dt;
        if (pos_log_timer_ >= 1.0f) {
            pos_log_timer_ = 0.0f;
            glm::vec2 hv(player_.velocity.x, player_.velocity.z);
            log::infof("player pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f) "
                       "speed_h=%.3f grounded=%d feet_y=%.3f fps=%.1f ticks=%d",
                       player_.position.x, player_.position.y, player_.position.z,
                       player_.velocity.x, player_.velocity.y, player_.velocity.z,
                       glm::length(hv),
                       player_.on_ground ? 1 : 0,
                       player_.position.y - game::Player::kHalfExtents.y,
                       ema_fps_, last_physics_ticks_);
            log::infof("state  proj=%zu particles=%zu dyn_props=%zu "
                       "tlas_n=%u frame=%llu vk_warn=%llu vk_err=%llu",
                       projectiles_.size(), particles_.size(),
                       dyn_props_.size(), last_tlas_n_,
                       static_cast<unsigned long long>(frame_number_),
                       static_cast<unsigned long long>(g_validation_warning_count.load()),
                       static_cast<unsigned long long>(g_validation_error_count.load()));
            if (gpu_query_pool_) {
                log::infof("gpu    frame=%.2f ms (TDR limit ~2000 ms)",
                           gpu_timers_.scene_ms);
                prof_log();
            }
        }
    }
}

// Per-frame upkeep between the simulation and the frame: jobs that finished in the
// background (replanting, baking), rebuilds a changed setting asked for, the environment
// (time of day, weather), the listener's position for the sound.
void VulkanEngine::loop_upkeep(InputFrame& in, float frame_dt) {
    // Voxel destruction: debounced heavy work (collapse + collision
    // rebuild) once per frame, off the per-bullet impact path.
    process_voxel_updates(frame_dt);
    tick_flashlight(frame_dt);       // F key, sway, beam direction
    // Test hooks: VIBE_TEST_ZONE=<zone> paints a 70 m disc of that zone
    // 90 m ahead of the camera on frame 5; VIBE_TEST_EDIT=1 opens edit
    // mode with the zone tool (for UI captures).
    {
        static const char* kTestZone = std::getenv("VIBE_TEST_ZONE");
        static const bool kTestEdit = std::getenv("VIBE_TEST_EDIT") != nullptr;
        static const bool kTestMenu = std::getenv("VIBE_TEST_MENU") != nullptr;
        if (kTestMenu && frame_number_ == 5) state_ = State::Paused;
        if (frame_number_ == 5 && (kTestZone || kTestEdit)) {
            const glm::vec3 fw(std::sin(player_.yaw), 0.0f, -std::cos(player_.yaw));
            terrain_brush_world_pos_ = player_.position + fw * 90.0f;
            terrain_brush_world_pos_.y = sample_terrain_height(terrain_brush_world_pos_.x, terrain_brush_world_pos_.z);
            terrain_brush_has_hit_ = true;
            if (kTestZone) {
                const float keep = terrain_brush_radius_;
                terrain_brush_radius_ = 70.0f;
                paint_active_ = std::clamp(std::atoi(kTestZone), 1, int(paint_.presets.size()) - 1);
                paint_veg(false);
                terrain_brush_radius_ = keep;
                request_paint_apply(false);
            }
            if (kTestEdit) { terrain_edit_mode_ = true; terrain_brush_mode_ = TerrainBrushMode::ZonePaint; }
        }
    }
    if (veg_replant_request_ && !veg_job_active_) {
        veg_replant_request_ = false;
        start_veg_replant();                 // placement runs in the background
    }
    wv_nf_poll();
    hw_poll();
    hw_objects_poll();
    river_stream::poll(allocator_, player_.eye_position(), globe::current().home, rt_.water_level, std::clamp(rt_.terrain_raymarch_octaves, 4, 24));   // (the river's plants beyond the home tile)
    if (veg_job_active_ && veg_job_.valid() &&
        veg_job_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        VegJobResult r = veg_job_.get();
        veg_job_active_ = false;
        finish_veg_replant(std::move(r));
        // Ground cover depends on where the tree crowns are - but that
        // rebuild stalls the game for most of a second, so while editing
        // it waits until edit mode is left.
        if (terrain_edit_mode_) scatter_after_edit_ = true;
        else scatter_rebuild_request_ = true;
    }
    if (scatter_after_edit_ && !terrain_edit_mode_) {
        scatter_after_edit_ = false;
        scatter_rebuild_request_ = true;
    }
    if (scatter_rebuild_request_) {
        scatter_rebuild_request_ = false;
        rebuild_scatter();
    }

    // Castle gate animation — proximity trigger + lerp toward target
    // angles. Runs once per frame (the lerp is dt-relative so it's
    // not tied to the physics fixed-step).
    update_doors(frame_dt);

    // Recompute the sun shadow-map light view-proj before update_scene_ubo
    // so the matrix copied into the UBO this frame matches the geometry
    // we'll draw into the shadow target.
    // The sun's shadow map and the small plants' are redrawn in turn, one a frame (each
    // keeps the light matrix it was drawn with until its next turn).
    if ((frame_number_ & 1) == 0) update_sun_shadow_light_vp();
    // If the user changed the shadow-map resolution slider, recreate the
    // image at the new size. Cheap (one vkDeviceWaitIdle + a 4-16MB
    // alloc); only fires when the slider actually moves.
    if (rt_.shadow_map_resolution != sun_->shadow_dim) {
        vkDeviceWaitIdle(device_);
        destroy_sun_shadow_resources();
        init_sun_shadow_resources();
    }
    // (The chunk-LOD stride rebake + near-densify systems are gone —
    // the quadtree CLOD terrain has no per-chunk geometry to rebake.
    // Density follows the camera continuously via per-frame node
    // selection in update_terrain_clod_instances.)
    // Heightmap-bake supersample changed → recreate texture and
    // restart the worker so subsequent jobs run at the new ss.
    // The user can flip 1x/2x/4x freely; texture realloc is cheap.
    if (rt_.terrain_bake_supersample != terrain_shadow_active_ss_) {
        vkDeviceWaitIdle(device_);
        stop_terrain_shadow_worker();
        destroy_terrain_shadow_texture();
        rebuild_terrain_shadow_texture();
        start_terrain_shadow_worker();
        // Trigger an immediate re-bake at the new ss so the worker
        // refines beyond the SS=1 fallback the sync-bake replicated.
        float p_rad = glm::radians(rt_.sun_pitch_deg);
        float y_rad = glm::radians(rt_.sun_yaw_deg);
        glm::vec3 cur(std::sin(y_rad) * std::cos(p_rad),
                      std::sin(p_rad),
                      std::cos(y_rad) * std::cos(p_rad));
        enqueue_terrain_shadow_rebake(cur);
    }
    // update_scene_ubo() moved into draw(), after the prev-frame fence
    // wait: writing it here raced the GPU still reading it for the
    // previous frame (single host-visible UBO shared by both slots).
    // Heightmap sun-shadow is sun-direction-dependent. Instead of
    // re-baking the whole 1024² texture in one go (~100 ms hitch),
    // we tile the bake (kShadowTileSize per side), sort tiles by
    // distance to the camera, and drain a small budget per frame.
    // Near-camera tiles update first so grass shadows under the
    // player respond promptly to the sun slider; distant tiles
    // catch up over the next handful of frames. Threshold ~1.4°
    // — enough to absorb continuous slider drag without thrashing
    // the queue.
    // Lazy-build for the rasterised mesh/chunks/grass — only fires
    // the first time the user toggles a raymarch flag off (init
    // skips the build when raymarch is on to save VRAM).
    if (!rt_.terrain_raymarch_enabled) ensure_terrain_raster_built();
    if (rt_.grass_enabled && !rt_.grass_raymarch_enabled) {
        ensure_grass_raster_built();
    }
    // TAAU toggle: re-point compose's history sampler at taau_->view or
    // history_view_. Cheap when state matches.
    rewrite_compose_for_taau();
    // FSR2 toggle: takes priority over TAAU/TAA in compose's history
    // binding when the context is valid. wait_idle is paid only on the
    // actual switch.
    rewrite_compose_for_fsr2();
    // FSR3 toggle: only meaningful when fsr2_enabled is on AND the
    // backend is 1 (FSR3). Same wait-idle-only-on-actual-switch semantics.
    rewrite_compose_for_fsr3();
    // DLSS: output (re)creation + compose rebind, never mid-recording.
    ensure_dlss_bindings();
    ensure_cloud_target();
    // Day / night clock + weather drive the sun, sky, clouds, fog and wind.
    { CPU_SCOPE("env"); apply_environment(state_ == State::Playing ? frame_dt : 0.0f); }
    tick_sun_horizon();      // far-mountain sun shadow follows the light

    // Heightmap-shadow texture is sampled by cube.frag's mesh-terrain
    // branch and grass.vert (raster grass). When raymarch terrain is on
    // AND raster grass is off, nothing reads it — skip the rebake.
    if (!terrain_data_.heights.empty() && needs_heightmap_shadow()) {
        float p_rad = glm::radians(rt_.sun_pitch_deg);
        float y_rad = glm::radians(rt_.sun_yaw_deg);
        glm::vec3 cur_sun(std::sin(y_rad) * std::cos(p_rad),
                          std::sin(p_rad),
                          std::cos(y_rad) * std::cos(p_rad));
        if (glm::distance(cur_sun, terrain_shadow_target_sun_dir_) > 0.025f) {
            enqueue_terrain_shadow_rebake(cur_sun);
        }
        tick_terrain_shadow_progressive();
    }
    // Menu: 'Recalculate shadows' - force a full re-bake of everything
    // precomputed: the heightmap sun-shadow texture (mesh terrain, grass,
    // deferred terrain), the curvature / sky-visibility aux map, and the
    // terrain BLAS + collision.
    if (shadow_rebake_request_ && !terrain_data_.heights.empty()) {
        shadow_rebake_request_ = false;
        const float p_rad = glm::radians(rt_.sun_pitch_deg), y_rad = glm::radians(rt_.sun_yaw_deg);
        const glm::vec3 cur_sun(std::sin(y_rad) * std::cos(p_rad), std::sin(p_rad),
                                std::cos(y_rad) * std::cos(p_rad));
        vkDeviceWaitIdle(device_);
        destroy_terrain_aux_texture();
        init_terrain_aux_texture();
        refresh_terrain_blas();
        refresh_terrain_collision();
        enqueue_terrain_shadow_rebake(cur_sun);
        log::info("[shadow] full re-bake requested from the menu");
    }
    // Re-upload the heightmap delta texture if the user sculpted /

    // added plateau noise this frame. Cheap one-shot upload of
    // dim*dim*4 bytes; happens only on the frame the dirty bit
    // is set, not every frame the brush is held.
    if (terrain_height_dirty_) {
        refresh_terrain_height_texture();
        refresh_terrain_height_full_texture();
    }
    // Reupload grass eligibility mask if a GrassAdd/GrassRemove
    // stroke modified the CPU mirror this frame. Same one-shot
    // upload pattern as the heightmap delta texture above.
    if (grass_mask_dirty_) {
        reupload_grass_mask();
    }
    // Audio listener follows the camera. Reaping drained one-shots
    // also runs here so the active-voice list stays small.
    if (audio_) {
        audio_->set_listener(player_.eye_position(), player_.forward());
        car_audio_tick(last_frame_dt_);
    }
}

} // namespace vibe
