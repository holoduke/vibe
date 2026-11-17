// LAN multiplayer, game side: feeds the local player into the net session
// (engine/net/net.h), poses and draws the other players, replays their
// shots, decides hits for the local player's shots, health and respawn.
//
// Other players use the same soldier + pose solver as the local body
// (player_body.cpp): for each of them the solver is run with THEIR state
// swapped in, writing into their own vertex buffer.
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/net/net.h"
#include "engine/audio.h"
#include "engine/log.h"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace qlike {

namespace {
constexpr float kHitDamage = 20.0f;

bool seg_aabb(glm::vec3 a, glm::vec3 b, glm::vec3 mn, glm::vec3 mx, float& t_hit) {
    const glm::vec3 d = b - a;
    float t0 = 0.0f, t1 = 1.0f;
    for (int k = 0; k < 3; ++k) {
        if (std::abs(d[k]) < 1e-7f) { if (a[k] < mn[k] || a[k] > mx[k]) return false; continue; }
        float u0 = (mn[k] - a[k]) / d[k], u1 = (mx[k] - a[k]) / d[k];
        if (u0 > u1) std::swap(u0, u1);
        t0 = std::max(t0, u0); t1 = std::min(t1, u1);
        if (t0 > t1) return false;
    }
    t_hit = t0;
    return true;
}
}  // namespace

void VulkanEngine::mp_init() {
    // Off for automated runs (--frames / screenshots) unless QLIKE_NET=1, and with QLIKE_NO_NET.
    const bool automated = opts_.max_frames > 0 || !opts_.screenshot_path.empty();
    if (std::getenv("QLIKE_NO_NET") || (automated && !std::getenv("QLIKE_NET"))) return;
    const char* cn = std::getenv("COMPUTERNAME");
    net_ = new net::Session();
    if (!net_->start(cn && *cn ? cn : "player")) { delete net_; net_ = nullptr; return; }
    mp_spawn_ = player_.position;
    mp_health_ = 100.0f;
}

void VulkanEngine::mp_shutdown() {
    if (net_) { net_->stop(); delete net_; net_ = nullptr; }
    for (auto& a : mp_avatars_) {
        for (uint32_t i = 0; i < kFrameOverlap; ++i)
            if (a.vbuf[i]) { vmaDestroyBuffer(allocator_, a.vbuf[i], a.valloc[i]); a.vbuf[i] = VK_NULL_HANDLE; a.mapped[i] = nullptr; }
        a.id = 0;
    }
}

void VulkanEngine::mp_feed(const std::string& line) {
    mp_feed_.push_back({ line, 6.0f });
    if (mp_feed_.size() > 5) mp_feed_.erase(mp_feed_.begin());
    log::infof("[net] %s", line.c_str());
}

// Once per frame, after the local body was posed (frame.cpp).
void VulkanEngine::mp_update(float dt) {
    if (!net_) return;
    dt = glm::clamp(dt, 0.0f, 0.1f);
    net::PlayerState ls{};
    ls.px = player_.position.x; ls.py = player_.position.y; ls.pz = player_.position.z;
    ls.vx = player_.velocity.x; ls.vy = player_.velocity.y; ls.vz = player_.velocity.z;
    ls.yaw = player_.yaw; ls.pitch = player_.pitch; ls.crouch = player_.crouch_factor;
    ls.flags = uint8_t((player_.on_ground ? 1 : 0) | (player_swimming_ ? 2 : 0) | (flashlight_on_ ? 4 : 0));
    ls.health = uint8_t(glm::clamp(mp_health_, 0.0f, 100.0f));
    net_->update(glfwGetTime(), ls);

    for (auto& f : mp_feed_) f.ttl -= dt;
    mp_feed_.erase(std::remove_if(mp_feed_.begin(), mp_feed_.end(), [](const MpFeed& f) { return f.ttl <= 0.0f; }), mp_feed_.end());
    mp_hurt_ = std::max(0.0f, mp_hurt_ - dt * 2.5f);

    // QLIKE_TEST_WX=<seconds>:<type>: pick that weather type after that many seconds (sync tests).
    if (static const char* tw = std::getenv("QLIKE_TEST_WX"); tw) {
        static bool done = false;
        float at = 0.0f; int ty = 0;
        if (!done && std::sscanf(tw, "%f:%d", &at, &ty) == 2 && glfwGetTime() > double(at)) { done = true; rt_.weather_enabled = true; rt_.weather_preset = ty; }
    }
    mp_sync_world(dt);

    // ---- events ----
    for (const net::Event& e : net_->take_events()) {
        if (e.kind == net::Event::Fire) {
            const glm::vec3 o(e.a[0], e.a[1], e.a[2]), d(e.b[0], e.b[1], e.b[2]);
            mp_remote_shot_ = true;
            fire_projectile(o, d);
            mp_remote_shot_ = false;
            if (audio_) audio_->play_at(ClipID::Laser, o, 0.8f);
            for (auto& a : mp_avatars_) if (a.id == e.from) { a.recoil_t = kRecoilDuration; a.muzzle_t = kMuzzleFlashDuration; }
        } else if (e.kind == net::Event::Hit) {
            if (e.to == net_->id() && mp_health_ > 0.0f) {
                mp_health_ -= e.v;
                mp_hurt_ = 1.0f;
                if (audio_) audio_->play_local(ClipID::Impact, 0.8f);
                if (mp_health_ <= 0.0f) {
                    net::Event d{}; d.kind = net::Event::Death; d.from = net_->id(); d.to = e.from;
                    net_->send_event(d);
                    mp_feed(net_->name_of(e.from) + " killed " + net_->name_of(net_->id()));
                    // Respawn where this session started.
                    player_.position = mp_spawn_;
                    player_.velocity = glm::vec3(0.0f);
                    mp_health_ = 100.0f;
                }
            }
        } else if (e.kind == net::Event::Death) {
            if (e.to == net_->id()) ++mp_kills_;
            mp_feed(net_->name_of(e.to) + " killed " + net_->name_of(e.from));
        } else if (e.kind == net::Event::Leave) {
            mp_feed("a player left");
        }
    }

    // ---- avatars: one slot per remote player ----
    const auto& players = net_->players();
    for (auto& a : mp_avatars_) {
        bool alive = false;
        for (const auto& p : players) if (p.id == a.id && a.id) alive = true;
        if (!alive) { a.id = 0; a.drawn = false; }
    }
    for (const auto& p : players) {
        if (!p.fresh) continue;
        MpAvatar* av = nullptr;
        for (auto& a : mp_avatars_) if (a.id == p.id) av = &a;
        if (!av) {
            for (auto& a : mp_avatars_) if (!a.id) { av = &a; break; }
            if (!av) continue;
            MpAvatar fresh{};
            for (uint32_t i = 0; i < kFrameOverlap; ++i) { fresh.vbuf[i] = av->vbuf[i]; fresh.valloc[i] = av->valloc[i]; fresh.mapped[i] = av->mapped[i]; }
            *av = fresh;
            av->id = p.id;
            av->fresh_pose = true;
            mp_feed(net_->name_of(p.id) + " joined");
        }
        av->st = p.state;
    }

}

// Pose the other players with the local solver (their state swapped in).
void VulkanEngine::mp_pose_avatars(float dt) {
    if (!net_ || body_bones_.empty()) return;
    dt = glm::clamp(dt, 0.0f, 0.1f);
    const size_t nv = body_verts_.size();
    struct Saved {
        glm::vec3 pos, vel; float yaw, pitch, crouch; bool ground, swim, flash;
        glm::vec3 eye; glm::mat4 model; glm::vec3 root; float phase, move, air, recoil; glm::vec2 dir; bool drawn;
    } sv{ player_.position, player_.velocity, player_.yaw, player_.pitch, player_.crouch_factor, player_.on_ground,
          player_swimming_, flashlight_on_, current_frame_view_.eye_pos, body_model_, body_root_off_, body_phase_,
          body_move_, body_air_, body_recoil_, body_dir_, body_drawn_ };
    const uint32_t slot = frame_number_ % kFrameOverlap;
    for (auto& a : mp_avatars_) {
        a.drawn = false;
        if (!a.id) continue;
        a.recoil_t = std::max(0.0f, a.recoil_t - dt);
        a.muzzle_t = std::max(0.0f, a.muzzle_t - dt);
        for (uint32_t i = 0; i < kFrameOverlap; ++i) {
            if (a.vbuf[i]) continue;
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sizeof(Vertex) * nv;
            bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo ai{};
            if (vmaCreateBuffer(allocator_, &bci, &aci, &a.vbuf[i], &a.valloc[i], &ai) != VK_SUCCESS) { a.vbuf[i] = VK_NULL_HANDLE; continue; }
            a.mapped[i] = ai.pMappedData;
        }
        if (!a.mapped[slot]) continue;
        const net::PlayerState& s = a.st;
        player_.position = glm::vec3(s.px, s.py, s.pz);
        player_.velocity = glm::vec3(s.vx, s.vy, s.vz);
        player_.yaw = s.yaw; player_.pitch = s.pitch; player_.crouch_factor = s.crouch;
        player_.on_ground = (s.flags & 1) != 0;
        player_swimming_ = (s.flags & 2) != 0;
        flashlight_on_ = false;
        const glm::vec3 eye = player_.eye_position();
        current_frame_view_.eye_pos = eye;
        body_phase_ = a.phase; body_move_ = a.move; body_air_ = a.air; body_dir_ = a.dir; body_root_off_ = a.root;
        body_recoil_ = recoil_kick(a.recoil_t, kRecoilDuration, kRecoilStroke);
        mp_pose_dst_ = static_cast<Vertex*>(a.mapped[slot]);
        update_player_body(dt);
        mp_pose_dst_ = nullptr;
        a.prev_model = a.fresh_pose ? body_model_ : a.model;
        a.fresh_pose = false;
        a.model = body_model_;
        a.phase = body_phase_; a.move = body_move_; a.air = body_air_; a.dir = body_dir_; a.root = body_root_off_;
        // Their camera: where their gun and head are drawn from.
        const float cy = std::cos(s.yaw), sy = std::sin(s.yaw), cp = std::cos(s.pitch), sp = std::sin(s.pitch);
        const glm::vec3 f(sy * cp, sp, -cy * cp), r(cy, 0.0f, sy), u = glm::cross(r, f);
        a.view_inv = glm::mat4(glm::vec4(r, 0.0f), glm::vec4(u, 0.0f), glm::vec4(-f, 0.0f), glm::vec4(eye, 1.0f));
        a.eye = eye;
        a.drawn = body_drawn_;
    }
    player_.position = sv.pos; player_.velocity = sv.vel; player_.yaw = sv.yaw; player_.pitch = sv.pitch;
    player_.crouch_factor = sv.crouch; player_.on_ground = sv.ground;
    player_swimming_ = sv.swim; flashlight_on_ = sv.flash;
    current_frame_view_.eye_pos = sv.eye;
    body_model_ = sv.model; body_root_off_ = sv.root; body_phase_ = sv.phase; body_move_ = sv.move; body_air_ = sv.air;
    body_recoil_ = sv.recoil; body_dir_ = sv.dir; body_drawn_ = sv.drawn;
}

// Weather + clock are one shared state. Whoever changes it (a weather
// button, the clock) tells the host; the host tells everybody, at once and
// again every second. Every machine then runs the same transition from the
// same moment (environment.cpp eases toward the same targets at the same rate).
void VulkanEngine::mp_sync_world(float dt) {
    auto snapshot = [&]() {
        net::WorldState w{};
        w.weather_on = rt_.weather_enabled; w.day_night = rt_.day_night_enabled;
        w.preset = int8_t(rt_.weather_preset); w.manual = rt_.weather_manual;
        w.season = int8_t(rt_.season); w.transition_s = rt_.weather_transition_s;
        w.time_of_day = rt_.time_of_day; w.day_len_min = rt_.day_length_minutes; w.change_min = rt_.weather_change_minutes;
        const float* t = &wx_tgt_.cover; const float* c = &wx_cur_.cover;
        for (int i = 0; i < 9; ++i) { w.tgt[i] = t[i]; w.cur[i] = c[i]; }
        return w;
    };
    auto same_choice = [](const net::WorldState& a, const net::WorldState& b) {
        return a.weather_on == b.weather_on && a.day_night == b.day_night && a.preset == b.preset && a.manual == b.manual &&
               a.season == b.season && a.transition_s == b.transition_s &&
               a.day_len_min == b.day_len_min && a.change_min == b.change_min;
    };
    const bool host = net_->role() == net::Role::Host, client = net_->role() == net::Role::Client;
    if (!client) mp_host_tgt_valid_ = false;
    // ---- incoming ----
    net::WorldState in{};
    if (net_->take_world(in)) {
        const bool choice_changed = !mp_world_init_ || !same_choice(in, mp_world_last_);
        rt_.weather_enabled = in.weather_on != 0; rt_.day_night_enabled = in.day_night != 0;
        if (rt_.weather_preset != in.preset) wx_last_preset_ = -2;      // start that type (and its clock turn) here too
        rt_.weather_preset = in.preset; rt_.weather_manual = in.manual;
        rt_.season = in.season; rt_.weather_transition_s = in.transition_s;
        rt_.day_length_minutes = in.day_len_min; rt_.weather_change_minutes = in.change_min;
        if (client) {
            // The clock: follow the host unless a weather type is turning it right now (both turn it the same way).
            const float d = std::fmod(in.time_of_day - rt_.time_of_day + 36.0f, 24.0f) - 12.0f;
            if (wx_clock_target_ < 0.0f && !choice_changed && std::abs(d) > 0.01f) rt_.time_of_day = in.time_of_day;
            float* t = &mp_host_tgt_.cover; float* c = &wx_cur_.cover;
            float worst = 0.0f;
            for (int i = 0; i < 9; ++i) { t[i] = in.tgt[i]; worst = std::max(worst, std::abs(c[i] - in.cur[i])); }
            mp_host_tgt_valid_ = true;
            if (worst > 0.12f && !choice_changed) for (int i = 0; i < 9; ++i) c[i] = in.cur[i];     // joined late or drifted: take the sky of the host
        } else {
            rt_.time_of_day = in.time_of_day;      // host: a request (the clock set by hand included)
        }
        mp_world_last_ = snapshot();
        mp_world_init_ = true;
        if (host) { net_->send_world(mp_world_last_); mp_world_timer_ = 1.0f; }    // pass the request on now
        return;
    }
    // ---- outgoing ----
    const net::WorldState now_w = snapshot();
    // (the clock was set by hand: more than it can have moved by itself)
    const float moved = std::abs(std::fmod(now_w.time_of_day - mp_world_last_.time_of_day + 36.0f, 24.0f) - 12.0f);
    const bool clock_set = mp_world_init_ && wx_clock_target_ < 0.0f && moved > 0.25f;
    const bool changed = mp_world_init_ && (!same_choice(now_w, mp_world_last_) || clock_set);
    mp_world_timer_ -= dt;
    if (host && (changed || mp_world_timer_ <= 0.0f)) { net_->send_world(now_w); mp_world_timer_ = 1.0f; }
    else if (client && changed) net_->send_world(now_w);
    mp_world_last_ = now_w;
    mp_world_init_ = true;
}

// The local player fired (called from fire_projectile).

void VulkanEngine::mp_on_local_fire(glm::vec3 origin, glm::vec3 dir) {
    if (!net_ || mp_remote_shot_) return;
    net::Event e{};
    e.kind = net::Event::Fire; e.from = net_->id();
    e.a[0] = origin.x; e.a[1] = origin.y; e.a[2] = origin.z;
    e.b[0] = dir.x; e.b[1] = dir.y; e.b[2] = dir.z;
    net_->send_event(e);
}

// A local bolt moved from a to b this tick: did it pass through another
// player? Tested against where that player is DRAWN on this machine - what
// the shooter aimed at - so a hit on screen is a hit.
bool VulkanEngine::mp_bolt_hits(glm::vec3 a, glm::vec3 b, glm::vec3& hit) {
    if (!net_) return false;
    for (auto& av : mp_avatars_) {
        if (!av.id || !av.drawn) continue;
        const glm::vec3 c(av.st.px, av.st.py, av.st.pz);
        const float top = game::Player::kHalfExtents.y + 0.25f - 0.9f * av.st.crouch;     // up to the helmet
        float t = 0.0f;
        if (!seg_aabb(a, b, c - glm::vec3(0.36f, game::Player::kHalfExtents.y, 0.36f), c + glm::vec3(0.36f, top, 0.36f), t)) continue;
        hit = a + (b - a) * t;
        net::Event e{};
        e.kind = net::Event::Hit; e.from = net_->id(); e.to = av.id; e.v = kHitDamage;
        e.a[0] = hit.x; e.a[1] = hit.y; e.a[2] = hit.z;
        net_->send_event(e);
        mp_hitmark_ = 0.25f;
        return true;
    }
    return false;
}

void VulkanEngine::draw_remote_players(VkCommandBuffer cmd, const glm::mat4& vp) {
    if (!net_) return;
    const float sv_recoil = recoil_timer_, sv_muzzle = muzzle_flash_timer_, sv_body_recoil = body_recoil_;
    const int sv_weapon = weapon_; const bool sv_flash = flashlight_on_;
    for (auto& a : mp_avatars_) {
        if (!a.id || !a.drawn) continue;
        recoil_timer_ = a.recoil_t; muzzle_flash_timer_ = a.muzzle_t; weapon_ = 0; flashlight_on_ = false;
        mp_draw_ = &a;
        draw_viewmodel(cmd, vp, a.view_inv);
        mp_draw_ = nullptr;
    }
    recoil_timer_ = sv_recoil; muzzle_flash_timer_ = sv_muzzle; weapon_ = sv_weapon; flashlight_on_ = sv_flash;
    body_recoil_ = sv_body_recoil;
}

void VulkanEngine::mp_hud() {
    if (!net_) return;
    mp_hitmark_ = std::max(0.0f, mp_hitmark_ - ImGui::GetIO().DeltaTime);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    const int others = int(net_->players().size());
    char line[160];
    if (others > 0) {
        if (net_->role() == net::Role::Client)
            std::snprintf(line, sizeof(line), "LAN: %s  |  %d players  |  ping %u ms  |  kills %d", net_->role_name(), others + 1, net_->ping_ms(), mp_kills_);
        else
            std::snprintf(line, sizeof(line), "LAN: %s  |  %d players  |  kills %d", net_->role_name(), others + 1, mp_kills_);
    } else {
        std::snprintf(line, sizeof(line), "LAN: %s for other players...", net_->role_name());
    }
    const ImVec2 ts = ImGui::CalcTextSize(line);
    const ImVec2 p0(disp.x - ts.x - 26.0f, 22.0f);
    dl->AddRectFilled(ImVec2(p0.x - 8.0f, p0.y - 4.0f), ImVec2(p0.x + ts.x + 8.0f, p0.y + ts.y + 4.0f), IM_COL32(0, 0, 0, 120), 4.0f);
    dl->AddText(p0, others > 0 ? IM_COL32(150, 255, 170, 255) : IM_COL32(220, 220, 220, 200), line);
    float y = p0.y + ts.y + 12.0f;
    for (const auto& f : mp_feed_) {
        const ImVec2 fs = ImGui::CalcTextSize(f.text.c_str());
        dl->AddText(ImVec2(disp.x - fs.x - 18.0f, y), IM_COL32(255, 255, 255, int(255.0f * glm::clamp(f.ttl, 0.0f, 1.0f))), f.text.c_str());
        y += fs.y + 3.0f;
    }
    if (others > 0) {
        if (mp_hurt_ > 0.0f)
            dl->AddRect(ImVec2(0, 0), disp, IM_COL32(255, 30, 20, int(170.0f * mp_hurt_)), 0.0f, 0, 28.0f);
        if (mp_hitmark_ > 0.0f) {
            const ImVec2 c(disp.x * 0.5f, disp.y * 0.5f);
            const ImU32 col = IM_COL32(255, 80, 60, 255);
            for (float sx : { -1.0f, 1.0f }) for (float sy : { -1.0f, 1.0f })
                dl->AddLine(ImVec2(c.x + sx * 6.0f, c.y + sy * 6.0f), ImVec2(c.x + sx * 13.0f, c.y + sy * 13.0f), col, 2.0f);
        }
        // Name tags over the other players.
        const glm::mat4 vpm = current_frame_view_.vp;
        for (const auto& a : mp_avatars_) {
            if (!a.id || !a.drawn) continue;
            const glm::vec4 c = vpm * glm::vec4(a.eye + glm::vec3(0.0f, 0.38f, 0.0f), 1.0f);
            if (c.w < 0.3f) continue;
            const ImVec2 sp((c.x / c.w * 0.5f + 0.5f) * disp.x, (c.y / c.w * 0.5f + 0.5f) * disp.y);
            char tag[64];
            std::snprintf(tag, sizeof(tag), "%s  %d", net_->name_of(a.id).c_str(), int(a.st.health));
            const ImVec2 tsz = ImGui::CalcTextSize(tag);
            dl->AddText(ImVec2(sp.x - tsz.x * 0.5f + 1.0f, sp.y - tsz.y + 1.0f), IM_COL32(0, 0, 0, 200), tag);
            dl->AddText(ImVec2(sp.x - tsz.x * 0.5f, sp.y - tsz.y), IM_COL32(255, 235, 150, 255), tag);
        }
    }
}

}  // namespace qlike
