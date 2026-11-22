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
#include "engine/net/net_anim.h"
#include "anim/anim_lod.h"
#include "engine/audio.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace vibe {

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
    // Off for automated runs (--frames / screenshots) unless VIBE_NET=1, and with VIBE_NO_NET.
    const bool automated = opts_.max_frames > 0 || !opts_.screenshot_path.empty();
    if (std::getenv("VIBE_NO_NET") || (automated && !std::getenv("VIBE_NET"))) return;
    const char* cn = std::getenv("COMPUTERNAME");
    net_ = new net::Session();
    if (!net_->start(cn && *cn ? cn : "player")) { delete net_; net_ = nullptr; return; }
    mp_spawn_ = player_.position;
    mp_health_ = 100.0f;
}

void VulkanEngine::mp_shutdown() {
    if (net_) { net_->stop(); delete net_; net_ = nullptr; }
    for (auto& a : mp_avatars_) {
        a.animator.reset();
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
    // A respawn is told as such (one state long), so nobody slides the body across the map.
    // (held for a moment: states go out 30 times a second, this runs every frame)
    if (glm::length(player_.position - mp_last_sent_pos_) > net::kTeleportDistance + 0.1f * glm::length(player_.velocity)) mp_teleport_hold_ = 0.12f;
    mp_last_sent_pos_ = player_.position;
    const bool jumped = mp_teleport_hold_ > 0.0f;
    mp_teleport_hold_ = std::max(0.0f, mp_teleport_hold_ - dt);
    ls.flags = uint8_t((player_.on_ground ? net::kFlagGround : 0) | (player_swimming_ ? net::kFlagSwim : 0) | (flashlight_on_ ? net::kFlagTorch : 0) |
                       (body_reload_left_ > 0.0f ? net::kFlagReload : 0) | (body_aim_ ? net::kFlagAim : 0) | (jumped ? net::kFlagTeleport : 0));
    ls.health = uint8_t(glm::clamp(mp_health_, 0.0f, 100.0f));
    ls.weapon = uint8_t(weapon_);
    // The movement keys: what this player wants, so others can predict starts, stops and turns as the controller will make them.
    ls.move = net::move_bits(last_pin_);
    ls.shot_seq = uint16_t(body_shot_seq_);
    net_->update(glfwGetTime(), ls);

    for (auto& f : mp_feed_) f.ttl -= dt;
    mp_feed_.erase(std::remove_if(mp_feed_.begin(), mp_feed_.end(), [](const MpFeed& f) { return f.ttl <= 0.0f; }), mp_feed_.end());
    mp_hurt_ = std::max(0.0f, mp_hurt_ - dt * 2.5f);

    // VIBE_TEST_WX=<seconds>:<type>: pick that weather type after that many seconds (sync tests).
    if (static const char* tw = std::getenv("VIBE_TEST_WX"); tw) {
        static bool done = false;
        float at = 0.0f; int ty = 0, se = -1;
        if (!done && std::sscanf(tw, "%f:%d:%d", &at, &ty, &se) >= 2 && glfwGetTime() > double(at)) {
            done = true; rt_.weather_enabled = true; rt_.weather_preset = ty;
            if (se >= 0) rt_.season = se;            // VIBE_TEST_WX=<seconds>:<type>:<season>
        }
    }
    mp_sync_world(dt);

    // ---- events ----
    for (const net::Event& e : net_->take_events()) {
        if (e.kind == net::Event::Fire) {
            const glm::vec3 o(e.a[0], e.a[1], e.a[2]), d(e.b[0], e.b[1], e.b[2]);
            mp_remote_shot_ = true;
            for (const auto& a : mp_avatars_) if (a.id == e.from && a.animator) shot_weapon_offset_ = &a.fire_offset;
            fire_projectile(o, d);
            shot_weapon_offset_ = nullptr;
            mp_remote_shot_ = false;
            if (audio_) audio_->play_at(ClipID::Laser, o, 0.8f);
            for (auto& a : mp_avatars_) if (a.id == e.from) { a.recoil_t = kRecoilDuration; a.muzzle_t = kMuzzleFlashDuration; a.combat_t = 0.0f; }
        } else if (e.kind == net::Event::Hit) {
            // (where and from where they were hit: the push their body gets if this kills them)
            for (auto& a : mp_avatars_)
                if (a.id == e.to) {
                    a.hit_pos = glm::vec3(e.a[0], e.a[1], e.a[2]); a.hit_dir = glm::vec3(e.b[0], e.b[1], e.b[2]);
                    a.hit_power = e.v / kHitDamage; a.hit_age = 0.0f;
                    ++a.hit_seq;
                }
            if (e.to == net_->id() && mp_health_ > 0.0f) {
                mp_health_ -= e.v;
                ++body_hit_seq_;
                body_hit_dir_ = glm::vec3(e.b[0], e.b[1], e.b[2]);
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
        } else if (e.kind == net::Event::Airstrike) {
            spawn_airstrike(glm::vec3(e.a[0], e.a[1], e.a[2]), glm::normalize(glm::vec3(e.b[0], e.b[1], e.b[2]) + glm::vec3(1e-5f, 0.0f, 0.0f)), true);
            mp_feed(net_->name_of(e.from) + " called an airstrike");
        } else if (e.kind == net::Event::Death) {
            for (auto& a : mp_avatars_) if (a.id == e.from) a.die = true;      // their body falls where it is (ragdoll.cpp)
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
            *av = MpAvatar{};
            av->id = p.id;
            av->fresh_pose = true;
            mp_feed(net_->name_of(p.id) + " joined");
        }
        av->st = p.state;
        av->teleported = av->teleported || p.teleported;
    }

}

// The other players: each has an animator of its own, told what the network says about them
// (interpolated ~100 ms in the past). Their pose is reconstructed here from position, velocity,
// view, movement keys, stance, what they hold and their shot counter - no pose data is received.
// Nothing of the local player's state is involved.
void VulkanEngine::mp_pose_avatars(float dt) {
    if (!net_ || body_bones_.empty() || !anim_set_.ok()) return;
    CPU_SCOPE("remote bodies");
    dt = glm::clamp(dt, 0.0f, 0.1f);
    const glm::vec3 cam = current_frame_view_.eye_pos;
    const float tan_half = current_frame_view_.proj[1][1] != 0.0f ? 1.0f / std::abs(current_frame_view_.proj[1][1]) : 0.7f;
    struct Posed { MpAvatar* a; anim::CharacterAnimationInput in; net::RemoteFrame rf; bool skip; };
    Posed posed[net::kMaxPlayers];
    int n_posed = 0;
    for (auto& a : mp_avatars_) {
        a.drawn = false;
        a.present = false;
        if (!a.id) continue;
        a.present = true;
        a.recoil_t = std::max(0.0f, a.recoil_t - dt);
        a.muzzle_t = std::max(0.0f, a.muzzle_t - dt);
        a.combat_t += dt;
        a.hit_age += dt;
        a.palette_prev = a.fresh_pose ? -1 : a.palette;       // (a pose from nowhere has no motion)
        a.palette = -1;
        if (!a.animator) { a.animator = std::make_unique<anim::CharacterAnimator>(anim_set_); apply_anim_tuning(a.animator->tuning); a.fresh_pose = true; }
        // Killed: for a moment the body still stands where it was hit and buckles (they themselves
        // are at their spawn already); then it is the physics'. Until then it is posed from the
        // state it had, slowing down.
        static const float kDying = std::getenv("VIBE_DYING") ? float(std::atof(std::getenv("VIBE_DYING"))) : 0.38f;
        if (a.die && a.dying_t < 0.0f && kDying > 0.0f && !a.fresh_pose) { a.die = false; a.dying_t = 0.0f; a.dying_st = a.st; }
        if (a.dying_t >= 0.0f) {
            a.dying_t += dt;
            net::PlayerState& d = a.dying_st;
            const float k = std::exp(-dt * 5.0f);
            d.vx *= k; d.vz *= k;
            if (d.flags & net::kFlagGround) d.vy = 0.0f; else d.vy -= game_.gravity * dt;
            d.px += d.vx * dt; d.py += d.vy * dt; d.pz += d.vz * dt;
            d.flags = uint8_t(d.flags & ~(net::kFlagAim | net::kFlagReload | net::kFlagTeleport));
            if (a.dying_t >= kDying) { a.die = true; }                  // hand over below, with this pose
        }
        const bool dying = a.dying_t >= 0.0f;
        const net::PlayerState& s = dying ? a.dying_st : a.st;
        anim::CharacterAnimationInput in;
        const net::RemoteFrame rf = net::remote_animation_input(s, in);
        if (dying) in.dying = glm::clamp(a.dying_t / std::max(kDying, 1e-3f), 0.0f, 1.0f);
        const glm::vec3 eye = rf.eye;
        const glm::mat4 model = rf.model;

        // Level of detail: by size on screen, whether in view, and whether we are fighting them.
        anim::LodInput li;
        li.distance = glm::length(eye - cam);
        li.screen_height = float(swapchain_extent_.height);
        li.tan_half_fov_y = tan_half;
        {
            const glm::vec4 c = current_frame_view_.vp * glm::vec4(eye - glm::vec3(0.0f, 0.7f, 0.0f), 1.0f);
            const float m = 1.25f * c.w + 1.5f;                     // generous: limbs, and what turns into view next frame
            li.in_view = c.w > -1.5f && std::abs(c.x) < m && std::abs(c.y) < m;
        }
        li.combat = a.combat_t < 4.0f;
        li.previous = a.lod;
        static const int kForceLod = std::getenv("VIBE_ANIM_LOD") ? std::atoi(std::getenv("VIBE_ANIM_LOD")) : -1;
        a.lod = kForceLod >= 0 ? glm::clamp(kForceLod, 0, 3) : anim::select_lod(li);

        in.dt = dt;
        const bool reloading = in.reloading;
        if (reloading && !a.was_reloading) ++a.reload_seq;          // a reload began (the flag is state: it survives lost packets)
        a.was_reloading = reloading;
        in.reload_seq = a.reload_seq;
        in.variation = a.id;
        {   // somebody worth a look: the nearest other soldier, us included
            glm::vec3 best(0.0f);
            float best_d2 = 30.0f * 30.0f;
            auto consider = [&](glm::vec3 p) { const glm::vec3 d = p - eye; const float d2 = glm::dot(d, d); if (d2 > 0.25f && d2 < best_d2) { best_d2 = d2; best = p; in.has_look_at = true; } };
            consider(cam);
            for (const auto& o : mp_avatars_) if (o.id && o.id != a.id && o.present) consider(o.eye);
            if (in.has_look_at) in.look_at = glm::vec3(glm::inverse(model) * glm::vec4(best, 1.0f));
        }
        in.hit_seq = a.hit_seq;
        in.hit_power = a.hit_power;
        {   // (world -> their model space: +x their right, -z the way they face)
            const float sy = std::sin(s.yaw), cy = std::cos(s.yaw);
            in.hit_dir = glm::vec3(a.hit_dir.x * cy + a.hit_dir.z * sy, a.hit_dir.y, a.hit_dir.x * sy - a.hit_dir.z * cy);
            in.hit_dir.z = -in.hit_dir.z;
        }
        in.teleported = (a.teleported || a.fresh_pose) && !dying;        // (the respawn is not this body's)
        a.teleported = false;
        in.lod = a.lod;

        // Level 2 is posed every other frame; in between, the last pose is sent again.
        // (Level 3 is not posed at all; its animator only keeps time.)
        const bool skip = a.lod == 2 && !a.fresh_pose && !a.die && a.skipped == 0 && (frame_number_ & 1u) && !a.pose_rows.empty();
        if (skip) {
            a.skipped = 1;
            a.skip_dt = dt;
        } else {
            in.dt = dt + a.skip_dt;
            a.skipped = 0;
            a.skip_dt = 0.0f;
            if (a.lod >= 3) a.pose_rows.clear();
        }
        Posed& r = posed[n_posed++];
        r.a = &a;
        r.in = in;
        r.rf = rf;
        r.skip = skip;
    }
    // Every animator that has a pose to make this frame, at once (job threads), ...
    {
        PoseJob jobs[net::kMaxPlayers];
        int nj = 0;
        for (int i = 0; i < n_posed; ++i) if (!posed[i].skip) jobs[nj++] = { posed[i].a->animator.get(), &posed[i].in };
        pose_update_all(jobs, nj);
    }
    // ... then each taken over in turn: palette, where what they carry is, a death.
    for (int pi = 0; pi < n_posed; ++pi) {
        Posed& r = posed[pi];
        MpAvatar& a = *r.a;
        anim::CharacterAnimationInput& in = r.in;
        const net::RemoteFrame& rf = r.rf;
        const net::PlayerState& s = a.dying_t >= 0.0f ? a.dying_st : a.st;
        const glm::vec3 eye = rf.eye;
        const glm::mat4 model = rf.model;
        if (r.skip) {
            a.palette = body_push_rows(a.pose_rows);
        } else {
            pose_finish(*a.animator, in, a.lod >= 3 ? nullptr : &a.palette, &a.pose_rows);
            a.weapon_offset = a.animator->weapon_offset();
            a.fire_offset = a.animator->weapon_fire_offset();
        }
        a.prev_model = a.fresh_pose ? model : a.model;
        a.fresh_pose = false;
        a.model = model;
        if (a.die) {
            const bool was_dying = a.dying_t >= 0.0f;
            a.dying_t = -1.0f;
            if (was_dying) { a.fresh_pose = true; a.teleported = true; }      // (the living player is somewhere else by now)
            // Killed: this pose, with the speed they had, goes to the physics; the push comes
            // from the shot that did it (if we saw it), at the part of the body it hit.
            a.die = false;
            if (a.lod >= 3) { in.lod = 2; in.dt = 0.0f; pose_character(*a.animator, in, nullptr); }       // (unseen: a pose is needed now)
            const bool seen = a.hit_age < 1.0f && glm::dot(a.hit_dir, a.hit_dir) > 0.5f;
            corpse_spawn(model, glm::mat4(glm::vec4(std::cos(s.yaw), 0.0f, std::sin(s.yaw), 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                                          glm::vec4(-std::sin(s.yaw), 0.0f, std::cos(s.yaw), 0.0f), glm::vec4(eye, 1.0f)),
                         in.velocity, seen ? a.hit_pos : eye - glm::vec3(0.0f, 0.4f, 0.0f),
                         seen ? a.hit_dir : glm::vec3(0.0f), seen ? std::max(a.hit_power, 0.5f) : 0.0f);
        }
        a.view_inv = rf.view_inv;          // their camera: where their gun and head are drawn from
        a.eye = eye;
        a.mesh_lod = body_mesh_lod(eye, a.mesh_lod);
        a.drawn = a.lod < 3 && a.palette >= 0;
    }
    if (std::getenv("VIBE_ANIM_LOG") && frame_number_ % 20 == 0) {
        // (what each remote animator is doing: the evidence for a two-instance run)
        for (const auto& a : mp_avatars_) {
            if (!a.id || !a.animator) continue;
            const anim::AnimDebug& d = a.animator->debug();
            log::infof("[anim] remote %u: L%d %s clip %s  speed %.2f rate %.2f  air %s  contact %d%d  pos %.1f %.1f %.1f", a.id, a.lod, d.mode,
                       d.clip >= 0 ? anim_set_.runtime->clip(d.clip).name.c_str() : "-", d.speed, d.rate, anim::air_phase_name(d.air_phase),
                       int(d.contact[0]), int(d.contact[1]), a.st.px, a.st.py, a.st.pz);
        }
    }
    if (std::getenv("VIBE_ANIM_LOG") && frame_number_ % 120 == 0) {
        const net::Session::Stats& ns = net_->stats();
        log::infof("[anim] bodies: animation %.3f ms, skinning %.3f ms this frame; levels 0/1/2/3 = %d/%d/%d/%d; net tx %llu B rx %llu B, dup events %u",
                   anim_ms_, skin_ms_, anim_lod_count_[0], anim_lod_count_[1], anim_lod_count_[2], anim_lod_count_[3],
                   static_cast<unsigned long long>(ns.tx_bytes), static_cast<unsigned long long>(ns.rx_bytes), ns.duplicate_events);
    }
}

void VulkanEngine::draw_remote_players(VkCommandBuffer cmd, const glm::mat4& vp) {
    if (!net_) return;
    const float sv_recoil = recoil_timer_, sv_muzzle = muzzle_flash_timer_;
    const int sv_weapon = weapon_; const bool sv_flash = flashlight_on_;
    for (auto& a : mp_avatars_) {
        if (!a.id || !a.drawn) continue;
        // What they hold, as replicated: the laser gun or the tablet (a torch is not drawn for others).
        recoil_timer_ = a.recoil_t; muzzle_flash_timer_ = a.muzzle_t; weapon_ = a.st.weapon == 1 ? 1 : 0; flashlight_on_ = false;
        mp_draw_ = &a;
        draw_viewmodel(cmd, vp, a.view_inv);
        mp_draw_ = nullptr;
    }
    recoil_timer_ = sv_recoil; muzzle_flash_timer_ = sv_muzzle; weapon_ = sv_weapon; flashlight_on_ = sv_flash;
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
        // (whether they are drawn has nothing to do with it: animation detail never decides a hit)
        if (!av.id || !av.present) continue;
        const glm::vec3 c(av.st.px, av.st.py, av.st.pz);
        const float top = game::Player::kHalfExtents.y + 0.25f - 0.9f * av.st.crouch;     // up to the helmet
        float t = 0.0f;
        if (!seg_aabb(a, b, c - glm::vec3(0.36f, game::Player::kHalfExtents.y, 0.36f), c + glm::vec3(0.36f, top, 0.36f), t)) continue;
        hit = a + (b - a) * t;
        net::Event e{};
        e.kind = net::Event::Hit; e.from = net_->id(); e.to = av.id; e.v = kHitDamage;
        e.a[0] = hit.x; e.a[1] = hit.y; e.a[2] = hit.z;
        const glm::vec3 bd = glm::normalize(b - a + glm::vec3(1e-6f, 0.0f, 0.0f));
        e.b[0] = bd.x; e.b[1] = bd.y; e.b[2] = bd.z;
        av.hit_pos = hit; av.hit_dir = bd; av.hit_power = 1.0f; av.hit_age = 0.0f; av.combat_t = 0.0f;
        ++av.hit_seq;
        net_->send_event(e);
        mp_hitmark_ = 0.25f;
        return true;
    }
    return false;
}

void VulkanEngine::mp_hud() {
    // The hit marker: four ticks round the crosshair for a quarter of a second after a shot
    // of yours landed - on another player, an aircraft or a vehicle (also without a network).
    if (mp_hitmark_ > 0.0f && imgui_initialized_) {
        const ImVec2 dsz = ImGui::GetIO().DisplaySize;
        const ImVec2 c(dsz.x * 0.5f, dsz.y * 0.5f);
        const float k = dsz.y / 1080.0f, a = glm::clamp(mp_hitmark_ / 0.12f, 0.0f, 1.0f);
        const ImU32 col = IM_COL32(255, 80, 60, int(255.0f * a));
        for (float sx : { -1.0f, 1.0f }) for (float sy : { -1.0f, 1.0f })
            ImGui::GetForegroundDrawList()->AddLine(ImVec2(c.x + sx * 6.0f * k, c.y + sy * 6.0f * k),
                                                    ImVec2(c.x + sx * 13.0f * k, c.y + sy * 13.0f * k), col, 2.0f * k);
    }
    mp_hitmark_ = std::max(0.0f, mp_hitmark_ - ImGui::GetIO().DeltaTime);
    if (!net_) return;
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

}  // namespace vibe
