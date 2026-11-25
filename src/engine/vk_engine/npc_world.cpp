// The people and cars that live at the service station and on the coast highway.
//
// What they do is the NPC engine's business (src/npc: no drawing, no physics, no network in
// it). This file gives it its site and its road, steps it, and shows what it says:
//   * people as the soldiers other players are - the same animator, skinned mesh and draw
//     (multiplayer.cpp mp_pose_avatars is the pattern) - walking, standing, or sitting with the
//     animator's seat pose. Only the nearest are posed and drawn: the frame has a bone palette
//     for 48 bodies in all (kSkinCharacters), shared with the players and the fallen.
//   * cars as cars without physics (jeep.cpp npc_cars_place: a body matrix and rolling wheels,
//     as another machine's car is shown).
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/multiplayer.h"
#include "engine/highway.h"
#include "engine/net/net_anim.h"
#include "engine/terrain.h"
#include "engine/log.h"
#include "anim/anim_lod.h"
#include "npc/npc.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>

namespace vibe {

namespace {
constexpr int   kNpcBodies = 28;           // posed and drawn at most (of the 48 a frame can skin)
constexpr float kNpcBodyReach = 150.0f;    // metres from the eye
constexpr float kNpcCarBodyReach = 90.0f; // ... for those who sit in a car
// (a segment against a box: multiplayer.cpp's)
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
}

// HOW SOMEBODY WHO IS SHOT GOES DOWN. Not a doll thrown by the shot: the body first REACTS - a
// clip made for where the shot struck and what the person was doing (text to motion,
// docs/animation/actions.md "Death") - and at the moment that reaction loses its balance the
// physics takes the body, each part moving as the clip moved it (corpse_spawn's part_velocity),
// and lets it fall on whatever is there. `hand`: seconds into the clip at which the hips have
// begun to drop (from the take's own numbers; the clips are cut to begin at the reaction).
namespace {
struct ShotClip { const char* name; float hand; };
const ShotClip kShotFront = { "shot_front", 1.63f };      // staggers back, collapses on the back
const ShotClip kShotGut   = { "shot_gut",   1.07f };      // doubles over, to the knees, forward
const ShotClip kShotSpin  = { "shot_spin",  1.00f };      // spun round by a shoulder hit
const ShotClip kShotBack  = { "shot_back",  0.73f };      // thrown forward on the face
const ShotClip kShotHead  = { "shot_head",  0.63f };      // limp at once, straight down
const ShotClip kShotLeg   = { "shot_leg",   0.77f };      // the leg gives way, down sideways
const ShotClip kShotRun   = { "shot_run",   0.43f };      // trips at a run, tumbles
const ShotClip kShotWalk  = { "shot_walk",  1.30f };      // stops, staggers, collapses
}

// Which reaction: by where the shot struck (head, legs, trunk), from which side (the bolt's
// direction against the way the person faces) and how fast the person was moving. The same on
// every machine: all of it comes from the hit that was sent and the person's own state.
static const ShotClip& pick_shot_clip(float height, bool from_front, float speed, uint32_t id) {
    if (height > 1.50f) return kShotHead;
    if (speed > 2.4f) return kShotRun;
    if (height < 0.75f) return kShotLeg;
    if (!from_front) return kShotBack;
    if (speed > 0.5f) return kShotWalk;
    switch (id % 3u) { case 0: return kShotFront; case 1: return kShotGut; default: return kShotSpin; }
}

void VulkanEngine::npc_dying_pose(float dt) {
    if (npc_dying_.empty() || body_bones_.empty() || !anim_set_.ok()) { npc_dying_.clear(); return; }
    dt = glm::clamp(dt, 1e-4f, 0.1f);
    const anim::IAnimationRuntime& rtm = *anim_set_.runtime;
    for (size_t i = 0; i < npc_dying_.size();) {
        NpcDying& d = npc_dying_[i];
        if (d.fresh) { d.fresh = false; ++i; continue; }      // (posed this frame already, as it stood)
        d.t += dt;
        anim::CharacterAnimationInput in = d.in;
        in.dt = dt;
        in.action_clip = d.clip;
        in.action_time = std::min(d.t, rtm.clip(d.clip).duration);
        in.action_in_place = false; in.action_warp = false; in.action_warp_full = false;
        in.action_warp_from_start = false; in.action_end_at_origin = false;
        in.velocity = in.desired_velocity = glm::vec3(0.0f);
        in.has_look_at = false; in.teleported = false; in.seated = false;
        in.lod = std::min(d.av.lod, 1);
        d.av.palette_prev = d.av.palette;
        d.av.palette = -1;
        pose_character(*d.av.animator, in, &d.av.palette, &d.av.pose_rows);
        glm::vec3 now[kCorpseParts];
        const bool have = corpse_part_centres(d.model, now);
        if (d.t >= d.hand || !have) {
            // The physics' from here: every part goes on as the clip moved it (no faster than a
            // falling body moves), and the shot itself adds nothing - the reaction was the clip.
            glm::vec3 vel[kCorpseParts];
            for (int k = 0; k < kCorpseParts; ++k) {
                vel[k] = d.have_prev && have ? (now[k] - d.parts_prev[k]) / dt : glm::vec3(0.0f);
                const float sp = glm::length(vel[k]);
                if (sp > 7.0f) vel[k] *= 7.0f / sp;
            }
            glm::mat4 vi = d.view_inv;
            if (have) vi[3] = glm::vec4(now[2], 1.0f);        // (the head, where it is now)
            corpse_spawn(d.model, vi, glm::vec3(0.0f), d.hit_pos, d.hit_dir, 0.0f, vel);
            npc_dying_.erase(npc_dying_.begin() + ptrdiff_t(i));
            continue;
        }
        for (int k = 0; k < kCorpseParts; ++k) d.parts_prev[k] = now[k];
        d.have_prev = true;
        d.av.prev_model = d.av.model = d.model;
        d.av.eye = now[2];
        d.av.view_inv = d.view_inv; d.av.view_inv[3] = glm::vec4(now[2], 1.0f);
        d.av.mesh_lod = body_mesh_lod(d.av.eye, d.av.mesh_lod);
        d.av.drawn = d.av.palette >= 0;
        ++i;
    }
}

void VulkanEngine::npc_tick(float dt) {
    // Made once the road and the station are there (the road is built on another thread).
    if (!npc_) {
        if (!highway::ready() || station_pos_ == glm::vec3(0.0f) || std::getenv("VIBE_NO_NPC")) return;
        npc_ = std::make_unique<npc::World>();
        if (!npc_->load_site("assets/gasstation/gasstation.nav", station_pos_))
            log::warnf("[npc] assets/gasstation/gasstation.nav missing - no people at the station (blender -b --python tools/gasstation/make_gasstation.py)");
        npc::Road road;
        road.length = highway::course_length();
        // (the course is points 20 m apart: the direction is taken over a car length, so that
        //  a car turns through a bend instead of at its corners)
        road.at = [](float s, glm::vec3& pos, glm::vec2& dir) {
            glm::vec3 a, b; glm::vec2 d;
            if (!highway::point_along(s, pos, dir) || !highway::point_along(s - 5.0f, a, d) || !highway::point_along(s + 5.0f, b, d)) return false;
            const glm::vec2 ab(b.x - a.x, b.z - a.z);
            if (glm::dot(ab, ab) > 1e-4f) dir = glm::normalize(ab);
            pos.y -= highway::cross_fall(1.9f);
            return true;
        };
        npc_->set_road(std::move(road));
        // The soldiers' run: from the launch pad along the coast road (on its edge, the side the
        // castle lane leaves it) and up the lane to the castle's car park - and back.
        if (highway::line_count() > 1 && !std::getenv("VIBE_NO_RUNNERS")) {
            const std::vector<glm::vec3>& C = highway::line_points(0);
            const std::vector<glm::vec3>& L = highway::line_points(1);
            if (C.size() > 8 && L.size() > 2) {
                const int nC = int(C.size());
                auto nearest_i = [&](glm::vec2 q) { int b = 0; float bd = 1e18f; for (int i = 0; i < nC; ++i) { const float d = glm::distance(q, glm::vec2(C[size_t(i)].x, C[size_t(i)].z)); if (d < bd) { bd = d; b = i; } } return b; };
                const int ia = nearest_i(glm::vec2(1600.0f, 830.0f)), ib = nearest_i(glm::vec2(L.front().x, L.front().z));
                const int fwd = (ib - ia + nC) % nC, step = fwd <= nC - fwd ? 1 : -1, count = std::min(fwd, nC - fwd);
                const float edge = std::max(0.5f * highway::line_width(0) - 1.5f, 2.6f);
                auto beside = [&](int i, float sgn) {
                    const glm::vec3 a = C[size_t((i - step + nC) % nC)], b = C[size_t((i + step + nC) % nC)];
                    const glm::vec2 d = glm::normalize(glm::vec2(b.x - a.x, b.z - a.z) + glm::vec2(1e-6f, 0.0f));
                    glm::vec3 p = C[size_t(i)] + glm::vec3(-d.y, 0.0f, d.x) * (edge * sgn);
                    p.y -= highway::cross_fall(edge);
                    return p;
                };
                const float sgn = glm::distance(beside(ib, 1.0f), L.front()) < glm::distance(beside(ib, -1.0f), L.front()) ? 1.0f : -1.0f;
                std::vector<glm::vec3> run;
                for (int k = 0; k <= count; ++k) run.push_back(beside((ia + step * k + nC * 4) % nC, sgn));
                for (const glm::vec3& p : L) run.push_back(p);
                float len = 0.0f; for (size_t i = 1; i < run.size(); ++i) len += glm::distance(run[i - 1], run[i]);
                log::infof("[npc] the soldiers' run: %zu points, %.0f m, from (%.0f, %.0f) to (%.0f, %.0f)", run.size(), double(len), run.front().x, run.front().z, run.back().x, run.back().z);
                npc_->set_run_path(std::move(run));
            }
        }
        npc_->populate(20261124u);
        // VIBE_NPC_WARM=<seconds>: so much of their life has passed already when the game starts
        // (tests: to be there at a chosen moment). With VIBE_LOG_NPC the moments a car turns in
        // to the station are logged on the way.
        if (const char* warm = std::getenv("VIBE_NPC_WARM")) {
            int at_site = npc_->census().cars_at_site;
            const int steps = int(std::atof(warm) / npc::kStep);
            for (int i = 0; i < steps; ++i) {
                npc_->update(npc::kStep);
                const int now = npc_->census().cars_at_site;
                if (now > at_site && std::getenv("VIBE_LOG_NPC")) log::infof("[npc] warm: a car turns in at %.1f s", double(i) * double(npc::kStep));
                at_site = now;
                if (std::getenv("VIBE_LOG_NPC"))                 // (and the moments somebody begins to sit down, get up, get in or out)
                    for (const npc::View& v : npc_->state())
                        if (v.kind == npc::Kind::Person && v.act != npc::Act::None && v.act_t < npc::kStep * 1.5f)
                            log::infof("[npc] warm: person %d begins act %d at %.1f s, at (%.1f, %.1f, %.1f) yaw %.2f", int(v.id), int(v.act), double(i) * double(npc::kStep), v.pos.x, v.pos.y, v.pos.z, v.yaw);
            }
        }
        npc_bodies_ = std::vector<MpAvatar>(size_t(kNpcBodies));
        const npc::World::Census c = npc_->census();
        log::infof("[npc] %d agents: %d people (%d in cars), %d cars; road %.0f m", npc_->agent_count(), c.people, c.hidden, c.cars, double(highway::course_length()));
    }
    // Not in another part of the planet (the site and the road are home's).
    if (!globe::current().home) { npc_views_.clear(); npc_cars_place(npc_views_, dt); for (MpAvatar& a : npc_bodies_) a.drawn = false; return; }
    // Multiplayer: the host's world is the world. A client takes in what the host sends and
    // only follows it (npc::World::apply makes it a replica); the host - and a machine that
    // plays alone - steps its own and tells the clients four times a second. A client that
    // has to take over (the host left) starts its world anew: a replica has only the picture.
    const net::Role role = net_ ? net_->role() : net::Role::Off;
    if (role == net::Role::Client) {
        std::vector<uint8_t> m;
        if (net_->take_npc(m) && !npc_->apply(m.data(), m.size())) log::warnf("[npc] a damaged state from the host (%zu bytes)", m.size());
    } else if (npc_->replica()) {
        npc_->populate(20261124u);
        log::infof("[npc] the host is gone: this machine runs the world now");
    } else if (role == net::Role::Host) {
        npc_send_ -= dt;
        if (npc_send_ <= 0.0f) {
            npc_send_ = 0.25f;
            std::vector<uint8_t> m;
            npc_->serialize(m);
            net_->send_npc(m.data(), m.size());
        }
    }
    // (VIBE_NPC_FAST=<n>: n times as fast - to watch an hour of their life in minutes)
    static const float kFast = std::getenv("VIBE_NPC_FAST") ? float(std::atof(std::getenv("VIBE_NPC_FAST"))) : 1.0f;
    // The players on foot, for the machine that steps the world: its cars do not drive into
    // them (npc::World::set_walkers) - which is also how a car on the road is made to stop, to
    // be taken. (VIBE_NPC_NO_WALKERS=1: the cars see nobody, as before.)
    if (!npc_->replica()) {
        static const bool kNoWalkers = std::getenv("VIBE_NPC_NO_WALKERS") != nullptr;
        std::vector<glm::vec3> feet;
        const float half = game::Player::kHalfExtents.y;
        if (!kNoWalkers) {
            if (!jeep_driving() && jeep_ride_ < 0 && !fly_mode_) feet.push_back(player_.position - glm::vec3(0.0f, half, 0.0f));
            if (mp_) for (const MpAvatar& a : mp_->avatars)
                if (a.id && a.present && !a.seated && a.ride == 0 && !a.corpse) feet.push_back(glm::vec3(a.st.px, a.st.py - half, a.st.pz));
        }
        npc_->set_walkers(feet);
    }
    for (float left = dt * kFast; left > 1e-5f; left -= 0.1f) npc_->update(std::min(left, 0.1f));
    npc_->views(npc_views_);
    for (float& t : npc_nodriver_) t = std::max(0.0f, t - dt);
    // (shot or taken a moment ago, and this copy of the host's world has not heard yet)
    for (size_t i = 0; i < npc_views_.size() && i < 100; ++i) {
        if (npc_gone_[i] <= 0.0f) continue;
        npc_gone_[i] -= dt;
        npc::View& v = npc_views_[i];
        v.hidden = true; v.car = 255; v.pos.y -= 20000.0f;
    }
    npc_cars_place(npc_views_, dt);
    npc_pose(dt);
    npc_dying_pose(dt);
    blood_update(dt);
    // VIBE_TEST_NPC_KILL=<frame>[,<skip>]: at that frame the six people nearest the eye that have a
    // body are shot from where the eye is (after the <skip> nearest) - tests of how they fall.
    if (static const char* tk = std::getenv("VIBE_TEST_NPC_KILL"); tk) {
        int at = 0, skip = 0; std::sscanf(tk, "%d,%d", &at, &skip);
        if (int(frame_number_) == at) {
            std::vector<std::pair<float, int>> who;
            for (const MpAvatar& a : npc_bodies_) if (a.present && a.id >= 1 && a.id <= 100 && !npc_views_[a.id - 1].hidden) who.push_back({ glm::distance(a.eye, current_frame_view_.eye_pos), int(a.id) - 1 });
            std::sort(who.begin(), who.end());
            for (size_t i = size_t(skip); i < who.size() && i < size_t(skip) + 6; ++i) {
                const npc::View& v = npc_views_[size_t(who[i].second)];
                const glm::vec3 chest = v.pos + glm::vec3(0.0f, v.pose == npc::Pose::Sit ? 0.45f : 1.25f, 0.0f);
                log::infof("[npc] test: person %d (pose %d style %d, speed %.1f) at %.1f m", who[i].second, int(v.pose), int(v.style), v.speed, double(who[i].first));
                npc_hit(who[i].second, chest, glm::normalize(chest - current_frame_view_.eye_pos));
            }
        }
    }
    // VIBE_NPC_CAM=<who>[,<metres>[,<degrees>[,<height>]]]: the eye is put beside somebody and
    // stays with them (tests and captures; use it with VIBE_TEST_HOLD). <who>: bar, table (eat,
    // talk), counter, look, walk or walk<0..7> (the way of walking), run, driver (a car that
    // drives), id<n>. The nearest to where the game started that fits. <degrees>: 0 looks them
    // in the face, 90 from their right side, 180 from behind; <height> above their feet.
    if (static const char* nc = std::getenv("VIBE_NPC_CAM"); nc) {
        char who[32] = { 0 }; float dist = 3.0f, ang = 35.0f, up = 1.45f;
        std::sscanf(nc, "%31[^,],%f,%f,%f", who, &dist, &ang, &up);
        const std::string w(who);
        static int held = -1;
        auto fits = [&](const npc::View& v) {
            if (v.kind == npc::Kind::Car) return w == "driver" && v.pose == npc::Pose::Drive && std::abs(v.speed) > 3.0f;
            if (v.hidden) return false;
            if (w.rfind("id", 0) == 0) return int(v.id) == std::atoi(w.c_str() + 2);
            if (v.act != npc::Act::None) return false;
            if (w == "bar") return v.pose == npc::Pose::Sit && v.style == 1;
            if (w == "table") return v.pose == npc::Pose::Sit && (v.style == 2 || v.style == 3);
            if (w == "eat") return v.pose == npc::Pose::Sit && v.style == 2;
            if (w == "talk") return v.pose == npc::Pose::Sit && v.style == 3;
            if (w == "counter") return v.pose == npc::Pose::Work && v.style == 1;
            if (w == "look") return v.pose == npc::Pose::Stand && v.style > 0;
            if (w == "run") return v.pose == npc::Pose::Run;
            if (w.rfind("walk", 0) == 0) return v.pose == npc::Pose::Walk && std::abs(v.speed) > 0.3f && (w.size() == 4 || int(v.look % 8) == std::atoi(w.c_str() + 4));
            return false;
        };
        if (held < 0 || size_t(held) >= npc_views_.size() || !fits(npc_views_[size_t(held)])) {
            held = -1; float bd = 1e18f;
            for (size_t i = 0; i < npc_views_.size(); ++i) {
                if (!fits(npc_views_[i])) continue;
                const float d = glm::distance(npc_views_[i].pos, player_.position);
                if (d < bd) { bd = d; held = int(i); }
            }
            if (held >= 0) {
                const npc::View& v = npc_views_[size_t(held)];
                log::infof("[npc] the eye goes to %s %d (pose %d style %d look %d) at (%.1f, %.1f, %.1f) yaw %.2f", v.kind == npc::Kind::Car ? "car" : "person", held, int(v.pose), int(v.style), int(v.look), v.pos.x, v.pos.y, v.pos.z, v.yaw);
            }
        }
        if (held >= 0) {
            const npc::View& v = npc_views_[size_t(held)];
            const bool sits = v.kind == npc::Kind::Person && v.pose == npc::Pose::Sit;
            glm::vec3 feet = v.pos;
            if (sits) feet.y -= float(npc::kSeatHeightCm[v.style == 1 ? 1 : 0]) * 0.01f;
            const float a = v.yaw + glm::radians(ang);
            const glm::vec3 eye = feet + glm::vec3(std::sin(a) * dist, up, -std::cos(a) * dist);
            const glm::vec3 d = glm::normalize(feet + glm::vec3(0.0f, sits ? 0.95f : 1.05f, 0.0f) - eye);
            player_.position = eye - (player_.eye_position() - player_.position);
            player_.velocity = glm::vec3(0.0f);
            player_.yaw = std::atan2(d.x, -d.z);
            player_.pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
            prev_player_position_ = player_.position;
            globe_moved_player_ = true;                    // (VIBE_TEST_HOLD holds this place now)
        }
    }
    if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg && frame_number_ % 300 == 0) {
        const npc::World::Census c = npc_->census();
        int drawn = 0; for (const MpAvatar& a : npc_bodies_) drawn += a.drawn ? 1 : 0;
        {
            int lent = 0, present = 0, lod3 = 0, nopal = 0; float near_car = 1e9f, near_man = 1e9f;
            for (const NpcCar& c : npc_cars_) lent += c.agent >= 0 ? 1 : 0;
            for (const MpAvatar& a : npc_bodies_) { present += a.present ? 1 : 0; lod3 += (a.present && a.lod >= 3) ? 1 : 0; nopal += (a.present && a.palette < 0) ? 1 : 0; }
            for (const npc::View& v : npc_views_) { const float d = glm::distance(v.pos, current_frame_view_.eye_pos); if (v.kind == npc::Kind::Car) near_car = std::min(near_car, d); else if (!v.hidden) near_man = std::min(near_man, d); }
            for (const npc::View& v : npc_views_) if (v.kind == npc::Kind::Person && !v.hidden && v.pose == npc::Pose::Walk) log::infof("[npc]   walker %d at %.1f %.1f %.1f yaw %.2f speed %.2f", int(v.id), v.pos.x, v.pos.y, v.pos.z, v.yaw, v.speed);
            for (const npc::View& v : npc_views_) if (v.kind == npc::Kind::Person && !v.hidden && v.pose != npc::Pose::Walk) log::infof("[npc]   person %d pose %d style %d at %.1f %.1f %.1f yaw %.2f", int(v.id), int(v.pose), int(v.style), v.pos.x, v.pos.y, v.pos.z, v.yaw);
            for (const npc::View& v : npc_views_) if (v.kind == npc::Kind::Car && v.pose == npc::Pose::Drive && glm::distance(v.pos, current_frame_view_.eye_pos) < 500.0f) log::infof("[npc]   driving car %d at %.1f %.1f %.1f yaw %.2f speed %.1f (frame %llu)", int(v.id), v.pos.x, v.pos.y, v.pos.z, v.yaw, v.speed, static_cast<unsigned long long>(frame_number_));
            { std::istringstream ds(npc_->describe_site()); std::string ln; while (std::getline(ds, ln)) log::infof("[npc]   %s", ln.c_str()); }
            log::infof("[npc] shown: car slots %zu lent %d, nearest car %.0f m; bodies present %d (level 3: %d, no palette: %d), nearest person %.0f m; bones %zu anim %d",
                       npc_cars_.size(), lent, double(near_car), present, lod3, nopal, double(near_man), body_bones_.size(), int(anim_set_.ok()));
        }
        log::infof("[npc] people %d: sitting %d walking %d in cars %d, drawn %d | cars %d: at the station %d (longest %.0f s)%s",
                   c.people, c.sitting, c.walking, c.hidden, drawn, c.cars, c.cars_at_site, double(c.longest_visit), npc_->replica() ? " - following the host" : "");
    }
}

void VulkanEngine::npc_pose(float dt) {
    for (MpAvatar& a : npc_bodies_) { a.present = false; a.drawn = false; }
    if (npc_views_.empty() || body_bones_.empty() || !anim_set_.ok()) return;
    dt = glm::clamp(dt, 0.0f, 0.1f);
    const glm::vec3 cam = current_frame_view_.eye_pos;
    const float tan_half = current_frame_view_.proj[1][1] != 0.0f ? 1.0f / std::abs(current_frame_view_.proj[1][1]) : 0.7f;

    // ---- who is shown: the nearest people on foot, and those in the cars that are shown - the
    //      driver and whoever sits beside him. A car that drives has a driver: where the engine
    //      has nobody at its wheel (the traffic the other way carries no people of its own),
    //      one is shown all the same.
    struct Who { float d; int view; int car; int seat; uint32_t id; };
    std::vector<Who> near_;
    std::vector<uint8_t> has_driver(npc_views_.size(), 0);
    for (size_t i = 0; i < npc_views_.size(); ++i) {
        const npc::View& v = npc_views_[i];
        if (v.kind != npc::Kind::Person) continue;
        if (v.car < npc_views_.size() && v.seat == 0) has_driver[v.car] = 1;
        const float d = glm::distance(v.pos, cam);
        if (!v.hidden) { if (d < kNpcBodyReach) near_.push_back({ d, int(i), -1, 0, uint32_t(v.id) + 1u }); }
        else if (v.car < npc_views_.size() && v.seat < 2 && d < kNpcCarBodyReach) near_.push_back({ d, int(i), int(v.car), int(v.seat), uint32_t(v.id) + 1u });
    }
    for (size_t i = 0; i < npc_views_.size(); ++i) {
        const npc::View& v = npc_views_[i];
        if (v.kind != npc::Kind::Car || has_driver[i] || v.pose != npc::Pose::Drive) continue;
        if (i < 100 && npc_nodriver_[i] > 0.0f) continue;      // (that driver was shot a moment ago: npc_hit)
        const float d = glm::distance(v.pos, cam);
        if (d < kNpcCarBodyReach) near_.push_back({ d, int(i), int(i), 0, 1000u + uint32_t(v.id) });
    }
    // (VIBE_NPC_NO_RIDERS=1: nobody in the cars - to compare a picture with and without them)
    if (static const bool none = std::getenv("VIBE_NPC_NO_RIDERS") != nullptr; none)
        near_.erase(std::remove_if(near_.begin(), near_.end(), [](const Who& w) { return w.car >= 0; }), near_.end());
    std::sort(near_.begin(), near_.end(), [](const Who& a, const Who& b) { return a.d < b.d; });
    if (near_.size() > size_t(kNpcBodies)) near_.resize(size_t(kNpcBodies));
    // ---- a body each: the one it had last frame, or a free one (which starts afresh)
    std::vector<int> body_of(near_.size(), -1);
    for (size_t k = 0; k < near_.size(); ++k) {
        const uint32_t id = near_[k].id;
        for (size_t b = 0; b < npc_bodies_.size(); ++b) if (npc_bodies_[b].id == id) { body_of[k] = int(b); npc_bodies_[b].present = true; break; }
    }
    for (MpAvatar& a : npc_bodies_) if (!a.present) a.id = 0;
    for (size_t k = 0; k < near_.size(); ++k) {
        if (body_of[k] >= 0) continue;
        for (size_t b = 0; b < npc_bodies_.size(); ++b) {
            MpAvatar& a = npc_bodies_[b];
            if (a.id) continue;
            a.id = near_[k].id;
            a.present = true; a.fresh_pose = true; a.unarmed = true;
            a.palette = a.palette_prev = -1; a.pose_rows.clear(); a.lod = 0; a.skipped = 0; a.skip_dt = 0.0f;
            if (a.animator) a.animator->reset();
            body_of[k] = int(b);
            break;
        }
    }

    // ---- the animators' inputs
    struct Posed { MpAvatar* a; anim::CharacterAnimationInput in; net::RemoteFrame rf; bool skip; };
    std::vector<Posed> posed;
    posed.reserve(near_.size());
    const anim::IAnimationRuntime& art = *anim_set_.runtime;
    const int hj = art.rig()[anim::Role::Hips];
    const glm::vec3 rest = hj >= 0 ? art.skeleton().rest_pos[size_t(hj)] : glm::vec3(0.0f, 0.95f, 0.0f);
    for (size_t k = 0; k < near_.size(); ++k) {
        if (body_of[k] < 0) continue;
        MpAvatar& a = npc_bodies_[size_t(body_of[k])];
        const npc::View& v = npc_views_[size_t(near_[k].view)];
        a.palette_prev = a.fresh_pose ? -1 : a.palette;
        a.palette = -1;
        if (!a.animator) { a.animator = std::make_unique<anim::CharacterAnimator>(anim_set_); apply_anim_tuning(a.animator->tuning); a.fresh_pose = true; }
        // WHAT IT PLAYS. Most of what the station's people do is a clip made for it, round and
        // round (docs/animation/actions.md): at the bar, eating or talking at a table, a
        // cashier, somebody standing about some way of his own, and four ways of walking
        // besides the plain one. `loop_nm`: that clip, or null (the plain walk and stand, the
        // plain seat).
        const anim::IAnimationRuntime& rtm0 = *anim_set_.runtime;
        const bool run = v.pose == npc::Pose::Run;
        const char* loop_nm = nullptr;
        bool loop_seat = false, loop_walk = false;
        if (near_[k].car < 0 && v.act == npc::Act::None) {
            static const char* kShow = std::getenv("VIBE_NPC_ACT");
            // Somebody who stays does not do one thing for ever: at the bar they drink, talk and
            // sit; at a table eat, drink, talk, lean back; the cashier scans, takes, waits. Each
            // in turn, a quarter of a minute or so, everybody at their own moment (`turn`: this
            // machine's own clock - the network sends where somebody is and what they do, not
            // which clip). A way of WALKING is the person's own and never changes (`look`).
            static double life_clock = 0.0;
            static uint64_t life_frame = ~0ull;
            if (life_frame != frame_number_) { life_frame = frame_number_; life_clock += double(dt); }
            auto turn = [&](int n, double period) { return int((uint32_t(v.id) * 7u + uint32_t((life_clock + double(v.id) * 3.7) / period)) % uint32_t(n)); };
            static const char* kBar[3] = { "bar_sit", "bar_talk", "bar_idle" };
            static const char* kEat[3] = { "table_eat", "table_drink", "table_eat" };
            static const char* kTalk[3] = { "table_talk", "table_idle", "table_drink" };
            static const char* kTill[3] = { "cashier", "cashier_take", "cashier_idle" };
            static const char* kLook[5][2] = { { nullptr, nullptr }, { "look_crossed", "look_around" }, { "look_hips", "look_hips" }, { "look_phone", "look_watch" }, { "look_far", "look_far" } };
            static const char* kWalkClip[8] = { nullptr, "walk_brisk", "walk_tired", "walk_pockets", "walk_phone", "walk_relaxed", "walk_swagger", nullptr };
            auto have = [&](const char* nm) { return nm && rtm0.find_clip(nm) >= 0; };
            if (v.pose == npc::Pose::Sit) {
                loop_nm = v.style == 1 ? kBar[turn(3, 17.0)] : v.style == 2 ? kEat[turn(3, 15.0)] : v.style == 3 ? kTalk[turn(3, 16.0)] : nullptr;
                if (!have(loop_nm)) loop_nm = v.style == 1 ? "bar_sit" : v.style == 2 ? "table_eat" : v.style == 3 ? "table_talk" : nullptr;      // (a set cooked before these were made)
                loop_seat = loop_nm != nullptr;
            } else if (v.pose == npc::Pose::Work) {
                loop_nm = v.style == 1 ? kTill[turn(3, 12.0)] : (v.style == 2 || v.car != 255) ? "fuel" : "till";
                if (!have(loop_nm) && v.style == 1) loop_nm = "cashier";
            } else if (v.pose == npc::Pose::Stand) {
                loop_nm = kLook[v.style % 5][v.id & 1];
                if (!have(loop_nm)) loop_nm = kLook[v.style % 5][0];
            } else if (v.pose == npc::Pose::Walk && std::abs(v.speed) > 0.3f) { loop_nm = kWalkClip[v.look % 8]; loop_walk = loop_nm != nullptr; }
            if (kShow && v.pose != npc::Pose::Walk && v.pose != npc::Pose::Run && !loop_seat) loop_nm = kShow;
            if (loop_nm && rtm0.find_clip(loop_nm) < 0) { loop_nm = nullptr; loop_seat = loop_walk = false; }
        }
        const glm::vec3 fwd(std::sin(v.yaw), 0.0f, -std::cos(v.yaw));
        net::PlayerState s{};
        s.px = v.pos.x; s.py = v.pos.y + game::Player::kHalfExtents.y; s.pz = v.pos.z;
        // (on a seat with a clip: the model stands on the floor in front of it, the clip sits)
        if (loop_seat) s.py -= float(npc::kSeatHeightCm[v.style == 1 ? 1 : 0]) * 0.01f;
        s.vx = fwd.x * v.speed; s.vz = fwd.z * v.speed;
        s.yaw = v.yaw; s.flags = net::kFlagGround;
        s.move = (v.pose == npc::Pose::Walk || run) ? net::kMoveFwd : 0;
        if (run) s.move |= net::kMoveSprint;
        anim::CharacterAnimationInput in;
        net::RemoteFrame rf = net::remote_animation_input(s, in);
        a.unarmed = !run;                               // (the soldiers on the run carry their rifles)
        if (!run) in.held = anim::HeldItem::None;       // (nothing in the hands)
        in.aiming = false;
        in.desired_velocity = in.velocity;              // (a stroll, not a soldier's pace)
        // Upright. The soldier's walk is a patrol walk: hunched, the head down over a rifle. Somebody
        // strolling to a shop door carries the head up - the spine, neck and head are raised by what
        // the aim's pitch raises them (upper_body), more the faster they walk.
        static const float kUp = std::getenv("VIBE_NPC_UPRIGHT") ? float(std::atof(std::getenv("VIBE_NPC_UPRIGHT"))) : 0.42f;
        in.aim_pitch = run ? 0.12f : kUp * (0.35f + 0.65f * glm::clamp(std::abs(v.speed) / 1.35f, 0.0f, 1.0f));
        a.seated = false;
        // On a seat: the hips on it, the animator's seat pose, the hands where they belong.
        auto sit = [&](const glm::mat4& hip, glm::vec3 hand_l, glm::vec3 hand_r, float legs) {
            rf.model = hip * glm::translate(glm::mat4(1.0f), -rest);
            const glm::mat4 inv = glm::inverse(rf.model);
            in.seat_hand_l = glm::vec3(inv * glm::vec4(hand_l, 1.0f));
            in.seat_hand_r = glm::vec3(inv * glm::vec4(hand_r, 1.0f));
            in.seat_legs = legs;
            in.seated = true;
            in.velocity = in.desired_velocity = glm::vec3(0.0f);
            rf.eye = glm::vec3(hip * glm::vec4(0.0f, 0.62f, 0.0f, 1.0f));
            rf.view_inv = hip; rf.view_inv[3] = glm::vec4(rf.eye, 1.0f);
            a.seated = true;
        };
        if (near_[k].car >= 0) {
            // In a car: at its wheel (the hands on it, going round with it) or beside the driver.
            glm::mat4 hip; glm::vec3 hl, hr; float legs = 0.0f;
            anim::CharacterAnimationInput::SeatExtras more;
            if (!npc_car_seat(near_[k].car, near_[k].seat, hip, hl, hr, legs, &more)) { a.present = false; a.id = 0; continue; }      // (that car is not shown)
            sit(hip, hl, hr, legs);
            in.take_seat(more, glm::inverse(rf.model));      // (the wheel's frame, the feet's places, the backrest: this car's own)
            if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg && frame_number_ % 300 == 0)
                log::infof("[npc]   in car %d seat %d: body %u (last frame: level %d, palette %d, drawn %d, mesh %d) at %.1f %.1f %.1f, %.0f m from the eye", near_[k].car, near_[k].seat, a.id, a.lod, a.palette_prev, int(a.drawn), a.mesh_lod, rf.eye.x, rf.eye.y, rf.eye.z, double(near_[k].d));
        } else if (v.pose == npc::Pose::Sit && !loop_seat) {
            // (a hand above the cushion's top; the hands on the knees)
            const glm::vec3 right(std::cos(v.yaw), 0.0f, std::sin(v.yaw));
            const glm::mat4 hip(glm::vec4(right, 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f), glm::vec4(-fwd, 0.0f), glm::vec4(v.pos + glm::vec3(0.0f, 0.10f, 0.0f), 1.0f));
            sit(hip, glm::vec3(hip * glm::vec4(-0.17f, 0.14f, -0.36f, 1.0f)), glm::vec3(hip * glm::vec4(0.17f, 0.14f, -0.36f, 1.0f)), 0.0f);
        }
        // Something it is in the middle of doing: a clip, played over the time the NPC engine
        // gives the act (npc.h kActTime), the hips carried to the seat this chair or car
        // really has (motion warping: character_animator.h action_*).
        static const char* kActClip[5] = { "", "sit_down", "stand_up", "car_enter", "car_exit" };
        const anim::IAnimationRuntime& rtm = *anim_set_.runtime;
        if (near_[k].car < 0 && v.act != npc::Act::None) {
            const anim::ClipId clip = rtm.find_clip(kActClip[int(v.act)]);
            if (clip >= 0) {
                in.action_clip = clip;
                in.action_time = glm::clamp(v.act_t / npc::kActTime[int(v.act)], 0.0f, 1.0f) * rtm.clip(clip).duration;
                glm::vec3 hips_at = v.aim + glm::vec3(0.0f, 0.10f, 0.0f);
                bool have = true;
                if (v.act == npc::Act::CarEnter || v.act == npc::Act::CarExit) {
                    glm::mat4 hip; glm::vec3 hl, hr; float legs = 0.0f;
                    have = v.car != 255 && npc_car_seat(int(v.car), int(v.seat), hip, hl, hr, legs);
                    if (have) hips_at = glm::vec3(hip[3]);
                }
                in.action_warp = have;
                in.action_hips = glm::vec3(glm::inverse(rf.model) * glm::vec4(hips_at, 1.0f));
                in.action_warp_from_start = v.act == npc::Act::StandUp || v.act == npc::Act::CarExit;
                in.action_end_at_origin = in.action_warp_from_start;
                in.has_look_at = false;
            }
        } else if (near_[k].car < 0 && !a.seated && loop_nm) {
            // A loop (see WHAT IT PLAYS above). A walk is played at the pace the body really
            // moves (its own travel left out: the NPC engine moves it), a seat's clip with the
            // hips on this seat.
            const anim::ClipId clip = rtm.find_clip(loop_nm);
            const float dur = std::max(rtm.clip(clip).duration, 0.1f);
            float rate = 1.0f;
            if (loop_walk) {
                const anim::Transform tr = a.animator->action_root(clip, dur);
                const float cs = glm::length(glm::vec2(tr.t.x, tr.t.z)) / dur;
                rate = cs > 0.2f ? glm::clamp(std::abs(v.speed) / cs, 0.4f, 1.8f) : 1.0f;
            }
            if (a.fresh_pose) a.npc_phase = float(a.id) * 0.73f;
            a.npc_phase = std::fmod(a.npc_phase + dt * rate, dur);
            in.action_clip = clip;
            in.action_time = a.npc_phase;
            in.action_in_place = loop_walk;
            if (loop_seat) {
                in.action_warp = true; in.action_warp_full = true;
                in.action_hips = glm::vec3(glm::inverse(rf.model) * glm::vec4(v.pos + glm::vec3(0.0f, 0.10f, 0.0f), 1.0f));
                in.velocity = in.desired_velocity = glm::vec3(0.0f);
            }
        }
        anim::LodInput li;
        li.distance = glm::length(rf.eye - cam);
        li.screen_height = float(swapchain_extent_.height);
        li.tan_half_fov_y = tan_half;
        {
            const glm::vec4 c = current_frame_view_.vp * glm::vec4(rf.eye - glm::vec3(0.0f, 0.7f, 0.0f), 1.0f);
            const float m = 1.25f * c.w + 1.5f;
            li.in_view = c.w > -1.5f && std::abs(c.x) < m && std::abs(c.y) < m;
        }
        li.combat = false;
        li.previous = a.lod;
        a.lod = std::max(anim::select_lod(li), near_[k].d < 12.0f ? 0 : 1);      // (full detail only close by: there are many of them)
        in.dt = dt;
        in.variation = a.id * 7919u;
        // (VIBE_LOG_NPC: what each body that is shown plays - a clip by its name, or which of the animator's own poses)
        if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg && frame_number_ % 300 == 0) {
            const char* plays = in.seated ? (near_[k].car >= 0 ? (near_[k].seat == 0 ? "the driver's pose" : "the passenger's pose") : "the seat pose")
                              : in.action_clip >= 0 ? rtm.clip(in.action_clip).name.c_str()
                              : run ? "the motion-matched run" : v.pose == npc::Pose::Walk ? "the motion-matched walk" : "the motion-matched stand";
            log::infof("[npc]   plays: %s %d (pose %d style %d look %d, %.0f m off) - %s at %.2f s", near_[k].car >= 0 ? "in car" : "person", near_[k].car >= 0 ? near_[k].car : int(v.id),
                       int(v.pose), int(v.style), int(v.look), double(near_[k].d), plays, in.action_clip >= 0 ? double(in.action_time) : 0.0);
        }
        // somebody near looks at the player
        if (near_[k].d < 7.0f && !a.seated && !run && !loop_seat) { in.has_look_at = true; in.look_at = glm::vec3(glm::inverse(rf.model) * glm::vec4(cam, 1.0f)); }
        in.teleported = a.fresh_pose && !a.seated;
        in.lod = a.lod;
        const bool skip = a.lod == 2 && !a.fresh_pose && !a.die && a.skipped == 0 && (frame_number_ & 1u) && !a.pose_rows.empty();
        if (skip) { a.skipped = 1; a.skip_dt = dt; }
        else { in.dt = dt + a.skip_dt; a.skipped = 0; a.skip_dt = 0.0f; if (a.lod >= 3) a.pose_rows.clear(); }
        posed.push_back({ &a, in, rf, skip });
    }
    // ---- posed on the job threads, eight at a time (pose_update_all's probes are that many)
    for (size_t i0 = 0; i0 < posed.size(); i0 += 8) {
        PoseJob jobs[8];
        int nj = 0;
        for (size_t i = i0; i < std::min(posed.size(), i0 + 8); ++i) if (!posed[i].skip) jobs[nj++] = { posed[i].a->animator.get(), &posed[i].in };
        if (nj) pose_update_all(jobs, nj);
    }
    for (Posed& r : posed) {
        MpAvatar& a = *r.a;
        if (r.skip) a.palette = body_push_rows(a.pose_rows);
        else pose_finish(*a.animator, r.in, a.lod >= 3 ? nullptr : &a.palette, &a.pose_rows);
        a.prev_model = a.fresh_pose ? r.rf.model : a.model;
        a.fresh_pose = false;
        a.model = r.rf.model;
        if (a.die) {
            // Shot (npc_bolt_hits): this pose goes to the physics as a body that falls, pushed by
            // the shot; the NPC engine has that person leave its world (the authority's; a copy
            // of it keeps the person out of sight until it has followed: npc_gone_).
            a.die = false;
            if (a.lod >= 3) { r.in.lod = 2; r.in.dt = 0.0f; pose_character(*a.animator, r.in, nullptr); }
            const glm::mat4 vi(glm::vec4(std::cos(a.dying_st.yaw), 0.0f, std::sin(a.dying_st.yaw), 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                               glm::vec4(-std::sin(a.dying_st.yaw), 0.0f, std::cos(a.dying_st.yaw), 0.0f), glm::vec4(r.rf.eye, 1.0f));
            if (npc_ && a.id >= 1 && a.id <= 100) { if (npc_->replica()) npc_gone_[a.id - 1] = 3.0f; else npc_->kill(int(a.id) - 1); }
            // (the driver of a car that has nobody of the NPC world at its wheel: the car's to lose)
            if (npc_ && a.id >= 1000u && a.id < 1100u) { if (npc_->replica()) npc_nodriver_[a.id - 1000u] = 3.0f; else npc_->kill(int(a.id - 1000u)); }
            // On its feet: the reaction first (npc_dying_pose). On a seat, or in the middle of
            // sitting down or getting into a car (the hips carried by a warp): the physics' at once.
            const glm::vec3 fwd(std::sin(a.dying_st.yaw), 0.0f, -std::cos(a.dying_st.yaw));
            const ShotClip& sc = pick_shot_clip(a.hit_pos.y - r.rf.model[3].y, glm::dot(a.hit_dir, fwd) < 0.0f, glm::length(r.in.velocity), a.id);
            const int clip = (r.in.seated || r.in.action_warp || std::getenv("VIBE_NO_DEATH_CLIPS")) ? -1 : anim_set_.runtime->find_clip(sc.name);
            if (clip < 0) {
                // (shot in a car that drives: the body goes on at the car's speed - npc_hit put it in dying_st)
                const glm::vec3 vel = r.in.seated ? glm::vec3(a.dying_st.vx, 0.0f, a.dying_st.vz) : r.in.velocity;
                corpse_spawn(r.rf.model, vi, vel, a.hit_pos, a.hit_dir, std::max(a.hit_power, 0.6f));
                a.id = 0; a.present = false; a.drawn = false; a.palette = -1;
                continue;
            }
            log::infof("[npc] person %u goes down: %s", a.id - 1, sc.name);
            NpcDying d;
            d.in = r.in; d.model = r.rf.model; d.view_inv = vi; d.clip = clip; d.hand = sc.hand;
            d.hit_pos = a.hit_pos; d.hit_dir = a.hit_dir;
            a.view_inv = r.rf.view_inv; a.eye = r.rf.eye; a.drawn = a.lod < 3 && a.palette >= 0;
            d.av = std::move(a);
            a = MpAvatar{};
            d.av.id = 5000u + d.av.id;
            if (npc_dying_.size() < 8) npc_dying_.push_back(std::move(d));
            continue;
        }
        a.view_inv = r.rf.view_inv;
        a.eye = r.rf.eye;
        a.mesh_lod = body_mesh_lod(r.rf.eye, a.mesh_lod);
        a.drawn = a.lod < 3 && a.palette >= 0;
    }
}

// A bolt of ours against the station's people (combat.cpp, beside mp_bolt_hits): whoever is on
// foot or on a seat can be shot - by any player. The shooter decides the hit against what is
// drawn on the shooter's screen (as for players), shows it at once (npc_hit) and tells
// everybody (Event::NpcHit); every machine shows the same, and the machine whose NPC world is
// the world takes that person out of it.
bool VulkanEngine::npc_bolt_hits(glm::vec3 a, glm::vec3 b, glm::vec3& hit) {
    if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg) log::infof("[npc] bolt (%.2f %.2f %.2f) -> (%.2f %.2f %.2f), %zu views", a.x, a.y, a.z, b.x, b.y, b.z, npc_views_.size());
    if (!npc_ || npc_views_.empty()) return false;
    const glm::vec3 lo = glm::min(a, b) - glm::vec3(1.0f, 2.2f, 1.0f), hi = glm::max(a, b) + glm::vec3(1.0f, 1.2f, 1.0f);
    int best = -1; float bt = 2.0f;
    for (size_t i = 0; i < npc_views_.size(); ++i) {
        const npc::View& v = npc_views_[i];
        if (v.kind != npc::Kind::Person || v.hidden) continue;
        if (v.pos.x < lo.x || v.pos.x > hi.x || v.pos.z < lo.z || v.pos.z > hi.z || v.pos.y < lo.y || v.pos.y > hi.y) continue;
        // (standing: the feet to the head; on a seat: `pos` is the seat's top - the legs below it, the trunk above)
        const bool sits = v.pose == npc::Pose::Sit;
        const glm::vec3 mn = v.pos + glm::vec3(-0.30f, sits ? -0.45f : 0.0f, -0.30f), mx = v.pos + glm::vec3(0.30f, sits ? 0.95f : 1.82f, 0.30f);
        float t = 0.0f;
        if (seg_aabb(a, b, mn, mx, t) && t < bt) { bt = t; best = int(i); }
    }
    // Whoever sits in a car that is shown - the driver, the one beside him: the body drawn there
    // (its head and trunk). A driver nobody of the NPC world sits for goes by his car's number.
    for (const MpAvatar& av : npc_bodies_) {
        if (!av.present || !av.id) continue;
        const int agent = av.id >= 1000u ? int(av.id - 1000u) : int(av.id) - 1;
        if (agent < 0 || size_t(agent) >= npc_views_.size() || agent >= 100) continue;
        if (av.id < 1000u && !npc_views_[size_t(agent)].hidden) continue;      // (on foot: above)
        const glm::vec3 mn = av.eye + glm::vec3(-0.27f, -0.70f, -0.27f), mx = av.eye + glm::vec3(0.27f, 0.22f, 0.27f);
        float t = 0.0f;
        if (seg_aabb(a, b, mn, mx, t) && t < bt) { bt = t; best = agent; }
    }
    if (best < 0) return false;
    hit = a + (b - a) * bt;
    const glm::vec3 dir = glm::normalize(b - a + glm::vec3(1e-6f, 0.0f, 0.0f));
    npc_hit(best, hit, dir);
    if (net_) {
        net::Event e{};
        e.kind = net::Event::NpcHit; e.from = net_->id(); e.to = uint32_t(best);
        e.a[0] = hit.x; e.a[1] = hit.y; e.a[2] = hit.z; e.b[0] = dir.x; e.b[1] = dir.y; e.b[2] = dir.z;
        net_->send_event(e);
    }
    mp_->hitmark = 0.25f;
    return true;
}

// Somebody was shot - by this machine's player or (told by the network) by another's. Blood
// where it hit; the body that is drawn for that person here is told to fall (npc_pose hands it
// to the physics, each machine its own fall from the same pose); and the person leaves the NPC
// world: the authority's kill(), which the copies follow. Until a copy has followed, the person
// is kept from being drawn again beside their own body (npc_gone_).
void VulkanEngine::npc_hit(int agent, glm::vec3 hit, glm::vec3 dir) {
    if (!npc_ || agent < 0 || size_t(agent) >= npc_views_.size() || agent >= 100) return;
    const npc::View& v = npc_views_[size_t(agent)];
    // Who: somebody on foot or on a seat; somebody in a car (hidden, with a car); or - `agent`
    // is a car - the driver of a car that drives with nobody of the NPC world at its wheel.
    const bool phantom = v.kind == npc::Kind::Car;
    const npc::View* car = phantom ? &v : (v.hidden && size_t(v.car) < npc_views_.size()) ? &npc_views_[v.car] : nullptr;
    if (phantom ? (v.hidden || v.pose != npc::Pose::Drive || npc_nodriver_[agent] > 0.0f) : (v.hidden && !car)) return;       // (gone already: two players shot the same one)
    spawn_blood(hit, dir);
    const uint32_t id = phantom ? 1000u + uint32_t(v.id) : uint32_t(v.id) + 1u;
    bool body = false;
    for (MpAvatar& av : npc_bodies_)
        if (av.id == id && av.present) {
            av.die = true; av.hit_pos = hit; av.hit_dir = dir; av.hit_power = 1.0f; av.hit_age = 0.0f;
            av.dying_st.yaw = v.yaw;
            av.dying_st.vx = car ? std::sin(car->yaw) * car->speed : 0.0f;       // (in a car: the body goes on as the car went)
            av.dying_st.vz = car ? -std::cos(car->yaw) * car->speed : 0.0f;
            body = true;
            break;
        }
    if (!body) {
        if (npc_->replica()) (phantom ? npc_nodriver_ : npc_gone_)[agent] = 3.0f; else npc_->kill(agent);
    }
    log::infof("[npc] %s %d shot at (%.1f, %.1f, %.1f)%s%s", phantom ? "the driver of car" : "person", agent, hit.x, hit.y, hit.z, car && !phantom ? " in a car" : "", body ? "" : " (no body drawn: gone at once)");
}

// VIBE_TEST_NPC_SHOOT=<frame>[,<who>[,<skip>]]: at that frame the player turns to somebody of
// the NPC world and fires one shot - a real bolt out of the gun, so the whole way from the
// trigger to the body on the ground is what is tested (VIBE_TEST_NPC_KILL calls npc_hit
// itself). who 0: the nearest on foot or on a seat; 1: the nearest who sits in a car (aimed
// ahead of it by the bolt's flight time). skip: after that many nearer ones. Called from
// loop_test_input, before the weapons.
void VulkanEngine::npc_test_input(InputFrame& in) {
    static const char* spec = std::getenv("VIBE_TEST_NPC_SHOOT");
    if (!spec || !npc_) return;
    int at = 0, who = 0, skip = 0;
    std::sscanf(spec, "%d,%d,%d", &at, &who, &skip);
    if (int(frame_number_) != at) return;
    const glm::vec3 eye = player_.eye_position();
    struct Target { float d; glm::vec3 p, v; int agent; };
    std::vector<Target> ts;
    for (const MpAvatar& a : npc_bodies_) {
        if (!a.present || !a.id) continue;
        const bool phantom = a.id >= 1000u;
        const int agent = phantom ? int(a.id - 1000u) : int(a.id) - 1;
        if (agent < 0 || size_t(agent) >= npc_views_.size()) continue;
        const npc::View& v = npc_views_[size_t(agent)];
        const bool in_car = phantom || v.hidden;
        if (in_car != (who == 1)) continue;
        const npc::View& mv = (!phantom && v.hidden && size_t(v.car) < npc_views_.size()) ? npc_views_[v.car] : v;
        const glm::vec3 p = in_car ? a.eye - glm::vec3(0.0f, 0.25f, 0.0f) : v.pos + glm::vec3(0.0f, v.pose == npc::Pose::Sit ? 0.45f : 1.25f, 0.0f);
        // (only somebody the bolt can reach: it is a body of the physics, and a table, a pane or a wall stops it)
        const float dist = glm::distance(p, eye);
        if (physics_ && dist > 0.6f) { const auto h = physics_->raycast(eye, (p - eye) / dist, dist); if (h.hit && h.distance < dist - 0.4f) continue; }
        ts.push_back({ dist, p, glm::vec3(std::sin(mv.yaw), 0.0f, -std::cos(mv.yaw)) * mv.speed, agent });
    }
    std::sort(ts.begin(), ts.end(), [](const Target& a, const Target& b) { return a.d < b.d; });
    if (size_t(skip) >= ts.size()) { log::warnf("[npc] test: nobody to shoot at (%zu %s)", ts.size(), who == 1 ? "in cars" : "on foot"); return; }
    const Target& t = ts[size_t(skip)];
    const glm::vec3 d = glm::normalize(t.p + t.v * (t.d / 220.0f) - eye);
    player_.yaw = std::atan2(d.x, -d.z);
    player_.pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
    in.fire = true; in.fire_held = true;
    const auto near_hit = physics_ ? physics_->raycast(eye, d, 0.15f) : PhysicsWorld::RayHit{};
    log::infof("[npc] test: a shot at %s %d, %.1f m away, moving %.1f m/s (yaw %.3f pitch %.3f)%s; weapon %d, cooldown %.2f", who == 1 ? "somebody in car / person" : "person", t.agent, double(t.d), double(glm::length(t.v)), player_.yaw, player_.pitch,
               near_hit.hit ? " - BUT something is right in front of the muzzle: no bolt will leave" : "", int(weapon_), double(fire_cooldown_));
}

// From the network (multiplayer.cpp hands them over).
void VulkanEngine::npc_event(const net::Event& e) {
    if (!npc_ || !net_) return;
    if (e.kind == net::Event::NpcHit) {
        const glm::vec3 hit(e.a[0], e.a[1], e.a[2]), dir(e.b[0], e.b[1], e.b[2]);
        if (!std::isfinite(hit.x + hit.y + hit.z + dir.x + dir.y + dir.z)) return;
        npc_hit(int(e.to), hit, dir);
    } else if (e.kind == net::Event::NpcCarTake) {
        // (the host's to answer; refused without a word when the car drove off meanwhile)
        if (net_->role() == net::Role::Host && e.to < 100) npc_host_take(int(e.to), e.from);
    } else if (e.kind == net::Event::NpcCarTaken) {
        const net::NpcCarTaken t = net::unpack_taken(e);
        if (npc_car_make(t) && t.taker == net_->id() && jeep_ && !jeep_driving() && jeep_ride_ < 0) {
            const bool in = jeep_use();                        // (the car we asked for: in)
            log::infof("[jeep] the host gave us NPC car %d: number %d, in it %d", int(t.agent), int(t.car), int(in && jeep_driving()));
        }
    }
}

// BLOOD. A shot that hits somebody throws a spray the way the bolt went (and a little back at
// the shooter), drops that fall and leave stains where they land. Drawn by fx.cpp (kind 7 of
// fx_ribbon.frag); the stains fade over two minutes.
void VulkanEngine::spawn_blood(glm::vec3 pos, glm::vec3 dir) {
    if (blood_.size() > 400) blood_.erase(blood_.begin(), blood_.begin() + 120);
    for (int i = 0; i < 16; ++i) {
        Blood d;
        const glm::vec3 rnd(frand_range(spawn_rng_state_, -1.0f, 1.0f), frand_range(spawn_rng_state_, -0.4f, 1.0f), frand_range(spawn_rng_state_, -1.0f, 1.0f));
        const bool back = i < 4;
        d.pos = pos + rnd * 0.05f;
        d.vel = (back ? -dir * frand_range(spawn_rng_state_, 0.6f, 1.8f) : dir * frand_range(spawn_rng_state_, 1.2f, 4.5f)) + rnd * 1.1f;
        d.size = i < 3 ? frand_range(spawn_rng_state_, 0.10f, 0.17f) : frand_range(spawn_rng_state_, 0.018f, 0.045f);       // (three puffs of mist, the rest drops)
        d.life = i < 3 ? 0.45f : 4.0f;
        d.seed = frand_range(spawn_rng_state_, 0.0f, 50.0f);
        blood_.push_back(d);
    }
}

void VulkanEngine::blood_update(float dt) {
    for (Blood& d : blood_) {
        d.t += dt;
        if (d.stain) continue;
        const bool mist = d.size > 0.09f;
        d.vel.y -= (mist ? 2.0f : 9.8f) * dt;
        if (mist) d.vel *= std::max(0.0f, 1.0f - 5.0f * dt);
        const glm::vec3 step = d.vel * dt;
        const float len = glm::length(step);
        if (!mist && physics_ && len > 1e-5f) {
            const auto h = physics_->raycast(d.pos, step / len, len + 0.02f);
            if (h.hit) {
                // (lands: a stain, a little bigger than the drop, lying on what it hit)
                d.pos = d.pos + step / len * std::max(h.distance - 0.012f, 0.0f);
                d.vel = h.normal; d.stain = true; d.t = 0.0f; d.life = 120.0f; d.size *= frand_range(spawn_rng_state_, 1.6f, 3.2f);
                continue;
            }
        }
        d.pos += step;
    }
    blood_.erase(std::remove_if(blood_.begin(), blood_.end(), [](const Blood& d) { return d.t > d.life; }), blood_.end());
}

void VulkanEngine::draw_npc_bodies(
VkCommandBuffer cmd, const glm::mat4& vp) {
    for (MpAvatar& a : npc_bodies_) {
        if (!a.id || !a.drawn) continue;
        mp_->draw = &a;
        draw_viewmodel(cmd, vp, a.view_inv);       // (the body alone: `unarmed`)
        mp_->draw = nullptr;
    }
    for (NpcDying& d : npc_dying_) {
        if (!d.av.drawn) continue;
        d.av.unarmed = true;
        mp_->draw = &d.av;
        draw_viewmodel(cmd, vp, d.av.view_inv);
        mp_->draw = nullptr;
    }
}

}  // namespace vibe
