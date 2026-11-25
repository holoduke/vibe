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
    for (float left = dt * kFast; left > 1e-5f; left -= 0.1f) npc_->update(std::min(left, 0.1f));
    npc_->views(npc_views_);
    npc_cars_place(npc_views_, dt);
    npc_pose(dt);
    blood_update(dt);
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
            static const char* kLook[5] = { nullptr, "look_crossed", "look_hips", "look_phone", "look_far" };
            static const char* kWalkClip[8] = { nullptr, "walk_brisk", "walk_tired", "walk_pockets", "walk_phone", nullptr, nullptr, nullptr };
            if (v.pose == npc::Pose::Sit) { loop_nm = v.style == 1 ? ((v.id & 1) ? "bar_talk" : "bar_sit") : v.style == 2 ? "table_eat" : v.style == 3 ? "table_talk" : nullptr; loop_seat = loop_nm != nullptr; }
            else if (v.pose == npc::Pose::Work) loop_nm = v.style == 1 ? ((v.id & 1) ? "cashier_take" : "cashier") : (v.style == 2 || v.car != 255) ? "fuel" : "till";
            else if (v.pose == npc::Pose::Stand) loop_nm = kLook[v.style % 5];
            else if (v.pose == npc::Pose::Walk && std::abs(v.speed) > 0.3f) { loop_nm = kWalkClip[v.look % 8]; loop_walk = loop_nm != nullptr; }
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
            if (!npc_car_seat(near_[k].car, near_[k].seat, hip, hl, hr, legs)) { a.present = false; a.id = 0; continue; }      // (that car is not shown)
            sit(hip, hl, hr, legs);
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
            // the shot; the NPC engine has that person leave its world (a replica cannot: there
            // the body falls and the host's person stands on - shots at NPCs are the host's).
            a.die = false;
            if (a.lod >= 3) { r.in.lod = 2; r.in.dt = 0.0f; pose_character(*a.animator, r.in, nullptr); }
            const glm::mat4 vi(glm::vec4(std::cos(a.dying_st.yaw), 0.0f, std::sin(a.dying_st.yaw), 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                               glm::vec4(-std::sin(a.dying_st.yaw), 0.0f, std::cos(a.dying_st.yaw), 0.0f), glm::vec4(r.rf.eye, 1.0f));
            corpse_spawn(r.rf.model, vi, r.in.velocity, a.hit_pos, a.hit_dir, std::max(a.hit_power, 0.6f));
            if (npc_ && a.id >= 1 && a.id < 1000) npc_->kill(int(a.id) - 1);
            a.id = 0; a.present = false; a.drawn = false; a.palette = -1;
            continue;
        }
        a.view_inv = r.rf.view_inv;
        a.eye = r.rf.eye;
        a.mesh_lod = body_mesh_lod(r.rf.eye, a.mesh_lod);
        a.drawn = a.lod < 3 && a.palette >= 0;
    }
}

// A bolt of ours against the station's people (combat.cpp, beside mp_bolt_hits): whoever is on
// foot or on a seat can be shot. Blood where it hits; the body that is drawn for that person is
// told to fall (npc_pose); somebody too far off to have a body is simply gone.
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
    if (best < 0) return false;
    hit = a + (b - a) * bt;
    const glm::vec3 dir = glm::normalize(b - a + glm::vec3(1e-6f, 0.0f, 0.0f));
    spawn_blood(hit, dir);
    const uint32_t id = uint32_t(npc_views_[size_t(best)].id) + 1u;
    bool body = false;
    for (MpAvatar& av : npc_bodies_)
        if (av.id == id && av.present) {
            av.die = true; av.hit_pos = hit; av.hit_dir = dir; av.hit_power = 1.0f; av.hit_age = 0.0f;
            av.dying_st.yaw = npc_views_[size_t(best)].yaw;
            body = true;
            break;
        }
    if (!body) npc_->kill(best);
    log::infof("[npc] person %d shot at (%.1f, %.1f, %.1f)%s", best, hit.x, hit.y, hit.z, body ? "" : " (no body drawn: gone at once)");
    mp_->hitmark = 0.25f;
    return true;
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
}

}  // namespace vibe
