// The soldier's body: the local player's own (seen from the eyes) and, through the same code,
// every other player's. A skinned mesh (built with MakeHuman / MPFB in Blender, tools/character/)
// posed by the animation library (src/anim, see docs/animation/architecture.md):
//
//   controller state  ->  anim::CharacterAnimator  ->  joint matrices  ->  bone palette  ->  GPU skinning (cube_skin.vert)
//
// This file only gathers what the animator is told (where the character is, how it moves, where
// it aims, what it holds), skins the result (about 43k vertices) into a per-frame vertex buffer
// and draws it with the ordinary mesh shader. The animator never touches the controller.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/multiplayer.h"
#include "engine/vk_engine/state/people.h"
#include "npc/npc.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"
#include "game/tri_collision.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace vibe {

namespace {

float env_float(const char* name, float def) {
    const char* s = std::getenv(name);
    return s ? float(std::atof(s)) : def;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct GroundProbe { const VulkanEngine* engine; float feet_y; };

}  // namespace

int VulkanEngine::body_bone(const char* name) const {
    for (size_t i = 0; i < body_bones_.size(); ++i)
        if (body_bones_[i].name == name) return int(i);
    return -1;
}

// Reads one .qchar. with_bones: this is the file the skeleton is taken from (and the scale that
// puts the model's eyes at the player's eye height); every other file must be on that skeleton.
bool VulkanEngine::load_skin_mesh(const char* file, SkinMesh& out, bool with_bones, bool rt_part) {
    std::string path;
    for (const char* dir : { "assets/character/", "../assets/character/" }) {
        std::ifstream t(std::string(dir) + file, std::ios::binary);
        if (t.good()) { path = std::string(dir) + file; break; }
    }
    if (path.empty()) return false;
    std::ifstream f(path, std::ios::binary);
    char magic[4]{};
    uint32_t nb = 0, nv = 0, ni = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&nb), 4);
    f.read(reinterpret_cast<char*>(&nv), 4);
    f.read(reinterpret_cast<char*>(&ni), 4);
    if (std::memcmp(magic, "QCH1", 4) != 0 || nb == 0 || nb > 255 || nv == 0 || nv > 2000000 || ni > 12000000) {
        log::warnf("[body] %s: bad character file", file);
        return false;
    }
    std::vector<BodyBone> bones(nb);
    for (BodyBone& b : bones) {
        char name[32]{};
        int32_t parent = -1;
        f.read(name, 32);
        f.read(reinterpret_cast<char*>(&parent), 4);
        f.read(reinterpret_cast<char*>(&b.head), 12);
        f.read(reinterpret_cast<char*>(&b.tail), 12);
        name[31] = 0;
        b.name = name;
        if (const size_t c = b.name.find(':'); c != std::string::npos) b.name = b.name.substr(c + 1);
        b.parent = parent;
    }
    struct FileVert { float p[3], n[3], uv[2]; uint8_t b[4], w[4]; };
    static_assert(sizeof(FileVert) == 40, "qchar vertex layout");
    std::vector<FileVert> fv(nv);
    f.read(reinterpret_cast<char*>(fv.data()), std::streamsize(nv) * sizeof(FileVert));
    std::vector<uint32_t> idx(ni);
    f.read(reinterpret_cast<char*>(idx.data()), std::streamsize(ni) * 4);
    if (!f.good()) {
        log::warnf("[body] %s: truncated", file);
        return false;
    }
    if (with_bones) {
        body_bones_ = std::move(bones);
        // Scale so the model's eyes are at the player's standing eye height.
        float top = 0.0f;
        for (const FileVert& v : fv) top = std::max(top, v.p[1]);
        const int head = body_bone("Head");
        // (the head is not in this mesh: its height is the neck joint plus a head)
        const float model_eye = head >= 0 ? body_bones_[head].head.y + 0.035f : top + 0.13f;
        const float stand_eye = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
        body_scale_ = stand_eye / std::max(model_eye, 0.5f);
        for (BodyBone& b : body_bones_) { b.head *= body_scale_; b.tail *= body_scale_; }
        log::infof("[body] soldier: %u bones, scale %.3f (model eye %.3f m)", nb, body_scale_, model_eye);
    } else {
        bool same = bones.size() == body_bones_.size();
        for (size_t i = 0; same && i < bones.size(); ++i) same = bones[i].name == body_bones_[i].name && bones[i].parent == body_bones_[i].parent;
        if (!same) {
            log::warnf("[body] %s is on another skeleton than the body: not used (export the character again)", file);
            return false;
        }
    }
    // Two static vertex streams: the rest pose, and each vertex's bones and weights. The GPU
    // skins (cube_skin.vert) with the bone palette of the character being drawn.
    std::vector<Vertex> stream(nv);
    for (uint32_t i = 0; i < nv; ++i)
        stream[i] = { glm::vec3(fv[i].p[0], fv[i].p[1], fv[i].p[2]) * body_scale_, glm::vec3(fv[i].n[0], fv[i].n[1], fv[i].n[2]), glm::vec2(fv[i].uv[0], fv[i].uv[1]) };
    out.rest = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_, stream.data(), nv, idx.data(), ni);
    const uint32_t rt_base = uint32_t(body_rt_rest_.size());
    if (rt_part) {                      // part of the mesh the ray tracer gets (body_rt.cpp)
        body_rt_rest_.insert(body_rt_rest_.end(), stream.begin(), stream.end());
        for (uint32_t i : idx) body_rt_idx_.push_back(rt_base + i);
    }
    for (uint32_t i = 0; i < nv; ++i) {
        float b[4], w[4];
        for (int k = 0; k < 4; ++k) { b[k] = float(std::min<uint8_t>(fv[i].b[k], uint8_t(nb - 1))); w[k] = fv[i].w[k] / 255.0f; }
        stream[i] = { glm::vec3(b[0], b[1], b[2]), glm::vec3(b[3], w[0], w[1]), glm::vec2(w[2], w[3]) };
    }
    out.skin = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_, stream.data(), nv, idx.data(), 3);
    if (rt_part) body_rt_skin_.insert(body_rt_skin_.end(), stream.begin(), stream.end());
    out.index_count = ni;
    out.vertex_count = nv;
    return true;
}

void VulkanEngine::init_player_body() {
    if (!load_skin_mesh("soldier_fp.qchar", body_lod_[0], true)) {
        log::info("[body] no assets/character/soldier_fp.qchar - no visible player body");
        body_bones_.clear();
        return;
    }
    const uint32_t nb = uint32_t(body_bones_.size());
    load_skin_mesh("soldier_fp_lod1.qchar", body_lod_[1], false);
    body_rt_rest_.clear(); body_rt_skin_.clear(); body_rt_idx_.clear();
    load_skin_mesh("soldier_fp_lod2.qchar", body_lod_[2], false, true);
    load_skin_mesh("soldier_head.qchar", head_lod_[0], false);
    load_skin_mesh("soldier_head_lod1.qchar", head_lod_[1], false);
    load_skin_mesh("soldier_head_lod2.qchar", head_lod_[2], false, true);
    init_body_rt();
    init_people();
    log::infof("[body] triangles by level of detail: body %u / %u / %u, head %u / %u / %u (0 = file missing: the level before is drawn)",
               body_lod_[0].index_count / 3, body_lod_[1].index_count / 3, body_lod_[2].index_count / 3,
               head_lod_[0].index_count / 3, head_lod_[1].index_count / 3, head_lod_[2].index_count / 3);
    if (nb > kSkinBones) log::warnf("[body] %u bones: more than the palette holds (%u)", nb, kSkinBones);
    body_R_.assign(nb, glm::mat3(1.0f));
    body_P_.assign(nb, glm::vec3(0.0f));

    // Animation: the cooked asset (tools/anim/cook_main.cpp -> vibe_animcook). Its skeleton must
    // be this mesh's, joint for joint, or the skin would follow the wrong bones.
    body_animator_.reset();
    std::string why;
    bool loaded = false;
    for (const char* p : { "assets/character/soldier.qanim", "../assets/character/soldier.qanim" }) {
        std::ifstream t(p, std::ios::binary);
        if (!t.good()) continue;
        loaded = anim_set_.load(p, &why);
        break;
    }
    if (loaded) {
        const anim::SkeletonDef& s = anim_set_.runtime->skeleton();
        bool same = size_t(s.size()) == body_bones_.size();
        for (size_t i = 0; same && i < body_bones_.size(); ++i)
            same = s.names[i] == body_bones_[i].name && s.parents[i] == body_bones_[i].parent &&
                   glm::length(s.rest_pos[i] - body_bones_[i].head) < 2e-3f;
        if (!same) { loaded = false; why = "its skeleton is not the character mesh's (cook the asset again)"; }
    }
    if (!loaded) {
        log::warnf("[body] no animation: assets/character/soldier.qanim %s - the body is not drawn",
                   why.empty() ? "is missing (run vibe_animcook)" : why.c_str());
        anim_set_ = anim::AnimationSet{};
        return;
    }
    body_animator_ = std::make_unique<anim::CharacterAnimator>(anim_set_);
    apply_anim_tuning(body_animator_->tuning);
    anim_debug_ = std::getenv("VIBE_ANIM_DEBUG") != nullptr;
    log::infof("[body] animation: %s, %d clips (%.1f MB compressed), motion database %u frames (%.1f MB), %zu jumps",
               anim_set_.runtime->name(), anim_set_.runtime->num_clips(), double(anim_set_.runtime->clip_bytes()) / 1048576.0,
               anim_set_.db.frames(), double(anim_set_.db.memory_bytes()) / 1048576.0, anim_set_.db.jumps().size());
}

// The animation's style, without a rebuild: VIBE_ANIM_TUNE="name=value,name=value". Names:
//   halflife (s, inertial blend)   run_speed (m/s the game's run is matched to in the capture)
//   max_rate  max_stride  max_warp (rad)  lean (0 = none)  foot_ik  foot_lock  stride_warp  orient_warp
//   hand_ik (0/1)   air (airborne pose strength)   air_sway   land_dip   reach_time (s)
//   search (s between searches)   lock_speed (m/s above which feet are not held)
void VulkanEngine::apply_anim_tuning(anim::AnimTuning& t) const {
    const char* s = std::getenv("VIBE_ANIM_TUNE");
    if (!s) return;
    std::string spec(s);
    size_t at = 0;
    while (at < spec.size()) {
        const size_t end = std::min(spec.find(',', at), spec.size());
        const std::string kv = spec.substr(at, end - at);
        at = end + 1;
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq);
        const float v = float(std::atof(kv.c_str() + eq + 1));
        if (k == "halflife") t.inertial_halflife = v;
        else if (k == "run_speed") t.matcher.run_speed = v;
        else if (k == "max_rate") t.max_rate = v;
        else if (k == "max_stride") t.max_stride = v;
        else if (k == "max_warp") t.max_orient_warp = v;
        else if (k == "lean") t.lean_strength = v;
        else if (k == "foot_ik") t.foot_ik = v != 0.0f;
        else if (k == "foot_lock") t.foot_lock = v != 0.0f;
        else if (k == "stride_warp") t.stride_warp = v != 0.0f;
        else if (k == "orient_warp") t.orient_warp = v != 0.0f;
        else if (k == "hand_ik") t.hand_ik = v != 0.0f;
        else if (k == "air") t.airborne_pose_strength = v;
        else if (k == "air_sway") t.air_sway = v;
        else if (k == "land_dip") t.land_dip = v;
        else if (k == "reach_time") t.air.reach_time = v;
        else if (k == "search") t.matcher.search_interval = v;
        else if (k == "face") t.face = v;
        else if (k == "blink") t.blink_hold = v;
        else if (k == "jaw") t.jaw_hold = v;
        else if (k == "eyes") t.eyes_hold = v;
        else if (k == "idle_look") t.idle_look = v;
        else if (k == "lock_speed") t.foot_lock_max_speed = v;
        else log::warnf("[body] VIBE_ANIM_TUNE: unknown name '%s'", k.c_str());
    }
}

void VulkanEngine::destroy_player_body() {
    destroy_body_rt();
    destroy_people();
    destroy_skin_blur();
    for (SkinMesh* set : { body_lod_, head_lod_ })
        for (int i = 0; i < kBodyLods; ++i) {
            destroy_mesh(allocator_, set[i].rest);
            destroy_mesh(allocator_, set[i].skin);
            set[i] = SkinMesh{};
        }
    body_animator_.reset();
    body_bones_.clear();
}

// What a foot at world (x, z) stands on, for a character whose feet are at feet_y: the terrain,
// or the top of a level box (floor, stair, bridge) under that point that the character could be
// standing on. Level boxes are known near the local player only (the collision broadphase).
float VulkanEngine::anim_ground_height(float x, float z, float feet_y) const {
    float g = sample_terrain_height(x, z);
    for (const collision::AABB& a : near_static_) {
        if (x < a.min.x || x > a.max.x || z < a.min.z || z > a.max.z) continue;
        if (a.max.y > feet_y + 0.6f) continue;               // above the knees: a wall, not a floor
        g = std::max(g, a.max.y);
    }
    // (a boulder's round top: the walker stands on the stone, not on its box - tri_collision.h)
    for (const collision::AABB& a : near_round_) {
        if (x < a.min.x || x > a.max.x || z < a.min.z || z > a.max.z) continue;
        const std::vector<tricol::Round> one{ tricol::round_in_box(a.min, a.max) };
        float y = 0.0f; glm::vec3 n;
        if (tricol::floor_below_rounds(one, x, z, feet_y + 0.6f, g, 0.5f, y, n)) g = std::max(g, y);
    }
    return g;
}

// One character, one frame: the animator is told what the controller (or the network) says,
// and its pose goes into the bone palette (`palette` gets its first row; nullptr: pose only).
// Leaves the pose in body_R_ / body_P_ (model space) for whoever needs it next (a ragdoll
// taking over).
void VulkanEngine::pose_character(anim::CharacterAnimator& a, anim::CharacterAnimationInput& in, int* palette, std::vector<glm::vec4>* keep) {
    PoseJob job{ &a, &in };
    pose_update_all(&job, 1);
    pose_finish(a, in, palette, keep);
}

void VulkanEngine::pose_update_all(PoseJob* jobs, int count) {
    if (count <= 0) return;
    static const auto ground_cb = [](void* user, float x, float z) -> float {
        const GroundProbe* p = static_cast<const GroundProbe*>(user);
        return p->engine->anim_ground_height(x, z, p->feet_y);
    };
    GroundProbe probes[1 + net::kMaxPlayers];
    count = std::min(count, int(std::size(probes)));
    for (int i = 0; i < count; ++i) {
        anim::CharacterAnimationInput& in = *jobs[i].in;
        probes[i] = GroundProbe{ this, in.position.y };
        // (level floors are known within the broadphase's reach of the local player; further out a
        // character's feet stay level rather than follow terrain that may lie under a floor)
        const glm::vec2 dc(in.position.x - near_static_center_.x, in.position.z - near_static_center_.y);
        if (glm::dot(dc, dc) < 20.0f * 20.0f) { in.ground_height = ground_cb; in.ground_user = &probes[i]; }
        in.gravity = game_.gravity;
        in.stand_eye_height = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
        ++anim_lod_count_[glm::clamp(in.lod, 0, 3)];
    }
    const double t0 = now_ms();
    static const bool kSerial = std::getenv("VIBE_ANIM_SERIAL") != nullptr;       // A/B: one after the other
    if (count == 1 || kSerial) {
        for (int i = 0; i < count; ++i) jobs[i].animator->update(*jobs[i].in);
    } else {
        jobs::parallel_for(count, 1, [&](int begin, int end) {
            for (int i = begin; i < end; ++i) jobs[i].animator->update(*jobs[i].in);
        });
    }
    anim_ms_ += float(now_ms() - t0);          // (wall time: what the frame waits for)
}

void VulkanEngine::pose_finish(anim::CharacterAnimator& a, const anim::CharacterAnimationInput& in, int* palette, std::vector<glm::vec4>* keep) {
    if (in.lod >= 3) return;
    const double t1 = now_ms();
    const std::span<const glm::mat4> m = a.model();
    const size_t nb = std::min(body_bones_.size(), m.size());
    for (size_t i = 0; i < nb; ++i) {
        body_R_[i] = glm::mat3(m[i]);
        body_P_[i] = glm::vec3(m[i][3]);
    }
    if (palette) {
        *palette = body_push_pose(keep);
        skin_ms_ += float(now_ms() - t1);
    }
}

// The local player's body for this frame. Called once the frame's camera is known (the body
// rides on the interpolated camera, like the flashlight).
void VulkanEngine::update_player_body(float dt) {
    body_drawn_ = false;
    anim_ms_ = skin_ms_ = 0.0f;
    body_palette_used_ = 0;                // (first of the frame's three: this, mp_pose_avatars, corpses_update)
    body_palette_prev_ = body_palette_;
    body_palette_ = -1;
    for (int& c : anim_lod_count_) c = 0;
    // The cosmetic reload runs out by itself; firing ends it at once (combat.cpp).
    if (body_reload_left_ > 0.0f) body_reload_left_ = std::max(0.0f, body_reload_left_ - glm::clamp(dt, 0.0f, 0.1f));
    if (!gun_in_hand()) body_reload_left_ = 0.0f;
    if (body_bones_.empty() || !body_animator_ || !game_.player_body) return;
    CPU_SCOPE("player body");
    dt = glm::clamp(dt, 0.0f, 0.05f);

    // ---- where the body stands ----
    const float yaw = player_.yaw, pitch = player_.pitch;
    const glm::vec3 fwd(std::sin(yaw), 0.0f, -std::cos(yaw)), right(std::cos(yaw), 0.0f, std::sin(yaw));
    const float eye_h = game::Player::kHalfExtents.y + player_.eye_offset_y();
    const float crouch = glm::clamp(player_.crouch_factor, 0.0f, 1.0f);
    const float look_down = glm::smoothstep(0.15f, 1.25f, -pitch);
    // The eye sits in front of the neck; further when looking down, so the view is never inside the chest.
    static const float kBack0 = env_float("VIBE_BODY_BACK", 0.11f);
    // (looking down, the bottom of the view looks back past straight down: the body stays behind that)
    static const float kBackDown = env_float("VIBE_BODY_BACK_DOWN", 0.30f);
    const float back = kBack0 + kBackDown * look_down + 0.25f * crouch;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    const glm::vec3 feet = eye - glm::vec3(0.0f, eye_h, 0.0f) - fwd * back;
    // Model space: +X right, +Y up, -Z forward; the eye is at (0, eye_h, -back).
    body_model_ = glm::mat4(glm::vec4(right, 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f), glm::vec4(-fwd, 0.0f), glm::vec4(feet, 1.0f));
    auto to_model = [&](glm::vec3 world) {
        const glm::vec3 d = world - feet;
        return glm::vec3(glm::dot(d, right), d.y, -glm::dot(d, fwd));
    };
    auto dir_to_model = [&](glm::vec3 d) { return glm::vec3(glm::dot(d, right), d.y, -glm::dot(d, fwd)); };

    // ---- what the controller says ----
    // Test hooks: VIBE_BODY_WALK=<m/s> moves the body as if walking forward at that speed,
    // VIBE_BODY_STRAFE=<m/s> adds a sideways speed (to the right), VIBE_BODY_AIR=<m/s> holds it
    // in the air rising at that speed, VIBE_BODY_FIRE=<shots/s> fires, VIBE_BODY_RELOAD=1 reloads
    // in a loop.
    // (VIBE_BODY_WALKBACK=<m/s>: backward at that speed)
    static const float kWalkBack = env_float("VIBE_BODY_WALKBACK", 0.0f);
    static const float kWalk = kWalkBack > 0.0f ? 0.0f : env_float("VIBE_BODY_WALK", -1.0f), kStrafe = env_float("VIBE_BODY_STRAFE", 0.0f);
    static const float kAir = env_float("VIBE_BODY_AIR", 1e9f), kFire = env_float("VIBE_BODY_FIRE", 0.0f);
    static const bool kReload = std::getenv("VIBE_BODY_RELOAD") != nullptr;
    const bool hooked = kWalk >= 0.0f || kAir < 1e8f;
    anim::CharacterAnimationInput in;
    in.dt = dt;
    in.position = feet;
    in.velocity = player_.velocity;
    in.facing_yaw = yaw;
    in.aim_pitch = pitch;
    in.grounded = player_.on_ground && !player_swimming_;
    in.swimming = player_swimming_;
    in.wants_jump = last_pin_.jump;
    in.crouch = crouch;
    in.eye_local = glm::vec3(0.0f, eye_h, -back);
    in.first_person = std::getenv("VIBE_BODY_FRONT") == nullptr;      // (looked at from outside by the test hook: as others see it)
    {
        glm::vec3 wish(0.0f);
        if (last_pin_.fwd) wish += fwd;
        if (last_pin_.back) wish -= fwd;
        if (last_pin_.right) wish += right;
        if (last_pin_.left) wish -= right;
        if (glm::dot(wish, wish) > 1e-6f && state_ == State::Playing && !fly_mode_) {
            const float mult = (last_pin_.sprint ? 1.5f : last_pin_.crawl ? 0.4f : 1.0f) * (1.0f - 0.55f * glm::clamp(last_pin_.wade, 0.0f, 1.0f));
            in.desired_velocity = glm::normalize(wish) * game::kPlayerMove.max_walk_speed * mult;
        }
    }
    if (hooked) {
        in.velocity = (kWalk >= 0.0f ? fwd * (kWalk - kWalkBack) + right * kStrafe : glm::vec3(0.0f)) + glm::vec3(0.0f, kAir < 1e8f ? kAir : 0.0f, 0.0f);
        in.desired_velocity = glm::vec3(in.velocity.x, 0.0f, in.velocity.z);
        in.grounded = kAir > 1e8f;
        in.swimming = false;
    }
    if (kFire > 0.0f) body_shot_seq_ = uint32_t(glfwGetTime() * double(kFire));
    if (kReload && body_reload_left_ <= 0.0f && std::fmod(glfwGetTime(), 3.0) < 0.1) { ++body_reload_seq_; body_reload_left_ = body_animator_->weapon.reload_time; }
    in.held = flashlight_on_ ? anim::HeldItem::Torch : (weapon_ == 1 ? anim::HeldItem::Tablet : anim::HeldItem::Rifle);
    if (flashlight_on_) {
        in.torch_pos = to_model(flash_pos_);
        in.torch_fwd = dir_to_model(flash_fwd_);
        in.torch_right = dir_to_model(flash_right_);
    }
    in.shot_seq = body_shot_seq_;
    in.reload_seq = body_reload_seq_;
    in.aiming = body_aim_;
    // VIBE_BODY_CARRY=1 (with VIBE_BODY_FRONT): the rifle carried low, as somebody else running with it is seen (net_anim.h)
    if (static const bool kCarry = std::getenv("VIBE_BODY_CARRY") != nullptr; kCarry) { in.carry = 1.0f; in.aiming = false; }
    {   // (VIBE_TEST_RAGDOLL: the 23 frames before it the body buckles, as a killed player's does)
        static const int kRagAt = int(env_float("VIBE_TEST_RAGDOLL", -1.0f));
        const int to_go = kRagAt - int(frame_number_);
        if (kRagAt >= 0 && to_go >= 0 && to_go < 23) { in.dying = 1.0f - float(to_go) / 23.0f; in.hit_dir = dir_to_model(fwd); in.first_person = false; }
    }
    in.hit_seq = body_hit_seq_;
    in.hit_dir = dir_to_model(body_hit_dir_);
    in.reloading = body_reload_left_ > 0.0f;
    // A respawn or a jump to another place: nothing remembered in world space survives it.
    static glm::vec3 last_feet(1e9f);
    in.teleported = glm::length(feet - last_feet) > 6.0f + 2.0f * glm::length(player_.velocity) * dt;
    last_feet = feet;

    // Aiming brings the held gun's sight onto the crosshair: each model's sight sits elsewhere.
    // (VIBE_ADS="x,y,z": where the aimed weapon goes in the player's own view, metres from its usual place)
    body_animator_->weapon.ads_shift_fp = weapon_ == 2 && !rifle_gltf_.empty() ? kRifleAdsShift : anim::default_rifle().ads_shift_fp;
    if (static const char* e = std::getenv("VIBE_ADS"); e) {
        glm::vec3 v = body_animator_->weapon.ads_shift_fp;
        if (std::sscanf(e, "%f,%f,%f", &v.x, &v.y, &v.z) == 3) body_animator_->weapon.ads_shift_fp = v;
    }
    // The support hand goes under THIS gun's fore-end (measured from the model: props.cpp).
    {
        const glm::vec4 fore = gun_fore_[weapon_ == 2 && !rifle_gltf_.empty() ? 1 : 0];
        body_animator_->weapon.grip_l = fore.w > 0.5f ? glm::vec3(fore) : anim::default_rifle().grip_l;
    }
    // (VIBE_READY="x,y,z": where the weapon is held when NOT aiming, in the player's own view - right, up, towards the eye)
    if (static const char* e = std::getenv("VIBE_READY"); e) {
        glm::vec3 v = body_animator_->weapon.ready_shift_fp;
        if (std::sscanf(e, "%f,%f,%f", &v.x, &v.y, &v.z) == 3) body_animator_->weapon.ready_shift_fp = v;
    }
    // At the wheel: the body sits in the car's seat, its hands on the steering wheel (jeep.cpp).
    body_seated_ = false;
    anim::CharacterAnimationInput::SeatExtras seat_more;
    if (glm::mat4 hip; jeep_driver(hip, in.seat_hand_l, in.seat_hand_r, in.seat_legs, &seat_more)) {
        const anim::IAnimationRuntime& art = *anim_set_.runtime;
        const int hj = art.rig()[anim::Role::Hips];
        const glm::vec3 rest = hj >= 0 ? art.skeleton().rest_pos[size_t(hj)] : glm::vec3(0.0f, 0.95f, 0.0f);
        body_model_ = hip * glm::translate(glm::mat4(1.0f), -rest);
        const glm::mat4 inv = glm::inverse(body_model_);
        in.take_seat(seat_more, inv);                    // (the wheel's frame, the feet's places, the backrest: this car's own)
        in.seat_hand_l = glm::vec3(inv * glm::vec4(in.seat_hand_l, 1.0f));
        in.seat_hand_r = glm::vec3(inv * glm::vec4(in.seat_hand_r, 1.0f));
        in.seated = true;
        in.first_person = false;
        in.teleported = false;
        in.dying = 0.0f;
        body_seated_ = true;
    }
    pose_character(*body_animator_, in, &body_palette_);
    body_weapon_offset_ = body_animator_->weapon_offset();

    body_fire_offset_ = body_animator_->weapon_fire_offset();

    // Test hooks: VIBE_BODY_FRONT=<m> stands the body that far in front of the camera (turned
    // by VIBE_BODY_TURN degrees) to look at the pose from outside; VIBE_BODY_REST=1 shows the
    // rest pose; VIBE_BODY_DEBUG=1 logs what the animator is doing once a second.
    static const float kFront = env_float("VIBE_BODY_FRONT", 0.0f), kTurn = glm::radians(env_float("VIBE_BODY_TURN", 180.0f));
    static const bool kRest = std::getenv("VIBE_BODY_REST") != nullptr, kDebug = std::getenv("VIBE_BODY_DEBUG") != nullptr;
    body_front_ = kFront > 0.0f;
    if (kFront > 0.0f) {
        const glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(eye.x, eye.y - eye_h, eye.z) + fwd * kFront);
        body_model_ = T * glm::rotate(glm::mat4(1.0f), kTurn - yaw, glm::vec3(0, 1, 0));
    }
    if (kRest) {
        for (size_t i = 0; i < body_bones_.size(); ++i) { body_R_[i] = glm::mat3(1.0f); body_P_[i] = body_bones_[i].head; }
        body_palette_ = body_push_pose();
    }
    if (kDebug && frame_number_ % 60 == 0) {
        const anim::AnimDebug& d = body_animator_->debug();
        log::infof("[body] %s clip %s t %.2f%s  speed %.2f (anim %.2f) rate %.2f stride %.2f warp %.2f  air %s ttc %.2f  contact %d%d lock %d%d  "
                   "cost %.2f (cur %.2f)%s  searches %u switches %u  invalid %u  %.3f ms (search %.3f)",
                   d.mode, d.clip >= 0 ? anim_set_.runtime->clip(d.clip).name.c_str() : "-", d.clip_time, d.mirrored ? " mirrored" : "",
                   d.speed, d.anim_speed, d.rate, d.stride, d.warp, anim::air_phase_name(d.air_phase), std::min(d.time_to_contact, 99.0f),
                   int(d.contact[0]), int(d.contact[1]), int(d.locked[0]), int(d.locked[1]), d.match.cost, d.current_cost,
                   d.fallback ? " fallback" : "", d.searches, d.switches, d.invalid_poses, d.update_ms, d.search_ms);
    }
    {
        static uint32_t reported = 0;
        const uint32_t bad = body_animator_->debug().invalid_poses;
        if (bad != reported) { log::warnf("[body] the animator produced %u pose(s) that were not finite; the last good pose was shown", bad); reported = bad; }
    }
    // Test hook: VIBE_TEST_RAGDOLL=<frame>: at that frame the body as posed here is shot (from
    // the camera, at chest height unless VIBE_TEST_RAGDOLL_Y=<m above the feet>) and falls.
    {
        static const int kRag = int(env_float("VIBE_TEST_RAGDOLL", -1.0f));
        if (kRag >= 0 && int(frame_number_) == kRag) {
            glm::mat4 bm = body_model_;
            bm[3].y = sample_terrain_height(bm[3].x, bm[3].z);          // (standing on the ground there)
            const glm::vec3 base = glm::vec3(bm[3]);
            const glm::vec3 hp = base + glm::vec3(0.0f, env_float("VIBE_TEST_RAGDOLL_Y", 1.25f), 0.0f);
            const glm::mat4 vi = bm * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, eye_h, -back));
            corpse_spawn(bm, vi, in.velocity, hp, glm::normalize(hp - eye), env_float("VIBE_TEST_RAGDOLL_POWER", 1.0f));
        }
    }
    body_drawn_ = body_palette_ >= 0;
}

// What the animation is doing, drawn over the game: the chosen clip and frame, the match's cost
// by feature group, the predicted path, foot contacts and locks, IK targets, phases and timings
// for the local player; levels of detail and cost for everybody. (VIBE_ANIM_DEBUG=1, or the menu.)
void VulkanEngine::anim_debug_overlay() {
    if (!anim_debug_ || !body_animator_) return;
    const anim::AnimDebug& d = body_animator_->debug();
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const ImVec2 disp = ImGui::GetIO().DisplaySize;
    char line[256];
    float y = disp.y * 0.30f;
    auto text = [&](ImU32 col) {
        dl->AddText(ImVec2(19.0f, y + 1.0f), IM_COL32(0, 0, 0, 220), line);
        dl->AddText(ImVec2(18.0f, y), col, line);
        y += ImGui::GetTextLineHeight() + 2.0f;
    };
    const ImU32 white = IM_COL32(235, 235, 235, 255), dim = IM_COL32(170, 200, 255, 255), warn = IM_COL32(255, 170, 90, 255);
    std::snprintf(line, sizeof(line), "ANIM  %s  |  %s  t %.2f s%s", d.mode, d.clip >= 0 ? anim_set_.runtime->clip(d.clip).name.c_str() : "-",
                  d.clip_time, d.mirrored ? "  (mirrored)" : "");
    text(white);
    std::snprintf(line, sizeof(line), "speed %.2f m/s  root velocity of the clip %.2f  ->  rate x%.2f  stride x%.2f  leg turn %.0f deg  lean %.1f / %.1f deg",
                  d.speed, d.anim_speed, d.rate, d.stride, glm::degrees(d.warp), glm::degrees(d.lean_roll), glm::degrees(d.lean_pitch));
    text(dim);
    std::snprintf(line, sizeof(line), "match cost %.2f (staying: %.2f)%s   feet %.2f  foot vel %.2f  hips %.2f  path %.2f  facing %.2f",
                  d.match.cost, d.current_cost, d.fallback ? "  FALLBACK: path only" : "", d.match.group_cost[0], d.match.group_cost[1],
                  d.match.group_cost[2], d.match.group_cost[3], d.match.group_cost[4]);
    text(d.fallback ? warn : dim);
    std::snprintf(line, sizeof(line), "searches %u  switches %u  inertial offset %.2f   air: %s  contact in %.2f s  impact %.1f m/s  flights %u",
                  d.searches, d.switches, d.inertial_energy, anim::air_phase_name(d.air_phase), std::min(d.time_to_contact, 99.0f), d.impact, d.flights);
    text(dim);
    std::snprintf(line, sizeof(line), "feet: contact L%d R%d  locked L%d R%d  pelvis %.3f m   weapon: recoil %.2f  reload %.2f  equip %.2f",
                  int(d.contact[0]), int(d.contact[1]), int(d.locked[0]), int(d.locked[1]), d.pelvis_offset, d.recoil, d.reload, d.equip);
    text(dim);
    std::snprintf(line, sizeof(line), "this frame: animation %.3f ms (local %.3f, search %.3f)  skinning %.3f ms   characters by level 0/1/2/3: %d / %d / %d / %d",
                  anim_ms_, d.update_ms, d.search_ms, skin_ms_, anim_lod_count_[0], anim_lod_count_[1], anim_lod_count_[2], anim_lod_count_[3]);
    text(white);
    if (d.invalid_poses) { std::snprintf(line, sizeof(line), "INVALID POSES: %u", d.invalid_poses); text(warn); }
    if (net_) {
        const net::Session::Stats& ns = net_->stats();
        std::snprintf(line, sizeof(line), "net: sent %.1f kB  received %.1f kB  duplicate events dropped %u  states dropped %u",
                      double(ns.tx_bytes) / 1024.0, double(ns.rx_bytes) / 1024.0, ns.duplicate_events, ns.dropped_states);
        text(dim);
    }

    // In the world: the predicted path, foot targets (green = planted and held), hand targets, the muzzle line.
    const glm::mat4 vp = current_frame_view_.vp;
    auto project = [&](glm::vec3 w, ImVec2& out) {
        const glm::vec4 c = vp * glm::vec4(w, 1.0f);
        if (c.w < 0.05f) return false;
        out = ImVec2((c.x / c.w * 0.5f + 0.5f) * disp.x, (c.y / c.w * 0.5f + 0.5f) * disp.y);
        return true;
    };
    auto model_pt = [&](glm::vec3 m) { return glm::vec3(body_model_ * glm::vec4(m, 1.0f)); };
    ImVec2 a, b;
    glm::vec3 prev(0.0f, 0.02f, 0.0f);
    for (int h = 0; h < anim::MotionSchema::kHorizons; ++h) {
        const glm::vec3 p(d.traj.pos[h].x, 0.02f, d.traj.pos[h].y);
        if (project(model_pt(prev), a) && project(model_pt(p), b)) {
            dl->AddLine(a, b, IM_COL32(90, 200, 255, 255), 2.0f);
            dl->AddCircleFilled(b, 4.0f, IM_COL32(90, 200, 255, 255));
            ImVec2 e;
            if (project(model_pt(p + glm::vec3(d.traj.dir[h].x, 0.0f, d.traj.dir[h].y) * 0.25f), e)) dl->AddLine(b, e, IM_COL32(255, 230, 90, 255), 1.5f);
        }
        prev = p;
    }
    for (int s = 0; s < 2; ++s) {
        if (project(model_pt(d.foot_target[s]), a))
            dl->AddCircle(a, 7.0f, d.locked[s] ? IM_COL32(80, 255, 120, 255) : (d.contact[s] ? IM_COL32(255, 220, 80, 255) : IM_COL32(255, 90, 90, 255)), 12, 2.0f);
        if (project(model_pt(d.hand_target[s]), a)) dl->AddRect(ImVec2(a.x - 4.0f, a.y - 4.0f), ImVec2(a.x + 4.0f, a.y + 4.0f), IM_COL32(255, 140, 255, 255));
    }
    const glm::vec3 mz = body_animator_->muzzle_position();
    if (project(model_pt(mz), a) && project(model_pt(mz + body_animator_->muzzle_direction() * 1.5f), b)) dl->AddLine(a, b, IM_COL32(255, 80, 60, 200), 1.5f);
    // Other players: their level of detail and clip over their heads.
    for (const auto& av : mp_->avatars) {
        if (!av.id || !av.animator || !av.drawn) continue;
        const anim::AnimDebug& ad = av.animator->debug();
        if (!project(av.eye + glm::vec3(0.0f, 0.55f, 0.0f), a)) continue;
        std::snprintf(line, sizeof(line), "L%d %s %s %.2f ms", av.lod, ad.mode, ad.clip >= 0 ? anim_set_.runtime->clip(ad.clip).name.c_str() : "-", ad.update_ms);
        dl->AddText(ImVec2(a.x - ImGui::CalcTextSize(line).x * 0.5f, a.y), IM_COL32(170, 255, 200, 255), line);
    }
}

// The pose in body_R_ / body_P_ as this frame's next palette entry: per bone the 3x4 matrix that
// takes a rest-pose vertex to its posed place (model space). Returns its first row for
// pc.grass_params.y, or -1 when the palette is full (the character is then not drawn).
int VulkanEngine::body_push_pose(std::vector<glm::vec4>* keep) {
    const size_t nb = std::min<size_t>(body_bones_.size(), kSkinBones);
    body_rows_.resize(nb * 3);
    for (size_t i = 0; i < nb; ++i) {
        const glm::mat3& R = body_R_[i];
        const glm::vec3 T = body_P_[i] - R * body_bones_[i].head;
        body_rows_[i * 3 + 0] = glm::vec4(R[0][0], R[1][0], R[2][0], T.x);
        body_rows_[i * 3 + 1] = glm::vec4(R[0][1], R[1][1], R[2][1], T.y);
        body_rows_[i * 3 + 2] = glm::vec4(R[0][2], R[1][2], R[2][2], T.z);
    }
    if (keep) *keep = body_rows_;
    return body_push_rows(body_rows_);
}

int VulkanEngine::body_push_rows(const std::vector<glm::vec4>& rows) {
    if (!body_palette_map_ || rows.empty() || body_palette_used_ >= kSkinCharacters) return -1;
    const uint32_t slot = frame_number_ % kPaletteSlots;
    const uint32_t first = (slot * kSkinCharacters + body_palette_used_++) * kSkinBones * 3;
    std::memcpy(body_palette_map_ + first, rows.data(), std::min<size_t>(rows.size(), kSkinBones * 3) * sizeof(glm::vec4));
    return int(first);
}

// Which of the soldier's meshes a character at `pos` (its eye) is drawn with: by its height on
// screen, with a margin against flipping back and forth at a boundary.
int VulkanEngine::body_mesh_lod(glm::vec3 pos, int previous) const {
    const float tan_half = current_frame_view_.proj[1][1] != 0.0f ? 1.0f / std::abs(current_frame_view_.proj[1][1]) : 0.7f;
    const float dist = std::max(glm::length(pos - current_frame_view_.eye_pos), 0.1f);
    const float px = 1.8f / (dist * tan_half) * 0.5f * float(swapchain_extent_.height);
    // full above 300 px, the middle one above 100
    const float up = 1.12f, down = 0.89f;
    int lod = px > 300.0f ? 0 : px > 100.0f ? 1 : 2;
    if (lod < previous && px < (previous == 1 ? 300.0f : 100.0f) * up) lod = previous;       // not yet clearly bigger
    if (lod > previous && px > (previous == 0 ? 300.0f : 100.0f) * down) lod = previous;     // not yet clearly smaller
    return lod;
}

void VulkanEngine::draw_player_body(
VkCommandBuffer cmd, const glm::mat4& vp, const glm::vec3& cam) {
    if (!mp_->draw && !body_drawn_) return;
    const int palette = mp_->draw ? mp_->draw->palette : body_palette_;
    const int palette_prev = mp_->draw ? mp_->draw->palette_prev : body_palette_prev_;
    if (palette < 0 || !skin_pipeline()) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skin_pipeline());
    // Which mesh: the player's own body is always the full one; others by their size on screen.
    static const int kForceMesh = std::getenv("VIBE_MESH_LOD") ? std::atoi(std::getenv("VIBE_MESH_LOD")) : -1;
    int lod = kForceMesh >= 0 ? kForceMesh : (mp_->draw ? mp_->draw->mesh_lod : 0);
    lod = glm::clamp(lod, 0, kBodyLods - 1);
    auto level = [&](const SkinMesh* set) -> const SkinMesh* {
        for (int i = lod; i >= 0; --i) if (set[i].index_count) return &set[i];
        return nullptr;
    };
    const SkinMesh* body = level(body_lod_);
    // The head: on everybody but the player himself (who looks out of it).
    const SkinMesh* head = (mp_->draw || body_front_ || body_seated_) ? level(head_lod_) : nullptr;      // (and on the driver, seen from behind the car)
    // PEOPLE (below): somebody of the NPC world is somebody - an outfit, a face, hair, headgear.
    const bool person = mp_->draw && mp_->draw->person != 0 && people_ && people_->ok;
    const PersonLook look = person ? people_look(mp_->draw->person, mp_->draw->dress) : people_soldier_look();
    const SkinMesh* extra[2] = { nullptr, nullptr };
    if (person) {
        if (look.outfit > 0) if (const SkinMesh* b = level(people_->bodies[look.outfit])) body = b;
        if (const SkinMesh* h = level(people_->heads[look.head])) head = h;
        if (look.hair) extra[0] = level(people_->hair[look.head]);
        if (look.gear == 1) extra[1] = level(people_->helmet);
        else if (look.gear == 2) extra[1] = level(people_->cap);
    }
    if (!body) return;
    // the look block: six rows after the bones in this character's region (shaders/people.glsl)
    if (body_palette_map_) std::memcpy(body_palette_map_ + size_t(palette) + kLookRow, look.row, sizeof(look.row));
    auto draw_mesh = [&](const SkinMesh& m) {
        const VkBuffer streams[2] = { m.rest.vertex_buffer, m.skin.vertex_buffer };
        const VkDeviceSize offs[2] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, streams, offs);
        vkCmdBindIndexBuffer(cmd, m.rest.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, m.index_count, 1, 0, 0, 0);
    };
    PushConstants pc{};
    // (somebody taller or broader than the soldier: the whole body, from the feet)
    const glm::mat4 build = glm::scale(glm::mat4(1.0f), glm::vec3(look.width, look.height, look.width));
    const glm::mat4 bm = (mp_->draw ? mp_->draw->model : body_model_) * build;
    // `vp` is relative to `cam` (draw_viewmodel): so is the matrix it multiplies; the lighting keeps the world one.
    glm::mat4 bm_rel = bm;
    bm_rel[3] = glm::vec4(glm::vec3(bm[3]) - cam, 1.0f);
    pc.mvp = vp * bm_rel;
    pc.model = bm;
    pc.prev_mvp = pc.mvp;                  // rides on the camera: no motion vectors
    if (mp_->draw) {
        // Another player: a real object in the world, with real motion.
        if (prev_view_proj_valid_) pc.prev_mvp = prev_view_proj_ * mp_->draw->prev_model * build;
    } else if (rt_.fsr2_enabled) {
        glm::mat4 J(1.0f);
        const glm::vec2 dj = current_frame_view_.jitter - prev_jitter_;
        J[3][0] = dj.x;
        J[3][1] = dj.y;
        pc.prev_mvp = J * pc.mvp;
    }
    pc.color = glm::vec4(1.0f);
    pc.emissive = glm::vec4(0.0f);
    // albedo = the soldier atlas, sampled with the mesh UVs (marker -8, cube.frag)
    // (.z: this character's first palette row - the fragment finds its look block there)
    pc.tex_params = glm::vec4(float(kTexSoldier), -8.0f, float(palette), 0.0f);
    // .y: first palette row; .z: 1 + last frame's (cube_skin.vert: the limbs' own motion)
    pc.grass_params = glm::vec4(0.0f, float(palette), palette_prev >= 0 ? float(palette_prev + 1) : 0.0f, 0.0f);
    vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
    draw_mesh(*body);
    if (head) draw_mesh(*head);
    for (const SkinMesh* x : extra) if (x) draw_mesh(*x);
    // Test hook: VIBE_BODY_CROWD=<n> draws the local body n more times in a block ahead
    // (the same pose): what n skinned characters cost the GPU (gpu-passes: w_viewmodel).
    static const int kCrowd = std::getenv("VIBE_BODY_CROWD") ? std::atoi(std::getenv("VIBE_BODY_CROWD")) : 0;
    if (kCrowd > 0 && !mp_->draw) {
        const int side = int(std::ceil(std::sqrt(float(kCrowd))));
        for (int i = 0; i < kCrowd; ++i) {
            const glm::vec3 off = glm::vec3(bm * glm::vec4(float(i % side - side / 2) * 1.2f, 0.0f, -2.0f - float(i / side) * 1.2f, 0.0f));
            glm::mat4 m = bm, mr = bm_rel;
            m[3] += glm::vec4(off, 0.0f);
            mr[3] += glm::vec4(off, 0.0f);
            pc.mvp = vp * mr;
            pc.prev_mvp = pc.mvp;
            pc.model = m;
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
            draw_mesh(*body);
            if (head) draw_mesh(*head);
        }
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());      // (the caller goes on with rigid parts)
}

// =====================================================================================
// PEOPLE. Everybody the NPC engine shows is somebody: an outfit, a face, a skin, hair, a beard,
// eyes, colours - all from the person's number (the same on every machine) and how the NPC
// engine has them dressed (npc::Dress). The meshes share the soldier's skeleton and atlas
// (tools/character/make_people.py); the colours are a block of six rows after the bones in the
// character's region of the bone palette (shaders/people.glsl). docs/ARCHITECTURE.md
// "People who look like people".
// =====================================================================================
namespace {

struct Rnd {
    uint64_t s;
    explicit Rnd(uint32_t seed) : s(uint64_t(seed) * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}
    uint32_t next() { s += 0x9E3779B97F4A7C15ull; uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return uint32_t((z ^ (z >> 31)) >> 32); }
    float uni() { return float(next() >> 8) * (1.0f / 16777216.0f); }
    float range(float a, float b) { return a + (b - a) * uni(); }
    int   pick(int n) { return int(next() % uint32_t(n)); }
    bool  chance(float p) { return uni() < p; }
};

// The faces (make_people.py FACES; 0 is the soldier's own) and how much melanin their origin has
// as a range: the skin is chosen in it, the hair and the eyes follow from it.
struct FaceDef { float m0, m1; bool old_skin; bool cards; float age; };
constexpr FaceDef kFaces[kPeopleHeads] = {
    { 0.00f, 0.20f, false, false, 0.3f },    // 0 the soldier (fair, young)
    { 0.70f, 0.98f, false, false, 0.3f },    // 1 of African origin
    { 0.02f, 0.25f, false, true,  0.6f },    // 2 forties, heavier
    { 0.18f, 0.38f, false, true,  0.25f },   // 3 of East Asian origin
    { 0.00f, 0.22f, true,  false, 1.0f },    // 4 near sixty
    { 0.35f, 0.65f, false, true,  0.45f },   // 5 of mixed origin
    { 0.00f, 0.18f, false, false, 0.1f },    // 6 very young
    { 0.30f, 0.62f, false, true,  0.55f },   // 7 South Asian / Middle Eastern
};

glm::vec3 skin_tone(float m, Rnd& r) {
    // linear albedo along the range of real skin, from very fair to very dark, a little warmer
    // or cooler from one person to the next
    const glm::vec3 fair(0.80f, 0.60f, 0.49f), mid(0.45f, 0.28f, 0.17f), dark(0.12f, 0.065f, 0.042f);
    glm::vec3 c = m < 0.5f ? glm::mix(fair, mid, m * 2.0f) : glm::mix(mid, dark, (m - 0.5f) * 2.0f);
    const float warm = r.range(-0.04f, 0.05f);
    c *= glm::vec3(1.0f + warm, 1.0f, 1.0f - warm * 1.5f);
    return c;
}

}  // namespace

PersonLook people_soldier_look() {
    PersonLook L;
    L.row[0] = glm::vec4(0.8769f, 0.6614f, 0.5376f, 0.0f);     // the young skin texture's own mean (make_people.py log)
    L.row[1] = glm::vec4(0.035f, 0.022f, 0.014f, 0.5f);        // dark brown, cropped
    L.row[2] = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    L.row[3] = glm::vec4(0.11f, 0.12f, 0.06f, 0.0f);
    L.row[4] = glm::vec4(0.06f, 0.06f, 0.05f, 0.25f);
    L.row[5] = glm::vec4(0.10f, 0.16f, 0.26f, 0.0f);
    return L;
}

PersonLook people_look(uint32_t person, uint8_t dress) {
    Rnd r(person * 2654435761u ^ 0x5bd1e995u);
    PersonLook L;
    // ---- the face, the skin
    L.head = r.pick(kPeopleHeads);
    const FaceDef& f = kFaces[L.head];
    const float m = r.range(f.m0, f.m1);
    const glm::vec3 skin = skin_tone(m, r);
    const bool old = f.old_skin || (f.age > 0.5f && r.chance(0.4f));
    L.row[0] = glm::vec4(skin, old ? 1.0f : 0.0f);
    // ---- the hair: dark where the skin is dark; any natural colour on the fair; grey with age
    static const glm::vec3 kHair[6] = { { 0.012f, 0.010f, 0.009f }, { 0.035f, 0.022f, 0.014f }, { 0.09f, 0.055f, 0.03f },
                                        { 0.30f, 0.21f, 0.11f }, { 0.20f, 0.065f, 0.028f }, { 0.24f, 0.235f, 0.225f } };
    glm::vec3 hair = m > 0.45f ? kHair[r.chance(0.8f) ? 0 : 1] : kHair[r.pick(100) < 25 ? 1 : r.pick(100) < 40 ? 2 : r.pick(100) < 55 ? 0 : r.chance(0.7f) ? 3 : 4];
    if (f.age > 0.9f || (f.age > 0.5f && r.chance(0.3f))) hair = glm::mix(hair, kHair[5], f.age > 0.9f ? 0.85f : 0.45f);
    const npc::Dress d = npc::Dress(dress);
    const bool military = d == npc::Dress::Soldier || d == npc::Dress::Garrison || (d == npc::Dress::Technician && r.chance(0.3f));
    // scalp: soldiers shaved, buzzed or cropped; civilians fuller (or cards, below)
    float scalp = military ? (r.chance(0.25f) ? 0.0f : r.chance(0.5f) ? 0.5f : 0.85f) : r.range(0.85f, 1.0f);
    if (f.age > 0.9f && r.chance(0.5f)) scalp = 0.0f;            // (the older man is bald or close to it)
    L.row[1] = glm::vec4(hair, scalp);
    // ---- the beard: soldiers at most stubble; others now and then a beard or a moustache
    float beard = 0.0f, mous = 0.0f, stubble = r.range(0.0f, 0.7f);
    if (!military) {
        const int b = r.pick(100);
        if (b < 22) beard = 1.0f; else if (b < 34) mous = 1.0f;
    }
    // ---- the outfit and the colours
    static const glm::vec3 kTops[10] = { { 0.70f, 0.70f, 0.68f }, { 0.25f, 0.25f, 0.25f }, { 0.03f, 0.05f, 0.12f }, { 0.35f, 0.04f, 0.04f },
                                         { 0.08f, 0.15f, 0.06f }, { 0.02f, 0.02f, 0.02f }, { 0.30f, 0.45f, 0.60f }, { 0.45f, 0.37f, 0.25f },
                                         { 0.45f, 0.30f, 0.05f }, { 0.12f, 0.20f, 0.30f } };
    static const glm::vec3 kJeans[5] = { { 0.06f, 0.09f, 0.18f }, { 0.12f, 0.17f, 0.28f }, { 0.03f, 0.04f, 0.07f }, { 0.02f, 0.02f, 0.02f }, { 0.30f, 0.25f, 0.15f } };
    static const glm::vec3 kMil[3] = { { 0.11f, 0.12f, 0.06f }, { 0.25f, 0.19f, 0.12f }, { 0.03f, 0.03f, 0.03f } };
    static const glm::vec3 kWork[4] = { { 0.05f, 0.10f, 0.25f }, { 0.60f, 0.20f, 0.02f }, { 0.12f, 0.13f, 0.07f }, { 0.20f, 0.20f, 0.21f } };
    static const glm::vec3 kJacket[5] = { { 0.03f, 0.04f, 0.09f }, { 0.18f, 0.18f, 0.18f }, { 0.40f, 0.33f, 0.22f }, { 0.12f, 0.13f, 0.07f }, { 0.02f, 0.02f, 0.02f } };
    static const glm::vec3 kTrousers[4] = { { 0.18f, 0.18f, 0.19f }, { 0.30f, 0.26f, 0.17f }, { 0.04f, 0.05f, 0.10f }, { 0.02f, 0.02f, 0.02f } };
    glm::vec3 top = kTops[r.pick(10)], bottom = kJeans[r.pick(5)];
    switch (d) {
    case npc::Dress::Soldier: L.outfit = 0; L.gear = 1; break;
    case npc::Dress::Garrison: L.outfit = r.chance(0.55f) ? 1 : 2; L.gear = r.chance(0.45f) ? 2 : 0; top = kMil[r.pick(3)]; break;
    case npc::Dress::Technician: L.outfit = r.chance(0.7f) ? 4 : 1; L.gear = r.chance(0.3f) ? 2 : 0; bottom = kWork[r.pick(4)]; break;
    case npc::Dress::Office: { const int o = r.pick(4); L.outfit = o < 2 ? 5 : o == 2 ? 3 : 2; if (L.outfit == 2) top = kMil[r.pick(3)]; break; }
    default: L.outfit = r.chance(0.6f) ? 3 : 5; break;
    }
    if (L.outfit == 5) { top = kJacket[r.pick(5)]; bottom = kTrousers[r.pick(4)]; }
    // the unit's shade: woodland as painted, a little faded or darker from one issue to the next
    const float fade = r.range(-0.08f, 0.12f);
    L.row[2] = glm::vec4(glm::vec3(1.0f + fade, 1.0f + fade * 0.9f, 1.0f + fade * 0.6f), beard);
    L.row[3] = glm::vec4(top * r.range(0.85f, 1.15f), mous);
    L.row[4] = glm::vec4(bottom * r.range(0.85f, 1.15f), stubble);
    // ---- the eyes: brown where the skin is dark; blue, green, grey, hazel or brown on the fair
    static const glm::vec3 kEyes[5] = { { 0.10f, 0.05f, 0.025f }, { 0.10f, 0.17f, 0.32f }, { 0.10f, 0.16f, 0.07f }, { 0.18f, 0.20f, 0.22f }, { 0.16f, 0.11f, 0.04f } };
    const glm::vec3 eye = (m > 0.35f || r.chance(0.35f)) ? kEyes[0] : kEyes[r.pick(5)];
    L.row[5] = glm::vec4(eye, r.uni());
    // hair cards only where there is no headgear and the hair is not cropped
    L.hair = f.cards && L.gear == 0 && !military;
    L.height = r.range(0.965f, 1.035f);
    L.width = L.height * r.range(0.97f, 1.05f);
    return L;
}

void VulkanEngine::init_people() {
    destroy_people();
    people_ = new PeopleState();
    PeopleState& P = *people_;
    static const char* kOutfit[kPeopleOutfits] = { "", "fatigues", "tee", "civil", "work", "jacket" };
    static const char* kLod[kBodyLods] = { "", "_lod1", "_lod2" };
    int files = 0;
    size_t bytes = 0;
    auto load = [&](const std::string& name, SkinMesh& m) {
        if (!load_skin_mesh((name + ".qchar").c_str(), m, false)) return false;
        ++files; bytes += size_t(m.vertex_count) * 2 * sizeof(Vertex) + size_t(m.index_count) * 4;
        return true;
    };
    for (int o = 1; o < kPeopleOutfits; ++o)
        for (int l = 0; l < kBodyLods; ++l) load(std::string("people_body_") + kOutfit[o] + kLod[l], P.bodies[o][l]);
    for (int h = 0; h < kPeopleHeads; ++h)
        for (int l = 0; l < kBodyLods; ++l) {
            load("people_head_" + std::to_string(h) + kLod[l], P.heads[h][l]);
            if (l < 2) load("people_hair_" + std::to_string(h) + kLod[l], P.hair[h][l]);
        }
    for (int l = 0; l < 2; ++l) load(std::string("people_helmet") + kLod[l], P.helmet[l]);
    load("people_cap", P.cap[0]);
    P.ok = P.heads[0][0].index_count > 0;
    if (body_bones_.size() > size_t(kLookRow / 3)) { log::warnf("[people] %zu bones: no room for the look block - everybody is the soldier", body_bones_.size()); P.ok = false; }
    log::infof("[people] %d meshes (%.1f MB on the GPU)%s", files, double(bytes) / 1048576.0,
               P.ok ? "" : " - not all there: everybody is the soldier (blender -b soldier.blend --python tools/character/make_people.py)");
}

void VulkanEngine::destroy_people() {
    if (!people_) return;
    PeopleState& P = *people_;
    auto kill = [&](SkinMesh& m) { destroy_mesh(allocator_, m.rest); destroy_mesh(allocator_, m.skin); m = SkinMesh{}; };
    for (auto& o : P.bodies) for (SkinMesh& m : o) kill(m);
    for (auto& o : P.heads) for (SkinMesh& m : o) kill(m);
    for (auto& o : P.hair) for (SkinMesh& m : o) kill(m);
    for (SkinMesh& m : P.helmet) kill(m);
    for (SkinMesh& m : P.cap) kill(m);
    delete people_;
    people_ = nullptr;
}

}  // namespace vibe
