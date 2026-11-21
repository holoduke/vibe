// The soldier's body: the local player's own (seen from the eyes) and, through the same code,
// every other player's. A skinned mesh (built with MakeHuman / MPFB in Blender, tools/character/)
// posed by the animation library (src/anim, see docs/animation/architecture.md):
//
//   controller state  ->  anim::CharacterAnimator  ->  joint matrices  ->  CPU skinning  ->  draw
//
// This file only gathers what the animator is told (where the character is, how it moves, where
// it aims, what it holds), skins the result (about 43k vertices) into a per-frame vertex buffer
// and draws it with the ordinary mesh shader. The animator never touches the controller.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"

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

namespace qlike {

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

void VulkanEngine::init_player_body() {
    std::string path;
    for (const char* p : { "assets/character/soldier_fp.qchar", "../assets/character/soldier_fp.qchar" }) {
        std::ifstream t(p, std::ios::binary);
        if (t.good()) { path = p; break; }
    }
    if (path.empty()) {
        log::info("[body] no assets/character/soldier_fp.qchar - no visible player body");
        return;
    }
    std::ifstream f(path, std::ios::binary);
    char magic[4]{};
    uint32_t nb = 0, nv = 0, ni = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&nb), 4);
    f.read(reinterpret_cast<char*>(&nv), 4);
    f.read(reinterpret_cast<char*>(&ni), 4);
    if (std::memcmp(magic, "QCH1", 4) != 0 || nb == 0 || nb > 255 || nv == 0 || nv > 2000000 || ni > 12000000) {
        log::warn("[body] bad character file");
        return;
    }
    body_bones_.resize(nb);
    for (BodyBone& b : body_bones_) {
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
        log::warn("[body] character file truncated");
        body_bones_.clear();
        return;
    }

    // Scale so the model's eyes are at the player's standing eye height.
    float top = 0.0f;
    for (const FileVert& v : fv) top = std::max(top, v.p[1]);
    const int head = body_bone("Head");
    // (the head is not in the mesh: its height is the neck joint plus a head)
    const float model_eye = head >= 0 ? body_bones_[head].head.y + 0.035f : top + 0.13f;
    const float stand_eye = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
    body_scale_ = stand_eye / std::max(model_eye, 0.5f);
    for (BodyBone& b : body_bones_) { b.head *= body_scale_; b.tail *= body_scale_; }
    body_verts_.resize(nv);
    for (uint32_t i = 0; i < nv; ++i) {
        BodyVert& o = body_verts_[i];
        o.pos = glm::vec3(fv[i].p[0], fv[i].p[1], fv[i].p[2]) * body_scale_;
        o.nrm = glm::vec3(fv[i].n[0], fv[i].n[1], fv[i].n[2]);
        o.uv = glm::vec2(fv[i].uv[0], fv[i].uv[1]);
        for (int k = 0; k < 4; ++k) {
            o.bone[k] = std::min<uint8_t>(fv[i].b[k], uint8_t(nb - 1));
            o.weight[k] = fv[i].w[k] / 255.0f;
        }
    }
    body_index_count_ = ni;
    body_dbg_idx_ = idx;

    // Index buffer (static) through the ordinary mesh upload; its vertex
    // buffer holds the rest pose and is not drawn.
    {
        std::vector<Vertex> rest(nv);
        for (uint32_t i = 0; i < nv; ++i) rest[i] = { body_verts_[i].pos, body_verts_[i].nrm, body_verts_[i].uv };
        body_mesh_ = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                           rest.data(), nv, idx.data(), ni);
    }
    for (uint32_t i = 0; i < kFrameOverlap; ++i) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(Vertex) * nv;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &body_vbuf_[i], &body_valloc_[i], &ai), "body vertex buffer");
        body_vmapped_[i] = ai.pMappedData;
    }
    body_skinned_.resize(nv);
    body_R_.assign(nb, glm::mat3(1.0f));
    body_P_.assign(nb, glm::vec3(0.0f));
    log::infof("[body] soldier: %u bones, %u vertices, %u triangles, scale %.3f (model eye %.3f m)",
               nb, nv, ni / 3, body_scale_, model_eye);

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
    anim_debug_ = std::getenv("QLIKE_ANIM_DEBUG") != nullptr;
    log::infof("[body] animation: %s, %d clips (%.1f MB compressed), motion database %u frames (%.1f MB), %zu jumps",
               anim_set_.runtime->name(), anim_set_.runtime->num_clips(), double(anim_set_.runtime->clip_bytes()) / 1048576.0,
               anim_set_.db.frames(), double(anim_set_.db.memory_bytes()) / 1048576.0, anim_set_.db.jumps().size());
}

// The animation's style, without a rebuild: QLIKE_ANIM_TUNE="name=value,name=value". Names:
//   halflife (s, inertial blend)   run_speed (m/s the game's run is matched to in the capture)
//   max_rate  max_stride  max_warp (rad)  lean (0 = none)  foot_ik  foot_lock  stride_warp  orient_warp
//   hand_ik (0/1)   air (airborne pose strength)   air_sway   land_dip   reach_time (s)
//   search (s between searches)   lock_speed (m/s above which feet are not held)
void VulkanEngine::apply_anim_tuning(anim::AnimTuning& t) const {
    const char* s = std::getenv("QLIKE_ANIM_TUNE");
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
        else if (k == "lock_speed") t.foot_lock_max_speed = v;
        else log::warnf("[body] QLIKE_ANIM_TUNE: unknown name '%s'", k.c_str());
    }
}

void VulkanEngine::destroy_player_body() {
    destroy_mesh(allocator_, body_mesh_);
    for (uint32_t i = 0; i < kFrameOverlap; ++i) {
        if (body_vbuf_[i]) vmaDestroyBuffer(allocator_, body_vbuf_[i], body_valloc_[i]);
        body_vbuf_[i] = VK_NULL_HANDLE;
        body_valloc_[i] = nullptr;
        body_vmapped_[i] = nullptr;
    }
    body_animator_.reset();
    body_bones_.clear();
    body_verts_.clear();
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
    return g;
}

// One character, one frame: the animator is told what the controller (or the network) says,
// and its pose is skinned into `dst` (nullptr: pose only). Leaves the pose in body_R_ / body_P_
// (model space) for whoever needs it next (a ragdoll taking over).
void VulkanEngine::pose_character(anim::CharacterAnimator& a, anim::CharacterAnimationInput& in, Vertex* dst) {
    static const auto ground_cb = [](void* user, float x, float z) -> float {
        const GroundProbe* p = static_cast<const GroundProbe*>(user);
        return p->engine->anim_ground_height(x, z, p->feet_y);
    };
    GroundProbe probe{ this, in.position.y };
    // (level floors are known within the broadphase's reach of the local player; further out a
    // character's feet stay level rather than follow terrain that may lie under a floor)
    const glm::vec2 dc(in.position.x - near_static_center_.x, in.position.z - near_static_center_.y);
    if (glm::dot(dc, dc) < 20.0f * 20.0f) { in.ground_height = ground_cb; in.ground_user = &probe; }
    in.gravity = game_.gravity;
    in.stand_eye_height = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
    const double t0 = now_ms();
    a.update(in);
    const double t1 = now_ms();
    anim_ms_ += float(t1 - t0);
    ++anim_lod_count_[glm::clamp(in.lod, 0, 3)];
    if (in.lod >= 3) return;
    const std::span<const glm::mat4> m = a.model();
    const size_t nb = std::min(body_bones_.size(), m.size());
    for (size_t i = 0; i < nb; ++i) {
        body_R_[i] = glm::mat3(m[i]);
        body_P_[i] = glm::vec3(m[i][3]);
    }
    if (dst) {
        body_skin_to(dst);
        skin_ms_ += float(now_ms() - t1);
    }
}

// The local player's body for this frame. Called once the frame's camera is known (the body
// rides on the interpolated camera, like the flashlight).
void VulkanEngine::update_player_body(float dt) {
    body_drawn_ = false;
    anim_ms_ = skin_ms_ = 0.0f;
    for (int& c : anim_lod_count_) c = 0;
    // The cosmetic reload runs out by itself; firing ends it at once (combat.cpp).
    if (body_reload_left_ > 0.0f) body_reload_left_ = std::max(0.0f, body_reload_left_ - glm::clamp(dt, 0.0f, 0.1f));
    if (weapon_ != 0 || flashlight_on_) body_reload_left_ = 0.0f;
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
    static const float kBack0 = env_float("QLIKE_BODY_BACK", 0.11f);
    const float back = kBack0 + 0.17f * look_down + 0.25f * crouch;
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
    // Test hooks: QLIKE_BODY_WALK=<m/s> moves the body as if walking forward at that speed,
    // QLIKE_BODY_STRAFE=<m/s> adds a sideways speed (to the right), QLIKE_BODY_AIR=<m/s> holds it
    // in the air rising at that speed, QLIKE_BODY_FIRE=<shots/s> fires, QLIKE_BODY_RELOAD=1 reloads
    // in a loop.
    static const float kWalk = env_float("QLIKE_BODY_WALK", -1.0f), kStrafe = env_float("QLIKE_BODY_STRAFE", 0.0f);
    static const float kAir = env_float("QLIKE_BODY_AIR", 1e9f), kFire = env_float("QLIKE_BODY_FIRE", 0.0f);
    static const bool kReload = std::getenv("QLIKE_BODY_RELOAD") != nullptr;
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
    in.first_person = true;
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
        in.velocity = (kWalk >= 0.0f ? fwd * kWalk + right * kStrafe : glm::vec3(0.0f)) + glm::vec3(0.0f, kAir < 1e8f ? kAir : 0.0f, 0.0f);
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
    in.reloading = body_reload_left_ > 0.0f;
    // A respawn or a jump to another place: nothing remembered in world space survives it.
    static glm::vec3 last_feet(1e9f);
    in.teleported = glm::length(feet - last_feet) > 6.0f + 2.0f * glm::length(player_.velocity) * dt;
    last_feet = feet;

    const uint32_t slot = frame_number_ % kFrameOverlap;
    pose_character(*body_animator_, in, static_cast<Vertex*>(body_vmapped_[slot]));
    body_weapon_offset_ = body_animator_->weapon_offset();

    // Test hooks: QLIKE_BODY_FRONT=<m> stands the body that far in front of the camera (turned
    // by QLIKE_BODY_TURN degrees) to look at the pose from outside; QLIKE_BODY_REST=1 shows the
    // rest pose; QLIKE_BODY_DEBUG=1 logs what the animator is doing once a second.
    static const float kFront = env_float("QLIKE_BODY_FRONT", 0.0f), kTurn = glm::radians(env_float("QLIKE_BODY_TURN", 180.0f));
    static const bool kRest = std::getenv("QLIKE_BODY_REST") != nullptr, kDebug = std::getenv("QLIKE_BODY_DEBUG") != nullptr;
    if (kFront > 0.0f) {
        const glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(eye.x, eye.y - eye_h, eye.z) + fwd * kFront);
        body_model_ = T * glm::rotate(glm::mat4(1.0f), kTurn - yaw, glm::vec3(0, 1, 0));
    }
    if (kRest) {
        for (size_t i = 0; i < body_bones_.size(); ++i) { body_R_[i] = glm::mat3(1.0f); body_P_[i] = body_bones_[i].head; }
        if (body_vmapped_[slot]) body_skin_to(static_cast<Vertex*>(body_vmapped_[slot]));
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
    // Test hook: QLIKE_TEST_RAGDOLL=<frame>: at that frame the body as posed here is shot (from
    // the camera, at chest height unless QLIKE_TEST_RAGDOLL_Y=<m above the feet>) and falls.
    {
        static const int kRag = int(env_float("QLIKE_TEST_RAGDOLL", -1.0f));
        if (kRag >= 0 && int(frame_number_) == kRag) {
            glm::mat4 bm = body_model_;
            bm[3].y = sample_terrain_height(bm[3].x, bm[3].z);          // (standing on the ground there)
            const glm::vec3 base = glm::vec3(bm[3]);
            const glm::vec3 hp = base + glm::vec3(0.0f, env_float("QLIKE_TEST_RAGDOLL_Y", 1.25f), 0.0f);
            const glm::mat4 vi = bm * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, eye_h, -back));
            corpse_spawn(bm, vi, in.velocity, hp, glm::normalize(hp - eye), env_float("QLIKE_TEST_RAGDOLL_POWER", 1.0f));
        }
    }
    body_drawn_ = body_vmapped_[slot] != nullptr;
}

// What the animation is doing, drawn over the game: the chosen clip and frame, the match's cost
// by feature group, the predicted path, foot contacts and locks, IK targets, phases and timings
// for the local player; levels of detail and cost for everybody. (QLIKE_ANIM_DEBUG=1, or the menu.)
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
    for (const auto& av : mp_avatars_) {
        if (!av.id || !av.animator || !av.drawn) continue;
        const anim::AnimDebug& ad = av.animator->debug();
        if (!project(av.eye + glm::vec3(0.0f, 0.55f, 0.0f), a)) continue;
        std::snprintf(line, sizeof(line), "L%d %s %s %.2f ms", av.lod, ad.mode, ad.clip >= 0 ? anim_set_.runtime->clip(ad.clip).name.c_str() : "-", ad.update_ms);
        dl->AddText(ImVec2(a.x - ImGui::CalcTextSize(line).x * 0.5f, a.y), IM_COL32(170, 255, 200, 255), line);
    }
}

// Skins the pose in body_R_ / body_P_ into a vertex buffer.
void VulkanEngine::body_skin_to(Vertex* dst) {
    const size_t nb = body_bones_.size();
    body_skin_R_.resize(nb);
    body_skin_T_.resize(nb);
    for (size_t i = 0; i < nb; ++i) {
        body_skin_R_[i] = body_R_[i];
        body_skin_T_[i] = body_P_[i] - body_R_[i] * body_bones_[i].head;
    }
    const size_t nv = body_verts_.size();
    for (size_t i = 0; i < nv; ++i) {
        const BodyVert& v = body_verts_[i];
        glm::vec3 p(0.0f), n(0.0f);
        for (int k = 0; k < 4; ++k) {
            const float w = v.weight[k];
            if (w <= 0.0f) continue;
            const glm::mat3& R = body_skin_R_[v.bone[k]];
            p += (R * v.pos + body_skin_T_[v.bone[k]]) * w;
            n += (R * v.nrm) * w;
        }
        dst[i] = { p, glm::normalize(n), v.uv };
    }
}

void VulkanEngine::draw_player_body(VkCommandBuffer cmd, const glm::mat4& vp, const glm::vec3& cam) {
    if (!mp_draw_ && !body_drawn_) return;
    const uint32_t slot = frame_number_ % kFrameOverlap;
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, mp_draw_ ? &mp_draw_->vbuf[slot] : &body_vbuf_[slot], &off);
    vkCmdBindIndexBuffer(cmd, body_mesh_.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    PushConstants pc{};
    const glm::mat4 bm = mp_draw_ ? mp_draw_->model : body_model_;
    // `vp` is relative to `cam` (draw_viewmodel): so is the matrix it multiplies; the lighting keeps the world one.
    glm::mat4 bm_rel = bm;
    bm_rel[3] = glm::vec4(glm::vec3(bm[3]) - cam, 1.0f);
    pc.mvp = vp * bm_rel;
    pc.model = bm;
    pc.prev_mvp = pc.mvp;                  // rides on the camera: no motion vectors
    if (mp_draw_) {
        // Another player: a real object in the world, with real motion.
        if (prev_view_proj_valid_) pc.prev_mvp = prev_view_proj_ * mp_draw_->prev_model;
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
    pc.tex_params = glm::vec4(float(kTexSoldier), -8.0f, 1.0f, 0.0f);
    vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
    vkCmdDrawIndexed(cmd, body_index_count_, 1, 0, 0, 0);
}

}  // namespace qlike
