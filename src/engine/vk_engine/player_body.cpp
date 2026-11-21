// The player's own body in first person: a skinned soldier (built with
// MakeHuman / MPFB in Blender, tools/character/) whose legs walk, whose
// knees bend when crouching and whose hands hold what the player carries.
//
// The pose is made here, not played back from animation files: legs and
// arms are solved with two-bone IK toward foot / hand targets, so the hands
// are always exactly on the weapon and the feet follow the real movement.
// The mesh is skinned on the CPU (about 12k vertices) into a per-frame
// vertex buffer and drawn with the ordinary mesh shader.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace qlike {

namespace {

// Shortest rotation taking unit vector a to unit vector b.
glm::mat3 rot_from_to(glm::vec3 a, glm::vec3 b) {
    const float d = glm::dot(a, b);
    if (d > 0.99999f) return glm::mat3(1.0f);
    glm::vec3 axis;
    if (d < -0.99999f) {
        axis = glm::cross(a, std::abs(a.x) < 0.8f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0));
    } else {
        axis = glm::cross(a, b);
    }
    axis = glm::normalize(axis);
    return glm::mat3(glm::rotate(glm::mat4(1.0f), std::acos(glm::clamp(d, -1.0f, 1.0f)), axis));
}

glm::mat3 rot_axis(glm::vec3 axis, float angle) {
    return glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::normalize(axis)));
}

// "x,y,z" from an environment variable (pose tuning without a rebuild).
glm::vec3 env_vec3(const char* name, glm::vec3 def) {
    const char* s = std::getenv(name);
    if (!s) return def;
    glm::vec3 v = def;
    if (std::sscanf(s, "%f,%f,%f", &v.x, &v.y, &v.z) != 3) return def;
    return v;
}

float env_float(const char* name, float def) {
    const char* s = std::getenv(name);
    return s ? float(std::atof(s)) : def;
}

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
    body_local_.assign(nb, glm::mat3(1.0f));
    body_set_.assign(nb, 0);
    log::infof("[body] soldier: %u bones, %u vertices, %u triangles, scale %.3f (model eye %.3f m)",
               nb, nv, ni / 3, body_scale_, model_eye);
    // Movement: the motion capture database (tools/anim/build_motion_db.py).
    for (const char* p : { "assets/character/motion.qmm", "../assets/character/motion.qmm" })
        if (body_motion_.load(p)) break;
    body_motion_bone_.clear();
    if (body_motion_.ok()) {
        for (const std::string& n : body_motion_.bone_names) body_motion_bone_.push_back(body_bone(n.c_str()));
        log::infof("[body] motion matching: %u frames (with mirrors), %u bones, %zu clips", body_motion_.frames,
                   body_motion_.bones, body_motion_.ranges.size());
    } else {
        log::info("[body] no assets/character/motion.qmm - the body moves by the built-in walk cycle");
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
    body_bones_.clear();
    body_verts_.clear();
}

// Forward kinematics: every joint follows its parent (times its local
// rotation) unless its orientation was set explicitly.
void VulkanEngine::body_fk() {
    for (size_t i = 0; i < body_bones_.size(); ++i) {
        const BodyBone& b = body_bones_[i];
        if (b.parent < 0) {
            body_P_[i] = b.head + body_root_off_;
            if (!body_set_[i]) body_R_[i] = body_local_[i];
        } else {
            const BodyBone& p = body_bones_[b.parent];
            body_P_[i] = body_P_[b.parent] + body_R_[b.parent] * (b.head - p.head);
            if (!body_set_[i]) body_R_[i] = body_R_[b.parent] * body_local_[i];
        }
    }
}

// Two-bone IK: joints a -> b -> c; c goes to `target` (as far as the limb
// reaches), the middle joint bends toward `pole`.
void VulkanEngine::body_ik(int a, int b, int c, glm::vec3 target, glm::vec3 pole) {
    if (a < 0 || b < 0 || c < 0) return;
    const glm::vec3 ab0 = body_bones_[b].head - body_bones_[a].head;
    const glm::vec3 bc0 = body_bones_[c].head - body_bones_[b].head;
    const float l1 = glm::length(ab0), l2 = glm::length(bc0);
    const glm::vec3 pa = body_P_[a];
    glm::vec3 d = target - pa;
    float dist = glm::length(d);
    if (dist < 1e-4f) { d = glm::vec3(0, -1, 0); dist = 1e-4f; }
    const glm::vec3 dir = d / dist;
    dist = glm::clamp(dist, std::abs(l1 - l2) + 0.01f, (l1 + l2) * 0.995f);
    const float ca = glm::clamp((l1 * l1 + dist * dist - l2 * l2) / (2.0f * l1 * dist), -1.0f, 1.0f);
    glm::vec3 pp = pole - dir * glm::dot(pole, dir);
    if (glm::dot(pp, pp) < 1e-8f) pp = glm::cross(dir, glm::vec3(1, 0, 0));
    pp = glm::normalize(pp);
    const glm::vec3 pb = pa + dir * (l1 * ca) + pp * (l1 * std::sqrt(std::max(0.0f, 1.0f - ca * ca)));
    const glm::vec3 pc = pa + dir * dist;
    const glm::mat3 parent = body_bones_[a].parent >= 0 ? body_R_[body_bones_[a].parent] : glm::mat3(1.0f);
    body_R_[a] = rot_from_to(glm::normalize(parent * ab0), glm::normalize(pb - pa)) * parent;
    body_R_[b] = rot_from_to(glm::normalize(body_R_[a] * bc0), glm::normalize(pc - pb)) * body_R_[a];
    body_set_[a] = body_set_[b] = 1;
    body_P_[b] = pb;
    body_P_[c] = pc;
}

// A leg fitted to a new ankle position, keeping what the pose had: thigh and shin are turned
// (the shortest way) from where they pointed to where the two-bone solution puts them, with
// the knee bending the way it already did.
void VulkanEngine::body_leg_fit(int up, int lo, int ft, glm::vec3 target) {
    const float l1 = glm::length(body_bones_[lo].head - body_bones_[up].head);
    const float l2 = glm::length(body_bones_[ft].head - body_bones_[lo].head);
    const glm::vec3 pa = body_P_[up], kb = body_P_[lo], fc = body_P_[ft];
    glm::vec3 d = target - pa;
    float dist = glm::length(d);
    if (dist < 1e-4f) return;
    const glm::vec3 dir = d / dist;
    dist = glm::clamp(dist, std::abs(l1 - l2) + 0.01f, (l1 + l2) * 0.998f);
    const float ca = glm::clamp((l1 * l1 + dist * dist - l2 * l2) / (2.0f * l1 * dist), -1.0f, 1.0f);
    glm::vec3 pole = (kb - pa) - dir * glm::dot(kb - pa, dir);
    if (glm::dot(pole, pole) < 1e-6f) pole = glm::vec3(0.0f, 0.0f, -1.0f) - dir * glm::dot(glm::vec3(0.0f, 0.0f, -1.0f), dir);
    pole = glm::normalize(pole);
    const glm::vec3 pb = pa + dir * (l1 * ca) + pole * (l1 * std::sqrt(std::max(0.0f, 1.0f - ca * ca)));
    const glm::vec3 pc = pa + dir * dist;
    body_R_[up] = rot_from_to(glm::normalize(kb - pa), glm::normalize(pb - pa)) * body_R_[up];
    body_R_[lo] = rot_from_to(glm::normalize(fc - kb), glm::normalize(pc - pb)) * body_R_[lo];
    body_set_[up] = body_set_[lo] = 1;
    body_P_[lo] = pb;
    body_P_[ft] = pc;
}

// Poses and skins the body for this frame. Called once the frame's camera
// is known (the body rides on the interpolated camera, like the flashlight).
void VulkanEngine::update_player_body(float dt) {
    body_drawn_ = false;
    if (body_bones_.empty() || (!game_.player_body && !mp_pose_dst_)) return;
    CPU_SCOPE("player body");
    dt = glm::clamp(dt, 0.0f, 0.05f);
    const float tnow = static_cast<float>(glfwGetTime());

    // ---- where the body stands ----
    const float yaw = player_.yaw, pitch = player_.pitch;
    const glm::vec3 fwd(std::sin(yaw), 0.0f, -std::cos(yaw)), right(std::cos(yaw), 0.0f, std::sin(yaw));
    const float eye_h = game::Player::kHalfExtents.y + player_.eye_offset_y();
    const float stand_eye = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
    const float crouch = glm::clamp(player_.crouch_factor, 0.0f, 1.0f);
    const float look_down = glm::smoothstep(0.15f, 1.25f, -pitch);
    // The eye sits in front of the neck; further when looking down, so the
    // view is never inside the chest.
    static const float kBack0 = env_float("QLIKE_BODY_BACK", 0.11f);
    const float back = kBack0 + 0.17f * look_down + 0.25f * crouch;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    const glm::vec3 feet = eye - glm::vec3(0.0f, eye_h, 0.0f) - fwd * back;
    body_model_ = glm::mat4(glm::vec4(right, 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f), glm::vec4(-fwd, 0.0f),
                            glm::vec4(feet, 1.0f));
    // Model space: +X right, +Y up, -Z forward; the eye is at (0, eye_h, -back).
    const glm::vec3 eye_m(0.0f, eye_h, -back);
    auto to_model = [&](glm::vec3 world) {
        const glm::vec3 d = world - feet;
        return glm::vec3(glm::dot(d, right), d.y, -glm::dot(d, fwd));
    };
    // A point given in view space (x right, y up, z toward the viewer).
    const glm::vec3 cam_f(0.0f, std::sin(pitch), -std::cos(pitch)), cam_u(0.0f, std::cos(pitch), std::sin(pitch));
    auto view_pt = [&](glm::vec3 v) { return eye_m + glm::vec3(v.x, 0.0f, 0.0f) + cam_u * v.y - cam_f * v.z; };
    auto view_dir = [&](glm::vec3 v) { return glm::vec3(v.x, 0.0f, 0.0f) + cam_u * v.y - cam_f * v.z; };

    // ---- movement ----
    // Test hook: QLIKE_BODY_WALK=<m/s> poses the body as if walking forward at that speed.
    // (QLIKE_BODY_STRAFE=<m/s> adds a sideways speed, to the right.)
    static const float kWalk = env_float("QLIKE_BODY_WALK", -1.0f), kStrafe = env_float("QLIKE_BODY_STRAFE", 0.0f);
    // (QLIKE_BODY_AIR=<m/s> poses it in the air, rising at that speed.)
    static const float kAir = env_float("QLIKE_BODY_AIR", 1e9f);
    const glm::vec3 vel = (kWalk >= 0.0f ? fwd * kWalk + right * kStrafe : player_.velocity) + glm::vec3(0.0f, kAir < 1e8f ? kAir : 0.0f, 0.0f);
    const glm::vec2 vm(glm::dot(vel, right), -glm::dot(vel, fwd));          // model x, z
    const float speed = glm::length(vm);
    const bool grounded = ((player_.on_ground && !player_swimming_) || kWalk >= 0.0f) && kAir > 1e8f;
    const float ease = 1.0f - std::exp(-dt * 9.0f);
    body_air_ += ((grounded ? 0.0f : 1.0f) - body_air_) * (1.0f - std::exp(-dt * 7.0f));
    body_move_ += (glm::clamp(speed / 1.2f, 0.0f, 1.0f) - body_move_) * ease;
    if (speed > 0.2f) body_dir_ = glm::normalize(glm::mix(body_dir_, vm / speed, ease));
    if (glm::dot(body_dir_, body_dir_) < 0.25f) body_dir_ = glm::vec2(0.0f, -1.0f);
    // Step length as people really walk and run: ~0.7 m at a stroll, ~1.25 m
    // at a run (was 0.48 / 0.77 m: twice too many small, hurried steps).
    const float stride = glm::clamp(0.44f + 0.185f * speed, 0.40f, 1.30f) * (1.0f - 0.45f * crouch);   // metres per step
    if (grounded) body_phase_ += 3.14159265f * speed * dt / std::max(stride, 0.2f);
    if (body_phase_ > 6.2831853f * 64.0f) body_phase_ -= 6.2831853f * 64.0f;

    // ---- reset the pose ----
    std::fill(body_set_.begin(), body_set_.end(), uint8_t(0));
    std::fill(body_local_.begin(), body_local_.end(), glm::mat3(1.0f));

    // ---- hips, spine and legs: from motion capture ----
    // (QLIKE_BODY_PROCEDURAL=1: the built-in walk cycle instead.)
    static const bool kNoMocap = std::getenv("QLIKE_BODY_PROCEDURAL") != nullptr;
    const bool mocap = body_motion_.ok() && !kNoMocap;
    if (mocap) {
        BodyAnim& an = *body_anim_;
        auto spring0 = [](glm::vec3& x, glm::vec3& v, float halflife, float h) {       // critically damped, toward 0
            const float y = 4.0f * 0.69314718f / halflife * 0.5f;
            const glm::vec3 j1 = v + x * y;
            const float e = std::exp(-y * h);
            x = e * (x + j1 * h);
            v = e * (v - j1 * y * h);
        };
        // How fast the body turns.
        float turn = 0.0f;
        if (an.has_yaw && dt > 1e-4f) {
            float d = yaw - an.prev_yaw;
            d -= 6.2831853f * std::round(d / 6.2831853f);
            turn = d / dt;
        }
        an.prev_yaw = yaw;
        an.has_yaw = true;
        an.turn += (glm::clamp(turn, -6.0f, 6.0f) - an.turn) * (1.0f - std::exp(-dt * 8.0f));
        // The game runs much faster than a person in a capture studio (8 m/s against 2): the
        // search is asked for the capture's equivalent, and the clip it finds is then played
        // faster and with longer strides until its feet cover the real ground speed.
        const float qs = speed <= 1.0f ? speed : 1.0f + 1.1f * (1.0f - std::exp(-(speed - 1.0f) / 2.5f));
        MotionQuery mq;
        mq.vel = speed > 0.05f ? vm / speed * qs : glm::vec2(0.0f);
        mq.turn_rate = an.turn;
        mq.tag = 0;
        if (crouch > 0.5f)
            for (const MotionDB::Range& rg : body_motion_.ranges) if (rg.tag == 1) { mq.tag = 1; break; }
        mq.rate = an.rate;
        // Jumps: captured ones. In the air the frame follows the flight - push-off while rising
        // fast, the top of the jump as the rise dies, reaching for the ground while falling -
        // because the game's jump lasts far longer than a person's. On touching down the
        // capture's landing plays out, then matching takes over again.
        const bool airborne = !grounded && !player_swimming_;
        if (!body_motion_.jumps.empty()) {
            if (airborne) {
                an.air_t += dt;
                if (an.jstate != 1 && an.air_t > 0.05f) {
                    an.jump = motion_find_jump(body_motion_, mq);
                    an.jstate = an.jump >= 0 ? 1 : 0;
                    if (an.jstate == 1) motion_goto(body_motion_, an.ms, int(body_motion_.jumps[an.jump].takeoff), 1.0f);
                }
            } else {
                if (an.jstate == 1) {
                    an.jstate = an.air_t > 0.20f ? 2 : 0;
                    an.land_t = 0.0f;
                    if (an.jstate == 2 && std::abs(an.ms.frame - int(body_motion_.jumps[an.jump].land)) > 2)
                        motion_goto(body_motion_, an.ms, int(body_motion_.jumps[an.jump].land), 1.0f);
                }
                an.air_t = 0.0f;
            }
        }
        if (an.jstate == 1) {
            const MotionDB::Jump& jp = body_motion_.jumps[an.jump];
            const float u = glm::clamp(0.5f - 0.5f * vel.y / 8.5f, 0.0f, 1.0f);
            const float fpos = float(jp.takeoff) + u * float(jp.land - jp.takeoff - 1);
            const float cur = float(an.ms.frame) + an.ms.frac;
            // (no faster than three times the capture's own pace, so nothing snaps)
            const float nf = cur + glm::clamp(fpos - cur, -dt * 90.0f, dt * 90.0f);
            an.ms.frame = int(std::floor(nf));
            an.ms.frac = nf - std::floor(nf);
            motion_pose(body_motion_, an.ms, dt, body_pose_);
        } else if (an.jstate == 2) {
            an.land_t += dt;
            an.ms.frac += dt * body_motion_.fps;
            while (an.ms.frac >= 1.0f) { an.ms.frac -= 1.0f; ++an.ms.frame; }
            motion_pose(body_motion_, an.ms, dt, body_pose_);
            if (an.land_t > 0.28f) { an.jstate = 0; an.ms.since_search = 1e9f; }
        } else {
            motion_update(body_motion_, an.ms, mq, dt, body_pose_);
        }
        const bool jumping = an.jstate != 0;
        const float as = glm::length(body_pose_.root_vel);
        float rate_t = 1.0f, stride_t = 1.0f, warp_t = 0.0f;
        if (speed > 0.3f && as > 0.3f && !jumping) {
            rate_t = glm::clamp(speed / as, 0.75f, 1.6f);
            stride_t = glm::clamp(speed / (as * rate_t), 0.7f, 2.4f);
            // Orientation warping: the legs turn to the way we really go (a diagonal, say, on a
            // forward or a sideways clip) while the chest keeps facing where we look.
            float a = std::atan2(vm.x, vm.y) - std::atan2(body_pose_.root_vel.x, body_pose_.root_vel.y);
            a -= 6.2831853f * std::round(a / 6.2831853f);
            warp_t = glm::clamp(a, -1.0f, 1.0f);
        }
        const float e2 = 1.0f - std::exp(-dt * 10.0f);
        an.rate += (rate_t - an.rate) * e2;
        an.stride += (stride_t - an.stride) * e2;
        an.warp += (warp_t - an.warp) * e2;

        const int hips_i = body_bone("Hips");
        const float hip_drop = (stand_eye - eye_h) * (mq.tag == 1 ? 0.50f : 0.62f);
        // A share of the look pitch goes into the spine (the arms follow the view); folded when crouched.
        const float bend = 0.10f * glm::clamp(pitch, -1.2f, 1.2f) - (mq.tag == 1 ? 0.25f : 0.55f) * crouch;
        const glm::mat3 W = rot_axis(glm::vec3(0, 1, 0), an.warp);
        for (size_t k = 0; k < body_motion_bone_.size(); ++k) {
            const int bi = body_motion_bone_[k];
            if (bi < 0) continue;
            const std::string& nm = body_bones_[bi].name;
            float up_frac = 1.0f;                                      // 0 = lower body, 1 = chest and above
            if (nm == "Hips" || nm.find("Leg") != std::string::npos || nm.find("Foot") != std::string::npos || nm.find("Toe") != std::string::npos) up_frac = 0.0f;
            else if (nm == "Spine") up_frac = 1.0f / 3.0f;
            else if (nm == "Spine1") up_frac = 2.0f / 3.0f;
            glm::mat3 R = glm::mat3_cast(body_pose_.rot[k]);
            if (up_frac < 1.0f) R = rot_axis(glm::vec3(0, 1, 0), an.warp * (1.0f - up_frac)) * R;
            if (up_frac > 0.0f) R = rot_axis(glm::vec3(1, 0, 0), bend * up_frac) * R;
            body_R_[bi] = R;
            body_set_[bi] = 1;
        }
        if (hips_i >= 0) {
            const glm::vec3 H = body_bones_[hips_i].head;
            const glm::vec3 hp = W * glm::vec3(body_pose_.hip.x, 0.0f, body_pose_.hip.z) * body_scale_;
            body_root_off_ = glm::vec3(hp.x, body_pose_.hip.y * body_scale_ - H.y - hip_drop, hp.z + 0.10f * crouch);
        }
        body_fk();
        const glm::vec3 hipP = hips_i >= 0 ? body_P_[hips_i] : glm::vec3(0.0f);
        const glm::vec2 mdir = speed > 0.3f ? vm / speed : glm::vec2(0.0f, -1.0f);
        for (int side = 0; side < 2; ++side) {
            const char* L = side == 0 ? "Left" : "Right";
            const int up = body_bone((std::string(L) + "UpLeg").c_str());
            const int lo = body_bone((std::string(L) + "Leg").c_str());
            const int ft = body_bone((std::string(L) + "Foot").c_str());
            if (up < 0 || lo < 0 || ft < 0) continue;
            glm::vec3 target = body_P_[ft];
            // Stride warping: the feet reach further along the way we go.
            glm::vec2 rel(target.x - hipP.x, target.z - hipP.z);
            rel += mdir * (glm::dot(rel, mdir) * (an.stride - 1.0f));
            target.x = hipP.x + rel.x;
            target.z = hipP.z + rel.y;
            // Foot locking: a foot the capture has on the ground stays where it stands in the
            // world; let go, what is left of the difference eases away.
            BodyAnim::Lock& lk = an.lock[side];
            const bool want = body_pose_.contact[side] > 0.5f && grounded && speed < 4.0f;
            const glm::vec3 tw = feet + right * target.x + glm::vec3(0.0f, target.y, 0.0f) - fwd * target.z;
            if (!want) lk.broken = false;
            if (want && !lk.on && !lk.broken) { lk.on = true; lk.world = tw + lk.off; lk.off_v = glm::vec3(0.0f); }
            if (lk.on) {
                lk.off = lk.world - tw;
                lk.off.y = 0.0f;
                if (!want || glm::length(lk.off) > 0.22f) { lk.on = false; lk.broken = want; }
            }
            if (!lk.on) spring0(lk.off, lk.off_v, 0.09f, dt);
            target += to_model(feet + lk.off);            // (the offset, turned into the body's frame)
            // In the air: legs hang, knees a little bent, one ahead of the other.
            const glm::vec3 air = body_bones_[ft].head + glm::vec3(0.0f, 0.16f, side == 0 ? -0.14f : 0.06f);
            // (only without captured jumps: with them the capture has the legs)
            const float air_k = body_motion_.jumps.empty() ? body_air_ : 0.0f;
            target = glm::mix(target, air, air_k);
            body_leg_fit(up, lo, ft, target);
            if (air_k > 0.01f) body_R_[ft] = rot_axis(glm::vec3(1, 0, 0), -0.35f * air_k) * body_R_[ft];
        }
        body_fk();
    } else {
    const int hips = body_bone("Hips");
    const int spine[3] = { body_bone("Spine"), body_bone("Spine1"), body_bone("Spine2") };
    // Hips: lower when crouching, a small bounce per step, a breath of sway.
    // The body is lowest when both feet are down and rides up over the
    // standing leg; more at a run.
    const float bounce = -(0.022f + 0.010f * glm::clamp(speed, 0.0f, 6.0f)) * (0.5f + 0.5f * std::cos(2.0f * body_phase_)) *
                         body_move_ * (1.0f - body_air_);
    const float hip_drop = (stand_eye - eye_h) * 0.62f;
    body_root_off_ = glm::vec3(0.012f * std::sin(body_phase_) * body_move_, -hip_drop + bounce, 0.10f * crouch);
    // Torso: leans into the run and folds forward when crouched; a little of
    // the look pitch goes into the spine so the arms follow the view.
    const float lean = -(0.035f * glm::clamp(speed, 0.0f, 7.0f) * (1.0f - body_air_) * (vm.y < 0.0f ? 1.0f : -0.4f) * 0.25f) -
                       0.55f * crouch + 0.10f * glm::clamp(pitch, -1.2f, 1.2f);
    const float breath = 0.006f * std::sin(tnow * 1.5f);
    for (int s : spine)
        if (s >= 0) body_local_[s] = rot_axis(glm::vec3(1, 0, 0), (lean + breath) / 3.0f);
    // Pelvis: turns with the stride and drops a little on the swinging side.
    if (hips >= 0) body_local_[hips] = rot_axis(glm::vec3(0, 1, 0), 0.09f * std::cos(body_phase_) * body_move_) *
                                       rot_axis(glm::vec3(0, 0, 1), 0.035f * std::sin(body_phase_) * body_move_);
    body_fk();

    // ---- legs ----
    const glm::vec3 dir3(body_dir_.x, 0.0f, body_dir_.y);
    for (int side = 0; side < 2; ++side) {
        const char* L = side == 0 ? "Left" : "Right";
        const int up = body_bone((std::string(L) + "UpLeg").c_str());
        const int lo = body_bone((std::string(L) + "Leg").c_str());
        const int ft = body_bone((std::string(L) + "Foot").c_str());
        if (up < 0 || lo < 0 || ft < 0) continue;
        const float ph = body_phase_ + (side == 0 ? 0.0f : 3.14159265f);
        // One cycle: the first half the foot stands (it moves back under the
        // body at exactly walking speed - no sliding), the second half it
        // swings forward, lifting early and reaching out before it lands.
        const float cyc = ph / 6.2831853f - std::floor(ph / 6.2831853f);
        const bool  stance = cyc < 0.5f;
        const float u = stance ? cyc / 0.5f : (cyc - 0.5f) / 0.5f;          // 0..1 within the half
        const float along = stance ? (0.5f - u) : (-0.5f + u * u * (3.0f - 2.0f * u));
        const float run = glm::clamp((speed - 2.0f) / 3.5f, 0.0f, 1.0f);
        // Lift: quick off the ground, peak before the middle, down to land.
        const float swing = stance ? 0.0f : std::sin(3.14159265f * std::pow(u, 0.75f));
        glm::vec3 target = body_bones_[ft].head;
        target += dir3 * (stride * along * body_move_);
        target.y += (0.055f + 0.16f * run) * swing * body_move_;
        // A runner's trailing foot kicks up behind.
        if (!stance) target.y += 0.10f * run * std::sin(3.14159265f * glm::clamp(u * 2.0f, 0.0f, 1.0f)) * body_move_;
        // Crouched: feet a little apart and forward.
        target += glm::vec3((side == 0 ? -1.0f : 1.0f) * 0.05f * crouch, 0.0f, -0.10f * crouch);
        // In the air: legs hang, knees a little bent, one ahead of the other.
        const glm::vec3 air = body_bones_[ft].head + glm::vec3(0.0f, 0.16f, side == 0 ? -0.14f : 0.06f);
        target = glm::mix(target, air, body_air_);
        const glm::vec3 pole(side == 0 ? -0.18f : 0.18f, 0.0f, -1.0f);
        body_ik(up, lo, ft, target, pole);
        // Foot: flat on the ground, toes down while it swings.
        // Foot roll: heel strikes first (toes up), flat through the stance,
        // heel lifts before push-off, toes hang down through the swing.
        float roll = 0.0f;
        if (stance) roll = 0.30f * (1.0f - glm::smoothstep(0.0f, 0.18f, u)) - 0.55f * glm::smoothstep(0.70f, 1.0f, u);
        else        roll = -0.55f * (1.0f - glm::smoothstep(0.0f, 0.55f, u)) + 0.30f * glm::smoothstep(0.65f, 1.0f, u);
        const float toe = roll * body_move_ * (1.0f - body_air_) - 0.35f * body_air_;
        body_R_[ft] = rot_axis(glm::vec3(1, 0, 0), toe);
        body_set_[ft] = 1;
    }
    body_fk();
    }

    // ---- arms ----
    // What the hands hold decides where they go (view space, metres).
    static const glm::vec3 kGunR = env_vec3("QLIKE_BODY_RH", glm::vec3(0.165f, -0.300f, -0.120f));
    static const glm::vec3 kGunL = env_vec3("QLIKE_BODY_LH", glm::vec3(0.178f, -0.243f, -0.430f));
    const float bob_x = 0.006f * std::sin(body_phase_) * body_move_;
    const float bob_y = -0.005f * (0.5f - 0.5f * std::cos(2.0f * body_phase_)) * body_move_;
    for (int side = 0; side < 2; ++side) {
        const char* L = side == 0 ? "Left" : "Right";
        const std::string S(L);
        const int ua = body_bone((S + "Arm").c_str()), fa = body_bone((S + "ForeArm").c_str()), hd = body_bone((S + "Hand").c_str());
        const int i1 = body_bone((S + "HandIndex1").c_str()), m1 = body_bone((S + "HandMiddle1").c_str()),
                  p1 = body_bone((S + "HandPinky1").c_str());
        if (ua < 0 || fa < 0 || hd < 0 || i1 < 0 || m1 < 0 || p1 < 0) continue;
        const float sx = side == 0 ? -1.0f : 1.0f;
        // The hand's own frame at rest: a = wrist to knuckles, n = out of the
        // palm, b = a x n.
        const glm::vec3 a0 = glm::normalize(body_bones_[m1].head - body_bones_[hd].head);
        glm::vec3 n0 = glm::normalize(glm::cross(a0, body_bones_[i1].head - body_bones_[p1].head));
        if (n0.y > 0.0f) n0 = -n0;                                   // palms face down in the rest pose
        const glm::vec3 b0 = glm::cross(a0, n0);
        const glm::mat3 F0(a0, n0, b0);

        glm::vec3 grip, a1, n1;          // palm centre, finger direction, palm normal (model space)
        glm::vec3 pole(sx * 0.9f, -1.0f, 0.5f);
        float curl[3] = { 0.9f, 1.15f, 0.8f };                       // finger joints (rad)
        float index_curl = 1.0f;
        bool  free_arm = false;
        bool  grip_thumb = false;        // thumb closes round the grip instead of sticking out
        if (flashlight_on_ && side == 1) {
            // Fist round the torch body, thumb forward along it.
            const glm::vec3 f = glm::normalize(glm::vec3(glm::dot(flash_fwd_, right), flash_fwd_.y, -glm::dot(flash_fwd_, fwd)));
            const glm::vec3 r = glm::normalize(glm::vec3(glm::dot(flash_right_, right), flash_right_.y, -glm::dot(flash_right_, fwd)));
            const glm::vec3 u = glm::cross(r, f);
            grip = to_model(flash_pos_) - f * 0.030f + r * 0.012f;
            pole = glm::vec3(0.30f, -1.0f, 0.85f);                    // elbow down and back, by the ribs
            a1 = glm::normalize(f * 0.80f - u * 0.60f);
            n1 = -r;
            curl[0] = 1.15f; curl[1] = 1.35f; curl[2] = 0.9f;
        } else if (!flashlight_on_ && weapon_ == 0 && side == 1) {
            // Right hand on the pistol grip, index finger on the trigger.
            grip = view_pt(kGunR + glm::vec3(bob_x, bob_y, body_recoil_));
            a1 = glm::normalize(view_dir(env_vec3("QLIKE_BODY_RA", glm::vec3(-0.10f, 0.25f, -1.0f))));
            n1 = glm::normalize(view_dir(env_vec3("QLIKE_BODY_RN", glm::vec3(-1.0f, 0.0f, 0.10f))));
            index_curl = 0.45f;
        } else if (!flashlight_on_ && weapon_ == 0 && side == 0) {
            // Left hand under the fore-end.
            grip = view_pt(kGunL + glm::vec3(bob_x, bob_y, body_recoil_));
            a1 = glm::normalize(view_dir(env_vec3("QLIKE_BODY_LA", glm::vec3(0.95f, 0.20f, -0.35f))));
            n1 = glm::normalize(view_dir(env_vec3("QLIKE_BODY_LN", glm::vec3(0.05f, 1.0f, 0.15f))));
            // Palm up under the fore-end, fingers wrapped up its far side, thumb up the near side.
            curl[0] = 1.05f; curl[1] = 1.30f; curl[2] = 0.85f;
            grip_thumb = true;
        } else if (!flashlight_on_ && weapon_ == 1 && side == 1) {
            // The tablet lies on the right hand.
            grip = view_pt(glm::vec3(0.170f, -0.200f + bob_y, -0.375f));
            a1 = glm::normalize(view_dir(glm::vec3(-0.35f, 0.55f, -0.75f)));
            n1 = glm::normalize(view_dir(glm::vec3(-0.25f, 0.70f, 0.65f)));
            curl[0] = 0.25f; curl[1] = 0.3f; curl[2] = 0.2f;
        } else {
            // Free arm: hangs at the side and swings against the legs.
            free_arm = true;
            const int sh = body_bone((S + "Shoulder").c_str());
            const glm::vec3 base = body_P_[sh >= 0 ? sh : ua];
            const float sw = std::sin(body_phase_ + (side == 0 ? 3.14159265f : 0.0f)) * body_move_;
            grip = base + glm::vec3(sx * 0.19f, -0.56f + 0.03f * std::abs(sw), -0.07f - 0.20f * sw - 0.10f * crouch);
            a1 = glm::normalize(glm::vec3(0.0f, -1.0f, -0.25f - 0.3f * sw));
            n1 = glm::vec3(-sx, 0.0f, 0.0f);
            pole = glm::vec3(sx * 0.4f, -0.3f, 1.0f);
            curl[0] = 0.35f; curl[1] = 0.45f; curl[2] = 0.3f;
        }
        n1 = glm::normalize(n1 - a1 * glm::dot(n1, a1));
        // The right-handed frame mirrors for the left hand.
        const glm::mat3 F1(a1, n1, glm::cross(a1, n1));
        const glm::mat3 G = F1 * glm::transpose(F0);
        // Wrist from the palm centre.
        const float palm = glm::length(body_bones_[m1].head - body_bones_[hd].head);
        const glm::vec3 wrist = grip - a1 * (palm * (free_arm ? 0.0f : 0.80f)) - n1 * (free_arm ? 0.0f : 0.028f);
        if (!(free_arm && mocap)) {                // (a free arm swings as the capture has it)
            body_ik(ua, fa, hd, wrist, pole);
            body_R_[hd] = G;
            body_set_[hd] = 1;
        }
        // Fingers close round the grip (rotation taking a toward n).
        for (const char* fn : { "Index", "Middle", "Ring", "Pinky" }) {
            const float k = (fn[0] == 'I') ? index_curl : 1.0f;
            for (int j = 0; j < 3; ++j) {
                const int fb = body_bone((S + "Hand" + fn + char('1' + j)).c_str());
                if (fb >= 0) body_local_[fb] = rot_axis(b0, curl[j] * k);
            }
        }
        // Thumb: swings in over the fingers.
        for (int j = 0; j < 3; ++j) {
            const int tb = body_bone((S + "HandThumb" + char('1' + j)).c_str());
            if (tb >= 0) body_local_[tb] = grip_thumb
                ? rot_axis(a0, -sx * (j == 0 ? 0.85f : 0.0f)) * rot_axis(b0, j == 0 ? 0.35f : 0.55f)
                : rot_axis(a0, -sx * (j == 0 ? 0.35f : 0.0f)) * rot_axis(n0, sx * (j == 0 ? 0.10f : 0.30f) * (free_arm ? 0.3f : 1.0f));
        }
    }
    body_fk();

    // Test hooks: QLIKE_BODY_FRONT=<m> stands the body that far in front of
    // the camera (turned by QLIKE_BODY_TURN degrees) to look at the pose from
    // outside; QLIKE_BODY_REST=1 shows the rest pose.
    static const float kFront = env_float("QLIKE_BODY_FRONT", 0.0f), kTurn = glm::radians(env_float("QLIKE_BODY_TURN", 180.0f));
    static const bool kRest = std::getenv("QLIKE_BODY_REST") != nullptr;
    if (kFront > 0.0f) {
        const glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(eye.x, eye.y - eye_h, eye.z) + fwd * kFront);
        body_model_ = T * glm::rotate(glm::mat4(1.0f), kTurn - yaw, glm::vec3(0, 1, 0));
    }
    if (kRest) {
        std::fill(body_set_.begin(), body_set_.end(), uint8_t(0));
        std::fill(body_local_.begin(), body_local_.end(), glm::mat3(1.0f));
        body_root_off_ = glm::vec3(0.0f);
        body_fk();
    }

    // ---- skin ----
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
        body_skinned_[i] = { p, glm::normalize(n), v.uv };
    }
    static const bool kDebug = std::getenv("QLIKE_BODY_DEBUG") != nullptr;
    if (kDebug && frame_number_ == 100) {
        std::vector<float> worst(nb, 0.0f);
        for (size_t i = 0; i < nv; ++i) {
            const BodyVert& v = body_verts_[i];
            const float d0 = glm::length(v.pos - body_bones_[v.bone[0]].head);
            const float d1 = glm::length(body_skinned_[i].position - body_P_[v.bone[0]]);
            worst[v.bone[0]] = std::max(worst[v.bone[0]], d1 - d0);
        }
        for (size_t t = 0, shown = 0; t + 2 < body_dbg_idx_.size() && shown < 25; t += 3) {
            for (int e = 0; e < 3; ++e) {
                const uint32_t i0 = body_dbg_idx_[t + e], i1 = body_dbg_idx_[t + (e + 1) % 3];
                const float r = glm::length(body_verts_[i0].pos - body_verts_[i1].pos);
                const float p = glm::length(body_skinned_[i0].position - body_skinned_[i1].position);
                if (p > r + 0.08f) {
                    const BodyVert& a = body_verts_[i0]; const BodyVert& b = body_verts_[i1];
                    log::infof("[body] edge %.3f -> %.3f  v%u %s %.2f/%s %.2f  v%u %s %.2f/%s %.2f", r, p,
                               i0, body_bones_[a.bone[0]].name.c_str(), a.weight[0], body_bones_[a.bone[1]].name.c_str(), a.weight[1],
                               i1, body_bones_[b.bone[0]].name.c_str(), b.weight[0], body_bones_[b.bone[1]].name.c_str(), b.weight[1]);
                    ++shown;
                    break;
                }
            }
        }
        for (size_t i = 0; i < nb; ++i) {
            const glm::mat3& R = body_R_[i];
            log::infof("[body] %-18s stretch %.3f  det %.3f  P %.2f %.2f %.2f", body_bones_[i].name.c_str(), worst[i],
                       glm::determinant(R), body_P_[i].x, body_P_[i].y, body_P_[i].z);
        }
    }
    // Test hook: QLIKE_TEST_RAGDOLL=<frame>: at that frame the body as posed here is shot (from
    // the camera, at chest height unless QLIKE_TEST_RAGDOLL_Y=<m above the feet>) and falls.
    {
        static const int kRag = int(env_float("QLIKE_TEST_RAGDOLL", -1.0f));
        if (kRag >= 0 && !mp_pose_dst_ && int(frame_number_) == kRag) {
            glm::mat4 bm = body_model_;
            bm[3].y = sample_terrain_height(bm[3].x, bm[3].z);          // (standing on the ground there)
            const glm::vec3 base = glm::vec3(bm[3]);
            const glm::vec3 hp = base + glm::vec3(0.0f, env_float("QLIKE_TEST_RAGDOLL_Y", 1.25f), 0.0f);
            const glm::mat4 vi = bm * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, eye_h, -back));
            corpse_spawn(bm, vi, vel, hp, glm::normalize(hp - eye), env_float("QLIKE_TEST_RAGDOLL_POWER", 1.0f));
        }
    }
    const uint32_t slot = frame_number_ % kFrameOverlap;
    void* dst = mp_pose_dst_ ? static_cast<void*>(mp_pose_dst_) : body_vmapped_[slot];     // a remote player's buffer (multiplayer.cpp)
    if (dst) {
        std::memcpy(dst, body_skinned_.data(), nv * sizeof(Vertex));
        body_drawn_ = true;
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
