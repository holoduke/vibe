#include "anim/character_animator.h"

#include "game/player.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace vibe::anim {

namespace {

constexpr float kPi = 3.14159265f;
constexpr float kDipHalflife = 0.10f;      // seconds: how quickly the pelvis gives under a landing and comes back

float smooth01(float x) { x = std::clamp(x, 0.0f, 1.0f); return x * x * (3.0f - 2.0f * x); }

glm::quat rot_x(float a) { return glm::angleAxis(a, glm::vec3(1.0f, 0.0f, 0.0f)); }

// The kick of a shot: quickly back to the full stroke, then forward again.
float recoil_curve(float t_remaining, float duration) {
    if (t_remaining <= 0.0f || duration <= 0.0f) return 0.0f;
    const float t = 1.0f - t_remaining / duration;
    if (t < 0.30f) return smooth01(t / 0.30f);
    return std::cos((t - 0.30f) / 0.70f * 1.57079632f);
}

glm::vec3 model_to_world_vec(glm::vec3 m, float yaw) {
    const float s = std::sin(yaw), c = std::cos(yaw);
    return glm::vec3(c * m.x - s * m.z, m.y, s * m.x + c * m.z);        // right * x + up * y - forward * z
}
glm::vec3 world_to_model_vec(glm::vec3 w, float yaw) {
    const float s = std::sin(yaw), c = std::cos(yaw);
    return glm::vec3(c * w.x + s * w.z, w.y, -s * w.x + c * w.z);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

const WeaponDef& default_rifle() {
    static const WeaponDef w{};
    return w;
}

bool AnimationSet::load(const char* path, std::string* why) {
    AnimAsset asset;
    if (!asset.load(path, why)) return false;
    if (!init(asset, why)) return false;
    asset_bytes = asset.skeleton_blob.size() + asset.motion_blob.size();
    for (const auto& b : asset.clip_blobs) asset_bytes += b.size();
    return true;
}

bool AnimationSet::init(const AnimAsset& asset, std::string* why) {
    runtime.reset();
    db = MotionDatabase{};
    std::unique_ptr<IAnimationRuntime> rt = create_runtime(asset, why);
    if (!rt) return false;
    if (!db.load(asset.motion_blob.data(), asset.motion_blob.size(), asset.skeleton.rig_id(), why)) return false;
    for (const MotionRange& r : db.ranges())
        if (r.clip < 0 || r.clip >= rt->num_clips()) { if (why) *why = "motion database names a clip the asset does not have"; db = MotionDatabase{}; return false; }
    if (!rt->rig().complete()) { if (why) *why = "the skeleton lacks a joint the animation needs"; db = MotionDatabase{}; return false; }
    const SkeletonDef& s = rt->skeleton();
    face = Face{};
    face.jaw = s.find("Jaw");
    for (int side = 0; side < 2; ++side) {
        const std::string S = side == 0 ? "Left" : "Right";
        face.eye[side] = s.find(S + "Eye");
        face.lid_up[side] = s.find(S + "UpperLid");
        face.lid_lo[side] = s.find(S + "LowerLid");
        face.brow[side] = s.find(S + "Brow");
    }
    face.ok = face.eye[0] >= 0 && face.eye[1] >= 0 && face.lid_up[0] >= 0 && face.lid_up[1] >= 0;
    static const char* const kFingers[4] = { "Index", "Middle", "Ring", "Pinky" };
    for (int side = 0; side < 2; ++side) {
        const std::string S = side == 0 ? "LeftHand" : "RightHand";
        Hand& h = hand[side];
        for (int f = 0; f < 4; ++f)
            for (int j = 0; j < 3; ++j) h.finger[f][j] = s.find(S + kFingers[f] + char('1' + j));
        for (int j = 0; j < 3; ++j) h.thumb[j] = s.find(S + "Thumb" + char('1' + j));
        const int hd = s.find(S), i1 = h.finger[0][0], m1 = h.finger[1][0], p1 = h.finger[3][0];
        h.ok = hd >= 0 && i1 >= 0 && m1 >= 0 && p1 >= 0;
        if (h.ok) {
            h.a0 = glm::normalize(s.rest_pos[size_t(m1)] - s.rest_pos[size_t(hd)]);
            h.n0 = glm::normalize(glm::cross(h.a0, s.rest_pos[size_t(i1)] - s.rest_pos[size_t(p1)]));
            if (h.n0.y > 0.0f) h.n0 = -h.n0;                       // palms face down in the rest pose
            h.b0 = glm::cross(h.a0, h.n0);
            h.palm = glm::length(s.rest_pos[size_t(m1)] - s.rest_pos[size_t(hd)]);
            h.index_side = glm::dot(h.b0, s.rest_pos[size_t(i1)] - s.rest_pos[size_t(p1)]) < 0.0f ? 1.0f : -1.0f;
        }
    }
    // Standing height of the hips: the median over the idle captures.
    std::vector<float> ys;
    for (const MotionRange& r : db.ranges())
        if ((r.tags & kTagIdle) && (r.tags & kTagLocomotion))
            for (uint32_t f = r.first; f < r.first + r.count; f += 4) ys.push_back(db.info(f).hips_y);
    stand_hips = s.rest_pos[size_t(rt->rig()[Role::Hips])].y;
    if (!ys.empty()) { std::nth_element(ys.begin(), ys.begin() + ptrdiff_t(ys.size() / 2), ys.end()); stand_hips = ys[ys.size() / 2]; }
    runtime = std::move(rt);
    build_masks();
    return true;
}

CharacterAnimator::CharacterAnimator(const AnimationSet& set) : set_(set) {
    tuning.move = game::kPlayerMove;
    const size_t n = size_t(set_.runtime->skeleton().size());
    ctx_ = set_.runtime->create_context(3);       // (0: what plays, 1: where it goes next, 2: the clip layer)
    pose_.resize(n); src_.resize(n); next_.resize(n); scratch_.resize(n); good_.resize(n);
    model_.resize(n); good_model_.resize(n);
    reset();
}

void CharacterAnimator::reset() {
    const IAnimationRuntime& rt = *set_.runtime;
    rt.rest_pose(pose_);
    rt.local_to_model(*ctx_, pose_, model_);
    inert_.reset(rt.skeleton().size());
    match_ = MatcherState{};
    air_ = AirborneState{};
    cur_ = MatcherOutput{};
    started_ = have_good_ = false;
    yaw_rate_ = prev_yaw_ = time_ = reload_u_ = 0.0f;
    prev_vel_ = glm::vec3(0.0f);
    shot_seq_ = reload_seq_ = 0;
    held_ = HeldItem::Rifle;
    muzzle_pos_ = glm::vec3(0.0f);
    muzzle_dir_ = glm::vec3(0.0f, 0.0f, -1.0f);
    accel_ = glm::vec3(0.0f);
    rate_ = stride_ = 1.0f;
    warp_ = lean_roll_ = lean_roll_v_ = lean_pitch_ = lean_pitch_v_ = 0.0f;
    snap_axis_ = 0.0f;
    snap_on_ = strafe_run_ = false;
    flinch_pitch_ = flinch_pitch_v_ = flinch_roll_ = flinch_roll_v_ = 0.0f;
    mag_visible_ = false;
    dip_ = dip_v_ = pelvis_ik_ = pelvis_ik_v_ = 0.0f;
    for (Lock& l : lock_) l = Lock{};
    recoil_t_ = 0.0f; reload_t_ = -1.0f; reload_w_ = 0.0f; equip_t_ = 1.0f;
    aim_w_ = 1.0f; since_shot_ = 1e9f;
    rng_ = 0x9E3779B9u; rng_seeded_ = false;
    gaze_mode_ = 0; gaze_left_ = 0.0f; gaze_yaw_t_ = gaze_pitch_t_ = 0.0f;
    gaze_half_ = 0.45f; gaze_sweep_ = gaze_sweep_to_ = 0.0f;
    gaze_yaw_ = gaze_yaw_v_ = gaze_pitch_ = gaze_pitch_v_ = gaze_w_ = 0.0f;
    shift_left_ = 3.0f; shift_t_ = shift_ = shift_v_ = 0.0f;
    blink_left_ = 2.0f; blink_t_ = -1.0f; lid_ = 0.0f; blinks_ = 0;
    eye_yaw_ = eye_pitch_ = dart_yaw_ = dart_pitch_ = 0.0f; dart_left_ = 1.0f;
    jaw_ = jaw_v_ = brow_ = brow_v_ = 0.0f; gaze_want_yaw_ = gaze_want_pitch_ = 0.0f;
    style_ = PersonStyle{}; style_of_ = 0; action_mirror_ = false;
    layer_clip_ = kNoClip; layer_mask_ = LayerMask::UpperBody; layer_t_ = layer_w_ = layer_wv_ = layer_eff_ = 0.0f;
    gest_ = Gesture::None; gest_side_ = 1; gest_t_ = gest_dur_ = gest_w_ = 0.0f; fidget_left_ = 6.0f;
    gesture_seq_ = look_id_ = 0; gest_look_yaw_ = gest_look_pitch_ = hips_shift_ = 0.0f;
    layer_prev_eff_ = gest_prev_w_ = 0.0f;
    layer_pa_.assign(pose_.size(), glm::quat(1.0f, 0.0f, 0.0f, 0.0f)); layer_pb_ = layer_pa_;
    weapon_fire_offset_ = glm::mat4(1.0f);
    stand_hips_ = set_.stand_hips;
    weapon_offset_ = glm::mat4(1.0f);
    dbg_ = AnimDebug{};
}

glm::quat CharacterAnimator::model_rotation(int joint) const {
    const std::vector<int>& par = set_.runtime->skeleton().parents;
    glm::quat q = pose_[size_t(joint)].r;
    for (int p = par[size_t(joint)]; p >= 0; p = par[size_t(p)]) q = pose_[size_t(p)].r * q;
    return q;
}

void CharacterAnimator::rotate_in_model(int joint, const glm::quat& q) {
    const int p = set_.runtime->skeleton().parents[size_t(joint)];
    const glm::quat P = p >= 0 ? model_rotation(p) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    pose_[size_t(joint)].r = glm::normalize(glm::inverse(P) * q * P * pose_[size_t(joint)].r);
}

float CharacterAnimator::ground_at(const CharacterAnimationInput& in, glm::vec3 m) const {
    if (!in.ground_height) return 0.0f;
    const glm::vec3 w = in.position + model_to_world_vec(m, in.facing_yaw);
    const float g = in.ground_height(in.ground_user, w.x, w.z) - in.position.y;
    return std::isfinite(g) ? g : 0.0f;
}

// Joints a -> b -> c: c goes to `target` as far as the limb reaches. Each bone is turned the
// shortest way from where it pointed to where the solution puts it, so the twist the pose had
// along the bones is kept (no rolled thighs or forearms); c keeps its orientation. The middle
// joint bends toward `pole` (model space), or the way it already bends if none is given.
void CharacterAnimator::fit_limb(int a, int b, int c, glm::vec3 target, const glm::vec3* pole, bool from_rest) {
    const glm::vec3 pa(model_[size_t(a)][3]), pb0(model_[size_t(b)][3]), pc0(model_[size_t(c)][3]);
    const float l1 = glm::length(pb0 - pa), l2 = glm::length(pc0 - pb0);
    if (l1 < 1e-5f || l2 < 1e-5f) return;
    glm::vec3 d = target - pa;
    float dist = glm::length(d);
    if (dist < 1e-4f) return;
    const glm::vec3 dir = d / dist;
    dist = std::clamp(dist, std::abs(l1 - l2) + 0.01f, (l1 + l2) * 0.998f);
    const float ca = std::clamp((l1 * l1 + dist * dist - l2 * l2) / (2.0f * l1 * dist), -1.0f, 1.0f);
    glm::vec3 pp = pole ? *pole : (pb0 - pa);
    pp -= dir * glm::dot(pp, dir);
    if (glm::dot(pp, pp) < 1e-8f) {
        pp = glm::vec3(0.0f, 0.0f, -1.0f) - dir * glm::dot(glm::vec3(0.0f, 0.0f, -1.0f), dir);
        if (glm::dot(pp, pp) < 1e-8f) pp = glm::vec3(1.0f, 0.0f, 0.0f) - dir * dir.x;
    }
    pp = glm::normalize(pp);
    const glm::vec3 pb = pa + dir * (l1 * ca) + pp * (l1 * std::sqrt(std::max(0.0f, 1.0f - ca * ca)));
    const glm::vec3 pc = pa + dir * dist;
    const glm::quat Ra = glm::quat_cast(glm::mat3(model_[size_t(a)])), Rb = glm::quat_cast(glm::mat3(model_[size_t(b)])),
                    Rc = glm::quat_cast(glm::mat3(model_[size_t(c)]));
    const SkeletonDef& sk = set_.runtime->skeleton();
    const int par = sk.parents[size_t(a)];
    const glm::quat P = par >= 0 ? glm::quat_cast(glm::mat3(model_[size_t(par)])) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::quat Ra2, Rb2;
    if (from_rest) {
        // A limb placed from scratch (an arm on a weapon): turned from its rest pose on the body,
        // not from where the capture has it. A running arm swings far behind the body, and the
        // shortest turn from there to the weapon goes over the top one frame and under the next.
        Ra2 = quat_from_to(P * glm::normalize(sk.rest_local(b)), glm::normalize(pb - pa)) * P;
        Rb2 = quat_from_to(Ra2 * glm::normalize(sk.rest_local(c)), glm::normalize(pc - pb)) * Ra2;
    } else {
        // A limb adjusted (a foot moved a little): the capture's own pose, turned the least.
        Ra2 = quat_from_to(glm::normalize(pb0 - pa), glm::normalize(pb - pa)) * Ra;
        Rb2 = quat_from_to(glm::normalize(pc0 - pb0), glm::normalize(pc - pb)) * Rb;
    }
    pose_[size_t(a)].r = glm::normalize(glm::inverse(P) * Ra2);
    pose_[size_t(b)].r = glm::normalize(glm::inverse(Ra2) * Rb2);
    pose_[size_t(c)].r = glm::normalize(glm::inverse(Rb2) * Rc);
    set_.runtime->local_to_model(*ctx_, pose_, model_, a);
}

void CharacterAnimator::update(const CharacterAnimationInput& in) {
    const double t0 = now_ms();
    const IAnimationRuntime& rt = *set_.runtime;
    const float dt = std::clamp(in.dt, 0.0f, 0.1f);
    time_ += dt;
    dbg_.lod = in.lod;
    dbg_.search_ms = 0.0f;

    // ---- how the controller moves ----
    if (!started_ || in.teleported) {
        prev_yaw_ = in.facing_yaw;
        prev_vel_ = in.velocity;
        yaw_rate_ = 0.0f;
        accel_ = glm::vec3(0.0f);
        for (Lock& l : lock_) l = Lock{};
        if (!started_) { shot_seq_ = in.shot_seq; reload_seq_ = in.reload_seq; hit_seq_ = in.hit_seq; held_ = in.held; gesture_seq_ = in.gesture_seq; look_id_ = in.look_id; }
    }
    if (dt > 1e-4f) {
        const float yr = std::clamp(wrap_angle(in.facing_yaw - prev_yaw_) / dt, -12.0f, 12.0f);
        yaw_rate_ += (yr - yaw_rate_) * (1.0f - std::exp(-dt * 10.0f));
        const glm::vec3 a = (in.velocity - prev_vel_) / dt;
        accel_ += (a - accel_) * (1.0f - std::exp(-dt * 12.0f));
    }
    prev_yaw_ = in.facing_yaw;
    prev_vel_ = in.velocity;
    const glm::vec2 v_local = world_to_facing(glm::vec2(in.velocity.x, in.velocity.z), in.facing_yaw);
    const glm::vec2 a_local = world_to_facing(glm::vec2(accel_.x, accel_.z), in.facing_yaw);
    const float speed = glm::length(v_local);
    dbg_.speed = speed;

    // ---- weapon timers run at every level of detail: events are never lost ----
    if (in.shot_seq != shot_seq_) { shot_seq_ = in.shot_seq; recoil_t_ = weapon.recoil_duration; since_shot_ = 0.0f; }
    since_shot_ += dt;
    if (!started_) aim_w_ = in.aiming ? 1.0f : 0.0f;
    // (up to the aim quickly, down from it at ease)
    aim_w_ += ((in.aiming ? 1.0f : 0.0f) - aim_w_) * (1.0f - std::exp(-dt / (in.aiming ? 0.07f : 0.14f)));
    recoil_t_ = std::max(0.0f, recoil_t_ - dt);
    if (in.reload_seq != reload_seq_) { reload_seq_ = in.reload_seq; reload_t_ = 0.0f; }
    if (reload_t_ >= 0.0f) {
        if (!in.reloading || in.held != HeldItem::Rifle) reload_t_ = -1.0f;            // interrupted
        else {
            reload_t_ += dt;
            reload_u_ = std::clamp(reload_t_ / std::max(weapon.reload_time, 0.1f), 0.0f, 1.0f);
            if (reload_t_ >= weapon.reload_time) reload_t_ = -1.0f;
        }
    }
    reload_w_ += ((reload_t_ >= 0.0f ? 1.0f : 0.0f) - reload_w_) * (1.0f - std::exp(-dt * 14.0f));
    if (in.held != held_) { held_ = in.held; equip_t_ = 0.0f; }
    equip_t_ = std::min(equip_t_ + dt / 0.22f, 1.0f);

    if (in.lod >= 3) {
        // Not seen: no pose. The playback position keeps moving so the pose is coherent when it is needed again.
        if (match_.frame >= 0 && set_.db.ok()) {
            const MotionRange& r = set_.db.range_of(uint32_t(match_.frame));
            match_.frac += dt * set_.db.sample_rate() * rate_;
            const int whole = int(std::floor(match_.frac));
            match_.frame = std::min(match_.frame + whole, int(r.first + r.count) - 2);
            match_.frac -= float(whole);
            match_.since_search += dt;
            match_.since_switch += dt;
        }
        air_.phase = (!in.grounded && !in.swimming) ? air_.phase : AirPhase::Ground;
        dbg_.mode = "culled";
        dbg_.update_ms = float(now_ms() - t0);
        return;
    }

    if (in.seated) seated(in);
    else if (in.action_clip >= 0 && in.action_clip < rt.num_clips()) action(in);
    else {
    float ground_dist = 1e9f;
    if (!in.grounded && in.ground_height) ground_dist = std::max(0.0f, -ground_at(in, glm::vec3(0.0f)));

    lower_body(in, v_local, speed, yaw_rate_, ground_dist);
    pelvis_and_warp(in, v_local, speed, yaw_rate_, a_local);
    upper_body(in);
    rt.local_to_model(*ctx_, pose_, model_);
    if (tuning.foot_ik && in.lod <= 1) feet(in, v_local, speed, glm::length(a_local));
    arms(in);
    rt.local_to_model(*ctx_, pose_, model_);
    }

    // A pose that is not finite is never handed on: the last good one stands in, and it is counted.
    if (!pose_is_valid(pose_) || !pose_is_valid(model_)) {
        ++dbg_.invalid_poses;
        if (have_good_) { pose_ = good_; model_ = good_model_; }
        else { rt.rest_pose(pose_); rt.local_to_model(*ctx_, pose_, model_); }
        inert_.reset(rt.skeleton().size());
    } else {
        good_ = pose_;
        good_model_ = model_;
        have_good_ = true;
    }
    dbg_.inertial_energy = inert_.energy();
    dbg_.update_ms = float(now_ms() - t0);
}

// On a seat (CharacterAnimationInput::seated): from the rest pose. Where the car says where its
// things are (seat_wheel, seat_feet, seat_back - jeep.cpp reads them from the car's file, which
// has them measured from its model) the body is fitted to THEM: the back against the backrest,
// the heels on the floor and the right foot on the accelerator, the hands round the rim.
// Without them: the thighs forward, the shins down, the wrists where the caller says.
void CharacterAnimator::seated(const CharacterAnimationInput& in) {
    const IAnimationRuntime& rt = *set_.runtime;
    const Rig& rig = rt.rig();
    const SkeletonDef& sk = rt.skeleton();
    rt.rest_pose(pose_);
    rt.local_to_model(*ctx_, pose_, model_);
    const glm::vec3 X(1.0f, 0.0f, 0.0f), Y(0.0f, 1.0f, 0.0f);
    const int hands[2] = { rig[Role::LeftHand], rig[Role::RightHand] };
    glm::vec3 fingers[2]{};
    glm::quat rest_rot[2]{};
    for (int s = 0; s < 2; ++s) {
        if (hands[s] < 0) continue;
        const glm::vec3 d = sk.rest_tail[size_t(hands[s])] - sk.rest_pos[size_t(hands[s])];
        fingers[s] = glm::dot(d, d) > 1e-8f ? glm::normalize(d) : glm::vec3(s == 0 ? -1.0f : 1.0f, 0.0f, 0.0f);
        rest_rot[s] = model_rotation(hands[s]);
    }
    auto turn = [&](Role r, float deg, const glm::vec3& axis) {
        const int j = rig[r];
        if (j >= 0) rotate_in_model(j, glm::angleAxis(glm::radians(deg), axis));
    };
    // (about +X a bone that hangs down swings forward, one that stands up leans back)
    const float legs = std::clamp(in.seat_legs, 0.0f, 1.0f);
    const float thigh = glm::mix(78.0f, 84.0f, legs), shin = glm::mix(-62.0f, -30.0f, legs);
    float back = in.seat_back >= 0.0f ? in.seat_back : glm::mix(9.0f, 22.0f, legs);
    const Role arm[2][3] = { { Role::LeftArm, Role::LeftForeArm, Role::LeftHand }, { Role::RightArm, Role::RightForeArm, Role::RightHand } };
    const Role leg[2][3] = { { Role::LeftUpLeg, Role::LeftLeg, Role::LeftFoot }, { Role::RightUpLeg, Role::RightLeg, Role::RightFoot } };
    // The feet at rest: how high the ankle stands over the sole, which way the sole points.
    glm::quat foot_rest[2]{};
    glm::vec3 sole0[2]{};
    float ankle_h[2] = { 0.09f, 0.09f };
    for (int s = 0; s < 2; ++s) {
        const int f = rig[leg[s][2]];
        if (f < 0) continue;
        foot_rest[s] = model_rotation(f);
        ankle_h[s] = std::clamp(model_[size_t(f)][3].y, 0.05f, 0.16f);
        glm::vec3 d = sk.rest_tail[size_t(f)] - sk.rest_pos[size_t(f)];
        d.y = 0.0f;
        sole0[s] = glm::dot(d, d) > 1e-8f ? glm::normalize(d) : glm::vec3(0.0f, 0.0f, -1.0f);
    }
    // WHERE THE HANDS GO. On a wheel (seat_wheel): the palm on the outside of the rim, the
    // knuckles along it, the fingers forward round it - the wrist is worked back from there, as
    // for a weapon's grip (arms()). Otherwise the caller's points are the wrists (on the knees).
    glm::vec3 wrist[2] = { in.seat_hand_l, in.seat_hand_r }, a1[2]{}, n1[2]{};
    bool grip[2] = { false, false };
    if (in.seat_wheel && glm::dot(in.seat_axis, in.seat_axis) > 1e-6f) {
        const glm::vec3 ax = glm::normalize(in.seat_axis);
        for (int s = 0; s < 2; ++s) {
            const AnimationSet::Hand& H = set_.hand[s];
            if (!H.ok) continue;
            const glm::vec3 P = s == 0 ? in.seat_hand_l : in.seat_hand_r;
            glm::vec3 er = (P - in.seat_hub) - ax * glm::dot(P - in.seat_hub, ax);
            if (glm::dot(er, er) < 1e-8f) continue;
            er = glm::normalize(er);
            glm::vec3 up = glm::cross(ax, er);                  // (along the rim, the way that goes up)
            if (up.y < 0.0f) up = -up;
            a1[s] = glm::normalize(-ax + up * 0.30f);
            n1[s] = glm::normalize(-er - a1[s] * glm::dot(-er, a1[s]));
            wrist[s] = P + er * 0.016f - a1[s] * (H.palm * 0.80f) - n1[s] * 0.028f;
            grip[s] = true;
        }
    }
    // THE TRUNK: against the backrest - unless the arms then do not reach the wheel with the
    // elbows still bent: then it comes forward off the backrest as far as it takes (a driver
    // whose wheel is far away sits up to it). Twice is enough: the reach changes little.
    auto trunk = [&](float lean) {
        rt.rest_pose(pose_);
        rt.local_to_model(*ctx_, pose_, model_);
        turn(Role::Spine, lean, X);
        rt.local_to_model(*ctx_, pose_, model_);
    };
    trunk(back);
    for (int it = 0; it < 3 && (grip[0] || grip[1]); ++it) {
        float short_by = 0.0f, lever = 0.4f;
        for (int s = 0; s < 2; ++s) {
            const int a = rig[arm[s][0]], b = rig[arm[s][1]], c = rig[arm[s][2]], sp = rig[Role::Spine];
            if (!grip[s] || a < 0 || b < 0 || c < 0) continue;
            const glm::vec3 pa(model_[size_t(a)][3]);
            const float reach = (glm::length(glm::vec3(model_[size_t(b)][3]) - pa) + glm::length(glm::vec3(model_[size_t(c)][3]) - glm::vec3(model_[size_t(b)][3]))) * 0.95f;
            short_by = std::max(short_by, glm::length(wrist[s] - pa) - reach);
            if (sp >= 0) lever = std::max(pa.y - model_[size_t(sp)][3].y, 0.2f);
        }
        if (short_by < 0.004f) break;
        const float fwd = glm::degrees(short_by / lever) * 1.15f;
        const float to = std::max(back - fwd, -8.0f);
        if (to >= back - 0.05f) break;
        back = to;
        trunk(back);
    }
    const float ph = float(in.variation % 1024u) * 0.0061f;
    // Somebody who sits somewhere (not at a wheel) and stays: the life layers - the posture of
    // this person, breathing, the look (life_layers.cpp). At a wheel: as it was.
    const bool alive = in.life_on_clip && !in.seat_wheel;
    if (alive) {
        turn(Role::Neck, -back * 0.8f, X);                                     // the head stays level
        life_layers(in, 1);
        face_layer(in);
    } else {
    turn(Role::Spine1, 0.5f * std::sin(time_ * 1.6f + ph), X);                 // breathing
    turn(Role::Neck, -back * 0.8f, X);                                         // the head stays level
    // (the head: into the bend with the wheel, and never quite still)
    turn(Role::Neck, -in.seat_steer * 7.0f + 1.5f * std::sin(time_ * 0.31f + ph), Y);
    turn(Role::Head, -in.seat_steer * 11.0f + 2.5f * std::sin(time_ * 0.23f + ph * 2.0f), Y);
    turn(Role::Head, 1.2f * std::sin(time_ * 0.41f + ph * 3.0f), X);
    }
    rt.local_to_model(*ctx_, pose_, model_);
    // THE LEGS. Where the feet have places (seat_feet): the heel on the floor, the sole from
    // there to where the ball of the foot presses, the knee up between. A leg too short for
    // its pedal keeps its heel on the floor and stops short of it (never a straight leg).
    if (in.seat_feet) {
        for (int s = 0; s < 2; ++s) {
            const int a = rig[leg[s][0]], b = rig[leg[s][1]], c = rig[leg[s][2]];
            if (a < 0 || b < 0 || c < 0) continue;
            const glm::vec3 heel = s == 0 ? in.seat_heel_l : in.seat_heel_r, ball = s == 0 ? in.seat_ball_l : in.seat_ball_r;
            glm::vec3 f1 = ball - heel;
            f1 = glm::dot(f1, f1) > 1e-8f ? glm::normalize(f1) : glm::vec3(0.0f, 0.0f, -1.0f);
            glm::vec3 up = glm::cross(X, f1);
            up = glm::dot(up, up) > 1e-8f ? glm::normalize(up) : Y;
            if (up.y < 0.0f) up = -up;
            glm::vec3 ankle = heel + f1 * 0.055f + up * ankle_h[s];
            const glm::vec3 pa(model_[size_t(a)][3]);
            const float len = glm::length(glm::vec3(model_[size_t(b)][3]) - pa) + glm::length(glm::vec3(model_[size_t(c)][3]) - glm::vec3(model_[size_t(b)][3]));
            const float most = len * 0.965f;
            if (glm::length(ankle - pa) > most) {
                const float dx = ankle.x - pa.x, dy = ankle.y - pa.y;
                ankle.z = pa.z - std::sqrt(std::max(most * most - dx * dx - dy * dy, 0.01f));      // (forward is -z)
            }
            const glm::vec3 pole(s == 0 ? -0.22f : 0.22f, 1.0f, -0.10f);
            fit_limb(a, b, c, ankle, &pole);
            rt.local_to_model(*ctx_, pose_, model_);
            const glm::mat3 F0(sole0[s], Y, glm::cross(sole0[s], Y)), F1(f1, up, glm::cross(f1, up));
            const glm::quat want = glm::quat_cast(F1 * glm::transpose(F0)) * foot_rest[s];
            pose_[size_t(c)].r = glm::normalize(glm::inverse(model_rotation(b)) * want);
            rt.local_to_model(*ctx_, pose_, model_);
        }
    } else {
        // (how this person places the feet: wider or closer, out in front or tucked under the seat)
        const float own = std::clamp(in.individuality, 0.0f, 1.0f);
        const float apart = 7.0f + std::clamp(style_.feet_apart * 130.0f, -5.0f, 6.0f) * own;
        const float out = std::clamp(style_.feet_fwd * 150.0f, -11.0f, 15.0f) * own;
        turn(Role::LeftUpLeg, thigh, X);  turn(Role::RightUpLeg, thigh, X);
        turn(Role::LeftUpLeg, apart, Y);  turn(Role::RightUpLeg, -apart, Y);       // knees a little apart
        rt.local_to_model(*ctx_, pose_, model_);
        turn(Role::LeftLeg, shin + out, X); turn(Role::RightLeg, shin + out * 0.8f, X);
        rt.local_to_model(*ctx_, pose_, model_);
        turn(Role::LeftFoot, -(thigh + shin) * 0.6f, X); turn(Role::RightFoot, -(thigh + shin) * 0.6f, X);
        rt.local_to_model(*ctx_, pose_, model_);
    }
    // THE ARMS: the elbows down and out.
    for (int s = 0; s < 2; ++s) {
        const int a = rig[arm[s][0]], b = rig[arm[s][1]], c = rig[arm[s][2]];
        if (a < 0 || b < 0 || c < 0) continue;
        const glm::vec3 pole(s == 0 ? -0.55f : 0.55f, -1.0f, 0.25f);
        // (an arm a clip layer has - somebody on a bench with a phone - belongs to the layer, not to the knee)
        const float layered = grip[s] ? 0.0f : layer_eff_ * set_.mask[int(layer_mask_)][size_t(a)];
        const glm::quat la = pose_[size_t(a)].r, lb = pose_[size_t(b)].r, lc = pose_[size_t(c)].r;
        fit_limb(a, b, c, wrist[s], &pole, grip[s]);
        rt.local_to_model(*ctx_, pose_, model_);
        if (grip[s]) {
            // The hand round the rim (its rest frame on to the grip's, as arms() does), the fingers closed.
            const AnimationSet::Hand& H = set_.hand[s];
            const glm::mat3 F0(H.a0, H.n0, H.b0), F1(a1[s], n1[s], glm::cross(a1[s], n1[s]));
            pose_[size_t(c)].r = glm::normalize(glm::inverse(glm::quat_cast(glm::mat3(model_[size_t(b)]))) * glm::quat_cast(F1 * glm::transpose(F0)));
            if (in.lod < 2) {
                static const float kCurl[3] = { 1.10f, 1.35f, 0.85f };
                for (int f = 0; f < 4; ++f)
                    for (int j = 0; j < 3; ++j)
                        if (H.finger[f][j] >= 0) pose_[size_t(H.finger[f][j])].r = glm::angleAxis(kCurl[j], H.b0);
                const float sx = s == 0 ? -1.0f : 1.0f;
                for (int j = 0; j < 3; ++j)
                    if (H.thumb[j] >= 0) pose_[size_t(H.thumb[j])].r = glm::angleAxis(-sx * (j == 0 ? 0.85f : 0.0f), H.a0) * glm::angleAxis(j == 0 ? 0.35f : 0.55f, H.b0);
            }
        } else {
            // A hand that lies somewhere (a knee): its fingers forward, down a little and inward.
            const glm::vec3 cur = glm::normalize(model_rotation(c) * glm::inverse(rest_rot[s]) * fingers[s]);
            const glm::vec3 want = in.seat_desk ? glm::normalize(glm::vec3(s == 0 ? 0.20f : -0.20f, -0.10f, -0.97f))
                                 : in.seat_wheel || in.seat_feet ? glm::normalize(glm::vec3(s == 0 ? 0.22f : -0.22f, -0.20f, -0.95f))
                                                                 : glm::normalize(glm::vec3(s == 0 ? 0.40f : -0.40f, 0.55f, -0.70f));
            const glm::vec3 ax = glm::cross(cur, want);
            const float sl = glm::length(ax), cl = glm::dot(cur, want);
            if (sl > 1e-5f) rotate_in_model(c, glm::angleAxis(std::atan2(sl, cl), ax / sl));
            // At a desk: the fingers a little bent over the keys - and, typing, each going down
            // and up at its own moment, in bursts (a pause now and then, as somebody thinks).
            const AnimationSet::Hand& H = set_.hand[s];
            if (in.seat_desk && H.ok && in.lod < 2) {
                const float busy = std::clamp(in.seat_typing, 0.0f, 1.0f) * (0.5f + 0.5f * std::sin(time_ * 0.9f + ph * 5.0f) > 0.22f ? 1.0f : 0.0f);
                for (int f = 0; f < 4; ++f) {
                    const float tap = busy * std::max(0.0f, std::sin(time_ * (9.0f + 2.3f * float(f)) + float(f) * 1.9f + float(s) * 2.6f + ph * 3.0f));
                    for (int j = 0; j < 3; ++j)
                        if (H.finger[f][j] >= 0) pose_[size_t(H.finger[f][j])].r = glm::angleAxis((j == 0 ? 0.22f + 0.30f * tap : 0.34f + 0.10f * tap), H.b0);
                }
            }
        }
        if (layered > 1e-3f) {
            pose_[size_t(a)].r = glm::normalize(glm::slerp(pose_[size_t(a)].r, la, layered));
            pose_[size_t(b)].r = glm::normalize(glm::slerp(pose_[size_t(b)].r, lb, layered));
            pose_[size_t(c)].r = glm::normalize(glm::slerp(pose_[size_t(c)].r, lc, layered));
        }
        rt.local_to_model(*ctx_, pose_, model_);
    }
    // (a hand to the head, a look at the wrist: over the arms as they were just placed)
    if (alive && in.lod <= 1) gesture_layer(in, std::clamp(in.dt, 0.0f, 0.1f), in.dying <= 0.0f, true, false);
    // (through the inertialiser: no pop from the end of "sitting down" into this, or out of it)
    src_ = pose_;
    if (last_mode_ != 1 && started_ && !in.teleported) inert_.transition(src_, src_, 0.0f);
    last_mode_ = 1;
    inert_.apply(src_, std::clamp(in.dt, 0.0f, 0.1f), tuning.inertial_halflife, pose_);
    started_ = true;
    rt.local_to_model(*ctx_, pose_, model_);
    dbg_.mode = "seated";
}

// An action clip (CharacterAnimationInput::action_clip): sampled at the caller's time, with its
// own travel from its first frame; inertialised in; the hips warped to where they must end.
void CharacterAnimator::action(const CharacterAnimationInput& in) {
    const IAnimationRuntime& rt = *set_.runtime;
    const Rig& rig = rt.rig();
    const SkeletonDef& sk = rt.skeleton();
    const ClipInfo& ci = rt.clip(in.action_clip);
    const float dt = std::clamp(in.dt, 0.0f, 0.1f);
    const float last = std::max(ci.duration - 1e-3f, 0.0f);
    const float t = std::clamp(in.action_time, 0.0f, last);
    const int hips = rig[Role::Hips];
    auto sample_at = [&](int slot, float tt, std::vector<Transform>& p) {
        rt.sample(*ctx_, slot, in.action_clip, tt, p);
        p[0] = rt.root_motion(*ctx_, in.action_clip, 0.0f, tt);          // (the clip's travel since its first frame)
        if (in.action_mirror) mirror_pose(rig, p, scratch_);             // (left for right, its travel too)
        if (in.action_in_place) p[0] = Transform{};
    };
    sample_at(0, t, src_);
    const float u = ci.duration > 0.05f ? t / ci.duration : 1.0f;
    if (in.action_end_at_origin) {
        const Transform all = rt.root_motion(*ctx_, in.action_clip, 0.0f, last);
        const float k = smooth01(u);
        src_[0].t.x -= (in.action_mirror ? -all.t.x : all.t.x) * k; src_[0].t.z -= all.t.z * k;
    }
    const bool looped = last_mode_ == 2 && action_clip_ == in.action_clip && t < action_last_t_ - 0.25f;
    action_last_t_ = t;
    if (looped && started_) inert_.transition(src_, src_, 0.0f);
    if (last_mode_ != 2 || action_clip_ != in.action_clip || action_mirror_ != in.action_mirror) {
        action_mirror_ = in.action_mirror;
        const float h = 1.0f / 30.0f;
        sample_at(1, std::min(t + h, last), next_);
        if (started_ && !in.teleported) inert_.transition(src_, next_, h);
        action_clip_ = in.action_clip;
        last_mode_ = 2;
        if (hips >= 0) {
            sample_at(1, in.action_warp_from_start ? 0.0f : last, next_);
            rt.local_to_model(*ctx_, next_, model_);
            action_ref_hips_ = glm::vec3(model_[size_t(hips)][3]);
        }
    }
    inert_.apply(src_, dt, tuning.inertial_halflife, pose_);
    started_ = true;
    rt.local_to_model(*ctx_, pose_, model_);
    // The feet as the clip has them, before anything moves the hips: a foot that stands (an
    // ankle near the ground) stays there through the shifts of weight of the life layers and
    // through the warp.
    const Role leg[2][3] = { { Role::LeftUpLeg, Role::LeftLeg, Role::LeftFoot }, { Role::RightUpLeg, Role::RightLeg, Role::RightFoot } };
    glm::vec3 foot[2]{};
    bool stands[2] = { false, false };
    for (int s = 0; s < 2; ++s) {
        const int f = rig[leg[s][2]];
        if (f < 0) continue;
        foot[s] = glm::vec3(model_[size_t(f)][3]);
        stands[s] = foot[s].y < 0.24f;
    }
    // THE HANDS. The action clips come from a body without fingers (22 joints): as sampled,
    // every hand is the rest pose's - flat, the fingers straight and spread, a doll's hand on a
    // body that walks. A hand that holds nothing hangs with its fingers a little curled (as a
    // free arm's does in arms()); a gesture then does what it likes with its own hand.
    if (in.held == HeldItem::None && in.lod < 2) {
        for (int side = 0; side < 2; ++side) {
            const AnimationSet::Hand& H = set_.hand[side];
            if (!H.ok) continue;
            const float sx = side == 0 ? -1.0f : 1.0f;
            static const float kCurl[3] = { 0.38f, 0.48f, 0.32f };
            for (int f = 0; f < 4; ++f)
                for (int j = 0; j < 3; ++j)
                    if (H.finger[f][j] >= 0) pose_[size_t(H.finger[f][j])].r = glm::angleAxis(kCurl[j] * (1.0f + 0.12f * float(f)), H.b0);
            for (int j = 0; j < 3; ++j)
                if (H.thumb[j] >= 0)
                    pose_[size_t(H.thumb[j])].r = glm::angleAxis(-sx * (j == 0 ? 0.35f : 0.0f), H.a0) * glm::angleAxis(sx * (j == 0 ? 0.10f : 0.30f) * 0.3f, H.n0);
        }
    }
    // The life layers (life_layers.cpp): posture always; the rest for somebody who stays.
    life_layers(in, 2);
    if (in.life_on_clip) face_layer(in);
    rt.local_to_model(*ctx_, pose_, model_);
    glm::vec3 delta(0.0f);
    const bool warp = in.action_warp && hips >= 0;
    if (warp) {
        float w = smooth01(u);
        if (in.action_warp_from_start) w = 1.0f - w;
        if (in.action_warp_full) w = 1.0f;
        delta = (in.action_hips - action_ref_hips_) * w;
        const int par = sk.parents[size_t(hips)];
        const glm::mat3 P = par >= 0 ? glm::mat3(model_[size_t(par)]) : glm::mat3(1.0f);
        pose_[size_t(hips)].t += glm::transpose(P) * delta;
        rt.local_to_model(*ctx_, pose_, model_);
    }
    if (warp || (std::abs(hips_shift_) > 1e-4f && !in.action_in_place)) {
        // (on a seat for good: how this person places the feet)
        const float own = in.action_warp_full ? std::clamp(in.individuality, 0.0f, 1.0f) : 0.0f;
        for (int s = 0; s < 2; ++s) {
            const int a = rig[leg[s][0]], b = rig[leg[s][1]], c = rig[leg[s][2]];
            if (!stands[s] || a < 0 || b < 0 || c < 0) continue;
            // (A foot stays where the clip put it while the hips are moved a little - sitting
            //  down on a chair a hand further back. Moved far - into a car seat a metre to
            //  the side - the feet go with the body: pinned, a leg stayed outside the car,
            //  through the door.)
            const float go = glm::smoothstep(0.18f, 0.55f, glm::length(glm::vec2(delta.x, delta.z)));
            const glm::vec3 place((s == 0 ? -1.0f : 1.0f) * style_.feet_apart * own, 0.0f, -style_.feet_fwd * own);
            fit_limb(a, b, c, foot[s] + delta * go + place, nullptr);
            rt.local_to_model(*ctx_, pose_, model_);
        }
    }
    dbg_.mode = "action";
    dbg_.clip = in.action_clip;
    dbg_.clip_time = t;
}

// An action clip's travel from its first frame to `t` (for whoever places the model: how far
// it goes, and how far it turns).
Transform CharacterAnimator::action_root(ClipId clip, float t) const {
    return set_.runtime->root_motion(*ctx_, clip, 0.0f, t);
}

void CharacterAnimator::lower_body(
const CharacterAnimationInput& in, glm::vec2 v_local, float speed, float yaw_rate, float ground_dist) {

    const IAnimationRuntime& rt = *set_.runtime;
    const MotionDatabase& db = set_.db;
    const float dt = std::clamp(in.dt, 0.0f, 0.1f);
    const bool airborne = !in.grounded && !in.swimming;
    const glm::vec2 wish(in.desired_velocity.x, in.desired_velocity.z);
    // Game speeds are mapped onto what the capture sustains in the direction of travel (its
    // back-pedal is slower than its run): asking for more finds single fast moments, never a
    // whole step, and the search would jump back to that moment for ever.
    MatcherParams mp = tuning.matcher;
    {
        const float ang = std::atan2(v_local.x, v_local.y);                    // 0 = backward, pi = forward
        const float a = std::abs(snap_on_ ? snap_axis_ : ang);
        const int axis = a > 2.36f ? 0 : a > 0.79f ? 1 : 2;
        const float cruise = db.cruise_speed(in.crouch > 0.5f, axis);
        if (cruise > 0.0f) {
            mp.run_speed = cruise;
            mp.walk_speed = std::min(mp.walk_speed, 0.6f * cruise);
        }
    }
    const float k = speed > 1e-3f ? capture_speed(speed, mp) / speed : 1.0f;

    AirborneInput ai;
    ai.dt = dt;
    ai.airborne = airborne && tuning.airborne_pose_strength > 0.0f;
    ai.vertical_speed = in.velocity.y;
    ai.ground_distance = ground_dist;
    ai.gravity = in.gravity;
    ai.velocity = v_local;
    ai.wants_move = in.wants_jump || glm::length(wish) > 1.0f;
    const AirborneOutput ao = update_airborne(db, air_, ai, tuning.air, k);
    dbg_.air_phase = air_.phase;
    dbg_.time_to_contact = air_.time_to_contact;
    dbg_.flights = air_.flights;

    bool switched = false;
    if (ao.landed) {
        // The pelvis gives under the landing: a visual dip only, the controller is not slowed.
        const float peak = std::min(ao.impact * tuning.land_dip, tuning.max_land_dip);
        const float y = 4.0f * 0.69314718f / kDipHalflife * 0.5f;
        dip_v_ += peak * y * 2.7182818f;
        dbg_.impact = ao.impact;
    }
    if (ao.active) {
        switched = ao.started;
        match_.frame = ao.frame;
        match_.frac = ao.frac;
        match_.mirrored = ao.mirrored;
        cur_ = matcher_output(db, match_);
        stand_hips_ = ao.stand_hips;
        dbg_.mode = air_.phase == AirPhase::Land ? "land" : "air";
    } else {
        TrajectoryInput ti;
        ti.velocity = glm::vec2(in.velocity.x, in.velocity.z);
        ti.desired_velocity = wish;
        ti.facing_yaw = in.facing_yaw;
        ti.yaw_rate = yaw_rate;
        ti.grounded = !airborne;
        const float hz[MotionSchema::kHorizons] = { db.horizon(0), db.horizon(1), db.horizon(2) };
        MatcherInput mi;
        mi.traj = predict_trajectory(ti, tuning.move, hz);
        mi.velocity = v_local;
        // Diagonals: the capture has forward, sideways and backward takes and nothing in between.
        // A request in between is turned onto the nearest of those for the search; the orientation
        // warp then turns the legs back to the way the character really goes. The direction in use
        // is kept until the request is well past the middle (W+D is exactly the middle).
        if (tuning.orient_warp && speed > 1.0f) {
            const float kQuarter = 1.57079633f;
            const float ang = std::atan2(v_local.x, v_local.y);
            float axis = std::round(ang / kQuarter) * kQuarter;
            if (snap_on_ && std::abs(wrap_angle(ang - snap_axis_)) < 0.96f) axis = snap_axis_;       // 55 degrees
            // A fast sidestep is not something legs do: the capture's sidestep is 1.4 m/s, and at
            // 8 the feet would have to cross four times a second, a stride each. At speed the
            // character runs instead, on legs turned the way it goes, the torso twisted back
            // onto the aim (forward when it goes sideways or ahead, backward otherwise).
            strafe_run_ = in.crouch < 0.5f && speed > (strafe_run_ ? tuning.strafe_run_speed - 1.0f : tuning.strafe_run_speed);
            if (strafe_run_ && std::abs(std::abs(axis) - kQuarter) < 0.1f) axis = std::abs(ang) > kQuarter - 0.09f ? 2.0f * kQuarter : 0.0f;
            snap_axis_ = axis;
            snap_on_ = true;
            const float w = wrap_angle(ang - axis), c = std::cos(-w), sn = std::sin(-w);
            auto turn = [&](glm::vec2 v) { return glm::vec2(v.x * c + v.y * sn, -v.x * sn + v.y * c); };
            mi.velocity = turn(mi.velocity);
            mi.traj.vel_end = turn(mi.traj.vel_end);
            for (int h = 0; h < MotionSchema::kHorizons; ++h) mi.traj.pos[h] = turn(mi.traj.pos[h]);
        } else {
            snap_on_ = false;
            strafe_run_ = false;
        }
        mi.require_tags = in.crouch > 0.5f ? uint32_t(kTagCrouch) : uint32_t(kTagLocomotion);
        // Standing still with no input is the idle, nothing else: a walk frozen in mid-step is not standing.
        if (speed < 0.25f && glm::length(wish) < 0.1f && !in.swimming) mi.require_tags |= kTagIdle;
        mi.rate = rate_;
        mi.force_search = ao.finished;
        if (in.lod == 1) { mp.search_interval *= 3.0f; mp.change_interval *= 3.0f; }
        else if (in.lod >= 2) { mp.search_interval = 0.5f; mp.change_interval = 0.25f; }
        const double s0 = now_ms();
        cur_ = update_matcher(db, match_, mi, mp, dt);
        if (cur_.searched) dbg_.search_ms = float(now_ms() - s0);
        switched = cur_.switched || ao.finished;
        dbg_.traj = mi.traj;
        dbg_.mode = in.swimming ? "swim" : "ground";
    }
    dbg_.clip = cur_.clip;
    dbg_.clip_time = cur_.time;
    dbg_.mirrored = cur_.mirrored;
    dbg_.match = match_.last_match;
    dbg_.current_cost = match_.last_current_cost;
    dbg_.fallback = match_.last_fallback;
    dbg_.searches = match_.searches;
    dbg_.switches = match_.switches;
    dbg_.contact[0] = cur_.contact[0];
    dbg_.contact[1] = cur_.contact[1];
    if (cur_.clip < 0) { rt.rest_pose(pose_); return; }

    const int hips = rt.rig()[Role::Hips];
    const bool jump_clip = match_.frame >= 0 && (db.range_of(uint32_t(match_.frame)).tags & kTagJump) != 0;
    auto prepare = [&](std::vector<Transform>& p) {
        if (cur_.mirrored) mirror_pose(rt.rig(), p, scratch_);
        p[0] = Transform{};                                     // the controller owns where the character is and faces
        // In a captured jump the performer really leaves the ground. The controller already
        // carries the body through the flight, so the capture's own rise is taken out: the hips
        // stay at standing height and keep half of the dip of the push-off and the landing.
        if (jump_clip) p[size_t(hips)].t.y = set_.stand_hips + 0.5f * std::min(p[size_t(hips)].t.y - stand_hips_, 0.0f);
    };
    rt.sample(*ctx_, 0, cur_.clip, cur_.time, src_);
    prepare(src_);
    if (last_mode_ != 0) { switched = true; last_mode_ = 0; action_clip_ = kNoClip; }      // (back from a seat or an action)
    if (switched && started_) {
        const float h = 1.0f / db.sample_rate();
        if (ao.active) inert_.transition(src_, src_, 0.0f);     // (a scrubbed flight has no pace of its own)
        else {
            rt.sample(*ctx_, 1, cur_.clip, cur_.time + h, next_);
            prepare(next_);
            inert_.transition(src_, next_, h / std::max(rate_, 0.1f));
        }
    }
    inert_.apply(src_, dt, tuning.inertial_halflife, pose_);
    started_ = true;

    // ---- how the capture is fitted to the real speed and direction ----
    const float as = glm::length(cur_.root_vel);
    float rate_t = 1.0f, stride_t = 1.0f, warp_t = 0.0f;
    if (!ao.active && speed > 0.3f && as > 0.3f) {
        rate_t = std::clamp(speed / as, tuning.min_rate, tuning.max_rate);
        stride_t = std::clamp(speed / (as * rate_t), tuning.min_stride, tuning.max_stride);
        warp_t = std::clamp(wrap_angle(std::atan2(v_local.x, v_local.y) - std::atan2(cur_.root_vel.x, cur_.root_vel.y)),
                            -tuning.max_orient_warp, tuning.max_orient_warp);
    }
    const float e = 1.0f - std::exp(-dt * 10.0f);
    rate_ += (rate_t - rate_) * e;
    stride_ += ((tuning.stride_warp ? stride_t : 1.0f) - stride_) * e;
    warp_ += ((tuning.orient_warp ? warp_t : 0.0f) - warp_) * e;
    dbg_.rate = rate_;
    dbg_.stride = stride_;
    dbg_.warp = warp_;
    dbg_.anim_speed = as;
}

void CharacterAnimator::pelvis_and_warp(const CharacterAnimationInput& in, glm::vec2 v_local, float speed, float, glm::vec2 a_local) {
    const Rig& rig = set_.runtime->rig();
    const float dt = std::clamp(in.dt, 0.0f, 0.1f);
    const int hips = rig[Role::Hips];
    const bool airborne = !in.grounded && !in.swimming;
    const bool crouch_clip = match_.frame >= 0 && (set_.db.range_of(uint32_t(match_.frame)).tags & kTagCrouch) != 0;

    // Crouch depth beyond what the capture has, and the dip of a landing.
    const float drop = std::max(in.stand_eye_height - in.eye_local.y, 0.0f) * (crouch_clip ? 0.50f : 0.62f);
    decay_spring(dip_, dip_v_, kDipHalflife, dt);
    pose_[size_t(hips)].t.y -= drop + std::max(dip_, 0.0f);
    pose_[size_t(hips)].t.z += 0.10f * std::clamp(in.crouch, 0.0f, 1.0f);
    dbg_.pelvis_offset = -(drop + std::max(dip_, 0.0f));
    if (in.dying > 0.0f) {
        const float d = std::clamp(in.dying, 0.0f, 1.0f);
        pose_[size_t(hips)].t.y -= 0.38f * d * d;                       // (slow to start, then it goes)
        pose_[size_t(hips)].t.z += 0.10f * d;
    }

    // Orientation warp: hips and legs turn to the way the character really goes; the spine
    // turns back in three steps so the chest stays on the aim.
    if (std::abs(warp_) > 1e-4f) {
        const glm::quat w = quat_yaw(warp_);
        pose_[size_t(hips)].t = glm::vec3((w * glm::vec3(pose_[size_t(hips)].t.x, 0.0f, pose_[size_t(hips)].t.z)).x, pose_[size_t(hips)].t.y,
                                          (w * glm::vec3(pose_[size_t(hips)].t.x, 0.0f, pose_[size_t(hips)].t.z)).z);
        rotate_in_model(hips, w);
        const glm::quat back = quat_yaw(-warp_ / 3.0f);
        for (Role r : { Role::Spine, Role::Spine1, Role::Spine2 }) rotate_in_model(rig[r], back);
    }

    // Lean: into acceleration (turning at speed included) and a little into the direction of
    // travel. Visual only; from the base of the spine up, so the feet stay where they are.
    float roll_t = 0.0f, pitch_t = 0.0f;
    if (tuning.lean_strength > 0.0f && !in.swimming) {
        const float g = airborne ? 0.4f : 1.0f;
        roll_t = -(a_local.x * tuning.accel_lean + v_local.x * tuning.speed_lean) * g;
        pitch_t = (a_local.y * tuning.accel_lean + v_local.y * tuning.speed_lean) * g;
        roll_t = std::clamp(roll_t * tuning.lean_strength, -tuning.max_lean, tuning.max_lean);
        pitch_t = std::clamp(pitch_t * tuning.lean_strength, -tuning.max_lean, tuning.max_lean);
    }
    if (in.swimming) pitch_t = -std::min(0.12f * speed, 0.7f);              // stretched out along the stroke
    spring_to(lean_roll_, lean_roll_v_, roll_t, 0.12f, dt);
    spring_to(lean_pitch_, lean_pitch_v_, pitch_t, 0.12f, dt);
    if (std::abs(lean_roll_) + std::abs(lean_pitch_) > 1e-4f)
        rotate_in_model(rig[Role::Spine], glm::angleAxis(lean_roll_, glm::vec3(0.0f, 0.0f, 1.0f)) * rot_x(lean_pitch_));
    // The chest stays near its place over the feet. What is carried hangs in the view, which
    // does not lean: a captured run leans the torso far enough forward to put the shoulder on
    // the weapon's grip, and an arm with no room folds to nothing and flips. Most of the lean
    // is taken back out at the base of the spine (the hips and legs keep theirs).
    if (tuning.chest_keep < 1.0f && !in.swimming) {
        const IAnimationRuntime& rt = *set_.runtime;
        rt.local_to_model(*ctx_, pose_, model_, hips);
        const int base = rig[Role::Spine], neck = rig[Role::Neck];
        const SkeletonDef& sk = rt.skeleton();
        const glm::vec3 pn(model_[size_t(neck)][3]), pb(model_[size_t(base)][3]);
        const float h = std::max(pn.y - pb.y, 0.15f);
        const float ex = pn.x - sk.rest_pos[size_t(neck)].x;
        const float ez = pn.z - (sk.rest_pos[size_t(neck)].z + 0.10f * std::clamp(in.crouch, 0.0f, 1.0f));
        const float k = (1.0f - std::clamp(tuning.chest_keep, 0.0f, 1.0f)) / h;
        rotate_in_model(base, glm::angleAxis(std::clamp(ex * k, -0.6f, 0.6f), glm::vec3(0.0f, 0.0f, 1.0f)) * rot_x(std::clamp(-ez * k, -0.6f, 0.6f)));
    }
    dbg_.lean_roll = lean_roll_;

    dbg_.lean_pitch = lean_pitch_;

    // A long fall is not a still picture: the limbs keep working a little, more the longer it lasts.
    if (airborne && air_.air_time > 0.35f && tuning.air_sway > 0.0f) {
        const float a = tuning.air_sway * 0.10f * std::min((air_.air_time - 0.35f) * 0.8f, 1.0f);
        for (int side = 0; side < 2; ++side) {
            const float ph = side == 0 ? 0.0f : kPi;
            const int up = rig[side == 0 ? Role::LeftUpLeg : Role::RightUpLeg], lo = rig[side == 0 ? Role::LeftLeg : Role::RightLeg];
            pose_[size_t(up)].r = glm::normalize(pose_[size_t(up)].r * rot_x(a * std::sin(time_ * 5.1f + ph)));
            pose_[size_t(lo)].r = glm::normalize(pose_[size_t(lo)].r * rot_x(a * 0.8f * (1.0f + std::sin(time_ * 4.3f + ph + 0.7f))));
        }
        rotate_in_model(rig[Role::Spine1], glm::angleAxis(a * 0.35f * std::sin(time_ * 3.1f), glm::vec3(0.0f, 0.0f, 1.0f)));
    }
}

float CharacterAnimator::rnd() {
    rng_ ^= rng_ << 13; rng_ ^= rng_ >> 17; rng_ ^= rng_ << 5;
    return float(rng_ & 0xFFFFFFu) / float(0x1000000);
}

void CharacterAnimator::upper_body(const CharacterAnimationInput& in) {
    const Rig& rig = set_.runtime->rig();
    const bool crouch_clip = match_.frame >= 0 && (set_.db.range_of(uint32_t(match_.frame)).tags & kTagCrouch) != 0;
    const float pitch = std::clamp(in.aim_pitch, -1.3f, 1.3f);
    const float share = in.first_person ? tuning.spine_pitch_share_fp : tuning.spine_pitch_share;
    // The aim's pitch over the spine (and neck and head for those who see the character from
    // outside); folded forward when crouched; a shot rocks the chest back a little.
    const float kick = recoil_curve(recoil_t_, weapon.recoil_duration);
    const float bend = share * pitch - (crouch_clip ? 0.25f : 0.55f) * std::clamp(in.crouch, 0.0f, 1.0f);
    const glm::quat step = rot_x(bend / 3.0f);
    for (Role r : { Role::Spine, Role::Spine1, Role::Spine2 }) rotate_in_model(rig[r], step);
    if (kick > 0.0f && held_ == HeldItem::Rifle) rotate_in_model(rig[Role::Spine2], rot_x(0.02f * kick));
    if (!in.first_person) {
        rotate_in_model(rig[Role::Neck], rot_x(0.20f * pitch));
        rotate_in_model(rig[Role::Head], rot_x(0.30f * pitch));
        // Aiming a rifle, the head leans over to the stock and down to the sights.
        const float sight = held_ == HeldItem::Rifle ? tuning.head_to_sights * smooth01(equip_t_) * (1.0f - reload_w_) * smooth01(aim_w_) : 0.0f;
        if (sight > 1e-3f) {
            rotate_in_model(rig[Role::Neck], glm::angleAxis(-0.10f * sight, glm::vec3(0.0f, 0.0f, 1.0f)));
            rotate_in_model(rig[Role::Head], glm::angleAxis(-0.12f * sight, glm::vec3(0.0f, 0.0f, 1.0f)) * rot_x(-0.07f * sight));
        }
    }
    // Not aiming, a character is not a statue: its own posture, breathing, the weight from one
    // leg to the other, a hand to the head, the head looking about (life_layers.cpp). All of it
    // stops the moment the aim comes up or a shot is fired.
    life_layers(in, 0);
    if (in.dying > 0.0f) {
        const float d = smooth01(std::clamp(in.dying, 0.0f, 1.0f));
        glm::vec2 hd(in.hit_dir.x, in.hit_dir.z);
        const float hl = glm::length(hd);
        hd = hl > 1e-4f ? hd / hl : glm::vec2(0.0f, -1.0f);
        // (shot from the front: over backward; from behind: folding forward; from a side: toward the other)
        const glm::quat fold = glm::angleAxis(-0.30f * hd.x * d, glm::vec3(0.0f, 0.0f, 1.0f)) * rot_x((hd.y > 0.0f ? 0.42f : -0.60f) * std::abs(hd.y) * d - 0.12f * d);
        rotate_in_model(rig[Role::Spine], glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), fold, 0.5f));
        rotate_in_model(rig[Role::Spine2], glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), fold, 0.5f));
        rotate_in_model(rig[Role::Neck], rot_x(-0.25f * d));
        rotate_in_model(rig[Role::Head], rot_x(-0.35f * d) * glm::angleAxis(0.20f * hd.x * d, glm::vec3(0.0f, 0.0f, 1.0f)));
    }
    // A hit: the torso gives the way the shot went and comes back (a look only: the controller
    // is not moved, the aim is not thrown).
    {
        const float dt = std::clamp(in.dt, 0.0f, 0.1f);
        if (in.hit_seq != hit_seq_) {
            hit_seq_ = in.hit_seq;
            const float kHalf = 0.09f;
            const float peak = tuning.flinch * std::clamp(in.hit_power, 0.2f, 2.0f);
            const float kickv = peak * (4.0f * 0.69314718f / kHalf * 0.5f) * 2.7182818f;
            glm::vec2 d(in.hit_dir.x, in.hit_dir.z);
            const float dl = glm::length(d);
            d = dl > 1e-4f ? d / dl : glm::vec2(0.0f, 1.0f);
            flinch_pitch_v_ = std::clamp(flinch_pitch_v_ + d.y * kickv, -3.0f * kickv, 3.0f * kickv);
            flinch_roll_v_ = std::clamp(flinch_roll_v_ + d.x * kickv, -3.0f * kickv, 3.0f * kickv);
            dip_v_ += 0.03f * (4.0f * 0.69314718f / kDipHalflife * 0.5f) * 2.7182818f;
        }
        decay_spring(flinch_pitch_, flinch_pitch_v_, 0.09f, dt);
        decay_spring(flinch_roll_, flinch_roll_v_, 0.09f, dt);
        if (std::abs(flinch_pitch_) + std::abs(flinch_roll_) > 1e-4f) {
            const float share = in.first_person ? 0.35f : 1.0f;          // (from the player's own eyes: a nudge of the arms, not a lurch)
            const glm::quat q = glm::angleAxis(-flinch_roll_ * share, glm::vec3(0.0f, 0.0f, 1.0f)) * rot_x(flinch_pitch_ * share);
            rotate_in_model(rig[Role::Spine1], q);
            if (!in.first_person) rotate_in_model(rig[Role::Head], glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), q, 0.6f));
        }
    }
    face_layer(in);
}

// The face: blinks, eyes that get where the head is going before it does and dart about a little
// in between, a jaw that drops when hit or out of breath, brows that go up with a hit. Seen from
// outside at full detail only; the joints are optional (AnimationSet::face).
void CharacterAnimator::face_layer(const CharacterAnimationInput& in) {
    const AnimationSet::Face& F = set_.face;
    if (!F.ok || in.first_person || in.lod > 0 || tuning.face <= 0.0f) return;
    if (!rng_seeded_) return;                                   // (the idle layer seeds the sequence)
    const float dt = std::clamp(in.dt, 0.0f, 0.1f);
    const float dying = std::clamp(in.dying, 0.0f, 1.0f);
    // ---- blinking: every few seconds, a tenth of a second; sometimes twice; with a turn of the eyes ----
    blink_left_ -= dt;
    if (blink_left_ <= 0.0f && blink_t_ < 0.0f) {
        blink_t_ = 0.0f;
        ++blinks_;
        blink_left_ = rnd() < 0.15f ? 0.30f + 0.15f * rnd() : 2.0f + 4.5f * rnd();
    }
    float shut = 0.0f;
    if (blink_t_ >= 0.0f) {
        blink_t_ += dt;
        const float kDown = 0.055f, kUp = 0.11f;                 // (down faster than up)
        shut = blink_t_ < kDown ? blink_t_ / kDown : 1.0f - (blink_t_ - kDown) / kUp;
        if (blink_t_ >= kDown + kUp) { blink_t_ = -1.0f; shut = 0.0f; }
        shut = smooth01(std::clamp(shut, 0.0f, 1.0f));
    }
    shut = std::max({ shut, smooth01(dying) * 0.85f, std::clamp(tuning.blink_hold, 0.0f, 1.0f) });
    lid_ = shut;
    // ---- eyes: at the target at once (the head follows), and small darts while nothing is looked at ----
    dart_left_ -= dt;
    if (dart_left_ <= 0.0f) {
        dart_left_ = 0.5f + 2.2f * rnd();
        dart_yaw_ = (rnd() - 0.5f) * 0.16f;
        dart_pitch_ = (rnd() - 0.5f) * 0.07f;
    }
    float ey = (gaze_want_yaw_ - gaze_yaw_) + dart_yaw_ * gaze_w_;
    float ep = (gaze_want_pitch_ - gaze_pitch_) + dart_pitch_ * gaze_w_;
    ey = std::clamp(ey + tuning.eyes_hold, -0.55f, 0.55f);
    ep = std::clamp(ep, -0.35f, 0.30f);
    // (an eye moves in a few hundredths of a second: no easing to speak of)
    const float ke = 1.0f - std::exp(-dt / 0.025f);
    eye_yaw_ += (ey - eye_yaw_) * ke;
    eye_pitch_ += (ep - eye_pitch_) * ke;
    // ---- jaw and brows: a hit, dying, out of breath ----
    const float hurt = std::clamp((std::abs(flinch_pitch_) + std::abs(flinch_roll_)) / std::max(tuning.flinch, 1e-3f), 0.0f, 1.0f);
    const float speed = glm::length(glm::vec2(in.velocity.x, in.velocity.z));
    const float breath = speed > 9.5f ? 0.035f + 0.02f * std::sin(time_ * 6.0f) : 0.0f;
    spring_to(jaw_, jaw_v_, std::max({ 0.16f * hurt, 0.22f * smooth01(dying), breath, tuning.jaw_hold }), 0.06f, dt);
    spring_to(brow_, brow_v_, std::max(hurt, 0.6f * smooth01(dying)), 0.06f, dt);

    const float k = std::clamp(tuning.face, 0.0f, 1.0f);
    const glm::vec3 kY(0.0f, 1.0f, 0.0f);
    for (int side = 0; side < 2; ++side) {
        // (+yaw looks right: about -Y; +pitch looks up: about +X; a lid comes down about -X)
        rotate_in_model(F.eye[side], glm::angleAxis(-eye_yaw_ * k, kY) * rot_x(eye_pitch_ * k));
        // the lids go with the eye's height, and shut
        rotate_in_model(F.lid_up[side], rot_x((0.6f * eye_pitch_ - 0.50f * lid_) * k));
        if (F.lid_lo[side] >= 0) rotate_in_model(F.lid_lo[side], rot_x((0.3f * eye_pitch_ + 0.10f * lid_) * k));
        if (F.brow[side] >= 0) pose_[size_t(F.brow[side])].t.y += 0.0045f * brow_ * k;
    }
    if (F.jaw >= 0) rotate_in_model(F.jaw, rot_x(-jaw_ * k));
}


void CharacterAnimator::feet(const CharacterAnimationInput& in, glm::vec2 v_local, float speed, float accel) {
    const IAnimationRuntime& rt = *set_.runtime;
    const Rig& rig = rt.rig();
    const float dt = std::clamp(in.dt, 0.0f, 0.1f);
    const bool on_ground = in.grounded && !in.swimming;
    const bool full = in.lod == 0;
    const int hips = rig[Role::Hips];
    const glm::vec3 hipP(model_[size_t(hips)][3]);
    const glm::vec2 mdir = speed > 0.3f ? v_local / speed : glm::vec2(0.0f, -1.0f);
    const float w = std::clamp(tuning.foot_ik_weight, 0.0f, 1.0f);
    glm::vec3 target[2];
    float slope[2] = { 0.0f, 0.0f };
    float lowest = 0.0f;
    for (int side = 0; side < 2; ++side) {
        const int ft = rig[side == 0 ? Role::LeftFoot : Role::RightFoot];
        const glm::vec3 base(model_[size_t(ft)][3]);
        glm::vec3 t = base;
        // Stride warp: the feet reach further along the way the character goes, so a foot on the
        // ground travels back under the body as fast as the ground does. A leg is only so long:
        // where the stride asks for more, the foot leaves the ground (it rises on the circle the
        // leg can reach) instead of being dragged over it. Contact gets shorter and the step gets a
        // flight, which is what running faster is. The hips come down a little to lengthen it.
        glm::vec2 rel(t.x - hipP.x, t.z - hipP.z);
        if (stride_ > 1.0f) {
            const int up = rig[side == 0 ? Role::LeftUpLeg : Role::RightUpLeg], lo = rig[side == 0 ? Role::LeftLeg : Role::RightLeg];
            const glm::vec3 pu = glm::vec3(model_[size_t(up)][3]) + glm::vec3(0.0f, pelvis_ik_, 0.0f);
            const float reach = (glm::length(glm::vec3(model_[size_t(lo)][3]) - glm::vec3(model_[size_t(up)][3])) +
                                 glm::length(base - glm::vec3(model_[size_t(lo)][3]))) * 0.985f;
            // (measured from under the foot's own hip joint: the width of the stance is not stride)
            rel += mdir * (glm::dot(glm::vec2(t.x - pu.x, t.z - pu.z), mdir) * (stride_ - 1.0f));
            glm::vec2 d(hipP.x + rel.x - pu.x, hipP.z + rel.y - pu.z);
            float dl = glm::length(d);
            const float cap = tuning.stride_reach * reach;
            if (dl > cap) {
                d *= cap / dl;
                rel = glm::vec2(pu.x + d.x - hipP.x, pu.z + d.y - hipP.z);
                dl = cap;
            }
            // (the circle of a leg that keeps a bend in the knee - a leg driven to its full length
            // snaps straight and back - or of the leg as long as the capture has it at this moment)
            const float own = std::min(glm::length(base - glm::vec3(model_[size_t(up)][3])), reach);
            const float allow = std::max(tuning.stride_leg * reach, own);
            t.y = std::max(t.y, pu.y - std::sqrt(std::max(allow * allow - dl * dl, 0.0f)));
        }
        t.x = hipP.x + rel.x;
        t.z = hipP.z + rel.y;
        // Foot lock: a foot the capture has on the ground stays where it stands in the world
        // (on what it stands on, if that moves); let go, the difference eases away. Never in
        // the air, at speed, or through a hard start or stop: that would pull like rubber.
        Lock& lk = lock_[side];
        const bool want = full && tuning.foot_lock && cur_.contact[side] && on_ground && air_.phase == AirPhase::Ground &&
                          speed < tuning.foot_lock_max_speed && accel < tuning.foot_lock_max_accel;
        const glm::vec3 tw = in.position + model_to_world_vec(t, in.facing_yaw);
        // (a lock given up because the body turned or was pushed away is taken up again where the
        // foot now stands, once the body is at rest)
        if (!want || (speed < 0.2f && std::abs(yaw_rate_) < 0.3f && glm::length(lk.off) < 0.02f)) lk.broken = false;
        if (lk.on) lk.world += in.ground_velocity * dt;
        if (want && !lk.on && !lk.broken) { lk.on = true; lk.world = tw + lk.off; lk.off_v = glm::vec3(0.0f); }
        if (lk.on) {
            lk.off = lk.world - tw;
            lk.off.y = 0.0f;
            if (!want || glm::length(lk.off) > tuning.foot_lock_break) { lk.on = false; lk.broken = want; }
        }
        if (!lk.on) decay_spring(lk.off, lk.off_v, 0.09f, dt);
        t += world_to_model_vec(lk.off, in.facing_yaw) * w;
        // The ground under the foot: up a step or a slope, down into a dip, within reach.
        float g = 0.0f;
        if (full && on_ground && in.ground_height) {
            // (ground further away than a leg can follow is a drop or a wall beside the foot, not
            // something to stand on: the foot stays level with the body)
            g = ground_at(in, t);
            if (std::abs(g) > tuning.foot_ground_range) g = 0.0f;
            else if (cur_.contact[side]) {
                const float gt = ground_at(in, t + glm::vec3(0.0f, 0.0f, -0.16f));
                if (std::abs(gt) <= tuning.foot_ground_range) slope[side] = std::clamp(std::atan2(gt - g, 0.16f), -0.5f, 0.5f);
            }
        }
        spring_to(lk.ground, lk.ground_v, g, 0.05f, dt);
        t.y += lk.ground * w;
        lowest = std::min(lowest, lk.ground * w);
        target[side] = t;
        dbg_.locked[side] = lk.on;
        dbg_.foot_target[side] = t;
    }
    // Pelvis compensation: when a foot has to go down, the hips go with it so the other leg is not
    // stretched; and a long stride is taken lower.
    if (on_ground) lowest -= std::clamp((stride_ - 1.0f) * tuning.stride_drop, 0.0f, tuning.max_stride_drop);
    spring_to(pelvis_ik_, pelvis_ik_v_, lowest, 0.08f, dt);
    if (std::abs(pelvis_ik_) > 1e-4f) {
        pose_[size_t(hips)].t.y += pelvis_ik_;
        rt.local_to_model(*ctx_, pose_, model_, hips);
    }
    for (int side = 0; side < 2; ++side) {
        const int up = rig[side == 0 ? Role::LeftUpLeg : Role::RightUpLeg], lo = rig[side == 0 ? Role::LeftLeg : Role::RightLeg],
                  ft = rig[side == 0 ? Role::LeftFoot : Role::RightFoot];
        // Which way the knee points: as the capture has it while its leg is bent; a leg the capture
        // has (nearly) straight says nothing about that - its knee lies on the line from hip to
        // foot, and whatever is left over swings round from frame to frame once the foot is moved.
        // There the knee points where the hips face.
        const glm::vec3 pa(model_[size_t(up)][3]), pk(model_[size_t(lo)][3]), pf(model_[size_t(ft)][3]);
        const float leg = glm::length(pk - pa) + glm::length(pf - pk);
        const float straight = smooth01((glm::length(pf - pa) / std::max(leg, 1e-4f) - 0.90f) / 0.07f);
        const glm::vec3 fwd = glm::mat3(model_[size_t(hips)]) * glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 own = pk - (pa + pf) * 0.5f;
        const float own_len = glm::length(own);
        own = own_len > 1e-4f ? own / own_len : fwd;
        const glm::vec3 knee = glm::mix(own, fwd, straight);
        fit_limb(up, lo, ft, target[side], &knee);
        if (std::abs(slope[side]) > 1e-3f) {                 // the sole follows the slope
            rotate_in_model(ft, rot_x(slope[side] * w));
            rt.local_to_model(*ctx_, pose_, model_, ft);
        }
    }
}

void CharacterAnimator::arms(const CharacterAnimationInput& in) {
    const IAnimationRuntime& rt = *set_.runtime;
    const Rig& rig = rt.rig();
    // ---- the weapon's frame: the view frame (gameplay aim), moved by recoil, reload and equip ----
    const float kick = held_ == HeldItem::Rifle ? recoil_curve(recoil_t_, weapon.recoil_duration) : 0.0f;
    const float lowered = std::max(1.0f - smooth01(equip_t_), smooth01(std::clamp(in.dying, 0.0f, 1.0f)));
    const float tilt = held_ == HeldItem::Rifle ? reload_w_ * smooth01(std::min(reload_u_ / 0.12f, (1.0f - reload_u_) / 0.12f)) : 0.0f;
    // (the turn is about the right hand's grip, so the hand stays on it)
    const glm::vec3 pivot = weapon.grip_r;
    auto offset_for = [&](float k) {
        glm::mat4 o(1.0f);
        o[3] = glm::vec4(0.0f, -0.22f * lowered - 0.03f * tilt, k * weapon.recoil_stroke, 1.0f);
        const glm::quat rq = rot_x(k * weapon.recoil_pitch - 0.5f * lowered + 0.22f * tilt) * glm::angleAxis(0.45f * tilt, glm::vec3(0.0f, 0.0f, 1.0f));
        return o * glm::translate(glm::mat4(1.0f), pivot) * glm::mat4_cast(rq) * glm::translate(glm::mat4(1.0f), -pivot);
    };
    glm::mat4 off = offset_for(kick);
    // Aiming or at the ready. `ready`: how far the weapon is down from the aim; `sunk`: how far
    // its muzzle hangs (not while a shot from the hip is fresh, a reload runs or it is being drawn).
    const float ready = held_ == HeldItem::Rifle ? (1.0f - smooth01(aim_w_)) * (1.0f - lowered) : 0.0f;
    const float level = 1.0f - smooth01(std::clamp((since_shot_ - weapon.hip_fire_hold) / 0.35f, 0.0f, 1.0f));
    const float sunk = ready * (1.0f - level) * (1.0f - reload_w_);
    auto about = [](glm::vec3 p, const glm::quat& q) {
        return glm::translate(glm::mat4(1.0f), p) * glm::mat4_cast(q) * glm::translate(glm::mat4(1.0f), -p);
    };
    const glm::vec3 kUp(0.0f, 1.0f, 0.0f), kBack(0.0f, 0.0f, 1.0f);
    // (the player's own view, aiming: to the middle of the screen)
    const float ads = in.first_person && held_ == HeldItem::Rifle ? smooth01(aim_w_) * (1.0f - lowered) * (1.0f - reload_w_) : 0.0f;
    if (ads > 1e-4f) off = glm::translate(glm::mat4(1.0f), weapon.ads_shift_fp * ads) * off;
    if (ready > 1e-4f) {
        if (in.first_person) {
            off = glm::translate(glm::mat4(1.0f), weapon.ready_shift_fp * ready) * off *
                  about(weapon.grip_r, glm::angleAxis(weapon.ready_yaw_fp * sunk, kUp) * rot_x(weapon.ready_pitch_fp * sunk) *
                                       glm::angleAxis(weapon.ready_roll_fp * ready, kBack));
        } else {
            off = off * about(weapon.butt, glm::angleAxis(weapon.ready_yaw * sunk, kUp) * rot_x(weapon.ready_pitch * sunk));
        }
    }
    dbg_.recoil = kick;
    dbg_.reload = reload_t_ >= 0.0f ? reload_u_ : 0.0f;
    dbg_.equip = lowered;

    const float p = in.aim_pitch;
    const glm::vec3 cam_f(0.0f, std::sin(p), -std::cos(p)), cam_u(0.0f, std::cos(p), std::sin(p));
    const glm::mat4 V(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), glm::vec4(cam_u, 0.0f), glm::vec4(-cam_f, 0.0f), glm::vec4(in.eye_local, 1.0f));
    glm::mat4 W = V * off;
    // Seen from outside, a rifle is held into the shoulder. The frame above is the one the
    // player sees his own weapon in: beside and below the eye, which on a body is a rifle
    // floating in front of the face at arm's length. For everybody else the weapon keeps its
    // direction (the aim) and is moved so that its butt sits in the right shoulder.
    if (!in.first_person && held_ == HeldItem::Rifle && tuning.shoulder_weapon) {
        const glm::vec3 shoulder(model_[size_t(rig[Role::RightArm])][3]);
        const glm::vec3 anchor = shoulder + glm::mat3(V) * (weapon.shoulder_offset + weapon.ready_shift * ready);
        // (where the butt is without the recoil: the kick still drives the weapon back from there)
        const glm::vec3 butt(V * offset_for(0.0f) * glm::vec4(weapon.butt, 1.0f));
        W[3] += glm::vec4(anchor - butt, 0.0f);
        off = glm::inverse(V) * W;
    }
    // The tablet likewise: seen from outside it is held in front of the chest, not at the eye.
    if (!in.first_person && held_ == HeldItem::Tablet && tuning.shoulder_weapon) {
        const glm::vec3 anchor = glm::vec3(model_[size_t(rig[Role::Neck])][3]) + weapon.tablet_chest;
        W[3] += glm::vec4(anchor - glm::vec3(W * glm::vec4(weapon.tablet_centre, 1.0f)), 0.0f);
        off = glm::inverse(V) * W;
    }
    weapon_offset_ = off;
    // Where a shot leaves from: this stance, level, before the recoil. (A weapon at the ready is
    // level by the time the shot shows: the shot itself brings it up.)
    {
        glm::mat4 f = offset_for(0.0f);
        if (ads > 1e-4f) f = glm::translate(glm::mat4(1.0f), weapon.ads_shift_fp * ads) * f;
        if (ready > 1e-4f && in.first_person)
            f = glm::translate(glm::mat4(1.0f), weapon.ready_shift_fp * ready) * f * about(weapon.grip_r, glm::angleAxis(weapon.ready_roll_fp * ready, kBack));
        glm::mat4 Wf = V * f;
        if (!in.first_person && held_ == HeldItem::Rifle && tuning.shoulder_weapon) {
            const glm::vec3 shoulder(model_[size_t(rig[Role::RightArm])][3]);
            const glm::vec3 anchor = shoulder + glm::mat3(V) * (weapon.shoulder_offset + weapon.ready_shift * ready);
            Wf[3] += glm::vec4(anchor - glm::vec3(V * offset_for(0.0f) * glm::vec4(weapon.butt, 1.0f)), 0.0f);
        }
        weapon_fire_offset_ = glm::inverse(V) * Wf;
    }
    mag_visible_ = false;
    const glm::mat3 Wr(W), Vr(V);
    auto wpt = [&](glm::vec3 v) { return glm::vec3(W * glm::vec4(v, 1.0f)); };
    muzzle_pos_ = wpt(weapon.muzzle);
    muzzle_dir_ = glm::normalize(Wr * glm::vec3(0.0f, 0.0f, -1.0f));
    if (!tuning.hand_ik || in.lod >= 3) return;

    for (int side = 0; side < 2; ++side) {
        const AnimationSet::Hand& H = set_.hand[side];
        const int sh = rig[side == 0 ? Role::LeftShoulder : Role::RightShoulder], ua = rig[side == 0 ? Role::LeftArm : Role::RightArm],
                  fa = rig[side == 0 ? Role::LeftForeArm : Role::RightForeArm], hd = rig[side == 0 ? Role::LeftHand : Role::RightHand];
        if (!H.ok) continue;
        const float sx = side == 0 ? -1.0f : 1.0f;
        glm::vec3 grip(0.0f), a1(0.0f, 0.0f, -1.0f), n1(0.0f, -1.0f, 0.0f);
        glm::vec3 pole(sx * 0.9f, -1.0f, 0.5f);
        float curl[3] = { 0.9f, 1.15f, 0.8f };
        float index_curl = 1.0f;
        bool free_arm = false, grip_thumb = false;
        const HandPose* hp = nullptr;                           // (the fingers one by one, where what is held has a pose for them)
        if (held_ == HeldItem::Torch && side == 1) {
            // Fist round the torch body, thumb forward along it.
            const glm::vec3 f = glm::normalize(in.torch_fwd), r = glm::normalize(in.torch_right), u = glm::cross(r, f);
            grip = in.torch_pos - f * 0.030f + r * 0.012f;
            pole = glm::vec3(0.30f, -1.0f, 0.85f);
            a1 = glm::normalize(f * 0.80f - u * 0.60f);
            n1 = -r;
            curl[0] = 1.15f; curl[1] = 1.35f; curl[2] = 0.9f;
        } else if (held_ == HeldItem::Rifle && side == 1) {
            // Right hand on the pistol grip, index finger on the trigger.
            grip = wpt(weapon.grip_r);
            a1 = glm::normalize(Wr * weapon.r_fingers);
            n1 = glm::normalize(Wr * weapon.r_palm);
            index_curl = 0.45f;
            hp = &weapon.pose_r;
        } else if (held_ == HeldItem::Rifle && side == 0) {
            // Left hand under the fore-end; during a reload it goes to the magazine and back.
            grip = wpt(weapon.grip_l);
            if (reload_w_ > 1e-3f) {
                const glm::vec3 a = weapon.grip_l, m = weapon.reload_hand, d = m + glm::vec3(0.0f, -0.12f, 0.02f);
                const float u = reload_u_;
                glm::vec3 path;
                if (u < 0.25f) path = glm::mix(a, m, smooth01(u / 0.25f));
                else if (u < 0.45f) path = glm::mix(m, d, smooth01((u - 0.25f) / 0.20f));
                else if (u < 0.75f) path = glm::mix(d, m, smooth01((u - 0.45f) / 0.30f));
                else path = glm::mix(m, a, smooth01((u - 0.75f) / 0.25f));
                grip = glm::mix(grip, wpt(path), reload_w_);
                // (the magazine: out with the hand on its way down, a fresh one back up with it)
                mag_visible_ = reload_w_ > 0.5f && u >= 0.25f && u <= 0.75f;
                mag_pos_ = path;
            }
            a1 = glm::normalize(Wr * weapon.l_fingers);
            n1 = glm::normalize(Wr * weapon.l_palm);
            curl[0] = 1.05f; curl[1] = 1.30f; curl[2] = 0.85f;
            grip_thumb = true;
            hp = &weapon.pose_l;
        } else if (held_ == HeldItem::Tablet && side == 1) {
            // The tablet lies on the right hand.
            grip = wpt(weapon.tablet_centre + glm::vec3(0.015f, -0.035f, 0.025f));
            a1 = glm::normalize(Vr * glm::vec3(-0.35f, 0.55f, -0.75f));
            n1 = glm::normalize(Vr * glm::vec3(-0.25f, 0.70f, 0.65f));
            curl[0] = 0.25f; curl[1] = 0.3f; curl[2] = 0.2f;
        } else {
            free_arm = true;                                    // swings as the capture has it
            curl[0] = 0.35f; curl[1] = 0.45f; curl[2] = 0.3f;
        }
        if (!free_arm) {
            n1 = glm::normalize(n1 - a1 * glm::dot(n1, a1));
            const glm::vec3 wrist = grip - a1 * (H.palm * 0.80f) - n1 * 0.028f;
            // The shoulder girdle goes a little of the way toward a far reach.
            // and as far as it takes (within what a shoulder can do) when the arm alone is too short.
            const float reach = (glm::length(glm::vec3(model_[size_t(fa)][3]) - glm::vec3(model_[size_t(ua)][3])) +
                                 glm::length(glm::vec3(model_[size_t(hd)][3]) - glm::vec3(model_[size_t(fa)][3]))) * 0.985f;
            float budget = 0.55f;
            for (int it = 0; it < 3 && budget > 1e-3f; ++it) {
                const glm::vec3 ps(model_[size_t(sh)][3]), pa(model_[size_t(ua)][3]);
                const glm::vec3 d0 = pa - ps, d1 = wrist - ps;
                if (glm::dot(d0, d0) < 1e-8f || glm::dot(d1, d1) < 1e-8f) break;
                const glm::vec3 full = quat_to_scaled_axis(quat_from_to(glm::normalize(d0), glm::normalize(d1)));
                const float full_len = glm::length(full);
                if (full_len < 1e-4f) break;
                const float deficit = glm::length(wrist - pa) - reach;
                float ang = it == 0 ? 0.18f * full_len : 0.0f;
                if (deficit > 0.0f) ang = std::max(ang, 1.3f * deficit / glm::length(d0));
                ang = std::min({ ang, full_len, budget });
                if (ang < 1e-4f) break;
                budget -= ang;
                rotate_in_model(sh, quat_from_scaled_axis(full * (ang / full_len)));
                rt.local_to_model(*ctx_, pose_, model_, sh);
                if (deficit <= 0.0f) break;
            }
            fit_limb(ua, fa, hd, wrist, &pole, true);
            // The hand's orientation: its rest frame (fingers, palm) onto the grip's.
            const glm::mat3 F0(H.a0, H.n0, H.b0), F1(a1, n1, glm::cross(a1, n1));
            const glm::quat G = glm::quat_cast(F1 * glm::transpose(F0));
            const glm::quat Rf = glm::quat_cast(glm::mat3(model_[size_t(fa)]));
            pose_[size_t(hd)].r = glm::normalize(glm::inverse(Rf) * G);
            dbg_.hand_target[side] = wrist;
        } else {
            dbg_.hand_target[side] = glm::vec3(model_[size_t(hd)][3]);
        }
        if (in.lod >= 2) continue;
        if (hp) {
            for (int f = 0; f < 4; ++f) {
                const float fan = hp->spread[f] * (f <= 1 ? 1.0f : -1.0f) * H.index_side;
                for (int j = 0; j < 3; ++j) {
                    if (H.finger[f][j] < 0) continue;
                    const glm::quat bend = glm::angleAxis(hp->curl[f][j], H.b0);
                    pose_[size_t(H.finger[f][j])].r = j == 0 ? glm::angleAxis(fan, H.n0) * bend : bend;
                }
            }
            for (int j = 0; j < 3; ++j)
                if (H.thumb[j] >= 0)
                    pose_[size_t(H.thumb[j])].r = glm::angleAxis(-sx * (j == 0 ? hp->thumb[0] : 0.0f), H.a0) * glm::angleAxis(j == 0 ? hp->thumb[1] : hp->thumb[2], H.b0);
            continue;
        }
        // Fingers close round the grip; the thumb swings in over them.
        for (int f = 0; f < 4; ++f) {
            const float k = f == 0 ? index_curl : 1.0f;
            for (int j = 0; j < 3; ++j)
                if (H.finger[f][j] >= 0) pose_[size_t(H.finger[f][j])].r = glm::angleAxis(curl[j] * k, H.b0);
        }
        for (int j = 0; j < 3; ++j) {
            if (H.thumb[j] < 0) continue;
            pose_[size_t(H.thumb[j])].r = grip_thumb
                ? glm::angleAxis(-sx * (j == 0 ? 0.85f : 0.0f), H.a0) * glm::angleAxis(j == 0 ? 0.35f : 0.55f, H.b0)
                : glm::angleAxis(-sx * (j == 0 ? 0.35f : 0.0f), H.a0) * glm::angleAxis(sx * (j == 0 ? 0.10f : 0.30f) * (free_arm ? 0.3f : 1.0f), H.n0);
        }
    }
}

}  // namespace vibe::anim
