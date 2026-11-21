#include "anim/character_animator.h"

#include "game/player.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace qlike::anim {

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
    return true;
}

CharacterAnimator::CharacterAnimator(const AnimationSet& set) : set_(set) {
    tuning.move = game::kPlayerMove;
    const size_t n = size_t(set_.runtime->skeleton().size());
    ctx_ = set_.runtime->create_context(2);
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
    dip_ = dip_v_ = pelvis_ik_ = pelvis_ik_v_ = 0.0f;
    for (Lock& l : lock_) l = Lock{};
    recoil_t_ = 0.0f; reload_t_ = -1.0f; reload_w_ = 0.0f; equip_t_ = 1.0f;
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
        if (!started_) { shot_seq_ = in.shot_seq; reload_seq_ = in.reload_seq; held_ = in.held; }
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
    if (in.shot_seq != shot_seq_) { shot_seq_ = in.shot_seq; recoil_t_ = weapon.recoil_duration; }
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

    float ground_dist = 1e9f;
    if (!in.grounded && in.ground_height) ground_dist = std::max(0.0f, -ground_at(in, glm::vec3(0.0f)));

    lower_body(in, v_local, speed, yaw_rate_, ground_dist);
    pelvis_and_warp(in, v_local, speed, yaw_rate_, a_local);
    upper_body(in);
    rt.local_to_model(*ctx_, pose_, model_);
    if (tuning.foot_ik && in.lod <= 1) feet(in, v_local, speed, glm::length(a_local));
    arms(in);
    rt.local_to_model(*ctx_, pose_, model_);

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

void CharacterAnimator::lower_body(const CharacterAnimationInput& in, glm::vec2 v_local, float speed, float yaw_rate, float ground_dist) {
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
    }
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
    const float lowered = 1.0f - smooth01(equip_t_);
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
        const glm::vec3 anchor = shoulder + glm::mat3(V) * weapon.shoulder_offset;
        // (where the butt is without the recoil: the kick still drives the weapon back from there)
        const glm::vec3 butt(V * offset_for(0.0f) * glm::vec4(weapon.butt, 1.0f));
        W[3] += glm::vec4(anchor - butt, 0.0f);
        off = glm::inverse(V) * W;
    }
    weapon_offset_ = off;
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
            }
            a1 = glm::normalize(Wr * weapon.l_fingers);
            n1 = glm::normalize(Wr * weapon.l_palm);
            curl[0] = 1.05f; curl[1] = 1.30f; curl[2] = 0.85f;
            grip_thumb = true;
        } else if (held_ == HeldItem::Tablet && side == 1) {
            // The tablet lies on the right hand.
            grip = wpt(glm::vec3(0.170f, -0.200f, -0.375f));
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

}  // namespace qlike::anim
