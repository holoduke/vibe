// THE LIFE LAYERS: what is put on top of whatever made the pose - locomotion, an action clip,
// the seat - so that a character is somebody and is alive, and so that two people who play the
// same clip are not the same picture (docs/animation/actions.md "Life layers").
//
//   a  posture     the person's own way of carrying the body: constant offsets (PersonStyle)
//   b  clip layer  a clip over part of the body (a mask a joint), eased in and out
//   c  breathing, and the weight going from one side to the other, on the person's own clocks
//   d  a gesture   a hand to the head, to the neck, a look at the wrist, a shift on the seat, a
//                  wave: one arm solved to a place on the body and blended over what it did
//   e  the look    head, neck, chest (and eyes: face_layer) to what is worth a look, within
//                  what a neck can do, at a head's pace, with glances away
//
// Everything here is cheap (a few joint rotations; one clip sample for b, one arm for d), keeps
// its own state in the animator, and never snaps: every weight is eased. What fits the body to
// the world comes after it (the callers: feet, the seat's hips, hands on a weapon or a wheel).
#include "anim/character_animator.h"

#include <algorithm>
#include <cmath>

namespace vibe::anim {

namespace {

constexpr float kPi = 3.14159265f;
const glm::vec3 kX(1.0f, 0.0f, 0.0f), kY(0.0f, 1.0f, 0.0f), kZ(0.0f, 0.0f, 1.0f);

float smooth01(float x) { x = std::clamp(x, 0.0f, 1.0f); return x * x * (3.0f - 2.0f * x); }
glm::quat rot_x(float a) { return glm::angleAxis(a, kX); }
// (about +Z: a trunk that stands leans to the LEFT, a bone that points right goes UP)
glm::quat rot_z(float a) { return glm::angleAxis(a, kZ); }

// Two poses of one joint blended, frame after frame, WITHOUT a jump. The shortest way from a
// to b is not always the same way: an arm at the mouth and the same arm solved to the head can
// be half a turn apart, and as the clip moves the shorter side changes - at half weight the arm
// would snap from one to the other. So the way round is chosen once, when the blend begins
// (`fresh`), and kept: each pose is held on the side it was on the frame before (pa, pb).
glm::quat blend_kept(glm::quat a, glm::quat b, float k, glm::quat& pa, glm::quat& pb, bool fresh) {
    if (!fresh) {
        if (glm::dot(a, pa) < 0.0f) a = -a;
        if (glm::dot(b, pb) < 0.0f) b = -b;
    }
    // (begun, or the two have come to be the same turn written both ways: the short way)
    if ((fresh && glm::dot(a, b) < 0.0f) || glm::dot(a, b) < -0.92f) b = -b;
    pa = a; pb = b;
    return glm::normalize(glm::mix(a, b, std::clamp(k, 0.0f, 1.0f)));
}

}  // namespace

// A weight a joint for each LayerMask. Feathered where the body carries on below the part: an
// upper body that is laid over other legs takes the lower spine only partly, or it would hinge
// at the hips.
void AnimationSet::build_masks() {
    const SkeletonDef& s = runtime->skeleton();
    const Rig& rig = runtime->rig();
    const size_t n = size_t(s.size());
    auto under = [&](int j, int top) {
        if (top < 0) return false;
        for (int p = j; p >= 0; p = s.parents[size_t(p)]) if (p == top) return true;
        return false;
    };
    for (std::vector<float>& m : mask) m.assign(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        const int j = int(i);
        float up = under(j, rig[Role::Spine2]) ? 1.0f : 0.0f;
        if (j == rig[Role::Spine2]) up = 0.85f;
        else if (j == rig[Role::Spine1]) up = 0.55f;
        else if (j == rig[Role::Spine]) up = 0.30f;
        mask[int(LayerMask::UpperBody)][i] = up;
        mask[int(LayerMask::Head)][i] = j == rig[Role::Neck] ? 0.6f : under(j, rig[Role::Head]) ? 1.0f : 0.0f;
        const float al = j == rig[Role::LeftShoulder] ? 0.5f : under(j, rig[Role::LeftShoulder]) ? 1.0f : 0.0f;
        const float ar = j == rig[Role::RightShoulder] ? 0.5f : under(j, rig[Role::RightShoulder]) ? 1.0f : 0.0f;
        mask[int(LayerMask::ArmLeft)][i] = al;
        mask[int(LayerMask::ArmRight)][i] = ar;
        mask[int(LayerMask::Arms)][i] = std::max(al, ar);
    }
}

void CharacterAnimator::life_layers(const CharacterAnimationInput& in, int base) {
    const Rig& rig = set_.runtime->rig();
    const float dt = std::clamp(in.dt, 0.0f, 0.1f);
    if (in.variation != style_of_) { style_of_ = in.variation; style_ = person_style(in.variation); }
    const bool sitting = base == 1 || (base == 2 && in.action_warp_full);
    hips_shift_ = 0.0f;
    layer_eff_ = 0.0f;

    posture_layer(in, sitting);                                   // a
    if (in.first_person || in.lod > 2) return;
    clip_layer(in, dt);                                           // b
    if (in.lod > 1) return;

    if (!rng_seeded_) {
        rng_seeded_ = true;
        rng_ = 0x9E3779B9u ^ (in.variation * 0x85EBCA6Bu + 0x1234567u);
        if (rng_ == 0) rng_ = 1;
        for (int i = 0; i < 4; ++i) rnd();
        gaze_left_ = 0.5f + 2.5f * rnd();
        shift_left_ = 2.0f + 5.0f * rnd();
        fidget_left_ = 2.0f + 12.0f * style_.phase;               // (not from rnd(): its sequence is the look's)
    }
    const float speed = glm::length(glm::vec2(in.velocity.x, in.velocity.z));
    const bool moving = speed > 0.6f || (base == 2 && in.action_in_place);
    // Locomotion: all of it stops the moment the aim comes up or a shot is fired. An action clip
    // or the seat: only where the caller says this is somebody who stays (life_on_clip), not
    // something done once.
    const bool busy = base == 0 ? (in.dying > 0.0f || in.aiming || since_shot_ < 1.6f || reload_w_ > 0.05f || held_ != in.held || in.swimming || !in.grounded)
                                : (!in.life_on_clip || in.dying > 0.0f);
    const float want = busy ? 0.0f : 1.0f;
    gaze_w_ += (want - gaze_w_) * (1.0f - std::exp(-dt / (busy ? 0.08f : 0.35f)));

    // ---- c: breathing, and the weight from one side to the other now and then ----
    const float still = gaze_w_ * (moving ? 0.0f : 1.0f) * tuning.idle_sway;
    const float own = std::clamp(in.individuality, 0.0f, 1.0f);
    spring_to(shift_, shift_v_, shift_t_ * still, 0.40f, dt);
    if (std::abs(shift_) > 1e-4f) {
        if (sitting) {
            // (on a seat the hips stay: the trunk goes over to one side, the head stays up)
            rotate_in_model(rig[Role::Spine], rot_z(-0.030f * shift_));
            rotate_in_model(rig[Role::Spine2], rot_z(-0.015f * shift_));
            rotate_in_model(rig[Role::Neck], rot_z(0.030f * shift_));
        } else {
            pose_[size_t(rig[Role::Hips])].t.x += 0.022f * shift_;      // (the feet stay: they are held, or solved again)
            hips_shift_ += 0.022f * shift_;
            rotate_in_model(rig[Role::Hips], rot_z(-0.035f * shift_));
            rotate_in_model(rig[Role::Spine1], rot_z(0.045f * shift_));
        }
    }
    {
        // (walking, only somebody in particular breathes visibly: the players' soldiers are as they were)
        const float w = gaze_w_ * tuning.idle_sway * (moving ? 0.5f * own : 1.0f);
        if (w > 1e-3f) {
            // (nobody in particular breathes as everybody did: the traits count as far as `individuality` says)
            const float b = std::sin((time_ + style_.clock * own) * glm::mix(1.7f, style_.breath_rate, own) + float(in.variation & 7u));
            const float depth = glm::mix(1.0f, style_.breath_depth, own) * (sitting ? 1.3f : 1.0f);
            rotate_in_model(rig[Role::Spine2], rot_x(0.012f * w * depth * b));
            if (own > 0.0f && in.held == HeldItem::None) {
                // the chest fills: the shoulders rise with it and come down again
                rotate_in_model(rig[Role::Spine1], rot_x(0.007f * w * depth * own * b));
                rotate_in_model(rig[Role::LeftShoulder], rot_z(-0.014f * w * depth * own * b));
                rotate_in_model(rig[Role::RightShoulder], rot_z(0.014f * w * depth * own * b));
            }
        }
    }

    // ---- d, e ----
    // (the seat pose fits its arms after this: it calls gesture_layer itself, when they are done)
    if (base != 1) gesture_layer(in, dt, !busy, sitting, moving);
    look_layer(in, dt, busy, moving, sitting);
    // (which side the weight goes to next: drawn after the look has drawn, as it always was -
    //  the two share one sequence of chances, and its order is part of who somebody is)
    shift_left_ -= dt;
    if (shift_left_ <= 0.0f) {
        const float r = rnd();
        shift_t_ = r < 0.38f ? -1.0f : r < 0.76f ? 1.0f : 0.0f;
        shift_left_ = 3.5f + 6.0f * rnd();
    }
}

// a: the person's own posture. Constant, so it is there at every level of detail and through
// every clip - somebody does not straighten up to sit down.
void CharacterAnimator::posture_layer(const CharacterAnimationInput& in, bool sitting) {
    const float k = std::clamp(in.individuality, 0.0f, 1.5f);
    if (k <= 0.0f) return;
    const Rig& rig = set_.runtime->rig();
    const PersonStyle& p = style_;
    // The back: rounded from the bottom up, the neck forward with it, the head lifted again by
    // half of that (somebody who slouches still looks ahead, a little from under the brows).
    const float s = p.slouch * k * (sitting ? 1.15f : 1.0f);
    rotate_in_model(rig[Role::Spine], rot_z(-p.lean * k) * rot_x(-0.35f * s));
    rotate_in_model(rig[Role::Spine1], rot_x(-0.35f * s));
    rotate_in_model(rig[Role::Spine2], rot_x(-0.30f * s));
    rotate_in_model(rig[Role::Neck], rot_z(0.5f * p.lean * k) * rot_x(-0.25f * s));
    rotate_in_model(rig[Role::Head], glm::angleAxis(-p.head_turn * k, kY) * rot_z(-p.head_tilt * k) * rot_x(0.55f * s + p.head_nod * k));
    // The shoulders (up, dropped, one higher) and how far the arms are held from the body. Not
    // with something in the hands: those arms are fitted to it.
    if (in.held == HeldItem::None) {
        const float up_l = (p.shoulders - p.shoulder_tilt) * k, up_r = (p.shoulders + p.shoulder_tilt) * k;
        rotate_in_model(rig[Role::LeftShoulder], rot_z(-up_l));
        rotate_in_model(rig[Role::RightShoulder], rot_z(up_r));
        // (the arm hangs as it did - the shoulder's turn taken out again - and a little outward)
        rotate_in_model(rig[Role::LeftArm], rot_z(up_l - p.arms_out * k));
        rotate_in_model(rig[Role::RightArm], rot_z(-up_r + p.arms_out * k));
    }
}

// b: a clip over part of the body. The animator owns its time and its weight: asked for, it
// comes in over a few tenths of a second; no longer asked for, or another one asked for, it
// goes out first. Where the clip wraps it lets go for a moment (the clips are not made to loop).
void CharacterAnimator::clip_layer(const CharacterAnimationInput& in, float dt) {
    const IAnimationRuntime& rt = *set_.runtime;
    const ClipId want = (in.layer_clip >= 0 && in.layer_clip < rt.num_clips() && in.layer_weight > 0.0f) ? in.layer_clip : kNoClip;
    if ((want != layer_clip_ || (want >= 0 && in.layer_mask != layer_mask_)) && layer_w_ < 0.02f) {
        layer_clip_ = want;
        layer_mask_ = in.layer_mask;
        layer_t_ = want >= 0 ? style_.phase * rt.clip(want).duration : 0.0f;
        layer_w_ = layer_wv_ = 0.0f;
    }
    const float target = (want >= 0 && want == layer_clip_ && in.layer_mask == layer_mask_) ? std::clamp(in.layer_weight, 0.0f, 1.0f) : 0.0f;
    spring_to(layer_w_, layer_wv_, target, tuning.layer_halflife, dt);
    layer_w_ = std::clamp(layer_w_, 0.0f, 1.0f);
    const bool fresh = layer_prev_eff_ < 0.02f;
    layer_prev_eff_ = 0.0f;
    if (layer_clip_ < 0 || layer_w_ < 1e-3f) return;
    const float dur = std::max(rt.clip(layer_clip_).duration, 0.1f);
    layer_t_ += dt * in.layer_rate;
    if (layer_t_ >= dur) layer_t_ = std::fmod(layer_t_, dur);
    const float w = layer_w_ * smooth01(std::min(layer_t_, dur - layer_t_) / 0.4f);
    if (w < 1e-3f) return;
    rt.sample(*ctx_, 2, layer_clip_, layer_t_, next_);
    if (in.action_mirror) mirror_pose(rt.rig(), next_, scratch_);
    const std::vector<float>& m = set_.mask[int(layer_mask_)];
    for (size_t j = 0; j < m.size() && j < pose_.size(); ++j) {
        const float k = m[j] * w;
        if (k > 1e-4f) pose_[j].r = blend_kept(pose_[j].r, next_[j].r, k, layer_pa_[j], layer_pb_[j], fresh);
    }
    layer_eff_ = layer_prev_eff_ = w;
}

// d: a gesture. One arm is solved from its rest to a place on the body (the side of the head,
// the back of the neck, in front of the chest, up beside the head) and blended over what the
// arm was doing, in and out; or the weight is moved on the seat. The animator starts one by
// itself now and then (a fidget) where the caller allows it, and any can be asked for.
void CharacterAnimator::gesture_layer(const CharacterAnimationInput& in, float dt, bool alive, bool sitting, bool moving) {
    const IAnimationRuntime& rt = *set_.runtime;
    const Rig& rig = rt.rig();
    gest_look_yaw_ = gest_look_pitch_ = 0.0f;
    const bool hands_free = held_ == HeldItem::None && in.held == HeldItem::None && (in.fidget_arms & 3u) != 0;
    auto start = [&](Gesture g) {
        if (g != Gesture::Shift && !hands_free) return;
        gest_ = g;
        gest_t_ = 0.0f;
        const float r = rnd();
        gest_dur_ = g == Gesture::Wave ? 2.3f + 0.9f * r : g == Gesture::CheckWrist ? 2.6f + 1.2f * r : g == Gesture::Shift ? 1.8f + 0.9f * r : 2.4f + 1.8f * r;
        // (the wrist that has the watch is the left one, the hand that waves the right - if it is free)
        int side = g == Gesture::CheckWrist ? 0 : g == Gesture::Wave ? 1 : (rnd() < 0.5f ? 0 : 1);
        if (!(in.fidget_arms & (1u << side))) side = 1 - side;
        gest_side_ = side;
    };
    if (in.gesture_seq != gesture_seq_) {
        gesture_seq_ = in.gesture_seq;
        if (alive && in.gesture != Gesture::None && in.gesture < Gesture::Count && gest_ == Gesture::None) start(in.gesture);
    }
    if (alive && in.fidgets > 0.0f && !moving && gest_ == Gesture::None) {
        fidget_left_ -= dt * in.fidgets * glm::mix(1.0f, style_.fidget, std::clamp(in.individuality, 0.0f, 1.0f));
        if (fidget_left_ <= 0.0f) {
            fidget_left_ = 7.0f + 13.0f * rnd();
            const float r = rnd();
            start(!hands_free ? Gesture::Shift : r < 0.30f ? Gesture::ScratchHead : r < 0.52f ? Gesture::RubNeck : r < 0.76f ? Gesture::CheckWrist : Gesture::Shift);
        }
    }
    const bool fresh = gest_prev_w_ < 0.02f;
    gest_prev_w_ = 0.0f;
    if (gest_ == Gesture::None) { gest_w_ = 0.0f; return; }
    gest_t_ += dt;
    if (!alive) gest_dur_ = std::min(gest_dur_, gest_t_ + 0.3f);            // (something came up: let go)
    const float env = gest_t_ >= gest_dur_ ? 0.0f : smooth01(gest_t_ / 0.5f) * smooth01((gest_dur_ - gest_t_) / 0.55f);
    gest_w_ += std::clamp(env - gest_w_, -4.0f * dt, 4.0f * dt);            // (never a jump, whatever cut it short)
    if (gest_t_ >= gest_dur_ && gest_w_ < 1e-3f) { gest_ = Gesture::None; gest_w_ = 0.0f; return; }
    const float w = gest_w_, t = gest_t_;
    const float sx = gest_side_ == 0 ? -1.0f : 1.0f;

    if (gest_ == Gesture::Shift) {
        // One way and back the other, the back straightened on the way (on a seat: lifted a little).
        const float u = std::clamp(t / std::max(gest_dur_, 0.1f), 0.0f, 1.0f);
        const float sway = std::sin(u * 2.0f * kPi) * w;
        const int hips = rig[Role::Hips];
        pose_[size_t(hips)].t.x += 0.018f * sx * sway;
        hips_shift_ += 0.018f * sx * sway;
        if (sitting) pose_[size_t(hips)].t.y += 0.010f * w;
        rotate_in_model(rig[Role::Spine], rot_z(-0.05f * sx * sway) * rot_x(-0.06f * w));
        rotate_in_model(rig[Role::Spine2], rot_x(0.07f * w));
        return;
    }

    const int ua = rig[sx < 0.0f ? Role::LeftArm : Role::RightArm], fa = rig[sx < 0.0f ? Role::LeftForeArm : Role::RightForeArm],
              hd = rig[sx < 0.0f ? Role::LeftHand : Role::RightHand];
    if (ua < 0 || fa < 0 || hd < 0) return;
    rt.local_to_model(*ctx_, pose_, model_);
    // (the body's own frame, flat: "in front" is in front of this chest, whichever way the clip has turned it)
    const glm::vec3 cf = glm::mat3(model_[size_t(rig[Role::Spine2])]) * glm::vec3(0.0f, 0.0f, -1.0f);
    const glm::quat B = glm::angleAxis(-std::atan2(cf.x, -cf.z), kY);
    const glm::vec3 head(model_[size_t(rig[Role::Head])][3]), neck(model_[size_t(rig[Role::Neck])][3]), shoulder(model_[size_t(ua)][3]);
    glm::vec3 target, pole;
    float curl = 0.5f;
    switch (gest_) {
    case Gesture::ScratchHead:
        // (the wrist beside the head: the hand goes on along the forearm, the fingers on the crown)
        target = head + B * glm::vec3(sx * 0.16f, -0.02f + 0.012f * std::sin(t * 11.0f), 0.03f + 0.010f * std::sin(t * 11.0f + 1.3f));
        pole = B * glm::vec3(sx * 0.9f, -0.5f, -0.4f);
        curl = 0.75f + 0.2f * std::sin(t * 11.0f);
        break;
    case Gesture::RubNeck:
        target = neck + B * glm::vec3(sx * (0.13f + 0.012f * std::sin(t * 5.0f)), 0.03f, 0.10f + 0.006f * std::sin(t * 5.0f + 1.0f));
        pole = B * glm::vec3(sx * 0.9f, 0.35f, -0.2f);
        curl = 0.55f;
        gest_look_pitch_ = -0.14f * w;                           // (the head bows into the hand)
        break;
    case Gesture::CheckWrist:
        target = neck + B * glm::vec3(0.0f, -0.24f, -0.26f);
        pole = B * glm::vec3(sx * 1.0f, -0.5f, 0.1f);
        curl = 0.5f;
        gest_look_yaw_ = sx * 0.06f * w;
        gest_look_pitch_ = -0.42f * w;                           // (the head goes down to it)
        break;
    default:   // Wave
        target = shoulder + B * glm::vec3(sx * 0.18f + 0.085f * std::sin(t * 8.5f), 0.34f + 0.02f * std::cos(t * 8.5f), -0.12f);
        pole = B * glm::vec3(sx * 0.8f, -0.7f, 0.2f);
        curl = 0.10f;
        break;
    }
    const glm::quat r_ua = pose_[size_t(ua)].r, r_fa = pose_[size_t(fa)].r, r_hd = pose_[size_t(hd)].r;
    fit_limb(ua, fa, hd, target, &pole, true);
    pose_[size_t(ua)].r = blend_kept(r_ua, pose_[size_t(ua)].r, w, gest_pa_[0], gest_pb_[0], fresh);
    pose_[size_t(fa)].r = blend_kept(r_fa, pose_[size_t(fa)].r, w, gest_pa_[1], gest_pb_[1], fresh);
    pose_[size_t(hd)].r = blend_kept(r_hd, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), w, gest_pa_[2], gest_pb_[2], fresh);     // (the hand along the forearm)
    gest_prev_w_ = w;
    const AnimationSet::Hand& H = set_.hand[gest_side_];
    if (H.ok && in.lod < 2)
        for (int f = 0; f < 4; ++f)
            for (int j = 0; j < 3; ++j)
                if (H.finger[f][j] >= 0)
                    pose_[size_t(H.finger[f][j])].r = glm::normalize(glm::slerp(pose_[size_t(H.finger[f][j])].r, glm::angleAxis(curl * (j == 0 ? 0.6f : 1.0f), H.b0), w));
    rt.local_to_model(*ctx_, pose_, model_, ua);
}

// e: the look. Not aiming, a character is not a statue: the head looks about - at what the
// caller says is worth a look (look_at: somebody, a car going by, the plate), off to a side, up
// or down, or nowhere for a while.
//   The turn is ADDED to the head the layers below made, and measured from it: a clip that
//   already has the head turned is not turned twice. What it may come to is limited from the
//   way the chest faces (AnimTuning::look_yaw_max, look_pitch_up / _down); the eyes do the rest.
void CharacterAnimator::look_layer(const CharacterAnimationInput& in, float dt, bool busy, bool moving, bool sitting) {
    const Rig& rig = set_.runtime->rig();
    const int head = rig[Role::Head];
    (void)sitting;
    // the head as it is without the look, and the chest it sits on
    const glm::vec3 hf = model_rotation(head) * glm::vec3(0.0f, 0.0f, -1.0f);
    const glm::vec3 cf = model_rotation(rig[Role::Spine1]) * glm::vec3(0.0f, 0.0f, -1.0f);
    const float base_yaw = std::atan2(hf.x, -hf.z), base_pitch = std::asin(std::clamp(hf.y, -1.0f, 1.0f));
    const float chest_yaw = std::atan2(cf.x, -cf.z);
    const float own_yaw = wrap_angle(base_yaw - chest_yaw);       // the head's turn that is already there
    // what to look at next
    gaze_left_ -= dt;
    // (where what is worth a look is, from the head: what has to be added to get there)
    float at_yaw = 0.0f, at_pitch = 0.0f;
    bool can_see = false;
    if (in.has_look_at) {
        const glm::vec3 d = in.look_at - glm::vec3(model_[size_t(head)][3]);
        const float flat = std::sqrt(d.x * d.x + d.z * d.z);
        const float ay = std::atan2(d.x, -d.z);
        at_yaw = wrap_angle(ay - base_yaw);
        at_pitch = std::atan2(d.y, std::max(flat, 1e-3f)) - base_pitch;
        can_see = flat > 0.2f && flat < 60.0f && std::abs(wrap_angle(ay - chest_yaw)) < tuning.look_yaw_max + 0.70f;
    }
    if (in.look_id != look_id_) {
        look_id_ = in.look_id;
        if (can_see) gaze_left_ = std::min(gaze_left_, 0.15f + 0.7f * rnd());      // (something new: it catches the eye soon)
    }
    // A head is heavy and mostly in no hurry. Small shifts of attention are made with the
    // eyes; the head follows the larger ones, slowly, and holds for seconds. So: long rests,
    // unhurried turns (a quick one only now and then, when something catches the eye), slow
    // sweeps across the view, and small angles far more often than large ones.
    if (gaze_left_ <= 0.0f || (gaze_mode_ == 2 && !can_see)) {
        const float r = rnd();
        const float p_look = !can_see ? 0.0f : in.look_weight >= 4.0f ? 1.0f
                           : std::min((moving ? 0.16f : 0.28f) * std::max(in.look_weight, 0.0f) * glm::mix(1.0f, style_.curious, std::clamp(in.individuality, 0.0f, 1.0f)), 0.9f);
        const float p_side = moving ? 0.14f : 0.20f;
        const float p_sweep = moving ? 0.10f : 0.18f;
        // how quickly: mostly slow, sometimes ordinary, rarely quick
        const float q = rnd();
        const float slow = 0.45f + 0.25f * rnd(), ordinary = 0.27f + 0.10f * rnd(), quick = 0.13f + 0.04f * rnd();
        gaze_sweep_ = 0.0f;
        if (r < p_look) {
            gaze_mode_ = 2;                                             // somebody: a little more alert
            gaze_half_ = q < 0.40f ? slow : q < 0.88f ? ordinary : quick;
            gaze_left_ = moving ? 2.0f + 2.5f * rnd() : 3.5f + 5.0f * rnd();
        } else if (r < p_look + p_side) {
            gaze_mode_ = 1;
            gaze_half_ = q < 0.68f ? slow : q < 0.94f ? ordinary : quick;
            const float side = rnd() < 0.5f ? -1.0f : 1.0f;
            const float far_look = rnd() < 0.2f ? 0.45f * rnd() : 0.0f;  // now and then well round
            gaze_yaw_t_ = side * (0.20f + 0.40f * rnd() + far_look) * (moving ? 0.75f : 1.0f);
            // level most of the time; sometimes up at the sky or down at the ground
            const float v = rnd();
            gaze_pitch_t_ = v < 0.27f ? 0.15f + 0.22f * rnd() : v < 0.52f ? -0.14f - 0.22f * rnd() : (rnd() - 0.5f) * 0.12f;
            if (moving) gaze_pitch_t_ *= 0.5f;
            gaze_left_ = moving ? 1.8f + 2.2f * rnd() : 2.5f + 4.0f * rnd();
        } else if (r < p_look + p_side + p_sweep) {
            gaze_mode_ = 3;                                             // a slow look across, from where the head is
            gaze_half_ = 0.30f;
            const float side = gaze_yaw_ > 0.05f ? -1.0f : gaze_yaw_ < -0.05f ? 1.0f : (rnd() < 0.5f ? -1.0f : 1.0f);
            gaze_sweep_to_ = side * (0.30f + 0.45f * rnd()) * (moving ? 0.7f : 1.0f);
            gaze_sweep_ = 0.10f + 0.16f * rnd();                        // 6 - 15 degrees a second
            gaze_yaw_t_ = gaze_yaw_;
            gaze_pitch_t_ = (rnd() - 0.5f) * 0.10f;
            gaze_left_ = std::abs(gaze_sweep_to_ - gaze_yaw_) / gaze_sweep_ + 0.8f + 1.5f * rnd();
        } else {
            gaze_mode_ = 0;
            gaze_half_ = q < 0.75f ? slow : ordinary;
            gaze_left_ = moving ? 4.0f + 6.0f * rnd() : 3.0f + 6.0f * rnd();
        }
    }
    if (gaze_mode_ == 3) {
        const float step = gaze_sweep_ * dt, d = gaze_sweep_to_ - gaze_yaw_t_;
        gaze_yaw_t_ += std::clamp(d, -step, step);
    }
    float ty = 0.0f, tp = 0.0f;
    if (gaze_mode_ == 2) { ty = at_yaw; tp = at_pitch; }
    else if (gaze_mode_ == 1 || gaze_mode_ == 3) { ty = gaze_yaw_t_; tp = gaze_pitch_t_; }
    // (never quite still: a degree or two of drift)
    const float ph = float(in.variation & 15u);
    ty += 0.022f * std::sin(time_ * 0.31f + ph) + 0.012f * std::sin(time_ * 0.73f + 2.0f * ph);
    tp += 0.010f * std::sin(time_ * 0.43f + 3.0f * ph);
    // (a gesture takes the head: down to the wrist)
    ty += gest_look_yaw_;
    tp += gest_look_pitch_;
    // The limits: what the head comes to, the clip's own turn included, from the chest. (A clip
    // that goes beyond them by itself is left as it is: the look adds nothing that way.)
    const float y_lo = std::min(-tuning.look_yaw_max - own_yaw, 0.0f), y_hi = std::max(tuning.look_yaw_max - own_yaw, 0.0f);
    const float p_lo = std::min(-tuning.look_pitch_down - base_pitch, 0.0f), p_hi = std::max(tuning.look_pitch_up - base_pitch, 0.0f);
    ty = std::clamp(ty, y_lo, y_hi) * gaze_w_ * tuning.idle_look;
    tp = std::clamp(tp, p_lo, p_hi) * gaze_w_ * tuning.idle_look;
    // (the aim, a shot: the head is back on it at once, whatever its pace was)
    const float half = busy ? 0.09f : gaze_half_;
    gaze_want_yaw_ = ty;
    gaze_want_pitch_ = tp;
    spring_to(gaze_yaw_, gaze_yaw_v_, ty, half, dt);
    spring_to(gaze_pitch_, gaze_pitch_v_, tp, half * 1.15f, dt);
    // (the clip turns the head the same way while the look is still on its way back: the limit holds all the same)
    if (gaze_yaw_ < y_lo || gaze_yaw_ > y_hi) { gaze_yaw_ = std::clamp(gaze_yaw_, y_lo, y_hi); gaze_yaw_v_ = 0.0f; }
    if (gaze_pitch_ < p_lo || gaze_pitch_ > p_hi) { gaze_pitch_ = std::clamp(gaze_pitch_, p_lo, p_hi); gaze_pitch_v_ = 0.0f; }
    if (std::abs(gaze_yaw_) + std::abs(gaze_pitch_) > 1e-4f) {
        // the eyes lead, the neck follows, the chest gives a little (+yaw looks right: about -Y);
        // up and down is about the head's own side-to-side axis as the layers below left it
        const glm::quat pitch_ax = glm::angleAxis(-base_yaw, kY);
        auto turn = [&](float yaw, float pitch) { return glm::angleAxis(-yaw, kY) * pitch_ax * rot_x(pitch) * glm::inverse(pitch_ax); };
        rotate_in_model(rig[Role::Spine2], glm::angleAxis(-0.12f * gaze_yaw_, kY));
        rotate_in_model(rig[Role::Neck], turn(0.33f * gaze_yaw_, 0.35f * gaze_pitch_));
        rotate_in_model(head, turn(0.55f * gaze_yaw_, 0.65f * gaze_pitch_));
    }
}

}  // namespace vibe::anim
