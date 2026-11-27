// HOLDING SOMETHING (CharacterAnimationInput::hold_*): the last layer before the pose is handed on.
// Whatever made the pose - motion matching, an action clip, the seat - and whatever the life
// layers put on it, a hand that holds something is taken to it: the arm solved to the grip (two
// bones, the elbow out and down), the hand turned so its fingers and palm lie as the grip says,
// the fingers closed round it. Eased in and out (hold_halflife), so a cup taken up or a box set
// down is a movement, not a jump; the caller moves the grip itself for what the hand DOES with
// it (a cup to the mouth, a box down on a table: npc_world.cpp "WHAT THEY CARRY").
#include "anim/character_animator.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>

namespace vibe::anim {

void CharacterAnimator::hold_layer(const CharacterAnimationInput& in, float dt) {
    const IAnimationRuntime& rt = *set_.runtime;
    const Rig& rig = rt.rig();
    const float e = in.hold_halflife > 1e-4f ? 1.0f - std::exp(-dt * 0.69314718f / in.hold_halflife) : 1.0f;
    bool any = false;
    for (int side = 0; side < 2; ++side) {
        const float want = (in.hold_hands >> side) & 1u ? 1.0f : 0.0f;
        if (!started_ || in.teleported) hold_w_[side] = want;
        hold_w_[side] += (want - hold_w_[side]) * e;
        if (hold_w_[side] < 1e-3f) hold_w_[side] = 0.0f;
        any = any || hold_w_[side] > 0.0f;
    }
    if (!any || in.lod >= 3) return;
    rt.local_to_model(*ctx_, pose_, model_);
    for (int side = 0; side < 2; ++side) {
        const float w = hold_w_[side];
        if (w <= 0.0f) continue;
        const AnimationSet::Hand& H = set_.hand[side];
        const int ua = rig[side == 0 ? Role::LeftArm : Role::RightArm], fa = rig[side == 0 ? Role::LeftForeArm : Role::RightForeArm],
                  hd = rig[side == 0 ? Role::LeftHand : Role::RightHand];
        if (!H.ok || ua < 0 || fa < 0 || hd < 0) continue;
        const glm::quat r_ua = pose_[size_t(ua)].r, r_fa = pose_[size_t(fa)].r, r_hd = pose_[size_t(hd)].r;
        const float sx = side == 0 ? -1.0f : 1.0f;
        glm::vec3 a1 = glm::length(in.hold_fingers[side]) > 1e-4f ? glm::normalize(in.hold_fingers[side]) : glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 n1 = in.hold_palm[side] - a1 * glm::dot(in.hold_palm[side], a1);
        n1 = glm::length(n1) > 1e-4f ? glm::normalize(n1) : glm::vec3(-sx, 0.0f, 0.0f);
        // the palm's middle is on the grip: the wrist a palm's length back along the fingers, and off the surface
        const glm::vec3 wrist = in.hold_grip[side] - a1 * (H.palm * 0.80f) - n1 * 0.028f;
        const glm::vec3 pole(sx * 0.9f, -1.0f, 0.5f);
        fit_limb(ua, fa, hd, wrist, &pole, true);
        const glm::mat3 F0(H.a0, H.n0, H.b0), F1(a1, n1, glm::cross(a1, n1));
        const glm::quat G = glm::quat_cast(F1 * glm::transpose(F0));
        const glm::quat Rf = glm::quat_cast(glm::mat3(model_[size_t(fa)]));
        pose_[size_t(hd)].r = glm::normalize(glm::inverse(Rf) * G);
        // eased: between where the arm was and where the grip wants it
        const float s = w * w * (3.0f - 2.0f * w);
        if (s < 0.999f) {
            pose_[size_t(ua)].r = glm::slerp(r_ua, pose_[size_t(ua)].r, s);
            pose_[size_t(fa)].r = glm::slerp(r_fa, pose_[size_t(fa)].r, s);
            pose_[size_t(hd)].r = glm::slerp(r_hd, pose_[size_t(hd)].r, s);
        }
        // the fingers round it, the thumb over them
        if (in.lod < 2) {
            const float c = std::clamp(in.hold_curl[side], 0.0f, 1.0f) * s;
            static const float kCurl[3] = { 1.05f, 1.30f, 0.85f };
            for (int f = 0; f < 4; ++f)
                for (int j = 0; j < 3; ++j)
                    if (H.finger[f][j] >= 0) pose_[size_t(H.finger[f][j])].r = glm::slerp(pose_[size_t(H.finger[f][j])].r, glm::angleAxis(kCurl[j] * (0.35f + 0.65f * c), H.b0), s);
            for (int j = 0; j < 3; ++j)
                if (H.thumb[j] >= 0)
                    pose_[size_t(H.thumb[j])].r = glm::slerp(pose_[size_t(H.thumb[j])].r,
                        glm::angleAxis(-sx * (j == 0 ? 0.70f * c : 0.0f), H.a0) * glm::angleAxis(j == 0 ? 0.30f : 0.45f * c, H.b0), s);
        }
        rt.local_to_model(*ctx_, pose_, model_);
    }
}

}  // namespace vibe::anim
