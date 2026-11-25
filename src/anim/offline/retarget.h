#pragma once
// Retargeting a motion capture skeleton onto the character's rig (offline only).
//
// The capture skeleton stands in a T pose, the soldier in an A pose with other proportions.
// Per bone, the capture's world rotation is taken relative to the pose in which its bones
// point the way the soldier's do at rest - so "the soldier's rest pose" maps to identity and
// the result is the rotation that takes each soldier bone from rest to the captured pose.
// Positions are scaled leg for leg; the lower ankle's typical height is put at the soldier's
// ankle rest height, so feet meet the ground.
//
// The root: the hips' path on the ground, smoothed, facing the smoothed hip direction. It is
// stored in joint 0 of the clip, so the body's own motion (hips, legs) is relative to a root
// that travels steadily - what motion matching and root-motion extraction need.
#include "anim/offline/bvh.h"
#include "anim/offline/raw_clip.h"

namespace vibe::anim::offline {

struct BoneMap {
    std::string target;          // soldier joint
    std::string source;          // capture joint
    std::string target_child;    // gives the soldier bone its direction ("" = no direction fix, "TAIL" = the bone's own tail)
    std::string source_child;    // the capture joint that does the same ("END" = the joint's end site)
};

struct RetargetParams {
    std::vector<BoneMap> map;
    glm::dmat3 axes{1.0};        // capture axes -> character axes (a rotation)
    float      out_rate = 30.0f; // Hz; the source rate must be a whole multiple
    float      root_smooth_s = 0.20f;
    // Which of the lower ankle's heights over the take is the ground: the middle one for takes that
    // mostly stand or walk, a low one for runs (their feet are in the air most of the time).
    float      ground_percentile = 0.5f;
    ContactParams contact;       // when a foot counts as on the ground
    std::string leg_upper = "LeftKnee", leg_lower = "LeftAnkle";      // capture joints whose offsets are the leg's two bones
};

// The mapping for the 100STYLE capture skeleton (+Z forward, +X left, centimetres).
RetargetParams style100_params();

// The mapping for the CMU Graphics Lab motion capture skeleton in its usual BVH conversion
// (+Z forward, +X left, units of about 5.6 cm; the scale comes from the legs).
RetargetParams cmu_params();

// The mapping for motions written by tools outside the repo as the SMPL-H body (22 joints named
// Hips, L_Hip, L_Knee ... as C:/Users/.../hy-motion/vibe_generate.py writes them: metres, +Z
// forward, +X the body's left, 30 Hz). See docs/animation/actions.md.
RetargetParams smplh_params();

// Frames [first, end) of the capture, resampled to params.out_rate. `skel` in metres.
bool retarget_bvh(const Bvh& bvh, int first, int end, const SkeletonDef& skel, const RetargetParams& params, RawClip& out,
                  std::string* why = nullptr);

}  // namespace vibe::anim::offline
