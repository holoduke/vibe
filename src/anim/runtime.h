#pragma once
// The skeletal animation runtime as the rest of the game sees it: a skeleton, compressed clips,
// sampling, blending (full-body, masked, additive), local -> model transforms, IK, the skinning
// palette. The implementation is ozz-animation (ozz_runtime.cpp); nothing outside that file
// knows.
//
// Poses are plain arrays, one Transform per joint in skeleton order (scale is always 1):
//   local pose  std::span<Transform>   each joint relative to its parent
//   model pose  std::span<glm::mat4>   each joint in character model space
//
// Thread safety: every method is const and may be called from several threads at once as long
// as each thread uses its own EvalContext and its own output arrays.
#include "anim/asset.h"

#include <memory>

namespace vibe::anim {

struct PoseLayer {
    std::span<const Transform> pose;
    float                      weight = 1.0f;
    std::span<const float>     joint_weights;      // optional: one factor per joint (a mask)
};

struct TwoBoneIk {
    int       start = -1, mid = -1, end = -1;      // e.g. thigh, shin, foot
    glm::vec3 target{0.0f};                        // model space: where `end` goes
    glm::vec3 pole{0.0f, 0.0f, -1.0f};             // model space: the side the middle joint bends to
    glm::vec3 mid_axis{1.0f, 0.0f, 0.0f};          // the middle joint's hinge axis in its own space, used while the limb
                                                   // is straight (bent, the axis is taken from the bones themselves)
    float     soften = 0.97f;                      // share of the limb's length reached without easing
    float     weight = 1.0f;
};

struct AimIk {
    int       joint = -1;
    glm::vec3 target{0.0f};                        // model space
    glm::vec3 forward{0.0f, 0.0f, -1.0f};          // joint space: the axis to point at the target
    glm::vec3 up{0.0f, 1.0f, 0.0f};                // joint space
    glm::vec3 pole{0.0f, 1.0f, 0.0f};              // model space: where `up` should lean
    float     weight = 1.0f;
};

// Per-character (or per-worker) evaluation state: sampling caches and scratch arrays. Reusing
// one context for the same clips frame after frame makes sampling cheap (the keyframe cursor
// moves forward instead of searching).
class EvalContext {
public:
    virtual ~EvalContext() = default;
};

class IAnimationRuntime {
public:
    virtual ~IAnimationRuntime() = default;

    virtual const SkeletonDef& skeleton() const = 0;
    virtual const Rig&         rig() const = 0;
    virtual int                num_clips() const = 0;
    virtual const ClipInfo&    clip(ClipId id) const = 0;
    virtual ClipId             find_clip(std::string_view name) const = 0;

    // `sample_slots`: how many clips this context samples per frame (one cache each).
    virtual std::unique_ptr<EvalContext> create_context(int sample_slots) const = 0;

    virtual void rest_pose(std::span<Transform> out) const = 0;
    // The clip at `time` seconds: wrapped if the clip loops, clamped to its ends otherwise.
    // The root joint carries the clip's own travel (see root_motion); callers that move the
    // character themselves overwrite it.
    virtual void sample(EvalContext& ctx, int slot, ClipId id, float time, std::span<Transform> out) const = 0;
    // Weighted average of `layers` (weights need not sum to 1; below a small total the rest
    // pose fills in), then each of `additive` applied on top: its pose is a difference
    // (translation added, rotation multiplied), scaled by its weight.
    virtual void blend(EvalContext& ctx, std::span<const PoseLayer> layers, std::span<const PoseLayer> additive,
                       std::span<Transform> out) const = 0;
    // from < 0: the whole skeleton. Otherwise only `from` and the joints below it are updated.
    virtual void local_to_model(EvalContext& ctx, std::span<const Transform> local, std::span<glm::mat4> model,
                                int from = -1) const = 0;
    // Both change `local` and bring `model` up to date for the joints they moved.
    // two_bone_ik returns false if the target is out of reach (the limb is then stretched toward it).
    virtual bool two_bone_ik(EvalContext& ctx, const TwoBoneIk& ik, std::span<Transform> local,
                             std::span<glm::mat4> model) const = 0;
    virtual void aim_ik(EvalContext& ctx, const AimIk& ik, std::span<Transform> local, std::span<glm::mat4> model) const = 0;
    // model pose -> matrices that take a rest-pose vertex (model space) to its posed place.
    virtual void skinning_matrices(std::span<const glm::mat4> model, std::span<glm::mat4> out) const = 0;
    // How the clip's root moves from t0 to t1 (seconds, t0 <= t1 within the clip; no wrapping),
    // expressed in the root's frame at t0.
    virtual Transform root_motion(EvalContext& ctx, ClipId id, float t0, float t1) const = 0;

    virtual size_t clip_bytes() const = 0;          // compressed clip data held in memory
    virtual const char* name() const = 0;
};

// The format string a runtime writes into / expects from AnimAsset::runtime_format.
const char* runtime_format_name();
// nullptr (and `why`) if the asset's blocks are not for this runtime or are damaged.
std::unique_ptr<IAnimationRuntime> create_runtime(const AnimAsset& asset, std::string* why = nullptr);

// ---- pose helpers that need no runtime ----
// Mirrors a local pose left <-> right (the model's x axis), in place. `scratch` is num_joints long.
void mirror_pose(const Rig& rig, std::span<Transform> pose, std::span<Transform> scratch);
bool pose_is_valid(std::span<const Transform> pose);      // finite, unit quaternions
bool pose_is_valid(std::span<const glm::mat4> pose);

}  // namespace vibe::anim
