#pragma once
// An uncompressed clip on the character's own skeleton (offline only): what the importers
// produce and what the cook step turns into runtime clips and motion-database frames.
#include "anim/skeleton_def.h"

namespace vibe::anim::offline {

struct RawClip {
    ClipInfo               info;
    int                    frames = 0;
    int                    joints = 0;
    std::vector<Transform> local;          // frames x joints; joint 0 (the root) carries the clip's travel
    std::vector<uint8_t>   contact;        // frames x 2: left, right foot on the ground (filled by label_contacts)

    Transform&       at(int f, int j) { return local[size_t(f) * size_t(joints) + size_t(j)]; }
    const Transform& at(int f, int j) const { return local[size_t(f) * size_t(joints) + size_t(j)]; }
    float            time_of(int f) const { return float(f) / info.sample_rate; }
};

// Model-space joint rotations and positions of one frame, the root's travel included
// ("clip space": where the performer was in the capture volume, re-based to start at the origin).
void clip_fk(const SkeletonDef& s, const RawClip& c, int frame, std::vector<glm::quat>& rot, std::vector<glm::vec3>& pos);

// Foot contacts from the motion itself: a foot is down while its ankle is near its rest
// height and nearly still. Short flickers are removed.
struct ContactParams {
    float max_speed = 0.32f;      // m/s
    float max_height = 0.065f;    // metres above the ankle's rest height
    float smooth_sigma = 1.2f;    // frames
};
void label_contacts(const SkeletonDef& s, RawClip& c, const ContactParams& p = {});

struct ClipReport {
    bool        ok = true;
    std::string errors;           // what makes the clip unusable
    std::string warnings;
    float       mean_speed = 0.0f, max_speed = 0.0f;       // of the root, m/s
    float       contact_left = 0.0f, contact_right = 0.0f; // share of frames
    float       max_limb_error = 0.0f;                     // metres a bone's length is off its rest length
    float       lowest_ankle = 0.0f;                       // metres relative to the ankle's rest height
};
// Checks a clip against the rig: joint count, finite unit quaternions, bone lengths, root
// speed within `max_root_speed`, feet not under the ground, frame rate, boundaries.
ClipReport validate_clip(const SkeletonDef& s, const RawClip& c, float max_root_speed = 14.0f);

}  // namespace vibe::anim::offline
