#pragma once
// Motion matching: the character's pose comes from a database of motion capture, not from
// clips wired into a state machine. Several times a second the database is searched for the
// frame that best continues the pose the character is in AND goes where the character is
// about to go (feet, their speeds, the hips' speed, and the path and facing over the next
// second); playback jumps there, and the jump is hidden by inertialisation - the difference
// between the old and the new pose is carried on as an offset that decays like a damped
// spring, so nothing ever blends two clips together (no foot-sliding cross-fades).
//
// The database (assets/character/motion.qmm) is built by tools/anim/build_motion_db.py from
// the 100STYLE capture set. Every frame also exists mirrored (made at load).
//
// Refs: Clavet, "Motion Matching and The Road to Next-Gen Animation" (GDC 2016); Bollo,
// "Inertialization" (GDC 2018); Holden et al., "Learned Motion Matching" (SIGGRAPH 2020).
#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace qlike {

struct MotionDB {
    struct Range { uint32_t first, end, tag; };
    static constexpr int kFeat = 27;
    uint32_t frames = 0, bones = 0;
    float    fps = 30.0f;
    std::vector<std::string> bone_names;
    std::vector<Range>       ranges;
    std::vector<float>       feat_offset, feat_scale;
    std::vector<float>       feat;          // frames x kFeat, normalised
    std::vector<float>       hdr;           // frames x 8: root vel x,z; turn rate; hip x,y,z; contact L,R
    std::vector<glm::quat>   rot;           // frames x bones: rotation from rest, root space
    std::vector<uint32_t>    range_of;      // frame -> its range

    bool load(const char* path);            // false (and empty) if the file is missing or bad
    bool ok() const { return frames > 0; }
};

struct MotionQuery {
    glm::vec2 vel{0.0f};      // wanted velocity, root space (x right, z back: forward is -z), m/s
    float     turn_rate = 0;  // rad/s, positive = turning right
    uint32_t  tag = 0;        // which style of the database
    float     rate = 1.0f;    // playback rate
};

struct MotionPose {
    std::vector<glm::quat> rot;       // per database bone: rotation from rest, root space
    glm::vec3 hip{0.0f};              // x,z offset from the rest hips, y absolute (file units)
    glm::vec2 root_vel{0.0f};         // the animation's own velocity at this moment (root space)
    float     contact[2]{};           // left, right foot on the ground
};

struct MotionState {
    int   frame = -1;
    float frac = 0.0f;
    float since_search = 1e9f;
    glm::vec2 last_vel{0.0f};
    uint32_t  last_tag = 0;
    std::vector<glm::vec3> off_x, off_v;     // per bone: rotation offset (scaled axis-angle) and its rate
    glm::vec3 hip_x{0.0f}, hip_v{0.0f};
};

// Advances the state by dt and gives the pose.
void motion_update(const MotionDB& db, MotionState& st, const MotionQuery& q, float dt, MotionPose& out);

}  // namespace qlike
