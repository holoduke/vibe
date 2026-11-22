#pragma once
// The canonical humanoid rig: the soldier's skeleton as the character file gives it (names,
// parents, rest joint positions), plus the roles the animation code needs by name.
#include "anim/types.h"

namespace vibe::anim {

enum class Role : int {
    Root, Hips, Spine, Spine1, Spine2, Neck, Head,
    LeftShoulder, LeftArm, LeftForeArm, LeftHand,
    RightShoulder, RightArm, RightForeArm, RightHand,
    LeftUpLeg, LeftLeg, LeftFoot, LeftToe,
    RightUpLeg, RightLeg, RightFoot, RightToe,
    Count
};
const char* role_name(Role r);

struct SkeletonDef {
    std::vector<std::string> names;
    std::vector<int>         parents;      // -1 = root; parents come before children
    std::vector<glm::vec3>   rest_pos;     // joint position at rest, model space, metres
    std::vector<glm::vec3>   rest_tail;    // the bone's far end at rest (direction of leaf bones)
    float                    scale = 1.0f; // file units -> metres, already applied to the positions

    int  size() const { return int(names.size()); }
    int  find(std::string_view name) const;          // -1 if absent
    // Parents before children, exactly one root, finite positions, unique names.
    bool validate(std::string* why = nullptr) const;
    // Identifies the rig a clip or a database was made for (names, hierarchy, rest positions to a millimetre).
    uint64_t rig_id() const;
    // Local rest translation of a joint (rest_pos relative to its parent's).
    glm::vec3 rest_local(int j) const { return parents[j] < 0 ? rest_pos[j] : rest_pos[j] - rest_pos[parents[j]]; }
};

// Reads the skeleton block of a character file ("QCH1", tools/character/export_fp.py).
// Positions stay in file units (scale = 1); see scale_skeleton.
bool load_qchar_skeleton(const char* path, SkeletonDef& out, std::string* why = nullptr);
void scale_skeleton(SkeletonDef& s, float factor);
// The factor that puts the model's eyes at `stand_eye_height` metres (the same rule the engine
// uses for the mesh: the eye is 3.5 cm above the head joint).
float eye_height_scale(const SkeletonDef& file_units, float stand_eye_height);

// Joint index per role (-1 if the skeleton lacks it), and the left/right partner of every joint.
struct Rig {
    int role[int(Role::Count)];
    std::vector<int> mirror;       // joint -> the joint of the other side (itself on the centre line)
    int operator[](Role r) const { return role[int(r)]; }
    bool complete() const;         // every role present
};
Rig make_rig(const SkeletonDef& s);

}  // namespace vibe::anim
