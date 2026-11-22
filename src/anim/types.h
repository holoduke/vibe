#pragma once
// Character animation library (vibe_anim): shared small types.
//
// Conventions (the engine's): right-handed, +Y up, metres, radians. Character model space is
// +X right, +Y up, -Z forward, feet at the origin. The soldier's joints have NO rest rotation:
// every joint's rest orientation is identity in model space, so a joint's local rotation is
// "how far it is turned from rest, seen from its parent".
//
// Nothing in src/anim includes Vulkan, GLFW or engine headers; ozz headers are included only by
// ozz_runtime.cpp and offline/.
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vibe::anim {

struct Transform {
    glm::vec3 t{0.0f};
    glm::quat r{1.0f, 0.0f, 0.0f, 0.0f};
};

using ClipId = int;
constexpr ClipId kNoClip = -1;

// What a clip (and every database frame cut from it) is good for.
enum ClipTag : uint32_t {
    kTagLocomotion = 1u << 0,   // standing, walking, running, their transitions
    kTagCrouch     = 1u << 1,   // crouched locomotion
    kTagJump       = 1u << 2,   // contains flights (take-off .. landing)
    kTagIdle       = 1u << 3,   // standing still
    kTagShortTake  = 1u << 4,   // a take of a second or two (a run cut after a few strides): for the search its end
                                // is continued at its last pace, and all but its last frames can be jumped to
};

struct ClipInfo {
    std::string name;
    float       duration = 0.0f;       // seconds
    float       sample_rate = 30.0f;   // the rate the source was resampled to
    bool        loop = false;
    uint32_t    tags = 0;
    std::string provenance;            // source file and frame range
    std::string licence;
};

// Rotation as "scaled axis-angle" (axis * angle) and back; the shortest arc.
glm::vec3 quat_to_scaled_axis(glm::quat q);
glm::quat quat_from_scaled_axis(glm::vec3 v);
// The rotation taking unit vector a to unit vector b the shortest way.
glm::quat quat_from_to(glm::vec3 a, glm::vec3 b);
inline glm::quat quat_yaw(float a) { return glm::angleAxis(a, glm::vec3(0.0f, 1.0f, 0.0f)); }
// Angle difference wrapped to (-pi, pi].
float wrap_angle(float a);
// Critically damped spring pulling (x, v) to zero; exact for any dt.
void decay_spring(glm::vec3& x, glm::vec3& v, float halflife, float dt);
void decay_spring(float& x, float& v, float halflife, float dt);
// Critically damped spring toward a target.
void spring_to(float& x, float& v, float target, float halflife, float dt);
void spring_to(glm::vec3& x, glm::vec3& v, glm::vec3 target, float halflife, float dt);

bool is_finite(const glm::vec3& v);
bool is_finite(const glm::quat& q);
bool is_finite(const glm::mat4& m);

}  // namespace vibe::anim
