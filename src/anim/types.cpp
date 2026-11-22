#include "anim/types.h"

#include <algorithm>
#include <cmath>

namespace vibe::anim {

glm::vec3 quat_to_scaled_axis(glm::quat q) {
    if (q.w < 0.0f) q = -q;
    const glm::vec3 v(q.x, q.y, q.z);
    const float s = glm::length(v);
    if (s < 1e-6f) return v * 2.0f;
    return v * (2.0f * std::atan2(s, q.w) / s);
}

glm::quat quat_from_scaled_axis(glm::vec3 v) {
    const float a = glm::length(v);
    if (a < 1e-6f) return glm::normalize(glm::quat(1.0f, v.x * 0.5f, v.y * 0.5f, v.z * 0.5f));
    const float s = std::sin(a * 0.5f) / a;
    return glm::quat(std::cos(a * 0.5f), v.x * s, v.y * s, v.z * s);
}

glm::quat quat_from_to(glm::vec3 a, glm::vec3 b) {
    const float d = glm::dot(a, b);
    if (d > 0.999999f) return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    if (d < -0.999999f) {
        glm::vec3 axis = glm::cross(a, std::abs(a.x) < 0.8f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0));
        axis = glm::normalize(axis);
        return glm::quat(0.0f, axis.x, axis.y, axis.z);
    }
    const glm::vec3 c = glm::cross(a, b);
    return glm::normalize(glm::quat(1.0f + d, c.x, c.y, c.z));
}

float wrap_angle(float a) {
    const float two_pi = 6.28318530718f;
    a -= two_pi * std::round(a / two_pi);
    return a;
}

void decay_spring(glm::vec3& x, glm::vec3& v, float halflife, float dt) {
    const float y = 4.0f * 0.69314718f / std::max(halflife, 1e-4f) * 0.5f;
    const glm::vec3 j1 = v + x * y;
    const float e = std::exp(-y * dt);
    x = e * (x + j1 * dt);
    v = e * (v - j1 * y * dt);
}

void decay_spring(float& x, float& v, float halflife, float dt) {
    const float y = 4.0f * 0.69314718f / std::max(halflife, 1e-4f) * 0.5f;
    const float j1 = v + x * y;
    const float e = std::exp(-y * dt);
    x = e * (x + j1 * dt);
    v = e * (v - j1 * y * dt);
}

void spring_to(float& x, float& v, float target, float halflife, float dt) {
    float d = x - target;
    decay_spring(d, v, halflife, dt);
    x = target + d;
}

void spring_to(glm::vec3& x, glm::vec3& v, glm::vec3 target, float halflife, float dt) {
    glm::vec3 d = x - target;
    decay_spring(d, v, halflife, dt);
    x = target + d;
}

bool is_finite(const glm::vec3& v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
bool is_finite(const glm::quat& q) { return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w); }
bool is_finite(const glm::mat4& m) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            if (!std::isfinite(m[c][r])) return false;
    return true;
}

}  // namespace vibe::anim
