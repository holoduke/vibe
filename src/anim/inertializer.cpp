#include "anim/inertializer.h"

#include <algorithm>
#include <cmath>

namespace vibe::anim {

namespace {
constexpr float kMaxAngVel = 40.0f;     // rad/s: a rate above this is a pop in the source, not motion
constexpr float kMaxLinVel = 12.0f;     // m/s

glm::vec3 clamp_len(glm::vec3 v, float m) {
    const float l = glm::length(v);
    return l > m ? v * (m / l) : v;
}

// The rate of change of a rotation vector phi that makes exp(phi) turn with angular velocity w
// (the inverse of the left Jacobian of SO(3)). For a small phi it is w itself; for the large
// offsets a transition can start with, using w directly would bend the carried-over motion.
glm::vec3 rotvec_rate(glm::vec3 phi, glm::vec3 w) {
    const float th = glm::length(phi);
    const glm::vec3 c1 = glm::cross(phi, w), c2 = glm::cross(phi, c1);
    float k = 1.0f / 12.0f;
    if (th > 1e-3f) {
        const float s = std::sin(th);
        k = std::abs(s) > 1e-5f ? 1.0f / (th * th) - (1.0f + std::cos(th)) / (2.0f * th * s) : 1.0f / (th * th);
    }
    return w - 0.5f * c1 + k * c2;
}
}  // namespace

void Inertializer::reset(int joints) {
    const size_t n = static_cast<size_t>(joints);
    rx_.assign(n, glm::vec3(0.0f)); rv_.assign(n, glm::vec3(0.0f));
    tx_.assign(n, glm::vec3(0.0f)); tv_.assign(n, glm::vec3(0.0f));
    last_.assign(n, Transform{}); prev_.assign(n, Transform{});
    have_ = 0;
    last_dt_ = 0.0f;
    energy_ = 0.0f;
}

void Inertializer::transition(std::span<const Transform> dst, std::span<const Transform> dst_next, float dst_dt) {
    if (have_ == 0) return;                               // nothing was shown yet: start on the new motion as it is
    const size_t n = std::min(rx_.size(), dst.size());
    const bool vel_ok = have_ >= 2 && last_dt_ > 1e-4f;
    const bool dvel_ok = dst_dt > 1e-4f && dst_next.size() >= n;
    for (size_t j = 1; j < n; ++j) {                      // joint 0 is the root
        // what is on screen, and how it was moving
        const glm::quat cur = last_[j].r;
        glm::vec3 cur_w(0.0f), cur_v(0.0f);
        if (vel_ok) {
            cur_w = clamp_len(quat_to_scaled_axis(cur * glm::inverse(prev_[j].r)) / last_dt_, kMaxAngVel);
            cur_v = clamp_len((last_[j].t - prev_[j].t) / last_dt_, kMaxLinVel);
        }
        glm::vec3 dst_w(0.0f), dst_v(0.0f);
        if (dvel_ok) {
            dst_w = clamp_len(quat_to_scaled_axis(dst_next[j].r * glm::inverse(dst[j].r)) / dst_dt, kMaxAngVel);
            dst_v = clamp_len((dst_next[j].t - dst[j].t) / dst_dt, kMaxLinVel);
        }
        rx_[j] = quat_to_scaled_axis(cur * glm::inverse(dst[j].r));
        rv_[j] = rotvec_rate(rx_[j], cur_w - dst_w);
        tx_[j] = last_[j].t - dst[j].t;
        tv_[j] = cur_v - dst_v;
    }
}

void Inertializer::apply(std::span<const Transform> src, float dt, float halflife, std::span<Transform> out) {
    const size_t n = std::min(rx_.size(), src.size());
    float e = 0.0f;
    for (size_t j = 0; j < n; ++j) {
        if (j == 0) { out[0] = src[0]; continue; }
        decay_spring(rx_[j], rv_[j], halflife, dt);
        decay_spring(tx_[j], tv_[j], halflife, dt);
        out[j].r = glm::normalize(quat_from_scaled_axis(rx_[j]) * src[j].r);
        out[j].t = src[j].t + tx_[j];
        e += glm::length(rx_[j]) + glm::length(tx_[j]);
    }
    energy_ = e;
    if (dt > 1e-5f || have_ == 0) {
        prev_.swap(last_);
        std::copy(out.begin(), out.begin() + ptrdiff_t(n), last_.begin());
        last_dt_ = dt;
        have_ = std::min(have_ + 1, 2);
    } else {
        std::copy(out.begin(), out.begin() + ptrdiff_t(n), last_.begin());      // a zero-length step: only the pose is refreshed
    }
}

}  // namespace vibe::anim
