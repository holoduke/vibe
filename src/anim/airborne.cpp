#include "anim/airborne.h"

#include <algorithm>
#include <cmath>

namespace qlike::anim {

const char* air_phase_name(AirPhase p) {
    switch (p) {
    case AirPhase::Ground: return "ground";
    case AirPhase::Rise:   return "rise";
    case AirPhase::Apex:   return "apex";
    case AirPhase::Fall:   return "fall";
    case AirPhase::Reach:  return "reach";
    case AirPhase::Land:   return "land";
    }
    return "?";
}

float time_to_contact(float h, float vy, float g) {
    if (h <= 0.0f) return 0.0f;
    if (!(h < 1e8f)) return 1e9f;
    if (g <= 1e-4f) return vy < -1e-4f ? h / -vy : 1e9f;
    // h = -vy t + g t^2 / 2
    const float disc = vy * vy + 2.0f * g * h;
    return (vy + std::sqrt(std::max(disc, 0.0f))) / g;
}

int find_jump(const MotionDatabase& db, glm::vec2 v, bool& mirrored) {
    int best = -1;
    float best_c = 1e30f;
    mirrored = false;
    const auto& jumps = db.jumps();
    for (size_t i = 0; i < jumps.size(); ++i) {
        for (int m = 0; m < 2; ++m) {
            const glm::vec2 jv(m ? -jumps[i].vel_x : jumps[i].vel_x, jumps[i].vel_z);
            const glm::vec2 d = jv - v;
            // prefer the higher of two equally fitting jumps: more pose range for the scrub
            const float c = glm::dot(d, d) - 0.25f * jumps[i].rise + (m ? 1e-4f : 0.0f);
            if (c < best_c) { best_c = c; best = int(i); mirrored = m != 0; }
        }
    }
    return best;
}

AirborneOutput update_airborne(const MotionDatabase& db, AirborneState& st, const AirborneInput& in, const AirborneParams& p,
                               float capture_scale) {
    AirborneOutput o;
    const bool have = !db.jumps().empty();
    const float dt = std::max(in.dt, 0.0f);

    if (in.airborne) {
        st.air_time += dt;
        st.last_vertical = in.vertical_speed;
        st.time_to_contact = time_to_contact(in.ground_distance, in.vertical_speed, in.gravity);
        if (st.phase == AirPhase::Ground || st.phase == AirPhase::Land) {
            if (st.phase == AirPhase::Land) {
                // jumped again out of a landing: no waiting
                st.air_time = std::max(st.air_time, p.min_air_time);
            }
            if (st.air_time >= p.min_air_time && have) {
                st.jump = find_jump(db, in.velocity * capture_scale, st.mirrored);
                const MotionJump& j = db.jumps()[size_t(st.jump)];
                st.frame = float(in.vertical_speed > 1.0f ? j.takeoff : j.apex);
                st.peak_up = std::max(in.vertical_speed, 0.0f);
                st.phase = in.vertical_speed > 0.0f ? AirPhase::Rise : AirPhase::Fall;
                ++st.flights;
                o.started = true;
            }
        }
        if (st.phase != AirPhase::Ground && st.phase != AirPhase::Land) {
            const MotionJump& j = db.jumps()[size_t(st.jump)];
            // A second push-off in the air (double jump) raises the reference at once.
            const bool boosted = in.vertical_speed > st.peak_up + 0.5f;
            st.peak_up = std::max(st.peak_up, in.vertical_speed);
            const float last = float(j.land) - 1.0f;
            float target;
            if (in.vertical_speed > 0.0f) {
                const float u = 1.0f - in.vertical_speed / std::max(st.peak_up, 1.0f);       // 0 at push-off, 1 at the top
                target = glm::mix(float(j.takeoff), float(j.apex), std::clamp(u, 0.0f, 1.0f));
                st.phase = u > 0.8f ? AirPhase::Apex : AirPhase::Rise;
            } else {
                const float reach = std::clamp(1.0f - st.time_to_contact / std::max(p.reach_time, 1e-3f), 0.0f, 1.0f);
                const float depth = std::clamp(-in.vertical_speed / std::max(p.fall_speed_ref, 1e-3f), 0.0f, 1.0f);
                const float u = std::max(reach, 0.35f * depth);
                target = glm::mix(float(j.apex), last, u);
                st.phase = reach > 0.0f ? AirPhase::Reach : (depth > 0.15f ? AirPhase::Fall : AirPhase::Apex);
            }
            const float max_step = p.max_scrub_rate * db.sample_rate() * dt;
            if (boosted && !o.started) { st.frame = target; o.started = true; }      // (a jump in the pose: inertialise)
            st.frame += std::clamp(target - st.frame, -max_step, max_step);
            st.frame = std::clamp(st.frame, float(j.takeoff), last);
        }
    } else {
        if (st.phase != AirPhase::Ground && st.phase != AirPhase::Land) {
            // touch-down
            st.impact = std::max(-st.last_vertical, 0.0f);
            o.landed = true;
            o.impact = st.impact;
            const bool show = st.impact >= p.land_min_impact && glm::length(in.velocity) <= p.land_max_move && !in.wants_move;
            if (show) {
                st.phase = AirPhase::Land;
                st.land_time = 0.0f;
                st.frame = float(db.jumps()[size_t(st.jump)].land);
                o.started = true;                      // (a jump in playback position: inertialise)
            } else {
                st.phase = AirPhase::Ground;
                o.finished = true;
            }
        } else if (st.phase == AirPhase::Land) {
            st.land_time += dt;
            st.frame += dt * db.sample_rate() * p.land_rate;
            const MotionRange& r = db.range_of(db.jumps()[size_t(st.jump)].land);
            const float end = float(r.first + r.count) - 2.0f;
            if (st.land_time >= p.land_time || in.wants_move || glm::length(in.velocity) > p.land_max_move || st.frame >= end) {
                st.frame = std::min(st.frame, end);
                st.phase = AirPhase::Ground;
                o.finished = true;
            }
        }
        st.air_time = 0.0f;
        st.time_to_contact = 0.0f;
    }

    o.active = st.phase != AirPhase::Ground || o.finished;
    if (st.jump >= 0 && (o.active)) {
        const MotionJump& j = db.jumps()[size_t(st.jump)];
        o.frame = int(std::floor(st.frame));
        o.frac = st.frame - float(o.frame);
        o.mirrored = st.mirrored;
        o.stand_hips = db.info(j.apex).hips_y - j.rise;
    }
    if (o.finished) o.active = false;
    return o;
}

}  // namespace qlike::anim
