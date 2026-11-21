#include "anim/motion_matcher.h"

#include <algorithm>
#include <cmath>

namespace qlike::anim {

float capture_speed(float s, const MatcherParams& p) {
    if (s <= p.walk_speed) return s;
    return p.walk_speed + (p.run_speed - p.walk_speed) * (1.0f - std::exp(-(s - p.walk_speed) / std::max(p.speed_knee, 1e-3f)));
}

MatcherOutput matcher_output(const MotionDatabase& db, const MatcherState& st) {
    MatcherOutput o;
    if (st.frame < 0 || !db.ok()) return o;
    const uint32_t f = uint32_t(st.frame);
    const MotionRange& r = db.range_of(f);
    const uint32_t f1 = std::min(f + 1, r.first + r.count - 1);
    const MotionFrameInfo& a = db.info(f);
    const MotionFrameInfo& b = db.info(f1);
    const float sx = st.mirrored ? -1.0f : 1.0f;
    o.clip = r.clip;
    o.time = db.time_in_clip(f, st.frac);
    o.mirrored = st.mirrored;
    o.root_vel = glm::vec2(sx * glm::mix(a.root_vel_x, b.root_vel_x, st.frac), glm::mix(a.root_vel_z, b.root_vel_z, st.frac));
    o.yaw_rate = sx * glm::mix(a.yaw_rate, b.yaw_rate, st.frac);
    o.hips_y = glm::mix(a.hips_y, b.hips_y, st.frac);
    const bool cl = (a.contact & 1) != 0, cr = (a.contact & 2) != 0;
    o.contact[0] = st.mirrored ? cr : cl;
    o.contact[1] = st.mirrored ? cl : cr;
    return o;
}

MatcherOutput matcher_goto(const MotionDatabase& db, MatcherState& st, int frame, bool mirrored) {
    st.frame = std::clamp(frame, 0, int(db.frames()) - 1);
    st.frac = 0.0f;
    st.mirrored = mirrored;
    st.since_switch = 0.0f;
    ++st.switches;
    MatcherOutput o = matcher_output(db, st);
    o.switched = true;
    return o;
}

MatcherOutput update_matcher(const MotionDatabase& db, MatcherState& st, const MatcherInput& in, const MatcherParams& p, float dt) {
    if (!db.ok()) return MatcherOutput{};
    // ---- advance ----
    bool at_end = false;
    if (st.frame >= 0) {
        const MotionRange& r = db.range_of(uint32_t(st.frame));
        st.frac += dt * db.sample_rate() * in.rate;
        const int whole = int(std::floor(st.frac));
        st.frame += whole;
        st.frac -= float(whole);
        const int last = int(r.first + r.count) - 2;                // the frame before the last: frame + 1 stays valid
        if (st.frame >= last) { st.frame = std::max(last, int(r.first)); st.frac = 0.0f; at_end = true; }
        // a range that no longer fits the request (stance changed) cannot be kept
        if ((r.tags & in.require_tags) != in.require_tags || (r.tags & in.exclude_tags) != 0) at_end = true;
    }
    st.since_search += dt;
    st.since_switch += dt;

    // ---- the request at capture scale ----
    const float speed = glm::length(in.velocity);
    float end_speed = glm::length(in.traj.vel_end);
    const float ref = std::max(speed, end_speed);
    const float k = ref > 1e-3f ? capture_speed(ref, p) / ref : 1.0f;
    const glm::vec2 request = in.traj.vel_end * k;

    const bool changed = glm::length(request - st.last_request) > p.change_speed || in.require_tags != st.last_tags;
    const bool due = st.frame < 0 || at_end || in.force_search || st.since_search >= p.search_interval ||
                     (changed && st.since_search >= p.change_interval);
    bool switched = false, searched = false;
    if (due) {
        searched = true;
        st.since_search = 0.0f;
        st.last_request = request;
        st.last_tags = in.require_tags;
        ++st.searches;
        MotionQuery q;
        q.require_tags = in.require_tags;
        q.exclude_tags = in.exclude_tags;
        if (st.frame >= 0) db.raw_features(uint32_t(st.frame), st.mirrored, q.raw);
        for (int h = 0; h < MotionSchema::kHorizons; ++h) {
            q.raw[15 + 2 * h] = in.traj.pos[h].x * k;
            q.raw[16 + 2 * h] = in.traj.pos[h].y * k;
            q.raw[21 + 2 * h] = in.traj.dir[h].x;
            q.raw[22 + 2 * h] = in.traj.dir[h].y;
        }
        float pose_w = st.frame < 0 ? 0.0f : 1.0f;
        if (speed > p.fast_speed) pose_w *= p.fast_pose_weight;
        q.group_weight[kFootPos] = q.group_weight[kFootVel] = q.group_weight[kHipVel] = pose_w;
        MotionMatch best = db.search(q);
        st.last_fallback = false;
        if (best.frame >= 0 && best.group_cost[kTrajPos] + best.group_cost[kTrajDir] > p.fallback_traj_cost && pose_w > 0.0f) {
            // Nothing captured continues this pose toward that path: choose by the path alone.
            q.group_weight[kFootPos] = q.group_weight[kFootVel] = q.group_weight[kHipVel] = 0.0f;
            best = db.search(q);
            st.last_fallback = true;
        }
        st.last_match = best;
        float cur_cost = 1e30f;
        const bool keepable = st.frame >= 0 && !at_end;
        if (keepable) cur_cost = db.cost(q, uint32_t(st.frame), st.mirrored).cost;
        st.last_current_cost = keepable ? cur_cost : 0.0f;
        if (best.frame >= 0) {
            const bool same_place = st.frame >= 0 && best.mirrored == st.mirrored &&
                                    &db.range_of(uint32_t(best.frame)) == &db.range_of(uint32_t(st.frame)) &&
                                    std::abs(best.frame - st.frame) < p.min_hop_frames;
            const bool better = best.cost < cur_cost * p.keep_ratio - p.keep_margin;
            const bool allowed = st.since_switch >= p.min_switch_interval;
            if (!keepable || (better && !same_place && allowed)) {
                if (!(same_place && !keepable && best.frame == st.frame)) {
                    st.frame = best.frame;
                    st.frac = 0.0f;
                    st.mirrored = best.mirrored;
                    st.since_switch = 0.0f;
                    ++st.switches;
                    switched = true;
                }
            }
        }
    }
    MatcherOutput o = matcher_output(db, st);
    o.switched = switched;
    o.searched = searched;
    o.speed_scale = k;
    return o;
}

}  // namespace qlike::anim
