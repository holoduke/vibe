#include "engine/motion_match.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace qlike {

namespace {

// Rotation as "scaled axis-angle" (axis * angle) and back.
glm::vec3 to_sav(glm::quat q) {
    if (q.w < 0.0f) q = -q;
    const glm::vec3 v(q.x, q.y, q.z);
    const float s = glm::length(v);
    if (s < 1e-6f) return v * 2.0f;
    return v * (2.0f * std::atan2(s, q.w) / s);
}
glm::quat from_sav(glm::vec3 v) {
    const float a = glm::length(v);
    if (a < 1e-6f) return glm::normalize(glm::quat(1.0f, v.x * 0.5f, v.y * 0.5f, v.z * 0.5f));
    const float s = std::sin(a * 0.5f) / a;
    return glm::quat(std::cos(a * 0.5f), v.x * s, v.y * s, v.z * s);
}
// A critically damped spring pulling x to zero.
void decay(glm::vec3& x, glm::vec3& v, float halflife, float dt) {
    const float y = 4.0f * 0.69314718f / std::max(halflife, 1e-4f) * 0.5f;
    const glm::vec3 j1 = v + x * y;
    const float e = std::exp(-y * dt);
    x = e * (x + j1 * dt);
    v = e * (v - j1 * y * dt);
}

}  // namespace

bool MotionDB::load(const char* path) {
    *this = MotionDB{};
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    char magic[4]{};
    uint32_t nf = 0, nb = 0, nfeat = 0, nr = 0;
    float fps_in = 30.0f;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&nf), 4);
    f.read(reinterpret_cast<char*>(&nb), 4);
    f.read(reinterpret_cast<char*>(&nfeat), 4);
    f.read(reinterpret_cast<char*>(&fps_in), 4);
    f.read(reinterpret_cast<char*>(&nr), 4);
    if (std::memcmp(magic, "QMM1", 4) != 0 || nfeat != uint32_t(kFeat) || nf == 0 || nf > 4000000 || nb == 0 || nb > 128 || nr == 0 || nr > 4096)
        return false;
    bone_names.resize(nb);
    for (auto& n : bone_names) {
        char name[32]{};
        f.read(name, 32);
        name[31] = 0;
        n = name;
    }
    std::vector<Range> src(nr);
    f.read(reinterpret_cast<char*>(src.data()), std::streamsize(nr) * sizeof(Range));
    feat_offset.resize(kFeat);
    feat_scale.resize(kFeat);
    f.read(reinterpret_cast<char*>(feat_offset.data()), kFeat * 4);
    f.read(reinterpret_cast<char*>(feat_scale.data()), kFeat * 4);
    // Twice the frames: the second half is the first, mirrored left to right.
    feat.resize(size_t(nf) * 2 * kFeat);
    hdr.resize(size_t(nf) * 2 * 8);
    rot.resize(size_t(nf) * 2 * nb);
    f.read(reinterpret_cast<char*>(feat.data()), std::streamsize(nf) * kFeat * 4);
    f.read(reinterpret_cast<char*>(hdr.data()), std::streamsize(nf) * 8 * 4);
    {
        std::vector<int16_t> q(size_t(nf) * nb * 4);
        f.read(reinterpret_cast<char*>(q.data()), std::streamsize(q.size()) * 2);
        if (!f.good()) { *this = MotionDB{}; return false; }
        for (size_t i = 0; i < size_t(nf) * nb; ++i)
            rot[i] = glm::normalize(glm::quat(q[i * 4 + 3] / 32767.0f, q[i * 4] / 32767.0f, q[i * 4 + 1] / 32767.0f, q[i * 4 + 2] / 32767.0f));
    }
    // The mirror: x -> -x. A rotation (x,y,z,w) becomes (x,-y,-z,w) and belongs to the bone of the other side.
    std::vector<uint32_t> other(nb);
    for (uint32_t b = 0; b < nb; ++b) {
        std::string n = bone_names[b];
        if (n.rfind("Left", 0) == 0) n = "Right" + n.substr(4);
        else if (n.rfind("Right", 0) == 0) n = "Left" + n.substr(5);
        other[b] = b;
        for (uint32_t k = 0; k < nb; ++k) if (bone_names[k] == n) other[b] = k;
    }
    for (uint32_t i = 0; i < nf; ++i) {
        const size_t m = size_t(nf) + i;
        for (uint32_t b = 0; b < nb; ++b) {
            const glm::quat q = rot[size_t(i) * nb + other[b]];
            rot[m * nb + b] = glm::quat(q.w, q.x, -q.y, -q.z);
        }
        const float* h = &hdr[size_t(i) * 8];
        float* hm = &hdr[m * 8];
        hm[0] = -h[0]; hm[1] = h[1]; hm[2] = -h[2]; hm[3] = -h[3]; hm[4] = h[4]; hm[5] = h[5]; hm[6] = h[7]; hm[7] = h[6];
        // features: feet (pos 0-2 | 3-5, vel 6-8 | 9-11) swap sides; every x changes sign
        float raw[kFeat], mr[kFeat];
        for (int k = 0; k < kFeat; ++k) raw[k] = feat[size_t(i) * kFeat + k] * feat_scale[k] + feat_offset[k];
        for (int k = 0; k < 3; ++k) { mr[k] = raw[3 + k]; mr[3 + k] = raw[k]; mr[6 + k] = raw[9 + k]; mr[9 + k] = raw[6 + k]; mr[12 + k] = raw[12 + k]; }
        for (int k = 15; k < kFeat; ++k) mr[k] = raw[k];
        for (int k : { 0, 3, 6, 9, 12, 15, 17, 19, 21, 23, 25 }) mr[k] = -mr[k];
        for (int k = 0; k < kFeat; ++k) feat[m * kFeat + k] = (mr[k] - feat_offset[k]) / feat_scale[k];
    }
    ranges = src;
    for (const Range& r : src) ranges.push_back({ r.first + nf, r.end + nf, r.tag });
    frames = nf * 2;
    bones = nb;
    fps = fps_in;
    range_of.resize(frames);
    for (uint32_t r = 0; r < ranges.size(); ++r)
        for (uint32_t i = ranges[r].first; i < ranges[r].end; ++i) range_of[i] = r;
    return true;
}

void motion_update(const MotionDB& db, MotionState& st, const MotionQuery& q, float dt, MotionPose& out) {
    const uint32_t nb = db.bones;
    out.rot.resize(nb);
    if (st.off_x.size() != nb) { st.off_x.assign(nb, glm::vec3(0.0f)); st.off_v.assign(nb, glm::vec3(0.0f)); }
    const int kTail = 31;                      // frames at a clip's end without a future to match
    const float fps = db.fps;

    // ---- advance ----
    bool at_end = false;
    if (st.frame >= 0) {
        st.frac += dt * fps * q.rate;
        const MotionDB::Range& r = db.ranges[db.range_of[st.frame]];
        while (st.frac >= 1.0f) { st.frac -= 1.0f; ++st.frame; }
        if (st.frame >= int(r.end) - 2) { st.frame = int(r.end) - 2; st.frac = 0.0f; at_end = true; }
    }
    st.since_search += dt;

    // ---- search ----
    const bool changed = glm::length(q.vel - st.last_vel) > 0.35f || q.tag != st.last_tag;
    if (st.frame < 0 || at_end || st.since_search > 0.20f || (changed && st.since_search > 0.05f)) {
        st.since_search = 0.0f;
        st.last_vel = q.vel;
        st.last_tag = q.tag;
        // What is asked: the pose we are in, and the path of the next second - the velocity
        // eases from the animation's to the wanted one while the body turns.
        float qf[MotionDB::kFeat];
        glm::vec2 v0(0.0f);
        if (st.frame >= 0) {
            std::memcpy(qf, &db.feat[size_t(st.frame) * MotionDB::kFeat], sizeof(qf));
            v0 = glm::vec2(db.hdr[size_t(st.frame) * 8], db.hdr[size_t(st.frame) * 8 + 1]);
        } else {
            for (int k = 0; k < MotionDB::kFeat; ++k) qf[k] = 0.0f;
        }
        {
            const float tau_v = 0.12f, tau_w = 0.25f;
            const float w = glm::clamp(q.turn_rate, -4.0f, 4.0f);
            glm::vec2 p(0.0f);
            float t = 0.0f;
            const float h = 1.0f / 30.0f;
            for (int k = 1; k <= 30; ++k) {
                t += h;
                const float th = w * tau_w * (1.0f - std::exp(-t / tau_w));        // how far the body has turned
                const glm::vec2 vb = q.vel + (v0 - q.vel) * std::exp(-t / tau_v);  // velocity in the body's frame then
                const float c = std::cos(th), s = std::sin(th);
                p += glm::vec2(vb.x * c - vb.y * s, vb.x * s + vb.y * c) * h;       // (x right, z back; turning right by th)
                if (k % 10 == 0) {
                    const int i = k / 10 - 1;
                    const float raw[4] = { p.x, p.y, s, -c };
                    const int at[4] = { 15 + 2 * i, 16 + 2 * i, 21 + 2 * i, 22 + 2 * i };
                    for (int j = 0; j < 4; ++j) qf[at[j]] = (raw[j] - db.feat_offset[at[j]]) / db.feat_scale[at[j]];
                }
            }
        }
        float best = 1e30f;
        int   best_i = -1;
        float cur_cost = 1e30f;
        for (const MotionDB::Range& r : db.ranges) {
            if (r.tag != q.tag || r.end < r.first + uint32_t(kTail) + 2) continue;
            const float* fp = &db.feat[size_t(r.first) * MotionDB::kFeat];
            for (uint32_t i = r.first; i < r.end - kTail; ++i, fp += MotionDB::kFeat) {
                float c = 0.0f;
                for (int k = 0; k < MotionDB::kFeat; ++k) { const float d = fp[k] - qf[k]; c += d * d; }
                if (c < best) { best = c; best_i = int(i); }
                if (int(i) == st.frame) cur_cost = c;
            }
        }
        // Stay where we are unless the jump is clearly better, and never for a hop of a few frames.
        const bool hop = st.frame >= 0 && best_i >= 0 && db.range_of[best_i] == db.range_of[st.frame] && std::abs(best_i - st.frame) < 8;
        if (best_i >= 0 && (st.frame < 0 || at_end || (!hop && best < cur_cost * 0.90f - 0.02f))) {
            if (st.frame >= 0) {
                // Inertialise: carry the difference (pose and rate of change) over as an offset.
                const int a = st.frame, b = best_i;
                for (uint32_t k = 0; k < nb; ++k) {
                    const glm::quat qa0 = db.rot[size_t(a) * nb + k], qa1 = db.rot[size_t(a + 1) * nb + k];
                    const glm::quat qb0 = db.rot[size_t(b) * nb + k], qb1 = db.rot[size_t(b + 1) * nb + k];
                    glm::quat qa = glm::normalize(glm::mix(qa0, glm::dot(qa0, qa1) < 0.0f ? -qa1 : qa1, st.frac));
                    const glm::quat cur = from_sav(st.off_x[k]) * qa;
                    st.off_x[k] = to_sav(cur * glm::inverse(qb0));
                    st.off_v[k] += (to_sav(qa1 * glm::inverse(qa0)) - to_sav(qb1 * glm::inverse(qb0))) * fps * q.rate;
                }
                const float* ha = &db.hdr[size_t(a) * 8];
                const float* ha1 = &db.hdr[size_t(a + 1) * 8];
                const float* hb = &db.hdr[size_t(b) * 8];
                const float* hb1 = &db.hdr[size_t(b + 1) * 8];
                const glm::vec3 pa = glm::mix(glm::vec3(ha[3], ha[4], ha[5]), glm::vec3(ha1[3], ha1[4], ha1[5]), st.frac);
                st.hip_x += pa - glm::vec3(hb[3], hb[4], hb[5]);
                st.hip_v += ((glm::vec3(ha1[3], ha1[4], ha1[5]) - glm::vec3(ha[3], ha[4], ha[5])) -
                             (glm::vec3(hb1[3], hb1[4], hb1[5]) - glm::vec3(hb[3], hb[4], hb[5]))) * fps * q.rate;
            }
            st.frame = best_i;
            st.frac = 0.0f;
        }
    }
    if (st.frame < 0) {
        for (auto& r : out.rot) r = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // ---- the pose: the database's, plus what is left of the offsets ----
    const float kHalf = 0.10f;
    const int a = st.frame;
    for (uint32_t k = 0; k < nb; ++k) {
        decay(st.off_x[k], st.off_v[k], kHalf, dt);
        const glm::quat q0 = db.rot[size_t(a) * nb + k], q1 = db.rot[size_t(a + 1) * nb + k];
        const glm::quat qd = glm::normalize(glm::mix(q0, glm::dot(q0, q1) < 0.0f ? -q1 : q1, st.frac));
        out.rot[k] = from_sav(st.off_x[k]) * qd;
    }
    decay(st.hip_x, st.hip_v, kHalf, dt);
    const float* h0 = &db.hdr[size_t(a) * 8];
    const float* h1 = &db.hdr[size_t(a + 1) * 8];
    out.hip = glm::mix(glm::vec3(h0[3], h0[4], h0[5]), glm::vec3(h1[3], h1[4], h1[5]), st.frac) + st.hip_x;
    out.root_vel = glm::mix(glm::vec2(h0[0], h0[1]), glm::vec2(h1[0], h1[1]), st.frac);
    out.contact[0] = h0[6];
    out.contact[1] = h0[7];
}

}  // namespace qlike
