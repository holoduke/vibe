#include "anim/offline/raw_clip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vibe::anim::offline {

void clip_fk(const SkeletonDef& s, const RawClip& c, int frame, std::vector<glm::quat>& rot, std::vector<glm::vec3>& pos) {
    const int n = s.size();
    rot.resize(size_t(n));
    pos.resize(size_t(n));
    for (int j = 0; j < n; ++j) {
        const Transform& t = c.at(frame, j);
        const int p = s.parents[size_t(j)];
        if (p < 0) { rot[size_t(j)] = t.r; pos[size_t(j)] = t.t; }
        else {
            rot[size_t(j)] = rot[size_t(p)] * t.r;
            pos[size_t(j)] = pos[size_t(p)] + rot[size_t(p)] * t.t;
        }
    }
}

namespace {
std::vector<float> gauss(const std::vector<float>& x, float sigma) {
    const int r = std::max(1, int(sigma * 3.0f));
    std::vector<float> k(size_t(2 * r + 1));
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) { k[size_t(i + r)] = std::exp(-0.5f * float(i * i) / (sigma * sigma)); sum += k[size_t(i + r)]; }
    const int n = int(x.size());
    std::vector<float> out(x.size(), 0.0f);
    for (int i = 0; i < n; ++i)
        for (int j = -r; j <= r; ++j) out[size_t(i)] += k[size_t(j + r)] / sum * x[size_t(std::clamp(i + j, 0, n - 1))];
    return out;
}
}  // namespace

void label_contacts(const SkeletonDef& s, RawClip& c, const ContactParams& p) {
    const int feet[2] = { s.find("LeftFoot"), s.find("RightFoot") };
    c.contact.assign(size_t(c.frames) * 2, 0);
    if (feet[0] < 0 || feet[1] < 0 || c.frames < 2) return;
    std::vector<glm::vec3> fp[2];
    std::vector<glm::quat> rot;
    std::vector<glm::vec3> pos;
    for (int f = 0; f < c.frames; ++f) {
        clip_fk(s, c, f, rot, pos);
        for (int k = 0; k < 2; ++k) fp[k].push_back(pos[size_t(feet[k])]);
    }
    const float dt = 1.0f / c.info.sample_rate;
    for (int k = 0; k < 2; ++k) {
        const float rest_y = s.rest_pos[size_t(feet[k])].y;
        std::vector<float> raw(size_t(c.frames));
        for (int f = 0; f < c.frames; ++f) {
            const int a = std::max(f - 1, 0), b = std::min(f + 1, c.frames - 1);
            const float speed = glm::length(fp[k][size_t(b)] - fp[k][size_t(a)]) / (float(b - a) * dt);
            raw[size_t(f)] = (speed < p.max_speed && fp[k][size_t(f)].y < rest_y + p.max_height) ? 1.0f : 0.0f;
        }
        const std::vector<float> sm = gauss(raw, p.smooth_sigma);
        for (int f = 0; f < c.frames; ++f) c.contact[size_t(f) * 2 + size_t(k)] = sm[size_t(f)] > 0.5f ? 1 : 0;
    }
}

ClipReport validate_clip(const SkeletonDef& s, const RawClip& c, float max_root_speed) {
    ClipReport r;
    auto err = [&](const std::string& m) { r.ok = false; r.errors += m + "; "; };
    auto warn = [&](const std::string& m) { r.warnings += m + "; "; };
    if (c.joints != s.size()) { err("joint count differs from the rig"); return r; }
    if (c.frames < 2) { err("fewer than two frames"); return r; }
    if (c.local.size() != size_t(c.frames) * size_t(c.joints)) { err("pose array has the wrong size"); return r; }
    if (!(c.info.sample_rate >= 10.0f && c.info.sample_rate <= 240.0f)) err("sample rate outside 10..240 Hz");
    if (std::abs(c.info.duration - float(c.frames - 1) / c.info.sample_rate) > 1e-3f) err("duration does not match frames / rate");
    for (const Transform& t : c.local) {
        if (!is_finite(t.t) || !is_finite(t.r)) { err("a transform is not finite"); return r; }
        if (std::abs(glm::dot(t.r, t.r) - 1.0f) > 1e-3f) { err("a rotation is not a unit quaternion"); return r; }
    }
    const int la = s.find("LeftFoot"), ra = s.find("RightFoot");
    std::vector<glm::quat> rot;
    std::vector<glm::vec3> pos, prev_pos;
    float speed_sum = 0.0f, low = 1e9f;
    for (int f = 0; f < c.frames; ++f) {
        clip_fk(s, c, f, rot, pos);
        for (int j = 1; j < s.size(); ++j) {
            const int p = s.parents[size_t(j)];
            if (p == 0) continue;                              // the hips move freely over the root
            const float len = glm::length(pos[size_t(j)] - pos[size_t(p)]);
            r.max_limb_error = std::max(r.max_limb_error, std::abs(len - glm::length(s.rest_local(j))));
        }
        if (la >= 0 && ra >= 0)
            low = std::min(low, std::min(pos[size_t(la)].y - s.rest_pos[size_t(la)].y, pos[size_t(ra)].y - s.rest_pos[size_t(ra)].y));
        if (f > 0) {
            const float sp = glm::length(glm::vec2(pos[0].x - prev_pos[0].x, pos[0].z - prev_pos[0].z)) * c.info.sample_rate;
            speed_sum += sp;
            r.max_speed = std::max(r.max_speed, sp);
        }
        prev_pos = pos;
    }
    r.mean_speed = speed_sum / float(c.frames - 1);
    r.lowest_ankle = low < 1e8f ? low : 0.0f;
    if (r.max_limb_error > 0.002f) err("bone lengths change (retargeting must keep them)");
    if (r.max_speed > max_root_speed) err("root moves faster than a person can");
    if (std::abs(c.at(0, 0).t.y) > 1e-3f) warn("root is not on the ground plane");
    if (r.lowest_ankle < -0.08f) warn("a foot goes more than 8 cm under the ground");
    if (c.contact.size() == size_t(c.frames) * 2) {
        size_t l = 0, rr = 0;
        for (int f = 0; f < c.frames; ++f) { l += c.contact[size_t(f) * 2]; rr += c.contact[size_t(f) * 2 + 1]; }
        r.contact_left = float(l) / float(c.frames);
        r.contact_right = float(rr) / float(c.frames);
        if ((c.info.tags & kTagJump) == 0 && r.contact_left + r.contact_right < 0.3f) warn("feet are hardly ever on the ground");
    } else {
        warn("no contact labels");
    }
    return r;
}

}  // namespace vibe::anim::offline
