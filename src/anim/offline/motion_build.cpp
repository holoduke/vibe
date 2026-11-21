#include "anim/offline/motion_build.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace qlike::anim::offline {

bool build_motion_features(const SkeletonDef& skel, const std::vector<RawClip>& clips, const MotionBuildParams& p,
                           MotionDatabaseBuilder& out, std::string* why) {
    out = MotionDatabaseBuilder{};
    auto fail = [&](const std::string& m) { if (why) *why = m; return false; };
    const int hips = skel.find("Hips"), la = skel.find("LeftFoot"), ra = skel.find("RightFoot");
    if (hips < 0 || la < 0 || ra < 0) return fail("rig has no hips / feet");
    if (clips.empty()) return fail("no clips");
    out.rig_id = skel.rig_id();
    out.rate = clips[0].info.sample_rate;
    float max_h = 0.0f;
    for (int i = 0; i < MotionSchema::kHorizons; ++i) { out.horizons[i] = p.horizons[i]; max_h = std::max(max_h, p.horizons[i]); }
    for (int g = 0; g < kFeatureGroups; ++g) out.group_weight[g] = p.group_weight[g];
    const int F = MotionSchema::kFeat;

    std::vector<glm::quat> rot;
    std::vector<glm::vec3> pos;
    for (size_t ci = 0; ci < clips.size(); ++ci) {
        const RawClip& c = clips[ci];
        if (c.joints != skel.size()) return fail("clip does not fit the rig: " + c.info.name);
        if (std::abs(c.info.sample_rate - out.rate) > 1e-3f) return fail("clips have different sample rates");
        if (c.contact.size() != size_t(c.frames) * 2) return fail("clip has no contact labels: " + c.info.name);
        const int n = c.frames;
        const size_t N = static_cast<size_t>(n);
        std::vector<glm::vec3> rp(N), fw(N), pl(N), pr(N), ph(N);
        std::vector<glm::quat> rr(N);
        std::vector<float> yaw(N);
        for (int f = 0; f < n; ++f) {
            clip_fk(skel, c, f, rot, pos);
            const size_t i = static_cast<size_t>(f);
            rp[i] = pos[0];
            rr[i] = rot[0];
            fw[i] = rot[0] * glm::vec3(0.0f, 0.0f, -1.0f);
            pl[i] = pos[size_t(la)];
            pr[i] = pos[size_t(ra)];
            ph[i] = pos[size_t(hips)];
            const float y = std::atan2(fw[i].x, -fw[i].z);
            yaw[i] = f == 0 ? y : yaw[i - 1] + wrap_angle(y - yaw[i - 1]);       // unwrapped
        }
        auto ddt = [&](const std::vector<glm::vec3>& x, int f) {
            const int a = std::max(f - 1, 0), b = std::min(f + 1, n - 1);
            return (x[size_t(b)] - x[size_t(a)]) * (out.rate / float(b - a));
        };
        const uint32_t first = uint32_t(out.info.size());
        for (int f = 0; f < n; ++f) {
            const size_t i = static_cast<size_t>(f);
            const glm::quat inv = glm::inverse(rr[i]);
            float feat[MotionSchema::kFeat];
            auto put3 = [&](int at, glm::vec3 v) { feat[at] = v.x; feat[at + 1] = v.y; feat[at + 2] = v.z; };
            put3(0, inv * (pl[i] - rp[i]));
            put3(3, inv * (pr[i] - rp[i]));
            put3(6, inv * ddt(pl, f));
            put3(9, inv * ddt(pr, f));
            put3(12, inv * ddt(ph, f));
            for (int h = 0; h < MotionSchema::kHorizons; ++h) {
                const int ix = std::min(f + int(std::lround(p.horizons[h] * out.rate)), n - 1);
                const glm::vec3 tp = inv * (rp[size_t(ix)] - rp[i]), td = inv * fw[size_t(ix)];
                feat[15 + 2 * h] = tp.x; feat[16 + 2 * h] = tp.z;
                feat[21 + 2 * h] = td.x; feat[22 + 2 * h] = td.z;
            }
            for (int k = 0; k < F; ++k)
                if (!std::isfinite(feat[k])) return fail("a feature is not finite in " + c.info.name);
            out.raw.insert(out.raw.end(), feat, feat + F);
            MotionFrameInfo fi{};
            const glm::vec3 rv = inv * ddt(rp, f);
            fi.root_vel_x = rv.x;
            fi.root_vel_z = rv.z;
            const int a = std::max(f - 1, 0), b = std::min(f + 1, n - 1);
            fi.yaw_rate = (yaw[size_t(b)] - yaw[size_t(a)]) * (out.rate / float(b - a));
            fi.hips_y = ph[i].y;
            fi.contact = uint8_t((c.contact[i * 2] ? 1 : 0) | (c.contact[i * 2 + 1] ? 2 : 0));
            out.info.push_back(fi);
        }
        MotionRange rg{};
        rg.clip = int32_t(ci);
        rg.first = first;
        rg.count = uint32_t(n);
        rg.tags = c.info.tags;
        rg.head = uint32_t(std::max(p.head_frames, 0));
        rg.tail = uint32_t(std::lround(max_h * out.rate)) + 1u;
        out.ranges.push_back(rg);

        // Flights: both feet off the ground for a while, hips clearly above where they stand.
        if (c.info.tags & kTagJump) {
            std::vector<float> stand;
            for (int f = 0; f < n; ++f)
                if (out.info[first + uint32_t(f)].contact != 0) stand.push_back(out.info[first + uint32_t(f)].hips_y);
            if (stand.empty()) continue;
            std::nth_element(stand.begin(), stand.begin() + ptrdiff_t(stand.size() / 2), stand.end());
            const float ref = stand[stand.size() / 2];
            const int after = int(std::lround(p.landing_frames_s * out.rate));
            int f = 0;
            while (f < n) {
                if (out.info[first + uint32_t(f)].contact != 0) { ++f; continue; }
                int e = f, apex = f;
                float top = -1e9f;
                while (e < n && out.info[first + uint32_t(e)].contact == 0) {
                    if (out.info[first + uint32_t(e)].hips_y > top) { top = out.info[first + uint32_t(e)].hips_y; apex = e; }
                    ++e;
                }
                if (e - f >= p.min_flight_frames && top > ref + p.min_jump_rise && f >= 3 && e + after < n) {
                    MotionJump j{};
                    j.takeoff = first + uint32_t(f);
                    j.apex = first + uint32_t(apex);
                    j.land = first + uint32_t(e);
                    j.rise = top - ref;
                    j.vel_x = out.info[j.takeoff].root_vel_x;
                    j.vel_z = out.info[j.takeoff].root_vel_z;
                    out.jumps.push_back(j);
                }
                f = e;
            }
        }
    }
    return true;
}

bool build_motion_database(const SkeletonDef& skel, const std::vector<RawClip>& clips, const MotionBuildParams& p,
                           std::vector<uint8_t>& blob, const std::function<void(const std::string&)>& log, std::string* why) {
    MotionDatabaseBuilder b;
    if (!build_motion_features(skel, clips, p, b, why)) return false;
    blob = b.serialize();
    MotionDatabase check;
    if (!check.load(blob.data(), blob.size(), skel.rig_id(), why)) return false;
    if (log) {
        size_t searchable = 0;
        for (const MotionRange& r : b.ranges)
            if (r.count > r.head + r.tail) searchable += r.count - r.head - r.tail;
        char line[200];
        std::snprintf(line, sizeof(line), "motion database: %zu frames (%zu can be jumped to, each also mirrored), %zu jumps, %.2f MB",
                      b.info.size(), searchable, b.jumps.size(), double(blob.size()) / 1048576.0);
        log(line);
    }
    return true;
}

}  // namespace qlike::anim::offline
