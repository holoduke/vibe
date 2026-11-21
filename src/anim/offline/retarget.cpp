#include "anim/offline/retarget.h"

#include <algorithm>
#include <cmath>

namespace qlike::anim::offline {

namespace {

glm::dmat3 rot_from_to(glm::dvec3 a, glm::dvec3 b) {
    a = glm::normalize(a);
    b = glm::normalize(b);
    const double c = glm::dot(a, b);
    if (c < -0.9999) {
        glm::dvec3 ax = glm::cross(a, glm::dvec3(1, 0, 0));
        if (glm::length(ax) < 1e-3) ax = glm::cross(a, glm::dvec3(0, 1, 0));
        ax = glm::normalize(ax);
        return 2.0 * glm::outerProduct(ax, ax) - glm::dmat3(1.0);
    }
    const glm::dvec3 v = glm::cross(a, b);
    const glm::dmat3 K(0.0, v.z, -v.y, -v.z, 0.0, v.x, v.y, -v.x, 0.0);     // columns; K * x = v cross x
    return glm::dmat3(1.0) + K + K * K * (1.0 / (1.0 + c));
}

// `extend`: beyond the ends the signal is continued on its own line (x[-i] = 2 x[0] - x[i]) instead
// of held, so a steady walk keeps its speed up to the first and the last frame.
template <typename T>
std::vector<T> gauss(const std::vector<T>& x, double sigma, bool extend = false) {
    const int r = std::max(1, int(sigma * 3.0));
    std::vector<double> k(size_t(2 * r + 1));
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) { k[size_t(i + r)] = std::exp(-0.5 * double(i * i) / (sigma * sigma)); sum += k[size_t(i + r)]; }
    const int n = int(x.size());
    std::vector<T> out(x.size(), T(0.0));
    for (int i = 0; i < n; ++i)
        for (int j = -r; j <= r; ++j) {
            const int q = i + j;
            T v = x[size_t(std::clamp(q, 0, n - 1))];
            if (extend && q < 0) v = x[0] * 2.0 - x[size_t(std::min(-q, n - 1))];
            if (extend && q > n - 1) v = x[size_t(n - 1)] * 2.0 - x[size_t(std::max(2 * (n - 1) - q, 0))];
            out[size_t(i)] += v * (k[size_t(j + r)] / sum);
        }
    return out;
}

}  // namespace

RetargetParams style100_params() {
    RetargetParams p;
    p.axes = glm::dmat3(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);       // +z forward, +x left -> -z forward, +x right
    p.map = {
        { "Hips", "Hips", "", "" }, { "Spine", "Chest", "", "" }, { "Spine1", "Chest2", "", "" },
        { "Spine2", "Chest4", "", "" }, { "Neck", "Neck", "", "" }, { "Head", "Head", "", "" },
    };
    for (const char* S : { "Left", "Right" }) {
        const std::string s(S);
        p.map.push_back({ s + "Shoulder", s + "Collar", s + "Arm", s + "Shoulder" });
        p.map.push_back({ s + "Arm", s + "Shoulder", s + "ForeArm", s + "Elbow" });
        p.map.push_back({ s + "ForeArm", s + "Elbow", s + "Hand", s + "Wrist" });
        p.map.push_back({ s + "Hand", s + "Wrist", s + "HandMiddle1", "END" });
        p.map.push_back({ s + "UpLeg", s + "Hip", s + "Leg", s + "Knee" });
        p.map.push_back({ s + "Leg", s + "Knee", s + "Foot", s + "Ankle" });
        p.map.push_back({ s + "Foot", s + "Ankle", s + "ToeBase", s + "Toe" });
        p.map.push_back({ s + "ToeBase", s + "Toe", "TAIL", "END" });
    }
    return p;
}

bool retarget_bvh(const Bvh& bvh, int first, int end, const SkeletonDef& skel, const RetargetParams& params, RawClip& out,
                  std::string* why) {
    out = RawClip{};
    auto fail = [&](const std::string& s) { if (why) *why = s; out = RawClip{}; return false; };
    const double src_rate = 1.0 / bvh.frame_time;
    const int step = int(std::lround(src_rate / double(params.out_rate)));
    if (step < 1 || std::abs(src_rate / double(step) - double(params.out_rate)) > 0.01 * double(params.out_rate))
        return fail("the capture's frame rate is not a whole multiple of the output rate");
    first = std::max(first, 0);
    end = std::min(end, bvh.frames);
    const int n = (end - first + step - 1) / step;
    if (n < 2) return fail("frame range is empty");

    const int nj = skel.size();
    const glm::dmat3& C = params.axes;
    const glm::dmat3 Ct = glm::transpose(C);

    // Per mapped bone: the capture joint and the fixed reference rotation.
    struct Mapped { int target, source; glm::dmat3 ref_t; };
    std::vector<Mapped> mapped;
    std::vector<int> map_of(static_cast<size_t>(nj), -1);
    for (const BoneMap& m : params.map) {
        const int tb = skel.find(m.target), sj = bvh.find(m.source);
        if (tb < 0) return fail("rig has no joint " + m.target);
        if (sj < 0) return fail("capture has no joint " + m.source);
        glm::dmat3 ref(1.0);
        if (!m.target_child.empty()) {
            glm::dvec3 tdir, sdir;
            if (m.target_child == "TAIL") tdir = glm::dvec3(skel.rest_tail[size_t(tb)] - skel.rest_pos[size_t(tb)]);
            else {
                const int tc = skel.find(m.target_child);
                if (tc < 0) return fail("rig has no joint " + m.target_child);
                tdir = glm::dvec3(skel.rest_pos[size_t(tc)] - skel.rest_pos[size_t(tb)]);
            }
            if (m.source_child == "END") {
                if (!bvh.joints[size_t(sj)].has_end) return fail("capture joint has no end site: " + m.source);
                sdir = C * bvh.joints[size_t(sj)].end;
            } else {
                const int sc = bvh.find(m.source_child);
                if (sc < 0) return fail("capture has no joint " + m.source_child);
                sdir = C * bvh.joints[size_t(sc)].offset;
            }
            if (glm::length(tdir) < 1e-6 || glm::length(sdir) < 1e-6) return fail("bone without a direction: " + m.target);
            ref = rot_from_to(sdir, tdir);
        }
        map_of[size_t(tb)] = int(mapped.size());
        mapped.push_back({ tb, sj, glm::transpose(ref) });
    }
    const int hips = skel.find("Hips"), la = skel.find("LeftFoot"), ra = skel.find("RightFoot");
    const int src_hips = bvh.find("Hips"), ju = bvh.find(params.leg_upper), jl = bvh.find(params.leg_lower);
    if (hips < 0 || la < 0 || ra < 0 || src_hips < 0 || ju < 0 || jl < 0 || map_of[size_t(hips)] < 0) return fail("hips / legs are not mapped");
    if (skel.parents[size_t(hips)] != 0) return fail("the hips must hang directly under the root joint");
    const int tl = skel.find("LeftLeg"), tu = skel.find("LeftUpLeg");
    if (tl < 0 || tu < 0) return fail("rig has no left leg");
    const double tgt_leg = double(glm::length(skel.rest_pos[size_t(tl)] - skel.rest_pos[size_t(tu)]) +
                                  glm::length(skel.rest_pos[size_t(la)] - skel.rest_pos[size_t(tl)]));
    const double src_leg = glm::length(bvh.joints[size_t(ju)].offset) + glm::length(bvh.joints[size_t(jl)].offset);
    if (src_leg < 1e-6) return fail("capture leg has no length");
    const double scale = tgt_leg / src_leg;

    // ---- rotations from rest (world, character axes) and the hips' path ----
    const size_t N = static_cast<size_t>(n);
    std::vector<std::vector<glm::dmat3>> D(N, std::vector<glm::dmat3>(mapped.size()));
    std::vector<glm::dvec3> hips_w(N);
    {
        std::vector<glm::dmat3> G;
        std::vector<glm::dvec3> P;
        for (int f = 0; f < n; ++f) {
            bvh_fk(bvh, first + f * step, G, P);
            for (size_t k = 0; k < mapped.size(); ++k) D[size_t(f)][k] = C * G[size_t(mapped[k].source)] * Ct * mapped[k].ref_t;
            hips_w[size_t(f)] = C * P[size_t(src_hips)] * scale;
        }
    }
    // The character's FK with those rotations (joints that are not mapped follow their parent).
    const size_t NJ = static_cast<size_t>(nj);
    std::vector<glm::dmat3> Rb(NJ);
    std::vector<glm::dvec3> Pb(NJ);
    auto fk = [&](int f, const glm::dmat3& root_rot, const glm::dvec3& root_pos) {
        for (int j = 0; j < nj; ++j) {
            const int p = skel.parents[size_t(j)];
            if (p < 0) { Rb[size_t(j)] = root_rot; Pb[size_t(j)] = root_pos; continue; }
            Rb[size_t(j)] = map_of[size_t(j)] >= 0 ? D[size_t(f)][size_t(map_of[size_t(j)])] : Rb[size_t(p)];
            Pb[size_t(j)] = j == hips ? hips_w[size_t(f)]
                                      : Pb[size_t(p)] + Rb[size_t(p)] * glm::dvec3(skel.rest_pos[size_t(j)] - skel.rest_pos[size_t(p)]);
        }
    };
    // The standing foot's ankle at its rest height.
    {
        std::vector<double> low(N);
        for (int f = 0; f < n; ++f) {
            fk(f, glm::dmat3(1.0), glm::dvec3(0.0));
            low[size_t(f)] = std::min(Pb[size_t(la)].y, Pb[size_t(ra)].y);
        }
        std::nth_element(low.begin(), low.begin() + n / 2, low.end());
        const double lift = double(skel.rest_pos[size_t(la)].y) - low[size_t(n / 2)];
        for (auto& h : hips_w) h.y += lift;
    }
    // ---- the root ----
    const double sigma = double(params.root_smooth_s) * double(params.out_rate);
    std::vector<glm::dvec3> flat(N), fw(N);
    const size_t hk = size_t(map_of[size_t(hips)]);
    for (int f = 0; f < n; ++f) {
        flat[size_t(f)] = glm::dvec3(hips_w[size_t(f)].x, 0.0, hips_w[size_t(f)].z);
        const glm::dvec3 d = D[size_t(f)][hk] * glm::dvec3(0.0, 0.0, -1.0);
        fw[size_t(f)] = glm::dvec3(d.x, 0.0, d.z);
    }
    const std::vector<glm::dvec3> rp = gauss(flat, sigma, true);
    fw = gauss(fw, sigma);
    for (auto& d : fw) {
        const double l = glm::length(d);
        d = l > 1e-9 ? d / l : glm::dvec3(0.0, 0.0, -1.0);
    }

    out.frames = n;
    out.joints = nj;
    out.local.resize(size_t(n) * size_t(nj));
    out.info.sample_rate = params.out_rate;
    out.info.duration = float(n - 1) / params.out_rate;
    for (int f = 0; f < n; ++f) {
        // root rotation: -z -> fw, y up. Columns: right, up, back.
        glm::dmat3 Rr;
        Rr[2] = -fw[size_t(f)];
        Rr[1] = glm::dvec3(0.0, 1.0, 0.0);
        Rr[0] = glm::cross(Rr[1], Rr[2]);
        fk(f, Rr, rp[size_t(f)] - rp[0]);
        Pb[size_t(hips)] -= rp[0];
        for (int j = 0; j < nj; ++j) {
            const int p = skel.parents[size_t(j)];
            Transform t;
            if (p < 0) {
                t.t = glm::vec3(Pb[size_t(j)]);
                t.r = glm::quat(glm::quat_cast(Rb[size_t(j)]));
            } else {
                const glm::dmat3 pt = glm::transpose(Rb[size_t(p)]);
                // Only the hips move against their parent; every other joint keeps its rest offset exactly.
                t.t = j == hips ? glm::vec3(pt * (Pb[size_t(j)] - Pb[size_t(p)])) : skel.rest_local(j);
                t.r = glm::quat(glm::quat_cast(pt * Rb[size_t(j)]));
            }
            t.r = glm::normalize(t.r);
            if (f > 0 && glm::dot(t.r, out.at(f - 1, j).r) < 0.0f) t.r = -t.r;       // one hemisphere through time
            out.at(f, j) = t;
        }
    }
    label_contacts(skel, out);
    return true;
}

}  // namespace qlike::anim::offline
