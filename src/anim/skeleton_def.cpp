#include "anim/skeleton_def.h"

#include <cmath>
#include <cstring>
#include <fstream>

namespace vibe::anim {

namespace {
const char* const kRoleNames[int(Role::Count)] = {
    "Root", "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
};
}  // namespace

const char* role_name(Role r) { return kRoleNames[int(r)]; }

int SkeletonDef::find(std::string_view name) const {
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == name) return int(i);
    return -1;
}

bool SkeletonDef::validate(std::string* why) const {
    auto fail = [&](const std::string& s) { if (why) *why = s; return false; };
    const size_t n = names.size();
    if (n == 0) return fail("no joints");
    if (parents.size() != n || rest_pos.size() != n || rest_tail.size() != n) return fail("array sizes differ");
    int roots = 0;
    for (size_t i = 0; i < n; ++i) {
        if (parents[i] < 0) ++roots;
        else if (parents[i] >= int(i)) return fail("joint " + names[i] + ": parent does not come before it");
        if (!is_finite(rest_pos[i]) || !is_finite(rest_tail[i])) return fail("joint " + names[i] + ": position is not finite");
        if (names[i].empty()) return fail("joint without a name");
        for (size_t k = 0; k < i; ++k)
            if (names[k] == names[i]) return fail("joint name used twice: " + names[i]);
    }
    if (roots != 1) return fail("expected exactly one root joint");
    if (parents[0] >= 0) return fail("the root must be joint 0");
    return true;
}

uint64_t SkeletonDef::rig_id() const {
    uint64_t h = 1469598103934665603ull;                    // FNV-1a
    auto mix = [&](const void* p, size_t len) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < len; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    for (size_t i = 0; i < names.size(); ++i) {
        mix(names[i].data(), names[i].size());
        const int32_t par = parents[i];
        mix(&par, sizeof(par));
        for (int k = 0; k < 3; ++k) {
            const int32_t mm = int32_t(std::lround(rest_pos[i][k] * 1000.0f));
            mix(&mm, sizeof(mm));
        }
    }
    return h;
}

bool load_qchar_skeleton(const char* path, SkeletonDef& out, std::string* why) {
    out = SkeletonDef{};
    auto fail = [&](const char* s) { if (why) *why = s; out = SkeletonDef{}; return false; };
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return fail("cannot open the character file");
    char magic[4]{};
    uint32_t nb = 0, nv = 0, ni = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&nb), 4);
    f.read(reinterpret_cast<char*>(&nv), 4);
    f.read(reinterpret_cast<char*>(&ni), 4);
    if (!f.good() || std::memcmp(magic, "QCH1", 4) != 0 || nb == 0 || nb > 255) return fail("not a QCH1 character file");
    for (uint32_t i = 0; i < nb; ++i) {
        char name[32]{};
        int32_t parent = -1;
        glm::vec3 head, tail;
        f.read(name, 32);
        f.read(reinterpret_cast<char*>(&parent), 4);
        f.read(reinterpret_cast<char*>(&head), 12);
        f.read(reinterpret_cast<char*>(&tail), 12);
        if (!f.good()) return fail("character file truncated");
        name[31] = 0;
        std::string n = name;
        if (const size_t c = n.find(':'); c != std::string::npos) n = n.substr(c + 1);
        out.names.push_back(n);
        out.parents.push_back(parent);
        out.rest_pos.push_back(head);
        out.rest_tail.push_back(tail);
    }
    std::string v;
    if (!out.validate(&v)) { if (why) *why = v; out = SkeletonDef{}; return false; }
    return true;
}

void scale_skeleton(SkeletonDef& s, float factor) {
    for (auto& p : s.rest_pos) p *= factor;
    for (auto& p : s.rest_tail) p *= factor;
    s.scale *= factor;
}

float eye_height_scale(const SkeletonDef& s, float stand_eye_height) {
    const int head = s.find("Head");
    float model_eye = 1.7f;
    if (head >= 0) model_eye = s.rest_pos[head].y / s.scale + 0.035f;
    return stand_eye_height / std::max(model_eye, 0.5f);
}

bool Rig::complete() const {
    for (int r : role) if (r < 0) return false;
    return true;
}

Rig make_rig(const SkeletonDef& s) {
    Rig rig{};
    for (int r = 0; r < int(Role::Count); ++r) rig.role[r] = s.find(kRoleNames[r]);
    rig.mirror.resize(s.names.size());
    for (size_t i = 0; i < s.names.size(); ++i) {
        std::string n = s.names[i];
        if (n.rfind("Left", 0) == 0) n = "Right" + n.substr(4);
        else if (n.rfind("Right", 0) == 0) n = "Left" + n.substr(5);
        const int o = s.find(n);
        rig.mirror[i] = o >= 0 ? o : int(i);
    }
    return rig;
}

}  // namespace vibe::anim
