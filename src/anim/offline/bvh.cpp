#include "anim/offline/bvh.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace vibe::anim::offline {

int Bvh::find(const std::string& name) const {
    for (size_t i = 0; i < joints.size(); ++i)
        if (joints[i].name == name) return int(i);
    return -1;
}

bool parse_bvh(const std::string& text, Bvh& out, std::string* why) {
    out = Bvh{};
    auto fail = [&](const std::string& s) { if (why) *why = s; out = Bvh{}; return false; };
    std::istringstream in(text);
    std::string tok;
    std::vector<int> stack;
    bool in_end = false, motion = false;
    while (in >> tok) {
        if (tok == "HIERARCHY") continue;
        if (tok == "ROOT" || tok == "JOINT") {
            Bvh::Joint j;
            if (!(in >> j.name)) return fail("joint without a name");
            j.parent = stack.empty() ? -1 : stack.back();
            out.joints.push_back(j);
            stack.push_back(int(out.joints.size()) - 1);
        } else if (tok == "End") {
            in >> tok;                                   // "Site"
            in_end = true;
        } else if (tok == "OFFSET") {
            glm::dvec3 v;
            if (!(in >> v.x >> v.y >> v.z) || stack.empty()) return fail("bad OFFSET");
            Bvh::Joint& j = out.joints[size_t(stack.back())];
            if (in_end) { j.has_end = true; j.end = v; }
            else j.offset = v;
        } else if (tok == "CHANNELS") {
            int n = 0;
            if (!(in >> n) || n < 0 || n > 6 || stack.empty()) return fail("bad CHANNELS");
            Bvh::Joint& j = out.joints[size_t(stack.back())];
            j.first_channel = out.columns;
            for (int k = 0; k < n; ++k) {
                if (!(in >> tok)) return fail("bad CHANNELS");
                const bool pos = tok.size() == 9 && tok.compare(1, 8, "position") == 0;
                const bool rot = tok.size() == 9 && tok.compare(1, 8, "rotation") == 0;
                if ((!pos && !rot) || (tok[0] != 'X' && tok[0] != 'Y' && tok[0] != 'Z')) return fail("unknown channel " + tok);
                j.channels.push_back(tok);
            }
            out.columns += n;
        } else if (tok == "{") {
        } else if (tok == "}") {
            if (in_end) in_end = false;
            else if (!stack.empty()) stack.pop_back();
            else return fail("unbalanced braces");
        } else if (tok == "MOTION") {
            motion = true;
            break;
        } else {
            return fail("unexpected token " + tok);
        }
    }
    if (!motion || out.joints.empty() || out.columns == 0) return fail("no MOTION block or no joints");
    if (!stack.empty()) return fail("unbalanced braces");
    if (!(in >> tok) || tok != "Frames:" || !(in >> out.frames) || out.frames <= 0) return fail("bad Frames:");
    std::string t2;
    if (!(in >> tok >> t2) || tok != "Frame" || t2 != "Time:" || !(in >> out.frame_time) || out.frame_time <= 0.0) return fail("bad Frame Time:");
    // The numbers: strtod over the rest of the text (far faster than stream extraction).
    const std::streampos at = in.tellg();
    if (at < 0) return fail("no frame data");
    const char* p = text.c_str() + size_t(at);
    const size_t want = size_t(out.frames) * size_t(out.columns);
    out.data.reserve(want);
    while (out.data.size() < want) {
        char* e = nullptr;
        const double v = std::strtod(p, &e);
        if (e == p) break;
        out.data.push_back(v);
        p = e;
    }
    if (out.data.size() != want) return fail("frame data is shorter than Frames: says");
    for (double v : out.data)
        if (!std::isfinite(v)) return fail("frame data holds a value that is not finite");
    return true;
}

bool load_bvh(const std::string& path, Bvh& out, std::string* why) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { if (why) *why = "cannot open " + path; out = Bvh{}; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    return parse_bvh(ss.str(), out, why);
}

namespace {
glm::dmat3 axis_rot(char axis, double deg) {
    const double a = glm::radians(deg), c = std::cos(a), s = std::sin(a);
    glm::dmat3 m(1.0);                                  // m[col][row]
    if (axis == 'X') { m[1][1] = c; m[2][1] = -s; m[1][2] = s; m[2][2] = c; }
    else if (axis == 'Y') { m[0][0] = c; m[2][0] = s; m[0][2] = -s; m[2][2] = c; }
    else { m[0][0] = c; m[1][0] = -s; m[0][1] = s; m[1][1] = c; }
    return m;
}
}  // namespace

void bvh_fk(const Bvh& b, int frame, std::vector<glm::dmat3>& rot, std::vector<glm::dvec3>& pos) {
    const size_t n = b.joints.size();
    rot.resize(n);
    pos.resize(n);
    for (size_t j = 0; j < n; ++j) {
        const Bvh::Joint& jt = b.joints[j];
        glm::dmat3 R(1.0);
        glm::dvec3 p(0.0);
        bool has_pos = false;
        for (size_t k = 0; k < jt.channels.size(); ++k) {
            const double v = b.at(frame, jt.first_channel + int(k));
            const std::string& c = jt.channels[k];
            if (c[1] == 'p') { p[c[0] - 'X'] = v; has_pos = true; }
            else R = R * axis_rot(c[0], v);
        }
        if (jt.parent < 0) {
            rot[j] = R;
            pos[j] = has_pos ? p : jt.offset;
        } else {
            rot[j] = rot[size_t(jt.parent)] * R;
            pos[j] = pos[size_t(jt.parent)] + rot[size_t(jt.parent)] * jt.offset;
        }
    }
}

}  // namespace vibe::anim::offline
