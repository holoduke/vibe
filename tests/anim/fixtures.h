#pragma once
// Test data for the animation tests: a small humanoid rig with every role joint, clips with
// known motion, and a synthetic capture file with the 100STYLE joint names. Nothing here
// needs the mocap dataset.
#include "anim/offline/cook.h"
#include "anim/runtime.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace fx {

using namespace qlike::anim;
using namespace qlike::anim::offline;

inline SkeletonDef make_skeleton() {
    SkeletonDef s;
    auto add = [&](const char* name, int parent, glm::vec3 pos, glm::vec3 tail_dir) {
        s.names.push_back(name);
        s.parents.push_back(parent);
        s.rest_pos.push_back(pos);
        s.rest_tail.push_back(pos + tail_dir * 0.1f);
        return int(s.names.size()) - 1;
    };
    const glm::vec3 up(0, 1, 0), fwd(0, 0, -1), down(0, -1, 0);
    const int root = add("Root", -1, { 0, 0, 0 }, up);
    const int hips = add("Hips", root, { 0, 0.93f, 0 }, up);
    const int sp0 = add("Spine", hips, { 0, 1.02f, 0.02f }, up);
    const int sp1 = add("Spine1", sp0, { 0, 1.10f, 0.01f }, up);
    const int sp2 = add("Spine2", sp1, { 0, 1.18f, 0.02f }, up);
    for (int side = 0; side < 2; ++side) {
        const float x = side == 0 ? -1.0f : 1.0f;
        const std::string S = side == 0 ? "Left" : "Right";
        const int sh = add((S + "Shoulder").c_str(), sp2, { x * 0.03f, 1.40f, -0.03f }, { x, 0, 0 });
        const int ar = add((S + "Arm").c_str(), sh, { x * 0.18f, 1.38f, -0.02f }, { x * 0.6f, -0.8f, 0 });
        const int fa = add((S + "ForeArm").c_str(), ar, { x * 0.35f, 1.19f, -0.02f }, { x * 0.5f, -0.5f, -0.7f });
        const int hd = add((S + "Hand").c_str(), fa, { x * 0.48f, 1.06f, -0.20f }, { x * 0.3f, -0.4f, -0.8f });
        add((S + "HandMiddle1").c_str(), hd, { x * 0.52f, 1.01f, -0.28f }, { x * 0.3f, -0.4f, -0.8f });
    }
    const int neck = add("Neck", sp2, { 0, 1.48f, -0.01f }, up);
    add("Head", neck, { 0, 1.57f, -0.04f }, up);
    for (int side = 0; side < 2; ++side) {
        const float x = side == 0 ? -1.0f : 1.0f;
        const std::string S = side == 0 ? "Left" : "Right";
        const int ul = add((S + "UpLeg").c_str(), hips, { x * 0.10f, 0.92f, 0 }, down);
        const int lg = add((S + "Leg").c_str(), ul, { x * 0.14f, 0.49f, -0.03f }, down);
        const int ft = add((S + "Foot").c_str(), lg, { x * 0.18f, 0.07f, -0.01f }, fwd);
        add((S + "ToeBase").c_str(), ft, { x * 0.18f, 0.01f, -0.14f }, fwd);
    }
    return s;
}

// A clip made by formula: the root travels along -Z at `speed` and turns at `yaw_rate`,
// the hips bob, the thighs swing about X in opposite phase, the spine sways.
inline RawClip make_clip(const SkeletonDef& s, const char* name, int frames, float rate, float speed, float yaw_rate = 0.0f,
                         bool loop = false) {
    RawClip c;
    c.info.name = name;
    c.info.sample_rate = rate;
    c.info.duration = float(frames - 1) / rate;
    c.info.loop = loop;
    c.info.tags = kTagLocomotion;
    c.info.provenance = "tests/anim/fixtures.h";
    c.info.licence = "MIT";
    c.frames = frames;
    c.joints = s.size();
    c.local.resize(size_t(frames) * size_t(c.joints));
    const int hips = s.find("Hips"), lu = s.find("LeftUpLeg"), ru = s.find("RightUpLeg"), sp = s.find("Spine");
    glm::vec3 pos(0.0f);
    float yaw = 0.0f;
    for (int f = 0; f < frames; ++f) {
        const float t = float(f) / rate;
        for (int j = 0; j < c.joints; ++j) c.at(f, j).t = s.rest_local(j);
        c.at(f, 0).t = pos;
        c.at(f, 0).r = quat_yaw(yaw);
        const float ph = 6.2831853f * 1.5f * t;
        const float amp = speed > 0.0f ? 0.5f : 0.0f;
        c.at(f, hips).t = s.rest_local(hips) + glm::vec3(0.0f, 0.02f * amp * std::cos(2.0f * ph), 0.0f);
        c.at(f, lu).r = glm::angleAxis(amp * std::sin(ph), glm::vec3(1, 0, 0));
        c.at(f, ru).r = glm::angleAxis(-amp * std::sin(ph), glm::vec3(1, 0, 0));
        c.at(f, sp).r = glm::angleAxis(0.1f * amp * std::sin(ph), glm::vec3(0, 1, 0));
        const glm::vec3 dir = quat_yaw(yaw) * glm::vec3(0.0f, 0.0f, -1.0f);
        pos += dir * (speed / rate);
        yaw += yaw_rate / rate;
    }
    label_contacts(s, c);
    return c;
}

struct TestAsset {
    AnimAsset            asset;
    std::vector<RawClip> raw;
};

inline TestAsset make_asset(bool optimize = true) {
    TestAsset t;
    const SkeletonDef s = make_skeleton();
    t.asset.skeleton = s;
    t.asset.runtime_format = runtime_format_name();
    std::string why;
    if (!build_runtime_skeleton(s, t.asset.skeleton_blob, &why)) throw std::runtime_error(why);
    t.raw.push_back(make_clip(s, "walk", 61, 30.0f, 1.5f));
    t.raw.push_back(make_clip(s, "still", 31, 30.0f, 0.0f));
    t.raw.push_back(make_clip(s, "turn", 61, 30.0f, 1.0f, 1.0f));
    t.raw.push_back(make_clip(s, "cycle", 41, 30.0f, 1.5f, 0.0f, true));       // 1.5 Hz x 40/30 s = 2 whole cycles
    CompressParams cp;
    cp.optimize = optimize;
    for (const RawClip& c : t.raw) {
        std::vector<uint8_t> blob;
        if (!build_runtime_clip(s, c, cp, blob, &why)) throw std::runtime_error(why);
        t.asset.clips.push_back(c.info);
        t.asset.clip_blobs.push_back(std::move(blob));
    }
    return t;
}

// A capture file with the 100STYLE joint names: T pose, +Z forward, +X left, centimetres,
// 60 fps. The performer walks along +Z at `speed_cm` and swings the thighs by `swing_deg`;
// `hop_cm` lifts the whole body in the middle third of the take (a jump without contacts).
inline std::string make_bvh(int frames, double speed_cm, double swing_deg, double hop_cm = 0.0, double frame_time = 1.0 / 60.0) {
    std::string h = "HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation\n";
    auto joint = [&](const char* name, double x, double y, double z) {
        char b[160];
        std::snprintf(b, sizeof(b), " JOINT %s\n {\n OFFSET %f %f %f\n CHANNELS 3 Yrotation Xrotation Zrotation\n", name, x, y, z);
        h += b;
    };
    auto end = [&](double x, double y, double z) {
        char b[96];
        std::snprintf(b, sizeof(b), " End Site\n {\n OFFSET %f %f %f\n }\n", x, y, z);
        h += b;
    };
    auto close = [&](int n) { for (int i = 0; i < n; ++i) h += " }\n"; };
    joint("Chest", 0, 13, 0); joint("Chest2", 0, 10, 0); joint("Chest3", 0, 9, 0); joint("Chest4", 0, 9, 0);
    joint("Neck", 0, 13, 0); joint("Head", 0, 9, 0); end(0, 17, 0); close(2);
    for (int side = 0; side < 2; ++side) {
        const double x = side == 0 ? -1.0 : 1.0;                    // right is -x in the capture
        const std::string S = side == 0 ? "Right" : "Left";
        joint((S + "Collar").c_str(), x * 3, 7, 0); joint((S + "Shoulder").c_str(), x * 14, 0, 0);
        joint((S + "Elbow").c_str(), x * 28, 0, 0); joint((S + "Wrist").c_str(), x * 25, 0, 0); end(x * 16, 0, 0); close(4);
    }
    close(4);
    for (int side = 0; side < 2; ++side) {
        const double x = side == 0 ? -1.0 : 1.0;
        const std::string S = side == 0 ? "Right" : "Left";
        joint((S + "Hip").c_str(), x * 9, 0, 0); joint((S + "Knee").c_str(), 0, -42, 0);
        joint((S + "Ankle").c_str(), 0, -42, 0); joint((S + "Toe").c_str(), 0, -6, 14); end(0, 0, 6); close(4);
    }
    h += "}\nMOTION\n";
    char b[96];
    std::snprintf(b, sizeof(b), "Frames: %d\nFrame Time: %f\n", frames, frame_time);
    h += b;
    // column layout: hips 6, then 3 per joint in declaration order
    const char* order[] = { "Chest", "Chest2", "Chest3", "Chest4", "Neck", "Head", "RightCollar", "RightShoulder", "RightElbow",
                            "RightWrist", "LeftCollar", "LeftShoulder", "LeftElbow", "LeftWrist", "RightHip", "RightKnee",
                            "RightAnkle", "RightToe", "LeftHip", "LeftKnee", "LeftAnkle", "LeftToe" };
    for (int f = 0; f < frames; ++f) {
        const double t = double(f) * frame_time;
        const bool hop = hop_cm > 0.0 && f >= frames / 3 && f < 2 * frames / 3;
        std::snprintf(b, sizeof(b), "0 %f %f 0 0 0", 90.0 + (hop ? hop_cm : 0.0), speed_cm * t);
        h += b;
        for (const char* n : order) {
            double xr = 0.0;
            // a negative X rotation swings a leg hanging down toward +Z (forward in the capture)
            if (std::string(n) == "LeftHip") xr = -swing_deg * std::sin(6.2831853 * t);
            if (std::string(n) == "RightHip") xr = swing_deg * std::sin(6.2831853 * t);
            std::snprintf(b, sizeof(b), " 0 %f 0", xr);
            h += b;
        }
        h += "\n";
    }
    return h;
}

inline float joint_error(const Transform& a, const Transform& b) {
    const glm::quat d = glm::inverse(a.r) * b.r;                 // (asin of the vector part: exact for tiny angles)
    return glm::length(a.t - b.t) + 2.0f * std::asin(std::min(glm::length(glm::vec3(d.x, d.y, d.z)), 1.0f));
}

}  // namespace fx

#include "anim/offline/motion_build.h"

namespace fx {

// A clip whose root travels at `vel` (character frame: x right, z back) while facing -Z; the
// thighs swing along the travel direction. `hop`: the body leaves the ground in the middle.
inline RawClip make_move_clip(const SkeletonDef& s, const char* name, int frames, float rate, glm::vec2 vel, uint32_t tags,
                              bool hop = false) {
    RawClip c = make_clip(s, name, frames, rate, 0.0f);
    c.info.tags = tags;
    const int hips = s.find("Hips"), lu = s.find("LeftUpLeg"), ru = s.find("RightUpLeg");
    const float speed = glm::length(vel);
    const glm::vec3 axis = speed > 1e-4f ? glm::normalize(glm::vec3(-vel.y, 0.0f, vel.x)) : glm::vec3(1, 0, 0);
    for (int f = 0; f < frames; ++f) {
        const float t = float(f) / rate;
        c.at(f, 0).t = glm::vec3(vel.x, 0.0f, vel.y) * t;
        const float ph = 6.2831853f * 1.5f * t, amp = (speed > 0.0f && !hop) ? 0.45f : 0.0f;     // (the hop slides on still legs: its feet are down outside the flight)
        c.at(f, lu).r = glm::angleAxis(amp * std::sin(ph), axis);
        c.at(f, ru).r = glm::angleAxis(-amp * std::sin(ph), axis);
        if (tags & kTagCrouch) c.at(f, hips).t.y -= 0.25f;
        if (hop && f >= frames / 3 && f < 2 * frames / 3) {
            const float u = float(f - frames / 3) / float(frames / 3);
            c.at(f, hips).t.y += 0.08f + 0.5f * 4.0f * u * (1.0f - u);
        }
    }
    label_contacts(s, c);
    return c;
}

struct MotionFixture {
    SkeletonDef          skel = make_skeleton();
    std::vector<RawClip> clips;
    MotionDatabase       db;
    std::vector<uint8_t> blob;
    MotionFixture() {
        clips.push_back(make_move_clip(skel, "idle", 120, 30.0f, { 0.0f, 0.0f }, kTagLocomotion | kTagIdle));
        clips.push_back(make_move_clip(skel, "walk_f", 300, 30.0f, { 0.0f, -1.0f }, kTagLocomotion));
        clips.push_back(make_move_clip(skel, "run_f", 300, 30.0f, { 0.0f, -2.0f }, kTagLocomotion));
        clips.push_back(make_move_clip(skel, "walk_b", 300, 30.0f, { 0.0f, 0.8f }, kTagLocomotion));
        clips.push_back(make_move_clip(skel, "strafe_r", 300, 30.0f, { 1.2f, 0.0f }, kTagLocomotion));
        clips.push_back(make_move_clip(skel, "crouch_f", 200, 30.0f, { 0.0f, -0.7f }, kTagCrouch));
        clips.push_back(make_move_clip(skel, "hop", 150, 30.0f, { 0.0f, -0.2f }, kTagJump, true));
        std::string why;
        if (!build_motion_database(skel, clips, MotionBuildParams{}, blob, nullptr, &why)) throw std::runtime_error(why);
        if (!db.load(blob.data(), blob.size(), skel.rig_id(), &why)) throw std::runtime_error(why);
    }
    int range_named(const char* name) const {
        for (size_t i = 0; i < clips.size(); ++i)
            if (clips[i].info.name == name) return int(i);
        return -1;
    }
    const char* clip_of(int frame) const { return frame < 0 ? "none" : clips[size_t(db.range_of(uint32_t(frame)).clip)].info.name.c_str(); }
};

}  // namespace fx
