#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "fixtures.h"

using namespace qlike::anim;
using namespace qlike::anim::offline;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("anim import: BVH parsing", "[anim][import]") {
    Bvh b;
    std::string why;
    REQUIRE(parse_bvh(fx::make_bvh(120, 120.0, 20.0), b, &why));
    CHECK(b.joints.size() == 23);
    CHECK(b.frames == 120);
    CHECK(b.columns == 6 + 22 * 3);
    CHECK_THAT(b.frame_time, WithinAbs(1.0 / 60.0, 1e-6));
    CHECK(b.joints[0].parent == -1);
    CHECK(b.joints[size_t(b.find("Head"))].has_end);
    CHECK(b.joints[size_t(b.find("LeftKnee"))].parent == b.find("LeftHip"));
    CHECK(b.joints[size_t(b.find("LeftCollar"))].parent == b.find("Chest4"));
    CHECK(b.joints[size_t(b.find("Chest"))].first_channel == 6);
    CHECK_THAT(b.at(60, 2), WithinAbs(120.0, 1e-3));          // z after one second
}

TEST_CASE("anim import: damaged BVH files are refused", "[anim][import]") {
    Bvh b;
    std::string why;
    const std::string good = fx::make_bvh(30, 100.0, 10.0);
    CHECK_FALSE(parse_bvh("", b, &why));
    CHECK_FALSE(parse_bvh("HIERARCHY\nROOT Hips\n{\n", b, &why));
    CHECK_FALSE(parse_bvh(good.substr(0, good.size() - 400), b, &why));       // frames missing
    std::string bad = good;
    bad.replace(bad.find("Zrotation"), 9, "Wrotation");
    CHECK_FALSE(parse_bvh(bad, b, &why));
    bad = good;
    bad.replace(bad.find("Frames: 30"), 10, "Frames: -3");
    CHECK_FALSE(parse_bvh(bad, b, &why));
    CHECK(b.joints.empty());
    CHECK_FALSE(load_bvh("no/such/file.bvh", b, &why));
}

TEST_CASE("anim import: BVH forward kinematics", "[anim][import]") {
    Bvh b;
    REQUIRE(parse_bvh(fx::make_bvh(4, 0.0, 0.0), b));
    std::vector<glm::dmat3> R;
    std::vector<glm::dvec3> P;
    bvh_fk(b, 0, R, P);
    const int la = b.find("LeftAnkle"), lw = b.find("LeftWrist");
    CHECK_THAT(P[size_t(la)].x, WithinAbs(9.0, 1e-9));
    CHECK_THAT(P[size_t(la)].y, WithinAbs(90.0 - 84.0, 1e-9));
    CHECK_THAT(P[size_t(lw)].x, WithinAbs(3.0 + 14.0 + 28.0 + 25.0, 1e-9));
    // Turn the root a quarter turn about Y: +x offsets go to -z.
    b.data[3] = 90.0;
    bvh_fk(b, 0, R, P);
    CHECK_THAT(P[size_t(la)].x, WithinAbs(0.0, 1e-9));
    CHECK_THAT(P[size_t(la)].z, WithinAbs(-9.0, 1e-9));
}

TEST_CASE("anim import: retargeting keeps the rig and converts the axes", "[anim][import]") {
    const SkeletonDef s = fx::make_skeleton();
    Bvh b;
    REQUIRE(parse_bvh(fx::make_bvh(241, 120.0, 25.0), b));
    RawClip c;
    std::string why;
    INFO(why);
    const bool retargeted = retarget_bvh(b, 0, 241, s, style100_params(), c, &why);
    INFO(why);
    REQUIRE(retargeted);
    CHECK(c.frames == 121);                                   // 60 -> 30 fps, both ends kept
    CHECK(c.joints == s.size());
    CHECK_THAT(c.info.duration, WithinAbs(4.0f, 1e-4));
    const ClipReport rep = validate_clip(s, c);
    INFO(rep.errors << rep.warnings);
    CHECK(rep.ok);
    CHECK(rep.max_limb_error < 1e-4f);

    // Walking +Z in the capture is walking -Z for the character, scaled leg for leg.
    const float leg_t = glm::length(s.rest_local(s.find("LeftLeg"))) + glm::length(s.rest_local(s.find("LeftFoot")));
    const float scale = leg_t / 84.0f;
    const Transform r0 = c.at(0, 0), r1 = c.at(c.frames - 1, 0);
    CHECK_THAT(r0.t.x, WithinAbs(0.0f, 1e-4));
    CHECK_THAT(r0.t.z, WithinAbs(0.0f, 1e-4));
    CHECK_THAT(r1.t.x, WithinAbs(0.0f, 1e-3));
    CHECK_THAT(r1.t.y, WithinAbs(0.0f, 1e-6));
    CHECK_THAT(r1.t.z, WithinRel(-120.0f * 4.0f * scale, 0.005f));
    CHECK_THAT(rep.mean_speed, WithinRel(120.0f * scale, 0.005f));
    // The root faces the way the hips face: straight down -Z.
    const glm::vec3 face = c.at(60, 0).r * glm::vec3(0, 0, -1);
    CHECK_THAT(face.z, WithinAbs(-1.0f, 1e-4));

    // Left stays left, and a leg that is forward in the capture is forward on the character.
    std::vector<glm::quat> rot;
    std::vector<glm::vec3> pos;
    std::vector<glm::dmat3> G;
    std::vector<glm::dvec3> P;
    const int lf = s.find("LeftFoot"), rf = s.find("RightFoot");
    int checked = 0;
    for (int f = 5; f < c.frames - 5; f += 7) {
        clip_fk(s, c, f, rot, pos);
        bvh_fk(b, f * 2, G, P);
        const double cap_left_fwd = P[size_t(b.find("LeftAnkle"))].z - P[0].z;       // + = forward in the capture
        if (std::abs(cap_left_fwd) < 5.0) continue;
        const float left_fwd = -(pos[size_t(lf)].z - pos[0].z), right_fwd = -(pos[size_t(rf)].z - pos[0].z);
        CHECK((left_fwd > 0.0f) == (cap_left_fwd > 0.0));
        CHECK((right_fwd > 0.0f) != (cap_left_fwd > 0.0));
        CHECK(pos[size_t(lf)].x < pos[size_t(rf)].x);
        ++checked;
    }
    CHECK(checked > 5);
    // The lower ankle's usual height is the rig's ankle rest height.
    // (the test capture swings stiff legs, so a vertical leg dips below the median a little)
    CHECK(std::abs(rep.lowest_ankle) < 0.06f);
}

TEST_CASE("anim import: frame ranges and rates", "[anim][import]") {
    const SkeletonDef s = fx::make_skeleton();
    Bvh b;
    REQUIRE(parse_bvh(fx::make_bvh(200, 100.0, 10.0), b));
    RawClip c;
    std::string why;
    REQUIRE(retarget_bvh(b, 40, 100, s, style100_params(), c, &why));
    CHECK(c.frames == 30);
    RetargetParams p = style100_params();
    p.out_rate = 60.0f;
    REQUIRE(retarget_bvh(b, 40, 100, s, p, c, &why));
    CHECK(c.frames == 60);
    p.out_rate = 25.0f;                                        // 60 / 25 is not whole
    CHECK_FALSE(retarget_bvh(b, 40, 100, s, p, c, &why));
    CHECK_FALSE(retarget_bvh(b, 100, 101, s, style100_params(), c, &why));
    // A rig without a mapped joint is an error, not a silent skip.
    SkeletonDef other = s;
    other.names[size_t(other.find("Neck"))] = "Throat";
    CHECK_FALSE(retarget_bvh(b, 0, 200, other, style100_params(), c, &why));
    CHECK(why.find("Neck") != std::string::npos);
}

TEST_CASE("anim import: foot contacts", "[anim][import]") {
    const SkeletonDef s = fx::make_skeleton();
    Bvh b;
    RawClip c;
    // Standing still: both feet down all the time.
    REQUIRE(parse_bvh(fx::make_bvh(180, 0.0, 0.0), b));
    REQUIRE(retarget_bvh(b, 0, 180, s, style100_params(), c));
    for (uint8_t v : c.contact) CHECK(v == 1);
    // A hop in the middle third: no contact there, contact before and after.
    REQUIRE(parse_bvh(fx::make_bvh(180, 0.0, 0.0, 40.0), b));
    REQUIRE(retarget_bvh(b, 0, 180, s, style100_params(), c));
    REQUIRE(c.frames == 90);
    CHECK(c.contact[size_t(10) * 2] == 1);
    CHECK(c.contact[size_t(45) * 2] == 0);
    CHECK(c.contact[size_t(45) * 2 + 1] == 0);
    CHECK(c.contact[size_t(80) * 2 + 1] == 1);
    // Walking: each foot is down part of the time (the swing leg moves fast).
    REQUIRE(parse_bvh(fx::make_bvh(240, 120.0, 30.0), b));
    REQUIRE(retarget_bvh(b, 0, 240, s, style100_params(), c));
    const ClipReport rep = validate_clip(s, c);
    CHECK(rep.contact_left < 0.9f);
    CHECK(rep.contact_right < 0.9f);
}

TEST_CASE("anim import: validation catches broken clips", "[anim][import]") {
    const SkeletonDef s = fx::make_skeleton();
    const RawClip good = fx::make_clip(s, "walk", 61, 30.0f, 1.5f);
    CHECK(validate_clip(s, good).ok);
    RawClip c = good;
    c.at(10, 4).r = glm::quat(2.0f, 0.0f, 0.0f, 0.0f);
    CHECK_FALSE(validate_clip(s, c).ok);
    c = good;
    c.at(3, 2).t.x = std::nanf("");
    CHECK_FALSE(validate_clip(s, c).ok);
    c = good;
    c.at(20, s.find("LeftLeg")).t.y -= 0.05f;                 // a stretched thigh
    CHECK_FALSE(validate_clip(s, c).ok);
    c = good;
    c.info.duration *= 2.0f;
    CHECK_FALSE(validate_clip(s, c).ok);
    c = fx::make_clip(s, "rocket", 61, 30.0f, 40.0f);
    CHECK_FALSE(validate_clip(s, c).ok);
    c = good;
    c.joints -= 1;
    CHECK_FALSE(validate_clip(s, c).ok);
}

TEST_CASE("anim import: the runtime skeleton keeps the character file's joint order", "[anim][import]") {
    SkeletonDef s = fx::make_skeleton();
    std::vector<uint8_t> blob;
    std::string why;
    CHECK(build_runtime_skeleton(s, blob, &why));
    CHECK(!blob.empty());
    // Move a leaf (the head) to the end: still parents-first, no longer depth-first.
    const int head = s.find("Head");
    auto move_last = [&](auto& v) { auto x = v[size_t(head)]; v.erase(v.begin() + head); v.push_back(x); };
    move_last(s.names); move_last(s.rest_pos); move_last(s.rest_tail);
    int parent = s.parents[size_t(head)];
    s.parents.erase(s.parents.begin() + head);
    for (int& p : s.parents) if (p > head) --p;
    s.parents.push_back(parent);
    REQUIRE(s.validate(&why));
    CHECK_FALSE(build_runtime_skeleton(s, blob, &why));
}
