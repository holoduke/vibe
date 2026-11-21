#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "anim/asset.h"
#include "fixtures.h"

#include <fstream>

using namespace qlike::anim;
using Catch::Matchers::WithinAbs;

namespace {
const char* kSoldier = QLIKE_SOURCE_DIR "/assets/character/soldier_fp.qchar";
}

TEST_CASE("anim: the soldier's skeleton loads and is a valid rig", "[anim][skeleton]") {
    SkeletonDef s;
    std::string why;
    REQUIRE(load_qchar_skeleton(kSoldier, s, &why));
    REQUIRE(s.validate(&why));
    CHECK(s.size() == 53);
    CHECK(s.parents[0] == -1);
    for (int i = 1; i < s.size(); ++i) {
        INFO(s.names[size_t(i)]);
        CHECK(s.parents[size_t(i)] >= 0);
        CHECK(s.parents[size_t(i)] < i);             // parents come first
    }
    const Rig rig = make_rig(s);
    CHECK(rig.complete());
    CHECK(s.parents[size_t(rig[Role::Hips])] == rig[Role::Root]);
    CHECK(s.parents[size_t(rig[Role::LeftLeg])] == rig[Role::LeftUpLeg]);
    CHECK(s.parents[size_t(rig[Role::LeftFoot])] == rig[Role::LeftLeg]);
    CHECK(s.parents[size_t(rig[Role::RightHand])] == rig[Role::RightForeArm]);
    // The character's left is -x, forward is -z, up is +y.
    CHECK(s.rest_pos[size_t(rig[Role::LeftFoot])].x < 0.0f);
    CHECK(s.rest_pos[size_t(rig[Role::RightFoot])].x > 0.0f);
    CHECK(s.rest_pos[size_t(rig[Role::LeftToe])].z < s.rest_pos[size_t(rig[Role::LeftFoot])].z);
    CHECK(s.rest_pos[size_t(rig[Role::Head])].y > s.rest_pos[size_t(rig[Role::Hips])].y);
}

TEST_CASE("anim: mirror map pairs left and right joints", "[anim][skeleton]") {
    SkeletonDef s;
    REQUIRE(load_qchar_skeleton(kSoldier, s));
    const Rig rig = make_rig(s);
    for (int i = 0; i < s.size(); ++i) {
        const int m = rig.mirror[size_t(i)];
        CHECK(rig.mirror[size_t(m)] == i);           // an involution
        const glm::vec3 a = s.rest_pos[size_t(i)], b = s.rest_pos[size_t(m)];
        INFO(s.names[size_t(i)]);
        CHECK_THAT(a.x, WithinAbs(-b.x, 2e-3));      // the model is symmetric
        CHECK_THAT(a.y, WithinAbs(b.y, 2e-3));
        CHECK_THAT(a.z, WithinAbs(b.z, 2e-3));
    }
    CHECK(rig.mirror[size_t(rig[Role::LeftHand])] == rig[Role::RightHand]);
    CHECK(rig.mirror[size_t(rig[Role::Hips])] == rig[Role::Hips]);
}

TEST_CASE("anim: scaling to the player's eye height", "[anim][skeleton]") {
    SkeletonDef s;
    REQUIRE(load_qchar_skeleton(kSoldier, s));
    const float k = eye_height_scale(s, 1.6f);
    scale_skeleton(s, k);
    CHECK_THAT(s.scale, WithinAbs(k, 1e-6));
    CHECK_THAT(s.rest_pos[size_t(s.find("Head"))].y + 0.035f * k, WithinAbs(1.6f, 1e-4));
}

TEST_CASE("anim: rig id follows the rig", "[anim][skeleton]") {
    SkeletonDef a = fx::make_skeleton(), b = fx::make_skeleton();
    CHECK(a.rig_id() == b.rig_id());
    b.rest_pos[3].y += 0.01f;
    CHECK(a.rig_id() != b.rig_id());
    b = a;
    b.names[4] = "Other";
    CHECK(a.rig_id() != b.rig_id());
}

TEST_CASE("anim: invalid skeletons are refused", "[anim][skeleton]") {
    std::string why;
    SkeletonDef s = fx::make_skeleton();
    REQUIRE(s.validate());
    SkeletonDef bad = s;
    bad.parents[2] = 5;                               // parent after child
    CHECK_FALSE(bad.validate(&why));
    bad = s;
    bad.parents[3] = -1;                              // two roots
    CHECK_FALSE(bad.validate(&why));
    bad = s;
    bad.names[5] = bad.names[4];
    CHECK_FALSE(bad.validate(&why));
    bad = s;
    bad.rest_pos[1].x = std::nanf("");
    CHECK_FALSE(bad.validate(&why));
    SkeletonDef out;
    CHECK_FALSE(load_qchar_skeleton(QLIKE_SOURCE_DIR "/CMakeLists.txt", out, &why));
    CHECK(out.size() == 0);
}

TEST_CASE("anim: asset round trip and damaged files", "[anim][asset]") {
    const fx::TestAsset t = fx::make_asset();
    AnimAsset a = t.asset;
    a.motion_blob = { 1, 2, 3, 4, 5 };
    const std::vector<uint8_t> bytes = a.serialize();
    AnimAsset b;
    std::string why;
    REQUIRE(b.deserialize(bytes.data(), bytes.size(), &why));
    CHECK(b.skeleton.names == a.skeleton.names);
    CHECK(b.skeleton.parents == a.skeleton.parents);
    CHECK(b.skeleton.rig_id() == a.skeleton.rig_id());
    CHECK(b.runtime_format == a.runtime_format);
    CHECK(b.skeleton_blob == a.skeleton_blob);
    REQUIRE(b.clips.size() == a.clips.size());
    for (size_t i = 0; i < a.clips.size(); ++i) {
        CHECK(b.clips[i].name == a.clips[i].name);
        CHECK(b.clips[i].duration == a.clips[i].duration);
        CHECK(b.clips[i].loop == a.clips[i].loop);
        CHECK(b.clips[i].tags == a.clips[i].tags);
        CHECK(b.clips[i].provenance == a.clips[i].provenance);
        CHECK(b.clips[i].licence == a.clips[i].licence);
        CHECK(b.clip_blobs[i] == a.clip_blobs[i]);
    }
    CHECK(b.motion_blob == a.motion_blob);
    CHECK(b.find_clip("turn") == 2);
    CHECK(b.find_clip("nope") == -1);

    // Every truncation is refused, none crashes.
    for (size_t cut : { size_t(0), size_t(3), size_t(8), bytes.size() / 3, bytes.size() / 2, bytes.size() - 1 }) {
        AnimAsset c;
        CHECK_FALSE(c.deserialize(bytes.data(), cut, &why));
        CHECK(c.clips.empty());
    }
    std::vector<uint8_t> other = bytes;
    other[4] ^= 0x40;                                 // version
    CHECK_FALSE(b.deserialize(other.data(), other.size(), &why));
    other = bytes;
    other[0] = 'X';
    CHECK_FALSE(b.deserialize(other.data(), other.size(), &why));
}
