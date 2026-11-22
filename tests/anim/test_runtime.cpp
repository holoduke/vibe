#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "fixtures.h"

#include <thread>

using namespace vibe::anim;
using namespace vibe::anim::offline;
using Catch::Matchers::WithinAbs;

namespace {

struct Rt {
    fx::TestAsset t = fx::make_asset();
    std::unique_ptr<IAnimationRuntime> rt;
    std::unique_ptr<EvalContext> ctx;
    int n = 0;
    std::vector<Transform> local, a, b, scratch;
    std::vector<glm::mat4> model, skin;
    Rt() {
        std::string why;
        rt = create_runtime(t.asset, &why);
        if (!rt) throw std::runtime_error(why);
        ctx = rt->create_context(4);
        n = rt->skeleton().size();
        local.resize(size_t(n)); a.resize(size_t(n)); b.resize(size_t(n)); scratch.resize(size_t(n));
        model.resize(size_t(n)); skin.resize(size_t(n));
    }
    glm::vec3 pos(int j) const { return glm::vec3(model[size_t(j)][3]); }
};

}  // namespace

TEST_CASE("anim runtime: skeleton and clips come through unchanged", "[anim][runtime]") {
    Rt r;
    const SkeletonDef s = fx::make_skeleton();
    REQUIRE(r.rt->skeleton().size() == s.size());
    for (int i = 0; i < s.size(); ++i) {
        CHECK(r.rt->skeleton().names[size_t(i)] == s.names[size_t(i)]);
        CHECK(r.rt->skeleton().parents[size_t(i)] == s.parents[size_t(i)]);
    }
    CHECK(r.rt->rig().complete());
    CHECK(r.rt->num_clips() == 4);
    CHECK(r.rt->find_clip("walk") == 0);
    CHECK(r.rt->find_clip("missing") == kNoClip);
    CHECK_THAT(r.rt->clip(0).duration, WithinAbs(2.0f, 1e-5));
    CHECK(r.rt->clip(3).loop);
    CHECK(r.rt->clip_bytes() > 0);
}

TEST_CASE("anim runtime: bind pose parity", "[anim][runtime]") {
    Rt r;
    const SkeletonDef& s = r.rt->skeleton();
    r.rt->rest_pose(r.local);
    REQUIRE(pose_is_valid(r.local));
    for (int i = 0; i < r.n; ++i) {
        CHECK(glm::length(r.local[size_t(i)].t - s.rest_local(i)) < 1e-6f);
        CHECK(std::abs(r.local[size_t(i)].r.w) > 0.999999f);
    }
    r.rt->local_to_model(*r.ctx, r.local, r.model);
    r.rt->skinning_matrices(r.model, r.skin);
    for (int i = 0; i < r.n; ++i) {
        INFO(s.names[size_t(i)]);
        CHECK(glm::length(r.pos(i) - s.rest_pos[size_t(i)]) < 1e-5f);
        // A vertex skinned with the rest pose does not move.
        const glm::vec3 v = s.rest_pos[size_t(i)] + glm::vec3(0.03f, -0.02f, 0.05f);
        CHECK(glm::length(glm::vec3(r.skin[size_t(i)] * glm::vec4(v, 1.0f)) - v) < 1e-5f);
    }
}

TEST_CASE("anim runtime: sampling at the boundaries and in between", "[anim][runtime]") {
    Rt r;
    const RawClip& raw = r.t.raw[0];
    const ClipId walk = r.rt->find_clip("walk");
    const float dur = r.rt->clip(walk).duration;
    auto check_frame = [&](float time, int frame, float tol) {
        r.rt->sample(*r.ctx, 0, walk, time, r.local);
        REQUIRE(pose_is_valid(r.local));
        for (int j = 0; j < r.n; ++j) {
            INFO("t " << time << " joint " << r.rt->skeleton().names[size_t(j)]);
            CHECK(fx::joint_error(r.local[size_t(j)], raw.at(frame, j)) < tol);
        }
    };
    check_frame(0.0f, 0, 3e-3f);
    check_frame(dur, raw.frames - 1, 3e-3f);
    check_frame(-5.0f, 0, 3e-3f);                    // before the start: the first frame
    check_frame(dur + 9.0f, raw.frames - 1, 3e-3f);  // past the end: the last frame
    for (int f : { 1, 7, 30, 44, 59 }) check_frame(float(f) / 30.0f, f, 5e-3f);
    // Between two frames: between their poses.
    r.rt->sample(*r.ctx, 0, walk, 10.5f / 30.0f, r.local);
    const int lu = r.rt->skeleton().find("LeftUpLeg");
    const float e0 = fx::joint_error(r.local[size_t(lu)], raw.at(10, lu)), e1 = fx::joint_error(r.local[size_t(lu)], raw.at(11, lu));
    const float span = fx::joint_error(raw.at(10, lu), raw.at(11, lu));
    CHECK(e0 < span);
    CHECK(e1 < span);
    // Going backwards in time works too (the sampling cache is rebuilt).
    check_frame(3.0f / 30.0f, 3, 5e-3f);
}

TEST_CASE("anim runtime: a looping clip wraps", "[anim][runtime]") {
    Rt r;
    const ClipId cyc = r.rt->find_clip("cycle");
    const float dur = r.rt->clip(cyc).duration;
    r.rt->sample(*r.ctx, 0, cyc, 0.4f, r.a);
    r.rt->sample(*r.ctx, 0, cyc, 0.4f + dur, r.b);
    for (int j = 1; j < r.n; ++j) CHECK(fx::joint_error(r.a[size_t(j)], r.b[size_t(j)]) < 1e-4f);
    r.rt->sample(*r.ctx, 0, cyc, 0.4f - 3.0f * dur, r.b);
    for (int j = 1; j < r.n; ++j) CHECK(fx::joint_error(r.a[size_t(j)], r.b[size_t(j)]) < 1e-4f);
}

TEST_CASE("anim runtime: quaternions stay normalised, transforms finite", "[anim][runtime]") {
    Rt r;
    for (ClipId c = 0; c < r.rt->num_clips(); ++c) {
        for (int k = 0; k <= 200; ++k) {
            const float t = r.rt->clip(c).duration * (float(k) / 100.0f - 0.25f);
            r.rt->sample(*r.ctx, 0, c, t, r.local);
            REQUIRE(pose_is_valid(r.local));
            r.rt->local_to_model(*r.ctx, r.local, r.model);
            REQUIRE(pose_is_valid(r.model));
            for (int j = 0; j < r.n; ++j) {
                const glm::mat3 R(r.model[size_t(j)]);
                CHECK_THAT(glm::determinant(R), WithinAbs(1.0f, 1e-3));
            }
        }
    }
}

TEST_CASE("anim runtime: root motion extraction", "[anim][runtime]") {
    Rt r;
    const ClipId walk = r.rt->find_clip("walk"), turn = r.rt->find_clip("turn");
    // 1.5 m/s along -Z for two seconds.
    Transform d = r.rt->root_motion(*r.ctx, walk, 0.0f, 2.0f);
    CHECK_THAT(d.t.x, WithinAbs(0.0f, 0.01));
    CHECK_THAT(d.t.z, WithinAbs(-3.0f, 0.01));
    CHECK(std::abs(d.r.w) > 0.9999f);
    d = r.rt->root_motion(*r.ctx, walk, 0.5f, 1.0f);
    CHECK_THAT(d.t.z, WithinAbs(-0.75f, 0.01));
    d = r.rt->root_motion(*r.ctx, walk, 0.7f, 0.7f);
    CHECK(glm::length(d.t) < 1e-5f);
    // A turning clip: the motion is told in the root's own frame, so it is "forward" at any time.
    d = r.rt->root_motion(*r.ctx, turn, 1.0f, 1.1f);
    CHECK_THAT(d.t.z, WithinAbs(-0.1f, 0.01));
    CHECK(std::abs(d.t.x) < 0.02f);
    const float yaw = 2.0f * std::atan2(d.r.y, d.r.w);
    CHECK_THAT(yaw, WithinAbs(0.1f, 0.005));
    // Summing pieces gives the whole.
    float z = 0.0f;
    for (int k = 0; k < 20; ++k) z += r.rt->root_motion(*r.ctx, walk, float(k) * 0.1f, float(k + 1) * 0.1f).t.z;
    CHECK_THAT(z, WithinAbs(-3.0f, 0.02));
}

TEST_CASE("anim runtime: blending, masks and additive layers", "[anim][runtime]") {
    Rt r;
    const int lu = r.rt->skeleton().find("LeftUpLeg"), ru = r.rt->skeleton().find("RightUpLeg"), sp = r.rt->skeleton().find("Spine");
    r.rt->rest_pose(r.a);
    r.rt->rest_pose(r.b);
    r.b[size_t(lu)].r = glm::angleAxis(0.8f, glm::vec3(1, 0, 0));
    r.b[size_t(sp)].r = glm::angleAxis(0.4f, glm::vec3(0, 1, 0));
    auto angle = [](glm::quat q) { return 2.0f * std::acos(std::min(std::abs(q.w), 1.0f)); };

    PoseLayer layers[2] = { { r.a, 0.5f, {} }, { r.b, 0.5f, {} } };
    r.rt->blend(*r.ctx, layers, {}, r.local);
    REQUIRE(pose_is_valid(r.local));
    CHECK_THAT(angle(r.local[size_t(lu)].r), WithinAbs(0.4f, 0.01));
    CHECK_THAT(angle(r.local[size_t(sp)].r), WithinAbs(0.2f, 0.01));
    CHECK(angle(r.local[size_t(ru)].r) < 1e-3f);

    layers[0].weight = 0.25f; layers[1].weight = 0.75f;
    r.rt->blend(*r.ctx, layers, {}, r.local);
    CHECK_THAT(angle(r.local[size_t(lu)].r), WithinAbs(0.6f, 0.02));

    // A mask: layer b only counts for the spine.
    std::vector<float> mask(size_t(r.n), 0.0f);
    mask[size_t(sp)] = 1.0f;
    PoseLayer masked[2] = { { r.a, 1.0f, {} }, { r.b, 1.0f, mask } };
    r.rt->blend(*r.ctx, masked, {}, r.local);
    CHECK(angle(r.local[size_t(lu)].r) < 1e-3f);
    CHECK_THAT(angle(r.local[size_t(sp)].r), WithinAbs(0.2f, 0.01));      // equal weights on the spine

    // Additive: a difference pose on top of the base, scaled by its weight.
    std::vector<Transform> delta(size_t(r.n));
    delta[size_t(ru)].r = glm::angleAxis(0.6f, glm::vec3(1, 0, 0));
    delta[1].t = glm::vec3(0.0f, -0.1f, 0.0f);
    PoseLayer base[1] = { { r.b, 1.0f, {} } };
    PoseLayer add[1] = { { delta, 0.5f, {} } };
    r.rt->blend(*r.ctx, base, add, r.local);
    CHECK_THAT(angle(r.local[size_t(ru)].r), WithinAbs(0.3f, 0.01));
    CHECK_THAT(angle(r.local[size_t(lu)].r), WithinAbs(0.8f, 0.01));
    CHECK_THAT(r.local[1].t.y, WithinAbs(r.b[1].t.y - 0.05f, 1e-4));

    // No weight at all: the rest pose.
    PoseLayer none[1] = { { r.b, 0.0f, {} } };
    r.rt->blend(*r.ctx, none, {}, r.local);
    CHECK(angle(r.local[size_t(lu)].r) < 1e-3f);
}

TEST_CASE("anim runtime: partial local-to-model equals the full one", "[anim][runtime]") {
    Rt r;
    r.rt->sample(*r.ctx, 0, 0, 0.6f, r.local);
    r.rt->local_to_model(*r.ctx, r.local, r.model);
    const int lu = r.rt->skeleton().find("LeftUpLeg");
    r.local[size_t(lu)].r = glm::angleAxis(1.0f, glm::vec3(1, 0, 0));
    std::vector<glm::mat4> full(size_t(r.n));
    r.rt->local_to_model(*r.ctx, r.local, full);
    r.rt->local_to_model(*r.ctx, r.local, r.model, lu);
    for (int j = 0; j < r.n; ++j)
        for (int c = 0; c < 4; ++c) CHECK(glm::length(full[size_t(j)][c] - r.model[size_t(j)][c]) < 1e-5f);
}

TEST_CASE("anim runtime: two-bone IK", "[anim][runtime]") {
    Rt r;
    const Rig& rig = r.rt->rig();
    r.rt->rest_pose(r.local);
    r.rt->local_to_model(*r.ctx, r.local, r.model);
    TwoBoneIk ik;
    ik.start = rig[Role::LeftUpLeg]; ik.mid = rig[Role::LeftLeg]; ik.end = rig[Role::LeftFoot];
    ik.pole = glm::vec3(0.0f, 0.0f, -1.0f);
    ik.soften = 1.0f;
    const glm::vec3 rest_foot = r.pos(ik.end), hip = r.pos(ik.start);
    for (glm::vec3 off : { glm::vec3(0.0f, 0.25f, -0.15f), glm::vec3(0.1f, 0.15f, 0.2f), glm::vec3(-0.05f, 0.4f, -0.3f) }) {
        r.rt->rest_pose(r.local);
        r.rt->local_to_model(*r.ctx, r.local, r.model);
        ik.target = rest_foot + off;
        CHECK(r.rt->two_bone_ik(*r.ctx, ik, r.local, r.model));
        REQUIRE(pose_is_valid(r.local));
        CHECK(glm::length(r.pos(ik.end) - ik.target) < 2e-3f);
        CHECK(glm::length(r.pos(ik.start) - hip) < 1e-5f);             // the hip did not move
        // The knee bends forward (toward the pole), not backward.
        const glm::vec3 knee = r.pos(ik.mid), mid = (hip + ik.target) * 0.5f;
        CHECK(knee.z < mid.z + 1e-3f);
        // Bone lengths are kept.
        CHECK_THAT(glm::length(knee - hip), WithinAbs(glm::length(r.rt->skeleton().rest_local(ik.mid)), 1e-4));
    }
    // Out of reach: says so, stretches toward it, stays finite.
    r.rt->rest_pose(r.local);
    r.rt->local_to_model(*r.ctx, r.local, r.model);
    ik.target = hip + glm::vec3(0.0f, -3.0f, -1.0f);
    CHECK_FALSE(r.rt->two_bone_ik(*r.ctx, ik, r.local, r.model));
    REQUIRE(pose_is_valid(r.local));
    REQUIRE(pose_is_valid(r.model));
    const glm::vec3 dir = glm::normalize(r.pos(ik.end) - hip);
    CHECK(glm::dot(dir, glm::normalize(ik.target - hip)) > 0.999f);
    // weight 0 leaves the pose alone
    r.rt->rest_pose(r.local);
    r.rt->local_to_model(*r.ctx, r.local, r.model);
    ik.weight = 0.0f;
    ik.target = rest_foot + glm::vec3(0.0f, 0.3f, 0.0f);
    r.rt->two_bone_ik(*r.ctx, ik, r.local, r.model);
    CHECK(glm::length(r.pos(ik.end) - rest_foot) < 1e-4f);
}

TEST_CASE("anim runtime: aim IK points a joint at a target", "[anim][runtime]") {
    Rt r;
    const int head = r.rt->rig()[Role::Head];
    r.rt->rest_pose(r.local);
    r.rt->local_to_model(*r.ctx, r.local, r.model);
    AimIk ik;
    ik.joint = head;
    ik.target = r.pos(head) + glm::vec3(2.0f, 1.0f, -2.0f);
    r.rt->aim_ik(*r.ctx, ik, r.local, r.model);
    REQUIRE(pose_is_valid(r.local));
    const glm::vec3 f = glm::mat3(r.model[size_t(head)]) * glm::vec3(0, 0, -1);
    CHECK(glm::dot(glm::normalize(f), glm::normalize(ik.target - r.pos(head))) > 0.999f);
}

TEST_CASE("anim runtime: mirroring a pose", "[anim][runtime]") {
    Rt r;
    const Rig& rig = r.rt->rig();
    r.rt->sample(*r.ctx, 0, 0, 0.25f, r.a);
    r.a[0] = Transform{};                              // (the root's travel is not part of the pose here)
    r.b = r.a;
    mirror_pose(rig, r.b, r.scratch);
    REQUIRE(pose_is_valid(r.b));
    std::vector<glm::mat4> ma(size_t(r.n)), mb(size_t(r.n));
    r.rt->local_to_model(*r.ctx, r.a, ma);
    r.rt->local_to_model(*r.ctx, r.b, mb);
    for (int j = 0; j < r.n; ++j) {
        const glm::vec3 p = glm::vec3(ma[size_t(j)][3]), q = glm::vec3(mb[size_t(rig.mirror[size_t(j)])][3]);
        INFO(r.rt->skeleton().names[size_t(j)]);
        CHECK(glm::length(glm::vec3(-p.x, p.y, p.z) - q) < 2e-3f);
    }
    mirror_pose(rig, r.b, r.scratch);                 // twice: back where it was
    for (int j = 0; j < r.n; ++j) CHECK(fx::joint_error(r.a[size_t(j)], r.b[size_t(j)]) < 1e-5f);
}

TEST_CASE("anim runtime: skinning follows a rotated joint", "[anim][runtime]") {
    Rt r;
    const Rig& rig = r.rt->rig();
    const SkeletonDef& s = r.rt->skeleton();
    r.rt->rest_pose(r.local);
    r.local[size_t(rig[Role::LeftArm])].r = glm::angleAxis(1.5707963f, glm::vec3(0, 0, 1));
    r.rt->local_to_model(*r.ctx, r.local, r.model);
    r.rt->skinning_matrices(r.model, r.skin);
    const glm::vec3 joint = s.rest_pos[size_t(rig[Role::LeftArm])];
    const glm::vec3 v = joint + glm::vec3(-0.2f, 0.0f, 0.0f);
    const glm::vec3 out = glm::vec3(r.skin[size_t(rig[Role::LeftArm])] * glm::vec4(v, 1.0f));
    CHECK(glm::length(out - (joint + glm::vec3(0.0f, -0.2f, 0.0f))) < 1e-5f);
    // Joints above it are untouched.
    const glm::vec3 w = s.rest_pos[size_t(rig[Role::Spine])] + glm::vec3(0.1f);
    CHECK(glm::length(glm::vec3(r.skin[size_t(rig[Role::Spine])] * glm::vec4(w, 1.0f)) - w) < 1e-5f);
}

TEST_CASE("anim runtime: assets for another runtime or damaged blocks are refused", "[anim][runtime]") {
    fx::TestAsset t = fx::make_asset();
    std::string why;
    AnimAsset a = t.asset;
    a.runtime_format = "something else";
    CHECK(create_runtime(a, &why) == nullptr);
    CHECK(!why.empty());
    a = t.asset;
    a.skeleton_blob.assign(a.skeleton_blob.size(), 0);
    CHECK(create_runtime(a, &why) == nullptr);
    a = t.asset;
    a.clip_blobs[1].clear();
    CHECK(create_runtime(a, &why) == nullptr);
    a = t.asset;
    a.skeleton.names[3] = "Renamed";                  // the asset's rig no longer matches the block
    CHECK(create_runtime(a, &why) == nullptr);
}

TEST_CASE("anim runtime: several threads, each with its own context, agree", "[anim][runtime]") {
    Rt r;
    r.rt->sample(*r.ctx, 0, 0, 0.9f, r.a);
    std::vector<std::vector<Transform>> out(4, std::vector<Transform>(size_t(r.n)));
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
        th.emplace_back([&, k] {
            auto ctx = r.rt->create_context(1);
            std::vector<Transform> tmp(size_t(r.n));
            for (int i = 0; i < 300; ++i) r.rt->sample(*ctx, 0, i % 4, float(i) * 0.01f, tmp);
            r.rt->sample(*ctx, 0, 0, 0.9f, out[size_t(k)]);
        });
    for (auto& t : th) t.join();
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < r.n; ++j) CHECK(fx::joint_error(out[size_t(k)][size_t(j)], r.a[size_t(j)]) < 1e-6f);
}
