#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "anim/inertializer.h"
#include "anim/motion_matcher.h"
#include "fixtures.h"
#include "game/player.h"

#include <chrono>
#include <random>

using namespace vibe;
using namespace vibe::anim;
using namespace vibe::anim::offline;
using Catch::Matchers::WithinAbs;

namespace {

MotionQuery path_query(glm::vec2 vel, float pose_weight = 0.0f) {
    MotionQuery q;
    for (int h = 0; h < MotionSchema::kHorizons; ++h) {
        const float t = float(h + 1) / 3.0f;
        q.raw[15 + 2 * h] = vel.x * t;
        q.raw[16 + 2 * h] = vel.y * t;
        q.raw[21 + 2 * h] = 0.0f;
        q.raw[22 + 2 * h] = -1.0f;
    }
    q.group_weight[kFootPos] = q.group_weight[kFootVel] = q.group_weight[kHipVel] = pose_weight;
    return q;
}

MatcherInput move_input(glm::vec2 vel_local, const MotionDatabase& db, uint32_t tags = kTagLocomotion) {
    // a steady request: moving at that velocity and staying on it (a straight path)
    MatcherInput in;
    for (int h = 0; h < MotionSchema::kHorizons; ++h) {
        in.traj.pos[h] = vel_local * db.horizon(h);
        in.traj.dir[h] = glm::vec2(0.0f, -1.0f);
    }
    in.traj.vel_end = vel_local;
    in.velocity = vel_local;
    in.require_tags = tags;
    return in;
}

}  // namespace

TEST_CASE("anim motion: features are what the clips do", "[anim][motion]") {
    fx::MotionFixture m;
    MotionDatabaseBuilder b;
    REQUIRE(build_motion_features(m.skel, m.clips, MotionBuildParams{}, b));
    size_t total = 0;
    for (const RawClip& c : m.clips) total += size_t(c.frames);
    CHECK(b.info.size() == total);
    CHECK(b.raw.size() == total * MotionSchema::kFeat);
    CHECK(m.db.frames() == total);
    // ranges tile the frames, in clip order, with their tags
    uint32_t at = 0;
    for (size_t i = 0; i < m.db.ranges().size(); ++i) {
        const MotionRange& r = m.db.ranges()[i];
        CHECK(r.first == at);
        CHECK(r.clip == int(i));
        CHECK(r.tags == m.clips[i].info.tags);
        CHECK(r.tail == 31u);
        at += r.count;
    }
    // run_f: 2 m/s straight ahead
    const MotionRange& run = m.db.ranges()[size_t(m.range_named("run_f"))];
    float raw[MotionSchema::kFeat];
    m.db.raw_features(run.first + 100, false, raw);
    CHECK_THAT(raw[15 + 4], WithinAbs(0.0f, 1e-3));          // x one second ahead
    CHECK_THAT(raw[16 + 4], WithinAbs(-2.0f, 1e-3));         // z one second ahead
    CHECK_THAT(raw[16], WithinAbs(-2.0f / 3.0f, 1e-3));
    CHECK_THAT(raw[21], WithinAbs(0.0f, 1e-4));
    CHECK_THAT(raw[22], WithinAbs(-1.0f, 1e-4));
    CHECK_THAT(m.db.info(run.first + 100).root_vel_z, WithinAbs(-2.0f, 1e-3));
    CHECK_THAT(m.db.info(run.first + 100).root_vel_x, WithinAbs(0.0f, 1e-3));
    CHECK_THAT(raw[14], WithinAbs(-2.0f, 0.05));             // the hips travel with the root
    CHECK(raw[0] < 0.0f);                                     // left foot on the left
    CHECK(raw[3] > 0.0f);
    // strafe: the path goes right, the facing stays ahead
    const MotionRange& st = m.db.ranges()[size_t(m.range_named("strafe_r"))];
    m.db.raw_features(st.first + 50, false, raw);
    CHECK_THAT(raw[15 + 4], WithinAbs(1.2f, 1e-3));
    CHECK_THAT(raw[16 + 4], WithinAbs(0.0f, 1e-3));
    CHECK_THAT(raw[22 + 4], WithinAbs(-1.0f, 1e-4));
    // mirrored: the same, to the left, feet swapped
    float mr[MotionSchema::kFeat];
    m.db.raw_features(st.first + 50, true, mr);
    CHECK_THAT(mr[15 + 4], WithinAbs(-1.2f, 1e-3));
    CHECK_THAT(mr[0], WithinAbs(-raw[3], 1e-5));
    CHECK_THAT(mr[1], WithinAbs(raw[4], 1e-5));
    CHECK_THAT(mr[9], WithinAbs(-raw[6], 1e-5));
    // time within the clip
    CHECK_THAT(m.db.time_in_clip(st.first + 45, 0.5f), WithinAbs(45.5f / 30.0f, 1e-5));
}

TEST_CASE("anim motion: yaw rate has the sign of the player's yaw", "[anim][motion]") {
    const SkeletonDef s = fx::make_skeleton();
    // fx::make_clip turns about +Y: counter-clockwise from above = to the left = negative yaw rate
    std::vector<RawClip> clips = { fx::make_clip(s, "turn_left", 120, 30.0f, 1.0f, 1.0f) };
    MotionDatabaseBuilder b;
    REQUIRE(build_motion_features(s, clips, MotionBuildParams{}, b));
    CHECK_THAT(b.info[40].yaw_rate, WithinAbs(-1.0f, 0.01));
    // and the facing one second ahead has turned left: x negative
    CHECK(b.raw[size_t(40) * MotionSchema::kFeat + 21 + 4] < -0.5f);
}

TEST_CASE("anim motion: flights are found in jump clips", "[anim][motion]") {
    fx::MotionFixture m;
    REQUIRE(m.db.jumps().size() == 1);
    const MotionJump& j = m.db.jumps()[0];
    const MotionRange& r = m.db.ranges()[size_t(m.range_named("hop"))];
    CHECK(j.takeoff >= r.first + 48);
    CHECK(j.takeoff <= r.first + 52);
    CHECK(j.land >= r.first + 98);
    CHECK(j.land <= r.first + 102);
    CHECK(j.apex > j.takeoff);
    CHECK(j.apex < j.land);
    CHECK(j.rise > 0.4f);
    CHECK_THAT(j.vel_z, WithinAbs(-0.2f, 0.01));
    CHECK((m.db.info(j.takeoff).contact) == 0);
    CHECK((m.db.info(j.land).contact) != 0);
}

TEST_CASE("anim motion: database round trip and damaged data", "[anim][motion]") {
    fx::MotionFixture m;
    MotionDatabase db;
    std::string why;
    REQUIRE(db.load(m.blob.data(), m.blob.size(), m.skel.rig_id(), &why));
    CHECK(db.frames() == m.db.frames());
    CHECK(db.sample_rate() == 30.0f);
    CHECK_THAT(db.horizon(2), WithinAbs(1.0f, 1e-6));
    CHECK(db.memory_bytes() > 0);
    MotionDatabaseBuilder b;
    REQUIRE(build_motion_features(m.skel, m.clips, MotionBuildParams{}, b));
    float raw[MotionSchema::kFeat];
    for (uint32_t f : { 0u, 77u, 500u, db.frames() - 1 }) {
        db.raw_features(f, false, raw);
        for (int k = 0; k < MotionSchema::kFeat; ++k) CHECK_THAT(raw[k], WithinAbs(b.raw[size_t(f) * MotionSchema::kFeat + size_t(k)], 2e-4));
    }
    CHECK_FALSE(db.load(m.blob.data(), m.blob.size(), m.skel.rig_id() + 1, &why));      // another rig
    CHECK_FALSE(db.ok());
    for (size_t cut : { size_t(0), size_t(6), size_t(100), m.blob.size() / 2, m.blob.size() - 4 })
        CHECK_FALSE(db.load(m.blob.data(), cut, m.skel.rig_id(), &why));
    std::vector<uint8_t> bad = m.blob;
    bad[4] ^= 0x10;                                           // version
    CHECK_FALSE(db.load(bad.data(), bad.size(), m.skel.rig_id(), &why));
    bad = m.blob;
    const float nan = std::nanf("");
    std::memcpy(&bad[bad.size() - 64], &nan, 4);              // inside the feature array
    CHECK_FALSE(db.load(bad.data(), bad.size(), m.skel.rig_id(), &why));
}

TEST_CASE("anim motion: search finds the clip that goes where asked", "[anim][motion]") {
    fx::MotionFixture m;
    struct Want { glm::vec2 vel; const char* clip; bool mirrored; };
    const Want wants[] = {
        { { 0.0f, 0.0f }, "idle", false }, { { 0.0f, -1.0f }, "walk_f", false }, { { 0.0f, -2.0f }, "run_f", false },
        { { 0.0f, 0.8f }, "walk_b", false }, { { 1.2f, 0.0f }, "strafe_r", false }, { { -1.2f, 0.0f }, "strafe_r", true },
    };
    for (const Want& w : wants) {
        const MotionMatch r = m.db.search(path_query(w.vel));
        INFO("velocity " << w.vel.x << " " << w.vel.y);
        REQUIRE(r.frame >= 0);
        CHECK(std::string(m.clip_of(r.frame)) == w.clip);
        if (w.vel.x != 0.0f) CHECK(r.mirrored == w.mirrored);
        CHECK(r.cost < 0.05f);
        // never inside a clip's exclusion zones
        const MotionRange& rg = m.db.range_of(uint32_t(r.frame));
        CHECK(uint32_t(r.frame) >= rg.first + rg.head);
        CHECK(uint32_t(r.frame) < rg.first + rg.count - rg.tail);
    }
    // tags: crouched requests only see crouched clips; jump clips are never locomotion
    MotionQuery q = path_query({ 0.0f, -2.0f });
    q.require_tags = kTagCrouch;
    CHECK(std::string(m.clip_of(m.db.search(q).frame)) == "crouch_f");
    q.require_tags = kTagLocomotion;
    q.exclude_tags = kTagIdle;
    CHECK(std::string(m.clip_of(m.db.search(path_query({ 0.0f, 0.0f })).frame)) == "idle");
    q = path_query({ 0.0f, 0.0f });
    q.exclude_tags = kTagIdle;
    CHECK(std::string(m.clip_of(m.db.search(q).frame)) != "idle");
    q.require_tags = 1u << 20;                                // nothing has it
    CHECK(m.db.search(q).frame == -1);
}

TEST_CASE("anim motion: the fast search equals the reference, and repeats", "[anim][motion]") {
    fx::MotionFixture m;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-2.5f, 2.5f), w(0.0f, 2.0f);
    for (int it = 0; it < 60; ++it) {
        MotionQuery q;
        for (float& v : q.raw) v = u(rng);
        for (float& g : q.group_weight) g = w(rng);
        const MotionMatch a = m.db.search(q), b = m.db.search_reference(q), again = m.db.search(q);
        REQUIRE(a.frame >= 0);
        CHECK(a.frame == again.frame);
        CHECK(a.mirrored == again.mirrored);
        CHECK(a.cost == again.cost);
        CHECK_THAT(a.cost, WithinAbs(b.cost, 1e-3f * std::max(1.0f, b.cost)));
        float sum = 0.0f;
        for (float g : a.group_cost) sum += g;
        CHECK_THAT(sum, WithinAbs(a.cost, 1e-3f * std::max(1.0f, a.cost)));
        // cost() of the found frame is the found cost
        CHECK_THAT(m.db.cost(q, uint32_t(a.frame), a.mirrored).cost, WithinAbs(a.cost, 1e-4f * std::max(1.0f, a.cost)));
    }
}

TEST_CASE("anim motion: extreme queries stay finite", "[anim][motion]") {
    fx::MotionFixture m;
    for (float big : { 1e3f, 1e6f, -1e6f, 0.0f }) {
        MotionQuery q;
        for (float& v : q.raw) v = big;
        const MotionMatch r = m.db.search(q);
        CHECK(r.frame >= 0);
        CHECK(std::isfinite(r.cost));
    }
}

TEST_CASE("anim motion: query time is bounded", "[anim][motion]") {
    // 120,000 frames (more than the shipped database), every one searchable, both mirror halves.
    const SkeletonDef s = fx::make_skeleton();
    MotionDatabaseBuilder b;
    b.rig_id = s.rig_id();
    const uint32_t n = 120000;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    b.raw.resize(size_t(n) * MotionSchema::kFeat);
    for (float& v : b.raw) v = u(rng);
    b.info.resize(n);
    b.ranges.push_back({ 0, 0, n, kTagLocomotion, 2, 31 });
    const std::vector<uint8_t> blob = b.serialize();
    MotionDatabase db;
    REQUIRE(db.load(blob.data(), blob.size(), s.rig_id()));
    MotionQuery q;
    for (float& v : q.raw) v = u(rng);
    (void)db.search(q);                                       // warm the cache
    const auto t0 = std::chrono::steady_clock::now();
    const int runs = 20;
    int sink = 0;
    for (int i = 0; i < runs; ++i) { q.raw[i % MotionSchema::kFeat] += 0.01f; sink += db.search(q).frame; }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;
    INFO("search over " << n << " frames x 2: " << ms << " ms");
    CHECK(sink >= 0);
    CHECK(ms < 25.0);                                         // generous: unoptimised builds run this too
}

TEST_CASE("anim trajectory: follows the controller's movement model", "[anim][trajectory]") {
    const float hz[3] = { 1.0f / 3.0f, 2.0f / 3.0f, 1.0f };
    const physics::MoveParams& mv = game::kPlayerMove;
    // standing, no input
    TrajectoryInput in;
    Trajectory t = predict_trajectory(in, mv, hz);
    for (int h = 0; h < 3; ++h) { CHECK(glm::length(t.pos[h]) < 1e-5f); CHECK_THAT(t.dir[h].y, WithinAbs(-1.0f, 1e-6)); }
    // from rest, holding forward: accelerates toward the walk speed, never beyond it
    in.desired_velocity = glm::vec2(0.0f, -mv.max_walk_speed);
    t = predict_trajectory(in, mv, hz);
    CHECK(t.pos[0].y < 0.0f);
    CHECK(t.pos[1].y < t.pos[0].y);
    CHECK(t.pos[2].y < t.pos[1].y);
    CHECK(-t.pos[2].y < mv.max_walk_speed * 1.0f);
    CHECK(-t.pos[2].y > 0.5f * mv.max_walk_speed);            // Quake acceleration is quick
    CHECK(glm::length(t.vel_end) <= mv.max_walk_speed + 1e-3f);
    // the same integration as the game: step the controller's own functions for one second
    {
        glm::vec3 v(0.0f), p(0.0f);
        const glm::vec3 wd(0.0f, 0.0f, -1.0f);
        for (int k = 0; k < 60; ++k) {
            v = physics::apply_friction(v, mv.ground_friction, mv.stop_speed, 1.0f / 60.0f, 0.05f);
            v = physics::accelerate(v, wd, mv.max_walk_speed, mv.ground_accel, 1.0f / 60.0f);
            p += v / 60.0f;
        }
        CHECK_THAT(t.pos[2].y, WithinAbs(p.z, 1e-3));
    }
    // running, input released: friction stops the character well within the second
    in = TrajectoryInput{};
    in.velocity = glm::vec2(0.0f, -8.0f);
    t = predict_trajectory(in, mv, hz);
    CHECK(-t.pos[2].y < 3.0f);
    CHECK(glm::length(t.vel_end) < 0.1f);
    // sprinting forward, input reversed: the path turns round inside the horizon
    in.desired_velocity = glm::vec2(0.0f, 8.0f);
    t = predict_trajectory(in, mv, hz);
    CHECK(t.vel_end.y > 6.0f);
    CHECK(t.pos[2].y > t.pos[0].y);
    // strafing right is +x in the character's frame, whatever the yaw
    for (float yaw : { 0.0f, 1.0f, -2.5f }) {
        in = TrajectoryInput{};
        in.facing_yaw = yaw;
        in.velocity = in.desired_velocity = facing_to_world(glm::vec2(6.0f, 0.0f), yaw);
        t = predict_trajectory(in, mv, hz);
        CHECK(t.pos[2].x > 4.0f);
        CHECK(std::abs(t.pos[2].y) < 0.2f);
        const glm::vec2 back = world_to_facing(in.velocity, yaw);
        CHECK_THAT(back.x, WithinAbs(6.0f, 1e-4));
        CHECK_THAT(back.y, WithinAbs(0.0f, 1e-4));
    }
    // in the air there is no friction and (almost) no acceleration: the path is a straight line
    in = TrajectoryInput{};
    in.grounded = false;
    in.velocity = glm::vec2(0.0f, -10.0f);
    in.desired_velocity = glm::vec2(8.0f, 0.0f);
    t = predict_trajectory(in, mv, hz);
    CHECK_THAT(t.pos[2].y, WithinAbs(-10.0f, 0.01));
    CHECK(t.pos[2].x > 0.0f);
    CHECK(t.pos[2].x < mv.air_wish_cap * 1.0f + 0.01f);       // air steering as the game defines it
    // turning right: the facing ahead turns to +x
    in = TrajectoryInput{};
    in.yaw_rate = 2.0f;
    t = predict_trajectory(in, mv, hz);
    CHECK(t.dir[2].x > 0.2f);
    CHECK_THAT(glm::length(t.dir[2]), WithinAbs(1.0f, 1e-5));
    in.yaw_rate = -2.0f;
    CHECK(predict_trajectory(in, mv, hz).dir[2].x < -0.2f);
}

TEST_CASE("anim matcher: speed mapping into the captured range", "[anim][matcher]") {
    MatcherParams p;
    CHECK(capture_speed(0.0f, p) == 0.0f);
    CHECK(capture_speed(0.6f, p) == 0.6f);
    float prev = 0.0f;
    for (float s = 0.0f; s < 40.0f; s += 0.25f) {
        const float c = capture_speed(s, p);
        CHECK(c >= prev);
        CHECK(c <= p.run_speed + 1e-5f);
        prev = c;
    }
    CHECK(capture_speed(12.0f, p) > 1.9f);
}

TEST_CASE("anim matcher: steady requests settle, changed requests switch quickly", "[anim][matcher]") {
    fx::MotionFixture m;
    MatcherParams p;
    MatcherState st;
    const float dt = 1.0f / 60.0f;
    auto run = [&](glm::vec2 vel, float seconds, uint32_t tags = kTagLocomotion) {
        MatcherOutput o;
        const MatcherInput in = move_input(vel, m.db, tags);
        for (float t = 0.0f; t < seconds; t += dt) {
            o = update_matcher(m.db, st, in, p, dt);
            REQUIRE(o.clip >= 0);
            const MotionRange& r = m.db.range_of(uint32_t(st.frame));
            REQUIRE(uint32_t(st.frame) + 1 < r.first + r.count);       // frame + 1 is always inside the clip
            REQUIRE(o.time >= 0.0f);
            REQUIRE(o.time <= m.clips[size_t(o.clip)].info.duration + 1e-4f);
        }
        return o;
    };
    // walking forward at the capture's own speed
    MatcherOutput o = run({ 0.0f, -1.0f }, 0.5f);
    CHECK(std::string(m.clips[size_t(o.clip)].info.name) == "walk_f");
    CHECK_THAT(o.root_vel.y, WithinAbs(-1.0f, 0.01));
    const uint32_t before = st.switches;
    run({ 0.0f, -1.0f }, 4.0f);
    CHECK(st.switches - before <= 1);                         // no thrashing while nothing changes
    CHECK(st.searches > 20);
    // the game's run speed maps into the captured run
    o = run({ 0.0f, -8.0f }, 0.4f);
    CHECK(std::string(m.clips[size_t(o.clip)].info.name) == "run_f");
    CHECK(o.speed_scale < 0.3f);
    // strafe left: the right-strafe capture, mirrored, within a few frames
    float t_switch = 0.0f;
    {
        const MatcherInput in = move_input({ -6.0f, 0.0f }, m.db);
        for (; t_switch < 1.0f; t_switch += dt) {
            o = update_matcher(m.db, st, in, p, dt);
            if (std::string(m.clips[size_t(o.clip)].info.name) == "strafe_r") break;
        }
    }
    CHECK(t_switch <= 0.12f);
    CHECK(o.mirrored);
    CHECK(o.root_vel.x < -1.0f);                              // the mirrored clip moves left
    // stop: the idle
    o = run({ 0.0f, 0.0f }, 0.5f);
    CHECK(std::string(m.clips[size_t(o.clip)].info.name) == "idle");
    // crouch: only crouched data; and back
    o = run({ 0.0f, -3.0f }, 0.3f, kTagCrouch);
    CHECK(std::string(m.clips[size_t(o.clip)].info.name) == "crouch_f");
    o = run({ 0.0f, -3.0f }, 0.3f);
    CHECK(std::string(m.clips[size_t(o.clip)].info.name) != "crouch_f");
}

TEST_CASE("anim matcher: rapid alternating strafes do not thrash or break", "[anim][matcher]") {
    fx::MotionFixture m;
    MatcherParams p;
    for (float hz : { 5.0f, 8.0f, 10.0f }) {
        MatcherState st;
        const float dt = 1.0f / 120.0f, seconds = 3.0f;
        int steps = 0;
        for (float t = 0.0f; t < seconds; t += dt, ++steps) {
            const bool right = std::fmod(t * hz, 1.0f) < 0.5f;
            const MatcherOutput o = update_matcher(m.db, st, move_input({ right ? 7.0f : -7.0f, 0.0f }, m.db), p, dt);
            REQUIRE(o.clip >= 0);
            REQUIRE(std::isfinite(o.time));
        }
        INFO(hz << " Hz: " << st.switches << " switches");
        // bounded by the minimum switch interval, whatever the input does
        CHECK(float(st.switches) <= seconds / p.min_switch_interval + 1.0f);
    }
}

TEST_CASE("anim matcher: the same inputs give the same result at any call pattern", "[anim][matcher]") {
    fx::MotionFixture m;
    MatcherParams p;
    MatcherState a, b;
    for (int i = 0; i < 600; ++i) {
        const glm::vec2 v = (i / 90) % 2 ? glm::vec2(0.0f, -6.0f) : glm::vec2(5.0f, 1.0f);
        const MatcherOutput oa = update_matcher(m.db, a, move_input(v, m.db), p, 1.0f / 60.0f);
        const MatcherOutput ob = update_matcher(m.db, b, move_input(v, m.db), p, 1.0f / 60.0f);
        REQUIRE(oa.clip == ob.clip);
        REQUIRE(oa.time == ob.time);
        REQUIRE(oa.mirrored == ob.mirrored);
    }
    // a clip's end is never run past, even with huge steps
    MatcherState c;
    for (int i = 0; i < 200; ++i) {
        const MatcherOutput o = update_matcher(m.db, c, move_input({ 0.0f, -2.0f }, m.db), p, 0.25f);
        const MotionRange& r = m.db.range_of(uint32_t(c.frame));
        REQUIRE(uint32_t(c.frame) + 1 < r.first + r.count);
        REQUIRE(o.clip >= 0);
    }
}

TEST_CASE("anim inertializer: a transition is continuous in pose and in rate", "[anim][inertializer]") {
    const int n = 6;
    std::vector<Transform> a(n), b(n), bn(n), out(n), prev(n);
    Inertializer in;
    in.reset(n);
    const float dt = 1.0f / 60.0f, half = 0.1f;
    // source: joint 2 swings at 2 rad/s about X, joint 1 (hips) drifts up at 0.3 m/s
    auto src = [&](float t) {
        for (auto& x : a) x = Transform{};
        a[2].r = glm::angleAxis(2.0f * t, glm::vec3(1, 0, 0));
        a[1].t = glm::vec3(0.0f, 1.0f + 0.3f * t, 0.0f);
    };
    // destination: a different pose, standing still
    for (auto& x : b) x = Transform{};
    b[2].r = glm::angleAxis(1.2f, glm::vec3(0, 0, 1));
    b[1].t = glm::vec3(0.1f, 0.9f, 0.0f);
    bn = b;
    float t = 0.0f;
    for (int i = 0; i < 30; ++i, t += dt) { src(t); prev = out; in.apply(a, dt, half, out); }
    CHECK_FALSE(in.active());
    const std::vector<Transform> shown = out;
    const glm::vec3 w_before = quat_to_scaled_axis(out[2].r * glm::inverse(prev[2].r)) / dt;
    const float v_before = (out[1].t.y - prev[1].t.y) / dt;
    in.transition(b, bn, 1.0f / 30.0f);
    // an instant later (dt -> 0) the pose is what was on screen
    in.apply(b, 0.0f, half, out);
    for (int j = 1; j < n; ++j) CHECK(fx::joint_error(out[size_t(j)], shown[size_t(j)]) < 1e-4f);
    CHECK(in.active());
    // a moment later it is still moving the way it moved (the rate is carried over): no kink
    prev = out;
    const float h = 5e-4f;
    in.apply(b, h, half, out);
    const glm::vec3 w_after = quat_to_scaled_axis(out[2].r * glm::inverse(prev[2].r)) / h;
    CHECK(glm::length(w_after - w_before) < 0.15f * glm::length(w_before));
    CHECK_THAT((out[1].t.y - prev[1].t.y) / h, WithinAbs(v_before, 0.15f * v_before));
    // and it arrives: after a second the output is the destination
    float worst_step = 0.0f;
    for (int i = 0; i < 100; ++i) {
        prev = out;
        in.apply(b, dt, half, out);
        REQUIRE(pose_is_valid(out));
        for (int j = 1; j < n; ++j) worst_step = std::max(worst_step, fx::joint_error(out[size_t(j)], prev[size_t(j)]));
    }
    for (int j = 1; j < n; ++j) CHECK(fx::joint_error(out[size_t(j)], b[size_t(j)]) < 2e-3f);
    CHECK(worst_step < 0.15f);                                // no pop on the way (1.3 rad + 0.1 m to cover: a few degrees per frame)
    CHECK_FALSE(in.active());
}

TEST_CASE("anim inertializer: quaternion double cover and odd steps are harmless", "[anim][inertializer]") {
    const int n = 4;
    std::vector<Transform> a(n), b(n), out(n);
    Inertializer in;
    in.reset(n);
    a[2].r = glm::angleAxis(0.3f, glm::vec3(0, 1, 0));
    for (int i = 0; i < 3; ++i) in.apply(a, 1.0f / 60.0f, 0.1f, out);
    // the same rotation with the opposite sign: no offset at all may appear
    b = a;
    b[2].r = -a[2].r;
    in.transition(b, b, 1.0f / 30.0f);
    in.apply(b, 0.0f, 0.1f, out);
    CHECK(in.energy() < 1e-4f);
    // a transition before anything was shown, zero and huge time steps, a tiny half-life
    Inertializer fresh;
    fresh.reset(n);
    fresh.transition(b, b, 0.0f);
    fresh.apply(b, 0.0f, 0.1f, out);
    REQUIRE(pose_is_valid(out));
    b[1].r = glm::angleAxis(3.1f, glm::vec3(1, 0, 0));       // almost half a turn away
    fresh.transition(b, b, 0.0f);
    for (float dt : { 0.0f, 1e-6f, 5.0f, 0.016f }) {
        fresh.apply(b, dt, 1e-6f, out);
        REQUIRE(pose_is_valid(out));
    }
    CHECK(fx::joint_error(out[1], b[1]) < 1e-3f);
    // the root is passed through untouched
    b[0].t = glm::vec3(5.0f, 0.0f, -3.0f);
    fresh.transition(a, a, 0.0f);
    fresh.apply(b, 0.016f, 0.1f, out);
    CHECK(out[0].t == b[0].t);
}

TEST_CASE("anim motion: a short take is continued at its last pace and can be jumped to almost to its end", "[anim][motion]") {
    using namespace offline;
    const SkeletonDef skel = fx::make_skeleton();
    // the same 1.2 s run twice: once as an ordinary clip, once marked as a short take
    std::vector<RawClip> clips;
    clips.push_back(fx::make_move_clip(skel, "plain", 36, 30.0f, { 0.0f, -3.0f }, kTagLocomotion));
    clips.push_back(fx::make_move_clip(skel, "short", 36, 30.0f, { 0.0f, -3.0f }, kTagLocomotion | kTagShortTake));
    std::vector<uint8_t> blob;
    std::string why;
    REQUIRE(build_motion_database(skel, clips, MotionBuildParams{}, blob, nullptr, &why));
    MotionDatabase db;
    REQUIRE(db.load(blob.data(), blob.size(), skel.rig_id(), &why));
    const MotionRange& plain = db.ranges()[0];
    const MotionRange& shrt = db.ranges()[1];
    CHECK(plain.count <= plain.head + plain.tail + 5);       // a second of future is cut off: next to nothing is left
    CHECK(shrt.count >= shrt.head + shrt.tail + 25);         // nearly all of it can be jumped to
    // one second ahead of a frame near the end: 3 m straight on, although the clip ends in 0.3 s
    float raw[MotionSchema::kFeat];
    db.raw_features(shrt.first + shrt.count - shrt.tail - 1, false, raw);
    CHECK(std::abs(raw[19]) < 0.05f);
    CHECK(std::abs(raw[20] + 3.0f) < 0.1f);
    db.raw_features(plain.first + plain.count - 8, false, raw);
    CHECK(raw[20] > -1.0f);                                  // the ordinary clip's path just stops there
}
