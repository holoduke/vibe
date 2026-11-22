// The character animator against the real character controller (game::update_player): the
// gameplay scenarios the animation has to survive, with the controller's trajectory checked to
// be untouched by it.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "anim/anim_lod.h"
#include "fixtures.h"
#include "game/player.h"

#include <functional>
#include <random>

using namespace vibe;
using namespace vibe::anim;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float kPi_ = 3.14159265f;
float flat_ground(void*, float, float) { return 0.0f; }
float sloped_ground(void*, float x, float) { return 0.25f * x; }            // rises to the east

struct Sim {
    game::Player p;
    std::vector<collision::AABB> world{ { glm::vec3(-2000.0f, -1.0f, -2000.0f), glm::vec3(2000.0f, 0.0f, 2000.0f) } };
    float time = 0.0f;
    Sim() { p.position = glm::vec3(0.0f, 0.9f, 0.0f); p.on_ground = true; }
    void tick(const game::PlayerInput& in, float dt) { game::update_player(p, in, world, dt); time += dt; }
};

CharacterAnimationInput make_input(const Sim& s, const game::PlayerInput& pin, float dt) {
    CharacterAnimationInput in;
    in.dt = dt;
    in.position = s.p.position - glm::vec3(0.0f, game::Player::kHalfExtents.y, 0.0f);
    in.velocity = s.p.velocity;
    const glm::vec3 f(std::sin(s.p.yaw), 0.0f, -std::cos(s.p.yaw)), r(std::cos(s.p.yaw), 0.0f, std::sin(s.p.yaw));
    glm::vec3 wish(0.0f);
    if (pin.fwd) wish += f;
    if (pin.back) wish -= f;
    if (pin.right) wish += r;
    if (pin.left) wish -= r;
    if (glm::dot(wish, wish) > 0.0f)
        in.desired_velocity = glm::normalize(wish) * game::kPlayerMove.max_walk_speed * (pin.sprint ? 1.5f : pin.crawl ? 0.4f : 1.0f);
    in.facing_yaw = s.p.yaw;
    in.aim_pitch = s.p.pitch;
    in.grounded = s.p.on_ground;
    in.wants_jump = pin.jump;
    in.crouch = s.p.crouch_factor;
    in.gravity = pin.gravity;
    in.eye_local = glm::vec3(0.0f, game::Player::kHalfExtents.y + s.p.eye_offset_y(), -0.11f);
    in.stand_eye_height = game::Player::kHalfExtents.y + game::Player::kEyeOffsetStand;
    in.ground_height = flat_ground;
    return in;
}

// Watches every pose the animator produces.
struct Watch {
    const AnimationSet& set;
    std::vector<Transform> prev;
    bool   have = false;
    float  max_ang_rate = 0.0f;         // rad/s, any joint
    float  max_bone_err = 0.0f;         // metres off the rest length
    int    frames = 0;
    std::string worst_joint;
    explicit Watch(const AnimationSet& s) : set(s) {}
    void see(const CharacterAnimator& a, float dt) {
        ++frames;
        REQUIRE(pose_is_valid(a.local()));
        REQUIRE(pose_is_valid(a.model()));
        REQUIRE(a.debug().invalid_poses == 0);
        const SkeletonDef& sk = set.runtime->skeleton();
        const int hips = set.runtime->rig()[Role::Hips];
        for (int j = 1; j < sk.size(); ++j) {
            if (j != hips) {
                const float len = glm::length(glm::vec3(a.model()[size_t(j)][3]) - glm::vec3(a.model()[size_t(sk.parents[size_t(j)])][3]));
                max_bone_err = std::max(max_bone_err, std::abs(len - glm::length(sk.rest_local(j))));
            }
            // (a hand keeps its orientation on the weapon while the forearm under it moves: its local
            // rotation is the forearm's in reverse, not a movement of its own)
            const bool is_hand = sk.names[size_t(j)] == "LeftHand" || sk.names[size_t(j)] == "RightHand";
            if (have && dt > 1e-5f && !is_hand) {
                const glm::quat d = glm::inverse(prev[size_t(j)].r) * a.local()[size_t(j)].r;
                const float ang = 2.0f * std::asin(std::min(glm::length(glm::vec3(d.x, d.y, d.z)), 1.0f));
                if (ang / dt > max_ang_rate) { max_ang_rate = ang / dt; worst_joint = sk.names[size_t(j)] + " in " + a.debug().mode; }
            }
        }
        prev.assign(a.local().begin(), a.local().end());
        have = true;
    }
};

using Script = std::function<void(float t, game::PlayerInput& in, Sim& sim, CharacterAnimationInput& extra)>;

// Runs controller + animator for `seconds`; the controller ticks at 120 Hz as in the game,
// the animator once per rendered frame of length `frame_dt(i)`.
void run(Sim& sim, CharacterAnimator& anim, Watch& w, float seconds, const Script& script,
         const std::function<float(int)>& frame_dt, const std::function<void(const CharacterAnimationInput&)>& each = {}) {
    const float tick = 1.0f / 120.0f;
    float acc = 0.0f, t = 0.0f;
    CharacterAnimationInput extra;
    for (int i = 0; t < seconds; ++i) {
        const float dt = frame_dt(i);
        game::PlayerInput pin;
        script(t, pin, sim, extra);
        acc += dt;
        while (acc >= tick) { sim.tick(pin, tick); acc -= tick; }
        CharacterAnimationInput in = make_input(sim, pin, dt);
        in.held = extra.held; in.shot_seq = extra.shot_seq; in.reload_seq = extra.reload_seq; in.reloading = extra.reloading;
        in.first_person = extra.first_person; in.lod = extra.lod;
        if (extra.ground_height) in.ground_height = extra.ground_height;
        in.ground_velocity = extra.ground_velocity;
        anim.update(in);
        w.see(anim, dt);
        if (each) each(in);
        t += dt;
    }
}

const std::function<float(int)> k60 = [](int) { return 1.0f / 60.0f; };

std::string clip_name(const fx::SetFixture& f, const CharacterAnimator& a) {
    return a.debug().clip >= 0 ? f.asset.clips[size_t(a.debug().clip)].name : "none";
}

}  // namespace

TEST_CASE("animator: idle, run, stop, strafe pick the fitting captures", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    Watch w(f.set);
    run(sim, a, w, 0.6f, [](float, game::PlayerInput&, Sim&, CharacterAnimationInput&) {}, k60);
    CHECK(clip_name(f, a) == "idle");
    run(sim, a, w, 1.0f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.fwd = true; }, k60);
    CHECK(clip_name(f, a) == "run_f");
    CHECK(glm::length(sim.p.velocity) > 7.5f);
    CHECK(a.debug().rate > 1.2f);                      // the capture is played faster ...
    CHECK(a.debug().stride > 1.5f);                    // ... and with longer strides
    CHECK(a.debug().rate <= a.tuning.max_rate + 1e-4f);
    CHECK(a.debug().stride <= a.tuning.max_stride + 1e-4f);
    // Sideways at this pace is run, not side-stepped: the forward run on legs turned to the side,
    // one way for right and the other for left.
    run(sim, a, w, 0.5f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.right = true; }, k60);
    CHECK(clip_name(f, a) == "run_f");
    const float warp_right = a.debug().warp;
    CHECK(std::abs(warp_right) > 1.1f);
    CHECK(std::abs(warp_right) <= a.tuning.max_orient_warp + 1e-3f);
    run(sim, a, w, 0.5f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.left = true; }, k60);
    CHECK(clip_name(f, a) == "run_f");
    CHECK(a.debug().warp * warp_right < -1.0f);
    // The sidestep itself (as below the strafe-run pace): the capture as it is to the right,
    // mirrored to the left.
    a.tuning.strafe_run_speed = 1e9f;
    run(sim, a, w, 0.5f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.right = true; }, k60);
    CHECK(clip_name(f, a) == "strafe_r");
    CHECK_FALSE(a.debug().mirrored);
    run(sim, a, w, 0.5f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.left = true; }, k60);
    CHECK(clip_name(f, a) == "strafe_r");
    CHECK(a.debug().mirrored);
    a.tuning.strafe_run_speed = AnimTuning{}.strafe_run_speed;
    run(sim, a, w, 0.5f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.back = true; }, k60);
    CHECK(clip_name(f, a) == "walk_b");
    run(sim, a, w, 0.8f, [](float, game::PlayerInput&, Sim&, CharacterAnimationInput&) {}, k60);
    CHECK(clip_name(f, a) == "idle");
    run(sim, a, w, 0.6f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.fwd = true; in.crawl = true; }, k60);
    CHECK(clip_name(f, a) == "crouch_f");
    INFO("max joint rate " << w.max_ang_rate << " rad/s at " << w.worst_joint);
    CHECK(w.max_bone_err < 2e-3f);
    // (a foot passing under the hips at 8 - 12 m/s turns knee and thigh at 30 - 45 rad/s by geometry
    // alone; a pop is 100 and more)
    CHECK(w.max_ang_rate < 50.0f);
}

TEST_CASE("animator: sprint, instant 180 degree reversal, sprint", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    Watch w(f.set);
    float t_new_dir = -1.0f;
    run(sim, a, w, 3.0f,
        [](float t, game::PlayerInput& in, Sim& s, CharacterAnimationInput&) {
            in.fwd = true; in.sprint = true;
            if (t >= 1.5f) s.p.yaw = kPi_;                  // the view snaps round in one frame
        },
        k60,
        [&](const CharacterAnimationInput& in) {
            // after the flick the character still slides the old way for a moment: travelling backward relative to the view
            if (sim.time > 1.5f && t_new_dir < 0.0f && clip_name(f, a) == "run_f" && glm::dot(glm::vec2(in.velocity.x, in.velocity.z), glm::vec2(std::sin(in.facing_yaw), -std::cos(in.facing_yaw))) > 2.0f)
                t_new_dir = sim.time;
        });
    CHECK(glm::length(sim.p.velocity) > 11.0f);
    CHECK(sim.p.velocity.z > 11.0f);                     // going the new way at sprint speed
    CHECK(clip_name(f, a) == "run_f");
    CHECK(t_new_dir > 0.0f);
    INFO("max joint rate " << w.max_ang_rate << " rad/s at " << w.worst_joint);
    CHECK(w.max_ang_rate < 50.0f);
    CHECK(a.debug().switches < 40);
}

TEST_CASE("animator: rapid alternating strafes, 5 to 10 Hz", "[anim][animator]") {
    fx::SetFixture f;
    for (float hz : { 5.0f, 7.0f, 10.0f }) {
        CharacterAnimator a(f.set);
        Sim sim;
        Watch w(f.set);
        run(sim, a, w, 3.0f,
            [hz](float t, game::PlayerInput& in, Sim&, CharacterAnimationInput&) {
                const bool right = std::fmod(t * hz, 1.0f) < 0.5f;
                in.right = right; in.left = !right;
            }, k60);
        INFO(hz << " Hz: " << a.debug().switches << " switches, max joint rate " << w.max_ang_rate);
        CHECK(float(a.debug().switches) <= 3.0f / a.tuning.matcher.min_switch_interval + 2.0f);
        CHECK(w.max_ang_rate < 40.0f);
        CHECK(w.max_bone_err < 2e-3f);
    }
}

TEST_CASE("animator: jumps with normal, 2x and 3x airtime never run out of clip", "[anim][animator]") {
    fx::SetFixture f;
    const MotionJump& j = f.set.db.jumps()[0];
    for (float gravity : { 25.0f, 12.5f, 8.33f }) {
        CharacterAnimator a(f.set);
        Sim sim;
        Watch w(f.set);
        float air = 0.0f, land_t = -1.0f;
        bool saw[6] = {};
        float last_frame_change = 0.0f, longest_still = 0.0f, prev_clip_time = -1.0f;
        run(sim, a, w, 3.5f,
            [gravity](float t, game::PlayerInput& in, Sim&, CharacterAnimationInput&) {
                in.gravity = gravity; in.fwd = true; in.jump = t > 0.5f && t < 0.6f;
            },
            k60,
            [&](const CharacterAnimationInput& in) {
                saw[int(a.debug().air_phase)] = true;
                if (!in.grounded) {
                    air += in.dt;
                    // the frame shown is inside the captured flight, whatever the airtime
                    const float fr = a.debug().clip_time * f.set.db.sample_rate() + float(f.set.db.range_of(j.takeoff).first);
                    if (air > 0.1f) { CHECK(fr >= float(j.takeoff) - 0.01f); CHECK(fr <= float(j.land) - 0.99f); }
                    // and the pose keeps moving (limbs work through a long fall): never a frozen picture
                    if (a.debug().clip_time != prev_clip_time) last_frame_change = sim.time;
                    longest_still = std::max(longest_still, sim.time - last_frame_change);
                } else if (air > 0.2f && land_t < 0.0f) land_t = sim.time;
                prev_clip_time = a.debug().clip_time;
            });
        INFO("gravity " << gravity << ": airtime " << air << " s, max joint rate " << w.max_ang_rate << " at " << w.worst_joint);
        CHECK(air > 0.6f * 25.0f / gravity);                    // 2 * 8.5 / g
        CHECK(saw[int(AirPhase::Rise)]);
        CHECK(saw[int(AirPhase::Apex)]);
        CHECK(saw[int(AirPhase::Reach)]);
        CHECK(a.debug().flights == 1);
        CHECK(land_t > 0.0f);
        CHECK(a.debug().air_phase == AirPhase::Ground);
        CHECK(clip_name(f, a) == "run_f");                      // landed into the run
        CHECK(w.max_ang_rate < 40.0f);
        CHECK(w.max_bone_err < 2e-3f);
    }
}

TEST_CASE("animator: a long fall keeps the body alive and reaches for the ground at the end", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    sim.p.position.y = 60.0f;                                   // dropped from a tower
    sim.p.on_ground = false;
    Watch w(f.set);
    float moved = 0.0f;
    std::vector<Transform> last;
    bool reached_before_contact = false;
    run(sim, a, w, 2.6f, [](float, game::PlayerInput&, Sim&, CharacterAnimationInput&) {}, k60,
        [&](const CharacterAnimationInput& in) {
            if (!in.grounded && sim.time > 0.8f && sim.time < 1.8f) {
                const int lu = f.set.runtime->rig()[Role::LeftUpLeg];
                if (!last.empty()) moved += fx::joint_error(last[size_t(lu)], a.local()[size_t(lu)]);
                last.assign(a.local().begin(), a.local().end());
                CHECK(a.debug().air_phase == AirPhase::Fall);
            }
            if (!in.grounded && a.debug().air_phase == AirPhase::Reach) {
                reached_before_contact = true;
                CHECK(a.debug().time_to_contact <= a.tuning.air.reach_time + 0.02f);
            }
        });
    CHECK(sim.p.on_ground);
    CHECK(moved > 0.2f);                                        // radians of thigh movement in one second of falling
    CHECK(reached_before_contact);
    CHECK(a.debug().impact > 20.0f);
    CHECK(w.max_ang_rate < 40.0f);
}

TEST_CASE("animator: landing never holds the player - jump again at once, or run on", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    Watch w(f.set);
    int flights_seen = 0;
    bool was_ground = true;
    float worst_ground_stay = 0.0f, ground_since = 0.0f;
    run(sim, a, w, 4.0f,
        [](float t, game::PlayerInput& in, Sim& s, CharacterAnimationInput&) {
            in.fwd = true;
            in.jump = s.p.on_ground && std::fmod(t, 0.05f) < 0.02f;      // bunny hop: press the moment the ground is there
        },
        k60,
        [&](const CharacterAnimationInput& in) {
            if (!in.grounded && was_ground) ++flights_seen;
            if (in.grounded) ground_since += in.dt; else { worst_ground_stay = std::max(worst_ground_stay, ground_since); ground_since = 0.0f; }
            was_ground = in.grounded;
            CHECK(a.debug().air_phase != AirPhase::Land);              // moving: the captured landing never plays
        });
    CHECK(flights_seen >= 4);
    CHECK(worst_ground_stay < 0.12f);                           // the controller left the ground again within a few ticks
    CHECK(a.debug().flights >= uint32_t(flights_seen) - 1);
    CHECK(w.max_ang_rate < 40.0f);

    // Standing still, a hard landing shows the capture's landing and a pelvis dip - and input ends it.
    CharacterAnimator b(f.set);
    Sim s2;
    s2.p.position.y = 6.0f;
    s2.p.on_ground = false;
    Watch w2(f.set);
    bool land_seen = false;
    float deepest = 0.0f;
    run(s2, b, w2, 1.6f, [](float, game::PlayerInput&, Sim&, CharacterAnimationInput&) {}, k60,
        [&](const CharacterAnimationInput&) {
            land_seen = land_seen || b.debug().air_phase == AirPhase::Land;
            deepest = std::min(deepest, b.debug().pelvis_offset);
        });
    CHECK(land_seen);
    CHECK(deepest < -0.1f);
    CHECK(deepest >= -b.tuning.max_land_dip - 1e-3f);
    CHECK(b.debug().air_phase == AirPhase::Ground);             // over by itself within its time limit
    CHECK(std::abs(b.debug().pelvis_offset) < 0.01f);           // and the dip has gone
}

TEST_CASE("animator: double jump and a 180 degree turn in the air", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    Watch w(f.set);
    float top = 0.0f;
    run(sim, a, w, 3.0f,
        [](float t, game::PlayerInput& in, Sim& s, CharacterAnimationInput& x) {
            in.fwd = true; in.sprint = true;
            in.jump = (t > 0.8f && t < 0.85f) || (t > 1.1f && t < 1.15f);          // jump, then the air jump
            if (t > 1.25f && t < 1.3f) s.p.yaw = kPi_;                              // spin round in the air
            if (t > 1.3f) x.shot_seq = uint32_t((t - 1.3f) * 10.0f);                // and fire, 10 shots a second
        },
        k60, [&](const CharacterAnimationInput& in) { top = std::max(top, in.position.y); });
    CHECK(top > 2.0f);                                          // higher than one jump can go (1.45 m)
    CHECK(sim.p.on_ground);
    CHECK(a.debug().flights == 1);
    CHECK(w.max_ang_rate < 40.0f);
    CHECK(w.max_bone_err < 2e-3f);
}

TEST_CASE("animator: both hands stay on the weapon, and the weapon on the aim", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    Watch w(f.set);
    const Rig& rig = f.set.runtime->rig();
    float worst_hand = 0.0f, worst_aim = 0.0f;
    int out_of_reach = 0, hand_checks = 0;
    run(sim, a, w, 6.0f,
        [](float t, game::PlayerInput& in, Sim& s, CharacterAnimationInput& x) {
            in.fwd = std::fmod(t, 2.0f) < 1.4f; in.right = std::fmod(t, 1.3f) < 0.5f; in.sprint = t > 3.0f;
            in.jump = std::fmod(t, 1.7f) < 0.05f;
            s.p.pitch = 1.1f * std::sin(t * 1.3f);              // look far up and far down
            s.p.yaw += 0.03f * std::sin(t * 0.7f);
            x.shot_seq = uint32_t(t * 10.0f);                   // firing all the time
        },
        k60,
        [&](const CharacterAnimationInput& in) {
            for (int side = 0; side < 2; ++side) {
                const int hd = rig[side == 0 ? Role::LeftHand : Role::RightHand], fa = rig[side == 0 ? Role::LeftForeArm : Role::RightForeArm],
                          ua = rig[side == 0 ? Role::LeftArm : Role::RightArm];
                const glm::vec3 ph(a.model()[size_t(hd)][3]), pf(a.model()[size_t(fa)][3]), pu(a.model()[size_t(ua)][3]);
                const glm::vec3 tgt = a.debug().hand_target[side];
                const float reach = glm::length(pf - pu) + glm::length(ph - pf);
                if (glm::length(tgt - pu) <= reach * 0.98f) worst_hand = std::max(worst_hand, glm::length(ph - tgt));
                else {
                    // beyond the arm's reach (this test rig has short arms): stretched straight at it
                    CHECK(glm::dot(glm::normalize(ph - pu), glm::normalize(tgt - pu)) > 0.999f);
                    ++out_of_reach;
                }
                ++hand_checks;
            }
            const glm::vec3 aim(0.0f, std::sin(in.aim_pitch), -std::cos(in.aim_pitch));
            worst_aim = std::max(worst_aim, std::acos(std::min(glm::dot(aim, a.muzzle_direction()), 1.0f)));
            CHECK(a.debug().recoil >= 0.0f);
        });
    INFO("worst wrist error " << worst_hand << " m, worst muzzle angle " << worst_aim << " rad");
    INFO(out_of_reach << " of " << hand_checks << " hand checks were beyond reach");
    CHECK(worst_hand < 0.02f);                                  // wrists on their grips
    CHECK(out_of_reach * 2 < hand_checks);
    CHECK(worst_aim <= a.weapon.recoil_pitch + 1e-3f);          // the muzzle is off the aim by the recoil climb at most
    CHECK(w.max_bone_err < 2e-3f);
}

TEST_CASE("animator: recoil, reload, interruption and weapon switch", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    auto step = [&](int n) { for (int i = 0; i < n; ++i) a.update(in); };
    step(30);
    CHECK(a.debug().recoil == 0.0f);
    const glm::vec3 rest_muzzle = a.muzzle_position();
    // a shot kicks the weapon back and returns within the recoil's duration
    in.shot_seq = 1;
    step(3);
    CHECK(a.debug().recoil > 0.5f);
    CHECK(a.muzzle_position().z > rest_muzzle.z + 0.02f);       // +z is toward the shooter
    step(12);
    CHECK(a.debug().recoil == 0.0f);
    CHECK(glm::length(a.muzzle_position() - rest_muzzle) < 1e-4f);
    // the same shot number again is not a new shot (duplicated network event)
    step(5);
    CHECK(a.debug().recoil == 0.0f);
    // reload: the support hand leaves the fore-end and comes back; the firing hand stays
    const Rig& rig = f.set.runtime->rig();
    const glm::vec3 lh0(a.model()[size_t(rig[Role::LeftHand])][3]), rh0(a.model()[size_t(rig[Role::RightHand])][3]);
    in.reload_seq = 1; in.reloading = true;
    float far_l = 0.0f, far_r = 0.0f;
    for (int i = 0; i < 60; ++i) {
        a.update(in);
        far_l = std::max(far_l, glm::length(glm::vec3(a.model()[size_t(rig[Role::LeftHand])][3]) - lh0));
        far_r = std::max(far_r, glm::length(glm::vec3(a.model()[size_t(rig[Role::RightHand])][3]) - rh0));
        REQUIRE(pose_is_valid(a.model()));
    }
    CHECK(far_l > 0.15f);
    CHECK(far_r < 0.08f);
    CHECK(a.debug().reload > 0.5f);
    // interrupted (fired): the hand is back on the weapon in a moment, the reload does not resume
    in.reloading = false;
    in.shot_seq = 2;
    step(20);
    CHECK(a.debug().reload == 0.0f);
    CHECK(glm::length(glm::vec3(a.model()[size_t(rig[Role::LeftHand])][3]) - lh0) < 0.02f);
    // a full reload ends by itself
    in.reload_seq = 2; in.reloading = true;
    step(int(a.weapon.reload_time * 60.0f) + 30);
    CHECK(a.debug().reload == 0.0f);
    CHECK(glm::length(glm::vec3(a.model()[size_t(rig[Role::LeftHand])][3]) - lh0) < 0.02f);
    // switching what is held lowers it and brings the new item up
    in.reloading = false;
    in.held = HeldItem::Tablet;
    step(2);
    CHECK(a.debug().equip > 0.5f);
    step(30);
    CHECK(a.debug().equip == 0.0f);
    // the free arm is the capture's: far from the rifle's fore-end
    CHECK(glm::length(glm::vec3(a.model()[size_t(rig[Role::LeftHand])][3]) - lh0) > 0.15f);
    in.held = HeldItem::None;
    step(30);
    REQUIRE(pose_is_valid(a.model()));
}

TEST_CASE("animator: planted feet do not slide; locks let go in the air and at speed", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    Watch w(f.set);
    const Rig& rig = f.set.runtime->rig();
    glm::vec3 held[2];
    bool was[2] = {};
    float worst_slide = 0.0f;
    int locked_frames = 0;
    // a slow walk: crawl speed is 3.2 m/s, under the lock limit
    run(sim, a, w, 4.0f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.fwd = true; in.crawl = true; }, k60,
        [&](const CharacterAnimationInput& in) {
            const float s = std::sin(in.facing_yaw), c = std::cos(in.facing_yaw);
            for (int side = 0; side < 2; ++side) {
                const glm::vec3 m(a.model()[size_t(rig[side == 0 ? Role::LeftFoot : Role::RightFoot])][3]);
                const glm::vec3 world = in.position + glm::vec3(c * m.x - s * m.z, m.y, s * m.x + c * m.z);
                if (a.debug().locked[side]) {
                    if (was[side]) worst_slide = std::max(worst_slide, glm::length(glm::vec2(world.x - held[side].x, world.z - held[side].z)));
                    else held[side] = world;
                    ++locked_frames;
                }
                was[side] = a.debug().locked[side];
            }
        });
    INFO("locked frames " << locked_frames << ", worst slide " << worst_slide << " m");
    CHECK(locked_frames > 20);
    CHECK(worst_slide < 0.015f);
    // sprinting: never locked. In the air: never locked.
    int bad = 0;
    run(sim, a, w, 2.0f, [](float t, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.fwd = true; in.sprint = true; in.jump = std::fmod(t, 0.9f) < 0.05f; }, k60,
        [&](const CharacterAnimationInput& in) {
            if ((glm::length(in.velocity) > a.tuning.foot_lock_max_speed + 0.5f || !in.grounded) && (a.debug().locked[0] || a.debug().locked[1])) ++bad;
        });
    CHECK(bad == 0);
    // switched off: no locks at all
    a.tuning.foot_lock = false;
    int any = 0;
    run(sim, a, w, 1.5f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.back = true; in.crawl = true; }, k60,
        [&](const CharacterAnimationInput&) { any += a.debug().locked[0] || a.debug().locked[1]; });
    CHECK(any == 0);
}

TEST_CASE("animator: feet follow a slope and a moving platform", "[anim][animator]") {
    fx::SetFixture f;
    const Rig& rig = f.set.runtime->rig();
    // Standing across a slope that rises to the right: the right foot is higher than the left.
    CharacterAnimator a(f.set);
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = sloped_ground;
    for (int i = 0; i < 90; ++i) a.update(in);
    const float ly = a.model()[size_t(rig[Role::LeftFoot])][3].y, ry = a.model()[size_t(rig[Role::RightFoot])][3].y;
    const float lx = a.model()[size_t(rig[Role::LeftFoot])][3].x, rx = a.model()[size_t(rig[Role::RightFoot])][3].x;
    CHECK_THAT(ry - ly, WithinAbs(0.25f * (rx - lx), 0.02));
    CHECK(a.debug().pelvis_offset == 0.0f);
    CHECK(a.model()[size_t(rig[Role::Hips])][3].y < f.set.stand_hips);       // the hips came down with the lower foot
    REQUIRE(pose_is_valid(a.model()));
    // switched off: flat feet
    a.tuning.foot_ik = false;
    for (int i = 0; i < 60; ++i) a.update(in);
    CHECK(std::abs(a.model()[size_t(rig[Role::RightFoot])][3].y - a.model()[size_t(rig[Role::LeftFoot])][3].y) < 0.01f);

    // A platform carrying the character sideways at 3 m/s: standing on it, the feet stay under the body.
    CharacterAnimator b(f.set);
    CharacterAnimationInput pin;
    pin.dt = 1.0f / 60.0f;
    pin.ground_height = flat_ground;
    for (int i = 0; i < 120; ++i) {
        pin.ground_velocity = glm::vec3(3.0f, 0.0f, 0.0f);
        pin.position.x += 3.0f * pin.dt;                         // carried along; no velocity of its own over the ground
        b.update(pin);
        REQUIRE(pose_is_valid(b.model()));
        if (i > 30) {
            CHECK(std::abs(b.model()[size_t(rig[Role::LeftFoot])][3].x - f.set.runtime->skeleton().rest_pos[size_t(rig[Role::LeftFoot])].x) < 0.06f);
        }
    }
}

TEST_CASE("animator: variable frame rates give valid, similar motion", "[anim][animator]") {
    fx::SetFixture f;
    const Script script = [](float t, game::PlayerInput& in, Sim& s, CharacterAnimationInput& x) {
        in.fwd = t > 0.3f && t < 2.2f; in.left = t > 1.4f && t < 2.8f; in.jump = t > 1.0f && t < 1.3f;
        s.p.pitch = 0.4f * std::sin(t);
        x.shot_seq = uint32_t(t * 5.0f);
    };
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> jitter(1.0f / 240.0f, 1.0f / 20.0f);
    const std::function<float(int)> rates[] = {
        [](int) { return 1.0f / 30.0f; }, [](int) { return 1.0f / 60.0f; }, [](int) { return 1.0f / 144.0f; }, [](int) { return 1.0f / 500.0f; },
        [&](int) { return jitter(rng); }, [](int i) { return i % 50 == 49 ? 0.25f : 1.0f / 60.0f; },      // hitches of a quarter second
    };
    std::vector<std::string> end_clip;
    for (const auto& rate : rates) {
        CharacterAnimator a(f.set);
        Sim sim;
        Watch w(f.set);
        run(sim, a, w, 3.6f, script, rate);
        end_clip.push_back(clip_name(f, a));
        CHECK(w.max_bone_err < 2e-3f);
        CHECK(a.debug().flights == 1);
    }
    for (const std::string& c : end_clip) CHECK(c == "idle");
    // a zero-length frame and a stalled frame are harmless
    CharacterAnimator a(f.set);
    CharacterAnimationInput in;
    for (float dt : { 0.0f, 0.0f, 1.0f / 60.0f, 3.0f, 0.0f, 1e-7f, 1.0f / 60.0f }) {
        in.dt = dt;
        in.velocity = glm::vec3(0.0f, 0.0f, -6.0f);
        a.update(in);
        REQUIRE(pose_is_valid(a.model()));
    }
}

TEST_CASE("animator: the controller's trajectory is identical with and without animation", "[anim][animator]") {
    fx::SetFixture f;
    const Script script = [](float t, game::PlayerInput& in, Sim& s, CharacterAnimationInput&) {
        in.fwd = std::fmod(t, 1.1f) < 0.8f; in.right = std::fmod(t, 0.7f) < 0.3f; in.left = std::fmod(t, 0.9f) > 0.6f;
        in.sprint = t > 2.0f; in.crawl = t > 4.0f && t < 4.6f;
        in.jump = std::fmod(t, 0.8f) < 0.04f;
        s.p.yaw = 2.0f * std::sin(t * 1.9f);
        s.p.pitch = 0.6f * std::sin(t * 0.8f);
    };
    // with animation
    Sim with;
    std::vector<glm::vec3> pos_a, vel_a;
    std::vector<int> jump_tick_a;
    {
        CharacterAnimator a(f.set);
        Watch w(f.set);
        bool was = true;
        int i = 0;
        run(with, a, w, 6.0f, script, k60, [&](const CharacterAnimationInput&) {
            pos_a.push_back(with.p.position); vel_a.push_back(with.p.velocity);
            if (was && !with.p.on_ground) jump_tick_a.push_back(i);
            was = with.p.on_ground; ++i;
        });
    }
    // without: the same loop, no animator
    Sim without;
    std::vector<glm::vec3> pos_b, vel_b;
    std::vector<int> jump_tick_b;
    {
        const float tick = 1.0f / 120.0f;
        float acc = 0.0f, t = 0.0f;
        bool was = true;
        CharacterAnimationInput extra;
        for (int i = 0; t < 6.0f; ++i) {
            game::PlayerInput pin;
            script(t, pin, without, extra);
            acc += 1.0f / 60.0f;
            while (acc >= tick) { without.tick(pin, tick); acc -= tick; }
            pos_b.push_back(without.p.position); vel_b.push_back(without.p.velocity);
            if (was && !without.p.on_ground) jump_tick_b.push_back(i);
            was = without.p.on_ground;
            t += 1.0f / 60.0f;
        }
    }
    REQUIRE(pos_a.size() == pos_b.size());
    for (size_t i = 0; i < pos_a.size(); ++i) {
        REQUIRE(pos_a[i] == pos_b[i]);                           // bit for bit
        REQUIRE(vel_a[i] == vel_b[i]);
    }
    CHECK(jump_tick_a == jump_tick_b);                           // every jump left the ground on the same frame
    CHECK(jump_tick_a.size() >= 5);
}

TEST_CASE("animator: the same inputs give the same poses", "[anim][animator]") {
    fx::SetFixture f;
    const Script script = [](float t, game::PlayerInput& in, Sim& s, CharacterAnimationInput& x) {
        in.fwd = t > 0.2f; in.right = std::fmod(t, 0.6f) < 0.3f; in.jump = std::fmod(t, 1.3f) < 0.05f;
        s.p.yaw = std::sin(t); x.shot_seq = uint32_t(t * 8.0f);
    };
    CharacterAnimator a(f.set), b(f.set);
    Sim sa, sb;
    Watch wa(f.set), wb(f.set);
    run(sa, a, wa, 3.0f, script, k60);
    run(sb, b, wb, 3.0f, script, k60);
    for (size_t j = 0; j < a.model().size(); ++j) CHECK(a.model()[j] == b.model()[j]);
    // reset() brings an animator back to a fresh one
    a.reset();
    Sim sc;
    Watch wc(f.set);
    run(sc, a, wc, 3.0f, script, k60);
    for (size_t j = 0; j < a.model().size(); ++j) CHECK(a.model()[j] == b.model()[j]);
}

TEST_CASE("animator: levels of detail", "[anim][lod]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    Sim sim;
    Watch w(f.set);
    const Script go = [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput&) { in.fwd = true; };
    for (int lod : { 0, 1, 2 }) {
        run(sim, a, w, 0.8f, [lod](float, game::PlayerInput& in, Sim&, CharacterAnimationInput& x) { in.fwd = true; x.lod = lod; }, k60);
        CHECK(clip_name(f, a) == "run_f");
        CHECK(a.debug().lod == lod);
    }
    // level 2 searches far less often than level 0
    const uint32_t s0 = a.debug().searches;
    run(sim, a, w, 2.0f, [](float, game::PlayerInput& in, Sim&, CharacterAnimationInput& x) { in.fwd = true; x.lod = 2; }, k60);
    const uint32_t s2 = a.debug().searches - s0;
    run(sim, a, w, 2.0f, go, k60);
    const uint32_t sfull = a.debug().searches - s0 - s2;
    CHECK(s2 * 3 < sfull);
    // level 3: no pose work, but shots are not lost and the pose is coherent when it comes back
    const std::vector<glm::mat4> before(a.model().begin(), a.model().end());
    const float t_before = a.debug().clip_time;
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f; in.lod = 3; in.velocity = glm::vec3(0.0f, 0.0f, -8.0f); in.desired_velocity = in.velocity; in.shot_seq = 7;
    for (int i = 0; i < 30; ++i) a.update(in);
    for (size_t j = 0; j < before.size(); ++j) CHECK(a.model()[j] == before[j]);       // untouched
    in.lod = 0;
    a.update(in);
    CHECK(a.debug().recoil == 0.0f);                              // the shot happened while unseen: its kick is over
    CHECK(a.debug().clip_time != t_before);                       // playback moved on meanwhile
    REQUIRE(pose_is_valid(a.model()));

    // selection: by size on screen, with hysteresis; combat keeps full detail
    LodInput li;
    li.distance = 2.0f;
    CHECK(select_lod(li) == 0);
    li.distance = 20.0f;
    CHECK(select_lod(li) == 1);
    li.distance = 60.0f;
    CHECK(select_lod(li) == 2);
    li.distance = 2000.0f;
    CHECK(select_lod(li) == 3);
    li.combat = true;
    CHECK(select_lod(li) == 0);
    li.in_view = false;
    CHECK(select_lod(li) == 2);
    li.combat = false;
    CHECK(select_lod(li) == 3);
    li.in_view = true;
    // no flicker at a boundary
    li.distance = 1.8f / (0.7f * 110.0f) * 540.0f;                // exactly the level 0 / 1 boundary
    li.previous = 0; li.distance *= 1.05f;
    CHECK(select_lod(li) == 0);
    li.previous = 1;
    CHECK(select_lod(li) == 1);
    li.distance /= 1.05f * 1.05f;
    CHECK(select_lod(li) == 1);
    li.distance /= 1.3f;
    CHECK(select_lod(li) == 0);
    int prev = 0, changes = 0;
    for (int i = 0; i < 200; ++i) {                              // wobbling round a boundary
        li.distance = 1.8f / (0.7f * 45.0f) * 540.0f * (1.0f + 0.05f * std::sin(float(i)));
        li.previous = prev;
        const int l = select_lod(li);
        changes += l != prev;
        prev = l;
    }
    CHECK(changes <= 1);
}

TEST_CASE("animator: teleport drops world-space memory; swimming and no ground callback work", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    for (int i = 0; i < 60; ++i) a.update(in);
    in.position = glm::vec3(500.0f, 30.0f, -200.0f);
    in.teleported = true;
    a.update(in);
    in.teleported = false;
    REQUIRE(pose_is_valid(a.model()));
    const Rig& rig = f.set.runtime->rig();
    // feet under the body at the new place, not stretched back toward the old one
    CHECK(glm::length(glm::vec2(a.model()[size_t(rig[Role::LeftFoot])][3].x, a.model()[size_t(rig[Role::LeftFoot])][3].z)) < 0.4f);
    in.swimming = true; in.grounded = false; in.velocity = glm::vec3(0.0f, 0.0f, -3.0f);
    for (int i = 0; i < 60; ++i) { a.update(in); REQUIRE(pose_is_valid(a.model())); }
    CHECK(std::string(a.debug().mode) == "swim");
    CHECK(a.debug().air_phase == AirPhase::Ground);              // swimming is not falling
    in.swimming = false; in.ground_height = nullptr; in.velocity = glm::vec3(0.0f, -5.0f, 0.0f);
    for (int i = 0; i < 60; ++i) { a.update(in); REQUIRE(pose_is_valid(a.model())); }
    CHECK(a.debug().air_phase == AirPhase::Fall);                // ground unknown: falls, never "reaches"
}

// The cooked asset itself, at the paces the controller really holds. The synthetic clips of the
// other tests are too tame to show what real capture does at 8 - 12 m/s: a torso leaning onto
// the weapon, legs captured straight, a request faster than anything recorded.
TEST_CASE("real asset: steady crawl, walk and sprint in every direction", "[anim][animator][asset]") {
    AnimationSet set;
    std::string why;
    if (!set.load((std::string(VIBE_SOURCE_DIR) + "/assets/character/soldier.qanim").c_str(), &why)) {
        WARN("assets/character/soldier.qanim not loaded (" << why << "): skipped");
        return;
    }
    // (the capture's pace per direction is what the request is mapped onto)
    CHECK(set.db.cruise_speed(false, 0) > 1.5f);
    CHECK(set.db.cruise_speed(false, 0) < 4.6f);            // (2.0 with 100STYLE alone, 3.4 with the CMU runs)
    CHECK(set.db.cruise_speed(false, 2) > 0.8f);
    CHECK(set.db.cruise_speed(false, 2) < set.db.cruise_speed(false, 0));
    const Rig& rig = set.runtime->rig();
    const int foot[2] = { rig[Role::LeftFoot], rig[Role::RightFoot] };
    struct Case { const char* name; bool fwd, back, right, sprint, crawl; float max_slide; };
    const Case cases[] = {
        { "crawl forward", true, false, false, false, true, 0.06f },   { "crawl diagonal", true, false, true, false, true, 0.06f },
        { "crawl backward", false, true, false, false, true, 0.06f },  { "walk forward", true, false, false, false, false, 0.16f },
        { "walk diagonal", true, false, true, false, false, 0.16f },   { "walk sideways", false, false, true, false, false, 0.18f },
        { "walk backward", false, true, false, false, false, 0.35f },  { "sprint forward", true, false, false, true, false, 0.20f },
        { "sprint diagonal", true, false, true, true, false, 0.20f },  { "sprint backward", false, true, false, true, false, 0.30f },
        { "sprint sideways", false, false, true, true, false, 0.20f },
    };
    for (const Case& c : cases) {
        Sim sim;
        CharacterAnimator a(set);
        Watch w(set);
        std::vector<glm::vec3> track[2];
        std::vector<char> on[2];
        float low[2] = { 1e9f, 1e9f };
        int fallback = 0, frames = 0;
        std::vector<int> clips_seen;
        float t_now = 0.0f;
        run(sim, a, w, 7.0f, [&](float t, game::PlayerInput& in, Sim&, CharacterAnimationInput&) {
            t_now = t;
            in.fwd = c.fwd; in.back = c.back; in.right = c.right; in.sprint = c.sprint; in.crawl = c.crawl;
        }, k60, [&](const CharacterAnimationInput& in) {
            if (t_now < 2.0f) return;
            ++frames;
            fallback += a.debug().fallback ? 1 : 0;
            for (int k = 0; k < 2; ++k) {
                const glm::vec3 p = in.position + glm::vec3(a.model()[size_t(foot[k])][3]);      // yaw 0: model axes = world axes
                track[k].push_back(p);
                on[k].push_back(a.debug().contact[k] ? 1 : 0);
                low[k] = std::min(low[k], p.y);
            }
        });
        const float speed = glm::length(glm::vec2(sim.p.velocity.x, sim.p.velocity.z));
        double slid = 0.0;
        int standing = 0, contact_frames[2] = { 0, 0 };
        for (int k = 0; k < 2; ++k)
            for (size_t i = 1; i < track[k].size(); ++i) {
                if (!on[k][i] || !on[k][i - 1]) continue;
                ++contact_frames[k];
                if (track[k][i].y > low[k] + 0.03f) continue;
                ++standing;
                const glm::vec3 d = track[k][i] - track[k][i - 1];
                slid += double(std::sqrt(d.x * d.x + d.z * d.z)) * 60.0;
            }
        const double slide = standing ? slid / standing / double(speed) : 0.0;
        INFO(c.name << ": " << speed << " m/s, slide " << slide * 100.0 << " % of it, fastest joint " << w.max_ang_rate << " rad/s at " << w.worst_joint
                    << ", path-only searches " << fallback << " of " << frames << " frames");
        CHECK(slide < double(c.max_slide));
        CHECK(w.max_ang_rate < 45.0f);                       // no knee or elbow snapping over
        CHECK(w.max_bone_err < 2e-3f);
        CHECK(fallback * 10 < frames);                       // the request is one the capture can answer
        CHECK(contact_frames[0] > 10);                       // both feet take steps: a gait, not one moment replayed
        CHECK(contact_frames[1] > 10);
    }
}

TEST_CASE("animator: seen from outside the rifle's butt is in the shoulder; the player's own view is unchanged", "[anim][animator]") {
    fx::SetFixture f;
    const Rig& rig = f.set.runtime->rig();
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
    // third person, level aim, no recoil: the weapon frame has the view's axes, so the butt is
    // the muzzle minus their difference
    CharacterAnimator outside(f.set);
    for (int i = 0; i < 40; ++i) outside.update(in);
    const glm::vec3 butt = outside.muzzle_position() - (outside.weapon.muzzle - outside.weapon.butt);
    const glm::vec3 shoulder(outside.model()[size_t(rig[Role::RightArm])][3]);
    INFO("butt " << butt.x << " " << butt.y << " " << butt.z << ", shoulder " << shoulder.x << " " << shoulder.y << " " << shoulder.z);
    // (placed against the shoulder as the body has it before the arms are fitted; the shoulder
    // girdle then gives a few centimetres toward the grips)
    CHECK(glm::length(butt - (shoulder + outside.weapon.shoulder_offset)) < 0.05f);
    // first person: where the view has it, whatever the body does
    CharacterAnimator own(f.set);
    in.first_person = true;
    for (int i = 0; i < 40; ++i) own.update(in);
    // (aiming, in his own view the weapon comes to the middle of the screen: nobody else sees that)
    CHECK(glm::length(own.muzzle_position() - (in.eye_local + own.weapon.muzzle + own.weapon.ads_shift_fp)) < 1e-4f);
    CHECK(std::abs(own.muzzle_position().x) < 0.03f);
    // and with the switch off, third person is the same as that
    CharacterAnimator plain(f.set);
    plain.tuning.shoulder_weapon = false;
    in.first_person = false;
    for (int i = 0; i < 40; ++i) plain.update(in);
    CHECK(glm::length(plain.muzzle_position() - (in.eye_local + plain.weapon.muzzle)) < 1e-4f);
}

TEST_CASE("animator: a hit makes the torso give and come back; the aim is not thrown", "[anim][animator]") {
    fx::SetFixture f;
    const Rig& rig = f.set.runtime->rig();
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
    for (bool own : { false, true }) {
        in.first_person = own;
        in.hit_seq = 0;
        CharacterAnimator a(f.set);
        for (int i = 0; i < 60; ++i) a.update(in);
        const glm::vec3 head0(a.model()[size_t(rig[Role::Head])][3]);
        const glm::vec3 aim0 = a.muzzle_direction();
        in.hit_seq = 1;
        in.hit_dir = glm::vec3(0.0f, 0.0f, 1.0f);          // from the front: travelling toward the back
        float peak = 0.0f, worst_aim = 0.0f;
        glm::vec3 at_peak(0.0f);
        for (int i = 0; i < 20; ++i) {
            a.update(in);
            REQUIRE(pose_is_valid(a.model()));
            const glm::vec3 d = glm::vec3(a.model()[size_t(rig[Role::Head])][3]) - head0;
            if (glm::length(d) > peak) { peak = glm::length(d); at_peak = d; }
            worst_aim = std::max(worst_aim, glm::length(a.muzzle_direction() - aim0));
        }
        INFO((own ? "own view" : "seen from outside") << ": head moved " << peak << " m, by " << at_peak.x << " " << at_peak.y << " " << at_peak.z);
        CHECK(peak > (own ? 0.005f : 0.02f));            // it shows
        CHECK(peak < 0.25f);                             // and is a flinch, not a fall
        CHECK(at_peak.z > 0.0f);                         // backward: the way the shot went
        CHECK(worst_aim < 1e-4f);                        // the weapon stays on the view
        for (int i = 0; i < 60; ++i) a.update(in);       // the same hit number again is not a new hit
        CHECK(glm::length(glm::vec3(a.model()[size_t(rig[Role::Head])][3]) - head0) < 0.01f);
    }
}

TEST_CASE("animator: the magazine is in the support hand through the middle of a reload only", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimator a(f.set);
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    for (int i = 0; i < 30; ++i) a.update(in);
    CHECK_FALSE(a.magazine_in_hand());
    in.reload_seq = 1;
    in.reloading = true;
    int with = 0, frames = 0;
    bool first = false, last = false;
    for (float t = 0.0f; t < a.weapon.reload_time + 0.5f; t += in.dt, ++frames) {
        a.update(in);
        if (frames == 2) first = a.magazine_in_hand();
        if (a.magazine_in_hand()) ++with;
        last = a.magazine_in_hand();
    }
    CHECK_FALSE(first);                                  // not while the hand is still on its way
    CHECK_FALSE(last);                                   // gone when the reload is over
    CHECK(with > 20);
    CHECK(with < frames * 2 / 3);
}

TEST_CASE("animator: aiming the rifle is up; at the ready it is lower and points the same way", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
    const glm::vec3 fwd(0.0f, 0.0f, -1.0f);
    for (bool own : { false, true }) {
        in.first_person = own;
        in.shot_seq = 0;
        in.aiming = true;
        CharacterAnimator a(f.set);
        for (int i = 0; i < 40; ++i) a.update(in);
        const glm::vec3 muzzle_aim = a.muzzle_position();
        CHECK(glm::length(a.muzzle_direction() - fwd) < 1e-3f);          // aiming: along the view
        // let go of the aim: the weapon comes down, without a jump, and goes on pointing where the view does
        in.aiming = false;
        float worst_step = 0.0f;
        glm::vec3 prev = a.muzzle_position();
        for (int i = 0; i < 60; ++i) {
            a.update(in);
            worst_step = std::max(worst_step, glm::length(a.muzzle_position() - prev));
            prev = a.muzzle_position();
        }
        INFO((own ? "own view" : "seen from outside") << ": muzzle " << a.muzzle_position().y - muzzle_aim.y << " m lower, direction y " << a.muzzle_direction().y);
        CHECK(a.muzzle_position().y < muzzle_aim.y - 0.04f);             // lower
        CHECK(glm::length(a.muzzle_direction() - fwd) < 1e-3f);          // and still on the view's line
        CHECK(worst_step < 0.07f);                                       // eased, not snapped (the muzzle travels 0.4 m)
        // a shot from the hip: level again at once, and it stays so for a moment
        in.shot_seq = 1;
        for (int i = 0; i < 14; ++i) a.update(in);
        CHECK(std::abs(a.muzzle_direction().y) < 0.05f);
        CHECK(a.muzzle_position().y < muzzle_aim.y - 0.03f);             // (still from the hip, not from the shoulder)
        for (int i = 0; i < 90; ++i) a.update(in);
        CHECK(glm::length(a.muzzle_direction() - fwd) < 1e-3f);          // and stays so
        // aim again: back where it was
        in.aiming = true;
        for (int i = 0; i < 60; ++i) a.update(in);
        CHECK(glm::length(a.muzzle_position() - muzzle_aim) < 2e-3f);
        CHECK(glm::length(a.muzzle_direction() - fwd) < 1e-3f);
    }
}

TEST_CASE("animator: a shot from the hip leaves a level weapon that points where the view does", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
    in.aiming = false;
    for (bool own : { false, true }) {
        in.first_person = own;
        in.shot_seq = 0;
        CharacterAnimator a(f.set);
        for (int i = 0; i < 120; ++i) a.update(in);
        // at the ready the weapon points where the view does, and that is where a shot leaves from
        CHECK(glm::length(a.muzzle_direction() - glm::vec3(0.0f, 0.0f, -1.0f)) < 1e-3f);
        const glm::mat4 V(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0), glm::vec4(in.eye_local, 1.0f));
        const glm::mat4 fire = V * a.weapon_fire_offset();
        const glm::vec3 fire_dir = glm::normalize(glm::mat3(fire) * glm::vec3(0.0f, 0.0f, -1.0f));
        const glm::vec3 fire_muzzle(fire * glm::vec4(a.weapon.muzzle, 1.0f));
        INFO((own ? "own view" : "seen from outside") << ": fire direction " << fire_dir.x << " " << fire_dir.y << " " << fire_dir.z);
        CHECK(glm::length(fire_dir - glm::vec3(0.0f, 0.0f, -1.0f)) < 1e-3f);
        // the shot: one update later the weapon that is drawn is there (but for the recoil's kick)
        in.shot_seq = 1;
        a.update(in);
        CHECK(glm::length(a.muzzle_direction() - glm::vec3(0.0f, 0.0f, -1.0f)) < a.weapon.recoil_pitch + 0.01f);
        CHECK(glm::length(a.muzzle_position() - fire_muzzle) < a.weapon.recoil_stroke + 0.03f);
    }
}

TEST_CASE("animator: not aiming, the head looks about - at somebody, to a side, nowhere - and stops for the aim", "[anim][animator]") {
    fx::SetFixture f;
    const Rig& rig = f.set.runtime->rig();
    const int head = rig[Role::Head];
    int changes = 0;
    std::vector<float> rates;
    auto run_one = [&](uint32_t variation, bool moving, int counts[4], float& widest, float& highest, float& lowest, float& fastest, float& look_err) {
        CharacterAnimator a(f.set);
        CharacterAnimationInput in;
        in.dt = 1.0f / 60.0f;
        in.ground_height = flat_ground;
        in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
        in.aiming = false;
        in.variation = variation;
        in.has_look_at = true;
        in.look_at = glm::vec3(3.0f, 1.6f, -3.0f);            // somebody ahead and to the right
        if (moving) { in.velocity = glm::vec3(0.0f, 0.0f, -8.0f); in.desired_velocity = in.velocity; }
        glm::quat prev(1.0f, 0.0f, 0.0f, 0.0f);
        std::vector<float> yaws;
        int last_mode = 0;
        for (int i = 0; i < 60 * 180; ++i) {
            if (moving) in.position += in.velocity * in.dt;
            a.update(in);
            REQUIRE(pose_is_valid(a.model()));
            ++counts[a.gaze_mode()];
            if (i > 0 && a.gaze_mode() != last_mode) ++changes;
            last_mode = a.gaze_mode();
            if (i > 0 && !moving) rates.push_back(std::abs(a.gaze_yaw() - yaws.back()) / in.dt);
            widest = std::max(widest, std::abs(a.gaze_yaw()));
            highest = std::max(highest, a.gaze_pitch());
            lowest = std::min(lowest, a.gaze_pitch());
            const glm::quat q = a.local()[size_t(head)].r;
            if (i > 0 && !moving) {
                const glm::quat d = glm::inverse(prev) * q;
                fastest = std::max(fastest, 2.0f * std::asin(std::min(glm::length(glm::vec3(d.x, d.y, d.z)), 1.0f)) / in.dt);
            }
            prev = q;
            if (a.gaze_mode() == 2) look_err = std::min(look_err, std::abs(a.gaze_yaw() - 0.785f));
            yaws.push_back(a.gaze_yaw());
        }
        // the aim comes up: the head is back on it within half a second
        in.aiming = true;
        for (int i = 0; i < 40; ++i) a.update(in);
        CHECK(std::abs(a.gaze_yaw()) < 0.03f);
        CHECK(std::abs(a.gaze_pitch()) < 0.03f);
        return yaws;
    };
    int c[4] = { 0, 0, 0, 0 };
    float widest = 0.0f, highest = 0.0f, lowest = 0.0f, fastest = 0.0f, look_err = 1e9f;
    const std::vector<float> a1 = run_one(7, false, c, widest, highest, lowest, fastest, look_err);
    const int changes_standing = changes;
    std::sort(rates.begin(), rates.end());
    const float median_rate = rates[rates.size() / 2], p95_rate = rates[rates.size() * 95 / 100];
    INFO("standing 180 s: at rest " << c[0] << ", to a side " << c[1] << ", looking at them " << c[2] << ", sweeping " << c[3] << " frames; " << changes_standing
         << " changes; yaw rate median " << median_rate << ", 95 % below " << p95_rate << " rad/s; widest " << widest
         << " rad, up " << highest << ", down " << lowest << ", head at most " << fastest << " rad/s, nearest to them " << look_err << " rad");
    CHECK(c[0] > 300);                 // nothing for a while ...
    CHECK(c[1] > 300);                 // ... a look to a side ...
    CHECK(c[2] > 300);                 // ... at the other player ...
    CHECK(c[3] > 300);                 // ... and a slow sweep, each some of the time
    CHECK(c[0] > c[1]);                // at rest more than anything else to a side
    CHECK(changes_standing >= 12);     // a new idea every few seconds ...
    CHECK(changes_standing <= 60);     // ... not every second (3 minutes)
    CHECK(median_rate < 0.08f);        // most of the time the head is all but still
    CHECK(p95_rate < 0.9f);            // and its turns are unhurried: 50 degrees a second at the outside, nearly always
    CHECK(widest > 0.4f);
    CHECK(widest <= 1.06f);
    CHECK(look_err < 0.06f);           // when it looks at them, it looks at them
    CHECK(fastest < 6.0f);             // even the quick ones are a turn of the head, not a snap
    // the same character again does the same; another one does not
    int c2[4] = { 0, 0, 0, 0 }; float w2 = 0, h2 = 0, l2 = 0, f2 = 0, e2 = 1e9f;
    const std::vector<float> a2 = run_one(7, false, c2, w2, h2, l2, f2, e2);
    CHECK(a1 == a2);
    int c3[4] = { 0, 0, 0, 0 }; float w3 = 0, h3 = 0, l3 = 0, f3 = 0, e3 = 1e9f;
    const std::vector<float> a3 = run_one(8, false, c3, w3, h3, l3, f3, e3);
    CHECK(a1 != a3);
    // up at the sky and down at the ground are rare: one of the two characters did each in three minutes
    CHECK(std::max(highest, h3) > 0.1f);
    CHECK(std::min(lowest, l3) < -0.08f);
    // on the move: still some of it, less far
    int c4[4] = { 0, 0, 0, 0 }; float w4 = 0, h4 = 0, l4 = 0, f4 = 0, e4 = 1e9f;
    run_one(7, true, c4, w4, h4, l4, f4, e4);
    INFO("moving: nothing " << c4[0] << ", glance " << c4[1] << ", looking " << c4[2] << "; widest " << w4);
    CHECK(c4[0] > c4[1] + c4[3]);      // mostly looking where it goes
    CHECK(c4[1] + c4[2] + c4[3] > 300);
}

TEST_CASE("animator: aiming brings the weapon to the middle of the player's own view only", "[anim][animator]") {
    fx::SetFixture f;
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
    // his own view: beside the eye at the ready, under the crosshair when aiming
    in.first_person = true;
    in.aiming = false;
    CharacterAnimator own(f.set);
    for (int i = 0; i < 60; ++i) own.update(in);
    CHECK(own.muzzle_position().x > 0.12f);
    in.aiming = true;
    for (int i = 0; i < 60; ++i) own.update(in);
    CHECK(std::abs(own.muzzle_position().x) < 0.03f);                       // on the centre line
    CHECK(own.muzzle_position().y < in.eye_local.y - 0.05f);                // below the eye: the sights, not the barrel, are at the crosshair
    CHECK(glm::length(own.muzzle_direction() - glm::vec3(0.0f, 0.0f, -1.0f)) < 1e-3f);
    // and a shot leaves from there
    const glm::mat4 V(glm::vec4(1, 0, 0, 0), glm::vec4(0, 1, 0, 0), glm::vec4(0, 0, 1, 0), glm::vec4(in.eye_local, 1.0f));
    CHECK(glm::length(glm::vec3(V * own.weapon_fire_offset() * glm::vec4(own.weapon.muzzle, 1.0f)) - own.muzzle_position()) < 1e-3f);
    // everybody else: in the shoulder, to the right of the centre line, as before
    in.first_person = false;
    CharacterAnimator seen(f.set);
    for (int i = 0; i < 60; ++i) seen.update(in);
    CHECK(seen.muzzle_position().x > 0.08f);
}

TEST_CASE("animator: dying, the body buckles the way the shot went, without a jump", "[anim][animator]") {
    fx::SetFixture f;
    const Rig& rig = f.set.runtime->rig();
    for (float from_front : { 1.0f, -1.0f }) {
        CharacterAnimator a(f.set);
        CharacterAnimationInput in;
        in.dt = 1.0f / 60.0f;
        in.ground_height = flat_ground;
        in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
        for (int i = 0; i < 60; ++i) a.update(in);
        const glm::vec3 hips0(a.model()[size_t(rig[Role::Hips])][3]), head0(a.model()[size_t(rig[Role::Head])][3]);
        const glm::vec3 foot0(a.model()[size_t(rig[Role::LeftFoot])][3]);
        in.hit_dir = glm::vec3(0.0f, 0.0f, from_front);        // +z: the shot travelled toward their back (it came from the front)
        glm::vec3 prev = head0;
        float worst_step = 0.0f;
        const int frames = 23;                                  // 0.38 s
        for (int i = 1; i <= frames; ++i) {
            in.dying = float(i) / float(frames);
            a.update(in);
            REQUIRE(pose_is_valid(a.model()));
            const glm::vec3 h(a.model()[size_t(rig[Role::Head])][3]);
            worst_step = std::max(worst_step, glm::length(h - prev));
            prev = h;
        }
        const glm::vec3 hips1(a.model()[size_t(rig[Role::Hips])][3]), head1(a.model()[size_t(rig[Role::Head])][3]);
        const glm::vec3 foot1(a.model()[size_t(rig[Role::LeftFoot])][3]);
        INFO((from_front > 0.0f ? "shot from the front" : "shot from behind") << ": hips " << hips1.y - hips0.y << " m, head by "
             << head1.x - head0.x << " " << head1.y - head0.y << " " << head1.z - head0.z << ", worst step " << worst_step);
        CHECK(hips1.y < hips0.y - 0.25f);                       // the legs gave
        CHECK(head1.y < head0.y - 0.30f);                       // the head went down with them, and more
        CHECK((head1.z - head0.z) * from_front > 0.05f);        // backward when shot from the front, forward when from behind
        CHECK(glm::length(glm::vec2(foot1.x - foot0.x, foot1.z - foot0.z)) < 0.08f);   // on feet that stayed where they stood
        CHECK(worst_step < 0.06f);                              // a fall, not a cut
        CHECK(a.muzzle_position().y < 1.25f);                   // the weapon is out of the aim
    }
}

TEST_CASE("real asset: the face blinks, looks and lets go", "[anim][animator][asset]") {
    AnimationSet set;
    std::string why;
    if (!set.load((std::string(VIBE_SOURCE_DIR) + "/assets/character/soldier.qanim").c_str(), &why)) {
        WARN("assets/character/soldier.qanim not loaded (" << why << "): skipped");
        return;
    }
    REQUIRE(set.face.ok);
    REQUIRE(set.face.jaw >= 0);
    const Rig& rig = set.runtime->rig();
    CharacterAnimator a(set);
    CharacterAnimationInput in;
    in.dt = 1.0f / 60.0f;
    in.ground_height = flat_ground;
    in.eye_local = glm::vec3(0.0f, 1.6f, -0.11f);
    in.aiming = false;
    in.variation = 7;
    // how far a face joint is turned against the head it sits on
    auto turned = [&](int joint) {
        const glm::quat h = glm::quat_cast(glm::mat3(a.model()[size_t(rig[Role::Head])]));
        const glm::quat j = glm::quat_cast(glm::mat3(a.model()[size_t(joint)]));
        const glm::quat d = glm::inverse(h) * j;
        return 2.0f * std::acos(std::clamp(std::abs(d.w), 0.0f, 1.0f));
    };
    for (int i = 0; i < 30; ++i) a.update(in);
    const float lid_open = turned(set.face.lid_up[0]), jaw_shut = turned(set.face.jaw);
    int open_frames = 0, frames = 20 * 60;
    float lid_most = 0.0f, lid_turn_most = 0.0f, eye_most = 0.0f;
    for (int i = 0; i < frames; ++i) {
        a.update(in);
        REQUIRE(pose_is_valid(a.model()));
        if (a.lids() < 0.05f) ++open_frames;
        lid_most = std::max(lid_most, a.lids());
        lid_turn_most = std::max(lid_turn_most, std::abs(turned(set.face.lid_up[0]) - lid_open));
        eye_most = std::max(eye_most, glm::length(a.eyes()));
        CHECK(std::abs(a.eyes().x) <= 0.551f);
        CHECK(a.eyes().y <= 0.301f);
        CHECK(a.eyes().y >= -0.351f);
    }
    INFO("blinks " << a.blinks() << " in 20 s, eyes open " << 100.0f * float(open_frames) / float(frames) << " % of the time, lid turned "
         << lid_turn_most << " rad at most, eyes off centre " << eye_most << " rad at most");
    CHECK(a.blinks() >= 3);                                 // (people: 10 - 20 a minute)
    CHECK(a.blinks() <= 12);
    CHECK(float(open_frames) > 0.90f * float(frames));
    CHECK(lid_most > 0.9f);
    CHECK(lid_turn_most > 0.3f);                            // the joint itself moved, not only the number
    CHECK(eye_most > 0.02f);                                // the eyes are not painted on
    // killed: the eyes close and the jaw hangs
    for (int i = 1; i <= 23; ++i) { in.dying = float(i) / 23.0f; a.update(in); }
    for (int i = 0; i < 10; ++i) a.update(in);
    CHECK(a.lids() > 0.8f);
    CHECK(a.jaw() > 0.15f);
    CHECK(turned(set.face.jaw) > jaw_shut + 0.12f);
    // seen from inside there is no face to move
    CharacterAnimator b(set);
    in.dying = 0.0f;
    in.first_person = true;
    for (int i = 0; i < 600; ++i) b.update(in);
    CHECK(b.blinks() == 0);
}
