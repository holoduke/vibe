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

using namespace qlike;
using namespace qlike::anim;
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
    if (!set.load((std::string(QLIKE_SOURCE_DIR) + "/assets/character/soldier.qanim").c_str(), &why)) {
        WARN("assets/character/soldier.qanim not loaded (" << why << "): skipped");
        return;
    }
    // (the capture's pace per direction is what the request is mapped onto)
    CHECK(set.db.cruise_speed(false, 0) > 1.5f);
    CHECK(set.db.cruise_speed(false, 0) < 3.3f);
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
    CHECK(glm::length(own.muzzle_position() - (in.eye_local + own.weapon.muzzle)) < 1e-4f);
    CHECK(glm::length(glm::vec3(own.weapon_offset()[3])) < 1e-5f);
    // and with the switch off, third person is the same as that
    CharacterAnimator plain(f.set);
    plain.tuning.shoulder_weapon = false;
    in.first_person = false;
    for (int i = 0; i < 40; ++i) plain.update(in);
    CHECK(glm::length(plain.muzzle_position() - (in.eye_local + plain.weapon.muzzle)) < 1e-4f);
}
