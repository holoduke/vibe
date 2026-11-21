// Remote players: the replicated state, its wire form, buffering under bad network conditions,
// event handling, and the pose an observer reconstructs from it - the same code the game runs
// (net_state.h, net_anim.h, CharacterAnimator), with the network replaced by a model of one
// (latency, jitter, loss, duplication, reordering, stalls).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "engine/net/net_anim.h"
#include "fixtures.h"

#include <functional>
#include <random>

using namespace qlike;
using namespace qlike::anim;
using namespace qlike::net;
using Catch::Matchers::WithinAbs;

namespace {

constexpr float kPi_ = 3.14159265f;
float flat_ground(void*, float, float) { return 0.0f; }

// The sending player: the real controller, driven by a script.
struct Sender {
    game::Player p;
    std::vector<collision::AABB> world{ { glm::vec3(-4000.0f, -1.0f, -4000.0f), glm::vec3(4000.0f, 0.0f, 4000.0f) } };
    game::PlayerInput pin;
    uint16_t shots = 0;
    bool reloading = false, torch = false, teleport = false;
    uint8_t weapon = 0;
    int jumps = 0;
    bool was_ground = true;
    Sender() { p.position = glm::vec3(0.0f, 0.9f, 0.0f); p.on_ground = true; }
    void tick(float dt) {
        game::update_player(p, pin, world, dt);
        if (was_ground && !p.on_ground) ++jumps;
        was_ground = p.on_ground;
    }
    PlayerState state() const {
        PlayerState s;
        s.px = p.position.x; s.py = p.position.y; s.pz = p.position.z;
        s.vx = p.velocity.x; s.vy = p.velocity.y; s.vz = p.velocity.z;
        s.yaw = p.yaw; s.pitch = p.pitch; s.crouch = p.crouch_factor;
        s.flags = uint8_t((p.on_ground ? kFlagGround : 0) | (torch ? kFlagTorch : 0) | (reloading ? kFlagReload : 0) | (teleport ? kFlagTeleport : 0) | kFlagAim);     // (aiming: the weapon is on the view)
        s.weapon = weapon;
        s.move = move_bits(pin);
        s.shot_seq = shots;
        return s;
    }
};

struct LinkParams { float latency_ms = 0, jitter_ms = 0, loss = 0, duplicate = 0; };

struct Packet { float arrive; uint32_t t; PackedState s; };

struct Result {
    int   frames = 0, invalid = 0, extrapolated = 0, teleports = 0;
    float worst_step = 0.0f;          // metres the shown position moved in one frame beyond what its speed explains
    float worst_aim = 0.0f;           // radians between the shown muzzle direction and the sender's aim at that moment
    float worst_hand = 0.0f;
    int   recoils = 0, flights = 0;
    uint32_t dropped = 0;
    float delay_ms = 0.0f;
    std::string end_clip;
};

using SendScript = std::function<void(float t, Sender& s)>;

// Sender at 120 Hz, states at 30 Hz through the link, an observer at 60 fps.
Result observe(const fx::SetFixture& f, const LinkParams& link, float seconds, const SendScript& script, unsigned seed = 1,
               float stall_from = -1.0f, float stall_to = -1.0f) {
    Result r;
    Sender snd;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<Packet> wire;
    SnapBuffer buf;
    CharacterAnimator anim(f.set);
    // what the sender really did, for comparison: (time ms, yaw, pitch)
    struct Truth { float t, yaw, pitch; };
    std::vector<Truth> truth;
    const float tick = 1.0f / 120.0f, frame = 1.0f / 60.0f, send_every = 1.0f / 30.0f;
    float next_send = 0.0f, sim_t = 0.0f;
    glm::vec3 last_pos(0.0f);
    bool have_last = false, was_recoil = false, was_reloading = false;
    uint32_t reload_seq = 0;
    const Rig& rig = f.set.runtime->rig();
    for (float now = 0.0f; now < seconds; now += frame) {
        while (sim_t < now) {
            script(sim_t, snd);
            snd.tick(tick);
            sim_t += tick;
            truth.push_back({ sim_t * 1000.0f, snd.p.yaw, snd.p.pitch });
            if (sim_t >= next_send) {
                next_send += send_every;
                const bool stalled = sim_t >= stall_from && sim_t < stall_to;
                const int copies = (u01(rng) < link.loss || stalled) ? 0 : (u01(rng) < link.duplicate ? 2 : 1);
                for (int c = 0; c < copies; ++c)
                    wire.push_back({ sim_t * 1000.0f + link.latency_ms + u01(rng) * link.jitter_ms, uint32_t(sim_t * 1000.0f), pack_state(snd.state()) });
                snd.teleport = false;
            }
        }
        const uint32_t now_ms = uint32_t(now * 1000.0f);
        for (size_t i = 0; i < wire.size();) {                     // deliver in arrival order (the vector is not sorted: reordering)
            if (wire[i].arrive <= now * 1000.0f) { buf.add(wire[i].t, unpack_state(wire[i].s), now_ms); wire.erase(wire.begin() + ptrdiff_t(i)); }
            else ++i;
        }
        const uint32_t delay = buf.delay_ms();
        const uint32_t rt = now_ms > delay ? now_ms - delay : 0u;
        const SnapBuffer::Sample smp = buf.sample(rt);
        if (!smp.valid) continue;
        ++r.frames;
        r.extrapolated += smp.extrapolated;
        r.teleports += smp.teleported;
        CharacterAnimationInput in;
        const RemoteFrame rf = remote_animation_input(smp.s, in);
        in.dt = frame;
        in.ground_height = flat_ground;
        if (in.reloading && !was_reloading) ++reload_seq;
        was_reloading = in.reloading;
        in.reload_seq = reload_seq;
        in.teleported = smp.teleported || !have_last;
        anim.update(in);
        if (!pose_is_valid(anim.model()) || anim.debug().invalid_poses) ++r.invalid;
        // smoothness of the shown position
        const glm::vec3 pos(smp.s.px, smp.s.py, smp.s.pz);
        if (have_last && !smp.teleported) {
            const float sp = glm::length(glm::vec3(smp.s.vx, smp.s.vy, smp.s.vz));
            r.worst_step = std::max(r.worst_step, glm::length(pos - last_pos) - (sp + 4.0f) * frame * 1.5f);
        }
        last_pos = pos;
        have_last = true;
        // the weapon on the sender's aim at the moment shown
        if (!smp.extrapolated && !truth.empty() && in.held == HeldItem::Rifle) {
            size_t k = 0;
            while (k + 1 < truth.size() && truth[k + 1].t <= float(rt)) ++k;
            const Truth& tr = truth[k];
            const glm::vec3 aim(std::sin(tr.yaw) * std::cos(tr.pitch), std::sin(tr.pitch), -std::cos(tr.yaw) * std::cos(tr.pitch));
            const glm::vec3 shown = glm::mat3(rf.model) * anim.muzzle_direction();
            r.worst_aim = std::max(r.worst_aim, std::acos(std::clamp(glm::dot(aim, shown), -1.0f, 1.0f)));
            for (int side = 0; side < 2; ++side) {
                const int hd = rig[side == 0 ? Role::LeftHand : Role::RightHand], fa = rig[side == 0 ? Role::LeftForeArm : Role::RightForeArm],
                          ua = rig[side == 0 ? Role::LeftArm : Role::RightArm];
                const glm::vec3 ph(anim.model()[size_t(hd)][3]), pf(anim.model()[size_t(fa)][3]), pu(anim.model()[size_t(ua)][3]);
                const float reach = glm::length(pf - pu) + glm::length(ph - pf);
                if (glm::length(anim.debug().hand_target[side] - pu) <= reach * 0.98f)
                    r.worst_hand = std::max(r.worst_hand, glm::length(ph - anim.debug().hand_target[side]));
            }
        }
        const bool recoil = anim.debug().recoil > 0.0f;
        // (a new kick while the last is still running restarts it: count rises of the curve's start)
        if (recoil && (!was_recoil || anim.debug().recoil < 0.2f) && !was_recoil) ++r.recoils;
        was_recoil = recoil;
    }
    r.flights = int(anim.debug().flights);
    r.dropped = buf.dropped;
    r.delay_ms = float(buf.delay_ms());
    r.end_clip = anim.debug().clip >= 0 ? f.asset.clips[size_t(anim.debug().clip)].name : "none";
    return r;
}

// Sprint, a long jump with a 180 degree turn in the air, shots, landing, another jump at once,
// then strafing while looking 90 and 180 degrees away from the travel direction.
void combat_script(float t, Sender& s) {
    s.pin = game::PlayerInput{};
    if (t < 3.0f) {
        s.pin.fwd = true; s.pin.sprint = true;
        s.pin.jump = (t > 1.0f && t < 1.1f) || (t > 1.75f && t < 1.85f);
        if (t > 1.3f && t < 1.5f) s.p.yaw = kPi_ * (t - 1.3f) / 0.2f;          // half a turn in 0.2 s, in the air
        if (t > 1.4f && t < 2.6f) s.shots = uint16_t(1 + int((t - 1.4f) * 5.0f));   // five shots a second
    } else if (t < 5.0f) {
        s.pin.right = true;                                                  // strafing, the view 90 degrees off the path
        s.p.pitch = 0.5f * std::sin(t * 2.0f);
    } else if (t < 6.5f) {
        s.pin.back = true;                                                   // running one way, looking the other
    }
}

}  // namespace

TEST_CASE("net: wire form of a player state", "[net]") {
    CHECK(sizeof(PackedState) == 35);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> ang(-20.0f, 20.0f), pit(-1.55f, 1.55f), pos(-90000.0f, 90000.0f), vel(-2000.0f, 2000.0f);
    float worst_yaw = 0.0f, worst_pitch = 0.0f;
    for (int i = 0; i < 5000; ++i) {
        PlayerState s;
        s.px = pos(rng); s.py = pos(rng); s.pz = pos(rng); s.vx = vel(rng); s.vy = vel(rng); s.vz = vel(rng);
        s.yaw = ang(rng); s.pitch = pit(rng); s.crouch = float(i % 11) / 10.0f;
        s.flags = uint8_t(i); s.health = uint8_t(i % 101); s.weapon = uint8_t(i % 2); s.move = uint8_t(i * 7); s.shot_seq = uint16_t(i * 13);
        const PlayerState o = unpack_state(pack_state(s));
        REQUIRE(o.px == s.px); REQUIRE(o.py == s.py); REQUIRE(o.pz == s.pz);         // positions and velocities are exact
        REQUIRE(o.vx == s.vx); REQUIRE(o.vy == s.vy); REQUIRE(o.vz == s.vz);
        REQUIRE(o.flags == s.flags); REQUIRE(o.health == s.health); REQUIRE(o.weapon == s.weapon);
        REQUIRE(o.move == s.move); REQUIRE(o.shot_seq == s.shot_seq);
        worst_yaw = std::max(worst_yaw, std::abs(wrap_angle(o.yaw - s.yaw)));
        worst_pitch = std::max(worst_pitch, std::abs(o.pitch - s.pitch));
        REQUIRE(std::abs(o.crouch - s.crouch) <= 0.5f / 255.0f + 1e-6f);
    }
    // Aim precision: half a quantisation step. At the muzzle (0.72 m from the eye) that is
    // under a twentieth of a millimetre; at 100 m, under 5 mm.
    CHECK(worst_yaw <= 6.2831853f / 65536.0f * 0.5f + 1e-6f);
    CHECK(worst_pitch <= 1.5707963f / 32767.0f * 0.5f + 1e-6f);
    CHECK(worst_yaw * 0.72f < 0.00005f);
    CHECK(worst_yaw * 100.0f < 0.005f);
    // looking straight up / down and angles past a full turn survive
    PlayerState s;
    s.pitch = 1.5607963f; s.yaw = -7.0f * 6.2831853f + 0.3f;
    const PlayerState o = unpack_state(pack_state(s));
    CHECK_THAT(o.pitch, WithinAbs(s.pitch, 1e-4));
    CHECK_THAT(wrap_angle(o.yaw - 0.3f), WithinAbs(0.0f, 1e-4));
}

TEST_CASE("net: states that are not sane never enter the buffer", "[net]") {
    SnapBuffer b;
    PlayerState s;
    CHECK(b.add(100, s, 100));
    PlayerState bad = s;
    bad.px = std::nanf("");
    CHECK_FALSE(b.add(200, bad, 200));
    bad = s; bad.vy = INFINITY;
    CHECK_FALSE(b.add(200, bad, 200));
    bad = s; bad.pz = 1e9f;
    CHECK_FALSE(b.add(200, bad, 200));
    bad = s; bad.health = 200;
    CHECK_FALSE(b.add(200, bad, 200));
    CHECK(b.snaps.size() == 1);
    CHECK(b.dropped == 4);
    CHECK_FALSE(b.add(100, s, 300));           // duplicate
    CHECK_FALSE(b.add(60, s, 300));            // out of order
    CHECK(b.add(133, s, 300));
    CHECK(b.add(5, s, 100000 + 300) == false); // (still in the past of this clock)
    for (uint32_t i = 0; i < 200; ++i) b.add(200 + i * 33, s, 200 + i * 33);
    CHECK(b.snaps.size() <= 64);
}

TEST_CASE("net: sampling - interpolation, bounded extrapolation, teleports", "[net]") {
    SnapBuffer b;
    PlayerState a, c;
    a.px = 0.0f; a.vx = 6.0f; a.yaw = 6.2f; a.shot_seq = 3;
    c.px = 0.2f; c.vx = 6.0f; c.yaw = 0.1f; c.shot_seq = 4; c.flags = kFlagGround;
    b.add(1000, a, 1000);
    b.add(1033, c, 1033);
    SnapBuffer::Sample s = b.sample(1016);
    CHECK_FALSE(s.extrapolated);
    CHECK_THAT(s.s.px, WithinAbs(0.097f, 0.01));
    CHECK(std::abs(wrap_angle(s.s.yaw - 6.2f)) < 0.2f);            // the short way round through 2 pi
    CHECK(s.s.shot_seq == 4);                                       // a shot between two states is not lost
    CHECK(b.sample(900).s.px == 0.0f);
    // past the newest state: along the velocity, for a quarter second at most, then standing
    s = b.sample(1133);
    CHECK(s.extrapolated);
    CHECK_THAT(s.s.px, WithinAbs(0.2f + 6.0f * 0.1f, 1e-3));
    s = b.sample(1033 + 5000);
    CHECK_THAT(s.s.px, WithinAbs(0.2f + 6.0f * 0.25f, 1e-3));
    CHECK(s.s.vx == 0.0f);
    CHECK(s.s.move == 0);
    // a respawn: no slide across the map, and it is said
    PlayerState far = c;
    far.px = 300.0f;
    b.add(1066, far, 1066);
    s = b.sample(1050);
    CHECK(s.teleported);
    CHECK(s.s.px == 300.0f);
    // fast flight is movement, not a teleport
    SnapBuffer fl;
    PlayerState f0, f1;
    f0.vx = f1.vx = 2000.0f; f1.px = 66.0f;
    fl.add(0, f0, 0); fl.add(33, f1, 33);
    CHECK_FALSE(fl.sample(16).teleported);
    // the flag alone says so too
    SnapBuffer tf;
    PlayerState t0, t1;
    t1.px = 2.0f; t1.flags = kFlagTeleport;
    tf.add(0, t0, 0); tf.add(33, t1, 33);
    CHECK(tf.sample(16).teleported);
    // a restarted clock (new host) starts the buffer again
    CHECK(b.add(100000, a, 100000));
    CHECK(b.add(10, a, 10));
    CHECK(b.snaps.size() == 1);
}

TEST_CASE("net: the interpolation delay follows the link", "[net]") {
    SnapBuffer b;
    PlayerState s;
    for (uint32_t i = 0; i < 60; ++i) b.add(i * 33, s, i * 33 + 5);
    CHECK(b.delay_ms() == kInterpDelayMs);                          // a LAN: the minimum
    for (uint32_t i = 60; i < 120; ++i) b.add(i * 33, s, i * 33 + 200);
    CHECK(b.delay_ms() >= 250);                                     // a 200 ms link: behind it
    CHECK(b.delay_ms() <= kMaxInterpDelayMs);
    for (uint32_t i = 120; i < 2000; ++i) b.add(i * 33, s, i * 33 + 5);
    CHECK(b.delay_ms() < 130);                                      // and back, slowly
    b.add(2000 * 33, s, 2000 * 33 + 100000);
    CHECK(b.delay_ms() == kMaxInterpDelayMs);                       // never further back than the cap
}

TEST_CASE("net: events - duplicates dropped, order kept, shots wait for the picture", "[net]") {
    EventGate g;
    std::vector<Event> out;
    auto fire = [](uint32_t from, uint16_t seq, uint32_t t) { Event e{}; e.kind = Event::Fire; e.from = from; e.seq = seq; e.t = t; return e; };
    const auto rt = [](uint32_t now) { return [now](uint32_t) { return now; }; };
    CHECK(g.push(fire(7, 1, 1000)));
    CHECK(g.push(fire(7, 2, 1100)));
    CHECK_FALSE(g.push(fire(7, 2, 1100)));       // duplicate
    CHECK_FALSE(g.push(fire(7, 1, 1000)));       // stale repeat
    CHECK(g.push(fire(9, 1, 1050)));             // another sender has its own numbering
    CHECK(g.duplicates == 2);
    g.release(rt(990), out);
    CHECK(out.empty());                           // the picture has not reached any of them
    g.release(rt(1060), out);
    REQUIRE(out.size() == 2);
    CHECK(out[0].from == 7); CHECK(out[0].seq == 1);
    CHECK(out[1].from == 9);
    g.release(rt(1100), out);
    REQUIRE(out.size() == 3);
    CHECK(out[2].seq == 2);
    CHECK(g.waiting() == 0);
    // hits and deaths are gameplay: never held back
    Event h{}; h.kind = Event::Hit; h.from = 7; h.to = 9; h.t = 99999;
    Event d{}; d.kind = Event::Death; d.from = 9; d.t = 99999;
    g.push(h); g.push(d);
    out.clear();
    g.release(rt(0), out);
    CHECK(out.size() == 2);
    // clocks that disagree wildly do not hold a shot for ever
    CHECK(g.push(fire(7, 3, 50000)));
    out.clear();
    g.release(rt(1200), out);
    CHECK(out.size() == 1);
    // the sequence number wraps
    CHECK(g.push(fire(7, 30000, 1900)));
    CHECK(g.push(fire(7, 60000, 1950)));
    CHECK(g.push(fire(7, 65535, 2000)));
    CHECK(g.push(fire(7, 0, 2010)));
    CHECK(g.push(fire(7, 1, 2020)));
    CHECK_FALSE(g.push(fire(7, 65535, 2000)));
    // per-sender picture time: a slow sender's shots wait longer
    EventGate g2;
    g2.push(fire(1, 1, 1000)); g2.push(fire(2, 1, 1000));
    out.clear();
    g2.release([](uint32_t from) { return from == 1 ? 1000u : 800u; }, out);
    REQUIRE(out.size() == 1);
    CHECK(out[0].from == 1);
    // someone who left and comes back starts numbering again
    g.forget(7);
    CHECK(g.push(fire(7, 1, 3000)));
}

TEST_CASE("net: an observer sees the full body, on the aim, through the whole fight", "[net][animator]") {
    fx::SetFixture f;
    const Result r = observe(f, LinkParams{ 5.0f, 3.0f, 0.0f, 0.0f }, 7.5f, combat_script);
    INFO("aim " << r.worst_aim << " rad, hand " << r.worst_hand << " m, step " << r.worst_step << " m, recoils " << r.recoils
         << ", flights " << r.flights << ", delay " << r.delay_ms << " ms");
    CHECK(r.frames > 400);
    CHECK(r.invalid == 0);
    CHECK(r.worst_hand < 0.02f);
    // The weapon follows the replicated aim: off by the recoil climb plus how far the aim moves
    // between two states (the half turn in the air is 15.7 rad/s; elsewhere under a degree).
    CHECK(r.worst_aim < 0.6f);
    CHECK(r.worst_step <= 0.0f);                  // the shown position never jumps
    CHECK(r.flights == 2);                        // both jumps seen as flights
    CHECK(r.recoils >= 5);                        // the shots kicked (6 fired; two close together may merge)
    CHECK(r.end_clip == "idle");
}

TEST_CASE("net: the aim is exact when it is steady or turns slowly", "[net][animator]") {
    fx::SetFixture f;
    const Result r = observe(f, LinkParams{ 20.0f, 10.0f, 0.0f, 0.0f }, 5.0f, [](float t, Sender& s) {
        s.pin = game::PlayerInput{};
        s.pin.right = true;
        s.p.yaw = 1.0f + 0.5f * std::sin(t * 1.5f);            // up to 0.75 rad/s
        s.p.pitch = 0.6f * std::sin(t * 1.1f);
    });
    INFO("aim " << r.worst_aim << " rad");
    CHECK(r.invalid == 0);
    CHECK(r.worst_aim < 0.02f);                   // about a degree, at the muzzle 1.4 cm
    CHECK(r.worst_hand < 0.02f);
}

TEST_CASE("net: latency, jitter, loss, duplication and reordering", "[net][animator]") {
    fx::SetFixture f;
    for (float latency : { 0.0f, 50.0f, 100.0f, 200.0f }) {
        for (float loss : { 0.0f, 0.02f, 0.05f, 0.10f }) {
            LinkParams link{ latency, 0.4f * latency + 10.0f, loss, 0.03f };
            const Result r = observe(f, link, 7.5f, combat_script, unsigned(latency) * 7u + unsigned(loss * 100.0f) + 1u);
            INFO("latency " << latency << " ms, loss " << loss * 100.0f << "%: delay " << r.delay_ms << " ms, extrapolated frames "
                 << r.extrapolated << "/" << r.frames << ", step " << r.worst_step << " m, flights " << r.flights << ", recoils " << r.recoils
                 << ", dropped " << r.dropped);
            CHECK(r.invalid == 0);
            CHECK(r.worst_hand < 0.02f);
            CHECK(r.dropped > 0);                         // duplicates and late packets were thrown away, not played
            CHECK(r.flights >= 1);                        // (a short second hop can fall between lost states at 10 % loss)
            CHECK(r.flights <= 2);
            CHECK(r.recoils >= 4);                        // shots are seen through the shot counter even when states are lost
            CHECK(r.end_clip == "idle");
            // the picture stays behind the data: hardly ever guessed, and never far
            CHECK(r.extrapolated * 5 < r.frames);
            CHECK(r.worst_step < 0.5f);
            CHECK(r.delay_ms >= latency);
        }
    }
}

TEST_CASE("net: a stall, then recovery", "[net][animator]") {
    fx::SetFixture f;
    const Result r = observe(f, LinkParams{ 30.0f, 10.0f, 0.0f, 0.0f }, 7.5f, combat_script, 3, 3.5f, 4.7f);      // 1.2 s without a packet
    INFO("extrapolated " << r.extrapolated << ", step " << r.worst_step);
    CHECK(r.invalid == 0);
    CHECK(r.extrapolated > 30);
    // while nothing arrives the player runs on for a quarter second and stands; when data returns it
    // catches up over the following states (strafing at 8 m/s for a second is 8 m: under the
    // teleport threshold, so it is a fast slide of one interpolation interval, not a snap of the pose)
    CHECK(r.worst_hand < 0.02f);
    CHECK(r.end_clip == "idle");
}

TEST_CASE("net: respawn, late join, weapon switch and reload as seen by an observer", "[net][animator]") {
    fx::SetFixture f;
    // respawn far away at t = 2
    Result r = observe(f, LinkParams{ 10.0f, 5.0f, 0.02f, 0.0f }, 4.0f, [](float t, Sender& s) {
        s.pin = game::PlayerInput{};
        s.pin.fwd = true;
        static bool done = false;
        if (t < 0.1f) done = false;
        if (t >= 2.0f && !done) { done = true; s.p.position = glm::vec3(500.0f, 0.9f, -300.0f); s.p.velocity = glm::vec3(0.0f); s.teleport = true; }
    });
    CHECK(r.invalid == 0);
    CHECK(r.teleports >= 1);
    CHECK(r.worst_step <= 0.0f);                  // the jump itself is flagged, nothing else jumps
    // late join: the first thing ever seen of them is the middle of a jump
    SnapBuffer b;
    PlayerState s;
    s.py = 3.0f; s.vy = 4.0f; s.vz = -9.0f; s.move = kMoveFwd; s.shot_seq = 4000;
    b.add(50000, s, 50010);
    CharacterAnimator a(f.set);
    CharacterAnimationInput in;
    remote_animation_input(b.sample(50005).s, in);
    in.dt = 1.0f / 60.0f; in.teleported = true;
    for (int i = 0; i < 20; ++i) { a.update(in); REQUIRE(pose_is_valid(a.model())); in.teleported = false; }
    CHECK(a.debug().recoil == 0.0f);              // their earlier shots are history, not a kick now
    CHECK(a.debug().air_phase != AirPhase::Ground);
    // tablet / torch / reload flags reach the animator
    s.weapon = 1;
    remote_animation_input(s, in);
    CHECK(in.held == HeldItem::Tablet);
    s.flags = kFlagTorch;
    remote_animation_input(s, in);
    CHECK(in.held == HeldItem::Tablet);            // (a torch is not drawn for others: what they carry besides it is)
    s.weapon = 0; s.flags = kFlagReload | kFlagGround;
    remote_animation_input(s, in);
    CHECK(in.held == HeldItem::Rifle);
    CHECK(in.reloading);
    CHECK(in.grounded);
    CHECK_FALSE(in.aiming);                        // at the ready unless they hold the aim
    s.flags = kFlagAim | kFlagGround;
    remote_animation_input(s, in);
    CHECK(in.aiming);
    CHECK_FALSE(in.reloading);
    CHECK_FALSE(in.first_person);
    // their camera looks where they look
    s.yaw = 0.7f; s.pitch = -0.4f;
    const RemoteFrame rf = remote_animation_input(s, in);
    const glm::vec3 look = -glm::vec3(rf.view_inv[2]);
    CHECK_THAT(look.y, WithinAbs(std::sin(-0.4f), 1e-5));
    CHECK_THAT(look.x, WithinAbs(std::sin(0.7f) * std::cos(0.4f), 1e-5));
    // and the wish direction is rebuilt from their keys in their frame
    s.move = kMoveRight | kMoveSprint;
    remote_animation_input(s, in);
    CHECK_THAT(in.desired_velocity.x, WithinAbs(std::cos(0.7f) * 12.0f, 1e-4));
    CHECK_THAT(in.desired_velocity.z, WithinAbs(std::sin(0.7f) * 12.0f, 1e-4));
}

TEST_CASE("net: what is sent holds no pose - bytes per player per second", "[net]") {
    // State: 35 bytes, 30 times a second from each client; the host relays 60 snapshots a second.
    // (ENet adds its own headers; these are payload sizes, the ones Session::stats() counts.)
    const size_t state_msg = 1 + 4 + 4 + sizeof(PackedState);
    const size_t snap_entry = 4 + 4 + sizeof(PackedState);
    CHECK(state_msg == 44);
    CHECK(snap_entry == 43);
    CHECK(state_msg * 30 == 1320);                // bytes a client sends per second
    CHECK(sizeof(Event) == 43);
    // no field of the state or of an event can carry a bone: the largest is three floats
    CHECK(sizeof(PackedState) < 53 * 4);          // less than one float per joint
}
