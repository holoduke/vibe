// The NPC engine (src/npc): people and cars that live their own life, with no renderer.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "npc/npc.h"

using namespace vibe;

namespace {

// A straight north-south road at x = 40 that closes far away: 2 km round, a "ring" of two
// straights 6 m apart is not needed - only that s wraps.
npc::Road ring_road() {
    npc::Road r;
    r.length = 2000.0f;
    r.at = [](float s, glm::vec3& pos, glm::vec2& dir) {
        // a circle of 2 km round whose west-most point lies at (40, 0): there it runs toward +z
        const float R = 2000.0f / 6.2831853f;
        const float a = s / R;
        pos = glm::vec3(40.0f + R - R * std::cos(a), 0.0f, R * std::sin(a));
        dir = glm::vec2(std::sin(a), std::cos(a));
        return true;
    };
    return r;
}

// A small site west of that road: a shop node chain, seats, spots, two pump stops in one lane,
// bays on an aisle, ways in and out.
const char* kSite =
    "node w_a -6 0 0\nnode w_b -6 0 10\nnode shop -12 0 10\nnode till -14 0 6\nnode fc 3 0 0\nnode pump13a 10 0 -3\nnode pump13b 10 0 2\n"
    "link w_a w_b\nlink w_b shop\nlink shop till\nlink w_a fc\nlink fc pump13a\nlink fc pump13b\n"
    "seat booth -13 0.5 12 0\nseat booth -13 0.5 13 3.1416\nseat stool -15 0.8 11 1.57\nseat picnic -8 0.5 -6 0\nseat bench -7 0.5 -8 0\nseat chair -14 0.5 14 0\n"
    "spot browse -12 0 8 0\nspot browse -11 0 8 0\nspot pay -14 0 6 1.57\nspot cashier -16 0 6 -1.57\nspot cook -18 0 12 0\nspot serve -16 0 12 0\n"
    "bay -1.25 0 -4 -1.5708\nbay -1.25 0 0 -1.5708\nbay -1.25 0 4 -1.5708\n"
    "stop pump13a 13 0 -2.4 3.1416\nstop pump13b 13 0 2.4 3.1416\n"
    "path in13 38.1 0 -16\npath in13 30 0 -6\npath in13 22 0 -8\npath in13 14 0 -8\npath in13 13 0 -2.4\n"
    "path mid13 13 0 -2.4\npath mid13 13 0 2.4\n"
    "path out13 13 0 2.4\npath out13 14 0 9\npath out13 24 0 8\npath out13 30 0 6\npath out13 38.1 0 17\n"
    "path inpark 38.1 0 -16\npath inpark 30 0 -6\npath inpark 20 0 -12\npath inpark 4 0 -12\npath inpark 3.3 0 -9\n"
    "path aisle 3.3 0 -9\npath aisle 3.3 0 9\n"
    "path outpark 3.3 0 9\npath outpark 8 0 12\npath outpark 24 0 9\npath outpark 30 0 6\npath outpark 38.1 0 17\n";

void run(npc::World& w, float seconds) {
    for (float t = 0.0f; t < seconds; t += 0.05f) w.update(0.05f);
}

}  // namespace

TEST_CASE("npc: a world from a site and a road stays whole for half an hour", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(7);
    REQUIRE(w.agent_count() > 20);
    REQUIRE(w.agent_count() <= npc::kMaxAgents);
    REQUIRE(w.sane());
    int max_at_site = 0, max_walking = 0, max_sitting = 0;
    for (int minute = 0; minute < 30; ++minute) {
        run(w, 60.0f);
        REQUIRE(w.sane());
        const npc::World::Census c = w.census();
        max_at_site = std::max(max_at_site, c.cars_at_site);
        max_walking = std::max(max_walking, c.walking);
        max_sitting = std::max(max_sitting, c.sitting);
        REQUIRE(c.cars_moving > 0);                    // the traffic never stands still as a whole
    }
    // cars pulled in, people walked and sat
    REQUIRE(max_at_site >= 2);
    REQUIRE(max_walking >= 1);
    REQUIRE(max_sitting >= 1);
    REQUIRE(w.step_count() > 17900);                   // (fixed steps: ten a second whatever the frame time)
    REQUIRE(w.step_count() < 18100);
}

TEST_CASE("npc: the same seed gives the same world", "[npc]") {
    npc::World a, b;
    for (npc::World* w : { &a, &b }) {
        REQUIRE(w->load_site_text(kSite, glm::vec3(100.0f, 5.0f, -30.0f)));
        w->set_road(ring_road());
        w->populate(1234);
    }
    // (stepped in different slices of time: the steps are fixed)
    for (int i = 0; i < 6000; ++i) a.update(0.02f);
    for (int i = 0; i < 1200; ++i) b.update(0.1f);
    std::vector<npc::View> va, vb;
    REQUIRE(a.step_count() > 1000);
    // bring both to the same step
    while (b.step_count() < a.step_count()) b.update(npc::kStep);
    while (a.step_count() < b.step_count()) a.update(npc::kStep);
    std::vector<uint8_t> sa, sb;
    a.serialize(sa); b.serialize(sb);
    REQUIRE(sa == sb);
}

TEST_CASE("npc: every car that pulls in drives out again", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(99);
    const int parked0 = w.census().cars_at_site;       // those that stand parked from the start
    int seen_more = 0; float longest = 0.0f;
    for (int i = 0; i < 60 * 90; ++i) {                // 90 minutes, a second at a time
        run(w, 1.0f);
        const npc::World::Census c = w.census();
        if (c.cars_at_site > parked0) ++seen_more;
        longest = std::max(longest, c.longest_visit);
    }
    REQUIRE(seen_more > 0);
    REQUIRE(longest > 20.0f);                          // somebody did stop for a while
    REQUIRE(longest < 600.0f);                         // and nobody for good: the longest visit (a meal) is minutes
    REQUIRE(w.sane());
}

TEST_CASE("npc: a state goes over the wire and a replica follows it", "[npc]") {
    npc::World host, peer;
    REQUIRE(host.load_site_text(kSite, glm::vec3(0.0f)));
    host.set_road(ring_road());
    host.populate(5);
    run(host, 30.0f);
    std::vector<uint8_t> msg;
    host.serialize(msg);
    REQUIRE(msg.size() == 4 + size_t(host.agent_count()) * 28);
    REQUIRE(peer.apply(msg.data(), msg.size()));
    REQUIRE(peer.replica());
    REQUIRE(peer.agent_count() == host.agent_count());
    // a second state a quarter of a second later: the replica blends from the first to it
    run(host, 0.25f);
    peer.update(0.25f);
    host.serialize(msg);
    REQUIRE(peer.apply(msg.data(), msg.size()));
    std::vector<npc::View> hv, pv;
    peer.update(10.0f);                                // (long past the second state: it holds there)
    host.views(hv); peer.views(pv);
    REQUIRE(hv.size() == pv.size());
    for (size_t i = 0; i < hv.size(); ++i) {
        REQUIRE(pv[i].kind == hv[i].kind);
        REQUIRE(pv[i].hidden == hv[i].hidden);
    }
    // damaged messages are refused
    REQUIRE_FALSE(peer.apply(msg.data(), msg.size() - 1));
    msg[0] = 'X';
    REQUIRE_FALSE(peer.apply(msg.data(), msg.size()));
    REQUIRE_FALSE(peer.apply(nullptr, 0));
}

TEST_CASE("npc: a replica that has to take over starts its own world", "[npc]") {
    npc::World host, peer;
    for (npc::World* w : { &host, &peer }) {
        REQUIRE(w->load_site_text(kSite, glm::vec3(0.0f)));
        w->set_road(ring_road());
        w->populate(5);
    }
    run(host, 5.0f);
    std::vector<uint8_t> msg;
    host.serialize(msg);
    REQUIRE(peer.apply(msg.data(), msg.size()));
    REQUIRE(peer.replica());
    const int steps = peer.step_count();
    run(peer, 2.0f);
    REQUIRE(peer.step_count() == steps);               // (a replica does not step)
    peer.populate(5);                                  // the host left
    REQUIRE_FALSE(peer.replica());
    run(peer, 2.0f);
    REQUIRE(peer.step_count() > 15);
    REQUIRE(peer.sane());
}

TEST_CASE("npc: a car only ever drives - it is never put somewhere, and never turns on the spot", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(99);
    std::vector<npc::View> last = w.state();
    float worst_step = 0.0f, worst_turn = 0.0f, worst_side = 0.0f, worst_askew = 0.0f;
    int parked_in_bay = 0;
    float worst_astray = 0.0f;
    for (int i = 0; i < 10 * 60 * 40; ++i) {           // 40 minutes, step by step
        w.update(npc::kStep);
        const std::vector<npc::View>& now = w.state();
        REQUIRE(now.size() == last.size());
        for (size_t k = 0; k < now.size(); ++k) {
            if (now[k].kind != npc::Kind::Car) continue;
            const glm::vec2 d(now[k].pos.x - last[k].pos.x, now[k].pos.z - last[k].pos.z);
            float turn = now[k].yaw - last[k].yaw;
            while (turn > 3.14159265f) turn -= 6.2831853f;
            while (turn < -3.14159265f) turn += 6.2831853f;
            // (how far it went against how fast it says it goes; how far of that was sideways)
            worst_step = std::max(worst_step, glm::length(d) - (std::abs(now[k].speed) * npc::kStep + 0.12f));
            worst_turn = std::max(worst_turn, std::abs(turn));
            const glm::vec2 right(std::cos(now[k].yaw), std::sin(now[k].yaw));
            worst_side = std::max(worst_side, std::abs(glm::dot(d, right)));
            // (it stays on the road or the site: the ring is a circle of 318 m round (358, 0), the site lies inside x -20..40)
            const float from_ring = std::abs(glm::length(glm::vec2(now[k].pos.x - (40.0f + 2000.0f / 6.2831853f), now[k].pos.z)) - 2000.0f / 6.2831853f);
            const bool on_site = now[k].pos.x > -25.0f && now[k].pos.x < 45.0f && std::abs(now[k].pos.z) < 30.0f;
            if (!on_site) worst_astray = std::max(worst_astray, from_ring);
            // (a car that has come to rest in a bay stands straight in it: the bays face -pi/2 at x = -1.25)
            if (now[k].pose == npc::Pose::Park && last[k].pose == npc::Pose::Drive && std::abs(now[k].pos.x + 1.25f) < 0.8f) {
                float askew = now[k].yaw + 1.5708f;
                while (askew > 3.14159265f) askew -= 6.2831853f;
                while (askew < -3.14159265f) askew += 6.2831853f;
                worst_askew = std::max(worst_askew, std::abs(askew));
                ++parked_in_bay;
            }
        }
        last = now;
    }
    REQUIRE(worst_step <= 0.0f);                       // no jump
    REQUIRE(worst_turn < 0.11f);                       // what its steering lock allows at the site's 4.2 m/s (v / wheelbase x tan 0.6 = 1.06 rad/s)
    REQUIRE(worst_side < 0.16f);                       // it goes where it points: the body's middle, 1.35 m ahead of the rear axle, swings 14 cm a step at full lock
    REQUIRE(worst_astray < 6.0f);                      // no car leaves the road for the country (one did: its target behind it, it drove straight on)
    REQUIRE(parked_in_bay > 0);
    REQUIRE(worst_askew < 0.21f);                      // within 12 degrees of the bay's line                       // it goes where it points (the body's middle swings a little round the rear axle)
}

TEST_CASE("npc: nobody walks through a car or through somebody else", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(99);
    float deepest_in_car = 0.0f, nearest_pair = 1e9f;
    int walkers_seen = 0;
    for (int i = 0; i < 10 * 60 * 40; ++i) {           // 40 minutes, step by step
        w.update(npc::kStep);
        const std::vector<npc::View>& v = w.state();
        for (size_t a = 0; a < v.size(); ++a) {
            if (v[a].kind != npc::Kind::Person || v[a].hidden || v[a].pose != npc::Pose::Walk) continue;
            ++walkers_seen;
            for (size_t b = 0; b < v.size(); ++b) {
                if (a == b) continue;
                if (v[b].kind == npc::Kind::Person) {
                    if (v[b].hidden || v[b].pose == npc::Pose::Sit) continue;
                    nearest_pair = std::min(nearest_pair, glm::distance(glm::vec2(v[a].pos.x, v[a].pos.z), glm::vec2(v[b].pos.x, v[b].pos.z)));
                } else if (size_t(v[a].car) != b && v[b].pose == npc::Pose::Park) {
                    // (a car that stands; its own people excepted. Inside its outline by how much?)
                    const glm::vec2 f(std::sin(v[b].yaw), -std::cos(v[b].yaw)), r(std::cos(v[b].yaw), std::sin(v[b].yaw));
                    const glm::vec2 d(v[a].pos.x - v[b].pos.x, v[a].pos.z - v[b].pos.z);
                    const float in = std::min(1.02f - std::abs(glm::dot(d, r)), 2.35f - std::abs(glm::dot(d, f)));
                    deepest_in_car = std::max(deepest_in_car, in);
                }
            }
        }
    }
    REQUIRE(walkers_seen > 1000);
    REQUIRE(deepest_in_car <= 0.0f);                   // never inside a standing car's outline
    REQUIRE(nearest_pair > 0.30f);                     // never closer than two bodies side by side
    REQUIRE(w.sane());
}

TEST_CASE("npc: somebody shot is gone, and later somebody new is there; the world stays whole", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(11);
    run(w, 120.0f);
    std::vector<npc::View> v;
    w.views(v);
    int shot = 0;
    for (const npc::View& p : v)
        if (p.kind == npc::Kind::Person && !p.hidden && shot < 6) { REQUIRE(w.kill(int(p.id))); REQUIRE_FALSE(w.kill(int(p.id))); ++shot; }
    REQUIRE(shot >= 4);
    w.views(v);
    int on_foot = 0;
    for (const npc::View& p : v) if (p.kind == npc::Kind::Person && !p.hidden) ++on_foot;
    run(w, 600.0f);                                    // (their cars leave without them; new people come)
    REQUIRE(w.sane());
    w.views(v);
    int later = 0;
    for (const npc::View& p : v) if (p.kind == npc::Kind::Person && !p.hidden) ++later;
    REQUIRE(later > on_foot);
    REQUIRE(w.census().longest_visit < 420.0f);        // nobody's car waits for the dead
}

TEST_CASE("npc: a standing car with nobody in it can be taken, and the traffic gets another", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(5);
    std::vector<npc::View> v;
    int parked = -1, moving = -1;
    // (until a car stands at the site with its people out of it)
    for (int i = 0; i < 2400 && parked < 0; ++i) {
        run(w, 0.5f);
        w.views(v);
        for (const npc::View& c : v) if (c.kind == npc::Kind::Car) { if (parked < 0 && w.can_take_car(int(c.id))) parked = int(c.id); if (c.pose == npc::Pose::Drive && moving < 0) moving = int(c.id); }
    }
    REQUIRE(parked >= 0);
    moving = -1;
    for (const npc::View& c : w.state()) if (c.kind == npc::Kind::Car && c.speed > 5.0f && moving < 0) moving = int(c.id);
    REQUIRE(moving >= 0);
    REQUIRE_FALSE(w.take_car(moving));                 // (not one that drives)
    REQUIRE(w.take_car(parked));
    REQUIRE_FALSE(w.take_car(parked));
    w.views(v);
    REQUIRE(v[size_t(parked)].hidden);
    run(w, 200.0f);
    w.views(v);
    REQUIRE_FALSE(v[size_t(parked)].hidden);           // back, as traffic
    REQUIRE(v[size_t(parked)].pose == npc::Pose::Drive);
    run(w, 900.0f);
    REQUIRE(w.sane());
}

TEST_CASE("npc: the soldiers run their path there and back, in step", "[npc]") {
    npc::World w;
    w.set_road(ring_road());
    w.set_run_path({ glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(100.0f, 0.0f, 0.0f), glm::vec3(100.0f, 0.0f, 80.0f) });
    w.populate(3);
    std::vector<npc::View> v, v0;
    w.views(v0);
    int runners = 0;
    for (const npc::View& p : v0) if (p.kind == npc::Kind::Person && p.pose == npc::Pose::Run) ++runners;
    REQUIRE(runners == 12);
    float far_off = 0.0f, step_max = 0.0f;
    std::vector<npc::View> last = v0;
    for (int i = 0; i < 2400; ++i) {                   // two minutes: more than once there and back
        w.update(0.05f);
        w.views(v);
        for (size_t k = 0; k < v.size(); ++k) {
            if (v[k].pose != npc::Pose::Run) continue;
            step_max = std::max(step_max, glm::distance(v[k].pos, last[k].pos));
            // (never off the path by more than its file's place beside it)
            const glm::vec3 p = v[k].pos;
            const float d1 = glm::distance(glm::vec2(p.x, p.z), glm::vec2(std::clamp(p.x, 0.0f, 100.0f), 0.0f)), d2 = glm::distance(glm::vec2(p.x, p.z), glm::vec2(100.0f, std::clamp(p.z, 0.0f, 80.0f)));
            far_off = std::max(far_off, std::min(d1, d2));
        }
        last = v;
    }
    REQUIRE(step_max < 0.5f);                          // 3.4 m/s: nobody jumps (the turn at an end included)
    REQUIRE(far_off < 2.0f);
    // the wire carries the pose and the style
    std::vector<uint8_t> bytes;
    w.serialize(bytes);
    npc::World r;
    REQUIRE(r.apply(bytes.data(), bytes.size()));
    REQUIRE(r.state().size() == w.state().size());
    for (size_t k = 0; k < r.state().size(); ++k) { REQUIRE(r.state()[k].pose == w.state()[k].pose); REQUIRE(r.state()[k].style == w.state()[k].style); }
}

TEST_CASE("npc: a car on the road stops for a player on its lane, and can be taken with its people in it", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(7);
    run(w, 30.0f);
    std::vector<npc::View> v;
    w.views(v);
    // a place on the lane of the cars that carry people: where one of them is now, far from the site
    glm::vec3 P(0.0f); bool found = false;
    for (const npc::View& p : v) {
        if (p.kind != npc::Kind::Person || !p.hidden || p.car == 255) continue;
        const npc::View& c = v[p.car];
        if (c.pose == npc::Pose::Drive && c.speed > 15.0f && std::abs(c.pos.z) > 200.0f) { P = c.pos; found = true; break; }
    }
    REQUIRE(found);
    run(w, 3.0f);                                      // (that car has gone on)
    w.set_walkers({ P });
    int stopped = -1; float nearest = 1e9f;
    for (int i = 0; i < 2400 && stopped < 0; ++i) {
        w.update(0.05f);
        for (const npc::View& c : w.state()) {
            if (c.kind != npc::Kind::Car || c.hidden) continue;
            const float d = glm::distance(c.pos, P);
            nearest = std::min(nearest, d);
            if (d < 14.0f && std::abs(c.speed) < 0.05f) stopped = int(c.id);
        }
    }
    REQUIRE(stopped >= 0);                             // one came, and stands in front of the player
    REQUIRE(nearest > 3.0f);                           // ... and nobody was driven into
    run(w, 5.0f);
    REQUIRE(std::abs(w.state()[size_t(stopped)].speed) < 0.05f);      // it waits as long as he stands there
    const glm::vec3 where = w.state()[size_t(stopped)].pos;
    std::vector<int> inside;
    for (const npc::View& p : w.state()) if (p.kind == npc::Kind::Person && p.hidden && int(p.car) == stopped) inside.push_back(int(p.id));
    REQUIRE(!inside.empty());
    REQUIRE(w.can_take_car(stopped));
    REQUIRE(w.take_car(stopped));
    REQUIRE(w.state()[size_t(stopped)].hidden);
    for (int id : inside) {                            // its people stand beside where it stood
        const npc::View& p = w.state()[size_t(id)];
        REQUIRE_FALSE(p.hidden);
        REQUIRE(p.car == 255);
        REQUIRE(glm::distance(p.pos, where) < 3.0f);
        REQUIRE(glm::distance(p.pos, where) > 1.0f);
    }
    w.set_walkers({});
    run(w, 60.0f);                                     // out on the road there is nowhere to walk to: they are gone
    for (int id : inside) REQUIRE(w.state()[size_t(id)].hidden);
    run(w, 900.0f);
    REQUIRE(w.sane());
    REQUIRE(w.census().cars_moving > 10);              // the traffic drives on
}

TEST_CASE("npc: a driver shot in his car - it rolls to a stop, the others get out, and it is cleared away", "[npc]") {
    npc::World w;
    REQUIRE(w.load_site_text(kSite, glm::vec3(0.0f)));
    w.set_road(ring_road());
    w.populate(21);
    run(w, 20.0f);
    // ---- a driver of the world's own
    int driver = -1, car = -1;
    for (const npc::View& p : w.state()) {
        if (p.kind != npc::Kind::Person || !p.hidden || p.car == 255 || p.seat != 0) continue;
        const npc::View& c = w.state()[p.car];
        if (c.pose == npc::Pose::Drive && c.speed > 15.0f && std::abs(c.pos.z) > 200.0f) { driver = int(p.id); car = int(p.car); break; }
    }
    REQUIRE(driver >= 0);
    std::vector<int> others;
    for (const npc::View& p : w.state()) if (p.kind == npc::Kind::Person && p.hidden && int(p.car) == car && int(p.id) != driver) others.push_back(int(p.id));
    REQUIRE_FALSE(w.can_take_car(car));                // (it drives)
    REQUIRE_FALSE(w.take_car(car));
    REQUIRE(w.kill(driver));
    REQUIRE_FALSE(w.kill(driver));
    REQUIRE(w.state()[size_t(car)].pose == npc::Pose::Park);       // (nobody at the wheel from this moment)
    const glm::vec3 from = w.state()[size_t(car)].pos;
    float step_max = 0.0f; glm::vec3 last = from;
    for (int i = 0; i < 300; ++i) { w.update(0.05f); const glm::vec3 p = w.state()[size_t(car)].pos; step_max = std::max(step_max, glm::distance(p, last)); last = p; }
    REQUIRE(std::abs(w.state()[size_t(car)].speed) < 0.05f);       // it stands, some way on
    REQUIRE(glm::distance(w.state()[size_t(car)].pos, from) > 20.0f);
    REQUIRE(step_max < 2.5f);                                      // rolled there: not put
    for (int id : others) REQUIRE_FALSE(w.state()[size_t(id)].hidden);      // whoever sat beside him got out
    REQUIRE(w.can_take_car(car));                      // a car that stands can be taken
    // ---- the driver of a car nobody of the world sits in: by the car's number
    int ghost = -1;
    for (const npc::View& c : w.state()) {
        if (c.kind != npc::Kind::Car || c.pose != npc::Pose::Drive || c.speed < 15.0f || int(c.id) == car) continue;
        bool people = false;
        for (const npc::View& p : w.state()) if (p.kind == npc::Kind::Person && int(p.car) == int(c.id)) people = true;
        if (!people) { ghost = int(c.id); break; }
    }
    REQUIRE(ghost >= 0);
    REQUIRE(w.kill(ghost));
    REQUIRE_FALSE(w.kill(ghost));
    REQUIRE(w.state()[size_t(ghost)].pose == npc::Pose::Park);
    run(w, 15.0f);
    REQUIRE(std::abs(w.state()[size_t(ghost)].speed) < 0.05f);
    // ---- two minutes on both are cleared away, and later the traffic has them again
    run(w, 400.0f);
    REQUIRE(w.state()[size_t(car)].pose == npc::Pose::Drive);
    REQUIRE(w.state()[size_t(ghost)].pose == npc::Pose::Drive);
    REQUIRE_FALSE(w.state()[size_t(car)].hidden);
    run(w, 900.0f);
    REQUIRE(w.sane());
}

// ---------------------------------------------------------------------------------------------
// The traffic: drivers of different pace, following, overtaking, the police, what is left on the road.
namespace {

// An oval: two straights of 1500 m (x = 0 driven toward +z, x = 240 driven toward -z) joined by
// half circles of 120 m. A car driving the way s grows keeps right: on the first straight at x = -1.9.
constexpr float kOvalS = 1500.0f, kOvalR = 120.0f;
npc::Road oval_road() {
    npc::Road r;
    r.length = 2.0f * kOvalS + 2.0f * 3.14159265f * kOvalR;
    r.at = [](float s, glm::vec3& pos, glm::vec2& dir) {
        const float arc = 3.14159265f * kOvalR;
        if (s < kOvalS) { pos = glm::vec3(0.0f, 0.0f, s); dir = glm::vec2(0.0f, 1.0f); return true; }
        s -= kOvalS;
        if (s < arc) { const float a = s / kOvalR; pos = glm::vec3(kOvalR - kOvalR * std::cos(a), 0.0f, kOvalS + kOvalR * std::sin(a)); dir = glm::vec2(std::sin(a), std::cos(a)); return true; }
        s -= arc;
        if (s < kOvalS) { pos = glm::vec3(2.0f * kOvalR, 0.0f, kOvalS - s); dir = glm::vec2(0.0f, -1.0f); return true; }
        s -= kOvalS;
        const float a = s / kOvalR;
        pos = glm::vec3(kOvalR + kOvalR * std::cos(a), 0.0f, -kOvalR * std::sin(a)); dir = glm::vec2(-std::sin(a), -std::cos(a));
        return true;
    };
    return r;
}
// The nearest two cars on the road come to each other (centre to centre). Side by side in two
// lanes they are 3.8 m apart, nose to tail 4.7 m: under 2 m they are in each other.
float nearest_pair(const std::vector<npc::View>& v, std::string* what = nullptr) {
    float best = 1e9f;
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t k = i + 1; k < v.size(); ++k)
            if (v[i].kind == npc::Kind::Car && v[k].kind == npc::Kind::Car && !v[i].hidden && !v[k].hidden) {
                const float d = glm::distance(v[i].pos, v[k].pos);
                if (d >= best) continue;
                best = d;
                if (what) {
                    char b[256];
                    std::snprintf(b, sizeof(b), "car %d at (%.1f, %.1f) yaw %.2f %.1f m/s style %d and car %d at (%.1f, %.1f) yaw %.2f %.1f m/s style %d", int(i), v[i].pos.x, v[i].pos.z, v[i].yaw, v[i].speed,
                                  int(v[i].style), int(k), v[k].pos.x, v[k].pos.z, v[k].yaw, v[k].speed, int(v[k].style));
                    *what = b;
                }
            }
    return best;
}
// ... keeping the worst moment of a run.
struct Nearest {
    float d = 1e9f; std::string what; int step = 0;
    void look(const std::vector<npc::View>& v, int at) { std::string s; const float n = nearest_pair(v, &s); if (n < d) { d = n; what = s; step = at; } }
};

}  // namespace

TEST_CASE("npc traffic: a faster car follows a slower one at a distance, and stops for a car left on its lane", "[npc][traffic]") {
    // (on the ring: a bend all the way round, where nobody overtakes)
    npc::World w;
    w.set_road(ring_road());
    w.populate(3, 0);
    const int slow = w.add_car(300.0f, 1, 12.0f);
    const int fast = w.add_car(100.0f, 1, 22.0f);
    REQUIRE(slow >= 0);
    REQUIRE(fast >= 0);
    float nearest = 1e9f;
    for (int i = 0; i < 10 * 90; ++i) { w.update(npc::kStep); nearest = std::min(nearest, nearest_pair(w.state())); }
    const float gap = glm::distance(w.state()[size_t(slow)].pos, w.state()[size_t(fast)].pos);
    INFO("after 90 s: " << gap << " m apart, the follower at " << w.state()[size_t(fast)].speed << " m/s; nearest ever " << nearest);
    REQUIRE(std::abs(w.state()[size_t(fast)].speed - 12.0f) < 1.0f);      // at the pace of the one in front
    REQUIRE(gap > 12.0f);                                                // 4.7 m of car, 2.5 m and over a second at 12 m/s
    REQUIRE(gap < 40.0f);
    REQUIRE(nearest > 9.0f);
    REQUIRE(w.traffic().overtakes == 0);
    // A car left standing on the lane 150 m ahead of the slow one: both stop short of it.
    const npc::View lead = w.state()[size_t(slow)];
    npc::Obstacle o; o.id = 9;
    {
        glm::vec3 p; glm::vec2 d;
        const npc::Road r = ring_road();
        const float R = 2000.0f / 6.2831853f;
        float a = std::atan2(lead.pos.z, 40.0f + R - lead.pos.x);              // where the slow car is on the ring
        if (a < 0.0f) a += 6.2831853f;
        r.at(std::fmod(a * R + 150.0f, 2000.0f), p, d);
        o.pos = p + glm::vec3(-d.y, 0.0f, d.x) * 1.9f; o.yaw = std::atan2(d.x, -d.y);
    }
    npc::Obstacle o2 = o; o2.id = 10;                 // (and one beside it in the other lane: the road is shut)
    {
        glm::vec3 p; glm::vec2 d;
        const float R = 2000.0f / 6.2831853f;
        float a = std::atan2(lead.pos.z, 40.0f + R - lead.pos.x);
        if (a < 0.0f) a += 6.2831853f;
        ring_road().at(std::fmod(a * R + 150.0f, 2000.0f), p, d);
        o2.pos = p - glm::vec3(-d.y, 0.0f, d.x) * 1.9f;
    }
    float nearest_o = 1e9f;
    for (int i = 0; i < 10 * 60; ++i) {
        w.set_obstacles({ o, o2 });
        w.update(npc::kStep);
        for (int c : { slow, fast }) nearest_o = std::min(nearest_o, glm::distance(w.state()[size_t(c)].pos, o.pos));
    }
    INFO("nearest to the car left on the lane: " << nearest_o << " m; speeds " << w.state()[size_t(slow)].speed << ", " << w.state()[size_t(fast)].speed);
    REQUIRE(nearest_o > 4.7f + 3.0f);                 // not into it
    REQUIRE(nearest_o < 30.0f);                       // and it did come up to it
    REQUIRE(std::abs(w.state()[size_t(slow)].speed) < 0.3f);
    REQUIRE(std::abs(w.state()[size_t(fast)].speed) < 0.3f);
    REQUIRE(w.sane());
}

TEST_CASE("npc traffic: a car left on a straight road is gone round when nothing comes", "[npc][traffic]") {
    npc::World w;
    w.set_road(oval_road());
    w.populate(3, 0);
    const int c = w.add_car(100.0f, 1, 20.0f);
    npc::Obstacle o; o.id = 4; o.pos = glm::vec3(-1.9f, 0.0f, 400.0f); o.yaw = 3.14159265f;      // on the lane, 300 m ahead
    float nearest = 1e9f, slowest = 1e9f;
    for (int i = 0; i < 10 * 60; ++i) {
        w.set_obstacles({ o });
        w.update(npc::kStep);
        nearest = std::min(nearest, glm::distance(w.state()[size_t(c)].pos, o.pos));
        slowest = std::min(slowest, std::abs(w.state()[size_t(c)].speed));
    }
    const npc::View v = w.state()[size_t(c)];
    INFO("nearest to it " << nearest << " m, slowest " << slowest << " m/s, after a minute at z " << v.pos.z << ", x " << v.pos.x << ", " << v.speed << " m/s");
    REQUIRE(nearest > 2.25f);                         // beside it, clear: two car widths are 2.04 m centre to centre
    REQUIRE(v.pos.z > 500.0f);                        // past it
    REQUIRE(std::abs(-v.pos.x - 1.9f) < 0.5f);        // and back in its lane
    REQUIRE(w.traffic().overtakes == 1);
}

TEST_CASE("npc traffic: a car held up overtakes - after what comes the other way has passed", "[npc][traffic]") {
    npc::World w;
    w.set_road(oval_road());
    w.populate(3, 0);
    const int slow = w.add_car(200.0f, 1, 10.0f);
    const int fast = w.add_car(120.0f, 1, 24.0f);
    const int other = w.add_car(620.0f, -1, 21.0f);             // 400 m ahead of the slow one, coming: they meet in 13 s
    float nearest = 1e9f, out_at = -1.0f, met_at = -1.0f, back_at = -1.0f, most_left = 1e9f, lane_after = 1e9f, speed_after = 0.0f, ahead_after = 0.0f;
    for (int i = 0; i < 10 * 80; ++i) {
        w.update(npc::kStep);
        const std::vector<npc::View>& v = w.state();
        nearest = std::min(nearest, nearest_pair(v));
        const float t = float(i) * npc::kStep;
        const npc::View& f = v[size_t(fast)];
        if (met_at < 0.0f && v[size_t(other)].pos.z < f.pos.z) met_at = t;                   // (the first straight runs along +z)
        if (out_at < 0.0f && (f.style & npc::kCarOvertaking)) out_at = t;
        if (out_at >= 0.0f && back_at < 0.0f && !(f.style & npc::kCarOvertaking)) back_at = t;
        if (out_at >= 0.0f && back_at < 0.0f) most_left = std::min(most_left, -f.pos.x);      // (right of the centre line: -x)
        if (back_at >= 0.0f && t < back_at + 6.0f) { lane_after = -f.pos.x; speed_after = f.speed; ahead_after = f.pos.z - v[size_t(slow)].pos.z; }      // (six seconds on: still on the straight)
    }
    const std::vector<npc::View>& v = w.state();
    INFO("pulled out at " << out_at << " s, the oncoming car passed at " << met_at << " s, back in lane at " << back_at << " s; furthest left " << most_left
         << " m right of the centre; nearest two cars ever " << nearest << " m; overtakes " << w.traffic().overtakes << ", given up " << w.traffic().given_up);
    REQUIRE(met_at > 0.0f);
    REQUIRE(out_at > met_at - 1.0f);                  // not out in front of the car that came
    REQUIRE(back_at > out_at);
    REQUIRE(most_left < -1.2f);                       // it did drive in the other lane
    REQUIRE(nearest > 3.0f);                          // and nobody touched anybody
    (void)v;
    REQUIRE(ahead_after > 8.0f);                      // past it
    REQUIRE(std::abs(lane_after - 1.9f) < 0.5f);      // and back in its own lane
    REQUIRE(speed_after > 20.0f);
    REQUIRE(w.traffic().overtakes == 1);
    REQUIRE(w.sane());
}

TEST_CASE("npc traffic: somebody comes while a car is out passing - it gets back in, and nobody is hit", "[npc][traffic]") {
    for (float ahead : { 120.0f, 200.0f, 300.0f }) {
        npc::World w;
        w.set_road(oval_road());
        w.populate(3, 0);
        const int slow = w.add_car(200.0f, 1, 10.0f);
        const int fast = w.add_car(150.0f, 1, 24.0f);
        (void)slow;
        float nearest = 1e9f;
        bool added = false;
        for (int i = 0; i < 10 * 60; ++i) {
            w.update(npc::kStep);
            const std::vector<npc::View>& v = w.state();
            // (the moment it is fully out: a car appears in that lane, coming)
            if (!added && (v[size_t(fast)].style & npc::kCarOvertaking) && -v[size_t(fast)].pos.x < -1.0f) {
                added = true;
                REQUIRE(w.add_car(v[size_t(fast)].pos.z + ahead, -1, 21.0f) >= 0);
            }
            nearest = std::min(nearest, nearest_pair(w.state()));
        }
        INFO("a car appearing " << ahead << " m ahead: nearest two cars ever " << nearest << " m; overtakes " << w.traffic().overtakes << ", given up " << w.traffic().given_up);
        REQUIRE(added);
        REQUIRE(nearest > 2.6f);
        REQUIRE(w.sane());
    }
}

TEST_CASE("npc traffic: slow drivers, speeders and the police share the road for half an hour without touching", "[npc][traffic]") {
    npc::World w;
    w.set_road(oval_road());
    w.populate(11);
    const npc::World::Traffic t0 = w.traffic();
    REQUIRE(t0.speeders >= 1);
    REQUIRE(t0.police == 2);
    float nearest = 1e9f, fastest = 0.0f;
    int siren_steps = 0;
    Nearest nn;
    for (int i = 0; i < 10 * 60 * 30; ++i) {
        w.update(npc::kStep);
        nn.look(w.state(), i); nearest = nn.d;
        const npc::World::Traffic t = w.traffic();
        fastest = std::max(fastest, t.fastest);
        siren_steps += t.sirens;
    }
    const npc::World::Traffic t = w.traffic();
    INFO("overtakes " << t.overtakes << " (given up " << t.given_up << "), fastest " << fastest << " m/s, siren-seconds " << siren_steps / 10 << ", nearest two cars ever " << nearest << " m: at step " << nn.step << " " << nn.what);
    REQUIRE(t.overtakes > 10);
    REQUIRE(fastest > 34.0f);                         // the speeder got to drive (the rest want 13 .. 28 m/s)
    REQUIRE(siren_steps > 0);
    REQUIRE(nearest > 2.6f);
    REQUIRE(w.sane());
}

TEST_CASE("npc traffic: the others make way for a siren, and the police car gets through", "[npc][traffic]") {
    npc::World w;
    w.set_road(oval_road());
    w.populate(3, 0);
    const int a = w.add_car(260.0f, 1, 16.0f), b = w.add_car(300.0f, 1, 16.0f);
    const int police = w.add_car(100.0f, 1, 20.0f, npc::kCarPolice | npc::kCarSiren);
    float nearest = 1e9f, most_right = 0.0f;
    Nearest nn;
    for (int i = 0; i < 10 * 45; ++i) {
        w.update(npc::kStep);
        nn.look(w.state(), i); nearest = nn.d;
        most_right = std::max(most_right, -w.state()[size_t(a)].pos.x);
    }
    const std::vector<npc::View>& v = w.state();
    INFO("the car behind which the siren came went " << most_right << " m right of the centre; police at z " << v[size_t(police)].pos.z << ", the others at " << v[size_t(a)].pos.z << ", " << v[size_t(b)].pos.z << "; nearest " << nearest << " at step " << nn.step << ": " << nn.what);
    REQUIRE((v[size_t(police)].style & npc::kCarSiren) != 0);
    REQUIRE(most_right > 2.6f);                       // to the edge of the road
    REQUIRE(v[size_t(police)].pos.z > v[size_t(b)].pos.z + 10.0f);
    REQUIRE(nearest > 2.3f);                          // (down the middle past a car at the edge: about 3 m centre to centre)
}

TEST_CASE("npc traffic: a car that is struck is pushed, and drives on", "[npc][traffic]") {
    npc::World w;
    w.set_road(oval_road());
    w.populate(3, 0);
    const int c = w.add_car(300.0f, 1, 20.0f);
    run(w, 2.0f);
    const npc::View before = w.state()[size_t(c)];
    REQUIRE(w.push_car(c, glm::vec3(4.0f, 0.0f, -6.0f), 0.8f));          // from the left front: sideways and back
    run(w, 0.5f);
    const npc::View hit = w.state()[size_t(c)];
    REQUIRE(hit.speed < before.speed - 4.0f);         // slowed by what went against it
    REQUIRE(hit.pos.x > before.pos.x + 0.8f);         // and slid sideways
    run(w, 40.0f);
    const npc::View after = w.state()[size_t(c)];
    REQUIRE(after.speed > 18.0f);                     // it drives on
    REQUIRE(std::abs(-after.pos.x - 1.9f) < 0.6f);    // in its lane
    REQUIRE(w.sane());
}

// An added site (the pavilion, the pad): desks with a keyboard each, spots to stand on, a way
// between them - 300 m from the first site.
namespace {
const char* kOffice =
    "seat desk 0 0.5 0 0 0 0.75 -0.6\nseat desk 1.5 0.5 0 0 1.5 0.75 -0.6\nseat desk 3 0.5 0 0 3 0.75 -0.6\nseat desk 4.5 0.5 0 0 4.5 0.75 -0.6\n"
    "seat meet 9 0.5 4 0\nseat sofa 9 0.5 -4 3.1416\n"
    "spot point 0.7 0 0.2 0 0 1.1 -0.9\nspot point 3.7 0 0.2 0 3 1.1 -0.9\nspot window 2 0 6 3.1416 2 20 80\nspot coffee 8 0 0 0\nspot wall -3 0 2 0 -3 2 -2\nspot rail 6 0 6 0\nspot terrace -3 0 6 0\n"
    "spot group 20 0 0 0\nspot group 21 0 0.6 0\nspot group 20.9 0 -0.7 0\nspot watch 24 0 4 0 30 40 0\nspot watch 25 0 5 0 30 40 0\nspot edge 26 0 -5 0\nspot edge 27 0 -6 0\nspot tech 16 0 8 0\nspot tech 17 0 -8 0\n"
    "node a 0 0 2\nnode b 4.5 0 2\nnode c 9 0 2\nnode d 2 0 5\nnode e 20 0 2\nnode f 25 0 2\n"
    "link a b\nlink b c\nlink a d\nlink c e\nlink e f\n";
}

TEST_CASE("npc: an added site's people are there only while a player is near, and the first site is not changed by them", "[npc]") {
    const glm::vec3 at(300.0f, 0.0f, 0.0f);
    npc::World plain, w;
    for (npc::World* x : { &plain, &w }) { REQUIRE(x->load_site_text(kSite, glm::vec3(0.0f))); x->set_road(ring_road()); }
    REQUIRE(w.add_site_text(kOffice, at));
    plain.populate(99); w.populate(99);
    const int n0 = plain.agent_count(), n1 = w.agent_count();
    REQUIRE(n1 > n0 + 8);
    REQUIRE(n1 <= npc::kMaxAgents);
    // Nobody near: none of them is there, and nothing of the first site's life differs.
    run(plain, 120.0f); run(w, 120.0f);
    REQUIRE(w.census().asleep == n1 - n0);
    for (int i = 0; i < n0; ++i) {
        REQUIRE(glm::distance(plain.state()[size_t(i)].pos, w.state()[size_t(i)].pos) < 1e-4f);
        REQUIRE(plain.state()[size_t(i)].pose == w.state()[size_t(i)].pose);
    }
    for (int i = n0; i < n1; ++i) REQUIRE(w.state()[size_t(i)].hidden);
    // A player comes near: they are at their places - at desks (style 4, the keyboard within
    // reach in front), standing where there is something to do - and live on without harm.
    w.set_eyes({ at + glm::vec3(50.0f, 0.0f, 0.0f) });
    run(w, 1.0f);
    REQUIRE(w.census().asleep == 0);
    int at_desk = 0, standing = 0;
    for (int i = n0; i < n1; ++i) {
        const npc::View& v = w.state()[size_t(i)];
        REQUIRE(!v.hidden);
        REQUIRE(glm::distance(v.pos, at) < 60.0f);
        if (v.pose == npc::Pose::Sit && v.style == 4) { ++at_desk; REQUIRE(glm::distance(v.aim, v.pos) < 1.0f); REQUIRE(glm::distance(v.aim, v.pos) > 0.3f); }
        if (v.pose == npc::Pose::Stand && v.style >= 5) ++standing;
    }
    REQUIRE(at_desk >= 2);
    REQUIRE(standing >= 3);
    int walked = 0, pointed = 0;
    for (int minute = 0; minute < 20; ++minute) {
        for (int s = 0; s < 60; ++s) {
            run(w, 1.0f);
            for (int i = n0; i < n1; ++i) {
                const npc::View& v = w.state()[size_t(i)];
                walked += v.pose == npc::Pose::Walk ? 1 : 0;
                pointed += (v.pose == npc::Pose::Stand && v.style == 5) ? 1 : 0;
                REQUIRE(glm::distance(v.pos, at) < 60.0f);
            }
        }
        REQUIRE(w.sane());
    }
    REQUIRE(walked > 50);
    REQUIRE(pointed > 0);
    // The whole world crosses the wire and comes out the same.
    std::vector<uint8_t> m; w.serialize(m);
    npc::World copy;
    REQUIRE(copy.apply(m.data(), m.size()));
    REQUIRE(copy.agent_count() == n1);
    // The player leaves: they are gone again; and back: there again.
    w.set_eyes({ at + glm::vec3(2000.0f, 0.0f, 0.0f) });
    run(w, 1.0f);
    REQUIRE(w.census().asleep == n1 - n0);
    REQUIRE(w.sane());
    w.set_eyes({ at });
    run(w, 1.0f);
    REQUIRE(w.census().asleep == 0);
    REQUIRE(w.sane());
    // Somebody shot at a desk is replaced there after a while.
    int victim = -1;
    for (int i = n0; i < n1; ++i) if (w.state()[size_t(i)].pose == npc::Pose::Sit) { victim = i; break; }
    REQUIRE(victim >= 0);
    REQUIRE(w.kill(victim));
    run(w, 1.0f);
    REQUIRE(w.state()[size_t(victim)].hidden);
    run(w, 80.0f);
    REQUIRE(!w.state()[size_t(victim)].hidden);
    REQUIRE(glm::distance(w.state()[size_t(victim)].pos, at) < 60.0f);
    REQUIRE(w.sane());
}
