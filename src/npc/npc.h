#pragma once
// The NPC engine: up to a hundred people and cars that live a small life of their own.
//
// It knows nothing of drawing, physics, Vulkan or the network. It is given
//   * a SITE: a text file of where one can walk (nodes and links), sit (seats), stand and do
//     something (spots), park (bays), stop at a pump (stops) and drive (paths) - written by the
//     script that models the place (tools/gasstation/make_gasstation.py -> gasstation.nav);
//   * a ROAD: a closed course, as a function from metres along it to a point and a direction;
// and from a seed it makes its people and cars and steps them in fixed steps:
//   people   staff who work their shift, locals who sit outside and wander in for something,
//            travellers who arrive in a car, fill up, pay, perhaps eat, and drive on;
//   cars     traffic round the course in both directions (each keeps to its right-hand lane and
//            follows the car in front at a time gap and overtakes it when the other lane is free:
//            npc.cpp "THE TRAFFIC"); some of those that pass the site pull in - to a pump or to a
//            parking bay - wait for their people and leave again; a few stand parked.
//
// MULTIPLAYER. One machine is the AUTHORITY: it calls update(). It serialises the state a few
// times a second (serialize(): 27 bytes an agent - no bones or clips: place, heading, speed, pose, act, style); the others apply() what they receive and
// call update() too, which for them only moves the clock that views() blends two states with.
// Nothing here depends on frame rate or on who is watching: the same seed and the same number
// of steps give the same world (tests/test_npc.cpp).
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace vibe::npc {

constexpr int   kMaxAgents = 160;              // (100 until 2026-11-26: the pavilion and the launch pad got people of their own)
constexpr float kStep = 0.1f;                  // seconds a step
constexpr float kTakeSpeed = 1.5f;             // m/s: a car no faster than this can be taken (take_car)

enum class Kind : uint8_t { Person = 0, Car = 1 };
enum class Pose : uint8_t {
    Stand = 0,        // a person standing idle
    Walk  = 1,
    Sit   = 2,        // on a seat: `pos` is the seat's top, `yaw` the way it faces
    Work  = 3,        // standing, busy with the hands (a till, a pump's nozzle, a stove)
    Drive = 4,        // a car that moves
    Park  = 5,        // a car that stands
    Run   = 6,        // a person running (the soldiers on the road)
};

// Something a person does once, over a fixed time (the renderer plays a clip over it):
enum class Act : uint8_t { None = 0, SitDown = 1, StandUp = 2, CarEnter = 3, CarExit = 4 };
constexpr float kActTime[5] = { 0.0f, 3.2f, 2.6f, 3.6f, 3.4f };      // seconds each takes

// What a renderer needs of one agent, blended to the present moment.
// View::style of a CAR: what kind it is and what it is doing (bits).
constexpr uint8_t kCarPolice = 1, kCarSiren = 2, kCarOvertaking = 4;

// Something on or by the road that is not this world's: a player's car, driven or left standing.
// The traffic keeps behind it, stops for it, and goes round it when the other lane is free.
struct Obstacle {
    uint32_t  id = 0;                 // the same thing has the same number from call to call
    glm::vec3 pos{0.0f};
    float     yaw = 0.0f;             // as a View's
    glm::vec3 vel{0.0f};
    float     half_w = 1.02f, half_l = 2.35f;
};

struct View {
    uint8_t   id = 0;
    Kind      kind = Kind::Person;
    Pose      pose = Pose::Stand;
    uint8_t   look = 0;          // which of the person's or the car's appearances (the renderer's to map)
    bool      hidden = false;    // a person inside a car
    uint8_t   car = 255;         // a person: the car (an agent's id) it came in - it sits in it while hidden; 255 none
    uint8_t   seat = 0;          // ... and on which seat: 0 the driver's, 1 beside, 2 behind
    glm::vec3 pos{0.0f};
    Act       act = Act::None;   // a person: what it is in the middle of doing (it stands at `pos` meanwhile) ...
    float     act_t = 0.0f;      // ... for how long already (seconds; kActTime says how long it takes)
    uint8_t   style = 0;         // a person: which way of sitting / standing / working (below): the renderer's clip. A car: kCarPolice, kCarSiren, kCarOvertaking
    glm::vec3 aim{0.0f};         // ... and the seat it is sitting down on or getting up from (a car's seat: `car`, `seat`)
    float     yaw = 0.0f;        // radians; 0 looks toward -z, as the player's
    float     speed = 0.0f;      // m/s (negative: a car reversing)
};

// The road: a closed course of `length` metres. at(s) gives the centre line's point and its
// unit direction (x, z) for increasing s; s wraps.
// View::style. Sitting: 0 plain, 1 at the bar (a high stool), 2 eating at a table, 3 talking at a
// table, 4 working at a desk (View::aim: the middle of its keyboard). Standing: 0 plain, 1 arms
// crossed, 2 hands on the hips, 3 at a phone, 4 looking far, 5 pointing at something (View::aim:
// a colleague's screen), 6 talking with whoever stands near, 7 looking at something (View::aim:
// the rocket, the hall below). Working: 0 at a machine (a till's buttons, a tank's valves), 1 a
// cashier, 2 at a pump.
// How somebody walks: `look % 8` - 0, 6 and 7 plainly, 1 briskly, 2 and 5 at ease, 3 hands in the
// pockets, 4 at a phone - and how fast that is, of the plain pace.
constexpr float kPace[8] = { 1.0f, 0.85f, 0.82f, 0.67f, 0.81f, 0.82f, 1.0f, 1.0f };      // (the clips' own paces, a little quicker: they were made slow. 2 was 0.42, the "tired" shuffle, until 2026-11-26)
constexpr int kSeatHeightCm[2] = { 45, 75 };      // a chair's seat above its floor, a bar stool's

struct Road {
    float length = 0.0f;
    std::function<bool(float s, glm::vec3& pos, glm::vec2& dir)> at;
};

class World {
public:
    World();
    ~World();
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // The site file, placed at `origin`. False (and no site) when it cannot be read.
    bool load_site(const std::string& nav_file, glm::vec3 origin);
    // The same from text (tests).
    bool load_site_text(const std::string& text, glm::vec3 origin);
    // ANOTHER SITE beside the first (after load_site, before populate): its nodes, seats and
    // spots - the pavilion by the launch pad, the pad itself. It has people of its own
    // (npc.cpp "THE ADDED SITES"), who are only there while a player is near it (set_eyes):
    // nobody near, they are hidden and cost nothing. A link in its file may name a node of a
    // site read before it. No cars. False (and nothing added) when it cannot be read.
    bool add_site(const std::string& nav_file, glm::vec3 origin);
    bool add_site_text(const std::string& text, glm::vec3 origin);
    // Where the players are, on foot or not (the authority's, given every frame): what decides
    // whether an added site's people are there.
    void set_eyes(const std::vector<glm::vec3>& at);
    void set_road(Road road);
    // Makes the people and the cars. Call after the site and the road are set.
    // The way the soldiers run, there and back (before populate; empty: none).
    void set_run_path(std::vector<glm::vec3> pts);
    // `traffic`: how many cars drive the road (tests make their own with add_car).
    void populate(uint32_t seed, int traffic = 26);
    // One more car on the road (tests and scenarios; the authority's, after populate): `s` metres
    // along it, `dir` +1 / -1, wanting to drive `cruise` m/s. `style`: kCarPolice and kCarSiren.
    // Its number, or -1.
    int  add_car(float s, int dir, float cruise, uint8_t style = 0);
    // SHOT (the authority's): that person is dead - gone from the world (the renderer has the
    // body), and after a while somebody new walks on to the site. Somebody on foot, on a seat or
    // IN A CAR; and `id` may be a car that drives with nobody of this world at its wheel (the
    // traffic the other way): then it is that car's driver. A car whose driver is shot rolls to
    // a stop on its lane (pose Park from that moment: no driver is shown), whoever else is in it
    // gets out, and two minutes later it is cleared away (the traffic gets another). False:
    // nobody there to shoot.
    bool kill(int id);
    // TAKEN (the authority's): a car that stands or barely rolls (kTakeSpeed) is the player's now
    // - it leaves this world; later another joins the traffic. Whoever sits in it is put out
    // beside it, its people who are out on foot stay behind: at the site they go about as the
    // locals do, out on the road they stand there a while and are gone. False: not a car that
    // can be taken now (it drives, or it is gone).
    bool take_car(int id);
    bool can_take_car(int id) const;
    // The players on foot (where their feet are; the authority's, given every frame): a car
    // does not drive into them - on the road it brakes in good time and waits while one stands
    // on its lane, on the site it waits as it does for its own people.
    void set_walkers(const std::vector<glm::vec3>& feet);
    // The players' cars near the road (the authority's, given every frame): see Obstacle.
    void set_obstacles(const std::vector<Obstacle>& obs);
    // STRUCK (the authority's): something hit that car - its velocity changes by `dv` (world,
    // m/s) and it turns by `spin` (rad/s). What goes along the car changes its speed, the rest
    // slides it sideways and dies away; after a hard knock its driver stands a few seconds
    // before he drives on. A parked car is shoved and stays where it comes to rest.
    bool push_car(int id, glm::vec3 dv, float spin);
    // Tests: every police car's siren on (and it stays on), or off.
    void set_sirens(bool on);

    // Authority: steps the world in fixed steps for the time passed. Replica: moves its clock.
    void update(float frame_dt);
    void views(std::vector<View>& out) const;
    // The state after the last step, not blended (tests).
    const std::vector<View>& state() const;

    // The whole state, for the other machines (authority) / taking it in (replica: from then on
    // this world only follows).
    void serialize(std::vector<uint8_t>& out) const;
    bool apply(const uint8_t* data, size_t size);
    bool replica() const;

    int    agent_count() const;
    int    step_count() const;
    // For tests and logs: how many people sit, walk; how many cars stand at the site.
    struct Census {
        int people = 0, sitting = 0, walking = 0, hidden = 0, cars = 0, cars_at_site = 0, cars_moving = 0;
        int asleep = 0;                  // people of added sites nobody is near (not counted as hidden)
        float longest_visit = 0.0f;      // seconds the car that has been at the site longest (not the parked ones) has been there
    };
    Census census() const;
    // The traffic: overtakes begun, how many of them were given up (somebody came), the kinds of
    // driver on the road now, sirens on, and the fastest car's speed.
    struct Traffic { int overtakes = 0, given_up = 0, overtaking = 0, slow = 0, speeders = 0, police = 0, sirens = 0; float fastest = 0.0f; };
    Traffic traffic() const;
    // True when no seat, spot, bay or stop is held by two agents and no number is not finite.
    bool   sane() const;
    // One line a car at the site (what it is doing): for logs and for finding what stands still.
    std::string describe_site() const;

private:
    struct Impl;
    Impl* d_;
};

}  // namespace vibe::npc
