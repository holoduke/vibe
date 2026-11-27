// The jeep: a vehicle to get in and drive.
//
//   model      assets/jeep/{body,wheel,steering}.mesh + jeep.json (tools/jeep/make_jeep.py,
//              Blender): one mesh group per material, the wheel once (drawn four times).
//   physics    Jolt's wheeled vehicle (PhysicsWorld::vehicle_*): a 1.25 t body on four sprung
//              wheels, an engine with a torque curve, clutch, four gears and reverse, a
//              differential on each axle, tyres whose grip falls off with slip, anti-roll bars.
//              It is an ordinary rigid body: it rolls over, bounces off trees and rails, and
//              lands as hard as it fell.
//   the world  The home tile has a heightfield in Jolt; the rest of the planet does not. So the
//              ground round the jeep is given to Jolt as a patch (254 m square at 2 m, made on
//              a worker thread from the same formula everything else stands on - the road's
//              shelf, crown and undulation included) that follows the jeep, and what stands
//              near it as boxes: tree trunks, the road's guard rail, buildings.
//   driving    Use (E) at the door gets in and out. Forward / back: accelerate, brake, and -
//              once stopped - reverse. Left / right or the stick: steering, lighter at speed.
//              Jump (space): handbrake - with the throttle from a standstill a burnout. The
//              camera follows behind and a little above (the middle mouse button: nearer,
//              farther, from the driver's seat); the right mouse button looks round, and it
//              swings back when let go.
//
// Drawn through the static-mesh path (CastleMeshGpu with a transform), so it is lit, shadowed
// and ray traced like everything else. Home's chart only (the course lies within it).
#include "engine/vk_engine/internal.h"
#include "engine/highway.h"
#include "engine/audio.h"
#include "engine/sound_space.h"
#include "engine/net/net.h"
#include "engine/terrain.h"
#include "engine/globe.h"
#include "engine/vk_engine/state/environment.h"
#include "engine/siren_synth.h"
#include "engine/tyre_synth.h"
#include "engine/vk_engine/state/car_traffic.h"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <sstream>
#include <string>

namespace vibe {

struct VulkanEngine::JeepState {
    bool loaded = false;
    std::string dir;                   // assets/<dir>
    size_t draw_first = 0;             // its first part in jeep_draw_
    bool posed = false;                // it stands somewhere (st is valid)
    PhysicsWorld::VehicleDesc tune;    // what kind of car it is
    // Tyre marks (per wheel: points along the ground, strength 0 = the end of a mark) and smoke.
    struct Mark { glm::vec3 p, side; float a; };
    std::vector<Mark> marks[4];
    struct Puff { glm::vec3 p, v; float age, life, size, seed, bright; };
    std::vector<Puff> puffs;
    // The body's box in the car's frame (between the wheels' outer faces, nose to tail): what smoke stays out of.
    void body_box(float& hx, float& z0, float& z1) const {
        hx = 0.0f; z0 = 0.0f; z1 = 0.0f;
        for (int w = 0; w < 4; ++w) {
            hx = std::max(hx, std::abs(wheel_pos[w].x) + 0.5f * wheel_width);
            z0 = std::min(z0, wheel_pos[w].z - wheel_radius - 0.45f); z1 = std::max(z1, wheel_pos[w].z + wheel_radius + 0.45f);
        }
        hx += 0.12f;
    }
    float puff_acc = 0.0f;
    float test_t = 0.0f;
    float tc = 1.0f;                   // traction control: the share of the throttle let through
    // Turbo boost (Sprint, while on the throttle): a tank of 3.2 s of extra push. It fills again by
    // itself once the boost has been off for a moment; run dry, it has to come back to a quarter first.
    float turbo = 1.0f, turbo_idle = 0.0f;
    bool  turbo_dry = false, boosting = false;
    // A burnout (handbrake + throttle from a standstill: jeep_tick): the front brakes hold, the
    // rear wheels spin. Held until the handbrake is let go - then it launches.
    bool  burnout = false;
    float squeal = 0.0f;               // how hard its tyres slide on a hard surface now (jeep_marks): the squeal's level
    // The view widens a few degrees with speed and with the boost (jeep_fov_extra), and shakes a
    // little while boosting.
    float fov_extra = 0.0f, boost_k = 0.0f;
    // Snow lying on it (winter: scene UBO car_snow, scene_ubo.glsl snow_cover): 1 all there, 0 gone.
    // Blown off by driving - speed x time, gone after about a minute - and settling again on a
    // car that stands in falling snow.
    float snow = 1.0f;
    // Lights: head lights on / off (L), brake lights by the pedal. Where they are (the car's
    // frame), and which drawn parts are lamps (1 head, 2 tail).
    bool      lights_on = false, prev_l = false, prev_mm = false;
    // Its sound (car_audio_tick): the sample set, and what the engine does.
    int       snd_set = -2;            // -2: not asked for yet; -1: none
    bool      snd_running = false, snd_has_last = false;
    float     snd_rpm = 900.0f, snd_load = 0.0f, throttle_in = 0.0f;
    float     snd_vol = 0.0f;                   // the loudness its voice has now: it follows what it should be, never jumps
    bool      snd_far = false;                  // ... and whether it is the cheap far voice
    int       snd_gear = 0;
    glm::vec3 snd_last{0.0f}, snd_vel{0.0f};
    bool      dusk_lit = false;        // its lights were put on by the dusk (once: L still switches them)
    float     brake_in = 0.0f;
    glm::vec3 head[2]{ { -0.4f, 0.85f, -1.8f }, { 0.4f, 0.85f, -1.8f } }, tail[2]{ { -0.6f, 0.6f, 1.8f }, { 0.6f, 0.6f, 1.8f } };
    std::vector<uint8_t> kind;
    std::vector<uint8_t> painted;      // which drawn parts carry the body's paint
    int       cam_mode = 1;            // the view: 0 near, 1 medium, 2 far, 3 from the driver's seat (middle mouse button)
    // The chase view's own: how much the car tumbles (0 on its wheels .. 1 rolling / in the air)
    // and the way it is heading on the ground plan, both followed smoothly (jeep_after_sim).
    float     cam_tumble = 0.0f;
    glm::vec2 cam_head{0.0f, -1.0f};
    float     cockpit_yaw = 0.0f, cockpit_pitch = 0.0f;       // the driver's head turned (right mouse button), seat view
    float     cam_h = 1.45f;
    // The driver: where his hips are (the car's frame), how far his legs are stretched
    // (0 an upright seat .. 1), and the steering wheel's rim radius.
    glm::vec3 hip{-0.4f, 0.76f, 0.42f};
    float     legs = 0.0f, rim_r = 0.18f;
    // Where the driver's feet and back go, measured from the car's model and written in its
    // file (docs/animation/actions.md "At the wheel"): "pedal_gas" the middle of the
    // accelerator's pad, "floor_y" the footwell's floor, "foot_rest" x and z of the left heel,
    // "seat_back_deg" how far the backrest leans. has_feet: the file has them.
    glm::vec3 pedal{0.0f};
    glm::vec2 foot_rest{0.0f};
    float     floor_y = 0.0f, back_deg = -1.0f;
    bool      has_feet = false;
    // What is drawn: the body's parts, four wheels' and the steering wheel's - in this order
    // in jeep_draw_. `local`: each part's bounds in its own frame.
    size_t n_body = 0, n_wheel = 0, n_steer = 0;
    // THE FRONT DOORS open. The models have no doors of their own - a body is one skin with
    // the gaps drawn on it - so the two front doors are cut out of body.mesh when the car is
    // loaded: every triangle whose middle lies in the door's box (beside the front seat, from
    // the sill up, from the windscreen pillar back to the one behind the seat) is the door's,
    // outer skin, glass, mirror and inner trim alike. Each turns on a hinge down its front
    // edge. Shut, it is the same triangles in the same place: nothing shows. After the
    // steering wheel in jeep_draw_: the left door's parts, then the right's.
    size_t    n_door[2] = { 0, 0 };
    glm::vec3 door_hinge[2]{};
    float     door[2] = { 0.0f, 0.0f };        // 0 shut .. 1 open
    float     door_want[2] = { 0.0f, 0.0f };   // what somebody getting in or out asks (npc_cars_place)
    float     door_hold = 0.0f;                // the player got in or out: the driver's door stands open this long yet
    bool      was_driving = false;
    std::vector<std::pair<glm::vec3, glm::vec3>> local;
    // From jeep.json.
    glm::vec3 wheel_pos[4]{};
    float     wheel_radius = 0.40f, wheel_width = 0.26f;
    glm::vec3 seat{-0.4f, 1.28f, 0.42f}, exit_left{-1.92f, 0.0f, 0.15f}, exit_right{1.92f, 0.0f, 0.15f};
    glm::vec3 steering_hub{-0.4f, 1.0f, -0.02f};
    float     steering_tilt = 38.0f;
    std::vector<PhysicsWorld::StaticBox> boxes;
    // Running.
    bool      spawned = false, driving = false;
    PhysicsWorld::VehicleState st{};
    glm::mat4 body_prev{1.0f};
    float     steer = 0.0f, steer_wheel = 0.0f;
    float     upside_t = 0.0f;
    // The camera.
    float     cam_dist = 4.6f, look_idle = 10.0f;
    float     cam_yaw = 0.0f, orbit_yaw = 0.0f, orbit_pitch = 0.0f;      // the view: behind the jeep, + where the driver looks
    bool      look_held = false;
    glm::vec3 cam_pos{0.0f};
    bool      cam_init = false;
    // The ground patch and the obstacles.
    struct Patch { std::vector<float> h; glm::vec2 org{0.0f}; glm::vec2 centre{0.0f}; };
    std::future<Patch> patch_job;
    // Multiplayer (jeep_net_*): who simulates it. net_mine: this machine, and it tells the
    // others. net_owner: another player's machine (their id) - it is drawn from their states.
    bool      net_mine = false, net_driven = false;      // (net_driven: its owner sits at the wheel)
    uint32_t  net_owner = 0;
    float     net_roll[4]{ 0.0f, 0.0f, 0.0f, 0.0f };     // the wheels' turn, carried on from their speed
    float     net_send = 0.0f;                           // seconds until its next state goes out
    bool      braking() const { return (driving || net_owner != 0) && brake_in > 0.05f; }
    bool      patch_busy = false, patch_valid = false;
    glm::vec2 patch_centre{1e9f};
    glm::vec2 obst_centre{1e9f};
    size_t    obst_sig = 0;
};

namespace {

constexpr int   kPatchN = 128;
constexpr float kPatchCell = 2.0f;
bool g_car_tune_old = false;            // VIBE_CAR_TUNE=old: the handling before 2026-11-27 (init_jeep)

// How much light falls on the ground from the sky by day, in lux, for the sun's height (degrees)
// and the cloud's share of the direct light (1 clear): a clear noon ~100,000, sunset ~400, the end
// of civil twilight (-6) ~3, night under 0.3. What a car's head light is measured against.
float daylight_lux(float sun_deg, float direct) {
    if (sun_deg > 0.0f) {
        const float s = std::sin(glm::radians(sun_deg));
        return 400.0f + (100000.0f * s * (0.25f + 0.75f * direct) + 18000.0f * std::sqrt(s));
    }
    if (sun_deg > -6.0f) return 3.0f * std::pow(400.0f / 3.0f, (sun_deg + 6.0f) / 6.0f);
    return std::max(0.25f, 3.0f * std::pow(0.1f, (-6.0f - sun_deg) / 6.0f));
}

struct JMat { const char* name; int tex; glm::vec3 tint; float tile; glm::vec3 emissive; bool full; };
// (-1 painted, -2 bare metal, -3 glass; 29 = the pavilion's woven fabric)
const JMat kJMats[] = {
    { "j_paint",       34, { 0.80f, 0.84f, 0.74f }, 0.9f, {}, false },
    { "j_paint_dark",  34, { 0.40f, 0.42f, 0.38f }, 0.9f, {}, false },
    { "j_steel",       -1, { 0.030f, 0.032f, 0.036f }, 1.0f, {}, false },
    { "j_metal",       -2, { 0.52f, 0.53f, 0.55f }, 1.0f, {}, false },
    { "j_rim",         34, { 0.70f, 0.74f, 0.66f }, 0.5f, {}, false },
    { "j_tyre",        -1, { 0.016f, 0.016f, 0.017f }, 1.0f, {}, false },
    { "j_seat",        35, { 0.95f, 0.92f, 0.86f }, 0.45f, {}, false },
    { "j_rubber",      -1, { 0.014f, 0.014f, 0.015f }, 1.0f, {}, false },
    { "j_glass",       -3, { 0.045f, 0.075f, 0.080f }, 1.0f, {}, false },
    { "j_light",       -1, { 0.90f, 0.89f, 0.82f }, 1.0f, { 0.9f, 0.86f, 0.70f }, true },
    { "j_light_red",   -1, { 0.55f, 0.03f, 0.02f }, 1.0f, { 0.8f, 0.04f, 0.03f }, true },
    { "j_light_amber", -1, { 0.80f, 0.38f, 0.03f }, 1.0f, {}, false },
    { "j_dash",        -1, { 0.022f, 0.024f, 0.022f }, 1.0f, {}, false },
    { "j_gauge",       -1, { 0.78f, 0.77f, 0.70f }, 1.0f, {}, false },
};
// The sports car's (assets/sportscar).
const JMat kSMats[] = {
    { "s_paint",       -6, { 0.62f, 0.022f, 0.018f }, 1.0f, {}, false },
    { "s_paint_dark",  -1, { 0.020f, 0.020f, 0.022f }, 1.0f, {}, false },
    { "s_carbon",      -1, { 0.030f, 0.031f, 0.034f }, 1.0f, {}, false },
    { "s_glass",       -3, { 0.045f, 0.075f, 0.080f }, 1.0f, {}, false },
    { "s_rim",         -2, { 0.60f, 0.61f, 0.63f }, 1.0f, {}, false },
    { "s_tyre",        -1, { 0.016f, 0.016f, 0.017f }, 1.0f, {}, false },
    { "s_brake",       -2, { 0.42f, 0.42f, 0.44f }, 1.0f, {}, false },
    { "s_caliper",     -1, { 0.85f, 0.62f, 0.03f }, 1.0f, {}, false },
    { "s_seat",        35, { 0.78f, 0.52f, 0.30f }, 0.5f, {}, false },
    { "s_dash",        -1, { 0.022f, 0.022f, 0.024f }, 1.0f, {}, false },
    { "s_metal",       -4, { 0.70f, 0.71f, 0.73f }, 1.0f, {}, false },
    { "s_rubber",      -1, { 0.014f, 0.014f, 0.015f }, 1.0f, {}, false },
    { "s_light",       -1, { 0.92f, 0.92f, 0.88f }, 1.0f, { 0.9f, 0.88f, 0.76f }, true },
    { "s_light_red",   -1, { 0.60f, 0.02f, 0.02f }, 1.0f, { 0.9f, 0.03f, 0.02f }, true },
    { "s_light_amber", -1, { 0.80f, 0.38f, 0.03f }, 1.0f, {}, false },
    { "s_gauge",       -1, { 0.80f, 0.79f, 0.74f }, 1.0f, {}, false },
};
const JMat* jmat(const std::string& n) {
    for (const JMat& m : kSMats) if (n == m.name) return &m;

    for (const JMat& m : kJMats) if (n == m.name) return &m;
    return &kJMats[2];
}

// The numbers after "key" in the JSON text (as many as there are before the next letter).
std::vector<float> json_nums(const std::string& t, const std::string& key, size_t from = 0, size_t* end = nullptr) {
    std::vector<float> out;
    size_t i = t.find("\"" + key + "\"", from);
    if (i == std::string::npos) return out;
    i = t.find(':', i);
    if (i == std::string::npos) return out;
    ++i;
    while (i < t.size()) {
        const char c = t[i];
        if (c == '-' || c == '.' || (c >= '0' && c <= '9')) {
            char* e = nullptr;
            out.push_back(std::strtof(t.c_str() + i, &e));
            i = size_t(e - t.c_str());
        } else if (c == '"' || c == '}' ) break;
        else ++i;
    }
    if (end) *end = i;
    return out;
}
glm::vec3 v3(const std::vector<float>& v, size_t at, glm::vec3 def) { return v.size() >= at + 3 ? glm::vec3(v[at], v[at + 1], v[at + 2]) : def; }

float hash_f(float v) { return std::floor(v * 4.0f); }
float rnd01() { static uint32_t s = 0x9E3779B9u; s = s * 1664525u + 1013904223u; return float(s >> 8) / 16777216.0f; }

// A place between two places (rigid: turned the short way, moved in a straight line).
glm::mat4 mix_pose(const glm::mat4& a, const glm::mat4& b, float t) {
    const glm::mat3 ra(a), rb(b);
    if (glm::determinant(ra) <= 0.0f || glm::determinant(rb) <= 0.0f) return t < 0.5f ? a : b;
    glm::mat4 m = glm::mat4_cast(glm::slerp(glm::quat_cast(ra), glm::quat_cast(rb), t));
    m[3] = glm::vec4(glm::mix(glm::vec3(a[3]), glm::vec3(b[3]), t), 1.0f);
    return m;
}

}  // namespace

bool VulkanEngine::jeep_driving() const { return jeep_ && jeep_->driving; }

void VulkanEngine::init_jeep() {
    if (JeepState* c = jeep_load("assets/jeep", "jeep.json")) {
        c->tune.com_height = 0.54f; c->tune.front_weight = 0.52f;
        c->tune.grip = 0.90f; c->tune.drag = 0.75f;                 // (all-terrain tyres, a brick in the wind)
        // (a flat four of the old kind: pulls from low down, done by 4600; a slow, sure box)
        c->tune.engine_idle_rpm = 800.0f; c->tune.torque_peak = 0.35f;
        c->tune.shift_up_full = 0.90f; c->tune.shift_up_easy = 0.50f; c->tune.shift_down_easy = 0.26f; c->tune.shift_down_full = 0.42f;
        c->tune.shift_time = 0.35f; c->tune.engine_loudness = 0.80f;
        jeep_all_.push_back(c);
    }
    // The others: a kind (its folder and what it drives like) in one or more colours.
    auto add = [&](const char* dir, glm::vec3 paint, const std::function<void(PhysicsWorld::VehicleDesc&)>& tune, float legs) {
        JeepState* c = jeep_load(dir, "car.json", &paint);
        if (!c) return;
        tune(c->tune);
        if (legs >= 0.0f) c->legs = legs;
        jeep_all_.push_back(c);
    };
    // A mid-engined sports car: low, stiff, rear-wheel drive. A flat-plane V8 that revs to 9000,
    // pulls hardest high up and is held to 8600 before the box - fast, a tenth of a second - changes.
    auto sports = [](PhysicsWorld::VehicleDesc& t) {
        t.mass = 1480.0f; t.com_height = 0.40f; t.front_weight = 0.42f;
        t.front_drive = 0.0f;
        t.gears = { 3.08f, 2.19f, 1.63f, 1.29f, 1.03f, 0.84f };
        t.engine_torque = 610.0f; t.engine_max_rpm = 9000.0f; t.final_drive = 4.2f;
        t.engine_idle_rpm = 1000.0f; t.engine_inertia = 0.20f; t.torque_peak = 0.68f;
        t.shift_up_full = 0.955f; t.shift_up_easy = 0.40f; t.shift_down_easy = 0.20f; t.shift_down_full = 0.45f;
        t.shift_time = 0.11f; t.engine_loudness = 1.0f;
        t.suspension_rest = 0.20f; t.suspension_travel_up = 0.07f; t.suspension_travel_down = 0.08f;
        t.suspension_hz = 2.3f; t.suspension_damping = 0.62f; t.antiroll = 9000.0f;
        t.max_steer = 0.50f; t.brake_torque = 4200.0f; t.handbrake_torque = 6500.0f;
        t.grip = 1.30f; t.wheel_mass = 20.0f;
        t.downforce = 1.9f; t.drag = 0.45f;
    };
    // An executive saloon: heavy, soft; most of its power to the rear. A four-cylinder that is
    // hardly heard: it changes up early and smoothly and only revs out when the pedal is down.
    auto saloon = [](PhysicsWorld::VehicleDesc& t) {
        t.mass = 1750.0f; t.com_height = 0.52f; t.front_weight = 0.54f;
        t.front_drive = 0.30f;
        t.gears = { 3.50f, 2.10f, 1.45f, 1.10f, 0.90f, 0.74f };
        t.engine_torque = 430.0f; t.engine_max_rpm = 6500.0f; t.final_drive = 3.5f;
        t.engine_idle_rpm = 750.0f; t.engine_inertia = 0.30f; t.torque_peak = 0.45f;
        t.shift_up_full = 0.92f; t.shift_up_easy = 0.34f; t.shift_down_easy = 0.17f; t.shift_down_full = 0.42f;
        t.shift_time = 0.28f; t.engine_loudness = 0.42f;
        t.suspension_rest = 0.24f; t.suspension_travel_up = 0.09f; t.suspension_travel_down = 0.10f;
        t.suspension_hz = 1.5f; t.suspension_damping = 0.58f; t.antiroll = 5500.0f;
        t.max_steer = 0.55f; t.brake_torque = 3400.0f; t.handbrake_torque = 5600.0f;
        t.grip = 1.08f; t.downforce = 0.35f; t.drag = 0.38f;
    };
    // A pickup: tall, heavy at the front, long travel, four-wheel drive. A big V6 that pulls from
    // idle and is done by 4800; a slow box that changes early.
    auto pickup = [](PhysicsWorld::VehicleDesc& t) {
        t.mass = 2150.0f; t.com_height = 0.70f; t.front_weight = 0.57f;
        t.front_drive = 0.40f;
        t.gears = { 3.80f, 2.30f, 1.50f, 1.10f, 0.85f };
        t.engine_torque = 520.0f; t.engine_max_rpm = 4800.0f; t.final_drive = 3.9f;
        t.engine_idle_rpm = 650.0f; t.engine_inertia = 0.50f; t.torque_peak = 0.38f;
        t.shift_up_full = 0.90f; t.shift_up_easy = 0.46f; t.shift_down_easy = 0.24f; t.shift_down_full = 0.40f;
        t.shift_time = 0.40f; t.engine_loudness = 0.85f;
        t.suspension_rest = 0.30f; t.suspension_travel_up = 0.13f; t.suspension_travel_down = 0.16f;
        t.suspension_hz = 1.5f; t.suspension_damping = 0.55f; t.antiroll = 3600.0f;
        t.max_steer = 0.56f; t.brake_torque = 3600.0f; t.handbrake_torque = 6200.0f;
        t.wheel_mass = 28.0f;
        t.grip = 0.97f; t.drag = 0.80f;
    };
    add("assets/sportscar", { 0.62f, 0.022f, 0.018f }, sports, 1.0f);
    add("assets/saloon",    { 0.20f, 0.21f, 0.23f }, saloon, -1.0f);
    add("assets/pickup",    { 0.72f, 0.72f, 0.70f }, pickup, -1.0f);
    add("assets/sportscar", { 0.80f, 0.52f, 0.02f }, sports, 1.0f);
    add("assets/saloon",    { 0.025f, 0.045f, 0.11f }, saloon, -1.0f);
    add("assets/pickup",    { 0.045f, 0.10f, 0.065f }, pickup, -1.0f);
    add("assets/saloon",    { 0.70f, 0.70f, 0.68f }, saloon, -1.0f);
    if (jeep_all_.empty()) return;
    // HANDLING, 2026-11-27 ("feels a bit sluggish, as if gravity is too low"). Every kind: it
    // falls at 1.45 g (lands sooner off a crest, floats less), springs a little stiffer and
    // better damped (the body settles in one swing), the tyres a little more grip, the engine
    // spins up faster. Steering is quicker in jeep_tick (kSteerRate). Measured with
    // VIBE_TEST_JEEP_PERF (docs/ARCHITECTURE.md "Handling, 2026-11-27"). VIBE_CAR_TUNE=old: as before.
    g_car_tune_old = std::getenv("VIBE_CAR_TUNE") && std::string(std::getenv("VIBE_CAR_TUNE")) == "old";
    if (!g_car_tune_old)
        for (JeepState* c : jeep_all_) {
            PhysicsWorld::VehicleDesc& t = c->tune;
            t.gravity = 14.2f;
            t.suspension_hz *= 1.12f;
            t.suspension_damping = std::min(t.suspension_damping + 0.06f, 0.75f);
            t.grip *= 1.06f;
            t.engine_inertia *= 0.80f;
        }
    const int pick = std::getenv("VIBE_TEST_CAR") ? std::atoi(std::getenv("VIBE_TEST_CAR")) : 0;      // (tests: which one is simulated first)
    jeep_ = jeep_all_[size_t(std::clamp(pick, 0, int(jeep_all_.size()) - 1))];
    npc_cars_init();
}

VulkanEngine::JeepState* VulkanEngine::jeep_load(const std::string& dir, const std::string& json, const glm::vec3* paint) {
    JeepState* jp = new JeepState();
    JeepState& J = *jp;
    J.dir = dir;
    J.draw_first = jeep_draw_.size();
    std::ifstream jf(dir + "/" + json);
    if (!jf) { log::warnf("[jeep] %s/%s missing - no such car", dir.c_str(), json.c_str()); delete jp; return nullptr; }
    std::stringstream ss; ss << jf.rdbuf();
    const std::string js = ss.str();
    {
        const size_t w = js.find("\"wheels\"");
        const char* names[4] = { "front_left", "front_right", "rear_left", "rear_right" };
        const glm::vec3 def[4] = { { -0.75f, 0.4f, -1.2f }, { 0.75f, 0.4f, -1.2f }, { -0.75f, 0.4f, 1.2f }, { 0.75f, 0.4f, 1.2f } };
        for (int i = 0; i < 4; ++i) J.wheel_pos[i] = v3(json_nums(js, names[i], w == std::string::npos ? 0 : w), 0, def[i]);
        if (auto v = json_nums(js, "wheel_radius"); !v.empty()) J.wheel_radius = v[0];
        if (auto v = json_nums(js, "wheel_width"); !v.empty()) J.wheel_width = v[0];
        J.seat = v3(json_nums(js, "seat_driver"), 0, J.seat);
        J.exit_left = v3(json_nums(js, "exit_left"), 0, J.exit_left);
        J.exit_right = v3(json_nums(js, "exit_right"), 0, J.exit_right);
        J.steering_hub = v3(json_nums(js, "steering_hub"), 0, J.steering_hub);
        { const auto v = json_nums(js, "headlights"); J.head[0] = v3(v, 0, J.head[0]); J.head[1] = v3(v, 3, J.head[1]); }
        { const auto v = json_nums(js, "tail_lights"); J.tail[0] = v3(v, 0, J.tail[0]); J.tail[1] = v3(v, 3, J.tail[1]); }
        if (auto v = json_nums(js, "steering_tilt_deg"); !v.empty()) J.steering_tilt = v[0];
        size_t at = js.find("\"collision_boxes\"");
        while (at != std::string::npos) {
            size_t e1 = 0, e2 = 0;
            const auto c = json_nums(js, "center", at, &e1);
            if (c.size() < 3) break;
            const auto h = json_nums(js, "half_extents", e1, &e2);
            if (h.size() < 3) break;
            J.boxes.push_back({ glm::vec3(c[0], c[1], c[2]), glm::vec3(h[0], h[1], h[2]) });
            at = e2;
        }
    }
    if (J.boxes.empty()) J.boxes.push_back({ glm::vec3(0.0f, 0.75f, 0.0f), glm::vec3(0.8f, 0.35f, 1.8f) });

    // `part`: 0 the whole file but the doors, 1 / 2 the left / right front door alone, -1 everything.
    const bool has_doors = dir.find("jeep") == std::string::npos && !std::getenv("VIBE_NO_CAR_DOORS");      // (the jeep is open-sided)
    auto door_of = [&](const glm::vec3& c) -> int {
        if (!has_doors) return 0;
        const float sx = std::abs(J.seat.x);
        if (c.y < 0.30f || c.y > 2.4f || c.z < J.seat.z - 1.05f || c.z > J.seat.z + 0.50f) return 0;
        return c.x < -(sx + 0.30f) ? 1 : c.x > sx + 0.30f ? 2 : 0;
    };
    auto add_file = [&](const std::string& path, int copies, int part = -1) -> size_t {
        auto groups = game::load_mesh_groups(path.c_str());
        if (part >= 0)
            for (auto& g : groups) {
                // keep only this part's triangles (and the corners they use)
                game::LevelMesh& lm = g.mesh;
                std::vector<uint32_t> idx, remap(lm.verts.size(), 0xFFFFFFFFu);
                std::vector<game::LevelVertex> verts;
                for (size_t t = 0; t + 2 < lm.idx.size(); t += 3) {
                    const glm::vec3 c = (lm.verts[lm.idx[t]].p + lm.verts[lm.idx[t + 1]].p + lm.verts[lm.idx[t + 2]].p) / 3.0f;
                    if (door_of(c) != part) continue;
                    for (int k = 0; k < 3; ++k) {
                        uint32_t& r = remap[lm.idx[t + size_t(k)]];
                        if (r == 0xFFFFFFFFu) { r = uint32_t(verts.size()); verts.push_back(lm.verts[lm.idx[t + size_t(k)]]); }
                        idx.push_back(r);
                    }
                }
                lm.verts.swap(verts); lm.idx.swap(idx);
            }
        size_t n = 0;
        std::vector<CastleMeshGpu> made;
        std::vector<std::pair<glm::vec3, glm::vec3>> bounds;
        std::vector<uint8_t> kinds;
        std::vector<uint8_t> paints;
        for (auto& g : groups) {
            if (g.mesh.idx.empty()) continue;
            const JMat* m = jmat(g.name);
            game::LevelMesh& lm = g.mesh;
            lm.tex_albedo = lm.tex_normal = m->tex;
            lm.tint = m->tint; lm.fallback = m->tex >= 0 ? m->tint * 0.5f : m->tint;
            if (paint && g.name == "s_paint") lm.tint = lm.fallback = *paint;
            lm.emissive = m->emissive; lm.full_emissive = m->full;
            lm.relief = 0.0f;
            lm.bmin = glm::vec3(1e9f); lm.bmax = glm::vec3(-1e9f);
            for (auto& v : lm.verts) { v.u /= m->tile; v.v /= m->tile; lm.bmin = glm::min(lm.bmin, v.p); lm.bmax = glm::max(lm.bmax, v.p); }
            made.push_back(make_mesh_part(lm));
            kinds.push_back(g.name.find("light_red") != std::string::npos ? 2 : (g.name.size() > 6 && g.name.compare(g.name.size() - 6, 6, "_light") == 0) ? 1 : 0);
            bounds.push_back({ lm.bmin, lm.bmax });
            paints.push_back(g.name == "s_paint" ? 1 : 0);
            ++n;
        }
        build_mesh_part_blas(made);
        for (auto& g : made) jeep_owned_.push_back(g);
        for (int c = 0; c < copies; ++c)
            for (size_t i = 0; i < made.size(); ++i) { jeep_draw_.push_back(made[i]); J.local.push_back(bounds[i]); J.kind.push_back(kinds[i]); J.painted.push_back(paints[i]); }
        return n;
    };
    J.n_body = add_file(dir + "/body.mesh", 1, 0);
    J.n_wheel = add_file(dir + "/wheel.mesh", 4);
    J.n_steer = add_file(dir + "/steering.mesh", 1);
    if (!J.n_body || !J.n_wheel) { log::warnf("[jeep] %s/*.mesh missing - no such car", dir.c_str()); jeep_draw_.resize(J.draw_first); delete jp; return nullptr; }
    J.loaded = true;
    J.lights_on = std::getenv("VIBE_TEST_JEEP_LIGHTS") != nullptr;
    J.hip = J.seat + glm::vec3(0.0f, -0.52f, 0.0f);
    J.hip = v3(json_nums(js, "hip_driver"), 0, J.hip);
    if (auto v = json_nums(js, "legs"); !v.empty()) J.legs = glm::clamp(v[0], 0.0f, 1.0f);
    if (auto p = json_nums(js, "pedal_gas"), f = json_nums(js, "floor_y"), r = json_nums(js, "foot_rest"); p.size() >= 3 && !f.empty() && r.size() >= 2) {
        J.pedal = glm::vec3(p[0], p[1], p[2]); J.floor_y = f[0]; J.foot_rest = glm::vec2(r[0], r[1]); J.has_feet = true;
    }
    if (auto v = json_nums(js, "seat_back_deg"); !v.empty()) J.back_deg = glm::clamp(v[0], 0.0f, 40.0f);
    if (J.n_steer) {
        float r = 0.0f;
        for (size_t i = J.n_body + 4 * J.n_wheel; i < J.local.size(); ++i)
            r = std::max({ r, std::abs(J.local[i].first.x), std::abs(J.local[i].second.x) });
        if (r > 0.08f) J.rim_r = r - 0.015f;
    }
    for (int sd = 0; sd < 2 && has_doors; ++sd) {
        const size_t first = J.local.size();
        J.n_door[sd] = add_file(dir + "/body.mesh", 1, sd + 1);
        glm::vec3 lo(1e9f), hi(-1e9f);
        for (size_t i = first; i < J.local.size(); ++i) { lo = glm::min(lo, J.local[i].first); hi = glm::max(hi, J.local[i].second); }
        // (the hinge: down the door's front edge - forward is -z - a little inside its outer skin)
        if (J.n_door[sd]) J.door_hinge[sd] = glm::vec3(sd == 0 ? lo.x + 0.06f : hi.x - 0.06f, 0.0f, lo.z + 0.02f);
    }
    for (int i = 0; i < 4; ++i) J.tune.wheel_pos[i] = J.wheel_pos[i];
    J.tune.wheel_radius = J.wheel_radius; J.tune.wheel_width = J.wheel_width;
    J.tune.boxes = J.boxes;
    log::infof("[jeep] %s loaded: %zu body parts, %zu wheel parts, %zu steering parts, %zu collision boxes", dir.c_str(), J.n_body, J.n_wheel, J.n_steer, J.boxes.size());
    return jp;
}

void VulkanEngine::destroy_jeep() {
    for (JeepState* c : jeep_all_) if (c->patch_busy && c->patch_job.valid()) c->patch_job.wait();
    if (physics_ && jeep_) physics_->vehicle_remove();
    jeep_draw_.clear();
    destroy_mesh_parts(jeep_owned_);
    for (JeepState* c : jeep_all_) delete c;
    if (audio_) for (auto& kv : car_voices_) audio_->engine_stop(kv.second);
    car_voices_.clear();
    if (cars_x_) { if (audio_) for (auto& kv : cars_x_->sirens) audio_->source_stop(kv.second.src); cars_x_->sirens.clear(); if (audio_ && cars_x_->squeal_src) audio_->source_stop(cars_x_->squeal_src); cars_x_->squeal_src = 0; }
    for (NpcCar& c : npc_cars_) delete c.car;
    npc_cars_.clear();
    jeep_all_.clear();
    jeep_ = nullptr;
}

// A car standing somewhere, not simulated: its parts where they would be at rest.
void VulkanEngine::jeep_park(JeepState& J, glm::vec3 pos, float yaw) {
    J.st = PhysicsWorld::VehicleState{};
    J.st.body = glm::translate(glm::mat4(1.0f), pos) * glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    for (int i = 0; i < 4; ++i) J.st.wheel[i] = J.st.body * glm::translate(glm::mat4(1.0f), J.wheel_pos[i]);
    J.body_prev = J.st.body;
    J.posed = true;
    jeep_place(J, J.st, 0.0f);
}

// WHAT A WHEEL STANDS ON at (x, z): the same ground the simulated car's physics is given
// (jeep_feed_world) - inside the home tile its own heightfield, outside the world's formula with
// the road's shelf and crown; a paved place (drive_pads_) and the kerb round it; a bridge's deck.
// Cars that are placed rather than simulated (the NPC traffic, a car standing where it was left,
// one that is struck and slides) stand on this too, so that they are neither in the road nor
// over it (2026-11-27: the NPC cars took their height from the course's centre line, which runs
// in 20 m pieces and knows nothing of a bridge's deck or a region's ground).
float VulkanEngine::car_ground(float x, float z) const {
    float gh = (std::abs(x) < 1022.0f && std::abs(z) < 1022.0f) ? sample_terrain_height(x, z) : world_ground_height(x, z, rt_.water_level, 9);
    for (const DrivePad& p : drive_pads_) {
        const float dx = std::max({ p.x0 - x, x - p.x1, 0.0f }), dz = std::max({ p.z0 - z, z - p.z1, 0.0f });
        const float d = std::sqrt(dx * dx + dz * dz);
        if (d > 1.5f) continue;
        const float t = glm::clamp((x - p.x0) / std::max(p.x1 - p.x0, 0.01f), 0.0f, 1.0f);
        gh = std::max(gh, glm::mix(glm::mix(p.h0, p.h1, t), gh, d / 1.5f));
    }
    if (float dh = 0.0f; highway::deck(x, z, dh)) gh = std::max(gh, dh);
    return gh;
}

// A car standing on that ground at (pos.x, pos.z), turned by `yaw` (as jeep_park): the ground is
// found under each of its four wheels, and the body lies on the plane through them - height,
// pitch along it and roll across it. `tilt` (if given) gets (pitch, roll) in radians.
glm::mat4 VulkanEngine::car_on_ground(const JeepState& J, glm::vec3 pos, float yaw, glm::vec2* tilt) const {
    const glm::mat4 R = glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    float h[4];
    for (int w = 0; w < 4; ++w) {
        const glm::vec3 p = glm::vec3(R * glm::vec4(J.wheel_pos[w].x, 0.0f, J.wheel_pos[w].z, 0.0f));
        h[w] = car_ground(pos.x + p.x, pos.z + p.z);
    }
    const float zf = 0.5f * (J.wheel_pos[0].z + J.wheel_pos[1].z), zr = 0.5f * (J.wheel_pos[2].z + J.wheel_pos[3].z);
    const float xl = 0.5f * (J.wheel_pos[0].x + J.wheel_pos[2].x), xr = 0.5f * (J.wheel_pos[1].x + J.wheel_pos[3].x);
    const float hf = 0.5f * (h[0] + h[1]), hr = 0.5f * (h[2] + h[3]), hl = 0.5f * (h[0] + h[2]), hri = 0.5f * (h[1] + h[3]);
    const float base = std::max(std::abs(zr - zf), 0.5f), track = std::max(std::abs(xr - xl), 0.5f);
    // (front is -z: a front higher than the rear is the nose up, a positive turn about +x;
    //  the right side higher is a positive turn about +z)
    const float pitch = std::atan2(hf - hr, base), roll = std::atan2(hri - hl, track);
    // (the height of the plane under the origin)
    const float y = hr + (hf - hr) * (0.0f - zr) / (zf - zr != 0.0f ? zf - zr : 1.0f) + (hri - hl) * (0.0f - 0.5f * (xl + xr)) / track;
    if (tilt) *tilt = glm::vec2(pitch, roll);
    return glm::translate(glm::mat4(1.0f), glm::vec3(pos.x, y, pos.z)) * R *
           glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1.0f, 0.0f, 0.0f)) * glm::rotate(glm::mat4(1.0f), roll, glm::vec3(0.0f, 0.0f, 1.0f));
}

// A car standing on the ground where it is put (car_on_ground), not simulated.
void VulkanEngine::car_settle(JeepState& J, glm::vec3 pos, float yaw) {
    jeep_park(J, pos, yaw);
    J.st.body = car_on_ground(J, pos, yaw);
    for (int i = 0; i < 4; ++i) J.st.wheel[i] = J.st.body * glm::translate(glm::mat4(1.0f), J.wheel_pos[i]);
    J.body_prev = J.st.body;
    jeep_place(J, J.st, 0.0f);
}

// The simulated car: made in the physics where it stands.
void VulkanEngine::jeep_spawn() {

    JeepState& J = *jeep_;
    if (!J.loaded || !physics_ || !launch_site_placed_) return;
    if (!J.posed) return;
    PhysicsWorld::VehicleDesc d = J.tune;
    const glm::vec3 p(J.st.body[3]);
    const glm::vec3 f0 = -glm::vec3(J.st.body[2]);
    d.position = p + glm::vec3(0.0f, 0.12f, 0.0f);
    d.yaw = std::atan2(-f0.x, -f0.z);
    J.spawned = physics_->vehicle_create(d) != 0;
    J.patch_valid = false; J.patch_centre = glm::vec2(1e9f); J.obst_centre = glm::vec2(1e9f);
    if (J.spawned) {
        physics_->vehicle_state(J.st);
        J.body_prev = J.st.body;
        log::infof("[jeep] parked at (%.1f, %.1f, %.1f)", p.x, p.y, p.z);
    }
}

// The ground and the obstacles Jolt needs round the jeep (see the file's header).
void VulkanEngine::jeep_feed_world() {
    JeepState& J = *jeep_;
    const glm::vec3 pos(J.st.body[3]);
    const glm::vec2 p2(pos.x, pos.z);
    // -- the ground patch --
    if (J.patch_busy && J.patch_job.valid() && J.patch_job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        JeepState::Patch P = J.patch_job.get();
        J.patch_busy = false;
        physics_->vehicle_set_ground(std::move(P.h), kPatchN, P.org, kPatchCell);
        J.patch_centre = P.centre; J.patch_valid = true;
    }
    if (!J.patch_busy && (!J.patch_valid || glm::distance(p2, J.patch_centre) > 48.0f)) {
        const glm::vec2 c = glm::round(p2 / 16.0f) * 16.0f;
        const float wl = rt_.water_level;
        const std::vector<DrivePad> pads = drive_pads_;
        J.patch_busy = true;
        J.patch_job = std::async(std::launch::async, [c, wl, pads]() {
            JeepState::Patch P;
            P.centre = c;
            P.org = c - glm::vec2(0.5f * float(kPatchN - 1) * kPatchCell);
            P.h.resize(size_t(kPatchN) * kPatchN);
            for (int j = 0; j < kPatchN; ++j)
                for (int i = 0; i < kPatchN; ++i) {
                    const float x = P.org.x + float(i) * kPatchCell, z = P.org.y + float(j) * kPatchCell;
                    // (inside the home tile its own heightfield is the ground: a hole here)
                    float gh = (std::abs(x) < 1022.0f && std::abs(z) < 1022.0f) ? 1e30f : world_ground_height(x, z, wl, 9);
                    // (paved places: their own height, and round them the ground comes up to
                    //  it over a metre and a half - a kerb a wheel rolls over, not a wall)
                    if (gh < 1e29f)
                        for (const DrivePad& p : pads) {
                            const float dx = std::max({ p.x0 - x, x - p.x1, 0.0f }), dz = std::max({ p.z0 - z, z - p.z1, 0.0f });
                            const float d = std::sqrt(dx * dx + dz * dz);
                            if (d > 1.5f) continue;
                            const float t = glm::clamp((x - p.x0) / std::max(p.x1 - p.x0, 0.01f), 0.0f, 1.0f);
                            const float ph = glm::mix(p.h0, p.h1, t);
                            gh = std::max(gh, glm::mix(ph, gh, d / 1.5f));
                        }
                    if (float dh = 0.0f; gh < 1e29f && highway::deck(x, z, dh)) gh = std::max(gh, dh);       // (a bridge's deck)
                    P.h[size_t(j) * kPatchN + i] = gh;

                }
            return P;
        });
        if (!J.patch_valid) {                       // (the first one: it must be there before the jeep falls)
            JeepState::Patch P = J.patch_job.get();
            J.patch_busy = false;
            physics_->vehicle_set_ground(std::move(P.h), kPatchN, P.org, kPatchCell);
            J.patch_centre = P.centre; J.patch_valid = true;
        }
    }
    // -- what stands near --
    const size_t sig = wv_cols_.size() * 31u + world_.aabbs.size() + lane_rock_cols_.size() * 7u;
    if (glm::distance(p2, J.obst_centre) > 22.0f || sig != J.obst_sig) {
        J.obst_centre = p2; J.obst_sig = sig;
        std::vector<PhysicsWorld::OrientedBox> ob;
        const float reach = 70.0f;
        auto take = [&](const collision::AABB& a) {
            const glm::vec3 c = 0.5f * (a.min + a.max), h = 0.5f * (a.max - a.min);
            if (std::abs(c.x - pos.x) > reach + h.x || std::abs(c.z - pos.z) > reach + h.z) return;
            if (h.x > 400.0f || h.z > 400.0f) return;
            if (a.max.y < pos.y + 0.45f && (h.x > 0.24f && h.z > 0.24f) && (h.x > 2.0f || h.z > 2.0f)) return;       // (a floor under the wheels: that is the ground patch's, not a wall)
            ob.push_back({ c, h, 0.0f });
        };
        for (const auto& a : wv_cols_) take(a);
        for (const auto& a : world_.aabbs) take(a);
        hw_rail_boxes(pos, reach, ob);
        // (the rock formations are in the physics world as their own surface: rock_formations_gpu.cpp)
        if (ob.size() > 900) ob.resize(900);
        physics_->vehicle_set_obstacles(ob.data(), ob.size());
    }
}

// Use at the jeep: in, or out. True when the key was taken.
bool VulkanEngine::jeep_use() {
    if (!jeep_ || !jeep_->spawned) return false;
    // Riding beside another player's driver: Use gets out, on the passenger's side.
    if (jeep_ride_ >= 0) {
        if (size_t(jeep_ride_) < jeep_all_.size()) {
            const JeepState& R = *jeep_all_[size_t(jeep_ride_)];
            glm::vec3 out = glm::vec3(R.st.body * glm::vec4(R.exit_right, 1.0f));
            out.y = std::max(out.y, sample_terrain_height(out.x, out.z)) + 1.0f;
            player_.position = out;
            player_.velocity = glm::vec3(0.0f);
            prev_player_position_ = player_.position;
        }
        jeep_ride_ = -1;
        return true;
    }
    // A car another player is driving, within reach: in beside them.
    if (!jeep_->driving) {
        int best_i = -1; float bd = 3.6f;
        for (size_t i = 0; i < jeep_all_.size(); ++i) {
            const JeepState* c = jeep_all_[i];
            if (!c->posed || !(c->net_owner && c->net_driven)) continue;
            const float dd = glm::distance(player_.eye_position(), glm::vec3(c->st.body[3]) + glm::vec3(0.0f, 1.0f, 0.0f));
            if (dd < bd) { bd = dd; best_i = int(i); }
        }
        if (best_i >= 0) { jeep_ride_ = best_i; fly_mode_ = false; return true; }
    }
    // Somebody else's car that stands with nobody in it (the station's traffic): it is taken.
    if (!jeep_->driving) npc_take_car(3.6f);
    if (!jeep_->driving) {
        // The nearest car within reach. One that is not the simulated one becomes it; the
        // other stays where it stands.
        JeepState* best = nullptr; float bd = 3.6f;
        for (JeepState* c : jeep_all_) {
            if (!c->posed || (c->net_owner && c->net_driven)) continue;      // (not one somebody else is driving)
            const float dd = glm::distance(player_.eye_position(), glm::vec3(c->st.body[3]) + glm::vec3(0.0f, 1.0f, 0.0f));
            if (dd < bd) { bd = dd; best = c; }
        }
        if (!best) return false;
        if (best != jeep_) {
            if (jeep_->patch_busy && jeep_->patch_job.valid()) jeep_->patch_job.wait();
            jeep_->patch_busy = false;
            physics_->vehicle_remove();
            jeep_->spawned = false;
            jeep_ = best;
            jeep_spawn();
            if (!jeep_->spawned) return false;
            jeep_feed_world();
        }
        // (one that another machine simulated: it is made here, where it stands, and is ours)
        jeep_->net_owner = 0; jeep_->net_driven = false;
        if (!jeep_->spawned) {
            jeep_spawn();
            if (!jeep_->spawned) return false;
            jeep_feed_world();
        }
    }
    JeepState& J = *jeep_;
    const glm::vec3 pos(J.st.body[3]);
    if (J.driving) {

        // Out, on the driver's side - or the other when that is blocked by the ground.
        glm::vec3 out = glm::vec3(J.st.body * glm::vec4(J.exit_left, 1.0f));
        const float g = sample_terrain_height(out.x, out.z);
        out.y = std::max(out.y, g) + 1.0f;
        J.driving = false;
        player_.position = out;
        player_.velocity = glm::vec3(0.0f);
        prev_player_position_ = player_.position;
        J.burnout = false; physics_->vehicle_burnout(false);
        physics_->vehicle_input(0.0f, 0.0f, 0.0f, 1.0f);
        return true;
    }
    if (glm::distance(player_.eye_position(), pos + glm::vec3(0.0f, 1.0f, 0.0f)) > 3.6f) return false;
    J.driving = true;
    J.net_mine = true;
    J.cam_init = false;
    J.look_idle = 10.0f;
    fly_mode_ = false;
    return true;
}

// Before the physics ticks: the driver's controls to the vehicle.
void VulkanEngine::jeep_tick(InputFrame& in, float dt) {
    if (!jeep_ || !jeep_->loaded || !physics_) return;
    npc_car_test(in);                       // (VIBE_TEST_NPC_CAR: into a car of the NPC world, and away)
    // (VIBE_TEST_RIDE=1: gets in beside the first driver another machine tells of - tests)
    if (static const bool tr = std::getenv("VIBE_TEST_RIDE") != nullptr; tr && jeep_ride_ < 0)
        for (size_t i = 0; i < jeep_all_.size(); ++i)
            if (jeep_all_[i]->posed && jeep_all_[i]->net_owner && jeep_all_[i]->net_driven) { jeep_ride_ = int(i); break; }
    if (jeep_ride_ >= 0) {                  // (a passenger: looks round and may shoot, does not walk)
        in.fwd = in.back = in.left = in.right = in.jump = in.sprint = in.crawl = false;
        in.move_x = in.move_y = 0.0f;
    }
    // VIBE_TEST_USE=<frame>: the use key at that frame (tests: into the car the player stands at).
    if (static const int use_at = std::getenv("VIBE_TEST_USE") ? std::atoi(std::getenv("VIBE_TEST_USE")) : -1; use_at >= 0 && int(frame_number_) == use_at)
        log::infof("[jeep] test: use -> %d; driving %d, %zu cars", int(jeep_use()), int(jeep_->driving), jeep_all_.size());
    JeepState& J = *jeep_;
    if (!J.posed) {
        // Parked outside the pavilion: on the concrete in front of its terrace, in a row.
        if (!(launch_site_placed_ && globe::current().home)) return;
        for (size_t i = 0; i < jeep_all_.size(); ++i) {
            glm::vec3 p(office_pos_.x + 15.0f - 9.0f * float(i), 0.0f, office_pos_.z + 25.0f);
            float yaw = 1.5708f;
            if (!parking_bays_.empty()) {
                // (spread over the bays, the same every run. Each car its own bay: with 42
                //  bays a step of 7 came round after six cars, and the seventh and eighth
                //  stood inside the first and second - so a taken bay is passed over)
                size_t bay = (3 + i * 7) % parking_bays_.size();
                for (size_t tries = 0; tries < parking_bays_.size(); ++tries) {
                    bool taken = false;
                    for (size_t j = 0; j < i && !taken; ++j)
                        taken = glm::distance(glm::vec3(jeep_all_[j]->st.body[3]) * glm::vec3(1.0f, 0.0f, 1.0f),
                                              glm::vec3(parking_bays_[bay].x, 0.0f, parking_bays_[bay].z)) < 1.0f;
                    if (!taken) break;
                    bay = (bay + 1) % parking_bays_.size();
                }
                const glm::vec4 b = parking_bays_[bay];
                p = glm::vec3(b.x, 0.0f, b.z); yaw = b.w;
            }
            if (const char* at = std::getenv("VIBE_TEST_JEEP_AT"); at && jeep_all_[i] == jeep_)      // "<x>,<z>[,<yaw>]": somewhere else (tests)
                std::sscanf(at, "%f,%f,%f", &p.x, &p.z, &yaw);
            p.y = world_ground_height(p.x, p.z, rt_.water_level, 9);
            for (const auto& a : world_.aabbs)                    // (a slab it stands on: the apron)
                if (p.x > a.min.x && p.x < a.max.x && p.z > a.min.z && p.z < a.max.z && a.max.y > p.y && a.max.y < p.y + 1.0f) p.y = a.max.y;
            car_settle(*jeep_all_[i], p, yaw);        // (on its four wheels: the bay may slope)
        }
    }
    cars_collide(dt);                       // (the other cars are solid to this one, and what is struck moves: CARS STRIKE CARS, below)
    if (!J.spawned) { if (J.net_owner) return; jeep_spawn(); if (!J.spawned) return; }      // (another player's machine simulates it: jeep_net_receive)
    if (!globe::current().home) return;                    // (another part of the planet: it waits where it was left)
    jeep_feed_world();
    if (state_ != State::Playing) return;
    if (!J.driving) {
        // VIBE_TEST_JEEP=<throttle>[,<steer>]: drives itself (tests).
        static const char* test = std::getenv("VIBE_TEST_JEEP");
        if (test) {
            float th = 0.0f, stv = 0.0f; std::sscanf(test, "%f,%f", &th, &stv);
            if (frame_number_ > 90) { J.driving = true; J.net_mine = true; J.cam_init = false; }
            (void)th; (void)stv;
        }
        J.burnout = false; physics_->vehicle_burnout(false);
        physics_->vehicle_input(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    {   // L: the head lights. The middle mouse button: the view nearer / farther.
        GLFWwindow* w = window_->handle();
        const bool l = glfwGetKey(w, GLFW_KEY_L) == GLFW_PRESS;
        if (l && !J.prev_l) J.lights_on = !J.lights_on;
        if (static const bool tl = std::getenv("VIBE_TEST_JEEP_LIGHTS") != nullptr; tl) J.lights_on = true;      // (tests: the head lights on)
        if (static const char* ts = std::getenv("VIBE_TEST_CAR_SNOW"); ts && frame_number_ < 100) J.snow = float(std::atof(ts));     // (tests: the share of its snow the driven car starts with)
        J.prev_l = l;
        // (Dusk: a car that is driven puts its lights on by itself - once; after that L is the driver's.)
        if (cars_need_lights()) { if (!J.dusk_lit) { J.dusk_lit = true; J.lights_on = true; } }
        else { if (J.dusk_lit) J.lights_on = false; J.dusk_lit = false; }      // (and off again by itself at dawn: 2026-11-27, lights put on in the dusk burned all day)
        const bool mm = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        if (mm && !J.prev_mm) J.cam_mode = (J.cam_mode + 1) % 4;      // (near, medium, far, from the driver's seat)
        if (static const char* tv = std::getenv("VIBE_TEST_JEEP_VIEW")) J.cam_mode = std::clamp(std::atoi(tv), 0, 3);       // (tests: which view)
        J.prev_mm = mm;
    }
    float throttle = (in.fwd ? 1.0f : 0.0f) - (in.back ? 1.0f : 0.0f);
    float steer_in = (in.right ? 1.0f : 0.0f) - (in.left ? 1.0f : 0.0f);
    if (std::abs(in.move_y) > 0.05f) throttle = glm::clamp(in.move_y, -1.0f, 1.0f);
    if (std::abs(in.move_x) > 0.05f) steer_in = glm::clamp(in.move_x, -1.0f, 1.0f);
    float hand = in.jump ? 1.0f : 0.0f;
    if (static const char* test = std::getenv("VIBE_TEST_JEEP")) {
        std::sscanf(test, "%f,%f", &throttle, &steer_in); hand = 0.0f;
        // (steer 9: it follows the course by itself, in the right-hand lane, at up to <throttle> x 30 m/s)
        if (steer_in > 5.0f) {
            const glm::vec3 p(J.st.body[3]);
            const glm::vec3 f0 = -glm::vec3(J.st.body[2]);
            const float sp = glm::length(J.st.velocity);
            float d = 0.0f, h = 0.0f, s = 0.0f;
            steer_in = 0.0f;
            if (highway::nearest(p.x, p.z, 300.0f, d, h, s)) {
                glm::vec3 tp; glm::vec2 td;
                const glm::vec2 f2 = glm::normalize(glm::vec2(f0.x, f0.z));
                // (the way round it already faces)
                static float way = 0.0f;
                if (way == 0.0f && highway::point_along(s, tp, td)) way = glm::dot(td, f2) >= 0.0f ? 1.0f : -1.0f;
                if (highway::point_along(s + way * (12.0f + sp * 0.9f), tp, td)) {
                    td *= way;
                    const glm::vec2 lane = glm::vec2(tp.x, tp.z) + glm::vec2(-td.y, td.x) * 1.8f;
                    const glm::vec2 to = glm::normalize(lane - glm::vec2(p.x, p.z));
                    steer_in = glm::clamp((f2.x * to.y - f2.y * to.x) * 3.0f, -1.0f, 1.0f);
                }
            }
            const float top = throttle * 30.0f;
            throttle = sp < top ? 1.0f : (sp > top + 2.0f ? -0.3f : 0.0f);
        }
    }

    // VIBE_TEST_JEEP_SKID=<s>: after <s> seconds at the wheel, full lock and the handbrake (tests).
    if (static const char* sk = std::getenv("VIBE_TEST_JEEP_SKID")) {
        J.test_t += dt;
        if (J.test_t > float(std::atof(sk))) { hand = 1.0f; steer_in = 1.0f; throttle = 0.0f; }
    }
    // VIBE_TEST_JEEP_TURN=<s>: after <s> seconds at the wheel, full lock, the throttle held (tests: how hard it corners).
    if (static const char* tn = std::getenv("VIBE_TEST_JEEP_TURN")) {
        if (!std::getenv("VIBE_TEST_JEEP_SKID")) J.test_t += dt;
        if (J.test_t > float(std::atof(tn))) {
            steer_in = 1.0f; throttle = 0.6f;
            if (frame_number_ % 12 == 0)
                log::infof("[jeep] turn: %.0f km/h, %.2f g across, steer %.2f", double(glm::length(J.st.velocity) * 3.6f),
                           double(glm::length(glm::cross(J.st.angular_velocity, J.st.velocity)) / 9.81f), double(J.steer));
        }
    }
    // VIBE_TEST_JEEP_PERF=1: how the car goes, as numbers (tests; with VIBE_TEST_JEEP_AT on a long
    // straight). From rest: full throttle straight on until 100 km/h ("0-100 in"); then held at
    // 60 km/h for two seconds; then full lock with the throttle held for four seconds (the most
    // sideways g, the body's most roll); then the brakes to a stop (100-0 is not measured: from 60).
    if (static const bool perf = std::getenv("VIBE_TEST_JEEP_PERF") != nullptr; perf) {
        static int phase = 0; static float t = 0.0f, g_max = 0.0f, roll_max = 0.0f, g_lat = 0.0f, g_sum = 0.0f, g_n = 0.0f; static glm::vec3 v_was(0.0f);
        {   // (the sideways acceleration: the change of the velocity across the car, smoothed over a fifth of a second)
            const glm::vec3 a = dt > 0.0f ? (J.st.velocity - v_was) / dt : glm::vec3(0.0f);
            v_was = J.st.velocity;
            const glm::vec3 rgt = glm::normalize(glm::vec3(J.st.body[0]));
            g_lat += (std::abs(glm::dot(a, rgt)) / 9.81f - g_lat) * std::min(1.0f, dt / 0.2f);
        }
        const float sp = glm::length(J.st.velocity);
        t += dt; hand = 0.0f;
        {   // (along the road, in its right-hand lane, as the autopilot of VIBE_TEST_JEEP=<t>,9 steers: it stays on asphalt)
            steer_in = 0.0f;
            glm::vec2 f2(-J.st.body[2].x, -J.st.body[2].z);
            f2 = glm::length(f2) > 1e-3f ? glm::normalize(f2) : glm::vec2(0.0f, -1.0f);
            // (VIBE_TEST_JEEP_PERF=2: on open ground - straight on the way it faced at the start, and a steady
            //  full lock to the right in the third part instead of left and right)
            static const int pmode = std::atoi(std::getenv("VIBE_TEST_JEEP_PERF"));
            static glm::vec2 h0(0.0f);
            if (h0 == glm::vec2(0.0f)) h0 = f2;
            if (pmode == 2) steer_in = glm::clamp((f2.x * h0.y - f2.y * h0.x) * 4.0f, -1.0f, 1.0f);
            else {
            const glm::vec3 p(J.st.body[3]);
            float d = 0.0f, h = 0.0f, s = 0.0f;
            static float way = 0.0f;
            glm::vec3 tp; glm::vec2 td;
            if (highway::nearest(p.x, p.z, 300.0f, d, h, s)) {
                if (way == 0.0f && highway::point_along(s, tp, td)) way = glm::dot(td, f2) >= 0.0f ? 1.0f : -1.0f;
                if (highway::point_along(s + way * (12.0f + sp * 0.9f), tp, td)) {
                    td *= way;
                    const glm::vec2 to = glm::normalize(glm::vec2(tp.x, tp.z) + glm::vec2(-td.y, td.x) * 1.8f - glm::vec2(p.x, p.z));
                    steer_in = glm::clamp((f2.x * to.y - f2.y * to.x) * 3.0f, -1.0f, 1.0f);
                }
            }
            }
        }
        if (phase == 0 && std::atoi(std::getenv("VIBE_TEST_JEEP_PERF")) == 2 && sp * 3.6f >= 60.0f) { phase = 1; t = 3.0f; }      // (on open ground: only up to 60, and a second there - the room is the apron's)
        if (phase == 0) { throttle = 1.0f; if (sp * 3.6f >= 100.0f) { log::infof("[jeep] perf: %s 0-100 km/h in %.2f s", J.dir.c_str(), double(t)); phase = 1; t = 0.0f; } else if (t > 30.0f) { log::infof("[jeep] perf: %s 0-100 not reached in 30 s (%.0f km/h)", J.dir.c_str(), double(sp * 3.6f)); phase = 1; t = 0.0f; } }
        else if (phase == 1) { throttle = sp * 3.6f > 62.0f ? -0.6f : (sp * 3.6f < 58.0f ? 0.7f : 0.25f); if (t > 4.0f) { phase = 2; t = 0.0f; } }
        else if (phase == 2) {
            steer_in = (std::atoi(std::getenv("VIBE_TEST_JEEP_PERF")) == 2 || std::fmod(t, 2.4f) < 1.2f) ? 1.0f : -1.0f; throttle = 0.45f;      // (full lock one way, then the other, 1.2 s each: it stays on the road)
            if (t > 0.8f) {
                g_max = std::max(g_max, g_lat); if (t > 2.0f) { g_sum += g_lat; g_n += 1.0f; }
                roll_max = std::max(roll_max, std::abs(std::asin(glm::clamp(J.st.body[0].y, -1.0f, 1.0f))));
            }
            if (t > 4.8f) { log::infof("[jeep] perf: %s full lock %s from 60 km/h: up to %.2f g sideways (%.2f g on average over the last 2.8 s), body roll up to %.1f deg, %.0f km/h at the end", J.dir.c_str(), std::atoi(std::getenv("VIBE_TEST_JEEP_PERF")) == 2 ? "to the right" : "left and right", double(g_max), double(g_sum / std::max(g_n, 1.0f)), double(glm::degrees(roll_max)), double(sp * 3.6f)); phase = 3; t = 0.0f; }
        } else { throttle = sp > 0.5f ? -1.0f : 0.0f; }
    }
    // VIBE_TEST_JEEP_FLIP=<s>: after <s> seconds at the wheel the car is thrown over - a blow up
    // under its left side (tests: the chase view in a roll, VIBE_LOG_JEEP_CAM).
    if (static const char* fl = std::getenv("VIBE_TEST_JEEP_FLIP"); fl) {
        static float ft = 0.0f; static bool done = false;
        ft += dt;
        if (!done && ft > float(std::atof(fl))) {
            done = true;
            const glm::vec3 com(J.st.body * glm::vec4(0.0f, J.tune.com_height, 0.0f, 1.0f));
            physics_->vehicle_impulse(glm::vec3(0.0f, J.tune.mass * 6.5f, 0.0f), com - glm::normalize(glm::vec3(J.st.body[0])) * 0.9f);
            log::infof("[jeep] test: thrown over at %.0f km/h", double(glm::length(J.st.velocity) * 3.6f));
        }
    }
    const glm::vec3 fwd = -glm::vec3(J.st.body[2]);
    const float v_fwd

 = glm::dot(J.st.velocity, fwd), speed = glm::length(J.st.velocity);
    // A BURNOUT: the handbrake (space) and the throttle together from a standstill. The front
    // brakes hold the car, the rear wheels spin under full power - the engine screams, the rear
    // tyres smoke and squeal (jeep_marks, car_tyre_audio). Let go of the handbrake with the
    // throttle still down and it launches. Held until the handbrake is let go; at speed the
    // handbrake is the handbrake (a slide). VIBE_TEST_BURNOUT=<s0>,<s1>: held from s0 to s1
    // seconds at the wheel, the throttle down until 4 s after (tests).
    {
        static const char* tb = std::getenv("VIBE_TEST_BURNOUT");
        if (tb) {
            static float s0 = 1.0f, s1 = 4.0f, tt = 0.0f; static bool rd = false;
            if (!rd) { rd = true; std::sscanf(tb, "%f,%f", &s0, &s1); }
            tt += dt;
            hand = (tt > s0 && tt < s1) ? 1.0f : 0.0f; throttle = (tt > s0 && tt < s1 + 4.0f) ? 1.0f : 0.0f; steer_in = 0.0f;
            if (static float lt = 0.0f; tt - lt > 0.5f && tt < s1 + 5.0f) {
                lt = tt;
                log::infof("[jeep] burnout test %.1f s: %s, %.1f km/h, %.0f rpm, gear %d, rear wheels %.1f / %.1f rad/s, front %.1f / %.1f, squeal %.2f, puffs %zu",
                           double(tt), J.burnout ? "burnout" : "-", double(speed * 3.6f), double(J.st.rpm), J.st.gear, double(J.st.wheel_spin[2]), double(J.st.wheel_spin[3]),
                           double(J.st.wheel_spin[0]), double(J.st.wheel_spin[1]), double(J.squeal), J.puffs.size());
            }
        }
        const bool want = hand > 0.5f && throttle > 0.05f;
        if (!want) J.burnout = false;
        else if (!J.burnout && std::abs(v_fwd) < 3.0f) J.burnout = true;
    }
    float forward = throttle, brake = 0.0f;
    if (throttle < -0.05f && v_fwd > 1.2f) { brake = -throttle; forward = 0.0f; }             // braking, then reverse
    else if (throttle > 0.05f && v_fwd < -1.2f) { brake = throttle; forward = 0.0f; }
    // Steering: the wheel takes time to turn, and turns less the faster the jeep goes.
    // How far the wheels turn at full lock on the key: as far as the tyres can use at this
    // speed, no further. The lock for the tightest circle the grip allows (wheelbase x the
    // sideways acceleration the tyres give / speed squared) plus the slip angle at which a
    // tyre grips best. More than that only scrubs the front tyres past their peak - the car
    // turns LESS - or, on a tall car, trips it over. (A stick or a wheel would need none of
    // this; a key is all or nothing.)
    const float wheelbase = std::abs(J.tune.wheel_pos[2].z - J.tune.wheel_pos[0].z);
    // (2026-11-23: a held key asks for four fifths of what the tyres can give, not all of it
    //  and a little more - at speed every touch of the key put the front tyres at their
    //  limit and the car slid at once. Holding the key for over half a second goes on to the
    //  full grip: a long hold is a driver who means it. And the wheel turns more slowly the
    //  faster the car goes - half the rate at 90 km/h - and comes back to the middle twice as
    //  fast as it goes out, so letting go of the key straightens the car.)
    static float held = 0.0f;
    held = std::abs(steer_in) > 0.5f ? held + dt : 0.0f;
    const float ask = glm::mix(0.80f, 1.02f, glm::smoothstep(0.55f, 1.4f, held));
    const float a_grip = (9.81f * J.tune.grip + J.tune.downforce * speed * speed / std::max(J.tune.mass, 1.0f)) * ask;
    const float lock = std::atan(wheelbase * a_grip / std::max(speed * speed, 1.0f)) + glm::mix(0.035f, 0.085f, glm::smoothstep(0.55f, 1.4f, held));
    const float limit = glm::clamp(lock / std::max(J.tune.max_steer, 0.1f), 0.05f, 1.0f);
    const float want = steer_in * limit;
    // (2026-11-27: 3.4 rad/s at a standstill, was 2.6 - "sluggish"; VIBE_CAR_TUNE=old: as it was)
    const float kSteerRate = g_car_tune_old ? 2.6f : 3.4f;
    const float rate = kSteerRate / (1.0f + speed / 25.0f) * (std::abs(want) < std::abs(J.steer) ? 2.0f : 1.0f);
    J.steer += glm::clamp(want - J.steer, -dt * rate, dt * rate);
    // Traction control, as every such car has: when a driven wheel spins (its slip past where
    // a tyre grips best) the throttle is eased until it bites again. Without it full throttle
    // in a low gear spun the rear wheels, and a spinning tyre holds nothing sideways: the
    // car stepped out on a straight road. The handbrake turns it off (a slide is asked for).
    {
        float slip = 0.0f;
        for (int w = 0; w < 4; ++w)
            if (J.st.wheel_contact[w] && (w >= 2 || J.tune.front_drive > 0.05f)) slip = std::max(slip, J.st.wheel_slip[w]);
        const float cut = (hand > 0.5f || speed < 4.0f) ? 1.0f : glm::clamp(1.0f - (slip - 0.22f) * 2.2f, 0.40f, 1.0f);      // (from a standstill the wheels may spin: that is how it gets away)
        J.tc += (cut - J.tc) * std::min(1.0f, dt * (cut < J.tc ? 30.0f : 6.0f));
        if (forward > 0.0f) forward *= J.tc;
    }
    {   // The turbo boost: a push along the car while Sprint is held and the throttle is down.
        int down = 0;
        for (int w = 0; w < 4; ++w) down += J.st.wheel_contact[w] ? 1 : 0;
        static const bool kTestTurbo = std::getenv("VIBE_TEST_JEEP_TURBO") != nullptr;      // (tests: held all the time)
        const bool want_boost = (in.sprint || kTestTurbo) && throttle > 0.05f && !J.turbo_dry && J.turbo > 0.0f;
        J.boosting = want_boost;
        if (want_boost) {
            J.turbo = std::max(0.0f, J.turbo - dt / 3.2f);
            J.turbo_idle = 0.0f;
            if (J.turbo <= 0.0f) J.turbo_dry = true;
            if (down >= 2) {
                const glm::vec3 com(J.st.body * glm::vec4(0.0f, J.tune.com_height, 0.0f, 1.0f));
                physics_->vehicle_impulse(fwd * (J.tune.mass * 7.5f * dt), com);
            }
            forward = std::max(forward, throttle);      // (no traction control in the way)
        } else {
            J.turbo_idle += dt;
            if (J.turbo_idle > 0.8f) J.turbo = std::min(1.0f, J.turbo + dt / 9.0f);
            if (J.turbo >= 0.25f) J.turbo_dry = false;
        }
    }
    physics_->vehicle_burnout(J.burnout);
    if (J.burnout) {
        // (all the engine has to the wheels; the brake pedal holds the front ones - the rear's
        //  brake torque is taken away by vehicle_burnout - and the handbrake is not used)
        forward = std::clamp(throttle, 0.0f, 1.0f); brake = 1.0f;
        J.throttle_in = forward;
        physics_->vehicle_input(forward, J.steer, brake, 0.0f, dt, forward);
        J.brake_in = 1.0f;
    } else {
    J.throttle_in = std::abs(forward);       // (what the engine is asked for: its sound's load)
    physics_->vehicle_input(forward, J.steer, brake, hand, dt, forward > 0.0f ? std::max(forward, std::clamp(throttle, 0.0f, 1.0f)) : 0.0f);
    J.brake_in = std::max(brake, hand);
    }
    // The driver's hands are full: no walking, shooting or jumping.
    in.fwd = in.back = in.left = in.right = in.jump = in.sprint = in.crawl = false;
    in.move_x = in.move_y = 0.0f;
    J.look_held = in.alt_fire_held;          // (the right mouse button: look round - jeep_after_sim)
    in.fire = in.fire_held = in.alt_fire_held = false;
}

// Every part of a car where the car's state says it is.
void VulkanEngine::jeep_place(JeepState& J, const PhysicsWorld::VehicleState& prev, float dt) {
    const glm::mat4& B = J.st.body;
    // -- the parts --
    auto place = [&](size_t i, const glm::mat4& m, const glm::mat4& m_prev) {
        CastleMeshGpu& g = jeep_draw_[i];
        g.xf_prev = m_prev; g.xf = m;
        // (lamps: lit when they are on - emissive.a = 2: by day as by night)
        const uint8_t kd = J.kind[i - J.draw_first];
        if (kd == 1) g.emissive = J.lights_on ? glm::vec4(2.8f, 2.7f, 2.3f, 2.0f) : glm::vec4(0.0f);
        if (kd == 2) {
            const float s = J.braking() ? 3.2f : (J.lights_on ? 0.75f : 0.0f);
            g.emissive = glm::vec4(s, s * 0.04f, s * 0.03f, s > 0.0f ? 2.0f : 0.0f);
        }
        glm::vec3 lo(1e9f), hi(-1e9f);
        const auto& b = J.local[i - J.draw_first];
        for (int k = 0; k < 8; ++k) {
            const glm::vec3 c((k & 1) ? b.second.x : b.first.x, (k & 2) ? b.second.y : b.first.y, (k & 4) ? b.second.z : b.first.z);
            const glm::vec3 w(m * glm::vec4(c, 1.0f));
            lo = glm::min(lo, w); hi = glm::max(hi, w);
        }
        g.bmin = lo; g.bmax = hi;
    };
    size_t at = J.draw_first;
    for (size_t i = 0; i < J.n_body; ++i) place(at++, B, J.body_prev);
    const glm::mat4 flip = glm::rotate(glm::mat4(1.0f), 3.14159265f, glm::vec3(0.0f, 1.0f, 0.0f));
    for (int w = 0; w < 4; ++w) {
        const bool left = (w % 2) == 0;                      // (the mesh's outer face looks to +x)
        const glm::mat4 m = left ? J.st.wheel[w] * flip : J.st.wheel[w];
        const glm::mat4 mp = left ? prev.wheel[w] * flip : prev.wheel[w];
        for (size_t i = 0; i < J.n_wheel; ++i) place(at++, m, mp);
    }
    J.steer_wheel += (J.st.steer - J.steer_wheel) * std::min(1.0f, dt * 12.0f);
    auto steer_m = [&](const glm::mat4& body, float a) {
        return body * glm::translate(glm::mat4(1.0f), J.steering_hub) *
               glm::rotate(glm::mat4(1.0f), glm::radians(-J.steering_tilt), glm::vec3(1.0f, 0.0f, 0.0f)) *
               glm::rotate(glm::mat4(1.0f), a * 9.0f, glm::vec3(0.0f, 0.0f, 1.0f));
    };
    for (size_t i = 0; i < J.n_steer; ++i) place(at++, steer_m(B, J.steer_wheel), steer_m(J.body_prev, J.steer_wheel));
    // The doors: the driver's stands open for a moment when the player gets in or out; either
    // opens for an NPC getting in or out (door_want). Open and shut in under half a second.
    if (J.driving != J.was_driving) { J.was_driving = J.driving; J.door_hold = 1.0f; }
    J.door_hold = std::max(0.0f, J.door_hold - dt);
    for (int sd = 0; sd < 2; ++sd) {
        const float want = std::max(J.door_want[sd], sd == 0 && J.door_hold > 0.0f ? 1.0f : 0.0f);
        J.door[sd] += std::clamp(want - J.door[sd], -dt / 0.45f, dt / 0.40f);
        const float k = J.door[sd] * J.door[sd] * (3.0f - 2.0f * J.door[sd]);
        const float ang = (sd == 0 ? -1.0f : 1.0f) * 1.12f * k;                    // 64 degrees, outward
        auto door_m = [&](const glm::mat4& body) {
            return body * glm::translate(glm::mat4(1.0f), J.door_hinge[sd]) * glm::rotate(glm::mat4(1.0f), ang, glm::vec3(0.0f, 1.0f, 0.0f)) *
                   glm::translate(glm::mat4(1.0f), -J.door_hinge[sd]);
        };
        for (size_t i = 0; i < J.n_door[sd]; ++i) place(at++, door_m(B), door_m(J.body_prev));
    }

}

// After the physics ticks: where everything is now - the parts, the driver, the camera.
void VulkanEngine::jeep_after_sim(float dt) {
    jeep_ride_place();
    cars_snow_tick(dt);
    if (!jeep_ || !jeep_->spawned || !physics_) return;
    JeepState& J = *jeep_;
    J.body_prev = J.st.body;
    PhysicsWorld::VehicleState prev = J.st;
    // The physics moves in steps of 1 / 120 s and a frame holds one of them, or two, or none:
    // drawn where the last step left it, the car moved by a different amount every frame
    // (at 100 km/h a step is 23 cm) - it shook against the road and the camera. So it is
    // drawn between the last two steps, as far along as the time not yet simulated
    // (the player is drawn the same way: world_render.cpp, render_pos).
    {
        PhysicsWorld::VehicleState cur, old;
        physics_->vehicle_state(cur);
        J.st = cur;
        static const bool kRaw = std::getenv("VIBE_TEST_JEEP_RAW") != nullptr;      // (tests: as it was, the last step's place)
        if (!kRaw && physics_->vehicle_state_prev(old)) {
            const float a = glm::clamp(physics_accumulator_ * 120.0f, 0.0f, 1.0f);
            J.st.body = mix_pose(old.body, cur.body, a);
            for (int w = 0; w < 4; ++w) J.st.wheel[w] = mix_pose(old.wheel[w], cur.wheel[w], a);
        }
    }
    const glm::mat4& B = J.st.body;
    jeep_place(J, prev, dt);
    jeep_marks(J, dt);

    // -- on its roof or its side and at rest: it is put back on its wheels --
    const float up_y = B[1].y;
    if (up_y < 0.35f && glm::length(J.st.velocity) < 0.8f) {
        J.upside_t += dt;
        if (J.upside_t > 2.5f) {
            const glm::vec3 p(B[3]);
            const glm::vec3 f = -glm::vec3(B[2]);
            physics_->vehicle_teleport(p + glm::vec3(0.0f, 1.0f, 0.0f), std::atan2(-f.x, -f.z));
            J.upside_t = 0.0f;
        }
    } else J.upside_t = 0.0f;

    // (how evenly it moves in the picture: the most a frame's step differed from speed x time)
    static float uneven = 0.0f;
    if (dt > 0.0f) uneven = std::max(uneven, std::abs(glm::length(glm::vec3(B[3]) - glm::vec3(J.body_prev[3])) - glm::length(J.st.velocity) * dt));
    if (static const bool lg2 = std::getenv("VIBE_LOG_JEEP") != nullptr; lg2 && frame_number_ % 60 == 0) {
        log::infof("[jeep] uneven by %.3f m at most over the last 60 frames (frame %.1f ms)", double(uneven), double(dt) * 1000.0);
        uneven = 0.0f;
    }
    // (VIBE_LOG_JEEP: every change of gear - from which engine speed, at what road speed, how hard the pedal was down)
    if (static const bool lg3 = std::getenv("VIBE_LOG_JEEP") != nullptr; lg3) {
        static int gear_was = 0; static float rpm_was = 0.0f, rpm_top = 0.0f;
        rpm_top = std::max(rpm_top, J.st.rpm);
        if (J.st.gear != gear_was) {
            log::infof("[jeep] %s: gear %d -> %d from %.0f rpm (the most in that gear %.0f of %.0f) at %.0f km/h, pedal %.2f", J.dir.c_str(), gear_was, J.st.gear,
                       double(rpm_was), double(rpm_top), double(J.tune.engine_max_rpm), double(glm::length(J.st.velocity) * 3.6f), double(J.throttle_in));
            gear_was = J.st.gear; rpm_top = 0.0f;
        }
        rpm_was = J.st.rpm;
    }
    if (static const bool lg = std::getenv("VIBE_LOG_JEEP") != nullptr; lg && frame_number_ % 60 == 0)
        log::infof("[jeep] pos (%.1f, %.2f, %.1f) speed %.1f km/h gear %d rpm %.0f contact %d%d%d%d up %.2f driving %d | springs %.2f %.2f %.2f %.2f ground %.2f",
                   double(B[3].x), double(B[3].y), double(B[3].z), double(glm::length(J.st.velocity) * 3.6f), J.st.gear, double(J.st.rpm),
                   int(J.st.wheel_contact[0]), int(J.st.wheel_contact[1]), int(J.st.wheel_contact[2]), int(J.st.wheel_contact[3]), double(B[1].y), int(J.driving),
                   double(J.st.suspension[0]), double(J.st.suspension[1]), double(J.st.suspension[2]), double(J.st.suspension[3]),
                   double(world_ground_height(B[3].x, B[3].z, rt_.water_level, 9)));
    if (!J.driving) { J.fov_extra = 0.0f; J.boost_k = 0.0f; return; }
    // -- the camera: behind and above, looking where the player looks --
    // The player's yaw / pitch are the camera's (the mouse and the stick turn them as on
    // foot). Left alone for a moment while moving, the view swings back behind the jeep.
    const glm::vec3 f = -glm::vec3(B[2]);
    const float speed = glm::length(J.st.velocity);
    // -- the speed in the view: a few degrees wider with speed, more with the boost, eased in
    //    and out; and a small shake while the boost pushes (jeep_fov_extra, world_render.cpp) --
    {
        J.boost_k += ((J.boosting ? 1.0f : 0.0f) - J.boost_k) * (1.0f - std::exp(-dt * (J.boosting ? 4.0f : 1.6f)));
        static const float kFovSpeed = std::getenv("VIBE_CAR_FOV_SPEED") ? float(std::atof(std::getenv("VIBE_CAR_FOV_SPEED"))) : 5.0f;
        static const float kFovBoost = std::getenv("VIBE_CAR_FOV_BOOST") ? float(std::atof(std::getenv("VIBE_CAR_FOV_BOOST"))) : 6.0f;
        const float want = kFovSpeed * glm::smoothstep(15.0f, 55.0f, speed) + kFovBoost * J.boost_k;
        J.fov_extra += (want - J.fov_extra) * (1.0f - std::exp(-dt * (want > J.fov_extra ? 3.0f : 1.5f)));
    }
    static const bool kCamOld = std::getenv("VIBE_CAR_CAM_OLD") != nullptr;      // (tests: the chase view as before 2026-11-27 - no easing in a roll, the origin as pivot)
    // HOW MUCH IT TUMBLES (0 on its wheels .. 1 rolling or in the air): on its side, wheels off
    // the ground, turning fast about a level axis. The chase view goes soft while it does - it
    // keeps the horizon and its own heading and lets the car turn over in front of it, instead
    // of following every turn of the body (2026-11-27: "when the car is rolling the camera moves
    // erratically"). On its wheels again it closes in as before.
    {
        int down = 0;
        for (int w = 0; w < 4; ++w) down += J.st.wheel_contact[w] ? 1 : 0;
        const glm::vec3 av = J.st.angular_velocity;
        const float level_spin = glm::length(glm::vec2(av.x, av.z));
        const float want = std::max({ 1.0f - glm::smoothstep(0.55f, 0.85f, B[1].y), down == 0 ? glm::smoothstep(0.2f, 0.6f, std::abs(J.st.velocity.y) + level_spin * 0.3f) : 0.0f,
                                       glm::smoothstep(1.2f, 3.0f, level_spin) });
        J.cam_tumble += (want - J.cam_tumble) * (1.0f - std::exp(-dt * (want > J.cam_tumble ? 10.0f : 1.6f)));
        if (kCamOld) J.cam_tumble = 0.0f;
    }
    // The way it heads on the ground plan: its nose flattened - but a nose that points at the sky
    // or the ground says nothing of that, so then the way it moves, and with neither the last one.
    {
        glm::vec2 h(f.x, f.z);
        const float hl = glm::length(h);
        glm::vec2 want = J.cam_head;
        const glm::vec2 vel(J.st.velocity.x, J.st.velocity.z);
        const float w_nose = glm::smoothstep(0.25f, 0.6f, hl) * (1.0f - J.cam_tumble), w_vel = glm::smoothstep(2.0f, 6.0f, glm::length(vel)) * J.cam_tumble;
        if (w_nose + w_vel > 1e-3f) {
            glm::vec2 m = (hl > 1e-3f ? h / hl : glm::vec2(0.0f)) * w_nose + (glm::length(vel) > 1e-3f ? glm::normalize(vel) : glm::vec2(0.0f)) * w_vel;
            if (glm::length(m) > 1e-3f) want = glm::normalize(m);
        }
        if (!J.cam_init) J.cam_head = want;
        J.cam_head = glm::normalize(J.cam_head + (want - J.cam_head) * std::min(1.0f, dt * glm::mix(12.0f, 2.0f, J.cam_tumble)) + glm::vec2(1e-6f, 0.0f));
    }
    const float car_yaw = kCamOld ? std::atan2(f.x, -f.z) : std::atan2(J.cam_head.x, -J.cam_head.y);
    // As in other driving games: the view stays behind the jeep by itself and turns with it,
    // a little late. Holding the right mouse button the mouse looks round the jeep; let go
    // and the view comes back behind it.
    static float last_yaw = 0.0f, last_pitch = 0.0f;
    const float kPitch = -0.20f;
    if (!J.cam_init) { J.cam_yaw = car_yaw; J.orbit_yaw = J.orbit_pitch = 0.0f; J.cockpit_yaw = J.cockpit_pitch = 0.0f; last_yaw = player_.yaw; last_pitch = player_.pitch; }
    const float d_yaw = std::remainder(player_.yaw - last_yaw, 6.2831853f), d_pitch = player_.pitch - last_pitch;
    // FROM THE DRIVER'S SEAT (the fourth view of the middle mouse button): the eye where the
    // driver's is, looking along the car - with its pitch, not its roll (the horizon stays level).
    // The right mouse button held: the head turns where the mouse says, as far as a neck turns
    // (110 degrees each way, 60 up, 45 down); let go and it comes back to the road.
    // VIBE_COCKPIT_EYE=<x>,<y>,<z>: where the eye is from the driver's hips (the car's frame).
    if (J.cam_mode == 3) {
        if (J.look_held) {
            J.cockpit_yaw = glm::clamp(J.cockpit_yaw + d_yaw, -1.92f, 1.92f);
            J.cockpit_pitch = glm::clamp(J.cockpit_pitch + d_pitch, -0.78f, 1.05f);
        } else {
            J.cockpit_yaw -= J.cockpit_yaw * std::min(1.0f, dt * 5.0f);
            J.cockpit_pitch -= J.cockpit_pitch * std::min(1.0f, dt * 5.0f);
        }
        static const glm::vec3 kEye = [] { glm::vec3 e(0.0f, 0.70f, -0.10f); if (const char* s = std::getenv("VIBE_COCKPIT_EYE")) std::sscanf(s, "%f,%f,%f", &e.x, &e.y, &e.z); return e; }();
        const glm::vec3 eye(B * glm::vec4(J.hip + kEye, 1.0f));
        const float car_pitch = std::asin(glm::clamp(f.y / std::max(glm::length(f), 1e-3f), -1.0f, 1.0f));
        const float yaw_now = std::atan2(f.x, -f.z);
        player_.yaw = yaw_now + J.cockpit_yaw;
        player_.pitch = glm::clamp(car_pitch * 0.85f + J.cockpit_pitch - 0.04f, -1.45f, 1.45f);
        last_yaw = player_.yaw; last_pitch = player_.pitch;
        J.cam_yaw = car_yaw; J.orbit_yaw = J.orbit_pitch = 0.0f;
        J.cam_pos = eye; J.cam_init = true;
        player_.position = eye - (player_.eye_position() - player_.position);
        player_.velocity = J.st.velocity;
        player_.on_ground = true;
        prev_player_position_ = player_.position;
        return;
    }
    if (J.look_held) {
        J.orbit_yaw = std::remainder(J.orbit_yaw + d_yaw, 6.2831853f);
        J.orbit_pitch = glm::clamp(J.orbit_pitch + d_pitch, -0.9f, 0.55f);
    } else {
        J.orbit_yaw -= J.orbit_yaw * std::min(1.0f, dt * 7.0f);
        J.orbit_pitch -= J.orbit_pitch * std::min(1.0f, dt * 7.0f);
    }
    // (it swings round after the car more slowly while the car tumbles)
    J.cam_yaw += std::remainder(car_yaw - J.cam_yaw, 6.2831853f) * std::min(1.0f, dt * glm::mix(4.5f, 1.2f, J.cam_tumble));
    // (VIBE_TEST_JEEP_ORBIT=<yaw>[,<pitch>][,<distance>]: the view held round the car - tests)
    if (static const char* ob = std::getenv("VIBE_TEST_JEEP_ORBIT")) { float d = J.cam_dist; std::sscanf(ob, "%f,%f,%f", &J.orbit_yaw, &J.orbit_pitch, &d); J.cam_dist = d; }
    player_.yaw = J.cam_yaw + J.orbit_yaw;
    player_.pitch = kPitch + J.orbit_pitch;
    last_yaw = player_.yaw; last_pitch = player_.pitch;
    const glm::vec3 look(std::sin(player_.yaw) * std::cos(player_.pitch), std::sin(player_.pitch), -std::cos(player_.yaw) * std::cos(player_.pitch));
    // (2026-11-23: nearer than the first 4.3 / 6.4 / 10.5; 2.7 / 3.3 / 4.0 was too near)
    static const float kDist[3] = { 3.6f, 4.6f, 5.8f }, kHigh[3] = { 1.30f, 1.45f, 1.65f };
    const int vm = std::clamp(J.cam_mode, 0, 2);
    if (!std::getenv("VIBE_TEST_JEEP_ORBIT")) J.cam_dist += (kDist[vm] - J.cam_dist) * std::min(1.0f, dt * 6.0f);
    J.cam_h += (kHigh[vm] - J.cam_h) * std::min(1.0f, dt * 6.0f);
    const float dist = J.cam_dist + std::min(speed * 0.03f, 1.2f) + 0.9f * J.cam_tumble;      // (it drops back a little at speed, and while it tumbles)
    // (the pivot is the car's centre of mass, not its origin on the ground line: rolling over,
    //  the origin swings round the centre by a metre and the view swung with it)
    const glm::vec3 com(B * glm::vec4(0.0f, J.tune.com_height, 0.5f * (J.wheel_pos[0].z + J.wheel_pos[2].z), 1.0f));
    const glm::vec3 pivot = kCamOld ? glm::vec3(B[3]) + glm::vec3(0.0f, J.cam_h, 0.0f) : com + glm::vec3(0.0f, J.cam_h - J.tune.com_height, 0.0f);
    glm::vec3 want = pivot - look * dist;
    // (not under the ground, and not through a hill between it and the car: the ground along
    //  the line from the car to the view is looked at, and the view goes up over it)
    {
        const float wl = rt_.water_level;
        for (int k = 1; k <= 4; ++k) {
            const float s = float(k) / 4.0f;
            const glm::vec3 q = glm::mix(pivot, want, s);
            const float g = std::max(world_ground_height(q.x, q.z, wl, 7), wl) + 0.6f;
            if (q.y < g) want.y += (g - q.y) / s;
        }
    }
    if (!J.cam_init) { J.cam_pos = want; J.cam_init = true; }
    // (the position follows stiffly: a lagging camera on a bumpy road makes one seasick.
    //  It first goes along with the car, then closes what is left: following alone it hung
    //  back by speed x a time that depends on the frame's length, so with uneven frames
    //  the car moved back and forth in the picture. While it tumbles it follows softly - the
    //  centre's own speed, and a slower closing.)
    {
        J.cam_pos += J.st.velocity * dt;          // (the body's velocity is its centre of mass's)
        J.cam_pos += (want - J.cam_pos) * (1.0f - std::exp(-dt * glm::mix(14.0f, 3.5f, J.cam_tumble)));
        const float g = std::max(world_ground_height(J.cam_pos.x, J.cam_pos.z, rt_.water_level, 7), rt_.water_level) + 0.45f;
        if (J.cam_pos.y < g) J.cam_pos.y = g;
    }
    // (VIBE_LOG_JEEP_CAM=1: how calmly the chase view moves - every 30 frames the most it turned in
    //  one frame and the most its place jumped in one frame beyond what the car's own motion explains)
    if (static const bool lc = std::getenv("VIBE_LOG_JEEP_CAM") != nullptr; lc) {
        static glm::vec3 last_p(0.0f), last_look(0.0f, 0.0f, -1.0f); static float turn_max = 0.0f, jump_max = 0.0f; static bool has = false;
        if (has) {
            turn_max = std::max(turn_max, glm::degrees(std::acos(glm::clamp(glm::dot(look, last_look), -1.0f, 1.0f))));
            jump_max = std::max(jump_max, glm::length((J.cam_pos - last_p) - J.st.velocity * dt));
        }
        last_p = J.cam_pos; last_look = look; has = true;
        if (frame_number_ % 30 == 0) {
            log::infof("[jeep] cam: tumble %.2f, up %.2f, most turn in a frame %.2f deg, most jump %.3f m, %.1f m/s", double(J.cam_tumble), double(B[1].y), double(turn_max), double(jump_max), double(speed));
            turn_max = jump_max = 0.0f;
        }
    }
    // (the boost: a small shake, a few centimetres, of the view - not of the aim)
    glm::vec3 shake(0.0f);
    if (J.boost_k > 0.01f) {
        const float t = float(frame_clock_s_);
        shake = glm::vec3(std::sin(t * 37.0f) + 0.6f * std::sin(t * 71.0f + 1.3f), std::sin(t * 43.0f + 0.7f) + 0.5f * std::sin(t * 83.0f), 0.0f) * (0.012f * J.boost_k * glm::smoothstep(8.0f, 30.0f, speed));
    }
    player_.position = J.cam_pos + shake - (player_.eye_position() - player_.position);
    player_.velocity = J.st.velocity;
    player_.on_ground = true;
    prev_player_position_ = player_.position;
}

void VulkanEngine::jeep_hud() {
    if (!jeep_ || !jeep_->spawned || state_ != State::Playing) return;
    JeepState& J = *jeep_;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImFont* fs = ImGui::GetFont();
    const ImVec2 ds = ImGui::GetIO().DisplaySize;
    const float s = ds.y / 1080.0f;
    if (J.driving) {
        // The speedometer: a dial with a needle, the speed in figures, the gear, the engine's
        // revs as a thin inner arc, and round the outside the turbo's tank.
        char buf[64];
        const float kmh = glm::length(J.st.velocity) * 3.6f;
        const ImVec2 c(ds.x - 175.0f * s, ds.y - 310.0f * s);       // (above the weapon panel)
        const float R = 118.0f * s;
        const float a0 = 0.75f * 3.14159265f, a1 = 2.25f * 3.14159265f;      // (270 degrees, open at the bottom)
        const float vmax = J.tune.engine_max_rpm > 7000.0f ? 320.0f : 240.0f;
        auto arc = [&](float r, float f0, float f1, ImU32 col, float w) {
            if (f1 <= f0) return;
            dl->PathArcTo(c, r, a0 + (a1 - a0) * f0, a0 + (a1 - a0) * f1, 48);
            dl->PathStroke(col, 0, w);
        };
        dl->AddCircleFilled(c, R + 22.0f * s, IM_COL32(8, 11, 15, 170), 64);
        arc(R, 0.0f, 1.0f, IM_COL32(255, 255, 255, 60), 3.0f * s);
        for (int v = 0; v <= int(vmax); v += 20) {
            const float a = a0 + (a1 - a0) * float(v) / vmax;
            const ImVec2 d(std::cos(a), std::sin(a));
            const bool big = v % 40 == 0;
            dl->AddLine(ImVec2(c.x + d.x * (R - (big ? 13.0f : 7.0f) * s), c.y + d.y * (R - (big ? 13.0f : 7.0f) * s)),
                        ImVec2(c.x + d.x * R, c.y + d.y * R), IM_COL32(255, 255, 255, big ? 230 : 130), (big ? 2.5f : 1.5f) * s);
            if (big) {
                std::snprintf(buf, sizeof buf, "%d", v);
                const ImVec2 ts = fs->CalcTextSizeA(15.0f * s, 1e9f, 0.0f, buf);
                dl->AddText(fs, 15.0f * s, ImVec2(c.x + d.x * (R - 30.0f * s) - ts.x * 0.5f, c.y + d.y * (R - 30.0f * s) - ts.y * 0.5f), IM_COL32(255, 255, 255, 190), buf);
            }
        }
        // revs (inner arc, red near the limit)
        const float rv = glm::clamp(J.st.rpm / std::max(J.tune.engine_max_rpm, 1.0f), 0.0f, 1.0f);
        arc(R - 48.0f * s, 0.0f, 1.0f, IM_COL32(255, 255, 255, 30), 4.0f * s);
        arc(R - 48.0f * s, 0.0f, rv, rv > 0.9f ? IM_COL32(255, 80, 60, 230) : IM_COL32(255, 190, 90, 210), 4.0f * s);
        // the turbo's tank (outer arc): blue, orange while it is in use, grey while it has run dry
        arc(R + 13.0f * s, 0.0f, 1.0f, IM_COL32(255, 255, 255, 35), 7.0f * s);
        arc(R + 13.0f * s, 0.0f, J.turbo, J.turbo_dry ? IM_COL32(150, 150, 150, 200) : (J.boosting ? IM_COL32(255, 170, 40, 255) : IM_COL32(70, 170, 255, 235)), 7.0f * s);
        // the needle
        {
            const float a = a0 + (a1 - a0) * glm::clamp(kmh / vmax, 0.0f, 1.0f);
            const ImVec2 d(std::cos(a), std::sin(a));
            dl->AddLine(ImVec2(c.x - d.x * 14.0f * s, c.y - d.y * 14.0f * s), ImVec2(c.x + d.x * (R - 16.0f * s), c.y + d.y * (R - 16.0f * s)), IM_COL32(255, 70, 50, 255), 3.5f * s);
            dl->AddCircleFilled(c, 7.0f * s, IM_COL32(230, 230, 230, 255), 20);
        }
        std::snprintf(buf, sizeof buf, "%.0f", double(kmh));
        ImVec2 ts = fs->CalcTextSizeA(40.0f * s, 1e9f, 0.0f, buf);
        dl->AddText(fs, 40.0f * s, ImVec2(c.x - ts.x * 0.5f, c.y + 26.0f * s), IM_COL32(255, 255, 255, 245), buf);
        ts = fs->CalcTextSizeA(15.0f * s, 1e9f, 0.0f, "km/h");
        dl->AddText(fs, 15.0f * s, ImVec2(c.x - ts.x * 0.5f, c.y + 66.0f * s), IM_COL32(255, 255, 255, 150), "km/h");
        const char* gears[] = { "R", "N", "1", "2", "3", "4", "5", "6" };
        const char* g = gears[std::clamp(J.st.gear + 1, 0, 7)];
        ts = fs->CalcTextSizeA(26.0f * s, 1e9f, 0.0f, g);
        dl->AddText(fs, 26.0f * s, ImVec2(c.x - ts.x * 0.5f, c.y - 62.0f * s), IM_COL32(255, 220, 140, 235), g);
        const char* tl = J.turbo_dry ? "TURBO  recharging" : "TURBO";
        ts = fs->CalcTextSizeA(14.0f * s, 1e9f, 0.0f, tl);
        dl->AddText(fs, 14.0f * s, ImVec2(c.x - ts.x * 0.5f, c.y + R + 26.0f * s), J.boosting ? IM_COL32(255, 180, 60, 255) : IM_COL32(120, 190, 255, 220), tl);
        const std::string k = bind_label(Action::Use) + " out   Shift turbo   L lights   MMB view";
        ts = fs->CalcTextSizeA(16.0f * s, 1e9f, 0.0f, k.c_str());
        dl->AddText(fs, 15.0f * s, ImVec2(c.x - ts.x * 0.47f, c.y + R + 46.0f * s), IM_COL32(255, 255, 255, 140), k.c_str());
        return;
    }
    if (!globe::current().home) return;
    bool near_one = false;
    for (const JeepState* c : jeep_all_)
        if (c->posed && glm::distance(player_.eye_position(), glm::vec3(c->st.body[3]) + glm::vec3(0.0f, 1.0f, 0.0f)) < 3.6f) near_one = true;
    if (!near_one) return;
    const std::string t = bind_label(Action::Use) + "   DRIVE";
    const ImVec2 sz = fs->CalcTextSizeA(30.0f * s, 1e9f, 0.0f, t.c_str());
    const ImVec2 p(ds.x * 0.5f - sz.x * 0.5f, ds.y * 0.62f);
    dl->AddRectFilled(ImVec2(p.x - 14.0f * s, p.y - 6.0f * s), ImVec2(p.x + sz.x + 14.0f * s, p.y + sz.y + 6.0f * s), IM_COL32(10, 14, 18, 190), 6.0f * s);
    dl->AddText(fs, 30.0f * s, p, IM_COL32(255, 255, 255, 240), t.c_str());
}

// For the ray-tracing structure's signature: the jeep's place, to a quarter of a metre.
uint64_t VulkanEngine::jeep_signature() const {
    uint64_t h = 1469598103934665603ull;
    for (const JeepState* c : jeep_all_) {
        if (!c->posed) continue;
        const glm::mat4& B = c->st.body;
        for (float v : { B[3].x * 16.0f, B[3].y * 16.0f, B[3].z * 16.0f, B[0].x * 64.0f, B[2].x * 64.0f, B[1].y * 64.0f, c->st.steer * 32.0f })
            h = (h ^ uint64_t(int64_t(hash_f(v)))) * 1099511628211ull;
    }
    for (const NpcCar& c : npc_cars_) {
        if (c.agent < 0) continue;
        const glm::mat4& B = c.car->st.body;
        for (float v : { B[3].x * 16.0f, B[3].y * 16.0f, B[3].z * 16.0f, B[0].x * 64.0f, B[2].x * 64.0f, float(c.agent) })
            h = (h ^ uint64_t(int64_t(hash_f(v)))) * 1099511628211ull;
    }
    return h;
}

// Tyre marks and smoke: where a tyre slides over the ground - sideways in a drift, or spinning
// or locked along it - it lays rubber on a hard surface and throws up smoke (dust off-road).
void VulkanEngine::jeep_marks(JeepState& J, float dt) {
    if (dt <= 0.0f) return;
    const glm::mat4& B = J.st.body;
    const glm::vec3 up = glm::normalize(glm::vec3(B[1]));
    const glm::vec3 com(B * glm::vec4(0.0f, J.tune.com_height, 0.0f, 1.0f));
    J.squeal = 0.0f;
    // (a burnout: the rear tyres spin on the spot - thick smoke that hangs, three times as much, bigger puffs)
    const bool burn = J.burnout && J.driving;
    for (int w = 0; w < 4; ++w) {
        const glm::vec3 c(J.st.wheel[w][3]);
        const glm::vec3 axle = glm::normalize(glm::vec3(J.st.wheel[w][0]));
        const glm::vec3 fw = glm::normalize(glm::cross(up, axle));
        const glm::vec3 v = J.st.velocity + glm::cross(J.st.angular_velocity, c - com);
        const float v_lat = std::abs(glm::dot(v, axle)), v_long = std::abs(glm::dot(v, fw));
        const float slip = std::abs(std::abs(J.st.wheel_spin[w]) * J.wheel_radius - v_long);
        float k = 0.0f;
        // (sideways: by the tyre's slip ANGLE - past where it grips best, 8 degrees - and not
        //  by sideways speed alone: at 100 km/h a tyre cornering well inside its grip already
        //  moves 2 - 3 m/s sideways, and every turn of the wheel at speed drew marks and smoke)
        const float ang = std::atan2(v_lat, std::max(v_long, 2.5f));
        if (J.st.wheel_contact[w]) k = std::max(glm::smoothstep(0.19f, 0.34f, ang) * glm::smoothstep(1.2f, 2.5f, v_lat), glm::smoothstep(2.2f, 7.0f, slip));
        const glm::vec3 g = c - up * (J.wheel_radius - 0.035f);
        const bool hard = highway::distance(g.x, g.z) < 5.7f || g.y > world_ground_height(g.x, g.z, rt_.water_level, 9) + 0.15f;
        if (hard) J.squeal = std::max(J.squeal, glm::smoothstep(0.12f, 0.7f, k));
        auto& M = J.marks[w];
        if (k > 0.08f && hard) {
            if (M.empty() || M.back().a <= 0.0f || glm::distance(M.back().p, g) > 0.30f)
                M.push_back({ g, axle * (0.42f * J.wheel_width), std::min(1.0f, k * 1.25f) });
        } else if (!M.empty() && M.back().a > 0.0f) M.push_back({ g, axle * (0.42f * J.wheel_width), 0.0f });
        if (M.size() > 700) M.erase(M.begin(), M.begin() + 150);
        if (k > 0.22f) {
            J.puff_acc += dt * k * (hard ? 22.0f : 16.0f) * (burn && w >= 2 ? 3.0f : 1.0f);
            while (J.puff_acc >= 1.0f) {
                J.puff_acc -= 1.0f;
                if (J.puffs.size() >= (burn ? 480u : 220u)) break;
                JeepState::Puff p;
                p.p = g + up * 0.12f + glm::vec3(rnd01() - 0.5f, 0.0f, rnd01() - 0.5f) * 0.25f;
                p.v = v * 0.18f + glm::vec3(rnd01() - 0.5f, 0.9f + rnd01() * 0.8f, rnd01() - 0.5f) * 0.9f;
                p.v += axle * ((J.wheel_pos[w].x < 0.0f ? -1.0f : 1.0f) * (glm::dot(axle, glm::vec3(B[0])) < 0.0f ? -1.0f : 1.0f) * (0.7f + 0.8f * rnd01()));      // (thrown out from under the car, to its side)
                p.age = 0.0f; p.life = (hard ? 1.9f : 1.2f) * (0.7f + 0.6f * rnd01());
                p.size = 0.28f + 0.2f * rnd01(); p.seed = rnd01() * 40.0f;
                p.bright = hard ? 1.0f : 0.62f;
                if (burn && w >= 2) { p.life *= 1.7f; p.size += 0.15f; p.v.y += 0.35f; }
                J.puffs.push_back(p);
            }
        }
    }
    // SMOKE STAYS OUTSIDE THE CAR. A puff grows to a metre and more and drifts; the body is in
    // its way. One that would be inside the body's box (between the wheels' outer faces, nose to
    // tail, floor to roof) is put out at the nearer side and loses its speed towards the car.
    const glm::mat4 Binv = glm::inverse(B);
    float hx, z0, z1; J.body_box(hx, z0, z1);
    for (auto& p : J.puffs) {
        const glm::vec3 l(Binv * glm::vec4(p.p, 1.0f));
        if (std::abs(l.x) >= hx || l.z <= z0 || l.z >= z1 || l.y >= 2.1f || l.y < 0.0f) continue;
        const float side = l.x < 0.0f ? -1.0f : 1.0f;
        const glm::vec3 out = glm::normalize(glm::vec3(B[0])) * side;
        p.p += out * (hx - std::abs(l.x));
        const float in = glm::dot(p.v - J.st.velocity, -out);
        if (in > 0.0f) p.v += out * in;
    }
    for (auto& p : J.puffs) {
        p.age += dt; p.p += p.v * dt;
        p.v *= std::exp(-2.2f * dt); p.v.y += 0.5f * dt;
        p.size += (1.5f - 0.5f * std::min(p.age, 1.0f)) * dt;
    }
    J.puffs.erase(std::remove_if(J.puffs.begin(), J.puffs.end(), [](const JeepState::Puff& p) { return p.age >= p.life; }), J.puffs.end());
}

// A car's beam: the lit car nearest the camera (one beam at a time, like the torch).
bool VulkanEngine::jeep_headlights(glm::vec4& pos, glm::vec4& dir, glm::vec4& col) const {
    if (!globe::current().home) return false;
    const JeepState* best = nullptr; float bd = 350.0f;
    for (const JeepState* c : jeep_all_) {
        if (!c->posed || !c->lights_on) continue;
        const float d = glm::distance(glm::vec3(c->st.body[3]), player_.eye_position()) - (c == jeep_ && c->driving ? 1000.0f : 0.0f);
        if (d < bd) { bd = d; best = c; }
    }
    if (!best) return false;
    const glm::mat4& B = best->st.body;
    const glm::vec3 f = -glm::normalize(glm::vec3(B[2])), up = glm::normalize(glm::vec3(B[1]));
    const glm::vec3 p = glm::vec3(B * glm::vec4(0.5f * (best->head[0] + best->head[1]), 1.0f)) + f * 0.12f;
    // Dipped: aimed a little down, wide and flat - the shape is the shader's (scene_ubo.glsl
    // car_beam_pattern: a sharp cut-off just under the horizon, a wide spread, the brightest
    // part just under the cut), asked for by col.w > 1.5.
    pos = glm::vec4(p, 270.0f);        // (was 95 m: the road was dark a few car lengths ahead)
    dir = glm::vec4(glm::normalize(f - up * 0.030f), std::cos(glm::radians(40.0f)));
    // HOW BRIGHT AGAINST THE DAY (2026-11-27: "car lights during the day are way too bright, it
    // lights the entire grass field"). A dipped beam puts some tens of lux on the road ahead;
    // the sun and sky put 400 lux on the ground at sunset and up to 100,000 at noon. The beam's
    // strength here (52) is the one that looks right at night, where the eye is adapted to a
    // few lux; against the day it is that times (a few lux / the day's lux): at noon about a
    // ten-thousandth - not seen on the ground at all, as by day; at sunset a hundredth; in the
    // dusk it comes up to full. VIBE_CAR_BEAM_DAY=1: as before (full strength by day too).
    static const bool kOld = std::getenv("VIBE_CAR_BEAM_DAY") != nullptr;
    const float sun_deg = env_->sun_elev > -999.0f ? env_->sun_elev : rt_.sun_pitch_deg;
    const float k = kOld ? 1.0f : std::min(1.0f, 5.0f / daylight_lux(sun_deg, env_->cloud_direct));
    if (k < 2e-4f) return false;       // (nothing that would show: the slot is the torch's again)
    col = glm::vec4(glm::vec3(1.0f, 0.96f, 0.88f) * (52.0f * k), kOld ? std::cos(glm::radians(13.0f)) : 2.0f);
    if (kOld) dir.w = std::cos(glm::radians(34.0f));
    return true;
}

// Hot air behind running cars (lens_fx.cpp: the heat shimmer): one source a car, at the tail
// low down, stronger the harder the engine works. Returns whether the camera is in a car's
// cabin (the driver's nearest view: no rain on the lens there).
bool VulkanEngine::jeep_heat_sources(std::vector<glm::vec4>& out, glm::vec3 eye) const {
    if (!globe::current().home) return false;
    auto add = [&](const JeepState* c) {
        if (!c || !c->posed || !c->snd_running) return;
        const glm::mat4& B = c->st.body;
        const glm::vec3 tail = 0.5f * (c->tail[0] + c->tail[1]);
        const glm::vec3 p = glm::vec3(B * glm::vec4(c->tail[1].x * 0.55f, 0.42f, tail.z + 0.35f, 1.0f));
        if (glm::distance(p, eye) > 70.0f) return;
        const float rpm = glm::clamp(c->snd_rpm / std::max(c->tune.engine_max_rpm, 1000.0f), 0.0f, 1.0f);
        const float k = 0.55f + 0.9f * rpm + 0.5f * glm::clamp(c->snd_load, 0.0f, 1.0f);
        out.push_back(glm::vec4(p + glm::vec3(0.0f, 0.25f, 0.0f), 0.55f));
        out.push_back(glm::vec4(k, 1.6f, 2.0f, 9.0f));
    };
    for (const JeepState* c : jeep_all_) add(c);
    for (const NpcCar& nc : npc_cars_)
        if (nc.agent >= 0 && std::find(jeep_all_.begin(), jeep_all_.end(), nc.car) == jeep_all_.end()) add(nc.car);
    return jeep_ && jeep_->driving && jeep_->cam_mode == 3;       // (2026-11-27: the seat view; the near chase view was taken for the cabin before there was one)
}

// ---------------------------------------------------------------------------------------
// CAR SOUND. Every car that runs has an engine (audio.h "ENGINES"; the loops come out of
// an engine simulation - docs/ARCHITECTURE.md "Car sound"): the jeep a flat four, the sports
// car a flat-plane V8 to 9000, the saloon a quiet four-cylinder, the pickup a V6. The player's car, the
// NPC engine's traffic and the cars other players drive: the nearest kVoices of them that can
// be heard sound.
//   speed of the engine: the driven car's own (the physics); any other car's from its road
//     speed through its gears, changed by the rule the driven car's box has
//     (VehicleDesc::shift_up_rpm / shift_down_rpm: each kind of car its own).
//   load: the driver's throttle; another car's from how it gains or loses speed.
//   where it is heard from: the spatialiser's (audio.h "SOUND IN SPACE", sound_space.h) -
//     quieter and duller with the distance down to silence, Doppler, the side it is on, a hill
//     or a wall in the way, the echo of the surroundings. This function only says where each
//     car is, how fast it goes and how loud its engine is beside it.
//   in the cabin (the driver's nearest view): the own engine through 2.4 kHz, everything
//     outside through 1.4 kHz and at less than half (sound_env.cpp: Space::outside_*).
void VulkanEngine::car_audio_tick(float dt) {
    if (!audio_ || !audio_->ok() || dt <= 0.0f) return;
    constexpr size_t kVoices = 8;           // cars heard at once (one that is let go fades out first: a few more may sound meanwhile)
    // What a car is as a source (sound_space.h): full loudness within about 6 m; from outside
    // its top end is gone already beside it (exhaust and body, not the intake the driver hears).
    constexpr sound_space::Shape kCarShape{ 6.0f, 4500.0f };
    // VIBE_LOG_SOUND_CURVE=1: the curves as a table (the one in docs/ARCHITECTURE.md "Sound in space").
    if (static bool told = false; !told && std::getenv("VIBE_LOG_SOUND_CURVE")) {
        told = true;
        for (float d : { 0.0f, 3.0f, 6.0f, 12.0f, 25.0f, 50.0f, 75.0f, 100.0f, 150.0f, 200.0f, 250.0f, 300.0f, 350.0f, 400.0f, 430.0f, 450.0f }) {
            const float g = sound_space::gain(d, kCarShape.ref_m);
            log::infof("[sound-space] curve at %3.0f m: gain %.5f (%6.1f dB), low-pass %4.0f Hz, reverb share %.4f (%6.1f dB); a car passing at 30 m/s there: Doppler %.3f coming, %.3f going", double(d), double(g),
                       g > 0.0f ? 20.0 * std::log10(double(g)) : -999.0, double(sound_space::cutoff_hz(d, kCarShape.top_hz)), double(sound_space::send(g)),
                       g > 0.0f ? 20.0 * std::log10(double(sound_space::send(g))) : -999.0, double(sound_space::doppler(30.0f, 0.0f, d)), double(sound_space::doppler(-30.0f, 0.0f, d)));
        }
    }
    static const float kGain = std::getenv("VIBE_CAR_SOUND_GAIN") ? float(std::atof(std::getenv("VIBE_CAR_SOUND_GAIN"))) : 1.0f;
    static const bool off = std::getenv("VIBE_NO_CAR_SOUND") != nullptr;
    const glm::vec3 eye = player_.eye_position();
    {   // the listener's own speed (Doppler): the camera's, smoothed
        const glm::vec3 v = car_ear_has_ ? (eye - car_ear_last_) / dt : glm::vec3(0.0f);
        car_ear_vel_ += ((glm::length(v) < 150.0f ? v : glm::vec3(0.0f)) - car_ear_vel_) * std::min(1.0f, dt * 6.0f);
        car_ear_last_ = eye; car_ear_has_ = true;
        audio_->set_listener_velocity(car_ear_vel_);
    }
    struct Cand { JeepState* J; float d; bool wanted = true; float rank = 0.0f; const NpcCar* npc = nullptr; };
    std::vector<Cand> cand;
    const bool home = globe::current().home;
    const bool cabin = jeep_ && jeep_->driving && jeep_->cam_mode == 3;      // (the view from the driver's seat)
    sound_env_tick(dt, cabin);
    if (home && !off) {
        for (JeepState* c : jeep_all_) {
            c->snd_running = c->posed && (c->driving || (c->net_owner && c->net_driven));
            if (c->snd_running) cand.push_back({ c, c->driving ? 0.0f : glm::distance(glm::vec3(c->st.body[3]), eye) });
        }
        for (NpcCar& nc : npc_cars_)
            if (nc.agent >= 0 && nc.car->posed && nc.car->snd_running) cand.push_back({ nc.car, glm::distance(glm::vec3(nc.car->st.body[3]), eye), true, 0.0f, &nc });
    }
    // The nearest kVoices that can be heard at all. A car that has a voice counts as a little
    // nearer than it is: two cars equally far do not take the last voice from each other in turn.
    for (Cand& c : cand) c.rank = c.d * (car_voices_.count(c.J) ? 0.85f : 1.0f);
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.rank < b.rank; });
    // (heard at all: the distance alone leaves it above the floor - twice the floor to be given a voice)
    cand.erase(std::remove_if(cand.begin(), cand.end(), [&](const Cand& c) {
        return sound_space::gain(c.d, kCarShape.ref_m) < (car_voices_.count(c.J) ? 1.0f : 2.0f) * sound_space::kFloor; }), cand.end());
    if (cand.size() > kVoices) cand.resize(kVoices);
    // NOTHING STOPS AT ONCE. A voice whose car is no longer one of the heard (it parked and
    // switched off, a nearer car took its place, it left the reach) is turned down to nothing
    // first (the spatialiser follows its loudness down) and only then let go, once it is under
    // the floor and not mixed any more: a voice cut while it sounds is a click and a hole in
    // the traffic's noise.
    for (auto it = car_voices_.begin(); it != car_voices_.end();) {
        bool keep = false;
        for (const Cand& c : cand) if (c.J == it->first) { keep = true; break; }
        if (keep) { ++it; continue; }
        if (it->first->snd_vol < sound_space::kFloor || !home || off || car_voices_.size() > kVoices + 6) { it->first->snd_vol = 0.0f; audio_->engine_stop(it->second); it = car_voices_.erase(it); continue; }
        cand.push_back({ it->first, glm::distance(glm::vec3(it->first->st.body[3]), eye), false });
        ++it;
    }
    for (const Cand& c : cand) {
        JeepState& J = *c.J;
        if (J.snd_set == -2) {
            const std::string kind = J.dir.substr(J.dir.find_last_of("/\\") + 1);
            J.snd_set = audio_->engine_set_load("assets/sounds/engines/" + kind + ".txt");
        }
        if (J.snd_set < 0) continue;
        auto it = car_voices_.find(&J);
        if (it == car_voices_.end()) {
            const uint32_t h = audio_->engine_start(J.snd_set, kCarShape);
            if (!h) continue;
            it = car_voices_.emplace(&J, h).first;
            J.snd_has_last = false; J.snd_rpm = 900.0f; J.snd_load = 0.0f; J.snd_vol = 0.0f; J.snd_far = false;
        }
        const glm::vec3 pos = glm::vec3(J.st.body * glm::vec4(0.0f, 0.7f, -1.2f, 1.0f));       // (the engine: ahead of the cabin)
        // How fast it goes. A car of the NPC engine: the speed the NPC engine drives it at, along
        // its nose - exact. (From where it was a frame ago, as for the others, its speed came out
        // up to a quarter wrong for a frame now and then: heard as a flutter in the engine's
        // note and in the Doppler shift.)
        glm::vec3 vel_now = J.snd_has_last ? (pos - J.snd_last) / dt : glm::vec3(0.0f);
        if (c.npc) vel_now = -glm::normalize(glm::vec3(J.st.body[2])) * c.npc->last_speed;
        const float sp_prev = glm::length(J.snd_vel);
        J.snd_vel += ((glm::length(vel_now) < 120.0f ? vel_now : glm::vec3(0.0f)) - J.snd_vel) * std::min(1.0f, dt * 8.0f);
        J.snd_last = pos; J.snd_has_last = true;
        const float sp = glm::length(J.snd_vel), limit = std::max(J.tune.engine_max_rpm, 3000.0f);
        float rpm_t, load_t;
        if (J.driving && &J == jeep_) {
            rpm_t = std::max(std::abs(J.st.rpm), J.tune.engine_idle_rpm);
            load_t = std::clamp(J.throttle_in, 0.0f, 1.0f);
        } else {
            // through its gears
            const size_t ng = std::max<size_t>(J.tune.gears.size(), 1);
            auto rpm_in = [&](int g) {
                const float ratio = J.tune.gears.empty() ? 1.5f : J.tune.gears[size_t(std::clamp(g, 0, int(ng) - 1))];
                return sp / (6.2831853f * std::max(J.wheel_radius, 0.2f)) * 60.0f * ratio * J.tune.final_drive;
            };
            J.snd_gear = std::clamp(J.snd_gear, 0, int(ng) - 1);
            // (as a driver changes: early when he is only keeping his speed, late when he is on it)
            const float up = J.tune.shift_up_rpm(J.snd_load), down = J.tune.shift_down_rpm(J.snd_load);
            if (rpm_in(J.snd_gear) > up && J.snd_gear + 1 < int(ng)) ++J.snd_gear;
            else if (rpm_in(J.snd_gear) < down && J.snd_gear > 0) --J.snd_gear;
            rpm_t = std::max(rpm_in(J.snd_gear), J.tune.engine_idle_rpm);
            const float accel = (sp - sp_prev) / dt;
            load_t = std::clamp(0.24f + accel * 0.30f, 0.0f, 1.0f);
        }
        J.snd_rpm += (rpm_t - J.snd_rpm) * std::min(1.0f, dt * 7.0f);
        J.snd_load += (load_t - J.snd_load) * std::min(1.0f, dt * 9.0f);
        const bool own = J.driving && &J == jeep_;
        const float d = own ? glm::distance(pos, eye) : c.d;
        // How loud the engine is beside it: its load and its speed, the kind of car. What is heard
        // of that from where the listener is - distance, tone, Doppler, side, a hill in the way,
        // the echo - is the spatialiser's (audio.h "SOUND IN SPACE"; before 2026-11-25's fifth pass
        // this function had its own loudness and low-pass curves against the distance).
        // The car the player drives is not a source in space: the camera's distance is not the
        // driver's. It is in the middle, through the chase view's 12 kHz or the cabin's low-pass.
        float level = 0.62f * (0.50f + 0.50f * J.snd_load) * (0.60f + 0.40f * std::clamp(J.snd_rpm / limit, 0.0f, 1.0f)) * kGain * J.tune.engine_loudness;
        float ridden_hz = 0.0f;
        if (own) { level *= 0.42f; ridden_hz = 12000.0f; if (cabin) { ridden_hz = 2400.0f * (0.55f + 0.45f * J.tune.engine_loudness); level *= 0.9f; } }
        if (!c.wanted) level = 0.0f;
        // A quiet car plays one loop instead of four (the blend is not heard under the others).
        // By loudness, not by distance, and with a margin: the change is made where it is
        // quietest and not back and forth. It is brought up to date every frame all the same -
        // every fourth frame made its fading a staircase.
        if (own) J.snd_far = false; else if (J.snd_vol < 0.006f) J.snd_far = true; else if (J.snd_vol > 0.010f) J.snd_far = false;
        const AudioEngine::Heard heard = audio_->engine_update(it->second, pos, J.snd_vel, J.snd_rpm, J.snd_load, level, ridden_hz, J.snd_far);
        J.snd_vol = heard.gain;       // (the loudness its voice has now: followed by the spatialiser, never jumps)
        if (static const bool lg = std::getenv("VIBE_LOG_CAR_SOUND") != nullptr; lg && frame_number_ % 90 == 0)
            log::infof("[car-sound] %s%s: %.0f m, %.1f m/s, gear %d, %.0f rpm, load %.2f, level %.3f, heard %.4f%s%s, low-pass %.0f Hz", J.dir.c_str(), own ? " (driven)" : "", double(d), double(sp),
                       J.snd_gear + 1, double(J.snd_rpm), double(J.snd_load), double(level), double(heard.gain), J.snd_far ? " far voice" : "", c.wanted ? "" : " fading out", double(heard.cutoff_hz));
        // VIBE_LOG_SOUND_SPACE=1: what is heard of every car that has a voice, six times a second
        // - a pass-by read as numbers (VIBE_LOG_SOUND_SPACE=<n>, n > 1: every n-th frame; 1: every tenth).
        static const int every = [] { const char* e = std::getenv("VIBE_LOG_SOUND_SPACE"); const int n = e ? std::atoi(e) : 0; return !e ? 0 : n > 1 ? n : 10; }();
        if (every > 0 && frame_number_ % every == 0 && !own)
            log::infof("[sound-space] car %p %s: %6.1f m, gain %.5f (%6.1f dB), low-pass %4.0f Hz, doppler %.4f, pan %+.2f, reverb %.5f, blocked %.2f, %.1f m/s%s", static_cast<void*>(&J),
                       J.dir.substr(J.dir.find_last_of("/\\") + 1).c_str(), double(heard.distance), double(heard.gain), heard.gain > 0.0f ? 20.0 * std::log10(double(heard.gain)) : -999.0,
                       double(heard.cutoff_hz), double(heard.doppler), double(heard.pan), double(heard.send), double(heard.occlusion), double(sp), c.wanted ? "" : " (let go)");
    }
    car_tyre_audio(dt);         // (the driven car's tyres: a burnout, a slide)
    siren_audio_tick(dt);       // (the sirens: sources of the same spatialiser, after the listener's speed and the surroundings are set)
    // VIBE_LOG_AUDIO_COST=1: every 300 frames, what the mixer cost since the last line (all the
    // engine's sound, on the audio thread) and how many engine voices and loops were in it.
    if (static const bool lg = std::getenv("VIBE_LOG_AUDIO_COST") != nullptr; lg && frame_number_ % 300 == 0) {
        const AudioEngine::MixerCost mc = audio_->mixer_cost_take();
        int voices = 0, loops = 0;
        audio_->engine_counts(voices, loops);
        if (mc.audio_ms > 0.0)
            log::infof("[audio-cost] mixer %.2f ms for %.0f ms of sound = %.3f %% of a core (%.1f us a block, %llu blocks); engine voices %d, loops mixed %d",
                       mc.mix_ms, mc.audio_ms, 100.0 * mc.mix_ms / mc.audio_ms, 1000.0 * mc.mix_ms / double((std::max<uint64_t>)(mc.blocks, 1)), static_cast<unsigned long long>(mc.blocks), voices, loops);
    }
}

// Evening: when a car that drives has its lights on. The clock's lantern factor when the day
// runs (it comes up through the dusk), else the sun's height: on from when the sun is under
// about 12 degrees.
bool VulkanEngine::cars_need_lights() const {
    const float night = env_->night >= 0.0f ? env_->night : 1.0f - glm::smoothstep(4.0f, 16.0f, rt_.sun_pitch_deg);
    return night > 0.10f || rt_.sun_pitch_deg < 12.0f;
}

void VulkanEngine::jeep_fx(std::vector<Vertex>& marks, std::vector<Vertex>& puffs, std::vector<Vertex>& glow, glm::vec3 eye, glm::vec3 R, glm::vec3 U) {
    if (!globe::current().home) return;
    static const bool smoke_old = std::getenv("VIBE_TEST_SMOKE_OLD") != nullptr;       // (tests: as it was, to compare)
    // (a head lamp's glow by day: a third of the night's - a lamp that is on shows, it does not blaze; the dusk brings it up)
    const float head_day = 0.35f + 0.65f * glm::smoothstep(0.001f, 0.2f, std::min(1.0f, 5.0f / daylight_lux(env_->sun_elev > -999.0f ? env_->sun_elev : rt_.sun_pitch_deg, env_->cloud_direct)));
    auto flare = [&](glm::vec3 p, glm::vec3 out, float r, glm::vec3 tint) {
        const glm::vec3 to = eye - p;
        const float d = glm::length(to);
        const float facing = d > 1e-3f ? glm::dot(to / d, out) : 0.0f;       // (a lamp shines to its own side)
        if (facing < 0.05f || d > 900.0f) return;
        const float k = glm::smoothstep(0.05f, 0.5f, facing);
        const float rr = std::max(r, 0.012f * d) * (0.6f + 0.4f * k);       // (never smaller than a few pixels: a light far off is still a point of light)
        const glm::vec3 t = tint * k, a = R * rr, b = U * rr;
        const Vertex q0{ p - a - b, t, { -1.0f, -1.0f } }, q1{ p + a - b, t, { 1.0f, -1.0f } };
        const Vertex q2{ p + a + b, t, { 1.0f, 1.0f } }, q3{ p - a + b, t, { -1.0f, 1.0f } };
        glow.insert(glow.end(), { q0, q1, q2, q0, q2, q3 });
    };
    // (the NPC engine's cars: the same lamps - head lights when they are on, tail lights, brake lights)
    for (const NpcCar& nc : npc_cars_) {
        if (nc.agent < 0 || !nc.car->posed) continue;
        const JeepState* c = nc.car;
        const glm::mat4& B = c->st.body;
        const glm::vec3 f = -glm::normalize(glm::vec3(B[2]));
        const bool braking = c->braking();
        for (int s = 0; s < 2; ++s) {
            if (c->lights_on) flare(glm::vec3(B * glm::vec4(c->head[s], 1.0f)) + f * 0.05f, f, 0.42f, glm::vec3(1.0f, 0.95f, 0.85f) * 2.4f * head_day);
            if (c->lights_on || braking)
                flare(glm::vec3(B * glm::vec4(c->tail[s], 1.0f)) - f * 0.05f, -f, braking ? 0.30f : 0.20f, glm::vec3(1.0f, 0.06f, 0.04f) * (braking ? 2.6f : 0.9f));
        }
    }
    // (the police cars' light bars: the lamp that is lit, seen from all round)
    if (cars_x_)
        for (const CarTrafficState::Bar& b : cars_x_->bars)
            for (int s = 0; s < 2 && b.agent >= 0; ++s)
                if (b.lit[s]) { const glm::vec3 to = eye - b.lamp[s]; if (glm::dot(to, to) > 1e-4f) flare(b.lamp[s], glm::normalize(to), 0.50f, (s == 0 ? glm::vec3(0.10f, 0.28f, 1.0f) : glm::vec3(1.0f, 0.06f, 0.05f)) * 3.4f); }
    for (const JeepState* c : jeep_all_) {
        if (c->posed) {
            const glm::mat4& B = c->st.body;
            const glm::vec3 f = -glm::normalize(glm::vec3(B[2]));
            const bool braking = c->braking();
            for (int s = 0; s < 2; ++s) {
                if (c->lights_on) flare(glm::vec3(B * glm::vec4(c->head[s], 1.0f)) + f * 0.05f, f, 0.42f, glm::vec3(1.0f, 0.95f, 0.85f) * 2.4f * head_day);
                if (c->boosting) flare(glm::vec3(B * glm::vec4(c->tail[s], 1.0f)) - f * 0.08f, -f, 0.36f, glm::vec3(0.35f, 0.65f, 1.0f) * 3.2f);
                if (c->lights_on || braking)
                    flare(glm::vec3(B * glm::vec4(c->tail[s], 1.0f)) - f * 0.05f, -f, braking ? 0.30f : 0.20f, glm::vec3(1.0f, 0.06f, 0.04f) * (braking ? 2.6f : 0.9f));
            }
        }
        for (int w = 0; w < 4; ++w) {
            const auto& M = c->marks[w];
            for (size_t i = 0; i + 1 < M.size(); ++i) {
                if (M[i].a <= 0.0f) continue;                       // (the end of one mark: a gap to the next)
                const glm::vec3 t(0.0f);
                const Vertex q0{ M[i].p - M[i].side, t, { M[i].a, -1.0f } }, q1{ M[i].p + M[i].side, t, { M[i].a, 1.0f } };
                const Vertex q2{ M[i + 1].p + M[i + 1].side, t, { M[i + 1].a, 1.0f } }, q3{ M[i + 1].p - M[i + 1].side, t, { M[i + 1].a, -1.0f } };
                marks.insert(marks.end(), { q0, q1, q2, q0, q2, q3 });
            }
        }
        // SMOKE IS NOT DRAWN INTO THE CAR. A puff is a flat round picture turned to the eye, grown
        // to a metre and a half: one at a rear wheel reached right across the car, and from behind
        // everything ahead of it - the seats, the driver - stood in smoke. A puff is drawn no wider
        // than its distance from the body's box allows (jeep_marks keeps its middle outside): small
        // at the tyre, full size once the car has left it behind or it has drifted clear.
        float hx, z0, z1; c->body_box(hx, z0, z1);
        const glm::mat4 Binv = c->puffs.empty() || !c->posed ? glm::mat4(1.0f) : glm::inverse(c->st.body);
        for (const auto& p : c->puffs) {
            const float f = p.age / p.life;
            const float a = 0.34f * std::min(1.0f, p.age * 9.0f) * (1.0f - f) * (1.0f - f);
            float size = p.size;
            if (c->posed && !smoke_old) {
                const glm::vec3 l(Binv * glm::vec4(p.p, 1.0f));
                const glm::vec3 o(std::max(std::abs(l.x) - hx, 0.0f), std::max(std::max(-l.y, l.y - 1.6f), 0.0f), std::max(std::max(z0 - l.z, l.z - z1), 0.0f));
                size = std::min(size, glm::length(o) + 0.30f);
            }
            const glm::vec3 t(a, p.bright, p.seed);
            const glm::vec3 r = R * size, u = U * size;
            const Vertex q0{ p.p - r - u, t, { -1.0f, -1.0f } }, q1{ p.p + r - u, t, { 1.0f, -1.0f } };
            const Vertex q2{ p.p + r + u, t, { 1.0f, 1.0f } }, q3{ p.p - r + u, t, { -1.0f, 1.0f } };
            puffs.insert(puffs.end(), { q0, q1, q2, q0, q2, q3 });
        }
    }
}

// ---- Multiplayer: cars (net_state.h "Cars") ---------------------------------------------------
// A car is simulated where its driver sits, and stays that machine's after the driver has got
// out, until somebody else gets in. That machine sends its state (30 times a second while it
// moves, once a second standing - so a player who joins later sees it where it is); every
// other machine draws it from those states and does not simulate it.

// After the simulation (vk_engine.cpp, behind jeep_after_sim): the states of the cars this
// machine simulates, as they are in this frame, stamped with this frame's time. (The session's
// update runs earlier in the frame, before the physics: sent from there, a state was last
// frame's with this frame's time on it - off by one frame's travel, a different amount every
// frame, which other machines saw as a car that jerks.)
void VulkanEngine::jeep_net_send(float dt) {
    // The time these states are of: the frame's, less what the simulation is behind the clock
    // (sim_debt_s_). After a long frame the car has moved less than the clock: stamped with
    // the clock it reached others as a car that slowed down and sped up again. The debt
    // drains at 1 % of the time passing (the car looks 1 % slow meanwhile); too much to make
    // up (the game stood still for a moment) is told as a jump.
    sim_debt_s_ = std::max(0.0f, sim_debt_s_ - 0.01f * dt);
    const bool stood = sim_debt_s_ > 0.4f;
    if (stood) sim_debt_s_ = 0.0f;
    if (!net_) return;
    for (size_t i = 0; i < jeep_all_.size() && i < size_t(net::kMaxCars); ++i) {
        JeepState& J = *jeep_all_[i];
        if (!J.net_mine || !J.posed) continue;
        // (live: it moves - driven or rolling here, or struck and sliding: cars_collide)
        const bool live = (&J == jeep_ && J.spawned && (J.driving || glm::length(J.st.velocity) > 0.15f)) ||
                          (&J != jeep_ && cars_x_ && cars_x_->shove.count(&J) && glm::length(J.st.velocity) > 0.05f);
        J.net_send -= dt;
        if (live) J.net_send = std::min(J.net_send, 1.0f / 30.0f);
        if (J.net_send > 0.0f) continue;
        J.net_send = std::max(0.0f, J.net_send + (live ? 1.0f / 30.0f : 1.0f));
        net::CarState c;
        c.car = uint8_t(i);
        const glm::mat4& B = J.st.body;
        const glm::quat q = glm::normalize(glm::quat_cast(glm::mat3(B)));
        const glm::mat4 inv = glm::inverse(B);
        for (int k = 0; k < 3; ++k) { c.p[k] = B[3][k]; c.v[k] = J.st.velocity[k]; c.w[k] = J.st.angular_velocity[k]; }
        c.q[0] = q.x; c.q[1] = q.y; c.q[2] = q.z; c.q[3] = q.w;
        c.steer = J.st.steer;
        c.flags = uint8_t((J.driving ? net::kCarDriven : 0) | (J.lights_on ? net::kCarLights : 0) | (J.braking() ? net::kCarBrake : 0));
        for (int w = 0; w < 4; ++w) {
            c.susp[w] = (inv * J.st.wheel[w][3]).y - J.wheel_pos[w].y;
            c.spin[w] = live ? J.st.wheel_spin[w] : 0.0f;
            if (J.st.wheel_contact[w]) c.flags |= uint8_t(net::kCarContact0 << w);
        }
        if (stood) c.flags |= net::kCarTeleport;
        net_->queue_car(c, frame_clock_s_ - double(sim_debt_s_));
        // VIBE_TEST_CAR_ECHO=1: what a receiver makes of this state against what is drawn here
        // (the wheels' places and axles, and which way they roll) - once a second.
        if (static const bool echo = std::getenv("VIBE_TEST_CAR_ECHO") != nullptr; echo && live) {
            static float roll_prev[4] = { 0, 0, 0, 0 };
            const net::CarState r = net::unpack_car(net::pack_car(c));
            float worst_p = 0.0f, worst_ax = 0.0f, rate[4];
            for (int w = 0; w < 4; ++w) {
                const glm::mat4 m = jeep_wheel_from(J, B, w, r.steer, r.susp[w], 0.0f);
                worst_p = std::max(worst_p, glm::distance(glm::vec3(m[3]), glm::vec3(J.st.wheel[w][3])));
                worst_ax = std::max(worst_ax, glm::distance(glm::vec3(m[0]), glm::vec3(J.st.wheel[w][0])));
                // the drawn wheel's roll: its y axis in the frame of the steered, unrolled wheel
                const glm::vec3 y = glm::vec3(glm::inverse(m) * J.st.wheel[w][1]);
                const float roll = std::atan2(y.z, y.y);
                rate[w] = std::remainder(roll - roll_prev[w], 6.2831853f);
                roll_prev[w] = roll;
            }
            if (frame_number_ % 30 == 0)
                log::infof("[jeep] echo: wheel place off by %.3f m, axle by %.3f; steer %.2f; roll step %.3f (spin x dt %.3f)",
                           double(worst_p), double(worst_ax), double(c.steer), double(rate[2]), double(c.spin[2] / 30.0f));
        }
    }
}

// A car's wheel from what is sent about it: its place under the body, steered, turned.
// (Jolt's wheel frame with the axle along +x: steering is a turn about y, rolling about x.)
glm::mat4 VulkanEngine::jeep_wheel_from(const JeepState& J, const glm::mat4& body, int w, float steer, float susp, float roll) {
    return body * glm::translate(glm::mat4(1.0f), J.wheel_pos[w] + glm::vec3(0.0f, susp, 0.0f)) *
           glm::rotate(glm::mat4(1.0f), w < 2 ? steer : 0.0f, glm::vec3(0.0f, 1.0f, 0.0f)) *
           glm::rotate(glm::mat4(1.0f), roll, glm::vec3(1.0f, 0.0f, 0.0f));
}

// After the session's update: the cars other machines simulate, where their states put them.
void VulkanEngine::jeep_net_receive(float dt) {
    if (!net_) return;
    for (const net::RemoteCar& rc : net_->cars()) {
        if (!rc.fresh || size_t(rc.car) >= jeep_all_.size()) continue;
        JeepState& J = *jeep_all_[rc.car];
        const net::CarState& s = rc.state;
        const bool driven = (s.flags & net::kCarDriven) != 0 && rc.age < 3.0;
        if (J.net_mine) {
            // Two machines with the same car. Old news (it was theirs before it became ours) is
            // passed over. Otherwise whoever sits in it has it; both sitting in it (they got in
            // in the same moment): the lower id, the same answer on both machines.
            if (rc.age > 3.0) continue;
            if (cars_x_) { const auto sh = cars_x_->shove.find(&J); if (sh != cars_x_->shove.end() && sh->second.claim > 0.0f) continue; }      // (struck and taken over a moment ago: cars_collide)
            if (J.driving && (!driven || net_->id() < rc.sender)) continue;
            J.net_mine = false;
            if (J.driving && &J == jeep_) jeep_use();                  // (out of it)
        }
        if (&J == jeep_ && J.spawned) {                                // (no longer simulated here)
            if (J.patch_busy && J.patch_job.valid()) J.patch_job.wait();
            J.patch_busy = false;
            physics_->vehicle_remove();
            J.spawned = false;
        }
        const bool jump = rc.teleported || J.net_owner != rc.sender || !J.posed;
        J.net_owner = rc.sender;
        J.net_driven = driven;
        const PhysicsWorld::VehicleState prev = J.st;
        glm::mat4 B = glm::mat4_cast(glm::quat(s.q[3], s.q[0], s.q[1], s.q[2]));
        B[3] = glm::vec4(s.p[0], s.p[1], s.p[2], 1.0f);
        J.st.body = B;
        J.st.velocity = glm::vec3(s.v[0], s.v[1], s.v[2]);
        J.st.angular_velocity = glm::vec3(s.w[0], s.w[1], s.w[2]);
        J.st.steer = s.steer;
        for (int w = 0; w < 4; ++w) {
            J.st.wheel_contact[w] = (s.flags & (net::kCarContact0 << w)) != 0;
            J.st.wheel_spin[w] = s.spin[w];
            // (the wheels' turn is carried on here from their speed: a wheel goes round several
            //  times between two states, so its angle cannot be sent and blended)
            J.net_roll[w] = std::remainder(J.net_roll[w] - s.spin[w] * dt, 6.2831853f);       // (the mesh's frame turns the other way to Jolt's wheel angle)
            J.st.wheel[w] = jeep_wheel_from(J, B, w, s.steer, s.susp[w], J.net_roll[w]);
        }
        J.lights_on = (s.flags & net::kCarLights) != 0;
        J.brake_in = (s.flags & net::kCarBrake) ? 1.0f : 0.0f;
        J.body_prev = jump ? B : prev.body;
        J.posed = true;
        jeep_place(J, jump ? J.st : prev, dt);
        jeep_marks(J, dt);
        if (static const bool lg = std::getenv("VIBE_LOG_JEEP") != nullptr; lg) {
            // (how evenly it moves here. A frame's step against speed x time: `off`. A steady
            //  difference is a car shown a per cent slow or fast while the delay or the sender's
            //  clock is brought in line - nobody sees that. What is seen as a jerk is that
            //  difference CHANGING from one frame to the next: `worst`.)
            static float worst = 0.0f, off_last = 0.0f;
            static double shown_last = 0.0;
            const float off = glm::length(glm::vec3(B[3]) - glm::vec3(prev.body[3])) - glm::length(J.st.velocity) * dt;
            const float err = std::abs(off - off_last);
            off_last = off;
            if (!jump) worst = std::max(worst, err);
            if (static const char* lv = std::getenv("VIBE_LOG_JEEP_NET"); lv && lv[0] == '2')
                log::infof("[jeep] SH %.3f %.4f %.4f %.4f %.3f %.3f %.3f %.3f", rc.shown_ms, double(s.p[0]), double(s.p[1]), double(s.p[2]), double(s.v[0]), double(s.v[1]), double(s.v[2]), double(dt) * 1000.0);
            if (!jump && err > 0.01f && std::getenv("VIBE_LOG_JEEP_NET"))
                log::infof("[jeep] net step off by %.3f m: frame %.2f ms, shown time moved %.2f ms, extrapolated %d, newest state %.1f ms ahead, %zu states",
                           double(err), double(dt) * 1000.0, rc.shown_ms - shown_last, int(rc.extrapolated), rc.buf.snaps.empty() ? 0.0 : rc.buf.snaps.back().t - rc.shown_ms, rc.buf.snaps.size());
            shown_last = rc.shown_ms;
            if (frame_number_ % 60 == 0) {
                log::infof("[jeep] remote car %d from %08X: pos (%.1f, %.2f, %.1f) %.1f km/h driven %d, a frame's step changed by %.3f m at most",
                           int(rc.car), rc.sender, double(B[3].x), double(B[3].y), double(B[3].z), double(glm::length(J.st.velocity) * 3.6f), int(driven), double(worst));
                worst = 0.0f;
            }
        }
    }
}

// The seat beside the driver: the driver's, mirrored to the other side. The hands lie on the knees.
bool VulkanEngine::jeep_passenger_seat(int car, glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs, anim::CharacterAnimationInput::SeatExtras* more) const {
    if (car < 0 || size_t(car) >= jeep_all_.size() || !jeep_all_[size_t(car)]->posed) return false;
    const JeepState& J = *jeep_all_[size_t(car)];
    const glm::vec3 seat(-J.hip.x, J.hip.y, J.hip.z);
    hip = J.st.body * glm::translate(glm::mat4(1.0f), seat);
    hand_l = glm::vec3(J.st.body * glm::vec4(seat + glm::vec3(-0.17f, 0.14f, -0.36f), 1.0f));
    hand_r = glm::vec3(J.st.body * glm::vec4(seat + glm::vec3( 0.17f, 0.14f, -0.36f), 1.0f));
    legs = J.legs;
    if (more) jeep_seat_feet(J, true, *more);
    return true;
}

// WHERE A CAR HAS ITS OCCUPANT'S FEET AND BACK (the car's file: JeepState::pedal ..). The
// driver: the right foot's ball on the accelerator's pad, its heel on the floor behind it; the
// left foot on the floor beside the pedals, the toes up the footwell's slope. Beside the
// driver: both feet so, in front of that seat. Nothing where the file does not say.
void VulkanEngine::jeep_seat_feet(const JeepState& J, bool passenger, anim::CharacterAnimationInput::SeatExtras& e) {
    e.back = J.back_deg;
    if (!J.has_feet) return;
    const glm::mat4& B = J.st.body;
    auto w = [&](glm::vec3 p) { return glm::vec3(B * glm::vec4(p, 1.0f)); };
    const float foot = 0.19f;                                   // heel to the ball of the foot
    const glm::vec3 toes(0.0f, foot * 0.42f, -foot * 0.907f);   // (a resting foot: 25 degrees up)
    e.feet = true;
    if (passenger) {
        for (int s = 0; s < 2; ++s) {
            const glm::vec3 heel(-J.hip.x + (s == 0 ? -0.11f : 0.11f), J.floor_y, J.foot_rest.y);
            (s == 0 ? e.heel_l : e.heel_r) = w(heel);
            (s == 0 ? e.ball_l : e.ball_r) = w(heel + toes);
        }
        return;
    }
    const float rise = glm::clamp(J.pedal.y - J.floor_y, 0.02f, foot * 0.9f);
    const glm::vec3 heel_r(J.pedal.x, J.floor_y, J.pedal.z + std::sqrt(foot * foot - rise * rise));
    e.heel_r = w(heel_r);
    e.ball_r = w(glm::vec3(J.pedal.x, J.floor_y + rise, J.pedal.z));
    const glm::vec3 heel_l(J.foot_rest.x, J.floor_y, J.foot_rest.y);
    e.heel_l = w(heel_l);
    e.ball_l = w(heel_l + toes);
}

// The local player riding beside another player's driver: carried with the car, eye at the
// seat's head height. (Called after the simulation, so the walk's own step - gravity - does
// not show: the place is the seat's, every frame.)
void VulkanEngine::jeep_ride_place() {
    if (jeep_ride_ < 0) return;
    if (size_t(jeep_ride_) >= jeep_all_.size() || !jeep_all_[size_t(jeep_ride_)]->posed) { jeep_ride_ = -1; return; }
    const JeepState& J = *jeep_all_[size_t(jeep_ride_)];
    const glm::vec3 eye(J.st.body * glm::vec4(-J.hip.x, J.hip.y + 0.74f, J.hip.z, 1.0f));
    player_.position = eye - (player_.eye_position() - player_.position);
    player_.velocity = J.st.velocity;
    player_.on_ground = true;
    prev_player_position_ = player_.position;
}

// The seat of the car a player drives (another player's: multiplayer.cpp poses them in it).
bool VulkanEngine::jeep_seat_of(uint32_t player, glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs, anim::CharacterAnimationInput::SeatExtras* more) const {
    for (const JeepState* c : jeep_all_)
        if (c->posed && c->net_owner == player && c->net_driven) { jeep_seat(*c, hip, hand_l, hand_r, legs, more); return true; }
    return false;
}

// (`more`: for a body that is fitted to the car itself - CharacterAnimationInput::SeatExtras.
//  The hands are then points ON the rim, at a quarter to three, and the wheel's frame, the
//  feet's places and the backrest's lean come with them.)
void VulkanEngine::jeep_seat(const JeepState& J, glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs, anim::CharacterAnimationInput::SeatExtras* more) {
    const glm::mat4& B = J.st.body;
    hip = B * glm::translate(glm::mat4(1.0f), J.hip);
    // The wheel's own frame: +z along the column toward the driver. The hands hold the rim at
    // ten to two and go round with it - up to a quarter turn, then the wheel slides through them.
    const glm::mat4 W = B * glm::translate(glm::mat4(1.0f), J.steering_hub) *
                        glm::rotate(glm::mat4(1.0f), glm::radians(-J.steering_tilt), glm::vec3(1.0f, 0.0f, 0.0f));
    const float phi = glm::clamp(J.steer_wheel * 9.0f, -1.5f, 1.5f);
    auto at = [&](float deg) {
        const float th = glm::radians(deg) + phi;
        return glm::vec3(W * glm::vec4(J.rim_r * std::cos(th), J.rim_r * std::sin(th), 0.07f, 1.0f));
    };
    hand_l = at(160.0f);
    hand_r = at(20.0f);
    legs = J.legs;
    if (more) {
        auto rim = [&](float deg) {
            const float th = glm::radians(deg) + phi;
            return glm::vec3(W * glm::vec4(J.rim_r * std::cos(th), J.rim_r * std::sin(th), 0.0f, 1.0f));
        };
        hand_l = rim(172.0f);
        hand_r = rim(8.0f);
        more->wheel = true;
        more->hub = glm::vec3(W[3]);
        more->axis = glm::vec3(W[2]);
        more->steer = -phi / 1.5f;
        jeep_seat_feet(J, false, *more);
    }
}

bool VulkanEngine::jeep_driver(glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs, anim::CharacterAnimationInput::SeatExtras* more) const {
    if (!jeep_ || !jeep_->driving || !jeep_->spawned) return false;
    jeep_seat(*jeep_, hip, hand_l, hand_r, legs, more);
    return true;
}


// The cars the grass lies down under (scene UBO car_press, grass_raymarch.frag): the one driven
// here and the nearest other that moves or was just left. Behind a car the grass is still down
// over the stretch it drove in the last second and a half (25 m at most): a straight stretch
// behind where it faces now - in a bend the far end of it lies a little beside the real track.
void VulkanEngine::jeep_grass_press(glm::vec4 out[4]) const {
    for (int i = 0; i < 4; ++i) out[i] = glm::vec4(0.0f);
    if (!globe::current().home) return;                        // (cars stand in home's chart)
    const glm::vec3 eye = player_.eye_position();
    const JeepState* pick[2] = { nullptr, nullptr };
    float best[2] = { 1e9f, 1e9f };
    for (const JeepState* c : jeep_all_) {
        if (!c->posed) continue;
        const float d = glm::distance(glm::vec3(c->st.body[3]), eye);
        if (d > 250.0f) continue;
        for (int k = 0; k < 2; ++k)
            if (d < best[k]) {
                if (k == 0) { best[1] = best[0]; pick[1] = pick[0]; }
                best[k] = d; pick[k] = c;
                break;
            }
    }
    for (int k = 0; k < 2; ++k) {
        const JeepState* c = pick[k];
        if (!c) continue;
        const glm::mat4& B = c->st.body;
        glm::vec2 f(-B[2].x, -B[2].z);
        const float fl = glm::length(f);
        if (fl < 1e-3f || B[1].y < 0.3f) continue;             // (on its side or roof: nothing under it)
        f /= fl;
        const float half_w = 0.5f * std::abs(c->wheel_pos[1].x - c->wheel_pos[0].x) + 0.5f * c->wheel_width + 0.12f;
        const float half_l = 0.5f * std::abs(c->wheel_pos[2].z - c->wheel_pos[0].z) + c->wheel_radius + 0.35f;
        const glm::vec3 mid(B * glm::vec4(0.0f, 0.0f, 0.5f * (c->wheel_pos[0].z + c->wheel_pos[2].z), 1.0f));
        const float along = glm::dot(glm::vec2(c->st.velocity.x, c->st.velocity.z), f);      // (reversing: no stretch behind)
        out[2 * k]     = glm::vec4(mid.x, mid.z, f.x, f.y);
        out[2 * k + 1] = glm::vec4(half_w, half_l, glm::clamp(along * 1.5f, 0.0f, 25.0f), 1.0f);
    }
}


// THE VIEW WIDENS WITH SPEED (2026-11-27, "nicer effect when using boost, a little more FOV"):
// degrees added to the view's angle while driving - up to VIBE_CAR_FOV_SPEED (5) with the speed
// from 55 to 200 km/h, VIBE_CAR_FOV_BOOST (6) more while the boost pushes; eased in over a
// third of a second, out over two thirds (jeep_after_sim). Not in the seat view.
float VulkanEngine::jeep_fov_extra() const {
    if (!jeep_ || !jeep_->driving || !jeep_->spawned) return 0.0f;
    return jeep_->cam_mode == 3 ? 0.4f * jeep_->fov_extra : jeep_->fov_extra;
}

// SNOW ON THE CARS (2026-11-27). Winter lays snow on what faces up (scene_ubo.glsl snow_cover),
// cars included; a car that drives loses it: the share it still carries goes down by speed x
// time - 60 s at 72 km/h, never faster than 15 s - and comes back on one that stands in falling
// snow over about three minutes (by how hard it snows). The NPC cars' in npc_cars_place.
void VulkanEngine::cars_snow_tick(float dt) {
    if (dt <= 0.0f) return;
    for (JeepState* c : jeep_all_) {
        if (!c->posed) continue;
        const float sp = glm::length(c->st.velocity);
        if (sp > 1.0f) c->snow = std::max(0.0f, c->snow - dt * std::min(sp / 20.0f, 4.0f) / 60.0f);
        else if (env_->snowfall > 0.05f) c->snow = std::min(1.0f, c->snow + dt * env_->snowfall / 180.0f);
    }
}

// The cars' snow for the scene block (scene_ubo.glsl car_snow): the eight posed cars nearest the
// eye within 300 m, two vec4 each - [0] xy the middle (world xz), zw the way it faces; [1] x half
// its width, y half its length, z its floor's height (world y), w the share of its snow left.
// An unused slot has x = 0 (the shader stops there). A car not in the list keeps all its snow.
void VulkanEngine::jeep_snow_fill(glm::vec4 out[16]) const {
    for (int i = 0; i < 16; ++i) out[i] = glm::vec4(0.0f);
    if (!globe::current().home) return;
    const glm::vec3 eye = player_.eye_position();
    std::vector<std::pair<float, const JeepState*>> near_;
    auto add = [&](const JeepState* c) {
        if (!c || !c->posed || c->snow > 0.995f) return;
        const float d = glm::distance(glm::vec3(c->st.body[3]), eye);
        if (d < 300.0f) near_.push_back({ d, c });
    };
    for (const JeepState* c : jeep_all_) add(c);
    for (const NpcCar& nc : npc_cars_) if (nc.agent >= 0) add(nc.car);
    std::sort(near_.begin(), near_.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t k = 0; k < near_.size() && k < 8; ++k) {
        const JeepState* c = near_[k].second;
        const glm::mat4& B = c->st.body;
        glm::vec2 f(-B[2].x, -B[2].z);
        const float fl = glm::length(f);
        f = fl > 1e-3f ? f / fl : glm::vec2(0.0f, -1.0f);
        float hx, z0, z1; c->body_box(hx, z0, z1);
        const glm::vec3 mid(B * glm::vec4(0.0f, 0.0f, 0.5f * (z0 + z1), 1.0f));
        out[2 * k]     = glm::vec4(mid.x, mid.z, f.x, f.y);
        out[2 * k + 1] = glm::vec4(hx + 0.05f, 0.5f * (z1 - z0) + 0.05f, mid.y, c->snow);
    }
}

// THE TYRES SQUEAL (2026-11-27): one sound for the car that is driven - a burnout, a slide, a
// locked wheel on a hard surface (jeep_marks' `squeal`, 0..1). A source of the spatialiser at the
// car (audio.h "SOUND IN SPACE"); its loudness follows the slide, its pitch rises with it. The
// sound is made in code (engine/tyre_synth.h: 6 s that loop), written to
// assets/sounds/tyre_squeal.wav the first time it is needed. VIBE_NO_TYRE_SQUEAL=1: none.
void VulkanEngine::car_tyre_audio(float dt) {
    if (!cars_x_ || !audio_ || !audio_->ok() || dt <= 0.0f) return;
    CarTrafficState& X = *cars_x_;
    static const bool off = std::getenv("VIBE_NO_TYRE_SQUEAL") != nullptr;
    const bool on = !off && jeep_ && jeep_->driving && jeep_->spawned && globe::current().home;
    float want = on ? jeep_->squeal : 0.0f;
    X.squeal_level += (want - X.squeal_level) * (1.0f - std::exp(-dt * (want > X.squeal_level ? 14.0f : 6.0f)));
    if (X.squeal_level < 0.01f && want < 0.01f) {
        if (X.squeal_src) { audio_->source_stop(X.squeal_src); X.squeal_src = 0; }
        return;
    }
    if (!X.squeal_src) {
        if (X.squeal_clip == 0) {
            const char* path = "assets/sounds/tyre_squeal.wav";
            std::error_code ec;
            const bool made = !std::filesystem::exists(path, ec) && tyre::write_squeal_wav(path);
            X.squeal_clip = std::filesystem::exists(path, ec) ? 1 : -1;
            if (X.squeal_clip > 0) audio_->load_clip(ClipID::TyreSqueal, path);
            log::infof("[jeep] the tyres' squeal: %s%s", X.squeal_clip > 0 ? path : "could not be written - none", made ? " (made now: engine/tyre_synth.h)" : "");
        }
        if (X.squeal_clip <= 0 || !on) return;
        X.squeal_src = audio_->source_start(ClipID::TyreSqueal, sound_space::Shape{ 6.0f, 9000.0f }, true, 0.0f);
        if (!X.squeal_src) return;
    }
    if (!jeep_) return;
    const JeepState& J = *jeep_;
    const glm::vec3 pos = glm::vec3(J.st.body * glm::vec4(0.0f, 0.3f, 0.5f * (J.wheel_pos[2].z + J.wheel_pos[0].z), 1.0f));
    const float lvl = 0.55f * X.squeal_level;
    audio_->source_update(X.squeal_src, pos, J.st.velocity, lvl, 0.92f + 0.16f * X.squeal_level);
}

// The cars as obstacles for a player on foot (props.cpp, the per-tick collision list). The
// player collides with axis-aligned boxes only and a car stands at any angle, so each of a
// car's collision boxes is cut along its length into pieces about as long as it is wide, and
// each piece's own axis-aligned bound goes in: a stair-stepped outline no more than a hand
// wider than the car, instead of one bound of the whole car (which at 45 degrees is a square
// of its length). Not the car this player sits in. Until 2026-11-24 a car was nothing to a
// walker: one walked through it.
void VulkanEngine::jeep_player_colliders(std::vector<collision::AABB>& out) const {
    constexpr float kReach = 24.0f;
    // (the cars of the NPC engine too: the ones shown)
    std::vector<const JeepState*> cars;
    for (size_t i = 0; i < jeep_all_.size(); ++i) {
        const JeepState& J = *jeep_all_[i];
        if (!J.loaded || !J.posed) continue;
        if (J.driving || int(i) == jeep_ride_) continue;            // (the one this player is in)
        cars.push_back(&J);
    }
    for (const NpcCar& c : npc_cars_) if (c.agent >= 0) cars.push_back(c.car);
    for (const JeepState* jp : cars) {
        const JeepState& J = *jp;
        const glm::vec3 c(J.st.body[3]);
        if (std::abs(c.x - player_.position.x) > kReach || std::abs(c.z - player_.position.z) > kReach) continue;
        for (const auto& b : J.boxes) {
            const glm::vec3 h = b.half_extents;
            const bool along_z = h.z >= h.x;
            const float len = along_z ? h.z : h.x, wid = along_z ? h.x : h.z;
            const int n = glm::clamp(int(std::ceil(len / std::max(wid, 0.2f))), 1, 8);
            const float seg = len / float(n);
            for (int k = 0; k < n; ++k) {
                const float off = -len + seg * (2.0f * float(k) + 1.0f);
                glm::vec3 lc = b.center, lh = h;
                if (along_z) { lc.z += off; lh.z = seg; } else { lc.x += off; lh.x = seg; }
                glm::vec3 mn, mx;
                world_aabb_of_box(J.st.body * glm::translate(glm::mat4(1.0f), lc), lh, mn, mx);
                out.push_back({ mn, mx });
            }
        }
    }
}

// The cars that have no physics of their own, as bodies for what has fallen beside them
// (ragdoll.cpp corpse_obstacles; PhysicsWorld::corpse_cars_set): the parked ones and the NPC
// engine's, within `reach` of `at`. Not the driven one (it is a vehicle of the physics already).
void VulkanEngine::corpse_car_bodies(glm::vec3 at, float reach, std::vector<PhysicsWorld::CorpseCar>& out) const {
    auto add = [&](const JeepState& J, uint32_t key) {
        if (!J.loaded || !J.posed || J.boxes.empty()) return;
        const glm::vec3 c(J.st.body[3]);
        if (std::abs(c.x - at.x) > reach || std::abs(c.z - at.z) > reach || std::abs(c.y - at.y) > reach) return;
        out.push_back({ key, J.st.body, J.boxes.data(), J.boxes.size() });
    };
    for (size_t i = 0; i < jeep_all_.size(); ++i)
        if (!(jeep_all_[i]->driving && jeep_all_[i] == jeep_)) add(*jeep_all_[i], 1u + uint32_t(i));
    for (size_t i = 0; i < npc_cars_.size(); ++i)
        if (npc_cars_[i].agent >= 0) add(*npc_cars_[i].car, 1000u + uint32_t(i));
}

// ---------------------------------------------------------------------------------------
// The cars of the NPC engine (src/npc; npc_world.cpp steps it). They have no physics: a body
// matrix on the road and wheels that roll, as another machine's car is shown. There can be
// thirty of them round the course, so a few cars are made once (copies of the loaded kinds:
// the same meshes, their own places) and lent each frame to the NPC cars nearest the eye.
// One that is lent to nobody stands far under the ground.
// EVERY CAR CAN BE DRIVEN, BY EVERY PLAYER. A car of the NPC engine's that stands or barely
// rolls (npc::kTakeSpeed) - parked in a bay, at a pump, or stopped on the road because a player
// stands in its way (the cars do not drive into players on foot: npc::World::set_walkers) - is
// taken by walking up to it and getting in. Whoever sits in it is put out beside it
// (npc::World::take_car). A model taken out of the pool of lent cars is replaced by a new one
// (npc_car_add), so the traffic keeps its looks however many are taken; the only limit is the
// number of players' cars the network can name (net::kMaxCars).
// Three steps, the same alone and on a network:
//   1. whoever wants it asks (npc_take_car): the host and a machine that plays alone ask
//      themselves; a client sends Event::NpcCarTake and waits;
//   2. the authority decides (npc_host_take): npc::World::take_car lets the car go (its people
//      stay behind on foot, and later another car joins the traffic), the car gets the next
//      number in the list of players' cars, and everybody is told (Event::NpcCarTaken: which
//      agent, which number, kind, paint, where it stands, who gets in);
//   3. every machine makes it (npc_car_make): a model of that kind leaves the pool of lent
//      cars and joins jeep_all_ under that number - from then on it is a car like the others
//      (jeep_net_send / jeep_net_receive carry it). The one who asked gets in.
// Somebody who joins later is told of all of them again (npc_taken_tick).
namespace {
const glm::vec3 kNpcPaint[10] = {
    { 0.55f, 0.56f, 0.58f }, { 0.03f, 0.03f, 0.035f }, { 0.70f, 0.70f, 0.68f }, { 0.36f, 0.02f, 0.02f }, { 0.02f, 0.06f, 0.20f },
    { 0.10f, 0.11f, 0.12f }, { 0.04f, 0.14f, 0.08f }, { 0.45f, 0.32f, 0.16f }, { 0.30f, 0.33f, 0.37f }, { 0.62f, 0.40f, 0.05f } };
const glm::vec3 kPolicePaint(0.075f, 0.095f, 0.055f);      // the military police's: olive drab (THE MILITARY POLICE, below)
}

bool VulkanEngine::npc_take_car(float reach) {
    if (!npc_ || npc_cars_.empty() || jeep_all_.size() >= size_t(net::kMaxCars)) return false;
    const bool replica = npc_->replica();
    // (what a copy of the host's world can see of it: it stands, or barely rolls)
    auto stands_empty = [&](int agent) {
        if (!replica) return npc_->can_take_car(agent);
        if (agent < 0 || size_t(agent) >= npc_views_.size()) return false;
        const npc::View& v = npc_views_[size_t(agent)];
        return v.kind == npc::Kind::Car && !v.hidden && std::abs(v.speed) <= npc::kTakeSpeed;
    };
    int best = -1; float bd = reach;
    for (size_t k = 0; k < npc_cars_.size(); ++k) {
        const NpcCar& c = npc_cars_[k];
        if (c.agent < 0 || !c.car->posed || !stands_empty(c.agent)) continue;
        const float dd = glm::distance(player_.eye_position(), glm::vec3(c.car->st.body[3]) + glm::vec3(0.0f, 1.0f, 0.0f));
        if (dd < bd) { bd = dd; best = int(k); }
    }
    if (best < 0) return false;
    // (not when one of the players' own cars is nearer: that one is meant)
    for (const JeepState* c : jeep_all_)
        if (c->posed && glm::distance(player_.eye_position(), glm::vec3(c->st.body[3]) + glm::vec3(0.0f, 1.0f, 0.0f)) < bd) return false;
    const int agent = npc_cars_[size_t(best)].agent;
    if (!replica) return npc_host_take(agent, net_ ? net_->id() : 0u);
    if (!net_ || npc_take_asked_ > 0.0f) return false;
    net::Event e{};
    e.kind = net::Event::NpcCarTake; e.from = net_->id(); e.to = uint32_t(agent);
    net_->send_event(e);
    npc_take_asked_ = 0.6f;
    log::infof("[jeep] asked the host for NPC car %d", agent);
    return false;                                              // (not ours yet: the host's answer puts us in it)
}

// The authority's part. `taker`: the player who gets in (this machine's id, or a client's).
bool VulkanEngine::npc_host_take(int agent, uint32_t taker) {
    if (!npc_ || npc_->replica() || npc_cars_.empty() || jeep_all_.size() >= size_t(net::kMaxCars) || jeep_all_.size() < 4) return false;
    if (agent < 0 || size_t(agent) >= npc_->state().size() || !npc_->can_take_car(agent)) return false;
    const npc::View v = npc_->state()[size_t(agent)];          // (before it is let go: that moves it away)
    net::NpcCarTaken t;
    t.agent = uint8_t(agent); t.car = uint8_t(jeep_all_.size()); t.look = v.look; t.kind = uint8_t(v.look % 3);
    t.p[0] = v.pos.x; t.p[1] = v.pos.y; t.p[2] = v.pos.z; t.yaw = -v.yaw; t.taker = taker;
    // (the model it is shown with here says what kind it is: the look only asks for one)
    for (const NpcCar& c : npc_cars_)
        if (c.agent == agent)
            for (uint8_t k = 0; k < 3; ++k) if (jeep_all_[1 + size_t(k)]->dir == c.car->dir) t.kind = k;
    if (!npc_->take_car(agent)) return false;
    if (!npc_car_make(t)) return false;
    if (net_) net_->send_event(net::pack_taken(t));
    return true;
}

// Every machine's part: the car with that number is made of a model from the pool. True when
// it is there afterwards (made now, or known already).
bool VulkanEngine::npc_car_make(const net::NpcCarTaken& t) {
    if (!net::taken_is_sane(t, net::kMaxCars) || jeep_all_.size() < 4) return false;
    if (size_t(t.car) < jeep_all_.size()) return true;        // (told again: we have it)
    if (size_t(t.car) > jeep_all_.size()) {                    // (the ones in front of it first: npc_taken_tick)
        for (const net::NpcCarTaken& w : npc_taken_wait_) if (w.car == t.car) return false;
        npc_taken_wait_.push_back(t);
        return false;
    }
    if (npc_cars_.empty()) return false;
    const std::string& kind = jeep_all_[1 + size_t(t.kind)]->dir;
    // the model that is shown as this very car, else a free one of its kind, else one of its kind, else any
    int pick = -1, rank = 0;
    for (size_t k = 0; k < npc_cars_.size(); ++k) {
        const NpcCar& c = npc_cars_[k];
        const bool same = c.car->dir == kind;
        const int r = (same && c.agent == int(t.agent)) ? 4 : (same && c.agent < 0) ? 3 : same ? 2 : 1;
        if (r > rank) { rank = r; pick = int(k); }
    }
    JeepState* J = npc_cars_[size_t(pick)].car;
    npc_cars_.erase(npc_cars_.begin() + pick);
    npc_car_add(*jeep_all_[1 + size_t(t.kind)]);              // (the pool gets another of that kind: the traffic keeps its looks)
    const glm::vec3 paint = kNpcPaint[(t.look / 3) % 10];
    for (size_t i = 0; i < J->painted.size(); ++i)
        if (J->painted[i]) { CastleMeshGpu& g = jeep_draw_[J->draw_first + i]; g.tint = paint; g.fallback = paint; }
    J->net_owner = 0; J->net_driven = false; J->net_mine = false; J->snd_running = false; J->lights_on = false; J->brake_in = 0.0f;
    J->door_want[0] = J->door_want[1] = 0.0f; J->door_hold = 0.0f;
    // (upright where it stands: the lean and dip it was shown with are the NPC engine's)
    const glm::vec3 p(t.p[0], t.p[1], t.p[2]);
    car_settle(*J, p, t.yaw);                                  // (on its four wheels where it stands: not at the NPC engine's height, which is the lane's)
    jeep_all_.push_back(J);
    npc_taken_.push_back(t);
    // (a copy of the host's world still has the car for a moment: not shown twice)
    if (npc_ && npc_->replica() && t.agent < 100) npc_gone_[t.agent] = 3.0f;
    log::infof("[jeep] NPC car %d (%s) at (%.1f, %.1f, %.1f) is a players' car now: number %d of %zu", int(t.agent), J->dir.c_str(), p.x, p.y, p.z, int(t.car), jeep_all_.size());
    return true;
}

// Once a frame (multiplayer.cpp): the cars that waited for the ones in front of them; and a
// second after somebody joined, every machine tells of the cars it knows - where they stand
// now - so the newcomer's list of cars is the others'.
void VulkanEngine::npc_taken_tick(float dt) {
    npc_take_asked_ = std::max(0.0f, npc_take_asked_ - dt);
    for (bool again = true; again && !npc_taken_wait_.empty();) {
        again = false;
        for (size_t i = 0; i < npc_taken_wait_.size(); ++i) {
            const net::NpcCarTaken t = npc_taken_wait_[i];
            if (size_t(t.car) > jeep_all_.size()) continue;
            npc_taken_wait_.erase(npc_taken_wait_.begin() + ptrdiff_t(i));
            npc_car_make(t);
            again = true;
            break;
        }
    }
    if (!net_) return;
    const size_t now = net_->players().size();
    if (now > npc_players_seen_ && !npc_taken_.empty()) npc_taken_resend_ = 1.0f;
    npc_players_seen_ = now;
    if (npc_taken_resend_ < 0.0f) return;
    npc_taken_resend_ -= dt;
    if (npc_taken_resend_ >= 0.0f) return;
    for (net::NpcCarTaken t : npc_taken_) {
        if (size_t(t.car) >= jeep_all_.size()) continue;
        const JeepState& J = *jeep_all_[t.car];
        const glm::vec3 p(J.st.body[3]), f = -glm::vec3(J.st.body[2]);
        t.p[0] = p.x; t.p[1] = p.y; t.p[2] = p.z; t.yaw = std::atan2(-f.x, -f.z); t.taker = 0;
        net_->send_event(net::pack_taken(t));
    }
    log::infof("[jeep] told the newcomer of %zu taken NPC cars", npc_taken_.size());
}

// VIBE_TEST_NPC_CAR=<frame>[,<throttle>[,<mode>]]: the player takes a car of the NPC world
// and drives it (tests; not with VIBE_TEST_HOLD or VIBE_TEST_JEEP).
//   mode 0  at that frame the player is put at the driver's door of the nearest NPC car that
//           stands (a parked one, or one at a pump) and presses Use;
//   mode 1  at that frame the player is put on the lane of the nearest car that drives on the
//           road, 75 m ahead of it, and stands there; when a car has stopped for him he goes to
//           its door and presses Use (whoever sits in it is put out).
// From then on: <throttle> (default 0.6) straight ahead. The log says each step ("[jeep] test:").
void VulkanEngine::npc_car_test(InputFrame& in) {
    static const char* spec = std::getenv("VIBE_TEST_NPC_CAR");
    if (!spec || !npc_ || !jeep_) return;
    static int at = -1, mode = 0, phase = 0;
    static float throttle = 0.6f;
    static glm::vec3 stand(0.0f);
    if (at < 0) { at = 0; std::sscanf(spec, "%d,%f,%d", &at, &throttle, &mode); }
    const float half = game::Player::kHalfExtents.y;
    auto put = [&](glm::vec3 feet) {
        player_.position = feet + glm::vec3(0.0f, half + 0.05f, 0.0f);
        player_.velocity = glm::vec3(0.0f);
        prev_player_position_ = player_.position;
    };
    // (to the driver's door of that car of the pool, and Use)
    auto take = [&](const NpcCar& c) {
        const JeepState& J = *c.car;
        glm::vec3 door = glm::vec3(J.st.body * glm::vec4(-1.75f, 0.0f, 0.15f, 1.0f));
        door.y = glm::vec3(J.st.body[3]).y;
        put(door);
        const int agent = c.agent;
        const npc::View v = npc_views_[size_t(agent)];
        int inside = 0;
        for (const npc::View& p : npc_views_) if (p.kind == npc::Kind::Person && p.hidden && int(p.car) == agent) ++inside;
        const size_t cars = jeep_all_.size();
        const bool used = jeep_use();
        log::infof("[jeep] test: at the door of NPC car %d (pose %d, %.1f m/s, %d inside) at (%.1f, %.1f, %.1f): use -> %d, driving %d, players' cars %zu -> %zu",
                   agent, int(v.pose), double(v.speed), inside, v.pos.x, v.pos.y, v.pos.z, int(used), int(jeep_driving()), cars, jeep_all_.size());
        phase = jeep_driving() ? 2 : 3;
    };
    if (phase == 0 && int(frame_number_) >= at) {
        const glm::vec3 eye = player_.eye_position();
        const NpcCar* best = nullptr; float bd = 1e9f;
        for (const NpcCar& c : npc_cars_) {
            if (c.agent < 0 || !c.car->posed || size_t(c.agent) >= npc_views_.size()) continue;
            const npc::View& v = npc_views_[size_t(c.agent)];
            if (mode == 0 ? !(v.pose == npc::Pose::Park && std::abs(v.speed) < 0.1f) : !(v.pose == npc::Pose::Drive && v.speed > 8.0f)) continue;
            const float d = glm::distance(v.pos, eye);
            if (d < bd) { bd = d; best = &c; }
        }
        if (!best) { if (int(frame_number_) == at) log::warnf("[jeep] test: no NPC car %s near the player yet", mode == 0 ? "that stands" : "that drives"); if (int(frame_number_) > at + 1200) phase = 3; return; }
        if (mode == 0) { take(*best); return; }
        // on its lane, 75 m ahead of it
        const npc::View& v = npc_views_[size_t(best->agent)];
        float d = 0.0f, h = 0.0f, s = 0.0f;
        glm::vec3 c0; glm::vec2 dir;
        if (!highway::nearest(v.pos.x, v.pos.z, 30.0f, d, h, s) || !highway::point_along(s, c0, dir)) return;      // (not on the road just now: next frame)
        const float sgn = glm::dot(dir, glm::vec2(std::sin(v.yaw), -std::cos(v.yaw))) >= 0.0f ? 1.0f : -1.0f;
        const float len = std::max(highway::course_length(), 1.0f);
        if (!highway::point_along(std::fmod(s + sgn * 75.0f + len, len), c0, dir)) return;
        dir *= sgn;
        stand = c0 + glm::vec3(-dir.y, 0.0f, dir.x) * 1.9f;
        put(stand);
        log::infof("[jeep] test: standing on the lane at (%.1f, %.1f, %.1f), %.0f m ahead of NPC car %d that drives at %.1f m/s", stand.x, stand.y, stand.z, double(glm::distance(stand, v.pos)), best->agent, double(v.speed));
        phase = 1;
        return;
    }
    if (phase == 1) {
        put(stand);
        in.fwd = in.back = in.left = in.right = in.jump = false; in.move_x = in.move_y = 0.0f;
        for (const NpcCar& c : npc_cars_) {
            if (c.agent < 0 || !c.car->posed || size_t(c.agent) >= npc_views_.size()) continue;
            const npc::View& v = npc_views_[size_t(c.agent)];
            const float d = glm::distance(v.pos, stand);
            if (frame_number_ % 60 == 0 && d < 120.0f) log::infof("[jeep] test: NPC car %d is %.1f m from the player, %.1f m/s", c.agent, double(d), double(v.speed));
            if (d < 14.0f && std::abs(v.speed) < 0.3f) { log::infof("[jeep] test: NPC car %d stopped %.1f m from the player", c.agent, double(d)); take(c); return; }
        }
        if (int(frame_number_) > at + 2400) { log::warnf("[jeep] test: no car stopped for the player"); phase = 3; }
        return;
    }
    if (phase == 2 && jeep_driving()) {
        in.move_y = throttle; in.move_x = 0.0f;
        if (frame_number_ % 60 == 0) {
            const glm::vec3 p(jeep_->st.body[3]);
            log::infof("[jeep] test: driving the taken car (%s): at (%.1f, %.1f, %.1f), %.1f m/s", jeep_->dir.c_str(), p.x, p.y, p.z, double(glm::length(jeep_->st.velocity)));
        }
    }
}

void VulkanEngine::npc_cars_init() {
    if (jeep_all_.size() < 2 || std::getenv("VIBE_NO_NPC")) return;
    constexpr int kCars = 15;
    for (int k = 0; k < kCars; ++k)
        npc_car_add(*jeep_all_[1 + size_t(k) % std::min<size_t>(3, jeep_all_.size() - 1)]);      // sports car, saloon, pickup
    police_init();                          // (the light bars of the police cars among them)
}

// One more car without physics in the pool lent to the NPC cars: a copy of a loaded kind (the
// same meshes, its own place). Also called when one of the pool became a players' car.
void VulkanEngine::npc_car_add(const JeepState& T) {
    {
        JeepState* c = new JeepState();
        c->loaded = true; c->dir = T.dir; c->tune = T.tune;
        c->head[0] = T.head[0]; c->head[1] = T.head[1]; c->tail[0] = T.tail[0]; c->tail[1] = T.tail[1];
        c->kind = T.kind; c->painted = T.painted; c->local = T.local;
        c->hip = T.hip; c->legs = T.legs; c->rim_r = T.rim_r;
        c->pedal = T.pedal; c->foot_rest = T.foot_rest; c->floor_y = T.floor_y; c->back_deg = T.back_deg; c->has_feet = T.has_feet;
        c->n_body = T.n_body; c->n_wheel = T.n_wheel; c->n_steer = T.n_steer;
        for (int sd = 0; sd < 2; ++sd) { c->n_door[sd] = T.n_door[sd]; c->door_hinge[sd] = T.door_hinge[sd]; }
        for (int w = 0; w < 4; ++w) c->wheel_pos[w] = T.wheel_pos[w];
        c->wheel_radius = T.wheel_radius; c->wheel_width = T.wheel_width;
        c->seat = T.seat; c->steering_hub = T.steering_hub; c->steering_tilt = T.steering_tilt;
        c->boxes = T.boxes;
        c->net_owner = ~0u;                                   // (so that its brake lights work: braking())
        c->draw_first = jeep_draw_.size();
        const size_t parts = T.n_body + 4 * T.n_wheel + T.n_steer + T.n_door[0] + T.n_door[1];
        for (size_t i = 0; i < parts; ++i) { CastleMeshGpu g = jeep_draw_[T.draw_first + i]; jeep_draw_.push_back(g); }
        NpcCar nc; nc.car = c;
        npc_cars_.push_back(nc);
        jeep_park(*c, glm::vec3(0.0f, -20000.0f, 0.0f), 0.0f);
        c->posed = false;
    }
}

void VulkanEngine::npc_cars_place(const std::vector<npc::View>& views, float dt) {
    if (npc_cars_.empty()) return;
    const glm::vec3 eye = player_.eye_position();
    constexpr float kReach = 700.0f;
    // ---- which NPC cars are shown: those lent a car already (while in reach), then the nearest
    std::vector<std::pair<float, int>> want;
    for (size_t i = 0; i < views.size(); ++i)
        if (views[i].kind == npc::Kind::Car && !views[i].hidden) { const float d = glm::distance(views[i].pos, eye); if (d < kReach) want.push_back({ d, int(i) }); }
    std::sort(want.begin(), want.end());
    auto wanted = [&](int agent) { for (const auto& w : want) if (w.second == agent) return true; return false; };
    for (NpcCar& c : npc_cars_) {
        if (c.agent >= 0 && !(c.agent < int(views.size()) && wanted(c.agent))) {
            c.agent = -1;
            c.car->snd_running = false;
            jeep_park(*c.car, glm::vec3(0.0f, -20000.0f, 0.0f), 0.0f);
            c.car->posed = false;
        }
    }
    const glm::vec3* kPaint = kNpcPaint;
    for (const auto& w : want) {
        bool has = false;
        for (const NpcCar& c : npc_cars_) if (c.agent == w.second) { has = true; break; }
        if (has) continue;
        // a free car, of the kind its look asks for if there is one
        const npc::View& v = views[size_t(w.second)];
        int pick = -1;
        for (size_t k = 0; k < npc_cars_.size(); ++k) {
            if (npc_cars_[k].agent >= 0) continue;
            if (pick < 0) pick = int(k);
            if (int(k % 3) == int(v.look % 3)) { pick = int(k); break; }
        }
        if (pick < 0) break;                                   // (all lent: the far ones are not shown)
        NpcCar& c = npc_cars_[size_t(pick)];
        c.agent = w.second; c.roll = 0.0f; c.pitch = 0.0f; c.steer = 0.0f; c.last = v.pos; c.last_yaw = v.yaw; c.last_speed = v.speed;
        c.car->snow = (v.pose == npc::Pose::Drive && std::abs(v.speed) > 2.0f) ? 0.0f : 1.0f;      // (one that is under way has driven its snow off)
        c.car->posed = false;                                  // (it jumps there: no motion from where it was)
        const glm::vec3 paint = (v.style & npc::kCarPolice) ? kPolicePaint : kPaint[(v.look / 3) % 10];      // (the military police: olive drab)
        for (size_t i = 0; i < c.car->painted.size(); ++i)
            if (c.car->painted[i]) { CastleMeshGpu& g = jeep_draw_[c.car->draw_first + i]; g.tint = paint; g.fallback = paint; }
    }
    // ---- where they are
    const bool dark = cars_need_lights();
    dt = std::max(dt, 1e-4f);
    for (NpcCar& c : npc_cars_) {
        if (c.agent < 0) continue;
        const npc::View& v = views[size_t(c.agent)];
        JeepState& J = *c.car;
        // What the body does on its springs, from how the car moves: it lies on the slope it
        // climbs, leans out of a bend with the sideways pull, squats under power and dips its
        // nose under the brakes. The front wheels point where the turn says they must.
        const glm::vec3 d = v.pos - c.last;
        const float dh = glm::length(glm::vec2(d.x, d.z));
        const float dir = v.speed < 0.0f ? -1.0f : 1.0f;
        if (dh > 0.02f && dh < 5.0f) c.pitch += (std::atan2(d.y * dir, dh) - c.pitch) * std::min(1.0f, dt * 5.0f);
        float turn = v.yaw - c.last_yaw;
        while (turn > 3.14159265f) turn -= 6.2831853f;
        while (turn < -3.14159265f) turn += 6.2831853f;
        const float yaw_rate = turn / dt;
        const float wheelbase = std::abs(J.wheel_pos[2].z - J.wheel_pos[0].z);
        const float steer = std::abs(v.speed) > 0.3f ? glm::clamp(-std::atan(yaw_rate * wheelbase / v.speed), -0.6f, 0.6f) : c.steer;
        c.steer += (steer - c.steer) * std::min(1.0f, dt * 8.0f);
        c.roll = std::remainder(c.roll - v.speed / std::max(J.wheel_radius, 0.1f) * dt, 6.2831853f);
        const bool jump = !J.posed;
        const float accel = jump ? 0.0f : glm::clamp((v.speed - c.last_speed) / dt, -8.0f, 8.0f) * dir;
        const float lean_want = glm::clamp(v.speed * yaw_rate * 0.011f, -0.07f, 0.07f);
        const float dip_want = glm::clamp(accel * 0.009f, -0.035f, 0.03f);
        c.lean += (lean_want - c.lean) * std::min(1.0f, dt * 3.5f);
        c.dip += (dip_want - c.dip) * std::min(1.0f, dt * 3.0f);
        if (jump) { c.lean = 0.0f; c.dip = 0.0f; }
        const PhysicsWorld::VehicleState prev = J.st;
        // (the wheels stand on the road; the body turns about a line a little above the axles)
        // ON ITS FOUR WHEELS (2026-11-27, "NPC cars a bit inside the road or floating"): the NPC
        // engine's height is its lane's centre line - points 20 m apart, eased toward a point ahead,
        // and nothing of a bridge's deck, a paved place, the road's crown where it overtakes, a region's
        // ground; nor is it moved when the car is struck or stands with its driver shot. So within
        // kFitReach of the eye the body is laid on the ground under its four wheels (car_on_ground:
        // the same ground the simulated car stands on), height, pitch and roll; beyond, the lane's
        // height and the pitch of its way are good enough. VIBE_NPC_CAR_LANE_Y=1: as before everywhere.
        static const bool kLaneY = std::getenv("VIBE_NPC_CAR_LANE_Y") != nullptr;
        constexpr float kFitReach = 320.0f;
        glm::mat4 G;
        if (!kLaneY && glm::distance(v.pos, eye) < kFitReach) {
            static double fit_us = 0.0; static int fit_n = 0;
            const auto t0 = std::chrono::steady_clock::now();
            G = car_on_ground(J, v.pos, -v.yaw);
            fit_us += double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()) * 1e-3; ++fit_n;
            if (static const bool lg = std::getenv("VIBE_LOG_NPC_GROUND") != nullptr; lg && frame_number_ % 300 == 0 && fit_n > 0) {
                log::infof("[npc-cars] on the ground: %d cars laid in 300 frames, %.1f us each, %.3f ms a frame; car %d at (%.1f, %.2f, %.1f): the lane said y %.2f, the wheels' ground %.2f",
                           fit_n, fit_us / fit_n, fit_us / 300.0 * 1e-3, c.agent, double(v.pos.x), double(G[3].y), double(v.pos.z), double(v.pos.y), double(G[3].y));
                fit_us = 0.0; fit_n = 0;
            }
        } else {
            G = glm::translate(glm::mat4(1.0f), v.pos) * glm::rotate(glm::mat4(1.0f), -v.yaw, glm::vec3(0.0f, 1.0f, 0.0f)) *
                glm::rotate(glm::mat4(1.0f), c.pitch, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        {   // (its snow: blown off as it drives, settling again where it stands in falling snow - cars_snow_tick)
            const float sp = std::abs(v.speed);
            if (sp > 1.0f) J.snow = std::max(0.0f, J.snow - dt * std::min(sp / 20.0f, 4.0f) / 60.0f);
            else if (env_->snowfall > 0.05f) J.snow = std::min(1.0f, J.snow + dt * env_->snowfall / 180.0f);
        }
        const glm::vec3 pivot(0.0f, J.wheel_radius + 0.15f, 0.0f);
        const glm::mat4 B = G * glm::translate(glm::mat4(1.0f), pivot) * glm::rotate(glm::mat4(1.0f), c.lean, glm::vec3(0.0f, 0.0f, 1.0f)) *
                            glm::rotate(glm::mat4(1.0f), c.dip, glm::vec3(1.0f, 0.0f, 0.0f)) * glm::translate(glm::mat4(1.0f), -pivot);
        J.st.body = B;
        J.st.steer = c.steer;
        for (int w = 0; w < 4; ++w) J.st.wheel[w] = jeep_wheel_from(J, G, w, c.steer, 0.0f, c.roll);
        J.lights_on = dark && v.pose == npc::Pose::Drive;
        J.snd_running = v.pose == npc::Pose::Drive;
        // (its doors: open while somebody of its people gets in or out by them - shut again for the last half second)
        J.door_want[0] = J.door_want[1] = 0.0f;
        for (const npc::View& p : views) {
            if (p.kind != npc::Kind::Person || int(p.car) != c.agent || (p.act != npc::Act::CarEnter && p.act != npc::Act::CarExit)) continue;
            if (p.act_t > 0.05f && p.act_t < npc::kActTime[int(p.act)] - 0.55f) J.door_want[p.seat & 1] = 1.0f;
        }
        J.brake_in = (accel < -0.8f && std::abs(v.speed) > 0.2f) ? 1.0f : 0.0f;
        J.body_prev = jump ? B : prev.body;
        J.posed = true;
        jeep_place(J, jump ? J.st : prev, dt);
        c.last = v.pos; c.last_yaw = v.yaw; c.last_speed = v.speed;
    }
}

// A seat of the car lent to an NPC car (0 the driver's, 1 beside): where the hips are and the
// hands go - on the wheel, or on the knees. False when that car is not shown.
bool VulkanEngine::npc_car_seat(int agent, int seat, glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs, anim::CharacterAnimationInput::SeatExtras* more) const {
    for (const NpcCar& c : npc_cars_) {
        if (c.agent != agent || !c.car->posed) continue;
        const JeepState& J = *c.car;
        if (seat == 0) { jeep_seat(J, hip, hand_l, hand_r, legs, more); return true; }
        if (seat != 1) return false;
        const glm::vec3 s(-J.hip.x, J.hip.y, J.hip.z);
        hip = J.st.body * glm::translate(glm::mat4(1.0f), s);
        hand_l = glm::vec3(J.st.body * glm::vec4(s + glm::vec3(-0.17f, 0.14f, -0.36f), 1.0f));
        hand_r = glm::vec3(J.st.body * glm::vec4(s + glm::vec3( 0.17f, 0.14f, -0.36f), 1.0f));
        legs = J.legs;
        if (more) jeep_seat_feet(J, true, *more);
        return true;
    }
    return false;
}

// Where the cars stand (their bodies' origins): the trail in the grass is pressed under them
// (grass_trail.cpp).
void VulkanEngine::jeep_positions(std::vector<glm::vec3>& out) const {
    out.clear();
    for (const JeepState* c : jeep_all_)
        if (c->posed) out.push_back(glm::vec3(c->st.body[3]));
}

// =======================================================================================
// CARS STRIKE CARS. One car is simulated on this machine (jeep_); every other car - the
// players' cars that stand, the ones other players drive, the NPC engine's traffic - is a
// place somebody else decides. So that they are solid to each other:
//   * every other car near the simulated one is given to the physics as a body that is moved,
//     not simulated (PhysicsWorld::vehicle_set_cars): the simulated car runs into it, is
//     struck by it and bounces off it as off anything else. In multiplayer every machine does
//     this for its own car against where the others are said to be - two drivers who meet
//     each bounce off the other on their own machine;
//   * a car that is struck must move too, and whoever owns it moves it. A players' car that
//     nobody simulates (it stands where it was left) is this machine's when this machine left
//     it there, or nobody's - then the machine whose car struck it takes it over (it tells the
//     others of it from then on: jeep_net_send). It slides and turns on the ground, less and
//     less, and stops at a wall. An NPC car is the NPC engine's: the blow is handed to it
//     (npc::World::push_car) - by the host directly, by a client as Event::NpcCarPush;
//   * the blow itself: outlines on the ground plan (the car's box between its wheels), the
//     gap between two of them and the speed they close at. What is struck goes off with the
//     striker's share of that speed by weight - what a blow without bounce gives.
// And the NPC engine's traffic is told where the players' cars are (npc_feed_obstacles): it
// keeps behind them, stops for them, and goes round one left on the road when it can.
// Hooks: VIBE_LOG_CARS=1 (every blow), VIBE_NO_CAR_COLLIDE=1 (as before: nothing is solid),
// VIBE_TEST_CAR_CRASH=<frame>[,<metres>] (at that frame another car is stood in the simulated
// car's way), VIBE_NPC_NO_OBSTACLES=1 (the traffic does not see the players' cars).
void VulkanEngine::cars_collide(float dt) {
    if (!jeep_ || !physics_ || !cars_x_ || dt <= 0.0f) return;
    CarTrafficState& X = *cars_x_;
    static const bool off = std::getenv("VIBE_NO_CAR_COLLIDE") != nullptr;
    static const bool lg = std::getenv("VIBE_LOG_CARS") != nullptr;
    JeepState& me = *jeep_;
    const bool home = globe::current().home;
    const bool sim = me.spawned && me.posed && home;
    for (auto& kv : X.shove) { kv.second.cool = std::max(0.0f, kv.second.cool - dt); kv.second.claim = std::max(0.0f, kv.second.claim - dt); }
    for (float& t : X.npc_cool) t = std::max(0.0f, t - dt);
    if (off || !home) { if (me.spawned) physics_->vehicle_set_cars(nullptr, 0, dt); return; }
    const float wl = rt_.water_level;
    // (what a car stands on: the ground, or a bridge's deck over it)
    auto surface = [wl](float x, float z) {
        float h = world_ground_height(x, z, wl, 9), dh = 0.0f;
        if (highway::deck(x, z, dh)) h = std::max(h, dh);
        return h;
    };
    if (static const char* tc = std::getenv("VIBE_TEST_CAR_CRASH"); tc && sim) {
        int at = 0; float ahead = 45.0f;
        std::sscanf(tc, "%d,%f", &at, &ahead);
        if (int(frame_number_) == at && jeep_all_.size() > 1) {
            JeepState* t = jeep_all_[jeep_all_[0] == jeep_ ? 1 : 0];
            const glm::vec3 f = -glm::vec3(me.st.body[2]);
            glm::vec3 p = glm::vec3(me.st.body[3]) + glm::normalize(glm::vec3(f.x, 0.0f, f.z)) * ahead;
            p.y = surface(p.x, p.z);
            jeep_park(*t, p, std::atan2(-f.x, -f.z) + 0.6f);
            log::infof("[cars] test: %s stood %.0f m ahead of the simulated car, at (%.1f, %.1f, %.1f)", t->dir.c_str(), double(ahead), p.x, p.y, p.z);
        }
    }

    // A car's outline on the ground plan: its middle, its two axes, half its width and length.
    struct Box2 { glm::vec2 c{0.0f}, ax{1.0f, 0.0f}, az{0.0f, 1.0f}; float hx = 1.0f, hz = 2.3f; };
    auto box_of = [](const JeepState& J) {
        float hx, z0, z1; J.body_box(hx, z0, z1);
        const glm::mat4& B = J.st.body;
        const glm::vec3 c(B * glm::vec4(0.0f, 0.0f, 0.5f * (z0 + z1), 1.0f));
        Box2 b;
        b.c = glm::vec2(c.x, c.z);
        const glm::vec2 ax(B[0].x, B[0].z), az(B[2].x, B[2].z);
        if (glm::dot(ax, ax) > 1e-6f) b.ax = glm::normalize(ax);
        if (glm::dot(az, az) > 1e-6f) b.az = glm::normalize(az);
        b.hx = hx; b.hz = 0.5f * (z1 - z0);
        return b;
    };
    // How far apart two outlines are (under 0: how deep in each other) and the way from a to b.
    auto box_gap = [](const Box2& a, const Box2& b, glm::vec2& n) {
        float best = -1e9f;
        const glm::vec2 d = b.c - a.c;
        for (const glm::vec2& axis : { a.ax, a.az, b.ax, b.az }) {
            const float ra = a.hx * std::abs(glm::dot(a.ax, axis)) + a.hz * std::abs(glm::dot(a.az, axis));
            const float rb = b.hx * std::abs(glm::dot(b.ax, axis)) + b.hz * std::abs(glm::dot(b.az, axis));
            const float along = glm::dot(d, axis);
            const float sep = std::abs(along) - ra - rb;
            if (sep > best) { best = sep; n = along >= 0.0f ? axis : -axis; }
        }
        return best;
    };
    auto shoving = [&](const JeepState* c) {
        const auto it = X.shove.find(c);
        return it != X.shove.end() && (glm::dot(it->second.v, it->second.v) > 0.0025f || std::abs(it->second.w) > 0.05f);
    };

    // ---- everything that has an outline here: the players' cars, and the NPC cars that are shown
    struct Car { JeepState* J = nullptr; int agent = -1; uint32_t key = 0; glm::vec2 v{0.0f}; Box2 box; bool mover = false, target = false; float mass = 1500.0f; };
    std::vector<Car> cars;
    const glm::vec3 at = sim ? glm::vec3(me.st.body[3]) : player_.position;
    for (size_t i = 0; i < jeep_all_.size(); ++i) {
        JeepState* c = jeep_all_[i];
        if (!c->posed || !c->loaded) continue;
        Car k; k.J = c; k.key = uint32_t(i); k.box = box_of(*c); k.mass = c->tune.mass;
        const bool mine_sim = c == jeep_ && sim;
        const bool slides = shoving(c);
        if (mine_sim || c->net_owner != 0) k.v = glm::vec2(c->st.velocity.x, c->st.velocity.z);
        else if (slides) k.v = X.shove[c].v;
        k.mover = mine_sim || (c->net_owner != 0 && c->net_driven) || slides;
        // (to be moved here: nobody simulates it, and it is this machine's - or nobody's)
        k.target = !mine_sim && !c->driving && !(c == jeep_ && c->spawned) && (c->net_mine || c->net_owner == 0);
        cars.push_back(k);
    }
    for (const NpcCar& nc : npc_cars_) {
        if (nc.agent < 0 || !nc.car->posed || size_t(nc.agent) >= npc_views_.size()) continue;
        const glm::vec3 p(nc.car->st.body[3]);
        if (std::abs(p.x - at.x) > 90.0f || std::abs(p.z - at.z) > 90.0f) continue;
        const npc::View& v = npc_views_[size_t(nc.agent)];
        Car k; k.J = nc.car; k.agent = nc.agent; k.key = 1000u + uint32_t(nc.agent); k.box = box_of(*nc.car); k.mass = nc.car->tune.mass;
        k.v = glm::vec2(std::sin(v.yaw), -std::cos(v.yaw)) * v.speed;
        k.mover = std::abs(v.speed) > 0.5f;
        cars.push_back(k);
    }

    // ---- 1. the others as bodies for the simulated car
    if (sim) {
        std::vector<PhysicsWorld::CarBody> bodies;
        for (const Car& k : cars) {
            if (k.J == jeep_ || k.J->boxes.empty()) continue;
            const glm::vec3 p(k.J->st.body[3]);
            if (std::abs(p.x - at.x) > 70.0f || std::abs(p.z - at.z) > 70.0f || std::abs(p.y - at.y) > 30.0f) continue;
            PhysicsWorld::CarBody b;
            b.key = k.key; b.body = k.J->st.body; b.boxes = k.J->boxes.data(); b.n_boxes = k.J->boxes.size();
            bodies.push_back(b);
        }
        physics_->vehicle_set_cars(bodies.data(), bodies.size(), dt);
    }

    // ---- 2. who strikes whom
    const bool authority = !npc_ || !npc_->replica();
    for (size_t a = 0; a < cars.size(); ++a) {
        const Car A = cars[a];
        if (!A.mover) continue;
        const bool a_mine = A.J == jeep_ && sim;
        if (A.agent >= 0 && !authority) continue;                  // (what an NPC car strikes is the host's to say)
        for (size_t b = 0; b < cars.size(); ++b) {
            if (a == b) continue;
            Car& T = cars[b];
            const bool t_npc = T.agent >= 0;
            if (t_npc ? !a_mine : !T.target) continue;             // (an NPC car is pushed by this machine's own car only)
            if (glm::distance(A.box.c, T.box.c) > 9.0f || std::abs(A.J->st.body[3].y - T.J->st.body[3].y) > 2.5f) continue;
            glm::vec2 n(0.0f);
            const float gap = box_gap(A.box, T.box, n);
            const float closing = glm::dot(A.v - T.v, n);
            if (gap > 0.14f + std::max(closing, 0.0f) * dt * 2.0f) continue;
            // (nobody's car: the machine whose car struck it takes it over)
            const bool a_ours = a_mine || A.agent >= 0 || A.J->net_mine;
            if (!t_npc && !T.J->net_mine && !a_ours) continue;
            auto take_over = [&](CarTrafficState::Shove& S) {
                if (T.J->net_mine) return;
                T.J->net_mine = true; T.J->net_owner = 0; T.J->net_driven = false; T.J->net_send = 0.0f;
                S.claim = 1.5f;
            };
            if (closing > 0.6f) {
                if (t_npc ? X.npc_cool[size_t(T.agent) % 100] > 0.0f : X.shove[T.J].cool > 0.0f) continue;
                const glm::vec2 dv = n * (closing * A.mass / (A.mass + T.mass) * 1.05f);
                // (where it is struck: the point of its outline nearest the striker - off its middle, it turns)
                const glm::vec2 rel = A.box.c - T.box.c;
                const glm::vec2 hit = T.box.ax * std::clamp(glm::dot(rel, T.box.ax), -T.box.hx, T.box.hx) + T.box.az * std::clamp(glm::dot(rel, T.box.az), -T.box.hz, T.box.hz);
                const float spin = std::clamp((hit.y * dv.x - hit.x * dv.y) / 2.4f, -2.0f, 2.0f);      // about +y
                if (t_npc) {
                    X.npc_cool[size_t(T.agent) % 100] = 0.3f;
                    // (the NPC engine's yaw turns the other way round)
                    if (authority) { if (npc_) npc_->push_car(T.agent, glm::vec3(dv.x, 0.0f, dv.y), -spin); }
                    else if (net_) {
                        net::Event e{};
                        e.kind = net::Event::NpcCarPush; e.from = net_->id(); e.to = uint32_t(T.agent);
                        e.a[0] = dv.x; e.a[2] = dv.y; e.b[0] = -spin;
                        net_->send_event(e);
                    }
                } else {
                    CarTrafficState::Shove& S = X.shove[T.J];
                    take_over(S);
                    S.v += dv; S.w = std::clamp(S.w + spin, -2.5f, 2.5f); S.cool = 0.25f;
                    T.v = S.v;
                }
                ++X.strikes;
                if (lg) log::infof("[cars] %s%s struck %s%s at %.1f m/s (gap %.2f m): it goes off at %.1f m/s, turning %.2f rad/s", A.J->dir.c_str(), a_mine ? " (simulated here)" : A.agent >= 0 ? " (NPC)" : "",
                                   T.J->dir.c_str(), t_npc ? " (NPC)" : "", double(closing), double(gap), double(glm::length(dv)), double(spin));
            } else if (a_mine && !t_npc && gap < 0.10f && me.driving && me.throttle_in > 0.2f) {
                // (leaning on it under power: it gives way slowly)
                glm::vec3 f = -glm::vec3(me.st.body[2]);
                if (me.st.gear < 0) f = -f;
                if (glm::dot(glm::vec2(f.x, f.z), n) < 0.5f) continue;
                CarTrafficState::Shove& S = X.shove[T.J];
                take_over(S);
                if (glm::dot(S.v, n) < 1.8f) S.v += n * (9.0f * dt);
                T.v = S.v;
            }
        }
    }

    // ---- 3. the struck cars slide, less and less
    for (auto it = X.shove.begin(); it != X.shove.end();) {
        CarTrafficState::Shove& S = it->second;
        JeepState* c = nullptr;
        for (JeepState* k : jeep_all_) if (k == it->first) c = k;
        const float sp = glm::length(S.v);
        const bool simulated = c && c == jeep_ && c->spawned;
        if (!c || simulated || c->driving || c->net_owner != 0 || (sp < 0.03f && std::abs(S.w) < 0.03f)) {
            S.v = glm::vec2(0.0f); S.w = 0.0f;
            if (c && !simulated && c->net_owner == 0 && c->posed) {    // (at rest: no speed, and drawn where it stands)
                c->st.velocity = c->st.angular_velocity = glm::vec3(0.0f);
                c->body_prev = c->st.body;
                jeep_place(*c, c->st, dt);
            }
            if (S.claim <= 0.0f && S.cool <= 0.0f) { it = X.shove.erase(it); continue; }
            ++it;
            continue;
        }
        const PhysicsWorld::VehicleState prev = c->st;
        const glm::vec3 p0(prev.body[3]);
        glm::vec3 p1 = p0 + glm::vec3(S.v.x, 0.0f, S.v.y) * dt;
        // (not through a wall, a tree or a rock: it stops there)
        bool blocked = false;
        {
            float hx, z0, z1; c->body_box(hx, z0, z1);
            const glm::vec3 mid = glm::vec3(prev.body * glm::vec4(0.0f, 0.0f, 0.5f * (z0 + z1), 1.0f)) + (p1 - p0) + glm::vec3(S.v.x, 0.0f, S.v.y) * (0.9f / std::max(sp, 0.1f));
            auto in = [&](const collision::AABB& a) {
                if (a.max.y < p0.y + 0.45f || a.min.y > p0.y + 1.6f) return false;
                if (a.max.x - a.min.x > 800.0f || a.max.z - a.min.z > 800.0f) return false;
                return mid.x > a.min.x - 0.9f && mid.x < a.max.x + 0.9f && mid.z > a.min.z - 0.9f && mid.z < a.max.z + 0.9f;
            };
            for (const auto& a : wv_cols_) if (in(a)) { blocked = true; break; }
            if (!blocked) for (const auto& a : world_.aabbs) if (in(a)) { blocked = true; break; }
        }
        if (blocked) { S.v = glm::vec2(0.0f); S.w = 0.0f; p1 = p0; }
        p1.y += surface(p1.x, p1.z) - surface(p0.x, p0.z);
        const glm::mat4 M = glm::translate(glm::mat4(1.0f), p1) * glm::rotate(glm::mat4(1.0f), S.w * dt, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::translate(glm::mat4(1.0f), -p0);
        c->st.body = M * prev.body;
        for (int w = 0; w < 4; ++w) c->st.wheel[w] = M * prev.wheel[w];
        {   // (on its four wheels where it has slid to: 2026-11-27 - it kept the tilt and the height of where
            //  it was struck, and over a dip or a rise it hung in the air or sank in)
            const glm::vec3 fw = -glm::vec3(c->st.body[2]);
            c->st.body = car_on_ground(*c, p1, std::atan2(-fw.x, -fw.z));
            for (int w = 0; w < 4; ++w) c->st.wheel[w] = c->st.body * glm::translate(glm::mat4(1.0f), c->wheel_pos[w]);
        }
        c->st.velocity = glm::vec3(S.v.x, 0.0f, S.v.y);
        c->st.angular_velocity = glm::vec3(0.0f, S.w, 0.0f);
        c->body_prev = prev.body;
        jeep_place(*c, prev, dt);
        const float left = glm::length(S.v);
        S.v = left > 6.0f * dt ? S.v * ((left - 6.0f * dt) / left) : glm::vec2(0.0f);
        S.w = std::abs(S.w) > 4.0f * dt ? S.w - (S.w > 0.0f ? 4.0f : -4.0f) * dt : 0.0f;
        if (lg && frame_number_ % 30 == 0) log::infof("[cars] %s slides: at (%.1f, %.2f, %.1f), %.1f m/s", c->dir.c_str(), p1.x, p1.y, p1.z, double(left));
        ++it;
    }
    if (lg && sim && frame_number_ % 60 == 0)
        log::infof("[cars] the simulated car: %.1f m/s at (%.1f, %.1f); %zu cars with an outline here, %d blows dealt so far", double(glm::length(me.st.velocity)), at.x, at.z, cars.size(), X.strikes);
}

// (the host's) a client's car struck an NPC car.
void VulkanEngine::npc_car_push(const net::Event& e) {
    if (!npc_ || npc_->replica() || e.to >= 100u) return;
    glm::vec3 dv(e.a[0], 0.0f, e.a[2]);
    if (!std::isfinite(dv.x + dv.z + e.b[0])) return;
    const float l = glm::length(dv);
    if (l > 30.0f) dv *= 30.0f / l;
    npc_->push_car(int(e.to), dv, std::clamp(e.b[0], -2.5f, 2.5f));
}

// The players' cars on or by the road, for the NPC engine's traffic (the machine that steps it).
void VulkanEngine::npc_feed_obstacles() {
    if (!npc_ || npc_->replica()) return;
    static const bool off = std::getenv("VIBE_NPC_NO_OBSTACLES") != nullptr;
    std::vector<npc::Obstacle> obs;
    for (size_t i = 0; !off && i < jeep_all_.size(); ++i) {
        const JeepState& J = *jeep_all_[i];
        if (!J.loaded || !J.posed) continue;
        float hx, z0, z1; J.body_box(hx, z0, z1);
        const glm::vec3 c(J.st.body * glm::vec4(0.0f, 0.0f, 0.5f * (z0 + z1), 1.0f));
        if (highway::distance(c.x, c.z) > 20.0f) continue;
        npc::Obstacle o;
        o.id = uint32_t(i) + 1u; o.pos = c;
        const glm::vec3 f = -glm::vec3(J.st.body[2]);
        o.yaw = std::atan2(f.x, -f.z);
        const bool moves = (&J == jeep_ && J.spawned) || J.net_owner != 0 || (cars_x_ && cars_x_->shove.count(&J));
        if (moves && std::isfinite(J.st.velocity.x + J.st.velocity.z)) o.vel = J.st.velocity;
        o.half_w = hx; o.half_l = 0.5f * (z1 - z0);
        obs.push_back(o);
    }
    npc_->set_obstacles(obs);
}

// =======================================================================================
// THE MILITARY POLICE. Two of the NPC engine's cars are the police's (npc.cpp make_driver):
// saloons in olive drab with a light bar on the roof. They patrol with the traffic; now and
// then the siren goes on - then they drive fast, the bar flashes blue and red, and the others
// make way (all of that is the NPC engine's: View::style says which car is the police's and
// whether its siren is on, on every machine).
//   the bar    three small parts made here (a base and two lamps), drawn on the roof of the
//              model lent to a police car; the lamps are lit in turn while the siren is on
//              and throw a glow (jeep_fx);
//   the siren  engine/siren_synth.h makes the sound - wail and yelp, no recording - and
//              siren_audio_tick plays it from the car.
// Hooks: VIBE_LOG_TRAFFIC=1 (who pulls out to pass, who is back, sirens on and off, what is
// heard: each siren's distance and gain every half second), VIBE_TEST_SIREN=1 (every siren on;
// =stay: and the eye is put beside one and stays - with VIBE_TEST_HOLD), VIBE_NO_SIREN=1 (no sound), VIBE_SIREN_GAIN=<x>,
// VIBE_NPC_CAM=police / passing (the eye goes with a police car / a car that is passing).
namespace {
// The bar's three boxes in its own frame (the roof's middle at 0): middle and half size.
const glm::vec3 kBarC[3] = { { 0.0f, 0.03f, 0.0f }, { -0.30f, 0.10f, 0.0f }, { 0.30f, 0.10f, 0.0f } };
const glm::vec3 kBarH[3] = { { 0.58f, 0.03f, 0.15f }, { 0.24f, 0.05f, 0.12f }, { 0.24f, 0.05f, 0.12f } };
const glm::vec3 kBarLamp[2] = { { 0.05f, 0.16f, 1.0f }, { 1.0f, 0.05f, 0.04f } };      // blue on the left, red on the right
}

void VulkanEngine::police_init() {
    if (!cars_x_ || npc_cars_.empty()) return;
    std::vector<CastleMeshGpu> made;
    for (int p = 0; p < 3; ++p) {
        game::LevelMesh lm;
        for (int k = 0; k < 3; ++k)
            for (int s = -1; s <= 1; s += 2) {
                glm::vec3 n(0.0f), u(0.0f), v(0.0f);
                n[k] = float(s); u[(k + 1) % 3] = 1.0f; v[(k + 2) % 3] = 1.0f;
                const float hu = kBarH[p][(k + 1) % 3], hv = kBarH[p][(k + 2) % 3];
                const glm::vec3 o = kBarC[p] + n * kBarH[p][k];
                glm::vec3 a = o - u * hu - v * hv, b = o + u * hu - v * hv, c = o + u * hu + v * hv, d = o - u * hu + v * hv;
                if (glm::dot(glm::cross(b - a, d - a), n) < 0.0f) std::swap(b, d);
                const uint32_t i0 = uint32_t(lm.verts.size());
                lm.verts.push_back({ a, n, 0.0f, 0.0f }); lm.verts.push_back({ b, n, 1.0f, 0.0f });
                lm.verts.push_back({ c, n, 1.0f, 1.0f }); lm.verts.push_back({ d, n, 0.0f, 1.0f });
                lm.idx.insert(lm.idx.end(), { i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3 });
            }
        lm.tex_albedo = lm.tex_normal = -1;
        lm.tint = lm.fallback = p == 0 ? glm::vec3(0.03f, 0.03f, 0.035f) : kBarLamp[p - 1] * 0.45f;
        if (p > 0) { lm.emissive = kBarLamp[p - 1]; lm.full_emissive = true; }
        lm.relief = 0.0f;
        lm.bmin = kBarC[p] - kBarH[p]; lm.bmax = kBarC[p] + kBarH[p];
        made.push_back(make_mesh_part(lm));
    }
    build_mesh_part_blas(made);
    for (auto& g : made) jeep_owned_.push_back(g);
    const glm::mat4 away = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -20000.0f, 0.0f));
    for (int k = 0; k < 2; ++k) {
        CarTrafficState::Bar b;
        b.first = jeep_draw_.size();
        for (int p = 0; p < 3; ++p) {
            CastleMeshGpu g = made[size_t(p)];
            g.xf = g.xf_prev = away; g.bmin = g.bmax = glm::vec3(0.0f, -20000.0f, 0.0f); g.emissive = glm::vec4(0.0f);
            jeep_draw_.push_back(g);
        }
        cars_x_->bars.push_back(b);
    }
}

void VulkanEngine::police_place(float dt) {
    if (!cars_x_) return;
    CarTrafficState& X = *cars_x_;
    X.flash_t += dt;
    static const bool lg = std::getenv("VIBE_LOG_TRAFFIC") != nullptr;
    // (the log: what changed about a car since the last frame)
    for (size_t i = 0; i < npc_views_.size() && i < 100; ++i) {
        const npc::View& v = npc_views_[i];
        const uint8_t st = v.kind == npc::Kind::Car && !v.hidden ? v.style : uint8_t(0);
        const uint8_t ch = uint8_t(st ^ X.style_was[i]);
        if (lg && ch) {
            if (ch & npc::kCarOvertaking) log::infof("[traffic] car %d %s at (%.0f, %.0f), %.1f m/s%s", int(i), (st & npc::kCarOvertaking) ? "pulls out to pass" : "is back in its lane", v.pos.x, v.pos.z, double(v.speed), (st & npc::kCarPolice) ? " (police)" : "");
            if (ch & npc::kCarSiren) log::infof("[traffic] police car %d: siren %s at (%.0f, %.0f), %.0f m from the eye", int(i), (st & npc::kCarSiren) ? "ON" : "off", v.pos.x, v.pos.z, double(glm::distance(v.pos, player_.eye_position())));
        }
        X.style_was[i] = st;
    }
    if (lg && npc_ && !npc_->replica() && frame_number_ % 600 == 0) {
        const npc::World::Traffic t = npc_->traffic();
        log::infof("[traffic] on the road: %d slow, %d speeders, %d police (%d with the siren on), %d passing now; overtakes so far %d, given up %d; the fastest drives %.1f m/s",
                   t.slow, t.speeders, t.police, t.sirens, t.overtaking, t.overtakes, t.given_up, double(t.fastest));
    }
    // ---- the light bars: one on each police car that is shown
    const glm::mat4 away = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -20000.0f, 0.0f));
    std::vector<int> was;
    for (CarTrafficState::Bar& b : X.bars) { was.push_back(b.agent); b.agent = -1; b.lit[0] = b.lit[1] = false; }
    size_t k = 0;
    for (const NpcCar& nc : npc_cars_) {
        if (k >= X.bars.size()) break;
        if (nc.agent < 0 || !nc.car->posed || size_t(nc.agent) >= npc_views_.size()) continue;
        const npc::View& v = npc_views_[size_t(nc.agent)];
        if (v.kind != npc::Kind::Car || v.hidden || !(v.style & npc::kCarPolice)) continue;
        const JeepState& J = *nc.car;
        CarTrafficState::Bar& b = X.bars[k];
        b.agent = nc.agent; b.siren = (v.style & npc::kCarSiren) != 0;
        float top = 0.0f;
        for (size_t i = 0; i < J.n_body && i < J.local.size(); ++i) top = std::max(top, J.local[i].second.y);
        const glm::mat4 on_roof = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, top - 0.012f, J.seat.z + 0.12f));
        const glm::mat4 M = J.st.body * on_roof, Mp = was[k] == nc.agent ? J.body_prev * on_roof : M;
        // (the siren on: the two lamps in turn, each a double flash)
        if (b.siren) {
            const float ph = X.flash_t * 1.7f - std::floor(X.flash_t * 1.7f);
            const float sub = ph * 8.0f - std::floor(ph * 8.0f);
            b.lit[0] = ph < 0.5f && sub < 0.62f;
            b.lit[1] = ph >= 0.5f && sub < 0.62f;
        }
        for (int p = 0; p < 3; ++p) {
            CastleMeshGpu& g = jeep_draw_[b.first + size_t(p)];
            g.xf_prev = Mp; g.xf = M;
            world_aabb_of_box(M * glm::translate(glm::mat4(1.0f), kBarC[p]), kBarH[p], g.bmin, g.bmax);
            if (p > 0) g.emissive = b.lit[p - 1] ? glm::vec4(kBarLamp[p - 1] * 9.0f, 2.0f) : glm::vec4(0.0f);
        }
        for (int s = 0; s < 2; ++s) b.lamp[s] = glm::vec3(M * glm::vec4(kBarC[s + 1] + glm::vec3(0.0f, 0.06f, 0.0f), 1.0f));
        ++k;
    }
    for (size_t i = k; i < X.bars.size(); ++i)
        for (int p = 0; p < 3; ++p) {
            CastleMeshGpu& g = jeep_draw_[X.bars[i].first + size_t(p)];
            g.xf = g.xf_prev = away; g.bmin = g.bmax = glm::vec3(0.0f, -20000.0f, 0.0f); g.emissive = glm::vec4(0.0f);
        }
}

// THE SIREN'S SOUND: a source of the one spatialiser (audio.h "A SOURCE": source_start /
// source_update / source_stop), as the engines are - the distance law down to silence, the
// air's low-pass, Doppler, the side it is on, a hill in the way, the echo, the cabin. This
// function only says where each siren is, how fast it moves and how loud it is beside it.
//   whose: every police car of the NPC engine whose siren is on (View::style, the same on
//     every machine), BY ITS AGENT NUMBER and from the NPC world's own place and speed -
//     whether a car model is lent to it or not (only the nearest 15 cars within 700 m are
//     drawn; a siren carries a kilometre: sound_space::kSirenShape).
//   starts: when it comes within hearing (the gain is 0 there: out of silence) or is switched
//     on (the spatialiser brings it up in a tenth of a second).
//   stops: switched off, the car taken or gone - source_stop fades it where it was and frees
//     it; driven out of hearing - it has faded to nothing by the distance law before that.
// The clip (engine/siren_synth.h, 16 s) loops; each car begins at another place in it. It is
// written to assets/sounds/siren_wail.wav the first time it is needed and not there.
// (Before 2026-11-25 this was a tracked voice a roof-bar slot with its own law - half at 40 m,
// by the square of the distance: 41 dB down at 300 m - that only sounded while the car was
// drawn and changed car when the bars were dealt out anew.)
// Called from car_audio_tick, every frame.
void VulkanEngine::siren_audio_tick(float dt) {
    if (!cars_x_ || !audio_ || !audio_->ok() || dt <= 0.0f) return;
    CarTrafficState& X = *cars_x_;
    static const bool off = std::getenv("VIBE_NO_SIREN") != nullptr;
    static const bool lg = std::getenv("VIBE_LOG_TRAFFIC") != nullptr;
    static const float kGain = std::getenv("VIBE_SIREN_GAIN") ? float(std::atof(std::getenv("VIBE_SIREN_GAIN"))) : 1.0f;
    constexpr float kLevel = 0.55f;                     // how loud it is beside the car (1 = the clip as written)
    const sound_space::Shape S = sound_space::kSirenShape;
    const uint8_t kOn = npc::kCarPolice | npc::kCarSiren;
    auto sounds = [&](const npc::View& v) { return v.kind == npc::Kind::Car && !v.hidden && (v.style & kOn) == kOn; };
    // VIBE_TEST_SIREN=stay: once, the eye is put 12 m beside the nearest police car that drives
    // with its siren on, and stays there (with VIBE_TEST_HOLD) - the car drives away: the fade
    // read as numbers (VIBE_LOG_TRAFFIC).
    if (static const bool stay = std::getenv("VIBE_TEST_SIREN") && std::string(std::getenv("VIBE_TEST_SIREN")) == "stay"; stay && !X.siren_test_placed && frame_number_ > 30) {
        int best = -1; float bd = 1e18f;
        for (size_t i = 0; i < npc_views_.size(); ++i) {
            const npc::View& v = npc_views_[i];
            if (!sounds(v) || v.speed < 15.0f) continue;
            const float d = glm::distance(v.pos, player_.position);
            if (d < bd) { bd = d; best = int(i); }
        }
        if (best >= 0) {
            const npc::View& v = npc_views_[size_t(best)];
            player_.position = v.pos + glm::vec3(std::cos(v.yaw), 0.0f, std::sin(v.yaw)) * 12.0f + glm::vec3(0.0f, 1.2f, 0.0f);
            player_.velocity = glm::vec3(0.0f);
            prev_player_position_ = player_.position;
            globe_moved_player_ = true;                 // (VIBE_TEST_HOLD holds this place now)
            X.siren_test_placed = true;
            log::infof("[traffic] siren test: the eye stands at (%.1f, %.1f, %.1f), 12 m beside police car %d that drives %.1f m/s", player_.position.x, player_.position.y, player_.position.z, best, double(v.speed));
            return;                                     // (the listener is there from the next frame)
        }
    }
    const glm::vec3 eye = player_.eye_position();
    bool seen[npc::kMaxAgents] = {};
    if (!off && globe::current().home) {
        for (size_t i = 0; i < npc_views_.size() && i < size_t(npc::kMaxAgents); ++i) {
            const npc::View& v = npc_views_[i];
            if (!sounds(v)) continue;
            const glm::vec3 pos = v.pos + glm::vec3(0.0f, 1.5f, 0.0f);      // (the roof)
            const float d = glm::distance(pos, eye);
            auto it = X.sirens.find(int(i));
            // (an agent's number that now belongs to another car somewhere else: not this siren
            // moved - the old one fades where it was, the new one comes up where it is)
            if (it != X.sirens.end() && glm::distance(pos, it->second.pos) > 60.0f) { audio_->source_stop(it->second.src); X.sirens.erase(it); it = X.sirens.end(); }
            if (it == X.sirens.end()) {
                if (d >= S.hear_m) continue;
                if (X.siren_clip == 0) {
                    const char* path = "assets/sounds/siren_wail.wav";
                    std::error_code ec;
                    const bool made = !std::filesystem::exists(path, ec) && siren::write_siren_wav(path);
                    X.siren_clip = std::filesystem::exists(path, ec) ? 1 : -1;
                    if (X.siren_clip > 0) audio_->load_clip(ClipID::Siren, path);
                    log::infof("[traffic] the siren's sound: %s%s", X.siren_clip > 0 ? path : "could not be written - no siren", made ? " (made now: 16 s, engine/siren_synth.h)" : "");
                }
                if (X.siren_clip <= 0) continue;
                const uint32_t h = audio_->source_start(ClipID::Siren, S, true, float(i % 7) * 2.3f);
                if (!h) continue;
                CarTrafficState::Siren s; s.src = h;
                it = X.sirens.emplace(int(i), s).first;
                if (lg) log::infof("[traffic] siren of police car %d: begins to sound, %.0f m from the eye", int(i), double(d));
            } else if (d > S.hear_m + 60.0f) {          // (silent since hear_m: let go, with a margin so that it is not started and stopped at the edge)
                audio_->source_stop(it->second.src);
                X.sirens.erase(it);
                if (lg) log::infof("[traffic] siren of police car %d: out of hearing, %.0f m from the eye", int(i), double(d));
                continue;
            }
            seen[i] = true;
            CarTrafficState::Siren& s = it->second;
            s.pos = pos;
            // (its speed: the NPC engine's own, along its nose - exact, as for the engines' sound)
            const glm::vec3 vel = glm::vec3(std::sin(v.yaw), 0.0f, -std::cos(v.yaw)) * v.speed;
            const AudioEngine::Heard heard = audio_->source_update(s.src, pos, vel, kLevel * kGain, 1.0f);
            s.log_t -= dt;
            if (lg && s.log_t <= 0.0f) {
                s.log_t += 0.5f; if (s.log_t < 0.0f) s.log_t = 0.5f;
                bool drawn = false;
                for (const NpcCar& nc : npc_cars_) if (nc.agent == int(i)) { drawn = true; break; }
                log::infof("[traffic] siren of police car %d: %6.1f m from the eye, gain %.5f (%6.1f dB), low-pass %4.0f Hz, pitch x%.4f, pan %+.2f, reverb %.5f, blocked %.2f, %.1f m/s, car %s", int(i), double(heard.distance),
                           double(heard.gain), heard.gain > 0.0f ? 20.0 * std::log10(double(heard.gain)) : -999.0, double(heard.cutoff_hz), double(heard.doppler), double(heard.pan), double(heard.send),
                           double(heard.occlusion), double(v.speed), drawn ? "drawn" : "not drawn");
            }
        }
    }
    // (switched off, taken, gone, another part of the planet: it fades out where it was)
    for (auto it = X.sirens.begin(); it != X.sirens.end();) {
        if (it->first >= 0 && it->first < npc::kMaxAgents && seen[it->first]) { ++it; continue; }
        audio_->source_stop(it->second.src);
        if (lg) log::infof("[traffic] siren of police car %d: off", it->first);
        it = X.sirens.erase(it);
    }
}

}  // namespace vibe
