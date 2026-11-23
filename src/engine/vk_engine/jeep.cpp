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
//              Jump (space): handbrake. The camera follows behind and a little above; the mouse
//              or the right stick looks round it, and it swings back behind when left alone.
//
// Drawn through the static-mesh path (CastleMeshGpu with a transform), so it is lit, shadowed
// and ray traced like everything else. Home's chart only (the course lies within it).
#include "engine/vk_engine/internal.h"
#include "engine/highway.h"
#include "engine/net/net.h"
#include "engine/terrain.h"
#include "engine/globe.h"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
    float puff_acc = 0.0f;
    float test_t = 0.0f;
    float tc = 1.0f;                   // traction control: the share of the throttle let through
    // Lights: head lights on / off (L), brake lights by the pedal. Where they are (the car's
    // frame), and which drawn parts are lamps (1 head, 2 tail).
    bool      lights_on = false, prev_l = false, prev_mm = false;
    float     brake_in = 0.0f;
    glm::vec3 head[2]{ { -0.4f, 0.85f, -1.8f }, { 0.4f, 0.85f, -1.8f } }, tail[2]{ { -0.6f, 0.6f, 1.8f }, { 0.6f, 0.6f, 1.8f } };
    std::vector<uint8_t> kind;
    int       cam_mode = 1;            // the view: 0 near, 1 medium, 2 far (middle mouse button)
    float     cam_h = 1.55f;
    // The driver: where his hips are (the car's frame), how far his legs are stretched
    // (0 an upright seat .. 1), and the steering wheel's rim radius.
    glm::vec3 hip{-0.4f, 0.76f, 0.42f};
    float     legs = 0.0f, rim_r = 0.18f;
    // What is drawn: the body's parts, four wheels' and the steering wheel's - in this order
    // in jeep_draw_. `local`: each part's bounds in its own frame.
    size_t n_body = 0, n_wheel = 0, n_steer = 0;
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
    float     cam_dist = 6.4f, look_idle = 10.0f;
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
    // A mid-engined sports car: low, stiff, rear-wheel drive, a high-revving engine.
    auto sports = [](PhysicsWorld::VehicleDesc& t) {
        t.mass = 1480.0f; t.com_height = 0.40f; t.front_weight = 0.42f;
        t.front_drive = 0.0f;
        t.gears = { 3.08f, 2.19f, 1.63f, 1.29f, 1.03f, 0.84f };
        t.engine_torque = 640.0f; t.engine_max_rpm = 8400.0f; t.final_drive = 4.2f;
        t.suspension_rest = 0.20f; t.suspension_travel_up = 0.07f; t.suspension_travel_down = 0.08f;
        t.suspension_hz = 2.3f; t.suspension_damping = 0.62f; t.antiroll = 9000.0f;
        t.max_steer = 0.50f; t.brake_torque = 4200.0f; t.handbrake_torque = 6500.0f;
        t.grip = 1.30f; t.wheel_mass = 20.0f;
        t.downforce = 1.9f; t.drag = 0.45f;
    };
    // An executive saloon: heavy, quiet, soft; most of its power to the rear.
    auto saloon = [](PhysicsWorld::VehicleDesc& t) {
        t.mass = 1750.0f; t.com_height = 0.52f; t.front_weight = 0.54f;
        t.front_drive = 0.30f;
        t.gears = { 3.50f, 2.10f, 1.45f, 1.10f, 0.90f, 0.74f };
        t.engine_torque = 430.0f; t.engine_max_rpm = 6500.0f; t.final_drive = 3.5f;
        t.suspension_rest = 0.24f; t.suspension_travel_up = 0.09f; t.suspension_travel_down = 0.10f;
        t.suspension_hz = 1.5f; t.suspension_damping = 0.58f; t.antiroll = 5500.0f;
        t.max_steer = 0.55f; t.brake_torque = 3400.0f; t.handbrake_torque = 5600.0f;
        t.grip = 1.08f; t.downforce = 0.35f; t.drag = 0.38f;
    };
    // A pickup: tall, heavy at the front, long travel, four-wheel drive, a slow strong engine.
    auto pickup = [](PhysicsWorld::VehicleDesc& t) {
        t.mass = 2150.0f; t.com_height = 0.70f; t.front_weight = 0.57f;
        t.front_drive = 0.40f;
        t.gears = { 3.80f, 2.30f, 1.50f, 1.10f, 0.85f };
        t.engine_torque = 520.0f; t.engine_max_rpm = 4600.0f; t.final_drive = 3.9f;
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
    const int pick = std::getenv("VIBE_TEST_CAR") ? std::atoi(std::getenv("VIBE_TEST_CAR")) : 0;      // (tests: which one is simulated first)
    jeep_ = jeep_all_[size_t(std::clamp(pick, 0, int(jeep_all_.size()) - 1))];
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

    auto add_file = [&](const std::string& path, int copies) -> size_t {
        auto groups = game::load_mesh_groups(path.c_str());
        size_t n = 0;
        std::vector<CastleMeshGpu> made;
        std::vector<std::pair<glm::vec3, glm::vec3>> bounds;
        std::vector<uint8_t> kinds;
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
            ++n;
        }
        build_mesh_part_blas(made);
        for (auto& g : made) jeep_owned_.push_back(g);
        for (int c = 0; c < copies; ++c)
            for (size_t i = 0; i < made.size(); ++i) { jeep_draw_.push_back(made[i]); J.local.push_back(bounds[i]); J.kind.push_back(kinds[i]); }
        return n;
    };
    J.n_body = add_file(dir + "/body.mesh", 1);
    J.n_wheel = add_file(dir + "/wheel.mesh", 4);
    J.n_steer = add_file(dir + "/steering.mesh", 1);
    if (!J.n_body || !J.n_wheel) { log::warnf("[jeep] %s/*.mesh missing - no such car", dir.c_str()); jeep_draw_.resize(J.draw_first); delete jp; return nullptr; }
    J.loaded = true;
    J.lights_on = std::getenv("VIBE_TEST_JEEP_LIGHTS") != nullptr;
    J.hip = J.seat + glm::vec3(0.0f, -0.52f, 0.0f);
    J.hip = v3(json_nums(js, "hip_driver"), 0, J.hip);
    if (auto v = json_nums(js, "legs"); !v.empty()) J.legs = glm::clamp(v[0], 0.0f, 1.0f);
    if (J.n_steer) {
        float r = 0.0f;
        for (size_t i = J.n_body + 4 * J.n_wheel; i < J.local.size(); ++i)
            r = std::max({ r, std::abs(J.local[i].first.x), std::abs(J.local[i].second.x) });
        if (r > 0.08f) J.rim_r = r - 0.015f;
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
    const size_t sig = wv_cols_.size() * 31u + world_.aabbs.size();
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
        if (ob.size() > 900) ob.resize(900);
        physics_->vehicle_set_obstacles(ob.data(), ob.size());
    }
}

// Use at the jeep: in, or out. True when the key was taken.
bool VulkanEngine::jeep_use() {
    if (!jeep_ || !jeep_->spawned) return false;
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
            jeep_park(*jeep_all_[i], p, yaw);
        }
    }
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
        physics_->vehicle_input(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }
    {   // L: the head lights. The middle mouse button: the view nearer / farther.
        GLFWwindow* w = window_->handle();
        const bool l = glfwGetKey(w, GLFW_KEY_L) == GLFW_PRESS;
        if (l && !J.prev_l) J.lights_on = !J.lights_on;
        J.prev_l = l;
        const bool mm = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        if (mm && !J.prev_mm) J.cam_mode = (J.cam_mode + 1) % 3;
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
    const glm::vec3 fwd = -glm::vec3(J.st.body[2]);
    const float v_fwd

 = glm::dot(J.st.velocity, fwd), speed = glm::length(J.st.velocity);
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
    const float a_grip = 9.81f * J.tune.grip * 1.05f + J.tune.downforce * speed * speed / std::max(J.tune.mass, 1.0f);
    const float lock = std::atan(wheelbase * a_grip / std::max(speed * speed, 1.0f)) + 0.085f;
    const float limit = glm::clamp(lock / std::max(J.tune.max_steer, 0.1f), 0.06f, 1.0f);
    const float want = steer_in * limit;
    J.steer += glm::clamp(want - J.steer, -dt * 2.6f, dt * 2.6f);
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
    physics_->vehicle_input(forward, J.steer, brake, hand);
    J.brake_in = std::max(brake, hand);
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

}

// After the physics ticks: where everything is now - the parts, the driver, the camera.
void VulkanEngine::jeep_after_sim(float dt) {
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
    if (static const bool lg = std::getenv("VIBE_LOG_JEEP") != nullptr; lg && frame_number_ % 60 == 0)
        log::infof("[jeep] pos (%.1f, %.2f, %.1f) speed %.1f km/h gear %d rpm %.0f contact %d%d%d%d up %.2f driving %d | springs %.2f %.2f %.2f %.2f ground %.2f",
                   double(B[3].x), double(B[3].y), double(B[3].z), double(glm::length(J.st.velocity) * 3.6f), J.st.gear, double(J.st.rpm),
                   int(J.st.wheel_contact[0]), int(J.st.wheel_contact[1]), int(J.st.wheel_contact[2]), int(J.st.wheel_contact[3]), double(B[1].y), int(J.driving),
                   double(J.st.suspension[0]), double(J.st.suspension[1]), double(J.st.suspension[2]), double(J.st.suspension[3]),
                   double(world_ground_height(B[3].x, B[3].z, rt_.water_level, 9)));
    if (!J.driving) return;
    // -- the camera: behind and above, looking where the player looks --
    // The player's yaw / pitch are the camera's (the mouse and the stick turn them as on
    // foot). Left alone for a moment while moving, the view swings back behind the jeep.
    const glm::vec3 pos(B[3]);
    const glm::vec3 f = -glm::vec3(B[2]);
    const float car_yaw = std::atan2(f.x, -f.z);
    const float speed = glm::length(J.st.velocity);
    // As in other driving games: the view stays behind the jeep by itself and turns with it,
    // a little late. Holding the right mouse button the mouse looks round the jeep; let go
    // and the view comes back behind it.
    static float last_yaw = 0.0f, last_pitch = 0.0f;
    const float kPitch = -0.20f;
    if (!J.cam_init) { J.cam_yaw = car_yaw; J.orbit_yaw = J.orbit_pitch = 0.0f; last_yaw = player_.yaw; last_pitch = player_.pitch; }
    const float d_yaw = std::remainder(player_.yaw - last_yaw, 6.2831853f), d_pitch = player_.pitch - last_pitch;
    if (J.look_held) {
        J.orbit_yaw = std::remainder(J.orbit_yaw + d_yaw, 6.2831853f);
        J.orbit_pitch = glm::clamp(J.orbit_pitch + d_pitch, -0.9f, 0.55f);
    } else {
        J.orbit_yaw -= J.orbit_yaw * std::min(1.0f, dt * 7.0f);
        J.orbit_pitch -= J.orbit_pitch * std::min(1.0f, dt * 7.0f);
    }
    J.cam_yaw += std::remainder(car_yaw - J.cam_yaw, 6.2831853f) * std::min(1.0f, dt * 4.5f);
    // (VIBE_TEST_JEEP_ORBIT=<yaw>[,<pitch>][,<distance>]: the view held round the car - tests)
    if (static const char* ob = std::getenv("VIBE_TEST_JEEP_ORBIT")) { float d = J.cam_dist; std::sscanf(ob, "%f,%f,%f", &J.orbit_yaw, &J.orbit_pitch, &d); J.cam_dist = d; }
    player_.yaw = J.cam_yaw + J.orbit_yaw;
    player_.pitch = kPitch + J.orbit_pitch;
    last_yaw = player_.yaw; last_pitch = player_.pitch;
    const glm::vec3 look(std::sin(player_.yaw) * std::cos(player_.pitch), std::sin(player_.pitch), -std::cos(player_.yaw) * std::cos(player_.pitch));
    static const float kDist[3] = { 4.3f, 6.4f, 10.5f }, kHigh[3] = { 1.35f, 1.55f, 1.95f };
    if (!std::getenv("VIBE_TEST_JEEP_ORBIT")) J.cam_dist += (kDist[J.cam_mode] - J.cam_dist) * std::min(1.0f, dt * 6.0f);
    J.cam_h += (kHigh[J.cam_mode] - J.cam_h) * std::min(1.0f, dt * 6.0f);
    const float dist = J.cam_dist + std::min(speed * 0.05f, 2.0f);                  // (it drops back a little at speed)
    const glm::vec3 pivot = pos + glm::vec3(0.0f, J.cam_h, 0.0f);
    glm::vec3 want = pivot - look * dist;
    // (not under the ground, and not through a hill between it and the jeep)
    for (int k = 0; k < 2; ++k) {
        const float g = std::max(world_ground_height(want.x, want.z, rt_.water_level, 9), rt_.water_level) + 0.6f;
        if (want.y < g) want.y = g;
    }
    if (!J.cam_init) { J.cam_pos = want; J.cam_init = true; }
    // (the position follows stiffly: a lagging camera on a bumpy road makes one seasick.
    //  It first goes along with the car, then closes what is left: following alone it hung
    //  back by speed x a time that depends on the frame's length, so with uneven frames
    //  the car moved back and forth in the picture)
    J.cam_pos += J.st.velocity * dt;
    J.cam_pos += (want - J.cam_pos) * (1.0f - std::exp(-dt * 14.0f));
    player_.position = J.cam_pos - (player_.eye_position() - player_.position);
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
        char buf[64];
        const float kmh = glm::length(J.st.velocity) * 3.6f;
        std::snprintf(buf, sizeof buf, "%3.0f km/h", double(kmh));
        dl->AddText(fs, 44.0f * s, ImVec2(ds.x - 300.0f * s, ds.y - 150.0f * s), IM_COL32(255, 255, 255, 235), buf);
        const char* gears[] = { "R", "N", "1", "2", "3", "4", "5" };
        std::snprintf(buf, sizeof buf, "%s   %4.0f rpm", gears[std::clamp(J.st.gear + 1, 0, 6)], double(J.st.rpm));
        dl->AddText(fs, 24.0f * s, ImVec2(ds.x - 300.0f * s, ds.y - 100.0f * s), IM_COL32(255, 255, 255, 200), buf);
        const std::string k = bind_label(Action::Use) + "  get out   L  lights   MMB  view";
        dl->AddText(fs, 20.0f * s, ImVec2(ds.x - 300.0f * s, ds.y - 68.0f * s), IM_COL32(255, 255, 255, 150), k.c_str());
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
    return h;
}

// Tyre marks and smoke: where a tyre slides over the ground - sideways in a drift, or spinning
// or locked along it - it lays rubber on a hard surface and throws up smoke (dust off-road).
void VulkanEngine::jeep_marks(JeepState& J, float dt) {
    if (dt <= 0.0f) return;
    const glm::mat4& B = J.st.body;
    const glm::vec3 up = glm::normalize(glm::vec3(B[1]));
    const glm::vec3 com(B * glm::vec4(0.0f, J.tune.com_height, 0.0f, 1.0f));
    for (int w = 0; w < 4; ++w) {
        const glm::vec3 c(J.st.wheel[w][3]);
        const glm::vec3 axle = glm::normalize(glm::vec3(J.st.wheel[w][0]));
        const glm::vec3 fw = glm::normalize(glm::cross(up, axle));
        const glm::vec3 v = J.st.velocity + glm::cross(J.st.angular_velocity, c - com);
        const float v_lat = std::abs(glm::dot(v, axle)), v_long = std::abs(glm::dot(v, fw));
        const float slip = std::abs(std::abs(J.st.wheel_spin[w]) * J.wheel_radius - v_long);
        float k = 0.0f;
        if (J.st.wheel_contact[w]) k = std::max(glm::smoothstep(1.6f, 5.0f, v_lat), glm::smoothstep(2.2f, 7.0f, slip));
        const glm::vec3 g = c - up * (J.wheel_radius - 0.035f);
        const bool hard = highway::distance(g.x, g.z) < 5.7f || g.y > world_ground_height(g.x, g.z, rt_.water_level, 9) + 0.15f;
        auto& M = J.marks[w];
        if (k > 0.08f && hard) {
            if (M.empty() || M.back().a <= 0.0f || glm::distance(M.back().p, g) > 0.30f)
                M.push_back({ g, axle * (0.42f * J.wheel_width), std::min(1.0f, k * 1.25f) });
        } else if (!M.empty() && M.back().a > 0.0f) M.push_back({ g, axle * (0.42f * J.wheel_width), 0.0f });
        if (M.size() > 700) M.erase(M.begin(), M.begin() + 150);
        if (k > 0.22f) {
            J.puff_acc += dt * k * (hard ? 22.0f : 16.0f);
            while (J.puff_acc >= 1.0f) {
                J.puff_acc -= 1.0f;
                if (J.puffs.size() >= 220) break;
                JeepState::Puff p;
                p.p = g + up * 0.12f + glm::vec3(rnd01() - 0.5f, 0.0f, rnd01() - 0.5f) * 0.25f;
                p.v = v * 0.18f + glm::vec3(rnd01() - 0.5f, 0.9f + rnd01() * 0.8f, rnd01() - 0.5f) * 0.9f;
                p.age = 0.0f; p.life = (hard ? 1.9f : 1.2f) * (0.7f + 0.6f * rnd01());
                p.size = 0.28f + 0.2f * rnd01(); p.seed = rnd01() * 40.0f;
                p.bright = hard ? 1.0f : 0.62f;
                J.puffs.push_back(p);
            }
        }
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
    // Dipped: aimed a little down, wide and flat as far as one cone can be.
    pos = glm::vec4(p, 95.0f);
    dir = glm::vec4(glm::normalize(f - up * 0.055f), std::cos(glm::radians(34.0f)));
    col = glm::vec4(glm::vec3(1.0f, 0.96f, 0.88f) * 52.0f, std::cos(glm::radians(13.0f)));
    return true;
}

void VulkanEngine::jeep_fx(std::vector<Vertex>& marks, std::vector<Vertex>& puffs, std::vector<Vertex>& glow, glm::vec3 eye, glm::vec3 R, glm::vec3 U) {
    if (!globe::current().home) return;
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
    for (const JeepState* c : jeep_all_) {
        if (c->posed) {
            const glm::mat4& B = c->st.body;
            const glm::vec3 f = -glm::normalize(glm::vec3(B[2]));
            const bool braking = c->braking();
            for (int s = 0; s < 2; ++s) {
                if (c->lights_on) flare(glm::vec3(B * glm::vec4(c->head[s], 1.0f)) + f * 0.05f, f, 0.42f, glm::vec3(1.0f, 0.95f, 0.85f) * 2.4f);
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
        for (const auto& p : c->puffs) {
            const float f = p.age / p.life;
            const float a = 0.34f * std::min(1.0f, p.age * 9.0f) * (1.0f - f) * (1.0f - f);
            const glm::vec3 t(a, p.bright, p.seed);
            const glm::vec3 r = R * p.size, u = U * p.size;
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
        const bool live = &J == jeep_ && J.spawned && (J.driving || glm::length(J.st.velocity) > 0.15f);
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

// The seat of the car a player drives (another player's: multiplayer.cpp poses them in it).
bool VulkanEngine::jeep_seat_of(uint32_t player, glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs) const {
    for (const JeepState* c : jeep_all_)
        if (c->posed && c->net_owner == player && c->net_driven) { jeep_seat(*c, hip, hand_l, hand_r, legs); return true; }
    return false;
}

void VulkanEngine::jeep_seat(const JeepState& J, glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs) {
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
}

bool VulkanEngine::jeep_driver(glm::mat4& hip, glm::vec3& hand_l, glm::vec3& hand_r, float& legs) const {
    if (!jeep_ || !jeep_->driving || !jeep_->spawned) return false;
    jeep_seat(*jeep_, hip, hand_l, hand_r, legs);
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

}  // namespace vibe
