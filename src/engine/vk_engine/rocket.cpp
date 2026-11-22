// The launch site and its rocket.
//
// The site (assets/rocket/site.mesh, house.mesh - built by tools/rocket/make_rocket.py in
// Blender) stands on a terrace on the coast east of home (terrain.h kLaunchSite*): a concrete
// apron, the launch mount, the tower with its arms, a tank farm, light masts, and up the
// slope a control house with the launch button. They are static level meshes like the chapel.
//
// The rocket (rocket.mesh) is drawn with a model matrix. Pressing E at the button starts a
// ten second count; the flight itself is RocketSim (engine/rocket_sim.h): real orbital
// mechanics in double precision - up to a 180 km orbit, once round the planet, a retrograde
// burn, entry and a powered landing back on the mount. The rocket's place in the world is
// the simulation's, on the round planet: downrange it sinks below the horizon, and a
// revolution later it comes back from the other side.
//
// Far away the rocket is drawn as a proxy: on the line of sight, at a distance the depth
// buffer covers, scaled by the same ratio (so it has its true size on screen), and only
// when neither the planet nor the landscape stands in the way.
// Time runs faster while the rocket is out of sight (a real orbit takes an hour and a half).

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/audio.h"
#include "engine/log.h"
#include "engine/terrain.h"
#include "engine/globe.h"
#include "game/level.h"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace qlike {

namespace {
constexpr float kMountTop = 20.0f;                       // the mount's table above the terrace (make_rocket.py MZ)
constexpr float kHouseFloor = 0.30f;                     // (make_rocket.py F)
const glm::vec3 kButton(2.55f, kHouseFloor + 1.09f, 0.0f);   // the button's top, house-local (make_rocket.py BUTTON)
constexpr float kRocketLen = 76.0f;

struct Mat {
    const char* name;
    int tex;                 // albedo / normal slot; -1 = painted, -2 = bare steel (cube_surface_helpers.glsl)
    glm::vec3 tint;
    float tile;              // metres per texture repeat
    glm::vec3 emissive;
    bool full_emissive;
};
// Slots: 5 = Tiles130, 23 = hull steel, 24 = concrete.
const Mat kMats[] = {
    { "hull",        23, { 1.18f, 1.18f, 1.20f }, 3.6f, {}, false },
    { "dark",        -1, { 0.030f, 0.030f, 0.034f }, 1.0f, {}, false },
    { "vent",        -1, { 0.008f, 0.008f, 0.008f }, 1.0f, {}, false },
    { "fin",         -2, { 0.46f, 0.47f, 0.49f }, 1.0f, {}, false },
    { "engine",      -2, { 0.30f, 0.26f, 0.23f }, 1.0f, {}, false },
    { "engine_in",   -1, { 0.050f, 0.040f, 0.036f }, 1.0f, {}, false },
    { "text",        -1, { 0.020f, 0.020f, 0.022f }, 1.0f, {}, false },
    { "concrete",    24, { 1.00f, 1.00f, 1.00f }, 8.0f, {}, false },
    { "concrete_w",  24, { 0.96f, 0.95f, 0.93f }, 4.0f, {}, false },
    { "steel_plate", -2, { 0.16f, 0.15f, 0.15f }, 1.0f, {}, false },
    { "steel_grey",  -1, { 0.40f, 0.42f, 0.45f }, 1.0f, {}, false },
    { "steel_dark",  -1, { 0.060f, 0.064f, 0.072f }, 1.0f, {}, false },
    { "pipe",        -1, { 0.42f, 0.13f, 0.08f }, 1.0f, {}, false },
    { "white",       -1, { 0.84f, 0.85f, 0.86f }, 1.0f, {}, false },
    { "hazard",      -1, { 0.80f, 0.56f, 0.04f }, 1.0f, {}, false },
    { "paint",       -1, { 0.78f, 0.78f, 0.75f }, 1.0f, {}, false },
    { "lamp",        -1, { 1.0f, 0.96f, 0.86f }, 1.0f, { 3.4f, 3.2f, 2.7f }, true },
    { "lamp_in",     -1, { 1.0f, 0.96f, 0.88f }, 1.0f, { 5.0f, 4.7f, 4.0f }, true },
    { "lamp_red",    -1, { 1.0f, 0.10f, 0.06f }, 1.0f, { 3.4f, 0.22f, 0.14f }, true },
    { "wall_in",     -1, { 0.62f, 0.62f, 0.58f }, 1.0f, { 0.10f, 0.10f, 0.09f }, false },      // (a little of its own light: the room has no window to the sun)
    { "floor",        5, { 0.62f, 0.62f, 0.60f }, 2.0f, { 0.05f, 0.05f, 0.045f }, false },
    { "glass",       -1, { 0.035f, 0.055f, 0.070f }, 1.0f, {}, false },
    { "door",        -1, { 0.16f, 0.22f, 0.20f }, 1.0f, {}, false },
    { "roof",        24, { 0.42f, 0.42f, 0.42f }, 4.0f, {}, false },
    { "console",     -1, { 0.42f, 0.44f, 0.46f }, 1.0f, { 0.06f, 0.065f, 0.07f }, false },
    { "console_top", -1, { 0.06f, 0.06f, 0.07f }, 1.0f, {}, false },
    { "screen",      -1, { 0.10f, 0.60f, 0.35f }, 1.0f, { 0.5f, 2.6f, 1.5f }, true },
    { "button",      -1, { 0.85f, 0.04f, 0.03f }, 1.0f, { 3.6f, 0.10f, 0.06f }, true },
};
const Mat* find_mat(const std::string& n) {
    for (const auto& m : kMats) if (n == m.name) return &m;
    return nullptr;
}
// Dresses a loaded group: material, uv tile, and its place in the world.
void dress(game::LevelMesh& m, const std::string& name, glm::vec3 base) {
    const Mat* mt = find_mat(name);
    static const Mat kDefault{ "", -1, { 0.5f, 0.5f, 0.5f }, 1.0f, {}, false };
    if (!mt) mt = &kDefault;
    m.tex_albedo = m.tex_normal = mt->tex;
    m.tint = mt->tint; m.fallback = mt->tex >= 0 ? mt->tint * 0.5f : mt->tint;
    m.emissive = mt->emissive; m.full_emissive = mt->full_emissive;
    m.relief = 0.0f;
    m.bmin = glm::vec3(1e9f); m.bmax = glm::vec3(-1e9f);
    for (auto& v : m.verts) {
        v.p += base; v.u /= mt->tile; v.v /= mt->tile;
        m.bmin = glm::min(m.bmin, v.p); m.bmax = glm::max(m.bmax, v.p);
    }
}
const char* phase_name(int p) {
    switch (p) {
        case RocketSim::Ascent:  return "ASCENT";
        case RocketSim::Coast:   return "ORBIT";
        case RocketSim::Deorbit: return "DEORBIT BURN";
        case RocketSim::Entry:   return "ENTRY";
        case RocketSim::Landing: return "LANDING BURN";
        case RocketSim::Landed:  return "LANDED";
        case RocketSim::Crashed: return "LOST";
        default:                 return "ON THE PAD";
    }
}
float hash1(float n) { return glm::fract(std::sin(n * 127.1f) * 43758.5453f); }
}  // namespace

// ------------------------------------------------------------------ the site
void VulkanEngine::place_launch_site() {
    const glm::vec3 pad(kLaunchSiteX, kLaunchSiteH, kLaunchSiteZ);
    auto site = game::load_mesh_groups("assets/rocket/site.mesh");
    if (site.empty()) { log::warnf("[rocket] assets/rocket/site.mesh missing - no launch site"); return; }
    for (auto& g : site) { dress(g.mesh, g.name, pad); world_.meshes.push_back(std::move(g.mesh)); }
    auto col = [&](glm::vec3 base, glm::vec3 lo, glm::vec3 hi) { world_.aabbs.push_back({ base + lo, base + hi }); };
    // (file space is Blender's x, z, -y: the tower stands at z = -14.5)
    col(pad, { -70.0f, -2.5f, -70.0f }, { 70.0f, 0.35f, 70.0f });                       // apron
    for (int k = 0; k < 6; ++k) {                                                       // mount legs
        const float a = 6.2831853f * (float(k) + 0.5f) / 6.0f;
        const glm::vec3 c(7.8f * std::cos(a), 0.0f, -7.8f * std::sin(a));
        col(pad, c + glm::vec3(-1.6f, 0.0f, -1.6f), c + glm::vec3(1.6f, kMountTop - 3.6f, 1.6f));
    }
    for (int k = 0; k < 12; ++k) {                                                      // the table (a ring)
        const float a = 6.2831853f * float(k) / 12.0f;
        const glm::vec3 c(7.3f * std::cos(a), 0.0f, 7.3f * std::sin(a));
        col(pad, c + glm::vec3(-2.1f, kMountTop - 3.6f, -2.1f), c + glm::vec3(2.1f, kMountTop, 2.1f));
    }
    col(pad, { -5.0f, 0.0f, -19.5f }, { 5.0f, 108.6f, -9.5f });                          // tower
    for (int i = 0; i < 5; ++i) col(pad, { -61.2f + 8.2f * float(i), 0.0f, 48.8f }, { -54.8f + 8.2f * float(i), 19.0f, 55.2f });
    for (int i = 0; i < 2; ++i) col(pad, { -60.6f + 20.0f * float(i), 0.0f, 35.8f }, { -43.4f + 20.0f * float(i), 5.4f, 40.2f });
    col(pad, { -66.0f, 0.0f, 40.0f }, { -62.0f, 3.6f, 48.0f });                          // pump house
    for (float sx : { -1.0f, 1.0f }) for (float sz : { -1.0f, 1.0f })
        col(pad, { sx * 63.0f - 1.0f, 0.0f, sz * 63.0f - 1.0f }, { sx * 63.0f + 1.0f, 34.0f, sz * 63.0f + 1.0f });

    // The control house, up the slope west of the pad, its window to the rocket.
    glm::vec3 hp(kLaunchSiteX - 156.0f, 0.0f, kLaunchSiteZ - 14.0f);
    float top = -1e9f;
    for (float dx : { -3.8f, 0.0f, 3.8f }) for (float dz : { -3.3f, 0.0f, 3.3f })
        top = std::max(top, world_ground_height(hp.x + dx, hp.z + dz, rt_.water_level, 9));
    hp.y = top - 0.05f;
    auto house = game::load_mesh_groups("assets/rocket/house.mesh");
    for (auto& g : house) { dress(g.mesh, g.name, hp); world_.meshes.push_back(std::move(g.mesh)); }
    const float F = kHouseFloor, W = F + 3.0f;
    col(hp, { -3.8f, -3.0f, -3.3f }, { 3.8f, F, 3.3f });                                 // plinth + floor
    col(hp, { -3.5f, F, -3.0f }, { -3.2f, W, 3.0f });                                    // west wall
    col(hp, { 3.2f, F, -3.0f }, { 3.5f, W, 3.0f });                                      // east wall (window)
    col(hp, { -3.5f, F, -3.0f }, { 3.5f, W, -2.7f });                                    // north wall
    col(hp, { -3.5f, F, 2.7f }, { -2.5f, W, 3.0f });                                     // south wall, either side of the door
    col(hp, { -1.5f, F, 2.7f }, { 3.5f, W, 3.0f });
    col(hp, { -2.5f, F + 2.1f, 2.7f }, { -1.5f, W, 3.0f });
    col(hp, { -4.0f, W, -3.5f }, { 4.0f, W + 0.34f, 3.5f });                             // roof
    col(hp, { 2.05f, F, -2.2f }, { 3.2f, F + 0.96f, 2.2f });                             // console
    col(hp, { -3.2f, F, -2.7f }, { -2.1f, F + 1.9f, -2.15f });                           // locker
    col(hp, { -2.8f, -1.5f, 3.0f }, { -1.2f, F - 0.02f, 3.9f });                         // steps
    col(hp, { -2.8f, -1.5f, 3.9f }, { -1.2f, F - 0.17f, 4.6f });

    launch_pad_ = pad;
    launch_house_ = hp;
    launch_button_ = hp + kButton;
    launch_site_placed_ = true;
    rocket_.sim.reset(RocketSim::kPlanetR + double(pad.y) + double(kMountTop));
    log::infof("[rocket] launch site at (%.0f, %.1f, %.0f), control house at (%.0f, %.1f, %.0f)",
               pad.x, pad.y, pad.z, hp.x, hp.y, hp.z);
}

void VulkanEngine::init_rocket_meshes() {
    if (!launch_site_placed_) return;
    auto groups = game::load_mesh_groups("assets/rocket/rocket.mesh");
    size_t tris = 0;
    for (auto& g : groups) {
        dress(g.mesh, g.name, glm::vec3(0.0f));
        const game::LevelMesh& lm = g.mesh;
        if (lm.idx.empty()) continue;
        CastleMeshGpu p{};
        p.mesh = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                       reinterpret_cast<const Vertex*>(lm.verts.data()),
                                       uint32_t(lm.verts.size()), lm.idx.data(), uint32_t(lm.idx.size()));
        p.tex_albedo = lm.tex_albedo; p.tex_normal = lm.tex_normal;
        p.tint = lm.tint; p.fallback = lm.fallback;
        p.bmin = lm.bmin; p.bmax = lm.bmax;
        tris += lm.idx.size() / 3;
        rocket_parts_.push_back(p);
        rocket_part_glow_.push_back(g.name == "engine_in" ? 1 : 0);
    }
    log::infof("[rocket] vehicle: %zu mesh parts, %zu tris", rocket_parts_.size(), tris);

    // QLIKE_ROCKET_SIMTEST=1: fly the whole mission now, in numbers (the log).
    if (std::getenv("QLIKE_ROCKET_SIMTEST")) {
        RocketSim s = rocket_.sim;
        s.liftoff();
        int last = -1; double next = 0.0;
        for (long i = 0; i < 4000000 && s.phase != RocketSim::Landed && s.phase != RocketSim::Crashed; ++i) {
            s.step();
            if (int(s.phase) != last || s.t >= next) {
                log::infof("[rocket] sim t=%.0f %s alt %.2f km v %.0f m/s downrange %.1f km prop %.1f%% g %.2f",
                           s.t, phase_name(s.phase), s.alt / 1000.0, s.speed, s.downrange() / 1000.0, s.propellant() * 100.0, s.gload);
                last = int(s.phase); next = s.t + 120.0;
            }
        }
        log::infof("[rocket] sim end: %s at t=%.0f s, %.2f m from the pad", phase_name(s.phase), s.t, s.x);
    }
}

void VulkanEngine::destroy_rocket() {
    for (auto& g : rocket_parts_) {
        if (g.blas) g_rt.destroy_as(device_, g.blas, nullptr);
        if (g.blas_buf) vmaDestroyBuffer(allocator_, g.blas_buf, g.blas_alloc);
        if (g.mesh.vertex_buffer) destroy_mesh(allocator_, g.mesh);
    }
    rocket_parts_.clear();
    rocket_part_glow_.clear();
}

// ------------------------------------------------------------------ the button
bool VulkanEngine::rocket_button_in_reach() const {
    if (!launch_site_placed_) return false;
    const glm::vec3 eye = player_.eye_position();
    const glm::vec3 to = launch_button_ - eye;
    const float d = glm::length(to);
    if (d > 2.6f) return false;
    const glm::vec3 fwd(std::sin(player_.yaw) * std::cos(player_.pitch), std::sin(player_.pitch), -std::cos(player_.yaw) * std::cos(player_.pitch));
    return d < 0.6f || glm::dot(to / d, fwd) > 0.55f;
}

bool VulkanEngine::rocket_press_button() {
    if (!rocket_button_in_reach()) return false;
    if (rocket_.stage == 0 && (rocket_.sim.phase == RocketSim::Pad || rocket_.sim.phase == RocketSim::Landed)) {
        rocket_.stage = 1;
        rocket_.countdown = 10.0f;
        rocket_.beep_at = 10;
        rocket_.button_t = static_cast<float>(glfwGetTime());
        if (audio_) audio_->play_at(ClipID::Beep, launch_button_, 0.9f, 0.0f, 0.0f);
        log::info("[rocket] launch button pressed: T-10");
    }
    return true;       // (the key was meant for the button either way)
}

// ------------------------------------------------------------------ the globe: charts
// (engine/globe.h) The engine's coordinates are a flat chart of part of the planet. Home's
// chart reaches 300 km; beyond it the player is given a chart of their own, re-anchored
// under them every 40 km, and home's is taken up again on the way back.
glm::dvec3 VulkanEngine::globe_planet_pos(glm::vec3 p) const {
    return globe::to_planet(globe::current(), double(p.x), double(p.z)) * (globe::kR + double(p.y));
}

void VulkanEngine::globe_enter(const GlobeChartRef& ref) {
    const globe::Chart old = globe::current();
    const globe::Chart& nw = ref.c;
    const glm::dvec3 G = globe::to_planet(old, double(player_.position.x), double(player_.position.z));
    const glm::dmat3 fo = globe::frame_at(old, double(player_.position.x), double(player_.position.z));
    const glm::dvec2 np = globe::to_chart(nw, G);
    const glm::dmat3 fn = globe::frame_at(nw, np.x, np.y);
    // the way the player faces, kept on the planet
    const glm::dvec3 fwd = fo[0] * double(std::sin(player_.yaw)) - fo[2] * double(std::cos(player_.yaw));
    player_.yaw = float(std::atan2(glm::dot(fwd, fn[0]), -glm::dot(fwd, fn[2])));
    const glm::dvec3 vel = fo[0] * double(player_.velocity.x) + fo[2] * double(player_.velocity.z);
    player_.velocity.x = float(glm::dot(vel, fn[0])); player_.velocity.z = float(glm::dot(vel, fn[2]));
    player_.position.x = float(np.x); player_.position.z = float(np.y);
    prev_player_position_ = player_.position;
    globe::set_current(nw);
    globe_moved_player_ = true;
    wv_reset();                                          // the streamed woods are laid out in chart coordinates
    rocket_.cam_yaw = player_.yaw;
    const glm::dvec2 hz = globe::home_az(G);
    if (nw.home != old.home) log::infof("[globe] %s chart, %.0f km from home (%.0f km east, %.0f km south)", nw.home ? "home" : "far",
               glm::length(hz) * 0.001, hz.x * 0.001, hz.y * 0.001);
}

void VulkanEngine::globe_go_home() {
    if (globe::current().home) return;
    GlobeChartRef h;
    globe_enter(h);
}

void VulkanEngine::globe_update() {
    const globe::Chart& c = globe::current();
    const double x = double(player_.position.x), z = double(player_.position.z);
    if (c.home) {
        if (x * x + z * z < globe::kHomeReach * globe::kHomeReach) return;
    } else {
        const glm::dvec3 G = globe::to_planet(c, x, z);
        if (glm::length(globe::home_az(G)) < globe::kHomeBack) { globe_go_home(); return; }
        const double lx = x - c.ox();
        if (lx * lx + z * z < globe::kFarReach * globe::kFarReach) return;
    }
    GlobeChartRef nw;
    nw.c = globe::far_chart(globe::frame_at(c, x, z));
    globe_enter(nw);
}

// ------------------------------------------------------------------ the rocket camera
void VulkanEngine::rocket_toggle_camera() {
    if (!launch_site_placed_) return;
    Rocket& R = rocket_;
    if (!R.cam) {
        if (R.stage != 1 && R.stage != 2) return;        // (something to watch: from the count on)
        R.saved_pos = player_.position; R.saved_yaw = player_.yaw; R.saved_pitch = player_.pitch; R.saved_fly = fly_mode_;
        R.saved_chart = globe::current();
        R.cam = true;
        R.cam_yaw = player_.yaw; R.cam_pitch = player_.pitch;
        R.first = true;
    } else {
        R.cam = false;
        { GlobeChartRef back; back.c = R.saved_chart; globe_enter(back); }
        player_.position = R.saved_pos; prev_player_position_ = player_.position; player_.yaw = R.saved_yaw; player_.pitch = R.saved_pitch;
        player_.velocity = glm::vec3(0.0f);
        fly_mode_ = R.saved_fly;
        R.first = true;
    }
}

// ------------------------------------------------------------------ flight
void VulkanEngine::tick_rocket(float dt) {
    if (!launch_site_placed_) return;
    Rocket& R = rocket_;
    RocketSim& S = R.sim;
    const float now = static_cast<float>(glfwGetTime());

    // QLIKE_TEST_ROCKET=<s>: press the button after that many seconds (captures).
    if (static const char* tr = std::getenv("QLIKE_TEST_ROCKET"); tr && R.stage == 0 && !R.test_done && now > float(std::atof(tr))) {
        R.test_done = true;
        R.stage = 1; R.countdown = 10.0f; R.beep_at = 10;
        if (const char* skip = std::getenv("QLIKE_TEST_ROCKET_T")) R.test_skip = std::atof(skip);
    }

    if (R.stage == 1) {                                   // the count
        R.countdown -= dt;
        if (R.countdown < float(R.beep_at) - 1.0f && R.beep_at > 1) {
            --R.beep_at;
            if (audio_) audio_->play_at(ClipID::Beep, launch_button_, 0.8f, 0.0f, 0.0f);
        }
        // Ignition at T-3: the engines run up against the hold-down clamps.
        R.thr_vis = glm::clamp((3.0f - R.countdown) / 2.4f, 0.0f, 1.0f);
        if (R.countdown <= 0.0f) {
            S.liftoff();
            R.stage = 2; R.acc = 0.0; R.warp = 1.0f;
            R.px = S.x; R.py = S.y; R.pux = S.ux; R.puy = S.uy;
            R.trail.clear();
            log::info("[rocket] lift-off");
            if (std::getenv("QLIKE_TEST_ROCKET_CAM") && !R.cam) rocket_toggle_camera();
            if (R.test_skip > 0.0) { while (S.t < R.test_skip && S.phase != RocketSim::Landed && S.phase != RocketSim::Crashed) S.step(); R.px = S.x; R.py = S.y; R.pux = S.ux; R.puy = S.uy; R.test_skip = 0.0; }
        }
    } else if (R.stage == 2) {
        // Time runs faster while there is nothing to see: the climb above the air, the orbit,
        // the long fall back to the air.
        float want = 1.0f;
        switch (S.phase) {
            case RocketSim::Ascent:  want = S.alt > 75.0e3 ? 5.0f : 1.0f; break;
            case RocketSim::Coast:   want = 60.0f; break;
            case RocketSim::Deorbit: want = 6.0f; break;
            case RocketSim::Entry:   want = S.alt > 110.0e3 ? 60.0f : (S.alt > 55.0e3 ? 5.0f : 1.0f); break;
            default: break;
        }
        R.warp += (want - R.warp) * glm::clamp(dt * (want > R.warp ? 0.5f : 2.5f), 0.0f, 1.0f);
        if (std::abs(want - R.warp) < 0.02f) R.warp = want;
        R.acc += double(dt) * double(R.warp);
        int steps = 0;
        while (R.acc >= RocketSim::kDt && steps < 4000) {
            R.px = S.x; R.py = S.y; R.pux = S.ux; R.puy = S.uy;
            const int before = int(S.phase);
            S.step();
            R.acc -= RocketSim::kDt; ++steps;
            if (int(S.phase) != before)
                log::infof("[rocket] T+%.0f s: %s (alt %.1f km, %.0f m/s, %.0f km downrange, propellant %.1f%%)",
                           S.t, phase_name(S.phase), S.alt / 1000.0, S.speed, S.downrange() / 1000.0, S.propellant() * 100.0);
            if (S.phase == RocketSim::Landed || S.phase == RocketSim::Crashed) { R.acc = 0.0; break; }
        }
        // the plume follows the throttle (all 33 engines = 1; the landing burns a few)
        const float thr = float(S.throttle);
        const float tv = thr > 0.0f ? glm::clamp(std::sqrt(thr) * 1.05f, 0.25f, 1.0f) : 0.0f;
        R.thr_vis += (tv - R.thr_vis) * glm::clamp(dt * 6.0f, 0.0f, 1.0f);
        if (S.phase == RocketSim::Landed) { R.stage = 0; R.landed_t = now; R.warp = 1.0f; }
        if (S.phase == RocketSim::Crashed) {
            if (std::abs(S.x) < 20000.0 && globe::current().home) spawn_explosion(glm::vec3(R.pos), 6.0f, true);
            R.stage = 3; R.landed_t = now; R.warp = 1.0f;
        }
    } else {
        R.thr_vis += (0.0f - R.thr_vis) * glm::clamp(dt * 3.0f, 0.0f, 1.0f);
        // A lost rocket is replaced on the mount after a while.
        if (R.stage == 3 && now - R.landed_t > 20.0f) { S.reset(S.rs); R.stage = 0; }
    }

    // Where it is now (between the last two steps), on the planet: the simulation's plane is
    // the one through the pad's up and east.
    const double al = R.stage == 2 ? glm::clamp(R.acc / RocketSim::kDt, 0.0, 1.0) : 1.0;
    const double sx = R.stage == 2 ? R.px + (S.x - R.px) * al : S.x, sy = R.stage == 2 ? R.py + (S.y - R.py) * al : S.y;
    glm::dvec2 ax(R.pux + (S.ux - R.pux) * al, R.puy + (S.uy - R.puy) * al);
    if (R.stage != 2) ax = glm::dvec2(0.0, 1.0);
    ax = glm::normalize(ax);
    const globe::Chart home_chart{};
    const glm::dmat3 fp = globe::frame_at(home_chart, double(launch_pad_.x), double(launch_pad_.z));
    R.planet_pos = fp[0] * sx + fp[1] * sy;
    R.planet_axis = fp[0] * ax.x + fp[1] * ax.y;
    R.alt = R.stage == 2 ? float(S.alt) : 0.0f;
    if (R.cam) {
        // Riding along: the camera is put beside the rocket, in whatever chart the engine is in
        // (and that follows - globe_update - as the two go round the planet).
        const glm::dvec3 G = glm::normalize(R.planet_pos);
        const double alt_r = glm::length(R.planet_pos) - globe::kR;
        const glm::dvec2 cp = globe::to_chart(globe::current(), G);
        const glm::dmat3 fr = globe::frame_at(globe::current(), cp.x, cp.y);
        const glm::dvec3 lax(glm::dot(R.planet_axis, fr[0]), glm::dot(R.planet_axis, fr[1]), glm::dot(R.planet_axis, fr[2]));
        // the mouse turns the camera round the rocket
        R.cam_az += player_.yaw - R.cam_yaw;
        R.cam_el = glm::clamp(R.cam_el - (player_.pitch - R.cam_pitch), R.alt < 150.0f ? 0.03f : -1.2f, 1.35f);
        const double dist = 190.0;
        const glm::dvec3 off(std::cos(R.cam_el) * std::sin(R.cam_az) * dist, std::sin(R.cam_el) * dist, -std::cos(R.cam_el) * std::cos(R.cam_az) * dist);
        const glm::dvec3 eye_d = glm::dvec3(cp.x, alt_r, cp.y) + lax * 36.0 + off;
        const glm::vec3 look = glm::normalize(glm::vec3(-off));
        fly_mode_ = true;
        player_.velocity = glm::vec3(0.0f);
        player_.position = glm::vec3(eye_d) - (player_.eye_position() - player_.position);
        prev_player_position_ = player_.position;        // (the view is drawn between the last two physics positions)
        player_.yaw = std::atan2(look.x, -look.z);
        player_.pitch = std::asin(glm::clamp(look.y, -1.0f, 1.0f));
        globe_update();
        R.cam_yaw = player_.yaw; R.cam_pitch = player_.pitch;
        if (R.stage != 1 && R.stage != 2 && now - R.landed_t > 12.0f) rocket_toggle_camera();      // the flight is over: back to the player
    }
    // Its place in the engine's coordinates: relative to the eye, in the eye's own east / up /
    // south (the true straight line to it on the round planet - downrange it sinks below the
    // horizon, and it comes back from the other side).
    {
        const globe::Chart& gc = globe::current();
        if (&gc != R.last_chart) { R.first = true; R.last_chart = &gc; }
        const glm::vec3 pe = player_.eye_position();
        const glm::dmat3 fe = globe::frame_at(gc, double(pe.x), double(pe.z));
        const glm::dvec3 rel = R.planet_pos - fe[1] * (globe::kR + double(pe.y));
        R.prev_pos = R.pos; R.prev_axis = R.axis;
        R.pos = glm::dvec3(pe) + glm::dvec3(glm::dot(rel, fe[0]), glm::dot(rel, fe[1]), glm::dot(rel, fe[2]));
        R.axis = glm::normalize(glm::vec3(float(glm::dot(R.planet_axis, fe[0])), float(glm::dot(R.planet_axis, fe[1])), float(glm::dot(R.planet_axis, fe[2]))));
        R.rel = R.pos - glm::dvec3(pe);
    }
    if (R.first) { R.prev_pos = R.pos; R.prev_axis = R.axis; R.first = false; }
    const bool at_home = globe::current().home;

    const glm::vec3 eye = current_frame_view_.eye_pos;
    const float dist = float(glm::length(R.pos - glm::dvec3(eye)));
    const bool flying = R.stage == 2 && S.phase != RocketSim::Landed;

    // Smoke trail: the exhaust left behind in the air (below ~45 km), a sample every 0.12 s.
    R.trail_acc += dt;
    if (at_home && R.thr_vis > 0.2f && flying && R.alt < 45000.0f && R.warp < 1.5f && R.trail_acc > 0.12f) {
        R.trail_acc = 0.0f;
        const glm::dvec3 ex = R.pos - glm::dvec3(R.axis) * 6.0;
        // (how much smoke: all engines on the way up, a few for the landing)
        R.trail.push_back({ glm::vec3(ex), now, R.alt, glm::vec3(-R.axis) * (30.0f * glm::clamp(float(S.throttle) * 1.6f, 0.12f, 1.0f)) });
        if (R.trail.size() > 700) R.trail.erase(R.trail.begin(), R.trail.begin() + 100);
    }
    while (!R.trail.empty() && now - R.trail.front().t > 82.0f) R.trail.erase(R.trail.begin());

    // Clouds round the pad: while the engines burn close to the ground.
    const float near_ground = glm::clamp(1.0f - (float(R.pos.y) - (launch_pad_.y + kMountTop)) / 500.0f, 0.0f, 1.0f);
    const bool blasting = R.thr_vis > 0.05f && near_ground > 0.0f && (R.stage == 1 || flying);
    if (blasting) {
        if (R.pad_smoke_t0 < 0.0f || now - R.pad_smoke_end > 25.0f) R.pad_smoke_t0 = now;
        R.pad_smoke_end = now;
        R.pad_smoke_k = std::min(1.0f, R.pad_smoke_k + dt * 0.5f * R.thr_vis * near_ground);
    } else {
        R.pad_smoke_k = std::max(0.0f, R.pad_smoke_k - dt / 45.0f);
    }

    // Sound: the roar, by distance (none above the air), and the ground shaking near the pad.
    if (audio_) {
        const float air = 1.0f - glm::smoothstep(40000.0f, 70000.0f, R.alt);
        float vol = R.thr_vis * air * 2.4f / (1.0f + std::pow(dist / 260.0f, 1.25f));
        if (R.stage == 2 && R.warp > 1.5f) vol = 0.0f;
        vol = std::min(vol, 1.6f);
        const glm::vec3 sp = eye + glm::normalize(glm::vec3(R.pos - glm::dvec3(eye)) + glm::vec3(0.0f, 1e-3f, 0.0f)) * std::min(dist, 60.0f);
        if (vol < 0.004f) {
            for (auto& h : R.voice) if (h) { audio_->stop_voice(h, 0.6f); h = 0; }
        } else {
            for (int i = 0; i < 2; ++i) {
                if (!R.voice[i] || !audio_->voice_playing(R.voice[i])) {
                    const bool second = (i == 1 && R.voice[1] == 0);
                    R.voice[i] = audio_->play_tracked(ClipID::RocketRoar, sp, vol, 1.0f, second ? 4.0f : 0.0f);
                } else {
                    audio_->set_voice(R.voice[i], sp, vol);
                }
            }
        }
        cam_shake_ = std::max(cam_shake_, R.thr_vis * air * 0.55f / (1.0f + std::pow(dist / 180.0f, 2.0f)));
    }
}

// The rocket as the camera sees it: its transform, or `false` when it is out of sight.
bool VulkanEngine::rocket_view(glm::dvec3 pos, glm::vec3 axis, glm::vec3 eye, glm::mat4& model, float& scale, glm::vec3* rel_out) const {
    const glm::dvec3 rel = pos - glm::dvec3(eye);
    const double d = glm::length(rel);
    scale = 1.0f;
    glm::vec3 at = glm::vec3(pos);
    const double lim = 0.9 * double(object_far_);
    if (d > lim) {
        // Behind the planet? (a sphere through the sea, its centre straight below the pad)
        const glm::dvec3 c(double(eye.x), -RocketSim::kPlanetR, double(eye.z));        // (the planet's centre: straight below the eye)
        const glm::dvec3 dir = rel / d, oc = glm::dvec3(eye) - c;
        const double rsea = RocketSim::kPlanetR + double(rt_.water_level);
        const double b = glm::dot(oc, dir), cc = glm::dot(oc, oc) - rsea * rsea;
        const double disc = b * b - cc;
        if (disc > 0.0) { const double t0 = -b - std::sqrt(disc); if (t0 > 0.0 && t0 < d) return false; }
        scale = float(lim / d);
        at = eye + glm::vec3(dir * lim);
        if (rel_out) *rel_out = glm::vec3(dir * lim);
    } else if (rel_out) {
        *rel_out = glm::vec3(rel);
    }
    const glm::vec3 y = glm::normalize(axis), z(0.0f, 0.0f, 1.0f);
    const glm::vec3 x = glm::normalize(glm::cross(y, z));
    model = glm::mat4(glm::vec4(x * scale, 0.0f), glm::vec4(y * scale, 0.0f), glm::vec4(z * scale, 0.0f), glm::vec4(at, 1.0f));
    return true;
}

void VulkanEngine::draw_rocket(VkCommandBuffer cmd, const glm::mat4& vp, const glm::mat4& prev_vp,
                               bool depth_only, VkPipelineLayout shadow_layout) {
    if (rocket_parts_.empty() || rocket_.stage == 3) return;
    const Rocket& R = rocket_;
    const glm::vec3 eye = current_frame_view_.eye_pos;
    glm::mat4 model, prev_model; float s = 1.0f, ps = 1.0f;
    glm::vec3 rel(0.0f), prel(0.0f);
    if (!rocket_view(R.pos, R.axis, eye, model, s, &rel)) return;
    if (!rocket_view(R.prev_pos, R.prev_axis, eye, prev_model, ps, &prel)) { prev_model = model; prel = rel; }
    // Far from the origin (riding along in orbit) a float world position is metres coarse and
    // view-projection x model cancels badly: draw relative to the eye instead.
    const bool far_eye = !shadow_layout && glm::length(eye) > 20000.0f;
    glm::mat4 mvp_rel(1.0f), pmvp_rel(1.0f);
    if (far_eye) {
        const glm::mat4 vr = current_frame_view_.proj * glm::mat4(glm::mat3(current_frame_view_.view));
        glm::mat4 m = model, pm = prev_model;
        m[3] = glm::vec4(rel, 1.0f); pm[3] = glm::vec4(prel, 1.0f);
        mvp_rel = vr * m; pmvp_rel = vr * pm;
    }
    if (shadow_layout && s < 1.0f) return;               // (a far proxy casts no shadow here)
    if (s < 1.0f) {
        // Landscape in the way? March the line of sight over the ground the CPU knows.
        const glm::vec3 to = glm::vec3(R.pos - glm::dvec3(eye));
        const float d = glm::length(to);
        const glm::vec3 dir = to / d;
        const float reach = std::min(d, 12000.0f);
        for (float t = 60.0f; t < reach; t += 40.0f + t * 0.03f) {
            const glm::vec3 p = eye + dir * t;
            if (p.y > 400.0f && dir.y > 0.0f) break;
            if (p.y < sample_terrain_height(p.x, p.z)) return;
        }
    }
    const bool tex_on = rt_.textures_enabled;
    const float night = 1.0f - glm::smoothstep(4.0f, 16.0f, rt_.sun_pitch_deg);
    VkDeviceSize off = 0;
    for (size_t i = 0; i < rocket_parts_.size(); ++i) {
        const CastleMeshGpu& g = rocket_parts_[i];
        vkCmdBindVertexBuffers(cmd, 0, 1, &g.mesh.vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, g.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
        if (shadow_layout) {
            const glm::mat4 mvp = vp * model;
            vkCmdPushConstants(cmd, shadow_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &mvp);
        } else {
            PushConstants pc{};
            pc.mvp = far_eye ? mvp_rel : vp * model;
            pc.model = model;
            pc.prev_mvp = far_eye ? pmvp_rel : prev_vp * prev_model;
            pc.grass_params = glm::vec4(-1.0f, 0.0f, 0.0f, -1.0f);      // static-mesh mode, vertex UVs, flat
            if (!depth_only) {
                pc.color = glm::vec4(tex_on ? g.tint : g.fallback, 1.0f);
                pc.tex_params = tex_on ? glm::vec4(float(g.tex_albedo), float(g.tex_normal), 1.0f, 0.0f) : glm::vec4(-1.0f, -1.0f, 1.0f, 0.0f);
                glm::vec3 em(0.0f);
                if (rocket_part_glow_[i]) em = glm::vec3(9.0f, 5.0f, 2.2f) * R.thr_vis;             // the bells' insides
                // Entry: the air in front of the engines glows; so does the skirt.
                const float heat = R.stage == 2 ? float(glm::clamp(R.sim.heat, 0.0, 1.0)) : 0.0f;
                if (heat > 0.02f && R.sim.phase == RocketSim::Entry) em += glm::vec3(2.2f, 0.7f, 0.25f) * heat * 0.25f;
                // far away a lit rocket is a point of light: keep it visible by day and night
                if (s < 0.2f) em += glm::vec3(pc.color) * glm::mix(0.6f, 0.05f, night);
                pc.emissive = glm::vec4(em, 0.0f);
            }
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
        }
        vkCmdDrawIndexed(cmd, g.mesh.index_count, 1, 0, 0, 0);
    }
}

// Exhaust, smoke and glow, as ribbons for the FX pass (fx.cpp render_fx).
void VulkanEngine::rocket_build_fx(glm::vec3 eye, float px_ang, std::vector<RocketFxStrip>& strips,
                                   std::vector<RocketFxFlash>& flashes) const {
    if (!launch_site_placed_) return;
    const Rocket& R = rocket_;
    const float now = static_cast<float>(glfwGetTime());
    const float lim = 0.9f * object_far_;
    // A point as the camera sees it (far points are pulled in along the line of sight).
    auto see = [&](glm::dvec3 p, float& s) {
        const glm::dvec3 rel = p - glm::dvec3(eye);
        const double d = glm::length(rel);
        s = d > double(lim) ? float(double(lim) / d) : 1.0f;
        return eye + glm::vec3(rel * double(s));
    };
    auto min_w = [&](glm::vec3 p, float w, float px) { return std::max(w, glm::distance(p, eye) * px_ang * px); };

    glm::mat4 model; float rs = 1.0f;
    const bool seen = R.stage != 3 && rocket_view(R.pos, R.axis, eye, model, rs);

    // ---- the exhaust ----
    if (seen && R.thr_vis > 0.01f) {
        const float k = R.thr_vis;
        const float vac = glm::smoothstep(8000.0f, 60000.0f, R.alt);            // the plume opens out as the air thins
        const float flick = 0.9f + 0.1f * std::sin(now * 61.0f) * std::sin(now * 37.0f + 1.3f);
        float s0;
        const glm::vec3 e0 = see(R.pos - glm::dvec3(R.axis) * 1.0, s0);
        const glm::vec3 back = -R.axis;
        const float len = (70.0f + 150.0f * vac) * k * flick * s0;
        const float w0 = (4.2f + 16.0f * vac) * s0;
        auto beam = [&](float l0, float l1, float w, float inten, glm::vec3 tint) {
            RocketFxStrip st; st.kind = 1; st.tint = tint;
            const int n = 6;
            for (int i = 0; i <= n; ++i) {
                const float f = float(i) / float(n), l = l0 + (l1 - l0) * f;
                const glm::vec3 p = e0 + back * l;
                st.p.push_back(p);
                st.w.push_back(min_w(p, w * (1.0f + (0.9f + 2.0f * vac) * f) * (1.0f - 0.55f * f * f), 1.5f));
                st.a.push_back(inten * (1.0f - f) * (1.0f - f) * (i == 0 ? 0.6f : 1.0f));
            }
            strips.push_back(std::move(st));
        };
        beam(0.0f, len, w0 * 1.25f, 1.5f * k * (1.0f - 0.8f * vac), glm::vec3(1.0f, 0.42f, 0.12f));      // the orange body
        beam(0.0f, len * 0.55f, w0 * 0.75f, 2.6f * k * (1.0f - 0.7f * vac), glm::vec3(1.0f, 0.74f, 0.42f));   // the bright core
        beam(0.0f, len * 0.22f, w0 * 0.5f, 3.2f * k, glm::vec3(0.75f, 0.80f, 1.0f));                      // shock-hot at the nozzles
        // glare
        flashes.push_back({ e0 + back * (2.0f * s0), min_w(e0, (11.0f + 4.0f * vac) * s0 * (0.5f + 0.5f * k), 6.0f) * flick,
                            glm::vec3(1.0f, 0.62f, 0.30f) * (1.5f * k * (1.0f - 0.7f * vac)) });
    }
    // ---- entry: a streak of glowing air ----
    if (seen && R.stage == 2 && R.sim.phase == RocketSim::Entry && R.sim.heat > 0.02) {
        const float h = float(glm::clamp(R.sim.heat, 0.0, 1.0));
        float s0;
        const glm::vec3 e0 = see(R.pos, s0);
        const glm::vec3 vdir = glm::normalize(glm::vec3(float(R.sim.vx), float(R.sim.vy), 0.0f));
        RocketFxStrip st; st.kind = 1; st.tint = glm::vec3(1.0f, 0.55f, 0.25f);
        const float len = (400.0f + 2600.0f * h) * s0;
        for (int i = 0; i <= 8; ++i) {
            const float f = float(i) / 8.0f;
            const glm::vec3 p = e0 - vdir * (len * f) + vdir * (20.0f * s0);
            st.p.push_back(p);
            st.w.push_back(min_w(p, (9.0f + 40.0f * f) * s0, 2.0f + 3.0f * h));
            st.a.push_back(1.5f * h * (1.0f - f) * (1.0f - f));
        }
        strips.push_back(std::move(st));
        flashes.push_back({ e0 + vdir * (30.0f * s0), min_w(e0, 34.0f * s0, 5.0f + 7.0f * h), glm::vec3(1.0f, 0.7f, 0.45f) * (1.5f * h) });
    }
    // ---- the smoke trail ----
    const bool fx_home = globe::current().home;
    if (fx_home && R.trail.size() >= 2) {
        RocketFxStrip st; st.kind = 0; st.tint = glm::vec3(0.93f, 0.92f, 0.90f);
        for (const auto& tp : R.trail) {
            const float age = now - tp.t;
            const float thin = 1.0f - glm::smoothstep(20000.0f, 45000.0f, tp.alt);
            const float amt = glm::length(tp.v) / 30.0f;             // 0.12 .. 1
            // the exhaust is thrown back, slows in the air, drifts with the wind and spreads
            const glm::vec3 wp = tp.p + tp.v * (1.0f - std::exp(-age * 0.7f)) / 0.7f + wind_vel_ * (0.5f * age) +
                                 glm::vec3(hash1(tp.t * 3.1f) - 0.5f, 0.15f, hash1(tp.t * 7.7f) - 0.5f) * (age * 1.2f);
            float s;
            const glm::vec3 p = see(glm::dvec3(wp), s);
            st.p.push_back(p);
            // (about half as wide as it was: at a minute old the trail was 400 m across. And it has
            // thinned to nothing by the time its oldest samples are dropped - the tail used to be
            // cut off, still a fifth opaque, twelve seconds of trail at a time.)
            st.w.push_back(min_w(p, (6.0f + 9.0f * std::sqrt(age) + 0.25f * age) * (0.35f + 0.65f * amt) * s, 1.0f));
            const float life = glm::clamp(1.0f - age / 80.0f, 0.0f, 1.0f);
            st.a.push_back(life * life * std::min(1.0f, age * 1.5f) * (0.25f + 0.75f * thin) * (0.3f + 0.7f * amt) * 0.95f);
        }
        strips.push_back(std::move(st));
    }
    // ---- clouds round the pad ----
    if (fx_home && R.pad_smoke_k > 0.005f && R.pad_smoke_t0 >= 0.0f) {
        const float age = now - R.pad_smoke_t0;                // since the engines lit
        const float fade = R.pad_smoke_k;
        const glm::vec3 c = launch_pad_ + glm::vec3(0.0f, 1.5f, 0.0f);
        const int kJets = 12;
        for (int j = 0; j < kJets; ++j) {
            const float a = 6.2831853f * (float(j) + 0.37f * hash1(float(j) * 5.3f)) / float(kJets);
            const glm::vec3 dir(std::cos(a), 0.0f, std::sin(a));
            const float reach = (60.0f + 60.0f * hash1(float(j) + 9.0f)) * (1.0f - std::exp(-age * 0.22f)) + 14.0f;
            RocketFxStrip st; st.kind = 0; st.tint = glm::vec3(0.90f, 0.89f, 0.87f);
            const int n = 9;
            for (int i = 0; i <= n; ++i) {
                const float f = float(i) / float(n);
                const float r = 9.0f + reach * f;
                const float billow = std::sin(now * 0.7f + float(j) * 2.1f + f * 5.0f);
                const float up = (3.0f + 26.0f * f * f) * (1.0f - std::exp(-age * 0.15f)) + 2.5f * billow * f;
                float s;
                const glm::vec3 p = see(glm::dvec3(c + dir * r + glm::vec3(0.0f, up, 0.0f) + wind_vel_ * (0.15f * age * f)), s);
                st.p.push_back(p);
                st.w.push_back((7.0f + 24.0f * f + 1.5f * billow) * s * (0.6f + 0.4f * std::min(1.0f, age * 0.3f)));
                st.a.push_back(fade * (i == 0 ? 0.0f : 1.0f) * (1.0f - f * f) * 0.95f);       // (thins to nothing at the tip)
            }
            strips.push_back(std::move(st));
        }
        // the column standing over the mount
        RocketFxStrip st; st.kind = 0; st.tint = glm::vec3(0.88f, 0.87f, 0.85f);
        for (int i = 0; i <= 6; ++i) {
            const float f = float(i) / 6.0f;
            float s;
            const glm::vec3 p = see(glm::dvec3(c + glm::vec3(0.0f, f * 55.0f * (1.0f - std::exp(-age * 0.12f)), 0.0f) + wind_vel_ * (0.4f * age * f * f)), s);
            st.p.push_back(p); st.w.push_back((24.0f + 16.0f * f) * s); st.a.push_back(fade * (1.0f - f) * 0.8f);
        }
        strips.push_back(std::move(st));
    }
}

// ------------------------------------------------------------------ HUD
void VulkanEngine::draw_rocket_hud() {
    if (!launch_site_placed_) return;
    const Rocket& R = rocket_;
    const RocketSim& S = R.sim;
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    const float W = io.DisplaySize.x, H = io.DisplaySize.y;
    const float s = glm::clamp(H / 1080.0f, 0.6f, 2.5f);
    ImFont* fs = font_hud_ ? font_hud_ : ImGui::GetFont();
    ImFont* fb = font_hud_big_ ? font_hud_big_ : ImGui::GetFont();
    const ImU32 kAccent = IM_COL32(255, 150, 40, 255), kPanel = IM_COL32(10, 12, 16, 175);
    const ImU32 kDim = IM_COL32(190, 196, 205, 220), kWhite = IM_COL32(245, 247, 250, 255);
    auto text = [&](ImFont* f, float px, ImVec2 p, ImU32 col, const char* t) {
        dl->AddText(f, px, ImVec2(p.x + 0.06f * px, p.y + 0.06f * px), IM_COL32(0, 0, 0, 170), t);
        dl->AddText(f, px, p, col, t);
    };
    auto centred = [&](ImFont* f, float px, float cx, float y, ImU32 col, const char* t) {
        const ImVec2 sz = f->CalcTextSizeA(px, 1e9f, 0.0f, t);
        text(f, px, ImVec2(cx - sz.x * 0.5f, y), col, t);
    };
    const float now = static_cast<float>(glfwGetTime());
    char buf[96];

    // The prompt at the button.
    if (state_ == State::Playing && R.stage == 0 && rocket_button_in_reach()) {
        const float y = H * 0.62f;
        const char* t = "LAUNCH";
        const float px = 30.0f * s;
        const ImVec2 sz = fs->CalcTextSizeA(px, 1e9f, 0.0f, t);
        // (the key or the controller's button it is bound to: menu, Controls)
        const std::string key = bind_label(Action::Use);
        const float kw = std::max(40.0f * s, fs->CalcTextSizeA(px, 1e9f, 0.0f, key.c_str()).x + 18.0f * s);
        const float kh = 40.0f * s, x0 = W * 0.5f - (kw + 14.0f * s + sz.x) * 0.5f;
        dl->AddRectFilled(ImVec2(x0, y), ImVec2(x0 + kw, y + kh), kPanel, 6.0f * s);
        dl->AddRect(ImVec2(x0, y), ImVec2(x0 + kw, y + kh), kAccent, 6.0f * s, 0, 2.0f * s);
        centred(fs, px, x0 + kw * 0.5f, y + 4.0f * s, kWhite, key.c_str());
        text(fs, px, ImVec2(x0 + kw + 14.0f * s, y + 4.0f * s), kWhite, t);
    }
    // The count.
    if (R.stage == 1) {
        const int n = std::max(0, int(std::ceil(R.countdown)));
        std::snprintf(buf, sizeof(buf), "T-%d", n);
        const float pulse = 1.0f + 0.08f * (1.0f - glm::fract(R.countdown));
        centred(fb, 120.0f * s * pulse, W * 0.5f, H * 0.16f, R.countdown < 3.0f ? kAccent : kWhite, buf);
        centred(fs, 24.0f * s, W * 0.5f, H * 0.16f + 132.0f * s, kDim, R.countdown < 3.0f ? "IGNITION" : "LAUNCH SEQUENCE");
        centred(fs, 19.0f * s, W * 0.5f, H * 0.16f + 166.0f * s, kDim, R.cam ? "C   BACK TO THE GROUND" : "C   ROCKET CAMERA");
        return;
    }
    const bool show = R.stage == 2 || (R.stage != 1 && R.landed_t > 0.0f && now - R.landed_t < 14.0f);
    if (!show) return;

    // Mission panel (top right): phase, clock, and the flight's numbers.
    const float x1 = W - 34.0f * s, x0 = x1 - 400.0f * s, y0 = 150.0f * s, y1 = y0 + 318.0f * s;
    dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), kPanel, 4.0f * s);
    dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y0 + 3.0f * s), kAccent);
    const int ph = R.stage == 3 ? int(RocketSim::Crashed) : int(S.phase);
    text(fs, 26.0f * s, ImVec2(x0 + 18.0f * s, y0 + 14.0f * s), ph == RocketSim::Crashed ? IM_COL32(255, 70, 55, 255) : kAccent, phase_name(ph));
    const int tt = int(S.t);
    std::snprintf(buf, sizeof(buf), "T+%02d:%02d:%02d", tt / 3600, (tt / 60) % 60, tt % 60);
    text(fb, 44.0f * s, ImVec2(x0 + 18.0f * s, y0 + 44.0f * s), kWhite, buf);
    if (R.warp > 1.05f) {
        std::snprintf(buf, sizeof(buf), "TIME x%d", int(R.warp + 0.5f));
        const ImVec2 sz = fs->CalcTextSizeA(20.0f * s, 1e9f, 0.0f, buf);
        text(fs, 20.0f * s, ImVec2(x1 - 18.0f * s - sz.x, y0 + 18.0f * s), kDim, buf);
    }
    auto row = [&](int i, const char* k, const char* v) {
        const float y = y0 + 104.0f * s + float(i) * 30.0f * s;
        text(fs, 19.0f * s, ImVec2(x0 + 18.0f * s, y + 3.0f * s), kDim, k);
        const ImVec2 sz = fs->CalcTextSizeA(25.0f * s, 1e9f, 0.0f, v);
        text(fs, 25.0f * s, ImVec2(x0 + 222.0f * s - sz.x + 30.0f * s, y), kWhite, v);
    };
    if (S.alt < 10000.0) std::snprintf(buf, sizeof(buf), "%.0f m", std::max(S.alt, 0.0)); else std::snprintf(buf, sizeof(buf), "%.1f km", S.alt / 1000.0);
    row(0, "ALTITUDE", buf);
    std::snprintf(buf, sizeof(buf), "%.0f km/h", S.speed * 3.6);
    row(1, "SPEED", buf);
    const double dr = std::abs(S.downrange());
    if (dr < 10000.0) std::snprintf(buf, sizeof(buf), "%.0f m", dr); else std::snprintf(buf, sizeof(buf), "%.0f km", dr / 1000.0);
    row(2, "FROM THE PAD", buf);
    std::snprintf(buf, sizeof(buf), "%.1f g", S.gload);
    row(3, "LOAD", buf);
    std::snprintf(buf, sizeof(buf), "%.1f %%", S.propellant() * 100.0);
    row(4, "PROPELLANT", buf);
    std::snprintf(buf, sizeof(buf), "%d %%", int(S.throttle * 100.0 + 0.5));
    row(5, "THRUST", buf);
    // The planet and where the rocket is round it (the pad on top).
    {
        const float cx = x1 - 70.0f * s, cy = y0 + 196.0f * s, pr = 44.0f * s;
        dl->AddCircleFilled(ImVec2(cx, cy), pr, IM_COL32(28, 52, 84, 230), 48);
        dl->AddCircle(ImVec2(cx, cy), pr, IM_COL32(120, 170, 220, 200), 48, 1.5f * s);
        dl->AddCircle(ImVec2(cx, cy), pr * float((RocketSim::kPlanetR + RocketSim::kOrbitAlt) / RocketSim::kPlanetR) + 3.0f * s,
                      IM_COL32(255, 255, 255, 50), 48, 1.0f * s);
        dl->AddTriangleFilled(ImVec2(cx, cy - pr - 1.0f * s), ImVec2(cx - 4.0f * s, cy - pr - 9.0f * s), ImVec2(cx + 4.0f * s, cy - pr - 9.0f * s), kWhite);
        const float a = float(S.angle());
        const float rr = pr * float((RocketSim::kPlanetR + std::max(S.alt, 0.0)) / RocketSim::kPlanetR) + 3.0f * s * float(std::min(S.alt / RocketSim::kOrbitAlt, 1.0));
        dl->AddCircleFilled(ImVec2(cx + std::sin(a) * rr, cy - std::cos(a) * rr), 4.0f * s, kAccent, 12);
    }
    if (R.stage == 0 && S.phase == RocketSim::Landed)
        centred(fs, 22.0f * s, (x0 + x1) * 0.5f, y1 - 34.0f * s, kWhite, "BACK ON THE MOUNT");
    else if (R.stage == 2)
        centred(fs, 19.0f * s, (x0 + x1) * 0.5f, y1 - 30.0f * s, kDim, R.cam ? "C   BACK TO THE GROUND" : "C   ROCKET CAMERA");
}

}  // namespace qlike
