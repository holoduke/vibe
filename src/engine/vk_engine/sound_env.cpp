// THE SURROUNDINGS, FOR THE EAR (docs/ARCHITECTURE.md "Sound in space").
//
// Two things the audio engine cannot know by itself, found here on the game thread with the
// tests the game already has - no ray tracing, nothing on the audio thread:
//
//   sound_env_tick   four times a second: how much comes back where the listener stands
//                    (sound_space::Space -> AudioEngine::set_space, which follows it smoothly).
//                      the land: its height in 8 directions at 6 distances (10 .. 160 m) - how
//                        much of the horizon stands above the ear (a valley, a cutting), and
//                        the nearest steep face (a cliff: one distinct echo, from its side);
//                      walls: the level's big collision boxes within 60 m - 8 rays of 40 m from
//                        the ear (between buildings: many hit; the nearest wall beyond 8 m
//                        gives the echo if no cliff does);
//                      a roof: a shelter over the ear (rain's own test). With walls on most
//                        sides it is a room - what is outside is quieter and dull; without, a
//                        canopy - early reflections only;
//                      the stone bridge over the ear; the woods' density round it.
//   sound_occlusion  for one source, about five times a second (the audio engine asks): is a
//                    hill or a wall in the line from the ear to it. 0 clear .. 1 blocked.
//
// The numbers (how much each adds) are in this file's kXxx constants, set by reasoning and
// the log (VIBE_LOG_SOUND_SPACE=1), not by ear.
#include "engine/vk_engine.h"
#include "engine/vk_engine/state/sound_env.h"

#include "engine/audio.h"
#include "engine/log.h"
#include "engine/terrain.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace vibe {

namespace {

constexpr float kLookEveryS   = 0.25f;
constexpr float kWallReach    = 250.0f;     // walls are gathered this far round the ear (and again once it has moved 40 m)
constexpr float kOpenAmount   = 0.03f;      // an open field: almost nothing comes back
constexpr float kLandAmount   = 0.55f;      // ... a closed valley adds this
constexpr float kWallAmount   = 0.45f;      // ... walls all round
constexpr float kWoodAmount   = 0.28f;      // ... closed woods
constexpr float kBridgeAmount = 0.70f;      // ... under the bridge
constexpr float kHillFullM    = 4.0f;       // this much hill above the line: fully blocked
constexpr float kWallBlock    = 0.85f;      // a wall in the line

// Where a segment a -> b enters a box (0..1 along it), or -1.
float segment_box(glm::vec3 a, glm::vec3 b, const collision::AABB& box) {
    float t0 = 0.0f, t1 = 1.0f;
    for (int k = 0; k < 3; ++k) {
        const float d = b[k] - a[k];
        if (std::abs(d) < 1.0e-6f) { if (a[k] < box.min[k] || a[k] > box.max[k]) return -1.0f; continue; }
        float u0 = (box.min[k] - a[k]) / d, u1 = (box.max[k] - a[k]) / d;
        if (u0 > u1) std::swap(u0, u1);
        t0 = (std::max)(t0, u0); t1 = (std::min)(t1, u1);
        if (t0 > t1) return -1.0f;
    }
    return t0;
}

bool inside(glm::vec3 p, const collision::AABB& b) {
    return p.x > b.min.x && p.x < b.max.x && p.y > b.min.y && p.y < b.max.y && p.z > b.min.z && p.z < b.max.z;
}

}  // namespace

float VulkanEngine::sound_occlusion(glm::vec3 ear, glm::vec3 source) const {
    // the land: the line over the height field (as a gun's report is tested, fx.cpp)
    const glm::vec3 b = source + glm::vec3(0.0f, 0.4f, 0.0f);
    float over = 0.0f;
    constexpr int n = 10;
    for (int i = 1; i < n; ++i) {
        const glm::vec3 p = glm::mix(ear, b, float(i) / float(n));
        over = (std::max)(over, sample_terrain_height(p.x, p.z) - p.y);
    }
    float occ = std::clamp(over / kHillFullM, 0.0f, 1.0f);
    // walls: a big box in the line that neither end is in (not from inside a room: its own
    // walls are in every line, and what they do is the room's - Space::outside_gain, outside_hz)
    if (sound_env_ && !sound_env_->room && occ < kWallBlock)
        for (const collision::AABB& w : sound_env_->walls) {
            if (inside(ear, w) || inside(b, w)) continue;
            if (segment_box(ear, b, w) >= 0.0f) { occ = kWallBlock; break; }
        }
    return occ;
}

void VulkanEngine::sound_env_tick(float dt, bool cabin) {
    if (!audio_ || !audio_->ok() || !sound_env_) return;
    SoundEnvState& E = *sound_env_;
    if (!E.probe_set) {
        E.probe_set = true;
        audio_->set_occlusion_probe([this](glm::vec3 ear, glm::vec3 source) { return sound_occlusion(ear, source); });
    }
    E.wait -= dt;
    if (E.wait > 0.0f && cabin == E.cabin) return;        // (getting in or out of a car is heard at once)
    E.wait = kLookEveryS;
    E.cabin = cabin;
    const glm::vec3 ear = player_.eye_position();
    const bool home = globe::current().home;

    // The walls near: the level's boxes that are big enough to be one (at least 3 m long and
    // 2 m high - not a trunk, a kerb, a parapet, a table).
    if (glm::distance(ear, E.walls_at) > 40.0f) {
        E.walls_at = ear;
        E.walls.clear();
        if (home)
            for (const collision::AABB& a : world_.aabbs) {
                const glm::vec3 size = a.max - a.min;
                if (size.y < 2.0f || (std::max)(size.x, size.z) < 3.0f) continue;
                const glm::vec3 c = glm::clamp(ear, a.min, a.max);
                if (glm::distance(c, ear) < kWallReach) E.walls.push_back(a);
            }
    }

    sound_space::Space s;
    // What may give the one echo: the best of the faces found (how good, how far, which way).
    float echo_face = 0.0f, echo_r = 0.0f; glm::vec2 echo_dir(0.0f);
    auto echo = [&](float face, float r, glm::vec2 dir) {
        const float score = face * (1.0f - r / 400.0f);
        if (score > echo_face * (1.0f - echo_r / 400.0f)) { echo_face = face; echo_r = r; echo_dir = dir; }
    };

    // The land.
    float land = 0.0f;
    static const float kR[6] = { 10.0f, 20.0f, 35.0f, 60.0f, 100.0f, 160.0f };
    for (int k = 0; k < 8; ++k) {
        const float a = (float(k) + 0.25f) * 0.78539816f;
        const glm::vec2 dir(std::cos(a), std::sin(a));
        float up = 0.0f;
        for (float r : kR) {
            const float h = (std::max)(sample_terrain_height(ear.x + dir.x * r, ear.z + dir.y * r), rt_.water_level);
            const float rise = h - ear.y;
            up = (std::max)(up, rise / r);
            // a face: at least a few metres above the ear and steep to look at (20 degrees and more)
            echo(std::clamp(rise / 6.0f, 0.0f, 1.0f) * std::clamp(rise / r / 0.36f, 0.0f, 1.0f), r, dir);
        }
        land += std::clamp(up / 0.45f, 0.0f, 1.0f) / 8.0f;       // (the horizon 24 degrees up in every direction: closed)
    }

    // Walls: 8 rays of 40 m at the ear's height.
    float walls = 0.0f;
    int   sides = 0;        // ... and how many of them end within 15 m (a room has most)
    for (int k = 0; k < 8; ++k) {
        const float a = float(k) * 0.78539816f;
        const glm::vec2 dir(std::cos(a), std::sin(a));
        const glm::vec3 to = ear + glm::vec3(dir.x, 0.0f, dir.y) * 40.0f;
        float t = 2.0f;
        for (const collision::AABB& w : E.walls) {
            if (inside(ear, w)) continue;
            const float u = segment_box(ear, to, w);
            if (u >= 0.0f) t = (std::min)(t, u);
        }
        if (t > 1.0f) continue;
        const float r = t * 40.0f;
        walls += (1.0f - 0.6f * t) / 8.0f;
        if (r < 15.0f) ++sides;
        if (r > 8.0f) echo(0.7f, r, dir);
    }

    // A roof over the ear (and so: in a room, or under a canopy).
    bool roof = false;
    {
        std::vector<Shelter> sh;
        shelters_near(ear, 1.0f, sh, 4);
        for (const Shelter& r : sh)
            if (ear.x > r.lo.x && ear.x < r.hi.x && ear.z > r.lo.y && ear.z < r.hi.y && ear.y < r.top) roof = true;
    }
    const bool room = roof && sides >= 5;
    E.room = room;

    // The stone bridge over the ear.
    float bridge = 0.0f;
    if (home && bridge_placed_) {
        const float deck = game::stone_bridge_deck_height(bridge_a_, bridge_b_, 2.0f * castle_road_.half_width, ear.x, ear.z);
        if (deck > ear.y + 0.8f) bridge = 1.0f;
    }

    // The woods round the ear.
    float wood = 0.0f;
    for (int k = 0; k < 5; ++k) {
        const glm::vec2 o = k == 0 ? glm::vec2(0.0f) : 22.0f * glm::vec2(std::cos(float(k) * 1.5707963f), std::sin(float(k) * 1.5707963f));
        const float x = ear.x + o.x, z = ear.z + o.y;
        wood += 0.2f * std::clamp(world_forest(x, z, sample_terrain_height(x, z), 1.0f, rt_.water_level), 0.0f, 1.0f);
    }

    s.amount  = std::clamp(kOpenAmount + kLandAmount * land + kWallAmount * walls + kWoodAmount * wood + kBridgeAmount * bridge + (roof ? 0.15f : 0.0f), 0.0f, 0.9f);
    s.decay_s = std::clamp(0.35f + 1.5f * land + 0.5f * walls + 0.3f * wood + 0.6f * bridge, 0.3f, 2.4f);
    s.early   = std::clamp(0.15f + 1.2f * walls + 0.8f * bridge + 0.3f * land + (roof ? 0.4f : 0.0f), 0.0f, 1.0f);
    s.damp_hz = glm::mix(4500.0f, 1800.0f, wood) + 1500.0f * (std::max)(walls, bridge);        // (leaves take the highs; stone and concrete give them back)
    if (echo_face > 0.15f && echo_r > 0.0f) {
        const glm::vec3 f = player_.forward();
        const glm::vec2 right = glm::vec2(-f.z, f.x) / (std::max)(std::sqrt(f.x * f.x + f.z * f.z), 0.05f);
        s.slap_s   = 2.0f * echo_r / sound_space::kSpeedOfSound;        // (there and back)
        s.slap     = 0.45f * echo_face;
        s.slap_pan = std::clamp(glm::dot(echo_dir, right), -1.0f, 1.0f);
    }
    if (cabin)     { s.outside_gain = 0.45f; s.outside_hz = 1400.0f; }                       // (the driver's seat: as before the spatialiser)
    else if (room) {       // (behind walls and glass; and a furnished room is no hall: what comes in from outside does not ring)
        s.outside_gain = 0.50f; s.outside_hz = 1800.0f; s.slap = 0.0f;
        s.amount = (std::min)(s.amount, 0.25f); s.decay_s = (std::min)(s.decay_s, 0.5f);
    }
    E.space = s;
    audio_->set_space(s);

    if (static const bool lg = std::getenv("VIBE_LOG_SOUND_SPACE") != nullptr; lg && E.looks++ % 8 == 0)
        log::infof("[sound-space] surroundings: land %.2f, walls %.2f (%d sides near, %zu boxes), woods %.2f, bridge %.0f, roof %d, room %d, cabin %d -> amount %.2f, tail %.2f s, early %.2f, damping %.0f Hz, echo %.2f after %.0f ms from %+.2f, outside x %.2f through %.0f Hz",
                   double(land), double(walls), sides, E.walls.size(), double(wood), double(bridge), int(roof), int(room), int(cabin), double(s.amount), double(s.decay_s), double(s.early), double(s.damp_hz),
                   double(s.slap), double(s.slap_s * 1000.0f), double(s.slap_pan), double(s.outside_gain), double(s.outside_hz));
}

}  // namespace vibe
