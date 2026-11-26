// Rock formations, the engine's side: the kit's meshes on the GPU, which formations are drawn
// at which level of detail, their colliders (the rule and the kit's file: rock_formations.cpp).
//
// STREAMED. The list of formations is a few numbers each (rockf::instances()). Once a frame
// (rockf_poll, from game_loop.cpp) every formation is given a level by its distance from the
// eye - level 0 within 55 m, 1 within 170 m, 2 within 900 m, none beyond (each a little further
// on the way out, so nothing flickers at a border). A piece's level is put on the GPU (mesh +
// ray-tracing structure) when a formation first wants it, one a frame, and given back when no
// formation has used it for fifteen seconds. What is drawn is a list of mesh parts that share
// the kit's buffers and carry the formation's place (CastleMeshGpu::xf) - the static mesh path
// draws, shadows and traces them like any other part (mesh_parts_now, castle_mesh.cpp), and
// the screen-space displacement pass moves their pixels (LevelMesh::ssdm).
// Home's chart only.
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/rock_formations.h"
#include "engine/highway.h"
#include "engine/terrain.h"
#include "engine/road.h"
#include "engine/globe.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace vibe {

namespace {
constexpr float kIn[3]  = { 55.0f, 170.0f, 900.0f };     // a level is taken within this distance ...
constexpr float kOut[3] = { 66.0f, 195.0f, 1000.0f };    // ... and kept to this
constexpr uint32_t kIdleFrames = 900;                    // an unused level is given back after this
constexpr float kBodyReach = 70.0f;                      // formations this near the player stand in the physics world
constexpr float kWalkReach = 14.0f;                      // ... and this near, the walker's triangles

// (k: the formation's reach - tall pieces hold their levels further out; never past the far plane)
int level_for(float d, const float* lim, float k) { return d < lim[0] * k ? 0 : (d < lim[1] * k ? 1 : (d < std::min(lim[2] * k, lim[2] + 480.0f) ? 2 : -1)); }
}

void VulkanEngine::rockf_poll() {
    static const bool off = std::getenv("VIBE_NO_ROCK_FORMATIONS") != nullptr;
    if (off) return;
    if (rockf_) {
        for (auto it = rockf_->retired.begin(); it != rockf_->retired.end();) {
            if (--it->first > 0) { ++it; continue; }
            destroy_mesh_parts(it->second);
            it = rockf_->retired.erase(it);
        }
    }
    if (!highway::ready() || !scene_desc_set_ || !launch_site_placed_ || terrain_data_.heights.empty()) return;
    if (!rockf_) rockf_ = new RockfState();
    RockfState& R = *rockf_;
    if (!globe::current().home) {
        if (!R.draw.empty()) { R.draw.clear(); R.tris = 0; std::fill(R.level.begin(), R.level.end(), int8_t(-1)); ++hw_gen_; }
        rockf_drop_colliders();
        return;
    }
    if (!R.tried) {
        R.tried = true;
        R.ok = rockf::load_kit("assets/rocks/formations.rockf");
        if (!R.ok) { log::warnf("[rocks] assets/rocks/formations.rockf missing or not understood - no rock formations (tools/rocks/make_formations.py makes it)"); return; }
        R.kit.resize(rockf::kit().size());
    }
    if (!R.ok) return;
    // The physics world's rock meshes: one per piece, built once on a worker thread.
    if (!R.phys_requested && physics_) {
        R.phys_requested = true;
        std::vector<std::vector<glm::vec3>> pieces(rockf::kit().size());
        for (size_t p = 0; p < pieces.size(); ++p) rockf::collision_triangles(int(p), pieces[p]);
        physics_->mesh_pieces_build_async(std::move(pieces));
    }
    const rockf::Ground ground = [this](float x, float z) {
        return (std::abs(x) < 1020.0f && std::abs(z) < 1020.0f) ? sample_terrain_height(x, z) : world_ground_height(x, z, rt_.water_level, 9);
    };
    if (!rockf::placed()) {
        rockf::Env env;
        env.ground = ground;
        env.water = rt_.water_level;
        // (the old castle road alone: the highways' lines are asked for by the rule itself)
        auto road = std::make_shared<Road>(castle_road_);
        road->with_highways = false;
        env.blocked = [this, road, ground](float x, float z, float r) {
            if (x > kCastlePark[0] - 8.0f - r && x < kCastlePark[2] + 8.0f + r && z > kCastlePark[1] - 8.0f - r && z < kCastlePark[3] + 8.0f + r) return true;
            if (x > kStationRect[0] - 12.0f - r && x < kStationRect[2] + 12.0f + r && z > kStationRect[1] - 12.0f - r && z < kStationRect[3] + 12.0f + r) return true;
            if (glm::length(glm::vec2(x, z)) < 95.0f + r) return true;                                        // the castle on its rock
            if (glm::distance(glm::vec2(x, z), glm::vec2(kLaunchSiteX, kLaunchSiteZ)) < 230.0f + r) return true;
            if (!road->empty() && road->distance(x, z) < road->half_width + road->shoulder + 1.0f + r) return true;
            const float gy = ground(x, z);
            for (const collision::AABB& a : world_.aabbs) {                                                    // buildings, bridges, towers
                if (a.max.x - a.min.x > 400.0f || a.max.z - a.min.z > 400.0f || a.max.y < gy - 1.0f) continue;
                const float dx = std::max(std::max(a.min.x - x, x - a.max.x), 0.0f), dz = std::max(std::max(a.min.z - z, z - a.max.z), 0.0f);
                if (dx * dx + dz * dz < (r + 1.5f) * (r + 1.5f)) return true;
            }
            return false;
        };
        rockf::place(env);
        R.level.assign(rockf::instances().size(), int8_t(-1));
        // The rocks in the water part the current (water_fx.cpp river_piers: the water shader's
        // pillow, wake and eddies round each, and the CPU's flow).
        for (const rockf::Inst& in : rockf::instances())
            if (in.why == 8) river_pier_add(glm::vec2(in.pos.x, in.pos.z), 0.75f * 0.5f * std::max(in.hi.x - in.lo.x, in.hi.z - in.lo.z));
    }
    const std::vector<rockf::Inst>& I = rockf::instances();
    if (I.empty()) return;
    const glm::vec3 eye = player_.eye_position();
    const uint32_t now = uint32_t(frame_number_);
    bool changed = false, made_one = false;
    for (size_t i = 0; i < I.size(); ++i) {
        const rockf::Inst& in = I[i];
        const float d = glm::distance(eye, glm::clamp(eye, in.lo, in.hi));
        const int cur = R.level[i], a = level_for(d, kIn, in.reach), b = level_for(d, kOut, in.reach);
        int want = (cur == a || cur == b) ? cur : a;
        if (want >= 0) {
            RockfState::Level& L = R.kit[size_t(in.piece)][size_t(want)];
            if (!L.made) {
                // (not while a level that was given back still waits: its buffers may be the ones in flight)
                if (made_one || !R.retired.empty()) want = cur;
                else {
                    const game::LevelMesh& lm = rockf::kit()[size_t(in.piece)].lod[want];
                    std::vector<CastleMeshGpu> one;
                    one.push_back(make_mesh_part(lm));
                    build_mesh_part_blas(one);
                    L.part = one[0]; L.made = true; L.tris = uint32_t(lm.idx.size() / 3);
                    made_one = true;
                }
            }
            if (want >= 0) R.kit[size_t(in.piece)][size_t(want)].last_used = now;
        }
        if (want != cur) { R.level[i] = int8_t(want); changed = true; }
    }
    // Levels nobody has wanted for a while go back.
    for (auto& piece : R.kit)
        for (RockfState::Level& L : piece)
            if (L.made && now - L.last_used > kIdleFrames) {
                R.retired.emplace_back(12, std::vector<CastleMeshGpu>{ L.part });
                L = RockfState::Level{};
                changed = true;
            }
    if (changed) {
        R.draw.clear(); R.tris = 0;
        for (size_t i = 0; i < I.size(); ++i) {
            if (R.level[i] < 0) continue;
            const rockf::Inst& in = I[i];
            const RockfState::Level& L = R.kit[size_t(in.piece)][size_t(R.level[i])];
            if (!L.made) { R.level[i] = -1; continue; }
            CastleMeshGpu g = L.part;                          // (shares the kit's buffers)
            g.xf = rockf::transform(in);
            g.xf_prev = g.xf;
            g.bmin = in.lo; g.bmax = in.hi;
            if (in.look == 1) { g.tex_normal = -10; g.ssdm = 0.10f; }       // (chalk: its blocks stand out a hand's width - rock_form.glsl rf_chalk_h)
            // (each formation a shade of its own: warmer or cooler, lighter or darker by a few percent)
            const uint32_t h = uint32_t(i) * 2654435761u;
            const float t0 = float(h & 255u) / 255.0f, t1 = float((h >> 8) & 255u) / 255.0f;
            g.tint = glm::mix(glm::vec3(1.04f, 1.0f, 0.94f), glm::vec3(0.96f, 0.99f, 1.03f), t0) * glm::mix(0.90f, 1.08f, t1);
            if (in.why >= 6) g.tint = glm::vec3(1.0f);                      // (the pieces of a run are one rock)
            R.draw.push_back(g);
            R.tris += L.tris;
        }
        ++hw_gen_;                                             // (the ray-tracing structure takes the new list)
        static const bool verbose = std::getenv("VIBE_LOG_ROCKS") != nullptr;
        if (verbose) log::infof("[rocks] drawn: %zu formations, %u triangles", R.draw.size(), R.tris);
    }
    // VIBE_TEST_ROCK_DROP=1: once, the player is dropped 3 m above the middle of the nearest
    // formation (the log's per-second position then shows where they come to rest).
    static bool dropped = std::getenv("VIBE_TEST_ROCK_DROP") == nullptr;
    if (!dropped && !I.empty()) {
        dropped = true;
        size_t best = 0; float bd = 1e30f;
        for (size_t i = 0; i < I.size(); ++i) {
            const float d = glm::distance(glm::vec2(0.5f * (I[i].lo.x + I[i].hi.x), 0.5f * (I[i].lo.z + I[i].hi.z)), glm::vec2(player_.position.x, player_.position.z));
            if (d < bd) { bd = d; best = i; }
        }
        const rockf::Inst& in = I[best];
        const glm::vec3 c = 0.5f * (in.lo + in.hi);
        player_.position = glm::vec3(c.x, in.hi.y + 3.0f, c.z);
        player_.velocity = glm::vec3(0.0f);
        R.cols_at = glm::vec3(1e9f);
        log::infof("[rocks] test drop onto %s at (%.1f, %.1f, %.1f): its top %.1f, the ground there %.1f",
                   rockf::kit()[size_t(in.piece)].name.c_str(), double(c.x), double(player_.position.y), double(c.z), double(in.hi.y),
                   double(sample_terrain_height(c.x, c.z)));
    }
    // Colliders: the rock's own surface for what stands near the player, looked at again when
    // the player has moved a few metres (or a piece's physics mesh has come in).
    const glm::vec3 pp = player_.position;
    bool pending = false;
    for (const auto& kv : R.bodies) if (kv.second == 0) { pending = true; break; }
    if (pending || glm::distance(glm::vec2(pp.x, pp.z), glm::vec2(R.cols_at.x, R.cols_at.z)) > 4.0f) {
        R.cols_at = pp;
        R.walk_near.clear();
        for (size_t i = 0; i < I.size(); ++i) {
            const rockf::Inst& in = I[i];
            const float dx = std::max(std::max(in.lo.x - pp.x, pp.x - in.hi.x), 0.0f), dz = std::max(std::max(in.lo.z - pp.z, pp.z - in.hi.z), 0.0f);
            const float d = std::sqrt(dx * dx + dz * dz);
            const int key = int(i);
            // physics bodies (the car, crates): within kBodyReach, given back beyond 1.5 x
            if (physics_) {
                auto b = R.bodies.find(key);
                if (d < kBodyReach) {
                    if (b == R.bodies.end()) b = R.bodies.emplace(key, 0u).first;
                    if (b->second == 0) b->second = physics_->add_mesh_instance(in.piece, in.pos, in.yaw, in.scale);
                } else if (b != R.bodies.end() && d > 1.5f * kBodyReach) {
                    if (b->second) physics_->remove_mesh_instance(b->second);
                    R.bodies.erase(b);
                }
            }
            // the walker's triangles: within kWalkReach, freed beyond 2 x
            auto w = R.walk.find(key);
            if (d < kWalkReach) {
                if (w == R.walk.end()) {
                    w = R.walk.emplace(key, std::make_unique<tricol::Mesh>()).first;
                    rockf::collision_mesh(in, *w->second);
                }
                R.walk_near.push_back(w->second.get());
            } else if (w != R.walk.end() && d > 2.0f * kWalkReach) {
                R.walk.erase(w);
            }
        }
    }
}

// All the rock colliders given back (left home, or shutting down).
void VulkanEngine::rockf_drop_colliders() {
    if (!rockf_) return;
    RockfState& R = *rockf_;
    if (physics_) for (const auto& kv : R.bodies) if (kv.second) physics_->remove_mesh_instance(kv.second);
    R.bodies.clear();
    R.walk.clear();
    R.walk_near.clear();
    R.cols_at = glm::vec3(1e9f);
}

// The parts to draw and trace this frame (castle_mesh.cpp mesh_parts_now).
void VulkanEngine::rockf_parts(std::vector<const CastleMeshGpu*>& out) const {
    if (!rockf_ || !globe::current().home) return;
    for (const CastleMeshGpu& g : rockf_->draw) out.push_back(&g);
}

// The triangles of the formations near the player, for the walker (game_loop.cpp).
const std::vector<const tricol::Mesh*>& VulkanEngine::rockf_walk_meshes() const {
    static const std::vector<const tricol::Mesh*> none;
    return rockf_ ? rockf_->walk_near : none;
}

void VulkanEngine::rockf_destroy() {
    if (!rockf_) return;
    rockf_drop_colliders();
    for (auto& r : rockf_->retired) destroy_mesh_parts(r.second);
    for (auto& piece : rockf_->kit)
        for (RockfState::Level& L : piece)
            if (L.made) { std::vector<CastleMeshGpu> one{ L.part }; destroy_mesh_parts(one); }
    delete rockf_;
    rockf_ = nullptr;
}

}  // namespace vibe
