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
#include "engine/water_course.h"

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
// (far2: how far this formation's coarsest level is drawn - by its size, to the trees' range: rockf_poll)
// (a piece whose far range is under the next levels' borders - the shore kit's cobbles and sticks,
//  far2 = 130 x their size - is not drawn past it at all)
int level_for(float d, const float* lim, float k, float far2) { if (d >= far2) return -1; return d < lim[0] * k ? 0 : (d < lim[1] * k ? 1 : 2); }
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
        R.ok = rockf::load_kit("assets/rocks/kit.rockf");
        if (!R.ok) { log::warnf("[rocks] assets/rocks/kit.rockf missing or not understood - no rock formations (tools/rocks/make_formations.py makes it)"); return; }
        // the shore kit (cobbles, driftwood, shingle slabs) as a second kit after it (rock_formations.cpp place_shore)
        if (!std::getenv("VIBE_NO_SHORE_KIT") && !rockf::load_kit("assets/shore/shore.rockf", true))
            log::warnf("[rocks] assets/shore/shore.rockf missing or not understood - no shore kit (tools/shore/make_shore_kit.py makes it)");
        // the cliff walls' kit as a third (rock_cliffs.cpp: steep ground round the camera)
        if (!std::getenv("VIBE_NO_CLIFF_WALLS") && !rockf::load_cliff_kit("assets/cliffs/cliffs.rockf"))
            log::warnf("[rocks] assets/cliffs/cliffs.rockf missing or not understood - no cliff walls (tools/rocks/make_formations.py --set cliffs makes it)");
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
        env.level = [wl = rt_.water_level](float x, float z) { return wl + water_level_off(x, z); };
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
        R.env = env; R.env_ok = true;
        R.level.assign(rockf::instances().size(), int8_t(-1));
        R.all_dirty = true;
        // The rocks in the water part the current (water_fx.cpp river_piers: the water shader's
        // pillow, wake and eddies round each, and the CPU's flow).
        for (const rockf::Inst& in : rockf::instances())
            if (in.why == 8 || (in.why == 9 && in.lo.y < rt_.water_level - 0.05f)) {      // (a boulder at the edge with its feet in the river too)
                // (a stone in the stream whose top lies under the water: a round stone the water runs
                //  over - its top goes with it; rock_formations.cpp place_stream_boulders)
                const float ss = wc::stream_surface(glm::vec2(in.pos.x, in.pos.z), rt_.water_level);
                river_pier_add(glm::vec2(in.pos.x, in.pos.z), 0.75f * 0.5f * std::max(in.hi.x - in.lo.x, in.hi.z - in.lo.z),
                               (in.why == 8 && ss > -1e8f && in.hi.y < ss) ? in.hi.y : 1e5f);
            }
    }
    // The cliff walls round the camera (rock_cliffs.cpp): their slots follow the fixed formations.
    if (R.env_ok && rockf::cliff_stream(R.env, player_.eye_position(), 1)) R.all_dirty = true;
    if (R.all_dirty) {
        R.all_dirty = false;
        const std::vector<rockf::Inst>& F = rockf::instances();
        const std::vector<rockf::Inst>& S = rockf::cliff_slots();
        const std::vector<uint32_t>& gen = rockf::cliff_slot_gen();
        R.all.assign(F.begin(), F.end());
        R.all.insert(R.all.end(), S.begin(), S.end());
        R.level.resize(R.all.size(), int8_t(-1));
        R.traced.resize(R.all.size(), uint8_t(1));
        R.slot_gen.resize(gen.size(), 0xFFFFFFFFu);
        for (size_t s = 0; s < gen.size(); ++s) {
            if (R.slot_gen[s] == gen[s]) continue;
            R.slot_gen[s] = gen[s];
            const int key = int(F.size() + s);
            R.level[size_t(key)] = -1;
            if (auto b = R.bodies.find(key); b != R.bodies.end()) { if (b->second && physics_) physics_->remove_mesh_instance(b->second); R.bodies.erase(b); }
            R.walk.erase(key);
            R.cols_at = glm::vec3(1e9f);                       // (the walker's list is made again below, this frame)
            R.force_draw = true;
        }
    }
    const std::vector<rockf::Inst>& I = R.all;
    if (I.empty()) return;
    rockf_tops_poll();
    const glm::vec3 eye = player_.eye_position();
    const uint32_t now = uint32_t(frame_number_);
    bool changed = R.force_draw, made_one = false;
    R.force_draw = false;
    for (size_t i = 0; i < I.size(); ++i) {
        const rockf::Inst& in = I[i];
        if (in.piece < 0) continue;                            // (an empty cliff slot)
        const float d = glm::distance(eye, glm::clamp(eye, in.lo, in.hi));
        // The coarsest level (0.8 - 2.5 thousand triangles) stays as the far form, out to where the
        // piece is a couple of pixels high and never past the range objects are drawn to (the
        // trees': object_far_, 3.8 km). VIBE_ROCK_FAR=<m> caps it (900 = as before).
        static const float far_cap = std::getenv("VIBE_ROCK_FAR") ? float(std::atof(std::getenv("VIBE_ROCK_FAR"))) : 1e9f;
        const float size = std::max(in.hi.y - in.lo.y, 0.5f * std::max(in.hi.x - in.lo.x, in.hi.z - in.lo.z));
        const float far2 = std::min(std::min(std::max(kIn[2] * in.reach, size * 130.0f), object_far_ - 150.0f), far_cap);
        const int cur = R.level[i], a = level_for(d, kIn, in.reach, far2), b = level_for(d, kOut, in.reach, far2 * 1.08f);
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
        // (far formations are drawn but not traced: shadows and bounce light only within 1.2 km)
        if (R.traced.size() != I.size()) R.traced.assign(I.size(), 1);
        const uint8_t tr = d < 1200.0f ? 1 : (d > 1320.0f ? 0 : R.traced[i]);
        if (tr != R.traced[i]) { R.traced[i] = tr; changed = true; }
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
            g.rock_piece = in.piece;
            if (i < R.traced.size() && !R.traced[i]) g.blas_addr = 0;
            if (in.look == 1) { g.tex_normal = -10; g.ssdm = 0.10f; }       // (chalk: its blocks stand out a hand's width - rock_form.glsl rf_chalk_h)
            // (each formation a shade of its own: warmer or cooler, lighter or darker by a few percent)
            const uint32_t h = uint32_t(i) * 2654435761u;
            const float t0 = float(h & 255u) / 255.0f, t1 = float((h >> 8) & 255u) / 255.0f;
            g.tint = glm::mix(glm::vec3(1.04f, 1.0f, 0.94f), glm::vec3(0.96f, 0.99f, 1.03f), t0) * glm::mix(0.90f, 1.08f, t1);
            if (in.why >= 6 && in.why != 9) g.tint = glm::vec3(1.0f);       // (the pieces of a run are one rock; boulders keep their own shade)
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
            if (I[i].piece < 0) continue;
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
            float d = std::sqrt(dx * dx + dz * dz);
            if (in.piece < 0 || rockf::no_collider(in)) d = 1e9f;                  // (the shore kit's small cobbles and sticks: walked over, not into)
            const int key = int(i);
            // physics bodies (the car, crates): within kBodyReach, given back beyond 1.5 x
            if (physics_) {
                auto b = R.bodies.find(key);
                if (d < kBodyReach) {
                    if (b == R.bodies.end()) b = R.bodies.emplace(key, 0u).first;
                    if (b->second == 0) b->second = physics_->add_mesh_instance(in.piece, in.pos, in.yaw, in.scale, in.tilt);
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

// THE TOPS FROM ABOVE, for the grass placement (tuft_place.comp binding 10): a picture of
// 96 m round the camera at 0.25 m, per texel the highest rock surface there and whether the
// ground lies on it (it faces up enough for the material's rule: N.y >= 0.82) or it is bare
// rock (then nothing grows through it). Made on the CPU from the formations' middle level -
// a few thousand triangles each, only those standing in the window - when the window moves on
// by 16 m or the set in it changes. Row 0 is the header: texel 0 = the window's corner (x, z),
// texel 1 = (metres a texel, the mark 77). VIBE_NO_ROCK_TOP_GRASS=1: the picture stays empty.
void VulkanEngine::rockf_tops_poll() {
    static const bool off = std::getenv("VIBE_NO_ROCK_TOP_GRASS") != nullptr || std::getenv("VIBE_NO_ROCK_TOPS") != nullptr;
    if (off || !rockf_ || !rockf_->ok || !rockf::placed() || !rock_tops_view()) return;
    RockfState& R = *rockf_;
    const int N = kRockTopsN;
    const float T = kRockTopsTexel, snap = 16.0f;
    const glm::vec3 eye = player_.eye_position();
    const glm::ivec2 c(int(std::floor(eye.x / snap + 0.5f)), int(std::floor(eye.z / snap + 0.5f)));
    const glm::vec2 org(float(c.x) * snap - 0.5f * float(N) * T, float(c.y) * snap - 0.5f * float(N) * T);
    const glm::vec2 end = org + glm::vec2(float(N) * T);
    const std::vector<rockf::Inst>& I = R.all;            // (the fixed formations and the cliff walls)
    std::vector<size_t> in_win;
    uint64_t sig = 1469598103934665603ull;
    for (size_t i = 0; i < I.size(); ++i) {
        const rockf::Inst& in = I[i];
        if (in.piece < 0) continue;
        if (in.hi.x < org.x || in.lo.x > end.x || in.hi.z < org.y || in.lo.z > end.y || !globe::current().home) continue;
        in_win.push_back(i);
        sig = (sig ^ uint64_t(i + 1)) * 1099511628211ull;
    }
    if (c == R.tops_c && sig == R.tops_sig) return;
    const bool was_empty = R.tops_empty;
    R.tops_c = c; R.tops_sig = sig; R.tops_empty = in_win.empty();
    if (in_win.empty() && was_empty && !R.tops_px.empty()) return;        // (nothing there before, nothing now: the picture on the GPU is still empty)
    R.tops_px.assign(size_t(N) * N * 2, 0.0f);
    std::vector<float>& P = R.tops_px;
    for (size_t idx : in_win) {
        const rockf::Inst& in = I[idx];
        const game::LevelMesh& m = rockf::kit()[size_t(in.piece)].lod[1];
        const glm::mat4 M = rockf::transform(in);
        for (size_t t = 0; t + 2 < m.idx.size(); t += 3) {
            glm::vec3 p[3];
            for (int k = 0; k < 3; ++k) p[k] = glm::vec3(M * glm::vec4(m.verts[m.idx[t + k]].p, 1.0f));
            glm::vec3 n = glm::cross(p[1] - p[0], p[2] - p[0]);
            const float nl = glm::length(n);
            if (nl < 1e-9f) continue;
            n /= nl;
            if (n.y < 0.0f) n = -n;
            const glm::vec2 a((p[0].x - org.x) / T, (p[0].z - org.y) / T), b((p[1].x - org.x) / T, (p[1].z - org.y) / T), q((p[2].x - org.x) / T, (p[2].z - org.y) / T);
            const float area = (b.x - a.x) * (q.y - a.y) - (q.x - a.x) * (b.y - a.y);
            if (std::abs(area) < 1e-6f) continue;
            const int x0 = std::max(0, int(std::floor(std::min({ a.x, b.x, q.x })))), x1 = std::min(N - 1, int(std::ceil(std::max({ a.x, b.x, q.x }))));
            const int z0 = std::max(1, int(std::floor(std::min({ a.y, b.y, q.y })))), z1 = std::min(N - 1, int(std::ceil(std::max({ a.y, b.y, q.y }))));
            for (int z = z0; z <= z1; ++z)
                for (int x = x0; x <= x1; ++x) {
                    const float px = float(x) + 0.5f, pz = float(z) + 0.5f;
                    const float w0 = ((b.x - px) * (q.y - pz) - (q.x - px) * (b.y - pz)) / area;
                    const float w1 = ((q.x - px) * (a.y - pz) - (a.x - px) * (q.y - pz)) / area;
                    const float w2 = 1.0f - w0 - w1;
                    if (w0 < -0.02f || w1 < -0.02f || w2 < -0.02f) continue;
                    const float y = p[0].y * w0 + p[1].y * w1 + p[2].y * w2;
                    float* o = &P[(size_t(z) * N + size_t(x)) * 2];
                    if (o[1] > 0.0f && y <= o[0]) continue;
                    o[0] = y; o[1] = n.y >= 0.82f ? 1.0f : 0.5f;
                }
        }
    }
    P[0] = org.x; P[1] = org.y; P[2] = T; P[3] = 77.0f;
    rock_tops_upload(P.data());
    static const bool verbose = std::getenv("VIBE_LOG_ROCKS") != nullptr;
    if (verbose) {
        size_t top = 0, bare = 0;
        for (size_t k = size_t(N); k < size_t(N) * N; ++k) { if (P[k * 2 + 1] > 0.75f) ++top; else if (P[k * 2 + 1] > 0.25f) ++bare; }
        log::infof("[rocks] tops picture round (%.0f, %.0f): %zu formations, %.0f m2 of top the ground lies on, %.0f m2 bare", double(org.x + 0.5f * N * T), double(org.y + 0.5f * N * T),
                   in_win.size(), double(top) * double(T * T), double(bare) * double(T * T));
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
