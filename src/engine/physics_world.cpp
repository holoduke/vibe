#include "engine/physics_world.h"

#include "engine/log.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <cstring>
#include <map>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Vehicle/VehicleConstraint.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/PhysicsStepListener.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/CollisionCollector.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <thread>
#include <unordered_map>
#include <vector>

JPH_SUPPRESS_WARNINGS

namespace vibe {

namespace {

namespace Layers {
    // Four object layers so bullets and sparks can be made transparent to
    // each other (the user's request: bullets shouldn't deflect off
    // hit-spark spheres). Broadphase still has just NON_MOVING vs MOVING
    // since the per-pair filtering is what matters for game logic.
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING     = 1;
    static constexpr JPH::ObjectLayer BULLET     = 2;
    static constexpr JPH::ObjectLayer SPARK      = 3;
    // Destructible voxel building. Static, collides with crates/debris so
    // they pile on it, but NOT bullets — bullet-vs-voxel destruction uses a
    // live voxel raycast (VoxelWorld::raycast), so the debounced collision
    // body must not stop bullets at a stale (pre-hole) surface.
    static constexpr JPH::ObjectLayer VOXEL      = 4;
    static constexpr JPH::ObjectLayer NUM        = 5;
}

namespace BPLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr unsigned NUM = 2;
}

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl() {
        map_[Layers::NON_MOVING] = BPLayers::NON_MOVING;
        map_[Layers::MOVING]     = BPLayers::MOVING;
        map_[Layers::BULLET]     = BPLayers::MOVING;
        map_[Layers::SPARK]      = BPLayers::MOVING;
        map_[Layers::VOXEL]      = BPLayers::NON_MOVING;  // static
    }
    unsigned int GetNumBroadPhaseLayers() const override { return BPLayers::NUM; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer l) const override {
        return map_[l];
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer) const override {
        return "vibe-bp";
    }
#endif
private:
    JPH::BroadPhaseLayer map_[Layers::NUM]{};
};

class ObjectVsBPFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer obj, JPH::BroadPhaseLayer bp) const override {
        switch (obj) {
            case Layers::NON_MOVING:
            case Layers::VOXEL:
                // Static only collides with anything that moves.
                return bp == BPLayers::MOVING;
            case Layers::MOVING:
            case Layers::BULLET:
            case Layers::SPARK:
                // Movables sweep the whole broadphase; pair filter below
                // does the actual game-logic culling.
                return true;
            default:
                return false;
        }
    }
};

class ObjectPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    ObjectPairFilterImpl() {
        // Precomputed symmetric collision matrix — ShouldCollide is called
        // thousands of times per frame across narrowphase pairs; a flat
        // table lookup beats the if-cascade by ~5-8 instructions per pair.
        //   STATIC vs MOVING / BULLET / SPARK : yes (world is solid)
        //   MOVING vs MOVING/BULLET/SPARK     : yes (crates collide with everything)
        //   BULLET vs BULLET                  : no  (bullets ignore each other)
        //   BULLET vs SPARK                   : no  (bullets pass through sparks)
        //   SPARK  vs SPARK                   : no  (sparks streak through each other)
        //   STATIC vs STATIC                  : no
        //   VOXEL  vs anything but MOVING     : no  (bullets carve via raycast)
        const bool yes = true;
        // STATIC
        matrix_[Layers::NON_MOVING][Layers::MOVING] = yes;
        matrix_[Layers::NON_MOVING][Layers::BULLET] = yes;
        matrix_[Layers::NON_MOVING][Layers::SPARK]  = yes;
        // MOVING
        matrix_[Layers::MOVING][Layers::NON_MOVING] = yes;
        matrix_[Layers::MOVING][Layers::MOVING]     = yes;
        matrix_[Layers::MOVING][Layers::BULLET]     = yes;
        matrix_[Layers::MOVING][Layers::SPARK]      = yes;
        matrix_[Layers::MOVING][Layers::VOXEL]      = yes;
        // BULLET
        matrix_[Layers::BULLET][Layers::NON_MOVING] = yes;
        matrix_[Layers::BULLET][Layers::MOVING]     = yes;
        // SPARK
        matrix_[Layers::SPARK][Layers::NON_MOVING]  = yes;
        matrix_[Layers::SPARK][Layers::MOVING]      = yes;
        // VOXEL
        matrix_[Layers::VOXEL][Layers::MOVING]      = yes;
    }

    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        if (a >= Layers::NUM || b >= Layers::NUM) return false;
        return matrix_[a][b];
    }

private:
    bool matrix_[Layers::NUM][Layers::NUM]{};
};

void trace_callback(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    log::infof("[jolt] %s", buf);
}

#ifdef JPH_ENABLE_ASSERTS
bool assert_callback(const char* expr, const char* msg, const char* file, JPH::uint line) {
    log::errorf("[jolt assert] %s:%u (%s) %s", file, line, expr, msg ? msg : "");
    return true;
}
#endif

std::atomic<int> g_init_count{0};

void global_init_once() {
    if (g_init_count.fetch_add(1) == 0) {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = &trace_callback;
#ifdef JPH_ENABLE_ASSERTS
        JPH::AssertFailed = &assert_callback;
#endif
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
        log::info("[jolt] global init complete");
    }
}

void global_shutdown_if_last() {
    if (g_init_count.fetch_sub(1) == 1) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        log::info("[jolt] global shutdown complete");
    }
}

} // namespace

struct PhysicsWorld::Impl {
    JPH::PhysicsSystem system;
    // 32 MB temp scratch — at 100 dyn props with up to 6 collision_steps
    // the constraint solver fits comfortably; previously 16 MB was tight
    // enough to occasionally fall back to the heap allocator.
    JPH::TempAllocatorImpl temp_alloc{ 32 * 1024 * 1024 };
    JPH::JobSystemThreadPool job_system{
        JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers,
        std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1) };

    BPLayerInterfaceImpl bp_layer_iface;
    ObjectVsBPFilterImpl obj_vs_bp;
    ObjectPairFilterImpl obj_pairs;

    // Single static body for the destructible voxel shape (rebuilt on carve).
    JPH::BodyID voxel_body;   // invalid until set_voxel_collision runs
    // Per-cell static box compounds (vegetation trunks + rocks), kept so a
    // cell can be rebuilt when blasts destroy trees in it.
    struct BoxCell { JPH::BodyID body; std::vector<PhysicsWorld::StaticBox> boxes; };
    // Terrain heightfield (region-updatable via SetHeights for craters).
    JPH::BodyID hf_body;
    std::thread hf_thread;            // add_static_heightfield_async
    int hf_samples = 0, hf_block = 8;
    std::map<std::pair<int, int>, BoxCell> box_cells;
    float box_cell = 128.0f;
    // Hash of the last voxel-collision geometry pushed. Lets us skip the
    // whole RemoveBody+DestroyBody+rebuild + AddShape×N storm when the
    // greedy decomposition produced the same boxes (e.g. carves outside
    // the collision-build radius, or no carves between two cadence ticks).
    size_t voxel_shape_hash = 0;
    struct Ragdoll { std::vector<JPH::BodyID> bodies; std::vector<JPH::Ref<JPH::Constraint>> joints; JPH::BodyID floor; };
    std::unordered_map<uint32_t, Ragdoll> ragdolls;
    std::unordered_map<uint32_t, JPH::BodyID> ids;
    // Reverse lookup so raycast hits → user id is O(1) instead of a
    // linear walk over `ids`. Keyed on BodyID's internal index value
    // (BodyID has no GetIndexAndSequenceNumber on older Jolt versions
    // — store the full GetIndex() which is the broadphase-stable key).
    std::unordered_map<uint32_t, uint32_t> id_by_body_index;
    uint32_t next_id = 1;
};

PhysicsWorld::PhysicsWorld() {
    global_init_once();
    impl_ = std::make_unique<Impl>();
    impl_->system.Init(
        /*maxBodies*/        2048,
        /*numBodyMutexes*/   0,
        /*maxBodyPairs*/     4096,
        /*maxContactConstraints*/ 2048,
        impl_->bp_layer_iface,
        impl_->obj_vs_bp,
        impl_->obj_pairs);
    impl_->system.SetGravity(JPH::Vec3(0, -25.0f, 0));
    // Tune the constraint solver. Default is 10 velocity / 2 position
    // iterations — fine for tall stacks of crates, overkill for our
    // shallow piles. 4 vel / 1 pos still resolves crate-on-floor stably
    // and shaves a measurable slice of step time at 100 dyn bodies.
    JPH::PhysicsSettings ps = impl_->system.GetPhysicsSettings();
    ps.mNumVelocitySteps = 4;
    ps.mNumPositionSteps = 1;
    impl_->system.SetPhysicsSettings(ps);
    log::info("PhysicsWorld initialized (Jolt)");
}

PhysicsWorld::~PhysicsWorld() {
    vehicle_remove();
    impl_.reset();
    global_shutdown_if_last();
}

void PhysicsWorld::add_static_heightfield_async(std::vector<float> samples, int dim,
                                                glm::vec2 origin_xz, float cell_size) {
    finish_async();
    impl_->hf_thread = std::thread([this, s = std::move(samples), dim, origin_xz, cell_size]() {
        add_static_heightfield(s.data(), dim, origin_xz, cell_size);
    });
}

void PhysicsWorld::finish_async() {
    if (impl_ && impl_->hf_thread.joinable()) impl_->hf_thread.join();
}

void PhysicsWorld::step(float dt) {
    finish_async();
    if (dt <= 0.0f) return;
    vehicle_keep_prev();
    constexpr float kFixed = 1.0f / 60.0f;
    int collision_steps = 1;
    int subdivisions = std::max(1, static_cast<int>(std::ceil(dt / kFixed)));
    float sub_dt = dt / subdivisions;
    for (int i = 0; i < subdivisions; ++i) {
        impl_->system.Update(sub_dt, collision_steps,
                             &impl_->temp_alloc, &impl_->job_system);
    }
}

int PhysicsWorld::body_count() const {
    return static_cast<int>(impl_->system.GetNumBodies());
}

// Shared convex-radius derivation. Jolt's default 0.05 fails on shapes
// thinner than that; we scale to box size, clamp to a 0.001 floor (so a
// 1 mm slab doesn't underflow), and cap at 0.04. Callers below stamp the
// same value out for static + dyn + batch + voxel paths.
static constexpr float calc_box_convex_radius(float min_he) {
    float r = min_he * 0.4f;
    if (r > 0.04f) r = 0.04f;
    if (r < 0.001f) r = 0.001f;
    return r;
}

void PhysicsWorld::add_static_box(glm::vec3 c, glm::vec3 he) {
    // Runtime calls are still allowed but cost a broadphase mutex per
    // body — flag it so future batch-spawn callers prefer add_static_boxes.
    log::warn("[jolt] add_static_box() called at runtime — prefer "
              "add_static_boxes() for batch efficiency");
    float min_he = std::min({he.x, he.y, he.z});
    float convex_radius = calc_box_convex_radius(min_he);
    JPH::BoxShapeSettings ss(JPH::Vec3(he.x, he.y, he.z), convex_radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] static box shape: %s", sr.GetError().c_str());
        return;
    }
    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z),
                                  JPH::Quat::sIdentity(),
                                  JPH::EMotionType::Static,
                                  Layers::NON_MOVING);
    impl_->system.GetBodyInterface().CreateAndAddBody(bcs, JPH::EActivation::DontActivate);
}

void PhysicsWorld::add_static_boxes(const StaticBox* boxes, size_t count) {
    if (count == 0) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    // Two-step: first allocate all bodies (bi.CreateBody, no add), then
    // batch-add via AddBodiesPrepare + AddBodiesFinalize. The prepare/
    // finalize pair takes the broadphase mutex once for the whole batch
    // instead of per body, which is the dominant cost when 150+ static
    // brushes are loaded at level start.
    std::vector<JPH::BodyID> ids;
    ids.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        glm::vec3 he = boxes[i].half_extents;
        float min_he = std::min({he.x, he.y, he.z});
        float convex_radius = calc_box_convex_radius(min_he);
        JPH::BoxShapeSettings ss(JPH::Vec3(he.x, he.y, he.z), convex_radius);
        ss.SetEmbedded();
        JPH::ShapeSettings::ShapeResult sr = ss.Create();
        if (sr.HasError()) {
            log::errorf("[jolt] static box shape (batch %zu): %s",
                        i, sr.GetError().c_str());
            continue;
        }
        glm::vec3 c = boxes[i].center;
        JPH::BodyCreationSettings bcs(sr.Get(),
                                      JPH::RVec3(c.x, c.y, c.z),
                                      JPH::Quat::sIdentity(),
                                      JPH::EMotionType::Static,
                                      Layers::NON_MOVING);
        if (JPH::Body* body = bi.CreateBody(bcs)) {
            ids.push_back(body->GetID());
        }
    }
    if (ids.empty()) return;
    JPH::BodyInterface::AddState state =
        bi.AddBodiesPrepare(ids.data(), static_cast<int>(ids.size()));
    bi.AddBodiesFinalize(ids.data(), static_cast<int>(ids.size()),
                          state, JPH::EActivation::DontActivate);
}

// Many small static boxes (tree trunks, rocks) merged into one static
// compound body per `cell` metres of XZ - keeps the body count far below
// Jolt's max_bodies (16k individual boxes exhausted it and the terrain
// heightfield then failed to create).
void PhysicsWorld::add_static_box_compounds(const StaticBox* boxes, size_t count, float cell) {
    if (count == 0) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    std::map<std::pair<int, int>, std::vector<size_t>> cells;
    for (size_t i = 0; i < count; ++i)
        cells[{ int(std::floor(boxes[i].center.x / cell)), int(std::floor(boxes[i].center.z / cell)) }].push_back(i);
    std::vector<JPH::BodyID> ids;
    impl_->box_cell = cell;
    for (const auto& kv : cells) {
        auto& bc = impl_->box_cells[kv.first];
        for (size_t i : kv.second) bc.boxes.push_back(boxes[i]);
    }
    for (const auto& kv : cells) {
        glm::vec3 org(0.0f);
        for (size_t i : kv.second) org += boxes[i].center;
        org /= float(kv.second.size());
        JPH::StaticCompoundShapeSettings cs;
        for (size_t i : kv.second) {
            glm::vec3 he = boxes[i].half_extents;
            float cr = calc_box_convex_radius(std::min({ he.x, he.y, he.z }));
            glm::vec3 p = boxes[i].center - org;
            cs.AddShape(JPH::Vec3(p.x, p.y, p.z), JPH::Quat::sIdentity(),
                        new JPH::BoxShapeSettings(JPH::Vec3(he.x, he.y, he.z), cr));
        }
        JPH::ShapeSettings::ShapeResult sr = cs.Create();
        if (sr.HasError()) { log::errorf("[jolt] compound: %s", sr.GetError().c_str()); continue; }
        JPH::BodyCreationSettings bcs(sr.Get(), JPH::RVec3(org.x, org.y, org.z), JPH::Quat::sIdentity(),
                                      JPH::EMotionType::Static, Layers::NON_MOVING);
        if (JPH::Body* body = bi.CreateBody(bcs)) {
            ids.push_back(body->GetID());
            impl_->box_cells[kv.first].body = body->GetID();
        }
    }
    if (ids.empty()) return;
    JPH::BodyInterface::AddState st = bi.AddBodiesPrepare(ids.data(), int(ids.size()));
    bi.AddBodiesFinalize(ids.data(), int(ids.size()), st, JPH::EActivation::DontActivate);
    log::infof("[jolt] %zu static boxes merged into %zu compound bodies", count, ids.size());
}

void PhysicsWorld::clear_static_box_compounds() {
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    for (auto& kv : impl_->box_cells)
        if (!kv.second.body.IsInvalid()) { bi.RemoveBody(kv.second.body); bi.DestroyBody(kv.second.body); }
    impl_->box_cells.clear();
}

size_t PhysicsWorld::remove_compound_boxes_at(const glm::vec2* xz, size_t n, float tol) {
    if (n == 0 || impl_->box_cells.empty()) return 0;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    const float cell = impl_->box_cell;
    size_t removed = 0;
    std::map<std::pair<int, int>, bool> dirty;
    for (size_t k = 0; k < n; ++k) {
        auto key = std::make_pair(int(std::floor(xz[k].x / cell)), int(std::floor(xz[k].y / cell)));
        auto it = impl_->box_cells.find(key);
        if (it == impl_->box_cells.end()) continue;
        auto& v = it->second.boxes;
        const size_t before = v.size();
        v.erase(std::remove_if(v.begin(), v.end(), [&](const StaticBox& b) {
                    float dx = b.center.x - xz[k].x, dz = b.center.z - xz[k].y;
                    return dx * dx + dz * dz < tol * tol;
                }), v.end());
        if (v.size() != before) { removed += before - v.size(); dirty[key] = true; }
    }
    for (const auto& kv : dirty) {
        auto& bc = impl_->box_cells[kv.first];
        if (!bc.body.IsInvalid()) { bi.RemoveBody(bc.body); bi.DestroyBody(bc.body); bc.body = JPH::BodyID(); }
        if (bc.boxes.empty()) continue;
        glm::vec3 org(0.0f);
        for (const auto& b : bc.boxes) org += b.center;
        org /= float(bc.boxes.size());
        JPH::StaticCompoundShapeSettings cs;
        for (const auto& b : bc.boxes) {
            float cr = calc_box_convex_radius(std::min({ b.half_extents.x, b.half_extents.y, b.half_extents.z }));
            glm::vec3 p = b.center - org;
            cs.AddShape(JPH::Vec3(p.x, p.y, p.z), JPH::Quat::sIdentity(),
                        new JPH::BoxShapeSettings(JPH::Vec3(b.half_extents.x, b.half_extents.y, b.half_extents.z), cr));
        }
        JPH::ShapeSettings::ShapeResult sr = cs.Create();
        if (sr.HasError()) continue;
        JPH::BodyCreationSettings bcs(sr.Get(), JPH::RVec3(org.x, org.y, org.z), JPH::Quat::sIdentity(),
                                      JPH::EMotionType::Static, Layers::NON_MOVING);
        bc.body = bi.CreateAndAddBody(bcs, JPH::EActivation::DontActivate);
    }
    return removed;
}

void PhysicsWorld::set_voxel_collision(const StaticBox* boxes, size_t count) {
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();

    // Hash-based early-out: the cadenced voxel-collision rebuild is
    // called every ~0.5 s while the player is near the tower, but the
    // box list often doesn't change between ticks (no carves, or carves
    // outside the rebuild radius). Hashing once and bailing skips the
    // RemoveBody+DestroyBody+AddShape×N storm — typically 2-5 ms saved
    // per avoided rebuild.
    size_t h = std::hash<size_t>{}(count);
    auto mix = [&](float v) {
        // Cheap commutativity-breaking mix; bit_cast keeps the entire
        // mantissa in the hash so two near-identical box configs don't
        // alias.
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        h ^= std::hash<uint32_t>{}(bits) + 0x9e3779b9 + (h << 6) + (h >> 2);
    };
    for (size_t i = 0; i < count; ++i) {
        mix(boxes[i].center.x);      mix(boxes[i].center.y);      mix(boxes[i].center.z);
        mix(boxes[i].half_extents.x); mix(boxes[i].half_extents.y); mix(boxes[i].half_extents.z);
    }
    if (h == impl_->voxel_shape_hash && !impl_->voxel_body.IsInvalid()) {
        return;     // geometry unchanged — skip rebuild
    }

    // Tear down the previous voxel body.
    if (!impl_->voxel_body.IsInvalid()) {
        bi.RemoveBody(impl_->voxel_body);
        bi.DestroyBody(impl_->voxel_body);
        impl_->voxel_body = JPH::BodyID();
    }
    if (count == 0) { impl_->voxel_shape_hash = h; return; }

    // Build each box shape, collect (shape, center). StaticCompoundShape
    // needs >= 2 sub-shapes; for 0/1 we special-case.
    std::vector<JPH::RefConst<JPH::Shape>> shapes;
    std::vector<JPH::Vec3> centers;
    shapes.reserve(count);
    centers.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        glm::vec3 he = boxes[i].half_extents;
        float min_he = std::min({he.x, he.y, he.z});
        float cr = calc_box_convex_radius(min_he);
        JPH::BoxShapeSettings ss(JPH::Vec3(he.x, he.y, he.z), cr);
        ss.SetEmbedded();
        JPH::ShapeSettings::ShapeResult sr = ss.Create();
        if (sr.HasError()) {
            log::errorf("[jolt] voxel box shape %zu: %s",
                        i, sr.GetError().c_str());
            continue;
        }
        shapes.push_back(sr.Get());
        centers.push_back(JPH::Vec3(boxes[i].center.x, boxes[i].center.y,
                                    boxes[i].center.z));
    }
    if (shapes.empty()) return;

    JPH::RefConst<JPH::Shape> final_shape;
    if (shapes.size() == 1) {
        // Single box: wrap nothing, place the body at the box center.
        final_shape = shapes[0];
        JPH::BodyCreationSettings bcs(final_shape, centers[0],
                                      JPH::Quat::sIdentity(),
                                      JPH::EMotionType::Static,
                                      Layers::VOXEL);
        if (JPH::Body* b = bi.CreateBody(bcs)) {
            bi.AddBody(b->GetID(), JPH::EActivation::DontActivate);
            impl_->voxel_body = b->GetID();
        }
        impl_->voxel_shape_hash = h;
        return;
    }
    JPH::StaticCompoundShapeSettings css;
    css.SetEmbedded();
    for (size_t i = 0; i < shapes.size(); ++i) {
        css.AddShape(centers[i], JPH::Quat::sIdentity(), shapes[i]);
    }
    JPH::ShapeSettings::ShapeResult cr = css.Create();
    if (cr.HasError()) {
        log::errorf("[jolt] voxel compound shape: %s", cr.GetError().c_str());
        return;
    }
    JPH::BodyCreationSettings bcs(cr.Get(), JPH::RVec3(0, 0, 0),
                                  JPH::Quat::sIdentity(),
                                  JPH::EMotionType::Static,
                                  Layers::VOXEL);
    if (JPH::Body* b = bi.CreateBody(bcs)) {
        bi.AddBody(b->GetID(), JPH::EActivation::DontActivate);
        impl_->voxel_body = b->GetID();
    }
    impl_->voxel_shape_hash = h;
}

void PhysicsWorld::add_static_heightfield(const float* samples, int dim,
                                           glm::vec2 origin_xz, float cell_size) {
    if (samples == nullptr || dim <= 0) return;
    // Jolt's HeightFieldShape requires the block size to:
    //   - be in the range [2, 8] (asserted by Jolt — outside that the
    //     shape Create() returns an error and the body is never added,
    //     so the player falls through forever);
    //   - divide mSampleCount = sample_count (= dim+1), not dim.
    // Pick the largest valid block size that divides sample_count.
    const int sample_count = dim + 1;
    int block_size = 2;
    for (int b = 8; b >= 2; b >>= 1) {
        if (sample_count % b == 0) { block_size = b; break; }
    }

    JPH::HeightFieldShapeSettings ss(
        samples, JPH::Vec3(origin_xz.x, 0.0f, origin_xz.y),
        JPH::Vec3(cell_size, 1.0f, cell_size),
        sample_count);
    ss.mBlockSize = static_cast<JPH::uint32>(block_size);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] heightfield shape: %s", sr.GetError().c_str());
        return;
    }
    JPH::BodyCreationSettings bcs(
        sr.Get(),
        JPH::RVec3(0.0f, 0.0f, 0.0f),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::NON_MOVING);
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    // Replace (not stack) a previous heightfield: sculpt-stroke refreshes
    // used to add a new body each time, leaving the old surface colliding.
    if (!impl_->hf_body.IsInvalid()) {
        bi.RemoveBody(impl_->hf_body);
        bi.DestroyBody(impl_->hf_body);
        impl_->hf_body = JPH::BodyID();
    }
    JPH::BodyID id = bi.CreateAndAddBody(bcs, JPH::EActivation::DontActivate);
    if (id.IsInvalid()) {
        log::error("[jolt] heightfield body create failed");
    } else {
        impl_->hf_body = id;
        impl_->hf_samples = sample_count;
        impl_->hf_block = block_size;
        log::infof("[jolt] heightfield: %dx%d samples, block=%d, cell=%.1fm",
                   sample_count, sample_count, block_size, cell_size);
    }
}

void PhysicsWorld::update_heightfield_region(const float* heights, int stride,
                                             int x0, int z0, int x1, int z1) {
    finish_async();
    if (impl_->hf_body.IsInvalid() || !heights) return;
    const int bs = impl_->hf_block, n = impl_->hf_samples;
    // SetHeights needs block-aligned start / size inside the sample grid.
    int ax0 = std::max(0, (x0 / bs) * bs), az0 = std::max(0, (z0 / bs) * bs);
    int ax1 = std::min(n, ((x1 + bs) / bs) * bs), az1 = std::min(n, ((z1 + bs) / bs) * bs);
    if (ax1 <= ax0 || az1 <= az0) return;
    const int w = ax1 - ax0, h = az1 - az0;
    std::vector<float> region(size_t(w) * size_t(h));
    for (int z = 0; z < h; ++z)
        std::memcpy(region.data() + size_t(z) * size_t(w),
                    heights + size_t(az0 + z) * size_t(stride) + size_t(ax0), size_t(w) * sizeof(float));
    {
        JPH::BodyLockWrite lock(impl_->system.GetBodyLockInterface(), impl_->hf_body);
        if (!lock.Succeeded()) return;
        JPH::Body& body = lock.GetBody();
        auto* hf = static_cast<JPH::HeightFieldShape*>(const_cast<JPH::Shape*>(body.GetShape()));
        hf->SetHeights(JPH::uint(ax0), JPH::uint(az0), JPH::uint(w), JPH::uint(h),
                       region.data(), w, impl_->temp_alloc);
    }
    impl_->system.GetBodyInterface().NotifyShapeChanged(impl_->hf_body, JPH::Vec3::sZero(), false,
                                                        JPH::EActivation::DontActivate);
}

uint32_t PhysicsWorld::add_dynamic_box(glm::vec3 c, glm::vec3 he,
                                       glm::vec3 euler_rad, float density) {
    float min_he = std::min({he.x, he.y, he.z});
    float convex_radius = calc_box_convex_radius(min_he);
    JPH::BoxShapeSettings ss(JPH::Vec3(he.x, he.y, he.z), convex_radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] dyn box shape: %s", sr.GetError().c_str());
        return 0;
    }
    glm::quat gq = glm::quat(euler_rad);
    JPH::Quat jq(gq.x, gq.y, gq.z, gq.w);

    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z), jq,
                                  JPH::EMotionType::Dynamic,
                                  Layers::MOVING);
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateMassAndInertia;
    bcs.mMassPropertiesOverride.mMass = density * he.x * he.y * he.z * 8.0f;

    JPH::BodyID bid = impl_->system.GetBodyInterface().CreateAndAddBody(
        bcs, JPH::EActivation::Activate);
    uint32_t id = impl_->next_id++;
    impl_->ids.emplace(id, bid);
    impl_->id_by_body_index[bid.GetIndex()] = id;
    return id;
}

uint32_t PhysicsWorld::add_dynamic_cylinder_ccd(glm::vec3 c, float radius,
                                                float half_length,
                                                glm::quat orientation,
                                                glm::vec3 linvel, float mass) {
    // Jolt's CylinderShape requires its convex radius to be smaller than both
    // the cylinder's radius and half-length. Default convex radius is 0.05;
    // for a 6-cm bullet that's already too large.
    float convex_radius = std::min({0.02f, radius * 0.4f, half_length * 0.4f});
    JPH::CylinderShapeSettings ss(half_length, radius, convex_radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] dyn cylinder shape: %s", sr.GetError().c_str());
        return 0;
    }
    JPH::Quat jq(orientation.x, orientation.y, orientation.z, orientation.w);

    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z), jq,
                                  JPH::EMotionType::Dynamic,
                                  Layers::BULLET);
    // *** Industry-standard CCD for fast small bodies. *** Jolt sweep-casts
    // the cylinder along its motion at the start of each step and reports the
    // first contact instead of integrating-then-resolving. Eliminates tunnel-
    // through against any thicker target.
    bcs.mMotionQuality = JPH::EMotionQuality::LinearCast;
    bcs.mLinearVelocity = JPH::Vec3(linvel.x, linvel.y, linvel.z);
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = mass;
    // Suppress bullet tumble: a thin cylinder has highly anisotropic inertia,
    // so any glancing contact spins it up enough that subsequent flight
    // veers off the firing axis. Strong angular drag dissipates that spin
    // each step; linear CCD behaviour is unaffected.
    bcs.mAngularDamping = 2.0f;
    // Jolt's per-body default cap is 500 m/s — anything above asserts/clamps
    // on creation. Raise it so the bullet-speed slider's 600 m/s upper bound
    // lands inside the legal range with comfortable headroom.
    bcs.mMaxLinearVelocity = 1000.0f;

    JPH::BodyID bid = impl_->system.GetBodyInterface().CreateAndAddBody(
        bcs, JPH::EActivation::Activate);
    uint32_t id = impl_->next_id++;
    impl_->ids.emplace(id, bid);
    impl_->id_by_body_index[bid.GetIndex()] = id;
    return id;
}

uint32_t PhysicsWorld::add_dynamic_sphere(glm::vec3 c, float radius,
                                          glm::vec3 linvel, float mass,
                                          float restitution, float friction) {
    JPH::SphereShapeSettings ss(radius);
    ss.SetEmbedded();
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) {
        log::errorf("[jolt] sphere shape: %s", sr.GetError().c_str());
        return 0;
    }
    JPH::BodyCreationSettings bcs(sr.Get(),
                                  JPH::RVec3(c.x, c.y, c.z),
                                  JPH::Quat::sIdentity(),
                                  JPH::EMotionType::Dynamic,
                                  Layers::SPARK);
    bcs.mLinearVelocity = JPH::Vec3(linvel.x, linvel.y, linvel.z);
    bcs.mRestitution = restitution;
    bcs.mFriction = friction;
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = mass;
    // Particles are tiny + numerous — Discrete motion quality is fine and
    // far cheaper than the LinearCast path used for bullets.
    bcs.mMotionQuality = JPH::EMotionQuality::Discrete;

    JPH::BodyID bid = impl_->system.GetBodyInterface().CreateAndAddBody(
        bcs, JPH::EActivation::Activate);
    uint32_t id = impl_->next_id++;
    impl_->ids.emplace(id, bid);
    impl_->id_by_body_index[bid.GetIndex()] = id;
    return id;
}

uint32_t PhysicsWorld::add_ragdoll(const RagdollPart* parts, size_t n, glm::vec3 linvel,
                                   const glm::vec3* floor_point, glm::vec3 floor_normal) {
    if (n == 0 || n > 32) return 0;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    Impl::Ragdoll rg;
    if (floor_point) {
        const glm::vec3 nrm = glm::normalize(floor_normal);
        const glm::vec3 ax = glm::cross(glm::vec3(0, 1, 0), nrm);
        const float sn = glm::length(ax);
        const glm::quat q = sn < 1e-5f ? glm::quat(1, 0, 0, 0) : glm::angleAxis(std::atan2(sn, nrm.y), ax / sn);
        const float thick = 0.5f;
        JPH::BoxShapeSettings fs(JPH::Vec3(8.0f, thick, 8.0f), 0.02f);
        fs.SetEmbedded();
        JPH::ShapeSettings::ShapeResult fr = fs.Create();
        if (!fr.HasError()) {
            const glm::vec3 c = *floor_point - nrm * thick;
            JPH::BodyCreationSettings fb(fr.Get(), JPH::RVec3(c.x, c.y, c.z), JPH::Quat(q.x, q.y, q.z, q.w),
                                         JPH::EMotionType::Static, Layers::NON_MOVING);
            fb.mFriction = 0.9f;
            if (JPH::Body* b = bi.CreateBody(fb)) { bi.AddBody(b->GetID(), JPH::EActivation::DontActivate); rg.floor = b->GetID(); }
        }
    }
    const uint32_t id = impl_->next_id++;
    JPH::Ref<JPH::GroupFilterTable> filter = new JPH::GroupFilterTable(static_cast<JPH::uint>(n));
    // No part collides with another of the same body: in a living pose they overlap (forearms
    // across the chest, thighs side by side) and would be thrown apart. The joint limits keep the shape.
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            filter->DisableCollision(static_cast<JPH::CollisionGroup::SubGroupID>(i), static_cast<JPH::CollisionGroup::SubGroupID>(j));
    auto perp = [](glm::vec3 d) {
        const glm::vec3 o = std::abs(d.y) < 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        return glm::normalize(glm::cross(d, o));
    };
    for (size_t i = 0; i < n; ++i) {
        const RagdollPart& p = parts[i];
        glm::vec3 d = p.b - p.a;
        const float len = std::max(glm::length(d), 1e-3f);
        d /= len;
        // the capsule's axis is its local y
        const glm::vec3 ax = glm::cross(glm::vec3(0, 1, 0), d);
        const float sn = glm::length(ax), cs = d.y;
        const glm::quat q = sn < 1e-5f ? (cs > 0.0f ? glm::quat(1, 0, 0, 0) : glm::quat(0, 1, 0, 0))
                                       : glm::angleAxis(std::atan2(sn, cs), ax / sn);
        const float rad = std::min(p.radius, len * 0.49f);
        JPH::CapsuleShapeSettings ss(std::max(len * 0.5f - rad, 0.005f), rad);
        ss.SetEmbedded();
        JPH::ShapeSettings::ShapeResult sr = ss.Create();
        if (sr.HasError()) { log::errorf("[jolt] ragdoll shape: %s", sr.GetError().c_str()); rg.bodies.push_back(JPH::BodyID()); continue; }
        const glm::vec3 c = (p.a + p.b) * 0.5f;
        JPH::BodyCreationSettings bcs(sr.Get(), JPH::RVec3(c.x, c.y, c.z), JPH::Quat(q.x, q.y, q.z, q.w),
                                      JPH::EMotionType::Dynamic, Layers::MOVING);
        bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        bcs.mMassPropertiesOverride.mMass = p.mass;
        bcs.mLinearVelocity = JPH::Vec3(linvel.x, linvel.y, linvel.z);
        bcs.mLinearDamping = 0.08f;
        bcs.mAngularDamping = 0.55f;
        bcs.mFriction = 0.9f;
        bcs.mRestitution = 0.05f;
        bcs.mMotionQuality = JPH::EMotionQuality::LinearCast;        // (a thrown limb must not pass through the ground)
        bcs.mCollisionGroup = JPH::CollisionGroup(filter, id, static_cast<JPH::CollisionGroup::SubGroupID>(i));
        JPH::Body* body = bi.CreateBody(bcs);
        if (!body) { rg.bodies.push_back(JPH::BodyID()); continue; }       // (the body limit)
        bi.AddBody(body->GetID(), JPH::EActivation::Activate);
        rg.bodies.push_back(body->GetID());
    }
    for (size_t i = 0; i < n; ++i) {
        const RagdollPart& p = parts[i];
        if (p.parent < 0 || size_t(p.parent) >= i || rg.bodies[i].IsInvalid() || rg.bodies[p.parent].IsInvalid()) continue;
        const glm::vec3 d = glm::normalize(p.b - p.a + glm::vec3(0.0f, 1e-6f, 0.0f));
        const JPH::RVec3 at(p.a.x, p.a.y, p.a.z);
        JPH::Ref<JPH::Constraint> c;
        if (p.hinge) {
            glm::vec3 ax = glm::normalize(p.axis);
            glm::vec3 nr = d - ax * glm::dot(d, ax);
            nr = glm::dot(nr, nr) < 1e-6f ? perp(ax) : glm::normalize(nr);
            JPH::HingeConstraintSettings hs;
            hs.mPoint1 = hs.mPoint2 = at;
            hs.mHingeAxis1 = hs.mHingeAxis2 = JPH::Vec3(ax.x, ax.y, ax.z);
            hs.mNormalAxis1 = hs.mNormalAxis2 = JPH::Vec3(nr.x, nr.y, nr.z);
            hs.mLimitsMin = std::min(p.lim_min, 0.0f);
            hs.mLimitsMax = std::max(p.lim_max, 0.0f);
            hs.mMaxFrictionTorque = 4.0f;
            JPH::Body* b1 = impl_->system.GetBodyLockInterfaceNoLock().TryGetBody(rg.bodies[p.parent]);
            JPH::Body* b2 = impl_->system.GetBodyLockInterfaceNoLock().TryGetBody(rg.bodies[i]);
            if (b1 && b2) c = hs.Create(*b1, *b2);
        } else {
            const glm::vec3 pl = perp(d);
            JPH::SwingTwistConstraintSettings st;
            st.mPosition1 = st.mPosition2 = at;
            st.mTwistAxis1 = st.mTwistAxis2 = JPH::Vec3(d.x, d.y, d.z);
            st.mPlaneAxis1 = st.mPlaneAxis2 = JPH::Vec3(pl.x, pl.y, pl.z);
            st.mNormalHalfConeAngle = st.mPlaneHalfConeAngle = p.cone;
            st.mTwistMinAngle = -0.5f;
            st.mTwistMaxAngle = 0.5f;
            st.mMaxFrictionTorque = 4.0f;
            JPH::Body* b1 = impl_->system.GetBodyLockInterfaceNoLock().TryGetBody(rg.bodies[p.parent]);
            JPH::Body* b2 = impl_->system.GetBodyLockInterfaceNoLock().TryGetBody(rg.bodies[i]);
            if (b1 && b2) c = st.Create(*b1, *b2);
        }
        if (c) { impl_->system.AddConstraint(c); rg.joints.push_back(c); }
    }
    impl_->ragdolls.emplace(id, std::move(rg));
    return id;
}

bool PhysicsWorld::ragdoll_part(uint32_t id, size_t part, glm::vec3& pos, glm::quat& rot) const {
    auto it = impl_->ragdolls.find(id);
    if (it == impl_->ragdolls.end() || part >= it->second.bodies.size() || it->second.bodies[part].IsInvalid()) return false;
    JPH::RVec3 p;
    JPH::Quat q;
    impl_->system.GetBodyInterface().GetPositionAndRotation(it->second.bodies[part], p, q);
    pos = glm::vec3(float(p.GetX()), float(p.GetY()), float(p.GetZ()));
    rot = glm::quat(q.GetW(), q.GetX(), q.GetY(), q.GetZ());
    return true;
}

void PhysicsWorld::ragdoll_impulse(uint32_t id, size_t part, glm::vec3 impulse, glm::vec3 at) {
    auto it = impl_->ragdolls.find(id);
    if (it == impl_->ragdolls.end() || part >= it->second.bodies.size() || it->second.bodies[part].IsInvalid()) return;
    impl_->system.GetBodyInterface().AddImpulse(it->second.bodies[part], JPH::Vec3(impulse.x, impulse.y, impulse.z),
                                                JPH::RVec3(at.x, at.y, at.z));
}

bool PhysicsWorld::ragdoll_active(uint32_t id) const {
    auto it = impl_->ragdolls.find(id);
    if (it == impl_->ragdolls.end()) return false;
    for (const JPH::BodyID& b : it->second.bodies)
        if (!b.IsInvalid() && impl_->system.GetBodyInterface().IsActive(b)) return true;
    return false;
}

void PhysicsWorld::remove_ragdoll(uint32_t id) {
    auto it = impl_->ragdolls.find(id);
    if (it == impl_->ragdolls.end()) return;
    for (auto& c : it->second.joints) impl_->system.RemoveConstraint(c);
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    if (!it->second.floor.IsInvalid()) {
        if (bi.IsAdded(it->second.floor)) bi.RemoveBody(it->second.floor);
        bi.DestroyBody(it->second.floor);
    }
    for (const JPH::BodyID& b : it->second.bodies) {
        if (b.IsInvalid()) continue;
        if (bi.IsAdded(b)) bi.RemoveBody(b);
        bi.DestroyBody(b);
    }
    impl_->ragdolls.erase(it);
}

void PhysicsWorld::remove_body(uint32_t id) {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    // (a body that could not be created - the body limit - has an invalid id: destroying it crashed)
    if (!it->second.IsInvalid()) {
        if (bi.IsAdded(it->second)) bi.RemoveBody(it->second);
        bi.DestroyBody(it->second);
        impl_->id_by_body_index.erase(it->second.GetIndex());
    }
    impl_->ids.erase(it);
}

PhysicsWorld::RayHit PhysicsWorld::raycast(glm::vec3 origin, glm::vec3 direction,
                                           float max_distance) const {
    RayHit out{};
    JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                      JPH::Vec3(direction.x, direction.y, direction.z) * max_distance);
    JPH::RayCastResult res;
    // Rays see the world, not the shots flying through it: without this a
    // bolt's impact point (re-cast along its aim line) landed on the NEXT
    // bolt still in flight on the same line - sparks and scorch marks in
    // mid-air in front of the target when firing fast.
    struct WorldOnly final : JPH::ObjectLayerFilter {
        bool ShouldCollide(JPH::ObjectLayer l) const override { return l != Layers::BULLET && l != Layers::SPARK; }
    } world_only;
    bool hit = impl_->system.GetNarrowPhaseQuery().CastRay(ray, res, JPH::BroadPhaseLayerFilter(), world_only);
    if (!hit) return out;

    out.hit = true;
    out.distance = res.mFraction * max_distance;
    out.position = origin + direction * out.distance;

    // BodyLockRead uses GetBodyLockInterface directly — no BodyInterface
    // needed. The previous `GetBodyInterface()` acquire took a mutex per
    // raycast for nothing (the `(void)bi` was just suppressing the unused
    // warning).
    JPH::BodyLockRead lock(impl_->system.GetBodyLockInterface(), res.mBodyID);
    if (lock.Succeeded()) {
        const JPH::Body& body = lock.GetBody();
        JPH::Vec3 n = body.GetWorldSpaceSurfaceNormal(res.mSubShapeID2,
                                                      JPH::RVec3(out.position.x,
                                                                 out.position.y,
                                                                 out.position.z));
        out.normal = glm::vec3(n.GetX(), n.GetY(), n.GetZ());
        if (body.GetMotionType() == JPH::EMotionType::Dynamic) {
            out.dynamic = true;
            // O(1) lookup via reverse map (was a linear walk over `ids`).
            auto rit = impl_->id_by_body_index.find(res.mBodyID.GetIndex());
            if (rit != impl_->id_by_body_index.end()) out.body_id = rit->second;
        }
    }
    return out;
}

void PhysicsWorld::apply_impulse(uint32_t id, glm::vec3 impulse) {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return;
    // Single-threaded mutations between Update() — same justification as
    // get_linear_velocity / get_body_world_matrix above; no body-mutex
    // needed.
    impl_->system.GetBodyInterfaceNoLock().AddImpulse(
        it->second, JPH::Vec3(impulse.x, impulse.y, impulse.z));
}

// Read-only queries use GetBodyInterfaceNoLock(): all our physics
// mutations (Update, AddImpulse, remove_body, spawn) happen on the main
// thread between Update() calls, so no other thread is racing with these
// reads. GetBodyInterface() takes a body mutex per call, which at 100+
// bodies × multiple queries/frame is a measurable fraction of CPU time.

glm::vec3 PhysicsWorld::get_linear_velocity(uint32_t id) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return glm::vec3(0.0f);
    JPH::Vec3 v = impl_->system.GetBodyInterfaceNoLock().GetLinearVelocity(it->second);
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

bool PhysicsWorld::is_body_active(uint32_t id) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return false;
    return impl_->system.GetBodyInterfaceNoLock().IsActive(it->second);
}

bool PhysicsWorld::get_body_world_matrix(uint32_t id, glm::mat4& out) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return false;
    // Single GetWorldTransform call returns position+rotation as one
    // RMat44, vs separate GetPosition + GetRotation which lock the body
    // mutex twice.
    JPH::RMat44 m = impl_->system.GetBodyInterfaceNoLock().GetWorldTransform(it->second);
    out = glm::mat4(
        m(0,0), m(1,0), m(2,0), m(3,0),
        m(0,1), m(1,1), m(2,1), m(3,1),
        m(0,2), m(1,2), m(2,2), m(3,2),
        m(0,3), m(1,3), m(2,3), m(3,3));
    return true;
}

// --- BodyHandle fast-path: caller stores the cached BodyID once, all
// subsequent reads skip the unordered_map.find. JPH::BodyID's underlying
// storage is a uint32 (index + sequence), so we can just round-trip it.

PhysicsWorld::BodyHandle PhysicsWorld::handle_of(uint32_t id) const {
    auto it = impl_->ids.find(id);
    if (it == impl_->ids.end()) return 0;
    return it->second.GetIndexAndSequenceNumber();
}

bool PhysicsWorld::get_body_world_matrix_h(BodyHandle h, glm::mat4& out) const {
    if (h == 0) return false;
    JPH::BodyID bid(h);
    // A removed body (evicted crate, expired spark) must report false:
    // GetWorldTransform returns identity for it, which callers drew at the
    // world origin (decals) instead of reaping. IsAdded also rejects a
    // stale handle whose slot was reused (sequence number mismatch).
    if (!impl_->system.GetBodyInterfaceNoLock().IsAdded(bid)) return false;
    JPH::RMat44 m = impl_->system.GetBodyInterfaceNoLock().GetWorldTransform(bid);
    out = glm::mat4(
        m(0,0), m(1,0), m(2,0), m(3,0),
        m(0,1), m(1,1), m(2,1), m(3,1),
        m(0,2), m(1,2), m(2,2), m(3,2),
        m(0,3), m(1,3), m(2,3), m(3,3));
    return true;
}

bool PhysicsWorld::is_body_active_h(BodyHandle h) const {
    if (h == 0) return false;
    return impl_->system.GetBodyInterfaceNoLock().IsActive(JPH::BodyID(h));
}

glm::vec3 PhysicsWorld::get_linear_velocity_h(BodyHandle h) const {
    if (h == 0) return glm::vec3(0.0f);
    JPH::Vec3 v = impl_->system.GetBodyInterfaceNoLock().GetLinearVelocity(JPH::BodyID(h));
    return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
}

void PhysicsWorld::apply_impulse_h(BodyHandle h, glm::vec3 impulse) {
    if (h == 0) return;
    impl_->system.GetBodyInterfaceNoLock().AddImpulse(
        JPH::BodyID(h), JPH::Vec3(impulse.x, impulse.y, impulse.z));
}

// =====================================================================================
// The vehicle (PhysicsWorld::vehicle_*): Jolt's wheeled vehicle - a rigid body on four
// suspended wheels, with an engine, a clutch, a gearbox, two differentials and tyres whose
// grip falls off as they slip. One vehicle at a time.
// =====================================================================================
struct PhysicsWorld::VehicleImpl {
    // The air on the car, every physics step: downforce along the car's own down, drag against its motion.
    struct Aero : JPH::PhysicsStepListener {
        JPH::BodyID body; float down = 0.0f, drag = 0.0f;
        void OnStep(const JPH::PhysicsStepListenerContext& c) override {
            JPH::BodyInterface& bi = c.mPhysicsSystem->GetBodyInterfaceNoLock();
            if (body.IsInvalid() || !bi.IsActive(body)) return;
            const JPH::Vec3 v = bi.GetLinearVelocity(body);
            const float sp = v.Length();
            if (sp < 1.0f) return;
            const JPH::Quat q = bi.GetRotation(body);
            const float vf = v.Dot(q * JPH::Vec3(0, 0, -1));
            bi.AddForce(body, -(q * JPH::Vec3(0, 1, 0)) * (down * vf * vf) - v * (drag * sp));
        }
    };
    // Where it was before the last physics step: the picture is drawn between the two
    // (vehicle_state_prev). Not valid across a teleport.
    PhysicsWorld::VehicleState prev{};
    bool prev_valid = false;
    Aero aero;
    JPH::BodyID body;
    JPH::Ref<JPH::VehicleConstraint> constraint;

    JPH::Ref<JPH::VehicleCollisionTester> tester;
    JPH::BodyID patch;                 // the ground under it, outside the home tile's heightfield
    JPH::BodyID obstacles;             // what stands near it (trees, rails, buildings)
    // The wheels' ground: everything but the obstacles. A wheel is a cylinder swept down its
    // suspension; beside a guard rail or a wall it found the rail's side as "ground" - a
    // contact on a vertical face, with the tyre's whole grip pushing into it. The car hung on
    // the rail it was sliding along. What stands beside the road is for the body to hit.
    struct WheelGround : JPH::BodyFilter {
        JPH::BodyID skip, self;       // (self: the car's own body - the tester's own filter, which this one replaces, left it out)
        bool ShouldCollide(const JPH::BodyID& id) const override { return id != skip && id != self; }
        bool ShouldCollideLocked(const JPH::Body& b) const override { return b.GetID() != skip && b.GetID() != self; }
    };
    WheelGround wheel_ground;
    std::vector<float> patch_samples;  // (Jolt keeps a pointer for the shape's life only: kept anyway)
};

uint32_t PhysicsWorld::vehicle_create(const VehicleDesc& d) {
    vehicle_remove();
    vehicle_ = std::make_unique<VehicleImpl>();
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    // The body: a few boxes, the centre of mass low between the axles.
    JPH::StaticCompoundShapeSettings cs;
    for (const StaticBox& b : d.boxes)
        cs.AddShape(JPH::Vec3(b.center.x, b.center.y, b.center.z), JPH::Quat::sIdentity(),
                    new JPH::BoxShape(JPH::Vec3(b.half_extents.x, b.half_extents.y, b.half_extents.z), 0.03f));
    // Over each axle a box out to the tyres' outer faces: on its side or its roof the car lies
    // on its wheels and arches, not on the narrower tub with the wheels sunk in the ground.
    for (int ax = 0; ax < 2; ++ax) {
        const glm::vec3 wl = d.wheel_pos[ax * 2], wr = d.wheel_pos[ax * 2 + 1];
        const float half_w = 0.5f * std::abs(wr.x - wl.x) + 0.5f * d.wheel_width;
        cs.AddShape(JPH::Vec3(0.0f, wl.y + 0.24f, 0.5f * (wl.z + wr.z)), JPH::Quat::sIdentity(),
                    new JPH::BoxShape(JPH::Vec3(half_w, 0.20f, d.wheel_radius * 0.95f), 0.03f));
    }
    JPH::ShapeSettings::ShapeResult sr0 = cs.Create();
    if (sr0.HasError()) { log::errorf("[jolt] vehicle shape: %s", sr0.GetError().c_str()); vehicle_.reset(); return 0; }
    // The centre of mass where a car's is: low, between the axles. The boxes' own is that of
    // a solid block of their shape - too high and too far forward: the car rolled about a
    // point at window height and tipped over far too easily.
    const float z_f0 = d.wheel_pos[0].z, z_r0 = d.wheel_pos[2].z;
    const JPH::Vec3 com_want(0.0f, d.com_height, z_r0 + (z_f0 - z_r0) * d.front_weight);
    JPH::OffsetCenterOfMassShapeSettings ocs(com_want - sr0.Get()->GetCenterOfMass(), sr0.Get());
    JPH::ShapeSettings::ShapeResult sr = ocs.Create();
    if (sr.HasError()) { log::errorf("[jolt] vehicle shape: %s", sr.GetError().c_str()); vehicle_.reset(); return 0; }
    JPH::BodyCreationSettings bcs(sr.Get(), JPH::RVec3(d.position.x, d.position.y, d.position.z),
                                  JPH::Quat::sRotation(JPH::Vec3::sAxisY(), d.yaw), JPH::EMotionType::Dynamic, Layers::MOVING);
    bcs.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = d.mass;
    bcs.mLinearDamping = 0.02f; bcs.mAngularDamping = 0.10f;
    bcs.mFriction = 0.6f; bcs.mRestitution = 0.12f;
    bcs.mMotionQuality = JPH::EMotionQuality::LinearCast;
    // (a rail is a row of boxes end to end, the body a set of boxes too: without this a corner
    //  catches on the joint between two of them as if it were a step)
    bcs.mEnhancedInternalEdgeRemoval = true;
    bcs.mAllowSleeping = true;
    // The game's gravity is 25 m/s2 (for the feel of the player's jumps). A vehicle built from
    // real numbers - mass, springs, tyres, engine - only behaves like one under the Earth's:
    // at 25 it sat on its bump stops, cornered at 2.5 g and stalled on the slightest hill.
    // So the vehicle falls at 9.81, and everything below is sized for that.
    const float grav = 9.81f;
    bcs.mGravityFactor = grav / std::max(impl_->system.GetGravity().Length(), 0.1f);
    JPH::Body* body = bi.CreateBody(bcs);
    if (!body) { vehicle_.reset(); return 0; }
    // (the centre of mass: where the engine, gearbox and frame are - low, a little forward)
    {
        JPH::MotionProperties* mp = body->GetMotionProperties();
        (void)mp;
    }
    bi.AddBody(body->GetID(), JPH::EActivation::Activate);
    vehicle_->body = body->GetID();

    JPH::VehicleConstraintSettings vs;
    vs.mUp = JPH::Vec3(0, 1, 0);
    vs.mForward = JPH::Vec3(0, 0, -1);                  // (the model's front is -z)
    vs.mMaxPitchRollAngle = JPH::JPH_PI;                // it may roll over
    const float com_z = sr.Get()->GetCenterOfMass().GetZ();
    for (int i = 0; i < 4; ++i) {
        JPH::WheelSettingsWV* w = new JPH::WheelSettingsWV;
        const bool front = i < 2;
        w->mPosition = JPH::Vec3(d.wheel_pos[i].x, d.wheel_pos[i].y + d.suspension_rest, d.wheel_pos[i].z);
        w->mSuspensionDirection = JPH::Vec3(0, -1, 0);
        w->mSteeringAxis = JPH::Vec3(0, 1, 0);
        w->mWheelUp = JPH::Vec3(0, 1, 0);
        w->mWheelForward = JPH::Vec3(0, 0, -1);
        // The spring, from the weight this wheel carries (the body's centre of mass is not
        // midway between the axles): stiff enough to swing at suspension_hz under that
        // weight, and as long as it takes to stand at the rest length under it. (Jolt's
        // frequency setting uses the mass the constraint sees at the wheel, which is much
        // less than the weight on it: the springs sat on their stops.)
        const float z_f = d.wheel_pos[0].z, z_r = d.wheel_pos[2].z;
        const float share_f = glm::clamp((z_r - com_z) / (z_r - z_f), 0.3f, 0.7f);
        const float m_w = 0.5f * d.mass * (front ? share_f : 1.0f - share_f);
        // (The game's gravity is not the Earth's - 25 m/s2, for the feel of the jumps - so the
        //  weight is taken from the physics, not from 9.81. The spring is stiffer by as much:
        //  the same sag as on Earth at suspension_hz, which is what the eye judges.)
        const float om = 6.2831853f * d.suspension_hz * std::sqrt(grav / 9.81f);
        const float sag = grav / (om * om);
        w->mSuspensionMinLength = d.suspension_rest - d.suspension_travel_up;
        w->mSuspensionMaxLength = d.suspension_rest + sag;
        w->mSuspensionSpring.mMode = JPH::ESpringMode::StiffnessAndDamping;
        w->mSuspensionSpring.mStiffness = m_w * om * om;
        w->mSuspensionSpring.mDamping = 2.0f * d.suspension_damping * m_w * om;
        w->mRadius = d.wheel_radius;
        w->mWidth = d.wheel_width;
        w->mInertia = 0.5f * d.wheel_mass * d.wheel_radius * d.wheel_radius;
        w->mAngularDamping = 0.25f;
        w->mMaxSteerAngle = front ? d.max_steer : 0.0f;
        w->mMaxBrakeTorque = front ? d.brake_torque : d.brake_torque * 0.6f;
        w->mMaxHandBrakeTorque = front ? 0.0f : d.handbrake_torque;
        // An all-terrain tyre: peak grip a little past 6 % slip, then it lets go.
        // (Grip is friction x weight, and the weight here is 2.5 x the Earth's: with a real
        //  tyre's friction the jeep cornered at 2.5 g and never slid. The friction is scaled so
        //  that the grip is a real tyre's in m/s2.)
        const float gk = d.grip * 9.81f / grav;
        // The tyre, as measured ones are (the "magic formula" shape): grip rises steeply with
        // slip, peaks - at about 12 % along, 8 degrees across - and then falls off only
        // gently: a sliding tyre still holds four fifths of its best. (It fell to half: once
        // past the peak the car ploughed straight on, whatever the wheel did.)
        w->mLongitudinalFriction.Clear();
        w->mLongitudinalFriction.AddPoint(0.00f, 0.00f);
        w->mLongitudinalFriction.AddPoint(0.04f, 0.80f * gk);
        w->mLongitudinalFriction.AddPoint(0.12f, 1.12f * gk);
        w->mLongitudinalFriction.AddPoint(0.30f, 1.00f * gk);
        w->mLongitudinalFriction.AddPoint(1.00f, 0.86f * gk);
        w->mLateralFriction.Clear();
        w->mLateralFriction.AddPoint(0.0f, 0.00f);
        w->mLateralFriction.AddPoint(2.0f, 0.62f * gk);
        w->mLateralFriction.AddPoint(5.0f, 1.00f * gk);
        w->mLateralFriction.AddPoint(8.0f, 1.08f * gk);
        w->mLateralFriction.AddPoint(14.0f, 1.02f * gk);
        w->mLateralFriction.AddPoint(30.0f, 0.92f * gk);
        w->mLateralFriction.AddPoint(90.0f, 0.82f * gk);
        (void)front;
        vs.mWheels.push_back(w);
    }
    // Anti-roll bars, front and rear.
    { JPH::VehicleAntiRollBar b; b.mLeftWheel = 0; b.mRightWheel = 1; b.mStiffness = d.antiroll; vs.mAntiRollBars.push_back(b); }
    { JPH::VehicleAntiRollBar b; b.mLeftWheel = 2; b.mRightWheel = 3; b.mStiffness = d.antiroll * 0.7f; vs.mAntiRollBars.push_back(b); }
    JPH::WheeledVehicleControllerSettings* cs2 = new JPH::WheeledVehicleControllerSettings;
    cs2->mEngine.mMaxTorque = d.engine_torque;
    cs2->mEngine.mMinRPM = 800.0f;
    cs2->mEngine.mMaxRPM = d.engine_max_rpm;
    cs2->mEngine.mInertia = 0.35f;
    cs2->mEngine.mNormalizedTorque.Clear();              // a low-revving four: strongest in the middle
    cs2->mEngine.mNormalizedTorque.AddPoint(0.0f, 0.72f);
    cs2->mEngine.mNormalizedTorque.AddPoint(0.35f, 1.0f);
    cs2->mEngine.mNormalizedTorque.AddPoint(0.75f, 0.92f);
    cs2->mEngine.mNormalizedTorque.AddPoint(1.0f, 0.60f);
    cs2->mTransmission.mMode = JPH::ETransmissionMode::Auto;
    cs2->mTransmission.mGearRatios = { 3.60f, 2.10f, 1.40f, 1.00f };
    if (!d.gears.empty()) { cs2->mTransmission.mGearRatios.clear(); for (float g : d.gears) cs2->mTransmission.mGearRatios.push_back(g); }
    cs2->mTransmission.mReverseGearRatios = { -3.40f };
    cs2->mTransmission.mSwitchTime = 0.35f;
    cs2->mTransmission.mClutchReleaseTime = 0.25f;
    cs2->mTransmission.mSwitchLatency = 0.4f;
    cs2->mTransmission.mShiftUpRPM = d.engine_max_rpm * 0.74f;
    cs2->mTransmission.mShiftDownRPM = d.engine_max_rpm * 0.36f;
    cs2->mTransmission.mClutchStrength = 12.0f;
    // Four-wheel drive: a differential on each axle, 40 % of the torque to the front.
    cs2->mDifferentials.resize(2);
    cs2->mDifferentials[0].mLeftWheel = 0; cs2->mDifferentials[0].mRightWheel = 1;
    cs2->mDifferentials[0].mDifferentialRatio = d.final_drive; cs2->mDifferentials[0].mEngineTorqueRatio = d.front_drive;
    cs2->mDifferentials[0].mLimitedSlipRatio = 1.6f;
    cs2->mDifferentials[1].mLeftWheel = 2; cs2->mDifferentials[1].mRightWheel = 3;
    cs2->mDifferentials[1].mDifferentialRatio = d.final_drive; cs2->mDifferentials[1].mEngineTorqueRatio = 1.0f - d.front_drive;
    cs2->mDifferentials[1].mLimitedSlipRatio = 1.6f;
    cs2->mDifferentialLimitedSlipRatio = 1.5f;
    vs.mController = cs2;

    vehicle_->constraint = new JPH::VehicleConstraint(*body, vs);
    // The tyre's curves ARE the grip. Jolt's default takes the mean of the tyre's friction and
    // that of the body it stands on - made for contacts between solids; on the home tile's
    // heightfield (friction 0.2 by default) it left a tyre with under half its grip.
    vehicle_->constraint->SetCombineFriction([](JPH::uint, float&, float&, const JPH::Body&, const JPH::SubShapeID&) {});
    // One tyre has one grip to spend: what it uses along (spinning under power, locked under
    // the brake) it has not across. (Jolt limits the two separately: a wheel spinning freely
    // still held the car sideways at full strength - no power slide, no locked-wheel plough.)
    static_cast<JPH::WheeledVehicleController*>(vehicle_->constraint->GetController())->SetTireMaxImpulseCallback(
        [](JPH::uint, float& out_long, float& out_lat, float susp, float f_long, float f_lat, float slip_long, float, float) {
            out_long = susp * f_long;
            const float spent = std::clamp((slip_long - 0.10f) / 0.30f, 0.0f, 1.0f);
            out_lat = susp * f_lat * (1.0f - 0.45f * spent);
        });
    vehicle_->aero.body = body->GetID();
    vehicle_->aero.down = d.downforce; vehicle_->aero.drag = d.drag;
    impl_->system.AddStepListener(&vehicle_->aero);
    vehicle_->tester = new JPH::VehicleCollisionTesterCastCylinder(Layers::MOVING, 0.08f);
    vehicle_->wheel_ground.self = body->GetID();
    vehicle_->tester->SetBodyFilter(&vehicle_->wheel_ground);

    vehicle_->constraint->SetVehicleCollisionTester(vehicle_->tester);
    impl_->system.AddConstraint(vehicle_->constraint);
    impl_->system.AddStepListener(vehicle_->constraint);
    return 1;
}

void PhysicsWorld::vehicle_remove() {
    if (!vehicle_) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    if (vehicle_->constraint) {
        impl_->system.RemoveStepListener(vehicle_->constraint);
        impl_->system.RemoveStepListener(&vehicle_->aero);
        impl_->system.RemoveConstraint(vehicle_->constraint);
    }
    for (JPH::BodyID id : { vehicle_->body, vehicle_->patch, vehicle_->obstacles })
        if (!id.IsInvalid()) { bi.RemoveBody(id); bi.DestroyBody(id); }
    vehicle_.reset();
}

bool PhysicsWorld::vehicle_exists() const { return vehicle_ && !vehicle_->body.IsInvalid(); }

void PhysicsWorld::vehicle_input(float forward, float right, float brake, float handbrake) {
    if (!vehicle_exists()) return;
    auto* c = static_cast<JPH::WheeledVehicleController*>(vehicle_->constraint->GetController());
    c->SetDriverInput(forward, right, brake, handbrake);
    if (forward != 0.0f || right != 0.0f || brake != 0.0f || handbrake != 0.0f)
        impl_->system.GetBodyInterface().ActivateBody(vehicle_->body);
}

void PhysicsWorld::vehicle_keep_prev() {
    if (vehicle_exists()) vehicle_->prev_valid = vehicle_state(vehicle_->prev);
}

// The state before the last step(): with vehicle_state() the two ends between which a frame
// that falls between two physics steps is drawn. false: there is none (just made, or moved by hand).
bool PhysicsWorld::vehicle_state_prev(VehicleState& s) const {
    if (!vehicle_exists() || !vehicle_->prev_valid) return false;
    s = vehicle_->prev;
    return true;
}

bool PhysicsWorld::vehicle_state(VehicleState& s) const {
    if (!vehicle_exists()) return false;
    const JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    const JPH::RMat44 m = bi.GetWorldTransform(vehicle_->body);
    auto to_glm = [](const JPH::RMat44& t) {
        glm::mat4 g(1.0f);
        for (int c = 0; c < 3; ++c) { const JPH::Vec3 v = t.GetColumn3(c); g[c] = glm::vec4(v.GetX(), v.GetY(), v.GetZ(), 0.0f); }
        const JPH::RVec3 p = t.GetTranslation();
        g[3] = glm::vec4(float(p.GetX()), float(p.GetY()), float(p.GetZ()), 1.0f);
        return g;
    };
    s.body = to_glm(m);
    const JPH::Vec3 v = bi.GetLinearVelocity(vehicle_->body), w = bi.GetAngularVelocity(vehicle_->body);
    s.velocity = glm::vec3(v.GetX(), v.GetY(), v.GetZ());
    s.angular_velocity = glm::vec3(w.GetX(), w.GetY(), w.GetZ());
    for (int i = 0; i < 4; ++i) {
        // (the wheel mesh: axle along +x, up +y)
        s.wheel[i] = to_glm(vehicle_->constraint->GetWheelWorldTransform(JPH::uint(i), JPH::Vec3::sAxisX(), JPH::Vec3::sAxisY()));
        const JPH::Wheel* wh = vehicle_->constraint->GetWheel(JPH::uint(i));
        s.wheel_contact[i] = wh->HasContact();
        s.wheel_spin[i] = wh->GetAngularVelocity();
        s.suspension[i] = wh->GetSuspensionLength();
        const auto* wv = static_cast<const JPH::WheelWV*>(wh);
        s.wheel_slip[i] = std::max(std::abs(wv->mLongitudinalSlip), std::abs(wv->mLateralSlip) * 0.1f);
    }
    s.steer = vehicle_->constraint->GetWheel(0)->GetSteerAngle();
    const auto* c = static_cast<const JPH::WheeledVehicleController*>(vehicle_->constraint->GetController());
    s.rpm = c->GetEngine().GetCurrentRPM();
    s.gear = c->GetTransmission().GetCurrentGear();
    s.active = bi.IsActive(vehicle_->body);
    return true;
}

void PhysicsWorld::vehicle_teleport(glm::vec3 position, float yaw) {
    if (!vehicle_exists()) return;
    vehicle_->prev_valid = false;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    bi.SetPositionAndRotation(vehicle_->body, JPH::RVec3(position.x, position.y, position.z),
                              JPH::Quat::sRotation(JPH::Vec3::sAxisY(), yaw), JPH::EActivation::Activate);
    bi.SetLinearAndAngularVelocity(vehicle_->body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
}

void PhysicsWorld::vehicle_shift(glm::vec3 delta) {
    if (!vehicle_exists()) return;
    vehicle_->prev_valid = false;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    const JPH::RVec3 p = bi.GetPosition(vehicle_->body);
    bi.SetPosition(vehicle_->body, p + JPH::RVec3(delta.x, delta.y, delta.z), JPH::EActivation::DontActivate);
}

void PhysicsWorld::vehicle_impulse(glm::vec3 impulse, glm::vec3 at) {
    if (!vehicle_exists()) return;
    impl_->system.GetBodyInterface().AddImpulse(vehicle_->body, JPH::Vec3(impulse.x, impulse.y, impulse.z), JPH::RVec3(at.x, at.y, at.z));
}

// The ground under the vehicle where the home tile's heightfield does not reach: a square of
// `n` x `n` samples `cell` metres apart, its first sample at `origin`. A sample of 1e30 or more
// is a hole (no ground there: the home tile's own heightfield has it). Replaces the last one.
void PhysicsWorld::vehicle_set_ground(std::vector<float> samples, int n, glm::vec2 origin, float cell) {
    if (!vehicle_) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    if (!vehicle_->patch.IsInvalid()) { bi.RemoveBody(vehicle_->patch); bi.DestroyBody(vehicle_->patch); vehicle_->patch = JPH::BodyID(); }
    if (n < 4 || samples.size() != size_t(n) * size_t(n)) return;
    for (float& h : samples) if (h >= 1e30f) h = JPH::HeightFieldShapeConstants::cNoCollisionValue;
    vehicle_->patch_samples = std::move(samples);
    int block = 2;
    for (int b = 8; b >= 2; b >>= 1) if (n % b == 0) { block = b; break; }
    JPH::HeightFieldShapeSettings ss(vehicle_->patch_samples.data(), JPH::Vec3(origin.x, 0.0f, origin.y), JPH::Vec3(cell, 1.0f, cell), JPH::uint32(n));
    ss.mBlockSize = JPH::uint32(block);
    JPH::ShapeSettings::ShapeResult sr = ss.Create();
    if (sr.HasError()) { log::errorf("[jolt] vehicle ground: %s", sr.GetError().c_str()); return; }
    JPH::BodyCreationSettings bcs(sr.Get(), JPH::RVec3(0.0f, 0.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
    bcs.mFriction = 0.9f;
    vehicle_->patch = bi.CreateAndAddBody(bcs, JPH::EActivation::DontActivate);
}

// What stands near the vehicle: boxes, each turned by `yaw` about the vertical. Replaces the last set.
void PhysicsWorld::vehicle_set_obstacles(const OrientedBox* boxes, size_t count) {
    if (!vehicle_) return;
    JPH::BodyInterface& bi = impl_->system.GetBodyInterface();
    if (!vehicle_->obstacles.IsInvalid()) { bi.RemoveBody(vehicle_->obstacles); bi.DestroyBody(vehicle_->obstacles); vehicle_->obstacles = JPH::BodyID(); }
    if (!count) return;
    JPH::StaticCompoundShapeSettings cs;
    for (size_t i = 0; i < count; ++i) {
        const OrientedBox& b = boxes[i];
        const JPH::Vec3 he(std::max(b.half_extents.x, 0.03f), std::max(b.half_extents.y, 0.03f), std::max(b.half_extents.z, 0.03f));
        cs.AddShape(JPH::Vec3(b.center.x, b.center.y, b.center.z), JPH::Quat::sRotation(JPH::Vec3::sAxisY(), b.yaw),
                    new JPH::BoxShape(he, std::min(0.02f, 0.5f * std::min({ he.GetX(), he.GetY(), he.GetZ() }))));
    }
    JPH::ShapeSettings::ShapeResult sr = cs.Create();
    if (sr.HasError()) { log::errorf("[jolt] vehicle obstacles: %s", sr.GetError().c_str()); return; }
    JPH::BodyCreationSettings bcs(sr.Get(), JPH::RVec3(0.0f, 0.0f, 0.0f), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);
    // Nearly no friction: a car's flank slides along a rail or a wall. With 0.5 (against the
    // body's 0.6) the rail held the corner that touched it - the car turned into the rail and
    // crawled along it at 10 - 20 km/h on full throttle.
    bcs.mFriction = 0.03f;
    bcs.mRestitution = 0.10f;
    vehicle_->obstacles = bi.CreateAndAddBody(bcs, JPH::EActivation::DontActivate);
    vehicle_->wheel_ground.skip = vehicle_->obstacles;
}

} // namespace vibe
