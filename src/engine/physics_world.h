#pragma once

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <memory>
#include <vector>
#include <vector>

namespace vibe {

// Thin Jolt wrapper. Owns a PhysicsSystem ready for non-player rigid bodies
// (dynamic crates, projectiles, ragdolls). The player movement is *not* run
// through this — Quake-feel needs the custom kinematic code.
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // Advance simulation by dt seconds. Internally subdivides if dt is large.
    void step(float dt);

    // Body count snapshot — handy for log diagnostics.
    int body_count() const;

    // Add a static box AABB to the world (for matching the level geometry to Jolt).
    void add_static_box(glm::vec3 center, glm::vec3 half_extents);

    // Batch-add many static boxes in one BodyInterface call. ~10x faster
    // than calling add_static_box per brush at level load (Jolt's
    // AddBodiesPrepare/Finalize amortizes the body-mutex acquire and the
    // broadphase tree rebuild). Bodies stay disabled until prepare returns.
    struct StaticBox { glm::vec3 center; glm::vec3 half_extents; };
    void add_static_boxes(const StaticBox* boxes, size_t count);
    void add_static_box_compounds(const StaticBox* boxes, size_t count, float cell = 128.0f);
    // Removes every compound added above (the forest is being re-planted).
    void clear_static_box_compounds();
    // Remove the compound boxes whose centre lies within `tol` (XZ) of any
    // given point and rebuild the affected cells. Returns boxes removed.
    size_t remove_compound_boxes_at(const glm::vec2* xz, size_t n, float tol);
    // Push edited heights (row-major, `stride` floats per row, same indexing
    // as the heightfield samples) for the sample rectangle [x0..x1]x[z0..z1]
    // into the live heightfield in place (craters). Block-aligned inside.
    void update_heightfield_region(const float* heights, int stride,
                                   int x0, int z0, int x1, int z1);

    // Voxel-building collision: replace the single static body that
    // represents the destructible voxel shape with a fresh one built from
    // `count` world-space boxes (one StaticCompoundShape → one Jolt body,
    // so it never threatens the body cap regardless of box count). Pass
    // count == 0 to just remove the existing body. Called at voxel init and
    // rebuilt after every carve. Projectiles collide with this for
    // shoot-to-destroy; the player uses the same boxes via game::collision.
    void set_voxel_collision(const StaticBox* boxes, size_t count);

    // Add a procedural heightmap as a single Jolt HeightFieldShape static
    // body. `dim` is the cell count per side; `samples` must be of size
    // (dim+1) * (dim+1) and laid out row-major (x outer, z inner — matching
    // Heightmap::at). `origin_xz` is the world-space corner the (0,0) cell
    // sits at; `cell_size` is the world distance between adjacent samples.
    void add_static_heightfield(const float* samples, int dim,
                                glm::vec2 origin_xz, float cell_size);
    // The same on a worker thread (start-up: the 0.7 s build overlaps the
    // texture loading). finish_async() waits for it; step() does too.
    void add_static_heightfield_async(std::vector<float> samples, int dim,
                                      glm::vec2 origin_xz, float cell_size);
    void finish_async();

    // Engine-side body handle (cached BodyID). Storing this on each
    // dynamic prop / particle / projectile lets the read-only queries skip
    // the unordered_map lookup in physics_world.cpp. 0 = invalid.
    //
    // *** HOT PATH ADVISORY ***
    // Any per-frame loop that touches multiple bodies (projectile update,
    // dyn-prop pose cache, particle TTL pass, AABB rebuild, TLAS instance
    // write) MUST cache `handle_of(id)` ONCE at spawn time and call only
    // the `_h` variants below thereafter. The id-keyed overloads
    // (get_body_world_matrix, get_linear_velocity, is_body_active,
    // apply_impulse) walk an unordered_map per call — fine for one-shot
    // queries, ruinous for combat where dozens of bodies are queried
    // every frame. Canonical examples: Projectile::jolt_handle (see
    // combat.cpp), DynamicProp::jolt_handle, Particle::jolt_handle —
    // they're cached at spawn for exactly this reason.
    using BodyHandle = uint32_t;
    BodyHandle handle_of(uint32_t id) const;
    bool get_body_world_matrix_h(BodyHandle h, glm::mat4& out) const;
    bool is_body_active_h(BodyHandle h) const;
    glm::vec3 get_linear_velocity_h(BodyHandle h) const;
    void apply_impulse_h(BodyHandle h, glm::vec3 impulse);

    // Spawn a dynamic box; returns an opaque body id (uint32_t).
    // `euler_radians` is XYZ Euler order (pitch, yaw, roll) — pass {0,0,0} for upright.
    uint32_t add_dynamic_box(glm::vec3 center, glm::vec3 half_extents,
                             glm::vec3 euler_radians = glm::vec3(0.0f),
                             float density = 1000.0f);

    // Spawn a dynamic cylinder (axis = Y in local space) with continuous
    // collision detection (Jolt's `LinearCast` motion quality). The body is
    // sweep-cast at the start of each physics step, so a fast projectile
    // CANNOT tunnel through a thinner box between ticks. Used for bullets.
    //   `orientation` is the quaternion that rotates the local +Y axis to the
    //   firing direction (so the cylinder points along travel).
    //   `linear_velocity` is set at creation (m/s, world space).
    //   `mass` is in kg (explicit, not density-derived).
    uint32_t add_dynamic_cylinder_ccd(glm::vec3 center,
                                      float radius, float half_length,
                                      glm::quat orientation,
                                      glm::vec3 linear_velocity,
                                      float mass);

    // Spawn a small dynamic sphere — used as a hit-effect particle. Bounces
    // realistically off everything in the world via Jolt's collision response
    // (this is exactly the "particles bounce off material and ground/walls"
    // requirement). `restitution` 0..1 (0=no bounce, 1=perfectly elastic).
    uint32_t add_dynamic_sphere(glm::vec3 center, float radius,
                                glm::vec3 linear_velocity,
                                float mass,
                                float restitution = 0.5f,
                                float friction = 0.4f);

    // World matrix (translation + rotation) for rendering. False if id missing.
    bool get_body_world_matrix(uint32_t id, glm::mat4& out) const;

    // True if the body is awake (Jolt is still simulating it). When a dynamic
    // body settles and Jolt sleeps it, its transform stops changing — callers
    // can skip recomputing per-frame derived state (AABB, render matrix). False
    // for unknown ids and for sleeping bodies.
    bool is_body_active(uint32_t id) const;

    // Current linear velocity in world space (m/s). Returns (0,0,0) if id is
    // missing. Used by projectile bookkeeping to detect bounce / impact.
    glm::vec3 get_linear_velocity(uint32_t id) const;

    // A ragdoll: capsules joined at their ends. Part i is the capsule from `a` to `b` (world);
    // it hangs on part `parent` (which comes before it; -1 for the first) at `a`. `hinge`: the
    // joint only bends about `axis` between lim_min and lim_max (rad, 0 = as it is now);
    // otherwise it may swing `cone` radians any way from how it is now, and twist a little.
    // Parts that are joined do not collide with each other.
    struct RagdollPart {
        glm::vec3 a{0.0f}, b{0.0f};
        float radius = 0.05f, mass = 5.0f;
        int   parent = -1;
        bool  hinge = false;
        glm::vec3 axis{1.0f, 0.0f, 0.0f};
        float lim_min = 0.0f, lim_max = 0.0f, cone = 0.6f;
    };
    // `floor_point` / `floor_normal`: a floor of its own under it (a slab 16 m across through
    // that point), for where the ground the game stands on is not the physics' ground - the
    // paved road, the world beyond the home terrain. Removed with the ragdoll.
    uint32_t add_ragdoll(const RagdollPart* parts, size_t n, glm::vec3 linear_velocity,
                         const glm::vec3* floor_point = nullptr, glm::vec3 floor_normal = glm::vec3(0.0f, 1.0f, 0.0f));
    bool ragdoll_part(uint32_t id, size_t part, glm::vec3& pos, glm::quat& rot) const;   // centre + orientation
    void ragdoll_impulse(uint32_t id, size_t part, glm::vec3 impulse, glm::vec3 at);
    bool ragdoll_active(uint32_t id) const;
    void remove_ragdoll(uint32_t id);

    // Remove and destroy a previously-added dynamic body.
    void remove_body(uint32_t id);

    struct RayHit {
        bool hit = false;
        bool dynamic = false;     // true if a MOVING-layer body was hit
        uint32_t body_id = 0;     // vibe id (only valid if `dynamic`)
        glm::vec3 position{0.0f}; // world-space hit point
        glm::vec3 normal{0.0f};   // surface normal at hit
        float distance = 0.0f;
    };

    // Cast a world-space ray. Returns the closest hit (statics + dynamics).
    RayHit raycast(glm::vec3 origin, glm::vec3 direction, float max_distance) const;

    // Apply a directional impulse (kg·m/s) to a dynamic body.
    void apply_impulse(uint32_t id, glm::vec3 impulse);

    // ---- The vehicle (one at a time): a wheeled vehicle with suspension, engine, gearbox and
    // tyres (Jolt's VehicleConstraint). Frame: +y up, the front toward -z, wheels 0/1 = front
    // left / right, 2/3 = rear left / right.
    struct VehicleDesc {
        glm::vec3 position{0.0f};          // the body's origin (on the ground under mid-wheelbase)
        float     yaw = 0.0f;
        std::vector<StaticBox> boxes;      // the body, in its own frame
        float     mass = 1250.0f;          // kg
        // The centre of mass: its height above the ground line, and the share of the weight on
        // the front axle. (Not the collision boxes' own: a windscreen and a roll bar weigh
        // little, an engine and a frame much.)
        float     com_height = 0.55f, front_weight = 0.52f;
        float     front_drive = 0.4f;      // share of the engine's torque to the front axle (0: rear-wheel drive)
        std::vector<float> gears;          // forward gear ratios (empty: four, 3.6 .. 1.0)
        glm::vec3 wheel_pos[4];            // wheel centres at rest, in the body's frame
        float     wheel_radius = 0.40f, wheel_width = 0.26f, wheel_mass = 22.0f;
        float     suspension_rest = 0.30f;            // spring length at rest (m)
        float     suspension_travel_up = 0.14f, suspension_travel_down = 0.16f;
        float     suspension_hz = 1.6f, suspension_damping = 0.55f;
        float     antiroll = 2600.0f;
        float     max_steer = 0.58f;                  // rad
        float     engine_torque = 310.0f, engine_max_rpm = 4600.0f, final_drive = 4.4f;
        float     brake_torque = 2400.0f, handbrake_torque = 5200.0f;
        float     grip = 1.0f;
    };
    struct VehicleState {
        glm::mat4 body{1.0f};
        glm::mat4 wheel[4];                // world; the wheel mesh's axle is +x
        bool      wheel_contact[4]{};
        float     wheel_spin[4]{}, wheel_slip[4]{}, suspension[4]{};
        glm::vec3 velocity{0.0f}, angular_velocity{0.0f};
        float     steer = 0.0f, rpm = 0.0f;
        int       gear = 0;
        bool      active = false;
    };
    struct OrientedBox { glm::vec3 center; glm::vec3 half_extents; float yaw; };
    uint32_t vehicle_create(const VehicleDesc& d);
    void     vehicle_remove();
    bool     vehicle_exists() const;
    void     vehicle_input(float forward, float right, float brake, float handbrake);
    bool     vehicle_state(VehicleState& out) const;
    void     vehicle_teleport(glm::vec3 position, float yaw);
    void     vehicle_shift(glm::vec3 delta);
    void     vehicle_impulse(glm::vec3 impulse, glm::vec3 at);
    void     vehicle_set_ground(std::vector<float> samples, int n, glm::vec2 origin, float cell);
    void     vehicle_set_obstacles(const OrientedBox* boxes, size_t count);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    struct VehicleImpl;
    std::unique_ptr<VehicleImpl> vehicle_;
};

} // namespace vibe
