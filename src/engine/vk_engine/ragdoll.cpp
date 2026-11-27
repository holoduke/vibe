// Ragdolls: a killed player's body, handed to the physics.
//
// At the moment of death the soldier's pose (whatever the motion capture had it doing) becomes
// eleven capsules joined at the joints - pelvis, torso, head, upper arms, forearms, thighs,
// shins - moving at the speed the player had. The shot that killed pushes the part of the body
// it hit, in the direction it flew, and gives the whole body a shove; from there it is the
// physics: it folds, spins and falls as its weight and the ground decide. Every frame the
// capsules' positions are read back into the skeleton and the mesh is skinned from them.
#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/multiplayer.h"
#include "engine/cpu_prof.h"
#include "engine/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace vibe {

namespace {

struct PartDef {
    const char* bone;       // the bone this part moves
    const char* from;       // capsule: from this bone's joint ...
    const char* to;         // ... to this one's ("" = along the bone itself)
    float radius, mass;
    int   parent;
    float cone;
    bool  hinge;
};
// (parents before children; the order is the part index)
const PartDef kParts[11] = {
    { "Hips",         "LeftUpLeg",    "RightUpLeg",   0.110f, 12.0f, -1, 0.0f,  false },
    { "Spine",        "Spine",        "Neck",         0.140f, 28.0f,  0, 0.45f, false },
    { "Neck",         "Neck",         "",             0.100f,  5.0f,  1, 0.60f, false },
    { "LeftArm",      "LeftArm",      "LeftForeArm",  0.050f,  2.5f,  1, 1.30f, false },
    { "RightArm",     "RightArm",     "RightForeArm", 0.050f,  2.5f,  1, 1.30f, false },
    { "LeftForeArm",  "LeftForeArm",  "LeftHandMiddle1",  0.045f, 2.0f, 3, 1.20f, false },
    { "RightForeArm", "RightForeArm", "RightHandMiddle1", 0.045f, 2.0f, 4, 1.20f, false },
    { "LeftUpLeg",    "LeftUpLeg",    "LeftLeg",      0.075f,  8.0f,  0, 0.90f, false },
    { "RightUpLeg",   "RightUpLeg",   "RightLeg",     0.075f,  8.0f,  0, 0.90f, false },
    { "LeftLeg",      "LeftLeg",      "LeftFoot",     0.055f,  5.0f,  7, 0.0f,  true  },
    { "RightLeg",     "RightLeg",     "RightFoot",    0.055f,  5.0f,  8, 0.0f,  true  },
};

float seg_dist(glm::vec3 p, glm::vec3 a, glm::vec3 b) {
    const glm::vec3 ab = b - a;
    const float t = glm::clamp(glm::dot(p - a, ab) / std::max(glm::dot(ab, ab), 1e-8f), 0.0f, 1.0f);
    return glm::length(p - (a + ab * t));
}

}  // namespace

// Where the ragdoll's parts are (their capsules' middles, in the world) in the pose that was
// made last (body_P_): two of these a frame apart say how fast each part moves.
bool VulkanEngine::corpse_part_centres(const glm::mat4& model, glm::vec3* out) const {
    if (body_bones_.empty()) return false;
    const glm::mat3 Mr(model);
    const glm::vec3 o(model[3]);
    for (int k = 0; k < kCorpseParts; ++k) {
        const PartDef& d = kParts[k];
        const int bf = body_bone(d.from), bt = d.to[0] ? body_bone(d.to) : -1;
        if (bf < 0) return false;
        const glm::vec3 a = o + Mr * body_P_[size_t(bf)];
        out[k] = bt >= 0 ? (a + o + Mr * body_P_[size_t(bt)]) * 0.5f : a;
    }
    return true;
}

void VulkanEngine::corpse_spawn(const glm::mat4& model, const glm::mat4& view_inv, glm::vec3 velocity,
                                glm::vec3 hit_pos, glm::vec3 hit_dir, float power, const glm::vec3* part_velocity) {
    if (!physics_ || body_bones_.empty()) return;
    static_assert(kCorpseParts == 11, "parts table");
    // A free slot, or the oldest.
    Corpse* c = nullptr;
    for (Corpse& k : corpses_) if (!k.rag) { c = &k; break; }
    if (!c) {
        c = &corpses_[0];
        for (Corpse& k : corpses_) if (k.age > c->age) c = &k;
        physics_->remove_ragdoll(c->rag);
        c->rag = 0;
    }
    const size_t nb = body_bones_.size();
    // The pose, in the world (positions relative to the feet: small numbers).
    const glm::mat3 Mr(model);
    c->origin = glm::vec3(model[3]);
    c->R0.resize(nb);
    c->P0.resize(nb);
    for (size_t i = 0; i < nb; ++i) {
        c->R0[i] = Mr * body_R_[i];
        c->P0[i] = Mr * body_P_[i];
    }
    c->eye0 = view_inv;
    c->eye0[3] = glm::vec4(glm::vec3(view_inv[3]) - c->origin, 1.0f);

    PhysicsWorld::RagdollPart parts[kCorpseParts];
    for (int k = 0; k < kCorpseParts; ++k) {
        const PartDef& d = kParts[k];
        const int bf = body_bone(d.from), bt = d.to[0] ? body_bone(d.to) : -1;
        if (bf < 0) return;
        PhysicsWorld::RagdollPart& p = parts[k];
        p.a = c->origin + c->P0[bf];
        p.b = bt >= 0 ? c->origin + c->P0[bt]
                      : p.a + c->R0[bf] * (body_bones_[bf].tail - body_bones_[bf].head) * 2.2f;      // (the head: neck to crown)
        p.radius = d.radius * body_scale_;
        p.mass = d.mass;
        p.parent = d.parent;
        p.cone = d.cone;
        p.hinge = d.hinge;
    }
    // The pelvis hangs on nothing; what hangs on it is joined where it starts. The torso starts
    // at the spine's base, between the hips.
    // Knees: a hinge about the thigh's sideways axis, from straight to folded - the limits are
    // given from the angle the knee has now.
    for (int k = 9; k <= 10; ++k) {
        const int th = body_bone(kParts[k - 2].bone);
        const glm::vec3 ax = glm::normalize(c->R0[th] * glm::vec3(1.0f, 0.0f, 0.0f));
        const glm::vec3 t = glm::normalize(parts[k - 2].b - parts[k - 2].a), s = glm::normalize(parts[k].b - parts[k].a);
        const float now = std::atan2(glm::dot(glm::cross(t, s), ax), glm::dot(t, s));
        parts[k].axis = ax;
        parts[k].lim_min = -2.4f - now;
        parts[k].lim_max = 0.05f - now;
    }
    // The ground it falls on: the ground the player walks on, which is not everywhere the
    // physics' own (the paved road lies higher; beyond the home terrain there is none).
    glm::vec3 fp = c->origin;
    const float e = 1.5f;
    fp.y = sample_terrain_height(fp.x, fp.z);
    const glm::vec3 fn = glm::normalize(glm::vec3(sample_terrain_height(fp.x - e, fp.z) - sample_terrain_height(fp.x + e, fp.z), 2.0f * e,
                                                  sample_terrain_height(fp.x, fp.z - e) - sample_terrain_height(fp.x, fp.z + e)));
    c->rag = physics_->add_ragdoll(parts, kCorpseParts, velocity, &fp, fn);
    if (!c->rag) return;
    for (int k = 0; k < kCorpseParts; ++k) {
        c->q0[k] = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        c->p0[k] = (parts[k].a + parts[k].b) * 0.5f;
        physics_->ragdoll_part(c->rag, size_t(k), c->p0[k], c->q0[k]);
        c->half0[k] = (parts[k].b - parts[k].a) * 0.5f; c->rad[k] = parts[k].radius; c->mass[k] = parts[k].mass;
        // (it was in the middle of a movement: each part goes on as it went)
        if (part_velocity) physics_->ragdoll_set_velocity(c->rag, size_t(k), part_velocity[k]);
    }
    // The shot: a hard knock on the part it hit, a shove for the whole body.
    if (power > 0.0f && glm::dot(hit_dir, hit_dir) > 0.25f) {
        const glm::vec3 dir = glm::normalize(glm::normalize(hit_dir) + glm::vec3(0.0f, 0.18f, 0.0f));
        int near_k = 1;
        float near_d = 1e9f;
        for (int k = 0; k < kCorpseParts; ++k) {
            const float dd = seg_dist(hit_pos, parts[k].a, parts[k].b) - parts[k].radius;
            if (dd < near_d) { near_d = dd; near_k = k; }
        }
        for (int k = 0; k < kCorpseParts; ++k)
            physics_->ragdoll_impulse(c->rag, size_t(k), dir * (parts[k].mass * 0.7f * power), c->p0[k]);
        // (a jolt, not a throw: 3.5 m/s for a limb, 2 m/s for the trunk - it was 7 and 5, and a body slid off)
        const float knock = std::min(parts[near_k].mass * 3.5f, 60.0f) * power;
        // (at the point the shot struck, brought onto the part: off its centre it spins it)
        const glm::vec3 ab = parts[near_k].b - parts[near_k].a;
        const float tt = glm::clamp(glm::dot(hit_pos - parts[near_k].a, ab) / std::max(glm::dot(ab, ab), 1e-8f), 0.0f, 1.0f);
        physics_->ragdoll_impulse(c->rag, size_t(near_k), dir * knock, parts[near_k].a + ab * tt);
        log::infof("[ragdoll] hit on %s, power %.2f", kParts[near_k].bone, power);
    }
    c->age = 0.0f;
    c->went = c->fastest = 0.0f; c->told = false; c->pelvis_last = c->p0[0];
    c->av.id = 1;
    c->av.corpse = true;
    c->av.drawn = true;
    c->av.model = c->av.prev_model = glm::translate(glm::mat4(1.0f), c->origin);
    c->av.view_inv = view_inv;
    c->av.muzzle_t = 0.0f;
    c->av.recoil_t = 0.0f;
    c->av.settle = 0.0f;        // frames skinned since the physics went to sleep
    corpse_obstacles(0.0f);     // (the cars beside it are there before it falls)
}

// A bolt through a body that fell or is falling. The bolt's own body does not touch it (the
// physics' CORPSE layer: it would throw it out of sight); the part it passes through gets a
// knock a body can take - it jerks, and bleeds. True: the bolt ends here.
bool VulkanEngine::corpse_bolt_hits(glm::vec3 a, glm::vec3 b, glm::vec3& hit) {
    if (!physics_ || std::getenv("VIBE_TEST_BOLTS_HIT_BODIES")) return false;
    const glm::vec3 d = b - a;
    const float dd = glm::dot(d, d);
    if (dd < 1e-8f) return false;
    Corpse* best_c = nullptr; int best_k = -1; float best_t = 2.0f;
    for (Corpse& c : corpses_) {
        if (!c.rag) continue;
        for (int k = 0; k < kCorpseParts; ++k) {
            glm::vec3 p = c.p0[k]; glm::quat q = c.q0[k];
            if (!physics_->ragdoll_part(c.rag, size_t(k), p, q)) continue;
            if (glm::distance(p, a) > glm::length(d) + 1.5f) continue;
            // (the nearest the bolt's path comes to the part's axis: a few samples along the capsule do)
            const glm::vec3 h = (q * glm::inverse(c.q0[k])) * c.half0[k];
            for (int s = -2; s <= 2; ++s) {
                const glm::vec3 w = p + h * (float(s) * 0.5f);
                const float t = glm::clamp(glm::dot(w - a, d) / dd, 0.0f, 1.0f);
                if (glm::distance(a + d * t, w) < c.rad[k] + 0.05f && t < best_t) { best_t = t; best_c = &c; best_k = k; }
            }
        }
    }
    if (!best_c) return false;
    hit = a + d * best_t;
    const glm::vec3 dir = glm::normalize(d);
    physics_->ragdoll_impulse(best_c->rag, size_t(best_k), dir * std::min(best_c->mass[best_k] * 2.5f, 22.0f), hit);
    spawn_blood(hit, dir);
    return true;
}

// WHAT A BODY THAT FELL LIES ON. The physics' world (Jolt, NON_MOVING) has the level's boxes -
// castle, walls, the station's and the pavilion's .col boxes (tables, counters, booths, stools,
// benches, desks), trees, rocks, bridges - the home terrain and each body's own floor slab; the
// driven car is a vehicle of the physics (layer CAR). It did NOT have what is no obstacle to a
// walker: the chairs (a chair is sat on: the station's 16 diner chairs, the pavilion's desk and
// meeting chairs, its sofas have no collider at all) and every car but the driven one (the
// parked cars and the NPC engine's have no physics). A body shot beside them fell through.
// Now, in the physics' CORPSE layer (only fallen bodies meet them; walkers, bolts and the driven
// car do not): a seat and a backrest box for every such seat of the sites' .nav files (made
// once), and the cars within 9 m of a body as bodies moved with them (PhysicsWorld::
// corpse_cars_set; while any body lies there). VIBE_LOG_CORPSE_OBST=1: what was added.
void VulkanEngine::corpse_obstacles(float dt) {
    if (!physics_) return;
    static const bool lg = std::getenv("VIBE_LOG_CORPSE_OBST") != nullptr;
    if (!corpse_props_made_) {
        corpse_props_made_ = true;
        std::vector<PhysicsWorld::CorpseProp> props;
        auto read = [&](const char* path, glm::vec3 base) {
            if (base == glm::vec3(0.0f)) return;                      // (the site is not placed)
            FILE* fp = std::fopen(path, "r");
            if (!fp) return;
            char line[256], kind[32];
            int n = 0;
            while (std::fgets(line, sizeof(line), fp)) {
                float x, y, z, yaw;
                if (std::sscanf(line, "seat %31s %f %f %f %f", kind, &x, &y, &z, &yaw) != 5) continue;
                const std::string k(kind);
                // seat half width / half depth / height under the top, the back's offset behind,
                // its height above the seat and its half width (the models: tools/gasstation
                // chair(), tools/office chair() and sofa())
                float hw, hd, h, back, bh, bw;
                if (k == "chair")                    { hw = 0.21f; hd = 0.21f; h = 0.50f; back = 0.215f; bh = 0.42f; bw = 0.20f; }
                else if (k == "desk" || k == "meet") { hw = 0.24f; hd = 0.22f; h = 0.40f; back = 0.25f;  bh = 0.55f; bw = 0.22f; }
                else if (k == "sofa")                { hw = 0.40f; hd = 0.30f; h = 0.42f; back = 0.36f;  bh = 0.40f; bw = 0.40f; }
                else continue;                        // (stools, booths, benches, picnic tables: colliders of their own)
                const glm::vec3 s = base + glm::vec3(x, y, z);
                const glm::vec3 fwd(std::sin(yaw), 0.0f, -std::cos(yaw));
                // (a box turned by -yaw about y has its x axis along the seat's right: (cos yaw, 0, sin yaw))
                props.push_back({ s - glm::vec3(0.0f, 0.5f * h, 0.0f), glm::vec3(hw, 0.5f * h, hd), -yaw });
                props.push_back({ s - fwd * back + glm::vec3(0.0f, 0.5f * bh + 0.05f, 0.0f), glm::vec3(bw, 0.5f * bh, 0.03f), -yaw });
                ++n;
            }
            std::fclose(fp);
            log::infof("[ragdoll] %d seats of %s are something a fallen body lies on", n, path);
        };
        read("assets/gasstation/gasstation.nav", station_pos_);
        read("assets/office/office.nav", office_pos_);
        if (!props.empty()) physics_->corpse_props_add(props.data(), props.size());
    }
    // The cars near any body that lies or falls.
    static std::vector<PhysicsWorld::CorpseCar> cars;
    cars.clear();
    for (const Corpse& c : corpses_) {
        if (!c.rag) continue;
        const size_t before = cars.size();
        corpse_car_bodies(c.p0[0], 9.0f, cars);
        // (one car near two bodies: once)
        for (size_t i = before; i < cars.size();) {
            bool dup = false;
            for (size_t j = 0; j < before; ++j) if (cars[j].key == cars[i].key) dup = true;
            if (dup) cars.erase(cars.begin() + ptrdiff_t(i)); else ++i;
        }
    }
    const size_t had = physics_->corpse_cars_count();
    physics_->corpse_cars_set(cars.data(), cars.size(), dt);
    if (lg && had != physics_->corpse_cars_count()) log::infof("[ragdoll] %zu car(s) are something a fallen body lies on", physics_->corpse_cars_count());
}

void VulkanEngine::corpses_update(float dt) {
    if (!physics_) return;
    corpse_obstacles(dt);
    static const float kLife = 25.0f;
    const size_t nb = body_bones_.size();
    for (Corpse& c : corpses_) {
        if (!c.rag) continue;
        c.av.palette_prev = c.age > 0.0f ? c.av.palette : -1;
        c.av.palette = -1;
        c.age += dt;
        if (c.age > kLife) {
            physics_->remove_ragdoll(c.rag);
            c.rag = 0;
            c.av.drawn = false;
            continue;
        }
        if (!c.told && c.age >= 3.0f) {
            c.told = true;
            if (c.went > 8.0f) log::warnf("[ragdoll] a body flew %.1f m from (%.1f, %.1f, %.1f), top speed %.1f m/s", double(c.went), c.p0[0].x, c.p0[0].y, c.p0[0].z, double(c.fastest));
            else log::infof("[ragdoll] fell within %.1f m of (%.1f, %.1f, %.1f), top speed %.1f m/s", double(c.went), c.p0[0].x, c.p0[0].y, c.p0[0].z, double(c.fastest));
        }
        // Lying still: the final pose is sent again, nothing more to do.
        if (!physics_->ragdoll_active(c.rag)) {
            if (c.av.settle >= 2.0f && !c.av.pose_rows.empty()) { c.av.palette = body_push_rows(c.av.pose_rows); continue; }
            c.av.settle += 1.0f;
        }
        else c.av.settle = 0.0f;
        CPU_SCOPE("ragdoll");
        if (std::getenv("VIBE_RAGDOLL_DEBUG") && (frame_number_ % 20) == 0) {
            glm::vec3 p; glm::quat q; physics_->ragdoll_part(c.rag, 0, p, q);
            const PhysicsWorld::RayHit h = physics_->raycast(c.origin + glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), 20.0f);
            log::infof("[ragdoll] age %.2f pelvis %.2f %.2f %.2f  origin %.2f %.2f %.2f  physics ground y %.2f (hit %d dyn %d)", c.age, p.x, p.y, p.z, c.origin.x, c.origin.y, c.origin.z, h.position.y, int(h.hit), int(h.dynamic));
        }
        glm::quat dq[kCorpseParts];
        glm::vec3 pp[kCorpseParts];
        for (int k = 0; k < kCorpseParts; ++k) {
            glm::vec3 p = c.p0[k];
            glm::quat q = c.q0[k];
            physics_->ragdoll_part(c.rag, size_t(k), p, q);
            dq[k] = q * glm::inverse(c.q0[k]);
            pp[k] = p;
        }
        // How far it went and how fast: said once, three seconds on. A body shot dead falls where
        // it stood - a few metres at most; one that flew is a fault to be found (WARN).
        c.went = std::max(c.went, glm::distance(pp[0], c.p0[0]));
        if (dt > 1e-4f) c.fastest = std::max(c.fastest, glm::distance(pp[0], c.pelvis_last) / dt);
        c.pelvis_last = pp[0];
        std::vector<int> part_of(nb, -1);
        for (int k = 0; k < kCorpseParts; ++k) {
            const int b = body_bone(kParts[k].bone);
            if (b >= 0) part_of[size_t(b)] = k;
        }
        if (nb > 0 && body_bones_[0].parent < 0) part_of[0] = 0;          // the root rides on the pelvis
        for (size_t i = 0; i < nb; ++i) {
            const int k = part_of[i];
            const int par = body_bones_[i].parent;
            if (k >= 0) {
                const glm::mat3 D = glm::mat3_cast(dq[k]);
                body_R_[i] = D * c.R0[i];
                body_P_[i] = (pp[k] - c.origin) + D * (c.P0[i] - (c.p0[k] - c.origin));
            } else if (par >= 0) {
                // carried by its parent, as it was held at the moment of death
                const glm::mat3 rel = body_R_[size_t(par)] * glm::transpose(c.R0[size_t(par)]);
                body_R_[i] = rel * c.R0[i];
                body_P_[i] = body_P_[size_t(par)] + rel * (c.P0[i] - c.P0[size_t(par)]);
            } else {
                body_R_[i] = c.R0[i];
                body_P_[i] = c.P0[i];
            }
        }
        c.av.palette = body_push_pose(&c.av.pose_rows);
        // the head (drawn as a part of its own, like the other players')
        const glm::mat3 Dh = glm::mat3_cast(dq[2]);
        glm::mat4 e(Dh * glm::mat3(c.eye0));
        e[3] = glm::vec4(pp[2] + Dh * (glm::vec3(c.eye0[3]) - (c.p0[2] - c.origin)), 1.0f);
        c.av.view_inv = e;
        c.av.eye = glm::vec3(e[3]);
        c.av.mesh_lod = body_mesh_lod(c.av.eye, c.av.mesh_lod);
    }
}

void VulkanEngine::draw_corpses(VkCommandBuffer cmd, const glm::mat4& vp) {
    const float sv_recoil = recoil_timer_, sv_muzzle = muzzle_flash_timer_;
    const int sv_weapon = weapon_;
    const bool sv_flash = flashlight_on_;
    for (Corpse& c : corpses_) {
        if (!c.rag || !c.av.drawn) continue;
        recoil_timer_ = 0.0f; muzzle_flash_timer_ = 0.0f; weapon_ = 0; flashlight_on_ = false;
        mp_->draw = &c.av;
        draw_viewmodel(cmd, vp, c.av.view_inv);
        mp_->draw = nullptr;
    }
    recoil_timer_ = sv_recoil; muzzle_flash_timer_ = sv_muzzle; weapon_ = sv_weapon; flashlight_on_ = sv_flash;
}

void VulkanEngine::corpses_destroy() {
    for (Corpse& c : corpses_) {
        if (c.rag && physics_) physics_->remove_ragdoll(c.rag);
        c.rag = 0;
    }
}

}  // namespace vibe
