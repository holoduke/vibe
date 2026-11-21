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
#include "engine/cpu_prof.h"
#include "engine/log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace qlike {

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

void VulkanEngine::corpse_spawn(const glm::mat4& model, const glm::mat4& view_inv, glm::vec3 velocity,
                                glm::vec3 hit_pos, glm::vec3 hit_dir, float power) {
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
    const size_t nb = body_bones_.size(), nv = body_verts_.size();
    for (uint32_t i = 0; i < kFrameOverlap; ++i) {
        if (c->av.vbuf[i]) continue;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = sizeof(Vertex) * nv;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        if (vmaCreateBuffer(allocator_, &bci, &aci, &c->av.vbuf[i], &c->av.valloc[i], &ai) != VK_SUCCESS) { c->av.vbuf[i] = VK_NULL_HANDLE; return; }
        c->av.mapped[i] = ai.pMappedData;
    }
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
            physics_->ragdoll_impulse(c->rag, size_t(k), dir * (parts[k].mass * 1.5f * power), c->p0[k]);
        const float knock = std::min(parts[near_k].mass * 7.0f, 150.0f) * power;
        // (at the point the shot struck, brought onto the part: off its centre it spins it)
        const glm::vec3 ab = parts[near_k].b - parts[near_k].a;
        const float tt = glm::clamp(glm::dot(hit_pos - parts[near_k].a, ab) / std::max(glm::dot(ab, ab), 1e-8f), 0.0f, 1.0f);
        physics_->ragdoll_impulse(c->rag, size_t(near_k), dir * knock, parts[near_k].a + ab * tt);
        log::infof("[ragdoll] hit on %s, power %.2f", kParts[near_k].bone, power);
    }
    c->age = 0.0f;
    c->av.id = 1;
    c->av.corpse = true;
    c->av.drawn = true;
    c->av.model = c->av.prev_model = glm::translate(glm::mat4(1.0f), c->origin);
    c->av.view_inv = view_inv;
    c->av.muzzle_t = 0.0f;
    c->av.recoil_t = 0.0f;
    c->av.phase = 0.0f;         // (used here as: frames skinned since the physics went to sleep)
}

void VulkanEngine::corpses_update(float dt) {
    if (!physics_) return;
    static const float kLife = 25.0f;
    const uint32_t slot = frame_number_ % kFrameOverlap;
    const size_t nb = body_bones_.size();
    for (Corpse& c : corpses_) {
        if (!c.rag) continue;
        c.age += dt;
        if (c.age > kLife) {
            physics_->remove_ragdoll(c.rag);
            c.rag = 0;
            c.av.drawn = false;
            continue;
        }
        // Lying still: both frames' buffers hold the final pose, nothing more to do.
        if (!physics_->ragdoll_active(c.rag)) { if (c.av.phase >= float(kFrameOverlap)) continue; c.av.phase += 1.0f; }
        else c.av.phase = 0.0f;
        CPU_SCOPE("ragdoll");
        if (std::getenv("QLIKE_RAGDOLL_DEBUG") && (frame_number_ % 20) == 0) {
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
        if (c.av.mapped[slot]) body_skin_to(static_cast<Vertex*>(c.av.mapped[slot]));
        // the head (drawn as a part of its own, like the other players')
        const glm::mat3 Dh = glm::mat3_cast(dq[2]);
        glm::mat4 e(Dh * glm::mat3(c.eye0));
        e[3] = glm::vec4(pp[2] + Dh * (glm::vec3(c.eye0[3]) - (c.p0[2] - c.origin)), 1.0f);
        c.av.view_inv = e;
        c.av.eye = glm::vec3(e[3]);
    }
}

void VulkanEngine::draw_corpses(VkCommandBuffer cmd, const glm::mat4& vp) {
    const float sv_recoil = recoil_timer_, sv_muzzle = muzzle_flash_timer_, sv_body_recoil = body_recoil_;
    const int sv_weapon = weapon_;
    const bool sv_flash = flashlight_on_;
    for (Corpse& c : corpses_) {
        if (!c.rag || !c.av.drawn) continue;
        recoil_timer_ = 0.0f; muzzle_flash_timer_ = 0.0f; weapon_ = 0; flashlight_on_ = false;
        mp_draw_ = &c.av;
        draw_viewmodel(cmd, vp, c.av.view_inv);
        mp_draw_ = nullptr;
    }
    recoil_timer_ = sv_recoil; muzzle_flash_timer_ = sv_muzzle; weapon_ = sv_weapon; flashlight_on_ = sv_flash;
    body_recoil_ = sv_body_recoil;
}

void VulkanEngine::corpses_destroy() {
    for (Corpse& c : corpses_) {
        if (c.rag && physics_) physics_->remove_ragdoll(c.rag);
        c.rag = 0;
        for (uint32_t i = 0; i < kFrameOverlap; ++i) {
            if (c.av.vbuf[i]) vmaDestroyBuffer(allocator_, c.av.vbuf[i], c.av.valloc[i]);
            c.av.vbuf[i] = VK_NULL_HANDLE;
            c.av.mapped[i] = nullptr;
        }
    }
}

}  // namespace qlike
