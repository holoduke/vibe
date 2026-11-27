// The people and cars that live at the service station and on the coast highway.
//
// What they do is the NPC engine's business (src/npc: no drawing, no physics, no network in
// it). This file gives it its site and its road, steps it, and shows what it says:
//   * people as the soldiers other players are - the same animator, skinned mesh and draw
//     (multiplayer.cpp mp_pose_avatars is the pattern) - walking, standing, or sitting with the
//     animator's seat pose. Only the nearest are posed and drawn: the frame has a bone palette
//     for 48 bodies in all (kSkinCharacters), shared with the players and the fallen.
//   * cars as cars without physics (jeep.cpp npc_cars_place: a body matrix and rolling wheels,
//     as another machine's car is shown).
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/multiplayer.h"
#include "engine/highway.h"
#include "engine/net/net_anim.h"
#include "engine/terrain.h"
#include "engine/log.h"
#include "anim/anim_lod.h"
#include "engine/vk_engine/state/people.h"
#include <glm/gtc/matrix_transform.hpp>
#include "npc/npc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>

namespace vibe {

namespace {
constexpr int   kNpcBodies = 40;           // posed and drawn at most (of the 48 a frame can skin; 28 until the pavilion and the pad had people)
constexpr float kNpcBodyReach = 150.0f;    // metres from the eye
constexpr float kNpcCarBodyReach = 90.0f; // ... for those who sit in a car
// (a segment against a box: multiplayer.cpp's)
bool seg_aabb(glm::vec3 a, glm::vec3 b, glm::vec3 mn, glm::vec3 mx, float& t_hit) {
    const glm::vec3 d = b - a;
    float t0 = 0.0f, t1 = 1.0f;
    for (int k = 0; k < 3; ++k) {
        if (std::abs(d[k]) < 1e-7f) { if (a[k] < mn[k] || a[k] > mx[k]) return false; continue; }
        float u0 = (mn[k] - a[k]) / d[k], u1 = (mx[k] - a[k]) / d[k];
        if (u0 > u1) std::swap(u0, u1);
        t0 = std::max(t0, u0); t1 = std::min(t1, u1);
        if (t0 > t1) return false;
    }
    t_hit = t0;
    return true;
}
}

// WHAT THEY CARRY (npc.h Item; docs/ARCHITECTURE.md "What people carry"). The NPC engine says what
// somebody has in the hands and what it has set down where it is, and when it sets a thing down
// or takes it up (acts PutDown / PickUp, at View::aim). Here: the things themselves - small meshes
// put together from the box and the cylinder - and how each is held. A HOLD is the item's frame in
// the body's model space (from the chest, last frame's) and, for each hand that holds it, the
// grip: the middle of the palm, the way the fingers point, the way the palm faces - which the
// animator's hold layer (anim/hold_layer.cpp) takes the arm to, last of everything. So the arm
// swing of a walk, a clip's arms, the seat pose keep the hand that is free; the other is on the
// cup, the box, the bag. What the hand does WITH it is moved here: the cup to the mouth now and
// then, the board and the tablet up to be read (and the head looks at them), the box down on
// the table and up again.
namespace carry {

struct Piece { int mesh; glm::vec3 at, size, color; float glow; };      // mesh: 0 the box, 1 the cylinder (size: radius, half height, radius)
struct Thing { const Piece* p; int n; };
constexpr Piece kBox[] = { { 0, { 0.0f, 0.0f, 0.0f }, { 0.40f, 0.28f, 0.30f }, { 0.36f, 0.24f, 0.12f }, 0.0f },
                           { 0, { 0.0f, 0.141f, 0.0f }, { 0.06f, 0.004f, 0.302f }, { 0.55f, 0.48f, 0.34f }, 0.0f } };
constexpr Piece kClipboard[] = { { 0, { 0.0f, 0.0f, 0.0f }, { 0.23f, 0.32f, 0.006f }, { 0.30f, 0.20f, 0.11f }, 0.0f },
                                 { 0, { 0.0f, -0.012f, 0.004f }, { 0.21f, 0.27f, 0.002f }, { 0.80f, 0.80f, 0.78f }, 0.0f },
                                 { 0, { 0.0f, 0.14f, 0.007f }, { 0.09f, 0.03f, 0.008f }, { 0.55f, 0.56f, 0.58f }, 0.0f } };
constexpr Piece kCup[] = { { 1, { 0.0f, 0.0f, 0.0f }, { 0.040f, 0.055f, 0.040f }, { 0.75f, 0.72f, 0.66f }, 0.0f },
                           { 1, { 0.0f, -0.005f, 0.0f }, { 0.042f, 0.022f, 0.042f }, { 0.30f, 0.17f, 0.08f }, 0.0f },
                           { 1, { 0.0f, 0.058f, 0.0f }, { 0.043f, 0.005f, 0.043f }, { 0.05f, 0.05f, 0.05f }, 0.0f } };
constexpr Piece kBag[] = { { 0, { 0.0f, 0.0f, 0.0f }, { 0.30f, 0.34f, 0.13f }, { 0.48f, 0.36f, 0.22f }, 0.0f },
                           { 0, { -0.06f, 0.21f, 0.0f }, { 0.012f, 0.10f, 0.012f }, { 0.40f, 0.30f, 0.18f }, 0.0f },
                           { 0, { 0.06f, 0.21f, 0.0f }, { 0.012f, 0.10f, 0.012f }, { 0.40f, 0.30f, 0.18f }, 0.0f },
                           { 0, { 0.0f, 0.255f, 0.0f }, { 0.13f, 0.012f, 0.012f }, { 0.40f, 0.30f, 0.18f }, 0.0f } };
constexpr Piece kToolbox[] = { { 0, { 0.0f, 0.0f, 0.0f }, { 0.20f, 0.20f, 0.46f }, { 0.55f, 0.05f, 0.03f }, 0.0f },
                               { 0, { 0.0f, 0.103f, 0.0f }, { 0.205f, 0.01f, 0.465f }, { 0.08f, 0.08f, 0.08f }, 0.0f },
                               { 0, { 0.0f, 0.17f, -0.10f }, { 0.02f, 0.13f, 0.02f }, { 0.06f, 0.06f, 0.06f }, 0.0f },
                               { 0, { 0.0f, 0.17f, 0.10f }, { 0.02f, 0.13f, 0.02f }, { 0.06f, 0.06f, 0.06f }, 0.0f },
                               { 0, { 0.0f, 0.23f, 0.0f }, { 0.03f, 0.025f, 0.22f }, { 0.06f, 0.06f, 0.06f }, 0.0f } };
constexpr Piece kPhone[] = { { 0, { 0.0f, 0.0f, 0.0f }, { 0.074f, 0.154f, 0.009f }, { 0.02f, 0.02f, 0.022f }, 0.0f },
                             { 0, { 0.0f, 0.0f, 0.0047f }, { 0.068f, 0.146f, 0.001f }, { 0.10f, 0.14f, 0.20f }, 0.35f } };
constexpr Piece kTablet[] = { { 0, { 0.0f, 0.0f, 0.0f }, { 0.25f, 0.175f, 0.008f }, { 0.12f, 0.12f, 0.13f }, 0.0f },
                              { 0, { 0.0f, 0.0f, 0.0042f }, { 0.23f, 0.155f, 0.001f }, { 0.12f, 0.17f, 0.24f }, 0.30f } };
inline Thing thing(uint8_t item) {
    switch (npc::Item(item)) {
    case npc::Item::Box: return { kBox, 2 };
    case npc::Item::Clipboard: return { kClipboard, 3 };
    case npc::Item::Cup: return { kCup, 3 };
    case npc::Item::Bag: return { kBag, 4 };
    case npc::Item::Toolbox: return { kToolbox, 5 };
    case npc::Item::Phone: return { kPhone, 2 };
    case npc::Item::Tablet: return { kTablet, 2 };
    default: return { nullptr, 0 };
    }
}
// How far its bottom is below its middle (what it stands on, set down).
inline float half_height(uint8_t item) {
    switch (npc::Item(item)) {
    case npc::Item::Box: return 0.14f; case npc::Item::Cup: return 0.055f; case npc::Item::Bag: return 0.17f;
    case npc::Item::Toolbox: return 0.10f; default: return 0.004f;      // (a board, a tablet, a phone lie flat)
    }
}

struct Hold {
    glm::mat4 M{1.0f};          // the item, model space
    uint8_t   hands = 0;        // 1 left, 2 right
    glm::vec3 grip[2]{}, fingers[2]{}, palm[2]{};
    float     curl[2]{ 1.0f, 1.0f };
    bool      read = false;     // the head looks at it
};

inline glm::mat4 frame(glm::vec3 p, float pitch) {
    return glm::translate(glm::mat4(1.0f), p) * glm::rotate(glm::mat4(1.0f), pitch, glm::vec3(1.0f, 0.0f, 0.0f));
}
inline void grip(Hold& h, int side, glm::vec3 at, glm::vec3 fingers, glm::vec3 palm, float curl) {
    h.hands |= uint8_t(1u << side);
    h.grip[side] = glm::vec3(h.M * glm::vec4(at, 1.0f));
    h.fingers[side] = glm::normalize(glm::mat3(h.M) * fingers);
    h.palm[side] = glm::normalize(glm::mat3(h.M) * palm);
    h.curl[side] = curl;
}

// How `item` is carried by somebody whose chest (Spine2) is at C in model space (+x right, +y up,
// -z forward). drink: 0 .. 1, how far a cup is on its way to the mouth; look: 0 .. 1, how far a
// board, a tablet or a phone is up to be read.
inline Hold carried(uint8_t item, glm::vec3 C, float drink, float look) {
    Hold h;
    switch (npc::Item(item)) {
    case npc::Item::Box:
        h.M = frame(C + glm::vec3(0.0f, -0.36f, -0.31f), 0.0f);
        grip(h, 0, { -0.205f, -0.03f, 0.03f }, { 0.0f, 0.15f, -1.0f }, { 1.0f, 0.0f, 0.0f }, 0.7f);
        grip(h, 1, { 0.205f, -0.03f, 0.03f }, { 0.0f, 0.15f, -1.0f }, { -1.0f, 0.0f, 0.0f }, 0.7f);
        break;
    case npc::Item::Cup: {
        const glm::vec3 low = C + glm::vec3(0.17f, -0.30f, -0.23f), mouth = C + glm::vec3(0.035f, 0.30f, -0.135f);
        const float d = drink * drink * (3.0f - 2.0f * drink);
        h.M = frame(glm::mix(low, mouth, d), 0.95f * d);
        grip(h, 1, { 0.046f, -0.005f, 0.0f }, { -0.25f, 0.0f, -1.0f }, { -1.0f, 0.0f, 0.0f }, 0.9f);
        break; }
    case npc::Item::Bag:
        h.M = frame(C + glm::vec3(0.235f, -0.78f, 0.03f), 0.0f);
        grip(h, 1, { 0.0f, 0.255f, 0.0f }, { -0.35f, -0.25f, -1.0f }, { -1.0f, 0.0f, 0.0f }, 1.0f);
        break;
    case npc::Item::Toolbox:
        h.M = frame(C + glm::vec3(0.255f, -0.79f, 0.0f), 0.0f);
        grip(h, 1, { 0.0f, 0.23f, 0.0f }, { -0.30f, -0.20f, -1.0f }, { -1.0f, 0.0f, 0.0f }, 1.0f);
        break;
    case npc::Item::Phone: {
        const float l = look * look * (3.0f - 2.0f * look);
        h.M = frame(glm::mix(C + glm::vec3(0.16f, -0.32f, -0.12f), C + glm::vec3(0.07f, -0.02f, -0.30f), l), glm::mix(-0.2f, -0.75f, l));
        grip(h, 1, { 0.0f, -0.035f, -0.02f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 0.75f);
        h.read = l > 0.5f;
        break; }
    case npc::Item::Tablet:
    case npc::Item::Clipboard: {
        // (held on the left forearm, the board's lower left corner in the hand; up to be read)
        const float l = look * look * (3.0f - 2.0f * look);
        const bool tab = item == uint8_t(npc::Item::Tablet);
        h.M = frame(glm::mix(C + glm::vec3(-0.16f, -0.36f, -0.16f), C + glm::vec3(-0.03f, -0.13f, -0.30f), l), glm::mix(-0.35f, -0.85f, l));
        grip(h, 0, { tab ? -0.105f : -0.095f, tab ? -0.05f : -0.11f, -0.012f }, { 0.55f, 0.45f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 0.65f);
        h.read = l > 0.5f;
        break; }
    default: break;
    }
    return h;
}

// The item set down at `place` (model space: where its bottom is), standing upright; a board, a
// tablet or a phone lying flat.
inline glm::mat4 set_down(uint8_t item, glm::vec3 place) {
    const bool flat = item == uint8_t(npc::Item::Clipboard) || item == uint8_t(npc::Item::Tablet) || item == uint8_t(npc::Item::Phone);
    glm::mat4 M = glm::translate(glm::mat4(1.0f), place + glm::vec3(0.0f, half_height(item), 0.0f));
    if (flat) M = M * glm::rotate(glm::mat4(1.0f), -1.5708f, glm::vec3(1.0f, 0.0f, 0.0f));
    return M;
}

// What each shown body carries this frame (draw_npc_bodies draws it): the item and its frame in
// the body's model space; and each body's chest last frame.
struct Shown { uint8_t item = 0; glm::mat4 M{1.0f}; uint32_t id = 0; glm::vec3 chest{0.0f}; bool chest_ok = false; };

}  // namespace carry
static carry::Shown g_carry_shown[kNpcBodies];      // (per shown body: npc_pose fills it, draw_npc_bodies draws it)

// HOW SOMEBODY WHO IS SHOT GOES DOWN. Not a doll thrown by the shot: the body first REACTS - a
// clip made for where the shot struck and what the person was doing (text to motion,
// docs/animation/actions.md "Death") - and at the moment that reaction loses its balance the
// physics takes the body, each part moving as the clip moved it (corpse_spawn's part_velocity),
// and lets it fall on whatever is there. `hand`: seconds into the clip at which the hips have
// begun to drop (from the take's own numbers; the clips are cut to begin at the reaction).
namespace {
struct ShotClip { const char* name; float hand; };
const ShotClip kShotFront = { "shot_front", 1.63f };      // staggers back, collapses on the back
const ShotClip kShotGut   = { "shot_gut",   1.07f };      // doubles over, to the knees, forward
const ShotClip kShotSpin  = { "shot_spin",  1.00f };      // spun round by a shoulder hit
const ShotClip kShotBack  = { "shot_back",  0.73f };      // thrown forward on the face
const ShotClip kShotHead  = { "shot_head",  0.63f };      // limp at once, straight down
const ShotClip kShotLeg   = { "shot_leg",   0.77f };      // the leg gives way, down sideways
const ShotClip kShotRun   = { "shot_run",   0.43f };      // trips at a run, tumbles
const ShotClip kShotWalk  = { "shot_walk",  1.30f };      // stops, staggers, collapses
// (2026-11-27, vibe_prompts/shot2.txt: shot_knees s37 frames 28 - 96, shot_side s11 frames 36 - 88)
const ShotClip kShotKnees = { "shot_knees", 1.80f };      // the knees buckle, a moment kneeling, then forward on the face
const ShotClip kShotSide  = { "shot_side",  1.07f };      // struck on the right of the chest: staggers right and back, down on that side
}

// Which reaction: by where the shot struck (head, legs, trunk), from which side (the bolt's
// direction against the way the person faces) and how fast the person was moving. The same on
// every machine: all of it comes from the hit that was sent and the person's own state.
// (2026-11-27) By WHERE on the body as well: a shot high on one side of the chest - a
// shoulder - spins the body round that side; a leg gives way on its own side; a chest from the
// front throws it back, the gut doubles it over; and every clip is then played left for right
// when the shot came from the other side than the take's (npc_shot_plan).
// Kinds of the plan: how the direction is fitted after the clip is chosen.
enum ShotFit : int { kFitNone = 0, kFitPush = 1, kFitTowardShooter = 2, kFitLegSide = 3, kFitSpin = 4, kFitSideHit = 5 };
struct ShotPick { const ShotClip* clip; int fit; float knock; };
static ShotPick pick_shot_clip(float height, float lateral, bool from_front, float speed, uint32_t id) {
    const bool shoulder = height > 1.20f && std::abs(lateral) > 0.13f;
    if (height > 1.50f && !shoulder) return { &kShotHead, kFitNone, 0.15f };
    if (speed > 2.4f) return { &kShotRun, kFitNone, 0.25f };
    if (height < 0.75f) return { &kShotLeg, kFitLegSide, 0.30f };
    if (shoulder) return { &kShotSpin, kFitSpin, 0.35f };
    if (!from_front) return { &kShotBack, kFitPush, 0.40f };
    if (speed > 0.5f) return { &kShotWalk, kFitPush, 0.35f };
    // (the chest a hand or more off the middle, below the shoulder: down on that side - the take is
    //  struck on the right, played left for right for the left)
    if (height >= 1.05f && std::abs(lateral) > 0.09f && id % 2u == 0u) return { &kShotSide, kFitSideHit, 0.35f };
    if (height < 1.05f) return id % 3u == 0u ? ShotPick{ &kShotFront, kFitPush, 0.40f }
                             : id % 3u == 1u ? ShotPick{ &kShotKnees, kFitTowardShooter, 0.20f } : ShotPick{ &kShotGut, kFitTowardShooter, 0.25f };
    return id % 4u == 0u ? ShotPick{ &kShotGut, kFitTowardShooter, 0.25f } : ShotPick{ &kShotFront, kFitPush, 0.45f };
}
// What the takes do by their prompts (vibe_prompts/shot.txt), for when a clip's own travel is
// too small to tell: model space (x right, -z forward), and the turn (+ = to the left).
static void shot_clip_authored(const char* name, glm::vec3& travel, float& yaw) {
    const std::string n(name);
    travel = glm::vec3(0.0f); yaw = 0.0f;
    if (n == "shot_front" || n == "shot_walk") travel = glm::vec3(0.0f, 0.0f, 1.0f);       // back
    else if (n == "shot_gut" || n == "shot_back" || n == "shot_run") travel = glm::vec3(0.0f, 0.0f, -1.0f);
    else if (n == "shot_leg") travel = glm::vec3(1.0f, 0.0f, 0.0f);                        // to the right (the right leg)
    else if (n == "shot_side") travel = glm::vec3(0.6f, 0.0f, 0.8f);                       // right and back
    else if (n == "shot_knees") travel = glm::vec3(0.0f, 0.0f, -1.0f);                     // forward
    else if (n == "shot_spin") yaw = 1.0f;                                                 // to the left (the left shoulder)
}

void VulkanEngine::npc_dying_pose(float dt) {
    if (npc_dying_.empty() || body_bones_.empty() || !anim_set_.ok()) { npc_dying_.clear(); return; }
    dt = glm::clamp(dt, 1e-4f, 0.1f);
    const anim::IAnimationRuntime& rtm = *anim_set_.runtime;
    for (size_t i = 0; i < npc_dying_.size();) {
        NpcDying& d = npc_dying_[i];
        if (d.fresh) { d.fresh = false; ++i; continue; }      // (posed this frame already, as it stood)
        d.t += dt;
        anim::CharacterAnimationInput in = d.in;
        in.dt = dt;
        in.action_clip = d.clip;
        in.action_time = std::min(d.t, rtm.clip(d.clip).duration);
        in.action_in_place = false; in.action_warp = false; in.action_warp_full = false;
        in.action_warp_from_start = false; in.action_end_at_origin = false;
        in.velocity = in.desired_velocity = glm::vec3(0.0f);
        in.has_look_at = false; in.teleported = false; in.seated = false;
        // (of the life layers only the posture stays; what was going on lets go by itself)
        in.life_on_clip = false; in.fidgets = 0.0f; in.layer_clip = anim::kNoClip; in.action_mirror = d.mirror;
        in.lod = std::min(d.av.lod, 1);
        // (turned the way the shot pushed, over the first third of a second: about where it stood)
        if (std::abs(d.turn) > 1e-3f) {
            const glm::vec3 o(d.model0[3]);
            const float k = glm::smoothstep(0.0f, 0.35f, d.t);
            d.model = glm::translate(glm::mat4(1.0f), o) * glm::rotate(glm::mat4(1.0f), d.turn * k, glm::vec3(0.0f, 1.0f, 0.0f)) *
                      glm::translate(glm::mat4(1.0f), -o) * d.model0;
        }
        d.av.palette_prev = d.av.palette;
        d.av.palette = -1;
        pose_character(*d.av.animator, in, &d.av.palette, &d.av.pose_rows);
        glm::vec3 now[kCorpseParts];
        const bool have = corpse_part_centres(d.model, now);
        if (have && d.hit_part < 0) {        // (the part the shot struck, in the first pose: it is followed from here)
            float best = 1e9f;
            for (int k = 0; k < kCorpseParts; ++k) { const float dd = glm::distance(now[k], d.hit_pos); if (dd < best) { best = dd; d.hit_part = k; } }
        }
        if (d.t >= d.hand || !have) {
            // The physics' from here: every part goes on as the clip moved it (no faster than a
            // falling body moves), and the shot itself adds nothing - the reaction was the clip.
            glm::vec3 vel[kCorpseParts];
            for (int k = 0; k < kCorpseParts; ++k) {
                vel[k] = d.have_prev && have ? (now[k] - d.parts_prev[k]) / dt : glm::vec3(0.0f);
                const float sp = glm::length(vel[k]);
                if (sp > 7.0f) vel[k] *= 7.0f / sp;
            }
            glm::mat4 vi = d.view_inv;
            if (have) vi[3] = glm::vec4(now[2], 1.0f);        // (the head, where it is now)
            // (and the shot's own push, on the part it struck as that part is now: the fall goes on the way it was driven)
            const glm::vec3 at = have && d.hit_part >= 0 ? now[d.hit_part] : d.hit_pos;
            corpse_spawn(d.model, vi, glm::vec3(0.0f), at, d.hit_dir, d.knock, vel);
            npc_dying_.erase(npc_dying_.begin() + ptrdiff_t(i));
            continue;
        }
        for (int k = 0; k < kCorpseParts; ++k) d.parts_prev[k] = now[k];
        d.have_prev = true;
        d.av.prev_model = d.av.model; d.av.model = d.model;
        d.av.eye = now[2];
        d.av.view_inv = d.view_inv; d.av.view_inv[3] = glm::vec4(now[2], 1.0f);
        d.av.mesh_lod = body_mesh_lod(d.av.eye, d.av.mesh_lod);
        d.av.drawn = d.av.palette >= 0;
        ++i;
    }
}

void VulkanEngine::npc_tick(float dt) {
    static_assert(sizeof(npc_gone_) / sizeof(npc_gone_[0]) >= size_t(npc::kMaxAgents) && sizeof(npc_nodriver_) / sizeof(npc_nodriver_[0]) >= size_t(npc::kMaxAgents), "a place for every agent");
    // Made once the road and the station are there (the road is built on another thread).
    if (!npc_) {
        if (!highway::ready() || station_pos_ == glm::vec3(0.0f) || std::getenv("VIBE_NO_NPC")) return;
        npc_ = std::make_unique<npc::World>();
        if (!npc_->load_site("assets/gasstation/gasstation.nav", station_pos_))
            log::warnf("[npc] assets/gasstation/gasstation.nav missing - no people at the station (blender -b --python tools/gasstation/make_gasstation.py)");
        npc::Road road;
        road.length = highway::course_length();
        // (the course is points 20 m apart: the direction is taken over a car length, so that
        //  a car turns through a bend instead of at its corners)
        road.at = [](float s, glm::vec3& pos, glm::vec2& dir) {
            glm::vec3 a, b; glm::vec2 d;
            if (!highway::point_along(s, pos, dir) || !highway::point_along(s - 5.0f, a, d) || !highway::point_along(s + 5.0f, b, d)) return false;
            const glm::vec2 ab(b.x - a.x, b.z - a.z);
            if (glm::dot(ab, ab) > 1e-4f) dir = glm::normalize(ab);
            pos.y -= highway::cross_fall(1.9f);
            return true;
        };
        npc_->set_road(std::move(road));
        // The pavilion by the launch pad and the pad itself: people of their own, who are only
        // there while a player is near (npc.cpp "THE ADDED SITES"). VIBE_NO_CROWD=1: without them.
        if (!std::getenv("VIBE_NO_CROWD") && launch_site_placed_) {
            const bool office = office_pos_ != glm::vec3(0.0f) && npc_->add_site("assets/office/office.nav", office_pos_);
            const bool pad = npc_->add_site("assets/rocket/site.nav", launch_pad_);
            if (!office) log::warnf("[npc] assets/office/office.nav missing - nobody in the pavilion (blender -b --python tools/office/make_office.py)");
            if (!pad) log::warnf("[npc] assets/rocket/site.nav missing - nobody on the launch pad");
        }
        // The soldiers' run: from the launch pad along the coast road (on its edge, the side the
        // castle lane leaves it) and up the lane to the castle's car park - and back.
        if (highway::line_count() > 1 && !std::getenv("VIBE_NO_RUNNERS")) {
            const std::vector<glm::vec3>& C = highway::line_points(0);
            const std::vector<glm::vec3>& L = highway::line_points(1);
            if (C.size() > 8 && L.size() > 2) {
                const int nC = int(C.size());
                auto nearest_i = [&](glm::vec2 q) { int b = 0; float bd = 1e18f; for (int i = 0; i < nC; ++i) { const float d = glm::distance(q, glm::vec2(C[size_t(i)].x, C[size_t(i)].z)); if (d < bd) { bd = d; b = i; } } return b; };
                const int ia = nearest_i(glm::vec2(1600.0f, 830.0f)), ib = nearest_i(glm::vec2(L.front().x, L.front().z));
                const int fwd = (ib - ia + nC) % nC, step = fwd <= nC - fwd ? 1 : -1, count = std::min(fwd, nC - fwd);
                const float edge = std::max(0.5f * highway::line_width(0) - 1.5f, 2.6f);
                auto beside = [&](int i, float sgn) {
                    const glm::vec3 a = C[size_t((i - step + nC) % nC)], b = C[size_t((i + step + nC) % nC)];
                    const glm::vec2 d = glm::normalize(glm::vec2(b.x - a.x, b.z - a.z) + glm::vec2(1e-6f, 0.0f));
                    glm::vec3 p = C[size_t(i)] + glm::vec3(-d.y, 0.0f, d.x) * (edge * sgn);
                    p.y -= highway::cross_fall(edge);
                    return p;
                };
                const float sgn = glm::distance(beside(ib, 1.0f), L.front()) < glm::distance(beside(ib, -1.0f), L.front()) ? 1.0f : -1.0f;
                std::vector<glm::vec3> run;
                for (int k = 0; k <= count; ++k) run.push_back(beside((ia + step * k + nC * 4) % nC, sgn));
                for (const glm::vec3& p : L) run.push_back(p);
                float len = 0.0f; for (size_t i = 1; i < run.size(); ++i) len += glm::distance(run[i - 1], run[i]);
                log::infof("[npc] the soldiers' run: %zu points, %.0f m, from (%.0f, %.0f) to (%.0f, %.0f)", run.size(), double(len), run.front().x, run.front().z, run.back().x, run.back().z);
                npc_->set_run_path(std::move(run));
            }
        }
        npc_->populate(20261124u);
        if (std::getenv("VIBE_TEST_SIREN")) npc_->set_sirens(true);      // (tests: every police car's siren on, and it stays on)
        // VIBE_NPC_WARM=<seconds>: so much of their life has passed already when the game starts
        // (tests: to be there at a chosen moment). With VIBE_LOG_NPC the moments a car turns in
        // to the station are logged on the way.
        if (const char* warm = std::getenv("VIBE_NPC_WARM")) {
            int at_site = npc_->census().cars_at_site;
            const int steps = int(std::atof(warm) / npc::kStep);
            for (int i = 0; i < steps; ++i) {
                npc_->update(npc::kStep);
                const int now = npc_->census().cars_at_site;
                if (now > at_site && std::getenv("VIBE_LOG_NPC")) log::infof("[npc] warm: a car turns in at %.1f s", double(i) * double(npc::kStep));
                at_site = now;
                if (std::getenv("VIBE_LOG_NPC"))                 // (and the moments somebody begins to sit down, get up, get in or out)
                    for (const npc::View& v : npc_->state())
                        if (v.kind == npc::Kind::Person && v.act != npc::Act::None && v.act_t < npc::kStep * 1.5f)
                            log::infof("[npc] warm: person %d begins act %d at %.1f s, at (%.1f, %.1f, %.1f) yaw %.2f", int(v.id), int(v.act), double(i) * double(npc::kStep), v.pos.x, v.pos.y, v.pos.z, v.yaw);
            }
        }
        npc_bodies_ = std::vector<MpAvatar>(size_t(kNpcBodies));
        const npc::World::Census c = npc_->census();
        log::infof("[npc] %d agents: %d people (%d in cars), %d cars; road %.0f m", npc_->agent_count(), c.people, c.hidden, c.cars, double(highway::course_length()));
    }
    // Not in another part of the planet (the site and the road are home's).
    if (!globe::current().home) { npc_views_.clear(); npc_cars_place(npc_views_, dt); for (MpAvatar& a : npc_bodies_) a.drawn = false; return; }
    // Multiplayer: the host's world is the world. A client takes in what the host sends and
    // only follows it (npc::World::apply makes it a replica); the host - and a machine that
    // plays alone - steps its own and tells the clients four times a second. A client that
    // has to take over (the host left) starts its world anew: a replica has only the picture.
    const net::Role role = net_ ? net_->role() : net::Role::Off;
    if (role == net::Role::Client) {
        std::vector<uint8_t> m;
        if (net_->take_npc(m) && !npc_->apply(m.data(), m.size())) log::warnf("[npc] a damaged state from the host (%zu bytes)", m.size());
    } else if (npc_->replica()) {
        npc_->populate(20261124u);
        log::infof("[npc] the host is gone: this machine runs the world now");
    } else if (role == net::Role::Host) {
        npc_send_ -= dt;
        if (npc_send_ <= 0.0f) {
            npc_send_ = 0.25f;
            std::vector<uint8_t> m;
            npc_->serialize(m);
            net_->send_npc(m.data(), m.size());
        }
    }
    // (VIBE_NPC_FAST=<n>: n times as fast - to watch an hour of their life in minutes)
    static const float kFast = std::getenv("VIBE_NPC_FAST") ? float(std::atof(std::getenv("VIBE_NPC_FAST"))) : 1.0f;
    // The players on foot, for the machine that steps the world: its cars do not drive into
    // them (npc::World::set_walkers) - which is also how a car on the road is made to stop, to
    // be taken. (VIBE_NPC_NO_WALKERS=1: the cars see nobody, as before.)
    if (!npc_->replica()) {
        static const bool kNoWalkers = std::getenv("VIBE_NPC_NO_WALKERS") != nullptr;
        std::vector<glm::vec3> feet;
        const float half = game::Player::kHalfExtents.y;
        if (!kNoWalkers) {
            if (!jeep_driving() && jeep_ride_ < 0 && !fly_mode_) feet.push_back(player_.position - glm::vec3(0.0f, half, 0.0f));
            if (mp_) for (const MpAvatar& a : mp_->avatars)
                if (a.id && a.present && !a.seated && a.ride == 0 && !a.corpse) feet.push_back(glm::vec3(a.st.px, a.st.py - half, a.st.pz));
        }
        npc_->set_walkers(feet);
        // (and where every player is, on foot or not: the people of the pavilion and the pad are
        //  there while one of them is near)
        std::vector<glm::vec3> eyes;
        eyes.push_back(player_.position);
        if (mp_) for (const MpAvatar& a : mp_->avatars) if (a.id && a.present) eyes.push_back(glm::vec3(a.st.px, a.st.py, a.st.pz));
        npc_->set_eyes(eyes);
        npc_feed_obstacles();                  // (and the players' cars: the traffic keeps behind them and goes round them - jeep.cpp)
    }
    // (VIBE_LOG_NPC: what the world's steps and the bodies' poses cost, as a mean over 300 frames)
    static double cost_step = 0.0, cost_pose = 0.0;
    const auto t_step = std::chrono::steady_clock::now();
    for (float left = dt * kFast; left > 1e-5f; left -= 0.1f) npc_->update(std::min(left, 0.1f));
    cost_step += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_step).count();
    npc_->views(npc_views_);
    for (float& t : npc_nodriver_) t = std::max(0.0f, t - dt);
    // (shot or taken a moment ago, and this copy of the host's world has not heard yet)
    for (size_t i = 0; i < npc_views_.size() && i < size_t(npc::kMaxAgents); ++i) {
        if (npc_gone_[i] <= 0.0f) continue;
        npc_gone_[i] -= dt;
        npc::View& v = npc_views_[i];
        v.hidden = true; v.car = 255; v.pos.y -= 20000.0f;
    }
    npc_cars_place(npc_views_, dt);
    police_place(dt);                          // (the police cars' light bars and sirens; VIBE_LOG_TRAFFIC - jeep.cpp)
    {
        const auto t_pose = std::chrono::steady_clock::now();
        npc_pose(dt);
        cost_pose += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pose).count();
    }
    npc_dying_pose(dt);
    blood_update(dt);
    // VIBE_TEST_NPC_KILL=<frame>[,<skip>]: at that frame the six people nearest the eye that have a
    // body are shot from where the eye is (after the <skip> nearest) - tests of how they fall.
    if (static const char* tk = std::getenv("VIBE_TEST_NPC_KILL"); tk) {
        int at = 0, skip = 0; std::sscanf(tk, "%d,%d", &at, &skip);
        if (int(frame_number_) == at) {
            std::vector<std::pair<float, int>> who;
            for (const MpAvatar& a : npc_bodies_) if (a.present && a.id >= 1 && a.id <= uint32_t(npc::kMaxAgents) && a.id <= npc_views_.size() && !npc_views_[a.id - 1].hidden) who.push_back({ glm::distance(a.eye, current_frame_view_.eye_pos), int(a.id) - 1 });
            std::sort(who.begin(), who.end());
            for (size_t i = size_t(skip); i < who.size() && i < size_t(skip) + 6; ++i) {
                const npc::View& v = npc_views_[size_t(who[i].second)];
                const glm::vec3 chest = v.pos + glm::vec3(0.0f, v.pose == npc::Pose::Sit ? 0.45f : 1.25f, 0.0f);
                log::infof("[npc] test: person %d (pose %d style %d, speed %.1f) at %.1f m", who[i].second, int(v.pose), int(v.style), v.speed, double(who[i].first));
                npc_hit(who[i].second, chest, glm::normalize(chest - current_frame_view_.eye_pos));
            }
        }
    }
    // VIBE_NPC_CAM=<who>[,<metres>[,<degrees>[,<height>]]]: the eye is put beside somebody and
    // stays with them (tests and captures; use it with VIBE_TEST_HOLD). <who>: bar, table (eat,
    // talk), counter, look, walk or walk<0..7> (the way of walking), run, driver (a car that
    // drives), desk, point, group, watch, tech (the pavilion's and the pad's people), id<n>. The
    // nearest to where the game started that fits. <degrees>: 0 looks them
    // in the face, 90 from their right side, 180 from behind; <height> above their feet.
    if (static const char* nc = std::getenv("VIBE_NPC_CAM"); nc) {
        char who[32] = { 0 }; float dist = 3.0f, ang = 35.0f, up = 1.45f;
        std::sscanf(nc, "%31[^,],%f,%f,%f", who, &dist, &ang, &up);
        const std::string w(who);
        static int held = -1;
        auto fits = [&](const npc::View& v) {
            if (v.kind == npc::Kind::Car) return v.pose == npc::Pose::Drive && std::abs(v.speed) > 3.0f && (w == "driver" || (w == "police" && (v.style & npc::kCarPolice)) || (w == "passing" && (v.style & npc::kCarOvertaking)));
            if (v.hidden) return false;
            if (w.rfind("id", 0) == 0) return int(v.id) == std::atoi(w.c_str() + 2);
            if (v.act != npc::Act::None) return false;
            if (w == "bar") return v.pose == npc::Pose::Sit && v.style == 1;
            if (w == "table") return v.pose == npc::Pose::Sit && (v.style == 2 || v.style == 3);
            if (w == "eat") return v.pose == npc::Pose::Sit && v.style == 2;
            if (w == "talk") return v.pose == npc::Pose::Sit && v.style == 3;
            if (w == "counter") return v.pose == npc::Pose::Work && v.style == 1;
            if (w == "look") return v.pose == npc::Pose::Stand && v.style > 0 && v.style < 5;
            if (w == "desk") return v.pose == npc::Pose::Sit && v.style == 4;
            if (w == "point") return v.pose == npc::Pose::Stand && v.style == 5;
            if (w == "group") return v.pose == npc::Pose::Stand && v.style == 6;
            if (w == "watch") return v.pose == npc::Pose::Stand && v.style == 7;
            if (w == "tech") return v.pose == npc::Pose::Work && v.style == 0 && v.car == 255 && glm::distance(v.pos, launch_pad_) < 120.0f;
            if (w == "run") return v.pose == npc::Pose::Run;
            if (w.rfind("walk", 0) == 0) return v.pose == npc::Pose::Walk && std::abs(v.speed) > 0.3f && (w.size() == 4 || int(v.look % 8) == std::atoi(w.c_str() + 4));
            return false;
        };
        if (held < 0 || size_t(held) >= npc_views_.size() || !fits(npc_views_[size_t(held)])) {
            held = -1; float bd = 1e18f;
            for (size_t i = 0; i < npc_views_.size(); ++i) {
                if (!fits(npc_views_[i])) continue;
                const float d = glm::distance(npc_views_[i].pos, player_.position);
                if (d < bd) { bd = d; held = int(i); }
            }
            if (held >= 0) {
                const npc::View& v = npc_views_[size_t(held)];
                log::infof("[npc] the eye goes to %s %d (pose %d style %d look %d) at (%.1f, %.1f, %.1f) yaw %.2f", v.kind == npc::Kind::Car ? "car" : "person", held, int(v.pose), int(v.style), int(v.look), v.pos.x, v.pos.y, v.pos.z, v.yaw);
            }
        }
        if (held >= 0) {
            const npc::View& v = npc_views_[size_t(held)];
            const bool sits = v.kind == npc::Kind::Person && v.pose == npc::Pose::Sit;
            glm::vec3 feet = v.pos;
            if (sits) feet.y -= float(npc::kSeatHeightCm[v.style == 1 ? 1 : 0]) * 0.01f;
            const float a = v.yaw + glm::radians(ang);
            const glm::vec3 eye = feet + glm::vec3(std::sin(a) * dist, up, -std::cos(a) * dist);
            const glm::vec3 d = glm::normalize(feet + glm::vec3(0.0f, sits ? 0.95f : 1.05f, 0.0f) - eye);
            player_.position = eye - (player_.eye_position() - player_.position);
            player_.velocity = glm::vec3(0.0f);
            player_.yaw = std::atan2(d.x, -d.z);
            player_.pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
            prev_player_position_ = player_.position;
            globe_moved_player_ = true;                    // (VIBE_TEST_HOLD holds this place now)
        }
    }
    if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg && frame_number_ % 300 == 0) {
        const npc::World::Census c = npc_->census();
        int drawn = 0; for (const MpAvatar& a : npc_bodies_) drawn += a.drawn ? 1 : 0;
        {
            int lent = 0, present = 0, lod3 = 0, nopal = 0; float near_car = 1e9f, near_man = 1e9f;
            for (const NpcCar& c : npc_cars_) lent += c.agent >= 0 ? 1 : 0;
            for (const MpAvatar& a : npc_bodies_) { present += a.present ? 1 : 0; lod3 += (a.present && a.lod >= 3) ? 1 : 0; nopal += (a.present && a.palette < 0) ? 1 : 0; }
            for (const npc::View& v : npc_views_) { const float d = glm::distance(v.pos, current_frame_view_.eye_pos); if (v.kind == npc::Kind::Car) near_car = std::min(near_car, d); else if (!v.hidden) near_man = std::min(near_man, d); }
            for (const npc::View& v : npc_views_) if (v.kind == npc::Kind::Person && !v.hidden && v.pose == npc::Pose::Walk) log::infof("[npc]   walker %d at %.1f %.1f %.1f yaw %.2f speed %.2f", int(v.id), v.pos.x, v.pos.y, v.pos.z, v.yaw, v.speed);
            for (const npc::View& v : npc_views_) if (v.kind == npc::Kind::Person && !v.hidden && v.pose != npc::Pose::Walk) log::infof("[npc]   person %d pose %d style %d at %.1f %.1f %.1f yaw %.2f", int(v.id), int(v.pose), int(v.style), v.pos.x, v.pos.y, v.pos.z, v.yaw);
            for (const npc::View& v : npc_views_) if (v.kind == npc::Kind::Car && v.pose == npc::Pose::Drive && glm::distance(v.pos, current_frame_view_.eye_pos) < 500.0f) log::infof("[npc]   driving car %d at %.1f %.1f %.1f yaw %.2f speed %.1f (frame %llu)", int(v.id), v.pos.x, v.pos.y, v.pos.z, v.yaw, v.speed, static_cast<unsigned long long>(frame_number_));
            { std::istringstream ds(npc_->describe_site()); std::string ln; while (std::getline(ds, ln)) log::infof("[npc]   %s", ln.c_str()); }
            log::infof("[npc] shown: car slots %zu lent %d, nearest car %.0f m; bodies present %d (level 3: %d, no palette: %d), nearest person %.0f m; bones %zu anim %d",
                       npc_cars_.size(), lent, double(near_car), present, lod3, nopal, double(near_man), body_bones_.size(), int(anim_set_.ok()));
        }
        log::infof("[npc] people %d: sitting %d walking %d in cars %d, not there (nobody near their site) %d, drawn %d | cars %d: at the station %d (longest %.0f s)%s",
                   c.people, c.sitting, c.walking, c.hidden, c.asleep, drawn, c.cars, c.cars_at_site, double(c.longest_visit), npc_->replica() ? " - following the host" : "");
        log::infof("[npc] cost: the world's steps %.3f ms a frame, the bodies' poses %.3f ms a frame (%d agents, %d bodies; mean of 300 frames)", cost_step / 300.0, cost_pose / 300.0, npc_->agent_count(), drawn);
        cost_step = cost_pose = 0.0;
    }
}

void VulkanEngine::npc_pose(float dt) {
    for (MpAvatar& a : npc_bodies_) { a.present = false; a.drawn = false; }
    if (npc_views_.empty() || body_bones_.empty() || !anim_set_.ok()) return;
    dt = glm::clamp(dt, 0.0f, 0.1f);
    const glm::vec3 cam = current_frame_view_.eye_pos;
    const float tan_half = current_frame_view_.proj[1][1] != 0.0f ? 1.0f / std::abs(current_frame_view_.proj[1][1]) : 0.7f;

    // ---- who is shown: the nearest people on foot, and those in the cars that are shown - the
    //      driver and whoever sits beside him. A car that drives has a driver: where the engine
    //      has nobody at its wheel (the traffic the other way carries no people of its own),
    //      one is shown all the same.
    struct Who { float d; int view; int car; int seat; uint32_t id; };
    std::vector<Who> near_;
    std::vector<uint8_t> has_driver(npc_views_.size(), 0);
    for (size_t i = 0; i < npc_views_.size(); ++i) {
        const npc::View& v = npc_views_[i];
        if (v.kind != npc::Kind::Person) continue;
        if (v.car < npc_views_.size() && v.seat == 0) has_driver[v.car] = 1;
        const float d = glm::distance(v.pos, cam);
        if (!v.hidden) { if (d < kNpcBodyReach) near_.push_back({ d, int(i), -1, 0, uint32_t(v.id) + 1u }); }
        else if (v.car < npc_views_.size() && v.seat < 2 && d < kNpcCarBodyReach) near_.push_back({ d, int(i), int(v.car), int(v.seat), uint32_t(v.id) + 1u });
    }
    for (size_t i = 0; i < npc_views_.size(); ++i) {
        const npc::View& v = npc_views_[i];
        if (v.kind != npc::Kind::Car || has_driver[i] || v.pose != npc::Pose::Drive) continue;
        if (i < 100 && npc_nodriver_[i] > 0.0f) continue;      // (that driver was shot a moment ago: npc_hit)
        const float d = glm::distance(v.pos, cam);
        if (d < kNpcCarBodyReach) near_.push_back({ d, int(i), int(i), 0, 1000u + uint32_t(v.id) });
    }
    // (VIBE_NPC_NO_RIDERS=1: nobody in the cars - to compare a picture with and without them)
    if (static const bool none = std::getenv("VIBE_NPC_NO_RIDERS") != nullptr; none)
        near_.erase(std::remove_if(near_.begin(), near_.end(), [](const Who& w) { return w.car >= 0; }), near_.end());
    std::sort(near_.begin(), near_.end(), [](const Who& a, const Who& b) { return a.d < b.d; });
    if (near_.size() > size_t(kNpcBodies)) near_.resize(size_t(kNpcBodies));
    // ---- a body each: the one it had last frame, or a free one (which starts afresh)
    std::vector<int> body_of(near_.size(), -1);
    for (size_t k = 0; k < near_.size(); ++k) {
        const uint32_t id = near_[k].id;
        for (size_t b = 0; b < npc_bodies_.size(); ++b) if (npc_bodies_[b].id == id) { body_of[k] = int(b); npc_bodies_[b].present = true; break; }
    }
    for (MpAvatar& a : npc_bodies_) if (!a.present) a.id = 0;
    for (size_t k = 0; k < near_.size(); ++k) {
        if (body_of[k] >= 0) continue;
        for (size_t b = 0; b < npc_bodies_.size(); ++b) {
            MpAvatar& a = npc_bodies_[b];
            if (a.id) continue;
            a.id = near_[k].id;
            a.present = true; a.fresh_pose = true; a.unarmed = true;
            a.palette = a.palette_prev = -1; a.pose_rows.clear(); a.lod = 0; a.skipped = 0; a.skip_dt = 0.0f;
            if (a.animator) a.animator->reset();
            body_of[k] = int(b);
            break;
        }
    }

    // ---- the animators' inputs
    struct Posed { MpAvatar* a; anim::CharacterAnimationInput in; net::RemoteFrame rf; bool skip; };
    std::vector<Posed> posed;
    posed.reserve(near_.size());
    const anim::IAnimationRuntime& art = *anim_set_.runtime;
    const int hj = art.rig()[anim::Role::Hips];
    const glm::vec3 rest = hj >= 0 ? art.skeleton().rest_pos[size_t(hj)] : glm::vec3(0.0f, 0.95f, 0.0f);
    static double life_now = 0.0;                  // (this machine's own clock for what is only looks)
    life_now += double(dt);
    for (size_t k = 0; k < near_.size(); ++k) {
        if (body_of[k] < 0) continue;
        MpAvatar& a = npc_bodies_[size_t(body_of[k])];
        const npc::View& v = npc_views_[size_t(near_[k].view)];
        // WHO IT IS TO LOOK AT (player_body.cpp people_look): the agent and its look byte - the same on
        // every machine; a car's driver that the NPC engine does not have is the car's. Dressed as the
        // NPC engine says (npc::Dress); a police car's driver is a soldier of the base.
        a.person = ((uint32_t(v.id) + (v.kind == npc::Kind::Car ? 1000u : 0u)) << 8 | uint32_t(v.look)) + 1u;
        a.dress = v.kind == npc::Kind::Person ? uint8_t(npc::dress_of(v.look)) : uint8_t((v.style & npc::kCarPolice) ? npc::Dress::Garrison : npc::Dress::Civilian);
        a.palette_prev = a.fresh_pose ? -1 : a.palette;
        a.palette = -1;
        if (!a.animator) { a.animator = std::make_unique<anim::CharacterAnimator>(anim_set_); apply_anim_tuning(a.animator->tuning); a.fresh_pose = true; }
        // WHAT IT PLAYS. Most of what the station's people do is a clip made for it, round and
        // round (docs/animation/actions.md): at the bar, eating or talking at a table, a
        // cashier, somebody standing about some way of his own, and four ways of walking
        // besides the plain one. `loop_nm`: that clip, or null (the plain walk and stand, the
        // plain seat).
        const anim::IAnimationRuntime& rtm0 = *anim_set_.runtime;
        const bool run = v.pose == npc::Pose::Run;
        // (VIBE_NPC_NO_LIFE=1: everybody as the clip has them, nobody in particular - to compare)
        static const bool kNoLife = std::getenv("VIBE_NPC_NO_LIFE") != nullptr;
        static const int kGesture = std::getenv("VIBE_NPC_GESTURE") ? std::clamp(std::atoi(std::getenv("VIBE_NPC_GESTURE")), 0, int(anim::Gesture::Count) - 1) : 0;
        const char* loop_nm = nullptr;
        bool loop_seat = false, loop_walk = false;
        // WHO IT IS (anim/person_style.h): where in a loop this person starts, how fast it is
        // played, whether left for right, and - in the animator - a posture, a breath and a
        // restlessness of its own. All from the body's number: the same person every time.
        const uint32_t who_id = a.id * 7919u;
        const anim::PersonStyle who = anim::person_style(who_id);
        // A second clip over part of the body (the animator's clip layer), and which hands are
        // free for a fidget (1 left, 2 right).
        const char* layer_nm = nullptr;
        anim::LayerMask layer_mask = anim::LayerMask::UpperBody;
        float fidgets = 0.0f;
        uint8_t free_hands = 3;
        // AT A DESK (a seat of style 4; the pavilion): what this person is doing just now - 0
        // typing, 1 reading (a hand on the mouse), 2 leaning back, 3 turned to the neighbour,
        // talking - and who that neighbour is. A turn of 13 s or so each, everybody at a moment
        // of his own, by this machine's clock (the network says "at a desk", not which).
        int desk_do = -1, desk_peer = -1;
        const bool at_desk = near_[k].car < 0 && v.act == npc::Act::None && v.pose == npc::Pose::Sit && v.style == 4;
        if (at_desk) {
            uint32_t hs = uint32_t((life_now + double(v.id) * 5.3) / 13.0) * 2654435761u ^ (uint32_t(v.id) * 40503u);
            hs ^= hs >> 13; hs *= 0x5bd1e995u; hs ^= hs >> 15;
            static const int kDo[10] = { 0, 0, 0, 0, 1, 1, 2, 3, 3, 0 };
            desk_do = kDo[hs % 10u];
            if (static const char* force = std::getenv("VIBE_NPC_DESK"); force) desk_do = std::clamp(std::atoi(force), 0, 3);
            if (desk_do == 3) {
                float bd = 2.0f;
                for (const npc::View& o : npc_views_) {
                    if (o.kind != npc::Kind::Person || o.hidden || o.id == v.id || o.pose != npc::Pose::Sit || o.style != 4) continue;
                    const float d = glm::distance(o.pos, v.pos);
                    if (d < bd) { bd = d; desk_peer = int(o.id); }
                }
                if (desk_peer < 0) desk_do = 1;                  // (nobody beside him: he reads)
            }
        }
        if (near_[k].car < 0 && v.act == npc::Act::None) {
            static const char* kShow = std::getenv("VIBE_NPC_ACT");
            // Somebody who stays does not do one thing for ever: at the bar they drink, talk and
            // sit; at a table eat, drink, talk, lean back; the cashier scans, takes, waits. Each
            // in turn, a quarter of a minute or so, everybody at their own moment (`turn`: this
            // machine's own clock - the network sends where somebody is and what they do, not
            // which clip). A way of WALKING is the person's own and never changes (`look`).
            static double life_clock = 0.0;
            static uint64_t life_frame = ~0ull;
            if (life_frame != frame_number_) { life_frame = frame_number_; life_clock += double(dt); }
            auto turn = [&](int n, double period) { return int((uint32_t(v.id) * 7u + uint32_t((life_clock + double(v.id) * 3.7) / period)) % uint32_t(n)); };
            static const char* kBar[3] = { "bar_sit", "bar_talk", "bar_idle" };
            static const char* kEat[3] = { "table_eat", "table_drink", "table_eat" };
            static const char* kTalk[3] = { "table_talk", "table_idle", "table_drink" };
            static const char* kTill[3] = { "cashier", "cashier_take", "cashier_idle" };
            static const char* kLook[5][2] = { { nullptr, nullptr }, { "look_crossed", "look_around" }, { "look_hips", "look_hips" }, { "look_phone", "look_watch" }, { "look_far", "look_far" } };
            // (2 was "walk_tired" and 6 "walk_swagger" until 2026-11-26: a shuffle at 0.57 m/s with the
            //  body hanging, and a walk with the arms held wide of the body - the two that read as
            //  the walking dead. Both clips are still in the set; nobody walks them.)
            static const char* kWalkClip[8] = { nullptr, "walk_brisk", "walk_relaxed", "walk_pockets", "walk_phone", "walk_relaxed", nullptr, nullptr };
            auto have = [&](const char* nm) { return nm && rtm0.find_clip(nm) >= 0; };
            if (v.pose == npc::Pose::Sit) {
                loop_nm = v.style == 1 ? kBar[turn(3, 17.0)] : v.style == 2 ? kEat[turn(3, 15.0)] : v.style == 3 ? kTalk[turn(3, 16.0)] : nullptr;
                if (!have(loop_nm)) loop_nm = v.style == 1 ? "bar_sit" : v.style == 2 ? "table_eat" : v.style == 3 ? "table_talk" : nullptr;      // (a set cooked before these were made)
                loop_seat = loop_nm != nullptr;
            } else if (v.pose == npc::Pose::Work) {
                loop_nm = v.style == 1 ? kTill[turn(3, 12.0)] : (v.style == 2 || v.car != 255) ? "fuel" : "till";
                if (!have(loop_nm) && v.style == 1) loop_nm = "cashier";
            } else if (v.pose == npc::Pose::Stand) {
                // (5 points at something, 6 talks in a group, 7 looks at something - the pavilion's and
                //  the pad's people: the arm, the hands and the head are the life layers', the body
                //  stands some way of its own or plainly)
                static const char* kTalkStand[4] = { nullptr, "look_hips", nullptr, "look_crossed" };
                static const char* kWatchStand[4] = { "look_far", "look_hips", nullptr, "look_crossed" };
                loop_nm = v.style == 5 ? nullptr : v.style == 6 ? kTalkStand[v.id & 3] : v.style == 7 ? kWatchStand[v.id & 3] : kLook[v.style % 5][v.id & 1];
                if (!have(loop_nm)) loop_nm = v.style >= 5 ? nullptr : kLook[v.style % 5][0];
            } else if (v.pose == npc::Pose::Walk && std::abs(v.speed) > 0.5f * 1.35f * npc::kPace[v.look % 8]) {
                // (a way of walking of its own once it is under way: its first and last steps are
                //  the plain walk's, which has starts and stops - a walking clip slowed to a
                //  third of its pace is a body wading)
                loop_nm = kWalkClip[v.look % 8]; loop_walk = loop_nm != nullptr;
            }
            if (kShow && v.pose != npc::Pose::Walk && v.pose != npc::Pose::Run && !loop_seat) loop_nm = kShow;
            if (loop_nm && rtm0.find_clip(loop_nm) < 0) { loop_nm = nullptr; loop_seat = loop_walk = false; }
            // COMBINED. On top of that clip, part of the time, another one over part of the body
            // (a second turn with a period of its own, so the two do not change together):
            // talking hands on somebody who leans back, a head that talks to the neighbour at
            // the bar, a head that looks about on a body that walks or stands, and on a plain
            // seat an upper body with a phone or one that sits at ease.
            auto is = [&](const char* nm) { return loop_nm && std::string(loop_nm) == nm; };
            const int turn2 = turn(3, 9.5);
            if (v.pose == npc::Pose::Sit) {
                fidgets = 1.0f;
                if (at_desk) { fidgets = desk_do == 2 ? 1.0f : 0.3f; free_hands = desk_do == 2 ? 3 : desk_do == 3 ? 0 : 2; }
                else if (!loop_nm) {
                    if (v.id % 3 == 0) { layer_nm = "look_phone"; free_hands = 0; }
                    else if (v.id % 3 == 1) layer_nm = "sit_idle";
                } else if (is("table_idle") && turn2 == 0) { layer_nm = "table_talk"; layer_mask = anim::LayerMask::Arms; }
                else if ((is("bar_idle") || is("bar_sit")) && turn2 == 0) { layer_nm = "bar_talk"; layer_mask = anim::LayerMask::Head; }
                // (the hand with the fork or the glass is the right one)
                if (is("table_eat") || is("table_drink") || is("bar_sit")) free_hands = 1;
            } else if (v.pose == npc::Pose::Work) {
                fidgets = is("cashier_idle") ? 0.8f : 0.25f;
                free_hands = is("cashier_idle") ? 3 : 1;
            } else if (v.pose == npc::Pose::Stand) {
                fidgets = 1.0f;
                if (is("look_phone")) free_hands = 0;
                else if (is("look_far")) free_hands = 1;
                if (v.style < 5 && (!loop_nm || is("look_hips") || is("look_far")) && turn2 == 0) { layer_nm = have("look_around") ? "look_around" : "look_crossed"; layer_mask = anim::LayerMask::Head; }
                if (v.style == 6) fidgets = 0.5f;
                if (v.style == 5 || v.style == 7) fidgets = 0.3f;
            } else if (v.pose == npc::Pose::Walk && std::abs(v.speed) > 0.3f) {
                // (walking, the head is the walk's own, and glances about by the look layer. Until
                //  2026-11-26 a third of the time the head of "look_around" was laid over the walk: a
                //  head turning on a clock of its own over a body that walks on.)
                if (is("walk_phone")) free_hands = 1;
            }
            if (!have(layer_nm)) layer_nm = nullptr;
        }
        const glm::vec3 fwd(std::sin(v.yaw), 0.0f, -std::cos(v.yaw));
        net::PlayerState s{};
        s.px = v.pos.x; s.py = v.pos.y + game::Player::kHalfExtents.y; s.pz = v.pos.z;
        // (on a seat with a clip: the model stands on the floor in front of it, the clip sits)
        if (loop_seat) s.py -= float(npc::kSeatHeightCm[v.style == 1 ? 1 : 0]) * 0.01f;
        s.vx = fwd.x * v.speed; s.vz = fwd.z * v.speed;
        s.yaw = v.yaw; s.flags = net::kFlagGround;
        s.move = (v.pose == npc::Pose::Walk || run) ? net::kMoveFwd : 0;
        if (run) s.move |= net::kMoveSprint;
        anim::CharacterAnimationInput in;
        net::RemoteFrame rf = net::remote_animation_input(s, in);
        a.unarmed = !run;                               // (the soldiers on the run carry their rifles)
        if (!run) in.held = anim::HeldItem::None;       // (nothing in the hands)
        in.aiming = false;
        if (run) in.carry = 1.0f;                       // (and carry them low across the body, not at the ready: anim WeaponDef carry_*)
        in.desired_velocity = in.velocity;              // (a stroll, not a soldier's pace)
        // Upright. The soldier's walk is a patrol walk: hunched, the head down over a rifle. Somebody
        // strolling to a shop door carries the head up - the spine, neck and head are raised by what
        // the aim's pitch raises them (upper_body), more the faster they walk.
        // (0.42 at a walk until 2026-11-26: that bent the back 7 degrees BACKWARD and tipped the
        //  head back 12 - somebody leaning away from where he goes, the chin in the air. Standing
        //  it was 0.15, which is what levels the head; a walk now gets 0.26.)
        static const float kUp = std::getenv("VIBE_NPC_UPRIGHT") ? float(std::atof(std::getenv("VIBE_NPC_UPRIGHT"))) : 0.26f;
        in.aim_pitch = run ? 0.12f : kUp * (0.56f + 0.44f * glm::clamp(std::abs(v.speed) / 1.35f, 0.0f, 1.0f));
        a.seated = false;
        // On a seat: the hips on it, the animator's seat pose, the hands where they belong.
        auto sit = [&](const glm::mat4& hip, glm::vec3 hand_l, glm::vec3 hand_r, float legs) {
            rf.model = hip * glm::translate(glm::mat4(1.0f), -rest);
            const glm::mat4 inv = glm::inverse(rf.model);
            in.seat_hand_l = glm::vec3(inv * glm::vec4(hand_l, 1.0f));
            in.seat_hand_r = glm::vec3(inv * glm::vec4(hand_r, 1.0f));
            in.seat_legs = legs;
            in.seated = true;
            in.velocity = in.desired_velocity = glm::vec3(0.0f);
            rf.eye = glm::vec3(hip * glm::vec4(0.0f, 0.62f, 0.0f, 1.0f));
            rf.view_inv = hip; rf.view_inv[3] = glm::vec4(rf.eye, 1.0f);
            a.seated = true;
        };
        if (near_[k].car >= 0) {
            // In a car: at its wheel (the hands on it, going round with it) or beside the driver.
            glm::mat4 hip; glm::vec3 hl, hr; float legs = 0.0f;
            anim::CharacterAnimationInput::SeatExtras more;
            if (!npc_car_seat(near_[k].car, near_[k].seat, hip, hl, hr, legs, &more)) { a.present = false; a.id = 0; continue; }      // (that car is not shown)
            sit(hip, hl, hr, legs);
            in.take_seat(more, glm::inverse(rf.model));      // (the wheel's frame, the feet's places, the backrest: this car's own)
            if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg && frame_number_ % 300 == 0)
                log::infof("[npc]   in car %d seat %d: body %u (last frame: level %d, palette %d, drawn %d, mesh %d) at %.1f %.1f %.1f, %.0f m from the eye", near_[k].car, near_[k].seat, a.id, a.lod, a.palette_prev, int(a.drawn), a.mesh_lod, rf.eye.x, rf.eye.y, rf.eye.z, double(near_[k].d));
        } else if (at_desk) {
            // AT A DESK: the seat pose (the body fitted to this chair and this desk by rule, not a
            // clip - no clip's keyboard is where this one is), the hips on the front half of the
            // chair, the wrists where the work is. `v.aim` is the middle of the keyboard's top
            // (the NPC engine has it from the pavilion's file); the mouse lies 30 cm to its left,
            // the screen stands a hand behind it. What the hands do follows desk_do; they go from
            // one place to the next at a hand's pace (eased here: the seat pose puts the wrists
            // exactly where it is told).
            const glm::vec3 right(std::cos(v.yaw), 0.0f, std::sin(v.yaw)), up(0.0f, 1.0f, 0.0f);
            const glm::vec3 K = v.aim;
            // (as far forward on the cushion as it takes to have the keyboard 47 cm before the
            //  hips - elbows bent, not arms held out straight: the chairs stand 55 - 66 cm from it)
            const float sit_fwd = glm::clamp(glm::dot(K - v.pos, fwd) - 0.47f, 0.05f, 0.18f);
            const glm::mat4 hip(glm::vec4(right, 0.0f), glm::vec4(up, 0.0f), glm::vec4(-fwd, 0.0f), glm::vec4(v.pos + fwd * sit_fwd + up * 0.10f, 1.0f));
            const float t = float(life_now) + float(a.id) * 1.37f;
            glm::vec3 hl, hr; float back = 1.5f, typing = 0.0f;
            if (desk_do == 0) {
                // typing: both wrists before the keys, each wandering a little along them
                hl = K - fwd * 0.105f - right * (0.105f + 0.022f * std::sin(t * 1.3f)) + up * (0.030f + 0.004f * std::sin(t * 7.0f));
                hr = K - fwd * 0.105f + right * (0.105f + 0.022f * std::sin(t * 1.7f + 1.0f)) + up * (0.030f + 0.004f * std::sin(t * 6.1f + 2.0f));
                typing = 1.0f;
            } else if (desk_do == 2) {
                // leaning back: the back into the chair's, the forearms on its armrests
                hl = v.pos - right * 0.265f + fwd * 0.13f + up * 0.215f;
                hr = v.pos + right * 0.265f + fwd * 0.13f + up * 0.215f;
                back = 21.0f;
            } else {
                // reading, or turned to the neighbour: a hand on the mouse (it moves it now and
                // then), the other at rest on the desk's edge
                hl = K - right * (0.30f + 0.020f * std::sin(t * 0.9f)) - fwd * (0.075f + 0.018f * std::sin(t * 1.4f + 0.7f)) + up * 0.032f;
                hr = K + right * 0.17f - fwd * 0.20f + up * 0.022f;
                back = desk_do == 3 ? 9.0f : 6.0f;
            }
            static glm::vec3 desk_l[kNpcBodies], desk_r[kNpcBodies];
            static float desk_back[kNpcBodies], desk_type[kNpcBodies];
            static uint32_t desk_of[kNpcBodies] = {};
            const size_t di = size_t(body_of[k]);
            if (a.fresh_pose || desk_of[di] != a.id) { desk_of[di] = a.id; desk_l[di] = hl; desk_r[di] = hr; desk_back[di] = back; desk_type[di] = typing; }
            const float e = 1.0f - std::exp(-dt / 0.22f);
            desk_l[di] += (hl - desk_l[di]) * e; desk_r[di] += (hr - desk_r[di]) * e;
            desk_back[di] += (back - desk_back[di]) * (1.0f - std::exp(-dt / 0.45f));
            // (the fingers stop while a hand is on its way, and begin when it is there)
            desk_type[di] += ((glm::distance(desk_l[di], hl) < 0.03f ? typing : 0.0f) - desk_type[di]) * e;
            sit(hip, desk_l[di], desk_r[di], 0.0f);
            in.seat_desk = true;
            in.seat_typing = desk_type[di];
            in.seat_back = desk_back[di];
            if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg && frame_number_ % 300 == 0)
                log::infof("[npc]   at a desk: person %d %s, the keyboard %.2f m in front of the hips, %.2f m to the side", int(v.id),
                           desk_do == 0 ? "types" : desk_do == 1 ? "reads" : desk_do == 2 ? "leans back" : "talks to the neighbour",
                           double(glm::dot(K - glm::vec3(hip[3]), fwd)), double(glm::dot(K - glm::vec3(hip[3]), right)));
        } else if (v.pose == npc::Pose::Sit && !loop_seat) {
            // (a hand above the cushion's top; the hands on the knees)
            const glm::vec3 right(std::cos(v.yaw), 0.0f, std::sin(v.yaw));
            const glm::mat4 hip(glm::vec4(right, 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f), glm::vec4(-fwd, 0.0f), glm::vec4(v.pos + glm::vec3(0.0f, 0.10f, 0.0f), 1.0f));
            sit(hip, glm::vec3(hip * glm::vec4(-0.17f, 0.14f, -0.36f, 1.0f)), glm::vec3(hip * glm::vec4(0.17f, 0.14f, -0.36f, 1.0f)), 0.0f);
        }
        // Something it is in the middle of doing: a clip, played over the time the NPC engine
        // gives the act (npc.h kActTime), the hips carried to the seat this chair or car
        // really has (motion warping: character_animator.h action_*).
        static const char* kActClip[5] = { "", "sit_down", "stand_up", "car_enter", "car_exit" };
        const anim::IAnimationRuntime& rtm = *anim_set_.runtime;
        if (near_[k].car < 0 && v.act != npc::Act::None && v.act <= npc::Act::CarExit) {      // (PutDown / PickUp: no clip - the hands, below)
            const anim::ClipId clip = rtm.find_clip(kActClip[int(v.act)]);
            if (clip >= 0) {
                in.action_clip = clip;
                in.action_time = glm::clamp(v.act_t / npc::kActTime[int(v.act)], 0.0f, 1.0f) * rtm.clip(clip).duration;
                glm::vec3 hips_at = v.aim + glm::vec3(0.0f, 0.10f, 0.0f);
                bool have = true;
                if (v.act == npc::Act::CarEnter || v.act == npc::Act::CarExit) {
                    glm::mat4 hip; glm::vec3 hl, hr; float legs = 0.0f;
                    have = v.car != 255 && npc_car_seat(int(v.car), int(v.seat), hip, hl, hr, legs);
                    if (have) hips_at = glm::vec3(hip[3]);
                }
                in.action_warp = have;
                in.action_hips = glm::vec3(glm::inverse(rf.model) * glm::vec4(hips_at, 1.0f));
                in.action_warp_from_start = v.act == npc::Act::StandUp || v.act == npc::Act::CarExit;
                in.action_end_at_origin = in.action_warp_from_start;
                in.has_look_at = false;
            }
        } else if (near_[k].car < 0 && !a.seated && loop_nm) {
            // A loop (see WHAT IT PLAYS above). A walk is played at the pace the body really
            // moves (its own travel left out: the NPC engine moves it), a seat's clip with the
            // hips on this seat.
            const anim::ClipId clip = rtm.find_clip(loop_nm);
            const float dur = std::max(rtm.clip(clip).duration, 0.1f);
            float rate = 1.0f;
            if (loop_walk) {
                const anim::Transform tr = a.animator->action_root(clip, dur);
                const float cs = glm::length(glm::vec2(tr.t.x, tr.t.z)) / dur;
                rate = cs > 0.2f ? glm::clamp(std::abs(v.speed) / cs, 0.4f, 1.8f) : 1.0f;
            }
            // (this person's own place in the loop and own pace - a walk's pace is its feet's -
            //  and, where the clip has no side that matters, left for right)
            if (!loop_walk) rate *= who.rate;
            if (a.fresh_pose) a.npc_phase = who.phase * dur;
            a.npc_phase = std::fmod(a.npc_phase + dt * rate, dur);
            in.action_clip = clip;
            in.action_time = a.npc_phase;
            {
                const std::string nm(loop_nm);
                const bool sided = nm.rfind("cashier", 0) == 0 || nm == "fuel" || nm == "till";      // (the till, the pump are on one side)
                in.action_mirror = who.mirror && !sided && !kNoLife;
            }
            in.action_in_place = loop_walk;
            if (loop_seat) {
                in.action_warp = true; in.action_warp_full = true;
                in.action_hips = glm::vec3(glm::inverse(rf.model) * glm::vec4(v.pos + glm::vec3(0.0f, 0.10f, 0.0f), 1.0f));
                in.velocity = in.desired_velocity = glm::vec3(0.0f);
            }
        }
        // WHAT THEY CARRY (carry::, at the top of this file): the item in the hands, how it is held,
        // what the hand does with it - and, setting it down or taking it up (the acts), the hands to
        // the place and the body bent down to a low one. From the chest as it was last frame.
        {
            carry::Shown& sh = g_carry_shown[size_t(body_of[k])];
            if (sh.id != a.id || a.fresh_pose) { sh = carry::Shown{}; sh.id = a.id; }
            sh.item = 0;
            const int sp2 = art.rig()[anim::Role::Spine2];
            const glm::vec3 C = sh.chest_ok ? sh.chest : (sp2 >= 0 ? art.skeleton().rest_pos[size_t(sp2)] : glm::vec3(0.0f, 1.25f, 0.0f));
            uint8_t item = near_[k].car >= 0 ? uint8_t(0) : uint8_t(v.item & 15);
            // (the ways of walking and standing that are at a phone: the phone)
            const bool phone_clip = v.act == npc::Act::None && ((v.pose == npc::Pose::Walk && v.look % 8 == 4) || (v.pose == npc::Pose::Stand && v.style == 3));
            if (!item && near_[k].car < 0 && phone_clip) item = uint8_t(npc::Item::Phone);
            if (run || a.seated || v.pose == npc::Pose::Sit) item = 0;
            const bool act_item = near_[k].car < 0 && (v.act == npc::Act::PutDown || v.act == npc::Act::PickUp);
            if (act_item && !item) item = uint8_t(v.item >> 4);              // (taking up: what lies there)
            static const bool kNoCarry = std::getenv("VIBE_NPC_NO_CARRY") != nullptr;
            if (kNoCarry) item = 0;
            if (item) {
                const float tl = float(life_now) + float(a.id) * 2.31f;
                float drink = 0.0f, look = 0.0f;
                const bool walking = v.pose == npc::Pose::Walk && std::abs(v.speed) > 0.3f;
                if (item == uint8_t(npc::Item::Cup) && !act_item) {
                    // a sip now and then: every 9 - 15 s, for 2.6 s (not while walking fast)
                    const float per = 9.0f + float(a.id % 7u), ph = std::fmod(tl, per);
                    if (ph < 2.6f && std::abs(v.speed) < 1.0f) drink = glm::clamp(std::sin(ph / 2.6f * 3.14159f) * 1.7f, 0.0f, 1.0f);
                }
                if ((item == uint8_t(npc::Item::Clipboard) || item == uint8_t(npc::Item::Tablet) || item == uint8_t(npc::Item::Phone)) && !act_item) {
                    // read for a while, lowered for a while; a glance at it while walking
                    const float per = 11.0f + float(a.id % 5u), ph = std::fmod(tl, per);
                    look = walking ? (ph < 3.0f ? 1.0f : 0.0f) : (ph < per * 0.62f ? 1.0f : 0.0f);
                    if (phone_clip) look = 1.0f;
                    // (eased: the board comes up over half a second)
                    const float e0 = glm::clamp(ph / 0.5f, 0.0f, 1.0f), e1 = glm::clamp(((walking ? 3.0f : per * 0.62f) - ph) / 0.5f, 0.0f, 1.0f);
                    if (look > 0.5f && !phone_clip) look = std::min(e0, e1);
                }
                carry::Hold h = carry::carried(item, C, drink, look);
                auto move = [](carry::Hold& x, const glm::mat4& D) {
                    x.M = D * x.M;
                    for (int s = 0; s < 2; ++s) {
                        x.grip[s] = glm::vec3(D * glm::vec4(x.grip[s], 1.0f));
                        x.fingers[s] = glm::mat3(D) * x.fingers[s];
                        x.palm[s] = glm::mat3(D) * x.palm[s];
                    }
                };
                if (act_item) {
                    // the hand reaches the place at half the act: setting down, the item is there
                    // from then on; taking up, it was there until then
                    const float u = glm::clamp(v.act_t / npc::kActTime[int(v.act)], 0.0f, 1.0f);
                    float reach = u < 0.5f ? u / 0.5f : (1.0f - u) / 0.5f;
                    reach = reach * reach * (3.0f - 2.0f * reach);
                    const glm::vec3 place = glm::vec3(glm::inverse(rf.model) * glm::vec4(v.aim, 1.0f));
                    const glm::mat4 Md = carry::set_down(item, place);
                    const bool lies = v.act == npc::Act::PutDown ? u >= 0.5f : u < 0.5f;
                    if (lies) { move(h, Md * glm::inverse(h.M)); if (reach < 0.3f) h.hands = 0; }
                    else {
                        glm::mat4 Mi = h.M;
                        Mi[3] = glm::mix(h.M[3], Md[3], reach);
                        for (int c = 0; c < 3; ++c) Mi[c] = glm::vec4(glm::normalize(glm::mix(glm::vec3(h.M[c]), glm::vec3(Md[c]), reach)), 0.0f);
                        move(h, Mi * glm::inverse(h.M));
                    }
                    // (a low place - the floor, a car's boot - is reached by bending down)
                    in.crouch = std::max(in.crouch, glm::clamp((1.0f - place.y) / 0.8f, 0.0f, 1.0f) * reach);
                    in.has_look_at = true; in.look_at = place; in.look_weight = 4.0f; in.look_id = 9;
                }
                in.hold_hands = h.hands;
                for (int s = 0; s < 2; ++s) { in.hold_grip[s] = h.grip[s]; in.hold_fingers[s] = h.fingers[s]; in.hold_palm[s] = h.palm[s]; in.hold_curl[s] = h.curl[s]; }
                in.fidget_arms &= uint8_t(~h.hands);
                if (h.read && !act_item) { in.has_look_at = true; in.look_at = glm::vec3(h.M[3]); in.look_weight = 3.0f; in.look_id = 8; }
                sh.item = item; sh.M = h.M;
            }
        }
        anim::LodInput li;
        li.distance = glm::length(rf.eye - cam);
        li.screen_height = float(swapchain_extent_.height);
        li.tan_half_fov_y = tan_half;
        {
            const glm::vec4 c = current_frame_view_.vp * glm::vec4(rf.eye - glm::vec3(0.0f, 0.7f, 0.0f), 1.0f);
            const float m = 1.25f * c.w + 1.5f;
            li.in_view = c.w > -1.5f && std::abs(c.x) < m && std::abs(c.y) < m;
        }
        li.combat = false;
        li.previous = a.lod;
        a.lod = std::max(anim::select_lod(li), near_[k].d < 12.0f ? 0 : 1);      // (full detail only close by: there are many of them)
        in.dt = dt;
        in.variation = a.id * 7919u;
        // (VIBE_LOG_NPC: what each body that is shown plays - a clip by its name, or which of the animator's own poses)
        if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg && frame_number_ % 300 == 0) {
            const char* plays = in.seated ? (near_[k].car >= 0 ? (near_[k].seat == 0 ? "the driver's pose" : "the passenger's pose") : "the seat pose")
                              : in.action_clip >= 0 ? rtm.clip(in.action_clip).name.c_str()
                              : run ? "the motion-matched run" : v.pose == npc::Pose::Walk ? "the motion-matched walk" : "the motion-matched stand";
            log::infof("[npc]   plays: %s %d (pose %d style %d look %d, %.0f m off) - %s at %.2f s%s%s%s; gesture %d (%.2f), head %.2f / %.2f rad (mode %d)", near_[k].car >= 0 ? "in car" : "person", near_[k].car >= 0 ? near_[k].car : int(v.id),
                       int(v.pose), int(v.style), int(v.look), double(near_[k].d), plays, in.action_clip >= 0 ? double(in.action_time) : 0.0,
                       in.action_mirror ? ", mirrored" : "", layer_nm ? ", over part of the body: " : "", layer_nm ? layer_nm : "",
                       int(a.animator->gesture()), double(a.animator->gesture_weight()), double(a.animator->gaze_yaw()), double(a.animator->gaze_pitch()), a.animator->gaze_mode());
        }
        // THE LIFE LAYERS (anim/life_layers.cpp, docs/animation/actions.md "Life layers"): this
        // person's posture, breathing, fidgets, the second clip chosen above - and the look.
        // WHAT IT LOOKS AT, the first that is there: the player close by; a car that drives
        // past (those outside); whoever is talked to (a table's talkers, the bar, the cashier's
        // customer); the plate or the glass; the player a little further off. How much of the
        // time is the weight - the head still glances away, and rests.
        const bool in_car = near_[k].car >= 0;
        const bool stays = v.act == npc::Act::None && !(in_car && near_[k].seat == 0);
        // (the wave's count: per body, cleared when somebody new has it, and told to the animator
        //  EVERY frame - a frame without it read as a new wave when it came back)
        static bool seen[kNpcBodies] = {};
        static uint32_t waves[kNpcBodies] = {};
        const size_t bi = size_t(body_of[k]);
        if (a.fresh_pose) { seen[bi] = true; waves[bi] = 0; }      // (not at the moment the body appears)
        in.gesture = anim::Gesture::Wave;
        in.gesture_seq = waves[bi];
        if (!kNoLife && !run) {
            in.individuality = 1.0f;
            in.life_on_clip = stays;
            in.fidgets = (stays && !in_car) ? fidgets : 0.0f;
            in.fidget_arms = in.action_mirror ? uint8_t(((free_hands & 1) << 1) | ((free_hands & 2) >> 1)) : free_hands;
            if (layer_nm && stays && !in_car) {
                in.layer_clip = rtm0.find_clip(layer_nm);
                in.layer_mask = layer_mask;
                in.layer_rate = who.rate;
            }
        }
        if (!kNoLife && stays && !run) {
            const bool sits = v.pose == npc::Pose::Sit || in_car;
            const bool outside = !sits && v.pose != npc::Pose::Work;
            const glm::vec3 head = in_car ? rf.eye : v.pos + glm::vec3(0.0f, sits ? 0.62f : 1.62f, 0.0f);
            glm::vec3 at(0.0f); float wt = 0.0f; uint32_t what = 0;
            const float dp = glm::distance(head, cam);
            if (dp < (sits ? 3.5f : 4.5f)) { at = cam; wt = 1.6f; what = 1; }
            // The pavilion's and the pad's people. At a desk: the screen (a hand behind the
            // keyboard, 30 cm over it) most of the time, or the neighbour talked to, all of the
            // time. Standing: what is pointed at, mostly; what is looked at - the rocket, the
            // hall below - all of the time (and the trunk goes back with a look that goes up);
            // whoever stands nearest in the group that talks.
            const bool stands_here = v.pose == npc::Pose::Stand && !in_car;
            if (wt == 0.0f && at_desk) {
                if (desk_do == 3 && desk_peer >= 0 && size_t(desk_peer) < npc_views_.size()) { at = npc_views_[size_t(desk_peer)].pos + glm::vec3(0.0f, 0.62f, 0.0f); wt = 4.0f; what = 200u + uint32_t(desk_peer); }
                else if (desk_do != 2) { at = v.aim + fwd * 0.33f + glm::vec3(0.0f, 0.30f, 0.0f); wt = 3.2f; what = 3; }
            }
            if (wt == 0.0f && stands_here && v.style == 5) { at = v.aim; wt = 3.0f; what = 4; }
            if (wt == 0.0f && stands_here && v.style == 7) {
                at = v.aim; wt = 4.0f; what = 5;
                const glm::vec3 d = at - head;
                in.aim_pitch += 0.45f * glm::clamp(std::atan2(d.y, std::max(glm::length(glm::vec2(d.x, d.z)), 0.5f)), -0.3f, 0.9f);
            }
            if (wt == 0.0f && stands_here && v.style == 6) {
                float bd = 2.8f;
                for (const npc::View& o : npc_views_) {
                    if (o.kind != npc::Kind::Person || o.hidden || o.id == v.id) continue;
                    const float d = glm::distance(o.pos, v.pos);
                    if (d < bd) { bd = d; at = o.pos + glm::vec3(0.0f, o.pose == npc::Pose::Sit ? 0.62f : 1.62f, 0.0f); wt = 2.6f; what = 200u + o.id; }
                }
            }
            if (wt == 0.0f && (outside || v.style == 2) && !in_car) {
                // (a car going by: the nearest that moves)
                float bd = 28.0f;
                for (const npc::View& c : npc_views_) {
                    if (c.kind != npc::Kind::Car || c.pose != npc::Pose::Drive || std::abs(c.speed) < 2.0f) continue;
                    const float d = glm::distance(c.pos, v.pos);
                    if (d < bd) { bd = d; at = c.pos + glm::vec3(0.0f, 1.0f, 0.0f); wt = v.pose == npc::Pose::Walk ? 0.8f : 1.3f; what = 100u + c.id; }
                }
                if (v.pose == npc::Pose::Work) wt *= 0.4f;         // (somebody at a pump has the nozzle to mind)
            }
            const bool talks = (v.pose == npc::Pose::Sit && (v.style == 3 || v.style == 1)) || (v.pose == npc::Pose::Work && v.style == 1);
            if (wt == 0.0f && talks && !in_car) {
                // (whoever is nearest: across the table, on the next stool, in front of the counter)
                float bd = v.pose == npc::Pose::Work ? 3.5f : v.style == 1 ? 1.9f : 2.6f;
                for (const npc::View& o : npc_views_) {
                    if (o.kind != npc::Kind::Person || o.hidden || o.id == v.id) continue;
                    const float d = glm::distance(o.pos, v.pos);
                    if (d < bd) { bd = d; at = o.pos + glm::vec3(0.0f, o.pose == npc::Pose::Sit ? 0.62f : 1.62f, 0.0f); wt = v.style == 1 && v.pose == npc::Pose::Sit ? 1.4f : 2.5f; what = 200u + o.id; }
                }
            }
            if (wt == 0.0f && v.pose == npc::Pose::Sit && (v.style == 1 || v.style == 2) && !in_car) {
                // (the plate on the table, the glass on the counter: in front of the seat, 30 cm over it)
                at = v.pos + fwd * 0.42f + glm::vec3(0.0f, 0.30f, 0.0f); wt = v.style == 2 ? 1.6f : 0.8f; what = 2;
            }
            if (wt == 0.0f && dp < 9.0f) { at = cam; wt = 1.0f; what = 1; }
            if (wt > 0.0f) {
                in.has_look_at = true;
                in.look_at = glm::vec3(glm::inverse(rf.model) * glm::vec4(at, 1.0f));
                in.look_weight = wt;
                in.look_id = what;
            }
            // A WAVE: somebody on foot who sees the player come close lifts a hand - once, and
            // not everybody (the more curious half). `seen`: was the player near last frame.
            const glm::vec3 to_cam = cam - head;
            const bool near_now = outside && dp < 6.0f && glm::dot(glm::vec2(to_cam.x, to_cam.z), glm::vec2(fwd.x, fwd.z)) > 0.35f * dp;
            if (near_now && !seen[bi] && who.curious > 1.0f && free_hands == 3) ++waves[bi];
            if (near_now) seen[bi] = true; else if (dp > 9.0f) seen[bi] = false;
            // (VIBE_NPC_GESTURE=<n>: everybody does gesture n - 1 a hand to the head, 2 to the neck,
            //  3 a look at the wrist, 4 a shift of weight, 5 a wave - every five seconds: to see them)
            // An arm that points (style 5: every six seconds or so, at what `aim` is), a hand
            // that talks (style 6, and at a desk turned to the neighbour: each at moments of his
            // own, so that in a group now one talks and now another).
            if (stands_here && v.style == 5) {
                in.gesture = anim::Gesture::Point; in.gesture_seq = 1000u + uint32_t((life_now + double(a.id) * 1.7) / 6.5);
                in.gesture_at = glm::vec3(glm::inverse(rf.model) * glm::vec4(v.aim, 1.0f));
            } else if ((stands_here && v.style == 6) || (at_desk && desk_do == 3)) {
                in.gesture = anim::Gesture::Talk; in.gesture_seq = 2000u + uint32_t((life_now + double(a.id) * 2.9) / (7.0 + double(a.id % 4u)));
                if (at_desk) in.fidget_arms = 2;                 // (the right hand: the left is on the mouse)
            }
            if (kGesture > 0) { in.gesture = anim::Gesture(kGesture); in.gesture_seq = uint32_t(life_now / 5.0 + double(a.id) * 0.37); }
        } else if (near_[k].d < 7.0f && !a.seated && !run && !loop_seat && v.act == npc::Act::None) {
            // (the soldiers on their run, and with VIBE_NPC_NO_LIFE everybody: somebody near looks at the player)
            in.has_look_at = true; in.look_at = glm::vec3(glm::inverse(rf.model) * glm::vec4(cam, 1.0f));
        }
        in.teleported = a.fresh_pose && !a.seated;
        in.lod = a.lod;
        const bool skip = a.lod == 2 && !a.fresh_pose && !a.die && a.skipped == 0 && (frame_number_ & 1u) && !a.pose_rows.empty();
        if (skip) { a.skipped = 1; a.skip_dt = dt; }
        else { in.dt = dt + a.skip_dt; a.skipped = 0; a.skip_dt = 0.0f; if (a.lod >= 3) a.pose_rows.clear(); }
        posed.push_back({ &a, in, rf, skip });
    }
    // ---- posed on the job threads, eight at a time (pose_update_all's probes are that many)
    for (size_t i0 = 0; i0 < posed.size(); i0 += 8) {
        PoseJob jobs[8];
        int nj = 0;
        for (size_t i = i0; i < std::min(posed.size(), i0 + 8); ++i) if (!posed[i].skip) jobs[nj++] = { posed[i].a->animator.get(), &posed[i].in };
        if (nj) pose_update_all(jobs, nj);
    }
    for (Posed& r : posed) {
        MpAvatar& a = *r.a;
        if (r.skip) a.palette = body_push_rows(a.pose_rows);
        else {
            pose_finish(*a.animator, r.in, a.lod >= 3 ? nullptr : &a.palette, &a.pose_rows);
            {   // (where the chest is: what is carried is held from it next frame)
                carry::Shown& sh = g_carry_shown[size_t(&a - npc_bodies_.data())];
                const int sp2 = anim_set_.runtime->rig()[anim::Role::Spine2];
                if (sp2 >= 0 && a.lod < 3 && size_t(sp2) < a.animator->model().size()) { sh.chest = glm::vec3(a.animator->model()[size_t(sp2)][3]); sh.chest_ok = true; }
            }
            // (where the rifle is drawn: the animator's frame for it - the soldiers on the run had
            //  theirs drawn at the view, level at the eye, while the hands were elsewhere)
            a.weapon_offset = a.animator->weapon_offset();
            a.fire_offset = a.animator->weapon_fire_offset();
        }
        a.prev_model = a.fresh_pose ? r.rf.model : a.model;
        a.fresh_pose = false;
        a.model = r.rf.model;
        if (a.die) {
            // Shot (npc_bolt_hits): this pose goes to the physics as a body that falls, pushed by
            // the shot; the NPC engine has that person leave its world (the authority's; a copy
            // of it keeps the person out of sight until it has followed: npc_gone_).
            a.die = false;
            if (a.lod >= 3) { r.in.lod = 2; r.in.dt = 0.0f; pose_character(*a.animator, r.in, nullptr); }
            const glm::mat4 vi(glm::vec4(std::cos(a.dying_st.yaw), 0.0f, std::sin(a.dying_st.yaw), 0.0f), glm::vec4(0.0f, 1.0f, 0.0f, 0.0f),
                               glm::vec4(-std::sin(a.dying_st.yaw), 0.0f, std::cos(a.dying_st.yaw), 0.0f), glm::vec4(r.rf.eye, 1.0f));
            if (npc_ && a.id >= 1 && a.id <= uint32_t(npc::kMaxAgents)) { if (npc_->replica()) npc_gone_[a.id - 1] = 3.0f; else npc_->kill(int(a.id) - 1); }
            // (the driver of a car that has nobody of the NPC world at its wheel: the car's to lose)
            if (npc_ && a.id >= 1000u && a.id < 1100u) { if (npc_->replica()) npc_nodriver_[a.id - 1000u] = 3.0f; else npc_->kill(int(a.id - 1000u)); }
            // On its feet: the reaction first (npc_dying_pose). On a seat, or in the middle of
            // sitting down or getting into a car (the hips carried by a warp): the physics' at once.
            const glm::vec3 fwd(std::sin(a.dying_st.yaw), 0.0f, -std::cos(a.dying_st.yaw));
            const glm::vec3 right(-fwd.z, 0.0f, fwd.x);
            const glm::vec3 origin(r.rf.model[3]);
            const float lateral = glm::dot(a.hit_pos - origin, right);        // + the body's right side
            const ShotPick pick = pick_shot_clip(a.hit_pos.y - origin.y, lateral, glm::dot(a.hit_dir, fwd) < 0.0f, glm::length(r.in.velocity), a.id);
            const ShotClip& sc = *pick.clip;
            const int clip = (r.in.seated || r.in.action_warp || std::getenv("VIBE_NO_DEATH_CLIPS")) ? -1 : anim_set_.runtime->find_clip(sc.name);
            if (clip < 0) {
                // (shot in a car that drives: the body goes on at the car's speed - npc_hit put it in dying_st)
                const glm::vec3 vel = r.in.seated ? glm::vec3(a.dying_st.vx, 0.0f, a.dying_st.vz) : r.in.velocity;
                corpse_spawn(r.rf.model, vi, vel, a.hit_pos, a.hit_dir, std::max(a.hit_power, 0.6f));
                a.id = 0; a.present = false; a.drawn = false; a.palette = -1;
                continue;
            }
            NpcDying d;
            d.in = r.in; d.model = r.rf.model; d.view_inv = vi; d.clip = clip; d.hand = sc.hand;
            d.hit_pos = a.hit_pos; d.hit_dir = a.hit_dir;
            d.model0 = r.rf.model; d.knock = pick.knock;
            {
                // WHICH WAY IT GOES DOWN (pick_shot_clip's fit). The take's own travel to the
                // hand-over (or, if it hardly moves, what its prompt asked for), in the world;
                // played left for right when the shot's side is the other one; then the body is
                // turned, over the first third of a second, by up to 35 degrees so that it goes
                // down the way the shot pushed it.
                const anim::Transform rm = a.animator ? a.animator->action_root(clip, sc.hand) : anim::Transform{};
                glm::vec3 trav(rm.t.x, 0.0f, rm.t.z);
                float yaw = glm::cross(glm::vec3(0.0f, 0.0f, -1.0f), rm.r * glm::vec3(0.0f, 0.0f, -1.0f)).y;
                if (glm::length(trav) < 0.25f) { glm::vec3 t0; float y0; shot_clip_authored(sc.name, t0, y0); trav = t0; if (std::abs(yaw) < 0.2f) yaw = y0; }
                const glm::mat3 M(r.rf.model);
                auto world_of = [&](glm::vec3 m, bool mir) { if (mir) m.x = -m.x; glm::vec3 w = M * m; w.y = 0.0f; return w; };
                glm::vec3 s(a.hit_dir.x, 0.0f, a.hit_dir.z);
                s = glm::length(s) > 1e-3f ? glm::normalize(s) : -fwd;
                glm::vec3 want = s;
                switch (pick.fit) {
                case kFitLegSide: d.mirror = (glm::dot(world_of(trav, false), right) > 0.0f) != (lateral > 0.0f); break;
                case kFitSideHit: d.mirror = (glm::dot(world_of(trav, false), right) > 0.0f) != (lateral > 0.0f); break;
                case kFitSpin: {
                    // (the struck shoulder is driven back: a turn about the up axis of r x s; + = to the left)
                    const float torque = glm::cross(right * lateral, s).y;
                    d.mirror = (yaw > 0.0f) != (torque > 0.0f);
                    break;
                }
                case kFitTowardShooter: want = -s; [[fallthrough]];
                case kFitPush: {
                    const float keep = glm::dot(world_of(trav, false), want), flip = glm::dot(world_of(trav, true), want);
                    d.mirror = std::abs(trav.x) > 0.15f && flip > keep + 0.05f;
                    break;
                }
                default: break;
                }
                if ((pick.fit == kFitPush || pick.fit == kFitTowardShooter) && glm::length(trav) > 0.05f) {
                    const glm::vec3 tw = glm::normalize(world_of(trav, d.mirror));
                    const float ang = std::atan2(glm::cross(tw, want).y, glm::dot(tw, want));     // + = about +y
                    d.turn = glm::clamp(ang, -0.61f, 0.61f);
                }
                log::infof("[npc] person %u goes down: %s%s, struck %.2f m up, %.2f m %s of the middle; turned %.0f deg", a.id - 1, sc.name,
                           d.mirror ? " (left for right)" : "", double(a.hit_pos.y - origin.y), double(std::abs(lateral)), lateral > 0.0f ? "right" : "left", double(glm::degrees(d.turn)));
            }
            a.view_inv = r.rf.view_inv; a.eye = r.rf.eye; a.drawn = a.lod < 3 && a.palette >= 0;
            d.av = std::move(a);
            a = MpAvatar{};
            d.av.id = 5000u + d.av.id;
            if (npc_dying_.size() < 8) npc_dying_.push_back(std::move(d));
            continue;
        }
        a.view_inv = r.rf.view_inv;
        a.eye = r.rf.eye;
        a.mesh_lod = body_mesh_lod(r.rf.eye, a.mesh_lod);
        a.drawn = a.lod < 3 && a.palette >= 0;
    }
}

// A bolt of ours against the station's people (combat.cpp, beside mp_bolt_hits): whoever is on
// foot or on a seat can be shot - by any player. The shooter decides the hit against what is
// drawn on the shooter's screen (as for players), shows it at once (npc_hit) and tells
// everybody (Event::NpcHit); every machine shows the same, and the machine whose NPC world is
// the world takes that person out of it.
bool VulkanEngine::npc_bolt_hits(glm::vec3 a, glm::vec3 b, glm::vec3& hit) {
    if (static const bool lg = std::getenv("VIBE_LOG_NPC") != nullptr; lg) log::infof("[npc] bolt (%.2f %.2f %.2f) -> (%.2f %.2f %.2f), %zu views", a.x, a.y, a.z, b.x, b.y, b.z, npc_views_.size());
    if (!npc_ || npc_views_.empty()) return false;
    const glm::vec3 lo = glm::min(a, b) - glm::vec3(1.0f, 2.2f, 1.0f), hi = glm::max(a, b) + glm::vec3(1.0f, 1.2f, 1.0f);
    int best = -1; float bt = 2.0f;
    for (size_t i = 0; i < npc_views_.size(); ++i) {
        const npc::View& v = npc_views_[i];
        if (v.kind != npc::Kind::Person || v.hidden) continue;
        if (v.pos.x < lo.x || v.pos.x > hi.x || v.pos.z < lo.z || v.pos.z > hi.z || v.pos.y < lo.y || v.pos.y > hi.y) continue;
        // (standing: the feet to the head; on a seat: `pos` is the seat's top - the legs below it, the trunk above)
        const bool sits = v.pose == npc::Pose::Sit;
        const glm::vec3 mn = v.pos + glm::vec3(-0.30f, sits ? -0.45f : 0.0f, -0.30f), mx = v.pos + glm::vec3(0.30f, sits ? 0.95f : 1.82f, 0.30f);
        float t = 0.0f;
        if (seg_aabb(a, b, mn, mx, t) && t < bt) { bt = t; best = int(i); }
    }
    // Whoever sits in a car that is shown - the driver, the one beside him: the body drawn there
    // (its head and trunk). A driver nobody of the NPC world sits for goes by his car's number.
    for (const MpAvatar& av : npc_bodies_) {
        if (!av.present || !av.id) continue;
        const int agent = av.id >= 1000u ? int(av.id - 1000u) : int(av.id) - 1;
        if (agent < 0 || size_t(agent) >= npc_views_.size() || agent >= npc::kMaxAgents) continue;
        if (av.id < 1000u && !npc_views_[size_t(agent)].hidden) continue;      // (on foot: above)
        const glm::vec3 mn = av.eye + glm::vec3(-0.27f, -0.70f, -0.27f), mx = av.eye + glm::vec3(0.27f, 0.22f, 0.27f);
        float t = 0.0f;
        if (seg_aabb(a, b, mn, mx, t) && t < bt) { bt = t; best = agent; }
    }
    if (best < 0) return false;
    hit = a + (b - a) * bt;
    const glm::vec3 dir = glm::normalize(b - a + glm::vec3(1e-6f, 0.0f, 0.0f));
    npc_hit(best, hit, dir);
    if (net_) {
        net::Event e{};
        e.kind = net::Event::NpcHit; e.from = net_->id(); e.to = uint32_t(best);
        e.a[0] = hit.x; e.a[1] = hit.y; e.a[2] = hit.z; e.b[0] = dir.x; e.b[1] = dir.y; e.b[2] = dir.z;
        net_->send_event(e);
    }
    mp_->hitmark = 0.25f;
    return true;
}

// Somebody was shot - by this machine's player or (told by the network) by another's. Blood
// where it hit; the body that is drawn for that person here is told to fall (npc_pose hands it
// to the physics, each machine its own fall from the same pose); and the person leaves the NPC
// world: the authority's kill(), which the copies follow. Until a copy has followed, the person
// is kept from being drawn again beside their own body (npc_gone_).
void VulkanEngine::npc_hit(int agent, glm::vec3 hit, glm::vec3 dir) {
    if (!npc_ || agent < 0 || size_t(agent) >= npc_views_.size() || agent >= npc::kMaxAgents) return;
    const npc::View& v = npc_views_[size_t(agent)];
    // Who: somebody on foot or on a seat; somebody in a car (hidden, with a car); or - `agent`
    // is a car - the driver of a car that drives with nobody of the NPC world at its wheel.
    const bool phantom = v.kind == npc::Kind::Car;
    const npc::View* car = phantom ? &v : (v.hidden && size_t(v.car) < npc_views_.size()) ? &npc_views_[v.car] : nullptr;
    if (phantom ? (v.hidden || v.pose != npc::Pose::Drive || npc_nodriver_[agent] > 0.0f) : (v.hidden && !car)) return;       // (gone already: two players shot the same one)
    spawn_blood(hit, dir);
    const uint32_t id = phantom ? 1000u + uint32_t(v.id) : uint32_t(v.id) + 1u;
    bool body = false;
    for (MpAvatar& av : npc_bodies_)
        if (av.id == id && av.present) {
            av.die = true; av.hit_pos = hit; av.hit_dir = dir; av.hit_power = 1.0f; av.hit_age = 0.0f;
            av.dying_st.yaw = v.yaw;
            av.dying_st.vx = car ? std::sin(car->yaw) * car->speed : 0.0f;       // (in a car: the body goes on as the car went)
            av.dying_st.vz = car ? -std::cos(car->yaw) * car->speed : 0.0f;
            body = true;
            break;
        }
    if (!body) {
        if (npc_->replica()) (phantom ? npc_nodriver_ : npc_gone_)[agent] = 3.0f; else npc_->kill(agent);
    }
    log::infof("[npc] %s %d shot at (%.1f, %.1f, %.1f)%s%s", phantom ? "the driver of car" : "person", agent, hit.x, hit.y, hit.z, car && !phantom ? " in a car" : "", body ? "" : " (no body drawn: gone at once)");
}

// VIBE_TEST_NPC_SHOOT=<frame>[,<who>[,<skip>]]: at that frame the player turns to somebody of
// the NPC world and fires one shot - a real bolt out of the gun, so the whole way from the
// trigger to the body on the ground is what is tested (VIBE_TEST_NPC_KILL calls npc_hit
// itself). who 0: the nearest on foot or on a seat; 1: the nearest who sits in a car (aimed
// ahead of it by the bolt's flight time). skip: after that many nearer ones. Called from
// loop_test_input, before the weapons.
void VulkanEngine::npc_test_input(InputFrame& in) {
    static const char* spec = std::getenv("VIBE_TEST_NPC_SHOOT");
    if (!spec || !npc_) return;
    int at = 0, who = 0, skip = 0;
    std::sscanf(spec, "%d,%d,%d", &at, &who, &skip);
    if (int(frame_number_) != at) return;
    const glm::vec3 eye = player_.eye_position();
    struct Target { float d; glm::vec3 p, v; int agent; };
    std::vector<Target> ts;
    for (const MpAvatar& a : npc_bodies_) {
        if (!a.present || !a.id) continue;
        const bool phantom = a.id >= 1000u;
        const int agent = phantom ? int(a.id - 1000u) : int(a.id) - 1;
        if (agent < 0 || size_t(agent) >= npc_views_.size()) continue;
        const npc::View& v = npc_views_[size_t(agent)];
        const bool in_car = phantom || v.hidden;
        if (in_car != (who == 1)) continue;
        const npc::View& mv = (!phantom && v.hidden && size_t(v.car) < npc_views_.size()) ? npc_views_[v.car] : v;
        const glm::vec3 p = in_car ? a.eye - glm::vec3(0.0f, 0.25f, 0.0f) : v.pos + glm::vec3(0.0f, v.pose == npc::Pose::Sit ? 0.45f : 1.25f, 0.0f);
        // (only somebody the bolt can reach: it is a body of the physics, and a table, a pane or a wall stops it)
        const float dist = glm::distance(p, eye);
        if (physics_ && dist > 0.6f) { const auto h = physics_->raycast(eye, (p - eye) / dist, dist); if (h.hit && h.distance < dist - 0.4f) continue; }
        ts.push_back({ dist, p, glm::vec3(std::sin(mv.yaw), 0.0f, -std::cos(mv.yaw)) * mv.speed, agent });
    }
    std::sort(ts.begin(), ts.end(), [](const Target& a, const Target& b) { return a.d < b.d; });
    if (size_t(skip) >= ts.size()) { log::warnf("[npc] test: nobody to shoot at (%zu %s)", ts.size(), who == 1 ? "in cars" : "on foot"); return; }
    const Target& t = ts[size_t(skip)];
    const glm::vec3 d = glm::normalize(t.p + t.v * (t.d / 220.0f) - eye);
    player_.yaw = std::atan2(d.x, -d.z);
    player_.pitch = std::asin(glm::clamp(d.y, -1.0f, 1.0f));
    in.fire = true; in.fire_held = true;
    const auto near_hit = physics_ ? physics_->raycast(eye, d, 0.15f) : PhysicsWorld::RayHit{};
    log::infof("[npc] test: a shot at %s %d, %.1f m away, moving %.1f m/s (yaw %.3f pitch %.3f)%s; weapon %d, cooldown %.2f", who == 1 ? "somebody in car / person" : "person", t.agent, double(t.d), double(glm::length(t.v)), player_.yaw, player_.pitch,
               near_hit.hit ? " - BUT something is right in front of the muzzle: no bolt will leave" : "", int(weapon_), double(fire_cooldown_));
}

// From the network (multiplayer.cpp hands them over).
void VulkanEngine::npc_event(const net::Event& e) {
    if (!npc_ || !net_) return;
    if (e.kind == net::Event::NpcHit) {
        const glm::vec3 hit(e.a[0], e.a[1], e.a[2]), dir(e.b[0], e.b[1], e.b[2]);
        if (!std::isfinite(hit.x + hit.y + hit.z + dir.x + dir.y + dir.z)) return;
        npc_hit(int(e.to), hit, dir);
    } else if (e.kind == net::Event::NpcCarTake) {
        // (the host's to answer; refused without a word when the car drove off meanwhile)
        if (net_->role() == net::Role::Host && e.to < 100) npc_host_take(int(e.to), e.from);
    } else if (e.kind == net::Event::NpcCarPush) {
        npc_car_push(e);
    } else if (e.kind == net::Event::NpcCarTaken) {
        const net::NpcCarTaken t = net::unpack_taken(e);
        if (npc_car_make(t) && t.taker == net_->id() && jeep_ && !jeep_driving() && jeep_ride_ < 0) {
            const bool in = jeep_use();                        // (the car we asked for: in)
            log::infof("[jeep] the host gave us NPC car %d: number %d, in it %d", int(t.agent), int(t.car), int(in && jeep_driving()));
        }
    }
}

// BLOOD. A shot that hits somebody throws a spray the way the bolt went (and a little back at
// the shooter), drops that fall and leave stains where they land. Drawn by fx.cpp (kind 7 of
// fx_ribbon.frag); the stains fade over two minutes.
void VulkanEngine::spawn_blood(glm::vec3 pos, glm::vec3 dir) {
    if (blood_.size() > 400) blood_.erase(blood_.begin(), blood_.begin() + 120);
    for (int i = 0; i < 16; ++i) {
        Blood d;
        const glm::vec3 rnd(frand_range(spawn_rng_state_, -1.0f, 1.0f), frand_range(spawn_rng_state_, -0.4f, 1.0f), frand_range(spawn_rng_state_, -1.0f, 1.0f));
        const bool back = i < 4;
        d.pos = pos + rnd * 0.05f;
        d.vel = (back ? -dir * frand_range(spawn_rng_state_, 0.6f, 1.8f) : dir * frand_range(spawn_rng_state_, 1.2f, 4.5f)) + rnd * 1.1f;
        d.size = i < 3 ? frand_range(spawn_rng_state_, 0.10f, 0.17f) : frand_range(spawn_rng_state_, 0.018f, 0.045f);       // (three puffs of mist, the rest drops)
        d.life = i < 3 ? 0.45f : 4.0f;
        d.seed = frand_range(spawn_rng_state_, 0.0f, 50.0f);
        blood_.push_back(d);
    }
}

void VulkanEngine::blood_update(float dt) {
    for (Blood& d : blood_) {
        d.t += dt;
        if (d.stain) continue;
        const bool mist = d.size > 0.09f;
        d.vel.y -= (mist ? 2.0f : 9.8f) * dt;
        if (mist) d.vel *= std::max(0.0f, 1.0f - 5.0f * dt);
        const glm::vec3 step = d.vel * dt;
        const float len = glm::length(step);
        if (!mist && physics_ && len > 1e-5f) {
            const auto h = physics_->raycast(d.pos, step / len, len + 0.02f);
            if (h.hit) {
                // (lands: a stain, a little bigger than the drop, lying on what it hit)
                d.pos = d.pos + step / len * std::max(h.distance - 0.012f, 0.0f);
                d.vel = h.normal; d.stain = true; d.t = 0.0f; d.life = 120.0f; d.size *= frand_range(spawn_rng_state_, 1.6f, 3.2f);
                continue;
            }
        }
        d.pos += step;
    }
    blood_.erase(std::remove_if(blood_.begin(), blood_.end(), [](const Blood& d) { return d.t > d.life; }), blood_.end());
}

void VulkanEngine::draw_npc_bodies(
VkCommandBuffer cmd, const glm::mat4& vp) {
    for (MpAvatar& a : npc_bodies_) {
        if (!a.id || !a.drawn) continue;
        mp_->draw = &a;
        draw_viewmodel(cmd, vp, a.view_inv);       // (the body alone: `unarmed`)
        mp_->draw = nullptr;
    }
    for (NpcDying& d : npc_dying_) {
        if (!d.av.drawn) continue;
        d.av.unarmed = true;
        mp_->draw = &d.av;
        draw_viewmodel(cmd, vp, d.av.view_inv);
        mp_->draw = nullptr;
    }
    // WHAT THEY CARRY (carry::): in the hands of the shown bodies, and what lies where somebody set
    // it down - a cup on a table, a toolbox beside a tank. The world pipeline, plain colours.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, world_pipeline());
    auto draw_thing = [&](uint8_t item, const glm::mat4& W) {
        const carry::Thing t = carry::thing(item);
        for (int i = 0; i < t.n; ++i) {
            const carry::Piece& p = t.p[i];
            const Mesh& m = p.mesh == 0 ? cube_mesh_ : cylinder_mesh_;
            if (!m.index_count) continue;
            const glm::mat4 M = W * glm::translate(glm::mat4(1.0f), p.at) * glm::scale(glm::mat4(1.0f), p.size);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertex_buffer, &off);
            vkCmdBindIndexBuffer(cmd, m.index_buffer, 0, VK_INDEX_TYPE_UINT32);
            PushConstants pc{};
            pc.mvp = vp * M;
            pc.model = M;
            pc.prev_mvp = prev_view_proj_valid_ ? prev_view_proj_ * M : pc.mvp;
            pc.color = glm::vec4(p.color * (1.0f + 2.5f * p.glow), 1.0f);
            pc.emissive = glm::vec4(0.0f);
            pc.tex_params = glm::vec4(-1.0f, -1.0f, 1.0f, -1.0f);
            vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, m.index_count, 1, 0, 0, 0);
        }
    };
    for (size_t b = 0; b < npc_bodies_.size() && b < size_t(kNpcBodies); ++b) {
        const MpAvatar& a = npc_bodies_[b];
        const carry::Shown& sh = g_carry_shown[b];
        if (!a.id || !a.drawn || !sh.item || sh.id != a.id) continue;
        glm::mat4 build(1.0f);
        if (a.person && people_ && people_->ok) {
            const PersonLook L = people_look(a.person, a.dress);
            build = glm::scale(glm::mat4(1.0f), glm::vec3(L.width, L.height, L.width));
        }
        draw_thing(sh.item, a.model * build * sh.M);
    }
    const glm::vec3 eye = current_frame_view_.eye_pos;
    for (const npc::View& v : npc_views_) {
        const uint8_t placed = uint8_t(v.item >> 4);
        if (!placed || v.kind != npc::Kind::Person || v.hidden || v.act == npc::Act::PickUp) continue;
        if (glm::distance(v.pos, eye) > 70.0f) continue;
        const int kind = v.pose == npc::Pose::Sit ? (v.style == 4 ? 2 : (v.style >= 1 && v.style <= 3) ? 1 : 3) : 0;
        const glm::vec3 at = npc::item_place(v.pos, v.yaw, kind, v.aim);
        draw_thing(placed, glm::translate(glm::mat4(1.0f), at) * glm::rotate(glm::mat4(1.0f), -v.yaw, glm::vec3(0.0f, 1.0f, 0.0f)) *
                           carry::set_down(placed, glm::vec3(0.0f)));
    }
}

}  // namespace vibe
