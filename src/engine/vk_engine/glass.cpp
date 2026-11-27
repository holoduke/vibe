// BREAKABLE GLASS, and the lenses of the weapons' sights.
//
// WHAT BREAKS. The glass of a building is one static-mesh part with material -3 (the office's
// `o_glass`, the station's `g_glass`): a bag of triangles. What breaks is a PANE - one piece of
// glass in its frame. Panes are found from the geometry alone (game/glass_panes.h): triangles
// in one plane that share an edge are a sheet; the mullions and transoms of the building's
// other parts that stand on a sheet cut it into panes. Sheets that are not rectangles (a
// sloping balustrade) stay whole.
//
// WHEN. Nothing is worked out for the world at start-up. A glass part is LOOKED AT
// (glass_adopt) when the player comes within 90 m of it, or when a bolt or a blast reaches it:
// then its sheets and panes are found (some milliseconds, once), a mesh of one rectangle a
// pane is made - it replaces the part's own mesh in the glass pass, so that a pane can be
// drawn by itself, or not at all - and the sheet's colliders are taken over (below). A pane's
// record is a few numbers; only a damaged pane has more (up to 7 impacts).
//
// A HIT (glass_ray: a bolt's path each physics step - combat.cpp; the test hooks) leaves an
// impact on the pane: a hole with cracks round it (the shader draws the web from the impact's
// seed: glass_pane.frag). Every hit weakens the pane; a hit near an earlier one makes the
// pieces round it drop out (the hole grows along the cracks); when the pane has no strength
// left - or a blast reaches it (glass_blast) - it BURSTS: its pieces fly and fall (Shard), its
// teeth stay in the frame for a minute. About one small pane in five is toughened glass: it
// holds one hit or none and goes all at once, into dice.
//
// WHAT GLASS STOPS. A bolt goes through glass: it leaves its mark and flies on (through a
// hole: no mark). The physics box a sheet had is removed when the part is looked at - so
// bolts are never stopped by glass that has been looked at. A WALKER is stopped by a sheet
// as before (its box in Level::aabbs) until its first pane bursts; from then on each pane
// that still stands has a box of its own (glass_player_colliders), and where one is out the
// way is open.
//
// MULTIPLAYER: nothing is sent. Every machine works the glass out for itself from the bolts
// it sees (the others' shots are replayed as bolts here), with seeds taken from where a bolt
// struck: the same shots give nearly the same damage on every machine, but nothing keeps them
// equal.
//
// LIFETIME: damage lasts as long as the part's level is loaded (home's: the whole run).
//
// THE LENSES: a held weapon's sight glass (its glTF material "Lens") is not drawn with the gun
// but here, as glass, after the lighting: draw_viewmodel leaves it in GlassState::lenses, the
// glass pass draws it with the weapon's own eye-relative matrices (glass.frag, mode 2).
//
// Hooks: VIBE_LOG_GLASS=1 (what was found, every hit and burst);
//        VIBE_TEST_GLASS_SHOTS=<n>[,<frame>[,<spread rad>]]  n shots at the glass in front of the camera, from that frame (120) on;
//        VIBE_TEST_GLASS_BREAK=<frame>                       the pane in front of the camera bursts at that frame (200);
//        VIBE_TEST_GLASS_BLAST=<frame>                       a blast 2 m in front of the camera at that frame.

#include "engine/vk_engine.h"
#include "engine/vk_engine/internal.h"
#include "engine/vk_engine/state/fx.h"
#include "engine/vk_engine/state/glass.h"
#include "engine/audio.h"
#include "engine/glass_synth.h"
#include "engine/globe.h"
#include "engine/log.h"
#include "engine/physics_world.h"
#include "game/glass_panes.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>

namespace vibe {

namespace {

// What a damaged pane's draw carries (glass_pane.vert / .frag): the same 256 bytes as the
// engine's push constants, read another way.
struct PanePush {
    glm::mat4 vp;
    glm::vec4 origin_w;
    glm::vec4 uaxis_h;
    glm::vec4 vaxis_s;
    glm::vec4 tint_n;
    glm::vec4 misc;
    glm::vec4 hits[7];
};
static_assert(sizeof(PanePush) == sizeof(PushConstants), "a pane's push constants must fill the engine's block");

float hash3(glm::vec3 p, float k) {
    const float s = std::sin(glm::dot(p, glm::vec3(12.9898f, 78.233f, 37.719f)) + k * 17.31f) * 43758.5453f;
    return s - std::floor(s);
}

bool seg_hits_box(glm::vec3 a, glm::vec3 d, glm::vec3 mn, glm::vec3 mx) {
    float t0 = 0.0f, t1 = 1.0f;
    for (int ax = 0; ax < 3; ++ax) {
        if (std::abs(d[ax]) < 1e-7f) { if (a[ax] < mn[ax] || a[ax] > mx[ax]) return false; continue; }
        float u0 = (mn[ax] - a[ax]) / d[ax], u1 = (mx[ax] - a[ax]) / d[ax];
        if (u0 > u1) std::swap(u0, u1);
        t0 = std::max(t0, u0); t1 = std::min(t1, u1);
        if (t0 > t1) return false;
    }
    return true;
}

// The size of the hole round an impact, near enough (the shader's is a polygon along its cracks).
float hole_radius(float fall) { return fall < 0.75f ? 0.012f : 0.055f * fall; }

}  // namespace

void VulkanEngine::glass_destroy() {
    if (!glass_) return;
    GlassState& G = *glass_;
    for (auto& p : G.parts) if (p->mesh.vertex_buffer) destroy_mesh(allocator_, p->mesh);
    G.parts.clear();
    G.part_of_mesh.clear();
    G.shards.clear();
    G.lenses.clear();
    if (G.pane_pipeline) vkDestroyPipeline(device_, G.pane_pipeline, nullptr);
    G.pane_pipeline = VK_NULL_HANDLE;
    for (VkShaderModule* m : { &G.pane_vert, &G.pane_frag }) {
        if (*m) vkDestroyShaderModule(device_, *m, nullptr);
        *m = VK_NULL_HANDLE;
    }
}

// ------------------------------------------------------------------ finding the panes
// Looks at one glass part of the level: its sheets and panes, the mesh that draws them, its
// colliders. Returns its index in GlassState::parts, -1 when there is nothing to it.
int VulkanEngine::glass_adopt(int lmi) {
    GlassState& G = *glass_;
    // (the level's glass parts: listed again when the level has grown; glass_adopt(-1) does only this)
    if (G.meshes_seen != world_.meshes.size()) {
        G.meshes_seen = world_.meshes.size();
        G.glass_meshes.clear();
        for (size_t i = 0; i < world_.meshes.size(); ++i)
            if (world_.meshes[i].tex_albedo == -3 && !world_.meshes[i].idx.empty()) G.glass_meshes.push_back(int(i));
    }
    if (lmi < 0 || size_t(lmi) >= world_.meshes.size()) return -1;
    if (G.part_of_mesh.size() < world_.meshes.size()) G.part_of_mesh.resize(world_.meshes.size(), -1);
    if (G.part_of_mesh[size_t(lmi)] != -1) return G.part_of_mesh[size_t(lmi)] >= 0 ? G.part_of_mesh[size_t(lmi)] : -1;
    G.part_of_mesh[size_t(lmi)] = -2;                        // (looked at; nothing, until shown otherwise)
    const game::LevelMesh& lm = world_.meshes[size_t(lmi)];
    if (size_t(lmi) >= castle_of_level_mesh_.size() || castle_of_level_mesh_[size_t(lmi)] < 0 || lm.idx.empty()) return -1;
    const double t_begin = glfwGetTime();

    auto P = std::make_unique<GlassState::Part>();
    P->level_mesh = lmi;
    P->gpu = &castle_meshes_[size_t(castle_of_level_mesh_[size_t(lmi)])];
    P->bmin = lm.bmin; P->bmax = lm.bmax; P->tint = lm.tint;

    const auto pos = [&](uint32_t i) { return lm.verts[i].p; };
    const std::vector<game::glass::Sheet> found = game::glass::find_sheets(pos, lm.idx.data(), lm.idx.size());

    // The bars on the sheets: the flat, narrow faces of the building's other parts.
    std::vector<size_t> rects;
    for (size_t i = 0; i < found.size(); ++i) if (found[i].rect) rects.push_back(i);
    std::vector<game::glass::Bars> bars(found.size());
    size_t tris_seen = 0;
    if (!rects.empty()) {
        const glm::vec3 lo = lm.bmin - glm::vec3(0.3f), hi = lm.bmax + glm::vec3(0.3f);
        for (size_t mi = 0; mi < world_.meshes.size(); ++mi) {
            const game::LevelMesh& o = world_.meshes[mi];
            if (o.tex_albedo == -3 || o.idx.empty()) continue;
            if (o.bmax.x < lo.x || o.bmin.x > hi.x || o.bmax.y < lo.y || o.bmin.y > hi.y || o.bmax.z < lo.z || o.bmin.z > hi.z) continue;
            for (size_t t = 0; t + 2 < o.idx.size(); t += 3) {
                const glm::vec3 a = o.verts[o.idx[t]].p, b = o.verts[o.idx[t + 1]].p, c = o.verts[o.idx[t + 2]].p;
                if (std::max({ a.x, b.x, c.x }) < lo.x || std::min({ a.x, b.x, c.x }) > hi.x ||
                    std::max({ a.z, b.z, c.z }) < lo.z || std::min({ a.z, b.z, c.z }) > hi.z ||
                    std::max({ a.y, b.y, c.y }) < lo.y || std::min({ a.y, b.y, c.y }) > hi.y) continue;
                ++tris_seen;
                for (size_t ri : rects) bars[ri].add(found[ri], a, b, c);
            }
        }
    }

    // The panes, and the mesh: a rectangle a pane (uv = metres from its lower left corner),
    // then what cannot break as it was.
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    for (size_t ri : rects) {
        const game::glass::Sheet& f = found[ri];
        GlassState::Sheet S;
        S.origin = f.origin; S.u = f.u; S.v = f.v; S.n = f.n; S.size = f.size;
        S.first_pane = int(P->panes.size());
        for (const auto& r : game::glass::split(f, bars[ri])) {
            GlassState::Pane p;
            p.lo = r.first; p.hi = r.second;
            p.sheet = int(P->sheets.size());
            const glm::vec2 mid = 0.5f * (p.lo + p.hi), sz = p.hi - p.lo;
            const glm::vec3 centre = S.origin + S.u * mid.x + S.v * mid.y;
            const float area = sz.x * sz.y;
            p.seed = hash3(centre, 1.0f);
            p.toughened = area < 4.5f && hash3(centre, 2.0f) < 0.20f;
            p.health = p.toughened ? (hash3(centre, 3.0f) < 0.5f ? 1.0f : 2.0f)
                                   : (5.0f + 9.0f * hash3(centre, 3.0f)) * (area > 5.0f ? 1.4f : 1.0f);
            const uint32_t i0 = uint32_t(verts.size());
            for (const glm::vec2& q : { glm::vec2(p.lo.x, p.lo.y), glm::vec2(p.hi.x, p.lo.y), glm::vec2(p.hi.x, p.hi.y), glm::vec2(p.lo.x, p.hi.y) })
                verts.push_back(Vertex{ S.origin + S.u * q.x + S.v * q.y, S.n, q - p.lo });
            idx.insert(idx.end(), { i0, i0 + 1, i0 + 2, i0, i0 + 2, i0 + 3 });
            P->panes.push_back(p);
        }
        S.n_panes = int(P->panes.size()) - S.first_pane;
        P->sheets.push_back(S);
    }
    P->loose_first = uint32_t(idx.size());
    for (const game::glass::Sheet& f : found) {
        if (f.rect) continue;
        for (uint32_t t : f.tris)
            for (int k = 0; k < 3; ++k) {
                const game::LevelVertex& v = lm.verts[lm.idx[size_t(t) * 3 + size_t(k)]];
                idx.push_back(uint32_t(verts.size()));
                verts.push_back(Vertex{ v.p, v.n, glm::vec2(v.u, v.v) });
            }
    }
    P->loose_count = uint32_t(idx.size()) - P->loose_first;
    if (verts.empty()) return -1;
    P->mesh = create_mesh_from_data(device_, allocator_, graphics_queue_, graphics_queue_family_,
                                    verts.data(), uint32_t(verts.size()), idx.data(), uint32_t(idx.size()));

    // The colliders: a thin box of the level that lies on a sheet and within it is that
    // sheet's. The physics' copy of it goes (bolts fly through glass); the walker's stays
    // until a pane is out.
    int cols = 0, bodies = 0;
    for (GlassState::Sheet& S : P->sheets) {
        const glm::vec3 an = glm::abs(S.n);
        const int k = an.x > 0.98f ? 0 : an.y > 0.98f ? 1 : an.z > 0.98f ? 2 : -1;
        if (k < 0) continue;
        const glm::vec3 far_corner = S.origin + S.u * S.size.x + S.v * S.size.y;
        const glm::vec3 smin = glm::min(S.origin, far_corner), smax = glm::max(S.origin, far_corner);
        const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
        const float sheet_area = S.size.x * S.size.y;
        for (size_t ai = world_.brushes.size(); ai < world_.aabbs.size(); ++ai) {
            const collision::AABB& a = world_.aabbs[ai];
            if (a.max[k] - a.min[k] > 0.5f) continue;
            if (S.origin[k] < a.min[k] - 0.05f || S.origin[k] > a.max[k] + 0.05f) continue;
            if (a.min[k1] < smin[k1] - 0.45f || a.max[k1] > smax[k1] + 0.45f || a.min[k2] < smin[k2] - 0.45f || a.max[k2] > smax[k2] + 0.45f) continue;
            if ((a.max[k1] - a.min[k1]) * (a.max[k2] - a.min[k2]) < 0.5f * sheet_area) continue;
            S.col = int(ai); S.col_box = a;
            ++cols;
            if (physics_ && physics_->remove_static_box(0.5f * (a.min + a.max), 0.5f * (a.max - a.min))) ++bodies;
            break;
        }
    }

    const int index = int(G.parts.size());
    log::infof("[glass] part %d of the level looked at: %zu sheets (%zu rectangles), %zu panes, %u triangles kept whole, %d colliders taken over (%d physics boxes removed), %zu triangles searched for bars, %.1f ms",
               lmi, found.size(), rects.size(), P->panes.size(), P->loose_count / 3, cols, bodies, tris_seen, (glfwGetTime() - t_begin) * 1000.0);
    G.parts.push_back(std::move(P));
    G.part_of_mesh[size_t(lmi)] = index;
    return index;
}

// ------------------------------------------------------------------ hits
void VulkanEngine::glass_sound(int kind, glm::vec3 at) {
    GlassState& G = *glass_;
    if (!audio_) return;
    if (!G.sounds_tried) {
        G.sounds_tried = true;
        struct Clip { ClipID id; const char* path; int what; };
        const Clip clips[3] = { { ClipID::GlassCrack1, "assets/sounds/glass_crack_1.wav", 0 },
                                { ClipID::GlassCrack2, "assets/sounds/glass_crack_2.wav", 1 },
                                { ClipID::GlassShatter, "assets/sounds/glass_shatter.wav", 2 } };
        G.sounds = true;
        for (const Clip& c : clips) {
            std::error_code ec;
            bool there = std::filesystem::exists(c.path, ec);
            if (!there) {
                there = glass_synth::write_wav(c.path, c.what == 2 ? glass_synth::shatter() : glass_synth::crack(c.what));
                log::infof("[glass] sound %s: %s (engine/glass_synth.h)", c.path, there ? "made now" : "could not be written");
            }
            if (there) audio_->load_clip(c.id, c.path); else G.sounds = false;
        }
    }
    if (!G.sounds) return;
    if (kind == 1) audio_->play_at(ClipID::GlassShatter, at, 1.0f, 0.08f, 0.06f);
    else           audio_->play_at((xorshift32(G.rng) & 1u) ? ClipID::GlassCrack1 : ClipID::GlassCrack2, at, 0.85f, 0.14f, 0.10f);
}

// The way from a to b (dir: the way the thing flies; energy: 1 = a bolt) against the glass.
// break_out: the pane struck bursts outright. True when glass was struck.
bool VulkanEngine::glass_ray(glm::vec3 a, glm::vec3 b, glm::vec3 dir, float energy, bool break_out) {
    GlassState& G = *glass_;
    if (!globe::current().home) return false;
    glass_adopt(-1);
    bool any = false;
    const glm::vec3 d = b - a;
    for (int lmi : G.glass_meshes) {
        const game::LevelMesh& lm = world_.meshes[size_t(lmi)];
        if (!seg_hits_box(a, d, lm.bmin - glm::vec3(0.05f), lm.bmax + glm::vec3(0.05f))) continue;
        const int pi = glass_adopt(lmi);
        if (pi < 0) continue;
        GlassState::Part& P = *G.parts[size_t(pi)];
        for (const GlassState::Sheet& S : P.sheets) {
            const float denom = glm::dot(S.n, d);
            if (std::abs(denom) < 1e-7f) continue;
            const float t = glm::dot(S.n, S.origin - a) / denom;
            if (t < 0.0f || t > 1.0f) continue;
            const glm::vec3 at = a + d * t;
            const glm::vec2 q(glm::dot(at - S.origin, S.u), glm::dot(at - S.origin, S.v));
            if (q.x < 0.0f || q.y < 0.0f || q.x > S.size.x || q.y > S.size.y) continue;
            for (int k = S.first_pane; k < S.first_pane + S.n_panes; ++k) {
                const GlassState::Pane& p = P.panes[size_t(k)];
                if (q.x < p.lo.x || q.x > p.hi.x || q.y < p.lo.y || q.y > p.hi.y) continue;
                if (p.state < 2) {
                    any = true;
                    if (break_out) glass_burst(pi, k, at, dir);
                    else           glass_hit(pi, k, q - p.lo, at, dir, energy);
                }
                break;
            }
        }
    }
    return any;
}

void VulkanEngine::glass_hit(int part, int pane, glm::vec2 uv, glm::vec3 at, glm::vec3 dir, float energy) {
    GlassState& G = *glass_;
    GlassState::Part& P = *G.parts[size_t(part)];
    GlassState::Pane& p = P.panes[size_t(pane)];
    if (p.state >= 2) return;
    // (a bolt's way is tested step after step, overlapping: the same point again is the same bolt)
    if (glm::distance(uv, p.last_uv) < 0.03f && G.now - p.last_time < 0.3) return;
    p.last_uv = uv; p.last_time = G.now;
    // Through a hole: nothing.
    for (int i = 0; i < p.n_hits; ++i)
        if (glm::distance(uv, p.hits[i].uv) < hole_radius(p.hits[i].fall)) return;

    if (p.state == 0) { p.state = 1; ++P.damaged; }
    p.health -= energy;
    float fall_before[GlassState::kMaxHits];
    for (int i = 0; i < p.n_hits; ++i) fall_before[i] = p.hits[i].fall;
    const int hits_before = p.n_hits;
    int nearest = -1; float nd = 1e9f;
    for (int i = 0; i < p.n_hits; ++i) {
        const float dd = glm::distance(uv, p.hits[i].uv);
        if (dd < nd) { nd = dd; nearest = i; }
        if (dd < 0.45f) p.hits[i].fall = std::min(3.2f, p.hits[i].fall + 0.28f * energy * (1.0f - dd / 0.45f));      // the pieces round it loosen
    }
    if (nearest >= 0 && (nd < 0.10f || p.n_hits >= GlassState::kMaxHits)) {
        p.hits[nearest].fall = std::min(3.2f, p.hits[nearest].fall + 0.55f * energy);       // the same place again: its hole grows
    } else {
        GlassState::Hit h;
        h.uv = uv;
        h.fall = std::min(3.2f, 0.25f * (energy - 1.0f));
        // (the seed from where it struck, to the centimetre: every machine draws the same web)
        const float s = std::sin(std::floor(uv.x * 100.0f) * 12.9898f + std::floor(uv.y * 100.0f) * 78.233f + p.seed * 37.7f) * 43758.5453f;
        h.seed = s - std::floor(s);
        p.hits[p.n_hits++] = h;
    }
    const GlassState::Sheet& S = P.sheets[size_t(p.sheet)];
    if (G.log) log::infof("[glass] hit: part %d pane %d at uv (%.2f, %.2f) of %.2f x %.2f m, %d impacts, strength left %.1f%s",
                          part, pane, uv.x, uv.y, p.hi.x - p.lo.x, p.hi.y - p.lo.y, p.n_hits, double(p.health), p.toughened ? " (toughened)" : "");
    if (p.health <= 0.0f) { glass_burst(part, pane, at, dir); return; }
    glass_sound(0, at);
    // Pieces thrown from a point of the pane: `n` of them, `lo` .. `hi` metres across, from
    // within `spread` metres of it.
    const float through = glm::dot(dir, S.n) >= 0.0f ? 1.0f : -1.0f;
    float floor_y = std::max(sample_terrain_height(at.x, at.z), at.y - 8.0f);
    if (physics_) {
        auto rh = physics_->raycast(at + S.n * (through * 0.3f), glm::vec3(0.0f, -1.0f, 0.0f), 12.0f);
        if (rh.hit) floor_y = rh.position.y;
    }
    auto pieces = [&](glm::vec3 from, int n, float lo, float hi, float spread, float speed) {
        for (int k = 0; k < n; ++k) {
            GlassState::Shard sh;
            const float sa = frand_range(G.rng, 0.0f, 6.28f), sr = spread * std::sqrt(frand(G.rng));
            sh.pos = from + S.u * (std::cos(sa) * sr) + S.v * (std::sin(sa) * sr);
            sh.ex = S.u; sh.ey = S.v;
            sh.pane_o = S.origin; sh.pane_n = S.n;
            const float back = (k % 4) == 0 ? -1.0f : 1.0f;          // (one in four toward the shooter)
            sh.vel = S.n * (through * back * frand_range(G.rng, 0.4f, 1.0f) * speed) + dir * (back > 0.0f ? frand_range(G.rng, 0.1f, 0.7f) * speed : 0.0f)
                   + S.u * frand_range(G.rng, -1.4f, 1.4f) + S.v * frand_range(G.rng, -0.8f, 1.6f);
            const float r = frand_range(G.rng, lo, hi) * 0.5f;
            float a0 = frand_range(G.rng, 0.0f, 6.28f);
            for (int c = 0; c < 3; ++c) { sh.c[c] = r * frand_range(G.rng, 0.6f, 1.0f) * glm::vec2(std::cos(a0), std::sin(a0)); a0 += frand_range(G.rng, 1.6f, 2.6f); }
            sh.axis = glm::normalize(S.u * frand_range(G.rng, -1.0f, 1.0f) + S.v * frand_range(G.rng, -1.0f, 1.0f) + S.n * 0.2f + glm::vec3(1e-3f));
            sh.spin = frand_range(G.rng, 6.0f, 24.0f);
            sh.life = frand_range(G.rng, 3.0f, 5.5f);
            sh.lie = frand_range(G.rng, 0.0f, 6.28f);
            sh.floor_y[0] = sh.floor_y[1] = floor_y;
            G.shards.push_back(sh);
        }
    };
    // Spall: splinters off the far side, a few back toward the shooter.
    pieces(at, 5 + int(xorshift32(G.rng) % 4u), 0.012f, 0.04f, 0.01f, 4.5f);
    // Where an earlier impact's hole has grown, the pieces that were round it drop out.
    const glm::vec3 corner = S.origin + S.u * p.lo.x + S.v * p.lo.y;
    for (int i = 0; i < hits_before; ++i) {
        const float r0 = hole_radius(fall_before[i]), r1 = hole_radius(p.hits[i].fall);
        if (r1 > r0 + 0.02f)
            pieces(corner + S.u * p.hits[i].uv.x + S.v * p.hits[i].uv.y, std::clamp(int((r1 * r1 - r0 * r0) * 900.0f), 3, 14), 0.03f, 0.09f, r1, 1.6f);
    }
    if (G.shards.size() > GlassState::kMaxShards)
        G.shards.erase(G.shards.begin(), G.shards.begin() + std::ptrdiff_t(G.shards.size() - GlassState::kMaxShards));
}

void VulkanEngine::glass_burst(int part, int pane, glm::vec3 at, glm::vec3 dir) {
    GlassState& G = *glass_;
    GlassState::Part& P = *G.parts[size_t(part)];
    GlassState::Pane& p = P.panes[size_t(pane)];
    if (p.state >= 2) return;
    GlassState::Sheet& S = P.sheets[size_t(p.sheet)];
    if (p.state == 0) ++P.damaged;
    p.state = 2; p.burst_t = 0.0f;
    // From now on each pane of this sheet that still stands is its own collider.
    if (S.col >= 0 && !S.col_off && size_t(S.col) < world_.aabbs.size()) {
        world_.aabbs[size_t(S.col)] = { glm::vec3(0.0f, -9000.0f, 0.0f), glm::vec3(0.0f, -9000.0f, 0.0f) };
        S.col_off = true;
    }
    const glm::vec2 sz = p.hi - p.lo, mid = 0.5f * (p.lo + p.hi);
    const glm::vec3 centre = S.origin + S.u * mid.x + S.v * mid.y;
    const glm::vec3 corner = S.origin + S.u * p.lo.x + S.v * p.lo.y;
    if (glm::dot(dir, dir) < 1e-6f) dir = S.n;
    dir = glm::normalize(dir);
    const float through = glm::dot(dir, S.n) >= 0.0f ? 1.0f : -1.0f;
    // The floor on either side of it.
    float floor_y[2];
    for (int side = 0; side < 2; ++side) {
        glm::vec3 from = centre + S.n * (side ? 0.5f : -0.5f);
        from.y = std::min(centre.y, corner.y + 0.3f);
        floor_y[side] = sample_terrain_height(from.x, from.z);
        if (physics_) {
            auto rh = physics_->raycast(from, glm::vec3(0.0f, -1.0f, 0.0f), 20.0f);
            if (rh.hit) floor_y[side] = rh.position.y;
        }
    }
    const float area = sz.x * sz.y;
    const int n = p.toughened ? std::clamp(int(area * 150.0f), 90, 420) : std::clamp(int(area * 40.0f), 26, 180);
    for (int k = 0; k < n; ++k) {
        const glm::vec2 q(frand_range(G.rng, 0.0f, sz.x), frand_range(G.rng, 0.0f, sz.y));
        bool in_hole = false;
        for (int i = 0; i < p.n_hits; ++i) if (glm::distance(q, p.hits[i].uv) < hole_radius(p.hits[i].fall)) in_hole = true;
        if (in_hole) continue;
        GlassState::Shard sh;
        sh.pos = corner + S.u * q.x + S.v * q.y;
        sh.ex = S.u; sh.ey = S.v;
        sh.pane_o = S.origin; sh.pane_n = S.n;
        sh.floor_y[0] = floor_y[0]; sh.floor_y[1] = floor_y[1];
        const glm::vec3 from_hit = sh.pos - at;
        const float dist = glm::length(from_hit);
        const glm::vec3 radial = dist > 1e-4f ? from_hit / dist : glm::vec3(0.0f);
        const float near_hit = std::max(0.0f, 1.0f - dist / 0.6f);
        sh.vel = S.n * (through * frand_range(G.rng, 0.3f, 2.6f)) + dir * (frand_range(G.rng, 0.0f, 1.2f) + near_hit * frand_range(G.rng, 2.0f, 7.0f))
               + radial * (frand_range(G.rng, 0.0f, 2.2f) / (1.0f + 1.5f * dist)) + glm::vec3(0.0f, frand_range(G.rng, -0.4f, 0.9f), 0.0f);
        // (the break runs out from where it was struck: what is further away lets go later)
        sh.wait = std::min(dist * 0.05f, 0.38f) + frand_range(G.rng, 0.0f, 0.04f);
        const float r = p.toughened ? frand_range(G.rng, 0.010f, 0.026f) : frand_range(G.rng, 0.05f, 0.16f);
        float a0 = frand_range(G.rng, 0.0f, 6.28f);
        for (int c = 0; c < 3; ++c) { sh.c[c] = r * frand_range(G.rng, 0.55f, 1.0f) * glm::vec2(std::cos(a0), std::sin(a0)); a0 += frand_range(G.rng, 1.5f, 2.7f); }
        sh.axis = glm::normalize(S.u * frand_range(G.rng, -1.0f, 1.0f) + S.v * frand_range(G.rng, -1.0f, 1.0f) + S.n * frand_range(G.rng, -0.3f, 0.3f) + glm::vec3(1e-3f));
        sh.spin = frand_range(G.rng, 2.5f, 13.0f) * (p.toughened ? 2.0f : 1.0f);
        sh.life = frand_range(G.rng, 4.5f, 7.5f);
        sh.lie = frand_range(G.rng, 0.0f, 6.28f);
        G.shards.push_back(sh);
    }
    if (G.shards.size() > GlassState::kMaxShards)
        G.shards.erase(G.shards.begin(), G.shards.begin() + std::ptrdiff_t(G.shards.size() - GlassState::kMaxShards));
    glass_sound(1, centre);
    if (G.log) log::infof("[glass] burst: part %d pane %d (%.2f x %.2f m%s) at (%.1f, %.1f, %.1f), %d pieces, floors %.2f / %.2f, %zu pieces alive",
                          part, pane, sz.x, sz.y, p.toughened ? ", toughened" : "", centre.x, centre.y, centre.z, n, floor_y[0], floor_y[1], G.shards.size());
}

// A blast: the panes near it burst, those further off are struck by what it throws.
void VulkanEngine::glass_blast(glm::vec3 pos, float radius) {
    GlassState& G = *glass_;
    if (!globe::current().home || radius <= 0.0f) return;
    glass_adopt(-1);
    for (int lmi : G.glass_meshes) {
        const game::LevelMesh& lm = world_.meshes[size_t(lmi)];
        const glm::vec3 c = glm::clamp(pos, lm.bmin, lm.bmax);
        if (glm::distance(c, pos) > radius) continue;
        const int pi = glass_adopt(lmi);
        if (pi < 0) continue;
        GlassState::Part& P = *G.parts[size_t(pi)];
        for (size_t k = 0; k < P.panes.size(); ++k) {
            GlassState::Pane& p = P.panes[k];
            if (p.state >= 2) continue;
            const GlassState::Sheet& S = P.sheets[size_t(p.sheet)];
            const glm::vec2 mid = 0.5f * (p.lo + p.hi), sz = p.hi - p.lo;
            const glm::vec3 centre = S.origin + S.u * mid.x + S.v * mid.y;
            const float dist = glm::distance(centre, pos);
            if (dist > radius) continue;
            const glm::vec3 dir = dist > 1e-3f ? (centre - pos) / dist : S.n;
            if (dist < 0.6f * radius) { glass_burst(pi, int(k), centre, dir); continue; }
            const int hits = 1 + int(xorshift32(G.rng) % 3u);
            for (int h = 0; h < hits && p.state < 2; ++h) {
                const glm::vec2 uv(frand_range(G.rng, 0.1f, 0.9f) * sz.x, frand_range(G.rng, 0.1f, 0.9f) * sz.y);
                glass_hit(pi, int(k), uv, S.origin + S.u * (p.lo.x + uv.x) + S.v * (p.lo.y + uv.y), dir, 1.6f);
            }
        }
    }
}

// ------------------------------------------------------------------ every step
void VulkanEngine::glass_tick(float dt) {
    GlassState& G = *glass_;
    static const bool kLog = std::getenv("VIBE_LOG_GLASS") != nullptr;
    G.log = kLog;
    G.now += double(dt);

    // The glass near the player is looked at before anything reaches it.
    G.scan_in -= dt;
    if (G.scan_in <= 0.0f && globe::current().home) {
        G.scan_in = 0.5f;
        const glm::vec3 eye = player_.eye_position();
        glass_adopt(-1);
        for (int lmi : G.glass_meshes) {
            const game::LevelMesh& lm = world_.meshes[size_t(lmi)];
            if (glm::distance(glm::clamp(eye, lm.bmin, lm.bmax), eye) < 90.0f) glass_adopt(lmi);
        }
    }

    for (auto& P : G.parts) {
        if (P->damaged == 0) continue;
        for (GlassState::Pane& p : P->panes)
            if (p.state == 2) { p.burst_t += dt; if (p.burst_t > GlassState::kTeethSeconds) p.state = 3; }
    }

    // The pieces.
    const float g = std::max(4.0f, game_.gravity);
    for (GlassState::Shard& sh : G.shards) {
        if (sh.wait > 0.0f) { sh.wait -= dt; continue; }
        sh.age += dt;
        if (sh.rest) continue;
        sh.vel.y -= g * dt;
        sh.vel *= std::max(0.0f, 1.0f - 0.5f * dt);
        sh.pos += sh.vel * dt;
        sh.ang += sh.spin * dt;
        // The floor under the piece where it IS (2026-11-27: "broken glass not always falls on the
        // ground but somewhere stuck in the air"). It was measured once, half a metre either side
        // of the pane's middle, at burst: a piece that flew further came to rest at that height -
        // over a lower lawn, beyond a sill or a step, off a balcony - in the air. Now: the
        // terrain or whatever the physics world holds (floors, steps, rocks, a car) straight
        // under it, measured again every 25 cm of travel (a ray each: a burst's 180 pieces cost
        // a few hundred rays over their fall). VIBE_GLASS_FLOOR_OLD=1: the old way.
        static const bool kFloorOld = std::getenv("VIBE_GLASS_FLOOR_OLD") != nullptr;
        float fy = sh.floor_y[glm::dot(sh.pos - sh.pane_o, sh.pane_n) > 0.0f ? 1 : 0];
        if (!kFloorOld) {
            const glm::vec2 xz(sh.pos.x, sh.pos.z);
            if (glm::distance(xz, sh.fy_at) > 0.25f) {
                float f = sample_terrain_height(sh.pos.x, sh.pos.z);
                if (physics_) {
                    // (from just above the piece: what it is falling onto, not a roof over it)
                    const glm::vec3 from(sh.pos.x, sh.pos.y + 0.05f, sh.pos.z);
                    const auto rh = physics_->raycast(from, glm::vec3(0.0f, -1.0f, 0.0f), std::max(from.y - f + 0.5f, 0.5f));
                    if (rh.hit && rh.distance > 1e-3f) f = std::max(f, rh.position.y);
                }
                sh.fy = f;
                sh.fy_at = xz;
            }
            fy = sh.fy;
        }
        if (sh.pos.y <= fy + 0.025f && sh.vel.y < 0.0f) {
            sh.pos.y = fy + 0.025f;
            if (!sh.bounced && sh.vel.y < -3.5f) {
                sh.bounced = true;
                sh.vel = glm::vec3(sh.vel.x * 0.45f, -sh.vel.y * 0.22f, sh.vel.z * 0.45f);
                sh.spin *= 0.6f;
            } else {
                sh.rest = true;
                sh.pos.y = fy + 0.022f + 0.006f * (sh.lie / 6.28f);       // (clear of the floor's finish: boards, a rug)
                static int rest_logs = 0;
                if (G.log && rest_logs < 12) {
                    ++rest_logs;
                    log::infof("[glass] a piece rests at (%.2f, %.2f, %.2f): the floor under it %.2f (at burst it was taken as %.2f), the terrain %.2f",
                               sh.pos.x, sh.pos.y, sh.pos.z, fy, sh.floor_y[glm::dot(sh.pos - sh.pane_o, sh.pane_n) > 0.0f ? 1 : 0], sample_terrain_height(sh.pos.x, sh.pos.z));
                }
            }
        }
    }
    G.shards.erase(std::remove_if(G.shards.begin(), G.shards.end(), [](const GlassState::Shard& s) { return s.age > s.life; }), G.shards.end());

    // Test hooks (once a frame).
    static uint64_t hook_frame = ~0ull;
    if (hook_frame != frame_number_) {
        hook_frame = frame_number_;
        static int shots = 0, shots_at = 120; static float spread = 0.035f; static bool init = false;
        static int break_at = -1, blast_at = -1;
        if (!init) {
            init = true;
            if (const char* e = std::getenv("VIBE_TEST_GLASS_SHOTS")) std::sscanf(e, "%d,%d,%f", &shots, &shots_at, &spread);
            if (const char* e = std::getenv("VIBE_TEST_GLASS_BREAK")) break_at = std::max(1, std::atoi(e) > 1 ? std::atoi(e) : 200);
            if (const char* e = std::getenv("VIBE_TEST_GLASS_BLAST")) blast_at = std::max(1, std::atoi(e) > 1 ? std::atoi(e) : 200);
            G.test_shots_left = shots;
        }
        const glm::vec3 eye = player_.eye_position(), fwd = glm::normalize(player_.forward());
        const glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 1.0f, 0.0f)) + glm::vec3(1e-5f, 0.0f, 0.0f));
        const glm::vec3 up = glm::cross(right, fwd);
        if (G.test_shots_left > 0 && int(frame_number_) >= shots_at && (frame_number_ % 4) == 0) {
            --G.test_shots_left;
            const glm::vec3 d = glm::normalize(fwd + right * frand_range(G.rng, -spread, spread) + up * frand_range(G.rng, -spread, spread));
            const bool hit = glass_ray(eye, eye + d * 80.0f, d, 1.0f);
            log::infof("[glass] test shot %d: %s", shots - G.test_shots_left, hit ? "struck glass" : "no glass that way");
        }
        if (int(frame_number_) == break_at) {
            const bool hit = glass_ray(eye, eye + fwd * 80.0f, fwd, 1.0f, true);
            log::infof("[glass] test break: %s", hit ? "a pane burst" : "no glass that way");
        }
        if (int(frame_number_) == blast_at) { glass_blast(eye + fwd * 2.0f, 9.0f); log::infof("[glass] test blast"); }
    }
}

// What a walker bumps into where a sheet has lost a pane: each pane that still stands.
void VulkanEngine::glass_player_colliders(std::vector<collision::AABB>& out) const {
    const GlassState& G = *glass_;
    if (!globe::current().home) return;
    for (const auto& P : G.parts) {
        if (P->damaged == 0) continue;
        if (glm::distance(glm::clamp(player_.position, P->bmin, P->bmax), player_.position) > 14.0f) continue;
        for (const GlassState::Sheet& S : P->sheets) {
            if (!S.col_off) continue;
            const glm::vec3 an = glm::abs(S.n);
            const int k = an.x > an.y && an.x > an.z ? 0 : an.y > an.z ? 1 : 2;
            for (int i = S.first_pane; i < S.first_pane + S.n_panes; ++i) {
                const GlassState::Pane& p = P->panes[size_t(i)];
                if (p.state >= 2) continue;
                const glm::vec3 a = S.origin + S.u * p.lo.x + S.v * p.lo.y, b = S.origin + S.u * p.hi.x + S.v * p.hi.y;
                collision::AABB box{ glm::min(a, b), glm::max(a, b) };
                box.min[k] = S.col_box.min[k]; box.max[k] = S.col_box.max[k];
                out.push_back(box);
            }
        }
    }
}

// ------------------------------------------------------------------ drawing
bool VulkanEngine::glass_draws(const CastleMeshGpu* gp) const {
    for (const auto& P : glass_->parts) if (P->gpu == gp) return true;
    return false;
}

// A part that has been looked at, in the glass pass (the FX pass's glass pipeline is bound,
// and is bound again on return): the whole panes and what cannot break in a few draws, then
// each damaged pane by itself.
void VulkanEngine::glass_draw_part(VkCommandBuffer cmd, const CastleMeshGpu* gp, const glm::mat4& vp) {
    GlassState& G = *glass_;
    const GlassState::Part* part = nullptr;
    for (const auto& P : G.parts) if (P->gpu == gp) { part = P.get(); break; }
    if (!part || !part->mesh.vertex_buffer) return;
    const GlassState::Part& P = *part;
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &P.mesh.vertex_buffer, &off);
    vkCmdBindIndexBuffer(cmd, P.mesh.index_buffer, 0, VK_INDEX_TYPE_UINT32);
    PushConstants pc{};
    pc.mvp = vp;
    pc.model = glm::mat4(1.0f);
    pc.prev_mvp = vp;
    pc.color = glm::vec4(P.tint, 1.0f);
    pc.emissive = glm::vec4(0.0f);
    pc.grass_params = glm::vec4(G.indoor, 0.0f, 0.0f, 0.0f);       // (glass.frag: the eye indoors)
    vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
    if (P.damaged == 0) {
        vkCmdDrawIndexed(cmd, P.mesh.index_count, 1, 0, 0, 0);
        return;
    }
    // runs of whole panes
    const size_t n = P.panes.size();
    for (size_t i = 0; i < n; ) {
        if (P.panes[i].state != 0) { ++i; continue; }
        size_t j = i;
        while (j < n && P.panes[j].state == 0) ++j;
        vkCmdDrawIndexed(cmd, uint32_t(j - i) * 6, 1, uint32_t(i) * 6, 0, 0);
        i = j;
    }
    if (P.loose_count) vkCmdDrawIndexed(cmd, P.loose_count, 1, P.loose_first, 0, 0);
    if (!G.pane_pipeline) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, G.pane_pipeline);
    for (size_t i = 0; i < n; ++i) {
        const GlassState::Pane& p = P.panes[i];
        if (p.state != 1 && p.state != 2) continue;
        const GlassState::Sheet& S = P.sheets[size_t(p.sheet)];
        PanePush pp{};
        pp.vp = vp;
        pp.origin_w = glm::vec4(S.origin + S.u * p.lo.x + S.v * p.lo.y, p.hi.x - p.lo.x);
        pp.uaxis_h = glm::vec4(S.u, p.hi.y - p.lo.y);
        pp.vaxis_s = glm::vec4(S.v, p.seed);
        pp.tint_n = glm::vec4(P.tint, float(p.n_hits));
        pp.misc = glm::vec4(float(p.state), p.burst_t, p.toughened ? 1.0f : 0.0f, G.indoor);
        for (int h = 0; h < p.n_hits && h < 7; ++h) pp.hits[h] = glm::vec4(p.hits[h].uv, p.hits[h].fall, p.hits[h].seed);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PanePush), &pp);
        vkCmdDrawIndexed(cmd, 6, 1, uint32_t(i) * 6, 0, 0);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fx_->glass_pipeline);
}

// The pieces in the air and on the floor: a triangle each (uv = its corners, for the pale edge).
void VulkanEngine::glass_shard_verts(std::vector<Vertex>& out) const {
    out.clear();
    const GlassState& G = *glass_;
    if (G.shards.empty() || !globe::current().home) return;
    out.reserve(G.shards.size() * 3);
    static const glm::vec2 kUv[3] = { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 0.0f, 1.0f } };
    for (const GlassState::Shard& sh : G.shards) {
        glm::vec3 X = sh.ex, Y = sh.ey;
        if (sh.rest) {
            X = glm::vec3(std::cos(sh.lie), 0.0f, std::sin(sh.lie));
            Y = glm::vec3(-std::sin(sh.lie), 0.0f, std::cos(sh.lie));
        } else if (sh.wait <= 0.0f) {
            const glm::quat q = glm::angleAxis(sh.ang, sh.axis);
            X = q * sh.ex; Y = q * sh.ey;
        }
        const glm::vec3 N = glm::cross(X, Y);
        const float left = std::clamp((sh.life - sh.age) / 0.6f, 0.0f, 1.0f);
        for (int c = 0; c < 3; ++c)
            out.push_back(Vertex{ sh.pos + (X * sh.c[c].x + Y * sh.c[c].y) * left, N, kUv[c] });
    }
}

// ------------------------------------------------------------------ the lenses
// A sight's lens of a weapon being drawn (draw_viewmodel): kept for the glass pass.
// model / vp: the weapon's own, relative to the eye. `own`: the player's own weapon.
void VulkanEngine::glass_lens(const ViewmodelMesh& vm, const glm::mat4& model, const glm::mat4& vp, bool own, bool rifle) {
    GlassState& G = *glass_;
    if (G.lens_frame != frame_number_) { G.lens_frame = frame_number_; G.lenses.clear(); }
    if (G.lenses.size() >= 64) return;
    GlassState::Lens L;
    L.mesh = &vm.mesh;
    L.model = model;
    L.to_clip = vp;
    L.mvp = vp * model;
    L.centre = glm::vec3(model * glm::vec4(vm.lens_centre, 1.0f));
    const glm::vec3 ax = glm::mat3(model) * vm.lens_axis;
    const float scale = glm::length(ax);
    L.axis = scale > 1e-6f ? ax / scale : glm::vec3(0.0f, 0.0f, 1.0f);
    L.radius = vm.lens_radius * scale;
    L.glow = glm::vec3(vm.base_color);
    L.aim = own ? glm::smoothstep(0.0f, 1.0f, aim_zoom_) : 0.0f;
    // The laser's sight enlarges a little and has a lit reticle; the rifle's is a plain window
    // on its red dot.
    L.mag = rifle ? 1.06f : 1.22f;
    L.reticle = (own && !rifle) ? 1.0f : 0.0f;
    G.lenses.push_back(L);
}

bool VulkanEngine::glass_lenses_pending() const { return glass_->lens_frame == frame_number_ && !glass_->lenses.empty(); }

// (the FX pass's glass pipeline is bound)
void VulkanEngine::glass_draw_lenses(VkCommandBuffer cmd) {
    GlassState& G = *glass_;
    if (!glass_lenses_pending()) return;
    VkDeviceSize off = 0;
    for (const GlassState::Lens& L : G.lenses) {
        PushConstants pc{};
        pc.mvp = L.mvp;
        pc.model = L.model;
        pc.prev_mvp = L.to_clip;
        pc.color = glm::vec4(L.glow, 1.0f);
        pc.emissive = glm::vec4(L.centre, 2.0f);
        pc.tex_params = glm::vec4(L.axis, L.radius);
        pc.grass_params = glm::vec4(L.aim, L.mag, L.reticle, 0.0f);
        vkCmdPushConstants(cmd, pipeline_layout_, kPushConstantStages, 0, sizeof(PushConstants), &pc);
        vkCmdBindVertexBuffers(cmd, 0, 1, &L.mesh->vertex_buffer, &off);
        vkCmdBindIndexBuffer(cmd, L.mesh->index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, L.mesh->index_count, 1, 0, 0, 0);
    }
    G.lenses.clear();
}

} // namespace vibe
