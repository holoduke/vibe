#include "game/level.h"

#include <cstdio>
#include <string>
#include <glm/vec2.hpp>
#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>

namespace qlike::game {

namespace {
// World-space inflation applied to SPOM-wall brush AABBs to cover the
// shell-mapping extrusion done by cube.vert (kShellThickness = 0.04 m
// along each face normal, gated to SPOM wall materials). Frustum culls
// using the raw size would clip wall edges at screen-side as the
// extruded geometry crosses the frustum plane a few cm earlier than
// the un-extruded box. Match the cube.vert constant exactly + a tiny
// safety margin for the bumpy brick silhouette extension.
//
// Applied ONLY to Level::render_aabbs. Level::aabbs stays tight so the
// player collision boxes match the un-extruded wall -- inflating the
// collision AABB would push the player back 5 cm from every wall,
// which reads as "I can't walk up to the wall."
constexpr float kShellInflation = 0.05f;

// SPOM wall materials -- mirrors height_idx_for_albedo() in cube.frag.
// Keep in sync; the extrusion gate in cube.vert uses the same list.
bool is_spom_wall_albedo(int tex_albedo) {
    return tex_albedo == 1 || tex_albedo == 4;
}

// Tight, physics-accurate AABB. Used for collision.
collision::AABB to_aabb(const Brush& b) {
    glm::vec3 half = b.size * 0.5f;
    return { b.center - half, b.center + half };
}

// Render-cull AABB. Inflated for SPOM wall brushes so the frustum cull
// doesn't clip the cube.vert shell extrusion at the screen edge.
collision::AABB to_render_aabb(const Brush& b) {
    glm::vec3 half = b.size * 0.5f;
    if (is_spom_wall_albedo(b.tex_albedo)) {
        // Walls extrude along their face normal only, but cheaper to
        // pad all 3 axes than to track which faces are walls vs floors
        // here -- the world-space cost is 5 cm on a 22 m wall
        // (< 0.25 % overshoot, no measurable frustum-cull regression
        // in the castle scene).
        half += glm::vec3(kShellInflation);
    }
    return { b.center - half, b.center + half };
}

// Sentinel for "no separate fallback" вЂ” substitutes `rgb` so brushes added
// without an explicit fallback look the same in textured and untextured mode
// (lanterns, lamps, anything whose `rgb` is already the desired color).
constexpr glm::vec3 kFallbackUseRgb(-1.0f, -1.0f, -1.0f);

void add(Level& lv, glm::vec3 center, glm::vec3 size, glm::vec3 rgb,
         int tex_albedo = -1, int tex_normal = -1, float uv_scale = 1.0f,
         glm::vec3 fallback = kFallbackUseRgb) {
    glm::vec3 fb = (fallback == kFallbackUseRgb) ? rgb : fallback;
    Brush b{ center, size, glm::vec4(rgb, 1.0f), glm::vec3(0.0f), false,
             tex_albedo, tex_normal, uv_scale,
             glm::vec4(fb, 1.0f) };
    lv.brushes.push_back(b);
    lv.aabbs.push_back(to_aabb(b));
    lv.render_aabbs.push_back(to_render_aabb(b));
}

void add_lamp(Level& lv, glm::vec3 center, glm::vec3 size, glm::vec3 rgb,
              glm::vec3 emissive) {
    // Full-emissive shading bypasses albedo/normal entirely вЂ” set fallback
    // to rgb so the box looks identical in both texture modes.
    Brush b{ center, size, glm::vec4(rgb, 1.0f), emissive, true,
             -1, -1, 1.0f, glm::vec4(rgb, 1.0f) };
    lv.brushes.push_back(b);
    lv.aabbs.push_back(to_aabb(b));
    lv.render_aabbs.push_back(to_render_aabb(b));
}

// ---------------------------------------------------------------------
// Mesh builders for the non-box castle parts. Everything is built in
// castle-local space (y = 0 at the courtyard floor) and lifted with the
// brushes in init_world. UVs are in texture repeats (tile_m metres per
// repeat), V pointing DOWN the world so NormalGL's green = world-up.
// ---------------------------------------------------------------------
constexpr float kPi = 3.14159265358979f;

void mesh_push(LevelMesh& m, const LevelVertex& v) {
    m.verts.push_back(v);
    m.bmin = glm::min(m.bmin, v.p);
    m.bmax = glm::max(m.bmax, v.p);
}
void mesh_quad(LevelMesh& m, const LevelVertex& a, const LevelVertex& b,
               const LevelVertex& c, const LevelVertex& d) {
    // The world pipelines cull back faces with the engine's cube winding
    // (front = cross(b - a, c - a) pointing outward). Builders pass the
    // intended outward normal per vertex; flip the winding whenever the
    // quad's geometric normal disagrees, so every builder is cull-safe.
    glm::vec3 g = glm::cross(b.p - a.p, c.p - a.p);
    if (glm::dot(g, g) < 1e-12f) g = glm::cross(c.p - a.p, d.p - a.p);
    const bool flip = glm::dot(g, a.n + b.n + c.n + d.n) < 0.0f;
    uint32_t base = static_cast<uint32_t>(m.verts.size());
    mesh_push(m, a); mesh_push(m, b); mesh_push(m, c); mesh_push(m, d);
    if (flip) m.idx.insert(m.idx.end(), { base, base + 2, base + 1, base, base + 3, base + 2 });
    else      m.idx.insert(m.idx.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
}

// Revolve a (radius, y) profile around the vertical axis through `c`.
// Hard edges between profile segments (flat along the profile, smooth
// around it). The number of texture repeats around is rounded to an
// integer so the seam wraps cleanly.
void lathe(LevelMesh& m, glm::vec3 c, const std::vector<glm::vec2>& prof,
           int segs, float tile_m) {
    float max_r = 0.0f;
    for (const auto& p : prof) max_r = (std::max)(max_r, p.x);
    // Even count: the masonry texture is 2:1, so the shader halves u (same
    // stone size as the straight walls) and the seam must still wrap.
    const float tiles_around =
        (std::max)(2.0f, 2.0f * std::round(kPi * max_r / tile_m));
    float s = 0.0f;
    for (size_t j = 0; j + 1 < prof.size(); ++j) {
        glm::vec2 p0 = prof[j], p1 = prof[j + 1];
        glm::vec2 d = p1 - p0;
        float len = glm::length(d);
        if (len < 1e-5f) continue;
        glm::vec2 n2 = glm::vec2(d.y, -d.x) / len;   // outward in (r, y)
        for (int i = 0; i < segs; ++i) {
            float a0 = 2.0f * kPi * float(i) / float(segs);
            float a1 = 2.0f * kPi * float(i + 1) / float(segs);
            auto vtx = [&](float a, glm::vec2 p, float sv, float ui) {
                glm::vec2 dir(std::cos(a), std::sin(a));
                return LevelVertex{
                    c + glm::vec3(dir.x * p.x, p.y, dir.y * p.x),
                    glm::normalize(glm::vec3(dir.x * n2.x, n2.y, dir.y * n2.x)),
                    ui, -sv / tile_m };
            };
            float u0 = tiles_around * float(i) / float(segs);
            float u1 = tiles_around * float(i + 1) / float(segs);
            mesh_quad(m, vtx(a0, p0, s, u0), vtx(a1, p0, s, u1),
                         vtx(a1, p1, s + len, u1), vtx(a0, p1, s + len, u0));
        }
        s += len;
    }
}

// Solid ring sector (merlon / corbel on a round tower).
void ring_sector(LevelMesh& m, glm::vec3 c, float r_in, float r_out,
                 float a0, float a1, float y0, float y1, int segs, float tile_m) {
    auto P = [&](float a, float r, float y) {
        return c + glm::vec3(std::cos(a) * r, y, std::sin(a) * r);
    };
    const glm::vec3 up(0, 1, 0);
    for (int i = 0; i < segs; ++i) {
        float b0 = a0 + (a1 - a0) * float(i) / float(segs);
        float b1 = a0 + (a1 - a0) * float(i + 1) / float(segs);
        glm::vec3 n0(std::cos(b0), 0, std::sin(b0)), n1(std::cos(b1), 0, std::sin(b1));
        float u0 = b0 * r_out / tile_m, u1 = b1 * r_out / tile_m;
        float v0 = -y0 / tile_m, v1 = -y1 / tile_m;
        mesh_quad(m, {P(b0, r_out, y0), n0, u0, v0}, {P(b1, r_out, y0), n1, u1, v0},
                     {P(b1, r_out, y1), n1, u1, v1}, {P(b0, r_out, y1), n0, u0, v1});
        mesh_quad(m, {P(b1, r_in, y0), -n1, u1, v0}, {P(b0, r_in, y0), -n0, u0, v0},
                     {P(b0, r_in, y1), -n0, u0, v1}, {P(b1, r_in, y1), -n1, u1, v1});
        float ri = r_in / tile_m, ro = r_out / tile_m;
        mesh_quad(m, {P(b0, r_in, y1), up, u0, ri}, {P(b0, r_out, y1), up, u0, ro},
                     {P(b1, r_out, y1), up, u1, ro}, {P(b1, r_in, y1), up, u1, ri});
        mesh_quad(m, {P(b1, r_in, y0), -up, u1, ri}, {P(b1, r_out, y0), -up, u1, ro},
                     {P(b0, r_out, y0), -up, u0, ro}, {P(b0, r_in, y0), -up, u0, ri});
    }
    auto cap = [&](float a, float side) {
        glm::vec3 n = glm::vec3(-std::sin(a), 0, std::cos(a)) * side;
        float ri = r_in / tile_m, ro = r_out / tile_m;
        mesh_quad(m, {P(a, r_in, y0), n, ri, -y0 / tile_m}, {P(a, r_out, y0), n, ro, -y0 / tile_m},
                     {P(a, r_out, y1), n, ro, -y1 / tile_m}, {P(a, r_in, y1), n, ri, -y1 / tile_m});
    };
    cap(a0, -1.0f);
    cap(a1, +1.0f);
}

// Crenellated ring: `count` merlons evenly spaced, `duty` = merlon share.
void merlon_ring(LevelMesh& m, glm::vec3 c, float r_in, float r_out,
                 float y0, float y1, int count, float duty, float tile_m) {
    for (int k = 0; k < count; ++k) {
        float a = 2.0f * kPi * float(k) / float(count);
        float w = 2.0f * kPi / float(count) * duty;
        ring_sector(m, c, r_in, r_out, a - w * 0.5f, a + w * 0.5f, y0, y1, 3, tile_m);
    }
}

// Corbels under a parapet: small radial brackets.
void corbel_ring(LevelMesh& m, glm::vec3 c, float r_in, float r_out,
                 float y0, float y1, int count, float tile_m) {
    for (int k = 0; k < count; ++k) {
        float a = 2.0f * kPi * (float(k) + 0.5f) / float(count);
        float w = 0.30f / r_out;
        ring_sector(m, c, r_in, r_out, a - w * 0.5f, a + w * 0.5f, y0, y1, 1, tile_m);
    }
}

// Round-arched gate passage: the part of the rectangular opening
// [-R, R] x [ys, ys + R] OUTSIDE the semicircle of radius R, extruded
// z0..z1, plus the curved soffit (vault underside).
void arch_infill(LevelMesh& m, float ys, float R, float z0, float z1, float tile_m) {
    const int N = 20;
    auto rect_pt = [&](float a) {
        float ca = std::cos(a), sa = std::sin(a);
        float t = 1e9f;
        if (std::abs(ca) > 1e-4f) t = (std::min)(t, R / std::abs(ca));
        if (sa > 1e-4f)           t = (std::min)(t, R / sa);
        return glm::vec2(ca * t, ys + sa * t);
    };
    auto V = [&](glm::vec2 q, float z, glm::vec3 n) {
        return LevelVertex{ glm::vec3(q.x, q.y, z), n, q.x / tile_m, -q.y / tile_m };
    };
    const glm::vec3 fz(0, 0, 1);
    float arc = 0.0f;
    for (int i = 0; i < N; ++i) {
        float a0 = kPi * float(i) / float(N), a1 = kPi * float(i + 1) / float(N);
        glm::vec2 c0(R * std::cos(a0), ys + R * std::sin(a0));
        glm::vec2 c1(R * std::cos(a1), ys + R * std::sin(a1));
        glm::vec2 p0 = rect_pt(a0), p1 = rect_pt(a1);
        mesh_quad(m, V(c0, z1, fz), V(p0, z1, fz), V(p1, z1, fz), V(c1, z1, fz));
        mesh_quad(m, V(c1, z0, -fz), V(p1, z0, -fz), V(p0, z0, -fz), V(c0, z0, -fz));
        glm::vec3 n0(-std::cos(a0), -std::sin(a0), 0), n1(-std::cos(a1), -std::sin(a1), 0);
        float seg = glm::length(c1 - c0);
        mesh_quad(m, {glm::vec3(c0, z0), n0, z0 / tile_m, -arc / tile_m},
                     {glm::vec3(c0, z1), n0, z1 / tile_m, -arc / tile_m},
                     {glm::vec3(c1, z1), n1, z1 / tile_m, -(arc + seg) / tile_m},
                     {glm::vec3(c1, z0), n1, z0 / tile_m, -(arc + seg) / tile_m});
        arc += seg;
    }
}

// Raised voussoir band framing the arch on a wall face (XY plane at z0,
// protruding to z1).
void arch_band(LevelMesh& m, float ys, float r_in, float r_out,
               float z0, float z1, float tile_m) {
    const int N = 20;
    auto V = [&](glm::vec3 p, glm::vec3 n) {
        return LevelVertex{ p, n, p.x / tile_m, -p.y / tile_m };
    };
    const glm::vec3 o(0, ys, 0), fz(0, 0, 1), Z0(0, 0, z0), Z1(0, 0, z1);
    for (int i = 0; i < N; ++i) {
        float a0 = kPi * float(i) / float(N), a1 = kPi * float(i + 1) / float(N);
        glm::vec3 d0(std::cos(a0), std::sin(a0), 0), d1(std::cos(a1), std::sin(a1), 0);
        mesh_quad(m, V(o + d0 * r_in + Z1, fz), V(o + d0 * r_out + Z1, fz),
                     V(o + d1 * r_out + Z1, fz), V(o + d1 * r_in + Z1, fz));
        mesh_quad(m, V(o + d0 * r_out + Z0, d0), V(o + d1 * r_out + Z0, d1),
                     V(o + d1 * r_out + Z1, d1), V(o + d0 * r_out + Z1, d0));
        mesh_quad(m, V(o + d1 * r_in + Z0, -d1), V(o + d0 * r_in + Z0, -d0),
                     V(o + d0 * r_in + Z1, -d0), V(o + d1 * r_in + Z1, -d1));
    }
}

// Four-sided hip (pyramid) roof over a rectangle, with eave underside.
void hip_roof(LevelMesh& m, glm::vec3 c, float hx, float hz, float y0,
              float apex_h, float tile_m) {
    glm::vec3 apex = c + glm::vec3(0, y0 + apex_h, 0);
    glm::vec3 k[4] = { c + glm::vec3(-hx, y0, hz), c + glm::vec3(hx, y0, hz),
                       c + glm::vec3(hx, y0, -hz), c + glm::vec3(-hx, y0, -hz) };
    for (int f = 0; f < 4; ++f) {
        glm::vec3 a = k[f], b = k[(f + 1) % 4];
        glm::vec3 n = glm::normalize(glm::cross(b - a, apex - a));
        if (n.y < 0.0f) n = -n;
        glm::vec3 e = glm::normalize(b - a);
        float slant = glm::length(apex - (a + b) * 0.5f);
        auto V = [&](glm::vec3 p, float vs) {
            return LevelVertex{ p, n, glm::dot(p - a, e) / tile_m, -vs / tile_m };
        };
        mesh_quad(m, V(a, 0.0f), V(b, 0.0f), V(apex, slant), V(apex, slant));
    }
    const glm::vec3 dn(0, -1, 0);
    mesh_quad(m, {k[3], dn, 0, 0}, {k[2], dn, 1, 0}, {k[1], dn, 1, 1}, {k[0], dn, 0, 1});
}

// Player collision for a round tower: a union of boxes whose corners lie
// on the circle (8 per quarter turn, so the flats between them cut at most
// about 1 % of the radius into the round - three boxes left gaps one could
// walk 30 cm into).
void tower_colliders(std::vector<collision::AABB>& out, glm::vec3 c, float r,
                     float y0, float y1) {
    constexpr int kN = 8;
    for (int k = 0; k < kN; ++k) {
        const float a = (float(k) + 0.5f) * (1.5707963f / float(kN));
        const float ex = r * std::cos(a), ez = r * std::sin(a);
        out.push_back({ glm::vec3(c.x - ex, c.y + y0, c.z - ez),
                        glm::vec3(c.x + ex, c.y + y1, c.z + ez) });
    }
}
}

Level make_arena(float r, float wh) {
    Level lv;

    // Texture indices match the engine's load order:
    //   0 = Ground054, 1 = Bricks067, 2 = Wood048,
    //   3 = Metal042A, 4 = PaintedBricks001 (keep walls),
    //   5 = Tiles130 (courtyard floor), 6 = Tiles074 (keep interior floor)
    constexpr int kGround = 0;
    constexpr int kBrick  = 1;
    constexpr int kPaint  = 4;
    constexpr int kTile1  = 5;     // courtyard floor
    constexpr int kTile2  = 6;     // keep interior floor

    // For each brush we pass two colors:
    //   `tint`    вЂ” multiplies the texture (near-white so the texture's hue
    //               dominates when textures are on)
    //   `proto`   вЂ” the saturated prototype color used when textures are off
    //
    // Without `proto`, switching textures off would leave near-white walls
    // everywhere, with no distinguishing detail to mask RT GI noise вЂ” the
    // user sees that as "stripes / weird texture / flicker on the wall."
    auto T = [](float r, float g, float b) { return glm::vec3(r, g, b); };

    // Floor + arena perimeter walls removed вЂ” replaced by the procedural
    // terrain heightmap (see src/engine/terrain.cpp). The castle below
    // sits on a flattened plateau in the heightmap so the brushes still
    // align at y=0..keep_h.
    (void)r; (void)wh; (void)kGround;

    // -----------------------------------------------------------------
    // Castle (centered at origin, entrance facing +Z).
    //   * 1.6 m thick, 7 m curtain walls with a walkable top, a plinth
    //     course along the outer faces and real crenellations.
    //   * Four ROUND corner towers (mesh): battered base, string course,
    //     corbelled parapet, merlon ring, conical tile roof.
    //   * Gatehouse: deep block with a round-arched passage (mesh arch +
    //     voussoir band), flanked by two round towers with overhanging
    //     conical roofs. The gate doors (props.cpp) sit at z = 11.
    //   * Keep: 8 x 8 m, 11 m tall, hip roof, four corbelled bartizans.
    // Straight walls stay brushes (box relief gives them real masonry
    // depth); everything curved is a LevelMesh.
    // -----------------------------------------------------------------
    {
        const float cr     = 11.0f;  // wall centre line half-extent
        const float wh     = 7.0f;   // wall-walk height
        const float wt     = 1.6f;   // wall thickness
        const float base_y = -1.5f;  // walls run below grade (uneven ground)
        const float ent_w  = 3.2f;   // gate opening width
        const float ent_h  = 3.6f;   // gate opening height (rect part)

        const auto castle_tint  = T(0.95f, 0.92f, 0.86f);  // warm sandstone
        const auto castle_proto = T(0.55f, 0.50f, 0.42f);
        const float wall_uv  = 0.22f;          // ~4.5 m texture repeat (~55 cm stones)
        const float cren_uv  = 0.30f;
        const float stone_tile = 1.0f / wall_uv;
        const auto stair_tint   = T(0.92f, 0.85f, 0.75f);
        const auto stair_proto  = T(0.50f, 0.35f, 0.20f);
        const auto crenel_proto = T(0.45f, 0.42f, 0.35f);
        constexpr int kRoof = 13;              // engine kTexRoof

        auto wall = [&](glm::vec3 c, glm::vec3 s) {
            add(lv, c, s, castle_tint, kBrick, kBrick, wall_uv, castle_proto);
        };
        auto merlon = [&](glm::vec3 c, glm::vec3 s) {
            add(lv, c, s, castle_tint, kBrick, kBrick, cren_uv, crenel_proto);
        };

        // Tower layout (needed to trim walls / merlons).
        const float tw_r   = 3.2f;                       // corner tower radius
        const float gt_r   = 2.4f;                       // gatehouse tower radius
        const float gt_x   = 4.4f;                       // gatehouse tower |x| (clear of the passage)
        const float gt_z   = cr + 0.9f;                  // gatehouse tower z
        const float gh_z0  = cr - wt * 0.5f;             // gatehouse block inner face
        const float gh_z1  = cr + 2.0f;                  // gatehouse block outer face
        const float gh_h   = 9.0f;

        // ---- Curtain walls ----
        const float wy = (wh + base_y) * 0.5f, wsy = wh - base_y;
        wall(glm::vec3(0.0f, wy, -cr), glm::vec3(2.0f * cr, wsy, wt));   // south
        wall(glm::vec3( cr, wy, 0.0f), glm::vec3(wt, wsy, 2.0f * cr));   // east
        wall(glm::vec3(-cr, wy, 0.0f), glm::vec3(wt, wsy, 2.0f * cr));   // west
        {   // north, either side of the gatehouse
            float x0 = gt_x, x1 = cr;
            float w = x1 - x0, cx = (x0 + x1) * 0.5f;
            wall(glm::vec3(-cx, wy, cr), glm::vec3(w, wsy, wt));
            wall(glm::vec3( cx, wy, cr), glm::vec3(w, wsy, wt));
        }
        // Plinth course: 0.45 m skirt along every outer face.
        {
            const float ph = 0.8f - base_y, py = (0.8f + base_y) * 0.5f, pd = 0.45f;
            const float po = cr + wt * 0.5f + pd * 0.5f;
            wall(glm::vec3(0.0f, py, -po), glm::vec3(2.0f * cr, ph, pd));
            wall(glm::vec3( po, py, 0.0f), glm::vec3(pd, ph, 2.0f * cr));
            wall(glm::vec3(-po, py, 0.0f), glm::vec3(pd, ph, 2.0f * cr));
            float x0 = gt_x, x1 = cr, cx = (x0 + x1) * 0.5f;
            wall(glm::vec3(-cx, py, po), glm::vec3(x1 - x0, ph, pd));
            wall(glm::vec3( cx, py, po), glm::vec3(x1 - x0, ph, pd));
        }
        // Crenellations on the outer edge of the wall-walk.
        {
            const float m_w = 0.95f, m_h = 1.15f, m_d = 0.5f, pitch = 1.65f;
            const float my = wh + m_h * 0.5f;
            const float edge = cr + wt * 0.5f - m_d * 0.5f;
            const float lim = cr - tw_r - 0.35f;          // stop at the towers
            auto run = [&](bool along_x, float fixed, float sign, float skip_lo) {
                for (float t = -lim + m_w * 0.5f; t <= lim - m_w * 0.5f + 1e-3f; t += pitch) {
                    if (std::abs(t) < skip_lo) continue;
                    glm::vec3 c = along_x ? glm::vec3(t, my, fixed * sign)
                                          : glm::vec3(fixed * sign, my, t);
                    glm::vec3 s = along_x ? glm::vec3(m_w, m_h, m_d)
                                          : glm::vec3(m_d, m_h, m_w);
                    merlon(c, s);
                }
            };
            run(true,  edge, -1.0f, 0.0f);                 // south
            run(false, edge, +1.0f, 0.0f);                 // east
            run(false, edge, -1.0f, 0.0f);                 // west
            run(true,  edge, +1.0f, gt_x + gt_r + 0.4f);   // north (gatehouse gap)
        }

        // ---- Gatehouse block with arched passage ----
        {
            const float d = gh_z1 - gh_z0, cz = (gh_z0 + gh_z1) * 0.5f;
            const float gy = (gh_h + base_y) * 0.5f, gsy = gh_h - base_y;
            const float side_w = gt_x - ent_w * 0.5f;
            const float side_x = ent_w * 0.5f + side_w * 0.5f;
            wall(glm::vec3(-side_x, gy, cz), glm::vec3(side_w, gsy, d));
            wall(glm::vec3( side_x, gy, cz), glm::vec3(side_w, gsy, d));
            wall(glm::vec3(0.0f, (gh_h + ent_h) * 0.5f, cz),
                 glm::vec3(ent_w, gh_h - ent_h, d));                  // over the passage
            // Battlement on the gatehouse front + back edges.
            for (float x : { -1.1f, 1.1f }) {
                merlon(glm::vec3(x, gh_h + 0.55f, gh_z1 - 0.25f), glm::vec3(0.95f, 1.1f, 0.5f));
                merlon(glm::vec3(x, gh_h + 0.55f, gh_z0 + 0.25f), glm::vec3(0.95f, 1.1f, 0.5f));
            }
            // Paved passage floor (the courtyard slab stops at the wall).
            add(lv, glm::vec3(0.0f, -0.20f, cz - 0.2f), glm::vec3(ent_w, 0.5f, d + 0.4f),
                T(0.96f, 0.94f, 0.90f), kTile1, kTile1, 0.4f, T(0.45f, 0.42f, 0.38f));
        }

        // ---- Mesh parts ----
        LevelMesh stone;  stone.tex_albedo = kBrick; stone.tex_normal = kBrick;
        stone.tint = castle_tint; stone.fallback = castle_proto;
        LevelMesh roof;   roof.tex_albedo = kRoof; roof.tex_normal = kRoof;
        roof.tint = T(0.92f, 0.86f, 0.82f); roof.fallback = T(0.55f, 0.25f, 0.18f);
        std::vector<collision::AABB> mesh_cols;
        const float roof_tile = 1.7f;

        // Corner towers.
        const glm::vec3 corners[4] = {
            { cr, 0.0f,  cr}, {-cr, 0.0f,  cr}, { cr, 0.0f, -cr}, {-cr, 0.0f, -cr} };
        for (const glm::vec3& c : corners) {
            const float r = tw_r, top = 12.0f;
            lathe(stone, c, {
                { r + 0.55f, -4.0f }, { r + 0.55f, 0.0f },           // buried footing
                { r, 2.6f },                                          // battered base
                { r, 6.8f }, { r + 0.14f, 6.8f }, { r + 0.14f, 7.1f },// string course
                { r, 7.1f }, { r, top - 0.6f },
                { r + 0.45f, top },                                   // machicolation
                { r + 0.45f, top + 0.55f }, { r - 0.4f, top + 0.55f },// parapet top
                { r - 0.4f, top + 0.2f }, { 0.0f, top + 0.2f },       // tower top
            }, 64, stone_tile);
            corbel_ring(stone, c, r - 0.05f, r + 0.42f, top - 1.15f, top - 0.45f, 28, stone_tile);
            merlon_ring(stone, c, r - 0.4f, r + 0.45f, top + 0.55f, top + 1.6f, 12, 0.55f, stone_tile);
            lathe(roof, c, { { r - 0.25f, top + 0.4f }, { 0.0f, top + 7.2f } }, 64, roof_tile);
            tower_colliders(mesh_cols, c, r + 0.2f, base_y, top + 1.6f);
        }

        // Gatehouse towers (overhanging conical roofs).
        for (float sx : { -1.0f, 1.0f }) {
            const glm::vec3 c(sx * gt_x, 0.0f, gt_z);
            const float r = gt_r, top = 10.4f;
            lathe(stone, c, {
                { r + 0.45f, -3.0f }, { r + 0.45f, 0.0f }, { r, 2.2f },
                { r, top - 0.7f }, { r + 0.35f, top - 0.2f }, { r + 0.35f, top },
                { 0.0f, top },
            }, 56, stone_tile);
            corbel_ring(stone, c, r - 0.05f, r + 0.33f, top - 1.25f, top - 0.6f, 22, stone_tile);
            lathe(roof, c, {
                { r + 0.35f, top - 0.1f }, { r + 0.75f, top - 0.1f },   // eave underside
                { 0.0f, top + 6.0f } }, 56, roof_tile);
            tower_colliders(mesh_cols, c, r + 0.2f, base_y, top);
        }

        // Arched gate passage (front half, outside the doors at z = 11)
        // + raised voussoir band on the gatehouse face.
        arch_infill(stone, ent_h - ent_w * 0.5f, ent_w * 0.5f, cr + 0.25f, gh_z1, stone_tile);
        arch_band(stone, ent_h - ent_w * 0.5f, ent_w * 0.5f, ent_w * 0.5f + 0.55f,
                  gh_z1, gh_z1 + 0.14f, stone_tile);

        // ---- Keep (donjon) ----
        const float kp_cx = 0.0f, kp_cz = -4.0f;
        const float kp_s = 8.0f, kp_wt = 1.0f, kp_h = 11.0f;
        const float door_w = 1.4f, door_h = 2.6f;
        {
            const float ky = (kp_h - 0.5f) * 0.5f, ksy = kp_h + 0.5f;
            const float h2 = kp_s * 0.5f;
            wall(glm::vec3(kp_cx, ky, kp_cz - h2 + kp_wt * 0.5f), glm::vec3(kp_s, ksy, kp_wt)); // S
            wall(glm::vec3(kp_cx + h2 - kp_wt * 0.5f, ky, kp_cz), glm::vec3(kp_wt, ksy, kp_s - 2.0f * kp_wt)); // E
            wall(glm::vec3(kp_cx - h2 + kp_wt * 0.5f, ky, kp_cz), glm::vec3(kp_wt, ksy, kp_s - 2.0f * kp_wt)); // W
            const float seg_w = (kp_s - door_w) * 0.5f;
            const float seg_x = (door_w + seg_w) * 0.5f;
            const float nz = kp_cz + h2 - kp_wt * 0.5f;
            wall(glm::vec3(kp_cx - seg_x, ky, nz), glm::vec3(seg_w, ksy, kp_wt));
            wall(glm::vec3(kp_cx + seg_x, ky, nz), glm::vec3(seg_w, ksy, kp_wt));
            wall(glm::vec3(kp_cx, (kp_h + door_h) * 0.5f, nz),
                 glm::vec3(door_w, kp_h - door_h, kp_wt));                  // over the door
            // Ceiling slab (the hip roof sits on it).
            wall(glm::vec3(kp_cx, kp_h - 0.15f, kp_cz), glm::vec3(kp_s - 2.0f * kp_wt, 0.3f, kp_s - 2.0f * kp_wt));
            // Interior floor.
            add(lv, glm::vec3(kp_cx, -0.17f, kp_cz),
                glm::vec3(kp_s - 2.0f * kp_wt, 0.5f, kp_s - 2.0f * kp_wt),
                T(0.96f, 0.94f, 0.90f), kTile2, kTile2, 0.5f, T(0.55f, 0.48f, 0.40f));
            hip_roof(roof, glm::vec3(kp_cx, 0.0f, kp_cz), h2 + 0.45f, h2 + 0.45f,
                     kp_h, 4.6f, roof_tile);
            // Corbelled corner bartizans with their own cone roofs.
            for (float sx : { -1.0f, 1.0f }) {
                for (float sz : { -1.0f, 1.0f }) {
                    glm::vec3 c(kp_cx + sx * (h2 + 0.15f), 0.0f, kp_cz + sz * (h2 + 0.15f));
                    lathe(stone, c, {
                        { 0.15f, kp_h - 4.2f }, { 1.05f, kp_h - 2.6f },
                        { 1.05f, kp_h + 1.0f }, { 0.0f, kp_h + 1.0f } }, 40, stone_tile);
                    lathe(roof, c, { { 1.0f, kp_h + 0.9f }, { 1.3f, kp_h + 0.9f },
                                     { 0.0f, kp_h + 3.9f } }, 40, roof_tile);
                }
            }
        }

        // ---- Stairway to the wall-walk (east wall, inside) ----
        {
            const int   n_steps = 17;                    // 0.41 m risers
            const float step_h  = wh / float(n_steps);
            const float step_d  = 0.5f;
            const float stair_w = 2.0f;
            const float stair_x = cr - wt * 0.5f - stair_w * 0.5f;
            const float z0 = -(float(n_steps) * 0.5f) * step_d;
            for (int i = 0; i < n_steps; ++i) {
                float h = step_h * float(i + 1);
                add(lv, glm::vec3(stair_x, (h + base_y) * 0.5f, z0 + float(i) * step_d),
                    glm::vec3(stair_w, h - base_y, step_d),
                    stair_tint, kBrick, kBrick, wall_uv, stair_proto);
            }
        }

        // ---- Courtyard floor (inside the curtain walls) ----
        {
            const float in_ext = (cr - wt * 0.5f) * 2.0f + 0.1f;
            add(lv, glm::vec3(0.0f, -0.20f, 0.0f), glm::vec3(in_ext, 0.5f, in_ext),
                T(0.96f, 0.94f, 0.90f), kTile1, kTile1, 0.4f, T(0.45f, 0.42f, 0.38f));
        }

        // Lanterns: inside the gate, and either side of the keep door.
        // (the same timber posts with a hanging lantern as along the road,
        // arms reaching in over the path between them)
        lv.lamp_spots.push_back({ glm::vec3( ent_w * 0.5f + 1.2f, 0.05f, gh_z0 - 1.2f), glm::vec3(-1.0f, 0.0f, 0.0f) });
        lv.lamp_spots.push_back({ glm::vec3(-ent_w * 0.5f - 1.2f, 0.05f, gh_z0 - 1.2f), glm::vec3( 1.0f, 0.0f, 0.0f) });
        lv.lamp_spots.push_back({ glm::vec3( door_w * 0.5f + 0.8f, 0.05f, kp_cz + kp_s * 0.5f + 0.8f), glm::vec3(-1.0f, 0.0f, 0.0f) });
        lv.lamp_spots.push_back({ glm::vec3(-door_w * 0.5f - 0.8f, 0.05f, kp_cz + kp_s * 0.5f + 0.8f), glm::vec3( 1.0f, 0.0f, 0.0f) });

        lv.meshes.push_back(std::move(stone));
        lv.meshes.push_back(std::move(roof));
        // Mesh colliders go AFTER every brush AABB (see Level::meshes).
        for (const auto& a : mesh_cols) lv.aabbs.push_back(a);
    }

    // Outer arena lanterns removed along with the floor + perimeter walls
    // вЂ” they used to anchor the corners of the brick arena. With the
    // open terrain there is nothing to corner.
    return lv;
}

// Free-standing outpost towers scattered over the foothills (world space,
// `base` = ground point under the centre). Styles:
//   0 watchtower  - tall round tower, machicolated parapet, conical roof
//   1 beacon      - slender, open crenellated top with a raised fire basket
//   2 ruin        - broken shell: wall segments of uneven height, no roof
//   3 keep        - squat drum with a second, narrower storey and a roof
void append_outpost_tower(Level& lv, glm::vec3 base, int style, uint32_t seed) {
    constexpr int kBrick = 1, kRoof = 13;
    const float stone_tile = 1.0f / 0.22f, roof_tile = 1.7f;
    LevelMesh stone; stone.tex_albedo = kBrick; stone.tex_normal = kBrick;
    stone.tint = glm::vec3(0.86f, 0.84f, 0.80f); stone.fallback = glm::vec3(0.55f, 0.50f, 0.42f);
    LevelMesh roof; roof.tex_albedo = kRoof; roof.tex_normal = kRoof;
    roof.tint = glm::vec3(0.92f, 0.86f, 0.82f); roof.fallback = glm::vec3(0.55f, 0.25f, 0.18f);
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return float((seed >> 8) & 0xFFFFu) / 65535.0f;
    };
    std::vector<collision::AABB> cols;
    const glm::vec3 c = base;
    if (style == 0) {
        const float r = 3.0f + rnd() * 0.6f, top = 15.0f + rnd() * 4.0f;
        lathe(stone, c, {
            { r + 0.9f, -6.0f }, { r + 0.9f, 0.0f }, { r, 3.4f },             // footing + batter
            { r, top * 0.5f }, { r + 0.14f, top * 0.5f }, { r + 0.14f, top * 0.5f + 0.3f },
            { r, top * 0.5f + 0.3f }, { r, top - 0.7f },
            { r + 0.5f, top }, { r + 0.5f, top + 0.55f }, { r - 0.4f, top + 0.55f },
            { r - 0.4f, top + 0.2f }, { 0.0f, top + 0.2f },
        }, 48, stone_tile);
        corbel_ring(stone, c, r - 0.05f, r + 0.46f, top - 1.2f, top - 0.45f, 22, stone_tile);
        merlon_ring(stone, c, r - 0.4f, r + 0.5f, top + 0.55f, top + 1.6f, 10, 0.55f, stone_tile);
        lathe(roof, c, { { r - 0.3f, top + 0.4f }, { r + 0.05f, top + 1.7f }, { 0.0f, top + 8.0f } }, 48, roof_tile);
        tower_colliders(cols, c, r + 0.2f, -1.0f, top + 1.6f);
    } else if (style == 1) {
        const float r = 2.1f + rnd() * 0.3f, top = 20.0f + rnd() * 5.0f;
        lathe(stone, c, {
            { r + 1.1f, -6.0f }, { r + 1.1f, 0.0f }, { r + 0.25f, 4.5f }, { r, top - 0.9f },
            { r + 0.65f, top }, { r + 0.65f, top + 0.5f }, { r + 0.25f, top + 0.5f },
            { r + 0.25f, top + 0.15f }, { 0.0f, top + 0.15f },
        }, 40, stone_tile);
        corbel_ring(stone, c, r - 0.05f, r + 0.6f, top - 1.3f, top - 0.5f, 16, stone_tile);
        merlon_ring(stone, c, r + 0.25f, r + 0.65f, top + 0.5f, top + 1.5f, 8, 0.5f, stone_tile);
        // Fire basket on a short pillar.
        lathe(stone, c, { { 0.35f, top + 0.15f }, { 0.35f, top + 1.3f }, { 0.75f, top + 1.6f },
                          { 0.75f, top + 1.95f }, { 0.55f, top + 1.95f }, { 0.0f, top + 1.75f } }, 16, stone_tile);
        tower_colliders(cols, c, r + 0.4f, -1.0f, top + 1.5f);
    } else if (style == 2) {
        const float r = 3.6f + rnd() * 0.8f, top = 7.0f + rnd() * 3.0f;
        // Solid plinth, then a hollow broken shell above it.
        lathe(stone, c, { { r + 0.8f, -6.0f }, { r + 0.8f, 0.0f }, { r, 2.2f }, { r, 2.6f }, { 0.0f, 2.6f } },
              40, stone_tile);
        const int segs = 14;
        float h = top;
        for (int k = 0; k < segs; ++k) {
            // Random-walk the wall height; one side collapsed almost to the plinth.
            h = glm::clamp(h + (rnd() - 0.5f) * 4.0f, 3.0f, top + 2.5f);
            float hk = (k >= segs - 4) ? 2.9f + rnd() * 1.4f : h;
            float a0 = 2.0f * kPi * float(k) / float(segs), a1 = 2.0f * kPi * float(k + 1) / float(segs);
            ring_sector(stone, c, r - 1.0f, r, a0, a1 + 0.002f, 2.55f, hk, 3, stone_tile);
        }
        // Fallen blocks around the foot.
        for (int k = 0; k < 7; ++k) {
            float a = rnd() * 2.0f * kPi, d = r + 1.0f + rnd() * 3.5f;
            glm::vec3 bc = c + glm::vec3(std::cos(a) * d, -0.2f, std::sin(a) * d);
            float s = 0.45f + rnd() * 0.5f;
            lathe(stone, bc, { { 0.0f, 0.0f }, { s, 0.0f }, { s * 0.9f, s * 0.9f }, { 0.0f, s * 0.9f } }, 5, stone_tile);
        }
        tower_colliders(cols, c, r + 0.1f, -1.0f, 2.6f);   // the solid plinth only: the shell is open
    } else {
        const float r = 4.6f + rnd() * 0.6f, mid = 8.0f + rnd() * 1.5f, r2 = r - 1.6f, top = mid + 6.5f + rnd() * 2.0f;
        lathe(stone, c, {
            { r + 1.0f, -6.0f }, { r + 1.0f, 0.0f }, { r, 2.8f }, { r, mid - 0.6f },
            { r + 0.45f, mid }, { r + 0.45f, mid + 0.5f }, { r - 0.35f, mid + 0.5f },
            { r - 0.35f, mid + 0.15f }, { r2, mid + 0.15f },                       // wall-walk
            { r2, top - 0.5f }, { r2 + 0.35f, top }, { r2 + 0.35f, top + 0.2f }, { 0.0f, top + 0.2f },
        }, 56, stone_tile);
        corbel_ring(stone, c, r - 0.05f, r + 0.42f, mid - 1.2f, mid - 0.45f, 30, stone_tile);
        merlon_ring(stone, c, r - 0.35f, r + 0.45f, mid + 0.5f, mid + 1.5f, 14, 0.55f, stone_tile);
        lathe(roof, c, { { r2 + 0.35f, top + 0.1f }, { r2 + 0.8f, top + 0.1f }, { 0.0f, top + 5.5f } }, 48, roof_tile);
        tower_colliders(cols, c, r + 0.2f, -1.0f, mid + 1.5f);
    }
    lv.meshes.push_back(std::move(stone));
    if (!roof.idx.empty()) lv.meshes.push_back(std::move(roof));
    for (const auto& a : cols) lv.aabbs.push_back(a);
}


// ---------------------------------------------------------------------
// The chapel. Geometry comes from Blender (tools/church/make_church.py ->
// assets/church/church.mesh): blocks of "mesh <name> <triangles>" followed
// by 3 lines per triangle: px py pz  nx ny nz  u v  (game axes, metres; uv
// in metres along the surface). Each block becomes one LevelMesh with the
// game's own materials: castle masonry (with stone relief), roof tiles,
// timber, iron, and dark glass that glows faintly at night.
// ---------------------------------------------------------------------

// A Roman stone bridge in WORLD space: a level-ish deck from `a` to `b` (deck surface heights
// in .y) carried on three semicircular arches between massive piers, with solid abutments at
// both ends and a parapet along each side. `width`: clear deck width. `ground(x, z)`: terrain
// height (the piers and abutments are carried 1.5 m into it). Adds its mesh and player
// colliders (the deck and parapets as a slab, the piers and abutments solid; under the arches
// it is open).
void append_stone_bridge(Level& lv, glm::vec3 a, glm::vec3 b, float width,
                         const std::function<float(float, float)>& ground) {
    constexpr int kBrick = 1;
    const float tile = 1.0f / 0.22f;
    const glm::vec2 a2(a.x, a.z), b2(b.x, b.z);
    const float L = glm::length(b2 - a2);
    if (L < 20.0f) return;
    const glm::vec2 d = (b2 - a2) / L, n(-d.y, d.x);
    const glm::vec3 D(d.x, 0.0f, d.y), N(n.x, 0.0f, n.y), U(0.0f, 1.0f, 0.0f);
    const float hw = width * 0.5f, par = 0.55f, ho = hw + par, par_h = 0.80f;     // (the coping stones come on top)
    // (a short ramp up from the road at the near end: the ground the road was levelled on, with
    // its close-range relief, otherwise shows through the first metres of the deck)
    auto deck = [&](float s) {
        const float t = s / L, r = std::clamp(s / 2.5f, 0.0f, 1.0f);
        return glm::mix(a.y, b.y, t) + 0.5f * std::sin(kPi * t) + 0.32f * r * r * (3.0f - 2.0f * r) * (1.0f - t);
    };
    auto gnd = [&](float s) {
        const glm::vec2 p = a2 + d * s;
        return (std::min)({ ground(p.x, p.y), ground(p.x + n.x * ho, p.y + n.y * ho), ground(p.x - n.x * ho, p.y - n.y * ho) });
    };
    // Three arches in the middle part; what is left at the ends are the abutments.
    const float R = (std::min)(7.0f, (L - 26.0f) / 6.0f), pier = 4.0f;
    const float open = 6.0f * R + 2.0f * pier, s_first = std::floor((L - open) * 0.42f * 2.0f) * 0.5f;
    float ac[3];
    for (int i = 0; i < 3; ++i) ac[i] = s_first + R + float(i) * (2.0f * R + pier);
    float crown = 1e9f;
    for (float s = ac[0] - R; s <= ac[2] + R; s += 0.5f) crown = (std::min)(crown, deck(s));
    crown -= 1.35f;                                    // (arch ring + road bed over the crown)
    const float spring = crown - R;
    auto arch_of = [&](float s) { for (int i = 0; i < 3; ++i) if (std::abs(s - ac[i]) < R) return i; return -1; };
    auto arch_y = [&](int i, float s) { const float x = s - ac[i]; return spring + std::sqrt((std::max)(R * R - x * x, 0.0f)); };

    LevelMesh m; m.tex_albedo = kBrick; m.tex_normal = kBrick;
    m.tint = glm::vec3(0.80f, 0.78f, 0.72f); m.fallback = glm::vec3(0.52f, 0.48f, 0.40f);
    LevelMesh road; road.tex_albedo = 14; road.tex_normal = 14;      // (engine kTexRoad)
    road.tint = glm::vec3(0.5f); road.fallback = glm::vec3(0.40f, 0.38f, 0.34f); road.relief = 0.5f;
    m.relief = 0.8f;
    auto P = [&](float s, float u, float y) { return glm::vec3(a.x, 0.0f, a.z) + D * s + N * u + U * y; };
    auto V = [&](float s, float u, float y, glm::vec3 nn, float tu, float tv) { return LevelVertex{ P(s, u, y), nn, tu / tile, tv / tile }; };
    const float ds = 0.5f;
    const int ns = int(std::ceil(L / ds));
    std::vector<collision::AABB> cols;
    for (int k = 0; k < ns; ++k) {
        const float s0 = float(k) * ds, s1 = (std::min)(L, s0 + ds), sm = 0.5f * (s0 + s1);
        const int ai = arch_of(sm);
        const float t0 = deck(s0), t1 = deck(s1);
        float y0, y1;
        if (ai >= 0) { y0 = arch_y(ai, s0); y1 = arch_y(ai, s1); }
        else { y0 = (std::min)(gnd(s0) - 1.5f, t0 - 1.0f); y1 = (std::min)(gnd(s1) - 1.5f, t1 - 1.0f); }
        for (float sg : { -1.0f, 1.0f }) {
            const glm::vec3 nn = N * sg;
            // Outer face (up to the top of the parapet).
            mesh_quad(m, V(s0, sg * ho, y0, nn, s0, -y0), V(s1, sg * ho, y1, nn, s1, -y1),
                         V(s1, sg * ho, t1 + par_h, nn, s1, -(t1 + par_h)), V(s0, sg * ho, t0 + par_h, nn, s0, -(t0 + par_h)));
            // Parapet: top and the face toward the road.
            mesh_quad(m, V(s0, sg * hw, t0 + par_h, U, s0, sg * hw), V(s1, sg * hw, t1 + par_h, U, s1, sg * hw),
                         V(s1, sg * ho, t1 + par_h, U, s1, sg * ho), V(s0, sg * ho, t0 + par_h, U, s0, sg * ho));
            mesh_quad(m, V(s0, sg * hw, t0, -nn, s0, -t0), V(s1, sg * hw, t1, -nn, s1, -t1),
                         V(s1, sg * hw, t1 + par_h, -nn, s1, -(t1 + par_h)), V(s0, sg * hw, t0 + par_h, -nn, s0, -(t0 + par_h)));
        }
        // The road: the castle road's cobbles (2.5 m to a repeat).
        {
            auto C = [&](float s, float u, float y) { return LevelVertex{ P(s, u, y), U, u / 2.5f, s / 2.5f }; };
            mesh_quad(road, C(s0, -hw, t0), C(s1, -hw, t1), C(s1, hw, t1), C(s0, hw, t0));
        }
        // Under the arch.
        if (ai >= 0) {
            const glm::vec3 n0 = -glm::normalize(D * (s0 - ac[ai]) + U * (y0 - spring) + U * 1e-4f);
            const glm::vec3 n1 = -glm::normalize(D * (s1 - ac[ai]) + U * (y1 - spring) + U * 1e-4f);
            const float w0 = R * std::asin(std::clamp((s0 - ac[ai]) / R, -1.0f, 1.0f)), w1 = R * std::asin(std::clamp((s1 - ac[ai]) / R, -1.0f, 1.0f));
            mesh_quad(m, V(s0, -ho, y0, n0, -ho, w0), V(s1, -ho, y1, n1, -ho, w1), V(s1, ho, y1, n1, ho, w1), V(s0, ho, y0, n0, ho, w0));
        }
    }
    // Pier and abutment faces below the springing, facing into each arch.
    for (int i = 0; i < 3; ++i)
        for (float sg : { -1.0f, 1.0f }) {
            const float s = ac[i] + sg * R;
            const float yb = gnd(s) - 1.5f;
            if (yb >= spring) continue;
            const glm::vec3 nn = -D * sg;
            mesh_quad(m, V(s, -ho, yb, nn, -ho, -yb), V(s, ho, yb, nn, ho, -yb), V(s, ho, spring, nn, ho, -spring), V(s, -ho, spring, nn, -ho, -spring));
        }
    // The ends (what stands clear of the ground there).
    for (float s : { 0.0f, L }) {
        const glm::vec3 nn = (s == 0.0f) ? -D : D;
        const float yb = gnd(s) - 1.5f, yt = deck(s);
        mesh_quad(m, V(s, -ho, yb, nn, -ho, -yb), V(s, ho, yb, nn, ho, -yb), V(s, ho, yt, nn, ho, -yt), V(s, -ho, yt, nn, -ho, -yt));
        for (float sg : { -1.0f, 1.0f })
            mesh_quad(m, V(s, sg * hw, yt, nn, sg * hw, -yt), V(s, sg * ho, yt, nn, sg * ho, -yt),
                         V(s, sg * ho, yt + par_h, nn, sg * ho, -(yt + par_h)), V(s, sg * hw, yt + par_h, nn, sg * hw, -(yt + par_h)));
    }
    // ---- Dressed stone, as real geometry (so every edge is a row of separate, slightly
    // uneven blocks instead of one ruled line - relief mapping cannot cut a thin wall's
    // outline without holes): coping stones on the parapets, a pillar with a cap at each of
    // their ends, a projecting course at deck level, and wedge stones around the arches.
    {
        uint32_t rs = 0x6D2B79F5u;
        auto rnd = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
        // A block in bridge space (s along, u across, y up), its top edges chamfered by `ch`.
        // The bottom is left open (it stands on masonry).
        auto block = [&](float sa, float sb, float ua, float ub, float ya, float yb, float ch) {
            const float sm[2] = { sa, sb }, um[2] = { ua, ub };
            const float si[2] = { sa + ch, sb - ch }, ui[2] = { ua + ch, ub - ch };
            const float ys = yb - ch;
            // sides
            for (int k = 0; k < 2; ++k) {
                const glm::vec3 ns = D * (k ? 1.0f : -1.0f), nu = N * (k ? 1.0f : -1.0f);
                mesh_quad(m, V(sm[k], ua, ya, ns, ua, -ya), V(sm[k], ub, ya, ns, ub, -ya), V(sm[k], ub, ys, ns, ub, -ys), V(sm[k], ua, ys, ns, ua, -ys));
                mesh_quad(m, V(sa, um[k], ya, nu, sa, -ya), V(sb, um[k], ya, nu, sb, -ya), V(sb, um[k], ys, nu, sb, -ys), V(sa, um[k], ys, nu, sa, -ys));
                // chamfers
                const glm::vec3 cs = glm::normalize(ns + U), cu = glm::normalize(nu + U);
                mesh_quad(m, V(sm[k], ua, ys, cs, ua, -ys), V(sm[k], ub, ys, cs, ub, -ys), V(si[k], ui[1], yb, cs, ui[1], -yb - ch), V(si[k], ui[0], yb, cs, ui[0], -yb - ch));
                mesh_quad(m, V(sa, um[k], ys, cu, sa, -ys), V(sb, um[k], ys, cu, sb, -ys), V(si[1], ui[k], yb, cu, si[1], -yb - ch), V(si[0], ui[k], yb, cu, si[0], -yb - ch));
            }
            // top
            mesh_quad(m, V(si[0], ui[0], yb, U, si[0], ui[0]), V(si[1], ui[0], yb, U, si[1], ui[0]), V(si[1], ui[1], yb, U, si[1], ui[1]), V(si[0], ui[1], yb, U, si[0], ui[1]));
        };
        for (float sg : { -1.0f, 1.0f }) {
            // Coping: stones of uneven length and height, a finger's gap between them.
            for (float s = 0.55f; s < L - 0.55f;) {
                const float len = (std::min)(0.85f + 0.65f * rnd(), L - 0.55f - s);
                const float h = 0.17f + 0.07f * rnd(), sh = (rnd() - 0.5f) * 0.035f, ov = 0.06f + 0.03f * rnd();
                const float y = deck(s + 0.5f * len) + par_h;
                const float u0 = sg > 0.0f ? hw - ov + sh : -ho - ov + sh, u1 = sg > 0.0f ? ho + ov + sh : -hw + ov + sh;
                if (len > 0.25f) block(s + 0.008f, s + len - 0.008f, u0, u1, y - 0.01f, y + h, 0.035f);
                s += len;
            }
            // End pillars with a cap stone.
            for (int e = 0; e < 2; ++e) {
                const float s0 = e ? L - 0.50f : -0.32f, s1 = e ? L + 0.32f : 0.50f;
                const float u0 = sg > 0.0f ? hw - 0.13f : -ho - 0.13f, u1 = sg > 0.0f ? ho + 0.13f : -hw + 0.13f;
                const float yd = deck(e ? L : 0.0f);
                block(s0, s1, u0, u1, yd - 1.6f, yd + par_h + 0.28f, 0.03f);
                block(s0 - 0.07f, s1 + 0.07f, u0 - 0.07f, u1 + 0.07f, yd + par_h + 0.27f, yd + par_h + 0.47f, 0.09f);
            }
            // The course at deck level: a row of blocks standing 7-10 cm proud of the wall.
            for (float s = 0.6f; s < L - 0.6f;) {
                const float len = (std::min)(0.9f + 0.8f * rnd(), L - 0.6f - s);
                const float out = 0.07f + 0.035f * rnd(), y = deck(s + 0.5f * len);
                const float u0 = sg > 0.0f ? ho - 0.02f : -ho - out, u1 = sg > 0.0f ? ho + out : -ho + 0.02f;
                if (len > 0.3f) {
                    block(s + 0.006f, s + len - 0.006f, u0, u1, y - 0.30f, y - 0.06f - 0.03f * rnd(), 0.02f);
                    // (its underside is seen from the river)
                    mesh_quad(m, V(s, u0, y - 0.30f, -U, s, u0), V(s + len, u0, y - 0.30f, -U, s + len, u0),
                                 V(s + len, u1, y - 0.30f, -U, s + len, u1), V(s, u1, y - 0.30f, -U, s, u1));
                }
                s += len;
            }
            // Wedge stones around each arch, the keystone larger.
            for (int i = 0; i < 3; ++i) {
                const int nv = (std::max)(9, int(kPi * R / 0.62f) | 1);
                for (int k = 0; k < nv; ++k) {
                    const bool key = (k == nv / 2);
                    const float a0 = kPi * (float(k) + 0.02f) / float(nv), a1 = kPi * (float(k) + 0.98f) / float(nv);
                    const float r0 = R - 0.015f, r1 = R + (key ? 0.92f : 0.56f + 0.16f * rnd());
                    const float out = key ? 0.13f : 0.06f + 0.03f * rnd();
                    const float ua = sg > 0.0f ? ho - 0.02f : -ho - out, ub = sg > 0.0f ? ho + out : -ho + 0.02f;
                    const float uf = sg > 0.0f ? ub : ua;                    // the face that shows
                    auto Q = [&](float ang, float r, float u, glm::vec3 nn) {
                        const float s = ac[i] - std::cos(ang) * r, y = spring + std::sin(ang) * r;
                        return V(s, u, y, nn, ang * R, -r);
                    };
                    const glm::vec3 nf = N * sg;
                    mesh_quad(m, Q(a0, r0, uf, nf), Q(a1, r0, uf, nf), Q(a1, r1, uf, nf), Q(a0, r1, uf, nf));
                    const float am = 0.5f * (a0 + a1);
                    const glm::vec3 rad = glm::normalize(-D * std::cos(am) + U * std::sin(am));
                    const glm::vec3 tan0 = glm::normalize(-D * std::sin(a0) - U * std::cos(a0)), tan1 = glm::normalize(D * std::sin(a1) + U * std::cos(a1));
                    mesh_quad(m, Q(a0, r1, ua, rad), Q(a1, r1, ua, rad), Q(a1, r1, ub, rad), Q(a0, r1, ub, rad));       // outer rim
                    mesh_quad(m, Q(a0, r0, ua, -rad), Q(a1, r0, ua, -rad), Q(a1, r0, ub, -rad), Q(a0, r0, ub, -rad));   // lip into the arch
                    mesh_quad(m, Q(a0, r0, ua, tan0), Q(a0, r1, ua, tan0), Q(a0, r1, ub, tan0), Q(a0, r0, ub, tan0));
                    mesh_quad(m, Q(a1, r0, ua, tan1), Q(a1, r1, ua, tan1), Q(a1, r1, ub, tan1), Q(a1, r0, ub, tan1));
                }
            }
        }
    }
    // Colliders: the bridge runs at an angle, the boxes are axis-aligned - so its footprint is
    // tiled with 1 m columns. Deck and parapets are a 1 m slab; piers and abutments go to the ground.
    {
        const glm::vec2 c0 = a2 - n * ho, c1 = a2 + n * ho, c2 = b2 - n * ho, c3 = b2 + n * ho;
        const int x0 = int(std::floor((std::min)({ c0.x, c1.x, c2.x, c3.x }))), x1 = int(std::ceil((std::max)({ c0.x, c1.x, c2.x, c3.x })));
        const int z0 = int(std::floor((std::min)({ c0.y, c1.y, c2.y, c3.y }))), z1 = int(std::ceil((std::max)({ c0.y, c1.y, c2.y, c3.y })));
        for (int z = z0; z < z1; ++z)
            for (int x = x0; x < x1; ++x) {
                const glm::vec2 p(float(x) + 0.5f, float(z) + 0.5f);
                const float s = glm::dot(p - a2, d), u = glm::dot(p - a2, n);
                if (s < -0.5f || s > L + 0.5f || std::abs(u) > ho + 0.2f) continue;
                const float sc = std::clamp(s, 0.0f, L);
                const float top = deck(sc) + ((std::abs(u) > hw - 0.1f) ? par_h + 0.2f : 0.0f);
                const float bot = (arch_of(sc) >= 0) ? deck(sc) - 1.0f : (std::min)(ground(p.x, p.y) - 0.5f, deck(sc) - 1.0f);
                cols.push_back({ glm::vec3(float(x), bot, float(z)), glm::vec3(float(x + 1), top, float(z + 1)) });
            }
    }
    lv.meshes.push_back(std::move(m));
    lv.meshes.push_back(std::move(road));
    for (const auto& c : cols) lv.aabbs.push_back(c);
}

bool append_church(Level& lv, glm::vec3 base) {
    FILE* fp = std::fopen("assets/church/church.mesh", "r");
    if (!fp) return false;
    constexpr int kBrick = 1, kRoof = 13, kWood = 17, kMetal = 3;
    char name[32];
    int tris = 0;
    while (std::fscanf(fp, " mesh %31s %d", name, &tris) == 2) {
        LevelMesh m;
        float tile = 1.0f;
        const std::string nm(name);
        if (nm == "floor")        { m.tex_albedo = 5; m.tex_normal = 5; m.tint = glm::vec3(0.72f, 0.70f, 0.66f); m.fallback = glm::vec3(0.45f, 0.42f, 0.38f); tile = 2.0f; }
        else if (nm == "wax")     { m.tint = glm::vec3(0.86f, 0.80f, 0.62f); m.fallback = m.tint; }
        else if (nm == "flame")   { m.tint = glm::vec3(1.0f, 0.72f, 0.30f); m.fallback = m.tint; m.emissive = glm::vec3(4.0f, 2.2f, 0.7f); m.full_emissive = true; }
        // Leaded stained glass: self-lit, so it glows from inside and out.
        else if (nm == "stained") { m.tex_albedo = 21; m.tex_normal = 21; m.tint = glm::vec3(1.5f); m.fallback = glm::vec3(0.3f, 0.2f, 0.4f); m.full_emissive = true; tile = 1.0f; }
        else if (nm == "stone" || nm == "trim" || nm == "altar") {
            // Walls: shallower relief than the castle (they are half as thick).
            // Trim (steps, plinth, buttresses): small parts, nearly flat.
            m.relief = nm == "stone" ? 0.40f : 0.10f; m.tex_albedo = kBrick; m.tex_normal = kBrick; m.tint = glm::vec3(0.80f, 0.84f, 0.90f); m.fallback = glm::vec3(0.5f, 0.47f, 0.42f); tile = 1.0f / 0.33f; }   // smaller, greyer stones than the castle
        // The boarding under the tiles (seen where a tile is missing, and from below the eaves).
        else if (nm == "roof")  { m.tex_albedo = kWood;  m.tex_normal = kWood;  m.tint = glm::vec3(0.22f, 0.17f, 0.13f); m.fallback = glm::vec3(0.12f, 0.09f, 0.07f); tile = 0.55f; }
        // Roof tiles: individually modelled, in three tones of weathered clay (lichen from the texture).
        else if (nm == "tiles_a") { m.tex_albedo = 20; m.tex_normal = 20; m.tint = glm::vec3(1.25f, 0.62f, 0.48f); m.fallback = glm::vec3(0.45f, 0.20f, 0.15f); tile = 0.8f; }
        else if (nm == "tiles_b") { m.tex_albedo = 20; m.tex_normal = 20; m.tint = glm::vec3(0.95f, 0.50f, 0.42f); m.fallback = glm::vec3(0.35f, 0.17f, 0.13f); tile = 0.8f; }
        else if (nm == "tiles_c") { m.tex_albedo = 20; m.tex_normal = 20; m.tint = glm::vec3(1.05f, 0.80f, 0.55f); m.fallback = glm::vec3(0.40f, 0.28f, 0.18f); tile = 0.8f; }
        else if (nm == "wood")  { m.tex_albedo = kWood;  m.tex_normal = kWood;  m.tint = glm::vec3(0.80f, 0.66f, 0.50f); m.fallback = glm::vec3(0.35f, 0.22f, 0.12f); tile = 0.55f; }
        else if (nm == "iron")  { m.tex_albedo = kMetal; m.tex_normal = kMetal; m.tint = glm::vec3(0.16f, 0.16f, 0.17f); m.fallback = glm::vec3(0.08f, 0.08f, 0.09f); tile = 0.5f; }
        else                    { m.tint = glm::vec3(0.035f, 0.05f, 0.08f); m.fallback = m.tint; m.emissive = glm::vec3(0.030f, 0.022f, 0.010f); }
        for (int t = 0; t < tris; ++t) {
            LevelVertex v[3];
            for (auto& q : v) {
                if (std::fscanf(fp, "%f %f %f %f %f %f %f %f", &q.p.x, &q.p.y, &q.p.z, &q.n.x, &q.n.y, &q.n.z, &q.u, &q.v) != 8) { std::fclose(fp); return false; }
                q.p += base;
                q.u /= tile; q.v /= tile;
            }
            // Front faces: cross(b - a, c - a) must point along the normal.
            const glm::vec3 g = glm::cross(v[1].p - v[0].p, v[2].p - v[0].p);
            const bool flip = glm::dot(g, v[0].n + v[1].n + v[2].n) < 0.0f;
            const uint32_t b0 = uint32_t(m.verts.size());
            mesh_push(m, v[0]); mesh_push(m, v[1]); mesh_push(m, v[2]);
            if (flip) m.idx.insert(m.idx.end(), { b0, b0 + 2, b0 + 1 });
            else      m.idx.insert(m.idx.end(), { b0, b0 + 1, b0 + 2 });
        }
        if (!m.idx.empty()) lv.meshes.push_back(std::move(m));
    }
    std::fclose(fp);
    // Player colliders (the chapel is closed): nave, apse, door step.
    auto col = [&](glm::vec3 lo, glm::vec3 hi) { lv.aabbs.push_back({ base + lo, base + hi }); };
    // The chapel is open: floor, four walls (a gap for the door), furniture.
    col(glm::vec3(-4.62f, -1.2f, -3.12f), glm::vec3(4.62f, 0.45f, 3.12f));            // floor
    col(glm::vec3(-4.62f, 0.45f, 2.45f), glm::vec3(4.62f, 4.3f, 3.12f));              // side walls
    col(glm::vec3(-4.62f, 0.45f, -3.12f), glm::vec3(4.62f, 4.3f, -2.45f));
    col(glm::vec3(3.95f, 0.45f, -3.12f), glm::vec3(4.62f, 7.6f, 3.12f));              // east wall
    col(glm::vec3(-4.62f, 0.45f, 0.675f), glm::vec3(-3.95f, 7.6f, 3.12f));            // west wall, either side of the door
    col(glm::vec3(-4.62f, 0.45f, -3.12f), glm::vec3(-3.95f, 7.6f, -0.675f));
    col(glm::vec3(-4.62f, 3.0f, -0.675f), glm::vec3(-3.95f, 7.6f, 0.675f));           // over the door
    col(glm::vec3(-2.95f, 0.45f, 0.55f), glm::vec3(1.5f, 1.3f, 2.25f));               // pew blocks
    col(glm::vec3(-2.95f, 0.45f, -2.25f), glm::vec3(1.5f, 1.3f, -0.55f));
    col(glm::vec3(2.55f, 0.45f, -1.7f), glm::vec3(3.95f, 0.6f, 1.7f));                // altar step
    col(glm::vec3(2.98f, 0.45f, -0.85f), glm::vec3(3.82f, 1.57f, 0.85f));             // altar
    col(glm::vec3(4.4f, -1.2f, -2.3f), glm::vec3(6.6f, 4.0f, 2.3f));
    col(glm::vec3(6.6f, -1.2f, -1.3f), glm::vec3(7.2f, 4.0f, 1.3f));
    col(glm::vec3(-5.35f, -1.2f, -1.05f), glm::vec3(-4.6f, 0.23f, 1.05f));
    return true;
}

// ---------------------------------------------------------------------
// The graveyard: ten stone models from Blender, placed ~26 times in loose
// rows on the south side of the chapel. Each stands on the ground where it
// is, turned a little and leaning as old stones do; some have sunk. All of
// them go into one mesh (one draw) with the gravestone texture.
// ---------------------------------------------------------------------
int append_graveyard(Level& lv, glm::vec3 church, const std::function<float(float, float)>& ground) {
    FILE* fp = std::fopen("assets/church/graves.mesh", "r");
    if (!fp) return 0;
    struct Model { std::string name; std::vector<LevelVertex> v; };
    std::vector<Model> models;
    char name[32]; int tris = 0;
    while (std::fscanf(fp, " mesh %31s %d", name, &tris) == 2) {
        Model m; m.name = name;
        m.v.resize(size_t(tris) * 3);
        for (auto& q : m.v)
            if (std::fscanf(fp, "%f %f %f %f %f %f %f %f", &q.p.x, &q.p.y, &q.p.z, &q.n.x, &q.n.y, &q.n.z, &q.u, &q.v) != 8) { std::fclose(fp); return 0; }
        models.push_back(std::move(m));
    }
    std::fclose(fp);
    if (models.empty()) return 0;
    auto find = [&](const char* n) { for (size_t i = 0; i < models.size(); ++i) if (models[i].name == n) return int(i); return 0; };
    const int i_stump = find("stump"), i_fallen = find("fallen"), i_ledger = find("ledger");

    LevelMesh mesh;
    mesh.tex_albedo = 20; mesh.tex_normal = 20;            // engine kTexGrave
    mesh.tint = glm::vec3(0.86f, 0.93f, 1.0f);              // cooler: grey stone, the lichen stays warm
    mesh.fallback = glm::vec3(0.42f, 0.40f, 0.36f);
    const float tile = 1.1f;                                // metres per texture repeat
    uint32_t rs = 0x6A4E57u;
    auto rnd = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return float(rs >> 8) / 16777216.0f; };
    int placed = 0;
    auto place = [&](int mi, float x, float z, float yaw, float lean, float lean_dir, float sink, float scale) {
        const Model& m = models[size_t(mi)];
        const float gy = ground(x, z) - sink;
        // yaw about y, then lean by `lean` about a horizontal axis.
        const float cy = std::cos(yaw), sy = std::sin(yaw);
        const glm::vec3 ax(std::cos(lean_dir), 0.0f, std::sin(lean_dir));
        const float cl = std::cos(lean), sl = std::sin(lean);
        auto xf = [&](glm::vec3 p) {
            p = glm::vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
            return p * cl + glm::cross(ax, p) * sl + ax * (glm::dot(ax, p) * (1.0f - cl));     // Rodrigues
        };
        const float us = rnd() * 3.0f, vs = rnd() * 3.0f;      // each stone its own patch of the texture
        glm::vec3 lo(1e9f), hi(-1e9f);
        for (size_t t = 0; t + 2 < m.v.size(); t += 3) {
            LevelVertex v[3];
            for (int k = 0; k < 3; ++k) {
                v[k] = m.v[t + size_t(k)];
                v[k].p = xf(v[k].p * scale) + glm::vec3(x, gy, z);
                v[k].n = xf(v[k].n);
                v[k].u = v[k].u * scale / tile + us; v[k].v = v[k].v * scale / tile + vs;
                lo = glm::min(lo, v[k].p); hi = glm::max(hi, v[k].p);
            }
            const glm::vec3 g = glm::cross(v[1].p - v[0].p, v[2].p - v[0].p);
            const bool flip = glm::dot(g, v[0].n + v[1].n + v[2].n) < 0.0f;
            const uint32_t b0 = uint32_t(mesh.verts.size());
            mesh_push(mesh, v[0]); mesh_push(mesh, v[1]); mesh_push(mesh, v[2]);
            if (flip) mesh.idx.insert(mesh.idx.end(), { b0, b0 + 2, b0 + 1 });
            else      mesh.idx.insert(mesh.idx.end(), { b0, b0 + 1, b0 + 2 });
        }
        // Collider: the part above the ground, a little slimmer than the stone.
        lo.y = gy - 0.2f;
        const glm::vec3 c = (lo + hi) * 0.5f, e = (hi - lo) * 0.5f * glm::vec3(0.8f, 1.0f, 0.8f);
        if (hi.y - gy > 0.3f) lv.aabbs.push_back({ c - e, c + e });
        ++placed;
    };
    // Four rows south of the chapel (its long axis runs along x; the stones
    // face west, toward the path, as the chapel's door does).
    const float face = -1.5708f;                  // carved face toward -x
    for (int row = 0; row < 4; ++row) {
        const float z = church.z + 6.3f + float(row) * 2.5f + (rnd() - 0.5f) * 0.3f;
        for (int col = 0; col < 7; ++col) {
            if (rnd() < 0.14f) continue;                           // gaps: graves without a stone left
            const float x = church.x - 5.6f + float(col) * 1.95f + (rnd() - 0.5f) * 0.5f;
            int mi = int(rnd() * float(models.size())) % int(models.size());
            if (mi == i_fallen) mi = i_stump;                      // the fallen top only goes with its stump
            const float yaw = face + (rnd() - 0.5f) * 0.35f;
            const float lean = mi == i_ledger ? (rnd() - 0.5f) * 0.05f : (rnd() * rnd()) * 0.30f;
            const float sink = mi == i_ledger ? 0.02f : rnd() * rnd() * 0.10f;
            const float sc = 1.0f + 0.3f * rnd();
            place(mi, x, z, yaw, lean, rnd() * 6.2832f, sink, sc);
            if (mi == i_stump)                                     // its broken-off top lies in front of it
                place(i_fallen, x - 0.75f * sc + (rnd() - 0.5f) * 0.2f, z + (rnd() - 0.5f) * 0.3f,
                      yaw + (rnd() - 0.5f) * 0.9f, (rnd() - 0.5f) * 0.12f, rnd() * 6.2832f, 0.03f, sc);
        }
    }
    if (!mesh.idx.empty()) lv.meshes.push_back(std::move(mesh));
    return placed;
}

// ---------------------------------------------------------------------
// Road lamp post: a hewn timber post with a braced arm and a hanging iron
// lantern (world space; `base` on the ground, the arm reaches along
// `arm_dir`, a horizontal unit vector, toward the road).
// ---------------------------------------------------------------------
namespace {

// Tapered prism along a -> b with `segs` sides (4 = square beam, more =
// round). UVs: u around, v along (texture repeats, tile_m metres each).
void lp_beam(LevelMesh& m, glm::vec3 a, glm::vec3 b, float r0, float r1, int segs,
             glm::vec3 ref, float tile_m, bool caps, float twist = 0.0f) {
    const glm::vec3 ax = glm::normalize(b - a);
    glm::vec3 u = glm::normalize(ref - ax * glm::dot(ref, ax));
    const glm::vec3 w = glm::cross(ax, u);
    const float len = glm::length(b - a);
    const float k = segs == 4 ? 1.41421356f : 1.0f;      // square: r is the half side
    auto dir = [&](int i, float t) {
        const float ang = 2.0f * kPi * (float(i) + (segs == 4 ? 0.5f : 0.0f)) / float(segs) + twist * t;
        return u * std::cos(ang) + w * std::sin(ang);
    };
    for (int i = 0; i < segs; ++i) {
        const glm::vec3 d0a = dir(i, 0.0f), d1a = dir(i + 1, 0.0f), d0b = dir(i, 1.0f), d1b = dir(i + 1, 1.0f);
        const glm::vec3 n = glm::normalize(d0a + d1a);
        const glm::vec3 n0 = segs <= 6 ? n : d0a, n1 = segs <= 6 ? n : d1a;      // flat faces on beams
        const float u0 = float(i) * 2.0f * r0 * k / tile_m, u1 = u0 + 2.0f * r0 * k / tile_m;
        mesh_quad(m, { a + d0a * (r0 * k), n0, u0, 0.0f }, { a + d1a * (r0 * k), n1, u1, 0.0f },
                     { b + d1b * (r1 * k), n1, u1, -len / tile_m }, { b + d0b * (r1 * k), n0, u0, -len / tile_m });
    }
    if (caps)
        for (int i = 1; i + 1 < segs; ++i) {
            mesh_quad(m, { b + dir(0, 1.0f) * (r1 * k), ax, 0.0f, 0.0f }, { b + dir(i, 1.0f) * (r1 * k), ax, 0.1f, 0.0f },
                         { b + dir(i + 1, 1.0f) * (r1 * k), ax, 0.1f, 0.1f }, { b + dir(i + 1, 1.0f) * (r1 * k), ax, 0.1f, 0.1f });
            mesh_quad(m, { a + dir(0, 0.0f) * (r0 * k), -ax, 0.0f, 0.0f }, { a + dir(i, 0.0f) * (r0 * k), -ax, 0.1f, 0.0f },
                         { a + dir(i + 1, 0.0f) * (r0 * k), -ax, 0.1f, 0.1f }, { a + dir(i + 1, 0.0f) * (r0 * k), -ax, 0.1f, 0.1f });
        }
}

}  // namespace

glm::vec3 append_lamp_post(Level& lv, glm::vec3 base, glm::vec3 arm_dir, uint32_t seed) {
    constexpr int kWood = 17, kMetal = 3;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return float((seed >> 8) & 0xFFFFu) / 65535.0f; };
    LevelMesh wood; wood.tex_albedo = kWood; wood.tex_normal = kWood;
    wood.tint = glm::vec3(0.95f, 0.90f, 0.84f); wood.fallback = glm::vec3(0.30f, 0.21f, 0.13f);
    LevelMesh iron; iron.tex_albedo = kMetal; iron.tex_normal = kMetal;
    iron.tint = glm::vec3(0.16f, 0.15f, 0.14f); iron.fallback = glm::vec3(0.09f, 0.09f, 0.09f);
    LevelMesh glass;                                     // warm, self-lit panes + flame
    glass.tint = glass.fallback = glm::vec3(1.0f, 0.74f, 0.36f);
    glass.emissive = glm::vec3(3.2f, 2.0f, 0.75f);
    glass.full_emissive = true;

    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const glm::vec3 A = glm::normalize(glm::vec3(arm_dir.x, 0.0f, arm_dir.z));
    const glm::vec3 S(-A.z, 0.0f, A.x);
    const float H = 3.55f + 0.25f * rnd();
    const glm::vec3 lean = (A * (rnd() - 0.5f) + S * (rnd() - 0.5f)) * 0.10f;   // posts are never plumb
    const glm::vec3 foot = base - up * 0.7f;
    const glm::vec3 top = base + up * H + lean;
    const float wt = 0.55f;                              // wood texture: metres per repeat

    // Post: hewn octagon, slightly tapered and twisted, with a chamfered top.
    lp_beam(wood, foot, top - up * 0.12f, 0.105f, 0.085f, 8, A, wt, false, 0.25f);
    lp_beam(wood, top - up * 0.12f, top, 0.085f, 0.05f, 8, A, wt, true, 0.0f);
    // Stone-packed footing collar.
    lp_beam(wood, base - up * 0.05f, base + up * 0.16f, 0.15f, 0.125f, 8, A, wt, true);
    // Arm (square beam) through the post, and a diagonal brace under it.
    const glm::vec3 arm_c = base + up * (H - 0.42f) + lean * 0.88f;
    const glm::vec3 arm_a = arm_c - A * 0.22f, arm_b = arm_c + A * 0.98f + up * 0.03f;
    lp_beam(wood, arm_a, arm_b, 0.052f, 0.045f, 4, up, wt, true);
    lp_beam(wood, base + up * (H - 1.18f) + lean * 0.67f + A * 0.07f, arm_c + A * 0.66f - up * 0.04f,
            0.036f, 0.036f, 4, S, wt, true);
    // Iron: straps round the joint, pegs, and the hook plate at the arm end.
    lp_beam(iron, arm_c - up * 0.075f, arm_c + up * 0.075f, 0.097f, 0.097f, 8, A, 0.3f, false);
    lp_beam(iron, arm_c + A * 0.62f - up * 0.062f, arm_c + A * 0.70f - up * 0.062f, 0.058f, 0.058f, 4, up, 0.3f, true);
    const glm::vec3 hook = arm_b - A * 0.10f;
    lp_beam(iron, hook - up * 0.06f - A * 0.03f, hook - up * 0.06f + A * 0.03f, 0.056f, 0.056f, 4, up, 0.3f, true);
    // Chain: four alternating links.
    glm::vec3 c = hook - up * 0.055f;
    for (int i = 0; i < 4; ++i) {
        const glm::vec3 side = (i & 1) ? A : S;
        lp_beam(iron, c - side * 0.012f, c - up * 0.055f - side * 0.012f, 0.0045f, 0.0045f, 4, side, 0.2f, false);
        lp_beam(iron, c + side * 0.012f, c - up * 0.055f + side * 0.012f, 0.0045f, 0.0045f, 4, side, 0.2f, false);
        c -= up * 0.05f;
    }
    // Lantern: ring, pyramid roof with a finial, four corner bars, base
    // tray, glass panes and a flame.
    const float lw = 0.105f, lh = 0.27f;                 // half width, body height
    const glm::vec3 lt = c - up * 0.10f;                 // top of the body
    const glm::vec3 lb = lt - up * lh;
    lp_beam(iron, c, lt + up * 0.12f, 0.008f, 0.012f, 6, A, 0.2f, false);
    lp_beam(iron, lt + up * 0.11f, lt + up * 0.135f, 0.022f, 0.016f, 6, A, 0.2f, true);
    lp_beam(iron, lt, lt + up * 0.115f, lw * 1.28f, 0.02f, 4, A, 0.3f, true);             // roof
    lp_beam(iron, lt - up * 0.018f, lt, lw * 1.12f, lw * 1.28f, 4, A, 0.3f, true);        // eave
    lp_beam(iron, lb - up * 0.03f, lb, lw * 0.80f, lw * 1.10f, 4, A, 0.3f, true);         // tray
    lp_beam(iron, lb - up * 0.05f, lb - up * 0.03f, 0.02f, lw * 0.80f, 4, A, 0.3f, true);
    for (int k = 0; k < 4; ++k) {
        const glm::vec3 d = (A * ((k & 1) ? 1.0f : -1.0f) + S * ((k & 2) ? 1.0f : -1.0f)) * lw;
        lp_beam(iron, lb + d, lt + d, 0.0085f, 0.0085f, 4, A, 0.2f, false);
        // mid rail on each side
        const glm::vec3 e = (k & 1) ? A : S, f = (k & 1) ? S : A;
        const glm::vec3 mc = lb + up * (lh * 0.5f) + e * (((k & 2) ? 1.0f : -1.0f) * lw);
        lp_beam(iron, mc - f * lw, mc + f * lw, 0.005f, 0.005f, 4, up, 0.2f, false);
    }
    lp_beam(glass, lb + up * 0.004f, lt - up * 0.004f, lw * 0.93f, lw * 0.93f, 4, A, 1.0f, false);   // panes
    lp_beam(glass, lb + up * 0.02f, lb + up * 0.13f, 0.022f, 0.004f, 6, A, 1.0f, true);              // flame

    // All posts share three meshes (wood / iron / glass): 3 draws and 3
    // BLAS geometries for the whole road instead of 3 per post.
    auto merge = [&](LevelMesh&& m, auto same) {
        for (LevelMesh& e : lv.meshes) {
            if (!same(e)) continue;
            const uint32_t off = uint32_t(e.verts.size());
            e.verts.insert(e.verts.end(), m.verts.begin(), m.verts.end());
            for (uint32_t i : m.idx) e.idx.push_back(i + off);
            e.bmin = glm::min(e.bmin, m.bmin); e.bmax = glm::max(e.bmax, m.bmax);
            return;
        }
        lv.meshes.push_back(std::move(m));
    };
    merge(std::move(wood),  [](const LevelMesh& e) { return e.tex_albedo == kWood && !e.full_emissive; });
    merge(std::move(iron),  [](const LevelMesh& e) { return e.tex_albedo == kMetal && e.tint == glm::vec3(0.16f, 0.15f, 0.14f); });
    merge(std::move(glass), [](const LevelMesh& e) { return e.full_emissive && e.emissive == glm::vec3(3.2f, 2.0f, 0.75f); });
    lv.aabbs.push_back({ base - glm::vec3(0.13f, 0.3f, 0.13f), base + glm::vec3(0.13f, H, 0.13f) });
    return lb + up * (lh * 0.5f);
}

} // namespace qlike::game

