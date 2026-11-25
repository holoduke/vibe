// Wind and trampling for trees and bushes - the reference game's foliage vertex shader (its
// foliage shader branch, frond and leaf vertex shaders, read from the compiled shaders: one body for all
// three, the same maths as its grass; this is our own code for it, shaders/tuft_cd.glsl is the
// grass's copy). In the game's order:
//   1. the wind here    the global wind, turned and strengthened by the wind field round the
//                       camera within 64 m (wind_fluid.cpp)
//   2. its fade         with distance, later for tall plants (a 10 m tree: to ~230 m)
//   3. the whole plant  the upper half leans a few millimetres - the trunk hardly moves
//   4. the branches     what is seen moving: each vertex along its branch's direction by its
//                       branch weight, a slow double swing with the branch's own phase
//   5. the leaves       a ripple along the leaf's normal, and the "tumble": the leaf card as
//                       a whole turning about the point it grows from
//   6. trampling        a low plant leans away from who walks through it (the trail picture,
//                       grass_trail.cpp); a tree does not - the lean is divided by
//                       (height - 1.5)^1.8
// The game's meshes carry the branch weight, phase and direction and the leaf's pivot in every
// vertex (authored). Ours: the weight from the vertex's place in the tree (veg.vert windW),
// the phase from the place too, the direction the wind's with a little rise and fall, the
// pivot the point the card grows from (the mesh has it).
//
// Needs from the including shader: scene, u_wind, u_trail, u_trail_h.

const float kVegWindN    = 48.0;      // the wind field: cells a side ...
const float kVegWindCell = 2.0;       // ... and metres a cell (wind_fluid.cpp)
const float kVegBranchW  = 0.06;      // branch weight of the crown's outside, per metre of the plant's height
const float kVegLeafW    = 0.5;       // the tumble's strength
const float kVegLeafK    = 0.9424779; // the lean when walked through: 0.3 pi

#include "wind_gust.glsl"

float vw_tri(float x)  { return abs(fract(x) * 2.0 - 1.0); }
float vw_sm(float x)   { return x * x * (3.0 - 2.0 * x) - 0.5; }            // -0.5 .. 0.5
float vw_s2(float x)   { return vw_sm(x) * 2.0; }
float vw_acos(float c) { return 1.570796 + (-0.8726646 - 0.6981317 * c * c) * c; }
vec3  vw_qrot(vec3 v, vec4 q) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
vec4  vw_qmul(vec4 a, vec4 b) { return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz)); }

// wp: the vertex at rest (world). base: the plant's foot. H: its height (m). w01: the vertex's
// wind weight 0 (trunk) .. 1 (the crown's outside). leaf: a leaf card, growing from `pivot`
// (world). phase: the plant's own. t: time. trail: the trail window (xy corner, z 1 / side,
// w on). n: the normal, turned with the leaf.
vec3 veg_wind(vec3 wp, vec3 base, float H, float w01, bool leaf, vec3 pivot, float phase, float t, vec4 trail,
              inout vec3 n) {
    float dist = distance(base, scene.camera_pos.xyz);
    vec3  rel  = wp - base;
    // 1. The wind here.
    vec3  Wd = normalize(vec3(0.871, 0.0, 0.490));
    float S  = clamp(scene.grass_rm_params.y * 12.5, 0.0, 1.0);
    S = min(S * wind_gust(base.xz, Wd.xz, t), 1.0);                       // (gust fronts travelling down the wind: wind_gust.glsl)
    if (dist < 64.0 + H) {
        vec3  centre = (floor(scene.camera_pos.xyz / kVegWindCell) + 0.5) * kVegWindCell;
        vec3  uvw = ((wp - centre) / kVegWindCell + 0.5 * kVegWindN) / kVegWindN;
        if (all(greaterThan(uvw, vec3(0.0))) && all(lessThan(uvw, vec3(1.0)))) {
            vec3  v  = textureLod(u_wind, uvw, 0.0).xyz;
            float lv = length(v);
            if (lv >= 0.001) {
                float f = clamp(lv, 0.0, 1.0);
                Wd = normalize(mix(Wd, v / lv, f) + vec3(1e-5, 0.0, 0.0));
                S  = mix(S, 1.0, f);
            }
        }
    }
    // 2. Its fade.
    float rH    = clamp(1.0 / max(H, 0.05), 0.0, 1.0);
    float wfade = 1.0 - clamp(rH * 12.0 - 15.0 + dist / 16.0, 0.0, 1.0);
    if (dist < 256.0 - 192.0 * rH && S > 0.0 && wfade > 0.0) {
        // 3. the whole plant
        float osc = vw_s2(vw_tri(t * 0.8 + base.y + 0.5)); osc = osc * osc + vw_s2(vw_tri(t + 0.5 + base.x));
        vec3  off = vec3(Wd.x, 0.0, Wd.z) * ((osc * S + S) * clamp(max(rel.y - H * 0.5, 0.0), 0.0, 1.0) * 0.005);
        // (ours: the lean under a gust, as for the grass)
        off += vec3(Wd.x, -0.15, Wd.z) * (kVegBranchW * H * w01 * w01 * 0.6 * S);
        // 4. its branch
        float bph = phase + dot(rel, vec3(0.9, 0.6, 1.3)) * 0.35;
        float T   = t * 0.5 + base.x + bph;
        float A   = vw_sm(vw_tri(T + 0.5)), B = vw_sm(vw_tri(T * 0.5 + 0.5));
        float o   = 4.0 * A * B, sg = o < 0.0 ? -1.0 : 1.0, ab = abs(o);
        vec3  bd  = normalize(Wd + vec3(0.0, 0.35 * sin(bph * 1.7), 0.0));
        off += bd * (kVegBranchW * H * w01 * w01 * 0.2) * ((mix(mix(o, sg, ab), sg, ab) + 2.0 * A) * S);
        if (leaf) {
            // 5. the leaf: the ripple of its face, and the tumble about where it grows from
            off += n * (S * 0.03 * vw_sm(vw_tri(dot(rel, vec3(10.0)) + t + 0.5)));
            vec3  P   = pivot - base;
            float ph  = dot(P, vec3(10.0)) + base.x;
            vec3  hx  = fract(sin(vec3(dot(P, vec3(12.9898, 78.233, 37.719)), dot(P, vec3(39.346, 11.135, 83.155)),
                                       dot(P, vec3(73.156, 52.235, 9.151)))) * 43758.5453) - 0.5;
            vec3  axl = normalize(hx + vec3(0.0, 0.6, 0.0));
            float A2 = vw_s2(vw_tri(ph + t + 0.5)), B2 = vw_s2(vw_tri(t * 0.75 + 0.5 - ph));
            float a1 = kVegLeafW * 0.25 * S * (B2 * B2 + A2);
            float dw = clamp(dot(Wd, axl), -1.0, 1.0);
            vec3  ax2 = normalize(cross(axl, Wd) + vec3(0.0, dw, 0.0));
            float fl = vw_s2(vw_tri(ph + 0.5 + (ph + t) * 0.87));
            fl = clamp((fl * fl * vw_s2(vw_tri(ph * ph + 0.5 + ph + t)) + 1.0) * 0.5, 0.0, 1.0);
            float a2 = kVegLeafW * 0.1 * (fl + (vw_acos(dw) - A2 * A2 + B2) * S) * S;
            vec4  q  = vw_qmul(vec4(ax2 * sin(a2), cos(a2)), vec4(axl * sin(a1), cos(a1)));
            vec3  d  = rel - P;
            vec3  r  = vw_qrot(d, q);
            float lr = length(r);
            if (lr > 1e-6) off += (P - rel) + r * (length(d) / lr);
            n = vw_qrot(n, q);
        }
        rel += off * wfade;
    }
    // 6. What walked through it.
    if (trail.w > 0.5 && dist < 48.0 && H < 8.0) {
        vec2 tuv = (base.xz - trail.xy) * trail.z;
        if (all(greaterThan(tuv, vec2(0.0))) && all(lessThan(tuv, vec2(1.0)))) {
            vec2  hh  = textureLod(u_trail_h, tuv, 0.0).xy;
            vec4  s   = textureLod(u_trail, tuv, 0.0);
            float amt = clamp(s.w, 0.0, 1.0);
            bool  hit = hh.x - 1.0 < base.y && hh.y + 0.2 > base.y;
            if (hit && amt > 0.0 && dot(s.xy, s.xy) > 1e-4) {
                vec3  D   = normalize(vec3(s.x, 0.0, s.y));
                vec3  ax  = normalize(cross(D, vec3(0.0, 1.0, 0.0)));
                float ang = amt * kVegLeafK * mix(sin(t * 7.0), 1.0, amt)
                          * clamp(abs(rel.y) * 0.25 - H * 0.03, 0.0, 1.0) / pow(max(H - 1.5, 1.0), 1.8);
                vec3  c = ax * dot(ax, rel);
                vec3  d = rel - c;
                rel = c + d * cos(ang) + cross(d, ax) * sin(ang);
            }
        }
    }
    return base + rel;
}
