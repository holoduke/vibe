// Grass - long tufts and turf: the vertex shader of the reference game's placed foliage (its
// foliage shader VSMain8Leaf, the permutation that reads the placement's records, global wind),
// read from the compiled shaders. This is our own code for it; formulas and constants are the
// game's, step for step and in its order:
//
//   1. the wind here                   the global wind, and within 64 m of the camera the wind
//                                      field (wind_fluid.cpp): where the air moves, the wind
//                                      turns that way and grows up to full strength
//   2. the wind's fade with distance   full to 48 m, none from 64 m (plants up to a metre)
//   3. wind, in the clump's own space  a sway of the upper half of the whole clump (1.5 cm per
//                                      unit of strength at most); a term per blade ("branch":
//                                      two triangle waves an octave apart pushed toward their
//                                      own sign, along the blade's own direction); the ripple
//                                      of the blade's face; and the "tumble": the blade turns
//                                      as a whole about its root - a twist about its own axis
//                                      and a lean down the wind, with a flutter of its own
//   4. into the world                  on the ground's normal, turned about it, scaled
//   5. every blade on the ground       within 64 m the blade is set on the ground under its
//                                      own root (so a clump follows the ground it spans), and
//                                      a blade of turf whose root has no grass under it is cut
//   6. what walked through it          the trail (grass_trail.cpp), read at the blade's root:
//                                      if the body that pushed stood at the root's height, the
//                                      blade turns about its root, away from the push, by up to
//                                      0.3 pi (54 degrees), wobbling as it comes back up
//
// What the game's meshes carry in every vertex and ours have no room for is worked out here
// from the blade's number (tuft_shape): the blade's root (the pivot), its direction, its
// "branch" weight and phase, its tumble axis. The numbers the game keeps per mesh are the
// kTuft* constants and pc.tex_params (ScatterSet::kTuftShape / kTurfShape: the mesh's height at
// size 1, blades, root radius, root depth).
//
// The hand-placed tufts (painted grass, river banks: plant instances from scatter.cpp) are the
// game's other kind of instance - the one with a seed: they also get a lean of their own
// (step 3b; the placed ones have none in the game either).
//
// Needs from the including shader: scene (camera_pos), pc.color (xy = wind direction, world x
// and z; z = wind strength 0..1), pc.emissive (the trail window), pc.grass_params.z (time),
// pc.tex_params, u_heightmap_full, u_grass_mask, u_trail, u_trail_h, u_wind.

const float kTuftBranchW = 0.67;      // wind weight of a blade's tip, per metre of the mesh's height (the root's is 0)
const float kTuftLeafW   = 1.0;       // the tumble's strength (the game: per vertex)
const float kTuftLeafK   = 0.9424779; // the turn when walked through: 0.3 pi
const float kTuftLean0   = 0.12;      // hand-placed tufts' lean: strength ...
const float kTuftLean1   = 0.05;      // ... the height it starts at (of the reference height) ...
const float kTuftLean23  = 2.2;       // ... how fast it grows with height ...
const float kTuftLean45  = 0.7;       // ... and how it levels off
const float kWindN       = 48.0;      // the wind field: cells a side ...
const float kWindCell    = 2.0;       // ... and metres a cell (wind_fluid.cpp)

#include "wind_gust.glsl"

float tw_tri(float x)  { return abs(fract(x) * 2.0 - 1.0); }
float tw_sm(float x)   { return x * x * (3.0 - 2.0 * x) - 0.5; }            // -0.5 .. 0.5
float tw_s2(float x)   { return tw_sm(x) * 2.0; }
float tw_acos(float c) { return 1.570796 + (-0.8726646 - 0.6981317 * c * c) * c; }
vec3  tw_rot(vec3 v, vec3 k, float a) {                                    // about unit axis k
    float c = cos(a), s = sin(a);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.0 - c));
}
vec3  tw_qrot(vec3 v, vec4 q) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
vec4  tw_qmul(vec4 a, vec4 b) { return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz)); }

float tuft_ground(vec2 xz) {
    float W = float(textureSize(u_heightmap_full, 0).x);
    return textureLod(u_heightmap_full, ((xz / 2048.0 + 0.5) * (W - 1.0) + 0.5) / W, 0.0).r;
}

// p: the mesh's vertex (y up, metres at size 1), n: its normal - both changed to the world's.
// base / size / yaw: the instance. sway01: 0 at a blade's root .. 1 at its tip.
// lean: a hand-placed tuft (it has a lean of its own). lawn: turf (cut where the lawn's mask
// has no grass). Returns the vertex in the world; kill = the blade is not there.
vec3 tuft_cd(vec3 p, inout vec3 n, vec3 base, float size, float yaw, float sway01, float blade_id, float dist,
             bool lean, bool lawn, out bool kill) {
    kill = false;
    float t  = pc.grass_params.z;
    float H0 = max(pc.tex_params.x, 0.05);
    float H  = max(H0 * size, 0.01);
    // The blade (scatter_plant_meshes.inl tuft_root: keep in step).
    float ba   = 2.39996323 * blade_id;
    vec3  bout = vec3(cos(ba), 0.0, sin(ba));                               // outward from the clump's middle
    vec3  P    = bout * (pc.tex_params.z * sqrt((blade_id + 0.5) / max(pc.tex_params.y, 1.0))) - vec3(0.0, pc.tex_params.w, 0.0);
    float wB   = kTuftBranchW * H0 * sway01 * sway01;                       // the game's windBranch.x
    float bB   = fract(sin(blade_id * 12.9898 + 4.1) * 43758.5453) * 16.0;  // ... and .y: its phase
    vec3  axl  = normalize(vec3(bout.x * 0.3, 1.0, bout.z * 0.3));          // the blade's own axis (the tumble's)

    // The instance's basis: the ground's normal (the home tile's height map; level elsewhere),
    // turned about it - built as the game's placement builds it.
    vec3  Y  = vec3(0.0, 1.0, 0.0);
    float g0 = base.y;
    bool  on_map = all(lessThan(abs(base.xz), vec2(1022.0)));
    if (on_map) {
        const float e = 2.0;
        Y = normalize(vec3(tuft_ground(base.xz - vec2(e, 0.0)) - tuft_ground(base.xz + vec2(e, 0.0)), 2.0 * e,
                           tuft_ground(base.xz - vec2(0.0, e)) - tuft_ground(base.xz + vec2(0.0, e))));
        g0 = tuft_ground(base.xz);
    }
    vec3 tt = normalize(vec3(0.0, -Y.z, Y.y));
    vec3 bb = cross(Y, tt);
    vec3 X  = bb * cos(yaw) + tt * sin(yaw);
    vec3 Z  = normalize(cross(X, Y));

    // 1. The wind here.
    vec3  Wd = normalize(vec3(pc.color.x, 0.0, pc.color.y) + vec3(1e-5, 0.0, 0.0));
    float S  = clamp(pc.color.z, 0.0, 1.0);
    S = min(S * wind_gust(base.xz, Wd.xz, t), 1.0);                       // (gust fronts travelling down the wind: wind_gust.glsl)
    if (dist < 64.0) {
        vec3  centre = (floor(scene.camera_pos.xyz / kWindCell) + 0.5) * kWindCell;
        vec3  v  = textureLod(u_wind, ((base - centre) / kWindCell + 0.5 * kWindN) / kWindN, 0.0).xyz;
        float lv = length(v);
        if (lv >= 0.001) {
            float f = clamp(lv, 0.0, 1.0);
            Wd = normalize(mix(Wd, v / lv, f) + vec3(1e-5, 0.0, 0.0));
            S  = mix(S, 1.0, f);
        }
    }
    // 2. Its fade.
    float rH    = clamp(1.0 / H, 0.0, 1.0);
    float wfade = 1.0 - clamp(rH * 12.0 - 15.0 + dist / 16.0, 0.0, 1.0);

    // 3. Wind.
    if (dist < 256.0 - 192.0 * rH && S > 0.0) {
        vec3  Wl = normalize(vec3(dot(Wd, X), dot(Wd, Y), dot(Wd, Z)));
        // the whole clump
        float osc = tw_s2(tw_tri(t * 0.8 + base.y + 0.5)); osc = osc * osc + tw_s2(tw_tri(t + 0.5 + base.x));
        vec3  off = vec3(Wl.x, 0.0, Wl.z) * ((osc * S + S) * clamp(max(p.y - H0 * 0.5, 0.0), 0.0, 1.0) * 0.005);
        if (dot(p, p) > 1e-8) {
            vec3  a = normalize(p), b = normalize(p + off);
            float ang = tw_acos(clamp(dot(a, b), -1.0, 1.0));
            vec3  ax = cross(b, a);
            if (ang > 0.01 && dot(ax, ax) > 1e-12) n = tw_rot(n, normalize(ax), ang * 3.0);
        }
        // (ours: the lean under a gust - the blade pressed down the wind as the front passes,
        // which is what shows the waves crossing a field)
        off += vec3(Wl.x, -0.22, Wl.z) * (sway01 * sway01 * H0 * 0.38 * S);
        // each blade
        float T  = t + base.x + bB;
        float A  = tw_sm(tw_tri(T + 0.5)), B = tw_sm(tw_tri(T * 0.5 + 0.5));
        float o  = 4.0 * A * B, sg = o < 0.0 ? -1.0 : 1.0, ab = abs(o);
        off += bout * (wB * 0.2) * ((mix(mix(o, sg, ab), sg, ab) + 2.0 * A) * S);
        // the ripple of the blade's face
        off += n * (S * 0.06 * tw_sm(tw_tri(dot(p, vec3(10.0)) + t + 0.5)));
        // the tumble: the blade as a whole, about its root
        {
            float ph = dot(P, vec3(10.0)) + base.x;
            float A2 = tw_s2(tw_tri(ph + t + 0.5)), B2 = tw_s2(tw_tri(t * 0.75 + 0.5 - ph));
            float a1 = kTuftLeafW * 0.25 * S * (B2 * B2 + A2);
            float dw = clamp(dot(Wl, axl), -1.0, 1.0);
            vec3  ax2 = normalize(cross(axl, Wl) + vec3(0.0, dw, 0.0));
            float fl = tw_s2(tw_tri(ph + 0.5 + (ph + t) * 0.87));
            fl = clamp((fl * fl * tw_s2(tw_tri(ph * ph + 0.5 + ph + t)) + 1.0) * 0.5, 0.0, 1.0);
            float a2 = kTuftLeafW * 0.1 * (fl + (tw_acos(dw) - A2 * A2 + B2) * S);
            vec4  q  = tw_qmul(vec4(ax2 * sin(a2), cos(a2)), vec4(axl * sin(a1), cos(a1)));
            vec3  d  = p - P;
            vec3  r  = tw_qrot(d, q);
            float lr = length(r);
            if (lr > 1e-6) off += (P - p) + r * (length(d) / lr);
            n = tw_qrot(n, q);
        }
        p += off * wfade;
    }

    // 3b. A hand-placed tuft's own lean (from its seed).
    if (lean) {
        float seed = floor(fract(sin(dot(base.xz, vec2(12.9898, 78.233))) * 43758.5453) * 65535.0);
        float r  = fract(seed * 9.430594e-06 + 1.118034);
        float h  = max(p.y - mix(0.01, kTuftLean1, r) * H0, 0.0);
        float dx = fract(seed * 1.151869e-05 + 1.254878) * 2.0 - 1.0;
        float dz = fract(seed * 8.695205e-06 + 1.06984) * 2.0 - 1.0;
        vec3  dn = normalize(vec3(dx, -1.0, dz));
        vec3  ax = normalize(vec3(-dn.z, 0.0, dn.x) + vec3(1e-5, 0.0, 0.0));
        float ang = r * 3.14159265 * max(kTuftLean0, 0.0) * sin(kTuftLean23 * h / (kTuftLean45 + 0.2 * h));
        p = tw_rot(p, ax, ang);
        n = tw_rot(n, ax, ang);
    }

    // 4. Into the world.
    vec3 rel  = (X * p.x + Y * p.y + Z * p.z) * size;
    vec3 relP = (X * P.x + Y * P.y + Z * P.z) * size;                       // the blade's root
    n = normalize(X * n.x + Y * n.y + Z * n.z);
    vec3 pivot = base + relP;

    // 5. Every blade on the ground under its own root.
    if (dist < 64.0 && on_map) {
        if (lawn && textureLod(u_grass_mask, pivot.xz / 2048.0 + 0.5, 0.0).r <= 0.02) { kill = true; return base; }
        float g = tuft_ground(pivot.xz);
        if (abs(g - g0) > 1.0) { kill = true; return base; }
        // (the root keeps its depth under the ground line; the clump's own offsets - the lift
        // onto the ground's relief - stay as they are)
        float dy = (g - g0) - (relP.y + pc.tex_params.w * size);
        rel.y += dy; relP.y += dy; pivot.y += dy;
    }

    // 6. What walked through it.
    if (pc.emissive.w > 0.5 && dist < 48.0) {
        vec2 tuv = (pivot.xz - pc.emissive.xy) * pc.emissive.z;
        if (all(greaterThan(tuv, vec2(0.0))) && all(lessThan(tuv, vec2(1.0)))) {
            vec2  hh  = textureLod(u_trail_h, tuv, 0.0).xy;
            vec4  s   = textureLod(u_trail, tuv, 0.0);
            float amt = clamp(s.w, 0.0, 1.0);
            bool  hit = hh.x - 1.0 < pivot.y && hh.y + 0.2 > pivot.y;
            if (hit && amt > 0.0 && dot(s.xy, s.xy) > 1e-4) {
                vec3 D  = normalize(vec3(s.x, 0.0, s.y));
                vec3 Aw = normalize(X * axl.x + Y * axl.y + Z * axl.z);
                vec3 ax = cross(D, Aw);
                if (dot(ax, ax) > 1e-6) {
                    ax = normalize(ax);
                    float ang = amt * kTuftLeafK * mix(sin(t * 7.0), 1.0, amt);
                    vec3  c = relP + ax * dot(ax, rel - relP);
                    vec3  d = rel - c;
                    vec3  turned = c + d * cos(ang) + cross(d, ax) * sin(ang);
                    rel += (turned - rel) * wfade;
                    n = tw_rot(n, ax, -ang * wfade);
                }
            }
        }
    }
    return base + rel;
}
