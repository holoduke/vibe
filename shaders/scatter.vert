#version 460
#extension GL_GOOGLE_include_directive : require

// Ground clutter (AAA Phase 3): instanced lumpy-rock mesh. Same
// instancing + shadow-sampling pattern as grass.vert; distance fade
// collapses far instances so the rasteriser drops them.

layout(location = 0) in vec3 inPos;        // rock-local, unit-ish radius
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(location = 3) in vec4 inPosScale;   // .xyz base world, .w scale
layout(location = 4) in vec4 inRotTint;    // .x rotY, .yzw tint

#include "scene_ubo.glsl"
#include "rock_sdf.glsl"

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;          // plants: xy = wind direction (world x, z), z = wind strength 0..1
    vec4 emissive;       // plants: the trail window - xy = world x, z of its corner, z = 1 / its side (m), w = 1 when there is one
    vec4 tex_params;
    vec4 grass_params;   // .x = max draw distance (m)
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec3 vWorldPos;
layout(location = 3) out float vSunShadow;
layout(location = 4) out float vCullKill;
layout(location = 5) out float vDistToCam;
layout(location = 6) out vec4 vPrevClip;
layout(location = 7) out vec2 vUv;
layout(location = 8) out vec3 vObj;            // rock: object-space hull point
layout(location = 9) flat out vec4 vInst;      // rock: base xyz, scale
layout(location = 10) flat out vec2 vRotSeed;  // rock: rotY, shape seed
layout(location = 11) out vec3 vStoneN;        // stone: object-space surface normal

layout(set = 0, binding = 6) uniform sampler2D u_terrain_shadow;
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;
layout(set = 0, binding = 17) uniform sampler2D u_heightmap_full;      // the home tile's heights (m), 2048 m square
layout(set = 0, binding = 41) uniform sampler2D u_trail;               // where people and cars have just been (grass_trail.cpp)

// ---- Grass tufts: the foliage model of Crimson Desert (its tree.hlsl vertex shaders, read from
// the compiled shaders; this is our own code for it). A tuft is an instanced clump of blades:
//   * it stands on the ground's own normal, not bolt upright on a slope,
//   * every tuft leans a little its own way (the same every frame),
//   * the wind is the game's two terms: a slow sway of the upper half of the whole clump and
//     a livelier one per blade, weighted from root (0) to tip (1), each blade with its own
//     direction and phase; both built from smoothed triangle waves, so a gust never jerks,
//   * what walks or drives through it lays it over: the tuft turns about its root, away from
//     the push read from the trail texture, and wobbles as it comes back up.
float tw_tri(float x) { return abs(fract(x) * 2.0 - 1.0); }
float tw_sm(float x)  { return x * x * (3.0 - 2.0 * x) - 0.5; }            // -0.5 .. 0.5
float tw_s2(float x)  { return tw_sm(x) * 2.0; }
vec3  tw_rot(vec3 v, vec3 k, float a) {                                    // about unit axis k
    float c = cos(a), s = sin(a);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.0 - c));
}
// lp: the vertex relative to the tuft's root, already turned about the vertical and scaled (m).
// sway01: 0 at a blade's root .. 1 at its tip. Returns the vertex relative to the root, placed.
vec3 tuft_place(vec3 lp, vec3 base, float size, float sway01, float blade_id, float dist) {
    float t    = pc.grass_params.z;
    float H    = max(0.9 * size, 0.05);                                    // the clump's height
    float seed = fract(sin(dot(base.xz, vec2(12.9898, 78.233))) * 43758.5453);
    // Its own lean: about a horizontal axis of its own, more the higher up the blade.
    {
        vec2  d  = vec2(fract(seed * 31.7), fract(seed * 57.3)) * 2.0 - 1.0;
        vec3  ax = normalize(vec3(-d.y, 0.0, d.x) + vec3(1e-4, 0.0, 0.0));
        float h  = max(lp.y, 0.0);
        lp = tw_rot(lp, ax, seed * 0.32 * sin(2.2 * h / (0.7 + 0.2 * h)));
    }
    // On the ground's normal (the home tile's height map; level elsewhere).
    vec3 up = vec3(0.0, 1.0, 0.0);
    {
        vec2 huv = base.xz / 2048.0 + 0.5;
        if (all(greaterThan(huv, vec2(0.002))) && all(lessThan(huv, vec2(0.998)))) {
            const float e = 2.0;
            float hl = textureLod(u_heightmap_full, huv - vec2(e / 2048.0, 0.0), 0.0).r;
            float hr = textureLod(u_heightmap_full, huv + vec2(e / 2048.0, 0.0), 0.0).r;
            float hd = textureLod(u_heightmap_full, huv - vec2(0.0, e / 2048.0), 0.0).r;
            float hu = textureLod(u_heightmap_full, huv + vec2(0.0, e / 2048.0), 0.0).r;
            up = normalize(mix(vec3(0.0, 1.0, 0.0), normalize(vec3(hl - hr, 2.0 * e, hd - hu)), 0.85));
        }
        if (up.y < 0.9999) lp = tw_rot(lp, normalize(cross(vec3(0.0, 1.0, 0.0), up)), acos(clamp(up.y, -1.0, 1.0)));
    }
    float h = max(dot(lp, up), 0.0);
    // Wind. (Nothing beyond 64 m: it is under a pixel there.)
    float wfade = 1.0 - clamp((dist - 48.0) / 16.0, 0.0, 1.0);
    float bh    = fract(sin(blade_id * 12.9898 + seed * 78.233) * 43758.5453);       // this blade's own number
    float ba    = 6.2831853 * bh;
    vec3  Wl    = normalize(vec3(pc.color.x, 0.0, pc.color.y) + vec3(1e-4, 0.0, 0.0));
    vec3  bdir  = normalize(mix(vec3(cos(ba), 0.0, sin(ba)), Wl, 0.6));              // mostly downwind, each a little aside
    float wgt   = sway01 * sway01;
    if (wfade > 0.0 && pc.color.z > 0.0) {
        // Gusts roll over the field along the wind.
        float S = clamp(pc.color.z * (0.62 + 0.38 * sin(dot(base.xz, Wl.xz) * 0.09 - t * 0.8 + sin(base.z * 0.05 + base.x * 0.03) * 2.0)), 0.0, 1.0);
        // The whole clump: its upper half sways.
        float osc = tw_s2(tw_tri(t * 0.8 + base.z + 0.5)); osc = osc * osc + tw_s2(tw_tri(t + 0.5 + base.x));
        vec3  off = Wl * ((osc * S + S) * clamp(max(h - H * 0.5, 0.0), 0.0, 1.0) * 0.015);
        // Each blade: two triangle waves an octave apart, pushed toward their own sign so the
        // blade lingers at the ends of its swing.
        float T  = t + base.x * 0.37 + base.z * 0.21 + bh;
        float A  = tw_sm(tw_tri(T + 0.5)), B = tw_sm(tw_tri(T * 0.5 + 0.5));
        float o  = 4.0 * A * B, sg = o < 0.0 ? -1.0 : 1.0, ab = abs(o);
        float amt = (mix(mix(o, sg, ab), sg, ab) + 2.0 * A) * S;
        off += bdir * (wgt * 0.2 * size * 0.6) * amt;
        // (a blade does not get longer when it is blown aside: it comes down a little)
        lp += (off - up * (dot(off, off) * 0.5 / max(h, 0.15))) * wfade;
    }
    // The trail: laid over, away from what pushed.
    if (pc.emissive.w > 0.5 && dist < 48.0) {
        vec2 tuv = (base.xz - pc.emissive.xy) * pc.emissive.z;
        if (all(greaterThan(tuv, vec2(0.0))) && all(lessThan(tuv, vec2(1.0)))) {
            vec4  s    = textureLod(u_trail, tuv, 0.0);
            float down = clamp(s.w, 0.0, 1.0);
            if (down > 0.01 && dot(s.xy, s.xy) > 1e-4) {
                vec3  D   = normalize(vec3(s.x, 0.0, s.y));
                vec3  kx  = cross(up, D);
                float ang = 1.25 * down * mix(0.75, 1.0, clamp(s.z, 0.0, 1.0))
                          * smoothstep(0.0, 0.6, h / H) * mix(sin(t * 7.0), 1.0, down);
                if (dot(kx, kx) > 1e-6) lp = tw_rot(lp, normalize(kx), ang);
                // ... and each blade trembles on the way back up.
                lp += bdir * (down * wgt * (0.6 - 0.4 * down) * mix(cos(t * 10.0), cos(t * 2.0), down) * 0.05);
            }
        }
    }
    return lp;
}

void main() {
    // Leaf blades (part 3) pack sway base + along-leaf coordinate in uv.y.
    float plant_sway = inUv.y;
    if (int(inUv.x + 0.5) == 8) plant_sway = fract(inUv.y * 32.0) / 0.98;        // grass blade: free tip
    else if (int(inUv.x + 0.5) >= 3) plant_sway = floor(inUv.y * 32.0) / 32.0 + 0.25 * fract(inUv.y * 32.0) / 0.98;
    vec3  base  = inPosScale.xyz;
    float scale = inPosScale.w;
    float rot   = inRotTint.x;

    float dist = distance(base, scene.camera_pos.xyz);
    // Plants on raymarched terrain: rise with its close-range ground
    // relief (whole turns of `rot` = lift in 4 mm steps, scatter.cpp),
    // faded over the same 10 - 26 m as terrain_raymarch.frag.
    if (pc.grass_params.y > 0.5 && pc.grass_params.w > 0.5)
        base.y += floor(rot / 6.2831853) * 0.004 * (1.0 - smoothstep(10.0, 26.0, dist));
    float max_d = max(pc.grass_params.x, 10.0);
    // Smooth shrink over the last 25% so the cull never pops; small
    // pebbles fade earlier than boulders (they're sub-pixel sooner).
    // Capped tighter than grass: a 0.5 m rock at 120 m is a 2-px dark
    // dot — pepper, not detail.
    float reach = max_d * clamp(scale * 2.2 + 0.22, 0.22, 0.85);
    float fade = 1.0 - smoothstep(reach * 0.75, reach, dist);
    // Plants thin out with distance instead of all vanishing at the range
    // limit: from 35 % of the reach a growing share (up to 70 %) shrinks
    // away, chosen by a per-plant hash, and the rest grows a little so a
    // flower drift keeps its colour.
    if (pc.grass_params.y > 0.5) {
        float thin = smoothstep(reach * 0.35, reach * 0.9, dist) * 0.7;
        float ph = fract(sin(dot(base.xz, vec2(12.9898, 78.233))) * 43758.5453);
        fade *= smoothstep(thin - 0.10, thin + 0.02, ph + 0.02) * (1.0 + 0.6 * thin);
    }
    vCullKill = (fade < 0.03) ? 1.0 : 0.0;
    if (vCullKill > 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vPrevClip   = gl_Position;
        vNormal = vec3(0, 1, 0); vColor = vec3(0);
        vWorldPos = vec3(0); vSunShadow = 0.0; vDistToCam = 0.0;
        vUv = vec2(0.0);
        return;
    }

    float c = cos(rot), s = sin(rot);
    mat3 R = mat3(vec3(c, 0, s), vec3(0, 1, 0), vec3(-s, 0, c));
    // Stones: the unit icosphere is pushed out to the stone's own shape
    // (rock_sdf.glsl stone_radius: a block broken along random planes, per-
    // stone seed). Real geometry with a real normal - the fragment shader
    // only adds fine detail (no ray march, no discard).
    vec3 lpos = inPos;
    vRotSeed = vec2(rot, fract(sin(dot(base.xz, vec2(41.3, 17.9))) * 951.135) * 40.0);
    vec3 squash = vec3(1.0);
    vec3 stone_n = vec3(0.0, 1.0, 0.0);
    if (pc.grass_params.y < 0.5) {
        float seed = vRotSeed.y;
        float sv = fract(seed * 0.618);
        vec3 d = normalize(inPos);
        // Surface point and two neighbours on the sphere -> analytic normal.
        vec3 t1 = normalize(cross(d, abs(d.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
        vec3 t2 = cross(d, t1);
        const float e = 0.06;
        vec3 d1 = normalize(d + t1 * e), d2 = normalize(d + t2 * e);
        vec3 p0 = d * stone_radius(d, seed), p1 = d1 * stone_radius(d1, seed), p2 = d2 * stone_radius(d2, seed);
        // Stone-local: centre 0.30 above the base (the lower part sits in
        // the ground). Flat pieces (sv < 0.22) are squashed.
        if (sv < 0.22) squash = vec3(1.0, 0.45, 1.0);
        lpos = vec3(0.0, 0.30, 0.0) + p0;
        vec3 nn = cross(p1 - p0, p2 - p0);
        if (dot(nn, d) < 0.0) nn = -nn;
        stone_n = normalize(nn / squash);
    }
    vObj = lpos;
    vStoneN = stone_n;
    vInst = vec4(base, scale * fade);
    vec3 world = R * (lpos * squash * scale * fade) + base;
    // Grass tufts: their own model (tuft_place above).
    bool tuft = pc.grass_params.y > 0.5 && int(inUv.x + 0.5) == 8;
    if (tuft)
        world = base + tuft_place(R * (lpos * scale * fade), base, scale * fade, clamp(plant_sway, 0.0, 1.0), floor(inUv.y * 32.0), dist);
    // Other plants: wind sway weighted by uv.y (0 at the root, 1 at the tips).
    if (!tuft && pc.grass_params.y > 0.5 && plant_sway > 0.0) {
        float t = pc.grass_params.z;
        float ph = dot(base.xz, vec2(0.37, 0.23));
        vec2 wd = normalize(vec2(0.8, 0.45));
        float s = sin(t * 2.1 + ph) * 0.7 + sin(t * 4.7 + ph * 1.9) * 0.3 + 0.25;
        world.xz += wd * (s * 0.06 * scale * plant_sway * plant_sway);
        world.y  -= abs(s) * 0.015 * scale * plant_sway;
    }
    vUv = inUv;

    gl_Position = pc.mvp * vec4(world, 1.0);
    vPrevClip   = pc.prev_mvp * vec4(world, 1.0);
    vWorldPos   = world;
    vNormal     = R * inNormal;
    vColor      = inRotTint.yzw;
    vDistToCam  = dist;

    // Hybrid sun shadow sampled at the rock's TOP, not its base — the
    // base sits exactly at the terrain depth in the shadow map, so a
    // base sample self-shadows (acne) and every rock rendered black.
    float shadow = 0.0;
    {
        vec3 probe = base + vec3(0.0, scale + 0.08, 0.0);
        vec4 lc = scene.light_vp * vec4(probe, 1.0);
        vec3 lndc = lc.xyz / lc.w;
        vec2 luv  = lndc.xy * 0.5 + 0.5;
        bool inFrustum = lndc.z >= 0.0 && lndc.z <= 1.0 &&
                         all(greaterThanEqual(luv, vec2(0.0))) &&
                         all(lessThanEqual(luv, vec2(1.0)));
        if (inFrustum) {
            shadow = 1.0 - textureLod(u_sun_shadow_map,
                                      vec3(luv, lndc.z - 0.00005), 0.0);
        } else {
            vec2 buv = (base.xz / 2048.0) + vec2(0.5);
            if (all(greaterThanEqual(buv, vec2(0.0))) &&
                all(lessThanEqual(buv, vec2(1.0)))) {
                shadow = step(0.5, textureLod(u_terrain_shadow, buv, 0.0).r);
            }
        }
    }
    vSunShadow = shadow;
}
