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
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;          // the lawn: r = grass, g = slope
layout(set = 0, binding = 41) uniform sampler2D u_trail;               // where people and cars have just been (grass_trail.cpp)
layout(set = 0, binding = 43) uniform sampler2D u_trail_h;             // ... and between which heights they stood
layout(set = 0, binding = 45) uniform sampler3D u_wind;                // the wind field round the camera (wind_fluid.cpp)

// Grass tufts: their own model (the reference game's foliage vertex shader).
#include "tuft_cd.glsl"

void main() {
    // Leaf blades (part 3) pack sway base + along-leaf coordinate in uv.y.
    float plant_sway = inUv.y;
    if (int(inUv.x + 0.5) == 8 || int(inUv.x + 0.5) == 10) plant_sway = fract(inUv.y * 32.0) / 0.98;        // grass blade: free tip
    else if (int(inUv.x + 0.5) >= 3) plant_sway = floor(inUv.y * 32.0) / 32.0 + 0.25 * fract(inUv.y * 32.0) / 0.98;
    vec3  base  = inPosScale.xyz;
    float scale = inPosScale.w;
    float rot   = inRotTint.x;
    vec3  tint  = inRotTint.yzw;
    // Grass tufts. Those placed on the GPU (tuft_place.comp) come with a negative size - they
    // are already faded by the placement - and with their dissolve packed into the red.
    bool  tuft = pc.grass_params.y > 0.5 && (int(inUv.x + 0.5) == 8 || int(inUv.x + 0.5) == 10);        // (8 = a blade, 10 = a flowering stalk; 9 is the water lily's pad)
    bool  placed = tuft && scale < 0.0;
    float dissolve = 1.0;
    if (tuft) {
        scale = abs(scale);
        float fb = floor(tint.r);
        if (fb > 0.0 || placed) { dissolve = (255.0 - fb) / 255.0; tint.r = fract(tint.r) / 0.99; }
    }

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
    if (placed) fade = 1.0;
    else if (pc.grass_params.y > 0.5) {
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
    vRotSeed = vec2(rot, tuft ? dissolve : fract(sin(dot(base.xz, vec2(41.3, 17.9))) * 951.135) * 40.0);
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
    // Grass tufts: their own model (tuft_cd.glsl).
    vec3 wnormal = R * inNormal;
    if (tuft) {
        wnormal = inNormal;
        bool gone;
        world = tuft_cd(lpos, wnormal, base, scale * fade, rot, clamp(plant_sway, 0.0, 1.0), floor(inUv.y * 32.0), dist,
                        !placed, pc.tex_params.x < 0.5, gone);
        if (gone) {
            vCullKill = 1.0;
            gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
            vPrevClip   = gl_Position;
            vNormal = vec3(0, 1, 0); vColor = vec3(0);
            vWorldPos = vec3(0); vSunShadow = 0.0; vDistToCam = 0.0;
            vUv = vec2(0.0);
            return;
        }
    }
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
    vNormal     = wnormal;
    vColor      = tint;
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
