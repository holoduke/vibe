#version 460
// ONE source for two passes:
//   * terrain_raymarch.frag.spv - procedural ray-marched terrain (default)
//   * water.frag.spv            - water surface, compiled with -DVIBE_WATER=1
//     (see CMakeLists.txt). The two used to be ~3100-line near-copies;
//     the pass-specific parts are the VIBE_WATER blocks below.
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require

// Procedural FBM heightfield ray-marched terrain — selectable alternate
// to the chunked rasterised terrain. Implements the technique from
// MiniMax-AI/skills shader-dev terrain-rendering reference: 2D Value
// Noise with analytical derivatives, FBM with derivative-erosion
// suppression, adaptive-step heightfield ray march, soft shadows,
// height/slope material blend, atmospheric extinction.
//
// Runs as a fullscreen triangle AFTER the rasterised cubes / castle /
// dyn-props have written depth. gl_FragDepth at the ray hit lets the
// rasterised geometry occlude correctly. Misses (sky) discard so the
// compose pass's sky still shows through.

layout(location = 0) in vec2 vNDC;
layout(location = 1) in vec4 vWNear;
layout(location = 2) in vec4 vWFar;
#ifdef VIBE_WATER
// Perspective-correct world position of the water surface at this
// pixel, output by water.vert from the actual plane mesh vertices.
// Replaces the per-frame inv(view_proj) reconstruction below — that
// was the source of the "water orientation drifts with camera" wobble.
layout(location = 3) in vec3 vWPos;
#endif /* VIBE_WATER */

layout(location = 0) out vec4 outColor;
#ifdef VIBE_MARCH_ONLY
// The low-resolution build (main: VIBE_MARCH_ONLY) writes the sun-shadow terms of its hit
// here: x self-shadow, y object shadow (both 0 lit .. 1 shadowed), z distance of the hit,
// w 1 = valid.
layout(location = 1) out vec4 outShadeLo;
#define outMotion outShadeLo.xy
#else
layout(location = 1) out vec2 outMotion;
#endif
#ifndef VIBE_WATER
// G-buffer dual writes (deferred migration). Raymarched terrain
// pixels carry material_id 8 = pre-shaded pass-through.
#else /* VIBE_WATER */
// G-buffer dual writes — Phase 2/5 of the deferred-render migration.
// Water pixels get material_id 7 so the deferred lighting pass sees the
// already-shaded water colour and passes it through unmodified.
#endif /* VIBE_WATER */
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

// Same push-constant layout as cube.frag's PushConstants. We use
//   mvp           — view_proj for gl_FragDepth at the hit point
//   model         — inverse(view_proj) for world-ray reconstruction
//   color.xy      — plateau centre (world XZ)
//   color.z       — plateau half-extent (m)
//   color.w       — plateau height
//   emissive.x    — march step factor (0.4..0.8)
//   tex_params.x  — march step count
//   tex_params.y  — shadow ray steps
//   tex_params.z  — march FBM octaves
//   tex_params.w  — normal FBM octaves
//   grass_params.x — fog strength multiplier (0 = off)
//   grass_params.y — relaxation flag (>0.5 = use relaxation step)
//   grass_params.z — fog god-ray flag (>0.5 = self-shadowed fog)
// The remaining fields are unused by this shader.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

// Heightmap delta texture (R32F). Stores `current_height -
// procedural_baseline_height` per cell, so sculpt edits + plateau
// noise + disk-loaded overlays appear on top of the shader's GLSL
// FBM. Uploaded once at level load and re-uploaded after sculpt /
// noise edits. World XZ → UV maps via the gameplay-plateau push
// constants (pc.color.xy = origin, pc.color.z is plateau-extent
// — we don't have the heightmap origin/side here, so use the well-
// known engine layout: terrain centred at origin, side = 2048 m).
layout(set = 0, binding = 8) uniform sampler2D u_terrain_height;
// FULL baked terrain height (NOT the delta) — the exact array the
// CDLOD mesh + physics are built from. The water path samples this in
// mesh-terrain mode so its shore / showthrough / reflection /
// self-shadow follow the real rasterised mesh surface with zero
// procedural FBM evaluation. Same dim+1 / 2048 m layout as binding 8.
layout(set = 0, binding = 17) uniform sampler2D u_terrain_height_full;
// Grass eligibility mask. R8 1024² covering the same 2048 m square as
// u_terrain_height. CPU-baked once at level load (presence + slope from
// the same FBM the GLSL terrain uses, then sample_terrain_height for an
// accurate slope finite-difference). Both this shader and
// grass_raymarch.frag sample binding 13 — single source of truth for
// "is there grass here", and avoids the per-step 9-cell noised() storm
// in the grass map() loop.
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;
// Depth pre-pass copy (objects, vegetation, rocks, castle): pixels they
// cover never need the terrain marched behind them.
layout(set = 0, binding = 27) uniform sampler2D u_prepass_depth;
// The low-resolution march (binding 34, R32F): for each block of pixels, how far a ray can
// go before the ground comes near (terrain_rm_heightfield.glsl raymarch_lo).
layout(set = 0, binding = 34) uniform sampler2D u_march_lo;
// ... and the sun-shadow terms at those blocks' hits (binding 35, RGBA16F; see outShadeLo).
layout(set = 0, binding = 35) uniform sampler2D u_shade_lo;
// Fog wisp pattern (R8 256², REPEAT). One textureLod() replaces the
// 3-octave noise2() weighted sum the fog density + godray probe
// loops used to compute per-tap. UV maps q→texture as `q / 16.0` so
// the texture tiles every 16 q-units (≈800 m of world space).
layout(set = 0, binding = 14) uniform sampler2D u_fog_wisp;
#ifndef VIBE_WATER
// Small-object (flowers / ferns / clutter rocks) sun shadow map, D32.
layout(set = 0, binding = 28) uniform sampler2DArray u_clutter_shadow;   // layer 0 = 64 m, 1 = 16 m near
// Smooth 3x3 PCF on one layer of the small-caster shadow map: the edge
// taps are weighted by the sub-texel position (4x4 fetches), so the
// result changes continuously across a texel instead of in texel-sized
// steps - without it close shadows look like stacked blocks.
float clutterOcc(vec2 uv, int layer, float zref) {
    vec2 st = uv * vec2(textureSize(u_clutter_shadow, 0).xy) - 0.5;
    ivec2 b = ivec2(floor(st));
    vec2 fr = st - vec2(b);
    float occ = 0.0;
    for (int y = -1; y <= 2; ++y) {
        float wy = (y == -1) ? 1.0 - fr.y : (y == 2 ? fr.y : 1.0);
        for (int x = -1; x <= 2; ++x) {
            float wx = (x == -1) ? 1.0 - fr.x : (x == 2 ? fr.x : 1.0);
            float zc = texelFetch(u_clutter_shadow, ivec3(b + ivec2(x, y), layer), 0).r;
            occ += (zc < zref) ? wx * wy : 0.0;
        }
    }
    return occ / 9.0;
}
#endif /* ! VIBE_WATER */
// (in a far chart - another part of the planet - there is no home tile: its side is nothing)
bool g_far_chart = false;        // (set first thing in main: scene.globe_info.x)
#define kHeightmapSide (g_far_chart ? 1.0e-3 : 2048.0)
// Forward decl — terrainDetailNoise is defined after `noised` + `m2`
// (further down in the file) so we can reuse them. We use a forward
// declaration here so sampleHeightDelta can call it without moving
// the whole noise toolkit above the declarations of u_terrain_height.
float terrainDetailNoise(vec2 worldXZ);
vec3  noised(in vec2 p);   // single-octave value noise + analytical derivatives

float sampleHeightDelta(vec2 worldXZ) {
    // Was: 4 compare branches before the fetch. Replaced with clamp +
    // textureLod 0 — sampler is CLAMP_TO_EDGE and the heightmap delta
    // is ~0 at the world boundary (no sculpting past the edge), so a
    // clamped read at out-of-world XZ returns the right value too.
    // Beyond the home tile there is no sculpting. The edge texels are NOT
    // all zero, and a clamped read drew them out into ruled furrows running
    // straight away from the tile (lines in the meadows east/west/north/south
    // of home): fade the delta out over the tile's last metres instead.
    vec2 aq = abs(worldXZ);
    float edge = kHeightmapSide * 0.5 - max(aq.x, aq.y);
    if (edge <= 0.0) return 0.0;
    vec2 uv = (worldXZ / kHeightmapSide) + vec2(0.5);
    return textureLod(u_terrain_height, uv, 0.0).r * smoothstep(0.0, 14.0, edge);
}

// === The open sea's wave field ======================================
// (the sea's wave field - sea_map - is further down, after scene_ubo.glsl: it reads the scene buffer)

// === Mesh-terrain height (water-only mode) ==========================
// Bilinear lookup of the FULL baked heightmap — identical to the
// surface the rasterised CDLOD mesh draws. Used INSTEAD of the FBM
// terrainM* functions throughout the water path when the engine runs
// the water-only pass over mesh terrain, so the water never reflects
// or shows procedural noise that disagrees with the mesh.
float meshHeight(vec2 worldXZ) {
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
    return textureLod(u_terrain_height_full, uv, 0.0).r;
}
vec3 meshNormal(vec2 worldXZ) {
    const float e = 1.5;   // ≈ heightmap cell — smooth mesh-scale normal
    float hL = meshHeight(worldXZ - vec2(e, 0.0));
    float hR = meshHeight(worldXZ + vec2(e, 0.0));
    float hD = meshHeight(worldXZ - vec2(0.0, e));
    float hU = meshHeight(worldXZ + vec2(0.0, e));
    return normalize(vec3(hL - hR, 2.0 * e, hD - hU));
}
// raymarchMesh / raymarchReflectMesh / calcShadowMesh are defined
// after the Scene UBO block (they read scene.terrain_local_info.y for
// the air early-out) — see below, just after raymarchReflect().

// Sun shadow map (single-cascade ortho D32). Rendered each frame
// from the light's POV with castle, dyn-props and terrain chunks
// as casters. Sampled here for water-surface shadow occlusion.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;

// TLAS — castle, dyn-props, terrain BLAS, etc. We fire shadow rays
// against it with cull-mask 0x02 so the terrain BLAS (instance
// mask 0x01) is excluded — terrain self-shadow is handled by
// calcShadow() against the FBM, and the rasterised terrain mesh sits
// below the BLAS detail anyway. With the mask, each ray only finds
// real dynamic-occluder hits (castle, boxes).
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

// Materials buffer — same layout cube.frag reads. Path-traced GI on
// the raymarched terrain looks up the BRDF colour at each bounce hit
// from this buffer using the closest-hit's instance custom index
// (with the kStaticBlasSentinel branch for the merged static BLAS).
struct Material {
    vec4 color;
    vec4 emissive;
    vec4 tex;       // x: albedo idx, y: normal idx, z: uv scale, w: spare
};
layout(set = 0, binding = 2, std430) readonly buffer Materials {
    Material materials[];
};
// Sentinel matching kStaticBlasInstSentinel in src/engine/vk_engine/rt.cpp
// — the merged-static castle BLAS uses this in instanceCustomIndex; the
// shader recovers the per-brush material via primitive_id / 12.
const int kStaticBlasSentinel = 0xFFFFFF;
const int kCubeTrisPerBox     = 12;

bool leaf_card_hit(rayQueryEXT rq);   // defined after the texture arrays
// Leaf cards as solid cards for the rays traced while this is set (as in
// deferred_lighting.frag, settings 'foliage rays'): the search for a shadow's blocker and
// the single far shadow ray do not need the leaf texture asked at every card they cross.
bool g_rq_leaf_solid = false;
// Mask 0x02 = "anything except the terrain BLAS instance".
bool any_hit_no_terrain(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT | (g_rq_leaf_solid ? gl_RayFlagsOpaqueEXT : 0u),
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
            leaf_card_hit(rq)) rayQueryConfirmIntersectionEXT(rq);
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Closest-hit variant for the PCSS blocker search — returns the hit
// distance so we can size the penumbra cone.
bool closest_hit_no_terrain(vec3 origin, vec3 dir, float t_max,
                             out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, g_rq_leaf_solid ? gl_RayFlagsOpaqueEXT : gl_RayFlagsNoneEXT,
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
            leaf_card_hit(rq)) rayQueryConfirmIntersectionEXT(rq);
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

// Closest-hit + material lookup (mask 0x02 — terrain BLAS skipped).
// Used by the path-traced GI loop to find castle / dyn-prop bounce
// hits and look up their albedo for the throughput tint.
bool closest_hit_material_no_terrain(vec3 origin, vec3 dir, float t_max,
                                     out float out_t,
                                     out int out_inst,
                                     out int out_prim) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, g_rq_leaf_solid ? gl_RayFlagsOpaqueEXT : gl_RayFlagsNoneEXT,
                          0x02, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
            leaf_card_hit(rq)) rayQueryConfirmIntersectionEXT(rq);
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t   = rayQueryGetIntersectionTEXT(rq, true);
    out_inst = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    out_prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    return true;
}

// Shadow-caster mask 0x01 — used FROM the GI bounce hit point on a
// castle / dyn-prop surface to test sun visibility. Includes the
// terrain BLAS (which IS marked 0x01) so a bounce hit deep in a
// valley is correctly shadowed by the surrounding terrain. NOT used
// for the primary terrain receiver shadow — that's the no-terrain
// mask above to avoid procedural-vs-BLAS LOD self-hits.
bool any_hit_shadow_caster(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT | (g_rq_leaf_solid ? gl_RayFlagsOpaqueEXT : 0u),
                          0x01, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
            leaf_card_hit(rq)) rayQueryConfirmIntersectionEXT(rq);
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Build a cosine-weighted sample direction in the hemisphere around
// `n`. With this distribution the Lambertian estimator simplifies to
// `albedo * Li` per sample (the cos(θ) and BRDF/π factors cancel
// against the pdf), which matches cube.frag's GI formulation.
vec3 cos_hemi(float u1, float u2, vec3 n) {
    float r   = sqrt(u1);
    float phi = 6.28318530718 * u2;
    vec3  d   = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
    vec3 up_a = abs(n.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t    = normalize(cross(up_a, n));
    vec3 b    = cross(n, t);
    return d.x * t + d.y * n + d.z * b;
}

// Sample-with-precomputed-basis variant. The tangent frame around `n`
// only depends on `n`, which is constant across an N-sample GI/AO
// loop — caller computes the basis once outside the loop and we
// avoid the cross+normalize+cross per sample.
vec3 cos_hemi_basis(float u1, float u2, vec3 n, vec3 t, vec3 b) {
    float r   = sqrt(u1);
    float phi = 6.28318530718 * u2;
    vec3  d   = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
    return d.x * t + d.y * n + d.z * b;
}

// Tiny xorshift32 hash — deterministic per-pixel jitter for the
// stratified shadow rays.
// Interleaved Gradient Noise — same dither pattern cube.frag uses
// for PCSS / RTAO. xorshift hashes (the previous rmRand) produce
// row-banding correlations that read as moving "lines in shadows"
// even with TAA. IGN is designed for stochastic rendering: low
// perceptual structure across pixels, well-distributed across the
// sample-index axis when you offset the input position.
float ignBase(vec2 p) {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
// PCG3D hash (Jarzynski & Olano 2020), same as deferred_lighting.frag.
// The previous IGN-based version put a sample's two coordinates on one
// 1D curve (diagonal hatching in penumbrae / AO) and lost float
// precision once the raw frame counter in seed.z grew large.
uvec3 rm_pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
float rmRand(uvec3 seed) {
    return float(rm_pcg3d(seed).x) * (1.0 / 4294967296.0);
}

// Scene UBO — same layout as cube.frag's binding 0. Only a handful of
// fields are read here (camera, sun, sky); the rest are unused.
layout(set = 0, binding = 32) uniform sampler2D u_cloud_shadow;   // sunlight through the cloud layer (clouds.comp)
#define CLOUD_SHADOW_MAP u_cloud_shadow
#include "scene_ubo.glsl"

#include "atmosphere.glsl"
#include "sun_horizon.glsl"

// After "Seascape" by Alexander Alekseev (TDM), shadertoy.com/view/Ms2SD1
// (CC BY-NC-SA 3.0): each octave is a sheet of sharp-crested waves made
// from |sin| and |cos| of a noise-bent coordinate, raised to a "choppy"
// power; two copies run against each other, and every octave is turned,
// 3.8x finer, 0.22x lower and a little less choppy than the last.
// Here it is evaluated on the water plane (metres) for the surface normal
// and the crest tint; an octave fades out where a pixel no longer resolves
// it (`foot`: metres per pixel there).
// The octave's coordinate comes in twice, for the same point: `uvn` for the noise lattice
// (with `cell0`, the lattice cell of the local origin) and `uvt` for the sines. Both are
// small numbers - see scene_ubo.glsl sea_a / sea_b: far from the world's origin the plain
// coordinate (up to 34 x the distance) left a float too few digits, and the surface cracked
// along the lattice.
float sea_hash(ivec2 c) {
    uvec2 q = uvec2(c & 4095);
    uint h = q.x * 1597334677u ^ q.y * 3812015801u;
    h *= 0x9E3779B1u; h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13;
    return float(h >> 8) * (1.0 / 16777216.0);
}
float sea_noise(vec2 x, ivec2 cell0) {
    vec2 fl = floor(x);
    vec2 f = x - fl;
    ivec2 c = ivec2(fl) + cell0;
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(sea_hash(c), sea_hash(c + ivec2(1, 0)), u.x),
               mix(sea_hash(c + ivec2(0, 1)), sea_hash(c + ivec2(1, 1)), u.x), u.y);
}
float sea_octave(vec2 uvn, vec2 uvt, ivec2 cell0, float choppy) {
    vec2 uv = uvt + (sea_noise(uvn, cell0) * 2.0 - 1.0);
    vec2 wv  = 1.0 - abs(sin(uv));
    vec2 swv = abs(cos(uv));
    wv = mix(wv, swv, wv);
    return pow(1.0 - pow(wv.x * wv.y, 0.65), choppy);
}
// `p`: the point on the water, in metres FROM scene.sea_org.xy.
float sea_map(vec2 p, float tm, float foot) {
    float freq = 0.16, amp = 0.6, choppy = 4.0, wlen = 19.6, h = 0.0;
    vec2 uv = vec2(p.x * 0.75, p.y);
    for (int i = 0; i < 5; ++i) {
        float keep = 1.0 - smoothstep(0.25, 0.7, foot / wlen);
        if (keep <= 0.0) break;
        vec4  A = scene.sea_a[i];
        ivec2 N = ivec2(scene.sea_b[i].xy);
        vec2  lp = (uv + tm) * freq, lm = (uv - tm) * freq;
        float d = sea_octave(A.zw + lp, A.xy + lp, N, choppy) + sea_octave(A.zw + lm, A.xy + lm, N, choppy);
        h += d * amp * keep;
        uv = mat2(1.6, 1.2, -1.2, 1.6) * uv;
        freq *= 1.9; amp *= 0.22; wlen /= 3.8;
        choppy = mix(choppy, 1.0, 0.2);
    }
    return h;
}
layout(set = 0, binding = 31) uniform sampler2D u_sun_horizon;   // far-mountain sun shadow
// Atmosphere sky-view LUT (atmosphere.comp; GENERAL layout).
layout(set = 0, binding = 30) uniform sampler2D u_skyview;

// Material texture arrays (same set-0 slots as cube.frag): only the castle
// road cobbles are sampled here - the rest of this terrain is procedural.
const int kRmTextureCount = 37;   // = cube.frag kTextureCount
const int kRmMatRoad = 14;        // = cube.frag kMatRoad (PavingStones131)
const int kRmMatAsphalt = 32;     // the highway's asphalt (Asphalt031)
const int kRmMatGravel  = 33;     // its verge (Gravel040)
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kRmTextureCount];
layout(set = 0, binding = 4) uniform sampler2D u_normal[kRmTextureCount];
#define LEAF_ATLAS u_albedo[15]
// Leaf-card cut-out for shadow rays. The vegetation BLAS keeps its leaf
// cards in a non-opaque geometry (two triangles per card, corners in a
// fixed order), so the card uv follows from the triangle's parity and the
// barycentrics; the instance's SBT offset carries the atlas cell + 1.
bool leaf_card_hit(rayQueryEXT rq) {
    uint cell1 = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rq, false);
    if (cell1 == 0u) return true;
    vec2 b = rayQueryGetIntersectionBarycentricsEXT(rq, false);
    vec2 uv = ((rayQueryGetIntersectionPrimitiveIndexEXT(rq, false) & 1) == 0) ? vec2(b.x + b.y, b.y)
                                                                              : vec2(b.x, b.x + b.y);
    float cell = float(cell1 - 1u);
    vec2 auv = vec2(mod(cell, 2.0) * 0.5 + clamp(uv.x, 0.006, 0.994) * 0.5,
                    (cell >= 2.0 ? 0.5 : 0.0) + (1.0 - clamp(uv.y, 0.006, 0.994)) * 0.5);
    return textureLod(LEAF_ATLAS, auv, 1.5).a > 0.33;
}

#ifndef VIBE_WATER
// Slab-test ray vs AABB. Used to check sun occlusion through the voxel
// castle AABB (voxels aren't in the TLAS — see voxel_origin comment).
bool ray_aabb_hit(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
    vec3 inv = 1.0 / rd;
    vec3 a = (bmin - ro) * inv;
    vec3 b = (bmax - ro) * inv;
    vec3 lo = min(a, b);
    vec3 hi = max(a, b);
    float t0 = max(max(lo.x, lo.y), lo.z);
    float t1 = min(min(hi.x, hi.y), hi.z);
    return t1 > max(t0, 0.0);
}
#endif /* ! VIBE_WATER */
// Voxel brickmap atlas + shape directory. Shared with cube.frag at the
// same binding slots — both shaders live in the same descriptor set.
// Suffixed _w so glslang doesn't see a block-name conflict if both
// shaders ever end up linked into one compilation unit.
struct VoxBrickW { uint occ[128]; uint pal[1024]; };
layout(set = 0, binding = 24, std430) readonly buffer VoxBrickAtlas_w {
    VoxBrickW vox_bricks[];
};
layout(set = 0, binding = 25, std430) readonly buffer VoxShapeDir_w {
    uint vox_entries[];
};
const uint kVoxEmptyW = 0xFFFFFFFFu;

// Voxel brickmap DDA for water reflections. The voxel tower lives
// OUTSIDE the TLAS (it's a raster pipeline + per-pixel DDA), so the
// closest-hit ray queries on `topLevelAS` miss it entirely. This march
// gives the water reflection path a way to see the tower as a mirror
// reflection. Disabled gate (scene.voxel_origin.w < 0.5) so the whole
// feature is near-zero cost when no voxel shape is live.
bool water_vox_march(vec3 origin_w, vec3 dir_w, float t_max,
                     out float out_t, out uint out_pal) {
    out_t = t_max; out_pal = 0u;
    if (scene.voxel_origin.w < 0.5) return false;

    vec3 lo = origin_w - scene.voxel_origin.xyz;
    vec3 ld = dir_w;
    float vs = scene.voxel_dims.w;
    float bs = scene.voxel_grid.w;
    ivec3 brick_dim = ivec3(scene.voxel_grid.xyz + 0.5);
    ivec3 voxel_dim = brick_dim * 16;
    vec3  shape_max = scene.voxel_dims.xyz;

    vec3 invd = 1.0 / max(abs(ld), vec3(1e-20)) * sign(ld);
    vec3 a = (vec3(0.0) - lo) * invd;
    vec3 b = (shape_max - lo) * invd;
    vec3 lov = min(a, b);
    vec3 hiv = max(a, b);
    float t0 = max(max(lov.x, lov.y), lov.z);
    float t1 = min(min(hiv.x, hiv.y), hiv.z);
    if (t1 <= max(t0, 0.0)) return false;
    float t = max(t0, 0.0) + 1e-4;
    if (t >= t_max) return false;

    vec3  pe = lo + t * ld;
    ivec3 vc = clamp(ivec3(floor(pe / vs)), ivec3(0), voxel_dim - ivec3(1));
    ivec3 stp = ivec3(ld.x >= 0.0 ? 1 : -1,
                      ld.y >= 0.0 ? 1 : -1,
                      ld.z >= 0.0 ? 1 : -1);
    vec3 abs_inv = vec3(
        (abs(ld.x) > 1e-8) ? 1.0 / abs(ld.x) : 1e30,
        (abs(ld.y) > 1e-8) ? 1.0 / abs(ld.y) : 1e30,
        (abs(ld.z) > 1e-8) ? 1.0 / abs(ld.z) : 1e30);
    vec3 tDelta = vs * abs_inv;
    vec3 face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
    vec3 tMax = vec3(
        (ld.x != 0.0) ? (face.x - lo.x) / ld.x : 1e30,
        (ld.y != 0.0) ? (face.y - lo.y) / ld.y : 1e30,
        (ld.z != 0.0) ? (face.z - lo.z) / ld.z : 1e30);

    const int max_steps = 96;
    for (int i = 0; i < max_steps; ++i) {
        if (t > t_max) return false;
        if (any(lessThan(vc, ivec3(0))) ||
            any(greaterThanEqual(vc, voxel_dim))) return false;

        ivec3 bc = vc >> 4;
        int   dir_i = (bc.z * brick_dim.y + bc.y) * brick_dim.x + bc.x;
        uint  bp = vox_entries[dir_i];
        if (bp == kVoxEmptyW) {
            vec3 brickFace = (vec3(bc) + max(vec3(stp), 0.0)) * bs;
            vec3 bMax = vec3(
                (ld.x != 0.0) ? (brickFace.x - lo.x) / ld.x : 1e30,
                (ld.y != 0.0) ? (brickFace.y - lo.y) / ld.y : 1e30,
                (ld.z != 0.0) ? (brickFace.z - lo.z) / ld.z : 1e30);
            t = min(min(bMax.x, bMax.y), bMax.z) + 1e-4;
            vec3 pn = lo + t * ld;
            vc = ivec3(floor(pn / vs));
            face = (vec3(vc) + max(vec3(stp), 0.0)) * vs;
            tMax = vec3(
                (ld.x != 0.0) ? (face.x - lo.x) / ld.x : 1e30,
                (ld.y != 0.0) ? (face.y - lo.y) / ld.y : 1e30,
                (ld.z != 0.0) ? (face.z - lo.z) / ld.z : 1e30);
            continue;
        }
        ivec3 lv = vc & ivec3(15);
        int   li = (lv.z * 256) + (lv.y * 16) + lv.x;
        if ((vox_bricks[bp].occ[li >> 5] & (1u << (li & 31))) != 0u) {
            int word = li >> 2;
            int shift = (li & 3) * 8;
            out_pal = (vox_bricks[bp].pal[word] >> shift) & 0xFFu;
            out_t = t;
            return true;
        }
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            t = tMax.x; vc.x += stp.x; tMax.x += tDelta.x;
        } else if (tMax.y < tMax.z) {
            t = tMax.y; vc.y += stp.y; tMax.y += tDelta.y;
        } else {
            t = tMax.z; vc.z += stp.z; tMax.z += tDelta.z;
        }
    }
    return false;
}

// Max terrain height of the 64 m cell containing world XZ. Branch-free
// clamp + packed-vec4 unpack. kHeightmapSide is the 2048 m world span.
float cellMaxHeight(vec2 wp) {
    vec2 uv = (wp / kHeightmapSide) + vec2(0.5);
    ivec2 c = clamp(ivec2(uv * 32.0), ivec2(0), ivec2(31));
    int idx = c.y * 32 + c.x;                  // 0..1023
    return scene.terrain_max_grid[idx >> 2][idx & 3];
}

// Standard exponential² distance fog. Returns 0..1 mix weight toward
// `distance_fog_color.rgb`. Caller does the actual mix so it can
// run AFTER all atmospheric / specular passes.
//   p_world: shaded point in world space
//   cam:     camera position
// The ground haze is tuned for the 1500 m seen from the ground. Beyond that
// (only reached from the air, looking down through thinner air) it grows slowly.
// And only the part of the ray inside the hazy air near the ground (2.5 km) counts:
// from high up most of the way down is through clear air.
// (ground_haze.glsl: shared with the lit geometry in deferred_lighting.frag)
#include "ground_haze.glsl"
float rm_haze_dist(float t) { return ground_haze_dist(t); }
// 0: the look of the land on the ground (occlusion terms, contrast curve). 1: its look from the air.
// By distance beyond the range of those terms - and by the camera's height, so that from the air
// the land right below is not a brighter, harder patch in the middle of the rest: the whole view
// goes from one look to the other between 250 and 1200 m up, and is one thing from there to space.
float rm_air_look(float t) {
    return max(smoothstep(1500.0, 3500.0, t), smoothstep(250.0, 1200.0, scene.camera_pos.y - scene.water_params.y));
}
float distanceFogAmount(vec3 p_world, vec3 cam) {
    float strength = scene.distance_fog_color.a;
    if (strength < 1e-3) return 0.0;
    float density   = clamp(scene.distance_fog_params.x, 0.0, 0.1);
    float start_d   = scene.distance_fog_params.y;
    float height_top = scene.distance_fog_params.z;
    float max_alpha  = scene.distance_fog_params.w;
    float d_cam = rm_haze_dist(length(p_world - cam));   // (from the air: less)
    float d_raw = max(0.0, d_cam - start_d);
    // Exp² falloff — natural-looking atmospheric fog. Cheaper than
    // Beer-Lambert with no visible difference at typical density.
    float dx = d_raw * density;
    float fog = 1.0 - exp(-dx * dx);
    if (height_top > 0.5) {
        // Optional height fog: thin out above height_top so mountain
        // peaks stay visible above the fog layer.
        float h_w = 1.0 - smoothstep(0.0, height_top, p_world.y);
        fog *= h_w;
    }
    return clamp(fog * strength, 0.0, max_alpha);
}

// Cheap directional sky model — horizon -> zenith ramp around the
// scene's sky colour, with a small sun-tinted lift near the horizon.
// Used as the "no-noise" reference for the GI softener and as the
// sample value for hemisphere rays that miss any occluder.
// Physical sky (sky-view LUT, scene binding 30) when sky_mode is on.
vec3 phys_sky(vec3 dir) {
    vec3 sun = normalize(scene.sun_direction.xyz);
    float elev = asin(clamp(dir.y, -1.0, 1.0));
    float az = 0.0;
    if (dot(dir.xz, dir.xz) > 1e-8 && dot(sun.xz, sun.xz) > 1e-8)
        az = acos(clamp(dot(normalize(dir.xz), normalize(sun.xz)), -1.0, 1.0));
    return textureLod(u_skyview, skyview_uv(elev, az), 0.0).rgb *
           scene.sun_color.rgb * scene.sun_color.a * scene.atmo_params.y;
}
vec3 sample_sky_atmosphere(vec3 dir) {
    if (scene.atmo_params.x > 0.5) return phys_sky(dir);
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, sqrt(up));   // pow(up, 0.45) approx
}

// Sky sample WITH the sun halo — used for path-traced GI rays that
// miss into the open. A bounce ray that points near the sun should
// pick up the disk's intensity (that's a sun-direct contribution
// without firing a separate sun ray). The halo is bounded by the
// pow(8) cosine and the 0.08 multiplier so it doesn't spike on
// near-grazing samples; matches cube.frag's path-trace estimator.
vec3 sample_sky(vec3 dir) {
    if (scene.atmo_params.x > 0.5) return phys_sky(dir);
    vec3  L = normalize(scene.sun_direction.xyz);
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    vec3 sky = mix(horizon, zenith, sqrt(up));   // pow(up, 0.45) approx
    // pow(x, 8) → 3 squarings instead of exp/log.
    float h1 = max(dot(dir, L), 0.0);
    float h2 = h1 * h1;
    float h4 = h2 * h2;
    float halo = h4 * h4;
    sky += scene.sun_color.rgb * scene.sun_color.a * 0.08 * halo;
    return sky;
}

#include "terrain_rm_noise_road.glsl"   // Noise, plateau weight, road distance
#include "terrain_rm_pebbles.glsl"   // Procedural pebbles, lake-bed stones, foam cells
#include "terrain_rm_heightfield.glsl"   // Terrain height (FBM + ground relief) and the ray march against it
#include "terrain_rm_shading_helpers.glsl"   // Normal, terrain / grass shadows, road weights, ground material, reflection marches
#include "terrain_rm_water.glsl"   // Water: flow, normals, foam pattern, ripples
// ---- The sun's shadow on the ground, as two functions ----
// They are called for every pixel by the full pass, or - settings: 'sun shadows at low
// resolution' - once per block of pixels by the low-resolution build, whose results the full
// pass then reads back (main: u_shade_lo). Both return 0 lit .. 1 shadowed.
// Terrain self-shadow: the walk over the baked heights, then the FBM march.
float rm_self_shadow_base(vec3 pos, vec3 nor, vec3 sunDir) {
    float self_shadow = 1.0;
    {
        float lit_g = 1.0;
#ifndef VIBE_WATER
        if (distance(pos, scene.camera_pos.xyz) < 220.0) {
            float tg = 0.10;
            float g_top = terrainMaxHeight();
            for (int i = 0; i < 30; ++i) {
                vec3 q = pos + sunDir * tg;
                if (q.y > g_top) break;
                if (any(greaterThan(abs(q.xz), vec2(kHeightmapSide * 0.5 - 2.0)))) break;
                float hg = q.y - meshHeight(q.xz);
                lit_g = min(lit_g, smoothstep(-(0.10 + 0.02 * tg), -(0.02 + 0.004 * tg), hg));
                if (lit_g < 0.01) break;
                tg *= (i < 14) ? 1.5 : 1.22;
            }
        }
#endif /* ! VIBE_WATER */
        // The FBM self-shadow march (the most expensive part of terrain
        // shading) only where the heightmap walk left the point lit.
        self_shadow = 1.0 - lit_g;
        if (lit_g > 0.01)
            self_shadow = max(self_shadow, 1.0 - calcShadow(pos + nor * 0.05, sunDir));
    }
    return self_shadow;
}
// Ray-traced shadow of objects (trees, castle, rocks): soft near the camera, one hard ray
// further out. `self_shadow`: the ground's own shadow there (where that is full, no far ray).
float rm_rt_sun_shadow(vec3 pos, vec3 nor, vec3 sunDir, float self_shadow) {
    float dyn_shadow = 0.0;
    float cam_dist_pos = distance(pos, scene.camera_pos.xyz);
    // Smooth-fade RT contributions across the LOD band rather than
    // leaving a visible cliff. lod_far comes from the user slider
    // (fog_band.w); lod_near is 60% of that. Effect ramps from 1.0
    // inside lod_near, to 0.0 at lod_far. Far pixels early-out.
    float lod_far  = max(50.0, scene.fog_band.w);
    float lod_near = lod_far * 0.6;
    float rt_lod_t = 1.0 - smoothstep(lod_near, lod_far, cam_dist_pos);
    bool do_rt = rt_lod_t > 0.005 && !RMDBG(32);

    // Gate on shadow_strength too — if the user dialled it to 0,
    // dyn_shadow gets multiplied by 0 at line ~2082 anyway, so the
    // 4 blocker rays + N_s shadow rays were burning BVH for nothing.
    if (scene.rt_flags.x != 0 && do_rt &&
        scene.rt_params.w > 1e-3) {
        // base_softness from the global Shadow softness slider,
        // scaled by the terrain-specific scale (terrain_params.w).
        float base_softness = scene.rt_params.x;
        if (scene.terrain_params.w > 0.0) {
            base_softness *= max(0.05, scene.terrain_params.w);
        }
        // Sample count from global Shadow samples slider × near
        // multiplier, but capped to 6 for the terrain shader. PCSS
        // on terrain at the user's full slider (13+) was the dominant
        // per-frame RT shadow cost and pushed total per-frame work
        // over the GPU TDR threshold once the TLAS grew with streamed
        // dyn-props. cube.frag PCSS still uses the full slider.
        float near_mult = max(1.0, scene.terrain_extra.y);
        int N_s_cap = max(2, int(scene.terrain_rt_extra.x));
        // Under DLSS Ray Reconstruction (rt_flags.x bit 1) the seed changes every frame and the
        // denoiser averages over time: one shadow ray a frame is enough (as deferred_lighting.frag).
        bool rr_denoise = (scene.rt_flags.x & 2) != 0;
        int N_s = clamp(int(ceil(float(scene.rt_flags.y) * near_mult)),
                        rr_denoise ? 1 : 2, N_s_cap);
        if (rr_denoise) N_s = max(1, N_s / 2);     // (half the budget a frame, the other half next frame: deferred_lighting.frag does the same)

        vec3 ref_l = abs(sunDir.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                            : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref_l, sunDir));
        vec3 tan_v = cross(sunDir, tan_u);
        // Bias along sun direction (NOT surface normal) — the FBM
        // normal varies high-frequency per pixel; using it offsets
        // each pixel's shadow cone slightly and produces dither.
        vec3 origin = pos + sunDir * 0.1;
        // Per-pixel + per-frame seed — TAA averages noisy samples.
#ifndef VIBE_WATER
        // World-anchored seed. The per-pixel + per-frame seed relied on
        // TAA to average it, but DLSS keeps it as detail: under tree
        // canopies (wide penumbra, every pixel partly lit) the ground was
        // a speckle that re-rolled every frame and crawled with the camera.
        // Hash the ground position on a grid matched to the pixel
        // footprint instead (power-of-two cell): the grain stays put on
        // the ground like texture.
        float rm_cell = exp2(ceil(log2(max(cam_dist_pos * 2.0 * scene.viewport.w, 1e-4))));
        ivec3 rm_q = ivec3(floor((pos + vec3(0.0, 0.25 * rm_cell, 0.0)) / rm_cell));
        uvec3 seed_base = rm_pcg3d(uvec3(rm_q) +
                                   uvec3(0u, 0u, uint(int(log2(rm_cell)) + 64) * 7919u));
        seed_base.z = 0u;
        // ...except under DLSS Ray Reconstruction (rt_flags.x bit 1): a
        // temporal denoiser needs noise that changes every frame; the
        // frozen grain read as texture and stayed as a blocky speckle.
        if ((scene.rt_flags.x & 2) != 0) seed_base = uvec3(gl_FragCoord.xy, scene.rt_flags.w);
#else /* VIBE_WATER */
        uvec3 seed_base = uvec3(gl_FragCoord.xy, scene.rt_flags.w);
#endif /* VIBE_WATER */

        // 1. Blocker search (4 rays, 4× softness cone).
        // (Ray Reconstruction: 2 blocker rays a frame estimate the same mean distance as 4 - and a
        // single one when the softness is so low that the rays are all but parallel anyway)
        int kBlocker = rr_denoise ? (base_softness < 0.004 ? 1 : 2) : 4;
        float sum_t = 0.0;
        int   hits  = 0;
        for (int i = 0; i < kBlocker; ++i) {
            float br1 = rmRand(seed_base + uvec3(i, 91u, 13u));
            float br2 = rmRand(seed_base + uvec3(i, 19u, 71u));
            float r   = sqrt(br1) * base_softness * 4.0;
            float ph  = 6.28318530718 * br2;
            vec3 j    = (cos(ph) * tan_u + sin(ph) * tan_v) * r;
            vec3 d    = normalize(sunDir + j);
            float t_b;
            // (exact leaf test here: with solid cards nearly every ray under a tree "finds a blocker"
            // and the full set of shadow rays is then traced for all those pixels - slower, not faster)
            if (closest_hit_no_terrain(origin, d, 100.0, t_b)) {
                sum_t += t_b;
                ++hits;
            }
        }
        if (hits > 0) {
            // 2. Penumbra estimate (cube.frag formulation).
            float avg_t = sum_t / float(hits);
            float t_norm = avg_t * 0.1;
            float curve_exp = mix(1.0, 3.0,
                                   clamp(scene.rt_params2.w, 0.0, 1.0));
            float scale = pow(max(t_norm, 0.0), curve_exp);
            float penumbra = clamp(base_softness * scale * 0.6,
                                    base_softness * 0.25,
                                    base_softness * 6.0);

            // 3. Stratified shadow rays.
            int strata = int(ceil(sqrt(float(N_s))));
            float inv = 1.0 / float(strata);
            int taken = 0;
            float blocked = 0.0;
            for (int sy = 0; sy < strata && taken < N_s; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s; ++sx) {
                    float r1 = rmRand(seed_base + uvec3(taken, 11u, 47u));
                    float r2 = rmRand(seed_base + uvec3(taken, 53u, 23u));
                    float u1 = (float(sx) + r1) * inv;
                    float u2 = (float(sy) + r2) * inv;
                    float r  = sqrt(u1) * penumbra;
                    float ph = 6.28318530718 * u2;
                    vec3 j   = (cos(ph) * tan_u + sin(ph) * tan_v) * r;
                    vec3 d   = normalize(sunDir + j);
                    if (any_hit_no_terrain(origin, d, 100.0)) blocked += 1.0;
                    ++taken;
                }
            }
            dyn_shadow = (blocked / float(taken)) * scene.rt_params.w;
        }
    }
    // 4. Max-combine: the FBM self-shadow is the analogue of
    //    cube.frag's heightmap bake — clean, no noise. Where the
    //    surface is in self-shadow, we take that value verbatim.
    //    Where it's lit, the RT cone fills in castle / cube
    //    contributions. Result reads identical to the rasterised
    //    terrain shadow path.
    // Far field: beyond the soft RT shadow band (RT LOD distance) the object
    // shadows (trees, castle, rocks) used to stop dead. Out to ~900 m one
    // hard shadow ray per pixel keeps them - a penumbra is sub-pixel there
    // anyway - blended in across the band.
    if (scene.rt_flags.x != 0 && scene.rt_params.w > 1e-3 &&
        rt_lod_t < 0.999 && cam_dist_pos < 900.0 && self_shadow < 0.99 && !RMDBG(32)) {
        float far_w = 1.0 - smoothstep(700.0, 900.0, cam_dist_pos);
        vec3  fo = pos + nor * (0.15 + cam_dist_pos * 0.002) + sunDir * (0.3 + cam_dist_pos * 0.002);
        g_rq_leaf_solid = scene.season.w <= 0.02 && ((scene.rt_flags.x >> 2) & 3) >= 1;
        float far_sh = any_hit_no_terrain(fo, sunDir, 220.0) ? scene.rt_params.w * far_w : 0.0;
        g_rq_leaf_solid = false;
        dyn_shadow = mix(far_sh, dyn_shadow, rt_lod_t);
    }
    return dyn_shadow;
}

#ifdef VIBE_LITE
// The "first start" build of this shader (terrain_raymarch_lite.frag.spv). The full shader is so
// large that a graphics driver needs a minute or more to compile it the first time; the game
// does not wait for that. It starts on this build - the same ground, found by the same march,
// so everything standing on it is where it belongs - shaded in the simplest way (slope and
// height give the colour, sun and sky the light, flat water), and switches to the full shader
// the moment the driver has it ready (finish_terrain_raymarch_pipeline).
void lite_out(vec3 p, vec3 col) {
    vec4 clip = pc.mvp * vec4(p, 1.0);
    gl_FragDepth = min(clip.z / max(clip.w, 1e-4), 0.999989);
    col = mix(col, scene.distance_fog_color.rgb, distanceFogAmount(p, scene.camera_pos.xyz));
    outColor = vec4(col, 1.0);
    outGBuffer0 = vec4(clamp(col, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
    outGBuffer1 = vec4(0.5, 0.5, 0.7, 0.0);
    vec4 prev_clip = pc.prev_mvp * vec4(p, 1.0);
    outMotion = (clip.w > 0.0 && prev_clip.w > 0.0)
        ? (clip.xy / clip.w - prev_clip.xy / prev_clip.w) * 0.5 - scene.restir_params.zw : vec2(0.0);
}
#endif

void main() {
    g_far_chart = scene.globe_info.x > 0.5;
    nf_setup();
    foot_bounds_init();      // footprints in the snow (winter)
#ifndef VIBE_WATER
    // Ray reconstruction: vWNear / vWFar (relative to the camera) were computed in the vert
    // shader (pc.model × ndc-near / ndc-far) and interpolated. Per
    // pixel we just need the homogeneous divide + subtract + normalize
    // — the matrix multiplies that used to dominate this stub are gone.
#else /* VIBE_WATER */
    // World position of the water surface at this pixel comes directly
    // from the rasterised plane's vertex interpolation (vWPos). The old
    // inv(view_proj) reconstruction (`pc.model * ndc...`) drifted by
    // µm-mm per frame as camera moved — the user saw this as the
    // shoreline orientation shifting with every camera move. Perspective-
    // correct interp of fixed vertex positions is bit-stable per camera
    // pose and matches the rasterised depth exactly. Ray dir & ray-t
    // are derived from (vWPos - camera) so reflection helpers below see
    // the same world geometry.
#endif /* VIBE_WATER */
    vec3 ro = scene.camera_pos.xyz;
#ifndef VIBE_WATER
    vec3 rd = normalize(vWFar.xyz / vWFar.w - vWNear.xyz / vWNear.w);
#else /* VIBE_WATER */
    vec3 _to_surf = vWPos - ro;
    vec3 rd = normalize(_to_surf);
#endif /* VIBE_WATER */

    // Water plane intersection FIRST — its `t` gives us a max distance
    // for the heightfield march: terrain past the water plane can't be
    // visible above water anyway (water is opaque-ish). Used as the
    // raymarch's t_max so far rays aren't wasted on terrain hidden
    // beneath water.
    bool water_on = scene.water_params.x > 0.5;
    float water_y = scene.water_params.y;
#ifndef VIBE_WATER
    float t_water_plane = -1.0;
    if (water_on && abs(rd.y) > 1e-4) {
        float tw = (water_y - ro.y) / rd.y;
        if (tw > 0.0) t_water_plane = tw;
    }
    // From high up (flying out): the water plane is the local lake, not an ocean to
    // the horizon - beyond the landscape the planet itself is drawn (compose.frag).
    // (No further than the terrain march reaches: past that there is no shore to bound it,
    // and from above 1.5 km it was a pale disc of "sea" under the camera.)
    // (The cut comes in from the horizon: at 220 m it lies beyond it and closes in to the march's
    // range by 1200 m. It used to drop to that range the moment the camera passed 220 m - all
    // the sea between 23 km and the horizon turned into the far field's colour in one frame.)
    if (ro.y > 220.0 && t_water_plane > rm_far_limit(ro.y) * mix(3.0, 1.0, smoothstep(220.0, 1200.0, ro.y))) t_water_plane = -1.0;
#else /* VIBE_WATER */
    // Distance to the water plane is just |vWPos - camera|. Replaces the
    // analytic ray-vs-plane intersection (which was algebraically equal
    // but numerically drifted with inv_vp, same root cause as wpos drift).
    float t_water_plane = water_on ? length(_to_surf) : -1.0;
#endif /* VIBE_WATER */

    // Water-only mode: the mesh terrain path already rasterised the
    // ground (with depth) this frame, so we must NOT raymarch terrain
    // again — just produce the water surface. scene.terrain_local_info.x
    // is 1.0 when the engine runs this pass purely for water on top of
    // mesh terrain. t = -1 (no terrain hit) makes the resolve below
    // treat the water plane as the surface; non-water pixels discard,
    // letting the rasterised mesh show, and the water fragment's
    // gl_FragDepth (written at the water surface) is hardware depth-
    // tested against the mesh depth so hills correctly occlude water.
    // Skips the expensive terrain march entirely — only the water
    // plane + the existing full water shading (all styles, foam,
    // showthrough, reflection) run. Identical water to the FBM path.
#ifndef VIBE_WATER
    bool water_only = scene.terrain_local_info.x > 0.5;
#else /* VIBE_WATER */
    // This shader is ONLY ever bound by the rasterised water-plane
    // pipeline, so water-only is unconditional — the mesh terrain
    // already owns the ground this frame; we just shade water.
    bool water_only = true;
#endif /* VIBE_WATER */
#ifdef VIBE_MARCH_ONLY
    {
        // This build IS the low-resolution march: one distance per block of pixels, nothing else.
        bool  air_lo = ro.y > terrainMaxHeight() + 60.0;
        // No further than the nearest object that covers the WHOLE block (the farthest of the
        // depth pre-pass's pixels under it): the full pass does not march behind trees, rocks
        // or walls either - without this the block's ray walked on to the ground behind them.
        float t_lim = t_water_plane;
        float t_obj_far = 1e9;
        {
            float dv = scene.rm_perf.y;
            vec2 c = gl_FragCoord.xy * dv * scene.viewport.zw;            // the block's centre
            vec2 o = (dv > 2.5 ? 0.25 * dv : 0.0) * scene.viewport.zw;     // (wider blocks: four gathers)
            vec4 g0 = textureGather(u_prepass_depth, c - o, 0);
            float zf = max(max(g0.x, g0.y), max(g0.z, g0.w));
            if (dv > 2.5) {
                vec4 g1 = textureGather(u_prepass_depth, c + o, 0), g2 = textureGather(u_prepass_depth, c + vec2(o.x, -o.y), 0),
                     g3 = textureGather(u_prepass_depth, c + vec2(-o.x, o.y), 0);
                zf = max(max(zf, max(max(g1.x, g1.y), max(g1.z, g1.w))),
                         max(max(max(g2.x, g2.y), max(g2.z, g2.w)), max(max(g3.x, g3.y), max(g3.z, g3.w))));
            }
            if (zf < 0.99999) {
                vec4 Pp = mix(vWNear, vWFar, zf);
                t_obj_far = length(Pp.xyz / Pp.w) * 1.02 + 0.3;
                if (t_lim < 0.0 || t_obj_far < t_lim) t_lim = t_obj_far;
            }
        }
        float tg = 0.0;
        float r = (water_only || air_lo) ? 0.0
                : raymarch_lo(ro, rd, t_lim, 1.68 * scene.viewport.w * scene.rm_perf.y * 0.75, tg);
        // (stopped at an object, or at the water: the grass may still stand in front of it)
        if (tg > 1e8 && t_lim > 0.0) tg = 0.0;
        outColor = vec4(r, tg, 0.0, 0.0);
        // The sun's shadow at this block's hit (rm_perf.x bit 4): march on to the ground, and
        // unless something stands in front of it there, work out the two shadow terms once
        // for the whole block.
        outShadeLo = vec4(0.0);
        // (Not where an object stands in front of the ground - trees, rocks, the castle: the
        // full pass does not shade the ground there either. The depth pre-pass tells, before
        // any marching is done.)
        bool  hidden = t_obj_far < r;
        // (rm_perf.x bit 8, "start from the low-resolution hit": the hit alone is kept, w = 0.25)
        if ((int(scene.rm_perf.x) & 12) != 0 && r > 0.0 && r < 1e8 && !hidden) {
            g_rm_no_relief = true;
            float th = raymarch(ro, rd, min(t_lim > 0.0 ? t_lim : 1400.0, 1400.0), r);
            g_rm_no_relief = false;
            if (th > 0.0 && th < 1400.0 && (t_water_plane < 0.0 || th < t_water_plane)) {
                vec3 sun = scene.sun_direction.xyz;
                outShadeLo = vec4(0.0, 0.0, th, 0.25);
                if ((int(scene.rm_perf.x) & 4) != 0) {
                    // (the baked landscape's normal: enough to tell which way the ground faces
                    // and to lift the shadow rays off it; the point itself a little above the
                    // relief-free hit, where the drawn ground is)
                    vec3 ph = ro + rd * th + vec3(0.0, 0.12 * (1.0 - smoothstep(10.0, 26.0, th)), 0.0);
                    vec3 nh = all(lessThan(abs(ph.xz), vec2(kHeightmapSide * 0.5 - 2.0))) ? meshNormal(ph.xz) : vec3(0.0, 1.0, 0.0);
                    // (well turned from the sun: left to the pixels themselves - none of them
                    // that is lit after all may inherit a block's "no light")
                    if (dot(nh, sun) > -0.30) {
                        float ss = RMDBG(4) ? 0.0 : rm_self_shadow_base(ph, nh, sun);
                        float ds = rm_rt_sun_shadow(ph, nh, sun, ss);
                        outShadeLo = vec4(ss, ds, th, 1.0);
                    }
                }
            }
        }
        return;
    }
#endif
    // The FBM heightfield IS the mesh terrain (the CDLOD mesh is baked
    // from this exact noise), so the water's FBM-driven shore depth,
    // showthrough and reflection all agree with the rasterised ground.
    // We still run the real terrain march here so water shades ONLY
    // where it is genuinely the closest surface — identical cost and
    // behaviour to the proven-stable FBM path (no t=-1 overdraw TDR).
    // In water-only mode the FBM terrain itself is never SHADED (the
    // mesh owns the ground); terrain-hit pixels are discarded below so
    // the rasterised mesh shows through.
#ifndef VIBE_WATER
    // Stop at the nearest pre-pass surface (full-resolution pass only: the
    // depth copy matches this target pixel for pixel). The depth test
    // already rejected the terrain behind trees / rocks / the castle - but
    // only after the whole march and the shading had been paid for. Not
    // when the water plane is nearer than the object (things standing in
    // the water are seen through it).
    float t_stop = t_water_plane;
    bool  obj_stop = false;
    if (pc.emissive.x < 0.0 && !water_only) {
        float zpre = texelFetch(u_prepass_depth, ivec2(gl_FragCoord.xy), 0).r;
        if (zpre < 0.99999) {
            vec4 Pp = mix(vWNear, vWFar, zpre);
            Pp.xyz += ro * Pp.w;                 // (vWNear / vWFar are relative to the camera)
            float t_obj = distance(Pp.xyz / Pp.w, ro) * 1.02 + 0.3;
            // Inside the castle: camera and surface both stand on the flat
            // plateau (a square, so the whole sight line does too) - the
            // ground cannot be in front of that surface. Without this every
            // courtyard pixel marched along just above the hidden ground.
            {
                vec3 Po = Pp.xyz / Pp.w;
                vec2 e  = vec2(pc.color.z);
                if (all(lessThan(abs(ro.xz - pc.color.xy), e)) && all(lessThan(abs(Po.xz - pc.color.xy), e)) &&
                    ro.y > pc.color.w + 0.02 && Po.y > pc.color.w + 0.02 &&
                    sampleHeightDelta(Po.xz) < 0.015 && sampleHeightDelta(0.5 * (Po.xz + ro.xz)) < 0.015)
                    discard;
            }
            if (t_water_plane < 0.0 || t_obj < t_water_plane) { t_stop = t_obj; obj_stop = true; }
        }
    }
    // From the air (the eye well above the highest ground) the whole landscape is found by
    // the far march: a careful walk made for the full-detail heights. The normal march takes
    // long strides at distance, counting on the smoothed far LOD - with the detail kept from
    // the air it stepped over thin ridges and left cracks and slivers in the land.
    // The same anywhere outside the home tile: the normal march leans on that tile's baked
    // heights to skip empty space, and without them lost the land beyond a hundred metres
    // (the water plane showed instead).
    // (Outside the home tile on the ground the normal march is used again: it lost the land there
    // only through the empty-space bug fixed in terrain_rm_heightfield.glsl, and the far march for
    // every pixel cost five times as much.)
    bool air = !water_only && ro.y > terrainMaxHeight() + 60.0;
    // Where the low-resolution march got to: the nearest of the four blocks round this
    // pixel (each is clear to there by more than its own width). 1e9: no ground in range.
    float t_start = 0.0;
    if ((int(scene.rm_perf.x) & 1) != 0 && pc.emissive.x < 0.0 && !water_only && !air) {
        vec4 g4 = textureGather(u_march_lo, gl_FragCoord.xy / (scene.rm_perf.y * vec2(textureSize(u_march_lo, 0))), 0);
        t_start = min(min(g4.x, g4.y), min(g4.z, g4.w));
        // "Start from the low-resolution hit" (rm_perf.x bit 8): that march went on to the
        // ground itself (without the close relief). Where the four blocks round this pixel
        // agree on how far it is - one unbroken slope, no edge between them - this pixel
        // starts a little short of the nearest of them and has only the last steps to do.
        // Checked: the starting point has to be clear of the ground, relief included.
        if ((int(scene.rm_perf.x) & 8) != 0 && t_start > 0.0 && t_start < 1e8) {
            vec2 suv = gl_FragCoord.xy / (scene.rm_perf.y * vec2(textureSize(u_shade_lo, 0)));
            vec4 h4 = textureGather(u_shade_lo, suv, 2), v4 = textureGather(u_shade_lo, suv, 3);
            if (min(min(v4.x, v4.y), min(v4.z, v4.w)) > 0.1) {
                float h_lo = min(min(h4.x, h4.y), min(h4.z, h4.w)), h_hi = max(max(h4.x, h4.y), max(h4.z, h4.w));
                if (h_hi - h_lo < 0.02 * h_lo + 0.5) {
                    float tc = h_lo - (0.6 / clamp(-rd.y, 0.1, 1.0) + 0.006 * h_lo);
                    if (tc > t_start) {
                        vec3 pc0 = ro + rd * tc;
                        if (pc0.y - terrainM_lod(pc0.xz, tc) > 0.34) t_start = tc;
                    }
                }
            }
        }
    }
    float t = water_only ? raymarchMesh(ro, rd, t_water_plane)
                         : ((air || t_start > 1e8) ? -1.0 : raymarch(ro, rd, t_stop, t_start));
    if (!air && obj_stop && t < 0.0) discard;      // an object owns this pixel
    // From the air (above 1 km): the landscape beyond the normal march's reach.
    if (t < 0.0 && !water_only && (air || !obj_stop)) {
        float fl = rm_far_limit(ro.y);
        t = raymarch_far(ro, rd, (t_stop > 0.0) ? min(fl, t_stop) : fl, air);     // (t_stop: the water plane or an object in front)
        if (obj_stop && t < 0.0) discard;
    }

#if !defined(VIBE_WATER) && !defined(VIBE_LITE)
    if (RMDBG(64)) {
        // Profiling (VIBE_RM_SKIP bit 64): the march alone. Nothing is shaded; the picture
        // shows how many steps each pixel's march took (black 0 .. white 64, red beyond).
        if (t < 0.0) discard;
        vec4 cd = pc.mvp * vec4(ro + rd * t, 1.0);
        gl_FragDepth = min(cd.z / max(cd.w, 1e-4), 0.999989);
        float sc = float(g_rm_steps) / 64.0;
        outColor = vec4(sc > 1.0 ? vec3(1.0, 0.0, 0.0) * min(sc - 1.0, 1.0) + vec3(0.0, 0.0, 0.2) : vec3(sc), 1.0);
        outMotion = vec2(0.0);
        outGBuffer0 = vec4(clamp(outColor.rgb, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
        outGBuffer1 = vec4(0.5, 0.5, 0.7, 0.0);
        return;
    }
#endif
    // Resolve water vs terrain ordering for the rest of the shader.
    float t_water = -1.0;
    if (t_water_plane > 0.0 && (t < 0.0 || t_water_plane < t)) {
        t_water = t_water_plane;
    }

    if (t < 0.0 && t_water < 0.0) {
        // Sky — let the existing compose-pass sky handle this pixel.
#else /* VIBE_WATER */
    // RASTERISED WATER PLANE: this shader is only ever bound by the
    // water-plane pipeline. The terrain mesh already wrote its depth
    // this frame, so occlusion (terrain in front of water) is a
    // pixel-exact HARDWARE depth test on real geometry. We must NOT
    // re-decide water-vs-land from the heightmap here: raymarchMesh()
    // disagrees with the rasterised+displaced terrain silhouette by a
    // few px, so discarding "land" punched a thin shore/far band that
    // neither water nor terrain filled → cleared sky bled through and
    // crawled as the camera moved. Always shade the water plane; the
    // depth test hides it wherever terrain is nearer. (Also drops the
    // expensive per-pixel terrain march from the water pass.)
    float t = -1.0;
    float t_water = t_water_plane;
    if (t_water <= 0.0) {
        // Ray never crosses the water plane (degenerate / looking away)
        // — nothing to shade. Plane geometry rarely produces these.
#endif /* VIBE_WATER */
        discard;
    }

#include "terrain_rm_main_water.glsl"   // main(), continued: the water surface: waves, refraction, reflection, foam, the bed seen through it

#include "terrain_rm_main_ground.glsl"   // main(), continued: the ground's material: road and path, the close-up detail, wetness
#include "terrain_rm_main_light.glsl"   // main(), continued: direct light: the sun's shadows (terrain, objects, clutter), ambient occlusion, sky light
#include "terrain_rm_main_gi.glsl"   // main(), continued: bounce light (path-traced GI), the lamps and other local lights, the final colour
#include "terrain_rm_main_fog.glsl"   // main(), continued: the ground fog (ray-marched) and the haze of distance
    // The castle floor brushes are 0.5 m thin slabs sitting ~5 cm above
    // plateau height. terrain_raymarch traces the FBM (which is mixed
    // to a flat plateau_height inside the plateau), and the marched hit
    // depth at a floor pixel is essentially equal to the floor brush's
    // rasterised depth — sub-pixel TAA jitter + float precision tips
    // the LESS_OR_EQUAL test one way or the other per frame, so the
    // raymarched ground colour bleeds through as "gray flicker on the
    // castle floor" when the camera moves. Discard inside the castle
    // footprint and let the cube floor brushes own those pixels. The
    // castle is centred at origin with half-extent cr = 11 m
    // (level.cpp); a small +0.05 margin matches the floor brush's
    // in_ext = cr*2 - 0.05 extent so the discard region exactly covers
    // the floor brush footprint.
    const float kCastleHalfExtent = 11.05;
    if (abs(pos.x) < kCastleHalfExtent && abs(pos.z) < kCastleHalfExtent) {
        discard;
    }

    // Write the hit-point depth so rasterised geometry that wrote
    // depth earlier in the same pass occludes the terrain correctly.
    // Cap below 0.9999 — the compose pass treats depth >= 0.99999 as
    // a sky pixel and paints `sample_sky(dir)`. For terrain hits past
    // ~800 m the projected depth crosses the threshold and compose
    // overwrites our colour with the skybox sampled below the horizon
    // (which is black for typical HDRIs). Result: distant raymarch
    // terrain renders as pure black. Capping keeps compose's sky-
    // detection from misfiring on far terrain hits.
#ifdef VIBE_WATER
    // Terrain branch is DEAD in water.frag (water_only is forced true →
    // discarded long before here). `clip` is still needed by the motion
    // block below, but we deliberately write NO gl_FragDepth anywhere in
    // this shader so it is not depth-replacing — the rasterised water
    // plane's hardware depth is used as-is (pixel-exact shoreline).
#endif /* VIBE_WATER */
    vec4 clip = pc.mvp * vec4(pos, 1.0);
#ifndef VIBE_WATER
    gl_FragDepth = min(clip.z / max(clip.w, 1e-4), 0.999989);
#endif /* ! VIBE_WATER */

    // Distance fog — final atmospheric mix, after all the other
    // shading, GI, water, volumetric fog and grass-shadow work.
    {
        float fa = distanceFogAmount(pos, scene.camera_pos.xyz);
        col = mix(col, scene.distance_fog_color.rgb, fa);
    }
    outColor = vec4(col, 1.0);
#ifndef VIBE_WATER
    outGBuffer0 = vec4(clamp(col, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
    outGBuffer1 = vec4(0.5, 0.5, 0.7, 0.0);
#else /* VIBE_WATER */
    // G-buffer pass-through. Water is already fully shaded; material_id 7
    // tells the deferred lighting pass to use albedo unmodified.
    outGBuffer0 = vec4(clamp(col, vec3(0.0), vec3(1.0)), 7.0 / 255.0);
    outGBuffer1 = vec4(0.5, 0.5, 0.2, 0.0);   // placeholder normal+rough
#endif /* VIBE_WATER */
    // Screen-space motion vector for TAA. World position is stationary
    // — only the camera moves — so prev_uv comes from prev_view_proj
    // applied to the same world hit point. Without this, TAA cannot
    // reproject and per-frame PCSS / fog jitter shows up as moving
    // square dither artefacts.
    {
        // clip-space derivation (resolution-independent) — see the
        // matching block in the water branch above.
        vec4 prev_clip = pc.prev_mvp * vec4(pos, 1.0);
        if (clip.w > 0.0 && prev_clip.w > 0.0) {
            vec2 cur_uv  = clip.xy      / clip.w      * 0.5 + 0.5;
            vec2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
            outMotion = (cur_uv - prev_uv) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
    }
}
