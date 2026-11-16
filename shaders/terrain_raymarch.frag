#version 460
// ONE source for two passes:
//   * terrain_raymarch.frag.spv - procedural ray-marched terrain (default)
//   * water.frag.spv            - water surface, compiled with -DQLIKE_WATER=1
//     (see CMakeLists.txt). The two used to be ~3100-line near-copies;
//     the pass-specific parts are the QLIKE_WATER blocks below.
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
#ifdef QLIKE_WATER
// Perspective-correct world position of the water surface at this
// pixel, output by water.vert from the actual plane mesh vertices.
// Replaces the per-frame inv(view_proj) reconstruction below — that
// was the source of the "water orientation drifts with camera" wobble.
layout(location = 3) in vec3 vWPos;
#endif /* QLIKE_WATER */

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
#ifndef QLIKE_WATER
// G-buffer dual writes (deferred migration). Raymarched terrain
// pixels carry material_id 8 = pre-shaded pass-through.
#else /* QLIKE_WATER */
// G-buffer dual writes — Phase 2/5 of the deferred-render migration.
// Water pixels get material_id 7 so the deferred lighting pass sees the
// already-shaded water colour and passes it through unmodified.
#endif /* QLIKE_WATER */
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
// Fog wisp pattern (R8 256², REPEAT). One textureLod() replaces the
// 3-octave noise2() weighted sum the fog density + godray probe
// loops used to compute per-tap. UV maps q→texture as `q / 16.0` so
// the texture tiles every 16 q-units (≈800 m of world space).
layout(set = 0, binding = 14) uniform sampler2D u_fog_wisp;
#ifndef QLIKE_WATER
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
#endif /* ! QLIKE_WATER */
const float kHeightmapSide = 2048.0;
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
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
    return textureLod(u_terrain_height, uv, 0.0).r;
}

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
// Mask 0x02 = "anything except the terrain BLAS instance".
bool any_hit_no_terrain(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT,
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
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT,
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
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT,
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
                          gl_RayFlagsTerminateOnFirstHitEXT,
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
layout(set = 0, binding = 31) uniform sampler2D u_sun_horizon;   // far-mountain sun shadow
// Atmosphere sky-view LUT (atmosphere.comp; GENERAL layout).
layout(set = 0, binding = 30) uniform sampler2D u_skyview;

// Material texture arrays (same set-0 slots as cube.frag): only the castle
// road cobbles are sampled here - the rest of this terrain is procedural.
const int kRmTextureCount = 19;   // = cube.frag kTextureCount
const int kRmMatRoad = 14;        // = cube.frag kMatRoad (PavingStones131)
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

#ifndef QLIKE_WATER
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
#endif /* ! QLIKE_WATER */
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
float distanceFogAmount(vec3 p_world, vec3 cam) {
    float strength = scene.distance_fog_color.a;
    if (strength < 1e-3) return 0.0;
    float density   = clamp(scene.distance_fog_params.x, 0.0, 0.1);
    float start_d   = scene.distance_fog_params.y;
    float height_top = scene.distance_fog_params.z;
    float max_alpha  = scene.distance_fog_params.w;
    float d_raw = max(0.0, length(p_world - cam) - start_d);
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
void main() {
#ifndef QLIKE_WATER
    // Ray reconstruction: vWNear / vWFar were computed in the vert
    // shader (pc.model × ndc-near / ndc-far) and interpolated. Per
    // pixel we just need the homogeneous divide + subtract + normalize
    // — the matrix multiplies that used to dominate this stub are gone.
#else /* QLIKE_WATER */
    // World position of the water surface at this pixel comes directly
    // from the rasterised plane's vertex interpolation (vWPos). The old
    // inv(view_proj) reconstruction (`pc.model * ndc...`) drifted by
    // µm-mm per frame as camera moved — the user saw this as the
    // shoreline orientation shifting with every camera move. Perspective-
    // correct interp of fixed vertex positions is bit-stable per camera
    // pose and matches the rasterised depth exactly. Ray dir & ray-t
    // are derived from (vWPos - camera) so reflection helpers below see
    // the same world geometry.
#endif /* QLIKE_WATER */
    vec3 ro = scene.camera_pos.xyz;
#ifndef QLIKE_WATER
    vec3 rd = normalize(vWFar.xyz / vWFar.w - vWNear.xyz / vWNear.w);
#else /* QLIKE_WATER */
    vec3 _to_surf = vWPos - ro;
    vec3 rd = normalize(_to_surf);
#endif /* QLIKE_WATER */

    // Water plane intersection FIRST — its `t` gives us a max distance
    // for the heightfield march: terrain past the water plane can't be
    // visible above water anyway (water is opaque-ish). Used as the
    // raymarch's t_max so far rays aren't wasted on terrain hidden
    // beneath water.
    bool water_on = scene.water_params.x > 0.5;
    float water_y = scene.water_params.y;
#ifndef QLIKE_WATER
    float t_water_plane = -1.0;
    if (water_on && abs(rd.y) > 1e-4) {
        float tw = (water_y - ro.y) / rd.y;
        if (tw > 0.0) t_water_plane = tw;
    }
#else /* QLIKE_WATER */
    // Distance to the water plane is just |vWPos - camera|. Replaces the
    // analytic ray-vs-plane intersection (which was algebraically equal
    // but numerically drifted with inv_vp, same root cause as wpos drift).
    float t_water_plane = water_on ? length(_to_surf) : -1.0;
#endif /* QLIKE_WATER */

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
#ifndef QLIKE_WATER
    bool water_only = scene.terrain_local_info.x > 0.5;
#else /* QLIKE_WATER */
    // This shader is ONLY ever bound by the rasterised water-plane
    // pipeline, so water-only is unconditional — the mesh terrain
    // already owns the ground this frame; we just shade water.
    bool water_only = true;
#endif /* QLIKE_WATER */
    // The FBM heightfield IS the mesh terrain (the CDLOD mesh is baked
    // from this exact noise), so the water's FBM-driven shore depth,
    // showthrough and reflection all agree with the rasterised ground.
    // We still run the real terrain march here so water shades ONLY
    // where it is genuinely the closest surface — identical cost and
    // behaviour to the proven-stable FBM path (no t=-1 overdraw TDR).
    // In water-only mode the FBM terrain itself is never SHADED (the
    // mesh owns the ground); terrain-hit pixels are discarded below so
    // the rasterised mesh shows through.
#ifndef QLIKE_WATER
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
    float t = water_only ? raymarchMesh(ro, rd, t_water_plane)
                         : raymarch(ro, rd, t_stop);
    if (obj_stop && t < 0.0) discard;      // an object owns this pixel

    // Resolve water vs terrain ordering for the rest of the shader.
    float t_water = -1.0;
    if (t_water_plane > 0.0 && (t < 0.0 || t_water_plane < t)) {
        t_water = t_water_plane;
    }

    if (t < 0.0 && t_water < 0.0) {
        // Sky — let the existing compose-pass sky handle this pixel.
#else /* QLIKE_WATER */
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
#endif /* QLIKE_WATER */
        discard;
    }

    // === Water surface shading ===========================================
    if (t_water > 0.0) {
#ifndef QLIKE_WATER
        vec3 wpos = ro + rd * t_water;
#else /* QLIKE_WATER */
        // Hardware depth-test (depth_test ON in setup.cpp) handles
        // "is terrain in front?" occlusion. wpos comes straight from
        // vWPos — perspective-correct world position from the plane
        // mesh vertices, no per-frame inv_vp drift, no need to snap.
        vec3 wpos = vWPos;
#endif /* QLIKE_WATER */
        float wave_str = scene.water_params.z;
        float wave_t   = scene.water_params.w;
        // wave-frequency multiplier — packed in water_color_shallow.w
        // by update_scene_ubo. Defaults to 1.0; the slider lets the
        // user dial finer ripples or bigger sweeping waves.
        float wave_scale = max(0.05, scene.water_color_shallow.w);
        vec3 wnor;
        float pm_foam_tex = 1.0;  // 1 = no extra foam; <1 darkens
        // Style 0 = engine default (directional sum-of-sines); 1 = P_Malin
        // river style (advected FBM + flow-driven foam). Packed in
        // water_foam_params.y as an int.
        int water_style = int(scene.water_foam_params.y + 0.5);
        if (water_style >= 1) {
            // Full P_Malin GetFlowRate — flow vector + foam amount in
            // one pass over the local depth field.
            vec3 flow_and_foam = pm_get_flow_rate(wpos.xz,
                                                   scene.water_params.y,
                                                   t_water);
            vec2 flow_rate = flow_and_foam.xy;
            float pm_foam_amount = flow_and_foam.z;

            // User knobs:
            //   water_foam_params.z = river flow speed (legacy slot)
            //   water_foam_params.w = river normal strength
            //   water_river_extra.x = flow direction angle (radians)
            //   water_river_extra.y = animation time speed
            //   water_river_extra.z = detail scale (UV multiplier)
            //   water_river_extra.w = foam amount multiplier
            float user_speed      = max(0.05, scene.water_foam_params.z);
            float user_normal_str = max(0.05, scene.water_foam_params.w);
            float flow_angle      = scene.water_river_extra.x;
            float time_speed      = scene.water_river_extra.y;
            float detail_scale    = scene.water_river_extra.z;
            float foam_amount     = scene.water_river_extra.w;

            // Rotate the flow vector by the user's flow_angle so the
            // user can pick any downstream direction. Default 0 keeps
            // P_Malin's +X bias.
            float ca = cos(flow_angle);
            float sa = sin(flow_angle);
            flow_rate = mat2(ca, sa, -sa, ca) * flow_rate * user_speed;
            pm_foam_amount *= foam_amount;

            // Time + UV scale are user-driven so the user can pick the
            // animation speed and the ripple density independently.
            float wave_time = wave_t * 0.4 * time_speed;
            vec2 uv_n = wpos.xz * wave_scale * detail_scale;
            // Pre-foam scaling for the FBM advection's gradient ascent.
            float foam_bias = clamp((pm_foam_amount - 0.2) * 1.5, 0.0, 1.0);
            foam_bias = foam_bias * foam_bias * 0.5;
            vec4 nf = pm_flowing_normal(uv_n, flow_rate, foam_bias, wave_time);
            float n_str = wave_str * user_normal_str;
            wnor = normalize(vec3(nf.x * n_str, 1.0, nf.z * n_str));
            // P_Malin's foam combines the per-pixel turbulence pattern
            // with the global foam amount: 1 - pow(foam_tex, foam_amt*5).
            float foam_tex = pm_foam_pattern(uv_n, flow_rate, wave_time);
            float foam_t = max(0.0, (foam_tex - 0.2) / 0.2);
            foam_tex = pow(0.5, foam_t);
            pm_foam_tex = pow(max(0.001, foam_tex), foam_bias * 5.0);
        } else if (water_style == 2) {
            // Lake style — distance-faded bumps, glassy at distance.
            // water_river_extra reused for tuning when style == 2:
            //   .x = (unused, kept compatible with river)
            //   .y = time speed
            //   .z = uv scale multiplier
            //   .w = (unused)
            // water_foam_params.w = bump strength (legacy slot, shared)
            // water_river_extinct.w * 100 = bump-fade distance (metres)
            float time_speed   = max(0.0, scene.water_river_extra.y);
            float uv_scale     = max(0.05, scene.water_river_extra.z);
            float bump_strength = max(0.0, scene.water_foam_params.w);
            float fade_dist    = max(5.0, scene.water_river_extinct.w * 100.0);
            wnor = lakeWaterNormal(wpos.xz, uv_scale * wave_scale,
                                    bump_strength * wave_str,
                                    wave_t * time_speed,
                                    t_water, fade_dist);
        } else {
            wnor = waterNormal(wpos.xz, wave_t, wave_str, wave_scale);
        }
        // Splash rings bend the surface normal.
        vec3 rip_w = water_ripples(wpos.xz);
        wnor = normalize(vec3(wnor.x - rip_w.x, wnor.y, wnor.z - rip_w.y));
        vec3 sunDirW = scene.sun_direction.xyz;
        // Camera below the surface: the water is seen from underneath.
        // Inside Snell's window (within ~48.6 deg of straight up) the sky
        // is refracted in, squeezed into a bright disc; outside it the
        // surface is a mirror of the depths (total internal reflection).
        // The waves (exaggerated: a small tilt moves the window edge a
        // lot) make the edge dance.
        bool cam_under = ro.y < water_y;
        vec3 under_col = vec3(0.0);
        if (cam_under) {
            vec3 nd = normalize(vec3(-wnor.x * 2.5, -wnor.y, -wnor.z * 2.5));
            vec3 rf = refract(rd, nd, 1.333);
            float sun_up = clamp(sunDirW.y, 0.0, 1.0);
            // The mirror shows the water column below: the same veil colour
            // compose.frag fades everything into under water.
            vec3 deep = vec3(0.07, 0.36, 0.44) *
                        (scene.sun_color.rgb * scene.sun_color.a * 0.22 * clamp(sun_up * 1.4, 0.0, 1.0) +
                         scene.sky_color.rgb * 0.30);
            if (dot(rf, rf) < 1e-6) {
                under_col = deep;
            } else {
                float ct = clamp(rf.y, 0.0, 1.0);
                float om = 1.0 - ct;
                float om2 = om * om;
                float F = 0.02 + 0.98 * om2 * om2 * om;
                vec3 sky_u = sample_sky_atmosphere(rf);
                float sdu = max(dot(rf, sunDirW), 0.0);
                float sd16 = pow(sdu, 16.0);
                sky_u += scene.sun_color.rgb * scene.sun_color.a *
                         (pow(sd16, 48.0) * 25.0 + sd16 * sd16 * 0.5);
                under_col = mix(sky_u, deep, F);
            }
        }

        // Schlick fresnel — F0 = 0.02 for water. Looking grazing-on
        // means full reflection; looking straight down в‰€ 2 % reflection.
        // pow(x, 5) → 3 multiplies (x²·x²·x). Capped at 0.88 so even
        // grazing-angle water keeps a sliver of water colour and
        // doesn't read as a pure-sky-blue line where the surface meets
        // the shore at distance.
        float cosV = clamp(-dot(rd, wnor), 0.0, 1.0);
        const float F0 = 0.02;
        float oneMinusV = 1.0 - cosV;
        float fv2 = oneMinusV * oneMinusV;
        float fv5 = fv2 * fv2 * oneMinusV;
        float fres = min(0.88, F0 + (1.0 - F0) * fv5);

        vec3 refl = reflect(rd, wnor);
        // Reflections lose sharpness with distance: one pixel of far water
        // covers many wavelets, each mirroring a slightly different
        // direction, and what it shows is the average - mostly a vertical
        // smear, as the wave slopes face the viewer. The reflected ray is
        // spread at random per pixel and frame (more with distance); the
        // temporal filter integrates that into a soft, stretched reflection.
        {
            float rdist = distance(wpos, ro);
            float spread = 0.004 + 0.085 * smoothstep(8.0, 260.0, rdist);
            vec2 rs = fract(sin(vec2(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)),
                                     dot(gl_FragCoord.xy, vec2(39.3468, 11.135))) +
                                float(scene.rt_flags.w & 63) * 1.618) * 43758.5453) - 0.5;
            vec3 side = normalize(cross(refl, vec3(0.0, 1.0, 0.0)) + vec3(1e-5));
            refl = normalize(refl + vec3(0.0, 1.0, 0.0) * (rs.y * spread * 2.2) + side * (rs.x * spread * 0.7));
            refl.y = max(refl.y, 0.004);
        }
        // Sky reflection — horizon→zenith gradient + sun halo. Used
        // directly when the reflection ray escapes the terrain or
        // when RT reflections are off.
        float skyT = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
        // Richer 3-stop sky: deep zenith → mid blue → bright hazy
        // horizon, then tinted toward the scene's actual sky colour so
        // the mirror matches the sky overhead. horizonGlow concentrates
        // brightness right at the waterline where real water reflections
        // are strongest — this is what brings the "nice" look back
        // without any FBM terrain in the reflection.
        float hb = 1.0 - abs(refl.y);
        hb *= hb; hb *= hb;                 // ≈ pow(1-|y|, 4) — horizon band
        vec3 cZenith = vec3(0.20, 0.34, 0.62);
        vec3 cMid    = vec3(0.42, 0.56, 0.78);
        vec3 cHaze   = vec3(0.80, 0.84, 0.88);
        vec3 sky_refl = mix(cMid, cZenith, skyT);
        sky_refl = mix(sky_refl, cHaze, hb * 0.65);
        sky_refl = mix(sky_refl, scene.sky_color.rgb, 0.22);
        // Physical sky on: mirror the REAL sky (time of day, dusk, night,
        // overcast grey) instead of the fixed daylight gradient.
        if (scene.atmo_params.x > 0.5) {
            vec3 rs = normalize(vec3(refl.x, max(refl.y, 0.03), refl.z));
            sky_refl = phys_sky(rs);
            float cov_w = scene.crater_info.y;
            sky_refl = mix(sky_refl, vec3(dot(sky_refl, vec3(0.2126, 0.7152, 0.0722))) * 1.05, cov_w * cov_w);
        }
        // pow(x, 60) ≈ pow(x, 64) = 6 squarings; visually identical
        // for a sun-halo lobe at this exponent. Reuses sh2/sh4 below for
        // the broad-glow term (was redundantly recomputing sg2 = sh1*sh1
        // → sg4 = sg2*sg2 right after this block — same values).
        float sh1 = max(dot(refl, sunDirW), 0.0);
        float sh2 = sh1 * sh1; float sh4 = sh2 * sh2;
        float sh8 = sh4 * sh4; float sh16 = sh8 * sh8;
        float sh32 = sh16 * sh16; float sunHalo = sh32 * sh32;
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sunHalo * 0.6;
        // Broad, soft sun glow (≈pow(.,4)) on top of the tight halo —
        // gives the wide specular sheet that made the old reflection
        // read as "lit water" rather than a flat gradient.
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sh4 * 0.30;

        vec3 reflCol = sky_refl;
        float refl_terrain_t = 1e9;     // terrain hit along the reflection (objects behind it stay hidden)
        // RT-style FBM-march reflection — picks up the actual mountain
        // / valley silhouettes in the water surface. Cheap because
        // we use a reduced-quality march; if it misses (refl ray
        // exits past the terrain or goes into the sky) fall back to
        // the analytical sky.
        bool do_rt_refl = scene.water_params.x > 0.5 &&
                          pc.grass_params.w > 0.5 && refl.y > 0.02;
        if (do_rt_refl) {
            // Lift the origin slightly along refl so we don't
            // immediately self-intersect the wave bumps below the
            // water plane.
            vec3 r_ro = wpos + refl * 0.5 + vec3(0.0, 0.05, 0.0);
            // Mesh-terrain mode reflects the REAL mesh surface (height-
            // map march + heightmap normal); FBM mode keeps the
            // procedural march. All the shading below (material, grass
            // tint, extinction, fog) is FBM-free and shared.
            float t_r = water_only ? raymarchReflectMesh(r_ro, refl)
                                   : raymarchReflectHM(r_ro, refl);
            if (t_r > 0.0) {
                refl_terrain_t = t_r;
                vec3 r_pos = r_ro + refl * t_r;
                vec3 r_nor = water_only ? meshNormal(r_pos.xz)
                                        : calcNormal(r_pos, t_r);
                // Lambert + sky ambient. No shadow march in the
                // reflection — too expensive for a free-running
                // surface effect, and the missing self-shadow is
                // visually fine on a wavy reflection.
                float r_dif = max(dot(r_nor, sunDirW), 0.0);
                float r_amb = 0.5 + 0.5 * r_nor.y;
                vec3  r_mat = getMaterial(r_pos, r_nor);
                vec3  r_lin = r_dif * scene.sun_color.rgb *
                                scene.sun_color.a +
                              r_amb * scene.sky_color.rgb * 0.75;
                vec3  r_col = r_mat * r_lin;
                // Grass tint boost in reflections — getMaterial already
                // includes a grass-eligibility tint, but reflections
                // see slopes from a low angle where the underlying rock
                // shading dominates. Re-blend the configured grass
                // ground colour at full strength when the hit point is
                // grass-eligible so the user sees their painted grass
                // mirrored in the water.
                float r_grass = grassEligibility(r_pos.xz, r_pos.y);
                if (r_grass > 0.05) {
                    // Distance-blend close↔far ground tint (mirrors
                    // getMaterial's primary-path ramp at line ~1077)
                    // so reflected far-distance terrain matches the
                    // surrounding land's tint instead of staying in
                    // the bright close-tint band.
                    //
                    // Distance must be the LIGHT-PATH length the
                    // viewer's eye actually traverses — camera → water
                    // surface → reflected hit. Straight-line camera→
                    // r_pos can be much shorter (think looking down
                    // at water from a hill: the bounce angle stretches
                    // the perceived distance ~2×). That mismatch is
                    // why the reflected mountains were reading in the
                    // close tint while the same mountains seen
                    // directly were already in the far band.
                    float r_path_d = distance(scene.camera_pos.xyz, wpos) +
                                     distance(wpos, r_pos);
                    float tint_far_r = max(50.0, scene.grass_color_ground_far.w);
                    float r_tint_t = smoothstep(0.0, tint_far_r, r_path_d);
                    vec3 r_ground_base = mix(scene.grass_color_ground.rgb,
                                              scene.grass_color_ground_far.rgb,
                                              r_tint_t);
                    // Same noise-darkening the primary path applies in
                    // getMaterial (line ~1081):
                    //   grass_top = mix(ground_base, ground_base*0.7, nz)
                    // — averages ~15 % darker than raw ground_base.
                    // Without it the reflection reads visibly brighter
                    // than the surrounding land at matching distance.
                    float r_nz = noise2(r_pos.xz * 0.04) *
                                  noise2(r_pos.xz * 0.005);
                    vec3 r_grass_top = mix(r_ground_base,
                                            r_ground_base * 0.7,
                                            r_nz);
                    vec3 g_col = r_grass_top * r_lin;
                    r_col = mix(r_col, g_col, r_grass * 0.7);
                }
                // Same atmospheric extinction as the primary
                // shading, applied to the reflection-ray distance.
                vec3  r_ext = exp(-t_r * 0.00025 *
                                  vec3(1.0, 1.3, 2.0));
                // pow(x, 8) → 3 squarings.
                float rf1 = max(dot(refl, sunDirW), 0.0);
                float rf2 = rf1 * rf1; float rf4 = rf2 * rf2; float rf8 = rf4 * rf4;
                vec3  r_fog = scene.crater_info.w * mix(vec3(0.55, 0.55, 0.58),
                                  vec3(1.0, 0.7, 0.3),
                                  0.3 * rf8);
                r_col = r_col * r_ext + r_fog * (1.0 - r_ext);
                reflCol = r_col;
            }
            // miss → keep sky_refl
        }

        // TLAS reflection — ray-query against castle / cubes / dyn
        // props (mask 0x02 skips the terrain BLAS). If the TLAS hit
        // is closer than whatever's in `reflCol` (FBM-march or sky),
        // override with a basic shaded stand-in. We don't do per-
        // instance material lookup here to keep the path cheap; a
        // mid-grey lit by sun + sky reads correctly for most stone /
        // wood / box surfaces and reflections in water are too low-
        // detail for the user to notice the missing texture.
        // In mesh-terrain (water-only) mode the FBM reflection march is
        // gone, so the TLAS reflection is the only geometry reflected.
        // Drive it from the RT-reflections toggle the user already has
        // on (pc.grass_params.w) so enabling reflections "just works"
        // there, in addition to the dedicated water-TLAS toggle.
        bool tlas_refl_on = scene.water_shore.z > 0.5 ||
                            (water_only && pc.grass_params.w > 0.5);
        if (tlas_refl_on && refl.y > 0.02) {
            vec3 r_ro = wpos + wnor * 0.05;
            // TLAS (castle, props, trees, vehicles, jets) plus the voxel
            // tower, which is not in the TLAS and needs its own march.
            // Same code for the raymarched-terrain water and the water
            // plane of the mesh terrain (the former used to skip the voxel
            // march and stop at 200 m: no tower, no aircraft in the lake).
            const float kReflRange = 900.0;
            float t_tlas = kReflRange;
            float t_vox  = kReflRange;
            uint  vox_pal = 0u;
            int   r_inst = 0, r_prim = 0;
            bool  tlas_hit = closest_hit_material_no_terrain(r_ro, refl, min(kReflRange, refl_terrain_t), t_tlas, r_inst, r_prim);
            bool  vox_hit  = water_vox_march(r_ro, refl,
                                              min(tlas_hit ? t_tlas : 200.0, min(200.0, refl_terrain_t)),
                                              t_vox, vox_pal);
            bool  use_vox  = vox_hit && (!tlas_hit || t_vox < t_tlas);
            if (tlas_hit || vox_hit) {
                float t_use = use_vox ? t_vox : t_tlas;
                // Face normal: the axis-aligned face opposing the ray's
                // dominant axis (exact for voxels / brushes, a stable
                // stand-in for everything else).
                vec3 hit_pos = r_ro + refl * t_use;
                vec3 ar = abs(refl);
                vec3 r_nor;
                if (ar.x > ar.y && ar.x > ar.z)
                    r_nor = vec3(refl.x > 0.0 ? -1.0 : 1.0, 0.0, 0.0);
                else if (ar.y > ar.z)
                    r_nor = vec3(0.0, refl.y > 0.0 ? -1.0 : 1.0, 0.0);
                else
                    r_nor = vec3(0.0, 0.0, refl.z > 0.0 ? -1.0 : 1.0);
                r_nor = normalize(mix(r_nor, vec3(0.0, 1.0, 0.0), 0.25));
                // Only near hits get a sun-shadow ray, and it dims rather than
                // blacks out (a reflected forest is mostly leaves in half light).
                float sun_vis = (t_use < 120.0 && any_hit_shadow_caster(hit_pos + r_nor * 0.08, sunDirW, 200.0)) ? 0.35 : 1.0;
                float r_dif = (0.35 + 0.65 * max(dot(r_nor, sunDirW), 0.0)) * sun_vis;
                float r_amb = 0.5 + 0.5 * r_nor.y;
                vec3 stone = vec3(0.55, 0.52, 0.48);
                // For voxel hits use the actual palette colour so the
                // tower's sandstone/wood reads back in the water; TLAS
                // hits stick with the generic stone fallback (cheap).
                vec3 albedo = stone;
                if (tlas_hit && !use_vox) {
                    // The hit instance's own colour (trees green, castle stone,
                    // vehicles) instead of one grey for everything.
                    int r_mi = (r_inst == kStaticBlasSentinel) ? (r_prim / kCubeTrisPerBox) : r_inst;
                    albedo = clamp(materials[r_mi].color.rgb, vec3(0.02), vec3(1.0));
                    if (materials[r_mi].tex.x >= 0.0) albedo *= vec3(0.55, 0.52, 0.48);
                }
                if (use_vox) {
                    if      (vox_pal == 1u) albedo = vec3(0.95, 0.78, 0.45);
                    else if (vox_pal == 2u) albedo = vec3(0.75, 0.30, 0.15);
                    else                    albedo = mix(vec3(0.85, 0.55, 0.30),
                                                        vec3(0.45, 0.22, 0.12),
                                                        float(max(int(vox_pal) - 3, 0)) / 12.0);
                }
                vec3 r_col = albedo * (r_dif * scene.sun_color.rgb *
                                       scene.sun_color.a +
                                       r_amb * scene.sky_color.rgb * 0.75);
                // Atmospheric extinction over reflection distance.
                vec3 r_ext = exp(-t_use * 0.00025 *
                                  vec3(1.0, 1.3, 2.0));
                // pow(x, 8) → 3 squarings — same trick as the rest
                // of the file; this water-TLAS-reflection branch was
                // the last stray pow.
                float rfL1 = max(dot(refl, sunDirW), 0.0);
                float rfL2 = rfL1 * rfL1;
                float rfL4 = rfL2 * rfL2;
                float rfL8 = rfL4 * rfL4;
                vec3 r_fog = scene.crater_info.w * mix(vec3(0.55, 0.55, 0.58),
                                  vec3(1.0, 0.7, 0.3),
                                  0.3 * rfL8);
                r_col = r_col * r_ext + r_fog * (1.0 - r_ext);
                reflCol = r_col;
            }
        }

        // Shore-aware base colour. We compute the actual water depth
        // at this surface point — water_level minus the FBM terrain
        // height directly below. Shallow water = `water_color_shallow`,
        // deep water = `water_color`, transition over `shore_blend`
        // metres. Mild noise on the depth breaks up the perfectly
        // geometric shoreline so it reads as natural beach.
        vec3 deep    = scene.water_color.rgb;
        vec3 shallow_tint = scene.water_color_shallow.rgb;
        float shore_blend = max(0.1, scene.water_shore.x);
        float shore_noise = scene.water_shore.y;
#ifdef QLIKE_WATER
        // wpos.xz was already 1 cm-snapped at the source (line ~1692)
        // so everything in this block — terrain_y, foam, shore noise,
        // wave normals — sees TAA-jitter-stable coordinates.
#endif /* QLIKE_WATER */
        float terrain_y = water_only ? meshHeight(wpos.xz)
                                      : terrainM_lod(wpos.xz, t_water);
        // Noise-free depth — the foam band uses this so the highlight
        // follows the actual shoreline geometry instead of wobbling
        // with the shore-noise. depth_m below adds the noise on top
        // for the wider shallow→deep tint transition only.
        float depth_clean = max(0.0, water_y - terrain_y);
        float depth_m   = depth_clean;
#ifndef QLIKE_WATER
        float dn = noise2(wpos.xz * 0.18) +
                    0.5 * noise2(wpos.xz * 0.42);
#else /* QLIKE_WATER */
        // Lower-frequency shore noise: the old metre-scale octaves made
        // the water/land boundary visibly crawl when the camera moved
        // along the shore. A slow meander reads as a natural coastline
        // without the swim.
        float dn = noise2(wpos.xz * 0.10) +
                    0.35 * noise2(wpos.xz * 0.26);
#endif /* QLIKE_WATER */
        depth_m += (dn - 0.6) * shore_blend * shore_noise;
        depth_m  = max(0.0, depth_m);
#ifndef QLIKE_WATER
        float water_blend = smoothstep(0.0, shore_blend, depth_m);
#else /* QLIKE_WATER */
        // Screen-space anti-alias the shore transition. A fixed
        // metre-width smoothstep collapses to <1 px at grazing angles
        // / distance, so the shoreline aliases and jumps frame-to-
        // frame as the camera moves. Never let the edge be narrower
        // than ~2.5 px of depth gradient.
        float sb_aa = max(shore_blend, fwidth(depth_m) * 2.5);
        float water_blend = smoothstep(0.0, sb_aa, depth_m);
#endif /* QLIKE_WATER */
        // P_Malin-style underwater extinction when river style is on.
        // exp2(-depth * ext_col) gives the characteristic green-river
        // look (red attenuates fastest, then green, blue lasts). Mixed
        // against the user's deep tint at 60% so the slider still has
        // colour authority.
        if (water_style >= 1) {
            // User-tunable extinction colour + density (defaults to
            // P_Malin's (0.5, 0.4, 0.1) × 3.0).
            vec3  ext_col = scene.water_river_extinct.rgb;
            float ext_den = scene.water_river_extinct.w * 3.0;
            vec3 ext = exp2(-depth_m * ext_col * ext_den);
            deep = mix(deep, deep * ext + vec3(0.05, 0.08, 0.06) * (1.0 - ext.r), 0.6);
        }
        vec3  shallow = mix(shallow_tint, deep, water_blend);
        // Distance fade to the deep tint so far water doesn't read
        // shallower than near water just because of the noise.
        float distFade = 1.0 - exp(-t_water * 0.012);
        shallow = mix(shallow, deep, distFade * 0.4);

        // Specular highlight on the wave crests when the sun is close
        // to the half-vector. Sharp lobe (~64) so it reads as glints.
        vec3 H = normalize(sunDirW - rd);
        // pow(x, 64) → 6 successive squarings.
        float sp1 = max(dot(wnor, H), 0.0);
        float sp2 = sp1 * sp1; float sp4 = sp2 * sp2;
        float sp8 = sp4 * sp4; float sp16 = sp8 * sp8;
        float sp32 = sp16 * sp16; float spec = sp32 * sp32;

        // Sun shadow on the water surface — bound-checked sample of
        // the sun shadow map (binding 7), already populated from
        // terrain chunks + castle + dyn-props. Dims the specular
        // (no glints in shadow) and slightly darkens the base tint.
        // Combined with calcShadow against the FBM so distant terrain
        // (mountains beyond the shadow-map cascade) also casts on
        // water — without it, water within sight of a tall ridge stayed
        // fully lit while the land beside it was correctly shadowed.
        float water_lit = 1.0;
        if (scene.water_color.w > 0.5) {
            vec4 lc = scene.light_vp * vec4(wpos, 1.0);
            vec3 lndc = lc.xyz / lc.w;
            if (lndc.z >= 0.0 && lndc.z <= 1.0) {
                vec2 luv = lndc.xy * 0.5 + 0.5;
                if (all(greaterThanEqual(luv, vec2(0.0))) &&
                    all(lessThanEqual(luv, vec2(1.0)))) {
                    const float kRecvBias = 0.00005;
                    water_lit = textureLod(u_sun_shadow_map,
                                            vec3(luv, lndc.z - kRecvBias), 0.0);
                }
            }
            // FBM self-shadow — terrain casts on the water surface.
            // Heavy: calcShadow does up to 32 FBM steps per call. Three
            // gates needed to keep this cheap from a high vantage where
            // water can fill 50 %+ of the screen (a 1080p shot then
            // fires 1M+ extra ray-style marches per frame and the
            // single submit can blow past Windows' 2 s TDR watchdog —
            // a confirmed device-lost repro from Y≈47 over the bay).
            //   1. Skip back-facing / grazing water (n·L > 0.05).
            //   2. Skip if the cascade shadow map already darkened the
            //      pixel below 0.95 — adding the FBM term won't visibly
            //      change a pixel that's already in shadow.
            //   3. Skip past 80 m view distance — the shadow-map
            //      cascade already covers everything within its frustum
            //      and beyond that the FBM detail is sub-pixel.
#ifndef QLIKE_WATER
            float w_dist = length(wpos - scene.camera_pos.xyz);
#else /* QLIKE_WATER */
            // t_water_plane (line 1765) already holds the camera-to-surface
            // distance — reuse it instead of recomputing a length() here.
            float w_dist = t_water_plane;
#endif /* QLIKE_WATER */
            if (water_lit > 0.95 && w_dist < 80.0 &&
                dot(wnor, sunDirW) > 0.05) {
                float self_lit = water_only
                    ? calcShadowMesh(wpos + wnor * 0.05, sunDirW)
                    : calcShadow(wpos + wnor * 0.05, sunDirW);
                water_lit = min(water_lit, self_lit);
            }
#ifndef QLIKE_WATER
            // Explicit TLAS shadow ray — catches brushes/dyn-props that
            // may not be in the sun shadow map cascade. Mask 0x01 = all
            // shadow casters (terrain + brushes + dyn-props).
            if (water_lit > 0.05 && dot(wnor, sunDirW) > 0.05) {
                if (any_hit_shadow_caster(wpos + wnor * 0.05,
                                          sunDirW, 200.0)) {
                    water_lit = 0.0;
                }
            }
            // Voxel castle is NOT in the TLAS (it's rendered via SSBO +
            // DDA, separate pipeline). The TLAS ray above misses it.
            // Use the shape's AABB as a cheap conservative occluder —
            // hits the "is the sun behind the castle?" question
            // correctly even if it can't tell which interior voxel is
            // blocking. Good enough for water glint suppression.
            if (water_lit > 0.05 && scene.voxel_origin.w > 0.5 &&
                dot(wnor, sunDirW) > 0.05) {
                vec3 bmin = scene.voxel_origin.xyz;
                vec3 bmax = bmin + scene.voxel_dims.xyz;
                if (ray_aabb_hit(wpos + wnor * 0.05, sunDirW,
                                 bmin, bmax)) {
                    water_lit = 0.0;
                }
            }
#endif /* ! QLIKE_WATER */
        }

        // Underwater showthrough — for shallow water we evaluate the
        // terrain at wpos.xz and pass its surface tint through the
        // water tinted by depth. Beer's-law-style attenuation:
        // T = exp(-depth * absorption) so deep water still hides
        // the bottom while shallow water reads transparent.
        vec3 baseTint = shallow;     // current shore-blended water colour
#ifdef QLIKE_WATER
        // Transparency / water clarity. The slider now controls how far
        // light penetrates: 0 = murky (bottom hidden within ~0.5 m),
        // 1 = very clear (seabed visible many metres down). Crucially
        // SHALLOW water is strongly see-through whenever clarity > ~0.2,
        // which is what reads as real shallow water over a sandy bottom.
#endif /* QLIKE_WATER */
        float trans_amt = scene.water_shore.w;
        if (trans_amt > 0.001) {
#ifndef QLIKE_WATER
            // Reuse the depth_m we already computed for the shore
            // blend. Absorption coefficient picks ~50 % attenuation
            // at a 2 m depth, ~95 % by 6 m.
            float absorp = 0.5;
#else /* QLIKE_WATER */
            // Two decoupled controls:
            //   trans_amt (Transparency slider) = master see-through
            //     strength: 0 = always opaque, 1 = full clarity.
            //   pc.color.y (Water clarity depth, m) = the depth at
            //     which water goes ~opaque, INDEPENDENT of strength.
            //     Small → only very shallow water is clear, deep stays
            //     opaque (what real shallow water looks like). Large →
            //     see far down.
            // absorp chosen so Tw = exp(-3) ≈ 0.05 (≈opaque) exactly at
            // `clarity_depth` metres; shallower water is clear.
            float clarity_depth = max(0.1, pc.color.y);
            float absorp = 3.0 / clarity_depth;
#endif /* QLIKE_WATER */
            float Tw = exp(-depth_m * absorp);
            // Visibility gate: skip the entire underwater shading
            // (calcNormal = 1× FBM, getMaterial = several `noise2`
            // taps, lambert + ambient) when the contribution is
            // below ~5 % of the surface tint. At 6 m depth Tw drops
            // to 0.05; deep-water pixels read as opaque tint anyway.
#ifndef QLIKE_WATER
            if (Tw * trans_amt > 0.05) {
#else /* QLIKE_WATER */
            if (Tw > 0.04) {
#endif /* QLIKE_WATER */
                // Underwater surface — terrain material at the under-
                // water point, lambert-shaded by sun (with shadow), no
                // shore noise so the look is calm.
                vec3 u_pos = vec3(wpos.x, terrain_y, wpos.z);
                vec3 u_nor = water_only ? meshNormal(u_pos.xz)
                                        : calcNormal(u_pos, t_water);
                vec3 u_mat = getMaterial(u_pos, u_nor);
                float bed_spec = 0.0;
                {
                    float bfp = max(t_water, 0.05) * 2.0 * scene.viewport.w;      // ~metres per pixel
                    float bwA = smoothstep(2.0, 5.0, 0.22 / bfp);
                    float bwB = smoothstep(2.0, 5.0, 0.075 / bfp);
                    if (bwA > 0.01) {
                        // Where the bed is seen through the water it lies a little
                        // further along the line of sight than straight below.
                        vec2  bp = u_pos.xz + rd.xz / max(-rd.y, 0.25) * min(depth_m, 1.5) * 0.6;
                        float be = max(0.004, 1.5 * bfp), bid, bk, tmp1, tmp2;
                        float hR = bed_h(bp + vec2(be, 0.0), bwA, bwB, tmp1, tmp2);
                        float hU = bed_h(bp + vec2(0.0, be), bwA, bwB, tmp1, tmp2);
                        float hC = bed_h(bp, bwA, bwB, bid, bk);
                        u_nor = normalize(u_nor + vec3(hC - hR, 0.0, hC - hU) / be * 0.9);
                        // Crevices between the stones are dark; stone tops stay open.
                        float crev = smoothstep(0.0, 0.045, hC) * smoothstep(0.0, 0.45, bk);
                        float gl = max(dot(u_mat, vec3(0.3, 0.5, 0.2)), 0.03);
                        vec3 sc = gl * mix(vec3(1.35, 1.15, 1.05), vec3(1.05, 1.0, 0.95), fract(bid * 5.3)) *
                                  mix(0.55, 1.45, fract(bid * 23.1));
                        sc = mix(sc, gl * vec3(1.75, 1.20, 1.05), step(0.70, fract(bid * 3.1)));     // pink-brown
                        sc = mix(sc, gl * vec3(0.55, 0.55, 0.60), step(0.86, fract(bid * 9.7)));     // dark
                        sc = mix(sc, gl * vec3(2.20, 2.10, 1.95), step(0.92, fract(bid * 17.9)));    // pale
                        sc *= mix(0.85, 1.15, noise2(bp * 90.0 + bid * 40.0));
                        u_mat = mix(u_mat, mix(u_mat * 0.45, sc, smoothstep(0.0, 0.2, bk)), bwA) * mix(1.0, mix(0.10, 1.0, crev), bwA);
                        // Wet stones: a tight highlight from the sun.
                        vec3 bh = normalize(sunDirW - rd);
                        bed_spec = pow(max(dot(u_nor, bh), 0.0), 90.0) * 0.6 * crev * bwA * water_lit;
                    }
                }
                float u_dif = max(dot(u_nor, sunDirW), 0.0) * water_lit;
                float u_amb = 0.5 + 0.5 * u_nor.y;
                vec3  u_col = u_mat * (u_dif * scene.sun_color.rgb *
                                        scene.sun_color.a * 0.8 +
                                        u_amb * scene.sky_color.rgb * 0.35) +
                             bed_spec * scene.sun_color.rgb * scene.sun_color.a;
                // Tint the underwater colour by the water tint so it
                // reads as "looking through water", not "no water".
                u_col *= mix(vec3(1.0), shallow * 1.5 + vec3(0.05),
                              0.5);
#ifndef QLIKE_WATER
                // Blend: shallow → underwater visible, deep → opaque
                // water_color. trans_amt scales the maximum showthrough.
#else /* QLIKE_WATER */
                // Beer's-law transmittance (depth fade governed by the
                // clarity-depth knob) × trans_amt (master strength). At
                // Transparency=1 + small clarity depth: shallow reads
                // crystal-clear, deep water stays fully opaque.
#endif /* QLIKE_WATER */
                baseTint = mix(baseTint, u_col, Tw * trans_amt);
            }
        }

        // Damp Fresnel toward zero in shallow shore water so the
        // very-grazing-angle band where water meets land doesn't read
        // as a thin sky-blue line. water_blend is 0 right at the
        // shore (depth = 0) → 1 once the water is `shore_blend`
        // metres deep; multiplying fres by water_blend means deep
        // water keeps its full reflection while shore water shows
        // its actual tint.
        // (Was fres * water_blend: the whole light shore band then had NO
        // reflection and read as paint. Keep a good part of it - wet sand
        // under a film of water mirrors the sky too - and let the foam
        // below take over where there is surf.)
        float fres_eff = fres * mix(0.5, 1.0, water_blend);
        float foam_cover = 0.0;   // how much opaque foam sits on the surface here
        // Shore-foam band: a thin lighter highlight at the very edge
        // of the water/land boundary that fades out within
        // `water_foam_params.x` metres of noise-free depth. Reads as
        // the natural foam / surf-line you see at real shorelines.
        // Tunable via UI: water_foam_color (rgb tint + a strength)
        // and water_foam_params.x (band width). Strength = 0 disables.
#ifndef QLIKE_WATER
        if (scene.water_foam_color.a > 0.001) {
#else /* QLIKE_WATER */
        float foam_amt = 0.0;   // hoisted so the final alpha can see it
        // Foam opacity (pc.color.w) is a MASTER override: it scales
        // both the foam colour AND its alpha, so 0 = no foam at all
        // (the shallow water shows through exactly like non-foam),
        // 1 = full opaque surf. Computed here so the colour blend can
        // use it too (previously only the alpha used it, leaving the
        // white foam tint visible at 0).
        float foam_op = clamp(pc.color.w, 0.0, 1.0);
        if (scene.water_foam_color.a > 0.001 && foam_op > 0.001) {
#endif /* QLIKE_WATER */
            float foam_w = max(0.05, scene.water_foam_params.x);
            float band   = exp(-depth_clean * (4.0 / foam_w));
            // Lapping wavelets instead of a solid stripe: thin foam lines
            // that run in toward the waterline (phase = depth), wobbling
            // along the shore and breaking up into patches, plus a narrow
            // solid line right at the edge.
            {
                float lap_n  = noise2(wpos.xz * 0.35 + vec2(wave_t * 0.05, 0.0));
                float lap_ph = depth_clean * (9.0 / foam_w) + wave_t * 1.1 + lap_n * 5.0;
                float lap    = pow(0.5 + 0.5 * sin(lap_ph), 3.0);
                float patchy = smoothstep(0.25, 0.7, noise2(wpos.xz * 1.7 - vec2(0.0, wave_t * 0.12)));
                float edge   = exp(-depth_clean * (22.0 / foam_w));
                band *= max(edge, lap * mix(0.35, 1.0, patchy));
                // Foam, not paint: a lace of bubble walls with open water in
                // between (two sizes, drifting and slowly reshaping). Thick
                // foam keeps narrow holes, thin foam is mostly holes - so the
                // bed shows through it.
                if (band > 0.01) {
                    vec2  fq = wpos.xz + vec2(wave_t * 0.06, -wave_t * 0.04) + (lap_n - 0.5) * 0.25;
                    float c1 = foam_cells(fq * 3.2 + 0.15 * sin(wave_t * 0.7 + fq.yx * 2.0));
                    float c2 = foam_cells(fq * 8.5 + 7.3);
                    float thick = clamp(band * 1.6, 0.0, 1.0);
                    float wall1 = 1.0 - smoothstep(0.04 + 0.22 * thick, 0.16 + 0.55 * thick, c1);
                    float wall2 = 1.0 - smoothstep(0.03 + 0.15 * thick, 0.14 + 0.40 * thick, c2);
                    float lace  = max(wall1, 0.75 * wall2 * mix(0.4, 1.0, wall1 + thick));
                    band = min(band * 1.5, 1.0) * mix(lace, 1.0, 0.10 * thick);
                }
            }
            // River style: blend in the flow-driven foam pattern on top
            // of the shore band. pm_foam_tex was set in the river branch
            // above (1.0 when river style is off, so no effect).
            float flow_foam = clamp(1.0 - pm_foam_tex, 0.0, 1.0);
            float foam_mix = max(band, flow_foam);
#ifdef QLIKE_WATER
            foam_amt = foam_mix;
#endif /* QLIKE_WATER */
#ifndef QLIKE_WATER
            foam_cover = foam_mix * scene.water_foam_color.a;
#else /* QLIKE_WATER */
            foam_cover = foam_mix * scene.water_foam_color.a * foam_op;
#endif /* QLIKE_WATER */
            baseTint = mix(baseTint, scene.water_foam_color.rgb, foam_cover);
        }
        // Reflection goes over the tinted water; foam is matte, so it
        // suppresses most (not all - it is wet) of the mirror.
        // The water body's own colour is light scattered back out of it:
        // it follows the light level (dark at night, dull under overcast).
        baseTint *= pow(clamp(scene.sun_color.a * 0.4, 0.0, 1.0), 0.6);
        vec3 col_w = mix(baseTint, reflCol, fres_eff * (1.0 - 0.7 * foam_cover));
        col_w += scene.sun_color.rgb * scene.sun_color.a * spec *
                  0.8 * water_lit;
        // Subtle darkening of the surface tint when in shadow.
        col_w *= mix(0.7, 1.0, water_lit);

        // Wave-peak foam — wnor.y drops on steep wave faces; fold in
        // a 2-octave drifting noise so the foam reads as a stochastic
        // froth rather than a clean steepness mask. Scrolling along
        // the wind direction gives the field motion. Foam intensity
        // also modulated by `water_lit` so shadowed water doesn't get
        // bright white speckles.
#ifndef QLIKE_WATER
        {
#else /* QLIKE_WATER */
        // Gated by the SAME foam-opacity master (foam_op) as the shore
        // band, and folded into foam_amt so the final alpha drops here
        // too — this whitecap foam was previously independent of the
        // slider, which is why low foam opacity still looked opaque.
        if (foam_op > 0.001) {
#endif /* QLIKE_WATER */
            float steepness = clamp(1.0 - wnor.y * 1.04, 0.0, 1.0);
            vec2 fp = wpos.xz * 0.55 + vec2(wave_t * 0.30, wave_t * 0.18);
            float fn = noise2(fp) * 0.6 + noise2(fp * 2.13) * 0.4;
            float foam = smoothstep(0.45, 0.85, fn * steepness * 1.6);
#ifdef QLIKE_WATER
            foam_amt = max(foam_amt, foam);
#endif /* QLIKE_WATER */
            col_w = mix(col_w, vec3(0.95, 0.97, 1.0),
#ifndef QLIKE_WATER
                        foam * 0.85 * water_lit);
#else /* QLIKE_WATER */
                        foam * 0.85 * water_lit * foam_op);
#endif /* QLIKE_WATER */
        }

        // Churned, aerated patch where something just hit the water.
        if (rip_w.z > 0.001) {
            vec2 fq = wpos.xz * 7.0;
            float fnz = noise2(fq) * 0.6 + noise2(fq * 2.3 + 3.1) * 0.4;
            float rf_amt = rip_w.z * smoothstep(0.25, 0.75, fnz + rip_w.z * 0.5);
#ifdef QLIKE_WATER
            foam_amt = max(foam_amt, rf_amt * 0.8);
#endif /* QLIKE_WATER */
            col_w = mix(col_w, vec3(0.95, 0.97, 1.0) * (0.35 + 0.65 * water_lit), rf_amt * 0.6);
        }
        // Ring crests read mostly through what they reflect: the face
        // tilted away from the eye mirrors the bright sky, the face tilted
        // toward it shows the darker water below. At steep view angles the
        // Fresnel term alone hardly moves, so add that contrast directly.
        {
            vec2 vdir = rd.xz / max(length(rd.xz), 1e-3);
            float tilt = dot(rip_w.xy, vdir);
            col_w += (sky_refl * 0.45 + 0.02) * clamp(tilt * 1.2, -0.22, 1.0) * (1.0 - 0.5 * foam_cover);
            col_w = max(col_w, vec3(0.0));
        }

        // Apply the same volumetric fog as terrain (reuse the
        // existing block by jumping to a tail). To keep the diff
        // tight we re-do the fog inline here using the same code path.
        float sundotW = clamp(dot(rd, sunDirW), 0.0, 1.0);
        vec3 ext_w = exp(-t_water * 0.00025 * vec3(1.0, 1.3, 2.0));
        // pow(x, 8) → 3 squarings.
        float sd2 = sundotW * sundotW;
        float sd4 = sd2 * sd2;
        float sd8 = sd4 * sd4;
        vec3 fogColW = scene.crater_info.w * mix(vec3(0.55, 0.55, 0.58),
                           vec3(1.0, 0.7, 0.3),
                           0.3 * sd8);
        col_w = col_w * ext_w + fogColW * (1.0 - ext_w);

#ifndef QLIKE_WATER
        // Write hit depth; rasterised geometry that's actually in
        // front (e.g. a cube on a pier) still occludes us.
#else /* QLIKE_WATER */
        // Depth is owned by the RASTERISED water plane (this is real
        // geometry now, not a fullscreen analytic pass) — do NOT write
        // gl_FragDepth. That makes the terrain↔water silhouette a
        // pixel-exact hardware depth test (the whole point of the
        // rasterised-plane port). clipW is still needed for the motion
        // vector below.
#endif /* QLIKE_WATER */
        vec4 clipW = pc.mvp * vec4(wpos, 1.0);
#ifndef QLIKE_WATER
        // See main terrain branch — cap below the compose sky-cutoff
        // threshold so far water doesn't get repainted as sky-below-
        // horizon (black). Guard against w<=0 (point behind camera) so
        // the depth divide doesn't produce NaN.
        gl_FragDepth = min(clipW.z / max(clipW.w, 1e-4), 0.999989);
#endif /* ! QLIKE_WATER */
        // Distance fog on the water surface too.
        {
            float fa = distanceFogAmount(wpos, scene.camera_pos.xyz);
            col_w = mix(col_w, scene.distance_fog_color.rgb, fa);
        }
#ifndef QLIKE_WATER
        if (cam_under) col_w = under_col;      // seen from below (compose adds the water fog)
        outColor = vec4(col_w, 1.0);
        outGBuffer0 = vec4(clamp(col_w, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
        outGBuffer1 = vec4(0.5, 0.5, 0.7, 0.0);
#else /* QLIKE_WATER */
        // Soft shore depth-feather. The flat water plane and the
        // terrain mesh INTERSECT at the shoreline, where their depths
        // are ~equal — the razor-thin coplanar edge z-fights and
        // jitters as the camera moves. Fade the water alpha out over
        // the last ~0.35 m of depth so it dissolves into the wet shore
        // (alpha-blended over the terrain already drawn this pass)
        // instead of fighting it. Deeper water keeps alpha = 1, so the
        // open-water look is unchanged.
        // See-through via the HARDWARE alpha blend. The real splatted
        // terrain is already in the framebuffer (drawn before the water
        // plane), so making the water partially transparent in
        // clear/shallow water reveals the actual ground — far better
        // than the shader's internal FBM showthrough, which doesn't
        // match the mesh. trans_amt (Transparency) = master strength;
        // pc.color.y (Clarity depth, m) sets how deep the see-through
        // reaches. Deep water → clarT→0 → fully opaque.
        float clar_d = max(0.1, pc.color.y);
        float clarT  = exp(-depth_m * (3.0 / clar_d));   // 1 shallow→0 deep
        // Grazing-angle opacity: at a low view angle the light path
        // through the water is long and Fresnel turns the surface
        // reflective, so you should see the SKY, not the bottom. cosV
        // is 1 looking straight down, →0 at grazing. Without this you
        // could see through deep water when glancing across it.
        float graze = smoothstep(0.10, 0.45, cosV);
        float see   = clamp(trans_amt, 0.0, 1.0) * clarT * graze;
        // Soft, screen-space-stable waterline. The fade width is the
        // shoreline-softness knob (pc.color.z, metres of depth), but
        // never narrower than ~2.5 px of the depth gradient — that
        // fwidth floor is what kills the frame-to-frame jitter from the
        // point-sampled heightmap vs the rasterised terrain edge.
        // Shore fade band, widened from the previous (0.02 floor, 2.5×
        // fwidth) to (2.0 metre floor, 8× fwidth). The narrow band
        // was producing visible boundary jitter as the user walked:
        // wpos.xz shifts a few cm per frame, terrain_y shifts the
        // same, depth_clean swings ~5 cm — within a narrow shore_w
        // that's a big alpha jump, but inside a 2 m soft band it's
        // a tiny gradient shift TAA can fully absorb.
        // 6 m floor (was 2 m). The visible shore line is the smoothstep
        // of depth_clean across this width. Anything narrower exposes
        // the cm-scale mismatch between the rasterised CDLOD/tessellated
        // terrain mesh (per-frame sub-pixel TAA jitter on triangle
        // edges) and the smooth heightmap meshHeight() this shader
        // samples — the user reads that mismatch as "shore jumps up
        // and down a few cm" as the camera moves. A 6 m fade band
        // makes those few cm < 1% of alpha change → below visual
        // threshold. Pc.color.z (UI slider) can still override upward.
        // Floor dropped 6 m -> 0.8 m: the 64x64 subdivided water plane
        // killed the shore wobble that the wide fade was masking, so
        // the see-through band can shrink to "wet shore" widths. UI
        // slider (pc.color.z) can widen if the user wants soft shores.
        float shore_soft = max(0.8, pc.color.z);
        float shore_w    = max(shore_soft, fwidth(depth_clean) * 8.0);
        float shore_alpha = smoothstep(0.0, shore_w, depth_clean);
        // Aggressive grazing-angle clamp: at shallow viewing angles
        // (cosV < ~0.35) Fresnel makes water near-fully-reflective and
        // the "you can see the terrain UNDER the lake from the shore"
        // bug is the see-through still being nonzero. Multiply shore
        // alpha back up by (1 - graze_terms) so the water plane is
        // hard-opaque at low view angles regardless of clarity.
        float opaque_kick = 1.0 - smoothstep(0.05, 0.40, cosV);
        shore_alpha = max(shore_alpha, opaque_kick * 0.95);
        // Foam forced opaque where present so it reads as a bright
        // white band at the shoreline regardless of underlying water
        // clarity. Safe — the software ray-vs-heightmap occlusion
        // test at the top of this branch already discarded any pixel
        // where terrain is in front of the water plane, so foam can
        // never bleed onto hill-occluded pixels.
        float water_a = shore_alpha * (1.0 - 0.92 * see) *
                        mix(1.0, foam_op, foam_amt);
        float a       = mix(water_a, 1.0, foam_amt);
        // The see-through part of the surface (1 - a) used to lose its
        // reflection and sun glints together with its tint, so the clear
        // shallow band looked like dry sand. A water surface reflects
        // whatever is under it: put the Fresnel reflection (a little
        // boosted - a rippled film mirrors more than flat water seen
        // from above) and the glints back on that part, fading out over
        // the last centimetres at the waterline.
        {
            float film = smoothstep(0.0, 0.12, depth_clean);
            float F2   = clamp(fres_eff + 0.07, 0.0, 1.0) * film;
            vec3  glint = scene.sun_color.rgb * scene.sun_color.a * spec * 0.8 * water_lit * film;
            float a2   = a + (1.0 - a) * F2;
            col_w = (col_w * a + (reflCol * F2 + glint) * (1.0 - a)) / max(a2, 1e-3);
            a = a2;
        }
        outColor = vec4(col_w, a);
        outGBuffer0 = vec4(clamp(col_w, vec3(0.0), vec3(1.0)), 7.0 / 255.0);
        outGBuffer1 = vec4(0.5, 0.5, 0.2, 0.0);
#endif /* QLIKE_WATER */
        // Motion vector for TAA reprojection. clip-space derivation
        // (resolution-independent) so the LR raymarch path doesn't
        // pick up a constant ~0.5 offset from scene.viewport being
        // full-res — that would kill TAA history every frame.
        vec4 prev_clipW = pc.prev_mvp * vec4(wpos, 1.0);
        if (clipW.w > 0.0 && prev_clipW.w > 0.0) {
            vec2 cur_uv  = clipW.xy      / clipW.w      * 0.5 + 0.5;
            vec2 prev_uv = prev_clipW.xy / prev_clipW.w * 0.5 + 0.5;
            outMotion = (cur_uv - prev_uv) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
        return;
    }

    // Water-only (mesh-terrain) mode: reaching here means water was not
    // the visible surface — the rasterised CDLOD mesh already owns this
    // pixel (sky pixels were discarded above). Never shade the FBM
    // terrain in this mode; let the mesh show through.
    if (water_only) discard;

    vec3 pos = ro + t * rd;
    // EARLY discard for castle-interior pixels. The original discard
    // at the end of the shader (~line 2524) fired AFTER calcNormal +
    // calcShadow + AO + GI + fog — all of that work was wasted on
    // pixels the rasterised castle floor brushes are about to own
    // anyway. Same kCastleHalfExtent constant as the late discard so
    // they can never disagree. Saves the entire shading payload for
    // the ~castle-footprint slice of the frame.
    {
        const float kCastleHalfExtentEarly = 11.05;
        if (abs(pos.x) < kCastleHalfExtentEarly &&
            abs(pos.z) < kCastleHalfExtentEarly) {
            discard;
        }
    }
    // Castle road: cobble normal map in road space, before any lighting.
    vec2 rm_road_uv, rm_road_dir;
    vec2 rm_road = roadWeights(pos.xz, rm_road_uv, rm_road_dir);
    if (rm_road.x + rm_road.y > 0.001 && rd.y < -0.0015) {
        // The march hit the procedural surface, which keeps decimetre FBM
        // lumps on top of the carved road. Slide the hit along the ray onto
        // the smooth baked heightmap so the paving lies flat.
        float t_s = t;
        // Also at grazing angles (a low eye looking along the road): the
        // old cut-off at rd.y = -0.02 left the lumpy FBM surface showing in
        // a band a few metres ahead. More, larger steps when the ray is flat.
        float graze = 1.0 - smoothstep(0.02, 0.15, -rd.y);
        int   n_it  = 4 + int(graze * 4.0);
        float lim   = 1.5;
        for (int i = 0; i < n_it; ++i) {
            vec3 ps = ro + t_s * rd;
            t_s += clamp((ps.y - meshHeight(ps.xz)) / -rd.y, -lim, lim) * 0.85;
        }
        t_s = max(t_s, 0.05);
        t = mix(t, t_s, clamp(rm_road.x + rm_road.y * 0.5, 0.0, 1.0));
        pos = ro + t * rd;
        rm_road = roadWeights(pos.xz, rm_road_uv, rm_road_dir);
    }
    g_rm_detail_mul = 1.0 - rm_road.x;
    // Hand-laid look: the stone rows wander and bulge (two scales of domain
    // warp on the paving uv), so the cobbles are not a perfect repeat.
    if (rm_road.x > 0.001) {
        vec2 wq = pos.xz;
        vec2 warp = vec2(noise2(wq * 0.31) - 0.5, noise2(wq * 0.31 + 17.7) - 0.5) * 0.24 +
                    vec2(noise2(wq * 1.13 + 5.3) - 0.5, noise2(wq * 1.13 + 41.1) - 0.5) * 0.08;
        rm_road_uv += warp;
    }
    vec3 nor = calcNormal(pos, t);
    float peb0_h = g_pebN_h, peb0_id = g_pebN_id, peb0_k = g_pebN_k, peb0_sz = g_pebN_sz;
    g_rm_detail_mul = 1.0;
    if (rm_road.x > 0.001) {
        // The paved surface is the smooth carved heightmap; the procedural
        // FBM lumps left over on top of it only read as noise on cobbles.
        const float e = 0.6;
        vec3 rn_s = normalize(vec3(meshHeight(pos.xz - vec2(e, 0.0)) - meshHeight(pos.xz + vec2(e, 0.0)), 2.0 * e,
                                   meshHeight(pos.xz - vec2(0.0, e)) - meshHeight(pos.xz + vec2(0.0, e))));
        nor = normalize(mix(nor, rn_s, rm_road.x));
    }
    float rm_road_lod = 0.0;
    float rm_road_vis = 1.0;   // cobble relief self-shadow (sun visibility)
    if (rm_road.x > 0.001) {
        // Mip from the pixel footprint (no screen derivatives: road_uv is
        // discontinuous at the road edge and across the march).
        float fp = t * 2.0 / max(scene.viewport.y, 1.0);
        float g_tile = max(scene.spom_params.z, 0.25);
        rm_road_lod = log2(max(fp / g_tile * float(textureSize(u_albedo[kRmMatRoad], 0).x), 1.0));
        vec3 dirw  = vec3(rm_road_dir.x, 0.0, rm_road_dir.y);
        vec3 sidew = vec3(-rm_road_dir.y, 0.0, rm_road_dir.x);
        vec3 rT = normalize(sidew - nor * dot(sidew, nor));
        vec3 rB = normalize(dirw  - nor * dot(dirw,  nor));
        // Parallax occlusion on the cobbles - the same march as the mesh
        // terrain's road (cube.frag): height in the normal map's alpha,
        // layered march + bisection, then a short self-shadow march
        // toward the sun so the joints between stones go dark.
        float pom_D = 0.035 * (1.0 - smoothstep(35.0, 70.0, t)) * smoothstep(0.3, 0.9, rm_road.x);
        if (pom_D > 0.002) {
            vec3 Vt = vec3(dot(-rd, rT), dot(-rd, rB), dot(-rd, nor));
            Vt.z = max(Vt.z, 0.22);
            int   steps = int(mix(24.0, 8.0, Vt.z));
            float layer = 1.0 / float(steps);
            vec2  dtex  = (Vt.xy / Vt.z) * pom_D * layer;
            vec2  uvc   = rm_road_uv;
            float curD  = 0.0;
            float hh    = textureLod(u_normal[kRmMatRoad], uvc, rm_road_lod).a;
            for (int i = 0; i < 24; ++i) {
                if (i >= steps || curD >= 1.0 - hh) break;
                uvc -= dtex;
                curD += layer;
                hh = textureLod(u_normal[kRmMatRoad], uvc, rm_road_lod).a;
            }
            vec2  lo = uvc + dtex;   float dLo = max(curD - layer, 0.0);
            vec2  hi = uvc;          float dHi = curD;
            for (int b = 0; b < 4; ++b) {
                vec2  mid  = (lo + hi) * 0.5;
                float dMid = (dLo + dHi) * 0.5;
                if (dMid < 1.0 - textureLod(u_normal[kRmMatRoad], mid, rm_road_lod).a) { lo = mid; dLo = dMid; }
                else                                                                    { hi = mid; dHi = dMid; }
            }
            rm_road_uv = (lo + hi) * 0.5;
            float h0 = 1.0 - 0.5 * (dLo + dHi);
            vec3  sd = scene.sun_direction.xyz;
            vec3  Lt = vec3(dot(sd, rT), dot(sd, rB), dot(sd, nor));
            if (Lt.z > 0.02 && h0 < 0.999) {
                vec2  span = (Lt.xy / max(Lt.z, 0.15)) * pom_D * (1.0 - h0);
                float occ = 0.0;
                for (int i = 1; i <= 6; ++i) {
                    float ts = float(i) / 6.0;
                    float sh = textureLod(u_normal[kRmMatRoad], rm_road_uv + span * ts, rm_road_lod).a;
                    occ = max(occ, (sh - (h0 + (1.0 - h0) * ts)) * (1.0 - ts));
                }
                rm_road_vis = mix(1.0, 1.0 - smoothstep(0.0, 0.16, occ), clamp(pom_D / 0.035 * 1.5, 0.0, 1.0));
            }
        }
        vec3 rn = textureLod(u_normal[kRmMatRoad], rm_road_uv, rm_road_lod).xyz * 2.0 - 1.0;
        nor = normalize(mix(nor, rT * rn.x + rB * rn.y + nor * rn.z, rm_road.x));
    }
    // ---- Close-up ground detail --------------------------------------
    // The procedural palette is flat colour; within ~45 m the photo-scanned
    // ground sets (soil on the flats, rock on the steeps) supply the fine
    // structure: a normal map for the lighting, the albedo as a tint
    // around its own mean (so the palette's colour is kept) and the
    // displacement as cavity darkening. Two scales - a ~2 m tile and a
    // ~0.5 m one that takes over right under the camera.
    vec3 rm_det_tint = vec3(1.0);
    {
        float dw = (1.0 - smoothstep(34.0, 95.0, t)) * (1.0 - rm_road.x);   // (was 18 - 45 m)
        if (dw > 0.01) {
            const int kSoil = 9, kRock = 7;
            float rockw = 1.0 - smoothstep(0.5, 0.8, nor.y);
            float fp = t * 2.0 * scene.viewport.w;                       // ~metres per pixel
            // Soil: planar from above. Rock: project along the wall.
            vec2 uvS = pos.xz * 0.45;
            // Parallax: the photographed height field shifts the texture
            // against the view, so stones and clods stand up from the ground
            // instead of being painted on it (two refinement steps, near only).
            if (t < 30.0) {
                vec3 pT = normalize(vec3(1.0, 0.0, 0.0) - nor * nor.x);
                vec3 pB = cross(nor, pT);
                vec2 vts = vec2(dot(-rd, pT), dot(-rd, pB)) / max(dot(-rd, nor), 0.25);
                float pk = 0.045 * (1.0 - smoothstep(14.0, 30.0, t));          // 10 cm of relief per 2 m tile
                float lodP = log2(max(t * 2.0 * scene.viewport.w * 0.45 * float(textureSize(u_albedo[kSoil], 0).x), 1.0));
                for (int pi = 0; pi < 2; ++pi)
                    uvS = pos.xz * 0.45 + vts * ((textureLod(u_normal[kSoil], uvS, lodP).a - 0.5) * pk);
            }
            vec2 uvR = (abs(nor.x) > abs(nor.z) ? pos.zy : pos.xy) * 0.30;
            uvR = mix(pos.xz * 0.30, uvR, smoothstep(0.35, 0.6, 1.0 - nor.y));
            float lodS = log2(max(fp * 0.45 * float(textureSize(u_albedo[kSoil], 0).x), 1.0));
            float lodR = log2(max(fp * 0.30 * float(textureSize(u_albedo[kRock], 0).x), 1.0));
            vec4 nS = textureLod(u_normal[kSoil], uvS, lodS), nR = textureLod(u_normal[kRock], uvR, lodR);
            vec3 aS = textureLod(u_albedo[kSoil], uvS, lodS).rgb / max(textureLod(u_albedo[kSoil], uvS, 12.0).rgb, vec3(0.02));
            vec3 aR = textureLod(u_albedo[kRock], uvR, lodR).rgb / max(textureLod(u_albedo[kRock], uvR, 12.0).rgb, vec3(0.02));
            vec4 nD = mix(nS, nR, rockw);
            vec3 aD = mix(aS, aR, rockw);
            // Fine layer under the camera.
            float fw = 1.0 - smoothstep(3.0, 12.0, t);
            if (fw > 0.01) {
                vec2 uvF = pos.xz * 1.9 + vec2(3.1, 7.7);
                float lodF = log2(max(fp * 1.9 * float(textureSize(u_albedo[kSoil], 0).x), 1.0));
                vec4 nF = textureLod(u_normal[kSoil], uvF, lodF);
                vec3 aF = textureLod(u_albedo[kSoil], uvF, lodF).rgb / max(textureLod(u_albedo[kSoil], uvF, 12.0).rgb, vec3(0.02));
                nD.xy = mix(nD.xy, (nD.xy - 0.5) + (nF.xy - 0.5) * 1.2 + 0.5, fw * (1.0 - rockw));
                nD.a  = mix(nD.a, nD.a * (0.5 + nF.a), fw * (1.0 - rockw));
                aD = mix(aD, aD * aF, fw * (1.0 - rockw) * 0.8);
            }
            vec3 dn = vec3((nD.xy * 2.0 - 1.0) * 1.35, 1.0);              // (was 0.85)
            vec3 dT = normalize(vec3(1.0, 0.0, 0.0) - nor * nor.x);
            vec3 dB = cross(nor, dT);
            nor = normalize(mix(nor, normalize(dT * dn.x + dB * dn.y + nor * dn.z), dw));
            float cav = mix(0.62, 1.10, smoothstep(0.15, 0.85, nD.a));  // pits darker, bumps a touch brighter
            rm_det_tint = mix(vec3(1.0), clamp(mix(vec3(1.0), aD, 0.55), vec3(0.6), vec3(1.5)) * cav, dw);
        }
    }
    vec3 sunDir = scene.sun_direction.xyz;

    float dif = clamp(dot(nor, sunDir), 0.0, 1.0);

    // Match cube.frag's terrain contrast power on n_dot_l. Without
    // this, the 'Terrain shading contrast' slider only affected the
    // chunked rasterised mesh path; raymarched terrain stayed pure
    // linear Lambert and looked flatter / lower-contrast than castle
    // floors with the same shadow underneath. Fast paths for the
    // common slider values mirror cube.frag so the two paths match
    // pixel-for-pixel where they meet.
    {
        float contrast = max(0.25, scene.terrain_extra.x);
        if      (contrast == 1.0) { /* linear — no-op */ }
        else if (contrast == 2.0) { dif = dif * dif; }
        else if (contrast == 3.0) { dif = dif * dif * dif; }
        else if (contrast == 0.5) { dif = sqrt(dif); }   // x^0.5 (was x^1.5)
        else if (contrast == 4.0) { float n2 = dif * dif; dif = n2 * n2; }
        else                      { dif = pow(dif, contrast); }
    }
    float amb = 0.5 + 0.5 * nor.y;
    float fre_in = clamp(1.0 + dot(rd, nor), 0.0, 1.0);
    float fre    = fre_in * fre_in;   // pow(x, 2) is just a square

    // Terrain self-shadow (FBM march). Bias is small (5 cm along
    // the surface normal) — large biases (the previous 50 cm) made
    // the shadow visibly detach from the contact line under
    // distant FBM normals.
    // calcShadow returns lit fraction; convert to "shadow amount"
    // (0 = lit, 1 = full shadow) so we can combine it with the RT
    // result the same way cube.frag does for the rasterised path.
    // Ground occlusion along the sun ray over the BAKED heightmap, first:
    // one texture fetch per step instead of an FBM evaluation, and it
    // reaches a whole hill away (14 steps x1.5 cover the first ~19 m, 16
    // more x1.22 carry on to ~450 m). The rendered surface never lies
    // below the bake, so a hit is a real blocker. Without it the ground in
    // front of a hill stayed sunlit at medium range and the ray-traced
    // object shadows (which ignore the terrain) were drawn onto it.
    float self_shadow = 1.0;
    if (dif > 0.0) {
        float lit_g = 1.0;
#ifndef QLIKE_WATER
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
#endif /* ! QLIKE_WATER */
        // The FBM self-shadow march (the most expensive part of terrain
        // shading) only where the heightmap walk left the point lit.
        self_shadow = 1.0 - lit_g;
        if (lit_g > 0.01)
            self_shadow = max(self_shadow, 1.0 - calcShadow(pos + nor * 0.05, sunDir));
    }


    // Fake grass-cast shadow: probe the grass mask along the sun's
    // XZ direction from this terrain pixel; cells with grass darken
    // the diffuse term. Strength + sample count + reach are all UI
    // sliders (0 strength = entirely off, no taps).
    float gcs_strength = scene.grass_shadow_params.x;
#ifndef QLIKE_WATER
    // Legacy mask-probe shadow only when the exact blade march is off.
    float grass_shadow = (scene.grass_rm_params2.y > 0.5) ? 0.0 : grassCastShadow(pos, sunDir);
#else /* QLIKE_WATER */
    float grass_shadow = grassCastShadow(pos, sunDir);
#endif /* QLIKE_WATER */
    // Combine multiplicatively with self_shadow as transmittances.
    // shadow_amount=1 → light_through=0, so combined-blocked
    // = 1 - (1-self)*(1-grass) (union of independent occluders).
    // Was `max(...)` which under-darkens partially-lit slopes that
    // also have grass nearby — the comment was right, the code wasn't.
    self_shadow = 1.0 - (1.0 - self_shadow) *
                        (1.0 - grass_shadow * gcs_strength);
#ifndef QLIKE_WATER
    // Flowers / ferns / clutter rocks: dedicated small-caster depth map
    // around the camera (same test as deferred_lighting.frag).
    if (scene.clutter_vp[3].w > 0.5 && dif > 0.0) {
        vec4 lc = scene.clutter_vp * vec4(pos, 1.0);
        vec2 luv = lc.xy * 0.5 + 0.5;
        if (all(greaterThan(luv, vec2(0.002))) && all(lessThan(luv, vec2(0.998))) &&
            lc.z > 0.0 && lc.z < 1.0) {
            float occ = clutterOcc(luv, 0, lc.z - 0.0004);
            // Near cascade (layer 1): same centre, 4x finer (0.8 cm texels
            // over 16 m) for crisp stem / petal shadows close by.
            vec2 nndc = lc.xy * 4.0;
            float nr = max(abs(nndc.x), abs(nndc.y));
            if (nr < 0.97)
                occ = mix(clutterOcc(nndc * 0.5 + 0.5, 1, lc.z - 0.00025), occ, smoothstep(0.80, 0.97, nr));
            float edge = min(min(luv.x, luv.y), min(1.0 - luv.x, 1.0 - luv.y));
            occ *= smoothstep(0.0, 0.08, edge) * 0.85;
            self_shadow = 1.0 - (1.0 - self_shadow) * (1.0 - occ);
        }
    }
    // Exact blade shadows (same blade field as grass_raymarch.frag and
    // the mesh terrain in cube.frag): sphere-traced near the camera,
    // averaged mask darkening further out where blades are sub-pixel.
    if (scene.grass_rm_params2.y > 0.5 && dif > 0.0 &&
        !(scene.water_params.x > 0.5 && pos.y < scene.water_params.y + 0.05)) {
        float gp = grassEligibility(pos.xz, pos.y);
        // Bare ground beside a grass patch is shaded by the blades that stand
        // toward the sun, so also look for grass a blade-shadow length that way
        // (without this the edge of a patch cast nothing onto the dirt).
        {
            float g_bh = clamp(scene.grass_rm_params.x, 0.05, 3.0) * 0.36;
            vec2  g_reach = sunDir.xz * (g_bh / max(sunDir.y, 0.2));
            gp = max(gp, max(grassEligibility(pos.xz + g_reach * 0.35, pos.y),
                             grassEligibility(pos.xz + g_reach * 0.80, pos.y)));
        }
        if (gp > 0.02) {
            float gstr = scene.grass_rm_params2.z;
            float cdist = distance(pos, scene.camera_pos.xyz);
            float near_w = 1.0 - smoothstep(13.0, 26.0, cdist);
            float far_vis = 1.0 - 0.45 * gstr * gp *
                            smoothstep(0.1, 0.6, sunDir.y) * scene.grass_rm_params.w;
            float gvis = far_vis;
            if (near_w > 0.01) {
                float exact = grass_blade_shadow(pos, sunDir);
                gvis = mix(far_vis, mix(1.0, exact, gstr * 0.6 + 0.4), near_w);
            }
            self_shadow = 1.0 - (1.0 - self_shadow) * gvis;
        }
    }
#endif /* ! QLIKE_WATER */

    // RT PCSS for castle / boxes / dyn-props. Mirrors cube.frag's
    // terrain-receiver shadow path 1:1 — the cube.frag terrain
    // shadows look clean, so the same algorithm with the same knobs
    // should work here. Origin biased along the sun direction so
    // the per-pixel FBM normal jitter doesn't enter (was the source
    // of the previous dither).
    //   1. 4 blocker rays in a wide cone
    //   2. PCSS penumbra estimate from avg blocker distance
    //   3. N_s stratified shadow rays in the size-adapted cone
    //   4. max-combine with `self_shadow` so where the FBM says we
    //      are in self-shadow we use that clean value (analogue of
    //      cube.frag's `max(shadow, sh_bake)`).
    // Distance-LOD: skip the RT work past 200 m. Far terrain pays
    // the bake/calcShadow penalty only — keeps the per-frame ray
    // budget bounded and prevents Windows TDR on a heavy view.
    float dyn_shadow = 0.0;
    float cam_dist_pos = distance(pos, scene.camera_pos.xyz);
    // Smooth-fade RT contributions across the LOD band rather than
    // leaving a visible cliff. lod_far comes from the user slider
    // (fog_band.w); lod_near is 60% of that. Effect ramps from 1.0
    // inside lod_near, to 0.0 at lod_far. Far pixels early-out.
    float lod_far  = max(50.0, scene.fog_band.w);
    float lod_near = lod_far * 0.6;
    float rt_lod_t = 1.0 - smoothstep(lod_near, lod_far, cam_dist_pos);
    bool do_rt = rt_lod_t > 0.005;

    // Gate on shadow_strength too — if the user dialled it to 0,
    // dyn_shadow gets multiplied by 0 at line ~2082 anyway, so the
    // 4 blocker rays + N_s shadow rays were burning BVH for nothing.
    if (dif > 0.0 && scene.rt_flags.x != 0 && do_rt &&
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
        int N_s = clamp(int(ceil(float(scene.rt_flags.y) * near_mult)),
                        2, N_s_cap);

        vec3 ref_l = abs(sunDir.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                            : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref_l, sunDir));
        vec3 tan_v = cross(sunDir, tan_u);
        // Bias along sun direction (NOT surface normal) — the FBM
        // normal varies high-frequency per pixel; using it offsets
        // each pixel's shadow cone slightly and produces dither.
        vec3 origin = pos + sunDir * 0.1;
        // Per-pixel + per-frame seed — TAA averages noisy samples.
#ifndef QLIKE_WATER
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
#else /* QLIKE_WATER */
        uvec3 seed_base = uvec3(gl_FragCoord.xy, scene.rt_flags.w);
#endif /* QLIKE_WATER */

        // 1. Blocker search (4 rays, 4× softness cone).
        const int kBlocker = 4;
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
    if (dif > 0.0 && scene.rt_flags.x != 0 && scene.rt_params.w > 1e-3 &&
        rt_lod_t < 0.999 && cam_dist_pos < 900.0 && self_shadow < 0.99) {
        float far_w = 1.0 - smoothstep(700.0, 900.0, cam_dist_pos);
        vec3  fo = pos + nor * (0.15 + cam_dist_pos * 0.002) + sunDir * (0.3 + cam_dist_pos * 0.002);
        float far_sh = any_hit_no_terrain(fo, sunDir, 220.0) ? scene.rt_params.w * far_w : 0.0;
        dyn_shadow = mix(far_sh, dyn_shadow, rt_lod_t);
    }
    // The stones cast shadows on the soil and on each other: a short walk
    // toward the sun over the stone heights.
    if (t < 6.0 && sunDir.y > 0.03 && self_shadow < 0.98 && peb0_h > -0.5) {
        float pfd = (1.0 - smoothstep(4.0, 6.0, t)) * (1.0 - plateauWeight(pos.xz));
        vec2  sxz = sunDir.xz;
        float sl  = length(sxz);
        if (pfd > 0.01 && sl > 1e-3) {
            sxz /= sl;
            float rise = sunDir.y / sl;
            g_peb_mode = 2;
            float h0 = max(peb0_h, 0.0);
            float occ = 0.0, sd = 0.011;
            for (int i = 0; i < 3; ++i) {
                sd *= 2.6;
                float hs = rm_pebbles(pos.xz + sxz * sd, t);
                occ = max(occ, smoothstep(0.0, 0.004 + 0.02 * sd, hs - (h0 + sd * rise)));
            }
            g_peb_mode = 0;
            self_shadow = max(self_shadow, occ * pfd * 0.9);
        }
    }
    float shadow = max(dyn_shadow, self_shadow);

    float sha = (1.0 - shadow) * rm_road_vis * sun_horizon_lit(u_sun_horizon, pos);

    // RT ambient occlusion against castle / boxes / dyn-props
    // (mask 0x02 — terrain BLAS skipped). Mirrors cube.frag's RTAO
    // path: cosine-hemisphere samples around the surface normal,
    // sqrt-shaped raw occlusion, ao_floor remap. AO darkens the
    // ambient term — corners between walls + boxes get visibly
    // shaded, just like the rasterised path.
    float ao = 1.0;
    int ao_mode = scene.rt_flags2.w;
    if (ao_mode > 0 && scene.rt_flags.z > 0 && do_rt) {
        int requested = (ao_mode == 1) ? min(scene.rt_flags.z, 2)
                                       : scene.rt_flags.z;
        int N_ao_full = clamp(requested, 1, 32);
        // Same distance-LOD ramp as GI: distant pixels drop toward
        // 1 sample as rt_lod_t fades to 0 at the cliff. Smooth instead
        // of full-or-nothing.
        int N_ao = max(1, int(ceil(float(N_ao_full) * rt_lod_t)));
        // Don't halve the AO radius in fast mode on the terrain shader
        // — the user's slider is typically dialled for cube.frag's
        // close-quarters AO (~0.7m) and halving puts it under 0.4m
        // which can't reach a 1m wall to find it as an occluder.
        // Lower-bounded at 2m so ground next to walls / boxes still
        // sees them as occluders.
        float ao_radius = max(scene.rt_params.y, 2.0);

        // Origin lifted along the surface normal a few cm so the
        // FBM-derived hit point doesn't self-intersect a BLAS
        // triangle (e.g. a castle pad).
        vec3 origin_ao = pos + nor * 0.05;
        uvec3 ao_seed = uvec3(gl_FragCoord.xy, scene.rt_flags.w);
        int taken = 0;
        float occluded = 0.0;
        // Tangent basis around `nor` is constant for all AO samples —
        // hoist outside the loop to skip cross+normalize+cross per sample.
        vec3 ao_up_a = abs(nor.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                          : vec3(1.0, 0.0, 0.0);
        vec3 ao_t    = normalize(cross(ao_up_a, nor));
        vec3 ao_b    = cross(nor, ao_t);
        for (int i = 0; i < N_ao; ++i) {
            float u1 = rmRand(ao_seed + uvec3(i, 7u, 53u));
            float u2 = rmRand(ao_seed + uvec3(i, 41u, 5u));
            vec3 d = cos_hemi_basis(u1, u2, nor, ao_t, ao_b);
            // BLAS occluders only (castle / dyn-props). The earlier
            // terrain_blocks_ray addition pushed per-pixel cost over
            // the GPU TDR threshold during long idle frames; analytical
            // ambient + sky_vis fallback carry the soft cavity look
            // without firing N_ao extra heightfield marches per pixel.
            if (any_hit_no_terrain(origin_ao, d, ao_radius)) occluded += 1.0;
            ++taken;
        }
        // Linear curve (no sqrt) — see commit 12f9ebc rationale.
        float raw = 1.0 - (occluded / float(taken));
        float ao_floor_v = scene.rt_lod.w;
        float ao_local = mix(ao_floor_v, 1.0, raw);
        // Fade AO toward 1.0 (no occlusion) as we approach the LOD
        // cutoff so distant terrain seamlessly drops back to the
        // analytical-only path.
        ao = mix(1.0, ao_local, rt_lod_t);
    }
    // Terrain's own cavities (the RT rays above skip the terrain).
    {
        float cav_w = 1.0 - smoothstep(180.0, 260.0, cam_dist_pos);
        if (cav_w > 0.01) ao *= mix(1.0, terrainCavityAO(pos.xz), cav_w);
    }

    vec3 mate = getMaterial(pos, nor) * rm_det_tint;
    // Pebbles: each stone its own colour (greys, tans, a few dark and a few
    // pale ones), darker round its rim where it sits in the soil.
    float peb_cover = 0.0;
    if (t < 26.0) {
        float pw = (1.0 - smoothstep(10.0, 26.0, t)) * (1.0 - plateauWeight(pos.xz));
        if (pw > 0.01 && peb0_h > 0.0) {
            float id = peb0_id, k = peb0_k;
            // Brightness relative to the soil they lie on (the ground palette
            // varies a lot), stone-grey rather than soil-coloured.
            float gl  = max(dot(mate, vec3(0.3, 0.5, 0.2)), 0.02);
            vec3 pc0 = gl * mix(vec3(1.25, 1.22, 1.18), vec3(1.55, 1.38, 1.15), fract(id * 5.3)) *
                       mix(0.80, 1.25, fract(id * 23.1));
            pc0 = mix(pc0, gl * vec3(0.50, 0.52, 0.58), step(0.82, fract(id * 3.1)));     // dark basalt
            pc0 = mix(pc0, gl * vec3(2.30, 2.25, 2.10), step(0.90, fract(id * 9.7)));     // pale quartz
            pc0 = mix(pc0, gl * vec3(1.70, 1.05, 0.80), step(0.87, fract(id * 17.9)));    // red sandstone
            pc0 = mix(pc0, gl * vec3(0.95, 1.05, 1.20), step(0.88, fract(id * 41.3)));    // blue-grey slate
            // Veins / banding across some stones.
            pc0 *= 1.0 + 0.22 * step(0.6, fract(id * 6.1)) *
                         sin(dot(pos.xz, vec2(cos(id * 40.0), sin(id * 40.0))) * 260.0 + id * 9.0);
            if (peb0_sz > 3.5) {
                // Big stones: weathered, moss and lichen on top.
                float ms = noise2(pos.xz * 11.0 + id * 20.0) * 0.6 + noise2(pos.xz * 37.0) * 0.4;
                pc0 = mix(pc0, gl * vec3(0.55, 1.05, 0.40), smoothstep(0.50, 0.72, ms) * smoothstep(0.35, 0.8, k) * step(0.35, fract(id * 2.7)));
                pc0 = mix(pc0, gl * vec3(1.9, 1.85, 1.5), smoothstep(0.80, 0.86, noise2(pos.xz * 55.0 + 3.0)) * 0.7);   // lichen spots
            }
            pc0 *= mix(0.85, 1.15, noise2(pos.xz * 140.0 + id * 50.0));                    // speckle
            pc0 *= mix(0.40, 1.0, smoothstep(0.0, 0.22, k));                               // sits in the soil
            peb_cover = smoothstep(0.0, 0.12, k) * pw;
            mate = mix(mate, pc0, peb_cover);
        }
    }
    // Road: trodden dirt verge, then the cobbles. The procedural palette
    // above is authored darker than texture albedo (rock 0.10 vs
    // ~0.33), so the cobble texture is scaled toward that range.
    if (rm_road.x + rm_road.y > 0.001) {
        const float kRmAlbedoScale = 0.50;
        // Trodden dirt verge: the scanned soil set for structure (grit,
        // pebbles, ruts) instead of one flat colour, broken up at two scales.
        {
            vec3 soil = textureLod(u_albedo[9], pos.xz * 0.55, rm_road_lod).rgb /
                        max(textureLod(u_albedo[9], pos.xz * 0.55, 12.0).rgb, vec3(0.03));
            float vn = noise2(pos.xz * 0.9) * 0.6 + noise2(pos.xz * 3.7) * 0.4;
            vec3 verge = vec3(0.105, 0.078, 0.052) * clamp(soil, vec3(0.45), vec3(1.8)) * mix(0.72, 1.25, vn);
            mate = mix(mate, verge, rm_road.y);
        }
        // Stone-to-stone colour variation: patches of lighter / darker and
        // slightly warmer / cooler stone, plus some sun-bleached wear.
        vec3 cob = textureLod(u_albedo[kRmMatRoad], rm_road_uv, rm_road_lod).rgb * kRmAlbedoScale;
        float cv1 = noise2(pos.xz * 6.5 + 9.1) * 0.6 + noise2(pos.xz * 2.1 + 1.7) * 0.4, cv2 = noise2(pos.xz * 0.55 + 3.3);
        cob *= mix(0.74, 1.22, cv1) * mix(vec3(1.06, 1.0, 0.92), vec3(0.94, 0.98, 1.05), cv2);
        mate = mix(mate, cob, rm_road.x);
    }

    // === Path-traced GI on the raymarched terrain =================
    // Mirrors cube.frag's GI loop: stratified cosine-hemisphere
    // samples, N_bounces of throughput tracking, sun-shadow rays at
    // bounce hits up to gi_shadow_max_bounce, sky fallback on miss.
    // The terrain BLAS is skipped (mask 0x02) for the bounce closest
    // hits — we only want castle/dyn-prop bounce contributions; a
    // procedural-terrain self-bounce would need running getMaterial
    // at the hit point which is much more expensive than the BLAS
    // material lookup. The sun-shadow ray FROM a bounce hit uses
    // mask 0x01 (the standard shadow-caster mask) so the terrain
    // CAN occlude the sun for a castle wall hit deep in a valley.
    //
    // Energy: cosine-weighted sampling makes each hit's contribution
    // = albedo * Li (the BRDF/π and cos/π factors cancel). We tag
    // the raw GI with the surface albedo (`mate`) at the end — same
    // structure as cube.frag — and lerp toward a smooth sky-at-N
    // estimate by `gi_softener` for low-sample noise control.
    vec3 gi_indirect = vec3(0.0);
    // Terrain GI sample count capped (was scene.rt_flags2.x directly).
    // 17+ samples * (1 BLAS hit + bounce shadow ray) per pixel was the
    // dominant per-frame RT cost and the smoking gun for the GPU TDR
    // observed during long idle frames as the TLAS grew with streamed
    // dyn-props. Slider still controls; we just clamp to 6 for the
    // raymarched terrain path. Cube.frag GI uses the full slider.
    int N_gi_user = scene.rt_flags2.x;
    int N_gi_cap  = max(0, int(scene.terrain_rt_extra.y));
    // Distance-LOD on sample count: rt_lod_t goes 1→0 across the
    // RT-LOD distance band. Scale the GI sample count smoothly so
    // mid-distance pixels pay 4 rays instead of the full 16, and
    // pixels just inside the LOD cliff pay 1. Avoids the binary
    // "full N or skip" cliff that produced visible noise rings.
    int N_gi_full = (N_gi_user > 0) ? min(N_gi_user, N_gi_cap) : 0;
    int N_gi = (N_gi_full > 0 && do_rt)
                ? max(1, int(ceil(float(N_gi_full) * rt_lod_t)))
                : 0;
    // Same pattern as the PCSS gate: gi_indirect is multiplied by
    // scene.rt_params2.x (the GI strength slider) at line ~2313 so
    // strength=0 makes the whole loop's output zero. Bail before the
    // ray fire instead of burning N_gi × N_bounces ray-queries to
    // produce a value that's about to be multiplied away.
    if (N_gi > 0 && scene.rt_params2.x > 1e-3) {
        int N_bounces     = max(1, int(scene.terrain_rt_extra.w));
        int gi_shadow_max = int(scene.rt_lod.z);
        // GI search radius — driven directly by the user's slider.
        // Earlier we forced a min of 80m so first-bounce rays could
        // reach distant castle walls, but combined with N_gi samples
        // the long-ray BLAS traversal pushed total per-frame work
        // over the GPU TDR threshold. Crank gi_radius in the UI to
        // 80m+ if you want stronger color bleed.
        float gi_radius   = max(scene.rt_params2.y, 1.0);
        // Soft sky fill for bounce hits that can't see the sun. Same
        // 5%-of-sky constant as cube.frag — keeps interior bounces
        // believably dark instead of glowing from a too-bright fill.
        vec3 sky_fill = scene.sky_color.rgb * 0.05;

        int strata = int(ceil(sqrt(float(N_gi))));
        float inv_strata = 1.0 / float(strata);
        int  taken = 0;
        int  first_bounce_misses = 0;
        vec3 sum   = vec3(0.0);
        uvec3 gi_seed = uvec3(gl_FragCoord.xy, scene.rt_flags.w);

        // Tangent basis around `nor` is constant across all GI samples
        // — precompute once, pass to cos_hemi_basis() to skip the
        // per-sample cross+normalize+cross.
        vec3 gi_up_a = abs(nor.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                          : vec3(1.0, 0.0, 0.0);
        vec3 gi_t    = normalize(cross(gi_up_a, nor));
        vec3 gi_b    = cross(nor, gi_t);

        for (int sy = 0; sy < strata && taken < N_gi; ++sy) {
            for (int sx = 0; sx < strata && taken < N_gi; ++sx) {
                float r1 = rmRand(gi_seed + uvec3(taken, 73u, 11u));
                float r2 = rmRand(gi_seed + uvec3(taken, 91u, 47u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;

                vec3 ray_dir    = cos_hemi_basis(u1, u2, nor, gi_t, gi_b);
                vec3 ray_origin = pos + nor * 0.05;
                vec3 throughput = vec3(1.0);
                vec3 path       = vec3(0.0);

                for (int b = 0; b < N_bounces; ++b) {
                    float t_hit;
                    int   inst_id;
                    int   prim_id;
                    if (!closest_hit_material_no_terrain(ray_origin, ray_dir,
                                                          gi_radius,
                                                          t_hit, inst_id, prim_id)) {
                        // BLAS miss → sky in this direction. Bound by
                        // the halo's pow(8) so no fireflies. Terrain
                        // self-occlusion is intentionally NOT checked
                        // here — for short gi_radius (~20 m) on flat
                        // terrain the heightfield march false-blocks
                        // sky-bound rays, killing the GI signal. AO
                        // already carries terrain occlusion via the
                        // ambient term; GI keeps its sky-escape
                        // contribution undamped.
                        path += throughput * sample_sky(ray_dir);
                        if (b == 0) ++first_bounce_misses;
                        break;
                    }
                    int mat_idx = (inst_id == kStaticBlasSentinel)
                                    ? (prim_id / kCubeTrisPerBox)
                                    : inst_id;
                    Material m  = materials[mat_idx];
                    vec3 hit_pos = ray_origin + ray_dir * t_hit;
                    // No vertex normal in ray queries; -ray_dir is a
                    // good local approximation for cosine sampling
                    // and the sun-shadow's n·L test (matches cube.frag).
                    vec3 hit_n   = -ray_dir;

                    vec3 hit_light = sky_fill;
                    if (b < gi_shadow_max) {
                        float n_dot_sun = dot(hit_n, scene.sun_direction.xyz);
                        if (n_dot_sun > 0.0) {
                            if (!any_hit_shadow_caster(
                                    hit_pos + hit_n * 0.01,
                                    scene.sun_direction.xyz, 200.0)) {
                                hit_light += scene.sun_color.rgb *
                                             scene.sun_color.a * n_dot_sun;
                            }
                        }
                    }
                    path += throughput * (m.emissive.rgb +
                                          m.color.rgb * hit_light);
                    if (b + 1 >= N_bounces) break;

                    // Tint throughput by the surface albedo, kick
                    // off the next bounce in a cosine hemisphere
                    // around the approximate hit normal.
                    throughput *= m.color.rgb;
                    ray_origin  = hit_pos + hit_n * 0.01;
                    float br1 = rmRand(gi_seed + uvec3(taken * 7 + b, 31u, 17u));
                    float br2 = rmRand(gi_seed + uvec3(taken * 7 + b, 7u, 91u));
                    ray_dir = cos_hemi(br1, br2, hit_n);
                }
                sum += path;
                ++taken;
            }
        }
        vec3 gi_raw = (taken > 0) ? (sum / float(taken)) : vec3(0.0);
        // Sky visibility: fraction of FIRST-bounce rays that escaped
        // all occluders. Used to attenuate the smooth fallback so a
        // surface deep in a cavity doesn't get a bright "sky from
        // normal" lift even when the softener slider is high.
        float sky_vis = (taken > 0)
                          ? (float(first_bounce_misses) / float(taken))
                          : 1.0;
        // Softener: lerp toward a noise-free sky-at-normal estimate.
        // 0 = raw path-traced GI (most accurate, noisiest at low N),
        // 1 = smooth fallback (no per-pixel variance).
        // Both terms are AO-modulated AND sky_vis-modulated so corners
        // and cavities go genuinely dark regardless of softener value.
        float ao_p     = max(scene.ambient.w, 0.5);
        float occl = pow(ao, ao_p) * mix(0.4, 1.0, sky_vis);
        vec3 gi_smooth = sample_sky_atmosphere(nor) * occl;
        float soft = clamp(scene.terrain_extra.z, 0.0, 1.0);
        gi_indirect = mate * mix(gi_raw, gi_smooth, soft) *
                      scene.rt_params2.x;
        // Hard ceiling to keep a single bright bounce (e.g. a sun-
        // facing white box wall) from spiking the HDR + bloom.
        gi_indirect = min(gi_indirect, vec3(4.0));
        // Smooth-fade the GI contribution past 150 m so distant
        // terrain doesn't show a hard boundary where the loop stops.
        gi_indirect *= rt_lod_t;
    }

    vec3 lin = vec3(0.0);
    float cl_sh = cloud_shadow(pos);
    lin += dif * sha * cl_sh * scene.sun_color.rgb * scene.sun_color.a;
    // Ambient + Fresnel rim term darkened by RTAO. ao_punch pow-curves
    // the AO so > 1 makes mid-AO drop fast (corners punch dark) while
    // 1.0 keeps the original linear behaviour. Driven by the
    // 'AO darkness (raymarch terrain)' slider.
    float ao_punch  = max(scene.ambient.w, 0.5);
    float ao_shaped = pow(ao, ao_punch);
    // Sky fill. 0.35 left sun-shadowed ground almost black (far darker than
    // the mesh terrain shows the same shadow), so the shadow of a trunk or
    // rock read as a hole in the ground rather than as shade.
    // Under trees / beside objects the sky is partly covered as well: the
    // shadow of a canopy takes most of the sky fill with it (a forest
    // floor is dark, an open slope in the shade of a hill is not).
    lin += amb * scene.sky_color.rgb * 0.75 * ao_shaped * mix(1.0, 0.30, dyn_shadow);
    lin += fre * scene.sky_color.rgb * 0.25 * ao_shaped;

    // Dynamic point light (muzzle flash or the brightest explosion, same UBO
    // light the deferred pass applies to everything else - it skips this
    // pre-lit material). Objects can block it; terrain occlusion is ignored.
    if (scene.muzzle_pos.w > 0.0) {
        vec3  lv = scene.muzzle_pos.xyz - pos;
        float d  = length(lv);
        float r  = max(scene.muzzle_color.w, 0.1);
        if (d < r && distance(pos, scene.camera_pos.xyz) > 1.5) {
            vec3  Lp  = lv / max(d, 1e-3);
            float att = 1.0 - d / r; att *= att;
            float ndl = max(dot(nor, Lp), 0.0);     // no light on slopes facing away (hills behind a blast)
            if (ndl > 0.0 && att * scene.muzzle_pos.w > 0.01 && !any_hit_no_terrain(pos + nor * 0.1, Lp, max(d - 0.8, 0.05)))
                lin += scene.muzzle_color.rgb * (scene.muzzle_pos.w * att * ndl);
        }
    }
    // Hand flashlight on the ground: shadowed by objects, bushes and trees,
    // and near the camera by the lawn's own blades.
    {
        vec3 fl_dir; float fl_dist;
        vec3 fl = flashlight_at(pos, fl_dir, fl_dist);
        float fl_ndl = max(dot(nor, fl_dir), 0.0);
        if (fl_ndl > 0.0 && (fl.r + fl.g + fl.b) > 1e-4 &&
            !any_hit_no_terrain(pos + nor * 0.08, fl_dir, max(fl_dist - 0.05, 0.05))) {
            float blade_vis = 1.0;
#ifndef QLIKE_WATER
            if (scene.grass_rm_params2.y > 0.5 && distance(pos, scene.camera_pos.xyz) < 20.0 &&
                grassEligibility(pos.xz, pos.y) > 0.02)
                blade_vis = mix(1.0, grass_blade_shadow(pos, fl_dir), scene.grass_rm_params2.z * 0.6 + 0.4);
#endif
            float plant_vis;
#ifndef QLIKE_WATER
            FLASH_PLANT_VIS(u_clutter_shadow, pos + nor * 0.03, plant_vis)
#else
            plant_vis = 1.0;
#endif
            lin += fl * (fl_ndl * blade_vis * plant_vis);
        }
    }
    lin += fx_lights_diffuse(pos, nor);   // tracers (no shadows)
    // Laser bolts and gun flashes: shadowed by objects, bushes and trees, and
    // near the camera by the lawn's own blades.
    {
        int fxn = int(scene.ripple_info.w + 0.5);
        for (int i = 0; i < fxn; ++i) {
            if (scene.fx_light_col[i].a < 0.5) continue;
            vec3 to = scene.fx_lights[i].xyz - pos;
            float r = scene.fx_lights[i].w;
            float d2 = dot(to, to);
            if (d2 >= r * r) continue;
            float d = sqrt(d2);
            vec3 ld = to / max(d, 1e-3);
            float w = max(0.3 + 0.7 * dot(nor, ld), 0.0);
            if (w <= 0.0) continue;
            float fall = 1.0 - d / r; fall *= fall;
            if (any_hit_no_terrain(pos + nor * 0.08, ld, max(d - 0.3, 0.05))) continue;
            float blade_vis = 1.0;
#ifndef QLIKE_WATER
            if (scene.grass_rm_params2.y > 0.5 && distance(pos, scene.camera_pos.xyz) < 20.0 &&
                grassEligibility(pos.xz, pos.y) > 0.02)
                blade_vis = mix(1.0, grass_blade_shadow(pos, ld), scene.grass_rm_params2.z * 0.6 + 0.4);
#endif
            lin += scene.fx_light_col[i].rgb * (fall * w * blade_vis);
        }
    }
    // Road lanterns (dusk / night): warm pools on the road and verge, with
    // a soft shadow (ray toward a random point of the lantern glass).
    {
        vec2 lseed = gl_FragCoord.xy + vec2(float(scene.rt_flags.w & 31)) * vec2(17.13, 9.71);
        for (int li = 0; li < 8; ++li) {
            float lI = scene.lamp_lights[li].w;
            if (lI < 1e-3) continue;
            vec3 lpos = scene.lamp_lights[li].xyz;
            vec3 to = lpos - pos;
            float d = length(to);
            if (d >= 13.0) continue;
            vec3 ld = to / max(d, 1e-3);
            float ndl = max(dot(nor, ld), 0.0);
            if (ndl <= 0.0) continue;
            float fall = 1.0 - d / 13.0; fall *= fall;
            vec3 jit = fract(sin(vec3(dot(lseed, vec2(12.9898, 78.233)) + float(li) * 3.1,
                                      dot(lseed, vec2(39.3468, 11.135)) + float(li) * 5.7,
                                      dot(lseed, vec2(73.1560, 52.235)) + float(li) * 9.3)) * 43758.5453) - 0.5;
            vec3 tj = lpos + jit * 0.34 - pos;
            float dj = length(tj);
            if (any_hit_no_terrain(pos + nor * 0.08, tj / max(dj, 1e-3), max(dj - 0.45, 0.05))) continue;
            // The lawn's own blades shade the ground from the lantern (the same
            // blade field the sun uses): exact near the camera, an averaged
            // darkening further out.
            float blade_vis = 1.0;
#ifndef QLIKE_WATER
            if (scene.grass_rm_params2.y > 0.5) {
                float gp = grassEligibility(pos.xz, pos.y);
                if (gp > 0.02) {
                    float gstr = scene.grass_rm_params2.z;
                    float near_w = 1.0 - smoothstep(13.0, 26.0, distance(pos, scene.camera_pos.xyz));
                    blade_vis = 1.0 - 0.45 * gstr * gp * scene.grass_rm_params.w;
                    if (near_w > 0.01)
                        blade_vis = mix(blade_vis, mix(1.0, grass_blade_shadow(pos, ld), gstr * 0.6 + 0.4), near_w);
                }
            }
#endif
            float lamp_pv = 1.0;
#ifndef QLIKE_WATER
            LAMP_PLANT_VIS(u_clutter_shadow, pos + nor * 0.03, li, lamp_pv)
#endif
            lin += vec3(1.0, 0.62, 0.28) * (lI * ndl * fall * blade_vis * lamp_pv);
        }
    }
    vec3 col = mate * lin + gi_indirect;
    // Rain: a wet sheen on the ground and standing water in the flats.
    // A puddle is a small mirror: flat normal (plus rain rings), darker
    // bed, Fresnel reflection of the sky and a sun glint.
    {
        // The beach where the water laps: the sand is soaked (darker, a
        // thin mirror of water on it) up to a waterline that moves in and
        // out with the wavelets.
        float shore = 0.0, shore_foam = 0.0;
        if (scene.water_params.x > 0.5) {
            float tw = scene.water_params.w;
            float lap = 0.045 * sin(tw * 0.8 + pos.x * 0.31 + pos.z * 0.23) + 0.03 * sin(tw * 1.37 + pos.x * 0.83 - pos.z * 0.57);
            float above = pos.y - scene.water_params.y - lap;
            float flat_w = smoothstep(0.55, 0.85, nor.y);
            shore = (1.0 - smoothstep(0.03, 0.50, above)) * flat_w;
            shore_foam = (1.0 - smoothstep(0.0, 0.022, abs(above - 0.010))) * flat_w;
            // The beach itself: bare sand with darker grit and pebbles from
            // the waterline up to about 70 cm above it (also under water).
            float beach = (1.0 - smoothstep(0.30, 0.75, above + 0.10 * noise2(pos.xz * 0.9))) * flat_w;
            if (beach > 0.01) {
                float grit = noise2(pos.xz * 6.0) * 0.6 + noise2(pos.xz * 19.0) * 0.4;
                float peb = smoothstep(0.62, 0.72, noise2(pos.xz * 11.0 + 3.7)) * smoothstep(0.35, 0.6, noise2(pos.xz * 2.3));
                vec3 sandc = mix(scene.terrain_sand_color.rgb, vec3(0.200, 0.172, 0.125), 0.65) * mix(0.78, 1.12, grit);
                sandc = mix(sandc, vec3(0.10, 0.097, 0.09) * mix(0.6, 1.3, noise2(pos.xz * 5.1)), peb * 0.8);
                // Lit with a smoothed normal: sand is fine-grained, the rocky
                // relief of the ground it replaces would show through as veins.
                vec3 bn = normalize(mix(nor, vec3(0.0, 1.0, 0.0), 0.8));
                vec3 blin = scene.sun_color.rgb * scene.sun_color.a * (max(dot(bn, sunDir), 0.0) * sha * cl_sh) +
                            scene.sky_color.rgb * (0.40 * ao_shaped);
                col = mix(col, sandc * blin + gi_indirect * 0.5, beach);
            }
        }
        float wet = max(scene.ripple_info.z, shore);
        float pd = distance(pos, ro);
        if (wet > 0.01 && pd < 400.0) {
            float flatn = smoothstep(0.80, 0.95, nor.y);            // (was 0.90..0.985: almost no ground qualified)
            float pn = noise2(pos.xz * 0.33) * 0.65 + noise2(pos.xz * 1.21 + 7.0) * 0.35;
            float thr = 0.66 - 0.34 * wet;
            float puddle = flatn * smoothstep(thr, thr + 0.06, pn) * smoothstep(0.2, 0.6, wet);
            vec3 pn_n = normalize(mix(nor, vec3(0.0, 1.0, 0.0), max(max(puddle, 0.5 * wet), 0.92 * shore)));
            if (puddle > 0.01 && pd < 60.0) {
                vec3 rw = water_ripples(pos.xz);
                pn_n = normalize(vec3(pn_n.x - rw.x * puddle, pn_n.y, pn_n.z - rw.y * puddle));
            }
            vec3 rr = reflect(rd, pn_n);
            rr.y = max(rr.y, 0.03);
            float cv = clamp(dot(-rd, pn_n), 0.0, 1.0);
            float om = 1.0 - cv; float om2 = om * om;
            float Fw = 0.02 + 0.98 * om2 * om2 * om;
            vec3 skyr = sample_sky_atmosphere(normalize(rr));
            float cov = scene.crater_info.y;
            skyr = mix(skyr, vec3(dot(skyr, vec3(0.2126, 0.7152, 0.0722))), cov);
            float film = max(puddle, 0.11 * wet * smoothstep(0.5, 0.95, nor.y));
            film = max(film, 0.40 * shore);                          // the glassy sheet on wet sand
            col *= mix(1.0, 0.55, puddle);
            col *= mix(1.0, 0.70, shore * (1.0 - puddle));           // soaked sand is darker
            col += skyr * Fw * film * ao_shaped;
            float sg = max(dot(normalize(rr), sunDir), 0.0);
            float sg2 = sg * sg; float sg8 = sg2 * sg2 * sg2 * sg2;
            float sg64 = sg8 * sg8 * sg8 * sg8 * sg8 * sg8 * sg8 * sg8;
            col += scene.sun_color.rgb * scene.sun_color.a * sg64 * sg64 * sha * cl_sh * max(puddle, 0.12 * film) * 1.5;
            // A thin line of foam where the water runs out.
            if (shore_foam > 0.01) {
                float fb = noise2(pos.xz * 9.0 + vec2(scene.water_params.w * 0.4, 0.0)) * 0.6 + noise2(pos.xz * 23.0) * 0.4;
                col = mix(col, (scene.sun_color.rgb * scene.sun_color.a * (0.25 + 0.75 * sha) + scene.sky_color.rgb * 0.6) * 0.55,
                          shore_foam * smoothstep(0.48, 0.74, fb) * 0.5);
            }
        }
    }

    // Final-colour AO multiplier — dampens the WHOLE pixel (sun term
    // included) by the shaped AO, making corner darkening visibly
    // pronounced even when the sun is hitting the ground. Without
    // this, AO only modulated the ambient term which is small
    // relative to the sun, so the corner-vs-open contrast was
    // imperceptible. Mirrors the visible "darker between objects"
    // look on cube.frag — there the higher-contrast albedos plus
    // GI hit-shading make the effect read; on terrain we have to
    // fold AO into the final colour to match.
    //
    // Strength comes from the user-facing 'AO final strength' slider
    // (terrain_rt_extra.z). 0 = no extra darkening, 1 = full punch.
    // ao_shaped drives the AO contour, so the 'AO darkness' slider
    // tunes the curve and 'AO final strength' tunes how much that
    // curve actually shows up on screen.
    float ao_final_str = clamp(scene.terrain_rt_extra.z, 0.0, 1.0);
    float ao_floor     = 1.0 - ao_final_str;
    col *= mix(ao_floor, 1.0, ao_shaped);

    // Contrast ('contrast (raymarch terrain)' slider): a power curve on
    // luminance around the brightness of average ground under the current
    // sun + sky, so it tracks time of day. Hue is kept (RGB scaled alike).
    {
        float rm_con = scene.grass_rm_params2.w;
        if (abs(rm_con - 1.0) > 0.005) {
            const vec3 kLum = vec3(0.2126, 0.7152, 0.0722);
            float piv = 0.12 * dot(scene.sun_color.rgb * scene.sun_color.a * 0.45 +
                                   scene.sky_color.rgb * 0.35, kLum);
            float l = dot(col, kLum);
            // Below the pivot the curve is much gentler: full strength there
            // crushed shadowed ground to black.
            float rel = max(l, 1e-5) / max(piv, 1e-4);
            col *= clamp(pow(rel, (rm_con - 1.0) * (rel < 1.0 ? 0.3 : 1.0)), 0.15, 4.0);
        }
    }

    // Debug viz: replace the final colour with the GI contribution
    // only (scaled). 0 = off, 1 = scaled gi_indirect, 2 = scaled raw
    // gi (no albedo modulation, no softener). Toggled from
    // Settings -> Lighting -> "GI debug viz (raymarch terrain)".
    if (scene.terrain_extra.w > 0.5) {
        col = gi_indirect * 4.0;
    }

    // Wavelength-dependent atmospheric extinction (blue attenuates
    // ~4× faster than red), with a sun-tinted fog colour near the sun
    // direction so the horizon glows warm where the sun touches it.
    vec3 extinction = exp(-t * 0.00025 * vec3(1.0, 1.3, 2.0));
    float sundot = clamp(dot(rd, sunDir), 0.0, 1.0);
    // pow(x, 8) → 3 squarings.
    float sn2 = sundot * sundot;
    float sn4 = sn2 * sn2;
    float sn8 = sn4 * sn4;
    vec3 fogCol = scene.crater_info.w * mix(vec3(0.55, 0.55, 0.58),
                      vec3(1.0, 0.7, 0.3),
                      0.3 * sn8);
    col = col * extinction + fogCol * (1.0 - extinction);

    // === Volumetric ground fog (real ray-march) =================
    // Multi-step march along the camera ray with Beer-Lambert
    // transmittance accumulation + Henyey-Greenstein phase-weighted
    // sun in-scattering. The Frostbite-style energy-conserving
    // integration:
    //   Sint = (S - S * exp(-σe * dt)) / σe    [exact for constant σ
    //                                            over the segment]
    //   trans *= exp(-σe * dt)
    //   col_out += trans_prev * Sint
    // Fog density = exp(-y / scaleHeight) × (1 + slow FBM scroll) so
    // the fog has subtle volumetric variation (wisps in the valleys)
    // instead of a uniform sheet. No per-step volumetric shadow ray
    // — the HG phase already gives the directional sun glow.
    {
        // Slider-driven strength; skip the entire march when 0.
        float fogStrength = pc.grass_params.x;
        bool  fogGodRays  = pc.grass_params.z > 0.5;
        if (fogStrength > 0.001) {
        // Distance-LOD on the fog march: full 12 steps near the
        // camera, drop to 4 past 150 m. Distant terrain pixels with
        // 16-step fog were the dominant per-pixel cost and the
        // direct cause of the GPU TDR — far fog can't visibly
        // resolve sub-step density anyway.
        // Linear ramp 12→4 across 100..250 m. The previous binary cliff
        // at 150 m caused wave divergence on neighbouring pixels and
        // visible TAA shimmer along the cutoff line.
        float dist_cam = distance(pos, scene.camera_pos.xyz);
        int kVolSteps = max(4, int(mix(12.0, 4.0,
                                        smoothstep(100.0, 250.0, dist_cam))));
        const float kVolDensityBase   = 0.030;  // σe at full density inside the band
        const float kVolPhaseG        = 0.65;   // forward-scattering bias
        // Band parameters from settings (fog_band slot in scene UBO).
        float fog_y_start = scene.fog_band.x;
        float fog_y_top   = scene.fog_band.y;
        float fog_noise   = clamp(scene.fog_band.z, 0.0, 1.0);

        vec3 fogTint   = vec3(0.78, 0.83, 0.90) * scene.crater_info.w;   // unlit fog is dark at night
        vec3 sunGlow   = scene.sun_color.rgb * scene.sun_color.a;

        // Henyey-Greenstein phase: forward-peaked when looking near
        // the sun. The denominator goes near-zero as cosTh→1, so
        // phase can spike past 100 and blow up the bloom + auto-
        // exposure (visible flash + sometimes drives a TDR).
        // Clamp the denominator's growth so phase stays bounded.
        float cosTh = clamp(dot(rd, sunDir), -1.0, 0.985);
        float gg    = kVolPhaseG * kVolPhaseG;
        float phase = (1.0 - gg) /
                      (4.0 * 3.14159265 *
                       // pow(x, 1.5) = x * sqrt(x) — one mul + one
                       // sqrt, way cheaper than the exp/log path pow
                       // takes for fractional exponents.
                       (1.0 + gg - 2.0 * kVolPhaseG * cosTh) *
                       sqrt(1.0 + gg - 2.0 * kVolPhaseG * cosTh));
        // Hard ceiling matches a typical "sun glow" magnitude.
        phase = min(phase, 4.0);
        // P16: hoist sunGlow * phase out of the per-step inner loop.
        // Both are loop-invariants — the scattering line at the hot
        // spot below was multiplying them every iteration. ~3 ALU/step
        // back × kVolSteps + early-out cap.
        vec3 sunPhaseTerm = sunGlow * phase;

        // Cap march length at the surface hit. Sky pixels never reach
        // here (we returned -1 above), so t is always finite.
        float t_end = t;
        float t_max = min(t_end, 600.0);   // cap so we don't waste taps in fog-noise pixels far away
        float dt    = t_max / float(kVolSteps);
        // Per-pixel jitter on the start position breaks step banding
        // — combined with Phase 5's surface jitter this stays dither-stable.
        // Stable per-pixel jitter (no frame number) so the volumetric
        // step pattern is fixed in screen space — TAA's spatial
        // filter erases it. Per-frame variation made the dither
        // pattern shift between frames and TAA couldn't keep up.
        float jStart = ignBase(gl_FragCoord.xy + vec2(39.1, 9.7));
        float t_v   = dt * jStart;

        vec3  scatter = vec3(0.0);
        float trans   = 1.0;
        for (int i = 0; i < kVolSteps; ++i) {
            vec3 p = ro + rd * t_v;
            // Density profile: 0 below y_start, ramps up over 1 m,
            // 1 inside the band, falls off over 4 m above y_top.
            // Compact band that hugs the valleys instead of a global
            // exp falloff.
            float profile = smoothstep(fog_y_start, fog_y_start + 1.0, p.y) *
                            (1.0 - smoothstep(fog_y_top, fog_y_top + 4.0, p.y));
            // Outside the fog layer the density is exactly 0: the step adds no
            // scatter and no extinction - skip its wisp fetch and god-ray taps.
            if (profile <= 0.0) { t_v += dt; continue; }
            // 3-octave wisp pattern, pre-baked into the 256² R8 fog
            // wisp texture (binding 14). One textureLod replaces the
            // 3 noise2 calls the inline FBM used to do (~12 hashes
            // per tap). REPEAT addressing handles time scrolling and
            // large camera moves seamlessly.
            vec2 q = p.xz * 0.020 + vec2(scene.water_params.w * 0.05);
            float w = textureLod(u_fog_wisp, q * (1.0 / 16.0), 0.0).r;
            // Centred around 0.5 so noise = 1.0 → "no modulation" and
            // noise = 0 → density ×0.2 (thin patch). Strength slider
            // controls how much the noise can carve gaps.
            float wisp = mix(1.0, 2.0 * w, fog_noise);
            wisp = clamp(wisp, 0.2, 2.0);
            float sigma_e = kVolDensityBase * profile * wisp * fogStrength;

            // Phase 7 — fog self-shadow / god-rays. Step a few small
            // distances toward the sun and accumulate optical depth
            // through the same density field. Surface (terrain /
            // castle) shadows the fog because the surface RT shadow
            // `sha` already gates the unshadowed mass; the per-step
            // sample additionally darkens fog that's behind a fog-
            // density gradient, giving the godray look.
            float lightTrans = sha;
            if (fogGodRays) {
                // 2 inner steps × 1-octave wisp. Was 4 steps × 3 octaves
                // = 12 noise2 calls per fog tap; with ~12 outer taps
                // that was 144 hash evals per pixel for the godray work
                // alone. Single-octave wisp is the dominant low-freq
                // mode; the higher-freq components alias-tolerate at
                // godray resolution under TAA.
                const int kFogShadowSteps = 2;
                float ds = 9.0;          // bigger base to cover same length in fewer steps
                float ld = ds * 0.5;
                for (int j = 0; j < kFogShadowSteps; ++j) {
                    vec3 sp = p + sunDir * ld;
                    float prof_s = smoothstep(fog_y_start, fog_y_start + 1.0, sp.y) *
                                    (1.0 - smoothstep(fog_y_top, fog_y_top + 4.0, sp.y));
                    vec2 qs = sp.xz * 0.020 + vec2(scene.water_params.w * 0.05);
                    float w_s = textureLod(u_fog_wisp, qs * (1.0 / 16.0), 0.0).r;
                    float wisp_s = clamp(mix(1.0, 2.0 * w_s, fog_noise), 0.2, 2.0);
                    float sig_s = kVolDensityBase * prof_s * wisp_s * fogStrength;
                    lightTrans *= exp(-sig_s * ds);
                    ds *= 1.7;          // bigger growth so 2 steps still cover far range
                    ld += ds;
                    if (lightTrans < 0.02) break;
                }
            }

            // Single-scattering: σs в‰€ σe (assume cloudy isotropic
            // scattering, no absorption). Phase function gives the
            // sun-aligned directional component; ambient fogTint
            // keeps back-lit fog from going pitch-black.
            vec3  Lin   = sunPhaseTerm * lightTrans + fogTint * 0.25;
            // P16: exp2(-x * 1.4427) ≈ exp(-x), and exp2 is a single
            // hardware instruction on most archs vs exp's exp/log
            // pair. Hoisted constant 1.4427 = 1/ln(2).
#ifndef QLIKE_WATER
            const float kLog2e = 1.44269504;   // log2(e) — converts exp() to exp2()
            float seg  = exp2(-sigma_e * dt * kLog2e);
#else /* QLIKE_WATER */
            float seg  = exp2(-sigma_e * dt * 1.44269504);
#endif /* QLIKE_WATER */
            vec3  Sint = (Lin - Lin * seg) / max(sigma_e, 1e-4);
            scatter += trans * Sint * sigma_e;   // re-multiply σs because we factored σe out
            trans   *= seg;
            t_v     += dt;
            if (trans < 0.02) break;             // early-out — fully foggy
        }
        // Apply: surface attenuated by trans, scatter added in front.
        col = col * trans + scatter;
        }   // fogStrength > 0
    }

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
#ifdef QLIKE_WATER
    // Terrain branch is DEAD in water.frag (water_only is forced true →
    // discarded long before here). `clip` is still needed by the motion
    // block below, but we deliberately write NO gl_FragDepth anywhere in
    // this shader so it is not depth-replacing — the rasterised water
    // plane's hardware depth is used as-is (pixel-exact shoreline).
#endif /* QLIKE_WATER */
    vec4 clip = pc.mvp * vec4(pos, 1.0);
#ifndef QLIKE_WATER
    gl_FragDepth = min(clip.z / max(clip.w, 1e-4), 0.999989);
#endif /* ! QLIKE_WATER */

    // Distance fog — final atmospheric mix, after all the other
    // shading, GI, water, volumetric fog and grass-shadow work.
    {
        float fa = distanceFogAmount(pos, scene.camera_pos.xyz);
        col = mix(col, scene.distance_fog_color.rgb, fa);
    }
    outColor = vec4(col, 1.0);
#ifndef QLIKE_WATER
    outGBuffer0 = vec4(clamp(col, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
    outGBuffer1 = vec4(0.5, 0.5, 0.7, 0.0);
#else /* QLIKE_WATER */
    // G-buffer pass-through. Water is already fully shaded; material_id 7
    // tells the deferred lighting pass to use albedo unmodified.
    outGBuffer0 = vec4(clamp(col, vec3(0.0), vec3(1.0)), 7.0 / 255.0);
    outGBuffer1 = vec4(0.5, 0.5, 0.2, 0.0);   // placeholder normal+rough
#endif /* QLIKE_WATER */
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
