#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require

// Raymarched grass — fullscreen pass that walks an SDF blade field.
// Adapted from MonterMan / iq's "grass field with blades" shadertoy
// (https://shadertoy.com/view/dd2cWh) for a Vulkan rasterized engine:
//
//   * Heightmap source is our scene_desc binding 8 (R32_SFLOAT) instead
//     of an inline FBM — blades sit on the actual terrain the rest of
//     the engine sees.
//   * Distance cap kills grass beyond `pc.color.x` so the per-pixel cost
//     stays bounded; the view fades to sky over the last 30% so the cap
//     doesn't read as a hard ring.
//   * gl_FragDepth from the marched hit lets cube/castle/dyn-prop depth
//     occlude grass via the LESS_OR_EQUAL test (depth pre-pass already
//     wrote those before this draw).
//   * Wind animation modulates per-cell blade rotation by `pc.color.z`
//     (time, seconds) × `pc.color.y` (wind strength).

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
// G-buffer pass-through outputs — see grass.frag for the design rationale.
// Material_id=5 makes deferred sample scene_color at this pixel instead
// of recomputing lighting, preserving the raymarched grass shading.
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

layout(set = 0, binding = 32) uniform sampler2D u_cloud_shadow;   // sunlight through the cloud layer (clouds.comp)
#define CLOUD_SHADOW_MAP u_cloud_shadow
#include "scene_ubo.glsl"
#include "sun_horizon.glsl"
layout(set = 0, binding = 31) uniform sampler2D u_sun_horizon;   // far-mountain sun shadow

// Heightmap (R32_SFLOAT) — shared with terrain raymarch + cube shaders.
// kTerrainExtent matches terrain.cpp's world span (2048 m square).
layout(set = 0, binding = 8) uniform sampler2D u_heightmap;

// Sun shadow map (binding 7). sampler2DShadow returns 0..1 PCF result.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;
// Scene TLAS - RT sun shadow on blades (trees / castle / props are not
// all in the raster shadow map).
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 3) uniform sampler2D u_albedo[19];
#define LEAF_ATLAS u_albedo[15]
#include "leaf_rt.glsl"

// Grass eligibility mask. R8 1024² — single source of truth shared
// with terrain_raymarch.frag. CPU-baked from the same FBM (presence)
// and a sample_terrain_height finite-difference (slope) the engine
// already runs at level load. Sampling this once per cell replaces a
// 9-call noised() storm in the inner loop of map() — that was the
// dominant cost in this pass per the latest perf review.
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;
#include "grass_blades.glsl"
// Depth PRE-PASS copy (binding 27). Bounds the march: nothing behind the
// nearest opaque surface can be visible, so stop there.
layout(set = 0, binding = 27) uniform sampler2D u_prepass_depth;
// The terrain's low-resolution march (binding 34, RG32F; terrain_rm_heightfield.glsl
// raymarch_lo): .g = for each block of pixels, how far a ray goes before it comes within a
// blade's height of the ground. This pass starts its march there.
layout(set = 0, binding = 34) uniform sampler2D u_march_lo;

// How far a blade at world xz is pressed down by a car: 1 under it, falling to 0 behind it
// over the stretch it has just driven (the grass comes up again), 0 elsewhere. The cars are
// in the scene block (car_press: two of them). Under the car the blades would otherwise
// stand up through its floor; behind it they rise again in the order it passed over them.
float car_press(vec2 xz) {
    float k = 0.0;
    for (int i = 0; i < 2; ++i) {
        vec4 c = scene.car_press[2 * i], e = scene.car_press[2 * i + 1];
        if (e.w < 0.5) continue;
        vec2 d = xz - c.xy;
        d -= 6400.0 * round(d / 6400.0);                 // (the blade lattice repeats every 6.4 km)
        float along  = dot(d, c.zw);                     // + toward its front
        float across = abs(dot(d, vec2(-c.w, c.z)));
        float side = 1.0 - smoothstep(e.x, e.x + 0.30, across);
        if (side <= 0.0) continue;
        float front = 1.0 - smoothstep(e.y, e.y + 0.35, along);
        // behind: fully down at its tail, up again at the end of the stretch
        float back = (along > -e.y) ? 1.0 : 1.0 - smoothstep(0.0, max(e.z, 0.5), -e.y - along);
        k = max(k, side * front * back);
    }
    return k;
}

layout(push_constant) uniform PC {
    mat4 mvp;          // view_proj — for gl_FragDepth at the hit
    mat4 model;        // inverse(view_proj) — for world-ray reconstruction
    mat4 prev_mvp;     // unused here, kept for layout match
    vec4 color;        // x: max distance (m), y: wind strength,
                       // z: time (s), w: density bias
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

// Heightmap texture mapping — matches terrain_raymarch.frag's
// kHeightmapSide so my grass sits on the same surface the terrain
// raymarch draws (otherwise grass appears UNDER the terrain and gets
// occluded by the LESS_OR_EQUAL depth test → no grass visible).
const float kHeightmapSide = 2048.0;

float sampleHeightDelta(vec2 worldXZ) {
    // Branchless boundary handling — same trick as terrain_raymarch.frag.
    vec2 uv = clamp((worldXZ / kHeightmapSide) + vec2(0.5), 0.0, 1.0);
    return textureLod(u_heightmap, uv, 0.0).r;
}

// === Procedural FBM terrain (ported from terrain_raymarch.frag) ===
// Match its terrainM() output so grass stands on the same y the
// terrain renderer draws. Heightmap texture stores DELTA from baseline
// (sculpt edits); the absolute height is the procedural FBM + delta.

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// Hermite cubic — MUST match terrain_raymarch.frag's noised() and
// the CPU rm_noised bake exactly so grass blades sit on the same
// FBM surface the terrain pass renders. (Reverted quintic experiment.)
vec3 noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = hash21(i + vec2(0.0, 0.0));
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    float value = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    vec2  deriv = du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    return vec3(value, deriv);
}

const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);
const float TERRAIN_SCALE  = 0.003;
const float TERRAIN_HEIGHT = 120.0;
// Plateau matches the gameplay default (engine init: castle area).
const vec2  kPlateauCenter = vec2(0.0, 0.0);
const float kPlateauExtent = 11.5;   // matches world.cpp HeightmapParams
const float kPlateauHeight = 22.0;
const float kPlateauBlend  = 20.0;

float plateauWeight(vec2 worldXZ) {
    vec2 d = abs(worldXZ - kPlateauCenter) - vec2(kPlateauExtent);
    float dout = max(max(d.x, 0.0), max(d.y, 0.0));
    return 1.0 - smoothstep(0.0, kPlateauBlend, dout);
}

// Same FBM as terrain_raymarch.frag::terrainM with the per-frame
// octave count fixed at 8 (middle of the engine's 4..9 quality slider).
// Returns absolute terrain height = procedural FBM + plateau blend +
// sculpt delta. MUST stay at 8 octaves so grass blades sit on the
// same surface the terrain raymarch renders — reducing octaves here
// produced a smoother grass-positioning shape that drifted above /
// below the actual terrain green tint, leaving visible gaps.
float terrainH(vec2 wp) {
    vec2 p = wp * TERRAIN_SCALE;
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kOct = 8;
    for (int i = 0; i < kOct; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    h = mix(h, kPlateauHeight, plateauWeight(wp));
    h += sampleHeightDelta(wp);
    return h;
}

// Distance-LOD variant — fades the per-march octave count from a fixed
// 8 at ray_t < lod_near_m down to lod_min_octaves past lod_far_m. Same
// LOD ramp endpoints + fractional-octave morph as terrain_raymarch.frag,
// so grass blades sit on the same surface the terrain raymarch draws
// even at distance. Params come from pc.emissive.{y,z,w} (mirrors the
// terrain shader's push-constant layout).
// Absolute heightmap (dim+1 texels over kHeightmapSide, any cell size,
// centred on the origin) - the surface the
// MESH terrain is built from, including sculpts, the road and craters.
layout(set = 0, binding = 17) uniform sampler2D u_heightmap_full;
// Beyond the home tile: the ground around the camera, baked by the engine as it moves
// (vk_engine/veg_stream.cpp, wv_nf_poll): 512 x 512 m at 1 m, the world formula's heights
// packed into RGB (24-bit fixed point, exact under bilinear filtering). Row 0 holds the
// window's origin (texels 0 and 1, in 64 m steps) and a mark that the bake is there (texel 2).
layout(set = 0, binding = 4) uniform sampler2D u_normal[23];
vec2 g_nf_org = vec2(0.0);
bool g_nf_ok  = false;
void nf_setup() {
    vec3 m = texelFetch(u_normal[22], ivec2(2, 0), 0).rgb * 255.0;
    g_nf_ok = scene.globe_info.z > 0.5 && abs(m.r - 77.0) < 0.5 && abs(m.g - 123.0) < 0.5 && abs(m.b - 201.0) < 0.5;
    if (!g_nf_ok) return;
    vec3 a = texelFetch(u_normal[22], ivec2(0, 0), 0).rgb * 255.0;
    vec3 b = texelFetch(u_normal[22], ivec2(1, 0), 0).rgb * 255.0;
    g_nf_org = (vec2(a.r * 65536.0 + a.g * 256.0 + a.b, b.r * 65536.0 + b.g * 256.0 + b.b) - 8388608.0) * 64.0;
}
bool nf_inside(vec2 wp) {
    vec2 l = wp - g_nf_org;
    return g_nf_ok && all(greaterThan(l, vec2(3.0))) && all(lessThan(l, vec2(508.0)));
}
float nf_height(vec2 wp) {
    vec3 c = textureLod(u_normal[22], (wp - g_nf_org + 0.5) / 512.0, 0.0).rgb;
    return c.r * 255.0 + c.g * (255.0 / 256.0) + c.b * (255.0 / 65536.0) - 64.0;
}
float terrainH_mesh(vec2 wp) {
    if (max(abs(wp.x), abs(wp.y)) > kHeightmapSide * 0.5) return nf_inside(wp) ? nf_height(wp) : -1e4;      // (no ground known: no grass)
    float W = float(textureSize(u_heightmap_full, 0).x);
    vec2 uv = ((wp / kHeightmapSide + 0.5) * (W - 1.0) + 0.5) / W;
    return textureLod(u_heightmap_full, uv, 0.0).r;
}

// 1 = also with the raymarched terrain: blades stand on the baked
// heightmap (9 octaves at 1 m, incl. road / sculpt / crater deltas) plus
// the close-range relief. Profiling: the 8-octave FBM evaluated on every
// march step was ~70 % of this pass (3.3 -> 1.0 ms at 3440x1369), and the
// bake follows the drawn 10-12 octave surface at least as well as those 8
// octaves did.
#define VIBE_GRASS_BAKED_HEIGHT 1
float terrainH_lod(vec2 wp, float ray_t) {
    // One bilinear fetch instead of 8 octaves of FBM per march step.
    if (pc.emissive.x > 0.5 || VIBE_GRASS_BAKED_HEIGHT == 1) return terrainH_mesh(wp);
    vec2 p = wp * TERRAIN_SCALE;
    const int oct_full = 8;
    int   oct_min = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f   = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f   = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct     = int(ceil(oct_f));
    float frac    = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    for (int i = 0; i < oct_full; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    h = mix(h, kPlateauHeight, plateauWeight(wp));
    h += sampleHeightDelta(wp);
    return h;
}

// Backwards-compat: rest of the shader still calls sampleHeight.
float sampleHeight(vec2 wp_xz) { return (pc.emissive.x > 0.5 || VIBE_GRASS_BAKED_HEIGHT == 1) ? terrainH_mesh(wp_xz) : terrainH(wp_xz); }
// Close-range ground relief of terrain_raymarch.frag (rm_ground_detail:
// rills + side cracks + sand ripples, the dirt texture's displacement at
// its mean; CPU twin raymarch_ground_relief). The raymarched ground
// rises by it within 26 m, so the blades must too - otherwise their
// lower ~17 cm vanished into the ground close to the camera.
float grass_ground_relief(vec2 wp, float t, float h) {
    float w = 1.0 - smoothstep(10.0, 26.0, t);
    if (w <= 0.0 || pc.emissive.x > 0.5) return 0.0;   // mesh terrain: no relief
    float wn = tile_noise2(wp, 0.21, vec2(0.0));       // (as terrain_rm_heightfield.glsl rm_ground_detail: keep in step)
    vec2  qo = vec2(wn, 1.0 - wn) * 1.7;
    float g1 = 1.0 - smoothstep(0.0, 0.22, abs(tile_noise2(wp, 0.55, qo) * 2.0 - 1.0));
    float g2 = 1.0 - smoothstep(0.0, 0.25, abs(tile_noise2(wp, 1.485, qo * 2.7 + 3.1) * 2.0 - 1.0));
    float d = 0.11 * (1.0 - g1) + 0.045 * (1.0 - g2) + 0.045;
    float sand = 1.0 - smoothstep(scene.water_params.y + 0.4, scene.water_params.y + 2.5, h);
    d += sand * 0.03 * (0.5 + 0.5 * sin(wp.x * TILE_W(9.0) + wp.y * TILE_W(3.5) + 6.0 * wn));
    return d * w;
}
float sampleHeight_lod(vec2 wp_xz, float ray_t) {
    float h = terrainH_lod(wp_xz, ray_t);
    return h + grass_ground_relief(wp_xz, ray_t, h);
}
// March variant: while the ray is well above anything the relief (max
// ~0.275 m) or a blade on it could reach, a constant upper bound replaces
// the three noise taps (most steps are in open air).
float sampleHeight_lod_ray(vec2 wp_xz, float ray_t, float ray_y) {
    float h = terrainH_lod(wp_xz, ray_t);
    if (ray_y - h > 1.6) return h + ((pc.emissive.x > 0.5) ? 0.0 : 0.30 * (1.0 - smoothstep(10.0, 26.0, ray_t)));
    return h + grass_ground_relief(wp_xz, ray_t, h);
}

// Hash helper still used by per-cell blade variation.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec4 hash42(vec2 p) {
    vec4 p4 = fract(vec4(p.xyxy) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

mat2 rotate2d(float t) {
    float c = cos(t), s = sin(t);
    return mat2(c, -s, s, c);
}

float opSubtraction(float d1, float d2) { return max(d1, -d2); }

vec2 opRepeat(vec2 p, vec2 period, out vec2 outId) {
    outId = floor((p + 0.5 * period) / period);
    return mod(p + 0.5 * period, period) - 0.5 * period;
}

float sdCircle(vec2 p, float r) { return length(p) - r; }

// 2D blade silhouette (per shadertoy): two circle subtractions clip the
// crescent shape, then the bottom half-plane truncates the base.
float sdGrassBlade2d(vec2 p) {
    float d = sdCircle(p - vec2(1.7, -1.3), 2.0);
    d = opSubtraction(d, sdCircle(p - vec2(1.7, -1.0), 1.8));
    d = opSubtraction(d, p.y + 1.0);
    d = opSubtraction(d, -p.x + 1.7);
    return d;
}

// Extrude the 2D shape to 3D with a thickness — sqrt(d² + z²) - thickness
// gives a smooth round bevel out of the slab.
float sdGrassBlade(vec3 p, float thickness) {
    p -= vec3(0, 1.0, 0);
    float d2 = max(0.0, sdGrassBlade2d(p.xy));
    return sqrt(d2 * d2 + p.z * p.z) - thickness;
}

// Distance to the grass field at a world-space point. terrainY is sampled
// once per call and reused for the cell-neighbor loop (the original SDF
// would call calcTerrainHeight per neighbor — ~9× more samples).
//
// out_blade_hit flags whether the closest feature is a grass blade
// (true) or just the terrain plane (false). Caller uses this to
// `discard` terrain-plane hits — those would z-fight with the actual
// rasterised / raymarched terrain renderer's output and read as a
// flickering green tint over bare ground.
// Per-pixel SDF invariants — slider-driven, identical for every map()
// call, so the outer raymarch loop hoists them once and passes via
// this struct. Cuts the per-step setup cost from 128× per pixel to 1×.
struct MapCtx {
    float slope_max;
    float density_eff_near;   // density at t = soft_d
    float density_eff_far;    // density at t = kMaxD
    float soft_d;
    float kMaxD;
    bool  underwater;         // global toggle (water_params.x > 0.5)
    float water_y;
};

// Rotation of the blade that won the last map() call (= the hit blade
// after the march): gives the blade face normal for lighting.
float g_blade_rot = 0.0;
vec3  g_blade_r   = vec3(0.5);   // that blade's hashes (colour variation)
int   g_blade_kind = 0;          // 0 blade, 1 seed head, 2 flower
// The march's place relative to the camera, on the ground plan (set before every map()):
// the blade lattice is worked out from it, not from the world position - see map().
vec2 g_rel_xz = vec2(0.0);
// How far the last map() sample stood above the tops of the blades (m); -1 when it was among them.
float g_above_top = -1.0;
float map(vec3 p, float ray_t, MapCtx ctx, out bool out_blade_hit) {
    out_blade_hit = false;
    g_above_top = -1.0;
    // Open air (raymarched terrain): one fetch of the baked heightmap
    // instead of the 8-octave FBM. The rendered surface is at most ~0.8 m
    // above the bake (finer octaves + close-range relief) and blades < 1 m,
    // so 2 m of margin is always clear. This is the same skip the terrain
    // pass uses; without it every step from the camera to the ground paid
    // for the full FBM (the bulk of this pass).
    float terrainY;
#if VIBE_GRASS_BAKED_HEIGHT == 1
    {
        // One fetch of the baked height serves the air skip AND the ground
        // (it used to be fetched twice per near-ground step).
        float hmesh = terrainH_mesh(p.xz);
        float hm = p.y - hmesh;
        if (pc.emissive.x < 0.5 && hm > 3.0) return 0.5 * (hm - 2.0);
        // The close relief (three noise look-ups) only where a blade can be. Not while the ray
        // is above the tallest blade standing on the HIGHEST the relief can be:
        float relief_ub = (pc.emissive.x > 0.5) ? 0.0 : 0.30 * (1.0 - smoothstep(10.0, 26.0, ray_t));
        const float top_ub = 1.2 * clamp(pc.grass_params.w, 0.05, 3.0) * 0.36 * kGsMaxH + 0.10 + 0.3;
        if (hm - relief_ub + 0.09 > top_ub) {
            float above = hm - relief_ub + 0.09 - (top_ub - 0.3);
            g_above_top = above;
            return 0.5 * above;
        }
        // And not over ground that carries no grass (road, sand, rock - the mask at this place,
        // before the gusts bend the lattice): there the ray only has to find the ground and be
        // thrown away. The baked height without relief lies under the drawn ground, so nothing
        // that could be seen is cut off. (A road through a meadow is half the screen: its rays
        // paid three noises a step all the way down, for nothing.)
        if (pc.emissive.x < 0.5 && max(abs(p.x), abs(p.z)) < kHeightmapSide * 0.5 - 1.0) {
            vec2 mg0 = textureLod(u_grass_mask, p.xz / kHeightmapSide + vec2(0.5), 0.0).rg;
            float e0 = mg0.r * (1.0 - smoothstep(ctx.slope_max * 0.7, ctx.slope_max * 1.2, mg0.g * 2.0));
            if (e0 + 0.175 * scene.grass_side_lit_params.w * smoothstep(0.04, 0.30, e0) < 0.07) return 0.5 * (hm + 0.09);
        }
        terrainY = hmesh + grass_ground_relief(p.xz, ray_t, hmesh);
    }
#else
    if (pc.emissive.x < 0.5) {
        float hm = p.y - terrainH_mesh(p.xz);
        if (hm > 3.0) return 0.5 * (hm - 2.0);
    }
    terrainY = sampleHeight_lod_ray(p.xz, ray_t, p.y);
#endif
    // Raymarched terrain: the blades stand on the 1 m baked height, the
    // drawn ground has finer bumps on top of it (+-several cm). Where the
    // bake was the higher one the blades ended in mid-air above the soil.
    // Root them a little lower - the real ground (depth test) hides what is
    // below it, so every blade now comes out of the ground.
    if (pc.emissive.x < 0.5) terrainY -= 0.09;
    float distToTerrain = 0.5 * (p.y - terrainY);

    // Above the tallest possible blade: no cell work at all (mask fetch,
    // hashes, rotations - 4 to 9 cells per step). The guard follows the
    // actual blade height (1.75 x the scaled base height covers every
    // blade); it used to be a fixed 2 m, sized for the slider's maximum -
    // an eye at 1.7 m was inside it for its whole ray.
    // (1.12 x the scaled height is the tip of the tallest blade of the
    // tallest grass type; + its seed head.)
    const float bh0 = clamp(pc.grass_params.w, 0.05, 3.0) * 0.36;
    {
        float blade_top = 1.2 * bh0 * kGsMaxH + 0.10;
        float above = p.y - terrainY;
        if (above > blade_top + 0.3) { g_above_top = above - blade_top; return 0.5 * (above - blade_top); }
    }

    vec2 cellId;
    const float kPeriod = TILE_CELL(0.17);        // ~35 blades per m²
    vec3 lp = p;
    lp.y -= terrainY;                  // ground-relative
    vec2 gust_sh = vec2(0.0);
    // Gusts (the wind of "Grass Scene #1" by Yusef28, shadertoy 7stSzf): the
    // space the blades stand in is bent by a travelling wave, more the higher
    // above the ground - so the blades of a patch lean together and waves
    // roll across the meadow. (Each blade's own small sway is added below.)
    {
        float gt = pc.color.z;
        vec2  gw = sin(gt * 1.6 + p.xz * TILE_W(0.30) - cos(gt * 0.45 - p.xz * TILE_W(0.13)));
        float gh = clamp(lp.y / (1.75 * clamp(pc.grass_params.w, 0.05, 3.0) * 0.36), 0.0, 1.0);
        gust_sh = (gw * 0.65 + vec2(0.35, 0.20)) * (pc.color.y * 0.22 * gh * gh);
    }
    {
        // The blade lattice, from the camera outward: the camera's cell and its place in it
        // come from the CPU in doubles (scene.grass_cam), the march's offset from the camera
        // is a small number, and so is everything below. Worked out from the world position
        // a blade - a few millimetres wide - broke into blocks wherever the coordinates were
        // large (a float resolves 1 mm at 14 km).
        vec2 v = scene.grass_cam.zw + g_rel_xz - gust_sh;
        vec2 cl = floor(v / kPeriod);
        cellId = scene.grass_cam.xy + cl;
        lp.xz = v - cl * kPeriod - 0.5 * kPeriod;
    }

    // Altitude + water gates — match terrain_raymarch.frag's
    // grassEligibility so the grass shader and the green terrain tint
    // agree about where grass exists.
    float alt_factor = 1.0 - smoothstep(55.0, 75.0, terrainY);
    // (not on the strip the water laps over either: the beach, see terrain_raymarch.frag)
    bool underwater  = ctx.underwater && terrainY < ctx.water_y + 0.36;
    if (alt_factor <= 0.001 || underwater) return distToTerrain;
    // Castle keep-out: the courtyard floor slabs sit 5 cm above the
    // terrain, and blades rooted on the terrain poke up THROUGH them —
    // the courtyard read as a swampy green weave over the tiles. The
    // castle footprint is the plateau extent (kPlateauExtent) centred
    // on origin; no blades inside it.
    if (abs(p.x) < kPlateauExtent + 0.5 &&
        abs(p.z) < kPlateauExtent + 0.5) return distToTerrain;

    float d = distToTerrain;
    const float kPi2 = 6.28318530718;
    // Density at this pixel's distance — interp between the two
    // pre-baked endpoints. Loop-`t` is the distance from camera (ro
    // is camera_pos and rd is normalised), so length(p - cam) was
    // wasted sqrt+dot per map() call.
    float dist_w      = smoothstep(ctx.soft_d, ctx.kMaxD, ray_t);
    float density_eff = mix(ctx.density_eff_near, ctx.density_eff_far, dist_w);
    // Dynamic close-range density: extra blades per cell near the camera,
    // fading out smoothly (each extra blade appears by the per-cell hash,
    // so the lawn thickens gradually as you approach - no ring).
    // (Applied per blade below from the blade cell's own distance - it used
    // to depend on the march sample, so blades shimmered in and out.)
    float near_boost = scene.grass_shore_params.y * step(0.05, ctx.density_eff_near);
    float density_max = density_eff + near_boost * (1.0 - smoothstep(3.5, 10.0, ray_t - 1.0));
    // Aggressive zero-density early-out. When distance has faded
    // density to near zero, the inner 9-cell loop almost always finds
    // every cell rejected by `r.x > density_eff` — cost without
    // benefit. 0.03 cutoff keeps a few stragglers visible past the
    // soft fade so it doesn't read as a hard ring.
    if (density_max < 0.03) return distToTerrain;
    const float kEligThresh = 0.10;
    // Eligibility (presence mask x slope) ONCE per map() call, at the
    // centre cell. The mask is 2 m per texel and bilinear; the 3x3 cell
    // block spans 0.5 m, so the nine per-cell fetches returned the same
    // value to within a few percent - at nine texture reads per march
    // step. Bare ground / rock / road now also skips the whole blade loop.
    float elig;
    {
        vec2 mg = textureLod(u_grass_mask, (cellId * kPeriod / kHeightmapSide) + vec2(0.5), 0.0).rg;
        float slope_w = 1.0 - smoothstep(ctx.slope_max * 0.7, ctx.slope_max * 1.2, mg.g * 2.0);
        elig = mg.r * slope_w;
        // Grass grows on the home landscape only (the mask and the heights it stands on are that
        // 2 km tile's). Beyond it the mask repeated and the heights were the tile's edge: blades
        // stood in the air, under the ground and on the sea bed of the wider world.
        vec2 gq = abs(cellId * kPeriod);
        if (max(gq.x, gq.y) > kHeightmapSide * 0.5) {
            // The wider world: grass by rule, on the baked ground around the camera - from just
            // above the shore up to where the turf ends, not on steep ground, thinning out
            // toward the edge of the baked window.
            vec2 cw = cellId * kPeriod;
            elig = 0.0;
            if (nf_inside(cw)) {
                float h0 = nf_height(cw);
                vec2  sg = vec2(nf_height(cw + vec2(2.0, 0.0)) - h0, nf_height(cw + vec2(0.0, 2.0)) - h0) * 0.5;
                float wl = scene.water_params.y;
                vec2  lc = abs(cw - g_nf_org - 256.0);
                elig = smoothstep(wl + 1.2, wl + 2.6, h0) * (1.0 - smoothstep(52.0, 70.0, h0)) *
                       (1.0 - smoothstep(ctx.slope_max * 0.7, ctx.slope_max * 1.2, length(sg))) *
                       (1.0 - smoothstep(200.0, 245.0, max(lc.x, lc.y))) *
                       smoothstep(84.0, 92.0, distance(cw, vec2(1600.0, 830.0)));       // (the launch site's apron: world_map.glsl wm_pad_c)
                // (a site's street and buildings: cut out in the bake's alpha - veg_stream.cpp)
                elig *= smoothstep(0.25, 0.75, textureLod(u_normal[22], (cw - g_nf_org + 0.5) / 512.0, 0.0).a);
            }
        } else if (!g_nf_ok) {
            // (thinning out over the last 170 m, where terrain_raymarch.frag brings in its turf;
            // with the world's grass beyond it the meadow simply carries on)
            elig *= 1.0 - smoothstep(kHeightmapSide * 0.5 - 170.0, kHeightmapSide * 0.5 - 10.0, max(gq.x, gq.y));
        }
    }
    // Soft patch border ('grass edge softness'): the mask alone gave a
    // wall of full-height grass next to bare ground. A low-frequency noise
    // frays the outline, and toward it the grass thins out and shortens.
    float edge_soft = scene.grass_side_lit_params.w;
    float elig_gate = smoothstep(0.04, 0.30, elig);     // no noise on cleared ground (road, rock)
    if (elig + 0.175 * edge_soft * elig_gate < kEligThresh) return distToTerrain;
    // The grass type of this patch (grass_blades.glsl): height, width, lean,
    // dry stalks. Once per call - it changes over metres, the cell block is
    // half a metre. Above this patch's own blades: no cell work either.
    const float type_n = gs_type_noise(cellId * kPeriod);
    const vec4  tp     = gs_type(type_n);
    {
        float top = 1.2 * bh0 * tp.x + 0.10;
        if (lp.y > top + 0.3) { g_above_top = lp.y - top; return min(distToTerrain, 0.5 * (lp.y - top)); }
    }
    // Small flowers stand in drifts that follow the type field.
    const bool flower_drift = fract(type_n * 9.0) < 0.35;
    // Drop the 8 neighbour cells past ~35 % of kMaxDist — at that
    // distance blades are sub-pixel and the neighbour offsets
    // contribute < 1 pixel of silhouette difference. Keeps centre
    // cell only → 9× cheaper per map() call on long-range pixels.
    // Scales with kMaxDist (was hardcoded 40 m which capped well
    // before kMaxDist when the user's grass-distance slider was set
    // above 100 m).
    int rad = (ray_t > ctx.kMaxD * 0.35) ? 0 : 1;
    // Centre cell first: its blades are usually the nearest, and a small
    // `d` early lets the slab test below reject most of the other blades.
    const ivec2 kCellOrder[9] = ivec2[9](ivec2(0, 0), ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1),
                                         ivec2(1, 1), ivec2(-1, 1), ivec2(1, -1), ivec2(-1, -1));
    // 3x3 only close up (blades are large on screen there and one cut off
    // at a cell border would show); from 8 m the four cells nearest the
    // sample (its own + the three toward the quadrant it sits in); past the
    // cutoff above just its own.
    const bool quad = rad > 0 && ray_t > 2.5;
    const int ncell = (rad > 0) ? (quad ? 4 : 9) : 1;
    const ivec2 qs = ivec2(lp.x >= 0.0 ? 1 : -1, lp.z >= 0.0 ? 1 : -1);
    for (int ci = 0; ci < ncell; ++ci) {
        {
            const ivec2 cofs = quad ? ivec2((ci & 1) != 0 ? qs.x : 0, (ci & 2) != 0 ? qs.y : 0) : kCellOrder[ci];
            const int dx = cofs.x, dy = cofs.y;
            vec2 nId = cellId + vec2(float(dx), float(dy));
            // Density-driven per-cell skip — only hash the cells we
            // didn't already reject by mask.
            // Up to two blades per cell: density 0..1 fills cells, 1..2 adds
            // a second blade (near rings only - the far single-cell path
            // keeps one). Each blade gets its own lean and width variation.
            // Per-BLADE-CELL edge noise and near boost (stable: they depend on
            // the cell, not on where along the ray this sample is).
            vec2  cw = nId * kPeriod;
            // The edge noise only matters near a patch border: well inside a
            // meadow (elig above the fade band whatever the noise) the two
            // hashes are skipped - this loop runs 9x per march step.
            float edge_k = 1.0;
            if (elig < kEligThresh + 0.55 + 0.18 * edge_soft) {
                float en = hash12(tile_wrap(floor(cw / TILE_CELL(0.9)), TILE_N(0.9))) * 0.6 + hash12(tile_wrap(floor(cw / TILE_CELL(2.7)), TILE_N(2.7)) + 17.0) * 0.4;
                float elig_n = elig + (en - 0.5) * 0.35 * edge_soft * elig_gate;
                if (elig_n < kEligThresh) continue;
                edge_k = mix(1.0, smoothstep(kEligThresh, kEligThresh + 0.55, elig_n), edge_soft);
            }
            // Near-density boost: only within its range (no sqrt beyond it).
            float dens_n = density_eff;
            if (near_boost > 0.0 && ray_t < 11.5) {
                float cdist = distance(vec3(cw.x, p.y, cw.y), scene.camera_pos.xyz);
                dens_n += near_boost * (1.0 - smoothstep(3.5, 10.0, cdist));
            }
            const int nb = (rad > 0) ? clamp(int(ceil(dens_n)), 1, 3) : 1;
            for (int bi = 0; bi < nb; ++bi) {
            vec4 r = hash42(tile_wrap(nId, TILE_N(0.17)) + vec2(31.7, 47.3) * float(bi));
            if (r.x > (dens_n - float(bi)) * mix(0.25, 1.0, edge_k)) continue;
            vec3 np = lp - vec3(float(dx), 0.0, float(dy)) * kPeriod;
            // Per-blade rotation + sub-cell xy offset.
            np.xz = rotate2d(r.z * kPi2) * np.xz;
            np.xz += (r.xy - 0.5) * kPeriod;
            float bh = bh0 * mix(0.45, 1.0, edge_k);
            // Under a car, and behind it where it has just been: the blade lies down (car_press).
            float press = car_press(cw);
            bh *= 1.0 - 0.90 * press;
            // The blade's own size: 0.3 .. 1 of its type's height (the short
            // ones fill the base of the lawn), narrower when shorter but not
            // in proportion, so a short blade is still a leaf and not a hair.
            float s  = gs_blade_s(r.w);
            float hs = bh * tp.x * s;
            float ws = bh * tp.y * (0.45 + 0.55 * s);
            // What stands on the tip: a seed head on the taller dry stalks, a
            // small flower here and there in the green grass. Close range only
            // (they shrink away with distance instead of sparkling).
            int kind = 0;
            if (ray_t < 28.0 && s > 0.55) {
                if (tp.w > 0.5) { if (fract(r.z * 17.3) < 0.55) kind = 1; }
                else if (flower_drift && fract(r.y * 23.7) < 0.022) kind = 2;
            }
            // Cheap reject before the wind / silhouette maths: a blade is a
            // thin slab in its local z (distance >= |z| - thickness) and has a
            // top (1.12 x its height, + the head). Further than the best so far: skip.
            if (abs(np.z) - (kind != 0 ? 0.022 : 0.008) >= d || np.y - (1.15 * hs + 0.08) >= d) continue;
            // Wind sway + a per-blade static lean (some blades stand, some
            // splay out), bending with height.
            float wind_t = pc.color.z + r.w * kPi2;
            float bend   = sin(wind_t * 1.4) * pc.color.y * 0.025 * (0.4 + 0.6 * s);
            // (coming up again behind the car it springs to and fro, most when half way up)
            bend += press * (1.0 - press) * 0.22 * sin(pc.color.z * 9.0 + r.w * 40.0);
            float lean   = (fract(r.w * 13.7 + r.z * 5.3) - 0.5) * 0.30 * tp.z * s;
            np.x += (bend + lean * bh) * smoothstep(0.0, 1.5 * hs, np.y);
            // Height and width scale separately: the distance is then a lower
            // bound (x the smaller of the two), which is what the march needs.
            float m     = min(hs, ws);
            float thick = 0.002 * (0.7 + 0.8 * fract(r.z * 7.1));
            float bladeDist = sdGrassBlade(np / vec3(ws, hs, ws), thick / m) * m;
            int   hit_kind  = 0;
            if (kind != 0) {
                float hf = 1.0 - smoothstep(16.0, 28.0, ray_t);
                vec3  q  = np - vec3(0.289 * ws, 1.117 * hs, 0.0);      // the blade's tip
                float hd;
                if (kind == 1) {            // seed head: a slim spike
                    q.y -= clamp(q.y, -0.012, 0.055 * hf);
                    hd = length(q) - 0.0065 * hf;
                } else {                    // flower: a small flattened ball
                    q.y *= 1.8;
                    hd = (length(q) - 0.017 * hf) / 1.8;
                }
                if (hd < bladeDist) { bladeDist = hd; hit_kind = kind; }
            }
            if (bladeDist < d) {
                d = bladeDist;
                out_blade_hit = true;
                g_blade_rot = r.z * kPi2;
                g_blade_r = r.yzw;
                g_blade_kind = hit_kind;
            }
            }
        }
    }
    return d;
}

vec3 calcTerrainNormalApprox(vec2 p, float yC, float ray_t) {
    const float e = 0.05;
    float yR = sampleHeight_lod(p + vec2(e, 0.0), ray_t);
    float yU = sampleHeight_lod(p + vec2(0.0, e), ray_t);
    return normalize(vec3(yC - yR, e, yC - yU));
}

vec3 sample_sky_grad(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, pow(up, 0.45));
}

void main() {
    nf_setup();
    // Reconstruct world ray from screen UV via inverse view-proj.
    vec2 uv  = gl_FragCoord.xy * scene.viewport.zw;
    vec4 ndc = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 ro  = scene.camera_pos.xyz;
    vec3 rd  = scene_ray(ndc.xy);          // (not from pc.model: see scene_ray - the grass came out in blocks far from the origin)

    // (Removed early-out on rd.y > 0. Mountains rise above eye height,
    // so an upward-pointing ray can still hit terrain on a peak — the
    // cut-out used to draw a hard horizon line at exactly the screen's
    // mid-height which read as a cliff in the grass coverage.)

    const float kMaxDist = max(pc.color.x, 4.0);
    // Cut the absolute step cap — 192 was overkill for most pixels and
    // burned the worst-case budget on distant grazing rays that never
    // converge. With the distance-ramped step factor below the loop
    // covers the same ground with fewer iterations at distance.
    // GPU-TDR safety cap: when the player stands IN dense grass and
    // looks horizontally across it, every pixel can burn the worst-
    // case budget × 9 cells × hash+sin+sqrt per step. Past a certain
    // budget the cost outruns the Windows TDR (~2 s on most drivers)
    // and the device is lost. 96 still gives plenty of convergence
    // headroom for normal play; the distance-aware breaks below
    // already collapse far rays to 16-32. Was 128.
    const int   kMaxSteps = 96;
    float t = 0.0;
    bool hit = false;
    bool blade_hit = false;
    float hit_d = 0.0;            // SDF distance at the converged hit
    float hit_thresh = 0.0;       // hit threshold at that t — alpha denom
    // Pixel size projected at unit distance — used to early-out when the
    // SDF distance is below sub-pixel detail at this t.
    float pixSize = 2.0 * scene.viewport.z;

    // Per-pixel map() invariants — slider-driven values that don't
    // change per step. Computing once here saves the per-step setup
    // cost (sqrt, smoothstep, several muls + UBO loads) × 128 steps.
    MapCtx ctx;
    {
        float slope_n_min = scene.grass_extra.z;
        ctx.slope_max = sqrt(max(1e-4,
                                  1.0 / max(slope_n_min * slope_n_min, 1e-4) - 1.0));
        float density   = clamp(scene.grass_color_top.w, 0.0, 2.0);   // >1: 2nd blade per cell
        float dist_dens = clamp(scene.grass_extra.w, 0.0, 1.0);
        float far_mul   = max(1.0 - dist_dens * 0.85, 0.0);
        ctx.density_eff_near = density;
        ctx.density_eff_far  = density * far_mul;
        ctx.kMaxD = max(pc.color.x, 4.0);
        ctx.soft_d = max(pc.color.w, 1.0);
        ctx.underwater = scene.water_params.x > 0.5;
        ctx.water_y    = scene.water_params.y;
    }

    // Distance-aware step budget. Past ~55 % / ~85 % of the user's
    // kMaxDist the worst-case grazing rays burn the full 128 budget
    // converging on sub-pixel blade edges that won't matter — TAA
    // absorbs the residual noise. Scale with kMaxDist so the user's
    // grass-distance slider keeps working at any value (was hardcoded
    // 60 m / 100 m which capped well before kMaxDist when the slider
    // pushed past 100 m).
    // Lossless march bound from the prepass depth: past the nearest
    // opaque surface every hit would fail the depth test anyway. Turns
    // castle walls / floors / props from a full 96-step march into ~0.
    float t_limit = kMaxDist;
    int   i_bias  = 0;
    {
        float zpre = texelFetch(u_prepass_depth, ivec2(gl_FragCoord.xy), 0).r;
        if (zpre < 0.99999) {
            vec4 sp = pc.model * vec4(ndc.xy, zpre, 1.0);
            float t_surf = distance(sp.xyz / sp.w, ro);
            t_limit = min(t_limit, t_surf + 0.05);
            // Skip the empty air: when the prepass surface IS the terrain,
            // blades (< ~0.7 m tall) can only touch this ray within a
            // short distance before it reaches the ground. Start the march
            // there instead of at the camera. At grazing angles that
            // stretch is ~0.9 m / cos(angle to the ground normal), capped
            // at 14 m (blade tips on ridges further back may be skipped -
            // sub-pixel at those angles). Was the bulk of the march cost:
            // ~all steps were spent sphere-tracing through air.
            // NOT when the prepass surface is under water: the depth there
            // is the lake BED (water isn't in the prepass), far behind the
            // shore blades that silhouette against the water - starting
            // the march near the bed skipped them all (no grass in front
            // of water).
            // ONLY with the mesh terrain (pc.emissive.x): the raymarched terrain
            // is not in the prepass, so there the prepass surface is always an
            // object. Its low parts (trunk bases, rocks, small bushes) pass the
            // 'near the ground' test below even when they stand BEHIND a hill -
            // the march then started out at the object and skipped every blade
            // on the hill in front, punching the object's outline through the
            // grass.
            // (the height under the object is asked only with the mesh terrain: with the
            //  raymarched one it was a fetch and three noises for every pixel under a tree)
            vec3  ps = sp.xyz / sp.w;
            float yS = (pc.emissive.x > 0.5) ? sampleHeight_lod(ps.xz, t_surf) : 1e9;
            if (pc.emissive.x > 0.5 &&
                abs(ps.y - yS) < 0.6 && ps.y > scene.water_params.y + 0.05) {
                vec3  Ns   = calcTerrainNormalApprox(ps.xz, yS, t_surf);
                float cosg = abs(dot(rd, Ns));
                float back = min(14.0, 0.9 / max(cosg, 0.06));
                t = max(t, t_surf - back);
                // Keep the old far-ray budget: a camera-start march spent
                // ~24 steps in air before reaching far ground, so the
                // distance caps below broke those rays early; without this
                // they converge on sub-pixel far blades (distant sparkle).
                i_bias = (t > kMaxDist * 0.55) ? 24 : 0;
            }
        }
    }
    // Skip the open air (raymarched terrain, settings 'low-resolution march first'): the
    // nearest of the four blocks round this pixel says where a ray first comes near the
    // ground. Nothing nearer than that can be a blade.
    if ((int(scene.rm_perf.x) & 1) != 0 && pc.emissive.x < 0.5) {
        vec4 g4 = textureGather(u_march_lo, gl_FragCoord.xy / (scene.rm_perf.y * vec2(textureSize(u_march_lo, 0))), 1);
        float tg = min(min(g4.x, g4.y), min(g4.z, g4.w));
        if (tg > 1e8) discard;
        if (tg > 0.5) t = max(t, tg - 0.3);
    }
    float step_cap_a = kMaxDist * 0.55;
    float step_cap_b = kMaxDist * 0.85;
    // A ray that skims the meadow travels tens of metres inside the layer of blades, a few
    // centimetres a step: it ran out of steps before it met a blade, and was thrown away - a
    // band without grass at middle distance (20 - 100 m), between the near grass (steep rays)
    // and the far (a coarse hit threshold). Such a ray has certainly met a blade by then: when
    // the march gives up while it is among the blades, that counts as the hit.
    bool gave_up_in_grass = false;
    bool last_blade = false;
    for (int i = 0; i < kMaxSteps; ++i) {
        if (t >= t_limit) break;
        if ((i + i_bias >= 32 && t > step_cap_a) || (i + i_bias >= 16 && t > step_cap_b) || i == kMaxSteps - 1) {
            gave_up_in_grass = last_blade;
            break;
        }
        vec3 p = ro + t * rd;
        bool step_blade;
        g_rel_xz = t * rd.xz;
        float d = map(p, t, ctx, step_blade);
        last_blade = step_blade && d < 0.12;
        // Hit threshold loosens with distance so the loop terminates on
        // sub-pixel features instead of trying to converge on a blade
        // edge from 50 m away.
        float thresh = 0.5 * pixSize * t + 0.0008;
        if (d < thresh) {
            hit = true;
            blade_hit = step_blade;
            hit_d = max(d, 0.0);
            hit_thresh = thresh;
            break;
        }
        // Distance-ramped step factor. Close-up needs the 0.7 cone-
        // safety factor for grazing rays through dense blade fields.
        // Past 50 m blades are sub-pixel and the cone-safety overhead
        // is pure cost — extend the ramp to 2.0 by 120 m so distant
        // rays cover 3× the ground per iteration.
        float step_f = 0.95;   // (was 0.70, then 0.82 near the camera: the blade field is close to a true distance - same picture, a tenth fewer steps)
        step_f = mix(step_f, 2.0, smoothstep(50.0, 120.0, t));
        float adv = d * step_f;
        // Above the blades the bound is the HEIGHT over their tops - for a ray that skims the
        // meadow, coming down a few centimetres a metre, a step of half that height closed 1 %
        // of the gap: 96 steps did not reach the grass (the band without blades at middle
        // distance). Along such a ray the tops are height / fall away - less, as the ground
        // ahead may rise (taken as up to 1 in 2).
        if (g_above_top > 0.0) adv = max(adv, min(g_above_top / (max(-rd.y, 0.0) + 0.5), 8.0));
        t += adv;
    }

    // Discard pure terrain-plane hits (no grass blade closer) so the
    // existing terrain renderer's pixel shows through unaltered. Without
    // this, no-grass cells (presence/slope rejected) painted the bare
    // ground green at z-fight depth → flickering green tint.
    if (gave_up_in_grass && !hit && t > 8.0) { hit = true; blade_hit = true; hit_d = 0.0; hit_thresh = 1.0; }
    if (!hit || !blade_hit) { discard; }

    vec3 p = ro + t * rd;
    vec3 L = normalize(scene.sun_direction.xyz);

    // Lighting: terrain normal at hit point + half-Lambert + sky fill.
    float terrainY = sampleHeight_lod(p.xz, t);
    vec3 N = calcTerrainNormalApprox(p.xz, terrainY, t);
    float n_dot_l = max(0.0, dot(N, L));

    // Sun shadow — sample the rasterised shadow map. PCF gives soft edges.
    float sun_lit = 1.0;
    if (!FLAT_GRASS) {
        vec4 lc  = scene.light_vp * vec4(p, 1.0);
        if (abs(lc.w) > 1e-5) {
            vec3 lndc = lc.xyz / lc.w;
            vec2 luv  = lndc.xy * 0.5 + 0.5;
            if (luv.x >= 0.0 && luv.x <= 1.0 &&
                luv.y >= 0.0 && luv.y <= 1.0 &&
                lndc.z >= 0.0 && lndc.z <= 1.0) {
                sun_lit = textureLod(u_sun_shadow_map,
                                      vec3(luv, lndc.z - 0.0005), 0.0);
            }
        }
    }
    // RT sun shadow: trees, castle and props cast onto the grass (the
    // raster shadow map misses most of them). One ray per grass pixel,
    // started a little up the sun direction so the blade's own ground
    // (terrain BLAS) can't self-hit at the base.
    if (sun_lit > 0.0 && L.y > 0.0) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS,
                              gl_RayFlagsTerminateOnFirstHitEXT,
                              0x01u, p + vec3(0.0, 0.04, 0.0), 0.3, L, 300.0);
        RQ_PROCEED_LEAVES(rq)
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
            sun_lit = 0.0;
    }

    // Per-blade colour: lerp between bottom (base) and top (tip) by
    // height above ground. Both ends are user-driven via UI sliders
    // (scene.grass_color_bottom / grass_color_top). Slight low-freq
    // hash modulation breaks regularity without needing two separate
    // "young/old" colour vars.
    float h_above = max(0.0, p.y - terrainY);
    // Normalise by the actual max blade height (1.7 SDF units × height
    // factor) so the base→tip gradient spans the whole blade. The old
    // /1.2 assumed the unscaled 1.2+ m blades — with lawn-scale blades
    // it left every pixel at the dark base colour.
    // (x the height of this patch's grass type, so short lawn and tall grass
    // both run the whole gradient.)
    float type_n  = gs_type_noise(p.xz);
    vec4  type_p  = gs_type(type_n);
    float bh_max  = 1.7 * clamp(pc.grass_params.w, 0.05, 3.0) * 0.36 * type_p.x;
    float h_t     = min(h_above / bh_max, 1.0);
    float clump   = hash12(tile_wrap(floor(p.xz / TILE_CELL(2.0)), TILE_N(2.0))) * 0.15;   // ±15% per-clump tint shift
    vec3 grassCol = mix(scene.grass_color_bottom.rgb,
                        scene.grass_color_top.rgb, h_t);
    grassCol *= 1.0 - clump;
    // Colour by grass type and by blade. All of it is relative to the two
    // colours of the settings, so those still set the look of the meadow.
    vec3 straw = dot(scene.grass_color_top.rgb, vec3(0.3333)) * vec3(1.90, 1.55, 0.75);
    {
        float lawn_w   = 1.0 - smoothstep(0.40, 0.47, type_n);
        float coarse_w = smoothstep(0.53, 0.60, type_n);
        grassCol *= mix(vec3(1.0), vec3(0.95, 1.10, 0.86), lawn_w);        // fine lawn: fresh
        grassCol *= mix(vec3(1.0), vec3(0.74, 0.86, 0.72), coarse_w);      // coarse grass: deep, dull
        grassCol  = mix(grassCol, straw, type_p.w * mix(0.25, 0.80, h_t)); // dry stalks: straw toward the tip
        // Each blade its own shade: yellower .. bluer, darker .. lighter, and
        // one in fifteen has dried out.
        vec3 br = g_blade_r;
        grassCol *= mix(vec3(1.13, 1.02, 0.80), vec3(0.89, 1.00, 1.12), fract(br.x * 19.3)) *
                    mix(0.82, 1.18, fract(br.y * 31.7));
        if (fract(br.z * 41.3) < 0.066) grassCol = mix(grassCol, straw, 0.25 + 0.5 * h_t);
    }
    // What stands on the tip of the blade.
    bool head = g_blade_kind != 0;
    if (g_blade_kind == 1) {
        grassCol = straw * mix(0.95, 1.30, fract(g_blade_r.x * 7.7));
    } else if (g_blade_kind == 2) {
        float fk = fract(g_blade_r.y * 57.1);
        grassCol = fk < 0.45 ? vec3(0.60, 0.60, 0.56) : (fk < 0.80 ? vec3(0.70, 0.52, 0.06) : vec3(0.36, 0.22, 0.55));
    }
    // Winter: snow lies on the grass, thickest on the tips.
    grassCol = mix(grassCol, vec3(0.86, 0.89, 0.93), scene_snow() * mix(0.45, 0.92, h_t));

    // Shoreline tint — UI sliders (rt_.grass_shore_color/strength/
    // distance). Cheap: 1 sub + 1 smoothstep + 1 mix per visible
    // grass pixel. Strength 0 disables the entire branch via the
    // mix weight collapsing to 0. terrainY is already in scope.
    {
        float strength = scene.grass_shore_color.a;
        if (strength > 1e-3) {
            float fade = max(0.05, scene.grass_shore_params.x);
            float h_above_water = max(0.0, terrainY - scene.water_params.y);
            float shore_w = 1.0 - smoothstep(0.0, fade, h_above_water);
            grassCol = mix(grassCol, scene.grass_shore_color.rgb,
                            shore_w * strength);
        }
    }

    // "Simplest shading" (scene_ubo.glsl FLAT_GRASS): the blade's colour and no more - all of
    // the lighting below is left out.
    vec3  col_out;
    float alpha_out = 1.0;
    if (FLAT_GRASS) {
        col_out = grassCol * flat_light();
    } else {
    // Height-from-ground AO. Slider-controlled floor (0 = darkest base,
    // 1 = no AO darkening); the lerp above already shifts colour with
    // height, so the AO term mostly adds extinction at the very base.
    float ao_floor = clamp(scene.grass_color_bottom.w, 0.0, 1.0);
    float ao = ao_floor + (1.0 - ao_floor) * pow(h_t, 0.6);
    if (head) ao = 1.0;      // on top of the grass: open to the sky

    // Per-blade "lean" variation — hash the cell ID and tilt the
    // effective n_dot_l so adjacent blades on the same flat ground
    // don't all shade identically. ±0.25 around base, then half-
    // Lambert wrap so the shadow side isn't pitch-black.
    float blade_var = hash12(tile_wrap(floor(p.xz / TILE_CELL(0.20)), TILE_N(0.20))) - 0.5;  // [-0.5, 0.5]
    // Blade face normal from the hit blade's rotation (the blade is a
    // flat strip with its face along local z), turned toward the viewer
    // and leaned toward the ground normal for the blade's curl. The face
    // turned to the sun is lit; the far side only gets light transmitted
    // through the leaf - a real dark side instead of every blade shaded
    // like the flat ground beneath it.
    vec3 Nf = vec3(-sin(g_blade_rot), 0.0, cos(g_blade_rot));
    if (dot(Nf, rd) > 0.0) Nf = -Nf;
    vec3 Nb = normalize(Nf + N * 0.45);
    float ndl_b = dot(Nb, L) + blade_var * 0.3;
    // Same scattering model as the tree leaves: wrap diffuse on the lit
    // face, and on the far face the sun transmitted through the blade
    // (yellow-green, stronger toward the thin tip).
    float n_dot_l_wrap = max(0.0, (ndl_b + 0.35) / 1.35);
    float blade_back   = clamp(-ndl_b, 0.0, 1.0);

    sun_lit *= cloud_shadow(p) * sun_horizon_lit(u_sun_horizon, p);
    vec3 sun_term = scene.sun_color.rgb * scene.sun_color.a *
                    (n_dot_l_wrap * sun_lit) +
                    scene.sun_color.rgb * scene.sun_color.a * vec3(1.25, 1.40, 0.45) *
                    (blade_back * 0.42 * sun_lit * mix(0.45, 1.0, h_t));
    // Sky fill. 0.18 left shaded grass about four times darker than the
    // shaded ground beside it (terrain uses 0.75 * hemisphere), so grass in
    // a tree or hill shadow went near black. A little ground bounce too.
    vec3 sky_term = scene.sky_color.rgb * 0.46 + scene.sun_color.rgb * scene.sun_color.a * 0.02;
    // Hand flashlight on the lawn, with a shadow ray.
    {
        vec3 fl_dir; float fl_dist;
        vec3 fl = flashlight_at(p, fl_dir, fl_dist);
        if ((fl.r + fl.g + fl.b) > 1e-4) {
            bool fl_blocked = false;
            if (fl_dist > 0.5) {
                rayQueryEXT flq;
                rayQueryInitializeEXT(flq, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT,
                                      0x01u, p + vec3(0.0, 0.04, 0.0), 0.3, fl_dir, fl_dist - 0.1);
                RQ_PROCEED_LEAVES(flq)
                fl_blocked = rayQueryGetIntersectionTypeEXT(flq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
            }
            if (!fl_blocked) {
                float bv = (t < 14.0) ? mix(1.0, grass_blade_vis(p, terrainY, fl_dir, 0.045), 0.65) : 1.0;
                sky_term += fl * (bv * (0.5 + 0.5 * abs(dot(Nb, fl_dir))));
            }
        }
    }
    sky_term += fx_lights_diffuse(p, vec3(0.0, 1.0, 0.0));   // tracers (no shadows)
    // Laser bolts and gun flashes on the lawn, with a shadow ray each.
    {
        int fxn = int(scene.ripple_info.w + 0.5);
        for (int i = 0; i < fxn; ++i) {
            if (scene.fx_light_col[i].a < 0.5) continue;
            vec3 fto = scene.fx_lights[i].xyz - p;
            float fr = scene.fx_lights[i].w;
            float fd2 = dot(fto, fto);
            if (fd2 >= fr * fr) continue;
            float fd = sqrt(fd2);
            vec3 fdir = fto / max(fd, 1e-3);
            float ffall = 1.0 - fd / fr; ffall *= ffall;
            if (fd > 0.7) {
                rayQueryEXT frq;
                rayQueryInitializeEXT(frq, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT,
                                      0x01u, p + vec3(0.0, 0.04, 0.0), 0.3, fdir, fd - 0.3);
                RQ_PROCEED_LEAVES(frq)
                if (rayQueryGetIntersectionTypeEXT(frq, true) != gl_RayQueryCommittedIntersectionNoneEXT) continue;
            }
            if (t < 14.0) ffall *= mix(1.0, grass_blade_vis(p, terrainY, fdir, 0.045), 0.65);
            sky_term += scene.fx_light_col[i].rgb * (ffall * (0.5 + 0.5 * abs(dot(Nb, fdir))));
        }
    }
    // Road lanterns on the lawn, with a shadow ray each: bushes, trees, posts
    // and walls keep the lantern light off the grass behind them.
    for (int li = 0; li < 8; ++li) {
        float lI = scene.lamp_lights[li].w;
        if (lI < 1e-3) continue;
        vec3 lto = scene.lamp_lights[li].xyz - p;
        float ld_len = length(lto);
        if (ld_len >= 13.0) continue;
        vec3 ldir = lto / max(ld_len, 1e-3);
        float lfall = 1.0 - ld_len / 13.0; lfall *= lfall;
        rayQueryEXT lrq;
        rayQueryInitializeEXT(lrq, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT,
                              0x01u, p + vec3(0.0, 0.04, 0.0), 0.3, ldir, max(ld_len - 0.45, 0.31));
        RQ_PROCEED_LEAVES(lrq)
        if (rayQueryGetIntersectionTypeEXT(lrq, true) != gl_RayQueryCommittedIntersectionNoneEXT) continue;
        // The blades between this one and the lantern.
        if (t < 14.0) lfall *= mix(1.0, grass_blade_vis(p, terrainY, ldir, 0.045), 0.65);
        // Blades stand upright: light from the side catches them fully.
        sky_term += vec3(1.0, 0.62, 0.28) * (lI * lfall * (0.5 + 0.5 * abs(dot(Nb, ldir))));
    }
    // Side-lit blade shading: sun-side brighter, away-side darker.
    // Uses RAW dot(N, L) so the sign carries direction (n_dot_l above
    // was max'd to 0 for Lambert, which would always tint upward).
    float side_tint = 1.0;
    {
        float sl_on  = scene.grass_side_lit_params.y;
        float sl_str = scene.grass_side_lit_params.x;
        if (sl_on > 0.5 && sl_str > 0.001) {
            float raw_ndl = dot(N, L);
            side_tint = clamp(1.0 + sl_str * raw_ndl,
                              1.0 - sl_str * 0.9,
                              1.0 + sl_str * 0.9);
        }
    }
    // The root darkening (ao) is gentler on the sky fill than on the sun.
    vec3 col = grassCol * (sun_term * side_tint * ao + sky_term * mix(1.0, ao, 0.6) * mix(0.55, 1.0, sun_lit));

    // View-dependent backlit / translucency — sun shining THROUGH the
    // blade from behind makes the leaves glow warm. Strongest when:
    //   * looking toward the sun (v_dot_l > 0)
    //   * the surface isn't already directly lit (1 - n_dot_l)
    //   * we're on the upper part of the blade (h_t)
    // Same trick the rasterised grass.frag uses — was missing here so
    // raymarched grass had no view-dependent tone shift, every blade
    // read identically from front vs back.
    float v_dot_l  = max(0.0, dot(rd, L));   // rd points away from camera
    float v_dot_l2 = v_dot_l * v_dot_l;
    float v_dot_l4 = v_dot_l2 * v_dot_l2;
    vec3 backlit = scene.sun_color.rgb * scene.sun_color.a *
                   (v_dot_l4 * (1.0 - n_dot_l) * h_t) *
                   vec3(0.65, 0.95, 0.35) * sun_lit;
    col += backlit * grassCol;

    // Fresnel-edge rim using the (per-blade-jittered) terrain normal.
    // pow(x, 3) → x*x*x (no exp/log).
    float fr_in = clamp(1.0 + dot(N, rd), 0.0, 1.0);
    // (Scaled by the light actually available: this rim and the distance fade
    // colour below were constants, so grass kept glowing on a black night.)
    float light_lvl = clamp(dot(sky_term + sun_term, vec3(0.3333)) / 0.22, 0.0, 1.0);
    vec3 fres = vec3(0.18, 0.18, 0.10) * ao * light_lvl *
                (fr_in * fr_in * fr_in);
    col += fres;

    // Distance fade colour target — eases blade colour INTO the
    // underlying ground tint so the silhouette transition is invisible.
    // Two ground colours blend by camera distance using the same
    // anchors the terrain shader uses (30..200m), keeping the two
    // passes in lock-step.
    // Far anchor comes from rt_.grass_ground_tint_far_distance (UI
    // slider, packed into grass_color_ground_far.w). Near anchor is
    // 0 so the close→far ramp is one continuous gradient over the
    // full 0..far range — no inflection band where the smoothstep
    // slope jumps. Reads as a true merge instead of a visible knee.
    float tint_far  = max(50.0, scene.grass_color_ground_far.w);
    float ground_t  = smoothstep(0.0, tint_far, t);
    vec3 fade_target = mix(scene.grass_color_ground.rgb,
                            scene.grass_color_ground_far.rgb,
                            ground_t);
    float fade = smoothstep(kMaxDist * 0.55, kMaxDist, t);
    col = mix(col, fade_target * light_lvl, fade);

    // Distance-driven opacity. Close-up blades are fully opaque
    // (alpha = 1). Past pc.color.w (the soft-distance setting) the
    // alpha ramps down so far blades become genuinely translucent
    // and the underlying terrain green tint shows through. The
    // user-cutoff slider controls HOW MUCH transparency the far end
    // reaches: cutoff = 0 → far blades stay opaque (alpha = 1);
    // cutoff = 0.6 → far blades drop to alpha ≈ 0.4 (heavy blend
    // with terrain).
    //
    // The earlier "edge alpha from SDF distance" approach was wrong:
    // the raymarch loop converges right AT the blade surface, so the
    // hit_d sample lands near hit_thresh for every hit (interior or
    // silhouette), producing uniformly low coverage and the "always
    // transparent close-up" symptom.
    float user_cut    = scene.grass_extra.y;
    float soft_dist   = max(pc.color.w, 1.0);
    float dist_w      = smoothstep(soft_dist, kMaxDist, t);
    float final_alpha = mix(1.0, 1.0 - user_cut, dist_w);

    // Distance fog — final atmospheric mix. Same exp² as terrain.
    {
        float strength = scene.distance_fog_color.a;
        if (strength > 1e-3) {
            float density   = clamp(scene.distance_fog_params.x, 0.0, 0.1);
            float start_d   = scene.distance_fog_params.y;
            float height_top = scene.distance_fog_params.z;
            float max_alpha  = scene.distance_fog_params.w;
            float d_raw = max(0.0, length(p - scene.camera_pos.xyz) - start_d);
            float dx = d_raw * density;
            float fog = 1.0 - exp(-dx * dx);
            if (height_top > 0.5) {
                fog *= 1.0 - smoothstep(0.0, height_top, p.y);
            }
            float fa = clamp(fog * strength, 0.0, max_alpha);
            col = mix(col, scene.distance_fog_color.rgb, fa);
        }
    }
    col_out = col; alpha_out = final_alpha;
    }
    outColor = vec4(col_out, alpha_out);
    outGBuffer0 = vec4(clamp(col_out, vec3(0.0), vec3(1.0)),
                       5.0 / 255.0);   // mat 5 = pass-through
    // Best-effort normal: use the cumulative marched hit normal if a
    // local var carries one; otherwise zero — the downstream deferred
    // pass-through reads scene_color, not gbuffer1, so this is just for
    // any future normal-aware consumer (SSR, etc).
    outGBuffer1 = vec4(0.5, 0.5, 0.5, 0.0);

    // Honour LESS_OR_EQUAL depth test so a closer wall already in
    // depth_image_ correctly occludes the marched grass.
    vec4 clip = pc.mvp * vec4(p, 1.0);
    gl_FragDepth = clamp(clip.z / max(clip.w, 1e-4), 0.0001, 0.9998);

    // Real screen-space motion vector for TAA reprojection. Grass is
    // world-static (per-blade wind sway is sub-pixel), so prev_uv
    // comes from prev_view_proj applied to the SAME world hit point.
    // Was hardcoded to vec2(0) — that meant TAA sampled the previous
    // frame at the SAME screen position, so when the camera strafed
    // grass blended in stale terrain colour from where it HAD been,
    // appearing to lag the rest of the scene by ~history-blend frames.
    // Using clip-space directly (not gl_FragCoord/viewport.zw) makes
    // it resolution-independent.
    vec4 prev_clip = pc.prev_mvp * vec4(p, 1.0);
    if (clip.w > 0.0001 && prev_clip.w > 0.0001) {
        vec2 cur_uv  = clip.xy      / clip.w      * 0.5 + 0.5;
        vec2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
        vec2 m = cur_uv - prev_uv - scene.restir_params.zw;   // cube.frag convention (unjittered)
        // NaN/inf guard + clamp to a sane range. A blade hit very
        // close to the camera with a glancing prev-frame projection
        // can produce |motion| > 1.0 which poisons TAA history and
        // (under enough motion) lands the next frame's reprojection
        // in undefined memory → device lost. Clamp ±0.5 = ±half-
        // screen, which is wider than any real per-frame motion.
        if (!any(isnan(m)) && !any(isinf(m))) {
            outMotion = clamp(m, vec2(-0.5), vec2(0.5));
        } else {
            outMotion = vec2(0.0);
        }
    } else {
        outMotion = vec2(0.0);
    }
}
