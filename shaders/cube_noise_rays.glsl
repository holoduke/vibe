// cube_noise_rays.glsl - part of cube.frag (included there, in place; not a standalone shader).
// Noise + random, ray-query helpers (any_hit*), sky sampling, voxel tower DDA

// Interleaved Gradient Noise (Jorge Jimenez, "Next Generation Post-Processing
// in Call of Duty: Advanced Warfare", SIGGRAPH 2014).
// Adjacent pixels see *smoothly varying* random values rather than hash-style
// uncorrelated jumps, so the per-pixel integration error has low spatial
// frequency — looks like a smooth gradient rather than a dotted dither.
float ign(vec2 p) {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Isotropic hashed value-noise for TERRAIN spatial break-up. ign()
// above is Interleaved Gradient Noise — a screen-space DITHER whose
// values are organised along one fixed direction; sampled as a spatial
// noise on world XZ it stamps parallel same-direction bands onto the
// ground (independent of height/geometry). tnoise() is a normal
// bilinear value-noise with no directional structure → no banding.
float thash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}
float tnoise(vec2 x) {
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float a = thash21(i);
    float b = thash21(i + vec2(1.0, 0.0));
    float c = thash21(i + vec2(0.0, 1.0));
    float d = thash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);   // [0,1], isotropic
}

// === Hash + value-noise with analytical derivatives ===
// Identical recipe to terrain_raymarch.frag::noised — the terrain
// micro-detail pass below uses these so the rasterised mesh inherits
// the same FBM-eroded look as the per-pixel raymarched terrain
// (otherwise sub-cell detail snaps to the 1 m vertex spacing).
float t_hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
vec3 t_noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = t_hash21(i + vec2(0.0, 0.0));
    float b = t_hash21(i + vec2(1.0, 0.0));
    float c = t_hash21(i + vec2(0.0, 1.0));
    float d = t_hash21(i + vec2(1.0, 1.0));
    float v = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    vec2  dv = du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    return vec3(v, dv);
}
const mat2 t_m2 = mat2(0.8, -0.6, 0.6, 0.8);

// ---- Raymarch-matched terrain micro-detail ----
// The heightmap bakes octaves 0..8 of the raymarcher's eroded FBM
// (terrain_raymarch.frag terrainM: scale 0.003, height 120, m2 rotation,
// erosion 1/(1+|d|^2)) at 1 m. The raymarcher adds octaves 9..19 per
// pixel - that IS its extra ridge / erosion detail. Same accumulator here,
// returning only the part ABOVE octave 8, with octaves faded out once
// their wavelength drops below ~2 pixels (`fp` = metres per pixel).
float terrain_fbm_detail(vec2 xz, float fp) {
    vec2  p = xz * 0.003;
    float a = 0.0, b = 1.0, a_base = 0.0;
    vec2  d = vec2(0.0);
    for (int i = 0; i < 20; ++i) {
        float wl = 333.333 * exp2(-float(i));          // octave wavelength (m)
        float w  = (i < 9) ? 1.0 : smoothstep(fp * 2.0, fp * 4.0, wl);
        if (w <= 0.0) break;
        vec3 n = t_noised(p);
        d += n.yz;
        a += b * n.x * w / (1.0 + dot(d, d));
        if (i == 8) a_base = a;
        b *= 0.5;
        p = t_m2 * p * 2.0;
    }
    return (a - a_base) * 120.0;
}

// FBM with derivative erosion — `n.x / (1 + |∇d|²)` damps high-freq
// octaves on steep slopes, producing the ridge/valley structure
// instead of fractal mush. Returns (value, ∂/∂x, ∂/∂z) where the
// derivatives are with respect to `world_xz` (Jacobian-corrected
// per octave so they stay in world space).
vec3 terrain_detail_fbm(vec2 wp, float scale, int octaves) {
    vec2 p = wp * scale;
    mat2 J = mat2(scale, 0.0, 0.0, scale);
    float a = 0.0, b = 1.0;
    vec2 dacc = vec2(0.0);
    for (int i = 0; i < 9; ++i) {
        if (i >= octaves) break;
        vec3 n = t_noised(p);
        // World-space derivative: J^T · local_deriv
        vec2 wd = J * n.yz;
        dacc += wd;
        float damp = 1.0 / (1.0 + dot(dacc, dacc));
        a += b * n.x * damp;
        b *= 0.5;
        // Per-octave 2× scale + rotation
        p = t_m2 * p * 2.0;
        J = t_m2 * J * 2.0;
    }
    // Approx-centred value (FBM tends toward ~0.5 mean) and
    // accumulated world-space gradient
    return vec3(a - 0.5, dacc.x, dacc.y);
}

// Per-sample IGN: shift the input position by the sample index along a known
// magic vector so each sample's jitter is uncorrelated within a pixel but
// stays smooth across neighbouring pixels.
// PCG3D hash - decorrelated per sample (IGN-of-shifted-pixel printed a
// fixed diagonal hatch into soft shadows; see deferred_lighting.frag).
uvec3 pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}
float rand(uvec3 seed) {
    return float(pcg3d(seed).x) * (1.0 / 4294967296.0);
}

vec3 cos_hemi(float u1, float u2, vec3 n) {
    float r = sqrt(u1);
    float phi = 6.28318530718 * u2;
    vec3 d = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
    vec3 up_a = abs(n.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 t = normalize(cross(up_a, n));
    vec3 b = cross(n, t);
    return d.x * t + d.y * n + d.z * b;
}

// Vogel disk taps (cos, sin of i × golden-angle). 32 entries; PCSS
// loops index `i & 31` so any sample count up to 32 maps cleanly. Per-
// pixel rotation (one sin/cos at the start of the loop) breaks the
// fixed-pattern aliasing while still saving 2N − 2 trig calls on an
// N-tap PCSS shadow.
const vec2 kVogelDisk[32] = vec2[32](
    vec2( 1.000000,  0.000000),
    vec2(-0.737369,  0.675490),
    vec2( 0.087426, -0.996171),
    vec2( 0.608439,  0.793601),
    vec2(-0.984713, -0.174182),
    vec2( 0.843755, -0.536728),
    vec2(-0.259604,  0.965715),
    vec2(-0.460907, -0.887448),
    vec2( 0.939321,  0.343039),
    vec2(-0.924346,  0.381556),
    vec2( 0.423846, -0.905734),
    vec2( 0.299284,  0.954164),
    vec2(-0.865211, -0.501408),
    vec2( 0.976676, -0.214719),
    vec2(-0.575129,  0.818062),
    vec2(-0.128511, -0.991708),
    vec2( 0.764649,  0.644447),
    vec2(-0.999146,  0.041318),
    vec2( 0.708829, -0.705380),
    vec2(-0.046191,  0.998933),
    vec2(-0.640709, -0.767784),
    vec2( 0.991069,  0.133347),
    vec2(-0.820858,  0.571132),
    vec2( 0.219481, -0.975617),
    vec2( 0.497181,  0.867647),
    vec2(-0.952693, -0.303935),
    vec2( 0.907791, -0.419423),
    vec2(-0.386061,  0.922473),
    vec2(-0.338452, -0.940984),
    vec2( 0.885189,  0.465231),
    vec2(-0.966970,  0.254890),
    vec2( 0.540838, -0.841127)
);

// Forward decls — voxel any-hit + raw DDA (defined with the DDA block
// further down). Let the shadow / AO / blocker-search helpers below fold
// the voxel tower into their result without reordering the whole file.
bool voxel_any_hit(vec3 origin_w, vec3 dir_w, float t_max);
bool vox_march(vec3 origin_w, vec3 dir_w, float t_max,
               out float out_t, out uint out_pal);

// Shadow + AO test. Cull-mask 0x01 picks up only instances marked as
// shadow-casters (bit 0). Sparks and projectiles are flagged 0xFE on the
// host — they're tiny visual effects whose hard shadow streaks looked wrong
// and whose AO darkening produced visible halos around hits.
bool any_hit(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT,
                          0x01, origin, 0.001, dir, t_max);
    RQ_PROCEED_LEAVES(rq)
    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT) return true;
    return voxel_any_hit(origin, dir, t_max);
}

// Shadow-ray variant for TERRAIN receivers. Uses cull-mask 0x02 which
// the terrain BLAS instance does NOT have set — so the ray only finds
// castle / dyn-prop occluders, not the terrain itself. The terrain BLAS
// would otherwise produce per-triangle false-hits because its full-
// resolution geometry sits above the rasterised LOD surface. Terrain
// self-shadow comes from the heightmap bake instead.
bool any_hit_no_terrain(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT,
                          0x02, origin, 0.001, dir, t_max);
    RQ_PROCEED_LEAVES(rq)
    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT) return true;
    return voxel_any_hit(origin, dir, t_max);
}

// Closest-hit variant of the terrain-receivers shadow query — used
// for the PCSS blocker search distance estimate.
bool closest_hit_no_terrain(vec3 origin, vec3 dir, float t_max,
                             out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT,
                          0x02, origin, 0.001, dir, t_max);
    RQ_PROCEED_LEAVES(rq)
    bool tlas = rayQueryGetIntersectionTypeEXT(rq, true) ==
                gl_RayQueryCommittedIntersectionTriangleEXT;
    float tt = tlas ? rayQueryGetIntersectionTEXT(rq, true) : t_max;
    float vt; uint vpal;
    if (vox_march(origin, dir, tt, vt, vpal) && (!tlas || vt < tt)) {
        out_t = vt; return true;
    }
    if (tlas) { out_t = tt; return true; }
    return false;
}

// Sky tint without the sun halo — for atmospheric fog where adding
// the localized halo brightness on distant terrain pixels makes them
// glow blindingly when looking near the sun. Same horizon→zenith
// gradient as `sample_sky` but no halo term.
vec3 sample_sky_atmosphere(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, sqrt(up));    // pow(up, 0.45) ≈ sqrt(up); 3% error, no exp/log
}

// Procedural sky: warm low horizon → cool zenith, brighter near sun.
vec3 sample_sky(vec3 dir) {
    vec3 L = normalize(scene.sun_direction.xyz);
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    vec3 sky = mix(horizon, zenith, sqrt(up));   // pow(up, 0.45) approx
    // pow(x, 8) → 3 squarings, no exp/log.
    float h1 = max(dot(dir, L), 0.0);
    float h2 = h1 * h1;
    float h4 = h2 * h2;
    float halo = h4 * h4;
    sky += scene.sun_color.rgb * scene.sun_color.a * 0.08 * halo;
    return sky;
}

// Mask-parameterised variants — let the PCSS loop hoist `is_terrain_pre`
// out into a single uniform mask instead of branching on a function call
// every tap (which forced two specialised loop bodies into the kernel).
bool any_hit_m(vec3 origin, vec3 dir, float t_max, uint mask) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT,
                          mask, origin, 0.001, dir, t_max);
    RQ_PROCEED_LEAVES(rq)
    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT) return true;
    return voxel_any_hit(origin, dir, t_max);
}
bool closest_hit_m(vec3 origin, vec3 dir, float t_max, uint mask,
                   out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT,
                          mask, origin, 0.001, dir, t_max);
    RQ_PROCEED_LEAVES(rq)
    bool tlas = rayQueryGetIntersectionTypeEXT(rq, true) ==
                gl_RayQueryCommittedIntersectionTriangleEXT;
    float tt = tlas ? rayQueryGetIntersectionTEXT(rq, true) : t_max;
    // Fold in the voxel tower so the PCSS blocker search detects it as an
    // occluder — without this, terrain pixels shadowed only by the tower
    // see hits==0 and short-circuit to "fully lit", skipping the penumbra
    // loop where voxel_any_hit lives.
    float vt; uint vpal;
    if (vox_march(origin, dir, tt, vt, vpal) && (!tlas || vt < tt)) {
        out_t = vt; return true;
    }
    if (tlas) { out_t = tt; return true; }
    return false;
}

// Closest-hit ray cast. Returns true on hit, fills out_t.
bool closest_hit(vec3 origin, vec3 dir, float t_max,
                 out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT,
                          0xFF, origin, 0.001, dir, t_max);
    RQ_PROCEED_LEAVES(rq)
    bool tlas = rayQueryGetIntersectionTypeEXT(rq, true) ==
                gl_RayQueryCommittedIntersectionTriangleEXT;
    float tt = tlas ? rayQueryGetIntersectionTEXT(rq, true) : t_max;
    float vt; uint vpal;
    if (vox_march(origin, dir, tt, vt, vpal) && (!tlas || vt < tt)) {
        out_t = vt; return true;
    }
    if (tlas) { out_t = tt; return true; }
    return false;
}

// Closest-hit + material lookup. Returns instance custom index AND
// primitive index. The merged-static BLAS instance carries
// kStaticBlasInstSentinel as its custom index; the shader recovers the
// per-brush material via primitive_id / 12 in that case (12 tris/brush).
// Dynamic instances carry their materials-buffer slot directly.
bool closest_hit_material(vec3 origin, vec3 dir, float t_max,
                          out float out_t, out int out_inst, out int out_prim) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT,
                          0xFF, origin, 0.001, dir, t_max);
    RQ_PROCEED_LEAVES(rq)
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    out_inst = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    out_prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    return true;
}

// ---- Voxel building DDA (Session B) ----
// Analytic intersection of a world-space ray against the single brickmap
// voxel shape, used by the shadow + GI ray helpers below. Works in
// shape-local space (translation-only transform). Mirrors voxel.frag's
// two-level DDA: empty bricks are skipped a brick at a time; occupied
// bricks are walked voxel-by-voxel. Whole feature is gated off when
// scene.voxel_origin.w < 0.5 (no shape) → near-zero cost.
bool vox_ray_aabb(vec3 lo, vec3 ld, vec3 bmin, vec3 bmax,
                  out float t0, out float t1) {
    vec3 invd = 1.0 / ld;
    vec3 a = (bmin - lo) * invd;
    vec3 b = (bmax - lo) * invd;
    vec3 lov = min(a, b);
    vec3 hiv = max(a, b);
    t0 = max(max(lov.x, lov.y), lov.z);
    t1 = min(min(hiv.x, hiv.y), hiv.z);
    return t1 > max(t0, 0.0);
}

// Core DDA. Returns the palette index of the first voxel hit and its t
// along `ld` (world units). want_closest is implicit — the march always
// stops at the first solid voxel, which IS the closest hit for a convex
// march from the entry point. Returns -1 (in out_pal) on miss.
bool vox_march(vec3 origin_w, vec3 dir_w, float t_max,
               out float out_t, out uint out_pal) {
    out_t = t_max; out_pal = 0u;
    if (scene.voxel_origin.w < 0.5) return false;       // disabled

    vec3 lo = origin_w - scene.voxel_origin.xyz;        // shape-local origin
    vec3 ld = dir_w;
    float vs = scene.voxel_dims.w;
    float bs = scene.voxel_grid.w;
    ivec3 brick_dim = ivec3(scene.voxel_grid.xyz + 0.5);
    ivec3 voxel_dim = brick_dim * 16;
    vec3  shape_max = scene.voxel_dims.xyz;

    float t0, t1;
    if (!vox_ray_aabb(lo, ld, vec3(0.0), shape_max, t0, t1)) return false;
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

    // Tight cap (was the worst-case diagonal, ~700 steps) so RT rays — fired
    // up to N×PCSS taps per pixel by cube.frag's shadow loop — can't pay the
    // worst-case cost when the player is close to the voxel tower. 96 covers
    // a ray that threads the grid via empty-brick fast-skips (each brick step
    // jumps 16 voxel cells at once), and bounds the in-solid case so a thick
    // chunk doesn't drag every shadow tap through hundreds of cells. Visually
    // invisible: shadows are soft and a few missed grazing voxels get absorbed
    // by the PCSS penumbra. Halved from 192 in the perf pass.
    const int max_steps = 96;
    for (int i = 0; i < max_steps; ++i) {
        if (t > t_max) return false;
        if (any(lessThan(vc, ivec3(0))) ||
            any(greaterThanEqual(vc, voxel_dim))) return false;

        ivec3 bc = vc >> 4;
        int   dir_i = (bc.z * brick_dim.y + bc.y) * brick_dim.x + bc.x;
        uint  bp = vox_entries[dir_i];

        if (bp == kVoxEmpty) {
            // Skip the rest of this brick in one step.
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

// Any-hit shadow test against the voxel shape.
bool voxel_any_hit(vec3 origin_w, vec3 dir_w, float t_max) {
    float t; uint pal;
    return vox_march(origin_w, dir_w, t_max, t, pal);
}

// Palette → linear RGB. Mirrors the C++ palette in voxel.cpp's
// update_voxel_camera_ubo (indices 1 = sandstone, 2 = wood accent).
vec3 voxel_albedo(uint pal) {
    if (pal == 1u) return vec3(0.95, 0.78, 0.45);
    if (pal == 2u) return vec3(0.75, 0.30, 0.15);
    return vec3(0.50);
}

// Sentinel for the merged-static BLAS instance — must match
// `kStaticBlasInstSentinel` in src/engine/vk_engine/rt.cpp. Picked at the
// top of the 24-bit instanceCustomIndex range so it never collides with a
// real materials-buffer slot.
const int kStaticBlasSentinel = 0xFFFFFF;
const int kCubeTrisPerBox     = 12;
#define BR_FRONT(x) (x)

