#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require
#extension GL_ARB_conservative_depth : require

// Conservative depth KEPT despite shell-mapping silhouette.
//
// cube.vert now extrudes SPOM wall faces outward by kShellThickness
// (4 cm) along their per-face normals; the raster face is therefore
// at the OUTER edge of the brick shell band. The SSDM block below
// either:
//   * Discards (gl_FragDepth = 1.0, the far plane -- ALWAYS > raster)
//     when the parallax-march brick top doesn't reach this fragment
//     -- this is the silhouette path, sky shows past the brush.
//   * Writes the parallax-hit surface depth, which lies AT or INSIDE
//     the raster face (the march walks inward from the raster peak),
//     so the depth is ALWAYS >= gl_FragCoord.z.
// Both branches honour `depth_greater`, so the driver can keep early-Z
// on cube.frag enabled -- the single biggest GPU win found by the last
// perf review (cube.frag was running on every fragment that the depth
// pre-pass should have culled). The user opted in to shell-mapping
// silhouette but we don't have to pay 0.5-1 ms of early-Z loss for it.
layout(depth_greater) out float gl_FragDepth;

layout(location = 0) in vec3 vNormal;
// flat — see cube.vert for the rationale (these are push-constant values
// that must not perspective-interpolate; the vTexParams.x precision drift
// was silently enabling texture sampling when textures were "off").
layout(location = 1) flat in vec3 vColor;
layout(location = 2) in vec3 vWorldPos;
layout(location = 3) flat in vec4 vEmissive;
layout(location = 4) in vec2 vUv;
layout(location = 5) flat in vec4 vTexParams;  // x: albedo idx, y: normal idx,
                                               // z: uv scale,
                                               // w: 0 = world-space triplanar
                                               //    1 = object-space triplanar
layout(location = 6) in vec3 vObjectPos;
layout(location = 7) in vec3 vObjectNormal;
layout(location = 8) in vec4 vPrevClip;  // prev_view_proj × prev_model × local_pos
// Pre-extrusion shell base. For SPOM wall brushes cube.vert pushes the
// rasterised vertex out by kShellThickness along the per-face normal;
// vShellBase carries the original (un-extruded) world position. The
// SSDM block uses dot(vWorldPos - vShellBase, geometric_normal) to read
// the shell thickness at this fragment (constant under uniform
// extrusion -- gives us a "how much shell band exists here" scalar so
// the parallax-march hit can be classified brick vs sky). For non-
// extruded fragments (floors, terrain, non-SPOM brushes, dyn props)
// vShellBase == vWorldPos, the dot product is 0, and the shell test is
// a no-op.
layout(location = 9) in vec3 vShellBase;
layout(location = 10) flat in vec4 vVegInfo;
layout(location = 0) out vec4 outColor;
// Per-pixel screen-space motion vector — current_uv minus prev_uv. Dynamic
// surfaces get correct reprojection because cube.vert applied prev_model
// (DynRender::prev_world × scale for dyn boxes; current model for static
// brushes / camera-attached viewmodel / sub-pixel particles). Future SVGF
// pass reads motion_vec_image_ + uses this delta to walk into history.
layout(location = 1) out vec2 outMotion;
// G-buffer outputs (Phase 2 of the deferred-render migration).
//   GBuffer0 RGBA8        : albedo.rgb + material_id.a (id / 255)
//   GBuffer1 R10G10B10A2  : octahedral_normal.rg + roughness.b + metallic.a
// Cube.frag writes both alongside the existing forward output. Phase 3
// will introduce a separate lighting pass that reads these; Phase 4
// strips the forward lighting block here.
//
// Material IDs: 0 = stone/brick (default), 1 = wood, 2 = metal,
//               3 = terrain, 4 = grass, 5 = emissive/sky-debug.
// Coarse — refined per-material in Phase 5 when terrain gets its own
// slot in the lighting branch.
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

// Octahedral normal encoding (Meyer et al. 2010). Stable, branchless,
// fits a unit normal into vec2 of [0,1] with ~10-bit precision per
// component — enough for diffuse, no banding under our materials.
vec2 octa_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = (n.z >= 0.0)
        ? n.xy
        : ((1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                     n.y >= 0.0 ? 1.0 : -1.0));
    return e * 0.5 + 0.5;
}

#include "scene_ubo.glsl"
#include "veg_leaf.glsl"
#include "rock_sdf.glsl"

// Half-rate shadow producer output, sampled when terrain_local_info.z
// is on. linear_sampler_ → bilinear interpolation across the 4 nearest
// half-res texels is the "bilateral" part of this minimum-viable upsample.
layout(set = 0, binding = 18) uniform sampler2D u_shadow_lr;

// SVGF GI denoiser raw-irradiance write target (Session 1 of
// docs/svgf_plan.md). cube.frag stores the post-temporal/spatial
// reservoir reuse irradiance HERE before it gets multiplied by
// albedo, so the future denoiser passes (sessions 2-4) operate on the
// clean lighting signal. Format MUST match the C++ image
// (VK_FORMAT_R16G16B16A16_SFLOAT). No reader in session 1.
layout(set = 0, binding = 19, rgba16f) uniform image2D u_svgf_gi;

// SVGF temporal accumulator history (Session 2). Ping-pong pair: on
// frame F, cube.frag reads from binding (20 + ((F+1) & 1)) and writes
// to binding (20 + (F & 1)). Race-free under kFrameOverlap == 2 — F's
// write target was last touched by F-2, which is retired by the F-1
// fence the CPU waits on before recording F. .rgb = accumulated
// irradiance, .a = M-count [1..kSvgfMmax] for the EMA weight.
layout(set = 0, binding = 20, rgba16f) uniform image2D u_svgf_hist0;
layout(set = 0, binding = 21, rgba16f) uniform image2D u_svgf_hist1;

// Distance-based sample LOD. fragments within lod_near get full samples;
// fragments past lod_far drop to a single ray. Standard "RT becomes a luxury
// at distance" trick used in commercial engines.
int lod_samples(int requested, float dist) {
    float lod_t = clamp((dist - scene.rt_lod.x) /
                        max(0.001, scene.rt_lod.y - scene.rt_lod.x), 0.0, 1.0);
    int n = int(round(mix(float(requested), 1.0, lod_t)));
    return max(1, n);
}

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct Material {
    vec4 color;
    vec4 emissive;
    vec4 tex;       // x: albedo idx, y: normal idx, z: uv scale, w: spare
};
layout(set = 0, binding = 2, std430) readonly buffer Materials {
    Material materials[];
};

// ---- Voxel building (Session B) ----
// Brick atlas + shape directory shared with voxel.frag. cube.frag's
// inline-RT shadow + GI rays DDA-march these to occlude / colour-bleed
// the procedural voxel tower. Layout MUST match voxel_world.h /
// voxel.frag exactly.
struct VoxBrick { uint occ[128]; uint pal[1024]; };
layout(set = 0, binding = 24, std430) readonly buffer VoxBrickAtlas {
    VoxBrick vox_bricks[];
};
// Flat global directory: all shapes' entries concatenated. cube.frag only
// tests the MAIN voxel shape (the static tower) for inline-RT shadows + GI,
// and its directory lives at offset 0, so this shader can read
// `vox_entries[dir_i]` directly without a per-shape base offset.
layout(set = 0, binding = 25, std430) readonly buffer VoxShapeDir {
    uint  vox_entries[];
};
const uint kVoxEmpty = 0xFFFFFFFFu;

// Bound texture arrays (must match kTextureCount on the C++ side).
const int kTextureCount = 25;     // +1 for kTexProcGrass (slot 12)
// Procedurally-baked terrain material slots (see bake_terrain_materials
// in world.cpp). Must match kTexRock/.. in vk_engine.h.
const int kMatRock  = 7;
const int kMatGrass = 8;
const int kMatDirt  = 9;
const int kMatSand  = 10;
const int kMatSnow  = 11;
const int kMatRoof  = 13;   // ambientCG RoofingTiles012A (castle roofs)
const int kMatRoad  = 14;   // ambientCG PavingStones131 (castle road)
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kTextureCount];
layout(set = 0, binding = 4) uniform sampler2D u_normal[kTextureCount];
#define LEAF_ATLAS u_albedo[15]
#include "leaf_rt.glsl"
// Pre-baked heightmap sun-shadow texture (R8, 0 = lit, 255 = shadowed).
// Sampled here as a distance fallback for terrain self-shadow — the
// BLAS holds the full heightmap detail, so RT shadow rays from a low-
// LOD rasterised surface false-hit BLAS peaks the raster missed. The
// bake was traced against the heightmap directly so it matches the
// rasterised LOD surface at distance with no false hits.
layout(set = 0, binding = 6) uniform sampler2D u_terrain_shadow;
// Sun shadow map (single-cascade ortho, rendered each frame from the
// sun's POV with castle / dyn-props / terrain as casters). 1024² over
// ~120m world half = ~12cm per texel close to the camera. We sample
// this for terrain receivers within the shadow map's frustum — much
// sharper than the bake's 1m texels.
layout(set = 0, binding = 7) uniform sampler2DShadow u_sun_shadow_map;

// Brick (slot 1) parallax-occlusion displacement. Sampled by the SPOM
// path below for albedo_idx == 1 (castle walls). Other materials don't
// have a height map and skip the parallax march entirely.
// SPOM (parallax-occlusion) displacement map array. cube.frag picks the
// right entry per fragment via height_idx_for_albedo() below; non-SPOM
// materials skip the parallax march entirely. Order MUST match the
// spom_height_textures_ array on the host (vk_engine.h). Slot [4] is the
// stylized-grass1 height for the kTexGrass / grass-band terrain
// material -- sampled in cube.frag's is_terrain block for SPOM-style
// pixel displacement AND in terrain_tess.tese for vertex displacement.
layout(set = 0, binding = 12) uniform sampler2D u_height[6];

// Grass eligibility mask (RG8). Same binding as in grass_raymarch.frag /
// terrain_raymarch.frag / water.frag -- the descriptor set layout
// already publishes this slot. R = paint mask (0 = no grass, 1 = full),
// G = slope clamp. cube.frag samples it in the is_terrain block to
// approximate grass-cast shadow on the ground by walking a few short
// steps along the sun's XZ direction and accumulating mask coverage.
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;
// Absolute terrain heightmap (R32F, same as terrain.vert binding 17).
layout(set = 0, binding = 17) uniform sampler2D u_heightmap_full;

// Terrain aux bake (AAA Phases 4/5): R = curvature (0.5 flat, <0.5
// concave gully, >0.5 convex ridge), G = sky visibility (horizon AO).
// 1024² over the 2048 m world, baked at load from the heightmap.
layout(set = 0, binding = 26) uniform sampler2D u_terrain_aux;

// ReSTIR GI reservoir buffers (session 2 — see docs/restir_plan.md).
// Layout MUST match the C++ Reservoir struct in restir.cpp (std430,
// 48 B / pixel). Indexed as [pix.y * render_w + pix.x]. Session 2
// only ships the write — temporal read of u_reservoir_prev lands in
// session 3.
struct Reservoir {
    vec3  sample_dir;   // surface normal N of the writing pixel
    vec3  radiance;     // temporally-accumulated GI radiance
    float W;
    float w_sum;
    uint  M;            // sample count (0 = invalid / never written)
    uint  pad;          // session 5: floatBitsToUint(cam_dist) of the
                        //   writing surface — used for depth-aware
                        //   disocclusion (reject reprojected/neighbour
                        //   samples whose surface is at a different
                        //   distance, i.e. a different object).
};
// Bindings 15/16 alias the SAME physical buffer, which holds THREE
// per-pixel reservoir regions (a ring). cube.frag selects the region
// by frame parity (see kResRingBase below): write region frame%3,
// read region (frame+2)%3. This makes prev/cur a genuine ping-pong
// with NO descriptor swap and is race-free under kFrameOverlap==2
// (see the restir.cpp file header for the proof). Both declarations
// omit readonly/writeonly so the compiler can't assume non-aliasing.
layout(set = 0, binding = 15, std430) buffer ReservoirPrev {
    Reservoir r[];
} u_reservoir_prev;
layout(set = 0, binding = 16, std430) buffer ReservoirCur {
    Reservoir r[];
} u_reservoir_cur;

// Map an albedo texture index to a SPOM material slot (or -1 for "no
// parallax"). Keep in sync with vk_engine.h's spom_height_textures_
// allocation order.
int height_idx_for_albedo(int a) {
    if (a == 1) return 0;   // Bricks078         — castle outer walls
    if (a == 4) return 1;   // PaintedBricks001  — keep walls
    if (a == 8) return 4;   // stylized-grass1 (kTexGrass / grass band).
                            // Used by the in-terrain SPOM pixel-displacement
                            // path inside the is_terrain block, NOT by the
                            // outer SPOM brush path (terrain bypasses that
                            // path via the is_terrain gate -- terrain uses
                            // its own splat-aware march, see below).
    // Floor tiles (courtyard): their slabs are 0.5 m thin, so parallax on
    // the SIDE faces showed depth past the slab - the brush path only
    // allows these on up-facing faces (see spom_path).
    if (a == 5) return 2;   // Tiles130
    if (a == 6) return 3;   // Tiles074
    return -1;
}

// Push constants — match cube.vert's PC layout exactly. Fragment shader
// only reads `model` (for the brush's world center + per-axis size used
// by the SPOM silhouette test); the rest are unused here but the layout
// must mirror the vertex shader because both stages share the range.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    // Already inside the shared push range (other shaders use it);
    // cube.frag just hadn't declared it. For terrain colour draws:
    //   .x = sand ripple scale, .y = grass line scale.
    vec4 grass_params;
} pc;

#include "cube_surface_helpers.glsl"   // Texturing helpers: triplanar sampling, terrain / mesh parallax (tpom_*, pom_*), grass blade shadow, army paint, roughness table
#include "cube_noise_rays.glsl"   // Noise + random, ray-query helpers (any_hit*), sky sampling, voxel tower DDA
#include "cube_wall_relief.glsl"   // Castle wall box relief (br_*): solidity margin, exit, open-air probe, silhouette clip
void main() {
    // Default depth = rasterized depth; SPOM silhouette path may bump
    // to 1.0 below. The depth_greater qualifier on the gl_FragDepth
    // declaration keeps early-Z enabled regardless.
    gl_FragDepth = gl_FragCoord.z;
    // Screen-space motion vector — current_uv в€’ prev_uv. prev_uv comes from
    // perspective-divided prev_clip (smooth-interpolated by the rasterizer).
    // Behind-camera prev pixels (prev_clip.w ≤ 0) have no valid prev_uv —
    // fall back to zero motion; the future SVGF pass treats sentinel as
    // "no history available" and rebuilds variance from current-frame
    // neighborhood instead of reprojecting.
    {
        vec2 current_uv = gl_FragCoord.xy * scene.viewport.zw;
        if (vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            vec2 prev_uv = prev_ndc * vec2(0.5, 0.5) + vec2(0.5);
            // Subtract jitter delta packed into restir_params.zw —
            // produces unjittered geometric motion that FSR3 expects.
            outMotion = (current_uv - prev_uv) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
    }

    // ---- Vegetation: LOD cross-fade dither + procedural impostors ----
    // vVegInfo.x = fade f, .y = 1 for the finer LOD of a fading pair,
    // .z = impostor kind (1 pine, 2 broadleaf). Screen-door dither with a
    // stable per-pixel noise: the coarse LOD keeps pixels n >= f, the fine
    // one n < f, so together they cover every pixel exactly once.
    if (vVegInfo.x > 0.0) {
        float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy + 5.588238 * float(uint(scene.rt_flags.w) & 7u),
                                                  vec2(0.06711056, 0.00583715))));
        if (vVegInfo.y > 0.5 ? (n >= vVegInfo.x) : (n < vVegInfo.x)) discard;
    }
    // ---- Rock (GPU vegetation, material 4) ------------------------------
    // The mesh IS the rock (world_props.cpp: chiselled boulders at up to 20k
    // triangles, flat-shaded slab beds). What is added here is only the
    // relief no mesh LOD carries - cracks, pitting, grain, the grikes of a
    // limestone pavement - as a small height field over the surface: its
    // gradient bends the mesh normal, its cavity term darkens the cracks.
    // No ray march, no discard, no depth write: silhouette, depth, ray-
    // traced shadow and collider are this one mesh from every angle.
    if (vVegInfo.z < -0.5 && pc.grass_params.w < -0.5 && pc.grass_params.x > 0.5) {
        float r_variant = -vVegInfo.z - 1.0;
        float rs = vObjectNormal.x, rysq = vObjectNormal.y, rrot = vObjectNormal.z;
        float rc = cos(rrot), rsn = sin(rrot);
        vec3  rscl = vec3(rs, rs * rysq, rs);
        float seed = 3.7 + 6.1 * r_variant;
        // Sculpted in Blender and baked (tools/rocks): normal, occlusion and
        // curvature come from its texture instead of the procedural detail.
        bool  baked = r_variant > 99.5;
        if (baked) r_variant -= 100.0;
        bool  slab = !baked && abs(mod(r_variant, 3.0) - 1.0) < 0.5;
        // Detail follows the pixel footprint (object units per pixel).
        float cam_d = distance(vWorldPos, scene.camera_pos.xyz);
        float fp = cam_d * 2.0 * scene.viewport.w / max(rs, 0.05);
        float fine = 1.0 - smoothstep(0.006, 0.03, fp);
        vec3 po = vObjectPos;
        vec3 wposr = vWorldPos;
        // Mesh normal, world -> object (inverse-transpose of rotate * scale).
        vec3 Ng = normalize(vNormal);
        vec3 ngo = normalize(vec3(rc * Ng.x + rsn * Ng.z, Ng.y, -rsn * Ng.x + rc * Ng.z) * rscl);
        float cav;
        float grike = slab ? smoothstep(0.75, 0.95, ngo.y) * (1.0 - smoothstep(0.02, 0.10, fp)) : 0.0;
        vec3 no;
        float bk_ao = 1.0, bk_curv = 0.5;
        if (baked) {
            vec2 buv = vec2(vUv.x, (vUv.y - 4.0) / 0.45);
            vec4 bk = texture(u_normal[19], buv);
            vec2 of = bk.rg * 2.0 - 1.0;
            vec3 on = vec3(of.x, of.y, 1.0 - abs(of.x) - abs(of.y));
            float ot = max(-on.z, 0.0);
            on.x += on.x >= 0.0 ? -ot : ot;
            on.y += on.y >= 0.0 ? -ot : ot;
            no = normalize(on);
            bk_ao = bk.b; bk_curv = bk.a;
            // Grain finer than the bake resolves, close up.
            float gc;
            vec3 fine_n = rock_detail_normal(po, no, seed, fine, 0.0, max(fp * 0.7, 0.0025), gc);
            no = normalize(mix(no, fine_n, 0.35 * fine));
            cav = clamp((1.0 - bk_ao) * 1.5 + max(0.5 - bk_curv, 0.0) * 1.2, 0.0, 1.0);
        } else {
            no = rock_detail_normal(po, ngo, seed, fine, grike, max(fp * 0.7, 0.0025), cav);
        }
        vec3 nl = no / rscl;
        vec3 Nr = normalize(vec3(rc * nl.x - rsn * nl.z, nl.y, rsn * nl.x + rc * nl.z));
        // Albedo: the rock set, triplanar by the MESH normal (weights from the
        // detail normal flickered per pixel into streaky static), around a
        // natural grey with large patches of warmer / cooler stone, darker
        // cracks, pale worn edges and moss on sheltered tops.
        vec3 tex = triplanar_sample(u_albedo[7], wposr * 0.22, Ng);
        float tl = dot(tex, vec3(0.333));
        vec3 alb = mix(vec3(tl), tex, 0.85) * vec3(1.10, 1.06, 1.0);
        alb = mix(alb, vec3(0.30, 0.29, 0.27), 0.25);                          // tame the texture contrast
        float big = rk_fbm(po * 0.9 + seed, 2);
        alb *= mix(0.82, 1.10, big) * mix(vec3(1.04, 1.0, 0.94), vec3(0.95, 0.98, 1.03), rk_vnoise(po * 0.55 - seed));
        alb *= 1.0 - 0.38 * clamp(cav, 0.0, 1.0);
        if (baked) {
            // Worn edges are paler (the weathered skin is chipped off there),
            // crevices hold dirt, the shaded underside stays damp and dark.
            float edge = smoothstep(0.56, 0.80, bk_curv);
            alb = mix(alb, alb * 1.32 + vec3(0.025), edge * 0.6);
            float dirt = smoothstep(0.75, 0.35, bk_ao) * smoothstep(0.35, 0.65, rk_vnoise(po * 3.3 + seed));
            alb = mix(alb, vec3(0.085, 0.072, 0.052), dirt * 0.55);
            alb *= mix(0.80, 1.0, smoothstep(-0.5, 0.3, no.y));
            // Lichen: pale grey-green and ochre blotches on the lit faces.
            float lic = smoothstep(0.60, 0.78, rk_fbm(po * 5.2 + seed * 2.0, 3)) * smoothstep(0.1, 0.6, no.y) * (1.0 - edge);
            alb = mix(alb, mix(vec3(0.36, 0.38, 0.30), vec3(0.42, 0.34, 0.16), step(0.5, rk_vnoise(po * 1.7 - seed))), lic * 0.45);
        }
        float moss = smoothstep(0.55, 0.9, Nr.y) * smoothstep(0.45, 0.75, rk_fbm(po * 2.6 - seed, 3)) * 0.55;
        alb = mix(alb, vec3(0.10, 0.17, 0.05), moss);
        float snr = smoothstep(scene.terrain_h_high.x + 10.0, scene.terrain_h_high.z, wposr.y) * smoothstep(0.35, 0.75, Nr.y);
        alb = mix(alb, vec3(0.88, 0.9, 0.95), clamp(snr, 0.0, 1.0));
        // Where the rock meets the ground: soil creeps up the foot (colour
        // and normal blend toward the ground) and the contact is darkened,
        // so the rock sits IN the terrain instead of ending at a hard line.
        {
            float hgt = textureLod(u_heightmap_full, wposr.xz / 2048.0 + 0.5, 0.0).r;
            float above = (wposr.y - hgt) / max(rs * rysq * 0.9, 0.2);   // relative to the rock height (flat pavements!)
            float gb = 1.0 - smoothstep(0.02, 0.14 + 0.08 * rk_vnoise(wposr * 3.1), above);
            vec3 soil = texture(u_albedo[9], wposr.xz * 0.45).rgb * vec3(0.95, 0.85, 0.68);
            // Flat tops stay rock even when they lie flush with the ground (a
            // limestone pavement); the soil climbs the sides and the foot only.
            float flat_top = smoothstep(0.6, 0.9, no.y);
            gb *= 1.0 - 0.85 * flat_top;
            alb = mix(alb, soil, gb * 0.7);
            alb *= mix(1.0, 0.72, (1.0 - smoothstep(0.0, 0.06, above)) * (1.0 - flat_top));
            Nr = normalize(mix(Nr, vec3(0.0, 1.0, 0.0), gb * 0.6));
        }
        g_veg_solid = true;
        outGBuffer0 = vec4(clamp(alb, vec3(0.0), vec3(1.0)), 1.0 / 255.0);
        outGBuffer1 = vec4(octa_encode(Nr), mix(0.9, 0.7, moss), gbuf_flags(true, 0.0));
        // Deferred path: the lighting pass shades it. Forward fallback: sun + sky.
        vec3 Lr = normalize(scene.sun_direction.xyz);
        outColor = vec4(alb * (scene.sun_color.rgb * scene.sun_color.a * max(dot(Nr, Lr), 0.0) +
                               scene.sky_color.rgb * 0.35 * (0.5 + 0.5 * Nr.y)), 1.0);
        return;
    }
    bool veg_imp = vVegInfo.z > 0.5;
    if (veg_imp) {
        // Billboard silhouette from uv (u -1..1 across, v 0..1 up).
        float u = vUv.x, v = vUv.y;
        float jag = (fract(sin(dot(floor(vec2(u * 9.0, v * 14.0)), vec2(12.9, 78.2))) * 43758.5) - 0.5);
        bool inside;
        if (vVegInfo.z < 1.5) {          // pine: tiered cone + trunk
            float tier = fract(v * 6.0);
            float w = (1.0 - v) * (0.75 + 0.25 * tier) + jag * 0.06;
            inside = (v > 0.12 && abs(u) < w) || (v <= 0.14 && abs(u) < 0.06);
        } else {                          // broadleaf: lumpy crown + trunk
            vec2 q = vec2(u, (v - 0.62) * 1.5);
            float rr = 0.78 + 0.12 * sin(atan(q.y, q.x) * 5.0 + vVegInfo.w) + jag * 0.08;
            inside = length(q) < rr || (v < 0.4 && abs(u) < 0.07);
        }
        if (!inside) discard;
    } else if (pc.grass_params.w < -0.5 && pc.grass_params.x > 0.5) {
        // GPU vegetation canopy / bush: leafy cut-out silhouette (veg_leaf.glsl).
        int vm = int(vUv.y + 0.5);
        if ((vm == 2 || vm == 3) && veg_leaf_cut(vObjectPos, vWorldPos, vNormal, scene.camera_pos.xyz)) discard;
        if (vm == 7) {
            g_card_tex = texture(u_albedo[15], veg_card_uv(vObjectNormal.xy, vObjectNormal.z, 0.5 * max(fwidth(vObjectNormal.x), fwidth(vObjectNormal.y))), -0.7);
            if (g_card_tex.a < 0.33) discard;      // (the same cut as the depth pre-pass, veg_depth.frag: a pixel that one keeps and the other drops has depth but no surface - a bright speck)
        }
    }
    vec3 N = normalize(vNormal);
    if (veg_imp) {
        // Fake a rounded canopy: bend the camera-facing normal outward by
        // the billboard uv so impostors shade like the 3D crowns.
        vec3 Vc = normalize(scene.camera_pos.xyz - vWorldPos);
        vec3 Rc = cross(vec3(0.0, 1.0, 0.0), Vc);   // zero when viewed from straight above
        vec3 Rt = dot(Rc, Rc) > 1e-8 ? normalize(Rc) : vec3(1.0, 0.0, 0.0);
        N = normalize(Vc * 0.6 + Rt * vUv.x * 0.7 + vec3(0.0, 1.0, 0.0) * (vUv.y - 0.35) * 0.9);
    }

    if (vEmissive.a > 0.5) {
        vec3 e = vColor + vEmissive.rgb;
        // Self-lit textured mesh (stained glass): the texture is the light.
        if (pc.grass_params.w < -0.5 && vTexParams.x > -0.5 && vTexParams.x < float(kTextureCount) - 0.5)
            e = texture(u_albedo[int(vTexParams.x + 0.5)], vUv * vTexParams.z).rgb * vColor + vEmissive.rgb;
        outColor = vec4(e, 1.0);
        // Also fill gbuffer with the lit emissive value tagged mat 5 so the
        // deferred shader's mat≥5 pass-through reads scene_color (which
        // has the HDR emissive) instead of treating this as a hole and
        // mis-shading the brush behind. Without this, emissive brushes
        // (lamp strips, neon trim) render as scattered single-pixel black
        // dots in the deferred path because alpha=0 in the cleared gbuffer
        // round-trips to material_id=0 with non-zero noise, slipping past
        // the albedo guard.
        outGBuffer0 = vec4(clamp(e, vec3(0.0), vec3(1.0)), 5.0 / 255.0);
        outGBuffer1 = vec4(octa_encode(N), 0.5, 0.0);
        return;
    }

    // Hoist the two scalar derivatives the shader uses repeatedly:
    //   * cam_dist appears 13× in branches; with the conditional code
    //     in between, the compiler often fails to CSE it.
    //   * L (sun direction unit vector) appears 4×.
    // Computing once at the top costs nothing extra on the fast paths
    // (early-out emissive returns above this line) and saves ~12 sqrt
    // and 3 inversesqrt per pixel on the heavy lit-surface path.
    float cam_dist = distance(vWorldPos, scene.camera_pos.xyz);
    vec3  L        = normalize(scene.sun_direction.xyz);
    // shading_pos = the effective surface position lighting sees.
    // For most fragments it equals vWorldPos. The SPOM path below
    // overrides it with the parallax-displaced point so RT shadow,
    // AO, and muzzle-flash rays trace from the actual brick crevice
    // instead of the flat geometric cube face — gives real
    // self-shadowing inside SPOM bricks at almost no cost.
    vec3 shading_pos = vWorldPos;

#ifdef QLIKE_TERRAIN_ONLY
    const bool is_terrain_pre = true;    // cube_gbuf_terrain variant (mesh terrain pipeline)
#else
    bool is_terrain_pre = vTexParams.w > 1.5;
#endif
    // Static castle mesh (round towers, roofs, arch): pc.grass_params.w = -1.
    // Vertex-UV texturing; skips every axis-aligned-box-only path.
    bool is_mesh_draw = !is_terrain_pre && pc.grass_params.w < -0.5;
    // Untextured world-prop mesh = foliage (canopies, needles, bushes):
    // flagged in gbuffer1.a so deferred lighting offsets its shadow / AO /
    // GI ray origins (the RT copy of a tree is static + lower detail while
    // the raster one sways - rays self-hit the canopy -> crawling patches).
    bool is_foliage_px = is_mesh_draw && pc.grass_params.z > 0.5 &&
                         (vTexParams.x < 0.0 || pc.grass_params.x > 0.5);   // .x = GPU veg draw
    {   // GPU vegetation bark (0) and rocks (4) are not leaves.
        int vm_f = int(vUv.y + 0.5);
        g_veg_solid = is_foliage_px && pc.grass_params.x > 0.5 && (vm_f == 0 || vm_f == 4) && vVegInfo.z < 0.5;
    }

    // ---- Terrain debug visualisation (set in pause-menu Graphics) ----
    // Bypasses everything except a basic Lambert / normal preview so we
    // can isolate which downstream effect (RT shadow, AO, GI, slope
    // blend, cavity, fog, triplanar) is producing visible artifacts.
    if (is_terrain_pre) {
        int dbg = int(scene.grass_extra2.w + 0.5);
        if (dbg == 1) {
            // Pure Lambert with constant grey albedo. No RT, no AO,
            // no slope blend, no fog. If the artifact persists here
            // it's a geometry/normal problem; if it disappears the
            // bug is in one of the disabled effects.
            vec3 L = normalize(scene.sun_direction.xyz);
            float ndl = max(dot(N, L), 0.0);
            vec3 lit = vec3(0.55) * (0.25 + 0.75 * ndl);
            outColor = vec4(lit, 1.0);
            return;
        } else if (dbg == 2) {
            // Per-vertex normal as RGB. Adjacent triangles with
            // mismatched corner normals will show as visible colour
            // jumps — direct view into the LOD/Gouraud issue.
            outColor = vec4(N * 0.5 + 0.5, 1.0);
            return;
        } else if (dbg == 3) {
            // Geometric face normal (cross of world-pos derivatives,
            // sign-guarded for Vulkan y-down). Each large LOD triangle
            // is a uniform colour — confirms what flat-shaded geometry
            // looks like at distance.
            vec3 fn = cross(dFdx(vWorldPos), dFdy(vWorldPos));
            if (fn.y < 0.0) fn = -fn;
            outColor = vec4(normalize(fn) * 0.5 + 0.5, 1.0);
            return;
        } else if (dbg >= 4) {
            // Incremental debug modes: each higher number adds one
            // feature on top of "Lambert + hybrid RT/bake shadow",
            // so the user can pinpoint which feature reintroduces
            // distant-terrain artifacts.
            //   4  = base Lambert + hybrid shadow.
            //   5  = + height-band layer albedo (sand/grass/dirt/rock/snow).
            //   6  = + slope-rock blend (faded by distance).
            //   7  = + cavity AO from screen-space N derivatives (faded).
            //   8  = + triplanar texture detail (faded at distance).
            //   9  = + RTAO (faded at distance).
            //   10 = + GI bounce (faded at distance).
            vec3 L = normalize(scene.sun_direction.xyz);
            float ndl = max(dot(N, L), 0.0);
            float dist_to_cam = cam_dist;   // P5: use hoisted distance

            // PURE BAKE with PCF — 5-tap kernel and no threshold so
            // shadow edges blend smoothly across triangles instead of
            // snapping at the 0.5 step boundary (which looked like
            // adjacent triangles flipping fully-lit / fully-shadowed).
            ivec2 sz = textureSize(u_terrain_shadow, 0);
            const float side_b = 2048.0; // terrain world extent (supersample-invariant)
            vec2 uv_b = (vWorldPos.xz / side_b) + vec2(0.5);
            float sh = 0.0;
            if (all(greaterThanEqual(uv_b, vec2(0.0))) &&
                all(lessThanEqual(uv_b, vec2(1.0)))) {
                vec2 texel = 1.0 / vec2(sz);
                float s = texture(u_terrain_shadow, uv_b).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2( 1.0,  0.0)).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2(-1.0,  0.0)).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2( 0.0,  1.0)).r;
                s += texture(u_terrain_shadow, uv_b + texel * vec2( 0.0, -1.0)).r;
                sh = s * 0.2;
            }

            // Albedo: grey at base, layer-blend from mode 5.
            vec3 base_col = vec3(0.55);
            if (dbg >= 5) {
                const vec3 sand   = vec3(0.78, 0.71, 0.50);
                const vec3 grass  = vec3(0.31, 0.45, 0.18);
                const vec3 dirt   = vec3(0.42, 0.30, 0.18);
                const vec3 rock   = vec3(0.40, 0.38, 0.36);
                const vec3 snow   = vec3(0.95, 0.95, 0.97);
                float h = vWorldPos.y;
                float t_sand = smoothstep(scene.terrain_h_low.x,  scene.terrain_h_low.y,  h);
                float t_dirt = smoothstep(scene.terrain_h_low.z,  scene.terrain_h_low.w,  h);
                float t_rock = smoothstep(scene.terrain_h_high.x, scene.terrain_h_high.y, h);
                float t_snow = smoothstep(scene.terrain_h_high.z, scene.terrain_h_high.w, h);
                base_col = mix(sand, grass, t_sand);
                base_col = mix(base_col, dirt, t_dirt);
                base_col = mix(base_col, rock, t_rock);
                base_col = mix(base_col, snow, t_snow);
            }
            // Slope-rock blend at mode 6+.
            if (dbg >= 6) {
                float slope = 1.0 - clamp(N.y, 0.0, 1.0);
                float steep = smoothstep(0.45, 0.75, slope);
                float steep_w = 1.0 - smoothstep(80.0, 200.0, dist_to_cam);
                base_col = mix(base_col, vec3(0.40, 0.38, 0.36), steep * steep_w);
            }
            // Cavity AO at mode 7+.
            if (dbg >= 7) {
                float curvature = -dot(dFdx(N), dFdx(vWorldPos)) -
                                  dot(dFdy(N), dFdy(vWorldPos));
                float cav_w  = 1.0 - smoothstep(80.0, 200.0, dist_to_cam);
                float cavity = clamp(0.5 - curvature * 0.4, 0.45, 1.0);
                base_col *= mix(1.0, cavity, cav_w);
            }
            // Triplanar texture detail at mode 8+.
            if (dbg >= 8) {
                vec3 detail = triplanar_sample(u_albedo[0], vWorldPos * 0.0625, N);
                float det_w = 1.0 - smoothstep(80.0, 200.0, dist_to_cam);
                base_col = mix(base_col, base_col * detail, det_w);
            }
            // AO at mode 9+ (single short ray, faded at distance).
            float ao_dbg = 1.0;
            if (dbg >= 9 && scene.rt_flags2.w > 0) {
                vec3 origin_ao = vWorldPos + N * 0.01;
                vec3 ao_dir = normalize(N + vec3(0.3, 0.0, 0.0));
                if (any_hit(origin_ao, ao_dir, 1.0)) ao_dbg = 0.7;
                float ao_far_t = smoothstep(80.0, 200.0, dist_to_cam);
                ao_dbg = mix(ao_dbg, 1.0, ao_far_t);
            }
            // GI at mode 10+ (single bounce, faded).
            vec3 gi_dbg = vec3(0.0);
            if (dbg >= 10) {
                vec3 origin_gi = vWorldPos + N * 0.01;
                vec3 gi_dir = normalize(N + vec3(0.0, 0.0, 0.5));
                if (!any_hit(origin_gi, gi_dir, 50.0)) {
                    gi_dbg = scene.sky_color.rgb * 0.2;
                }
                float gi_far_t = smoothstep(80.0, 200.0, dist_to_cam);
                gi_dbg = mix(gi_dbg, vec3(0.0), gi_far_t);
            }

            vec3 lit = base_col * (0.25 + 0.75 * ndl * (1.0 - sh)) * ao_dbg + gi_dbg;
            outColor = vec4(lit, 1.0);
            return;
        }
    }

    // (L = sun direction unit vector hoisted at top of main — used for
    // n_dot_l, shadow rays, GI bounces and slope material blends.)

    // --- Albedo + bump mapping (triplanar projection) ---
    // Choose object-space (texture glued to the model) for dynamic objects,
    // world-space (texture aligned to the world grid) for static brushes.
    int   albedo_idx_raw = int(vTexParams.x);
    int   normal_idx_raw = int(vTexParams.y);
    float scale          = max(0.001, vTexParams.z);
    // tex_params.w convention:
    //   0 = world-space triplanar (default — castle brushes)
    //   1 = object-space triplanar (dynamic crates so the texture
    //       rotates with the body)
    //   2 = terrain shading: sand→grass→dirt→rock→snow blended by
    //       world height + slope, with optional triplanar Ground054
    //       detail multiplied on top.
#ifdef QLIKE_TERRAIN_ONLY
    const bool is_terrain = true;
#else
    bool  is_terrain     = vTexParams.w > 1.5;
#endif
    bool  obj_space      = !is_terrain && vTexParams.w > 0.5;
    // Terrain uses world-space triplanar like the static brushes — the
    // mesh model is identity so vObjectPos == vWorldPos already, but we
    // pick world-space explicitly so the height read uses true world Y.
    vec3  sample_pos     = (obj_space ? vObjectPos : vWorldPos) * scale;
    vec3  proj_n         = obj_space ? normalize(vObjectNormal) : N;
    float mesh_pom_shade = 1.0;   // cavity + self-shadow, applied to albedo below
    // Surface curvature 1/R (|dN| / |dP|) for relief silhouette clipping on
    // curved meshes (round towers). Taken here, in uniform control flow.
    const float mesh_curv = length(fwidth(N)) / max(length(fwidth(vWorldPos)), 1e-5);
    // ---- Parallax occlusion for triplanar MESH surfaces ----
    // The brush SPOM path below needs brush face data and excludes mesh
    // draws. Steep-parallax march in the dominant triplanar plane; the
    // shifted sample_pos then drives albedo AND normal sampling.
    if (is_mesh_draw && pc.grass_params.z > 0.5 && albedo_idx_raw >= 0 &&
        scene.spom_params.x > 0.01) {
        int   hix   = height_idx_for_albedo(albedo_idx_raw);
        float fadeP = 1.0 - smoothstep(26.0, 50.0, distance(vWorldPos, scene.camera_pos.xyz));
        if (hix >= 0 && fadeP > 0.06) {
            vec3 an = abs(N);
            vec3 Tp, Bp, Fp;
            if (an.x >= an.y && an.x >= an.z)  { Tp = vec3(0, 0, 1); Bp = vec3(0, 1, 0); Fp = vec3(sign(N.x), 0, 0); }
            else if (an.y >= an.z)              { Tp = vec3(1, 0, 0); Bp = vec3(0, 0, 1); Fp = vec3(0, sign(N.y), 0); }
            else                                { Tp = vec3(1, 0, 0); Bp = vec3(0, 1, 0); Fp = vec3(0, 0, sign(N.z)); }
            vec3 Vw = normalize(scene.camera_pos.xyz - vWorldPos);
            vec3 Vt = vec3(dot(Vw, Tp), dot(Vw, Bp), dot(Vw, Fp));
            if (Vt.z > 0.12) {
                // uv = (dot(p,T), dot(p,B)) == triplanar_sample's wp.zy / wp.xz / wp.xy
                vec2  uv0 = vec2(dot(sample_pos, Tp), dot(sample_pos, Bp));
                vec2  gdx = dFdx(uv0), gdy = dFdy(uv0);
                // Relief depth in METRES like the brush box relief (stones ~4 cm
                // of relief per 1/scale m of tile), converted to uv units.
                float hs  = clamp(0.05 / max(scale, 0.01), 0.06, 0.3) * scale *
                            scene.spom_params.x * fadeP;
                vec2  uv  = pom_march(hix, uv0, Vt, hs, gdx, gdy);
                {   // silhouette clip (see pom_silhouette_clip)
                    float dm = (1.0 - textureGrad(u_height[hix], uv, gdx, gdy).r) * hs / max(scale, 1e-4);
                    if (pom_silhouette_clip(dm, dot(Vw, N), mesh_curv)) discard;
                }
                sample_pos += Tp * (uv.x - uv0.x) + Bp * (uv.y - uv0.y);
                vec3 Lt = vec3(dot(L, Tp), dot(L, Bp), dot(L, Fp));
                mesh_pom_shade = mix(1.0, pom_shade(hix, uv, Lt, hs, gdx, gdy, 0.0), fadeP);
            }
        }
    }
    bool  use_albedo     = albedo_idx_raw >= 0 && albedo_idx_raw < kTextureCount;
    bool  use_normal     = normal_idx_raw >= 0 && normal_idx_raw < kTextureCount &&
                           !obj_space && !is_terrain;
    // Texture sampling lives in branches gated on use_albedo / use_normal.
    // Sampler-array indexing on Vulkan requires a known-valid index; clamp
    // before indexing to keep the access well-defined regardless of any
    // hoisting the optimiser might do.
    vec3 albedo = vColor;
    if (int(vTexParams.y - 0.5) == -7) albedo = army_paint(vObjectPos, normalize(vObjectNormal), albedo);
    // Baked sky visibility for terrain pixels (set in the is_terrain
    // block, multiplied into the ambient terms downstream — Phase 5).
    float terrain_sky_vis = 1.0;
    // Splat-height + specular weight carried to the lighting section:
    // micro-shadowing (crevices lose grazing sun first) and a glancing
    // specular on rock/snow.
    float terrain_micro_h = 0.6;
    float terrain_pom_vis = 1.0;   // relief self-shadow (sun visibility)
    float terrain_grass_occ = 1.0; // soil under grass / flowers is dark (plants looked as if they floated on bright ground)
    float terrain_spec_w  = 0.0;
    if (is_terrain) {
#include "cube_main_terrain.glsl"   // main(): mesh-terrain surface (splat layers, road, terrain parallax + self-shadow)
    } else {
#include "cube_main_surfaces.glsl"   // main(): non-terrain surfaces (wall box relief march, castle meshes / towers, props)
    }

    // Tree bark: real bark reflects little light (albedo about 0.15 - 0.25);
    // the texture alone left trunks pale, glowing in the shade under their
    // own crown next to the dark forest floor. Deeper in the furrows.
    if (g_veg_solid && int(vUv.y + 0.5) == 0) {
        float lum = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
        albedo *= mix(0.42, 0.66, smoothstep(0.08, 0.40, lum));
    }
    // Where a trunk or a boulder meets the ground: no clean cut-out line.
    // The foot is stained with soil thrown up by rain, moss creeps up from
    // the ground (on boulders over the tops near the ground), and the last
    // hand-width sits in its own contact shade. Height = above the terrain
    // under this very pixel, so it follows slopes.
    if (g_veg_solid) {
        float gh = textureLod(u_heightmap_full, vWorldPos.xz / 2048.0 + 0.5, 0.0).r;
        float above = vWorldPos.y - gh;
        if (above < 1.8) {
            bool  rock = int(vUv.y + 0.5) == 4;
            float n1 = tnoise(vWorldPos.xz * 3.1 + vWorldPos.y * 2.3);
            float n2 = tnoise(vWorldPos.xz * 0.7 + 9.0);
            float n3 = tnoise(vWorldPos.xz * 1.9 + vWorldPos.y * 1.3 + 4.0);
            float soil_w = 1.0 - smoothstep(0.02, rock ? 0.30 : 0.40, above - 0.14 * n1 + 0.07);
            vec3  soil = mix(vec3(0.085, 0.070, 0.050), vec3(0.060, 0.082, 0.036), smoothstep(0.35, 0.70, n2));
            albedo = mix(albedo, soil, soil_w * 0.72);
            float moss_w = (1.0 - smoothstep(0.10, rock ? 0.95 : 1.50, above)) * smoothstep(0.42, 0.74, n3) *
                           (rock ? smoothstep(0.15, 0.80, N.y) : 0.65);
            albedo = mix(albedo, vec3(0.070, 0.105, 0.038) * mix(0.8, 1.25, n1), moss_w * 0.62);
            albedo *= mix(0.55, 1.0, smoothstep(0.0, 0.20, above));
            // Ease the shading normal toward the ground's at the very foot.
            N = normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.35 * (1.0 - smoothstep(0.0, 0.25, above))));
        }
    }
    // GPU-veg rocks: snow caps on up-facing surfaces in the rock/snow band.
    if (is_mesh_draw && pc.grass_params.x > 0.5 && use_albedo) {
        float sn = smoothstep(scene.terrain_h_high.x + 10.0, scene.terrain_h_high.z, vWorldPos.y) *
                   smoothstep(0.35, 0.75, N.y);
        sn *= 0.75 + 0.25 * tnoise(vWorldPos.xz * 1.7);
        albedo = mix(albedo, vec3(0.88, 0.9, 0.95), clamp(sn, 0.0, 1.0));
    }
    // Untextured world props (foliage): procedural leaf-cluster variation
    // - large-scale hue drift between plants + fine clumpy light/dark
    // breakup, and a darker interior (approximated by the normal facing
    // down / sideways) so canopies don't read as flat-coloured blobs.
    if (is_mesh_draw && pc.grass_params.z > 0.5 && !use_albedo) {
        float big  = tnoise(vWorldPos.xz * 0.07 + vec2(3.1, 7.7));
        float fine = tnoise(vWorldPos.xz * 2.6 + vWorldPos.y * 1.9) * 0.6 +
                     tnoise(vWorldPos.xz * 7.3 - vWorldPos.y * 5.1) * 0.4;
        albedo *= mix(vec3(0.85, 1.05, 0.80), vec3(1.15, 0.95, 1.10), big);
        albedo *= 0.65 + 0.7 * fine;
        albedo *= mix(0.55, 1.0, clamp(N.y * 0.5 + 0.6, 0.0, 1.0));
        // Seasons: this tree's leaves (veg.vert marks them in vEmissive.a).
        bool season_px = pc.grass_params.x > 0.5 && vEmissive.a > 0.01 && vEmissive.a < 0.5;
        if (season_px) albedo = season_leaf(albedo, (vEmissive.a - 0.05) / 0.40);
        // Leaf-scale facets (GPU vegetation, near): every ~8 cm cell is one
        // "leaf" with its own tilt and tone, so a canopy catches the sun
        // as a mass of leaves rather than as one smooth surface.
        // Depth inside the crown. A real canopy is a mass of leaf clumps
        // with dark gaps between and under them; lit evenly it reads as one
        // green blob. Three scales of 3-D structure in the tree's own space
        // (so it does not swim as the tree sways): boughs (~1.6 m), leaf
        // clumps (~0.5 m), sprays (~0.17 m) - the gaps go dark and cooler,
        // undersides and the side away from the sun darker still.
        float crown = 1.0;
        vec3  crown_tint = vec3(1.0);
        if (pc.grass_params.x > 0.5 && !g_veg_solid) {
            float cd = distance(vWorldPos, scene.camera_pos.xyz);
            float n1 = br_noise(vObjectPos * 0.62 + 11.0);
            float n2 = br_noise(vObjectPos * 2.0 + 3.0);
            float n3 = cd < 90.0 ? br_noise(vObjectPos * 5.9) : 0.5;
            float gaps = smoothstep(0.30, 0.62, n1 * 0.50 + n2 * 0.35 + n3 * 0.15);
            float under = clamp(N.y * 0.5 + 0.55, 0.0, 1.0);
            float sunny = clamp(dot(N, normalize(scene.sun_direction.xyz)) * 0.5 + 0.6, 0.0, 1.0);
            crown = mix(0.36, 1.16, gaps) * mix(0.68, 1.0, under) * mix(0.82, 1.0, sunny);
            // Shaded foliage is bluer and less saturated; sunlit tips yellower.
            crown_tint = mix(vec3(0.72, 0.88, 1.05), vec3(1.06, 1.02, 0.86), gaps * sunny);
        }
        if (g_card_tex.a >= 0.0) {
            // Leaf card: photographed leaves, tinted per leaf / per tree (vColor).
            // (Keeps the large-scale hue drift between trees; the old code
            // dropped every variation here, so all cards of a crown had one
            // flat colour.)
            albedo = g_card_tex.rgb * vColor * mix(vec3(0.85, 1.05, 0.80), vec3(1.15, 0.95, 1.10), big);
            if (season_px) albedo = season_leaf(albedo, (vEmissive.a - 0.05) / 0.40);
            // The darker parts of the leaf photo are the gaps between leaves:
            // push them down so single leaves stand out.
            float cl = dot(g_card_tex.rgb, vec3(0.3, 0.6, 0.1));
            albedo *= mix(0.55, 1.15, smoothstep(0.10, 0.45, cl));
            // Where the card is thinly covered (needle tips, the fringe of a spray) less of it is
            // leaf: darker there, so the twig's own structure shows inside its outline.
            albedo *= mix(0.58, 1.10, smoothstep(0.33, 0.95, g_card_tex.a));
            albedo *= crown * crown_tint;
        } else if (pc.grass_params.x > 0.5) {
            albedo *= crown * crown_tint;
            float lk = 1.0 - smoothstep(70.0, 150.0, distance(vWorldPos, scene.camera_pos.xyz));
            if (lk > 0.01) {
                vec3 lh;
                float ld = veg_leaf_cell(vObjectPos, lh);
                N = normalize(N + (lh - 0.5) * (0.9 * lk));
                albedo *= mix(1.0, 0.70 + 0.60 * lh.x, lk);
                // Leaf rim slightly darker / yellower than its midrib side.
                albedo *= mix(vec3(1.0), mix(vec3(1.0), vec3(0.80, 0.84, 0.62), smoothstep(0.35, 1.0, ld)), lk);
            }
        }
    }

    // Corner softening for axis-aligned brushes. Cube meshes have one
    // flat normal per face, so 90° outside corners (e.g. tower-to-
    // wall, gate-block-to-wall) read as a hard vertical seam — Lambert
    // n·L jumps abruptly across the edge. Bend the shading normal
    // toward the AVERAGE of (this face, the perpendicular face whose
    // edge we're close to) inside the last `kCornerWidth` of the
    // brush extent. Costs ~10 ALU per fragment; gives a smooth
    // ~25 m m-scale wrap around outside corners.
    // (Not for walls with relief: their corners are really worn round where
    // they are open, and this would also bend the light at the hidden edges
    // where two blocks join flush - a line up the wall in low sun.)
    if (!is_terrain && !obj_space && !is_mesh_draw && !g_br_relief) {
        vec3 face_size_w = vec3(length(pc.model[0].xyz),
                                length(pc.model[1].xyz),
                                length(pc.model[2].xyz));
        vec3 brush_center = vec3(pc.model[3].x, pc.model[3].y, pc.model[3].z);
        vec3 rel = vWorldPos - brush_center;
        // Normalised position along each axis: -1 (one face) .. +1 (other).
        vec3 norm_pos = rel / max(face_size_w * 0.5, vec3(1e-3));
        vec3 absN = abs(N);
        int dom = (absN.x >= absN.y)
                    ? (absN.x >= absN.z ? 0 : 2)
                    : (absN.y >= absN.z ? 1 : 2);

        const float kCornerWidth = 0.18;
        vec3 corner_target = N;
        float best_t = 0.0;
        // Walk the two non-dominant axes; each is a candidate "edge"
        // we might be approaching. Strongest contributor wins.
        for (int a = 0; a < 3; ++a) {
            if (a == dom) continue;
            float p      = norm_pos[a];
            float edge_t = clamp((abs(p) - (1.0 - kCornerWidth)) / kCornerWidth,
                                  0.0, 1.0);
            if (edge_t > best_t) {
                best_t = edge_t;
                vec3 nbr = vec3(0.0);
                nbr[a] = sign(p);
                corner_target = normalize(N + nbr);
            }
        }
        // 0.5 is "full lerp at the edge". Higher = rounder, lower =
        // sharper. 0.5 reads as a believable subtle bevel without
        // making corners look like a sphere.
        N = normalize(mix(N, corner_target, best_t * 0.5));
    }

    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    // Split ambient into a sky-derived component (fades when the surface
    // is enclosed — sky_factor below) and a constant ground/reflected
    // component that exists indoors too (a real room has a floor that
    // reflects light, not a black void). Combining them later avoids
    // the "AO * sky_factor" double-darkening that crushed interior
    // edges to black.
    vec3 ambient_ground = scene.ambient.rgb * scene.rt_params.z;
    // Warm-bias the ambient sky tint by mixing 12 % sun colour. The
    // raw sky tint here is the saturated Rayleigh blue used for the
    // sky dome (0.55, 0.72, 0.95), which is correct for the sky
    // itself but reads as steel-blue on pure-shadow surfaces (keep
    // walls, deep brush interiors) that get no sun direct. Real
    // shadow surfaces pick up bounce light off warm nearby geometry
    // and tonally drift toward neutral; the sky dome stays untouched
    // since this only affects the ambient term computed downstream.
    vec3 sky_tint       = mix(scene.sky_color.rgb, scene.sun_color.rgb, 0.12);
    vec3 ambient_sky    = sky_tint * 0.45 * scene.rt_params.z;
    // Phase 5: baked horizon AO — valley floors see less sky dome, so
    // their ambient drops. 0.45 floor keeps deep gorges readable.
    if (is_terrain) {
        float sv = mix(0.08, 1.0, terrain_sky_vis * terrain_sky_vis * (3.0 - 2.0 * terrain_sky_vis));   // matches deferred (was 0.45 floor: gullies far too bright)
        ambient_sky *= sv;
    }

    // Direct lighting uses pure Lambert (max(0, N·L)). The "wrap" lift
    // for back-of-mountain pixels is applied LATER as a sky-tinted
    // ambient bounce — applying it to direct here would produce the
    // "back-face fully lit while front-face shadow-rayed" artifact:
    // back faces have n_dot_l_raw = 0 so no shadow ray fires, but the
    // wrap pushes their effective n_dot_l above zero and the missing
    // shadow term means full sun colour leaks through. Front-facing
    // pixels resolved correctly via the shadow ray. Adjacent pixels
    // looking radically different is what the user reported.
    float n_dot_l_raw = max(dot(N, L), 0.0);
    float n_dot_l = n_dot_l_raw;
    // Terrain shading contrast: pow on n_dot_l. exp=1 → standard
    // Lambert; exp>1 sharpens the lit→shadow transition by darkening
    // grazing slopes faster. Brushes/dyn-props keep linear Lambert.
    if (is_terrain_pre) {
        float contrast = max(0.25, scene.terrain_extra.x);
        // Fast paths for the common slider values:
        //   contrast == 1 → no-op (default; most users)
        //   contrast == 2 → just n_dot_l²
        // Anything else falls through to the (slower) pow().
        // Common slider stops get cheap dedicated paths (avoids the
        // exp+log of pow). Anything not covered falls through to pow.
        if (contrast > 1.99 && contrast < 2.01) {
            n_dot_l = n_dot_l * n_dot_l;
        } else if (contrast < 1.01) {
            // identity — skip pow entirely
        } else if (contrast > 2.99 && contrast < 3.01) {
            // P11: x³ via two muls
            n_dot_l = n_dot_l * n_dot_l * n_dot_l;
        } else if (contrast > 1.49 && contrast < 1.51) {
            // P11: x^1.5 = x·sqrt(x), one mul + one sqrt vs full pow
            n_dot_l = n_dot_l * sqrt(n_dot_l);
        } else if (contrast > 3.99 && contrast < 4.01) {
            // P11: x⁴ via two squarings
            float n2 = n_dot_l * n_dot_l;
            n_dot_l = n2 * n2;
        } else {
            n_dot_l = pow(n_dot_l, contrast);
        }
    }

    // Per-pixel seed WITH the frame counter — TAA accumulates over ~8
    // frames, so animating the noise lets temporal averaging resolve the
    // residual dither into a clean signal. We tried a frame-stable seed
    // earlier (less moment-to-moment shimmer) but it left a fixed dither
    // pattern visible on flat surfaces; with TAA on (default) the temporal
    // average wins. We mod the frame counter by a small power-of-two so
    // the cycle is short and TAA always reaches steady state quickly.
    uvec3 seed_base = uvec3(uint(gl_FragCoord.x),
                            uint(gl_FragCoord.y),
                            uint(scene.rt_flags.w) & 7u);
    // FRAME-STABLE seed for shadows + AO. Those signals have NO
    // temporal accumulator (unlike GI, which EMA-accumulates via the
    // history images and NEEDS the cycling seed above to converge), so
    // per-frame seed variation made penumbras and contact darkening
    // visibly flicker. A static per-pixel pattern turns that into a
    // stationary dither the higher tap counts hide.
    uvec3 seed_stable = uvec3(uint(gl_FragCoord.x),
                              uint(gl_FragCoord.y), 0u);

    // (cam_dist already hoisted at the top of main — single source of
    // truth for distance-from-camera across LOD, terrain blending, AO
    // distance fade, GI sampling, etc.)

    // ---- Deferred fast path ----
    // When the deferred lighting pass owns material 1 (opaque world /
    // viewmodel) and 3 (mesh terrain) pixels it re-shades them from the
    // G-buffer and DISCARDS this shader's lit colour. Everything below
    // (PCSS shadow, path-traced GI, AO, muzzle light, fog) was ~200+ ray
    // queries per pixel of pure waste. Write the G-buffer and leave.
    // Emissive (mat 5) still needs the forward colour, so it falls through.
    // QLIKE_GBUFFER_ONLY (cube_gbuf.frag.spv) takes this path
    // unconditionally so the compiler strips every ray query, the GI
    // loops and the SSBO/image side-effect stores below.
#ifdef QLIKE_GBUFFER_ONLY
    if (true) {
#else
    if (scene.terrain_disp_params.w > 0.5 && vEmissive.a <= 0.5) {
#endif
        // Rain on the courtyard paving: the stone darkens as it gets wet and
        // pools of water gather in patches - flat, dark mirrors.
        float rain_pool = 0.0, rain_wet = 0.0;
        if (!is_terrain_pre && !is_mesh_draw && albedo_idx_raw == 5 && N.y > 0.85 && scene_wet() > 0.02) {
            rain_wet = scene_wet();
            float pn = br_noise(vec3(vWorldPos.xz * 0.42, 3.0)) * 0.65 + br_noise(vec3(vWorldPos.xz * 1.5, 9.0)) * 0.35;
            float thr = 0.66 - 0.30 * rain_wet;
            rain_pool = smoothstep(thr, thr + 0.05, pn) * smoothstep(0.2, 0.6, rain_wet);
            albedo *= mix(1.0, 0.72, rain_wet) * mix(1.0, 0.50, rain_pool);
            N = normalize(mix(N, vec3(0.0, 1.0, 0.0), rain_pool));
        }
        int fast_mat = is_terrain_pre ? 3 : 1;
        outColor    = vec4(albedo, 1.0);
        outGBuffer0 = vec4(clamp(albedo * terrain_grass_occ, vec3(0.0), vec3(1.0)),
                           float(fast_mat) / 255.0);
        vec2 rmf = surface_rough_metal(albedo_idx_raw, is_foliage_px, is_mesh_draw);
        rmf.x = mix(mix(rmf.x, rmf.x * 0.55, rain_wet), 0.04, rain_pool);
        outGBuffer1 = vec4(octa_encode(N), is_terrain_pre ? terrain_pom_vis : rmf.x, gbuf_flags(is_foliage_px, rmf.y));
        return;
    }

#include "cube_main_forward_lighting.glsl"   // main(): forward lighting path (RT sun shadow, AO, GI, final colour) - skipped when deferred lighting owns the frame
    int mat_id = is_terrain_pre ? 3 : 1;
    if (vEmissive.a > 0.5) mat_id = 5;     // emissive — handled above too
    outGBuffer0 = vec4(clamp(albedo * terrain_grass_occ, vec3(0.0), vec3(1.0)),
                       float(mat_id) / 255.0);
    vec2 rmf = surface_rough_metal(albedo_idx_raw, is_foliage_px, is_mesh_draw);
    outGBuffer1 = vec4(octa_encode(N), is_terrain_pre ? terrain_pom_vis : rmf.x, gbuf_flags(is_foliage_px, rmf.y));
}

