// cube_surface_helpers.glsl - part of cube.frag (included there, in place; not a standalone shader).
// Texturing helpers: triplanar sampling, terrain / mesh parallax (tpom_*, pom_*), grass blade shadow, army paint, roughness table

// Triplanar projection sample. Avoids the "what UV does this cube face
// use" problem and works correctly on every brush size — the texture stays
// world-aligned regardless of brush dimensions. Cost is 3 samples per pass,
// which is fine for the coverage we have.
vec3 triplanar_sample(sampler2D tex, vec3 wp, vec3 N) {
    // pow(abs(N), 4) via two squarings — saves one exp/log/mul per
    // triplanar call vs the pow() intrinsic, on every textured pixel.
    vec3 blend = abs(N); blend *= blend; blend *= blend;
    blend /= max(blend.x + blend.y + blend.z, 1e-3);
    // P15: dominant-axis early-out. When one weight ≥ 0.95, the other
    // two contribute ≤ 5 % blended → invisible. Sampling those two
    // textures wastes ~2 cache fetches on the very common cube/floor
    // pixels (axis-aligned brushes, flat terrain). Fall back to the
    // 3-tap blend when no single axis dominates.
    if (blend.x >= 0.95) return texture(tex, wp.zy).rgb;
    if (blend.y >= 0.95) return texture(tex, wp.xz).rgb;
    if (blend.z >= 0.95) return texture(tex, wp.xy).rgb;
    // Now that mipmaps exist (texture.cpp generates them via blit chain) and
    // vTexParams is `flat` so the sampler-array index is dynamically uniform,
    // standard `texture()` is correct — derivatives select the right mip and
    // far-distance moiré disappears.
    vec3 cx = texture(tex, wp.zy).rgb;
    vec3 cy = texture(tex, wp.xz).rgb;
    vec3 cz = texture(tex, wp.xy).rgb;
    return cx * blend.x + cy * blend.y + cz * blend.z;
}

// Triplanar tangent-space normal. For each projection axis, sample the
// tangent-space normal and transform into world space using a known basis
// for that axis. Then blend by face weight, same as the albedo. Cheap,
// works without explicit per-vertex tangents.
// Dominant-axis parallax-occlusion mapping for axis-aligned faces.
// `wp` is world position scaled to texture-UV space (the same scaled
// position triplanar_sample would consume), `N` is the surface normal.
// Returns the offset 2D UV inside the dominant axis projection plus
// (in `out_axis`) which axis won so the caller can reconstruct the
// tangent-space normal correctly.
//
// The march is 16 linear steps + 1 binary refine — enough to look like
// real depth on castle bricks at typical view angles without showing
// stair-stepping. Grazing-angle pixels (V_t.z < 0.2) skip the march
// because the parallax displacement explodes there and produces
// shimmer + texture swimming.
// `out_overhang_disc` is set true when the parallax march pushes the
// visible point past the brush face's geometric edge — caller should
// `discard` to make the silhouette read as bumpy bricks instead of a
// flat clipped edge (the "S" in SPOM).
vec2 spom_uv(vec3 wp_scaled, vec3 N, vec3 face_size_world, float uv_scale,
             int height_idx,
             out int out_axis,
             out vec3 out_T, out vec3 out_B, out vec3 out_face_n,
             out bool out_overhang_disc,
             out vec3 out_displaced_pos,
             out vec3 out_recessed_pos,
             out vec2 out_uv0) {
    out_overhang_disc = false;
    out_displaced_pos = vWorldPos;
    // Recessed surface point used by the SSDM depth-override path: the
    // lateral parallax offset PLUS the inward (-face_n) push down to the
    // height the ray actually hit. Kept separate from out_displaced_pos
    // (which intentionally omits the inward push to avoid RT self-
    // occlusion — see the note where it's assigned below). Initialised to
    // the rasterised point so every early-return leaves depth unchanged.
    out_recessed_pos = vWorldPos;
    vec3 absN = abs(N);
    int axis;
    vec2 uv0;
    vec3 T, Bb, face_n;
    if (absN.x >= absN.y && absN.x >= absN.z) {
        axis = 0;
        face_n = vec3(sign(N.x), 0.0, 0.0);
        // sign(N.x) on T flips the texture U handedness so the +X and -X
        // faces aren't mirrored versions of each other.
        T  = vec3(0.0, 0.0, sign(N.x));
        Bb = vec3(0.0, 1.0, 0.0);
        uv0 = vec2(wp_scaled.z * sign(N.x), wp_scaled.y);
    } else if (absN.y >= absN.z) {
        axis = 1;
        face_n = vec3(0.0, sign(N.y), 0.0);
        T  = vec3(1.0, 0.0, 0.0);
        Bb = vec3(0.0, 0.0, sign(N.y));
        uv0 = vec2(wp_scaled.x, wp_scaled.z * sign(N.y));
    } else {
        axis = 2;
        face_n = vec3(0.0, 0.0, sign(N.z));
        T  = vec3(sign(N.z), 0.0, 0.0);
        Bb = vec3(0.0, 1.0, 0.0);
        uv0 = vec2(wp_scaled.x * sign(N.z), wp_scaled.y);
    }
    // 2:1-aspect source textures (Bricks078 → height slot 0, Tiles130 →
    // slot 2) tile twice as often in U as in V when driven by square
    // world-space UVs, so the stones render squashed 2:1. Halve U for
    // those slots so texels are world-square. ssdm_disp.frag and
    // shell.frag apply the same factor — keep in sync.
    vec2 tex_aspect = (height_idx == 0 || height_idx == 2)
                      ? vec2(0.5, 1.0) : vec2(1.0);
    uv0 *= tex_aspect;
    out_axis = axis;
    out_T = T; out_B = Bb; out_face_n = face_n;
    out_uv0 = uv0;
    // Derivatives of the UN-shifted uv, taken before any early return: the
    // march below exits at data-dependent iterations per pixel, so implicit-
    // LOD texture() inside it had undefined derivatives (mip sparkle along
    // the parallax layer steps). Same approach as pom_march.
    const vec2 spom_gdx = dFdx(uv0), spom_gdy = dFdy(uv0);

    vec3 V_w = normalize(scene.camera_pos.xyz - vWorldPos);
    vec3 V_t = vec3(dot(V_w, T), dot(V_w, Bb), dot(V_w, face_n));
    // Grazing fade. The old hard cutoff (V_t.z < 0.2 → flat) switched
    // from FULL parallax to none across one pixel, and just above the
    // cutoff P = V_t.xy / V_t.z is near its largest — walls viewed
    // along their plane (looking up from the base) showed maximal
    // smearing right next to a flat band. Ramp the strength to zero
    // instead so displacement dies out smoothly toward grazing.
    float graze = smoothstep(0.12, 0.45, V_t.z);
    if (graze <= 0.001) return uv0;   // fully grazing — skip the march

    // Ramp parallax depth down at distance so far walls stay flat (no
    // visible texture swimming) and only close-up walls show the bumps.
    float d = distance(vWorldPos, scene.camera_pos.xyz);
    float depth_w = 1.0 - smoothstep(8.0, 30.0, d);
    if (depth_w <= 0.001) return uv0;

    // 4cm peak-to-trough at 1m view, scaled by the runtime slider
    // (scene.spom_params.x; 0 = flat, 1 = engine default). When the
    // multiplier is ~0 we'd march flat texture coords for nothing —
    // bail to the un-parallaxed path so SPOM cost goes to zero.
    const float kHeightScaleBase = 0.04;
    float spom_strength = max(0.0, scene.spom_params.x);
    if (spom_strength < 0.01) return uv0;
    float kHeightScale = kHeightScaleBase * spom_strength;
    // graze folds into the effective depth so the march, the recessed
    // depth-override and the silhouette test all fade together.
    float eff_depth = depth_w * graze;
    vec2  P = (V_t.xy / V_t.z) * (kHeightScale * eff_depth) * tex_aspect;

    // Step count ramps with depth_w: full 12 steps for the closest
    // walls (depth_w → 1), down to 4 steps for walls just inside the
    // SPOM fade kick-in. Cap is 12 (was 16) — TAA averages the
    // remaining step-quantisation noise away on close walls, and the
    // binary-search refine below adds sub-step precision. Each saved
    // step is one texture() in the castle-interior hot path.
    int kSteps = max(4, int(round(depth_w * 12.0)));
    float layer_step = 1.0 / float(kSteps);
    vec2  uv_step    = -P * layer_step;
    float current_layer = 0.0;
    vec2  cur_uv = uv0;
    float cur_h  = 1.0 - textureGrad(u_height[height_idx], cur_uv, spom_gdx, spom_gdy).r;
    for (int i = 0; i < 12; ++i) {
        if (i >= kSteps) break;
        if (current_layer >= cur_h) break;
        cur_uv += uv_step;
        current_layer += layer_step;
        cur_h = 1.0 - textureGrad(u_height[height_idx], cur_uv, spom_gdx, spom_gdy).r;
    }
    // Refine in two stages so the height steps don't read as visible
    // "stair lines" on the surface:
    //   1. Linear-interp between the last two layer samples (cheap).
    //   2. 2-iter binary search bracketed by (prev_uv, cur_uv) — each
    //      iteration halves the residual error, so 2 iters takes the
    //      step seam from ~6 % of one layer down to ~1.5 %. Visually
    //      fine after TAA averaging; costs 2 extra texture taps per
    //      SPOM-active pixel (was 4 — castle-interior perf tighten).
    vec2  prev_uv = cur_uv - uv_step;
    float prev_h  = 1.0 - textureGrad(u_height[height_idx], prev_uv, spom_gdx, spom_gdy).r;
    float prev_layer = current_layer - layer_step;
    float a = cur_h  - current_layer;
    float b = prev_h - prev_layer - a;
    float w = clamp(a / (b - a + 1e-4), 0.0, 1.0);
    vec2 final_uv = mix(cur_uv, prev_uv, w);
    float final_layer = mix(current_layer, prev_layer, w);
    {
        vec2 lo_uv = prev_uv;     // shallower (above surface)
        vec2 hi_uv = cur_uv;      // deeper (below surface)
        float lo_l = prev_layer;
        float hi_l = current_layer;
        for (int j = 0; j < 2; ++j) {
            vec2  m_uv = (lo_uv + hi_uv) * 0.5;
            float m_l  = (lo_l  + hi_l)  * 0.5;
            float m_h  = 1.0 - textureGrad(u_height[height_idx], m_uv, spom_gdx, spom_gdy).r;
            if (m_l < m_h) {       // still above the surface
                lo_uv = m_uv; lo_l = m_l;
            } else {                // below — bracket the upper half
                hi_uv = m_uv; hi_l = m_h;
            }
        }
        final_uv    = (lo_uv + hi_uv) * 0.5;
        final_layer = (lo_l  + hi_l)  * 0.5;
    }

    // SPOM silhouette test: convert the UV offset back to world units
    // along the face's tangent + bitangent. If the visible point lies
    // outside the cube face's geometric extent (face_size_world holds
    // the brush's full XYZ size), the bricks would be jutting past the
    // edge — make those pixels transparent so the silhouette looks
    // bumped at corners instead of flat-clipped.
    vec2 d_uv = final_uv - uv0;
    float d_T = d_uv.x / max(uv_scale * tex_aspect.x, 1e-4);
    float d_B = d_uv.y / max(uv_scale, 1e-4);
    // World-axis index for T and B: 0=X, 1=Y, 2=Z. T's axis depends on
    // the dominant face axis (see basis selection above).
    int t_axis = (axis == 0 || axis == 2) ? (axis == 0 ? 2 : 0) : 0;
    int b_axis = (axis == 1) ? 2 : 1;
    // Center of this brush in world space lives in pc.model[3].
    vec3 center = vec3(pc.model[3].x, pc.model[3].y, pc.model[3].z);
    vec3 rel = vWorldPos - center;
    float pos_T = (axis == 0) ? rel.z * sign(N.x) :
                  (axis == 1) ? rel.x :
                                rel.x * sign(N.z);
    float pos_B = (axis == 1) ? rel.z * sign(N.y) : rel.y;
    float vis_T = pos_T + d_T;
    float vis_B = pos_B + d_B;
    float ext_T = face_size_world[t_axis] * 0.5;
    float ext_B = face_size_world[b_axis] * 0.5;
    // Silhouette overhang: parallax has walked the visible point past
    // the geometric face edge. Caller draws this fragment with alpha
    // = 0 so the underlying scene_color (terrain / other walls drawn
    // earlier) shows through the brick cavity.
    // Silhouette test with tolerance. Pure ext_T/ext_B fired at joints
    // between two abutting brushes (e.g. tower-to-wall, gate-block-to-
    // wall) — the parallax displacement crossed the face edge but the
    // neighbour brush is RIGHT THERE filling the gap, so writing
    // alpha = 0 made the joint show the (often-shadowed) terrain
    // behind both brushes instead of the neighbour wall. Allowing the
    // displacement to extend up to 25 % past the geometric edge means
    // the test only fires when the parallax goes into a REAL gap (an
    // outside corner with no neighbour brush). Trade-off: bricks now
    // visibly poke a bit past the edge before silhouette kicks in;
    // visually fine for the brick depth scale we use.
    // Skip silhouette discard for top/bottom-facing surfaces (axis == 1).
    // Floors don't have visible silhouette edges from above — you're
    // looking AT them, not along them — but at grazing camera angles
    // the parallax displacement can still walk past the floor's XZ
    // extent and trigger the discard. That wrote alpha = 0 + depth =
    // 1.0 → compose substituted procedural sky for the downward ray
    // direction → grey patches that flickered with TAA jitter as the
    // player moved. SPOM displacement still works on floors; only the
    // alpha-cavity effect is gated to walls.
    // Tightened from 0.25 → 0.15: the wider pad let entire wall panes
    // read as "elevated outward" because all corner bricks could poke a
    // noticeable distance past the edge. 15 % is enough to absorb the
    // typical 1-cm parallax displacement without producing a visible
    // shelf at the geometric edge. The improved seam ray below catches
    // the extra silhouette pixels the tighter pad creates.
    // Restored to 0.15. Tightening to 0.05 made silhouettes fire so
    // eagerly that increasing the SPOM strength slider shrank the
    // whole wall outline (every edge pixel discarded to sky). 0.15
    // is the balanced value: silhouettes only fire on real outer-
    // corner overshoots, not on at-normal parallax shifts.
    float pad = 0.15;
    if (axis != 1 &&
        (abs(vis_T) > ext_T * (1.0 + pad) ||
         abs(vis_B) > ext_B * (1.0 + pad))) out_overhang_disc = true;
    // Displaced world position: lateral tangent offset only. The
    // `-face_n * depth` push (toward inside the cube) was tempting
    // for "true" 3D shadow origins but moved the origin INSIDE the
    // brush — every RT shadow ray then hit the cube's own back face,
    // dark patches everywhere. Keeping just the tangent shift means
    // shadow rays trace from a laterally-offset point on the cube
    // surface; that captures the brick-row alignment for falling sun
    // angles (visible self-shadow at brick edges) without the
    // self-occlusion bug.
    out_displaced_pos = vWorldPos + T * d_T + Bb * d_B;
    // Recessed point for SSDM: same lateral shift, plus the inward push
    // along the inward face normal by the world-space depth the ray hit.
    // The height field's full thickness along the normal is
    // (kHeightScale * depth_w) world units (the P vector was scaled by the
    // same factor); final_layer is the [0,1] fraction of that thickness at
    // the hit. Pushing INTO the surface (-face_n) keeps the displacement
    // inward-only, matching cube.frag's `layout(depth_greater)` contract
    // so gl_FragDepth writes never go nearer than the rasterised face.
    float depth_world = final_layer * kHeightScale * eff_depth;
    out_recessed_pos  = out_displaced_pos - face_n * depth_world;
    return final_uv;
}

// Height-aware splat blend (AAA terrain Phase 1). Instead of a linear
// cross-fade, the layer whose surface sits HIGHER at this texel wins the
// contested band — grass grows between stones, dirt settles into rock
// cracks. `w` is the plain blend weight, ha/hb the two layers' height
// samples (0..1, from the displacement packed in the normal alphas).
// Squared heights give a moderate sharpening; the 0.04 floor keeps the
// ratio finite and the blend smooth where height data is flat.
float height_lerp(float w, float ha, float hb) {
    float a = (1.0 - w) * (0.04 + ha * ha);
    float b = w * (0.04 + hb * hb);
    return b / (a + b);
}

// Terrain relief height fields (0 = deepest, 1 = top), in the material
// repeat UV `u`. MUST match what the splat below actually shows, or the
// parallax shifts the texture by bumps that aren't in it.
//   grass band: rocky-rugged (u_height[5]) where no grass grows, the
//               stylized grass ground (kMatGrass, x1.5 tile) where it does.
//   rock band : rock (x1) blended toward dirt (x1.25) by `pdw`.
// Mip level for the relief taps, set once per pixel before the march
// (explicit LOD: implicit derivatives are undefined inside the
// divergent march loop, and LOD 0 at range thrashes the texture cache).
float g_tpom_lod = 0.0;
float tpom_h_grass(vec2 u, float gp) {
    // Branch to ONE fetch where the grass mask is saturated (most pixels).
    if (gp < 0.02) return textureLod(u_height[5], u, g_tpom_lod).r;
    if (gp > 0.98) return textureLod(u_normal[kMatGrass], u * 1.5, g_tpom_lod).a;
    return mix(textureLod(u_height[5], u, g_tpom_lod).r,
               textureLod(u_normal[kMatGrass], u * 1.5, g_tpom_lod).a, gp);
}
float tpom_h_rock(vec2 u, float pdw) {
    if (pdw < 0.02) return textureLod(u_normal[kMatRock], u, g_tpom_lod).a;
    return mix(textureLod(u_normal[kMatRock], u, g_tpom_lod).a,
               textureLod(u_normal[kMatDirt], u * 1.25, g_tpom_lod).a, pdw);
}
float tpom_h(vec2 u, int kind, float k) {
    if (kind == 2) return textureLod(u_normal[kMatRoad], u, g_tpom_lod).a;
    return (kind == 0) ? tpom_h_grass(u, k) : tpom_h_rock(u, k);
}

// ---- Grass blade shadows on the terrain ----
// The lawn's blade field is grass_blades.glsl - the one the lawn itself is drawn from and the
// raymarched terrain shades with (this file had an older copy of it: one grass type, one blade
// height, so the shadows on the mesh terrain no longer matched the blades).
#include "grass_blades.glsl"
// Sun visibility 0..1 at ground point `p` (on the terrain surface).
float grass_blade_shadow(vec3 p, vec3 L) {
    if (L.y < 0.05) return 1.0;
    return grass_blade_vis(p, p.y, L, 0.01);
}

// Relief occlusion march (Tatarchuk 2006, "Practical POM" self-shadow):
// from the hit point, step toward the light through the height field
// and keep the largest "how far is the terrain above my ray" sample,
// weighted toward samples near the start (closer occluders = darker).
// Returns sun visibility 0..1. `Lt` = light dir in tangent space,
// `D` = relief depth in UV units (same scale the POM march used).
float tpom_shadow(vec2 uv, float h0, vec3 Lt, float D, int kind, float k) {
    if (Lt.z <= 0.02 || h0 >= 0.999) return 1.0;
    const int kN = 6;
    vec2  span = (Lt.xy / max(Lt.z, 0.15)) * D * (1.0 - h0);
    float occ = 0.0;
    for (int i = 1; i <= kN; ++i) {
        float t  = float(i) / float(kN);
        float rh = h0 + (1.0 - h0) * t;           // ray height at step
        float sh = tpom_h(uv + span * t, kind, k);
        occ = max(occ, (sh - rh) * (1.0 - t));
    }
    // ~0.06 of the full relief above the ray already reads as shadow.
    return 1.0 - smoothstep(0.0, 0.16, occ);
}

vec3 triplanar_normal(sampler2D ntex, vec3 wp, vec3 N) {
    // Same pow(abs(N), 4) → squarings trick as triplanar_sample.
    vec3 blend = abs(N); blend *= blend; blend *= blend;
    blend /= max(blend.x + blend.y + blend.z, 1e-3);

    // P15: dominant-axis early-out (matches triplanar_sample). Skip
    // 2 of 3 normal-map taps + their TBN math when one axis dominates.
    if (blend.x >= 0.95) {
        vec3 nx = texture(ntex, wp.zy).rgb * 2.0 - 1.0;
        return normalize(vec3(N.x + nx.z * sign(N.x), nx.y, nx.x));
    }
    if (blend.y >= 0.95) {
        vec3 ny = texture(ntex, wp.xz).rgb * 2.0 - 1.0;
        return normalize(vec3(ny.x, N.y + ny.z * sign(N.y), ny.y));
    }
    if (blend.z >= 0.95) {
        vec3 nz = texture(ntex, wp.xy).rgb * 2.0 - 1.0;
        return normalize(vec3(nz.x, nz.y, N.z + nz.z * sign(N.z)));
    }

    vec3 nx = texture(ntex, wp.zy).rgb * 2.0 - 1.0;
    vec3 ny = texture(ntex, wp.xz).rgb * 2.0 - 1.0;
    vec3 nz = texture(ntex, wp.xy).rgb * 2.0 - 1.0;

    // For each axis, build a world-space normal: the tangent-space sample's
    // .xy goes into the perpendicular plane, .z reinforces the axis.
    vec3 wnx = vec3(N.x + nx.z * sign(N.x), nx.y, nx.x);
    vec3 wny = vec3(ny.x, N.y + ny.z * sign(N.y), ny.y);
    vec3 wnz = vec3(nz.x, nz.y, N.z + nz.z * sign(N.z));

    return normalize(wnx * blend.x + wny * blend.y + wnz * blend.z);
}

// ---- Relief helpers shared by the brush SPOM, triplanar-mesh and UV-mesh
// parallax paths. `Vt` / `Lt` = view / light in the face's tangent frame
// (x along +u, y along +v, z = face normal); `hs` = relief depth in uv.
vec2 pom_march(int hix, vec2 uv0, vec3 Vt, float hs, vec2 gdx, vec2 gdy) {
    vec2  P   = Vt.xy / max(Vt.z, 0.12) * hs;
    int   nL  = int(mix(28.0, 10.0, clamp(Vt.z, 0.0, 1.0)));
    float lay = 1.0 / float(nL);
    vec2  duv = P * lay;
    vec2  uv  = uv0;
    float cur = 0.0;
    float hd  = 1.0 - textureGrad(u_height[hix], uv, gdx, gdy).r;
    for (int i = 0; i < 28; ++i) {
        if (i >= nL || cur >= hd) break;
        uv  -= duv;
        cur += lay;
        hd   = 1.0 - textureGrad(u_height[hix], uv, gdx, gdy).r;
    }
    vec2  puv    = uv + duv;
    float after  = hd - cur;
    float before = (1.0 - textureGrad(u_height[hix], puv, gdx, gdy).r) - (cur - lay);
    float wgt    = after / min(after - before, -1e-5);
    return mix(uv, puv, clamp(wgt, 0.0, 1.0));
}
// Relief silhouette clipping on curved surfaces. For a cylinder of radius
// R = 1/curv, a view ray hitting the surface with cos(angle) = c can reach
// at most R * (1 - sqrt(1 - c^2)) below it before it leaves the solid
// again - if the parallax hit is deeper, this pixel shows what is behind
// (bumpy stone outline instead of a perfectly smooth tower edge). Flat
// faces (curv ~ 0) are never clipped.
bool pom_silhouette_clip(float depth_m, float c, float curv);   // (below: needs the ray-query helpers)
// Cavity darkening x parallax self-shadow toward the sun (Tatarchuk 2006).
// Deferred lighting shades the FLAT surface, so these cues are what make
// the relief read as depth; they go into albedo, recomputed every frame.
float pom_shade(int hix, vec2 uv, vec3 Lt, float hs, vec2 gdx, vec2 gdy, float deep) {
    float h0  = textureGrad(u_height[hix], uv, gdx, gdy).r;
    float cav = mix(mix(0.80, 0.62, deep), 1.0, smoothstep(0.0, 0.6, h0));
    float vis = 1.0;
    if (Lt.z > 0.05 && h0 < 0.98) {
        vec2 sP = Lt.xy / max(Lt.z, 0.15) * hs * (1.0 - h0);
        float occ = 0.0;
        for (int k = 1; k <= 8; ++k) {
            float t  = float(k) / 8.0;
            float rh = h0 + (1.0 - h0) * t;
            float sh = textureGrad(u_height[hix], uv + sP * t, gdx, gdy).r;
            occ = max(occ, (sh - rh) * (1.0 - t));
        }
        vis = 1.0 - mix(0.40, 0.72, deep) * smoothstep(0.0, 0.12, occ);
    } else if (Lt.z <= 0.05) {
        vis = 0.9;   // face turned away from the sun: crevices stay a bit darker
    }
    return cav * vis;
}

// Per-surface PBR parameters for the G-buffer (deferred GGX). No roughness
// / metal maps ship with the brush textures, so they come from the texture
// slot (ambientCG set) - values from the sets' roughness-map means.
// Returns (roughness, metallic).
// Army paint for the vehicles: a three-colour disruptive pattern in the
// part's own space (it stays put when the turret turns), with what a
// vehicle in the field picks up - dust on the upward faces and low down,
// rain streaks down the sides, chipped paint in small specks. Parts that
// are already dark (tracks, barrels, tyres) keep their colour.
float ap_hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float ap_noise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(ap_hash(i), ap_hash(i + vec3(1, 0, 0)), f.x), mix(ap_hash(i + vec3(0, 1, 0)), ap_hash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(ap_hash(i + vec3(0, 0, 1)), ap_hash(i + vec3(1, 0, 1)), f.x), mix(ap_hash(i + vec3(0, 1, 1)), ap_hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
vec3 army_paint(vec3 op, vec3 on, vec3 base) {
    float lum = dot(base, vec3(0.2126, 0.7152, 0.0722));
    if (lum < 0.10) return base;                                // rubber, tracks, gun steel
    vec3 q = op * 1.35;
    float a = ap_noise(q) * 0.62 + ap_noise(q * 2.3 + 7.1) * 0.28 + ap_noise(q * 5.1 + 3.3) * 0.10;
    float b = ap_noise(q * 0.9 + 19.7) * 0.62 + ap_noise(q * 2.1 + 31.4) * 0.28 + ap_noise(q * 4.7 + 11.9) * 0.10;
    vec3 col = vec3(0.115, 0.140, 0.070);                       // olive drab
    col = mix(col, vec3(0.150, 0.115, 0.070), smoothstep(0.545, 0.565, a));   // earth brown
    col = mix(col, vec3(0.028, 0.032, 0.026), smoothstep(0.585, 0.605, b));   // black
    col *= clamp(lum / 0.16, 0.55, 1.5);                        // lighter and darker panels stay so
    // Weathering.
    float fine = ap_noise(op * 23.0) * 0.6 + ap_noise(op * 61.0) * 0.4;
    col *= 0.86 + 0.28 * fine;                                  // uneven matt paint
    float streak = ap_noise(vec3(op.x * 9.0, op.y * 0.7, op.z * 9.0));
    col *= 1.0 - 0.22 * smoothstep(0.55, 0.9, streak) * (1.0 - abs(on.y));      // rain streaks on the sides
    float dust = clamp(max(on.y, 0.0) * 0.55 + (1.0 - smoothstep(0.2, 1.3, op.y)) * 0.5, 0.0, 1.0) *
                 smoothstep(0.25, 0.75, ap_noise(op * 3.1 + 5.0));
    col = mix(col, vec3(0.21, 0.185, 0.14), dust * 0.55);       // dry mud and dust
    float chip = smoothstep(0.80, 0.86, ap_noise(op * 37.0)) * smoothstep(0.45, 0.7, ap_noise(op * 4.3 + 2.0));
    col = mix(col, vec3(0.30, 0.29, 0.27), chip * 0.7);         // bare primer where the paint chipped
    return col;
}

vec2 surface_rough_metal(int tex_idx, bool foliage, bool mesh_draw) {
    if (foliage) return vec2(0.55, 0.0);
    // Hand-held props (flashlight.cpp) pick a metal by a negative index.
    if (tex_idx == -2) return vec2(0.30, 1.0);               // brushed steel
    if (tex_idx == -3) return vec2(0.46, 1.0);               // anodised aluminium
    if (tex_idx == -4) return vec2(0.10, 1.0);               // chrome
    if (mesh_draw && tex_idx < 0) return vec2(0.62, 0.0);    // painted vehicles / jets
    switch (tex_idx) {
        case 0:  return vec2(0.88, 0.0);   // Ground054
        case 1:  return vec2(0.80, 0.0);   // Bricks078 (castle walls)
        case 2:  return vec2(0.62, 0.0);   // Wood048
        case 3:  return vec2(0.38, 1.0);   // Metal042A
        case 4:  return vec2(0.74, 0.0);   // PaintedBricks001
        case 5:  return vec2(0.42, 0.0);   // Tiles130 (glazed)
        case 6:  return vec2(0.50, 0.0);   // Tiles074
        case 13: return vec2(0.58, 0.0);   // RoofingTiles012A
        case 14: return vec2(0.82, 0.0);   // PavingStones131
        case 23: return vec2(0.27, 1.0);   // the rocket's stainless hull
        case 24: return vec2(0.86, 0.0);   // concrete
        default: return vec2(0.78, 0.0);   // rock / stone clutter etc.
    }
}
// gbuffer1.a (2 bits): 1.0 = leaves / needles, 1/3 = other vegetation
// (bark, rocks: foliage ray offsets but no translucency), 2/3 = metal,
// 0 = dielectric.
bool g_veg_solid = false;
vec4 g_card_tex = vec4(0.0, 0.0, 0.0, -1.0);   // leaf card: atlas colour (a >= 0 marks a card pixel)
float gbuf_flags(bool foliage, float metal) {
    return foliage ? (g_veg_solid ? 0.3333 : 1.0) : (metal > 0.5 ? 0.6667 : 0.0);
}

