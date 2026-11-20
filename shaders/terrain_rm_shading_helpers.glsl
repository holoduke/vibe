// terrain_rm_shading_helpers.glsl - part of terrain_raymarch.frag (included there, in place; not a standalone shader).
// Normal, terrain / grass shadows, road weights, ground material, reflection marches

// Terrain cavity occlusion from the baked heightmap: horizon scan in 6
// directions x 4 radii (0.8 .. 12 m), measured against the local tangent
// plane so an even slope reads as open and only gullies, ditches, bank
// feet and other narrow places darken. Texture taps only - no FBM, no rays.
float terrainCavityAO(vec2 p) {
    float h0 = meshHeight(p);
    const float ge = 0.75;
    vec2 g = vec2(meshHeight(p + vec2(ge, 0.0)) - meshHeight(p - vec2(ge, 0.0)),
                  meshHeight(p + vec2(0.0, ge)) - meshHeight(p - vec2(0.0, ge))) / (2.0 * ge);
    const float kR[4] = float[4](0.8, 2.0, 5.0, 12.0);
    float occ = 0.0;
    for (int i = 0; i < 6; ++i) {
        float a = 0.35 + float(i) * 1.04719755;
        vec2 d = vec2(cos(a), sin(a));
        float e = 0.0;
        for (int k = 0; k < 4; ++k) {
            float r = kR[k];
            e = max(e, (meshHeight(p + d * r) - h0) / r - dot(g, d));
        }
        occ += e * inversesqrt(1.0 + e * e);   // sin of the horizon angle
    }
    return clamp(1.0 - 1.5 * occ / 6.0, 0.12, 1.0);
}

float grassEligibility(vec2 worldXZ, float terrainY);
vec3 calcNormal(vec3 pos, float t) {
    // From the air (hits of raymarch_far, beyond the normal range): the slope of the
    // landscape itself over about a pixel. The screen-space path below is too coarse out
    // there (the hit positions are only that exact) and left the ground a flat sheet.
    if (t > 1500.0) {
        float e = max(0.0025 * t, 3.0);
        g_rm_fp_only = true;
        float h0 = terrainM_lod(pos.xz, t);
        vec3 nf = normalize(vec3(h0 - terrainM_lod(pos.xz + vec2(e, 0.0), t), e, h0 - terrainM_lod(pos.xz + vec2(0.0, e), t)));
        g_rm_fp_only = false;
        return nf;
    }
    // Far-distance fast path: derive the normal from screen-space
    // derivatives of the hit position itself (`dFdx(pos)` and
    // `dFdy(pos)` are the tangents along screen x/y). Zero terrainH
    // calls vs the 3× finite-difference path. Per-quad-constant — gives
    // a faceted look up close — but at lod_far_m+ the FBM is LOD'd to
    // few octaves anyway, so the surface has no per-pixel detail to lose.
    // Threshold is pc.emissive.z (the existing lod_far_m). Past this we
    // already snap to lod_min_octaves on the FBM, so swapping the normal
    // method costs nothing visible.
    if (t > pc.emissive.z) {
        vec3 dpdx = dFdx(pos);
        vec3 dpdy = dFdy(pos);
        vec3 Nfast = cross(dpdy, dpdx);
        // Cheap sign-flip in case the quad winding lands us pointing
        // into the ground.
        if (Nfast.y < 0.0) Nfast = -Nfast;
        return normalize(Nfast);
    }
    // 3-tap finite-difference is the only path that preserves the
    // per-octave `1/(1+|d|²)` damp variation — that variation IS what
    // makes rocks visible in snow and defines mountain ridges. Two
    // attempts at the analytic gradient (terrainH_grad_lod) were tried
    // (30 m gate and 200 m gate); both visibly lost detail. Until a
    // proper analytic gradient WITH the damp-derivative term is
    // implemented (needs running Hessian alongside the gradient — see
    // IQ's morenoise article), this stays at 3-tap FD. The cheaper
    // sampleHeightDelta gate inside the LOD FBM funcs is kept —
    // that's invisible past 80 m.
    float eps = 0.02 + 0.0008 * t;
    // Close up the stones are centimetres across: sample the slope at about
    // one pixel's width, or the pebbles would be smoothed away.
    // The 3 taps are within ~2 cm of each other — well below the
    // plateau blend's 20 m smoothstep width — so plateauWeight is
    // identical at all three points. Compute it once, pass it into
    // the FBM via terrainH_lod_pw, save 2 redundant evaluations.
    float tpl = plateauWeight(pos.xz);
    // (The stones are left out of these three full-terrain taps and added
    // below from three cheap stone-only taps at about one pixel's width.)
    g_pebN_h = -1.0; g_pebN_k = 0.0;   // -1: no stones looked up for this pixel
    g_peb_mode = -1;
    float hC = terrainH_lod_pw(pos.xz, t, tpl);
    float hR = terrainH_lod_pw(pos.xz + vec2(eps, 0.0), t, tpl);
    float hU = terrainH_lod_pw(pos.xz + vec2(0.0, eps), t, tpl);
    g_peb_mode = 0;
    vec3 gN = vec3(hC - hR, eps, hC - hU) / eps;
    if (t < 26.0 && tpl < 0.99) {
        float pw = (1.0 - smoothstep(10.0, 26.0, t)) * (1.0 - tpl) * g_rm_detail_mul;
        // Under a dense lawn the stones cannot be seen: skip all their work
        // there (most of the terrain).
        if (pw > 0.01) pw *= 1.0 - smoothstep(0.55, 0.80, grassEligibility(pos.xz, pos.y));
        if (pw > 0.01) {
            float pe = max(0.0015, 1.5 * t * 2.0 * scene.viewport.w);
            float pR = rm_pebbles(pos.xz + vec2(pe, 0.0), t);
            float pU = rm_pebbles(pos.xz + vec2(0.0, pe), t);
            float pC = rm_pebbles(pos.xz, t);          // last: its stone id / rim stay in the globals
            g_pebN_h = pC; g_pebN_id = g_peb_id; g_pebN_k = g_peb_k; g_pebN_sz = g_peb_sz;
            gN.xz += vec2(pC - pR, pC - pU) / pe * pw;
        }
    }
    vec3 N = normalize(gN);

    // Closeup micro-bump — two-octave high-frequency noise gradient
    // tilts the surface normal so near-camera ground reads as rocky
    // / pebbly instead of glass-smooth. Falls off past 60 m so it
    // only costs the always-near pixels and never aliases at depth.
    // Uses noised() which returns (value, dN/dx, dN/dy) in one call.
    // Tightened from 0.001 → 0.05: pixels in the 50–60 m band were
    // paying full noised() cost for sub-1 % blend weight. The lost
    // detail is invisible at the cutoff distance.
    float detail_w = (1.0 - smoothstep(15.0, 60.0, t)) * g_rm_detail_mul;
    if (detail_w > 0.05) {
        vec3 d1 = noised(pos.xz * 4.0);
        vec3 d2 = noised(pos.xz * 13.0);
        vec2 grad = (d1.yz * 0.30 + d2.yz * 0.18) * detail_w;
        // Project the noise gradient as a tangent-plane perturbation.
        N = normalize(N + vec3(-grad.x, 0.0, -grad.y) * 0.45);
    }

    return N;
}

// Classic heightfield soft-shadow march — `min(k·h/t)`. k ties to
// the global Shadow softness slider (rt_params.x) so a single
// control softens both terrain self-shadow AND PCSS castle/box
// shadows. Slider 0 → k=64 (razor sharp), slider 0.15 → k=4
// (very soft). Cubic smoothstep at the end softens the linear
// penumbra ramp.
float calcShadow(vec3 pos, vec3 sunDir) {
    // Driven by global Shadow softness slider × per-terrain
    // softness scale. Same combination the PCSS below uses so both
    // shadow types respond to the same UI controls in lock-step.
    float soft_g = scene.rt_params.x;
    if (scene.terrain_params.w > 0.0) {
        soft_g *= max(0.05, scene.terrain_params.w);
    }
    float soft = clamp(soft_g / 0.15, 0.0, 1.0);
    float k    = mix(64.0, 4.0, soft);
    float res = 1.0;
    int kSteps = int(pc.tex_params.y);
    // Receiver-distance shadow budget: shadows on far receivers are
    // sub-pixel at typical FOV — even 4 taps look identical to 96
    // after TAA absorbs the residual noise. Was already capped to
    // 10/6 past 200/500 m by P20; tighten further:
    //   8 past 80 m, 4 past 200 m. ~3× faster on long-shot vistas.
    float recv_dist = length(pos - scene.camera_pos.xyz);
    if (recv_dist > 200.0) kSteps = min(kSteps, 4);
    else if (recv_dist > 80.0)  kSteps = min(kSteps, 8);
    else if (recv_dist > 40.0)  kSteps = min(kSteps, 16);
    // Skip the receiver-adjacent first step. At t = 0.05 we're 5 cm
    // above the surface we just shaded, so terrain(p.xz) ≈ pos.y →
    // h ≈ 0.05·sun.y → res update is bounded near 1.0 (no shadow
    // contribution). Just go straight to min_step (the smallest
    // useful step size we'd take next anyway). Saves 1 FBM call per
    // shadowed pixel — shadow pixels are most of the screen so the
    // sum is meaningful.
    // min_step is a function of recv_dist only — loop-invariant. Hoist out
    // of the per-iteration body (was being recomputed up to kSteps times,
    // each call a mix+smoothstep). Single value used both for the start
    // offset and the clamp inside the loop — the two callsites computed
    // the same number, so collapse to one.
    float min_step = mix(0.5, 4.0, smoothstep(100.0, 500.0, recv_dist));
    float t   = max(0.5, min_step);
    // Open air: while the baked heightmap proves the ray is far enough
    // above the ground that this step cannot lower `res` (k * h / t >= 1
    // even with the bake's 2 m margin), one texture fetch replaces the full
    // FBM - the result is identical, the step merely a little shorter.
    // Those steps are free: only FBM steps count against `kSteps` (so the
    // march reaches at least as far as before).
    float maxH_sh = terrainMaxHeight();
    int fbm_steps = 0;
    for (int i = 0; i < 96; ++i) {
        vec3 p = pos + t * sunDir;
        if (p.y > maxH_sh) break;                         // above every summit: nothing left to hit
        if (all(lessThan(abs(p.xz), vec2(kHeightmapSide * 0.5 - 2.0)))) {
            float hb = p.y - meshHeight(p.xz) - 2.0;
            if (hb > 0.5 && k * hb >= t) {
                t += clamp(hb, min_step, 100.0);
                if (t > 800.0) break;
                continue;
            }
        }
        if (++fbm_steps > kSteps) break;
        // Coarse FBM (same user octaves, no sculpt-delta tap).
        // Local-Lipschitz overstep doesn't help here — the shadow
        // march already uses full `h` per step (no global kStep
        // factor to shrink), so the only effect would be SMALLER
        // steps on cliffs (i.e. a regression). Keep the simple
        // h-clamp behaviour.
        float h = p.y - terrainM_shadow(p.xz, t);
        if (h < 0.001) { res = 0.0; break; }
        res = min(res, k * h / t);
        // Bigger min-step and lower hard-cap at distance — same logic
        // as the primary march. Combined with the receiver-distance
        // step budget above, distant shadow contributions converge in
#ifndef QLIKE_WATER
        // 4 jumps instead of crawling 24+ tiny steps. min_step hoisted
        // above the loop (loop-invariant).
#else /* QLIKE_WATER */
        // 4 jumps instead of crawling 24+ tiny steps.
#endif /* QLIKE_WATER */
        t += clamp(h, min_step, 100.0);
        if (t > 800.0 || res < 0.001) break;
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

// Fake grass-cast shadow on the raymarched terrain. Walks the grass
// mask along the sun's XZ direction from the terrain pixel toward
// the sun; cells with grass presence accumulate a soft shadow.
// Cheap (≤8 mask taps), and "good enough" because grass blades are
// sub-pixel at the kind of distance where the player would notice
// shadow inaccuracy. Returns 0..1 (0 = no grass shadow, 1 = fully
// in a grass cell's shadow). Caller multiplies the diffuse term by
// (1 - grass_shadow * strength).
#ifndef QLIKE_WATER
// ---- Exact grass blade shadows (shared with cube.frag) ----
#include "grass_blades.glsl"
// Light visibility 0..1 at ground point `p` (on the terrain surface).
float grass_blade_shadow(vec3 p, vec3 L) {
    if (L.y < 0.05) return 1.0;
    return grass_blade_vis(p, p.y, L, 0.01);
}


#endif /* ! QLIKE_WATER */
float grassCastShadow(vec3 pos, vec3 sunDir) {
    float strength = scene.grass_shadow_params.x;
    int   N        = int(scene.grass_shadow_params.y);
    float reach    = max(scene.grass_shadow_params.z, 0.1);
    if (strength < 1e-3 || N <= 0) return 0.0;
    // Sun's XZ direction (from terrain toward sun). Skip when the
    // sun is near vertical (sun_xz_len → 0) — no horizontal shadow
    // projection then.
    vec2 sun_xz = sunDir.xz;
    float sun_xz_len = length(sun_xz);
    if (sun_xz_len < 0.05) return 0.0;
    sun_xz /= sun_xz_len;
    float shadow = 0.0;
    for (int i = 1; i <= 8; ++i) {
        if (i > N) break;
        float dist = reach * (float(i) / float(N));
        vec2 sample_xz = pos.xz + sun_xz * dist;
        vec2 uv = (sample_xz / kHeightmapSide) + vec2(0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;
        float presence = textureLod(u_grass_mask, uv, 0.0).r;
        // Falloff with distance: nearer grass casts a stronger
        // contribution; far samples are smaller blades / lower sun
        // angle penetration.
        float falloff = 1.0 - smoothstep(0.0, 1.0, dist / reach);
        shadow = max(shadow, presence * falloff);
    }
    return shadow;
}

// Sample the CPU-baked grass mask (binding 13, RG8). Single source of
// truth shared with grass_raymarch.frag.
//   R = presence  (fixed at bake time)
//   G = slope mag (raw |∇h| / 2, un-normalized here)
// Slope cutoff is applied per-frame so the slider (grass_slope_n_min)
// affects both blades and the green underground tint in lock-step.
// Altitude + water gating stay per-pixel since they need terrainY.
float grassEligibility(vec2 worldXZ, float terrainY) {
    vec2 mask_uv = (worldXZ / kHeightmapSide) + vec2(0.5);
    if (mask_uv.x < 0.0 || mask_uv.x > 1.0 ||
        mask_uv.y < 0.0 || mask_uv.y > 1.0) return 0.0;
    vec2 mg = textureLod(u_grass_mask, mask_uv, 0.0).rg;
    float slope_n_min = scene.grass_extra.z;
    float slope_max = sqrt(max(1e-4, 1.0 / max(slope_n_min * slope_n_min, 1e-4) - 1.0));
    float slope_mag = mg.g * 2.0;
    float slope_w   = 1.0 - smoothstep(slope_max * 0.7,
                                        slope_max * 1.2, slope_mag);
    float alt_factor = 1.0 - smoothstep(55.0, 75.0, terrainY);
    // No lawn in the water, and none on the strip the water laps over:
    // a band of bare sand and pebbles runs along every shore.
    float underwater = (scene.water_params.x > 0.5)
        ? smoothstep(scene.water_params.y + 0.22, scene.water_params.y + 0.50, terrainY) : 1.0;
    return mg.r * slope_w * alt_factor * underwater;
}

// Castle road, as on the mesh terrain (cube.frag "Castle road"): exact XZ
// distance to the road polyline in the scene UBO gives the cobble / dirt-
// verge weights; the bisector-partitioned second loop gives a road-space
// UV (across, along) that stays continuous through the bends, so the
// cobble rows follow the curve. Returns (road_w, verge_w).
// How much a bomb crater has destroyed the surface at q (1 in the bowl,
// fading out over the rim): paving and verge give way to the crater mud.
float crater_cover(vec2 q) {
    float c = 0.0;
    int nc = int(scene.crater_info.x + 0.5);
    for (int i = 0; i < nc && i < 24; ++i) {
        float r = length(q - scene.craters[i].xz) / max(scene.craters[i].w, 0.5);
        c = max(c, 1.0 - smoothstep(1.05, 1.45, r));
    }
    return c;
}
vec2 roadWeights(vec2 q, out vec2 road_uv, out vec2 r_dir) {
    road_uv = vec2(0.0);
    r_dir = vec2(0.0, 1.0);
    int rn = int(scene.road_info.x + 0.5);
    if (rn < 2 || any(lessThan(q, scene.road_bbox.xy)) || any(greaterThan(q, scene.road_bbox.zw)))
        return vec2(0.0);
    float best = 1e9;
    for (int i = 0; i < 63; ++i) {
        if (i + 1 >= rn) break;
        vec2 a = scene.road_pts[i].xz, b = scene.road_pts[i + 1].xz;
        vec2 ab = b - a;
        float s = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
        vec2 dv = q - (a + ab * s);
        best = min(best, dot(dv, dv));
    }
    float rd = sqrt(best);
    float hw = scene.road_info.y, vw = scene.road_info.z;
    if (rd > hw + vw + 1.5) return vec2(0.0);
    // The main road runs straight through a junction: wherever a pixel lies
    // on the main road's paving, the main road's rows are used; the side
    // road's paving only starts at the main road's edge.
    int sp0 = int(scene.road_info.w + 0.5) / 128, sp1 = int(scene.road_info.w + 0.5) % 128;
    float best_lat = 1e9, r_along = 0.0;
    float main_lat = 1e9, main_along = 0.0; vec2 main_dir = vec2(0.0, 1.0);
    for (int i = 0; i < 63; ++i) {
        if (i + 1 >= rn) break;
        vec2 a = scene.road_pts[i].xz, b = scene.road_pts[i + 1].xz;
        // The side road comes back along its own points: only its outward
        // half lays the paving rows (the way back would mirror them).
        int sp_tip = sp0 + (sp1 - sp0) / 2;
        if (i >= sp_tip && i < sp1) continue;
        vec2 sd = normalize(b - a);
        // At the dead end the neighbouring points coincide (the polyline
        // turns straight back): there the segment's own direction is used,
        // and the last segment carries on past its end to cover the tip.
        vec2 ta_r = b - scene.road_pts[max(i - 1, 0)].xz;
        vec2 tb_r = scene.road_pts[min(i + 2, rn - 1)].xz - a;
        bool dead_end = dot(tb_r, tb_r) < 1e-4;
        vec2 ta = dot(ta_r, ta_r) < 1e-4 ? sd : normalize(ta_r);
        vec2 tb = dead_end ? sd : normalize(tb_r);
        float da = dot(q - a, ta), db = dot(q - b, tb);
        if (da < 0.0 && i > 0) continue;
        if (db > 0.0 && i + 2 < rn && !dead_end) continue;
        float lat = sd.x * (q - a).y - sd.y * (q - a).x;
        float f = da / max(da - db, 1e-4);
        bool spur = i >= sp0 && i < sp1;
        if (!spur && abs(lat) < abs(main_lat)) {
            main_lat = lat;
            main_along = mix(scene.road_pts[i].w, scene.road_pts[i + 1].w, f);
            main_dir = normalize(mix(ta, tb, clamp(f, 0.0, 1.0)));
        }
        if (abs(lat) < abs(best_lat)) {
            best_lat = lat;
            r_along = mix(scene.road_pts[i].w, scene.road_pts[i + 1].w, f);
            r_dir = normalize(mix(ta, tb, clamp(f, 0.0, 1.0)) + sd * 1e-3);
        }
    }
    if (abs(main_lat) < hw + 0.35) { best_lat = main_lat; r_along = main_along; r_dir = main_dir; }
    float g_tile = max(scene.spom_params.z, 0.25);
    if (best_lat < 1e8) road_uv = vec2(best_lat, r_along) / g_tile;
    float jit = (noise2(q * 1.7) - 0.5) * 0.45 + (noise2(q * 0.31 + 17.0) - 0.5) * 0.6;
    float road_w  = 1.0 - smoothstep(hw - 0.12, hw + 0.12, rd + jit);
    float verge_w = (1.0 - smoothstep(hw + 0.4 * vw, hw + vw + 0.5, rd + jit * 1.6)) * (1.0 - road_w);
    return vec2(road_w, verge_w) * (1.0 - crater_cover(q));
}

// Layered material: rock base, grass on flats, snow up high (with
// noise-broken transition), sand near the floor. Multi-frequency
// noise (`nz`) breaks regularity in the transition lines.
vec3 getMaterial(vec3 pos, vec3 nor) {
    float slope = nor.y;
    float h = pos.y;
    float nz = noise2(pos.xz * 0.04) * noise2(pos.xz * 0.005);
    vec3 rock  = scene.rm_rock.rgb;
    // Dry earth .. green ground (settings palette). `green amount` pushes
    // the noise toward green (0 = the old mostly-brown mix).
    float gmix = clamp(mix(nz, 1.0, scene.rm_ground_dry.w) + (nz - 0.25) * 0.5, 0.0, 1.0);
    vec3 grass = mix(scene.rm_ground_dry.rgb, scene.rm_ground_green.rgb, gmix);
    vec3 snow  = vec3(0.62, 0.65, 0.70);
    vec3 sand  = scene.terrain_sand_color.rgb;
    vec3 col = rock;
    col = mix(col, grass, smoothstep(0.5, 0.8, slope));
    float snowMask = smoothstep(80.0 - 20.0 * nz, 90.0, h)
                   * smoothstep(0.3, 0.7, slope);
    col = mix(col, snow, snowMask);
    float beachMask = smoothstep(2.5, 1.0, h) * smoothstep(0.5, 0.9, slope);
    col = mix(col, sand, beachMask);
    // Grass-density green tint — drives the ground colour where the
    // raymarched grass pass (grass_raymarch.frag) would place blades.
    // Two colours (close + far) blend with camera distance so meadows
    // can read as bright lit-green near the player and a darker
    // muted hue at distance (or whatever the user picks).
    float grass_amt = grassEligibility(pos.xz, pos.y);
    float cam_d     = distance(pos, scene.camera_pos.xyz);
    // Red sand (terra rossa): irregular patches on flats and gentle slopes
    // below the snow, thinner where grass grows thick; a reddish stain
    // also creeps onto the dry slopes around them.
    float red_amt = scene.rm_ground_green.w;
    if (red_amt > 1e-3) {
        float rs = 1.0 / max(scene.rm_red_sand.w, 0.2);
        vec2  rq = pos.xz * rs;
        float rn = noise2(rq * 0.018 + 41.0) * 0.6 + noise2(rq * 0.071 + 7.0) * 0.28 + noise2(rq * 0.31) * 0.12;
        float patchm = smoothstep(0.66 - 0.16 * red_amt, 0.80 - 0.14 * red_amt, rn);
        float flat_w = smoothstep(0.55, 0.85, slope);
        float alt_w  = 1.0 - smoothstep(55.0, 80.0, h);
        float red_w  = patchm * flat_w * alt_w * (1.0 - 0.65 * grass_amt) * min(1.0, red_amt * 1.4) * 0.85;
        vec3 red = scene.rm_red_sand.rgb * mix(0.85, 1.15, noise2(pos.xz * 0.9));
        col = mix(col, red, red_w);
        // Stain on the bare dry ground around the patches.
        col = mix(col, col * mix(vec3(1.0), scene.rm_red_sand.rgb / max(dot(scene.rm_red_sand.rgb, vec3(0.333)), 1e-3), 0.35),
                  smoothstep(0.45, 0.65, rn) * (1.0 - grass_amt) * alt_w * red_amt * 0.35);
    }
    // Same anchor distances the grass-blade alpha ramp uses, so the
    // blade-fade and the ground-tint transition stay aligned.
    // Far anchor from rt_.grass_ground_tint_far_distance (UI slider,
    // packed into grass_color_ground_far.w). Near anchor is 0 so the
    // close→far ramp is one continuous gradient — no slope-spike
    // inflection band that reads as a hard line. Matches grass.
    float tint_far  = max(50.0, scene.grass_color_ground_far.w);
    float ground_t  = smoothstep(0.0, tint_far, cam_d);
    vec3 ground_base = mix(scene.grass_color_ground.rgb,
                            scene.grass_color_ground_far.rgb,
                            ground_t);
    vec3 grass_top  = mix(ground_base, ground_base * 0.7, nz);
    float tint_str  = clamp(scene.grass_color_ground.w, 0.0, 1.0);
    col = mix(col, grass_top, grass_amt * tint_str);

    // Two-region shore tint — grass-area gets one colour, bare-
    // terrain gets another. Both fade smoothly with height above the
    // water level. Each is gated by grass_amt so they never compete.
    float h_above_water = max(0.0, pos.y - scene.water_params.y);
    {
        // Grass-area shore tint (rt_.terrain_shore_*).
        float ts_strength = scene.terrain_shore_color.a;
        if (ts_strength > 1e-3 && grass_amt > 0.001) {
            float ts_fade = max(0.05, scene.terrain_shore_params.x);
            float ts_w = 1.0 - smoothstep(0.0, ts_fade, h_above_water);
            col = mix(col, scene.terrain_shore_color.rgb,
                      ts_w * ts_strength * grass_amt);
        }
    }
    {
        // Bare-terrain shore tint (rt_.terrain_shore_general_*).
        float gt_strength = scene.terrain_shore_general_color.a;
        if (gt_strength > 1e-3 && grass_amt < 0.999) {
            float gt_fade = max(0.05, scene.terrain_shore_general_params.x);
            float gt_w = 1.0 - smoothstep(0.0, gt_fade, h_above_water);
            col = mix(col, scene.terrain_shore_general_color.rgb,
                      gt_w * gt_strength * (1.0 - grass_amt));
        }
    }
    // Ground saturation (settings).
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = max(mix(vec3(lum), col, scene.rm_rock.w), vec3(0.0));
    // Bomb craters (scene.craters): churned, wet mud in the bowl - darkest
    // at the bottom where water collects - and dry thrown earth on the rim,
    // reaching out in irregular rays. Applied after everything else so no
    // grass / sand tint survives in the hole.
    {
        int nc = int(scene.crater_info.x + 0.5);
        for (int i = 0; i < nc && i < 24; ++i) {
            vec4  c  = scene.craters[i];
            vec2  dv = pos.xz - c.xz;
            float r  = length(dv) / max(c.w, 0.5);
            if (r > 2.1) continue;
            float ang = atan(dv.y, dv.x);
            float ray = noise2(vec2(ang * 4.0 + c.x * 0.37, c.z * 0.37)) * 0.65 +
                        noise2(vec2(ang * 11.0 - c.z * 0.21, c.x * 0.21)) * 0.35;      // ejecta rays
            float n1 = noise2(pos.xz * 1.7 + 3.0), n2 = noise2(pos.xz * 6.3);
            float bowl   = 1.0 - smoothstep(0.82, 1.06 + 0.12 * (n1 - 0.5), r);
            float ejecta = (1.0 - smoothstep(1.0, 1.15 + 0.9 * ray, r)) * (1.0 - bowl);
            vec3  soil = vec3(0.115, 0.085, 0.055) * mix(0.78, 1.22, n1);
            vec3  mud  = vec3(0.062, 0.044, 0.029) * mix(0.75, 1.25, n2);
            mud = mix(mud * 0.5, mud, smoothstep(0.0, 0.5, r));
            col = mix(col, soil, ejecta * 0.85);
            col = mix(col, mud, bowl);
        }
    }
    // Winter: snow on everything that faces up (not on the lake bed).
    {
        float sc = snow_cover(nor, pos);
        if (scene.water_params.x > 0.5 && pos.y < scene.water_params.y) sc = 0.0;
        col = mix(col, vec3(0.86, 0.89, 0.93), sc);
    }
    return col;
}

// Cheap raymarch for water-reflection rays. Same conservative-step
// heightfield trace as `raymarch()` but with a fixed reduced step
// count (the reflected rays usually hit within 30-50 steps because
// the surface is close to the camera). Skips jitter to avoid
// per-pixel reflection noise that wouldn't denoise temporally.
float raymarchReflect(vec3 ro, vec3 rd) {
    float maxH = terrainMaxHeight();
    float t = 0.0;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    if (ro.y > maxH) {
        t = (ro.y - maxH) / max(-rd.y, 1e-4);
    }
    const int kReflSteps = 64;
    const float kReflStep = 0.5;     // slightly aggressive
    for (int i = 0; i < kReflSteps; ++i) {
        vec3 pos = ro + t * rd;
        float h = pos.y - terrainM_lod(pos.xz, t);
        if (abs(h) < 0.005 * t) break;     // looser threshold than primary
        if (t > 800.0) return -1.0;        // shorter range than primary
        t += kReflStep * h;
    }
    return t;
}

// === Mesh-terrain heightmap marches (water-only mode) ===============
// Defined here (after the Scene UBO) so they can read
// scene.terrain_local_info.y = mesh max height + pad for the empty-air
// early-out. The baked mesh is smooth (bilinear texels), so an
// aggressive height-proportional step + the air skip keep these cheap
// — the same tricks the FBM marcher uses to stay off the TDR watchdog.
float raymarchMesh(vec3 ro, vec3 rd, float t_max) {
    float maxH = scene.terrain_local_info.y;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    float t = 0.0;
    if (ro.y > maxH) t = (ro.y - maxH) / max(-rd.y, 1e-4);
    for (int i = 0; i < 96; i++) {
        vec3 p = ro + rd * t;
        if (p.y > maxH && rd.y >= 0.0) return -1.0;
        float h = p.y - meshHeight(p.xz);
        if (h < 0.012 * t + 0.05) return t;
        t += max(0.5 + t * 0.012, h * 0.85);
        if (t > t_max) break;
    }
    return -1.0;
}
// Reflection ray over the baked heightmap (one texture fetch per step).
// Steps grow with distance so 96 of them reach the far mountains; a ray
// that runs out of steps or range is a MISS (the old FBM version returned
// wherever it had got to, which painted a dark sheet over the lake).
float raymarchReflectHM(vec3 ro, vec3 rd) {
    float maxH = terrainMaxHeight();
    if (rd.y >= 0.0 && ro.y > maxH) return -1.0;
    float t = 0.5, t_prev = 0.0;
    for (int i = 0; i < 96; ++i) {
        vec3 p = ro + rd * t;
        if (p.y > maxH && rd.y >= 0.0) return -1.0;
        if (any(greaterThan(abs(p.xz), vec2(kHeightmapSide * 0.5 - 2.0)))) return -1.0;
        float h = p.y - meshHeight(p.xz);
        if (h < 0.0) {
            float a = t_prev, b = t;
            for (int k = 0; k < 5; ++k) {
                float m = 0.5 * (a + b);
                vec3 q = ro + rd * m;
                if (q.y - meshHeight(q.xz) > 0.0) a = m; else b = m;
            }
            return a;
        }
        t_prev = t;
        t += max(0.6 + t * 0.03, h * 0.6);
        if (t > 1600.0) break;
    }
    return -1.0;
}
float raymarchReflectMesh(vec3 ro, vec3 rd) {

    float maxH = scene.terrain_local_info.y;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    float t = 0.0;
    if (ro.y > maxH) t = (ro.y - maxH) / max(-rd.y, 1e-4);
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        if (p.y > maxH && rd.y >= 0.0) return -1.0;
        float h = p.y - meshHeight(p.xz);
        if (h < 0.04 * t + 0.1) return t;
        t += max(0.8 + t * 0.02, h * 0.9);
        if (t > 700.0) break;
    }
    return -1.0;
}
float calcShadowMesh(vec3 pos, vec3 sunDir) {
    float t = 1.5;
    for (int i = 0; i < 18; i++) {
        vec3 p = pos + sunDir * t;
        if (p.y > scene.terrain_local_info.y) break;
        if (p.y - meshHeight(p.xz) < 0.0) return 0.0;
        t += 3.0 + t * 0.15;
        if (t > 140.0) break;
    }
    return 1.0;
}


// One of the photo-scanned ground sets at `uv`: normal + height (u_normal[idx]) and colour
// (u_albedo[idx]).
void rm_det_sample(int idx, vec2 uv, float lod, out vec4 n, out vec3 a) {
    n = textureLod(u_normal[idx], uv, lod);
    a = textureLod(u_albedo[idx], uv, lod).rgb;
}
