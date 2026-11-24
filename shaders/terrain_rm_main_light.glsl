// Part of main() in terrain_raymarch.frag: direct light: the sun's shadows (terrain, objects, clutter), ambient occlusion, sky light.
// Not a file of its own - it is included in the middle of main() and uses its local
// variables (ro, rd, t, pos, nor, col, ...). Split off only to keep that file readable;
// the order of the parts is the order of the #include lines there.

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
    if (dif > 0.0 && !RMDBG(4)) self_shadow = lo_ok ? lo_sh.x : rm_self_shadow_base(pos, nor, sunDir);


    // Fake grass-cast shadow: probe the grass mask along the sun's
    // XZ direction from this terrain pixel; cells with grass darken
    // the diffuse term. Strength + sample count + reach are all UI
    // sliders (0 strength = entirely off, no taps).
    float gcs_strength = scene.grass_shadow_params.x;
#ifndef VIBE_WATER
    // Legacy mask-probe shadow only when the exact blade march is off.
    float grass_shadow = (scene.grass_rm_params2.y > 0.5) ? 0.0 : grassCastShadow(pos, sunDir);
#else /* VIBE_WATER */
    float grass_shadow = grassCastShadow(pos, sunDir);
#endif /* VIBE_WATER */
    // Combine multiplicatively with self_shadow as transmittances.
    // shadow_amount=1 → light_through=0, so combined-blocked
    // = 1 - (1-self)*(1-grass) (union of independent occluders).
    // Was `max(...)` which under-darkens partially-lit slopes that
    // also have grass nearby — the comment was right, the code wasn't.
    self_shadow = 1.0 - (1.0 - self_shadow) *
                        (1.0 - grass_shadow * gcs_strength);
#ifndef VIBE_WATER
    // Flowers / ferns / clutter rocks: dedicated small-caster depth map
    // around the camera (same test as deferred_lighting.frag).
    if (scene.clutter_vp[3].w > 0.5 && dif > 0.0 && !RMDBG(16)) {
        vec4 lc = scene.clutter_vp * vec4(pos, 1.0);
        vec2 luv = lc.xy * 0.5 + 0.5;
        if (all(greaterThan(luv, vec2(0.002))) && all(lessThan(luv, vec2(0.998))) &&
            lc.z > 0.0 && lc.z < 1.0) {
            // Near cascade (layer 1): same centre, 4x finer (0.8 cm texels
            // over 16 m) for crisp stem / petal shadows close by.
            // (the wide layer only where it shows: inside the near one its weight is 0, and its
            //  sixteen fetches were thrown away)
            vec2 nndc = lc.xy * 4.0;
            float nr = max(abs(nndc.x), abs(nndc.y));
            float occ = (nr > 0.80) ? clutterOcc(luv, 0, lc.z - 0.0004) : 0.0;
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
    if (scene.grass_rm_params2.y > 0.5 && dif > 0.0 && !RMDBG(8) &&
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
#endif /* ! VIBE_WATER */

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
    float lod_far  = max(50.0, scene.fog_band.w);
    float lod_near = lod_far * 0.6;
    float rt_lod_t = 1.0 - smoothstep(lod_near, lod_far, cam_dist_pos);
    bool do_rt = rt_lod_t > 0.005 && !RMDBG(32);
    // (rm_rt_sun_shadow above: the soft band near the camera and the hard ray beyond it)
    // (Also for ground turned away from the sun: this shadow takes most of the sky's light
    //  with it as well - terrain_rm_main_gi.glsl, "the shadow of a canopy" - and asked only
    //  where the sun could reach, a face turned away kept all of its sky light and came out
    //  BRIGHTER than its shadowed neighbours. In the shade of a wood at a low sun, where the
    //  sand's fine normal turns to and from the sun from pixel to pixel, that was a ragged
    //  pattern of light and dark over the ripples.)
    // (`RMDBG` 65536: only where the ground faces the sun, as it was - to compare)
    if (dif > 0.0 || !RMDBG(65536)) dyn_shadow = lo_ok ? lo_sh.y : rm_rt_sun_shadow(pos, nor, sunDir, self_shadow);
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

    if (g_snow >= 0.01) rm_road_vis = mix(rm_road_vis, 1.0, snow_road_patch(pos.xz) * smoothstep(0.15, 0.6, g_snow));
    rm_road_vis = mix(rm_road_vis, 1.0, sand_cov);
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
        // (read from the home tile's baked heights: beyond the tile a clamped read drew its
        // edge row out as dark ruled lines over the land east / west / north / south of home)
        {
            vec2 cq = abs(pos.xz);
            cav_w *= 1.0 - smoothstep(kHeightmapSide * 0.5 - 40.0, kHeightmapSide * 0.5 - 6.0, max(cq.x, cq.y));
        }
        if (cav_w > 0.01) ao *= mix(1.0, terrainCavityAO(pos.xz), cav_w);
    }

    vec3 mate = getMaterial(pos, nor) * rm_det_tint;
#if !defined(VIBE_LITE) && !defined(VIBE_WATER)
    if (RMDBG(256)) mate = lo_ok ? vec3(0.1, 0.8, 0.1) : vec3(0.9, 0.1, 0.1);    // profiling: green = sun shadows taken from the low-resolution pass
#endif
    // The ground is darker where plants stand over it (leaf litter, damp soil,
    // less sky): a little under grass, more under bushes and trees, most where
    // both grow. Tree / bush cover is baked from the placed plants and blurred
    // (soft outer borders); the grass term is the meadow mask.
    {
        vec2 vuv = pos.xz / 2048.0 + 0.5;
        if (all(greaterThan(vuv, vec2(0.0))) && all(lessThan(vuv, vec2(1.0))) && rm_road.x < 0.99) {
            float trees = textureLod(u_albedo[22], vuv, 1.0).r;
            float grass = smoothstep(0.25, 0.75, grassEligibility(pos.xz, pos.y));
            float cover = 0.09 * grass + 0.26 * trees + 0.07 * grass * trees;
            mate *= 1.0 - cover * (1.0 - rm_road.x);
        }
    }
    // From the air: trees are drawn to 1.5 km. Beyond that (only seen from
    // above 1 km) the woods are a canopy colour on the ground - where they
    // really stand inside the landscape (the baked cover), and in the same
    // height band, in patches, outside it.
    if (t > 1200.0) {
        vec2 cuv = pos.xz / 2048.0 + 0.5;
        float woods;
        if (all(greaterThan(cuv, vec2(0.0))) && all(lessThan(cuv, vec2(1.0)))) {
            woods = smoothstep(0.06, 0.40, textureLod(u_albedo[22], cuv, 2.0).r);
        } else {
            woods = rm_forest(pos.xz, pos.y, nor.y, scene.water_params.y);     // (the rule the trees there are placed by)
        }
        vec3 canopy = mix(vec3(0.028, 0.052, 0.024), vec3(0.058, 0.082, 0.034), tile_noise2(pos.xz, 0.11, vec2(0.0)));
        mate = mix(mate, canopy, woods * smoothstep(1200.0, 1700.0, t));
    }
    // The sea's shore (beyond home's lakes): a beach of sand from just under the high-water
    // line up to a metre or so above it, with a ragged upper edge, and sand on the bed
    // below - the height layers put grass at the water level, which ran straight into the sea.
    {
        float sea_k = smoothstep(1350.0, 1750.0, rm_home_dist(pos.xz));
        // (only as high as the sand can reach - 2.45 m over the water: above that its weight
        //  is 0, and its four noises were worked out for every pixel of the wider world)
        if (sea_k > 0.0 && scene.water_params.x > 0.5 && pos.y < scene.water_params.y + 2.5) {
            float wl = scene.water_params.y;
            float edge = wl + 0.5 + 1.0 * tile_noise2(pos.xz, 0.045, vec2(0.0)) + 0.35 * tile_noise2(pos.xz, 0.31, vec2(0.0));
            float sand = sea_k * (1.0 - smoothstep(edge - 0.5, edge + 0.6, pos.y));
            // (a little lighter than the lake shore's sand, no more: at 0.56 the beach burnt out
            // in the sun and bloomed into a glowing patch)
            vec3 sandc = vec3(0.29, 0.245, 0.17) * (0.90 + 0.20 * tile_noise2(pos.xz, 0.35, vec2(0.0))) * (0.94 + 0.12 * tile_noise2(pos.xz, 3.1, vec2(0.0))) *
                         mix(1.0, 0.72, smoothstep(wl + 0.25, wl - 0.15, pos.y));          // wet at the water's edge
            mate = mix(mate, sandc, sand);
        }
    }
    // Turf where the home tile's grass thins out and beyond (grass_raymarch.frag fades its blades
    // over the last 170 m of the tile): the two cross over, so there is no line where the lawn
    // stops and no bare band between it and the turf of the wider world.
    {
        vec2 tq = abs(pos.xz);
        float out_k = smoothstep(kHeightmapSide * 0.5 - 170.0, kHeightmapSide * 0.5 - 10.0, max(tq.x, tq.y));
        if (out_k > 0.0) {
            float wl = scene.water_params.y;
            float turf = out_k * smoothstep(wl + 1.2, wl + 2.6, pos.y) * (1.0 - smoothstep(52.0, 70.0, pos.y)) *
                         (1.0 - smoothstep(0.25, 0.60, 1.0 - nor.y));
            // The patches of green and brown: value noise read straight has edges along the world's
            // grid (ruled lines across the meadow, seen from a little way up). Read on a bent copy
            // of the ground, in three turned layers, it has none.
            vec2 tp = pos.xz + (vec2(tile_noise2(pos.xz, 0.0091, vec2(0.0)), tile_noise2(pos.xz, 0.0091, vec2(23.0))) - 0.5) * 70.0;
            // (turned by the angles of whole-sided triangles: the same patches in every tile - scene_ubo.glsl)
            float tn = tile_noise2_rot(tp, vec3(4.0, 3.0, 5.0), vec2(0.027), vec2(0.0)) * 0.45 +
                       tile_noise2_rot(tp, vec3(7.0, -24.0, 25.0), vec2(0.071), vec2(4.0)) * 0.33 +
                       tile_noise2_rot(tp, vec3(-5.0, 12.0, 13.0), vec2(0.23), vec2(9.0)) * 0.22;
            vec3 turfc = mix(vec3(0.150, 0.118, 0.066), vec3(0.070, 0.112, 0.036), smoothstep(0.30, 0.52, tn)) *
                         (1.0 + 0.40 * (tile_noise2(pos.xz, 1.9, vec2(0.0)) - 0.5) * (1.0 - smoothstep(120.0, 400.0, t))) *
                         (1.0 + 0.16 * (tile_noise2(pos.xz, 11.0, vec2(0.0)) - 0.5) * (1.0 - smoothstep(18.0, 60.0, t)));   // (grain only while a pixel resolves it: finer than a pixel it turned into ruled lines)
            mate = mix(mate, turfc, turf * 0.85);
        }
    }
    // The lake bed seen directly (camera under water): the same stones as
    // the bed seen through the surface from above (water branch, bed_h).
    float bed_sub = 0.0;
    if (scene.water_params.x > 0.5 && pos.y < scene.water_params.y - 0.02) {
        bed_sub = smoothstep(0.02, 0.12, scene.water_params.y - pos.y);
        float bfp = max(t, 0.05) * 2.0 * scene.viewport.w;      // ~metres per pixel
        float bwA = smoothstep(2.0, 5.0, 0.22 / bfp) * bed_sub;
        float bwB = smoothstep(2.0, 5.0, 0.075 / bfp);
        if (bwA > 0.01) {
            vec2  bp = pos.xz;
            float be = max(0.004, 1.5 * bfp), bid, bk, tmp1, tmp2;
            float bhR = bed_h(bp + vec2(be, 0.0), bwA, bwB, tmp1, tmp2);
            float bhU = bed_h(bp + vec2(0.0, be), bwA, bwB, tmp1, tmp2);
            float bhC = bed_h(bp, bwA, bwB, bid, bk);
            nor = normalize(nor + vec3(bhC - bhR, 0.0, bhC - bhU) / be * 0.9);
            float crev = smoothstep(0.0, 0.045, bhC) * smoothstep(0.0, 0.45, bk);
            float gl = max(dot(mate, vec3(0.3, 0.5, 0.2)), 0.03);
            vec3 sc = gl * mix(vec3(1.35, 1.15, 1.05), vec3(1.05, 1.0, 0.95), fract(bid * 5.3)) *
                      mix(0.55, 1.45, fract(bid * 23.1));
            sc = mix(sc, gl * vec3(1.75, 1.20, 1.05), step(0.70, fract(bid * 3.1)));     // pink-brown
            sc = mix(sc, gl * vec3(0.55, 0.55, 0.60), step(0.86, fract(bid * 9.7)));     // dark
            sc = mix(sc, gl * vec3(2.20, 2.10, 1.95), step(0.92, fract(bid * 17.9)));    // pale
            sc *= mix(0.85, 1.15, noise2(bp * 90.0 + bid * 40.0));
            mate = mix(mate, mix(mate * 0.45, sc, smoothstep(0.0, 0.2, bk)), bwA) * mix(1.0, mix(0.10, 1.0, crev), bwA);
        }
    }
    // Pebbles: each stone its own colour (greys, tans, a few dark and a few
    // pale ones), darker round its rim where it sits in the soil.
    float peb_cover = 0.0;
    if (t < 26.0) {
        float pw = (1.0 - smoothstep(10.0, 26.0, t)) * (1.0 - plateauWeight(pos.xz)) * (1.0 - bed_sub);   // the bed has its own stones
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
                float ms = tile_noise2(pos.xz, 11.0, vec2(id * 20.0)) * 0.6 + tile_noise2(pos.xz, 37.0, vec2(0.0)) * 0.4;
                pc0 = mix(pc0, gl * vec3(0.55, 1.05, 0.40), smoothstep(0.50, 0.72, ms) * smoothstep(0.35, 0.8, k) * step(0.35, fract(id * 2.7)));
                pc0 = mix(pc0, gl * vec3(1.9, 1.85, 1.5), smoothstep(0.80, 0.86, tile_noise2(pos.xz, 55.0, vec2(3.0))) * 0.7);   // lichen spots
            }
            pc0 *= mix(0.85, 1.15, tile_noise2(pos.xz, 140.0, vec2(id * 50.0)));                    // speckle
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
            float vn = tile_noise2(pos.xz, 0.9, vec2(0.0)) * 0.6 + tile_noise2(pos.xz, 3.7, vec2(0.0)) * 0.4;
            vec3 verge = vec3(0.105, 0.078, 0.052) * clamp(soil, vec3(0.45), vec3(1.8)) * mix(0.72, 1.25, vn);
            mate = mix(mate, verge, rm_road.y);
        }
        // Stone-to-stone colour variation: patches of lighter / darker and
        // slightly warmer / cooler stone, plus some sun-bleached wear.
        vec3 cob = textureLod(u_albedo[kRmMatRoad], rm_road_uv, rm_road_lod).rgb * kRmAlbedoScale;
        float cv1 = tile_noise2(pos.xz, 6.5, vec2(9.1)) * 0.6 + tile_noise2(pos.xz, 2.1, vec2(1.7)) * 0.4, cv2 = tile_noise2(pos.xz, 0.55, vec2(3.3));
        cob *= mix(0.74, 1.22, cv1) * mix(vec3(1.06, 1.0, 0.92), vec3(0.94, 0.98, 1.05), cv2);
        mate = mix(mate, cob, rm_road.x);
        // Sand lying on it: pale and fine-grained, darker and damper where it is thin; a print's
        // floor is packed (darker) and the sun leaves part of it in shade.
        if (sand_cov > 0.001) {
            float sv = tile_noise2(pos.xz, 3.1, vec2(40.0)) * 0.6 + tile_noise2(pos.xz, 23.0, vec2(0.0)) * 0.4;
            vec3 sand = vec3(0.265, 0.215, 0.145) * mix(0.84, 1.12, sv) * mix(0.80, 1.0, sand_cov);
            // The grains themselves: pale quartz and dark mineral specks, coarser and darker in
            // the ripples' troughs (where the heavy grains collect), finer and paler on the crests.
            {
                float gk1 = 1.0 - smoothstep(1.5, 7.0, t), gk2 = 1.0 - smoothstep(0.6, 3.0, t);
                float g1 = tile_noise2(pos.xz, 190.0, vec2(3.0)), g2 = tile_noise2(pos.xz, 610.0, vec2(11.0));
                sand *= 1.0 + (g1 - 0.5) * 0.30 * gk1 + (g2 - 0.5) * 0.34 * gk2;
                float dark = smoothstep(0.74, 0.80, tile_noise2(pos.xz, 260.0, vec2(71.0))) * gk1 * mix(1.0, 0.45, sand_rip);
                sand = mix(sand, sand * vec3(0.42, 0.40, 0.40), dark * 0.75);
                float pale = smoothstep(0.80, 0.86, tile_noise2(pos.xz, 330.0, vec2(19.0))) * gk1;
                sand = mix(sand, sand * 1.35 + 0.02, pale * 0.6);
                sand *= mix(0.86, 1.07, sand_rip);
            }
            float dC = t < 30.0 ? foot_depth(pos.xz) : 0.0;
            if (dC > 0.002) {
                float shade = 0.0;
                vec3 sd3 = scene.sun_direction.xyz;
                if (sd3.y > 0.02) {
                    vec2 sdir = normalize(sd3.xz + vec2(1e-5));
                    float rise = sd3.y / max(length(sd3.xz), 1e-3);
                    for (int k = 1; k <= 4; ++k) {
                        float x = 0.02 * float(k);
                        shade = max(shade, smoothstep(0.0, 0.008, (dC - foot_depth(pos.xz + sdir * x)) * 0.45 - x * rise));
                    }
                }
                sand *= mix(1.0, 0.80, smoothstep(0.0, 0.05, dC)) * mix(1.0, 0.55, shade);
            }
            mate = mix(mate, sand, sand_cov);
        }
    }

    // A highway's surface (hw_* from terrain_rm_main_ground.glsl; rm_hw_asphalt).
    if (hw_pave + hw_verge > 0.001) {
        vec3 grav = textureLod(u_albedo[kRmMatGravel], pos.xz / TILE_CELL(1.7), hw_lod + 0.6).rgb * 0.50;
        grav *= mix(0.78, 1.12, tile_noise2(pos.xz, 0.6, vec2(2.0))) * vec3(1.0, 0.97, 0.92);
        // (the land's own cover comes back through it toward its outer edge)
        mate = mix(mate, grav, hw_verge * (1.0 - 0.55 * smoothstep(7.0, 8.6, abs(hw_d)) * tile_noise2(pos.xz, 1.9, vec2(0.0))));
        if (hw_pave > 0.001) {
            vec3 tex = textureLod(u_albedo[kRmMatAsphalt], hw_uv, hw_lod).rgb * 0.50;
            float gl = 0.0;
            vec3 asph = rm_hw_asphalt(pos.xz, hw_d, hw_s, t, tex, hw_hgt, gl);
            mate = mix(mate, asph, hw_pave);
            hw_gloss = gl * hw_pave;
        }
    }
    // Winter: snow lies on the road in patches (thicker toward the verges),
    // and feet leave prints in the snow - pressed in, with depth: the print
    // is looked up where the line of sight meets its floor (parallax), its
    // walls tilt the normal and the low sun leaves part of the floor in shade.
    {
        float sn = g_snow;
        if (sn > 0.01 && !(scene.water_params.x > 0.5 && pos.y < scene.water_params.y)) {
            float road_w = clamp(rm_road.x, 0.0, 1.0);
            float snow_patch = snow_road_patch(pos.xz) * smoothstep(0.15, 0.6, sn);
            float cover = snow_cov;        // (set before the lighting, with the snow's own normal)
            if (cover > 0.01) {
                mate = mix(mate, vec3(0.90, 0.93, 0.97), cover);
                // Footprints: the print itself is in the marched surface; here
                // the pressed snow of its floor (denser, bluer) and the shade
                // the low sun leaves inside it.
                float dC = t < 30.0 ? foot_depth(pos.xz) : 0.0;
                if (dC > 0.002) {
                    float shade = 0.0;
                    vec3 sd3 = scene.sun_direction.xyz;
                    if (sd3.y > 0.02) {
                        vec2 sdir = normalize(sd3.xz + vec2(1e-5));
                        float rise = sd3.y / max(length(sd3.xz), 1e-3);
                        for (int k = 1; k <= 4; ++k) {
                            float x = 0.025 * float(k);
                            shade = max(shade, smoothstep(0.0, 0.012, (dC - foot_depth(pos.xz + sdir * x)) - x * rise));
                        }
                    }
                    float inside = smoothstep(0.0, 0.05, dC) * cover;
                    mate = mix(mate, mate * vec3(0.66, 0.76, 0.93), inside);
                    mate *= mix(1.0, 0.5, shade * cover);
                }
            }
        }
    }

