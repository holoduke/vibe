// Part of main() in terrain_raymarch.frag: the ground's material: road and path, the close-up detail, wetness.
// Not a file of its own - it is included in the middle of main() and uses its local
// variables (ro, rd, t, pos, nor, col, ...). Split off only to keep that file readable;
// the order of the parts is the order of the #include lines there.

#if RM_PART == 1
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
#ifndef VIBE_BAKE      /* (the bake marks those texels 'not to be used' instead: a discarded texel would keep what was there) */
    {
        const float kCastleHalfExtentEarly = 11.05;
        if (abs(pos.x) < kCastleHalfExtentEarly &&
            abs(pos.z) < kCastleHalfExtentEarly) {
            discard;
        }
    }
#endif
#ifdef VIBE_LITE
    {
        vec3 n_l = calcNormal(pos, t);
        float up_l = n_l.y, wl = scene.water_params.y;
        float v_l = tile_noise2(pos.xz, 0.06, vec2(0.0)) * 0.6 + tile_noise2(pos.xz, 0.9, vec2(0.0)) * 0.4;
        vec3 grass_l = mix(vec3(0.060, 0.105, 0.030), vec3(0.105, 0.135, 0.045), v_l);
        vec3 rock_l  = vec3(0.165, 0.155, 0.140) * mix(0.8, 1.15, v_l);
        vec3 alb = mix(rock_l, grass_l, smoothstep(0.52, 0.74, up_l));
        alb = mix(alb, vec3(0.27, 0.235, 0.165), (1.0 - smoothstep(wl + 0.3, wl + 1.6, pos.y)) * float(scene.water_params.x > 0.5));
        alb = mix(alb, vec3(0.80, 0.83, 0.88), smoothstep(0.55, 0.75, up_l) * max(g_snow, smoothstep(250.0, 330.0, pos.y + 40.0 * v_l)));
        vec2 ruv_l, rdir_l;
        vec2 road_l = g_far_chart ? vec2(0.0) : roadWeights(pos.xz, ruv_l, rdir_l);
        alb = mix(alb, vec3(0.115, 0.095, 0.070), road_l.y);
        alb = mix(alb, vec3(0.20, 0.20, 0.19) * mix(0.85, 1.1, tile_noise2(pos.xz, 5.0, vec2(0.0))), road_l.x);
        float dif_l = max(dot(n_l, scene.sun_direction.xyz), 0.0) * sun_horizon_lit(u_sun_horizon, pos);
        vec3 lin_l = dif_l * scene.sun_color.rgb * scene.sun_color.a + (0.45 + 0.35 * up_l) * scene.sky_color.rgb * 0.75;
        if (FLAT_GROUND) lin_l = flat_light();        // ("simplest shading")
        lite_out(pos, alb * lin_l);
        return;
    }
#endif
    // ---- The ground's material: worked out below, or read from the baked ground texture ----
    // (terrain_rm_bake.glsl. What the lighting takes from the material is declared here; the
    //  material's own working is inside the braces, which close at the end of the colour part
    //  in terrain_rm_main_light.glsl. See RM_PART in terrain_raymarch.frag for the order.)
    vec3  nor = vec3(0.0, 1.0, 0.0), mate = vec3(0.0), sunDir = scene.sun_direction.xyz;
    bool  lo_ok = false;
    vec2  lo_sh = vec2(0.0);
    float peb0_h = -1.0, rm_road_vis = 1.0, sand_cov = 0.0, hw_pave = 0.0, hw_gloss = 0.0;
    bool  rm_baked = false;
#if !defined(VIBE_WATER) && !defined(VIBE_BAKE) && !defined(VIBE_MARCH_ONLY) && !defined(VIBE_LITE)
    if (g_bake_t0 > 0.0) rm_baked = rm_bake_fetch(pos, t, nor, mate, hw_pave, hw_gloss, sand_cov);
#endif
    if (!rm_baked) {
    // Castle road: cobble normal map in road space, before any lighting.
    vec2 rm_road_uv, rm_road_dir;
    vec2 rm_road = roadWeights(pos.xz, rm_road_uv, rm_road_dir);
    if (rm_road.x + rm_road.y > 0.001 && rd.y < -0.0015 && !rm_far_chart()) {      // (home only: a site's road lies on formula ground)
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
        // Newton steps on height-above-road along the ray. The rate at
        // which the ray closes on the surface is measured (two height
        // samples), not assumed to be the ray's own fall: on a road that
        // climbs or drops the old "/ -rd.y" was far off - and had the wrong
        // sign when looking up a hill - which left bands of smeared, dark
        // paving wherever the slope changed.
        for (int i = 0; i < n_it; ++i) {
            vec3  ps = ro + t_s * rd;
            float f0 = ps.y - meshHeight(ps.xz);
            vec3  p1 = ps + rd * 0.3;
            float df = ((p1.y - meshHeight(p1.xz)) - f0) / 0.3;
            df = min(df, -0.01);                       // the ray must be closing on the ground
            t_s += clamp(-f0 / df, -lim, lim) * 0.9;
        }
        t_s = max(t_s, 0.05);
        // (not where snow lies on the paving: that surface is the marched one)
        // (a real branch: `a ? b : c` may evaluate both sides, and the winter side is not free)
        float road_snow = 0.0;
        if (g_snow >= 0.01) road_snow = snow_road_patch(pos.xz) * smoothstep(0.15, 0.6, g_snow);
        if (t < 30.0) road_snow = max(road_snow, sand_road_patch(pos.xz));      // (nor where sand lies on it)
        t = mix(t, t_s, clamp(rm_road.x + rm_road.y * 0.5, 0.0, 1.0) * (1.0 - smoothstep(0.02, 0.25, road_snow)));
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
    nor = calcNormal(pos, t);
    // The sun-shadow terms from the low-resolution pass (settings: 'sun shadows at low
    // resolution'): the four blocks round this pixel, each used only if its hit lies on this
    // pixel's own surface - its distance must be what a neighbouring ray would find on the
    // plane through this point. Where none qualifies (an edge, something in front, a ridge)
    // the pixel works its shadows out itself, exactly as before.
#if !defined(VIBE_WATER) && !defined(VIBE_MARCH_ONLY)
    if ((int(scene.rm_perf.x) & 5) == 5 && pc.emissive.x < 0.0 && !water_only && t > 0.0 && t < 1400.0) {
        float dv = scene.rm_perf.y;
        vec2  tc = gl_FragCoord.xy / dv - 0.5;
        ivec2 i0 = ivec2(floor(tc));
        vec2  f  = fract(tc);
        ivec2 mx = textureSize(u_shade_lo, 0) - 1;
        vec3  Np = g_nb_ok ? g_nrm_plain : nor;
        float cg  = max(abs(dot(rd, Np)), 0.04);
        // (+ the relief the low-resolution hit leaves out, along this ray)
        float tol = t * (1.6 * dv * 1.68 * scene.viewport.w / cg + 0.004) + 0.05 + 0.32 * (1.0 - smoothstep(10.0, 26.0, t)) / cg;
        float ws = 0.0;
        for (int k = 0; k < 4; ++k) {
            ivec2 o = ivec2(k & 1, k >> 1);
            vec4 s = texelFetch(u_shade_lo, clamp(i0 + o, ivec2(0), mx), 0);
            float w = ((o.x == 1) ? f.x : 1.0 - f.x) * ((o.y == 1) ? f.y : 1.0 - f.y);
            w *= (s.w > 0.5 && abs(s.z - t) < tol) ? 1.0 : 0.0;
            lo_sh += s.xy * w; ws += w;
        }
        if (ws > 0.2) { lo_sh /= ws; lo_ok = true; }
    }
#endif
    peb0_h = g_pebN_h;
    float peb0_id = g_pebN_id, peb0_k = g_pebN_k, peb0_sz = g_pebN_sz;
    g_rm_detail_mul = 1.0;
    if (rm_road.x > 0.001 && rm_far_chart()) {
        nor = normalize(mix(nor, vec3(0.0, 1.0, 0.0), rm_road.x));       // (a site's street: on its level ground)
    } else if (rm_road.x > 0.001) {
        // The paved surface is the smooth carved heightmap; the procedural
        // FBM lumps left over on top of it only read as noise on cobbles.
        const float e = 0.6;
        vec3 rn_s = normalize(vec3(meshHeight(pos.xz - vec2(e, 0.0)) - meshHeight(pos.xz + vec2(e, 0.0)), 2.0 * e,
                                   meshHeight(pos.xz - vec2(0.0, e)) - meshHeight(pos.xz + vec2(0.0, e))));
        nor = normalize(mix(nor, rn_s, rm_road.x));
    }
    float rm_road_lod = 0.0;
    float rm_cob_h = 0.0;              // the cobbles' height 0..1 at this pixel
    float rm_road_wx = rm_road.x;      // the share of cobbles in the colour (by height at the paving's edge)
    float rm_cob_sink = 1.0, rm_cob_fill = 0.0, rm_cob_moss = 0.0, rm_cob_lvl = 0.0;   // the old paving: see "Old, hand-laid paving" below
    float gnd_det_h = 0.0, gnd_rockw = 0.0;      // the soil / rock scan's height 0..1 here, and the share of rock
    if (rm_road.x > 0.001) {
        // Mip from the pixel footprint (no screen derivatives: road_uv is
        // discontinuous at the road edge and across the march).
        float fp = t * 2.0 / max(scene.viewport.y, 1.0);
        float g_tile = RM_ROAD_TILE;
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
        // (with the screen-space displacement on there is no parallax: the reference game's ground
        //  shader marches only when its displacement pass is off - GND_SSDM, below)
        if (GND_SSDM) pom_D = 0.0;
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
        vec4 rn4 = textureLod(u_normal[kRmMatRoad], rm_road_uv, rm_road_lod);
        vec3 rn = rn4.xyz * 2.0 - 1.0;
        rm_cob_h = rn4.a;
        // Old, hand-laid paving (close by; colour and height together - the colour's side is in
        // terrain_rm_main_light.glsl "Road"). The scan is one even sheet of stones; what makes
        // it old is generated here, from noise at the stones' own scale laid on the scan's height:
        //  * the stones have settled in groups - some stand, some have sunk (rm_cob_sink);
        //  * a stone is missing here and there, earth in the hole (the paving's share goes to 0
        //    there - by height, so the hole ends at the joints round it);
        //  * earth stands in the joints, higher toward the edges and where the bed has sunk, and
        //    moss grows on it (rm_cob_fill, rm_cob_moss);
        //  * the edge is ragged: the paving's share is pushed about stone by stone before the
        //    earth and the stones are laid over each other by height (rm_height_blend), so
        //    single stones stand in the earth and tongues of earth reach in between them.
        float cob_keep = rm_road.x;                       // the paving's share before the height blend
        if (t < 60.0) {
            float nearw = 1.0 - smoothstep(30.0, 60.0, t);
            float lat_k = abs(rm_road_uv.x) * g_tile / max(scene.road_info.y, 0.5);      // 0 the middle .. 1 the edge
            float edge_k = smoothstep(0.35, 0.95, lat_k);
            float sink_n = tile_noise2(pos.xz, 2.3, vec2(83.0)) * 0.6 + tile_noise2(pos.xz, 0.7, vec2(19.0)) * 0.4;
            rm_cob_sink = mix(1.0, mix(0.62, 1.10, smoothstep(0.22, 0.72, sink_n)), nearw);
            float hole = smoothstep(0.80, 0.87, tile_noise2(pos.xz, 3.1, vec2(47.0))) *
                         smoothstep(0.45, 0.70, tile_noise2(pos.xz, 0.23, vec2(5.0))) * nearw;
            float rag = (tile_noise2(pos.xz, 5.5, vec2(29.0)) - 0.5) * 1.3 * (rm_road.x * (1.0 - rm_road.x) * 4.0);
            cob_keep = clamp(rm_road.x + rag * nearw, 0.0, 1.0) * (1.0 - hole);
            float lvl_s = rm_cob_h * rm_cob_sink;
            float fill_lvl = clamp(0.34 * edge_k + (tile_noise2(pos.xz, 1.3, vec2(71.0)) - 0.5) * 0.36 + 0.25 * (1.0 - rm_cob_sink), 0.0, 0.46) * nearw;
            rm_cob_fill = (1.0 - smoothstep(fill_lvl - 0.10, fill_lvl + 0.02, lvl_s)) * step(0.02, fill_lvl);
            rm_cob_lvl = max(lvl_s, fill_lvl);
            rm_cob_moss = 0.75 * rm_cob_fill * smoothstep(0.42, 0.68, tile_noise2(pos.xz, 0.9, vec2(13.0))) * smoothstep(0.15, 0.6, edge_k);
            rm_road_wx = mix(rm_height_blend(0.50, lvl_s, cob_keep), cob_keep, smoothstep(30.0, 60.0, t));
        } else {
            rm_cob_lvl = rm_cob_h;
        }
        // (the stones' own relief only where stone shows: earth in a joint or a hole lies flat)
        nor = normalize(mix(nor, rT * rn.x + rB * rn.y + nor * rn.z, rm_road.x * mix(1.0, rm_road_wx / max(rm_road.x, 1e-3), 0.85) * (1.0 - 0.8 * rm_cob_fill)));
    }
    // ---- Close-up ground detail --------------------------------------
    // The procedural palette is flat colour; within ~45 m the photo-scanned
    // ground sets (soil on the flats, rock on the steeps) supply the fine
    // structure: a normal map for the lighting, the albedo as a tint
    // around its own mean (so the palette's colour is kept) and the
    // displacement as cavity darkening. Two scales - a ~2 m tile and a
    // ~0.5 m one that takes over right under the camera.
    vec3 rm_det_tint = vec3(1.0);
    // A steep face is ROCK: its colour is the scanned rock itself, laid on the wall (not the
    // flat palette with the scan as a tint - that palette is worked out from the ground plan,
    // and on a face that stands up it smeared into streaks and blotches of moss). Beds run
    // across it, level; it is darker in its clefts and paler where it weathers.
    float rm_cliff = 0.0;
    vec3  rm_cliff_alb = vec3(0.0);
    vec3  rm_cliff_grad = vec3(0.0);        // the breaker pattern's slope on the face (rock_breaker.glsl)
    {
        // The ground sets carry on into the distance. A photo tiled at one size reads as a grid
        // as soon as a dozen repeats are in view, and is gone (averaged out by the mips) a little
        // further - leaving bare ground flat. So the tile grows with distance: each band four
        // times the last (2.2 m, 9 m, 36 m, 142 m ...), and everywhere two neighbouring sizes
        // are laid over each other, sliding from the one to the next. No more than a handful of
        // repeats of either is ever in view, the two never line up, and the ground keeps
        // structure of about the same size on screen out to the end of the landscape.
        float dw = (1.0 - smoothstep(1300.0, 2600.0, t)) * (1.0 - rm_road.x);
        if (dw > 0.01 && !RMDBG(1)) {
            const int kSoil = 9, kRock = 7;
            float rockw = 1.0 - smoothstep(0.5, 0.8, nor.y);
            float fp = t * 2.0 * scene.viewport.w;                       // ~metres per pixel
            float bL = max(log2(t / 6.0) * 0.5, 0.0);                    // band: 0 to 6 m, +1 per four times the distance
            float bi = floor(bL), bw = smoothstep(0.0, 1.0, bL - bi);
            float sA = pow(0.25, bi);
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
                if (GND_SSDM) pk = 0.0;                                         // (the screen-space displacement does it, below)
                float lodP = log2(max(t * 2.0 * scene.viewport.w * 0.45 * float(textureSize(u_albedo[kSoil], 0).x), 1.0));
                for (int pi = 0; pi < 2; ++pi)
                    uvS = pos.xz * 0.45 + vts * ((textureLod(u_normal[kSoil], uvS, lodP).a - 0.5) * pk);
            }
            vec2 par = uvS - pos.xz * 0.45;                              // (the parallax shift, nearest band only)
            vec2 rk = abs(nor.x) > abs(nor.z) ? pos.zy : pos.xy;
            float wallw = smoothstep(0.35, 0.6, 1.0 - nor.y);            // (the two projections are blended as samples, not as coordinates)
            float tsS = float(textureSize(u_albedo[kSoil], 0).x), tsR = float(textureSize(u_albedo[kRock], 0).x);
            vec3 mS = max(textureLod(u_albedo[kSoil], vec2(0.5), 12.0).rgb, vec3(0.02));
            vec3 mR = max(textureLod(u_albedo[kRock], vec2(0.5), 12.0).rgb, vec3(0.02));
            vec4 nD = vec4(0.0);
            vec3 aD = vec3(0.0), aK = vec3(0.0);
            for (int b = 0; b < 2; ++b) {
                float sc = b == 0 ? sA : sA * 0.25, wb = b == 0 ? 1.0 - bw : bw;
                if (wb < 0.004) continue;
                vec4 nS = vec4(0.5, 0.5, 1.0, 0.5), nR = nS;
                vec3 aS = vec3(1.0), aR = vec3(1.0);
                vec2 bo = vec2(17.37, 9.13) * (bi + float(b));           // (each size shifted: no common origin)
                if (rockw < 0.99) {
                    rm_det_sample(kSoil, pos.xz * (0.45 * sc) + bo + (sc > 0.99 ? par : vec2(0.0)), log2(max(fp * 0.45 * sc * tsS, 1.0)), nS, aS);
                    aS /= mS;
                }
                if (rockw > 0.01) {
                    float lodR = log2(max(fp * 0.30 * sc * tsR, 1.0));
                    if (wallw < 0.98) rm_det_sample(kRock, pos.xz * (0.30 * sc) + bo, lodR, nR, aR);
                    if (wallw > 0.02) {
                        vec4 nW; vec3 aW;
                        rm_det_sample(kRock, rk * (0.30 * sc) + bo, lodR, nW, aW);
                        nR = mix(nR, nW, wallw); aR = mix(aR, aW, wallw);
                    }
                    aR /= mR;
                }
                aK += aR * wb;
                // (soil against rock by their heights: the rock's ribs come through the soil first)
                float rwb = (GND_SSDM && rockw > 0.01 && rockw < 0.99) ? rm_height_blend(nS.a, nR.a, rockw) : rockw;
                nD += mix(nS, nR, rwb) * wb;
                aD += mix(aS, aR, rwb) * wb;
            }
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
            {
                float steep = smoothstep(0.56, 0.78, 1.0 - nor.y);
                if (steep > 0.01) {
                    // beds: level bands a metre or two thick, their line wandering along the face
                    float along = dot(pos.xz, vec2(0.071, 0.053));
                    float bed = pos.y * 0.42 + 1.7 * tile_noise2(pos.xz, 0.021, vec2(3.0)) + 0.35 * sin(along);
                    float bands = mix(0.80, 1.14, smoothstep(0.25, 0.75, fract(bed))) * mix(0.90, 1.08, fract(bed * 3.1 + 0.37));
                    float patchy = mix(0.82, 1.16, tile_noise2(pos.xz + pos.y * 0.6, 0.045, vec2(9.0)));
                    vec3  tone = mix(vec3(0.150, 0.140, 0.128), vec3(0.205, 0.188, 0.165), tile_noise2(pos.xz - pos.y * 0.4, 0.011, vec2(1.0)));
                    rm_cliff_alb = tone * clamp(aK, vec3(0.45), vec3(1.9)) * bands * patchy;
                    rm_cliff = steep * dw;
                    // The breaker pattern and the colour strip by altitude, shared with the rock
                    // formations (rock_breaker.glsl): a formation and the cliff behind it carry the same beds.
                    {
                        float rk_bm;
                        vec3  rk_tone = rk_breaker(pos, nor, t, fp, rm_cliff_grad, rk_bm);
                        rm_cliff_alb = rk_gradient(pos, rm_cliff_alb * rk_tone, 1.0 - clamp(nor.y, 0.0, 1.0), scene.water_params.y);
                        rm_cliff_grad *= rm_cliff;
                    }
                }
            }
            vec3 dn = vec3((nD.xy * 2.0 - 1.0) * mix(1.35, 0.95, smoothstep(80.0, 600.0, t)), 1.0);     // (the big far tiles: a little flatter)
            vec3 dT = normalize(vec3(1.0, 0.0, 0.0) - nor * nor.x);
            vec3 dB = cross(nor, dT);
            nor = normalize(mix(nor, normalize(dT * dn.x + dB * dn.y + nor * dn.z), dw));
            nor = normalize(nor - rm_cliff_grad);
            float cav = mix(0.62, 1.10, smoothstep(0.15, 0.85, nD.a));  // pits darker, bumps a touch brighter
            gnd_det_h = clamp(nD.a, 0.0, 1.0) * dw; gnd_rockw = rockw;
            rm_det_tint = mix(vec3(1.0), clamp(mix(vec3(1.0), aD, 0.55), vec3(0.6), vec3(1.5)) * cav, dw);
        }
    }

    // Winter: a clean sheet of snow. Its normal comes from the snow surface
    // itself (the marched height, sampled a few cm apart) - none of the
    // soil's grit or the cobbles' relief shows through; only its own drifts,
    // lumps and the prints pressed into it shade it. Before the lighting.
    // Sand on the road: the same - the normal of the sand's own surface (its drifts, ripples and
    // the prints in it), no cobble relief showing through.
    float sand_rip = 0.5;
    if (rm_road.x + rm_road.y > 0.001) {
        float sp = sand_road_patch(pos.xz) * clamp(rm_road.x + rm_road.y, 0.0, 1.0);
        if (sp > 0.005) {
            // Where it thins out it lies in the joints first and the stones' tops stand clear of
            // it: the edge follows the cobbles, grain by grain, instead of fading over them.
            float cob_h = rm_road.x > 0.001 ? textureLod(u_normal[kRmMatRoad], rm_road_uv, rm_road_lod).a * rm_road.x : 0.0;
            float edge_n = (tile_noise2(pos.xz, 38.0, vec2(3.0)) - 0.5) * 0.05 * (1.0 - smoothstep(4.0, 14.0, t));
            sand_cov = smoothstep(0.0, mix(0.05, 0.30, smoothstep(6.0, 25.0, t)), sp - 0.015 - 0.20 * cob_h * (1.0 - smoothstep(10.0, 30.0, t)) + edge_n);
            vec3 n_s = nor;
            if (t < 40.0) {
                float se = max(0.006, t * 0.003), stpl = plateauWeight(pos.xz);
                if (g_nb_ok) {
                    n_s = g_nrm_plain;
                } else {
                    vec4 h4 = rm_h4_cross(pos.xz, se, t, stpl);      // (terrain_rm_heightfield.glsl: one copy of the height in a loop)
                    float hx0 = h4.x, hx1 = h4.y;
                    float hz0 = h4.z, hz1 = h4.w;
                    n_s = normalize(vec3(hx0 - hx1, 2.0 * se, hz0 - hz1));
                }
            }
            // the fine wind ripples (their slope from two more samples), where the sand has some depth
            float rip_f = (1.0 - smoothstep(6.0, 18.0, t)) * smoothstep(0.25, 0.7, sp) * smoothstep(0.30, 0.55, tile_noise2(pos.xz, 0.9, vec2(31.0)));
            if (rip_f > 0.01) {
                float wn = tile_noise2(pos.xz, 0.6, vec2(3.0));
                const float re = 0.008;
                sand_rip = sand_fine_ripple(pos.xz, wn);
                float rx = sand_fine_ripple(pos.xz + vec2(re, 0.0), wn), rz = sand_fine_ripple(pos.xz + vec2(0.0, re), wn);
                n_s = normalize(n_s + vec3(sand_rip - rx, 0.0, sand_rip - rz) * (0.0022 / re) * rip_f);      // (gentle: at 0.0045 the ripples were hard stripes)
                sand_rip = mix(0.5, sand_rip, rip_f);
            }
            // grains: three sizes, each gone before it is finer than a pixel
            vec2 gq = pos.xz * 70.0;
            n_s = normalize(n_s + vec3(tile_noise2(pos.xz, 70.0, vec2(0.0)) - 0.5, 0.0, tile_noise2(pos.xz, 70.0, vec2(17.0)) - 0.5) * 0.20 * (1.0 - smoothstep(3.0, 14.0, t))
                                + vec3(tile_noise2(pos.xz, 70.0 * 3.7, vec2(5.0)) - 0.5, 0.0, tile_noise2(pos.xz, 70.0 * 3.7, vec2(29.0)) - 0.5) * 0.22 * (1.0 - smoothstep(1.2, 6.0, t))
                                + vec3(tile_noise2(pos.xz, 70.0 * 11.0, vec2(9.0)) - 0.5, 0.0, tile_noise2(pos.xz, 70.0 * 11.0, vec2(41.0)) - 0.5) * 0.18 * (1.0 - smoothstep(0.5, 2.5, t)));
            nor = normalize(mix(nor, n_s, sand_cov));
        }
    }
    // A highway (engine/highway.h): asphalt to about 5.5 m from the centre line, gravel under
    // its edge and out to 8 m. The surface's relief comes from the scans' normal maps.
    float hw_verge = 0.0, hw_d = 0.0, hw_s = 0.0, hw_lod = 0.0, hw_hgt = 0.5, hw_wk = 1.0, hw_ad = 64.0;
    vec2  hw_mk = vec2(1.0, 0.0);       // a junction's paint (rm_hw_jn_surface)
    RmAt  hw_at, gv_at;                 // the asphalt's and the gravel's layouts here (rm_at: anti-tiling)
    hw_at.n = 0; gv_at.n = 0;
    vec2  hw_uv = vec2(0.0);
    if (rm_hw_in(pos.xz) && !RMDBG(4194304)) {
        hw_d = rm_hw_dh(pos.xz).x;
        hw_ad = rm_hw_ad(pos.xz, hw_d);         // (at a junction: the two roads' paving and the corners between them)
        float ad = hw_ad, aa = max(t * 0.002, 0.015);
        // The asphalt's edge is not a ruled line: it wanders a hand's width and has broken away in places.
        float edge = 5.5 + (tile_noise2(pos.xz, 0.45, vec2(0.0)) - 0.5) * 0.30 + (tile_noise2(pos.xz, 3.1, vec2(0.0)) - 0.5) * 0.12 * (1.0 - smoothstep(30.0, 80.0, t))
                         - 0.35 * smoothstep(0.72, 0.9, tile_noise2(pos.xz, 0.17, vec2(5.0)));
        hw_pave = 1.0 - smoothstep(edge - aa, edge + aa, ad);
        hw_verge = 1.0 - smoothstep(6.8, 8.6, ad + (tile_noise2(pos.xz, 0.9, vec2(0.0)) - 0.5) * 1.6);
        if (hw_pave + hw_verge > 0.001) {
            hw_s = rm_hw_s(pos.xz);
            hw_wk = rm_hw_width(pos.xz);
            rm_hw_jn_surface(pos.xz, t, hw_d, hw_s, hw_wk, hw_mk);      // (at a junction: whose asphalt this is)
            float fp = t * 2.0 / max(scene.viewport.y, 1.0);
            hw_uv = pos.xz / TILE_CELL(2.6);
            hw_lod = log2(max(fp / TILE_CELL(2.6) * float(textureSize(u_albedo[kRmMatAsphalt], 0).x), 1.0));
            hw_at = rm_at(hw_uv, hw_lod);
            gv_at = rm_at(pos.xz / TILE_CELL(1.7), hw_lod + 0.6);
            vec4 na = rm_at_normal(kRmMatAsphalt, hw_at, hw_lod);
            vec4 ng = rm_at_normal(kRmMatGravel, gv_at, hw_lod + 0.6);
            vec3 rn = mix(ng.xyz, na.xyz, hw_pave) * 2.0 - 1.0;
            // Close by the asphalt's edge is a broken edge, not a line: over a hand's width the sheet
            // and the gravel are laid over each other by height - the gravel's stones show where they
            // stand higher than the thinning sheet (rm_height_blend).
            if (t < 45.0) {
                float soft = 1.0 - smoothstep(edge - 0.07, edge + 0.07, ad);
                hw_pave = mix(rm_height_blend(ng.a, 0.55 + 0.35 * na.a, soft), hw_pave, smoothstep(20.0, 45.0, t));
            }
            hw_hgt = mix(ng.a, na.a, hw_pave);
            // (the ground plan is the texture's plane: x east, y north)
            float k = mix(1.5, 1.0, hw_pave);
            vec3 base = normalize(mix(nor, vec3(0.0, 1.0, 0.0), 0.85 * hw_pave));
            nor = normalize(mix(nor, normalize(base + vec3(rn.x, 0.0, -rn.y) * k), max(hw_pave, hw_verge)));
        }
    }
    // Beaches (the palette's sand by the water): the wind's fine ripples, as on the sand that
    // lies on the road - in the normal here, and as height for the displacement below.
    float beach_w = 0.0, beach_rip = 0.5;
    if (t < 24.0 && !water_only && nor.y > 0.5 && pos.y < 2.5 && g_snow < 0.3) {
        beach_w = smoothstep(2.5, 1.0, pos.y) * smoothstep(0.5, 0.9, nor.y) * (1.0 - clamp(rm_road.x + rm_road.y, 0.0, 1.0));
        float rip_f = (1.0 - smoothstep(6.0, 18.0, t)) * beach_w * smoothstep(0.30, 0.55, tile_noise2(pos.xz, 0.9, vec2(31.0)));
        if (rip_f > 0.01) {
            float wn = tile_noise2(pos.xz, 0.6, vec2(3.0));
            const float re = 0.008;
            float r0 = sand_fine_ripple(pos.xz, wn);
            float rx = sand_fine_ripple(pos.xz + vec2(re, 0.0), wn), rz = sand_fine_ripple(pos.xz + vec2(0.0, re), wn);
            nor = normalize(nor + vec3(r0 - rx, 0.0, r0 - rz) * (0.0022 / re) * rip_f);
            beach_rip = mix(0.5, r0, rip_f);
        }
    }
    // ---- How far this pixel's material stands out of the surface the patches / the march give ----
    // Written to the G-buffer by main(); shaders/ssdm_ground.frag and the screen-space
    // displacement passes move the pixel by it. Metres, never negative: the ground only rises.
    // Every ground material has a height, and where two meet the higher one shows (the
    // reference game's rule). Here: all that lies under the snow; the snow, which fills the
    // low places of this first, follows.
    float gnd_h = 0.0, gnd_root = 0.0, gnd_lawn = 0.0;
    bool  gnd_h_on = t < kSsdmReach && !water_only;
    if (gnd_h_on) {
        gnd_lawn = grassEligibility(pos.xz, pos.y);
        // (not under a lawn: the blades stand on the surface and hide it)
        float bare = 1.0 - smoothstep(0.30, 0.65, gnd_lawn);
        // the scan of soil / rock: what stands above its middle - stones, clods, ribs
        float h = mix(0.07, 0.09, gnd_rockw) * max(gnd_det_h - 0.4, 0.0) / 0.6;
        // Bare earth has more on it than the scan carries: clods a hand across and crumbs,
        // most where the scan is flat; under trees, roots running across the forest floor.
        if (bare > 0.01 && t < 14.0 && rm_road.x < 0.99) {
            float cw = (1.0 - smoothstep(8.0, 14.0, t)) * (1.0 - gnd_rockw);
            float clod = smoothstep(0.52, 0.80, tile_noise2(pos.xz, 7.0, vec2(55.0))) * 0.022 +
                         smoothstep(0.60, 0.85, tile_noise2(pos.xz, 19.0, vec2(23.0))) * 0.010;
            h += clod * cw * (1.0 - 0.6 * smoothstep(0.4, 0.8, gnd_det_h));
            vec2 vuv = pos.xz / 2048.0 + 0.5;
            if (!g_far_chart && all(greaterThan(vuv, vec2(0.0))) && all(lessThan(vuv, vec2(1.0)))) {
                float trees = textureLod(u_albedo[22], vuv, 1.0).r;
                if (trees > 0.05) {
                    float r1 = abs(tile_noise2(pos.xz, 0.9, vec2(37.0)) - 0.5), r2 = abs(tile_noise2(pos.xz, 1.7, vec2(91.0)) - 0.5);
                    gnd_root = max(1.0 - smoothstep(0.0, 0.030, r1), 0.7 * (1.0 - smoothstep(0.0, 0.024, r2))) *
                               smoothstep(0.05, 0.35, trees) * cw * bare;
                    h = max(h, 0.045 * gnd_root);
                }
            }
        }
        h *= bare;
        // Where turf meets bare ground the sod's edge stands a few centimetres proud (a rim: deep
        // in the lawn the height is 0 again, the blades stand on the surface as it is).
        h += 0.03 * smoothstep(0.28, 0.42, gnd_lawn) * (1.0 - smoothstep(0.55, 0.9, gnd_lawn)) * (1.0 - gnd_rockw) * (1.0 - clamp(rm_road.x + rm_road.y, 0.0, 1.0));
        // The loose stones (terrain_rm_pebbles.glsl; the normal looked them up): with the march
        // they were part of the marched surface, with the patches they are raised here.
        if (peb0_h > 0.0) h = max(h, min(peb0_h, 0.10) * (1.0 - smoothstep(10.0, 20.0, t)) * (1.0 - plateauWeight(pos.xz)));
        // The paved road, worn: the stones stand 6.5 cm out of their joints; along the two wheel
        // lines (a cart's, 1.5 m apart) and where the bed has settled they are pressed down - the
        // road's height is a FACTOR on what is there, as the reference game's road decal does it
        // (it multiplies the ground's height, so a track can only press the ground down). The
        // trodden earth beside the paving the same. Earth in the joints and in a missing stone's
        // hole, sunk stones: rm_cob_lvl, "Old, hand-laid paving" above.
        if (rm_road.x + rm_road.y > 0.001) {
            float lat = abs(rm_road_uv.x) * RM_ROAD_TILE;
            float line = 1.0 - smoothstep(0.16, 0.46, abs(lat - min(0.75, 0.5 * scene.road_info.y)));
            line *= mix(0.45, 1.0, tile_noise2(pos.xz, 0.11, vec2(61.0)));                   // (they come and go)
            float settled = smoothstep(0.56, 0.80, tile_noise2(pos.xz, 0.55, vec2(77.0)));
            float press = (1.0 - 0.45 * line) * (1.0 - 0.35 * settled);
            h *= 1.0 - 0.55 * rm_road.y;
            h = mix(max(h, 0.020 * rm_road.x), 0.065 * rm_cob_lvl * press, rm_road_wx);     // (the castle road's cobbles: 6.5 cm from joint to top - 16 stretched them along the view, 9 was still a little much)
        }
        // Sand: a sheet a few centimetres thick with its ripples (the ones the normal is made
        // from), its lumps, and the prints pressed into it. Stones that stand higher than it
        // show through it (sand_cov already follows the cobbles' height on the road).
        if (sand_cov > 0.001 || beach_w > 0.01) {
            float lump = 0.012 * tile_noise2(pos.xz, 4.3, vec2(11.0)) + 0.006 * tile_noise2(pos.xz, 17.0, vec2(3.0));
            float fd = foot_depth(pos.xz);
            if (sand_cov > 0.001) h = mix(h, max(0.034 + 0.006 * (sand_rip - 0.5) + lump - 0.45 * fd, 0.0), sand_cov);
            if (beach_w > 0.01)   h = mix(h, max(0.030 + 0.006 * (beach_rip - 0.5) + lump - 0.45 * fd, 0.0), rm_height_blend(clamp(h / 0.08, 0.0, 1.0), 0.40, beach_w));
        }
        if (hw_pave + hw_verge > 0.001) h = mix(h, mix(0.03 * hw_hgt, 0.034 + 0.004 * hw_hgt, hw_pave), max(hw_pave, hw_verge));   // roadside gravel 3 cm; the asphalt a sheet whose top is 3.4 cm up, with its grain - so its edge is a step over the gravel (its cracks, ruts and patches: rm_hw_asphalt's `lvl`, terrain_rm_main_light.glsl)
        if (scene.water_params.x > 0.5 && pos.y < scene.water_params.y + 0.03) h = 0.0;
        gnd_h = h;
    }
    float snow_h = 0.0;                // the snow's own surface, as height above the same base
    float snow_cov = 0.0;
    {
        // (no snow lies in the wet band: snow x (1 - w'), the reference's rule - g_snow itself, so
        //  the material's snow colour, getMaterial, follows)
#ifndef VIBE_SUMMER
        if (SW_ON(64) && g_snow > 0.01) { float swl, swe; g_snow *= 1.0 - sw_terrain_band(pos, swl, swe); }
#endif
        float sn = g_snow;
        if (sn > 0.01 && !(scene.water_params.x > 0.5 && pos.y < scene.water_params.y)) {
            float road_w = clamp(rm_road.x, 0.0, 1.0);
            snow_cov = mix(smoothstep(0.1, 0.5, sn), snow_road_patch(pos.xz) * smoothstep(0.15, 0.6, sn), road_w);
            if (snow_cov > 0.01) {
                vec3 n_s = nor;
                if (t < 70.0) {
                    float se = max(0.02, t * 0.004), stpl = plateauWeight(pos.xz);
                    if (g_nb_ok) {
                        n_s = g_nrm_plain;
                    } else {
                        vec4 h4 = rm_h4_cross(pos.xz, se, t, stpl);      // (terrain_rm_heightfield.glsl: one copy of the height in a loop)
                    float hx0 = h4.x, hx1 = h4.y;
                        float hz0 = h4.z, hz1 = h4.w;
                        n_s = normalize(vec3(hx0 - hx1, 2.0 * se, hz0 - hz1));
                    }
                }
                snow_cov *= smoothstep(0.35, 0.7, n_s.y);
                // Close by the snow lies in the low places of what is under it first - the joints
                // of the paving, between stones and clods - and covers the tops last as the cover
                // grows (the reference game's rule): its surface is a level that rises with the
                // cover, with the snow's own lumps on it (the third term of snow_layer_h, which
                // the normal above is made from) and the prints pressed in.
                if (gnd_h_on) {
                    float lump = 0.030 * g_snow * noise2(pos.xz * 2.6 + 3.0) * smoothstep(0.0, 0.4, snow_cov);
                    float lvl = snow_cov * 0.11 + lump;
                    float in_low = smoothstep(-0.004, 0.008, lvl - gnd_h) * smoothstep(0.0, 0.08, snow_cov);
                    // (a full sheet needs no level of its own - only its lumps and a floor for the
                    //  prints to be pressed into: lifting every pixel of a white field by a hand's
                    //  width bought nothing and smeared at the screen's edge)
                    snow_h = max(mix(lvl, 0.05 + lump, smoothstep(0.6, 1.0, snow_cov)) - 0.8 * foot_depth(pos.xz), 0.0);
                    snow_cov = mix(in_low, snow_cov, smoothstep(14.0, 24.0, t));
                }
                // Fine grain of the snow surface (it is never a mirror-smooth sheet).
                vec2 gq = pos.xz * 38.0;
                n_s = normalize(n_s + vec3(tile_noise2(pos.xz, 38.0, vec2(0.0)) - 0.5, 0.0, tile_noise2(pos.xz, 38.0, vec2(17.0)) - 0.5) * 0.10 * (1.0 - smoothstep(4.0, 22.0, t)));
                // Tracks (post_fx.cpp): the snow's normal follows the ruts and their rims - the
                // displacement alone barely showed on a white sheet (its normal came from the
                // landscape's height, which knows nothing of them).
                if (t < 40.0 && pfx_trk.d.trk_on > 0.5) {
                    vec2 tr0 = track_dr(pos.xz);
                    if (tr0.x + tr0.y > 1e-4) {
                        const float te = 0.05;
                        float fx0 = foot_depth(pos.xz - vec2(te, 0.0)), fx1 = foot_depth(pos.xz + vec2(te, 0.0));
                        float fz0 = foot_depth(pos.xz - vec2(0.0, te)), fz1 = foot_depth(pos.xz + vec2(0.0, te));
                        n_s = normalize(n_s + vec3(fx1 - fx0, 0.0, fz1 - fz0) * (0.5 / te));
                    }
                }
                nor = normalize(mix(nor, n_s, snow_cov));
            }
        }
    }
    // ---- ... and with the snow on it: the height the displacement pass gets ----
    if (GND_SSDM && gnd_h_on) {
        float h = mix(gnd_h, snow_h, snow_cov);
        // (never quite 0 within reach: every ground pixel there is on the displacement's grid,
        //  moved or not - a grid with holes where the height happens to be 0 came out stippled)
        // (no fade here: the displacement pass fades it over the last quarter of its reach, as the
        //  reference game's does - its terrain shader writes the material's height as it is)
        g_gnd_ssdm_h = max(min(h, kGndSsdmMax), 0.006 * kGndSsdmMax);
    }
#endif /* RM_PART == 1 */
#if RM_PART == 3
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
#endif /* RM_PART == 3 */
