// Part of main() in terrain_raymarch.frag: the ground's material: road and path, the close-up detail, wetness.
// Not a file of its own - it is included in the middle of main() and uses its local
// variables (ro, rd, t, pos, nor, col, ...). Split off only to keep that file readable;
// the order of the parts is the order of the #include lines there.

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
        lite_out(pos, alb * lin_l);
        return;
    }
#endif
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
    vec3 nor = calcNormal(pos, t);
    // The sun-shadow terms from the low-resolution pass (settings: 'sun shadows at low
    // resolution'): the four blocks round this pixel, each used only if its hit lies on this
    // pixel's own surface - its distance must be what a neighbouring ray would find on the
    // plane through this point. Where none qualifies (an edge, something in front, a ridge)
    // the pixel works its shadows out itself, exactly as before.
    bool lo_ok = false;
    vec2 lo_sh = vec2(0.0);
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
    float peb0_h = g_pebN_h, peb0_id = g_pebN_id, peb0_k = g_pebN_k, peb0_sz = g_pebN_sz;
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
            vec3 aD = vec3(0.0);
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
                nD += mix(nS, nR, rockw) * wb;
                aD += mix(aS, aR, rockw) * wb;
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
            vec3 dn = vec3((nD.xy * 2.0 - 1.0) * mix(1.35, 0.95, smoothstep(80.0, 600.0, t)), 1.0);     // (the big far tiles: a little flatter)
            vec3 dT = normalize(vec3(1.0, 0.0, 0.0) - nor * nor.x);
            vec3 dB = cross(nor, dT);
            nor = normalize(mix(nor, normalize(dT * dn.x + dB * dn.y + nor * dn.z), dw));
            float cav = mix(0.62, 1.10, smoothstep(0.15, 0.85, nD.a));  // pits darker, bumps a touch brighter
            rm_det_tint = mix(vec3(1.0), clamp(mix(vec3(1.0), aD, 0.55), vec3(0.6), vec3(1.5)) * cav, dw);
        }
    }
    vec3 sunDir = scene.sun_direction.xyz;

    // Winter: a clean sheet of snow. Its normal comes from the snow surface
    // itself (the marched height, sampled a few cm apart) - none of the
    // soil's grit or the cobbles' relief shows through; only its own drifts,
    // lumps and the prints pressed into it shade it. Before the lighting.
    // Sand on the road: the same - the normal of the sand's own surface (its drifts, ripples and
    // the prints in it), no cobble relief showing through.
    float sand_cov = 0.0, sand_rip = 0.5;
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
                    float hx0 = terrainH_lod_pw(pos.xz - vec2(se, 0.0), t, stpl), hx1 = terrainH_lod_pw(pos.xz + vec2(se, 0.0), t, stpl);
                    float hz0 = terrainH_lod_pw(pos.xz - vec2(0.0, se), t, stpl), hz1 = terrainH_lod_pw(pos.xz + vec2(0.0, se), t, stpl);
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
                n_s = normalize(n_s + vec3(sand_rip - rx, 0.0, sand_rip - rz) * (0.0045 / re) * rip_f);
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
    float hw_pave = 0.0, hw_verge = 0.0, hw_d = 0.0, hw_s = 0.0, hw_lod = 0.0, hw_hgt = 0.5, hw_gloss = 0.0;
    vec2  hw_uv = vec2(0.0);
    if (rm_hw_in(pos.xz)) {
        hw_d = rm_hw_dh(pos.xz).x;
        float ad = abs(hw_d), aa = max(t * 0.002, 0.015);
        // The asphalt's edge is not a ruled line: it wanders a hand's width and has broken away in places.
        float edge = 5.5 + (tile_noise2(pos.xz, 0.45, vec2(0.0)) - 0.5) * 0.30 + (tile_noise2(pos.xz, 3.1, vec2(0.0)) - 0.5) * 0.12 * (1.0 - smoothstep(30.0, 80.0, t))
                         - 0.35 * smoothstep(0.72, 0.9, tile_noise2(pos.xz, 0.17, vec2(5.0)));
        hw_pave = 1.0 - smoothstep(edge - aa, edge + aa, ad);
        hw_verge = 1.0 - smoothstep(6.8, 8.6, ad + (tile_noise2(pos.xz, 0.9, vec2(0.0)) - 0.5) * 1.6);
        if (hw_pave + hw_verge > 0.001) {
            hw_s = rm_hw_s(pos.xz);
            float fp = t * 2.0 / max(scene.viewport.y, 1.0);
            hw_uv = pos.xz / TILE_CELL(2.6);
            hw_lod = log2(max(fp / TILE_CELL(2.6) * float(textureSize(u_albedo[kRmMatAsphalt], 0).x), 1.0));
            vec4 na = textureLod(u_normal[kRmMatAsphalt], hw_uv, hw_lod);
            vec4 ng = textureLod(u_normal[kRmMatGravel], pos.xz / TILE_CELL(1.7), hw_lod + 0.6);
            vec3 rn = mix(ng.xyz, na.xyz, hw_pave) * 2.0 - 1.0;
            hw_hgt = mix(ng.a, na.a, hw_pave);
            // (the ground plan is the texture's plane: x east, y north)
            float k = mix(1.5, 1.0, hw_pave);
            vec3 base = normalize(mix(nor, vec3(0.0, 1.0, 0.0), 0.85 * hw_pave));
            nor = normalize(mix(nor, normalize(base + vec3(rn.x, 0.0, -rn.y) * k), max(hw_pave, hw_verge)));
        }
    }
    float snow_cov = 0.0;
    {
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
                        float hx0 = terrainH_lod_pw(pos.xz - vec2(se, 0.0), t, stpl), hx1 = terrainH_lod_pw(pos.xz + vec2(se, 0.0), t, stpl);
                        float hz0 = terrainH_lod_pw(pos.xz - vec2(0.0, se), t, stpl), hz1 = terrainH_lod_pw(pos.xz + vec2(0.0, se), t, stpl);
                        n_s = normalize(vec3(hx0 - hx1, 2.0 * se, hz0 - hz1));
                    }
                }
                snow_cov *= smoothstep(0.35, 0.7, n_s.y);
                // Fine grain of the snow surface (it is never a mirror-smooth sheet).
                vec2 gq = pos.xz * 38.0;
                n_s = normalize(n_s + vec3(tile_noise2(pos.xz, 38.0, vec2(0.0)) - 0.5, 0.0, tile_noise2(pos.xz, 38.0, vec2(17.0)) - 0.5) * 0.10 * (1.0 - smoothstep(4.0, 22.0, t)));
                nor = normalize(mix(nor, n_s, snow_cov));
            }
        }
    }
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
