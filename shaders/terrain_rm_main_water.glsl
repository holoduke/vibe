// Part of main() in terrain_raymarch.frag: the water surface: waves, refraction, reflection, foam, the bed seen through it.
// Not a file of its own - it is included in the middle of main() and uses its local
// variables (ro, rd, t, pos, nor, col, ...). Split off only to keep that file readable;
// the order of the parts is the order of the #include lines there.

    // === Water surface shading ===========================================
    if (t_water > 0.0) {
#ifdef VIBE_LITE
        {
            vec3 wp = ro + rd * t_water;
            float fres = 0.03 + 0.97 * pow(1.0 - clamp(-rd.y, 0.0, 1.0), 5.0);
            vec3 sky_l = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
            vec3 deep_l = scene.water_color.rgb * (scene.sun_color.rgb * scene.sun_color.a * max(scene.sun_direction.y, 0.0) * 0.6 + scene.sky_color.rgb * 0.5);
            lite_out(wp, mix(deep_l, sky_l, fres) * mix(0.96, 1.04, noise2(wp.xz * 0.7 + scene.water_params.w * 0.4)));
            return;
        }
#endif
#ifndef VIBE_WATER
        vec3 wpos = ro + rd * t_water;
#else /* VIBE_WATER */
        // Hardware depth-test (depth_test ON in setup.cpp) handles
        // "is terrain in front?" occlusion. wpos comes straight from
        // vWPos — perspective-correct world position from the plane
        // mesh vertices, no per-frame inv_vp drift, no need to snap.
        vec3 wpos = vWPos;
#endif /* VIBE_WATER */
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
        if (water_style == 1) {   // (2, the lake style, has its own branch below)
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
        // The open sea (beyond home's lakes): swell. Five trains of waves from the
        // sea's quarter, 70 m down to 3 m long, each running at its own deep-water
        // speed; a train fades out where a pixel no longer resolves it, and a slow
        // noise breaks the ranks so the crests are not ruled lines.
        float rm_sea_crest = 0.0;      // the sea's wave height here, about its mean (m)
        {
            float sea_k = smoothstep(1350.0, 1750.0, rm_home_dist(wpos.xz));
            // Swell belongs to the sea. A pond inland has none: on a small lake the long
            // wave trains read as a ruled cross-hatch in the sun's glitter.
            if (sea_k > 0.0) sea_k *= 1.0 - smoothstep(0.0, 0.02, rm_continent(wpos.xz));
            if (sea_k > 0.0) {
                const float kAng[7] = float[7](2.83, 3.71, 3.19, 2.27, 4.02, 2.51, 3.47);          // where each train runs to (rad; the sea lies east)
                const float kLen[7] = float[7](83.0, 47.0, 26.0, 14.5, 8.1, 4.6, 2.7);
                const float kSlope[7] = float[7](0.070, 0.070, 0.065, 0.058, 0.050, 0.042, 0.036);
                // (along a shallow line of sight a pixel covers 1 / |rd.y| times more water: without that
                // the fine wave trains were kept far too long and shimmered as moire from the air)
                float foot = max(t_water, 1.0) * 0.0025 / max(abs(rd.y), 0.12);
                // Near and middle distance: the Seascape wave field (sea_map above). Its slope
                // is taken over about a pixel, so it smooths itself out with distance.
                // Far out, where a pixel is wider than its longest wave, the long swell
                // trains below take over (they keep the sea from going dead flat from the air).
                float sea_far = smoothstep(0.25, 0.7, foot / 19.6);
                // Seen from a height the wave field is a lattice: every crest the same, in ranks.
                // A real sea is not even. (1) The whole field is bent, slowly (30 m over a
                // hundred), so the ranks wander. (2) The wind does not blow evenly: the waves
                // stand higher in gusts a few hundred metres across and lie down in long
                // slicks drawn out along the wind - which is what the eye reads from the air.
                vec2 sea_p = wpos.xz;
                float sea_amp = 1.0;
                if (t_water > 60.0) {
                    float far_k = smoothstep(60.0, 400.0, t_water);
                    sea_p += (vec2(noise2(wpos.xz * 0.011), noise2(wpos.xz * 0.011 + 7.3)) - 0.5) * (30.0 * far_k);
                    vec2 wq = vec2(0.86 * wpos.x + 0.5 * wpos.z, -0.5 * wpos.x + 0.86 * wpos.z);       // along / across the wind
                    float gust  = noise2(wpos.xz * 0.0031 + 3.0) * 0.6 + noise2(wpos.xz * 0.0011 + 11.0) * 0.4;
                    float slick = noise2(wq * vec2(0.0016, 0.0095) + 21.0) * 0.65 + noise2(wq * vec2(0.0050, 0.031) + 5.0) * 0.35;
                    sea_amp = mix(1.0, mix(0.50, 1.35, smoothstep(0.25, 0.75, gust)) * mix(0.30, 1.0, smoothstep(0.34, 0.52, slick)), far_k);
                }
                vec2 g_sea = vec2(0.0);
                if (sea_far < 1.0) {
                    float tm = 1.0 + wave_t * 0.8;
                    float e  = max(foot, 0.04);
                    sea_p -= scene.sea_org.xy;      // (from the local origin: sea_map, scene_ubo.glsl sea_org)
                    float h0 = sea_map(sea_p, tm, foot);
                    g_sea = vec2(sea_map(sea_p + vec2(e, 0.0), tm, foot) - h0,
                                 sea_map(sea_p + vec2(0.0, e), tm, foot) - h0) / e * sea_amp;
                    rm_sea_crest = (h0 - 0.6) * sea_k * sea_amp;
                }
                // Seven trains at unrelated angles and lengths do not line up into a weave the way five
                // did. Each keeps its direction (a train bent by a noise turns into whorls and rings -
                // tried); only its height swells and sinks over a few hundred metres, as sets of waves do.
                float wa = noise2(wpos.xz * 0.0041), wb = noise2(wpos.xz * 0.0113 + 9.0);
                vec2 g = vec2(0.0);
                for (int i = 0; i < 7; ++i) {
                    float k = 6.2831853 / kLen[i];
                    float keep = sea_far * (1.0 - smoothstep(0.12, 0.45, foot / kLen[i]));
                    if (keep <= 0.0) continue;
                    float fi = float(i);
                    vec2 d = vec2(cos(kAng[i]), sin(kAng[i]));
                    float ph = dot(d, wpos.xz) * k + wave_t * sqrt(9.81 * k) + fi * 1.7;
                    keep *= 0.55 + 0.9 * mix(wa, wb, fract(fi * 0.37));
                    // (crests a little sharper than troughs)
                    float c = cos(ph);
                    g += d * (kSlope[i] * keep * sea_amp * (c + 0.35 * cos(2.0 * ph + 0.6)));
                }
                g += g_sea;
                // (the lake ripple of the surface under it gives way to the sea's own waves)
                wnor = normalize(vec3(wnor.x * (1.0 - 0.7 * sea_k) - g.x * sea_k, wnor.y, wnor.z * (1.0 - 0.7 * sea_k) - g.y * sea_k));
            }
        }
        // Splash rings bend the surface normal.
        vec3 rip_w = water_ripples(wpos.xz);
        wnor = normalize(vec3(wnor.x - rip_w.x, wnor.y, wnor.z - rip_w.y));
        // Winter: the lake freezes - no waves, a nearly flat sheet with a faint ripple in the surface.
        // (home's lakes; the open sea stays open)
        float ice_k = smoothstep(0.35, 0.75, scene.season.w) * (1.0 - smoothstep(1350.0, 1750.0, rm_home_dist(wpos.xz)));     // the lake freezes in winter (not from a snow shower in another season)
        if (ice_k > 0.001) {
            vec3 n_ice = normalize(vec3((noise2(wpos.xz * 1.7) - 0.5) * 0.05, 1.0, (noise2(wpos.xz * 1.7 + 31.0) - 0.5) * 0.05));
            wnor = normalize(mix(wnor, n_ice, ice_k));
        }
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
        // (Seascape: the water shows more of its own green-blue in the crests, close by)
        deep += vec3(0.48, 0.54, 0.36) * (0.09 * clamp(rm_sea_crest, -0.4, 1.2) * (1.0 - smoothstep(30.0, 180.0, t_water)));
        deep = max(deep, vec3(0.0));
        float shore_blend = max(0.1, scene.water_shore.x);
        float shore_noise = scene.water_shore.y;
#ifdef VIBE_WATER
        // wpos.xz was already 1 cm-snapped at the source (line ~1692)
        // so everything in this block — terrain_y, foam, shore noise,
        // wave normals — sees TAA-jitter-stable coordinates.
#endif /* VIBE_WATER */
        float terrain_y = water_only ? meshHeight(wpos.xz)
                                      : terrainM_lod(wpos.xz, t_water);
        // Noise-free depth — the foam band uses this so the highlight
        // follows the actual shoreline geometry instead of wobbling
        // with the shore-noise. depth_m below adds the noise on top
        // for the wider shallow→deep tint transition only.
        float depth_clean = max(0.0, water_y - terrain_y);
        float depth_m   = depth_clean;
#ifndef VIBE_WATER
        float dn = noise2(wpos.xz * 0.18) +
                    0.5 * noise2(wpos.xz * 0.42);
#else /* VIBE_WATER */
        // Lower-frequency shore noise: the old metre-scale octaves made
        // the water/land boundary visibly crawl when the camera moved
        // along the shore. A slow meander reads as a natural coastline
        // without the swim.
        float dn = noise2(wpos.xz * 0.10) +
                    0.35 * noise2(wpos.xz * 0.26);
#endif /* VIBE_WATER */
        depth_m += (dn - 0.6) * shore_blend * shore_noise;
        depth_m  = max(0.0, depth_m);
#ifndef VIBE_WATER
        float water_blend = smoothstep(0.0, shore_blend, depth_m);
#else /* VIBE_WATER */
        // Screen-space anti-alias the shore transition. A fixed
        // metre-width smoothstep collapses to <1 px at grazing angles
        // / distance, so the shoreline aliases and jumps frame-to-
        // frame as the camera moves. Never let the edge be narrower
        // than ~2.5 px of depth gradient.
        float sb_aa = max(shore_blend, fwidth(depth_m) * 2.5);
        float water_blend = smoothstep(0.0, sb_aa, depth_m);
#endif /* VIBE_WATER */
        // P_Malin-style underwater extinction when river style is on.
        // exp2(-depth * ext_col) gives the characteristic green-river
        // look (red attenuates fastest, then green, blue lasts). Mixed
        // against the user's deep tint at 60% so the slider still has
        // colour authority.
        if (water_style == 1) {   // (2, the lake style, has its own branch below)
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
#ifndef VIBE_WATER
            float w_dist = length(wpos - scene.camera_pos.xyz);
#else /* VIBE_WATER */
            // t_water_plane (line 1765) already holds the camera-to-surface
            // distance — reuse it instead of recomputing a length() here.
            float w_dist = t_water_plane;
#endif /* VIBE_WATER */
            if (water_lit > 0.95 && w_dist < 80.0 &&
                dot(wnor, sunDirW) > 0.05) {
                float self_lit = water_only
                    ? calcShadowMesh(wpos + wnor * 0.05, sunDirW)
                    : calcShadow(wpos + wnor * 0.05, sunDirW);
                water_lit = min(water_lit, self_lit);
            }
#ifndef VIBE_WATER
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
#endif /* ! VIBE_WATER */
        }

        // Underwater showthrough — for shallow water we evaluate the
        // terrain at wpos.xz and pass its surface tint through the
        // water tinted by depth. Beer's-law-style attenuation:
        // T = exp(-depth * absorption) so deep water still hides
        // the bottom while shallow water reads transparent.
        vec3 baseTint = shallow;     // current shore-blended water colour
#ifdef VIBE_WATER
        // Transparency / water clarity. The slider now controls how far
        // light penetrates: 0 = murky (bottom hidden within ~0.5 m),
        // 1 = very clear (seabed visible many metres down). Crucially
        // SHALLOW water is strongly see-through whenever clarity > ~0.2,
        // which is what reads as real shallow water over a sandy bottom.
#endif /* VIBE_WATER */
        float trans_amt = scene.water_shore.w;
        if (trans_amt > 0.001) {
#ifndef VIBE_WATER
            // Reuse the depth_m we already computed for the shore
            // blend. Absorption coefficient picks ~50 % attenuation
            // at a 2 m depth, ~95 % by 6 m.
            float absorp = 0.5;
#else /* VIBE_WATER */
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
#endif /* VIBE_WATER */
            float Tw = exp(-depth_m * absorp);
            // Visibility gate: skip the entire underwater shading
            // (calcNormal = 1× FBM, getMaterial = several `noise2`
            // taps, lambert + ambient) when the contribution is
            // below ~5 % of the surface tint. At 6 m depth Tw drops
            // to 0.05; deep-water pixels read as opaque tint anyway.
#ifndef VIBE_WATER
            if (Tw * trans_amt > 0.05) {
#else /* VIBE_WATER */
            if (Tw > 0.04) {
#endif /* VIBE_WATER */
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
#ifndef VIBE_WATER
                // Blend: shallow → underwater visible, deep → opaque
                // water_color. trans_amt scales the maximum showthrough.
#else /* VIBE_WATER */
                // Beer's-law transmittance (depth fade governed by the
                // clarity-depth knob) × trans_amt (master strength). At
                // Transparency=1 + small clarity depth: shallow reads
                // crystal-clear, deep water stays fully opaque.
#endif /* VIBE_WATER */
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
#ifndef VIBE_WATER
        if (scene.water_foam_color.a > 0.001) {
#else /* VIBE_WATER */
        float foam_amt = 0.0;   // hoisted so the final alpha can see it
        // Foam opacity (pc.color.w) is a MASTER override: it scales
        // both the foam colour AND its alpha, so 0 = no foam at all
        // (the shallow water shows through exactly like non-foam),
        // 1 = full opaque surf. Computed here so the colour blend can
        // use it too (previously only the alpha used it, leaving the
        // white foam tint visible at 0).
        float foam_op = clamp(pc.color.w, 0.0, 1.0);
        if (scene.water_foam_color.a > 0.001 && foam_op > 0.001) {
#endif /* VIBE_WATER */
            // (the sea's shore has surf: the same lapping foam, several times as wide as a lake's)
            float foam_w = max(0.05, scene.water_foam_params.x) * (1.0 + 5.0 * smoothstep(1350.0, 1750.0, rm_home_dist(wpos.xz)));
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
#ifdef VIBE_WATER
            foam_amt = foam_mix;
#endif /* VIBE_WATER */
#ifndef VIBE_WATER
            foam_cover = foam_mix * scene.water_foam_color.a;
#else /* VIBE_WATER */
            foam_cover = foam_mix * scene.water_foam_color.a * foam_op;
#endif /* VIBE_WATER */
            baseTint = mix(baseTint, scene.water_foam_color.rgb, foam_cover);
        }
        // Reflection goes over the tinted water; foam is matte, so it
        // suppresses most (not all - it is wet) of the mirror.
        // The water body's own colour is light scattered back out of it:
        // it follows the light level (dark at night, dull under overcast).
        baseTint *= pow(clamp(scene.sun_color.a * 0.4, 0.0, 1.0), 0.6);
        vec3 col_w = mix(baseTint, reflCol, fres_eff * (1.0 - 0.7 * foam_cover));
        if (ice_k > 0.001) {
            vec3 ia = ice_albedo(wpos.xz, rd, wnor, max(scene.water_params.y - terrain_y, 0.0));
            vec3 il = ia * (scene.sun_color.rgb * scene.sun_color.a * max(sunDirW.y, 0.0) * water_lit * 0.9 + scene.sky_color.rgb * 0.55);
            col_w = mix(col_w, mix(il, reflCol, fres_eff * 0.55), ice_k);
        }
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
#ifndef VIBE_WATER
        {
#else /* VIBE_WATER */
        // Gated by the SAME foam-opacity master (foam_op) as the shore
        // band, and folded into foam_amt so the final alpha drops here
        // too — this whitecap foam was previously independent of the
        // slider, which is why low foam opacity still looked opaque.
        if (foam_op > 0.001) {
#endif /* VIBE_WATER */
            float steepness = clamp(1.0 - wnor.y * 1.04, 0.0, 1.0);
            vec2 fp = wpos.xz * 0.55 + vec2(wave_t * 0.30, wave_t * 0.18);
            float fn = noise2(fp) * 0.6 + noise2(fp * 2.13) * 0.4;
            float foam = smoothstep(0.45, 0.85, fn * steepness * 1.6);
#ifdef VIBE_WATER
            foam_amt = max(foam_amt, foam);
#endif /* VIBE_WATER */
            col_w = mix(col_w, vec3(0.95, 0.97, 1.0),
#ifndef VIBE_WATER
                        foam * 0.85 * water_lit);
#else /* VIBE_WATER */
                        foam * 0.85 * water_lit * foam_op);
#endif /* VIBE_WATER */
        }

        // Churned, aerated patch where something just hit the water.
        if (rip_w.z > 0.001) {
            vec2 fq = wpos.xz * 7.0;
            float fnz = noise2(fq) * 0.6 + noise2(fq * 2.3 + 3.1) * 0.4;
            float rf_amt = rip_w.z * smoothstep(0.25, 0.75, fnz + rip_w.z * 0.5);
#ifdef VIBE_WATER
            foam_amt = max(foam_amt, rf_amt * 0.8);
#endif /* VIBE_WATER */
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
        vec3 ext_w = exp(-rm_haze_dist(t_water) * 0.00025 * vec3(1.0, 1.3, 2.0));
        // pow(x, 8) → 3 squarings.
        float sd2 = sundotW * sundotW;
        float sd4 = sd2 * sd2;
        float sd8 = sd4 * sd4;
        vec3 fogColW = ground_haze_air(rd, sunDirW);       // (ground_haze.glsl: the colour of the sky behind it)
        col_w = col_w * ext_w + fogColW * (1.0 - ext_w);

#ifndef VIBE_WATER
        // Write hit depth; rasterised geometry that's actually in
        // front (e.g. a cube on a pier) still occludes us.
#else /* VIBE_WATER */
        // Depth is owned by the RASTERISED water plane (this is real
        // geometry now, not a fullscreen analytic pass) — do NOT write
        // gl_FragDepth. That makes the terrain↔water silhouette a
        // pixel-exact hardware depth test (the whole point of the
        // rasterised-plane port). clipW is still needed for the motion
        // vector below.
#endif /* VIBE_WATER */
        vec4 clipW = pc.mvp * vec4(wpos, 1.0);
#ifndef VIBE_WATER
        // See main terrain branch — cap below the compose sky-cutoff
        // threshold so far water doesn't get repainted as sky-below-
        // horizon (black). Guard against w<=0 (point behind camera) so
        // the depth divide doesn't produce NaN.
        gl_FragDepth = min(clipW.z / max(clipW.w, 1e-4), 0.999989);
#endif /* ! VIBE_WATER */
        // Distance fog on the water surface too.
        {
            float fa = distanceFogAmount(wpos, scene.camera_pos.xyz);
            col_w = mix(col_w, scene.distance_fog_color.rgb, fa);
        }
#ifndef VIBE_WATER
        if (cam_under) col_w = under_col;      // seen from below (compose adds the water fog)
        outColor = vec4(col_w, 1.0);
        outGBuffer0 = vec4(clamp(col_w, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
        outGBuffer1 = vec4(0.5, 0.5, 0.7, 0.0);
#else /* VIBE_WATER */
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
#endif /* VIBE_WATER */
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
