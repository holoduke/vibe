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
#ifdef VIBE_GROUND
        return;      // (never reached: the ground build has no water pixels - the water pass shades them)
#else
#if !defined(VIBE_LITE) && !defined(VIBE_WATER)
        if (RMDBG(131072)) {       // (shading test, water "flat": as the first-start build above)
            vec3 wp = ro + rd * t_water;
            float fres = 0.03 + 0.97 * pow(1.0 - clamp(-rd.y, 0.0, 1.0), 5.0);
            vec3 sky_l = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
            vec3 deep_l = scene.water_color.rgb * (scene.sun_color.rgb * scene.sun_color.a * max(scene.sun_direction.y, 0.0) * 0.6 + scene.sky_color.rgb * 0.5);
            lite_out(wp, mix(deep_l, sky_l, fres));
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
        // The wind and the swell (terrain_rm_water.glsl wt_*; the swell's hit: terrain_raymarch.frag
        // just before this file). On the swell the surface stands where the ray met it: depths
        // below are taken from there. The bed under the point, the open sea's weight and the
        // depth are worked out once, here, for everything below.
        float wt_W = scene.water_wind.z;
#ifndef VIBE_WATER
        if (wt_sw_k > 1e-4) water_y = wpos.y;
#endif
        // The stream: its own surface (water_course.glsl; found in terrain_raymarch.frag).
        bool wt_in_stream = wt_stream_y > -1e8;
        WcAt wt_st;
        float wt_sl = 0.0;      // the stream's slope for its shading: a step's own, else smoothed across the points
        if (wt_in_stream) { water_y = wt_stream_y; wt_st = wc_stream_at(wpos.xz); wt_sl = wt_st.drop > 0.01 ? wt_st.slope : wt_st.sslope; }
        float wt_bed_y = water_only ? meshHeight(wpos.xz) : terrainM_lod(wpos.xz, t_water);
        float wt_depth = max(0.0, water_y - wt_bed_y);
        float wt_seak  = wt_sea_k(wpos.xz);
        wt_W = mix(wt_W, max(wt_W, scene.water_sun.z), wt_seak);     // (the open sea: its own wind, a sea breeze at least - environment.cpp sea_wind)
        float wave_str = scene.water_params.z * wt_lake();      // (calm in still air, rough in a gale: environment.cpp)
        float wave_t   = scene.water_params.w;
        // wave-frequency multiplier — packed in water_color_shallow.w
        // by update_scene_ubo. Defaults to 1.0; the slider lets the
        // user dial finer ripples or bigger sweeping waves.
        float wave_scale = max(0.05, scene.water_color_shallow.w);
        // The pixel's footprint on the water (m), along its two axes - across the view and along it
        // (stretched by the grazing angle): the FFT cascades are filtered over it through their mips
        // and the anisotropic filter (water_waves.glsl).
        vec2 wt_gx, wt_gy;
        {
            float px = max(t_water, 0.05) * 2.0 * scene.viewport.w;
            vec2  hz = length(rd.xz) > 1e-4 ? normalize(rd.xz) : vec2(1.0, 0.0);
            wt_gx = vec2(-hz.y, hz.x) * px;
            wt_gy = hz * (px / max(abs(rd.y), 0.03));
        }
        float wt_fold = 0.0;        // the wind sea's fold (> 0 where its surface crowds together: whitecaps)
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
            // (the ripples drift down the wind: water_swell.yz is the wind's direction summed over
            //  time - direction x clock swung the pattern round whenever the wind veered)
            wnor = lakeWaterNormal(wpos.xz - scene.water_swell.yz * 0.25, uv_scale * wave_scale,
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
                bool wt_fft = ww_fft_on();
                if (wt_fft) {
                    // The wind sea's four FFT cascades (water_waves.glsl; vk_engine/water_sim.cpp):
                    // their slopes filtered over the pixel through the mips are the sea's glitter and
                    // its far look; far out the gusts and slicks above still vary their strength.
                    // (They replace the analytic wave field and trains below: VIBE_NO_FFT=1 brings those back.)
                    vec3 sf = ww_fft_slope(wpos.xz, wt_gx, wt_gy, vec4(1.0));
                    g_sea = sf.xy * sea_amp;
                    wt_fold = sf.z * sea_k;
                } else if (sea_far < 1.0) {
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
                for (int i = 0; i < (wt_fft ? 0 : 7); ++i) {
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
                // (the wind raises them: a glassy sea in still air, steep chop in a gale - the
                //  cascades carry the wind in their spectrum already)
                g += g_sea;
                if (!wt_fft) { g *= wt_chop(); rm_sea_crest *= wt_chop(); }
                // (the lake ripple of the surface under it gives way to the sea's own waves)
                wnor = normalize(vec3(wnor.x * (1.0 - 0.7 * sea_k) - g.x * sea_k, wnor.y, wnor.z * (1.0 - 0.7 * sea_k) - g.y * sea_k));
            }
        }
        // The swell's slope (its height is in wpos already; terrain_raymarch.frag), and a crest
        // shows a little more of the water's own colour.
        if (wt_sw_k > 1e-4) {
            wnor = normalize(vec3(wnor.x - wt_sw_g.x, wnor.y, wnor.z - wt_sw_g.y));
            rm_sea_crest += clamp((water_y - scene.water_params.y + (SW_ON(8192) ? 1.0 : 0.0)) * 0.08, -0.3, 0.8);      // (over the sea's level: 1 m under the datum)
        }
        // The river's current (terrain_rm_water.glsl wt_river: wm_river's polyline): the ripple
        // pattern is carried along the channel, faster where it is deep (mid-channel), in two
        // phases half a cycle apart that are blended so neither one's restart shows (4 s cycle,
        // the reference game's pools); streaks of foam drawn out along the flow.
        // The local shallow-water solver round the camera (water_solver.glsl; vk_engine/water_solver.cpp):
        // inside its window the water really flows round the stones - its velocity bends the
        // carried patterns, its height gradient tilts the normal, its foam is the white water; the
        // analytic obstacle effects below (wt_obstacles) give way to it (and stay outside it).
        WsAt wt_sw = ws_at(wpos.xz);
        float wt_riv_foam = 0.0;
        float wt_obs_foam = 0.0;        // white water round the piers and rocks (wt_obstacles)
        float wt_obs_hole = 0.0;        // ... and the dark hole in the lee of each
        float wt_riv_B = 0.0;           // fast / narrow water: whiter foam and milky water (the reference game's B term)
        float wt_riv_k = 0.0;           // how much of this is the river's or the stream's running water
        float wt_riv_along = -1.0;      // metres down the river's course (its murk; -1: not the river)
        vec2  wt_riv_flow = vec2(0.0);  // the current here (m/s)
        float wt_fine_var = 0.0;        // the stream's wavelets too small for a pixel: slope variance for the glint
        vec2  wt_dbg_n = vec2(0.0);       // (the carried pattern alone: RMDBG 16777216)
        vec3  wt_dbg3 = vec3(0.0);        // (the stream's white by its source: RMDBG 8388608 + 16777216)
        {
            float riv_along = 0.0;
            vec2  riv_frame = vec2(0.0);     // metres down the course and across it: the frame the patterns are laid in (wt_river)
            vec4 rv = wt_in_stream ? vec4(wt_st.dir, wt_st.dist, wt_st.hw) : wt_river(wpos.xz, riv_along, riv_frame);
            if (wt_in_stream) riv_frame = vec2(wt_st.seg * 8.0, wt_st.lat);      // (the stream: its segments are about 8 m)
            if (rv.z >= 0.0) {
                float speed, riv_k;
                if (wt_in_stream) {
                    // The stream: the reference game's rule, 0.75 m/s on the level + 3 x the sine
                    // of the surface's fall (a step's face runs at up to 4.5), slower at the banks.
                    float sn = wt_sl / sqrt(1.0 + wt_sl * wt_sl);
                    speed = min(0.75 + 3.0 * sn, 4.5) * (1.0 - 0.5 * smoothstep(0.4, 1.0, wt_st.dist / max(wt_st.hw, 0.1)));
                    riv_k = 1.0;
                } else {
                    riv_k = (1.0 - smoothstep(rv.w + 4.0, rv.w + 20.0, rv.z)) *
                            smoothstep(30.0, 80.0, distance(wpos.xz, vec2(62.0, 36.0))) *       // (not out on home's lake)
                            (1.0 - smoothstep(1880.0, 1990.0, wpos.x)) * (1.0 - wt_seak);        // (nor into the sea)
                    float pool = 1.0 - smoothstep(22.0, 55.0, distance(wpos.xz, vec2(43.4, 297.4)));   // (the pool under the stone bridge)
                    speed = (0.30 + 1.10 * smoothstep(0.3, 2.2, wt_depth)) * (1.0 - 0.6 * pool);
                    // (the river falls to the sea in three short rapids - water_level.glsl: faster
                    //  there: 75 x the level's fall per metre, up to +2.2 m/s - over its stones a rapid is far steeper
                    //  than the 1 % its level falls on average; water_fx.cpp the same)
                    if (SW_ON(8192)) speed += 3.0 * clamp((wl_profile(riv_along - 6.0) - wl_profile(riv_along + 6.0)) / 12.0, 0.0, 0.03) * 25.0;
                    wt_riv_along = riv_along;
                }
                // Calm margins (the reference paints a calm-to-rough value across its river strips:
                // toward the bank the normal falls to 30 % and the streak foam goes): from half the
                // channel's half width out, and where the water is only a few decimetres deep (the
                // river's real bank is not at its nominal width). The current slows there too.
                // water_wet bit 1024 (VIBE_NO_CALM_MARGIN=1: off).
#ifndef RM_NO_CALM
                float wt_calm = SW_ON(1024)
                    ? max(smoothstep(0.5, 1.0, rv.z / max(rv.w, 0.1)), 1.0 - smoothstep(0.08, 0.5, wt_depth)) : 0.0;
#else
                float wt_calm = 0.0;
#endif
                speed *= 1.0 - 0.6 * wt_calm;
                if (riv_k > 0.001) {
                    wt_riv_k = riv_k;
                    vec2 fl = rv.xy * speed + scene.water_wind.xy * (0.012 * wt_W);           // (the wind pushes the surface a little)
                    // The piers and rocks standing in it (terrain_rm_water.glsl wt_obstacles): the current
                    // parts round them, eddies swirl behind them.
                    WtObs obs = wt_obstacles(wpos.xz, rv.xy, speed, wave_t);
                    // (inside the solver's window the solver's flow does this: the analytic pillow, wake,
                    //  hole and eddies fade out as it fades in)
                    { float ko = 1.0 - wt_sw.k; obs.dflow *= ko; obs.slope *= ko; obs.foam *= ko; obs.hole *= ko; }
                    fl += obs.dflow;
                    // The solver's current carries the patterns (the reference: S (t + 0.25 fade v_sw) on
                    // its sloped rivers, whose solver holds only the local part; here the solver holds the
                    // whole current, so its departure from the course's is added - half of it: carried
                    // over the 4 s cycle the full eddying flow winds the pattern into rings)
                    // (how far the solver's current departs from the course's here: behind and beside
                    //  a stone the course-frame streaks below give way to the solver's foam, which
                    //  really bends round it)
                    float wt_sw_dev = wt_sw.k * smoothstep(0.35, 0.9, length(wt_sw.vel - fl) / max(speed, 0.4));
                    fl += wt_sw.k * 0.5 * (wt_sw.vel - fl);
                    wt_riv_flow = fl * riv_k;
                    // Two phases half a 4 s cycle apart, blended so neither restart shows; each
                    // phase samples the pattern where the water there came from (here minus the
                    // flow x the phase's time), with a fresh offset every cycle.
                    float cyc = wave_t * 0.25;
                    float ph0 = fract(cyc), ph1 = fract(cyc + 0.5);
                    float w0 = 1.0 - abs(1.0 - 2.0 * ph0), w1 = 1.0 - w0;
                    vec2 h0 = pm_hash2(floor(cyc)) * 23.0, h1 = pm_hash2(floor(cyc + 0.5) + 7.0) * 23.0;
                    vec2 o0 = -fl * (ph0 * 4.0) + h0;
                    vec2 o1 = -fl * (ph1 * 4.0) + h1;
                    bool st2 = water_style == 2;
                    float us = st2 ? max(0.05, scene.water_river_extra.z) : 1.0;
                    // (running water is rougher than a pond: the faster, the more)
                    // (running water is roughened by its own flow, not by the wind: in still air a
                    //  stream still ripples - the wind's calm factor wt_lake is not applied here)
                    float bs = max(st2 ? scene.water_foam_params.w : 0.12, 0.12) * scene.water_params.z * 1.5 * (1.0 + 2.0 * smoothstep(0.3, 3.0, speed));
                    float fd = st2 ? max(5.0, scene.water_river_extinct.w * 100.0) : 120.0;
                    vec3 n0 = lakeWaterNormal(wpos.xz + o0, us * wave_scale, bs, wave_t * 0.2, t_water, fd);
                    vec3 n1 = lakeWaterNormal(wpos.xz + o1, us * wave_scale, bs, wave_t * 0.2, t_water, fd);
                    vec3 nr = normalize(n0 * w0 + n1 * w1);
                    // Near layers (the reference game's: 1.4 and 1.05 /m, carried at 0.89 and 1.19
                    // of the current, within 20 m): close by the surface moves at several speeds
                    // at once instead of sliding as one sheet.
                    float near_k = clamp(1.5 * (1.0 - t_water / 20.0), 0.0, 1.0);
                    if (near_k > 0.0) {
                        vec2 a0 = wt_near_slope((wpos.xz - fl * (ph0 * 4.0 * 0.89) + h0) * 1.4) * w0 + wt_near_slope((wpos.xz - fl * (ph1 * 4.0 * 0.89) + h1) * 1.4) * w1;
                        vec2 b0 = wt_near_slope((wpos.xz - fl * (ph0 * 4.0 * 1.19) + h1) * 1.05 + 17.0) * w0 + wt_near_slope((wpos.xz - fl * (ph1 * 4.0 * 1.19) + h0) * 1.05 + 17.0) * w1;
                        vec2 g = (a0 + b0) * (0.10 * near_k * (0.5 + 0.5 * smoothstep(0.3, 2.0, speed)));
                        nr = normalize(vec3(nr.x - g.x, nr.y, nr.z - g.y));
                    }
                    // The stream: running water is rough at the scale of a hand - wavelets of 15 - 35 cm
                    // carried with the current, rougher where it runs fast. Without them its surface
                    // was the lake's metres-wide ripple alone: a smooth mirror with soft bulges. Where a
                    // pixel is larger than the wavelets they go into the glint's width instead
                    // (wt_fine_var), as a rough surface seen from afar.
                    if (wt_in_stream) {
                        float px = max(t_water, 0.05) * 2.0 * scene.viewport.w;      // ~metres per pixel
                        float fa = 0.07 + 0.17 * smoothstep(0.5, 3.0, speed);
                        float k1 = 1.0 - smoothstep(0.07, 0.16, px), k2 = 1.0 - smoothstep(0.04, 0.09, px);
                        // (their own cycle, 1.2 s: carried for the 4 s of the coarse layers, the speed's
                        //  change across the stream - half at the banks - sheared them into fine stripes)
                        float fc = wave_t / 1.2, fp0 = fract(fc), fp1 = fract(fc + 0.5);
                        float fw0 = 1.0 - abs(1.0 - 2.0 * fp0), fw1 = 1.0 - fw0;
                        vec2 fh0 = pm_hash2(floor(fc) + 3.0) * 17.0, fh1 = pm_hash2(floor(fc + 0.5) + 11.0) * 17.0;
                        vec2 c0 = noised((wpos.xz - fl * (fp0 * 1.2 * 1.07) + fh0) * 3.1).yz * fw0 + noised((wpos.xz - fl * (fp1 * 1.2 * 1.07) + fh1) * 3.1).yz * fw1;
                        vec2 c1 = noised((wpos.xz - fl * (fp0 * 1.2 * 0.93) + fh1) * 5.6 + 31.0).yz * fw0 + noised((wpos.xz - fl * (fp1 * 1.2 * 0.93) + fh0) * 5.6 + 31.0).yz * fw1;
                        vec2 g = (c0 * k1 + c1 * (0.6 * k2)) * fa;
                        nr = normalize(vec3(nr.x - g.x, nr.y, nr.z - g.y));
                        wt_fine_var = fa * fa * 0.6 * ((1.0 - k1) + 0.36 * (1.0 - k2));
                    }
                    // The wind sea's cascades on running water (water_waves.glsl): the short ones a
                    // stretch of river has room for (fetch ~25 m), laid in the course's frame and
                    // carried in the same two phases, so they run down the current - rougher where
                    // it runs fast, a little more in a wind.
                    if (ww_fft_on()) {
                        vec2 fax = rv.xy, fay = vec2(-rv.y, rv.x);
                        vec2 fp0 = riv_frame + vec2(dot(o0, fax), dot(o0, fay)), fp1 = riv_frame + vec2(dot(o1, fax), dot(o1, fay));
                        float fa = 0.35 + 0.45 * smoothstep(0.3, 2.5, speed) + 0.25 * smoothstep(3.0, 12.0, wt_W);
                        vec4  fw = ww_fetch_w(25.0) * fa;
                        // (the finest cascade, 4 - 43 cm waves, fades out where a pixel covers more than three
                        //  of its texels: carried in two phases its mip-0 detail sparkled per pixel; what it
                        //  would have added goes into the glint's width, as a rough surface seen from afar)
                        float fpx = length(wt_gx);
                        float f3 = 1.0 - smoothstep(1.5, 3.0, fpx / (scene.water_fft_L.w / 64.0));
                        wt_fine_var += fw.w * fw.w * (1.0 - f3) * 0.25 * max(scene.water_fft.w, 0.01);
                        fw.w *= f3;
                        fw.z *= 1.0 - smoothstep(3.0, 6.0, fpx / (scene.water_fft_L.z / 64.0)) * 0.5;
                        vec3  s = ww_fft_slope_flow(fp0, fax, wt_gx, wt_gy, fw) * w0 + ww_fft_slope_flow(fp1, fax, wt_gx, wt_gy, fw) * w1;
                        nr = normalize(vec3(nr.x - s.x, nr.y, nr.z - s.y));
                    }
                    wt_dbg_n = nr.xz;
                    { float cs = mix(1.0, 0.3, wt_calm); nr = normalize(vec3(nr.x * cs, nr.y, nr.z * cs)); }   // (calm margin)
                    nr = normalize(vec3(nr.x - obs.slope.x, nr.y, nr.z - obs.slope.y));
                    wnor = normalize(mix(wnor, nr, riv_k));
                    // Foam streaks drawn out along the flow, carried like the pattern.
                    vec2 ax = rv.xy, ay = vec2(-rv.y, rv.x);
                    // (in the course's own frame - metres down it and across it - plus the drift along
                    //  and across the flow: not the absolute position projected on the flow's direction,
                    //  whose kilometre lever turned every small turn of the direction into metres - the
                    //  streaks fell apart into thin lines; wt_river)
                    vec2 q0 = riv_frame + vec2(dot(o0, ax), dot(o0, ay)), q1 = riv_frame + vec2(dot(o1, ax), dot(o1, ay));
                    float s0 = noise2(vec2(q0.x * 0.08, q0.y * 0.45)) * 0.7 + noise2(vec2(q0.x * 0.21, q0.y * 1.1) + 5.0) * 0.3;
                    float s1 = noise2(vec2(q1.x * 0.08, q1.y * 0.45)) * 0.7 + noise2(vec2(q1.x * 0.21, q1.y * 1.1) + 5.0) * 0.3;
                    float T = s0 * w0 + s1 * w1;
                    // The reference game's B: fast water (and narrow - the same flow through less
                    // width runs faster) foams and goes milky. Here from the speed, the river's
                    // width against its usual 7 m, and the steps of the stream.
                    float narrow = wt_in_stream ? 1.0 : clamp(7.0 / max(rv.w, 1.0), 0.5, 1.5);
                    float vB = speed * narrow;
                    wt_riv_B = 0.25 * pow(clamp((vB - 0.9) / 2.2, 0.0, 1.0), 1.5) + (wt_in_stream ? 0.35 * smoothstep(0.15, 0.6, wt_sl) : 0.0);
                    // (broken into patches along their length, and softer: unbroken they were thin white
                    //  ruled lines tens of metres long - a 2 m wide streak seen at a grazing angle is a
                    //  pixel tall; VIBE_RM_SKIP=25165824 shows them in blue)
                    float s_patch = smoothstep(0.35, 0.75, noise2(wpos.xz * 0.11 + o0 * 0.11) * w0 + noise2(wpos.xz * 0.11 + o1 * 0.11) * w1);
                    float foam_s = smoothstep(0.66, 0.95, T) * smoothstep(0.6, 1.3, speed) * 0.32 * s_patch;
                    foam_s *= 1.0 - wt_calm * (1.0 - smoothstep(1.0, 2.0, speed));      // (no streaks in a slow calm margin)
                    float foam_B = clamp(T * T * T * 0.5 * wt_riv_B * 8.0, 0.0, 1.0);
                    // (the stream: in patches where the streaks gather, not as a veil - T^3 x B lay a
                    //  thin even froth over every sheet; VIBE_RM_SKIP=25165824 shows it as red)
                    if (wt_in_stream) foam_B = smoothstep(0.58, 0.85, T) * clamp(wt_riv_B * 2.5, 0.0, 1.0);
                    wt_dbg3 = RMDBG(67108864) ? vec3(fract(q0.x * 0.08), fract(q0.y * 0.45), T) : vec3(foam_B, wt_in_stream ? 0.15 : 0.0, foam_s);
                    wt_riv_foam = riv_k * max(foam_s, foam_B) * (1.0 - wt_sw_dev);
                    // A step's face: white water all over, broken by the streak pattern.
                    if (wt_in_stream) {
                        float fall = smoothstep(0.25, 0.9, wt_st.slope) * step(0.01, wt_st.drop);      // (a step's face)
                        // (strands of white running down the face, carried with the water like the
                        //  streaks, finer; between them the clear water shows)
                        float f0 = noise2(vec2(q0.x * 0.55, q0.y * 3.0) + 11.0) * 0.65 + noise2(vec2(q0.x * 1.3, q0.y * 6.5) + 3.0) * 0.35;
                        float f1 = noise2(vec2(q1.x * 0.55, q1.y * 3.0) + 11.0) * 0.65 + noise2(vec2(q1.x * 1.3, q1.y * 6.5) + 3.0) * 0.35;
                        float strands = smoothstep(0.30, 0.72, f0 * w0 + f1 * w1);
                        // The water pours over a step in tongues between the stones of its lip (wc_n1 across
                        // the width, wavering): white where a tongue falls and churns, clear dark water
                        // over the rock between them.
                        float tongue = smoothstep(0.35, 0.70, wc_n1(wt_st.lat * 1.3 + wt_st.seg * 7.0 + 0.25 * sin(wave_t * 1.7 + wt_st.lat * 2.0)));
                        wt_riv_foam = max(wt_riv_foam, fall * tongue * (0.20 + 0.75 * strands));
                        // ... and boils white in the pool at its foot, the foam drifting out with the current.
                        wt_riv_foam = max(wt_riv_foam, wt_st.foot * smoothstep(0.25, 0.75, T * 0.6 + strands * 0.6) * (0.6 + 0.4 * tongue));
                        // A WATERFALL (a fall over 1.2 m, its face near sheer: water_course.cpp): the falling
                        // sheet in streaks - long vertical strands across the whole width, fine and coarse,
                        // running DOWN at the falling water's speed (sqrt(2 g h): 4 - 7 m/s), white where the
                        // water is torn to air, glassy darker strands between; aerated white over the whole
                        // pool where it lands (the mist over the foot: water_spray.cpp).
                        float wfall = smoothstep(1.0, 1.8, wt_st.drop) * smoothstep(0.6, 1.6, wt_st.slope);
                        if (wfall > 0.0) {
                            float vf = 5.0;
                            float sy = wpos.y + wave_t * vf;
                            float s1 = noise2(vec2(wt_st.lat * 4.5, sy * 0.35) + 7.0);
                            float s2 = noise2(vec2(wt_st.lat * 13.0, sy * 0.9) + 2.0);
                            float s3 = noise2(vec2(wt_st.lat * 31.0, sy * 2.2) + 5.0);
                            float st3 = s1 * 0.5 + s2 * 0.32 + s3 * 0.18;
                            float torn = smoothstep(0.38, 0.62, st3);
                            wt_riv_foam = max(wt_riv_foam, wfall * mix(0.35, 1.0, torn));
                            wnor = normalize(wnor + vec3(rv.y, 0.0, -rv.x) * ((s2 - 0.5) * 0.6 * wfall));      // (the strands ripple the sheet across)
                        }
                        float wland = wt_st.foot * smoothstep(1.0, 1.8, scene.stream_pts[max(int(wt_st.seg), 1) - 1].z - scene.stream_pts[max(int(wt_st.seg), 1)].z);
                        if (wland > 0.0) wt_riv_foam = max(wt_riv_foam, wland * mix(0.7, 1.0, smoothstep(0.3, 0.7, T * 0.5 + strands * 0.5)));
                        // A STEEP SHEET (the water following its bed down steep ground: water_course.cpp):
                        // white water that grows with the slope and the speed, broken by the streak
                        // pattern, and STANDING WAVES - short crests across the flow that stay where they
                        // are (fixed in the ground, not carried), white on their tops.
                        float sheet = smoothstep(0.06, 0.20, wt_sl) * (1.0 - step(0.01, wt_st.drop));
                        if (sheet > 0.0) {
                            float along = wt_st.seg * 8.0;
                            // (2026-11-26: the crests were one even ruled grating - every 1.9 m, straight
                            //  across the whole width, tilting the normal by up to 0.6 - which turned the
                            //  mirror into regular horizontal stripes over the rapid (sw_rapid3_zoom.png).
                            //  Now short-crested: the spacing wanders (a slow warp of the phase), the
                            //  crests come in patches across and along, a quarter of the tilt, and they
                            //  fade where a pixel is wider than a fifth of their spacing.)
                            float sw_ph = along * 3.3 + 3.0 * wc_n1(along * 0.15 + wt_st.lat * 0.9 + 3.0) + 1.7 * wc_n1(wt_st.lat * 0.8 + wt_st.seg) + 0.35 * sin(wave_t * 2.1 + wt_st.lat);
                            float crest = sin(sw_ph);
                            float sw_patch = smoothstep(0.30, 0.75, wc_n1(wt_st.lat * 1.6 + along * 0.23 + 5.0));
                            float sw_px = max(t_water, 0.05) * 2.0 * scene.viewport.w;
                            float amp = sheet * (0.08 + 0.10 * smoothstep(1.5, 4.0, speed)) * sw_patch * (1.0 - smoothstep(0.2, 0.45, sw_px / 1.9));
                            wnor = normalize(vec3(wnor.x - rv.x * amp * cos(sw_ph), wnor.y, wnor.z - rv.y * amp * cos(sw_ph)));
                            float rough = sheet * smoothstep(1.4, 3.8, speed);      // (white only where it really runs fast: a pale veil over the whole sheet read as a mirror)
                            float wwhite = rough * smoothstep(0.30, 0.85, strands * 0.8 + 0.35 * smoothstep(0.3, 0.95, crest));
                            wt_riv_foam = max(wt_riv_foam, clamp(wwhite, 0.0, 1.0) * (0.65 + 0.35 * tongue));
                            wt_dbg3.y = wwhite;
                        }
                        // (the surface tilts with the stream's own slope: the face of a step leans down it)
                        wnor = normalize(vec3(wnor.x + rv.x * wt_sl, wnor.y, wnor.z + rv.y * wt_sl));
                        // The surface's relief (water_relief.glsl: humps over the stones under it, standing
                        // waves, pillows in the gaps) tilts it as it stands - its slope from two taps a
                        // little apart (not the solver's part: its slope is in the normal below, ws_at).
                        if (SW_ON(4096)) {
                            float re = max(0.12, max(t_water, 0.05) * 2.0 * scene.viewport.w);
                            float r0 = wr_relief(wpos.xz, wt_st, false);
                            vec2  px = wpos.xz + vec2(re, 0.0), pz = wpos.xz + vec2(0.0, re);
                            vec2  rg = vec2(wr_relief(px, wc_stream_at(px), false) - r0, wr_relief(pz, wc_stream_at(pz), false) - r0) / re;
                            rg *= 1.0 / max(1.0, length(rg) / 0.8);
                            wnor = normalize(vec3(wnor.x - rg.x, wnor.y, wnor.z - rg.y));
                            // (a crest's top is where the water breaks: a little white on the highest humps)
                            wt_riv_foam = max(wt_riv_foam, smoothstep(0.12, 0.35, r0) * smoothstep(0.35, 0.8, T) * 0.6);
                        }
                    }
                    // THE RIVER'S RAPIDS (2026-11-27, they "barely read"): its level falls 0.23 m over
                    // ~8 m at 700, 1250 and 1800 m down the course (water_level.glsl wl_profile - the
                    // hit pass draws that step); the step's face runs white, broken by the streaks,
                    // and below it a train of standing waves (crests fixed in place, ~5.5 m apart -
                    // 2 pi U^2 / g at ~3 m/s - dying out over ~40 m) breaks white on its crests; the
                    // water there is milky. Less toward the calm margins. VIBE_NO_SLOPED_RIVER: none.
                    if (!wt_in_stream && SW_ON(8192)) {
                        float xr = riv_along - 700.0;
                        if (abs(riv_along - 1250.0) < abs(xr)) xr = riv_along - 1250.0;
                        if (abs(riv_along - 1800.0) < abs(xr)) xr = riv_along - 1800.0;
                        float face = 1.0 - smoothstep(4.5, 8.0, abs(xr));
                        float tail = smoothstep(0.0, 4.0, xr) * (1.0 - smoothstep(14.0, 55.0, xr));
                        // Foam LINES below the rapid (as on real rivers: the white water's leftovers
                        // drawn out by the current into long ribbons, 0.2 - 0.6 m wide, wandering,
                        // broken into lengths, thinning out over ~150 m). In the course's frame and
                        // carried in the same two phases as the streaks, so they drift at the
                        // surface speed; fewer toward the calm margins.
                        if (xr > 6.0 && xr < 200.0) {
                            float lk = smoothstep(6.0, 18.0, xr) * exp(-(xr - 18.0) / 70.0) * (1.0 - 0.8 * wt_calm) * (1.0 - wt_sw_dev);
                            float l0 = noise2(vec2(q0.x * 0.03, q0.y * 0.85 + 0.6 * noise2(vec2(q0.x * 0.05, 3.0)))) ;
                            float l1 = noise2(vec2(q1.x * 0.03, q1.y * 0.85 + 0.6 * noise2(vec2(q1.x * 0.05, 3.0)))) ;
                            float b0 = noise2(vec2(q0.x * 0.12, q0.y * 0.3) + 21.0), b1 = noise2(vec2(q1.x * 0.12, q1.y * 0.3) + 21.0);
                            float ln = smoothstep(0.66, 0.80, l0) * smoothstep(0.35, 0.6, b0) * w0 + smoothstep(0.66, 0.80, l1) * smoothstep(0.35, 0.6, b1) * w1;
                            wt_riv_foam = max(wt_riv_foam, riv_k * ln * lk * 0.75);
                        }
                        if (face + tail > 0.0) {
                            float side = 1.0 - 0.7 * wt_calm;
                            float lat = noise2(vec2(riv_frame.y * 0.35, xr * 0.12) + 9.0);
                            float amp = tail * (0.55 + 0.45 * lat) * side;
                            float ph = 6.2832 * (xr - 4.0) / 5.5 + 0.6 * sin(riv_frame.y * 0.45);
                            float dh = -0.18 * 1.14 * amp * sin(ph);                     // (slope of 0.18 cos(k x) along the flow)
                            wnor = normalize(vec3(wnor.x - dh * rv.x * riv_k, wnor.y, wnor.z - dh * rv.y * riv_k));
                            float crest = smoothstep(0.25, 0.85, cos(ph)) * amp;
                            float whit = max(face * side * mix(0.75, 1.0, smoothstep(0.3, 0.8, T)),
                                             crest * mix(0.35, 1.0, smoothstep(0.35, 0.75, T)));
                            wt_riv_foam = max(wt_riv_foam, riv_k * whit);
                            wt_riv_B = max(wt_riv_B, 0.30 * max(face, 0.6 * tail) * side);
                        }
                    }
                    // The white water round the piers and rocks: pillow, wake, trail, eddy spirals.
                    if (obs.foam > 0.0) wt_riv_foam = max(wt_riv_foam, riv_k * obs.foam * mix(0.55, 1.0, smoothstep(0.3, 0.8, T)));
                    wt_obs_foam = riv_k * obs.foam * mix(0.55, 1.0, smoothstep(0.3, 0.8, T));
                    // (the hole behind a rock breaks the white: a dark gap in a white sheet)
                    wt_riv_foam *= 1.0 - 0.9 * obs.hole;
                    wt_obs_hole = riv_k * obs.hole;
                }
            }
        }
        // The wind sea's cascades on a lake (water_waves.glsl): only the short waves its fetch
        // (~300 m) lets grow, and only with wind - glassy in still air (the castle lake's mirror),
        // a fine chop in a breeze. (The sea has them all, above; running water its own, above.)
        if (ww_fft_on()) {
            float lake_k = (1.0 - wt_seak) * (1.0 - wt_riv_k) * 0.3 * smoothstep(1.5, 9.0, wt_W);
            if (lake_k > 0.001) {
                vec3 s = ww_fft_slope(wpos.xz, wt_gx, wt_gy, ww_fetch_w(300.0) * lake_k);
                wnor = normalize(vec3(wnor.x - s.x, wnor.y, wnor.z - s.y));
            }
        }
        // Surf (after the reference game's): four wave fronts on a 20 s cycle roll in to the
        // beach, each a thin ridge exp(-40 d)(1 - exp(-80 d)) 0.4 m high with a noise mask along
        // the coast, leaving foam behind. The reference reads a distance-to-coast field; here
        // the depth stands in (s = depth / 2 m: the fronts rise at 2 m and run out at the
        // waterline - waves break by depth). Further out in a stronger wind.
        // (2026-11-26, later: by the DISTANCE to the coast where it is known - terrain_raymarch.frag
        //  wt_shore_d, water_waves.glsl ww_shore_dist: s = distance / 25 m, so the fronts come in
        //  evenly along a steep coast as along a flat beach; VIBE_NO_SHORE_SDF=1: the depth again.)
        float wt_surf_foam = 0.0;
        bool  wt_surf_sdf = scene.water_flags.z > 0.5 && wt_shore_d < 1e3;
        if (wt_seak > 0.01 && (wt_surf_sdf ? wt_shore_d < 25.0 : wt_depth < 2.2)) {
            float s = wt_surf_sdf ? wt_shore_d / 25.0 : wt_depth * 0.5;
            float sea_side = clamp(10.0 * (1.0 - s / (0.2 + 0.8 * clamp(wt_W / 30.0, 0.0, 1.0))), 0.0, 1.0) * wt_seak;
            if (sea_side > 0.0) {
                float m = -noise2(wpos.xz * 0.0225 + scene.water_swell.yz * 0.03);
                float jit = (noise2(wpos.xz * 0.021 + 3.7) - 0.5) * 0.25;
                float ridge = 0.0, dridge = 0.0, trail = 0.0;
                for (int j = 0; j < 4; ++j) {
                    float c = (float(j) + 0.5) * 0.25;
                    float msk = 2.0 * clamp(1.0 - 8.0 * abs(m + c), 0.0, 1.0);
                    if (msk <= 0.0) continue;
                    float fs = 1.0 - 2.0 * fract(0.05 * wave_t + jit + c);      // the front's place: s = 1 .. -1 (up the beach)
                    float d = s - fs;
                    if (d <= 0.0) continue;
                    float e1 = exp(-40.0 * d), e2 = exp(-80.0 * d);
                    float r = msk * e1 * (1.0 - e2);
                    if (r > ridge) { ridge = r; dridge = msk * (80.0 * e1 * e2 - 40.0 * e1 * (1.0 - e2)); }
                    trail = max(trail, msk * 0.45 * exp(-5.0 * d) * (1.0 - smoothstep(0.0, 0.6, -fs)));
                }
                ridge *= sea_side; dridge *= sea_side; trail *= sea_side;
                // The ridge's slope: 0.4 m x ridge over s, along the depth's gradient (from the
                // screen derivatives: no extra terrain taps).
                vec2 dx = dFdx(wpos.xz), dy = dFdy(wpos.xz);
                float ddx = dFdx(wt_depth), ddy = dFdy(wt_depth);
                float det = dx.x * dy.y - dx.y * dy.x;
                vec2 gd = abs(det) > 1e-10 ? vec2(ddx * dy.y - ddy * dx.y, dx.x * ddy - dy.x * ddx) / det : vec2(0.0);
                gd = (any(isnan(gd)) || dot(gd, gd) > 0.25) ? vec2(0.0) : gd;
                // (ds/dx: the depth's gradient / 2 m, or its direction / 25 m with the coast's distance)
                vec2 ds = wt_surf_sdf ? (dot(gd, gd) > 1e-10 ? normalize(gd) / 25.0 : vec2(0.0)) : gd * 0.5;
                vec2 sl = 0.4 * dridge * ds;
                sl *= 1.0 / max(1.0, length(sl));
                wnor = normalize(vec3(wnor.x - sl.x, wnor.y, wnor.z - sl.y));
                wt_surf_foam = clamp(ridge * 1.3 + trail, 0.0, 1.0) * 0.8 * (1.0 - exp(-4.0 * wt_depth)) * 1.6;
            }
        }
        // Soft water edge: over the last centimetres (8 at the sea, 4 elsewhere) the water
        // gives way - its normal goes flat, its reflection and glints fade, and the bed's own
        // colour takes over (below).
        float wt_edge = smoothstep(0.0, 0.08, wt_depth);      // (the reference: the last 8 cm everywhere - 4 cm at a lake drew a line)
        wnor = normalize(mix(vec3(0.0, 1.0, 0.0), wnor, wt_edge));
        // Splash rings bend the surface normal.
        vec3 rip_w = water_ripples(wpos.xz);
        // The ripple field round the camera (water_waves.glsl ww_ripple; vk_engine/water_sim.cpp):
        // wakes of waders and fording cars, splash rings that reflect off the banks and drift with
        // the current - its slope on every body of water, and a little foam where it is steep (the
        // reference's ripple foam). Inside its window it carries the splash rings (water_ripples
        // leaves their slope to it there and keeps their churned foam).
        {
            vec3 rf = ww_ripple(wpos.xz) * wt_edge;
            // (its slope is capped at 0.35: steeper, a facet mirrors a ray that goes DOWN - wt_rdown
            //  turns it to the water's own dark colour - and an impulse's trough read as a black hole)
            rf.xy *= 1.0 / max(1.0, length(rf.xy) / 0.35);
            rip_w.xy += rf.xy;
            float rs = length(rf.xy);
            rip_w.z = max(rip_w.z, smoothstep(0.18, 0.6, rs) * 0.45);
        }
        wnor = normalize(vec3(wnor.x - rip_w.x, wnor.y, wnor.z - rip_w.y));
        // The shallow-water solver: its surface's slope over the authored top (the water piled in
        // front of a stone, the dip behind it, bow waves and wakes) in the normal, and its foam as
        // white water - laid through a foam texture carried by its own current in two phases (the
        // reference's 1.5 soft saturate(T F - 0.015 (1 - F) + 0.1 F^3)).
        float wt_sw_white = 0.0;
        if (wt_sw.k > 0.001) {
            vec2 sg = wt_sw.grad * (wt_sw.k * wt_edge);
            wnor = normalize(vec3(wnor.x - sg.x, wnor.y, wnor.z - sg.y));
            // (x 1.8: the field's foam saturates at about half where the reference's would be near 1 -
            //  its many particles carry the rest of its white water; at 1 x the stones' white was a faint veil)
            float F = min(wt_sw.foam * wt_sw.k * 1.8, 1.0);
            if (F > 0.002) {
                float fc = wave_t / 1.5, fp0 = fract(fc), fp1 = fract(fc + 0.5);
                float fw0 = 1.0 - abs(1.0 - 2.0 * fp0);
                vec2 v = wt_sw.vel;
                float T0 = noise2((wpos.xz - v * (fp0 * 1.5) + pm_hash2(floor(fc)) * 13.0) * 2.3) * 0.65 + noise2((wpos.xz - v * (fp0 * 1.5)) * 6.1 + 3.0) * 0.35;
                float T1 = noise2((wpos.xz - v * (fp1 * 1.5) + pm_hash2(floor(fc + 0.5) + 5.0) * 13.0) * 2.3) * 0.65 + noise2((wpos.xz - v * (fp1 * 1.5)) * 6.1 + 3.0) * 0.35;
                float T = mix(T1, T0, fw0);
                wt_sw_white = clamp(1.5 * smoothstep(0.0, 1.0, clamp(T * F - 0.015 * (1.0 - F) + 0.1 * F * F * F, 0.0, 1.0)), 0.0, 1.0) * wt_edge;      // (was x 1.6 more: with the lace's own x 1.4 the stones' white was a solid sheet)
            }
        }
        // Winter: the lake freezes - no waves, a nearly flat sheet with a faint ripple in the surface.
        // (home's lakes; the open sea stays open)
        float ice_k = smoothstep(0.35, 0.75, scene.season.w) * (1.0 - smoothstep(1350.0, 1750.0, rm_home_dist(wpos.xz)));     // the lake freezes in winter (not from a snow shower in another season)
        if (wt_in_stream) ice_k = 0.0;      // (the stream runs on: fast water over steps and sheets does not freeze over - game_loop.cpp lets one wade in it then)
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
        bool cam_under = ro.y < scene.water_params.y + scene.water_swell.x;      // (the surface at the eye: the swell's CPU twin)
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
#ifndef VIBE_WATER
        // (the reference game: Schlick with F0 = 0.04 through its env BRDF, no cap - the colour of
        //  the water is its in-scatter, added below, not a share of a mix)
        float fres = 0.04 + 0.96 * fv5;
#else
        float fres = min(0.88, F0 + (1.0 - F0) * fv5);
#endif

        vec3 refl = reflect(rd, wnor);
        // A facet turned away from the eye mirrors a ray that goes DOWN: it meets the water again a
        // little further on (or the near bank), not the sky. It used to be lifted to the horizon
        // (refl.y >= 0.004 below) and so showed the brightest haze of the sky - with the Fresnel
        // of a grazing facet near 1. On running water, whose facets tilt a lot (a step's face, a
        // sheet's standing waves), half the surface was that haze: the stream read as a pale
        // mirror. Such a facet shows the water's own colour instead (wt_rdown, used below).
        float wt_rdown = 1.0 - smoothstep(-0.05, 0.03, refl.y);
        // Reflections lose sharpness with distance: one pixel of far water
        // covers many wavelets, each mirroring a slightly different
        // direction, and what it shows is the average - mostly a vertical
        // smear, as the wave slopes face the viewer. The reflected ray is
        // spread at random per pixel and frame (more with distance); the
        // temporal filter integrates that into a soft, stretched reflection.
        {
            float rdist = distance(wpos, ro);
#ifndef VIBE_WATER
            float spread = 0.004;           // (the reference game: a mirror, roughness 0.02, not widened by distance or wind)
#else
            float spread = 0.004 + 0.085 * smoothstep(8.0, 260.0, rdist) + 0.03 * smoothstep(6.0, 24.0, wt_W);
#endif
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
#ifdef VIBE_WATER
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sunHalo * 0.6;
        // Broad, soft sun glow (≈pow(.,4)) on top of the tight halo —
        // gives the wide specular sheet that made the old reflection
        // read as "lit water" rather than a flat gradient.
        sky_refl += scene.sun_color.rgb * scene.sun_color.a * sh4 * 0.30;
#else
        // (The raymarched water mirrors the sky as it is - the physical sky has the sun's
        //  aureole - and the sun itself through the glint (spec, a mirror GGX). The two lobes
        //  added here before - 0.6 sun at cos^64 and 0.3 sun at cos^4, i.e. a 60-degree-wide sheet
        //  of sunlight - lit every water facet that faced the sun's side: on running water,
        //  whose facets turn every way, a silver film over the whole surface. The reference's
        //  water has neither.)
        if (scene.atmo_params.x < 0.5) sky_refl += scene.sun_color.rgb * scene.sun_color.a * sunHalo * 0.6;
#endif

        vec3 reflCol = sky_refl;
        float refl_terrain_t = 1e9;     // terrain hit along the reflection (objects behind it stay hidden)
        // RT-style FBM-march reflection — picks up the actual mountain
        // / valley silhouettes in the water surface. Cheap because
        // we use a reduced-quality march; if it misses (refl ray
        // exits past the terrain or goes into the sky) fall back to
        // the analytical sky.
        // (the stream lies in a channel: its banks stand in the low reflections - march for them
        //  down to the horizon there; elsewhere a low ray goes on far over open water)
        bool do_rt_refl = scene.water_params.x > 0.5 &&
                          pc.grass_params.w > 0.5 && refl.y > (wt_in_stream ? 0.003 : 0.02);
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
                // The woods standing on that ground (the rule the trees are placed by: the home
                // tile's baked cover, else rm_forest) - in the mirror as a canopy, as the far
                // landscape draws them. Without it a wooded hillside on the far shore mirrored
                // as the bare meadow under the trees, bright lime green, wherever the ray did not
                // happen to meet a tree in the ray-tracing scene (most of them: the far trees
                // are not in it).
                {
                    vec2  cuv = r_pos.xz / 2048.0 + 0.5;
                    float woods = (all(greaterThan(cuv, vec2(0.0))) && all(lessThan(cuv, vec2(1.0))))
                                ? smoothstep(0.06, 0.40, textureLod(u_albedo[22], cuv, 2.0).r)
                                : rm_forest(r_pos.xz, r_pos.y, r_nor.y, scene.water_params.y);
                    vec3  canopy = mix(vec3(0.028, 0.052, 0.024), vec3(0.058, 0.082, 0.034), noise2(r_pos.xz * 0.11));
                    r_col = mix(r_col, canopy * (max(dot(vec3(0.0, 1.0, 0.0), sunDirW), 0.0) * scene.sun_color.rgb * scene.sun_color.a * 0.8 + scene.sky_color.rgb * 0.6), woods);
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
        // Running water: its murk along the course (wt_river_murk: clear at the spring, green in
        // the stone-bridge pool, silty at the mouth; the stream as clear as the spring) and the
        // reference game's milky fast water: albedo x (1 + 4.5 B).
        vec4 wt_murk = wt_in_stream ? vec4(0.045, 0.075, 0.045, 1.0) : wt_river_murk(max(wt_riv_along, 0.0));      // (the stream: a little milky - clear water over a mossy bed read as no water at all)
        if (wt_riv_k > 0.001) {
            float mk = wt_riv_k * (wt_in_stream || wt_riv_along >= 0.0 ? 1.0 : 0.0);
            deep = mix(deep, wt_murk.rgb, 0.7 * mk) * (1.0 + 4.5 * wt_riv_B * wt_riv_k);
            shallow_tint = mix(shallow_tint, mix(shallow_tint, wt_murk.rgb * 2.2, 0.5), 0.6 * mk) * (1.0 + 2.0 * wt_riv_B * wt_riv_k);
        }
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
        float terrain_y = wt_bed_y;        // (worked out at the top of this file)
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

        // The sun's glint: a normalised microfacet lobe, Fresnel at the half vector, / 4 N.V.
        // Its width is the surface's sub-pixel roughness from the wind (Cox-Munk: slope
        // variance 0.003 + 0.00512 W): a small bright sun image on calm water, a wide glitter
        // in a gale. (It was pow(N.H, 64) x 0.8 of the sun with no Fresnel - a lobe for a rough
        // sea at full mirror strength. On calm water it lit every sunlit pixel near the sun's
        // mirror direction to white: the white wedge under the stone bridge was sunlight
        // through an arch, seen against the sun, painted as one flat glare.)
        // (2026-11-26, the reference game's full water: its glint is a mirror GGX - alpha 4e-4,
        // alpha^2 = 1.6e-7 - with Vis and Fresnel (F0 0.04); the width of the glitter path comes
        // only from how the normals spread. Ours: that lobe, widened by the spread of the normal
        // between neighbouring pixels (what a pixel averages) and a little of the wind's
        // unresolved slope variance - our wave field has no FFT detail at the pixel scale.)
        vec3 H = normalize(sunDirW - rd);
        float spec;
        {
            vec3  dnx = dFdx(wnor), dny = dFdy(wnor);
            float var_px = clamp(0.5 * (dot(dnx, dnx) + dot(dny, dny)), 0.0, 0.25);
            float a2 = 1.6e-7 + 2.0 * var_px + 0.15 * (0.003 + 0.00512 * wt_W) + 2.0 * wt_fine_var;
            float NH = max(dot(wnor, H), 0.0), NL = max(dot(wnor, sunDirW), 0.0), NV = max(cosV, 1e-3);
            float dd = 1.0 + (a2 - 1.0) * NH * NH;
            float D = a2 / (3.14159265 * dd * dd);
            float al = sqrt(a2);
            float Vis = 0.5 / max(NL * (NV * (1.0 - al) + al) + NV * (NL * (1.0 - al) + al), 1e-5);
            float Fh = 0.04 + 0.96 * pow(1.0 - clamp(dot(-rd, H), 0.0, 1.0), 5.0);
            // (x 2.5: the sun's irradiance in the units the water's light uses - the ground's
            //  Lambert takes sun x 0.8 for E / pi; / 0.8: the glint line below multiplies by it)
            spec = min(NL * D * Vis * Fh * (2.5 / 0.8), 6000.0) * step(0.0, sunDirW.y) * wt_edge;
            spec *= scene.water_sun.x;      // (the glint is the sun's DISC: hidden by a closed overcast, whose light is diffuse)      // (no glints over the last centimetres)
        }

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
        // The clouds' shade slides over the water as over the land (scene_ubo.glsl cloud_shadow:
        // the cloud pass's map of the sunlight through the real cloud layer).
        water_lit *= cloud_shadow(wpos);

        // Underwater showthrough — for shallow water we evaluate the
        // terrain at wpos.xz and pass its surface tint through the
        // water tinted by depth. Beer's-law-style attenuation:
        // T = exp(-depth * absorption) so deep water still hides
        // the bottom while shallow water reads transparent.
        vec3 baseTint = shallow;     // current shore-blended water colour
#ifndef VIBE_WATER
        // THE WATER'S OWN COLOUR, the reference game's way (survey WATER_FULL.md 3.6 - 3.8): light
        // scattered back out of the water body, with GREY extinction - the hue is one scattering
        // albedo, the brightness the sun (two-sided Cornette-Shanks phase, g = 0.25, along the
        // view path plus an approximate refracted sun path) and the ambient. The bed below fades
        // with grey exp(-sigma d) (below, the see-through), and nothing is weighted by (1 - F):
        // reflection, in-scatter and the bed are added.
        //   albedo: the water colour setting (x 2.5: it was a lit tint, now an albedo) or the
        //   river's murk along its course, milkier where it runs fast (x (1 + 4.5 B));
        //   sigma: 0.5 /m (the old see-through's), the river's murk, thicker where it runs white.
        // (x 1.4 for a lake: at x 2.5 the castle lake's in-scatter was a milky green sheet - the owner:
        //  "foggy"; the sea and running water keep 2.5)
        vec3  wt_alb = clamp(mix(scene.water_color.rgb, wt_murk.rgb, (wt_riv_k > 0.001 && (wt_in_stream || wt_riv_along >= 0.0)) ? wt_riv_k : 0.0) * mix(1.4, 2.5, max(wt_seak, wt_riv_k)), vec3(0.01), vec3(0.9));
        wt_alb = min(wt_alb * (1.0 + 4.5 * min(wt_riv_B, wt_in_stream ? 0.08 : 1.0) * wt_riv_k), vec3(0.95));      // (the stream: its white water is drawn as froth; its body stays dark and clear)
        float wt_sig = mix(0.5, wt_murk.w, wt_riv_k) * (1.0 + 2.0 * min(wt_riv_B, wt_in_stream ? 0.08 : 1.0));
        vec3  wt_refr = refract(rd, wnor, 1.0 / 1.33);
        float wt_d = (dot(wt_refr, wt_refr) > 1e-6) ? depth_clean / max(-wt_refr.y, 0.15) : depth_clean;
        // REFRACTION, the reference's rule (WATER_FULL 3.6): only the DETAIL slopes bend the view
        // into the water - the tilt of the normal against the swell's (macro) normal, each way
        // clamped to +-0.2, faded out over the last 20 cm of depth (saturate(5 gap)) - and where the
        // bent ray comes out on ground above the water (a bank or stone in front) the unbent one is
        // used (its depth test). The reference shifts its picture of the scene behind by
        // 0.33 / (z + 1) of the screen; here the bed is worked out, so the same bend is applied to the
        // ray: refracted (Snell, 1.33) at the macro normal tilted by the clamped detail. Its path to
        // the bed is the in-scatter's and the bed's fade (wt_d). It replaces the earlier "bed wavers"
        // (the bed straight below, shifted by 0.6 x the whole normal's tilt x depth, up to 1.5 m:
        // swell tilt included, no clamp). VIBE_NO_WATER_REFRACT=1: that again.
        bool  wt_refr_ok = false;
        vec3  wt_bed_p = vec3(wpos.x, terrain_y, wpos.z);
        // (Ray Reconstruction's guides for a water pixel, as the reference's: the refracted bed's
        //  albedo and normal - its colour where the bed shows, the water's own albedo where deep)
        vec3  wt_guide_alb = wt_alb;
        vec3  wt_guide_n = vec3(0.0, 1.0, 0.0);
        if (scene.water_flags.x > 0.5 && depth_clean > 0.005 && exp(-mix(0.5, wt_murk.w, wt_riv_k) * wt_d) * scene.water_shore.w > 0.04) {
            vec3  nm  = wt_sw_k > 1e-4 ? normalize(vec3(-wt_sw_g.x, 1.0, -wt_sw_g.y)) : vec3(0.0, 1.0, 0.0);
            vec2  dtl = clamp(wnor.xz / max(wnor.y, 0.2) - nm.xz / nm.y, vec2(-0.2), vec2(0.2)) * clamp(5.0 * depth_clean, 0.0, 1.0);
            vec3  nr  = normalize(vec3(nm.x / nm.y + dtl.x, 1.0, nm.z / nm.y + dtl.y));
            vec3  rr  = refract(rd, nr, 1.0 / 1.33);
            if (dot(rr, rr) > 1e-6 && rr.y < -0.05) {
                float tb = depth_clean / -rr.y;
                vec3  pb = wpos;
                float hb = 0.0;
                RM_ONE_COPY for (int i = 0; i < 2; ++i) {      // (twice: the second time to the bed where the ray got to - one copy of the height in the code)
                    pb = wpos + rr * tb;
                    hb = water_only ? meshHeight(pb.xz) : terrainM_lod(pb.xz, t_water);
                    if (i == 0) tb = max(wpos.y - hb, 0.0) / -rr.y;
                }
                if (hb < water_y - 0.01) {
                    wt_refr_ok = true;
                    wt_bed_p = vec3(pb.x, hb, pb.z);
                    wt_d = distance(wpos, wt_bed_p);
                    wt_refr = rr;
                }
            }
        }
        {
            float c  = dot(-rd, sunDirW);
            float cs1 = 0.0542574 * (1.0 + c * c) / pow(1.0625 - 0.5 * c, 1.5);
            float cs2 = 0.0542574 * (1.0 + c * c) / pow(1.0625 + 0.5 * c, 1.5);
            float phase = max(cs1, cs2);
            float ks = wt_sig * (1.5 + 0.5 * c);
            vec3  Lsun = scene.sun_color.rgb * scene.sun_color.a * 2.5 * water_lit * step(0.0, sunDirW.y);
            vec3  Ein  = scene.sky_color.rgb * 1.1;
            // The ambient light on the water, near to far (the reference, WATER_FULL 3.5: its GI cache
            // under 64 m, its sky SH from 192 m, a blend between). Near: the light the lit land
            // beside it gets (the deferred pass's sky term - our GI cache holds objects only). Far:
            // the sky's irradiance at the water's normal, from the physical sky as it is drawn (five
            // taps of the sky-view table round the normal, cosine-weighted) and greyed by the cloud
            // deck as the mirror is - under overcast the far sea goes grey. VIBE_NO_WATER_SKY_AMB=1: off.
            if (scene.water_flags.w > 0.5 && scene.atmo_params.x > 0.5 && t_water > 64.0) {
                vec3  n  = wnor;
                vec3  t1 = normalize(cross(n, abs(n.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)));
                vec3  t2 = cross(n, t1);
                vec3  sum = phys_sky(normalize(vec3(n.x, max(n.y, 0.05), n.z)));
                for (int i = 0; i < 4; ++i) {
                    float a = 1.5707963 * float(i) + 0.4;
                    vec3  d = n * 0.5 + (t1 * cos(a) + t2 * sin(a)) * 0.866;
                    sum += 0.5 * phys_sky(normalize(vec3(d.x, max(d.y, 0.03), d.z)));
                }
                vec3  Ls = sum / 3.0;
                float cov = scene.crater_info.y;
                Ls = mix(Ls, vec3(dot(Ls, vec3(0.2126, 0.7152, 0.0722))) * 1.05, cov * cov);
                Ein = mix(Ein, 3.14159265 * Ls, clamp((t_water - 64.0) / 128.0, 0.0, 1.0));
            }
            vec3  S = wt_sig * wt_alb * (phase * Lsun * (1.0 - exp(-ks * wt_d)) / ks +
                                         0.2572201 * Ein * (1.0 - exp(-2.0 * wt_sig * wt_d)) / (2.0 * wt_sig));
            // THE LIGHT INSIDE THE WATER from the froxel fog volume (the reference: the volume's own
            // in-scatter between the surface and the bed, in the near 3/4 of its slices - so shafts
            // and shadows show in clear water). ADDED to the analytic in-scatter above, never in
            // place of it (replacing it turned the castle lake flat grey-blue): along the refracted
            // path to the bed, four taps of the volume's light - the sun's visibility there (its
            // shadow rays: piers, trees, the bank) against the surface's, and the lamps' light -,
            // each through the water's grey extinction, with the water's albedo.
            if (scene.water_grid.w > 0.5 && pfx_trk.d.fog2.w > 0.5 && wt_d > 0.05) {
                float sl = sqrt(clamp(t_water / max(pfx_trk.d.fog1.w, 1.0), 0.0, 1.0));
                float nk = 1.0 - clamp(4.0 * (sl - 0.75), 0.0, 1.0);
                if (nk > 0.0) {
                    vec3  wdir = wt_refr_ok ? normalize(wt_refr) : rd;
                    float Ld = min(wt_d, 12.0), ds = Ld * 0.25;
                    vec3  L0 = scene.sun_color.rgb * scene.sun_color.a * 2.5 * step(0.0, sunDirW.y);
                    float dvis = 0.0;
                    vec3  loc = vec3(0.0);
                    for (int i = 0; i < 4; ++i) {
                        float s = (float(i) + 0.5) * ds;
                        vec3  li;
                        float vi = wfx_light(wpos + wdir * s, li);
                        dvis += (vi - water_lit) * exp(-ks * s) * ds;
                        loc  += li * exp(-2.0 * wt_sig * s) * ds;
                    }
                    S = max(S + nk * wt_sig * wt_alb * (phase * L0 * dvis + 0.25 * loc), vec3(0.0));
                }
            }
            baseTint = S;
        }
#endif
        vec3 wt_bed_col = vec3(-1.0);   // the bed seen through the water, when worked out (the soft edge)
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
            absorp = mix(absorp, wt_murk.w, wt_riv_k) * (1.0 + 2.0 * wt_riv_B);      // (running water: its murk, thicker where it runs white)
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
#ifndef VIBE_WATER
            float Tw = exp(-wt_sig * wt_d);           // (grey, along the refracted path to the bed: WATER_FULL 3.6)
#else
            float Tw = exp(-depth_m * absorp);
#endif
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
#ifndef VIBE_WATER
                // Where the bent ray meets the bed (the refraction rule above, wt_refr_ok); without it
                // the earlier offset: the bed straight below, shifted by the surface's slopes.
                if (wt_refr_ok) u_pos = wt_bed_p;
                else u_pos.xz -= wnor.xz * (min(depth_clean, 1.5) * 0.6 * clamp(5.0 * depth_clean, 0.0, 1.0));
                // (the bed's stones and caustics were drawn "a little further along the line of sight";
                //  with the refracted ray the point is where it is)
                vec2 u_look = wt_refr_ok ? vec2(0.0) : rd.xz / max(-rd.y, 0.25) * min(depth_m, 1.5) * 0.6;
#else
                vec2 u_look = rd.xz / max(-rd.y, 0.25) * min(depth_m, 1.5) * 0.6;
#endif
                vec3 u_nor = water_only ? meshNormal(u_pos.xz)
                                        : calcNormal(u_pos, t_water);
                vec3 u_mat;
                bool u_baked = false;
#if !defined(VIBE_WATER) && !defined(VIBE_BAKE) && !defined(VIBE_MARCH_ONLY) && !defined(VIBE_LITE)
                // (2026-11-26, the owner: "foggy water, very sharp edges where the water stops". The bed
                //  was getMaterial's bare palette - no scan, no stones, often the lawn's green at the
                //  water level - while the bank beside it is the full ground from the baked ground
                //  texture: a flat green sheet meeting textured soil in a line. The bed now reads the
                //  SAME baked texture (its texels under the water are baked too, flagged 0.5:
                //  rm_bake_fetch_a), so the shallows show the bank's own material running under the
                //  water.)
                if (g_bake_t0 > 0.0) {
                    vec3 bn; float bp0, bp1, bp2;
                    float gsv = g_gnd_ssdm_h;
                    u_baked = rm_bake_fetch_a(u_pos, t_water + wt_d, 0.45, bn, u_mat, bp0, bp1, bp2);
                    g_gnd_ssdm_h = gsv;
                    if (u_baked) u_nor = normalize(mix(u_nor, bn, 0.7));
                }
#endif
                if (!u_baked) u_mat = getMaterial(u_pos, u_nor);
#ifndef VIBE_WATER
                // (no lawn under the water: a green that is grass's turns to silt a few cm down)
                {
                    float gr = clamp((u_mat.g - max(u_mat.r, u_mat.b)) * 10.0, 0.0, 1.0) * smoothstep(0.03, 0.25, depth_clean);
                    float lu = dot(u_mat, vec3(0.2126, 0.7152, 0.0722));
                    u_mat = mix(u_mat, lu * vec3(1.25, 1.0, 0.70), gr);
                }
#endif
#ifndef VIBE_WATER
                // The bed is WET: a submerged stone or soil is darker and a little more saturated than
                // the same dry (water fills its pores - about half the albedo; the reference retints
                // submerged ground and forces it wet). Dry, the stream's bed of pale stones under 20 cm of
                // clear water was brighter than the meadow beside it: the stream read as a pale sheet.
                if (SW_ON(1)) {
                    // (the reference: below the waterline the wet value is 1 - shore_wet.glsl's wet albedo)
                    u_mat = mix(u_mat, sw_wet_albedo(u_mat, 1.0), smoothstep(-0.02, 0.06, depth_clean));
                } else {
                u_mat = mix(vec3(dot(u_mat, vec3(0.3, 0.5, 0.2))), u_mat, 1.25) * 0.55 * smoothstep(-0.02, 0.06, depth_clean) + u_mat * (1.0 - smoothstep(-0.02, 0.06, depth_clean));
                }
                u_mat = max(u_mat, vec3(0.0));
#endif
                float bed_spec = 0.0;
                float bed_hn = 0.5;          // (the bed's own height 0..1 here, for the growth's hollows)
                bool  wb_done = false;
#ifndef VIBE_WATER
                // THE BED FROM SCANNED SETS (water_bed.glsl, slots 43..46; VIBE_NO_BED_SCAN=1: the
                // code-made pebbles and weed below): river stones where the water runs fast, silt in
                // slow and deep water, rock where the bed is steep, the growth scan on the up-facing
                // bed; the stones' relief marched along the refracted ray from their heights; the
                // last centimetres at the edge keep the bank's own material (no line at the water).
                if (wb_on()) {
                    wb_done = true;
                    float bfp = max(t_water, 0.05) * 2.0 * scene.viewport.w;      // ~metres per pixel
                    float below0 = water_y - u_pos.y;
                    float spd = length(wt_riv_flow);
                    WbW W = wb_weights(u_pos.xz, spd, below0, u_nor.y, bfp);
                    vec2 bp = u_pos.xz + u_look;
                    if (SW_ON(32) && bfp < 0.03 && depth_clean > 0.02) {
                        vec3  rdw = wt_refr_ok ? normalize(wt_refr) : rd;
                        const float kHmax = 0.09;
                        vec2  off = rdw.xz / max(-rdw.y, 0.2) * kHmax;
                        off *= 1.0 / max(1.0, length(off) / 0.45);
                        vec2  q0 = bp - off;
                        float sp = 0.0, fp = kHmax;
                        for (int i = 1; i <= 5; ++i) {
                            float s = float(i) / 5.0;
                            float f = kHmax * (1.0 - s) - wb_height(q0 + off * s, W);
                            if (f <= 0.0) { bp = q0 + off * mix(sp, s, fp / max(fp - f, 1e-5)); break; }
                            sp = s; fp = f;
                        }
                    }
                    WbMat M = wb_material(bp, W);
                    float gM = sw_growth(bp, M.hn, u_nor.y, below0) * (1.0 - 0.7 * smoothstep(0.6, 1.5, spd));      // (fast water scours it)
                    wb_growth(M, bp, W.lod, gM);
                    bed_hn = M.hn;
                    float bk = smoothstep(0.01, 0.15, depth_clean);
                    u_mat = mix(u_mat, SW_ON(1) ? sw_wet_albedo(M.alb, 1.0) : M.alb * 0.6, bk);
                    u_nor = normalize(u_nor + vec3(M.tn.x, 0.0, -M.tn.y) * bk);
                    float rgh = M.rough * 0.6;       // (wet: smoother than the dry scan)
                    vec3  bh = normalize(sunDirW - rd);
                    bed_spec = pow(max(dot(u_nor, bh), 0.0), mix(160.0, 12.0, rgh)) * 0.5 * (1.0 - rgh) * water_lit * bk;
                    float rk;
                    u_mat = sw_bed_tint(u_mat, below0, wt_alb, rk);
                    bed_spec *= 1.0 - 0.6 * rk;
                }
#endif
                if (!wb_done) {
                    float bfp = max(t_water, 0.05) * 2.0 * scene.viewport.w;      // ~metres per pixel
                    float bwA = smoothstep(2.0, 5.0, 0.22 / bfp);
                    float bwB = smoothstep(2.0, 5.0, 0.075 / bfp);
                    if (bwA > 0.01) {
                        // Where the bed is seen through the water it lies a little
                        // further along the line of sight than straight below.
                        vec2  bp = u_pos.xz + u_look;
                        float be = max(0.004, 1.5 * bfp), bid, bk, tmp1, tmp2;
#ifndef VIBE_WATER
                        // THE BED'S RELIEF under clear shallows (the reference gets it from displacing
                        // the bed in its G-buffer; here the bed is shaded in the water pixel): a short
                        // march of the stones' height along the refracted ray - from where the ray is
                        // as high as the tallest stone down to the bed's base - so the stones stand up
                        // and the near ones hide what lies behind them. 5 steps + one linear refine.
                        if (SW_ON(32) && bwA > 0.2 && depth_clean > 0.02) {
                            vec3  rdw = wt_refr_ok ? normalize(wt_refr) : rd;
                            const float kHmax = 0.12;
                            vec2  off = rdw.xz / max(-rdw.y, 0.2) * (kHmax * bwA);
                            off *= 1.0 / max(1.0, length(off) / 0.45);
                            vec2  q0 = bp - off;
                            float sp = 0.0, fp = kHmax * bwA;               // (the ray's height over the base at the last step)
                            for (int i = 1; i <= 5; ++i) {
                                float s = float(i) / 5.0;
                                float hb = bed_h(q0 + off * s, bwA, bwB, tmp1, tmp2);
                                float f = kHmax * bwA * (1.0 - s) - hb;
                                if (f <= 0.0) {
                                    float sh = mix(sp, s, fp / max(fp - f, 1e-5));
                                    bp = q0 + off * sh;
                                    break;
                                }
                                sp = s; fp = f;
                            }
                        }
#endif
                        float hR = bed_h(bp + vec2(be, 0.0), bwA, bwB, tmp1, tmp2);
                        float hU = bed_h(bp + vec2(0.0, be), bwA, bwB, tmp1, tmp2);
                        float hC = bed_h(bp, bwA, bwB, bid, bk);
                        bed_hn = clamp(hC / 0.10, 0.0, 1.0);
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
#ifndef VIBE_WATER
                // GROWTH UNDER WATER (the reference's submerged "moss", shore_wet.glsl sw_growth):
                // weed clumps on the up-facing bed below ~0.35 m, in the stones' hollows first;
                // then THE BED'S TINT by depth - toward the water's own albedo, its brightness
                // given back (the extinction does the darkening). The bed is not in the G-buffer,
                // so the growth's height reaches the displacement only where the ground is seen
                // directly under water (terrain_rm_main_light.glsl bed_sub).
                if (!wb_done) {
                    float below = water_y - u_pos.y;
                    float gM = sw_growth(u_pos.xz, bed_hn, u_nor.y, below);
                    if (gM > 0.0) {
                        float gh;
                        vec3  gc = sw_growth_col(u_pos.xz, gh);
                        u_mat = mix(u_mat, gc, gM);
                        bed_spec *= 1.0 - gM;
                        u_nor = normalize(mix(u_nor, vec3(0.0, 1.0, 0.0), 0.3 * gM));
                    }
                    float rk;
                    u_mat = sw_bed_tint(u_mat, below, wt_alb, rk);
                    bed_spec *= 1.0 - 0.6 * rk;        // (the roughness toward a fixed 0.3: a softer highlight)
                }
#endif
                float u_dif = max(dot(u_nor, sunDirW), 0.0) * water_lit;
#ifndef VIBE_WATER
                // (the sun reaches the bed through the water too: grey extinction along its refracted path)
                u_dif *= exp(-wt_sig * depth_clean / max(sqrt(max(1.0 - (1.0 - sunDirW.y * sunDirW.y) / 1.769, 0.0)), 0.2));
#endif
                // Caustics on the bed (the reference game splats them from the water's own
                // normals; here the pattern compose.frag shows under water, wt_caustic): the
                // waves focus the sun into a moving net on the bed of a clear shallow. Strongest
                // a little under the surface, gone by ~15 m, washed out by a wind-roughened
                // surface, by distance (sub-pixel lines) and by shade (water_lit is in u_dif).
                {
                    float cfp = max(t_water, 0.05) * 2.0 * scene.viewport.w;
                    float ck = clamp(sunDirW.y * 1.4, 0.0, 1.0) * exp(-depth_m * 0.16) * smoothstep(0.0, 0.25, depth_m) *
                               (1.0 - smoothstep(0.03, 0.12, cfp)) * (1.0 - 0.65 * smoothstep(5.0, 20.0, wt_W)) * clamp(trans_amt * 1.5, 0.0, 1.0);
                    if (ck > 0.002) {
                        vec2 cp = u_pos.xz + u_look +
                                  sunDirW.xz / max(sunDirW.y, 0.25) * depth_m;
                        float cc = wt_caustic(cp * 0.42, wave_t * 0.9 + 23.0);
                        u_dif *= 1.0 + (cc * 2.4 - 0.30) * ck * scene.water_sun.x;
                    }
                }
                float u_amb = 0.5 + 0.5 * u_nor.y;
                vec3  u_col = u_mat * (u_dif * scene.sun_color.rgb *
                                        scene.sun_color.a * 0.8 +
                                        u_amb * scene.sky_color.rgb * 0.35) +
                             bed_spec * scene.sun_color.rgb * scene.sun_color.a;
                wt_bed_col = u_col;        // (the soft edge: the bed as it is, untinted)
#ifndef VIBE_WATER
                wt_guide_alb = mix(wt_alb, u_mat, clamp(Tw * trans_amt, 0.0, 1.0));
                wt_guide_n = u_nor;
#endif
                // Tint the underwater colour by the water tint so it
                // reads as "looking through water", not "no water".
#ifdef VIBE_WATER
                u_col *= mix(vec3(1.0), shallow * 1.5 + vec3(0.05),
                              0.5);
#endif
#ifndef VIBE_WATER
                // Blend: shallow → underwater visible, deep → opaque
                // water_color. trans_amt scales the maximum showthrough.
#else /* VIBE_WATER */
                // Beer's-law transmittance (depth fade governed by the
                // clarity-depth knob) × trans_amt (master strength). At
                // Transparency=1 + small clarity depth: shallow reads
                // crystal-clear, deep water stays fully opaque.
#endif /* VIBE_WATER */
#ifndef VIBE_WATER
                baseTint += u_col * (Tw * trans_amt);      // (the bed, added: no (1 - F) on what comes through)
#else
                baseTint = mix(baseTint, u_col, Tw * trans_amt);
#endif
            }
        }

        // What a facet turned away from the eye mirrors: the water further on (wt_rdown above).
        // (On still water - a lake, the sea - seen low across it, a ray turned down by a ripple meets the
        //  next wave's face and is reflected up again at a grazing Fresnel near 1: the mirror there
        //  stays nearly the mirror. Turned to the water's own dark colour, a ripple field's trough seen
        //  from a swimmer's eye was a black hole: the in-scatter is dark in a deep lake. Running water
        //  keeps the water's colour - its tilted facets mirrored a pale haze over the whole stream.)
        reflCol = mix(reflCol, mix(baseTint, reflCol * 0.8, 1.0 - wt_riv_k), wt_rdown);
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
#ifndef VIBE_WATER
        float fres_eff = fres * wt_edge;      // (the reference: only the soft edge takes the reflection away)
        // Foam is an albedo, lit (the reference: GI + the light in the air at the surface) - the
        // sun on it and the sky.
        vec3 wt_white = scene.sun_color.rgb * scene.sun_color.a * max(sunDirW.y, 0.0) * (0.25 + 0.75 * water_lit) * 0.8 + scene.sky_color.rgb * 0.45;
#else
        float fres_eff = fres * mix(0.5, 1.0, water_blend) * wt_edge;      // (wt_edge: the soft water edge)
        vec3 wt_white = vec3(1.0);
#endif
        float foam_cover = 0.0;   // how much opaque foam sits on the surface here
        float dbg_wcap = 0.0, dbg_rip = 0.0, dbg_band0 = 0.0;      // (the white-term debug views below)
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
            float band   = wt_in_stream ? 0.0 : exp(-depth_clean * (4.0 / foam_w));      // (the lapping shore band is a lake's and the sea's: a stream is all shallow - its own foam is above)
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
                // (the lapping lines are a lake's and the sea's: on the river, whose banks shelve
                //  gently for metres, they followed the depth contours far out from the bank as
                //  long thin white arcs - the river keeps the edge line only)
                band *= max(edge, lap * mix(0.35, 1.0, patchy) * (1.0 - wt_riv_k));
                // (2026-11-26, the owner: the lace along every bank read as a hard white line. A lake's
                //  and the river's lace is thin, broken and only where there are ripples to make it:
                //  half of it in a breeze, none on still water; the sea keeps its surf foam.)
                band *= mix(0.5 * smoothstep(0.2, 0.8, wt_lake()) * mix(0.25, 1.0, patchy), 1.0, wt_seak);
                // (the surf's fronts and the foam they leave, the river's streaks: the same lace)
                band = max(band, wt_surf_foam);      // (the river's streaks: laced below, carried by the current; the stream's white water is its own, further down)
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
                // The river's foam streaks and lines through a lace CARRIED by the current (two phases,
                // wt_foam_lace) - laced with the shore's slowly drifting lace above, the bubbles stood
                // still while the streak moved under them.
                if (!wt_in_stream && wt_riv_foam > 0.002 && wt_riv_k > 0.001) {
                    vec2 rfl = mix(wt_riv_flow / wt_riv_k, wt_sw.vel, wt_sw.k);
                    band = max(band, wt_foam_lace(wpos.xz, rfl, wt_riv_foam, wave_t, max(t_water, 0.05) * 2.0 * scene.viewport.w));
                }
            }
            // River style: blend in the flow-driven foam pattern on top
            // of the shore band. pm_foam_tex was set in the river branch
            // above (1.0 when river style is off, so no effect).
            float flow_foam = clamp(1.0 - pm_foam_tex, 0.0, 1.0);
            dbg_band0 = band;
            float foam_mix = max(band, flow_foam);
#ifdef VIBE_WATER
            foam_amt = foam_mix;
#endif /* VIBE_WATER */
#ifndef VIBE_WATER
            foam_cover = foam_mix * scene.water_foam_color.a;
#else /* VIBE_WATER */
            foam_cover = foam_mix * scene.water_foam_color.a * foam_op;
#endif /* VIBE_WATER */
            // (the soft edge's foam term, the reference's 1 - e^(-4 gap) over its last 8 cm: no foam
            //  drawn on the dry edge; the normal, the extinction and the refraction already fade
            //  through wt_edge - the flattened normal, the depth, the bed mixed in at the end)
#ifndef RM_NO_CALM
            foam_cover *= 1.0 - exp(-4.0 * wt_depth / 0.08);
#endif
            baseTint = mix(baseTint, scene.water_foam_color.rgb * wt_white, foam_cover);
        }
        // Reflection goes over the tinted water; foam is matte, so it
        // suppresses most (not all - it is wet) of the mirror.
        // The water body's own colour is light scattered back out of it:
        // it follows the light level (dark at night, dull under overcast).
#ifndef VIBE_WATER
        // In-scatter + bed, through the surface with what the Fresnel lets through, + reflection x F.
        // (The reference adds the water's own light at full strength - no (1 - F). Looking along a
        // lake from its bank F is 0.5 - 0.8, and the lake's own bright green then lay over the
        // mirror at full strength: the castle lake was a flat pale green with no mirror at all.
        // Foam sits on the surface: it is not dimmed.)
        vec3 col_w = mix(baseTint * (1.0 - fres_eff), baseTint, foam_cover) + reflCol * (fres_eff * (1.0 - 0.7 * foam_cover));
#else
        baseTint *= pow(clamp(scene.sun_color.a * 0.4, 0.0, 1.0), 0.6);
        vec3 col_w = mix(baseTint, reflCol, fres_eff * (1.0 - 0.7 * foam_cover));
#endif
        if (ice_k > 0.001) {
            vec3 ia = ice_albedo(wpos.xz, rd, wnor, max(scene.water_params.y - terrain_y, 0.0));
            vec3 il = ia * (scene.sun_color.rgb * scene.sun_color.a * max(sunDirW.y, 0.0) * water_lit * 0.9 + scene.sky_color.rgb * 0.55);
            col_w = mix(col_w, mix(il, reflCol, fres_eff * 0.55), ice_k);
        }
        col_w += scene.sun_color.rgb * scene.sun_color.a * spec *
                  0.8 * water_lit;
#if !defined(VIBE_WATER) && !defined(VIBE_LITE)
        // LAMPS GLINTING ON THE WATER (the reference: one local light a pixel, picked by its weight,
        // the same mirror GGX as the sun, a short visibility test). The lamps the terrain pass has:
        // scene.lamp_lights (the lanterns and street lamps nearest the eye; the deferred pass's lamp
        // list, binding 26, is not in this pass's set). Every lamp's glint is weighed; one is picked
        // with a chance by its weight, and the sum is kept where a short ray (<= 10 m) toward it is
        // free - an estimate of the shadowed sum whose noise the upscaler averages. The lobe is the
        // sun's widened by the lamp's own size (0.17 m) as seen from the water.
        if (scene.water_grid.z > 0.5 && scene.lamp_info.y > 0.5 && wt_edge > 0.01) {
            vec3  dnx = dFdx(wnor), dny = dFdy(wnor);
            float a2b = 1.6e-7 + 2.0 * clamp(0.5 * (dot(dnx, dnx) + dot(dny, dny)), 0.0, 0.25) + 0.15 * (0.003 + 0.00512 * wt_W) + 2.0 * wt_fine_var;
            float NV = max(cosV, 1e-3);
            vec3  lsum = vec3(0.0);
            float wsum = 0.0, pick_d = 0.0;
            vec3  pick_l = vec3(0.0);
            float rnd = fract(sin(dot(gl_FragCoord.xy + float(scene.rt_flags.w & 63) * vec2(7.13, 3.71), vec2(12.9898, 78.233))) * 43758.5453);
            for (int li = 0, li_n = int(scene.lamp_info.y + 0.5); li < li_n; ++li) {
                float lI = scene.lamp_lights[li].w;
                if (lI < 1e-3) continue;
                vec3  to = scene.lamp_lights[li].xyz - wpos;
                float d = length(to);
                if (d >= 60.0 || d < 0.2) continue;
                vec3  L = to / d;
                float NL = max(dot(wnor, L), 0.0);
                if (NL <= 0.0) continue;
                vec3  Hl = normalize(L - rd);
                float a2 = a2b + 0.03 / (d * d);                    // (the lamp's glass, 0.17 m, seen from here)
                float NH = max(dot(wnor, Hl), 0.0);
                float dd = 1.0 + (a2 - 1.0) * NH * NH;
                float D = a2 / (3.14159265 * dd * dd);
                float al = sqrt(a2);
                float Vis = 0.5 / max(NL * (NV * (1.0 - al) + al) + NV * (NL * (1.0 - al) + al), 1e-5);
                float Fh = 0.04 + 0.96 * pow(1.0 - clamp(dot(-rd, Hl), 0.0, 1.0), 5.0);
                float fall = 1.0 - d / 60.0; fall *= fall;
                // (the ground's lantern light is lI x fall over ~34 m; a point light's glint goes as
                //  1 / d^2 - kept finite near the glass)
                vec3  c = vec3(1.0, 0.62, 0.28) * lI * fall * min(NL * D * Vis * Fh / max(d * d, 1.0), 2000.0);
                float w = dot(c, vec3(0.2126, 0.7152, 0.0722));
                if (w <= 0.0) continue;
                lsum += c;
                wsum += w;
                if (rnd * wsum < w) { pick_l = L; pick_d = d; }      // (a running pick: chance w / sum)
            }
            if (wsum > 1e-6 && pick_d > 0.0) {
                bool blocked = any_hit_no_terrain(wpos + vec3(0.0, 0.05, 0.0), pick_l, min(max(pick_d - 0.45, 0.05), 10.0));
                if (!blocked) col_w += lsum * wt_edge;
            }
        }
#endif
        // Subtle darkening of the surface tint when in shadow.
#ifdef VIBE_WATER
        col_w *= mix(0.7, 1.0, water_lit);
#endif

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
            vec2 fp = wpos.xz * 0.55 - scene.water_swell.yz * 0.35;          // (drifting down the wind)
            float fn = noise2(fp) * 0.6 + noise2(fp * 2.13) * 0.4;
            // Whitecaps come with the wind: on the sea from about 3 m/s (more and more of them
            // in a gale), on a lake or the river not before about 7.
            float wcap = mix(smoothstep(6.0, 16.0, wt_W), smoothstep(2.5, 9.0, wt_W) * (1.0 + 0.4 * smoothstep(12.0, 26.0, wt_W)), wt_seak);
            float foam = smoothstep(0.45, 0.85, fn * steepness * 1.6 * wcap);
            float foam_lit = water_lit;
#ifndef VIBE_WATER
            // The open sea's whitecaps come from the FOLD of the FFT cascades (the reference:
            // saturate(fold^4 x foam texture - 0.01)): where the choppy surface crowds together and
            // folds over, i.e. on the steepest crests - more of them the stronger the wind, as the
            // spectrum steepens (water_sim.cpp: its slope follows the Cox-Munk wind law). The foam
            // texture: the reference's two taps at 0.06 and 0.102 /m drifting with the wind, and a
            // finer lace. Foam is an albedo lit by the sun and the sky (wt_white): in shade it greys,
            // it does not vanish.
            if (ww_fft_on() && wt_seak > 0.01) {
                vec2  dr = scene.water_swell.yz;
                float tex = 0.35 * noise2(wpos.xz * 0.06 - dr * 0.025 * 0.06) + 0.35 * noise2(wpos.xz * 0.102 - dr * 0.05 * 0.102 + 5.0) + 0.3 * fn;
                float f = clamp(wt_fold, 0.0, 1.0);
                float f4 = f * f; f4 *= f4;
                float fold_foam = clamp(f4 * tex * 2.0 - 0.01, 0.0, 1.0);
                // (broken into lace: with the gale's 10 - 40 m waves a fold covers metres, and the foam on it
                //  was one wide, even, pale smear)
                fold_foam *= smoothstep(0.30, 0.62, fn) * 0.9;
                // Far out the mips average the fold away (its mean is 0) and the whitecaps with it;
                // there the share of the sea they cover is laid in instead - Monahan's law,
                // 3.84e-6 U^3.41 (0.2 % at 6 m/s, 7 % at 18) - as patches of the same texture.
                float cov = clamp(3.84e-6 * pow(max(wt_W, 0.0), 3.41), 0.0, 0.25);
                float far_k = smoothstep(0.4, 2.5, length(wt_gy));
                float thr = 0.5 + 0.12 * sqrt(-2.0 * log(max(cov * 2.5, 1e-4)));       // (the texture is about normal, mean 0.5, sd 0.12)
                float far_foam = cov > 1e-4 ? smoothstep(thr, thr + 0.08, tex) * 0.8 : 0.0;
                fold_foam = max(fold_foam * (1.0 - far_k), far_foam * far_k);
                // Streaks of foam drawn out along the wind (a gale, Beaufort 7 and up: the foam the
                // breaking crests leave is pulled into lanes down the wind): a noise stretched 25:1
                // along the wind, drifting with it, thresholded, from about 12 m/s.
                {
                    vec2  wa = scene.water_wind.xy, wc = vec2(-wa.y, wa.x);
                    vec2  q = vec2(dot(wpos.xz, wa), dot(wpos.xz, wc)) - vec2(dot(dr, wa) * 0.3, 0.0);
                    q.y += 9.0 * (noise2(q * vec2(0.004, 0.02)) - 0.5);      // (lanes wander: seen down the wind, straight ones were ruled vertical lines on the screen)
                    float st = 0.6 * noise2(vec2(q.x * 0.02, q.y * 0.9)) + 0.4 * noise2(vec2(q.x * 0.05, q.y * 2.2) + 7.0);
                    // (narrow lanes broken into lace: wide and smooth they read as milky sheets, not foam)
                    float lanes = smoothstep(0.62, 0.74, st) * smoothstep(11.0, 20.0, wt_W) * smoothstep(0.35, 0.65, fn) * (0.6 + 0.4 * smoothstep(0.3, 0.7, tex));
                    fold_foam = max(fold_foam, lanes * 0.7);
                }
                foam = mix(foam, fold_foam, wt_seak);
                foam_lit = 1.0;
            }
#endif
            dbg_wcap = foam;
#ifdef VIBE_WATER
            foam_amt = max(foam_amt, foam);
#endif /* VIBE_WATER */
            col_w = mix(col_w, vec3(0.95, 0.97, 1.0) * wt_white,
#ifndef VIBE_WATER
                        foam * 0.85 * foam_lit);
#else /* VIBE_WATER */
                        foam * 0.85 * water_lit * foam_op);
#endif /* VIBE_WATER */
        }

        // White water: the stream's steps and rapids and the river's fastest reaches are covered
        // in it (wt_riv_foam, the river block above) - as opaque froth, not through the thin
        // shore-foam lace (whose strength is a setting for the lapping at a lake's edge).
        float wt_dbg_white = 0.0;
        if ((wt_riv_k > 0.001 && (wt_in_stream || wt_riv_B > 0.02 || wt_obs_foam > 0.01)) || wt_sw_white > 0.001) {
            float white = clamp(max(wt_obs_foam, (wt_riv_k > 0.001 && (wt_in_stream || wt_riv_B > 0.02)) ? wt_riv_foam * (wt_in_stream ? 1.0 : 0.8) : 0.0), 0.0, 1.0);
            white = max(white, wt_sw_white);        // (the shallow-water solver's foam)
            // (the amount through the foam lace: bubble walls and open water, not smooth opaque
            //  blobs - terrain_rm_water.glsl wt_foam_lace)
            {
                vec2 lfl = wt_riv_k > 0.001 ? wt_riv_flow / wt_riv_k : wt_sw.vel;
                // (inside the solver's window the lace rides the solver's own velocity - the foam
                //  it laces is carried by it: a lace on the course's current slid straight over
                //  foam that bends round the stones)
                lfl = mix(lfl, wt_sw.vel, wt_sw.k);
                white = wt_foam_lace(wpos.xz, lfl, white, wave_t, max(t_water, 0.05) * 2.0 * scene.viewport.w);
            }
            wt_dbg_white = white;
#ifdef VIBE_WATER
            foam_amt = max(foam_amt, white);
#endif
            vec3 froth = vec3(0.90, 0.93, 0.95) * (scene.sun_color.rgb * scene.sun_color.a * max(sunDirW.y, 0.0) * (0.25 + 0.75 * water_lit) * 0.55 +
                                                    scene.sky_color.rgb * 0.45);
            col_w = mix(col_w, froth * (0.85 + 0.3 * clamp(wnor.y, 0.0, 1.0)), white * 0.85);
        }
        col_w *= 1.0 - 0.35 * wt_obs_hole;      // (the hole behind a rock: the water drops into it, dark)
        // Churned, aerated patch where something just hit the water.
        if (rip_w.z > 0.001) {
            vec2 fq = wpos.xz * 7.0;
            float fnz = noise2(fq) * 0.6 + noise2(fq * 2.3 + 3.1) * 0.4;
            float rf_amt = rip_w.z * smoothstep(0.25, 0.75, fnz + rip_w.z * 0.5);
            dbg_rip = rf_amt;
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

#ifndef VIBE_WATER
        // The soft water edge: over the last centimetres the bed itself (the mesh terrain's
        // water plane fades by its alpha instead, below).
        if (wt_bed_col.x >= 0.0 && wt_edge < 1.0) col_w = mix(wt_bed_col, col_w, wt_edge);
#endif
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
        if (RMDBG(8388608)) col_w = vec3(wt_obs_foam, wt_dbg_white, wt_riv_B);      // (DEBUG water terms)
        // (VIBE_DEBUG_SOLVER=1: the shallow-water solver - r the surface over the authored top x 10 (red up,
        //  blue down), g its foam, and the speed's stripes every 0.5 m/s faintly)
        if (scene.water_sw2.y > 0.5 && wt_sw.k > 0.0)
            col_w = mix(vec3(clamp(wt_sw.dh * 10.0, 0.0, 1.0), 0.0, clamp(-wt_sw.dh * 10.0, 0.0, 1.0)) + 0.15 * fract(length(wt_sw.vel) * 2.0),
                        vec3(0.1, 1.0, 0.1), clamp(wt_sw.foam * 1.5, 0.0, 1.0));
        if (RMDBG(16777216)) col_w = vec3(clamp(0.5 + 3.0 * wt_dbg_n.x, 0.0, 1.0), clamp(0.5 + 3.0 * wt_dbg_n.y, 0.0, 1.0), 0.0);      // (view: the running water's carried pattern alone - to measure how it moves)
        if (RMDBG(8388608) && RMDBG(16777216)) col_w = wt_dbg3;      // (view: the stream's white by its source - r fast-water foam, g a sheet's rough water, b the streaks)
        if (RMDBG(33554432)) col_w = reflCol;      // (view: what the water mirrors, before the Fresnel)
        // (view: the stream's water by its course segment - r, g from the segment's number, b how far along it;
        //  white where it is not the stream: where the water's footprint ends, which segment drew it)
        if (RMDBG(134217728)) col_w = wt_in_stream ? vec3(fract(floor(wt_st.seg) * 0.37), fract(floor(wt_st.seg) * 0.61), fract(wt_st.seg)) : vec3(1.0);
        // (the white terms one by one: with 33554432 r the shore band / lace, g the foam cover (band and
        //  the flow pattern), b the wave-peak / whitecap foam; with 67108864 r the white water (river,
        //  obstacles, solver), g the solver's foam alone, b the ripple churn)
        if (RMDBG(134217728) && RMDBG(33554432)) col_w = vec3(dbg_band0, foam_cover, dbg_wcap);
        if (RMDBG(134217728) && RMDBG(67108864)) col_w = vec3(wt_dbg_white, wt_sw_white, dbg_rip);
        outColor = vec4(col_w, 1.0);
        // The G-buffer of a water pixel: Ray Reconstruction's guides are the refracted bed's (the
        // reference's, WATER_FULL 3.11) - its albedo in rgb (the deferred pass passes material 8's
        // lit colour on from the scene colour and does not read this), its world normal in xy
        // (octahedral), and 0.66 in b marks water for rr_guides.comp and the upscalers (below the
        // ground's 0.7 + relief that ssdm_ground.frag reads, apart from the 0.6 of the enlarged ground).
        {
            vec3  gn = wt_guide_n / (abs(wt_guide_n.x) + abs(wt_guide_n.y) + abs(wt_guide_n.z));
            vec2  oe = gn.z >= 0.0 ? gn.xy : (1.0 - abs(gn.yx)) * vec2(gn.x >= 0.0 ? 1.0 : -1.0, gn.y >= 0.0 ? 1.0 : -1.0);
            outGBuffer0 = vec4(clamp(wt_guide_alb, vec3(0.0), vec3(1.0)), 8.0 / 255.0);
            outGBuffer1 = vec4(oe * 0.5 + 0.5, 0.66, 0.0);
            if (cam_under) { outGBuffer0.rgb = clamp(col_w, vec3(0.0), vec3(1.0)); outGBuffer1 = vec4(0.5, 0.5, 0.7, 0.0); }
        }
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
#endif /* ! VIBE_GROUND */
    }
