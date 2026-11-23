// Part of main() in terrain_raymarch.frag: bounce light (path-traced GI), the lamps and other local lights, the final colour.
// Not a file of its own - it is included in the middle of main() and uses its local
// variables (ro, rd, t, pos, nor, col, ...). Split off only to keep that file readable;
// the order of the parts is the order of the #include lines there.

    // === Path-traced GI on the raymarched terrain =================
    // Mirrors cube.frag's GI loop: stratified cosine-hemisphere
    // samples, N_bounces of throughput tracking, sun-shadow rays at
    // bounce hits up to gi_shadow_max_bounce, sky fallback on miss.
    // The terrain BLAS is skipped (mask 0x02) for the bounce closest
    // hits — we only want castle/dyn-prop bounce contributions; a
    // procedural-terrain self-bounce would need running getMaterial
    // at the hit point which is much more expensive than the BLAS
    // material lookup. The sun-shadow ray FROM a bounce hit uses
    // mask 0x01 (the standard shadow-caster mask) so the terrain
    // CAN occlude the sun for a castle wall hit deep in a valley.
    //
    // Energy: cosine-weighted sampling makes each hit's contribution
    // = albedo * Li (the BRDF/π and cos/π factors cancel). We tag
    // the raw GI with the surface albedo (`mate`) at the end — same
    // structure as cube.frag — and lerp toward a smooth sky-at-N
    // estimate by `gi_softener` for low-sample noise control.
    vec3 gi_indirect = vec3(0.0);
    // Terrain GI sample count capped (was scene.rt_flags2.x directly).
    // 17+ samples * (1 BLAS hit + bounce shadow ray) per pixel was the
    // dominant per-frame RT cost and the smoking gun for the GPU TDR
    // observed during long idle frames as the TLAS grew with streamed
    // dyn-props. Slider still controls; we just clamp to 6 for the
    // raymarched terrain path. Cube.frag GI uses the full slider.
    int N_gi_user = scene.rt_flags2.x;
    int N_gi_cap  = max(0, int(scene.terrain_rt_extra.y));
    // Distance-LOD on sample count: rt_lod_t goes 1→0 across the
    // RT-LOD distance band. Scale the GI sample count smoothly so
    // mid-distance pixels pay 4 rays instead of the full 16, and
    // pixels just inside the LOD cliff pay 1. Avoids the binary
    // "full N or skip" cliff that produced visible noise rings.
    int N_gi_full = (N_gi_user > 0) ? min(N_gi_user, N_gi_cap) : 0;
    int N_gi = (N_gi_full > 0 && do_rt)
                ? max(1, int(ceil(float(N_gi_full) * rt_lod_t)))
                : 0;
    // Under DLSS Ray Reconstruction (rt_flags.x bit 1) the seed changes every frame and the
    // denoiser averages over time: half the rays a frame, as for the sun's shadow rays.
    if ((scene.rt_flags.x & 2) != 0) N_gi = (N_gi + 1) / 2;
    // Same pattern as the PCSS gate: gi_indirect is multiplied by
    // scene.rt_params2.x (the GI strength slider) at line ~2313 so
    // strength=0 makes the whole loop's output zero. Bail before the
    // ray fire instead of burning N_gi × N_bounces ray-queries to
    // produce a value that's about to be multiplied away.
    if (N_gi > 0 && scene.rt_params2.x > 1e-3) {
        int N_bounces     = max(1, int(scene.terrain_rt_extra.w));
        int gi_shadow_max = int(scene.rt_lod.z);
        // GI search radius — driven directly by the user's slider.
        // Earlier we forced a min of 80m so first-bounce rays could
        // reach distant castle walls, but combined with N_gi samples
        // the long-ray BLAS traversal pushed total per-frame work
        // over the GPU TDR threshold. Crank gi_radius in the UI to
        // 80m+ if you want stronger color bleed.
        float gi_radius   = max(scene.rt_params2.y, 1.0);
        // Soft sky fill for bounce hits that can't see the sun. Same
        // 5%-of-sky constant as cube.frag — keeps interior bounces
        // believably dark instead of glowing from a too-bright fill.
        vec3 sky_fill = scene.sky_color.rgb * 0.05;

        // Bounce rays and the sun ray from where they land take leaf cards as solid cards
        // (settings 'foliage rays' below 'every leaf', and not in the leafless season - as the
        // far shadow ray does): in a wood each of them asked the leaf texture at every card it
        // crossed (0.7 ms of the 7.8 ms terrain pass there; the picture: 0.09 grey levels).
        bool gi_solid = scene.season.w <= 0.02 && ((scene.rt_flags.x >> 2) & 3) >= 1;
        int strata = int(ceil(sqrt(float(N_gi))));
        float inv_strata = 1.0 / float(strata);
        int  taken = 0;
        int  first_bounce_misses = 0;
        vec3 sum   = vec3(0.0);
        uvec3 gi_seed = uvec3(gl_FragCoord.xy, scene.rt_flags.w);

        // Tangent basis around `nor` is constant across all GI samples
        // — precompute once, pass to cos_hemi_basis() to skip the
        // per-sample cross+normalize+cross.
        vec3 gi_up_a = abs(nor.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                          : vec3(1.0, 0.0, 0.0);
        vec3 gi_t    = normalize(cross(gi_up_a, nor));
        vec3 gi_b    = cross(nor, gi_t);

        for (int sy = 0; sy < strata && taken < N_gi; ++sy) {
            for (int sx = 0; sx < strata && taken < N_gi; ++sx) {
                float r1 = rmRand(gi_seed + uvec3(taken, 73u, 11u));
                float r2 = rmRand(gi_seed + uvec3(taken, 91u, 47u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;

                vec3 ray_dir    = cos_hemi_basis(u1, u2, nor, gi_t, gi_b);
                vec3 ray_origin = pos + nor * 0.05;
                vec3 throughput = vec3(1.0);
                vec3 path       = vec3(0.0);

                for (int b = 0; b < N_bounces; ++b) {
                    float t_hit;
                    int   inst_id;
                    int   prim_id;
                    g_rq_leaf_solid = gi_solid;
                    bool gi_hit = closest_hit_material_no_terrain(ray_origin, ray_dir, gi_radius, t_hit, inst_id, prim_id);
                    g_rq_leaf_solid = false;
                    if (!gi_hit) {
                        // BLAS miss → sky in this direction. Bound by
                        // the halo's pow(8) so no fireflies. Terrain
                        // self-occlusion is intentionally NOT checked
                        // here — for short gi_radius (~20 m) on flat
                        // terrain the heightfield march false-blocks
                        // sky-bound rays, killing the GI signal. AO
                        // already carries terrain occlusion via the
                        // ambient term; GI keeps its sky-escape
                        // contribution undamped.
                        path += throughput * sample_sky(ray_dir);
                        if (b == 0) ++first_bounce_misses;
                        break;
                    }
                    int mat_idx = (inst_id == kStaticBlasSentinel)
                                    ? (prim_id / kCubeTrisPerBox)
                                    : inst_id;
                    Material m  = materials[mat_idx];
                    vec3 hit_pos = ray_origin + ray_dir * t_hit;
                    // No vertex normal in ray queries; -ray_dir is a
                    // good local approximation for cosine sampling
                    // and the sun-shadow's n·L test (matches cube.frag).
                    vec3 hit_n   = -ray_dir;

                    vec3 hit_light = sky_fill;
                    if (b < gi_shadow_max) {
                        float n_dot_sun = dot(hit_n, scene.sun_direction.xyz);
                        if (n_dot_sun > 0.0) {
                            g_rq_leaf_solid = gi_solid;
                            bool gi_sh = any_hit_shadow_caster(hit_pos + hit_n * 0.01, scene.sun_direction.xyz, 200.0);
                            g_rq_leaf_solid = false;
                            if (!gi_sh) {
                                hit_light += scene.sun_color.rgb *
                                             scene.sun_color.a * n_dot_sun;
                            }
                        }
                    }
                    path += throughput * (m.emissive.rgb +
                                          m.color.rgb * hit_light);
                    if (b + 1 >= N_bounces) break;

                    // Tint throughput by the surface albedo, kick
                    // off the next bounce in a cosine hemisphere
                    // around the approximate hit normal.
                    throughput *= m.color.rgb;
                    ray_origin  = hit_pos + hit_n * 0.01;
                    float br1 = rmRand(gi_seed + uvec3(taken * 7 + b, 31u, 17u));
                    float br2 = rmRand(gi_seed + uvec3(taken * 7 + b, 7u, 91u));
                    ray_dir = cos_hemi(br1, br2, hit_n);
                }
                sum += path;
                ++taken;
            }
        }
        vec3 gi_raw = (taken > 0) ? (sum / float(taken)) : vec3(0.0);
        // Sky visibility: fraction of FIRST-bounce rays that escaped
        // all occluders. Used to attenuate the smooth fallback so a
        // surface deep in a cavity doesn't get a bright "sky from
        // normal" lift even when the softener slider is high.
        float sky_vis = (taken > 0)
                          ? (float(first_bounce_misses) / float(taken))
                          : 1.0;
        // Softener: lerp toward a noise-free sky-at-normal estimate.
        // 0 = raw path-traced GI (most accurate, noisiest at low N),
        // 1 = smooth fallback (no per-pixel variance).
        // Both terms are AO-modulated AND sky_vis-modulated so corners
        // and cavities go genuinely dark regardless of softener value.
        float ao_p     = max(scene.ambient.w, 0.5);
        float occl = pow(ao, ao_p) * mix(0.4, 1.0, sky_vis);
        vec3 gi_smooth = sample_sky_atmosphere(nor) * occl;
        float soft = clamp(scene.terrain_extra.z, 0.0, 1.0);
        gi_indirect = mate * mix(gi_raw, gi_smooth, soft) *
                      scene.rt_params2.x;
        // Hard ceiling to keep a single bright bounce (e.g. a sun-
        // facing white box wall) from spiking the HDR + bloom.
        gi_indirect = min(gi_indirect, vec3(4.0));
        // Smooth-fade the GI contribution past 150 m so distant
        // terrain doesn't show a hard boundary where the loop stops.
        gi_indirect *= rt_lod_t;
    }

    vec3 lin = vec3(0.0);
    float cl_sh = cloud_shadow(pos);
    lin += dif * sha * cl_sh * scene.sun_color.rgb * scene.sun_color.a;
    // Ambient + Fresnel rim term darkened by RTAO. ao_punch pow-curves
    // the AO so > 1 makes mid-AO drop fast (corners punch dark) while
    // 1.0 keeps the original linear behaviour. Driven by the
    // 'AO darkness (raymarch terrain)' slider.
    float ao_punch  = max(scene.ambient.w, 0.5);
    float ao_shaped = pow(ao, ao_punch);
    // Sky fill. 0.35 left sun-shadowed ground almost black (far darker than
    // the mesh terrain shows the same shadow), so the shadow of a trunk or
    // rock read as a hole in the ground rather than as shade.
    // Under trees / beside objects the sky is partly covered as well: the
    // shadow of a canopy takes most of the sky fill with it (a forest
    // floor is dark, an open slope in the shade of a hill is not).
    lin += amb * scene.sky_color.rgb * 0.75 * ao_shaped * mix(1.0, 0.30, dyn_shadow);
    lin += fre * scene.sky_color.rgb * 0.25 * ao_shaped;

    // Dynamic point light (muzzle flash or the brightest explosion, same UBO
    // light the deferred pass applies to everything else - it skips this
    // pre-lit material). Objects can block it; terrain occlusion is ignored.
    if (scene.muzzle_pos.w > 0.0) {
        vec3  lv = scene.muzzle_pos.xyz - pos;
        float d  = length(lv);
        float r  = max(scene.muzzle_color.w, 0.1);
        if (d < r && distance(pos, scene.camera_pos.xyz) > 1.5) {
            vec3  Lp  = lv / max(d, 1e-3);
            float att = 1.0 - d / r; att *= att;
            float ndl = max(dot(nor, Lp), 0.0);     // no light on slopes facing away (hills behind a blast)
            if (ndl > 0.0 && att * scene.muzzle_pos.w > 0.01 && !any_hit_no_terrain(pos + nor * 0.1, Lp, max(d - 0.8, 0.05)))
                lin += scene.muzzle_color.rgb * (scene.muzzle_pos.w * att * ndl);
        }
    }
    // Hand flashlight on the ground: shadowed by objects, bushes and trees,
    // and near the camera by the lawn's own blades.
    {
        vec3 fl_dir; float fl_dist;
        vec3 fl = flashlight_at(pos, fl_dir, fl_dist);
        float fl_ndl = max(dot(nor, fl_dir), 0.0);
        if (fl_ndl > 0.0 && (fl.r + fl.g + fl.b) > 1e-4 &&
            !any_hit_no_terrain(pos + nor * 0.08, fl_dir, max(fl_dist - 0.05, 0.05))) {
            float blade_vis = 1.0;
#ifndef VIBE_WATER
            if (scene.grass_rm_params2.y > 0.5 && distance(pos, scene.camera_pos.xyz) < 20.0 &&
                grassEligibility(pos.xz, pos.y) > 0.02)
                blade_vis = mix(1.0, grass_blade_shadow(pos, fl_dir), scene.grass_rm_params2.z * 0.6 + 0.4);
#endif
            float plant_vis;
#ifndef VIBE_WATER
            FLASH_PLANT_VIS(u_clutter_shadow, pos + nor * 0.03, plant_vis)
#else
            plant_vis = 1.0;
#endif
            lin += fl * (fl_ndl * blade_vis * plant_vis);
        }
    }
    lin += fx_lights_diffuse(pos, nor);   // tracers (no shadows)
    // Laser bolts and gun flashes: shadowed by objects, bushes and trees, and
    // near the camera by the lawn's own blades.
    {
        int fxn = int(scene.ripple_info.w + 0.5);
        for (int i = 0; i < fxn; ++i) {
            if (scene.fx_light_col[i].a < 0.5) continue;
            vec3 to = scene.fx_lights[i].xyz - pos;
            float r = scene.fx_lights[i].w;
            float d2 = dot(to, to);
            if (d2 >= r * r) continue;
            float d = sqrt(d2);
            vec3 ld = to / max(d, 1e-3);
            float w = max(0.3 + 0.7 * dot(nor, ld), 0.0);
            if (w <= 0.0) continue;
            float fall = 1.0 - d / r; fall *= fall;
            if (any_hit_no_terrain(pos + nor * 0.08, ld, max(d - 0.3, 0.05))) continue;
            float blade_vis = 1.0;
#ifndef VIBE_WATER
            if (scene.grass_rm_params2.y > 0.5 && distance(pos, scene.camera_pos.xyz) < 20.0 &&
                grassEligibility(pos.xz, pos.y) > 0.02)
                blade_vis = mix(1.0, grass_blade_shadow(pos, ld), scene.grass_rm_params2.z * 0.6 + 0.4);
#endif
            lin += scene.fx_light_col[i].rgb * (fall * w * blade_vis);
        }
    }
    // Road lanterns (dusk / night): warm pools on the road and verge, with
    // a soft shadow (ray toward a random point of the lantern glass).
    {
        vec2 lseed = gl_FragCoord.xy + vec2(float(scene.rt_flags.w & 31)) * vec2(17.13, 9.71);
        for (int li = 0; li < 8; ++li) {
            float lI = scene.lamp_lights[li].w;
            if (lI < 1e-3) continue;
            vec3 lpos = scene.lamp_lights[li].xyz;
            vec3 to = lpos - pos;
            float d = length(to);
            if (d >= 13.0) continue;
            vec3 ld = to / max(d, 1e-3);
            float ndl = max(dot(nor, ld), 0.0);
            if (ndl <= 0.0) continue;
            float fall = 1.0 - d / 13.0; fall *= fall;
            vec3 jit = fract(sin(vec3(dot(lseed, vec2(12.9898, 78.233)) + float(li) * 3.1,
                                      dot(lseed, vec2(39.3468, 11.135)) + float(li) * 5.7,
                                      dot(lseed, vec2(73.1560, 52.235)) + float(li) * 9.3)) * 43758.5453) - 0.5;
            vec3 tj = lpos + jit * 0.34 - pos;
            float dj = length(tj);
            if (any_hit_no_terrain(pos + nor * 0.08, tj / max(dj, 1e-3), max(dj - 0.45, 0.05))) continue;
            // The lawn's own blades shade the ground from the lantern (the same
            // blade field the sun uses): exact near the camera, an averaged
            // darkening further out.
            float blade_vis = 1.0;
#ifndef VIBE_WATER
            if (scene.grass_rm_params2.y > 0.5) {
                float gp = grassEligibility(pos.xz, pos.y);
                if (gp > 0.02) {
                    float gstr = scene.grass_rm_params2.z;
                    float near_w = 1.0 - smoothstep(13.0, 26.0, distance(pos, scene.camera_pos.xyz));
                    blade_vis = 1.0 - 0.45 * gstr * gp * scene.grass_rm_params.w;
                    if (near_w > 0.01)
                        blade_vis = mix(blade_vis, mix(1.0, grass_blade_shadow(pos, ld), gstr * 0.6 + 0.4), near_w);
                }
            }
#endif
            float lamp_pv = 1.0;
#ifndef VIBE_WATER
            LAMP_PLANT_VIS(u_clutter_shadow, pos + nor * 0.03, li, lamp_pv)
#endif
            lin += vec3(1.0, 0.62, 0.28) * (lI * ndl * fall * blade_vis * lamp_pv);
        }
    }
    vec3 col = mate * lin + gi_indirect;
    // A highway's sheen: asphalt is matt seen from above and glares toward a low sun; the
    // polished wheel tracks and the tar more so.
    if (hw_pave > 0.001) {
        vec3  hv = normalize(sunDir - rd);
        float nh = max(dot(nor, hv), 0.0), vh = max(dot(-rd, hv), 0.0);
        float f5 = 1.0 - vh; f5 *= f5 * f5 * f5 * f5;
        float shin = mix(18.0, 260.0, hw_gloss);
        col += scene.sun_color.rgb * scene.sun_color.a * sha * (0.04 + 0.96 * f5) * pow(nh, shin) * (shin + 8.0) * 0.04
               * max(dot(nor, sunDir), 0.0) * hw_pave * (0.5 + 1.5 * hw_gloss);
        float g5 = fre_in * fre_in; g5 *= g5 * fre_in;
        col += scene.sky_color.rgb * (0.04 + 0.96 * g5) * (0.25 + hw_gloss) * hw_pave * ao_shaped;
    }
    // Rain: a wet sheen on the ground and standing water in the flats.
    // A puddle is a small mirror: flat normal (plus rain rings), darker
    // bed, Fresnel reflection of the sky and a sun glint.
    {
        // The beach where the water laps: the sand is soaked (darker, a
        // thin mirror of water on it) up to a waterline that moves in and
        // out with the wavelets.
        float shore = 0.0, shore_foam = 0.0;
        if (scene.water_params.x > 0.5) {
            float tw = scene.water_params.w;
            float lap = 0.045 * sin(tw * 0.8 + pos.x * 0.31 + pos.z * 0.23) + 0.03 * sin(tw * 1.37 + pos.x * 0.83 - pos.z * 0.57);
            float above = pos.y - scene.water_params.y - lap;
            float flat_w = smoothstep(0.55, 0.85, nor.y);
            shore = (1.0 - smoothstep(0.03, 0.50, above)) * flat_w;
            shore_foam = (1.0 - smoothstep(0.0, 0.022, abs(above - 0.010))) * flat_w;
            // The beach itself: bare sand with darker grit and pebbles from
            // the waterline up to about 70 cm above it (also under water).
            float beach = (1.0 - smoothstep(0.30, 0.75, above + 0.10 * tile_noise2(pos.xz, 0.9, vec2(0.0)))) * flat_w;
            if (beach > 0.01) {
                float grit = tile_noise2(pos.xz, 6.0, vec2(0.0)) * 0.6 + tile_noise2(pos.xz, 19.0, vec2(0.0)) * 0.4;
                float peb = smoothstep(0.62, 0.72, tile_noise2(pos.xz, 11.0, vec2(3.7))) * smoothstep(0.35, 0.6, tile_noise2(pos.xz, 2.3, vec2(0.0)));
                vec3 sandc = mix(scene.terrain_sand_color.rgb, vec3(0.200, 0.172, 0.125), 0.65) * mix(0.78, 1.12, grit);
                sandc = mix(sandc, vec3(0.10, 0.097, 0.09) * mix(0.6, 1.3, tile_noise2(pos.xz, 5.1, vec2(0.0))), peb * 0.8);
                // Lit with a smoothed normal: sand is fine-grained, the rocky
                // relief of the ground it replaces would show through as veins.
                vec3 bn = normalize(mix(nor, vec3(0.0, 1.0, 0.0), 0.8));
                vec3 blin = scene.sun_color.rgb * scene.sun_color.a * (max(dot(bn, sunDir), 0.0) * sha * cl_sh) +
                            scene.sky_color.rgb * (0.40 * ao_shaped);
                col = mix(col, sandc * blin + gi_indirect * 0.5, beach);
            }
        }
        float wet = max(scene_wet(), shore);
        float pd = distance(pos, ro);
        if (wet > 0.01 && pd < 400.0) {
            float flatn = smoothstep(0.80, 0.95, nor.y);            // (was 0.90..0.985: almost no ground qualified)
            float pn = tile_noise2(pos.xz, 0.33, vec2(0.0)) * 0.65 + tile_noise2(pos.xz, 1.21, vec2(7.0)) * 0.35;
            float thr = 0.66 - 0.34 * wet;
            float puddle = flatn * smoothstep(thr, thr + 0.06, pn) * smoothstep(0.2, 0.6, wet);
            vec3 pn_n = normalize(mix(nor, vec3(0.0, 1.0, 0.0), max(max(puddle, 0.5 * wet), 0.92 * shore)));
            if (puddle > 0.01 && pd < 60.0) {
                vec3 rw = water_ripples(pos.xz);
                pn_n = normalize(vec3(pn_n.x - rw.x * puddle, pn_n.y, pn_n.z - rw.y * puddle));
            }
            vec3 rr = reflect(rd, pn_n);
            rr.y = max(rr.y, 0.03);
            float cv = clamp(dot(-rd, pn_n), 0.0, 1.0);
            float om = 1.0 - cv; float om2 = om * om;
            float Fw = 0.02 + 0.98 * om2 * om2 * om;
            vec3 skyr = sample_sky_atmosphere(normalize(rr));
            float cov = scene.crater_info.y;
            skyr = mix(skyr, vec3(dot(skyr, vec3(0.2126, 0.7152, 0.0722))), cov);
            float film = max(puddle, 0.11 * wet * smoothstep(0.5, 0.95, nor.y));
            film = max(film, 0.40 * shore);                          // the glassy sheet on wet sand
            col *= mix(1.0, 0.55, puddle);
            col *= mix(1.0, 0.70, shore * (1.0 - puddle));           // soaked sand is darker
            col += skyr * Fw * film * ao_shaped;
            float sg = max(dot(normalize(rr), sunDir), 0.0);
            float sg2 = sg * sg; float sg8 = sg2 * sg2 * sg2 * sg2;
            float sg64 = sg8 * sg8 * sg8 * sg8 * sg8 * sg8 * sg8 * sg8;
            col += scene.sun_color.rgb * scene.sun_color.a * sg64 * sg64 * sha * cl_sh * max(puddle, 0.12 * film) * 1.5;
            // A thin line of foam where the water runs out.
            if (shore_foam > 0.01) {
                float fb = tile_noise2(pos.xz, 9.0, vec2(scene.water_params.w * 0.4, 0.0)) * 0.6 + tile_noise2(pos.xz, 23.0, vec2(0.0)) * 0.4;
                col = mix(col, (scene.sun_color.rgb * scene.sun_color.a * (0.25 + 0.75 * sha) + scene.sky_color.rgb * 0.6) * 0.55,
                          shore_foam * smoothstep(0.48, 0.74, fb) * 0.5);
            }
        }
    }

    // Final-colour AO multiplier — dampens the WHOLE pixel (sun term
    // included) by the shaped AO, making corner darkening visibly
    // pronounced even when the sun is hitting the ground. Without
    // this, AO only modulated the ambient term which is small
    // relative to the sun, so the corner-vs-open contrast was
    // imperceptible. Mirrors the visible "darker between objects"
    // look on cube.frag — there the higher-contrast albedos plus
    // GI hit-shading make the effect read; on terrain we have to
    // fold AO into the final colour to match.
    //
    // Strength comes from the user-facing 'AO final strength' slider
    // (terrain_rt_extra.z). 0 = no extra darkening, 1 = full punch.
    // ao_shaped drives the AO contour, so the 'AO darkness' slider
    // tunes the curve and 'AO final strength' tunes how much that
    // curve actually shows up on screen.
    float ao_final_str = clamp(scene.terrain_rt_extra.z, 0.0, 1.0);
    float ao_floor     = 1.0 - ao_final_str;
    col *= mix(ao_floor, 1.0, ao_shaped);
    // From the air: beyond the range of the occlusion and shadow terms above the ground
    // was fully lit, a step brighter than the land near the camera (snow went to plain
    // white). The average of what those terms take, faded in past their range.
    col *= mix(1.0, 0.74, rm_air_look(t));

    // Contrast ('contrast (raymarch terrain)' slider): a power curve on
    // luminance around the brightness of average ground under the current
    // sun + sky, so it tracks time of day. Hue is kept (RGB scaled alike).
    {
        float rm_con = scene.grass_rm_params2.w;
        if (abs(rm_con - 1.0) > 0.005) {
            const vec3 kLum = vec3(0.2126, 0.7152, 0.0722);
            float piv = 0.12 * dot(scene.sun_color.rgb * scene.sun_color.a * 0.45 +
                                   scene.sky_color.rgb * 0.35, kLum);
            float l = dot(col, kLum);
            // Below the pivot the curve is much gentler: full strength there
            // crushed shadowed ground to black.
            float rel = max(l, 1e-5) / max(piv, 1e-4);
            // (not on the ground seen from the air, beyond the normal range: with nothing dark left in
            // it the curve only pushed sunlit snow past white)
            col *= mix(clamp(pow(rel, (rm_con - 1.0) * (rel < 1.0 ? 0.3 : 1.0)), 0.15, 4.0), 1.0, rm_air_look(t));
        }
    }

    // Debug viz: replace the final colour with the GI contribution
    // only (scaled). 0 = off, 1 = scaled gi_indirect, 2 = scaled raw
    // gi (no albedo modulation, no softener). Toggled from
    // Settings -> Lighting -> "GI debug viz (raymarch terrain)".
    if (scene.terrain_extra.w > 0.5) {
        col = gi_indirect * 4.0;
    }

    // Wavelength-dependent atmospheric extinction (blue attenuates
    // ~4× faster than red), with a sun-tinted fog colour near the sun
    // direction so the horizon glows warm where the sun touches it.
    // (ground_haze.glsl: the same call the trees and buildings get in deferred_lighting.frag)
    col = ground_haze(col, t, rd, sunDir);

