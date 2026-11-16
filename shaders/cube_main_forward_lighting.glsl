// cube_main_forward_lighting.glsl - part of cube.frag (included there, in place; not a standalone shader).
// main(): forward lighting path (RT sun shadow, AO, GI, final colour) - skipped when deferred lighting owns the frame

    // --- Sun shadow (RT, contact-hardening / PCSS-style) ---
    // 1. BLOCKER SEARCH: a handful of cheap closest-hit rays in a wide cone
    //    around L; record the average distance to the occluder.
    // 2. PENUMBRA ESTIMATE: cone half-angle в€ќ avg_blocker_distance × softness.
    //    Close-by occluders give a small cone (sharp shadow); far ones give
    //    a wide cone (soft shadow). Real-world effect: shadow under a pole
    //    is razor-sharp, the same shadow stretched across the floor is fuzzy.
    // 3. SHADOW RAYS: stratified sampling inside the size-adapted cone.
    float shadow = 0.0;
    // Terrain receivers always fire RT shadows now — but they use
    // mask 0x02 so the ray queries skip the terrain BLAS entirely
    // (which is what used to false-hit at distance from LOD/BLAS
    // mismatch). The bake handles terrain self-shadow; the RT
    // ray-queries handle castle / dyn-props at every distance, so
    // box / castle shadows on terrain no longer fade out past 80 m.
    float terrain_dist = cam_dist;         // P5: use hoisted distance
    // P10: at near-terminator the surface is already at <4% direct
    // light, so PCSS softness is invisible. Skip the entire 4-blocker
    // + N_s shadow-ray block. shadow stays at the default 0; the
    // direct *= (1 - shadow) * n_dot_l math collapses to the Lambert
    // falloff, which is what dominates here anyway.
    // Gate on shadow_strength too — both `shadow` (line ~1501) and
    // `sh_bake` (line ~1529) are multiplied by rt_params.w at the
    // end. With strength=0 the entire blocker + PCSS + bake-PCF block
    // was wasting ~12 RT rays + 9 texture taps per pixel for a final
    // result of zero.
    if (n_dot_l_raw > 0.04 && scene.rt_flags.x != 0 &&
        scene.rt_params.w > 1e-3) {
      if (true) {   // (kept block for diff hygiene — always-on now)
        // Roadmap item #4 Phase 3: when the half-rate shadow toggle is
        // on AND this is a brush/dyn surface (NOT terrain — terrain
        // keeps its hybrid bake/RT path further below), sample the
        // pre-traced shadow_lr image at the matching screen UV. The
        // producer already multiplied by shadow_strength so the value
        // drops straight into `shadow`. Skips the entire blocker+PCSS
        // RT-ray block — the whole point of the half-rate pass.
        if (scene.terrain_local_info.z > 0.5 && !is_terrain_pre) {
            vec2 hr_uv = gl_FragCoord.xy / scene.viewport.xy;
            shadow = textureLod(u_shadow_lr, hr_uv, 0.0).r;
        } else {
        // Base sample count from slider, distance-LOD'd. terrain_extra.y
        // is a multiplier on the BASE count that's applied before LOD
        // reduction, so close fragments get N×multiplier rays while far
        // ones still drop toward 1.
        float near_mult = max(1.0, scene.terrain_extra.y);
        // Cap base sample count: TAA accumulates ~8 frames of jitter, so
        // anything past ~16 effective samples per pixel falls below the
        // perceptual noise floor. Ultra preset (slider=40) × near_mult=4
        // would otherwise hand the shader 160 → ~80 effective rays per
        // pixel, all but a handful of which are wasted work TAA blurs out.
        const int kMaxBase = 32;
        int  base_s = min(int(ceil(float(scene.rt_flags.y) * near_mult)), kMaxBase);
        int  N_s = lod_samples(max(1, base_s), cam_dist);
        // Per-pixel softness — terrain optionally tightens the cone via
        // the Settings UI to reduce PCSS dither. Tightening trades soft
        // shadow edges for cleaner per-pixel results at the same sample
        // count. Brushes/dyn props always use the global slider value.
        float base_softness = scene.rt_params.x;
        if (is_terrain_pre) {
            base_softness *= max(0.05, scene.terrain_params.w);
        }

        vec3 ref = abs(L.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref, L));
        vec3 tan_v = cross(L, tan_u);

        // Terrain shadow-ray bias has a non-linear ramp with camera
        // distance because the chunked raster mesh uses distance-LOD
        // (stride 1/2/4/8) while the BLAS holds the full heightmap.
        // At LOD 3 (≥320 m) the rasterised straight-line interpolation
        // can sit 5–15 m BELOW true heightmap peaks; shadow rays fired
        // upward then false-hit those missed peaks → dark patches on
        // distant ridges. We hold near bias tight (no peter-panning on
        // close shadows) and ramp hard past LOD 0's range:
        //   ≤ 80 m  : 0.04 m (matches the LOD-0 / BLAS gap)
        //   320 m   : ~6 m  (covers typical LOD-3 peak undershoot)
        //   ≥ 600 m : ~12 m (covers worst-case)
        float dist_to_cam = cam_dist;      // P5: use hoisted distance
        // Unclamped far_t so the bias keeps growing past 400 m. The
        // BLAS-vs-raster gap on LOD-2 stride-4 ridges keeps widening
        // with distance (the rasterised quad's straight-line interp
        // can sit further below sharp BLAS peaks the further out you
        // go), and the previous clamp at 1.0 capped the bias at 8 m —
        // not enough to clear the ~20–30 m peak undershoot at km-
        // range, so distant ridges kept reading as dark splatter
        // ("dark leaves") even after the first fix. Quadratic growth
        // soft-clamped at 32 m covers the worst case without blowing
        // out near-shadow precision.
        float far_t = max((dist_to_cam - 60.0) / 200.0, 0.0);
        float terrain_far_bias = min(far_t * far_t * 6.0, 32.0);
        float bias = is_terrain_pre
            ? (0.05 + 0.10 * (1.0 - n_dot_l_raw) + terrain_far_bias)
            : (0.005 + 0.02 * (1.0 - n_dot_l_raw));
        // For axis-aligned brushes (non-terrain), the shading normal
        // N has been bent by corner softening (~30° at outer corners)
        // and possibly perturbed by the SPOM/normal-map basis. Using
        // the bent N for the bias offset pushes the ray origin off
        // the face axis and partway TOWARD the adjacent brush body;
        // shadow rays then grazingly self-occlude on the neighbour
        // and produce the thin dark lines/panes the user sees along
        // wall↔tower and wall↔wall seams. Snap the bias direction to
        // the dominant face axis so the offset stays purely outward
        // along the face. Terrain keeps the FBM normal (its surfaces
        // aren't axis-aligned).
        vec3 bias_n = N;
        if (!is_terrain_pre && !is_mesh_draw) {
            vec3 absN = abs(N);
            if (absN.x >= absN.y && absN.x >= absN.z) {
                bias_n = vec3(sign(N.x), 0.0, 0.0);
            } else if (absN.y >= absN.z) {
                bias_n = vec3(0.0, sign(N.y), 0.0);
            } else {
                bias_n = vec3(0.0, 0.0, sign(N.z));
            }
        }
        vec3 origin = shading_pos + bias_n * bias;

        // 1. Blocker search — 2 rays in a wider cone (4× base softness) so we
        //    catch occluders even when the surface is close to lit. Was 4
        //    rays before; halving cuts ~25% of the per-pixel shadow ray
        //    budget (blocker:shadow ≈ 4:N_s_eff was 4:8 worst case, now
        //    2:8). Penumbra estimate degrades from 4-tap-mean to 2-tap-mean
        //    but the downstream `clamp(softness * 0.25 .. * 6)` is
        //    forgiving enough that the visible difference is invisible
        //    after TAA averaging.
        // Terrain receivers use the no-terrain variant so we don't get
        // false-hits from the BLAS terrain detail above the rasterised
        // LOD surface (the heightmap bake covers terrain self-shadow).
        // P9: scale RT distances by view distance. Surfaces at ~10 m
        // from the camera have visible occluders within ~30 m; firing
        // 200 m blocker rays into thin air past that just burns BVH
        // traversal. Cap blocker rays at 4× cam_dist (min 30 m for
        // close-up); cap shadow rays at 4× avg blocker hit distance
        // computed below.
        const float kBlockerTMax = clamp(cam_dist * 4.0, 30.0, 200.0);
        const int kBlockerSearch = 2;
        // Hoist the constant cone-width multiplier out of the per-iter
        // expression. base_softness is loop-invariant; kBlockerCone =
        // base_softness * 4.0 was being recomputed every iteration.
        const float kBlockerCone = base_softness * 4.0;
        float sum_t = 0.0;
        int   hits = 0;
        // One loop, mask hoisted to a uniform-per-pixel value. The earlier
        // two-specialised-loops form was a workaround for a per-tap
        // ternary on the ray-query function — closest_hit_m takes the mask
        // as an arg so a single function body inlines and the ray-query
        // call has no divergent control on it.
        const uint blocker_mask = is_terrain_pre ? 0x02u : 0xFFu;
        // Per-pixel rotation: ONE sin/cos covers the disk-rotation
        // dither for all blocker taps, instead of one sin/cos pair
        // per tap. Mixed with the Vogel constant table this drops the
        // per-pixel trig cost from 2·N to 2.
        // seed_stable (not seed_base): shadows have no temporal
        // accumulator, so frame-varying blocker jitter = flicker.
        float pp_phi_b = rand(seed_stable + uvec3(0, 99u, 0u)) * 6.28318530718;
        float pp_c_b = cos(pp_phi_b);
        float pp_s_b = sin(pp_phi_b);
        for (int i = 0; i < kBlockerSearch; ++i) {
            float br1 = rand(seed_stable + uvec3(i, 91u, 13u));
            float r   = sqrt(br1) * kBlockerCone;
            vec2  v   = kVogelDisk[i & 31];
            float vx  = pp_c_b * v.x - pp_s_b * v.y;
            float vy  = pp_s_b * v.x + pp_c_b * v.y;
            vec3 jitter = (vx * tan_u + vy * tan_v) * r;
            vec3 dir = normalize(L + jitter);
            float t;
            if (closest_hit_m(origin, dir, kBlockerTMax, blocker_mask, t)) { sum_t += t; ++hits; }
        }

        // Always trace the shadow rays: 2 blocker taps that thread a doorway
        // or gap reported "no occluder" and left shadowed interiors fully lit
        // (deferred_lighting.frag has the same fix). hits == 0 -> default
        // penumbra (avg blocker distance 50 m).
        {
            // 2. Penumbra estimate. Distance is normalised against a 10m
            //    reference, then raised to a power between 1 (linear, the
            //    classic PCSS) and 3 (cubic — very contact-sharp close-up,
            //    very soft far away). The exponent comes from the
            //    shadow_curve slider in the menu.
            float avg_t = (hits > 0) ? (sum_t / float(hits)) : 50.0;
            // P9: shadow rays only need to reach occluders that could
            // contribute to the penumbra estimate — anything past 4×
            // the average blocker hit can't possibly close the cone
            // around this surface point. Capped to existing 200 m so
            // we never extend the budget.
            float kShadowTMax = clamp(avg_t * 4.0, 20.0, 200.0);
            float t_norm = avg_t * 0.1;
            float curve_exp = mix(1.0, 3.0, clamp(scene.rt_params2.w, 0.0, 1.0));
            float scale = pow(max(t_norm, 0.0), curve_exp);
            // Terrain receivers need a much higher minimum penumbra
            // floor than walls. Boxes / dyn props sit ~1 m above the
            // ground (small blocker distance → scale ≈ 0.1), which
            // clamps the cone to ~0.25× base_softness — too tight for
            // ~10 half-rate samples to produce a smooth gradient, so
            // the edge reads as a hard line with dither dots. Walls
            // have far-away casters (long blocker distance) so they
            // don't hit the min clamp. Lift the terrain floor to 2.0×
            // so box-on-ground shadows always have a perceptibly soft
            // penumbra regardless of how low the blocker sits.
            float min_pen_mult = is_terrain_pre ? 2.0 : 0.25;
            float penumbra = clamp(base_softness * scale * 0.6,
                                   base_softness * min_pen_mult,
                                   base_softness * 6.0);

            // 3. Stratified shadow rays in the size-adapted cone.
            // FULL-RATE, FRAME-STABLE sampling. The old half-rate
            // checkerboard (alternating ray halves by frame parity)
            // and frame-cycled seeds both assumed TAA history would
            // average them out — at the history strengths this game
            // actually runs, they read as 2-frame shadow flicker.
            // Every frame now fires the full budget with a static
            // per-pixel pattern: penumbra noise is a stationary dither
            // instead of a shimmer.
            int   parity  = 0;
            int   N_s_eff = max(1, N_s);
            int strata = int(ceil(sqrt(float(N_s_eff))));
            float inv_strata = 1.0 / float(strata);
            float blocked = 0.0;
            int taken = 0;
            // One loop, mask hoisted (same trick as the blocker loop).
            const uint shadow_mask = is_terrain_pre ? 0x02u : 0x01u;
            // Same Vogel + per-pixel rotation trick as the blocker loop.
            // Trig drops from 2·N_s_eff to 2 across the whole shadow
            // pass; on heavy slider settings (32 samples) that's ~60
            // sin/cos ops removed per shadowed pixel.
            float pp_phi_s = rand(seed_stable + uvec3(1, 99u, 0u)) * 6.28318530718;
            float pp_c_s = cos(pp_phi_s);
            float pp_s_s = sin(pp_phi_s);
            for (int sy = 0; sy < strata && taken < N_s_eff; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s_eff; ++sx) {
                    int idx = taken * 2 + parity;
                    float r1 = rand(seed_stable + uvec3(idx, 11u, 47u));
                    float u1 = (float(sx) + r1) * inv_strata;
                    float r   = sqrt(u1) * penumbra;
                    vec2  v   = kVogelDisk[idx & 31];
                    float vx  = pp_c_s * v.x - pp_s_s * v.y;
                    float vy  = pp_s_s * v.x + pp_c_s * v.y;
                    vec3 jitter = (vx * tan_u + vy * tan_v) * r;
                    vec3 dir = normalize(L + jitter);
                    if (any_hit_m(origin, dir, kShadowTMax, shadow_mask)) blocked += 1.0;
                    ++taken;
                }
            }
            shadow = (blocked / float(taken)) * scene.rt_params.w;
        }
        }  // end half-rate else
      }  // end fire_rt_shadow
        // shadow_strength can exceed 1.0 via the slider — without this clamp
        // the (1 - shadow) factor at line ~808 goes negative and produces
        // black-and-then-some pixels (max() in compose can't recover them).
        shadow = clamp(shadow, 0.0, 1.0);

        // Terrain hybrid: RT (castle / dyn-props) + bake (terrain
        // self-shadow). `shadow` holds the RT no-terrain result, the
        // bake holds terrain-on-terrain occlusion. Take the MAX so
        // any blocker (castle or distant ridge) shadows the receiver.
        // Castle shadows on terrain therefore work at every distance.
        if (is_terrain_pre) {
            ivec2 sz_b = textureSize(u_terrain_shadow, 0);
            const float side_b = 2048.0;
            vec2 uv_b = (vWorldPos.xz / side_b) + vec2(0.5);
            float sh_bake = 0.0;
            if (all(greaterThanEqual(uv_b, vec2(0.0))) &&
                all(lessThanEqual(uv_b, vec2(1.0)))) {
                // PCSS on the baked map (matches deferred_lighting.frag).
                // G stores blocker distance along the sun ray (t / 400 m).
                // Blocker search (3x3, 3 m spacing) sets a WORLD-space
                // kernel radius that grows with blocker distance: tall
                // tower tops cast wide soft light penumbras, low walls
                // stay tight and dark at the base. The old constant
                // 4 m blur inverted this relationship.
                float cover = 0.0;
                float dsum  = 0.0;
                const float kSearchUv = 3.0 / side_b;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        vec2 sv = texture(u_terrain_shadow,
                                          uv_b + vec2(float(ox), float(oy)) *
                                                 kSearchUv).rg;
                        cover += sv.r;
                        dsum  += sv.g * sv.r;
                    }
                }
                if (cover > 1e-4) {
                    float avg_d = (dsum / cover) * 400.0;  // metres
                    float half_w = clamp(avg_d * 0.06, 0.6, 8.0);
                    float step_uv = (half_w * 0.5) / side_b;
                    // Pre-baked exp(-(ox² + oy²) * 0.35), ox/oy ∈ [-2,2].
                    const float kW[25] = float[25](
                        0.0608, 0.1738, 0.2466, 0.1738, 0.0608,
                        0.1738, 0.4966, 0.7047, 0.4966, 0.1738,
                        0.2466, 0.7047, 1.0000, 0.7047, 0.2466,
                        0.1738, 0.4966, 0.7047, 0.4966, 0.1738,
                        0.0608, 0.1738, 0.2466, 0.1738, 0.0608);
                    const float kWsum = 6.6172;
                    float s = 0.0;
                    for (int oy = -2; oy <= 2; ++oy) {
                        for (int ox = -2; ox <= 2; ++ox) {
                            vec2 d = vec2(float(ox), float(oy)) * step_uv;
                            s += texture(u_terrain_shadow, uv_b + d).r *
                                 kW[(oy + 2) * 5 + (ox + 2)];
                        }
                    }
                    sh_bake = s * (1.0 / kWsum);
                    // Distant blockers also lose peak density (sun-disk
                    // umbra collapse + open-sky fill): 100 m -> ~18%.
                    sh_bake *= 1.0 - 0.18 * clamp(avg_d / 100.0, 0.0, 1.0);
                    sh_bake *= scene.rt_params.w;
                }
            }
            shadow = max(shadow, sh_bake);
        }
        // Cubic smoothstep on the combined shadow factor (lit
        // fraction = 1 − shadow): softens the merged terrain-bake +
        // RT-castle visibility ramp into one continuous penumbra.
        float lit_c = clamp(1.0 - shadow, 0.0, 1.0);
        lit_c = lit_c * lit_c * (3.0 - 2.0 * lit_c);
        shadow = 1.0 - lit_c;

        // Grass-cast shadow on terrain: walk the grass eligibility mask
        // (binding 13) along the sun XZ direction for a few short steps,
        // accumulating presence as occlusion. Approximates the directional
        // soft shadow that an overhead grass layer would cast on the
        // ground without firing any rays. Cheap (N texture taps, gated
        // on the master toggle + strength). Only on terrain pixels.
        // Above-water gate: don't cast grass shadow onto lake-floor
        // pixels. The grass-eligibility mask still reads 1 in those
        // XZ cells (it doesn't know about water), and the resulting
        // shadow stacked with the see-through water tint reads as
        // black blotches when looking through the lake at a sandy
        // bottom near a grassy shore.
        float _wtr_lvl_gs = scene.water_params.y;
        float _wtr_on_gs  = scene.water_params.x;
        bool above_water_gs = (_wtr_on_gs < 0.5) ||
                              (vWorldPos.y > _wtr_lvl_gs + 0.05);
        if (is_terrain_pre &&
            above_water_gs &&
            scene.grass_shadow_on_terrain_params.w > 0.5 &&
            scene.grass_shadow_on_terrain_params.x > 0.001) {
            vec2  sun_xz = scene.sun_direction.xz;
            float sun_xz_len = length(sun_xz);
            if (sun_xz_len > 0.05) {
                sun_xz /= sun_xz_len;
                int   gs_N = int(scene.grass_shadow_on_terrain_params.y);
                float gs_reach = scene.grass_shadow_on_terrain_params.z;
                float gs_str   = scene.grass_shadow_on_terrain_params.x;
                // Heightmap is 2048 m square, centered at origin (matches
                // kHeightmapSide in terrain_raymarch.frag / water.frag).
                const float kGsSide = 2048.0;
                float gs_shadow = 0.0;
                // Hoist the per-loop divisor — gs_N is uniform across the
                // iterations, no need to divide each tap.
                float inv_gs_N = 1.0 / float(gs_N);
                for (int gi = 1; gi <= 8; ++gi) {
                    if (gi > gs_N) break;
                    float gd = gs_reach * (float(gi) * inv_gs_N);
                    vec2 sxz = vWorldPos.xz + sun_xz * gd;
                    vec2 gs_uv = (sxz / kGsSide) + vec2(0.5);
                    if (gs_uv.x < 0.0 || gs_uv.x > 1.0 ||
                        gs_uv.y < 0.0 || gs_uv.y > 1.0) continue;
                    float pres = textureLod(u_grass_mask, gs_uv, 0.0).r;
                    // Distance falloff: near grass shades stronger,
                    // far grass falls off (matches the raymarch path).
                    float fall = 1.0 - smoothstep(0.0, 1.0, gd / gs_reach);
                    gs_shadow = max(gs_shadow, pres * fall);
                }
                // Apply only where the receiver isn't already deeply
                // shadowed -- keeps the term from doubling-up on RT
                // shadow castles + grass shadow.
                float gs_term = gs_shadow * gs_str * (1.0 - shadow);
                shadow = clamp(shadow + gs_term, 0.0, 1.0);
            }
        }
    }

    vec3 direct = albedo * scene.sun_color.rgb * scene.sun_color.a *
                  n_dot_l * (1.0 - shadow);

    // ---- Terrain micro-shadow + glancing specular (AAA Phase 5) ----
    if (is_terrain) {
        // Micro-shadowing: low texels of the splat height field lose
        // grazing sun first — sells the material relief at zero ray
        // cost (the displacement-derived analogue of the classic
        // AO-based micro-shadow trick).
        float graze2 = (1.0 - n_dot_l) * (1.0 - n_dot_l);
        float micro = 1.0 - (1.0 - clamp(terrain_micro_h, 0.0, 1.0))
                            * graze2 * 0.65;
        direct *= clamp(micro, 0.3, 1.0);
        // Relief self-shadow from the POM sun march.
        direct *= terrain_pom_vis;
        // Glancing Blinn specular on rock/snow — low sun glints off
        // wet-looking stone and snow crust; pure-Lambert terrain reads
        // chalky by comparison.
        // Distance-faded: a pow-32 highlight on far rock shimmers per
        // frame (sub-pixel normals, no jitter) — keep it a near effect.
        float spec_fade = 1.0 - smoothstep(60.0, 140.0, cam_dist);
        if (terrain_spec_w * spec_fade > 0.005) {
            vec3 Vs = normalize(scene.camera_pos.xyz - vWorldPos);
            vec3 Hs = normalize(L + Vs);
            float spec = pow(max(dot(N, Hs), 0.0), 32.0);
            direct += scene.sun_color.rgb * scene.sun_color.a *
                      spec * terrain_spec_w * spec_fade *
                      n_dot_l * (1.0 - shadow);
        }
    }

    // --- Muzzle flash: dynamic point light + RT shadow ---
    // Active for kMuzzleFlashDuration on each shot; CPU sets intensity > 0
    // and a position just in front of the eye. One shadow ray per pixel,
    // gated by N·L > 0 and within the falloff radius — keeps the cost
    // negligible when no shot is firing. Inverse-square attenuation with
    // a smooth cutoff at the radius so contributions don't pop.
    // Tighter intensity gate (1e-3 vs > 0.0): the CPU clears intensity
    // to exactly 0 between flashes, but FP comparison with > 0.0 still
    // entered this block on epsilon residuals after physics writes.
    if (scene.muzzle_pos.w > 1e-3) {
        vec3  m_pos       = scene.muzzle_pos.xyz;
        float m_intensity = scene.muzzle_pos.w;
        vec3  m_color     = scene.muzzle_color.rgb;
        float m_radius    = max(0.5, scene.muzzle_color.w);

        vec3  to_light    = m_pos - shading_pos;
        // Squared-distance test first — saves the sqrt when this pixel
        // is outside the falloff radius (the common case for ground-far
        // pixels during a flash that lights only the local area).
        float dist2       = dot(to_light, to_light);
        float r2          = m_radius * m_radius;
        // Inner-radius cutoff: surfaces within ~1 m of the muzzle are
        // ALWAYS the viewmodel (gun barrel sits ~30–80 cm from the
        // muzzle origin). The inverse-square (1/(1+d²)) at d≈0.3 m
        // gives atten ≈ 0.92; even capped to 4.0 the gun visibly
        // glowed in flash colour during firing — the user explicitly
        // does not want that. Skip the muzzle dynamic light for
        // pixels closer than 1 m. World pixels are always farther
        // than the gun so they keep the flash lighting normally.
        const float kSelfLightRadius2 = 1.0;
        if (dist2 < r2 && dist2 > kSelfLightRadius2) {
            float dist     = sqrt(dist2);
            vec3 L_m       = to_light / max(dist, 1e-3);
            float n_dot_lm = max(dot(N, L_m), 0.0);
            if (n_dot_lm > 0.0) {
                // Smooth falloff: 1/(1 + d²)·(1 - d/r)². The window term zeroes
                // contribution exactly at the radius, no hard edge.
                float window = 1.0 - dist / m_radius;
                float atten  = (window * window) / (1.0 + dist * dist);

                // RT shadow: one any-hit ray to the light. t_max stops short
                // of the light origin so a triangle exactly at m_pos can't
                // self-occlude.
                float bias    = 0.005 + 0.02 * (1.0 - n_dot_lm);
                vec3  origin  = shading_pos + N * bias;
                bool  in_shadow = any_hit(origin, L_m, dist - 0.01);
                if (!in_shadow) {
                    // Cap the contribution per-pixel. Surfaces VERY
                    // close to the muzzle (the viewmodel itself, ~0.5
                    // m from the muzzle origin) hit the inverse-square
                    // singularity — the gun would self-light into the
                    // tens of thousands of nits and bloom the entire
                    // screen during firing. Hard ceiling at 4.0 keeps
                    // the muzzle-glow look without runaway HDR.
                    vec3 contrib = albedo * m_color * m_intensity * n_dot_lm * atten;
                    direct += min(contrib, vec3(4.0));
                }
            }
        }
    }

    // --- AO (stratified hemisphere) ---
    //
    // ao_mode (rt_flags2.w):
    //   0 = off — no AO, ao = 1.0 (lighting reads as fully lit at concavities)
    //   1 = "fast contact AO" — short hemisphere rays at HALF the radius and
    //       capped to 2 samples. Reads similar to GTAO at far less cost
    //       than full RTAO, but is still inline ray-traced (true GTAO needs
    //       a separate screen-space pass with depth+normal bound; deferred).
    //   2 = full RTAO — N hemisphere rays at user-set radius, the original
    //       quality path. Picks up off-screen occluders correctly.
    float ao = 1.0;
    int ao_mode = scene.rt_flags2.w;
    if (ao_mode > 0 && scene.rt_flags.z > 0) {
        // mode 1 caps to 2 samples; mode 2 honors the slider; mode 3
        // (HBAO) honors the slider as the direction count.
        int requested = (ao_mode == 1) ? min(scene.rt_flags.z, 2)
                                       : scene.rt_flags.z;
        int N_ao = lod_samples(requested, cam_dist);
        // mode 1 halves the search radius -- bias toward contact AO.
        // mode 3 also uses a tighter radius (0.6x) so the horizon-angle
        // search focuses on near-surface crevices the binary RTAO misses.
        float ao_radius = scene.rt_params.y *
            (ao_mode == 1 ? 0.5 : (ao_mode == 3 ? 0.6 : 1.0));
        // Distance-scaled origin lift for terrain — pushes the AO ray's
        // origin above the BLAS detail that the rasterised LOD-1+ raster
        // sits below. Without this, AO rays from the under-shooting
        // raster surface false-hit BLAS peaks → patchy darkening per
        // triangle. Lift ramps with distance, capped at ~6m which clears
        // typical LOD-2 peak undershoot.
        float ao_lift = 0.01;
        if (is_terrain_pre) {
            float t = clamp((cam_dist - 40.0) / 200.0, 0.0, 1.0);
            ao_lift = mix(0.01, 6.0, t);
        }
        vec3 origin_ao = shading_pos + N * ao_lift;
        // Unstratified random samples driven by IGN + temporal frame seed.
        // The stratified version (ceil(sqrt(N))²-cell grid) was biased
        // when N wasn't a perfect square — at N=2 strata=2 picked only
        // the cells with sy=0 → all rays in one half of the hemisphere
        // azimuth, which read as systematically darker edges. Random +
        // TAA temporal accumulation converges to true AO over ~8 frames
        // regardless of N.
        int taken = 0;
        float occluded = 0.0;
        // TBN hoist (same trick as the GI loop) — N is constant for all
        // AO rays at this pixel so the per-tap cross+normalize that
        // cos_hemi() does internally was redundant.
        vec3 ao_up  = abs(N.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 ao_tan = normalize(cross(ao_up, N));
        vec3 ao_bit = cross(N, ao_tan);
        // Vogel disk + per-pixel rotation for the hemisphere phi term.
        // The cosine-hemisphere y is still derived from u1 (sqrt-based);
        // only the disk-azimuth trig moves to the table. Saves N_ao
        // sin/cos pairs per pixel.
        // seed_stable: AO has no temporal accumulator — frame-varying
        // hemisphere jitter read as crawling noise in corners.
        float pp_phi_a = rand(seed_stable + uvec3(0, 99u, 1u)) * 6.28318530718;
        float pp_c_a = cos(pp_phi_a);
        float pp_s_a = sin(pp_phi_a);
        if (ao_mode == 3) {
            // HBAO branch -- horizon-angle accumulation. For each of
            // N_ao azimuth directions in the tangent plane, march a few
            // taps at increasing inclination and keep the steepest
            // closest-hit angle as the horizon. The unoccluded fraction
            // per direction is (1 - sin(horizon)); averaging across
            // directions gives a smoother contact-AO signal than the
            // binary occluded/not-occluded RTAO test, especially in
            // tight crevices (cube-cube contact lines, terrain creases)
            // where the binary RTAO either records every ray as a hit
            // (saturates) or misses thin features entirely.
            //
            // Cost: K taps per direction * N_ao directions inline rays;
            // we hold K=3 (one ray per inclination band) so total rays
            // = 3*N_dir. Cap the direction count at 8 (24 rays) so the
            // Ultra/Insane RTAO sample-count presets (16/32) don't
            // explode HBAO into 48-96 rays per pixel. The horizon-
            // angle average converges fast in azimuth (the steepest
            // hit dominates), so 8 directions is the visual quality
            // knee; more just costs perf without changing the look.
            const int  kHbaoSteps = 3;
            const int  kHbaoMaxDirs = 8;
            int N_dir = min(N_ao, kHbaoMaxDirs);
            // Inclination bands: low / mid / high above the tangent.
            // sin(angle) values pick where the ray points; cosine band
            // gives the horizontal component along the azimuth.
            const float kSinBand[3] = float[3](0.30, 0.55, 0.80);
            const float kCosBand[3] = float[3](0.9539, 0.8351, 0.6000);
            float horizon_sum = 0.0;
            int   dir_count   = 0;
            for (int i = 0; i < N_dir; ++i) {
                // Even azimuth distribution + per-pixel rotation. Using
                // the Vogel-disk x/y as the azimuth seed reuses the
                // existing table without extra trig.
                vec2  v   = kVogelDisk[i & 31];
                float vx  = pp_c_a * v.x - pp_s_a * v.y;
                float vy  = pp_s_a * v.x + pp_c_a * v.y;
                vec2  az  = normalize(vx * vx + vy * vy > 1e-6
                                       ? vec2(vx, vy) : vec2(1.0, 0.0));
                vec3 az_dir = az.x * ao_tan + az.y * ao_bit;
                float best_sin = 0.0;  // horizon = 0 -> fully visible
                for (int k = 0; k < kHbaoSteps; ++k) {
                    vec3 d = kCosBand[k] * az_dir + kSinBand[k] * N;
                    // closest_hit gives the distance; convert to a
                    // tangent-plane horizon-angle estimate. Closer hits
                    // raise the horizon more (steeper sin).
                    float t_hit;
                    if (closest_hit(origin_ao, d, ao_radius, t_hit)) {
                        // Inverse-distance-weighted horizon sin: a hit
                        // at t=0 reads as sin=1 (fully occluded); at
                        // t=ao_radius it reads as sin=0 (no contribution).
                        float w = 1.0 - clamp(t_hit / ao_radius, 0.0, 1.0);
                        // Blend the inclination band with the hit
                        // proximity so a near-tangent occluder produces
                        // a strong contact-AO signal even if its band
                        // is shallow.
                        float this_sin = mix(kSinBand[k], 1.0, w);
                        best_sin = max(best_sin, this_sin);
                    }
                }
                horizon_sum += (1.0 - best_sin);
                ++dir_count;
            }
            taken = max(dir_count, 1);
            // Repurpose `occluded` as (1 - avg_visibility) so the shared
            // shaping/floor code below works for both branches.
            occluded = float(taken) - horizon_sum;
        } else {
        for (int i = 0; i < N_ao; ++i) {
            float u1 = rand(seed_stable + uvec3(i, 7u, 53u));
            float r_h = sqrt(u1);
            float y   = sqrt(max(0.0, 1.0 - u1));
            vec2  v   = kVogelDisk[i & 31];
            float vx  = pp_c_a * v.x - pp_s_a * v.y;
            float vy  = pp_s_a * v.x + pp_c_a * v.y;
            vec3 d = (r_h * vx) * ao_tan + y * N + (r_h * vy) * ao_bit;
            if (any_hit(origin_ao, d, ao_radius)) occluded += 1.0;
            ++taken;
        }
        }
        // Two-step shaping to control corner overlap-darkness:
        //   1. sqrt curve on raw AO — compresses the [0..1] range so the
        //      delta between 30% occluded (1 edge) and 60% (2 edges
        //      overlapping) reads as a small darkening step instead of
        //      "twice as dark". Linear AO stacked occlusions
        //      multiplicatively, perceptually wrong at inner corners.
        //   2. ao_floor remap so even fully-occluded never crushes to
        //      pure black.
        float raw = 1.0 - (occluded / float(taken));
        raw = sqrt(raw);
        float ao_floor_v = scene.rt_lod.w;
        ao = mix(ao_floor_v, 1.0, raw);
        // Mild AO fade only — the origin-lift above is the primary
        // false-hit fix; this just softens any residual mid-distance
        // patchiness without flattening the look entirely.
        if (is_terrain_pre) {
            float ao_far_t = smoothstep(150.0, 400.0, cam_dist);  // P5
            ao = mix(ao, 1.0, ao_far_t);
        }
    }

    // --- Path-traced GI (1..N bounces) ---
    // Each sample follows a path of up to N_bounces ray segments.
    //   - On a surface hit: accumulate emissive + albedo×scene_light (a cheap
    //     local-light approximation), tint the throughput by surface albedo,
    //     bounce in a cosine-weighted hemisphere oriented around the
    //     approximate hit normal (-ray_dir).
    //   - On a miss (sky): accumulate throughput × procedural sky and stop.
    // Samples are stratified on a sqrt(N)×sqrt(N) grid for clean low-N look.
    vec3 gi_indirect = vec3(0.0);
    // Sky visibility — fraction of first-bounce GI rays that escape the
    // scene. Used after the GI loop to attenuate the (sky-derived) ambient
    // term: enclosed surfaces (e.g. inside the keep) get sky_vis в‰€ 0 and
    // their ambient drops to near zero, giving the room a real "interior"
    // feel instead of the same flat brightness as outdoors. Default 1.0
    // when GI is disabled so existing behaviour is unchanged.
    float sky_vis = 1.0;
    int sky_misses = 0;
    int sky_total  = 0;
    int N_gi = scene.rt_flags2.x > 0 ? lod_samples(scene.rt_flags2.x, cam_dist) : 0;
    int N_bounces = max(1, scene.rt_flags2.z);
    // Terrain skips GI entirely — adjacent triangles' N differs slightly
    // and cos_hemi() is N-relative, so per-triangle hit/miss flips on
    // BLAS detail produce patchy bright/dark faces. The earlier code
    // fired the rays then forced gi_indirect=0 / sky_vis=1; gating here
    // saves all those rays.
    //
    // gi_strength gate: when the slider is at 0 the loop's sole output
    // (gi_indirect *= rt_params2.x at line 1308) is multiplied by 0 —
    // every ray is pure waste. Skip the loop and keep sky_vis = 1.0
    // (ambient stays full, matching "GI off" expectation in the UI).
    // Cheap sky-vis probe for TERRAIN pixels (full GI loop is gated off
    // for terrain — adjacent triangle normals diverge enough that
    // cos_hemi sampling produces patchy bright/dark faces). Without
    // this, terrain inside enclosed structures (the little house, the
    // castle keep) stays at sky_vis = 1.0 → full sky ambient → bright
    // green floor while the cube walls/ceiling around it darken to
    // pitch black. Fire 4 hemisphere rays around +Y with the terrain
    // BLAS masked out (so terrain doesn't self-occlude). Distance-LOD'd
    // to near pixels — far terrain doesn't matter for interior look.
    // Gate also on gi_strength (rt_params2.x) — sky_vis output feeds
    // the ambient term that's later multiplied by gi_strength. When GI
    // is off the result is multiplied by 0 anyway; firing 4 BVH probe
    // rays per terrain pixel was pure waste.
    if (is_terrain_pre && scene.rt_flags.x != 0 &&
        scene.rt_params2.x > 1e-4 && cam_dist < 50.0) {
        const int kProbeN = 4;
        // Terrain instance mask is 0x05 (bits 0+2): 0xFD still hit it. 0xFA
        // = every bit except 0 and 2 -> castle / brushes / props only.
        const uint kProbeMask = 0xFAu;
        vec3 probe_origin = vWorldPos + vec3(0.0, 0.10, 0.0);
        int probe_misses = 0;
        for (int i = 0; i < kProbeN; ++i) {
            float r1 = rand(seed_base + uvec3(uint(i), 41u, 67u));
            float r2 = rand(seed_base + uvec3(uint(i), 13u, 89u));
            float r_h = sqrt(r1);
            float phi = 6.28318530718 * r2;
            vec3 dir = vec3(r_h * cos(phi),
                            sqrt(max(0.0, 1.0 - r1)),
                            r_h * sin(phi));
            if (!any_hit_m(probe_origin, dir, 60.0, kProbeMask)) {
                probe_misses += 1;
            }
        }
        sky_misses = probe_misses;
        sky_total  = kProbeN;
        sky_vis    = float(probe_misses) / float(kProbeN);
        sky_vis    = mix(sky_vis, 1.0, smoothstep(35.0, 50.0, cam_dist));   // no ring at 50 m
    }
    if (N_gi > 0 && !is_terrain_pre && scene.rt_params2.x > 1e-4) {
        float gi_radius = scene.rt_params2.y;

        // Sky-only ambient term — used when a GI bounce hits a surface
        // that CANNOT see the sun. Approximates the soft fill light from
        // the sky dome that reaches indoor surfaces near openings. 0.05
        // (~5% of sky color) keeps deep interiors believably dark; the
        // earlier 0.18 made every shadowed bounce hit accumulate more
        // light than a real enclosed room would have.
        vec3 sky_fill = scene.sky_color.rgb * 0.05;

        int strata = int(ceil(sqrt(float(N_gi))));
        float inv_strata = 1.0 / float(strata);
        int taken = 0;
        vec3 sum = vec3(0.0);
        // Hoist the receiver TBN out of the GI sample loop — N is constant
        // for all samples at this pixel, so the cross+normalize that
        // cos_hemi() does internally was being repeated N_gi times for
        // identical inputs. Bounce hemispheres still rebuild because
        // hit_n changes per bounce.
        vec3 N_up   = abs(N.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
        vec3 N_tan  = normalize(cross(N_up, N));
        vec3 N_bit  = cross(N, N_tan);
        vec3 ray_origin0 = shading_pos + N * 0.01;
        for (int sy = 0; sy < strata && taken < N_gi; ++sy) {
            for (int sx = 0; sx < strata && taken < N_gi; ++sx) {
                float r1 = rand(seed_base + uvec3(taken, 73u, 11u));
                float r2 = rand(seed_base + uvec3(taken, 91u, 47u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;

                // Inline cos_hemi using the hoisted TBN.
                float r_h = sqrt(u1);
                float phi = 6.28318530718 * u2;
                vec3 ld = vec3(r_h * cos(phi),
                               sqrt(max(0.0, 1.0 - u1)),
                               r_h * sin(phi));
                vec3 ray_dir = ld.x * N_tan + ld.y * N + ld.z * N_bit;
                vec3 ray_origin = ray_origin0;
                vec3 throughput = vec3(1.0);
                vec3 path = vec3(0.0);

                for (int b = 0; b < N_bounces; ++b) {
                    float t;
                    int idx;
                    int prim;
                    bool tlas_hit = closest_hit_material(ray_origin, ray_dir,
                                                         gi_radius, t, idx, prim);
                    // Voxel tower as a GI surface: march the brickmap and
                    // keep whichever hit is nearer. The voxel albedo bleeds
                    // its colour into the GI gather of nearby terrain/brushes.
                    float vt; uint vpal;
                    bool vox_hit = vox_march(ray_origin, ray_dir,
                                             tlas_hit ? t : gi_radius, vt, vpal);
                    bool use_vox = vox_hit && (!tlas_hit || vt < t);
                    if (!tlas_hit && !vox_hit) {
                        path += throughput * sample_sky(ray_dir);
                        // First-bounce miss = the surface can see sky in
                        // this direction. Tracks "open vs enclosed" for the
                        // ambient attenuation below.
                        if (b == 0) ++sky_misses;
                        break;
                    }
                    vec3 m_color;
                    vec3 m_emissive;
                    float hit_t;
                    if (use_vox) {
                        m_color    = voxel_albedo(vpal);
                        m_emissive = vec3(0.0);
                        hit_t      = vt;
                    } else {
                        // Static-castle hit: 1 BLAS instance covers all
                        // brushes — recover the per-brush material slot via
                        // primitive id.
                        int mat_idx = (idx == kStaticBlasSentinel)
                                        ? (prim / kCubeTrisPerBox)
                                        : idx;
                        Material m = materials[mat_idx];
                        m_color    = m.color.rgb;
                        m_emissive = m.emissive.rgb;
                        hit_t      = t;
                    }
                    vec3 hit_pos = ray_origin + ray_dir * hit_t;
                    vec3 hit_n = -ray_dir;  // approximate outward normal

                    // *** GI bounce lighting *** — fires a shadow ray
                    // from the hit point to the sun for bounce levels up
                    // to gi_shadow_max_bounce (rt_lod.z). Bounces beyond
                    // that fall back to sky_fill alone. 0 = no GI sun
                    // shadows (cheap, every shadowed bounce uses fill);
                    // 1 = first bounce only (default, the cheap one
                    // that gives the indoor-vs-outdoor delta we want);
                    // higher = more accurate at proportional cost.
                    int gi_shadow_max = int(scene.rt_lod.z);
                    vec3 hit_light = sky_fill;
                    if (b < gi_shadow_max) {
                        float n_dot_sun = dot(hit_n, scene.sun_direction.xyz);
                        if (n_dot_sun > 0.0) {
                            if (!any_hit(hit_pos + hit_n * 0.01,
                                         scene.sun_direction.xyz, 200.0)) {
                                hit_light += scene.sun_color.rgb *
                                             scene.sun_color.a * n_dot_sun;
                            }
                        }
                    }
                    path += throughput * (m_emissive +
                                          m_color * hit_light);
                    if (b + 1 >= N_bounces) break;

                    // Tint the path's throughput by the surface albedo and
                    // continue from the hit point in a new cosine-hemisphere.
                    throughput *= m_color;
                    ray_origin = hit_pos + hit_n * 0.01;
                    float br1 = rand(seed_base + uvec3(taken * 7 + b, 31u, 17u));
                    float br2 = rand(seed_base + uvec3(taken * 7 + b, 7u, 91u));
                    ray_dir = cos_hemi(br1, br2, hit_n);
                }
                sum += path;
                ++taken;
            }
        }
        vec3 cur_avg = sum / float(taken);

        // Session-3 temporal reservoir reuse. When restir is on, motion-
        // reproject the previous frame's reservoir at this pixel, gate
        // on the surface normal (sample_dir field carries N from the
        // writing pixel), and exponential-blend toward the prev radiance
        // with weight prev_M / (prev_M + 1). Cap M at restir_params.y so
        // a static camera doesn't lock onto a stale sample forever.
        ivec2 ipix = ivec2(gl_FragCoord.xy);
        uint  idx  = uint(ipix.y) * uint(scene.viewport.x) + uint(ipix.x);

        // Session-5 ring addressing. One SSBO, 3 per-pixel regions;
        // frame%3 is the write region, (frame+2)%3 the read region.
        // Reading a strictly older region than we write means a
        // spatial tap never sees this frame's partial writes (kills
        // the session-3 single-buffer aliasing race), and the
        // prev↔cur swap costs no descriptor update.
        uint  resPx   = uint(scene.viewport.x) * uint(scene.viewport.y);
        uint  resFrm  = uint(scene.rt_flags.w);
        uint  resCur  = (resFrm % 3u) * resPx;          // write base
        uint  resPrev = ((resFrm + 2u) % 3u) * resPx;   // read  base

        // Disocclusion thresholds (session 5). The session-3 quarantine
        // root cause #2 was normal-only acceptance pulling GI from a
        // different surface at the same orientation (parallel walls,
        // separate floor patches). We now also require the candidate's
        // recorded camera distance (Reservoir.pad) to match this
        // fragment's within a distance-scaled band. Constants, not UBO
        // slots — restir_params.zw is owned by FSR jitter.
        const float kResNrmCos  = 0.92;   // ~23° max normal deviation
        const float kResDepthRel = 0.04;  // 4% of camera distance …
        const float kResDepthAbs = 0.20;  //  … + 20 cm absolute slack
        float resDepthTol = kResDepthRel * cam_dist + kResDepthAbs;

        vec3 blended = cur_avg;
        uint M_new   = 1u;
        if (scene.restir_params.x > 0.5 && vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            vec2 prev_uv  = prev_ndc * 0.5 + 0.5;
            if (prev_uv.x > 0.0 && prev_uv.x < 1.0 &&
                prev_uv.y > 0.0 && prev_uv.y < 1.0) {
                ivec2 prev_pix = ivec2(prev_uv * scene.viewport.xy);
                prev_pix = clamp(prev_pix,
                                 ivec2(0),
                                 ivec2(scene.viewport.xy) - ivec2(1));
                uint prev_idx = uint(prev_pix.y) * uint(scene.viewport.x)
                              + uint(prev_pix.x);
                Reservoir prev = u_reservoir_prev.r[resPrev + prev_idx];
                float prev_d = uintBitsToFloat(prev.pad);
                if (prev.M > 0u &&
                    dot(prev.sample_dir, N) > kResNrmCos &&
                    abs(prev_d - cam_dist) < resDepthTol) {
                    uint M_max = uint(scene.restir_params.y);
                    M_new = min(prev.M + 1u, M_max);
                    float w_prev = float(M_new - 1u) / float(M_new);
                    blended = mix(cur_avg, prev.radiance, w_prev);
                }
            }
        }

        // Session-4 spatial neighbour merge: REMOVED in session 5.
        // The 3-tap variant added 3 divergent 48-B SSBO loads per
        // GI pixel and measured a 3.7× median / 12× castle GPU-time
        // regression — it cost far more than the ¼-sample ray saving
        // it was meant to converge. Temporal reuse alone recovers
        // most of the convergence at near-zero extra cost (one load +
        // one store), which is the point of ReSTIR here. If spatial
        // is ever revisited it must be a separate cheap compute pass,
        // not an inline per-fragment gather.
        vec3 shade_radiance = blended;

        // SVGF GI denoiser feed (Session 1). Store the unmodulated
        // irradiance signal (pre-albedo, post-temporal-reservoir-blend)
        // for future spatial-filter passes to denoise.
        imageStore(u_svgf_gi, ivec2(gl_FragCoord.xy),
                   vec4(shade_radiance, 1.0));

        // SVGF temporal accumulator (Session 2). When the toggle is on
        // (terrain_local_info.w), EMA-blend with the previous frame's
        // accumulated irradiance at the motion-reprojected pixel. .a
        // tracks M (sample count, 1..kSvgfMmax); EMA weight =
        // (M-1)/M. Frame parity picks read/write halves of the
        // ping-pong pair.
        vec3 svgf_out = shade_radiance;
        if (scene.terrain_local_info.w > 0.5) {
            const float kSvgfMmax = 8.0;
            uint   svFrame = uint(scene.rt_flags.w);
            ivec2  svPix   = ivec2(gl_FragCoord.xy);
            bool   parity0 = ((svFrame & 1u) == 0u);
            // Reprojected previous pixel from the existing vPrevClip
            // varying (already used by the motion-vec output).
            float  prevM   = 0.0;
            vec3   prevRgb = vec3(0.0);
            if (vPrevClip.w > 0.0) {
                vec2 pndc = vPrevClip.xy / vPrevClip.w;
                vec2 puv  = pndc * 0.5 + 0.5;
                if (puv.x > 0.0 && puv.x < 1.0 &&
                    puv.y > 0.0 && puv.y < 1.0) {
                    ivec2 ppx = ivec2(puv * scene.viewport.xy);
                    ppx = clamp(ppx, ivec2(0),
                                ivec2(scene.viewport.xy) - ivec2(1));
                    vec4 prev = parity0
                        ? imageLoad(u_svgf_hist1, ppx)
                        : imageLoad(u_svgf_hist0, ppx);
                    prevRgb = prev.rgb;
                    prevM   = prev.a;
                }
            }
            // DISOCCLUSION REJECTION. The reprojected history pixel may
            // have belonged to a DIFFERENT surface (sky/terrain that was
            // visible at this screen position before the camera moved).
            // Blindly EMA-blending that history painted the background's
            // irradiance onto walls — every wall read as semi-transparent
            // while moving, scaling with gi_samples. If the history
            // luminance is far outside the current estimate's range,
            // restart accumulation at this pixel (brief local noise at
            // disocclusion edges — the standard trade).
            {
                float lc = dot(shade_radiance, vec3(0.3333)) + 0.02;
                float lp = dot(prevRgb,        vec3(0.3333)) + 0.02;
                float ratio = max(lc / lp, lp / lc);
                if (ratio > 3.0) {
                    prevRgb = shade_radiance;
                    prevM   = 0.0;
                } else if (ratio > 1.8) {
                    // Soft zone: shorten history instead of full reset.
                    prevM = min(prevM, 3.0);
                }
            }
            float Mnew = min(prevM + 1.0, kSvgfMmax);
            float wPrev = (Mnew - 1.0) / Mnew;
            svgf_out = mix(shade_radiance, prevRgb, wPrev);
            // Write to the OPPOSITE slot of what we just read.
            vec4 storeVal = vec4(svgf_out, Mnew);
            if (parity0) imageStore(u_svgf_hist0, svPix, storeVal);
            else         imageStore(u_svgf_hist1, svPix, storeVal);
        }
        gi_indirect = albedo * svgf_out * scene.rt_params2.x;
        // Hard ceiling — a bounce ray happening to land on a sun-lit
        // white wall + multiple bounces can compound to 10×+ sun color
        // for a single sample. Without a cap the bright spike feeds
        // bloom + auto-exposure → next frame's exposure plummets →
        // next frame brightens → oscillation reads as surface flicker.
        // Clamp at 6 per channel: caps the firefly without crushing
        // legitimate strong bounces in the visible range.
        gi_indirect = min(gi_indirect, vec3(6.0));
        sky_total = taken;
        sky_vis = float(sky_misses) / float(max(1, sky_total));

        // Reservoir writeback for next frame's temporal combine. Stores
        // the pre-spatial-merge radiance so the spatial blur stays a
        // per-frame display effect, not a temporally-compounding one.
        Reservoir cur;
        cur.sample_dir = N;
        cur.radiance   = blended;
        cur.W          = 1.0;
        cur.w_sum      = float(M_new);
        cur.M          = M_new;
        cur.pad        = floatBitsToUint(cam_dist);  // depth disoccl. key
        u_reservoir_cur.r[resCur + idx] = cur;
    }

    // --- AO --- (also stratified to spread samples cleanly)
    // [moved below, recomputed from N_ao]

    // Sky-occlusion term. Surfaces that can't see the sky get a much
    // darker ambient — that's the natural look of an enclosed room. The
    // 0.15 floor stops pure-interior pixels from going completely black
    // (real rooms have some bounced light), and the smoothstep curve
    // rolls in the sky contribution faster than linear so the open arena
    // still reads bright. GI-disabled fallback (sky_vis=1.0) leaves the
    // outdoor look identical to before this change.
    // Combine: sky_factor only attenuates the sky-derived part of
    // ambient, the ground term carries through everywhere. AO multiplies
    // the whole thing so edges still darken, but never to zero (the
    // ground term keeps a baseline). 0.10 floor on sky_factor is a tiny
    // gesture toward "windowless rooms still have some light leakage."
    float sky_factor = mix(0.10, 1.0, smoothstep(0.0, 0.6, sky_vis));
    // Non-terrain receivers (brick walls, keep walls, tile floors) also
    // attenuate the constant ground term by sky_factor so deep cracks /
    // corners actually go dark — without this, the ground baseline keeps
    // walls bright even when GI rays mostly hit other walls and never
    // see the sky. Terrain keeps the original "ground term carries
    // through everywhere" behaviour because its GI is gated off and
    // sky_factor would otherwise stick at 0.10 forever.
    float ground_atten = is_terrain_pre ? 1.0
                                        : mix(0.25, 1.0, sky_factor);
    vec3 ambient_combined = mix(ambient_ground * ground_atten,
                                 ambient_sky * sky_factor, up);
    vec3 indirect = albedo * ambient_combined * ao + gi_indirect;
    // Terrain sky-bounce wrap. Back-facing pixels (-N·L > 0) pick up a
    // sky-tinted lift to model atmospheric scattering bouncing onto the
    // shadow side of mountains. Applied as ambient here (not as direct)
    // so it never overrides a real shadow-ray result. wrap_strength
    // tunes the intensity from the Settings UI.
    if (is_terrain_pre) {
        float wrap_amt = clamp(scene.terrain_params.y, 0.0, 1.0);
        float back = max(0.0, -dot(N, L));
        indirect += albedo * scene.sky_color.rgb * back * 0.30 * wrap_amt;
    }
    vec3 final = direct + indirect + vEmissive.rgb;

    // Realistic atmospheric distance fog. Was a terrain-only sky-tint
    // mix; now applies to ALL pixels (terrain + brushes + dyn props)
    // and uses the user-tunable distance_fog_* slots from the scene
    // UBO so the menu's fog sliders actually drive what you see.
    //   distance_fog_color (_scene_pad[17]): .rgb base fog tint,
    //     .a master strength (0 = disabled).
    //   distance_fog_params (_scene_pad[18]):
    //     .x density (exp^2 falloff per metre),
    //     .y start distance (no fog before this),
    //     .z height_top (>0 = fog density falls off above this height),
    //     .w max alpha (clamp on the fog amount).
    // Falls back to the previous sky-atmosphere mix at strength 0 so
    // the scene never loses depth cues even if the user zeroes the
    // fog slider.
    {
        vec4 dfog_col  = scene.distance_fog_color;
        vec4 dfog_par  = scene.distance_fog_params;
        vec3 view_dir  = normalize(vWorldPos - scene.camera_pos.xyz);
        float fog_t    = 0.0;
        vec3  fog_rgb  = dfog_col.rgb;
        // Realistic exp^2 atmospheric fog when the master strength is on.
        if (dfog_col.a > 1e-3) {
            float density  = max(0.0, dfog_par.x);
            float start_d  = max(0.0, dfog_par.y);
            float h_top    = dfog_par.z;
            float max_a    = clamp(dfog_par.w, 0.0, 1.0);
            float d_raw    = max(0.0, cam_dist - start_d);
            float dx       = d_raw * density;
            float fog      = 1.0 - exp(-dx * dx);
            // Height fog: density thins out above height_top so peaks
            // poke out of the layer. h_top <= 0 disables (uniform fog).
            if (h_top > 0.5) {
                float h_w = 1.0 - smoothstep(0.0, h_top, vWorldPos.y);
                fog *= h_w;
            }
            fog_t = clamp(fog * dfog_col.a, 0.0, max_a);
            // Mie scattering: warm forward-scatter halo toward the sun.
            // The fog tint shifts toward sun_color where the view direction
            // is aligned with the sun (anisotropic forward scattering).
            // Henyey-Greenstein approximation with g=0.76.
            vec3  Lf = scene.sun_direction.xyz;
            float mu  = clamp(dot(view_dir, Lf), -1.0, 1.0);
            const float g  = 0.76;
            const float g2 = g * g;
            // pow(x, 1.5) = x * sqrt(x) — one mul + one sqrt is much cheaper
            // than exp/log pair pow() takes for a fractional exponent.
            float hg_d = max(0.0001, 1.0 + g2 - 2.0 * g * mu);
            float hg = (1.0 - g2) / (12.566 * hg_d * sqrt(hg_d));
            // Normalised so isotropic = 1. hg peaks ~10 at forward.
            float mie = clamp(hg * 0.40, 0.0, 1.5);
            vec3  sun_tint = scene.sun_color.rgb * scene.sun_color.a;
            fog_rgb = mix(dfog_col.rgb,
                          dfog_col.rgb * 0.7 + sun_tint * 0.55,
                          clamp(mie, 0.0, 1.0));
        }
        // Aerial-perspective fallback / overlay: distant terrain always
        // gets some sky-tinted haze even with the user fog disabled,
        // so silhouettes don't pop in flatly at 1.5 km. terrain_params.x
        // is the legacy slider; keep it as the overlay strength.
        float legacy_fog = clamp(scene.terrain_params.x, 0.0, 1.0);
        if (is_terrain_pre && cam_dist > 250.0 && legacy_fog > 1e-3) {
            vec3 sky_col = sample_sky_atmosphere(view_dir);
            // Phase 6: forward-scatter tint on the aerial haze — looking
            // toward the sun, distant haze warms to the sun colour
            // (small Henyey-Greenstein lobe, g = 0.6).
            {
                float mu2 = clamp(dot(view_dir, scene.sun_direction.xyz),   // +1 = looking at the sun
                                  -1.0, 1.0);
                float hd = max(0.0001, 1.36 - 1.2 * mu2);
                float hg2 = 0.64 / (12.566 * hd * sqrt(hd));
                sky_col = mix(sky_col,
                              sky_col * 0.65 + scene.sun_color.rgb *
                              scene.sun_color.a * 0.5,
                              clamp(hg2 * 0.5, 0.0, 0.6));
            }
            float fl = clamp((cam_dist - 250.0) / 1100.0, 0.0, 0.95);
            fl = fl * fl * (3.0 - 2.0 * fl) * legacy_fog;
            // Take the MAX so the realistic fog wins where it's
            // active, but legacy aerial perspective shows on its own
            // if the master fog is off.
            if (fl > fog_t) {
                fog_t   = fl;
                fog_rgb = sky_col;
            }
        }
        // ---- Aerial colour grading (AAA Phase 6) ----
        // Before the haze takes over, distant surfaces desaturate and
        // cool — the scattering-in-depth look every AAA open world has.
        if (is_terrain_pre) {
            float ap = smoothstep(180.0, 1200.0, cam_dist) * 0.5;
            float lum_ap = dot(final, vec3(0.299, 0.587, 0.114));
            vec3 graded = mix(final, vec3(lum_ap), 0.45) *
                          vec3(0.93, 0.98, 1.08);
            final = mix(final, graded, ap);
            // Valley haze: a thin pool below the plateau altitude that
            // only reads at distance — dawn-valley layering.
            float vh = clamp((24.0 - vWorldPos.y) * 0.05, 0.0, 0.45) *
                       smoothstep(150.0, 550.0, cam_dist);
            if (vh > 1e-3) {
                vec3 vfc = mix(scene.sky_color.rgb,
                               scene.sun_color.rgb, 0.25) * 0.9;
                final = mix(final, vfc, vh);
            }
        }
        if (fog_t > 1e-4) final = mix(final, fog_rgb, fog_t);
    }

    outColor = vec4(final, 1.0);

    // G-buffer write — RAW albedo + material id so deferred_lighting.frag
    // can re-shade with its own sun/PCSS/RTAO/GI/point-light passes (RDR2
    // RAGE style). cube.frag still produces `final` above and ships it to
    // scene_color so the forward path is unchanged; the deferred path
    // (--deferred 1) reads gbuffer0/1 and recomputes its own lit output.
    //
    // material_id mapping:
    //   1 = generic world opaque (stone, brick, painted, tile)
    //   2 = wood
    //   3 = terrain (skips GI, boosts ambient)
    //   4 = grass / vegetation (extra wrap-around fill)
    // is_terrain_pre (vTexParams.w > 1.5) is the only branch we can
    // distinguish from the gbuffer write site without dragging more
    // vertex data through; the rest fall through to mat 1 and the
    // deferred shader's Lambert handles them. Per-material tightening
    // (wood/grass) can be added later via a per-brush material slot in
    // vColor.a or a 3rd gbuffer slot.
