// Part of main() in terrain_raymarch.frag: the ground fog (ray-marched) and the haze of distance.
// Not a file of its own - it is included in the middle of main() and uses its local
// variables (ro, rd, t, pos, nor, col, ...). Split off only to keep that file readable;
// the order of the parts is the order of the #include lines there.

    // === Volumetric ground fog (real ray-march) =================
    // Multi-step march along the camera ray with Beer-Lambert
    // transmittance accumulation + Henyey-Greenstein phase-weighted
    // sun in-scattering. The Frostbite-style energy-conserving
    // integration:
    //   Sint = (S - S * exp(-σe * dt)) / σe    [exact for constant σ
    //                                            over the segment]
    //   trans *= exp(-σe * dt)
    //   col_out += trans_prev * Sint
    // Fog density = exp(-y / scaleHeight) × (1 + slow FBM scroll) so
    // the fog has subtle volumetric variation (wisps in the valleys)
    // instead of a uniform sheet. No per-step volumetric shadow ray
    // — the HG phase already gives the directional sun glow.
    {
        // Slider-driven strength; skip the entire march when 0.
        float fogStrength = pc.grass_params.x;
        bool  fogGodRays  = pc.grass_params.z > 0.5;
        if (fogStrength > 0.001) {
        // Distance-LOD on the fog march: full 12 steps near the
        // camera, drop to 4 past 150 m. Distant terrain pixels with
        // 16-step fog were the dominant per-pixel cost and the
        // direct cause of the GPU TDR — far fog can't visibly
        // resolve sub-step density anyway.
        // Linear ramp 12→4 across 100..250 m. The previous binary cliff
        // at 150 m caused wave divergence on neighbouring pixels and
        // visible TAA shimmer along the cutoff line.
        float dist_cam = distance(pos, scene.camera_pos.xyz);
        int kVolSteps = max(4, int(mix(12.0, 4.0,
                                        smoothstep(100.0, 250.0, dist_cam))));
        const float kVolDensityBase   = 0.030;  // σe at full density inside the band
        const float kVolPhaseG        = 0.65;   // forward-scattering bias
        // Band parameters from settings (fog_band slot in scene UBO).
        float fog_y_start = scene.fog_band.x;
        float fog_y_top   = scene.fog_band.y;
        float fog_noise   = clamp(scene.fog_band.z, 0.0, 1.0);

        vec3 fogTint   = vec3(0.78, 0.83, 0.90) * scene.crater_info.w;   // unlit fog is dark at night
        vec3 sunGlow   = scene.sun_color.rgb * scene.sun_color.a;

        // Henyey-Greenstein phase: forward-peaked when looking near
        // the sun. The denominator goes near-zero as cosTh→1, so
        // phase can spike past 100 and blow up the bloom + auto-
        // exposure (visible flash + sometimes drives a TDR).
        // Clamp the denominator's growth so phase stays bounded.
        float cosTh = clamp(dot(rd, sunDir), -1.0, 0.985);
        float gg    = kVolPhaseG * kVolPhaseG;
        float phase = (1.0 - gg) /
                      (4.0 * 3.14159265 *
                       // pow(x, 1.5) = x * sqrt(x) — one mul + one
                       // sqrt, way cheaper than the exp/log path pow
                       // takes for fractional exponents.
                       (1.0 + gg - 2.0 * kVolPhaseG * cosTh) *
                       sqrt(1.0 + gg - 2.0 * kVolPhaseG * cosTh));
        // Hard ceiling matches a typical "sun glow" magnitude.
        phase = min(phase, 4.0);
        // P16: hoist sunGlow * phase out of the per-step inner loop.
        // Both are loop-invariants — the scattering line at the hot
        // spot below was multiplying them every iteration. ~3 ALU/step
        // back × kVolSteps + early-out cap.
        vec3 sunPhaseTerm = sunGlow * phase;

        // Cap march length at the surface hit. Sky pixels never reach
        // here (we returned -1 above), so t is always finite.
        float t_end = t;
        float t_max = min(t_end, 600.0);   // cap so we don't waste taps in fog-noise pixels far away
        float dt    = t_max / float(kVolSteps);
        // Per-pixel jitter on the start position breaks step banding
        // — combined with Phase 5's surface jitter this stays dither-stable.
        // Stable per-pixel jitter (no frame number) so the volumetric
        // step pattern is fixed in screen space — TAA's spatial
        // filter erases it. Per-frame variation made the dither
        // pattern shift between frames and TAA couldn't keep up.
        float jStart = ignBase(gl_FragCoord.xy + vec2(39.1, 9.7));
        float t_v   = dt * jStart;

        vec3  scatter = vec3(0.0);
        float trans   = 1.0;
        for (int i = 0; i < kVolSteps; ++i) {
            vec3 p = ro + rd * t_v;
            // Density profile: 0 below y_start, ramps up over 1 m,
            // 1 inside the band, falls off over 4 m above y_top.
            // Compact band that hugs the valleys instead of a global
            // exp falloff.
            float profile = smoothstep(fog_y_start, fog_y_start + 1.0, p.y) *
                            (1.0 - smoothstep(fog_y_top, fog_y_top + 4.0, p.y));
            // Outside the fog layer the density is exactly 0: the step adds no
            // scatter and no extinction - skip its wisp fetch and god-ray taps.
            if (profile <= 0.0) { t_v += dt; continue; }
            // 3-octave wisp pattern, pre-baked into the 256² R8 fog
            // wisp texture (binding 14). One textureLod replaces the
            // 3 noise2 calls the inline FBM used to do (~12 hashes
            // per tap). REPEAT addressing handles time scrolling and
            // large camera moves seamlessly.
            vec2 q = p.xz * 0.020 + vec2(scene.water_params.w * 0.05);
            float w = textureLod(u_fog_wisp, q * (1.0 / 16.0), 0.0).r;
            // Centred around 0.5 so noise = 1.0 → "no modulation" and
            // noise = 0 → density ×0.2 (thin patch). Strength slider
            // controls how much the noise can carve gaps.
            float wisp = mix(1.0, 2.0 * w, fog_noise);
            wisp = clamp(wisp, 0.2, 2.0);
            float sigma_e = kVolDensityBase * profile * wisp * fogStrength;

            // Phase 7 — fog self-shadow / god-rays. Step a few small
            // distances toward the sun and accumulate optical depth
            // through the same density field. Surface (terrain /
            // castle) shadows the fog because the surface RT shadow
            // `sha` already gates the unshadowed mass; the per-step
            // sample additionally darkens fog that's behind a fog-
            // density gradient, giving the godray look.
            float lightTrans = sha;
            if (fogGodRays) {
                // 2 inner steps × 1-octave wisp. Was 4 steps × 3 octaves
                // = 12 noise2 calls per fog tap; with ~12 outer taps
                // that was 144 hash evals per pixel for the godray work
                // alone. Single-octave wisp is the dominant low-freq
                // mode; the higher-freq components alias-tolerate at
                // godray resolution under TAA.
                const int kFogShadowSteps = 2;
                float ds = 9.0;          // bigger base to cover same length in fewer steps
                float ld = ds * 0.5;
                for (int j = 0; j < kFogShadowSteps; ++j) {
                    vec3 sp = p + sunDir * ld;
                    float prof_s = smoothstep(fog_y_start, fog_y_start + 1.0, sp.y) *
                                    (1.0 - smoothstep(fog_y_top, fog_y_top + 4.0, sp.y));
                    vec2 qs = sp.xz * 0.020 + vec2(scene.water_params.w * 0.05);
                    float w_s = textureLod(u_fog_wisp, qs * (1.0 / 16.0), 0.0).r;
                    float wisp_s = clamp(mix(1.0, 2.0 * w_s, fog_noise), 0.2, 2.0);
                    float sig_s = kVolDensityBase * prof_s * wisp_s * fogStrength;
                    lightTrans *= exp(-sig_s * ds);
                    ds *= 1.7;          // bigger growth so 2 steps still cover far range
                    ld += ds;
                    if (lightTrans < 0.02) break;
                }
            }

            // Single-scattering: σs в‰€ σe (assume cloudy isotropic
            // scattering, no absorption). Phase function gives the
            // sun-aligned directional component; ambient fogTint
            // keeps back-lit fog from going pitch-black.
            vec3  Lin   = sunPhaseTerm * lightTrans + fogTint * 0.25;
            // P16: exp2(-x * 1.4427) ≈ exp(-x), and exp2 is a single
            // hardware instruction on most archs vs exp's exp/log
            // pair. Hoisted constant 1.4427 = 1/ln(2).
#ifndef VIBE_WATER
            const float kLog2e = 1.44269504;   // log2(e) — converts exp() to exp2()
            float seg  = exp2(-sigma_e * dt * kLog2e);
#else /* VIBE_WATER */
            float seg  = exp2(-sigma_e * dt * 1.44269504);
#endif /* VIBE_WATER */
            vec3  Sint = (Lin - Lin * seg) / max(sigma_e, 1e-4);
            scatter += trans * Sint * sigma_e;   // re-multiply σs because we factored σe out
            trans   *= seg;
            t_v     += dt;
            if (trans < 0.02) break;             // early-out — fully foggy
        }
        // Apply: surface attenuated by trans, scatter added in front.
        col = col * trans + scatter;
        }   // fogStrength > 0
    }

