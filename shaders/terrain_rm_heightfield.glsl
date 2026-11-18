// terrain_rm_heightfield.glsl - part of terrain_raymarch.frag (included there, in place; not a standalone shader).
// Terrain height (FBM + ground relief) and the ray march against it

// Winter: the snow is a layer of its own on top of the ground, as in a
// ray-marched snow scene - a bumpy sheet smoothly joined (smooth maximum) with
// wind-blown drifts. It is part of the marched height, so it has a real
// outline, hides the rills under it, shades and shadows itself.
float snow_layer_h(vec2 wp) {            // metres of snow at full winter
    float d1 = noise2(wp * 0.15 + 31.0);                 // drifts, several metres across
    float d2 = noise2(wp * 0.50 + 7.0);                  // mounds
    float d3 = noise2(wp * 2.6 + 3.0);                   // lumps
    return 0.06 + 0.26 * d1 * d1 + 0.09 * d2 + 0.03 * d3;
}
float rm_smax(float a, float b, float k) { return log(exp(k * a) + exp(k * b)) / k; }
// Ground relief `d` (already 0 on the paving) with the snow on top.
// `road_keep`: 0 on the paving, 1 off it. On the paving the snow lies in
// patches a few cm thick; feet press prints into the snow.
float rm_snow_on(float d, vec2 wp, float h, float road_keep) {
    float sn = g_snow;
    if (sn < 0.01 || (scene.water_params.x > 0.5 && h < scene.water_params.y + 0.05)) return d;
    float open_h = rm_smax(d, sn * snow_layer_h(wp), 28.0);
    float road_h = 0.0;
    if (road_keep < 0.999)
        road_h = snow_road_patch(wp) * smoothstep(0.15, 0.6, sn) * (0.05 + 0.06 * noise2(wp * 1.7 + 5.0));
    float sh = mix(road_h, open_h, road_keep);
    float fd = foot_depth(wp);
    if (fd > 0.0) sh -= min(fd, max(sh - d, 0.0) * 0.9);     // never deeper than the snow is
    return sh;
}

float rm_ground_detail(vec2 wp, float t, float h) {
    float w = 1.0 - smoothstep(10.0, 26.0, t);
    if (w <= 0.0) return 0.0;
    // Not on the road: the paving is flat (crouching on it showed the
    // rills as noise in the cobbles). In winter the snow still lies on it.
    float road_keep = 1.0;
    {
        float keep = scene.road_info.y;
        float rdist = rm_road_dist_thr(wp, keep + 0.9);                 // paving only; the dirt verge keeps its relief
        road_keep = smoothstep(keep + 0.1, keep + 0.9, rdist);
        if (road_keep <= 0.0) {
            if (g_snow < 0.01) return 0.0;          // (a real branch, not ?: - the winter side has loops)
            return rm_snow_on(0.0, wp, h, 0.0) * w;
        }
    }
    float wn = noise2(wp * 0.21);                      // one noise warps both axes
    vec2  q  = wp * 0.55 + vec2(wn, 1.0 - wn) * 1.7;
    float g1 = 1.0 - smoothstep(0.0, 0.22, abs(noise2(q) * 2.0 - 1.0));              // main rills
    // While MARCHING (g_rm_ray_y set by raymarch()): the main rills only,
    // side cracks / soil displacement at their means - two noise taps
    // instead of three plus a texture fetch on every near-ground step. The
    // hit point lands within ~5 cm of the full surface; the normal and the
    // shading (called without the flag) still use the full relief.
    if (g_rm_ray_y > -1e8) return rm_snow_on((0.11 * (1.0 - g1) + 0.045 * 0.86 + 0.045 +
                                              rm_pebbles_march(wp, t)) * road_keep, wp, h, road_keep) * w;
    float lod = log2(max(t * 2.0 * scene.viewport.w * 0.45 * 2048.0, 1.0));
    float disp = textureLod(u_normal[9], wp * 0.45, lod).a;
    float g2 = 1.0 - smoothstep(0.0, 0.25, abs(noise2(q * 2.7 + 3.1) * 2.0 - 1.0));  // side cracks
    float d = 0.11 * (1.0 - g1) + 0.045 * (1.0 - g2) + 0.09 * disp;
    float sand = 1.0 - smoothstep(scene.water_params.y + 0.4, scene.water_params.y + 2.5, h);
    d += sand * 0.03 * (0.5 + 0.5 * sin(dot(wp, vec2(9.0, 3.5)) + 6.0 * wn));
    // (No stones here: this full variant is also what every terrain-shadow
    // and reflection step evaluates. The normal adds them itself, the march
    // has its own cut-down version above.)
    return rm_snow_on(d * road_keep, wp, h, road_keep) * w;
}

float terrainM(vec2 p) {

    vec2 wp = p;
    p *= TERRAIN_SCALE;
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    int kOct = int(pc.tex_params.z);
    for (int i = 0; i < kOct; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    // Option-2 brush amplifier: the brush's smooth bilinear delta
    // ALSO scales the local FBM amplitude. Brushed cells get more of
    // the SAME procedural FBM (per-pixel detail, same character as
    // the surrounding terrain) instead of a flat smooth bump. The
    // raise/lower offset is preserved at the end via `+= delta` so
    // the user can still sculpt arbitrary shapes; the amplifier is
    // additional, not destructive. AMP = 0.05 → 1 m raise gives +5 %
    // FBM amplitude; capped so a 60 m raise doesn't 4× the local
    // mountains. Inside the plateau the FBM is already mixed out, so
    // the amplifier has no effect there (plateau stays flat under
    // the brush — only the offset fires).
    float h = a * TERRAIN_HEIGHT;
    // Plateau blend — but instead of mixing toward a glass-flat
    // plateau_h, mix toward `plateau_h + small_detail_FBM` so the
    // plateau surface inherits the same procedural character as the
    // surrounding terrain (just smaller amplitude). Uses the SAME
    // accumulated `a` value so it's free, and reads as a continuous
    // scale-down of the surrounding mountains. The castle footprint
    // (|xz| < 11 m) is already discarded for raymarched terrain
    // earlier in the file, so castle siting isn't affected by this
    // detail; only the courtyard / plateau-rim area gains texture.
    // ±~1.5 m of FINE ground texture (2.5 m + 1.25 m wavelengths)
    // instead of scaled-down macro FBM (whose smallest feature is
    // 10 m → invisible at plateau scale).
    // Plateau is now tiny (11.5 m, hidden under castle) — no need
    // for extra ground-detail noise here.
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    h += sampleHeightDelta(wp);
    return h;
}


// Distance-LOD variant: drops octaves smoothly as the ray distance
// grows. The full octave count is the user's pc.tex_params.z; we
// fade down to ~2 octaves past 600 m. Same outer wrapping (plateau
// blend + sculpt delta) so close vs far results stay continuous —
// only the inner FBM converges to a coarser sum past the LOD start.
// Uses a uniform `kMaxOct` cap with an early-break so the loop body
// stays branch-friendly for the GPU.
// LOD ramp endpoints + minimum-octave floor come from push constants:
//   emissive.y = lod_near_m (full octaves below this distance)
//   emissive.z = lod_far_m  (lod_min_oct octaves above this distance)
//   emissive.w = lod_min_oct (floor, typically 2..8)
// Smoothstep ramp between the two endpoints.
// Height of the primary ray at the point being evaluated (set by raymarch()
// around its terrainM_lod calls, -1e9 otherwise = always exact). The close-
// range ground relief is at most ~0.275 m: while the ray is further above
// the base surface than that, its exact value cannot change the outcome,
// so a constant upper bound replaces the 3 noise taps + texture fetch
// (most march steps are in open air; this was ~40 % of the pass).
float terrainM_lod(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    // Fractional octave count — ceil() to include the partially-faded
    // top octave; `frac` is the [0..1] weight that octave gets so the
    // FBM is C0-continuous across the integer threshold.
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
#ifndef QLIKE_WATER
    // Pixel-footprint cap: octave i has wavelength 333 m / 2^i; drop
    // octaves smaller than ~1.5 pixels at this distance (sub-pixel
    // detail the march would only alias on). 1.68 = 2 tan(40 deg).
    {
        float fp = max(ray_t, 1.0) * 1.68 * scene.viewport.w;
        float oct_fp = log2(333.333 / (1.5 * fp)) + 1.0;
        oct_f = min(oct_f, max(oct_fp, float(oct_min)));
    }
#endif /* ! QLIKE_WATER */
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 24;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    // Plateau is now tiny (11.5 m, hidden under castle) — no need
    // for extra ground-detail noise here.
    float tpl = plateauWeight(wp);
    h = mix(h, pc.color.w, tpl);
    // Sculpt-delta texture fetch is sub-pixel past ~80 m and the
    // delta itself bilinear-samples to near-zero outside the brush
    // ring. Skip the texture tap on far raymarch steps — biggest
    // single texture-bandwidth win on terrain pixels.
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
    if (g_rm_ray_y - h > 0.34) {
        h += 0.30 * (1.0 - smoothstep(10.0, 26.0, ray_t)) * (1.0 - tpl);   // conservative bound
    } else {
        h += rm_ground_detail(wp, ray_t, h) * (1.0 - tpl);
    }
    return h;
}


// Variant of terrainH_lod that takes a precomputed plateau weight,
// so callers like calcNormal — which evaluates the FBM at three
// points within ~2 cm of each other — can compute plateauWeight
// once and pass it in instead of recomputing the same value 3×.
// Saves 2 × (2 abs + smoothstep) per shaded terrain pixel.
float terrainH_lod_pw(vec2 p, float ray_t, float tpl) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.w);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 32;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    h = mix(h, pc.color.w, tpl);
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
    h += rm_ground_detail(wp, ray_t, h) * (1.0 - tpl);
    return h;
}


// Generous headroom: sampleHeightDelta() (sculpt strokes + disk
// overlays) can push the FBM-derived terrain well above TERRAIN_HEIGHT.
// Underestimating max height triggers the open-air early-out in
// raymarch() while the camera is still inside the terrain volume,
// producing the "everything broken at high altitude" symptom.
//
// Must be the GLOBAL conservative cap. Tried clamping by a per-frame
// camera-local max (terrain_local_info.x) but that's unsafe: a shallow
// rising ray escapes once it's above the LOCAL max, yet a distant peak
// taller than that local max (outside the sample radius) would still
// be on the ray's path → the peak vanishes. A correct local cap needs
// a per-cell max sampled ALONG the ray (proper maximum-mipmap), not a
// single camera-vicinity value. terrain_local_info stays wired on the
// C++ side for when that lands; it is intentionally unused here.
float terrainMaxHeight() { return max(TERRAIN_HEIGHT, pc.color.w) + 80.0; }

// Shadow-march FBM. Identical to terrainM_lod (same user octave
// count + LOD ramp) EXCEPT the sampleHeightDelta texture tap is
// dropped — the sculpt overlay is sub-pixel for shadow boundaries
// and would just be filtered out by TAA anyway. Saves one texture
// fetch per shadow march step (up to 32 per pixel) without changing
// what blocks light. Octave count remains the user's quality
// setting so they keep control over shadow sharpness.
//
// Plateau term is kept (otherwise the castle area's sun shadow
// would mismatch the primary terrain there).
float terrainM_shadow(vec2 p, float ray_t) {
    vec2 wp = p;
    p *= TERRAIN_SCALE;
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 24;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct) break;
        vec3 n = noised(p);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    float h = a * TERRAIN_HEIGHT;
    float t = plateauWeight(wp);
    h = mix(h, pc.color.w, t);
    // The edits to the ground (castle site lowered, road cut and levelled,
    // sculpting) live in the delta texture. Without it the shadow march saw
    // the ORIGINAL hill there and threw a phantom shadow over the road.
    h += sampleHeightDelta(wp);
    return h;
}




// Coarse 3-octave heightfield — strictly for AO + GI occlusion
// raymarches where sub-meter detail doesn't matter. Cuts the per-step
// noise cost vs terrainM() (which runs the full march octave count,
// typically 4-6) while staying conservative-ish for occlusion (the
// gross terrain shape is what blocks light).

// (terrain_blocks_ray + terrainCoarse removed — per-pixel terrain self-
// occlusion was replaced by mask-0x02 TLAS-only AO once the sun shadow
// map covered the gross-shadow case. The 3-octave coarse heightfield
// here was unused and just bloating the shader.)

// === Adaptive-step heightfield ray march ===
// h = ray.y - terrain(ray.xz) is a conservative step bound for any
// heightfield. Step factor 0.4 stays safe on sharp ridges; the
// distance-adaptive precision threshold avoids over-iterating distant
// pixels where one screen pixel spans many world meters.
// raymarch with optional max-distance cap. Pass `t_max < 0` for
// unbounded (default 1500 m). Caller can pass a tighter bound — e.g.
// the water-plane intersection distance — so the loop bails as soon
// as any terrain hit past the bound is irrelevant.
float raymarch(vec3 ro, vec3 rd, float t_max) {
    float maxH = terrainMaxHeight();
    float t = 0.0;
    if (ro.y > maxH && rd.y >= 0.0) return -1.0;
    if (ro.y > maxH) {
        // Skip the open-air segment by jumping to the upper bound plane.
        t = (ro.y - maxH) / (-rd.y);
    }
    float t_limit = (t_max > 0.0) ? min(t_max, 1500.0) : 1500.0;
    // From high up (flying out) the landscape starts beyond the range of this march:
    // nothing to hit. (Without this the first sample, hundreds of km away, "hit" on
    // float precision and the whole view filled with fogged ground.)
    if (t > t_limit) return -1.0;
    // Phase 5 — blue-noise jittered start offset. Per-pixel hash-based
    // sub-step shift breaks up the per-fragment march alignment, so
    // contour banding from a too-aggressive step factor disperses into
    // noise (then washes out under TAA). Free; no extra texture taps.
    float jitter = ignBase(gl_FragCoord.xy);
    int kSteps = int(pc.tex_params.x);
    float kStep = abs(pc.emissive.x);   // sign = full-resolution pass flag (see main)
    bool relax = pc.grass_params.y > 0.5;
    // Tiny initial offset; max half-cell so the surface hit shifts
    // sub-pixel only.
    t += jitter * 0.5;
    float maxH_cap = terrainMaxHeight();
#ifndef QLIKE_WATER
    // Eye at or under the marched surface. The full FBM sits up to ~0.5 m
    // above the baked heightmap the player stands on (plus the close-range
    // ground detail), so a crouching / crawling eye can be INSIDE it: every
    // ray then "hit" at once and the ground in front turned to noise. March
    // the baked heightmap for this pixel instead - the eye is always above that.
    if (all(lessThan(abs(ro.xz), vec2(kHeightmapSide * 0.5 - 2.0))) &&
        ro.y - terrainM_lod(ro.xz, 0.0) < 0.12) {
        float th = 0.0;
        for (int i = 0; i < 96; ++i) {
            vec3 p = ro + rd * th;
            if (rd.y > 0.0 && p.y > maxH_cap) return -1.0;
            if (any(greaterThan(abs(p.xz), vec2(kHeightmapSide * 0.5 - 2.0)))) return -1.0;
            float hh = p.y - meshHeight(p.xz);
            if (hh < 0.002 * th + 0.004) return th;
            th += max(hh * 0.6, 0.02 + 0.02 * th);     // grows with distance: reaches the horizon
            if (th > t_limit) return -1.0;
        }
        return -1.0;                                  // never reached the ground: sky, not a fake hit
    }
#endif
    float t_prev = t;
    for (int i = 0; i < kSteps; i++) {

        vec3 pos = ro + t * rd;
        // Sky escape: any ray rising above the FBM ceiling with a
        // positive y-component cannot ever hit terrain — bail early
        // instead of walking the budget out to t=1500. Saves the worst-
        // case grazing-ray cost (which used to burn the full kSteps).
        if (rd.y > 0.0 && pos.y > maxH_cap) return -1.0;
        // NOTE: the per-cell hi-Z max-grid skip was REMOVED. It went
        // through four correctness regressions (black dots, FBM-budget
        // exhaustion, and twice skipping real medium-distance low
        // terrain so sky showed through) and never became reliable —
        // the coarse 64 m cell max can't safely bound the fine FBM
        // surface a ray grazes at medium distance regardless of the
        // safety margin. The proven plain marcher below is correct;
        // the local-Lipschitz overstep (terrainM_lod_d's grad term)
        // and the distance-octave LOD are the kept perf wins. The
        // terrain_max_grid UBO data + C++ bake stay wired but unused
        // (zero runtime cost) in case a correct screen-space hi-Z
        // prepass is built later.
        // Distance-LOD on the FBM octave count — full octaves close
        // to camera, smoothly fading down at distance (sub-pixel
        // detail) for a perf saving on long rays. (Reverted to the
        // plain terrainM_lod; the _d grad variant is unused now.)
#ifndef QLIKE_WATER
        // Empty-space skipping on the BAKED heightmap: one texture fetch
        // instead of up to 14 noise octaves. The bake holds octaves 0..8 of
        // this same FBM at 1 m (plus sculpt + road carving); the full
        // surface differs from its bilinear sample by the dropped octaves
        // (<= ~0.5 m) plus interpolation error, so `kHmMargin` above it
        // is provably clear. Full FBM only in the last stretch.
        bool coarse = false;
        float h;
        {
            // Distance-LOD drops octaves above `oct`, so the marched surface
            // can sit BELOW the 9-octave bake by up to 2 * H * 2^-oct.
            float lod_f = smoothstep(pc.emissive.y, pc.emissive.z, t);
            int   ofl   = int(pc.tex_params.z);
            int   omn   = max(2, min(ofl, int(pc.emissive.w)));
            float oct   = float(ofl) - lod_f * float(ofl - omn);
            // Dropped octaves only LOWER the marched surface (the noise is
            // 0..1), so the bake plus a constant margin for the octaves above
            // the bake (<= ~0.5 m) and interpolation error is always clear.
            // (A margin growing with the dropped octaves reached ~17 m far
            // out and switched the skip off exactly where it saves most.)
            float kHmMargin = 2.0;
            float hm = 1e9;
            if (all(lessThan(abs(pos.xz), vec2(kHeightmapSide * 0.5 - 2.0))))
                hm = pos.y - meshHeight(pos.xz) - kHmMargin;
            coarse = hm > 0.75;
            if (!coarse) g_rm_ray_y = pos.y;
            h = coarse ? hm : pos.y - terrainM_lod(pos.xz, t);
            g_rm_ray_y = -1e9;
            // On the road the marched surface IS the baked (levelled) one.
            // The FBM's extra octaves stand up to ~0.5 m proud of it; seen
            // from a low eye along the road they formed a lumpy, spiky
            // hump across the paving a few metres ahead.
            if (!coarse && h < 0.65 && t < 90.0 && rd.y > -0.16) {   // grazing rays only: steep ones are fixed by the slide in main()
                float keep = scene.road_info.y;
                float rdist = rm_road_dist_thr(pos.xz, keep + 0.9);
                if (rdist < keep + 0.9)
                    h = mix(pos.y - meshHeight(pos.xz), h, smoothstep(keep, keep + 0.9, rdist));
            }
        }
#else /* QLIKE_WATER */
        g_rm_ray_y = pos.y;
        float h = pos.y - terrainM_lod(pos.xz, t);
        g_rm_ray_y = -1e9;
#endif /* QLIKE_WATER */
        // P7: loosen the hit threshold with distance. At grazing angle a
        // typical pixel covers 0.3-1 m of world space past 200 m, so
        // demanding sub-millimetre convergence wastes ~30 % of the
        // march steps on far rays. mix 0.0015 → 0.005 over [150, 600] m.
        float hit_eps = mix(0.0015, 0.0025, smoothstep(150.0, 600.0, t)) * t;
#ifndef QLIKE_WATER
        if (!coarse && abs(h) < hit_eps) return t;   // only real-surface steps may hit (was: break + a full re-evaluation below)
#else /* QLIKE_WATER */
        if (abs(h) < hit_eps) break;
#endif /* QLIKE_WATER */
        if (t > t_limit) return -1.0;
        // Phase 6 — relaxation cone-stepping. Step grows with
        // distance so the same ray covers more ground in fewer
        // iterations. 0.7 attenuation on h prevents penetration
        // when the relaxation factor over-shoots.
        // Distance-ramped step factor — past 200 m the FBM is LOD'd
        // down to lod_min_octaves so the field is much smoother and
        // big strides won't punch through fine detail. Up to 2.5×
        // close-up step factor by 600 m → roughly halves the worst-
        // case grazing-ray iteration count.
        float kStep_eff = kStep * mix(1.0, 2.5, smoothstep(200.0, 600.0, t));
        // (Reverted the local-Lipschitz overstep experiment — back to
        // the proven conservative kStep_eff * h advance.)
        float advance;
        if (relax) {
            float rl = max(t * 0.02, 1.0);
            advance = kStep_eff * h * rl * 0.7;
        } else {
            advance = kStep_eff * h;
        }
        // Once h goes negative we've crossed the surface — the next step
        // would walk us back, and with rl > 1 the back-step grows enough to
        // ping-pong indefinitely until kSteps caps. Break out instead so
        // the caller gets the closest approach.
        if (advance < 0.0) {
            // Overstepped into the ground (steep slope facing the camera, or
            // the far-distance step ramp). Returning this point put the hit
            // INSIDE the hill: depth too far, ridges eroded, and the shadow
            // ray started underground (black speckle). Bisect back to the
            // surface between the last point above it and this one.
            float ta = t_prev, tb = t;
            for (int k = 0; k < 5; ++k) {
                float tm = 0.5 * (ta + tb);
                vec3  pm = ro + tm * rd;
                if (pm.y - terrainM_lod(pm.xz, tm) > 0.0) ta = tm; else tb = tm;
            }
            return ta;
        }
        t_prev = t;
        t += advance;
    }
    // Loop exhausted without converging. The fall-through is a real
    // hit when the ray is still close to the surface (just slow to
    // converge under the LOD'd FBM at distance), but NOT a real hit
    // when the ray has clearly gone above any possible terrain — that
    // produces ghost terrain projected into the sky.
    vec3 final_pos = ro + t * rd;
    float final_h  = final_pos.y - terrainM_lod(final_pos.xz, t);
    // Out of steps while still clearly above the ground: carry on over the
    // baked heightmap (one fetch per step) instead of accepting a point in
    // the air as the hit.
    if (final_h > 4.0 * 0.0025 * t) {
        for (int k = 0; k < 32; ++k) {
            vec3 q = ro + t * rd;
            if (rd.y > 0.0 && q.y > maxH_cap) return -1.0;
            if (t > t_limit) return -1.0;
            if (any(greaterThan(abs(q.xz), vec2(kHeightmapSide * 0.5 - 2.0)))) break;
            float hq = q.y - meshHeight(q.xz);
            if (hq < 0.0025 * t) return t;
            t += max(hq * 0.8, 0.002 * t);
        }
        final_pos = ro + t * rd;
        final_h   = final_pos.y - terrainM_lod(final_pos.xz, t);
    }
    // Treat as sky if we ended up well above the surface (>5m) AND
    // we were rising. Below that → accept as a hit (the FBM-LOD just
    // didn't converge tightly).
    if (final_h > 5.0 && rd.y > 0.0) return -1.0;
    return t;
}

#ifndef QLIKE_WATER
// How far the landscape is marched. 1500 m on the ground and up to 1 km
// altitude (unchanged); from there it grows with height, so that flying up
// the real landscape stays under the camera out to tens of kilometres (the
// planet's shaded far field in compose.frag takes over beyond it).
float rm_far_limit(float eye_y) {
    // (and from 25 km up it draws in again: by 60 km the planet's painted far field, with
    // its weather, has taken over - compose.frag has the same formula for the cross-fade)
    return clamp(1500.0 + (eye_y - 300.0) * 100.0, 1500.0, 80000.0) * (1.0 - smoothstep(25000.0, 60000.0, eye_y));
}
// The march for that: only used above 1 km, for rays the normal march
// cannot reach. Looking down from the air there is nothing near the eye,
// so this is a plain walk through the slab that holds the landscape.
float raymarch_far(vec3 ro, vec3 rd, float t_end) {
    if (rd.y > -1e-4) return -1.0;
    float t = max((ro.y - terrainMaxHeight()) / (-rd.y), 0.0);
    float t_prev = t;
    for (int i = 0; i < 160; ++i) {
        if (t > t_end) return -1.0;
        vec3 p = ro + rd * t;
        float h = p.y - terrainM(p.xz);          // all octaves: the heights (lakes, snow line) must not shift with distance
        if (h < 0.0012 * t) {
            if (h < 0.0) {
                float ta = t_prev, tb = t;
                for (int k = 0; k < 6; ++k) {
                    float tm = 0.5 * (ta + tb);
                    vec3  pm = ro + tm * rd;
                    if (pm.y - terrainM(pm.xz) > 0.0) ta = tm; else tb = tm;
                }
                return ta;
            }
            return t;
        }
        t_prev = t;
        t += max(h * 0.55, 0.0025 * t);
    }
    return t;      // out of steps just above the ground: that is where it is (never a hole to the water plane)
}
#endif

// Finite-difference normal with distance-scaled epsilon. Eps grows
// linearly with distance (was quadratic — that smeared distant
// detail far more than required for anti-aliasing) so distant
// terrain keeps edge crispness while close-up still uses sub-cm
// epsilon. The earlier "micro-noise on lit normal" path was removed
// — its `noise2(p) - noise2(p + 0.5)` cell-boundary derivative
// produced visible 8–16 pixel square blocks on the surface past
// 80 m. The FBM normal-octave slider (default 18) gives plenty of
// distance detail without the artefacts.
// Scales the close-up micro-bump (1 = full). The road sets it to 0 so the
// cobble normal map is not layered over pebbly terrain noise.
float g_rm_detail_mul = 1.0;
