// terrain_rm_noise_road.glsl - part of terrain_raymarch.frag (included there, in place; not a standalone shader).
// Noise, plateau weight, road distance

// === Hash + Value-Noise with analytical derivatives ===
// dot/fract pattern (avoids sin() precision artifacts on some GPUs).
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// Value Noise + ∂n/∂x, ∂n/∂y. Hermite smoothstep `3t²-2t³` for C¹
// continuity; `du = 6t(1-t)` is the chain-rule derivative used by the
// FBM-erosion accumulator. (Reverted from a quintic experiment —
// kept the proven cubic that matches the look the project shipped
// with and stays consistent with the CPU rm_noised bake.)
vec3 noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = hash21(i + vec2(0.0, 0.0));
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    float value = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    vec2  deriv = du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    return vec3(value, deriv);
}

float noise2(vec2 p) { return noised(p).x; }

// Per-octave ~37° rotation breaks axis-aligned banding. Unit
// determinant = pure rotation, no scaling.
const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);

const float TERRAIN_SCALE  = 0.003;
const float TERRAIN_HEIGHT = 120.0;

// The shape of the wider world. Home (the 2 km landscape and the ring of
// white mountains around it, out to 1.5 km from the centre) is the height
// formula as it is. Beyond that the land calms down to low hills - too low
// for the snow line - with a high range only here and there. `a`: the
// formula's value 0..1; returns metres. (compose.frag pl_terrain has the
// same function for the painted far field: change both.)
#ifndef RM_WATER_LEVEL
#define RM_WATER_LEVEL scene.water_params.y
#endif
float rm_world_h(float a, vec2 wp) {
    float h = a * TERRAIN_HEIGHT;
    float w = smoothstep(1500.0, 4000.0, length(wp));
    if (w <= 0.0) return h;
    float n = noise2(wp * 0.00008 + 7.3) * 0.65 + noise2(wp * 0.00021 + 3.1) * 0.35;
    float amp = mix(0.42, 1.12, smoothstep(0.60, 0.78, n));      // low hills; now and then a high range
    h = mix(h, 30.0 + (h - 30.0) * amp, w);                      // (scaled about 30 m: the lowlands stay above the water)
    // Oceans: a layer above the generator that overrules it. Where the continent map says
    // "sea" the ground goes down to a sea bed below the water level, whatever the hills
    // formula wanted there - with a shelf in between, so coasts have beaches and shallows.
    // Never within 5 km of home. (This is the place for more such areas: plains, deserts ...)
    float c = noise2(wp * 0.000035 + 11.7) * 0.6 + noise2(wp * 0.00009 + 4.2) * 0.4;
    float sea = smoothstep(0.50, 0.41, c) * smoothstep(5000.0, 12000.0, length(wp));
    return mix(h, RM_WATER_LEVEL - 24.0 - 14.0 * a, sea);
}


// High-frequency FBM-eroded noise used by sampleHeightDelta to paint
// procedural detail onto user-brushed cells. Same n.x / (1 + |∇d|²)
// erosion recipe as terrainM, but with a much higher base scale (0.05
// vs 0.003) so meter-scale ridges actually appear within the brush
// footprint. Centred ≈ 0 so it adds equal positive/negative variation
// — doesn't shift the mean brushed height. Forward-declared above
// sampleHeightDelta; the body lives here so it can use the noise
// helpers + m2 declared above.
float terrainDetailNoise(vec2 worldXZ) {
    vec2 p = worldXZ * 0.05;   // 20 m base; ≤ 30 cm at 6 octaves
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    for (int i = 0; i < 6; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    return a - 0.5;
}

// Plateau blend factor — 1 inside the gameplay plateau region, 0
// outside, smooth ramp across `kPlateauBlend` metres so the FBM
// surroundings ease into the flat castle pad. Mirrors what
// generate_heightmap() does in the rasterised path so the visible
// terrain shape matches what physics / castle assume.
const float kPlateauBlend = 20.0;  // smooth ramp from castle-flat to FBM mountains
float plateauWeight(vec2 worldXZ) {
    vec2 c = pc.color.xy;
    float ext = pc.color.z;
    vec2 d = abs(worldXZ - c) - vec2(ext);
    float dout = max(max(d.x, 0.0), max(d.y, 0.0));
    return 1.0 - smoothstep(0.0, kPlateauBlend, dout);
}

// Plateau ground-detail noise. The macro terrain FBM uses scales
// 0.003 → 0.096 (333 m → 10 m wavelengths) — its SMALLEST feature is
// ~10 m, so scaling it down for plateau use gives invisibly-small
// "detail" (a few cm). What the player perceives as "rocky ground"
// on the surrounding mountains is those 10 m ridges seen from far
// away; on the flat plateau they're not there at all.
//
// This separate noise fires AT FINER SCALES (2.5 m → 1.25 m
// wavelengths — safely above the ~0.8 m march-step alias floor)
// and adds visible "ground texture" to plateau cells without
// affecting anything outside. 2 octaves of plain noise (no
// derivative erosion damper — we WANT the high freqs to contribute
// fully). m2 (per-octave rotation declared further down) can't be
// forward-referenced — inlined.
float plateauGroundDetail(vec2 wp) {
    // 50/50 mix of plain noise (rolling) + soft ridge (1 - |2n-1|)
    // — gives natural rock character without the glass-flat smoothness
    // of pure FBM or the spike-needle look of multifractal-weighted
    // ridges. No squaring, no weight gating. 2 octaves at safe
    // wavelengths (2.5 m + 1.25 m).
    float a = 0.0, b = 1.0;
    vec2 p = wp * 0.4;
    for (int i = 0; i < 2; ++i) {
        float n = noised(p).x;
        float ridge  = 1.0 - abs(2.0 * n - 1.0);     // ∈ [0, 1]
        float smooth_n = n;                            // ∈ [0, 1]
        a += b * (mix(smooth_n, ridge, 0.5) - 0.5);  // re-centre on 0
        b *= 0.5;
        vec2 q = vec2( 0.8 * p.x - 0.6 * p.y,
                       0.6 * p.x + 0.8 * p.y);
        p = q * 2.0;
    }
    return a;
}

// Medium-detail FBM — used by the ray march. Derivative-erosion term
// `1/(1+dot(d,d))` suppresses high-frequency layers on steep slopes,
// producing realistic ridge/valley structure instead of fractal noise.
// Octave count is the per-frame quality knob (pc.tex_params.z, 4..9):
// last 3 of 9 contribute ~1.5 % of total amplitude so dropping them
// barely changes the shape but cuts ~30 % of the inner work.
// Close-range GEOMETRIC ground detail (real displacement of the marched
// surface, not a normal trick): erosion rills carved between raised
// ground, a finer second set, the scanned soil relief, lumps, and wind
// ripples on sand. Always >= 0 (the rendered surface must never drop
// below the baked heightmap that shadows / objects use) and faded out by
// 34 m, where it is sub-pixel anyway.
float rm_road_dist(vec2 q) {         // distance to the road centreline (1e9 outside its bounds)
    int rn = int(scene.road_info.x + 0.5);
    if (rn < 2 || any(lessThan(q, scene.road_bbox.xy)) || any(greaterThan(q, scene.road_bbox.zw))) return 1e9;
    float best = 1e9;
    for (int i = 0; i + 1 < rn && i < 63; ++i) {
        vec2 a = scene.road_pts[i].xz, ab = scene.road_pts[i + 1].xz - a;
        vec2 dv = q - (a + ab * clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0));
        best = min(best, dot(dv, dv));
    }
    return sqrt(best);
}
// Per-pixel cache: road distance is 1-Lipschitz, so d(q) >= d(q0) - |q - q0|.
// Callers only need the exact value below `thr`:
//  * far from the road: anything provably >= thr skips the loop;
//  * near / on the road (the eye standing on it: every near-ground march
//    step used to run all 63 segments): only the window of segments around
//    the last nearest one is tested. That is exact as long as it beats
//    the Lipschitz lower bound of every segment outside the window
//    (g_rd_out - |q - q0|); otherwise the full loop runs and re-anchors.
vec2  g_rd_q = vec2(1e9);
float g_rd_v = -1e9;
int   g_rd_i = -1;          // nearest segment at g_rd_q
float g_rd_out = 0.0;       // min distance at g_rd_q to segments outside the window
float rm_road_seg_d2(vec2 q, int i) {
    vec2 a = scene.road_pts[i].xz, ab = scene.road_pts[i + 1].xz - a;
    vec2 dv = q - (a + ab * clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0));
    return dot(dv, dv);
}
float rm_road_dist_thr(vec2 q, float thr) {
    float moved = length(q - g_rd_q);
    float lb = g_rd_v - moved;
    if (lb >= thr) return lb;
    int rn = int(scene.road_info.x + 0.5);
    const int kWin = 2;
    if (g_rd_i >= 0) {
        float d2 = 1e18;
        for (int i = max(g_rd_i - kWin, 0); i <= min(g_rd_i + kWin, rn - 2); ++i) d2 = min(d2, rm_road_seg_d2(q, i));
        float dl = sqrt(d2);
        if (dl <= g_rd_out - moved) return dl;        // provably the true nearest
    }
    g_rd_i = -1;
    if (rn < 2 || any(lessThan(q, scene.road_bbox.xy)) || any(greaterThan(q, scene.road_bbox.zw))) {
        // Outside the bbox: cache the distance to the box, a valid lower
        // bound for re-entry.
        vec2 ob = max(max(scene.road_bbox.xy - q, q - scene.road_bbox.zw), vec2(0.0));
        g_rd_q = q; g_rd_v = length(ob);
        return 1e9;
    }
    float best = 1e18; int bi = 0;
    for (int i = 0; i + 1 < rn && i < 63; ++i) {
        float d2 = rm_road_seg_d2(q, i);
        if (d2 < best) { best = d2; bi = i; }
    }
    float outd = 1e18;
    for (int i = 0; i + 1 < rn && i < 63; ++i)
        if (abs(i - bi) > kWin) outd = min(outd, rm_road_seg_d2(q, i));
    g_rd_q = q; g_rd_v = sqrt(best); g_rd_i = bi; g_rd_out = sqrt(outd);
    return g_rd_v;
}
float g_rm_ray_y = -1e9;   // see terrainM_lod


// Winter: where snow lies on the paving (0..1; irregular patches).
float snow_road_patch(vec2 wp) {
    float pn = noise2(wp * 0.8 + 4.0) * 0.55 + noise2(wp * 2.7 + 11.0) * 0.30 + noise2(wp * 9.0) * 0.15;
    return smoothstep(0.40, 0.56, pn);
}
