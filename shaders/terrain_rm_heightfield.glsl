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

// Set by calcNormal under a dense lawn (settings: simpler relief under grass): the main rills
// only, as while marching - the side cracks and the soil texture's grain are hidden by the blades.
bool g_rm_relief_lite = false;
// Set by the low-resolution build while it looks for its block's hit: the landscape without
// the close relief (a few FBM steps instead of the slow last stretch over rills and stones).
// The hit then lies up to ~0.3 m under the drawn ground - near enough to ask "is the sun
// blocked here", which is all it is used for.
bool g_rm_no_relief = false;
float rm_ground_detail_raw(vec2 wp, float t, float h);
// The close relief, less on a highway: none on the asphalt, growing over the verge (a hard stop
// at the asphalt's edge stood up as a wall of streaks).
float rm_ground_detail(vec2 wp, float t, float h) {
    float keep = 1.0;
    if (t < 26.0 && rm_hw_in(wp)) {
        keep = smoothstep(5.4, 8.4, abs(rm_hw_dh(wp).x));
        if (keep <= 0.0) return 0.0;
    }
    // No rills and stones under the service station's paving and its way in (terrain.h kStation*):
    // the paving lies a hand above the level ground, and the relief stood up through it.
    {
        float st = max(max(1585.0 - wp.x, wp.x - (abs(wp.y - 545.0) < 17.0 ? 1662.0 : 1643.6)), max(507.5 - wp.y, wp.y - 582.5));
        if (st < 1.0) { keep *= smoothstep(0.0, 1.0, st); if (keep <= 0.0) return 0.0; }
    }
    return rm_ground_detail_raw(wp, t, h) * keep;
}
float rm_ground_detail_raw(vec2 wp, float t, float h) {
    float w = 1.0 - smoothstep(10.0, 26.0, t);
    if (w <= 0.0 || RMDBG(1024) || g_rm_no_relief) return 0.0;
#ifndef VIBE_LITE
    if (g_rm_ray_y > -1e8 && RMDBG(128)) return 0.0;       // profiling: the march without the relief (before any of its work: this measures what the relief costs the march)
#endif
    // Not on the road: the paving is flat (crouching on it showed the
    // rills as noise in the cobbles). In winter the snow still lies on it.
    float road_keep = 1.0;
    {
        float keep = scene.road_info.y;
        float rdist = rm_road_dist_thr(wp, keep + 0.9);                 // paving only; the dirt verge keeps its relief
        road_keep = smoothstep(keep + 0.1, keep + 0.9, rdist);
        if (road_keep <= 0.0) {
            if (g_snow < 0.01) return sand_road_h(wp) * w;          // (a real branch, not ?: - the winter side has loops)
            return rm_snow_on(0.0, wp, h, 0.0) * w;
        }
    }
    // (only where there is road: this runs for every near step of the march, and the sand's
    // noise was being worked out - and thrown away - over all the open ground)
    float rsand = road_keep < 0.999 ? sand_road_h(wp) * (1.0 - road_keep) : 0.0;       // (the sand thins out over the verge)
    // (the same in every tile - scene_ubo.glsl tile_noise2 - so the rills stay where they are when
    // the engine re-centres; twins: grass_raymarch.frag grass_ground_relief, terrain.cpp raymarch_ground_relief)
    float wn = tile_noise2(wp, 0.21, vec2(0.0));       // one noise warps both axes
    vec2  qo = vec2(wn, 1.0 - wn) * 1.7;
    float g1 = 1.0 - smoothstep(0.0, 0.22, abs(tile_noise2(wp, 0.55, qo) * 2.0 - 1.0));              // main rills
    // While MARCHING (g_rm_ray_y set by raymarch()): the main rills only,
    // side cracks / soil displacement at their means - two noise taps
    // instead of three plus a texture fetch on every near-ground step. The
    // hit point lands within ~5 cm of the full surface; the normal and the
    // shading (called without the flag) still use the full relief.
    if (g_rm_ray_y > -1e8) return rm_snow_on((0.11 * (1.0 - g1) + 0.045 * 0.86 + 0.045 +
                                              (RMDBG(32768) ? 0.0 : rm_pebbles_march(wp, t))) * road_keep + rsand, wp, h, road_keep) * w;
    if (g_rm_relief_lite) return rm_snow_on((0.11 * (1.0 - g1) + 0.045 * 0.86 + 0.045) * road_keep + rsand, wp, h, road_keep) * w;
    float lod = log2(max(t * 2.0 * scene.viewport.w * 0.45 * 2048.0, 1.0));
    float disp = textureLod(u_normal[9], wp * TILE_FREQ(0.45), lod).a;
    float g2 = 1.0 - smoothstep(0.0, 0.25, abs(tile_noise2(wp, 1.485, qo * 2.7 + 3.1) * 2.0 - 1.0));  // side cracks
    float d = 0.11 * (1.0 - g1) + 0.045 * (1.0 - g2) + 0.09 * disp;
    float sand = 1.0 - smoothstep(scene.water_params.y + 0.4, scene.water_params.y + 2.5, h);
    d += sand * 0.03 * (0.5 + 0.5 * sin(wp.x * TILE_W(9.0) + wp.y * TILE_W(3.5) + 6.0 * wn));
    // (No stones here: this full variant is also what every terrain-shadow
    // and reflection step evaluates. The normal adds them itself, the march
    // has its own cut-down version above.)
    return rm_snow_on(d * road_keep + rsand, wp, h, road_keep) * w;
}

// Beyond the home tile: the ground round the camera as the engine bakes it while it moves
// (veg_stream.cpp wv_nf_poll; the grass stands on it): 512 x 512 m at 1 m, 9 octaves of the
// world formula, heights packed in RGB. The march uses it as it uses the home tile's bake: to
// stride through the air above the ground with one texture fetch a step instead of the FBM.
vec2 g_nf_org = vec2(0.0);
bool g_nf_ok  = false;
void nf_setup() {
    g_nf_ok = false;
    if (scene.globe_info.z < 0.5) return;
    vec3 m = texelFetch(u_normal[22], ivec2(2, 0), 0).rgb * 255.0;
    g_nf_ok = abs(m.r - 77.0) < 0.5 && abs(m.g - 123.0) < 0.5 && abs(m.b - 201.0) < 0.5;
    if (!g_nf_ok) return;
    vec3 a = texelFetch(u_normal[22], ivec2(0, 0), 0).rgb * 255.0;
    vec3 b = texelFetch(u_normal[22], ivec2(1, 0), 0).rgb * 255.0;
    g_nf_org = (vec2(a.r * 65536.0 + a.g * 256.0 + a.b, b.r * 65536.0 + b.g * 256.0 + b.b) - 8388608.0) * 64.0;
}
// Height of the bake at wp, or -1e9 where there is none (outside its window, or ground above
// what its packing holds).
float nf_height(vec2 wp) {
    vec2 l = wp - g_nf_org;
    if (!g_nf_ok || any(lessThan(l, vec2(3.0))) || any(greaterThan(l, vec2(508.0)))) return -1e9;
    vec3 c = textureLod(u_normal[22], (l + 0.5) / 512.0, 0.0).rgb;
    float h = c.r * 255.0 + c.g * (255.0 / 256.0) + c.b * (255.0 / 65536.0) - 64.0;
    return h > 188.0 ? -1e9 : h;
}

float terrainM(vec2 p) {

    vec2 wp = p;
    p = rm_flat_p0(p);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    int kOct = int(pc.tex_params.z);
    int kOctFlat = rm_flat_on() ? kOct : 0;              // (a chart beyond the flat formula's reach: the sphere's alone, rm_shape below)
    for (int i = 0; i < kOctFlat; i++) {
        vec3 n = rm_noised_i(p, i);
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
    a = rm_shape(a, wp, float(kOct));
    float h = rm_world_h(a, wp);
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
// Set while the far march (and the normal of its hits) reads the land: the detail then follows
// what a pixel resolves at that distance only - not the distance LOD of the view along the
// ground, which would move the heights with distance.
bool g_rm_fp_only = false;
float terrainM_lod(vec2 p, float ray_t) {
    vec2 wp = p;
    p = rm_flat_p0(p);
    int oct_full = int(pc.tex_params.z);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = g_rm_fp_only ? 0.0 : smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    // Fractional octave count — ceil() to include the partially-faded
    // top octave; `frac` is the [0..1] weight that octave gets so the
    // FBM is C0-continuous across the integer threshold.
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
#ifndef VIBE_WATER
    // Pixel-footprint cap: octave i has wavelength 333 m / 2^i; drop
    // octaves smaller than ~1.5 pixels at this distance (sub-pixel
    // detail the march would only alias on). 1.68 = 2 tan(40 deg).
    {
        float fp = max(ray_t, 1.0) * 1.68 * scene.viewport.w;
        float oct_fp = log2(333.333 / (1.5 * fp)) + 1.0;
        oct_f = min(oct_f, max(oct_fp, float(oct_min)));
    }
#endif /* ! VIBE_WATER */
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 24;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct || !rm_flat_on()) break;
        vec3 n = rm_noised_i(p, i);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    a = rm_shape(a, wp, oct_f);
    float h = rm_world_h(a, wp);
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
        if (!g_rm_no_relief) h += 0.30 * (1.0 - smoothstep(10.0, 26.0, ray_t)) * (1.0 - tpl);   // conservative bound
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
// The landscape without the close-range relief (terrainB) and with it (terrainH). The normal
// code reads the two apart: within a few centimetres the landscape is a plane, only the
// relief on it changes - so its slope is worked out once and reused (calcNormal, g_nb).
float terrainB_lod_pw(vec2 p, float ray_t, float tpl) {
    vec2 wp = p;
    p = rm_flat_p0(p);
    int oct_full = int(pc.tex_params.w);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    // Detail finer than the pixel (settings: 'normal detail limit'): those octaves only add
    // noise to the lighting at a distance. Same rule as the march above.
    if (scene.rm_perf.w > 0.0) {
        float fp = max(ray_t, 1.0) * 1.68 * scene.viewport.w;
        oct_f = min(oct_f, max(log2(333.333 / (scene.rm_perf.w * fp)) + 1.0, float(oct_min)));
    }
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    const int kMaxOct = 32;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct || !rm_flat_on()) break;
        vec3 n = rm_noised_i(p, i);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    a = rm_shape(a, wp, oct_f);
    float h = rm_world_h(a, wp);
    h = mix(h, pc.color.w, tpl);
    h += sampleHeightDelta(wp);   // (reverted 80 m gate — no near/far seam)
    return h;
}
// terrainB_lod_pw at wp and at wp + (e, 0) and wp + (0, e) - the three heights the normal is made
// from - with the landscape formula worked out ONCE. The formula's slope comes out of the same
// pass as its value: each octave's noise gives its gradient and second derivatives from the four
// lattice values it has already looked up, and the damping term 1 / (1 + |d|^2) is carried along
// with its own derivative (the term the earlier analytic attempts left out - without it the
// ridges and the rocks in snow went soft). What follows the formula (the world's shape, the
// road's shelf, sculpting) is still evaluated at the three points, with the formula's value
// stepped along its slope. Was three passes of up to 20 octaves a pixel: the largest single
// item of the shading (docs/ARCHITECTURE.md, where the frame goes).
float terrainB_slope(vec2 wp, float ray_t, float tpl, float e, out float bR, out float bU) {
    vec2 p = rm_flat_p0(wp);
    int oct_full = int(pc.tex_params.w);
    int oct_min  = max(2, min(oct_full, int(pc.emissive.w)));
    float lod_f  = smoothstep(pc.emissive.y, pc.emissive.z, ray_t);
    float oct_f  = float(oct_full) - lod_f * float(oct_full - oct_min);
    if (scene.rm_perf.w > 0.0) {
        float fp = max(ray_t, 1.0) * 1.68 * scene.viewport.w;
        oct_f = min(oct_f, max(log2(333.333 / (scene.rm_perf.w * fp)) + 1.0, float(oct_min)));
    }
    int   oct    = int(ceil(oct_f));
    float frac   = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    // J: how the octave's position moves with the world position; dD: how d does; ga: how a does.
    mat2 J = rm_far_chart() ? mat2(scene.ring_j.x, scene.ring_j.z, scene.ring_j.y, scene.ring_j.w) * 0.003 : mat2(0.003);
    mat2 dD = mat2(0.0);
    vec2 ga = vec2(0.0);
    const int kMaxOct = 32;
    for (int i = 0; i < kMaxOct; i++) {
        if (i >= oct || !rm_flat_on()) break;
        vec3 hs;
        vec3 n = rm_noised_h(p, i, hs);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        dD += (mat2(hs.x, hs.z, hs.z, hs.y) * J) * w;
        float q = 1.0 / (1.0 + dot(d, d));
        a += b * n.x * w * q;
        ga += (b * w * q) * ((n.yz * J) - (2.0 * n.x * q) * (d * dD));
        b *= 0.5;
        p = m2 * p * 2.0;
        J = (m2 * J) * 2.0;
    }
    vec2 wr = wp + vec2(e, 0.0), wu = wp + vec2(0.0, e);
    bR = mix(rm_world_h(rm_shape(a + ga.x * e, wr, oct_f), wr), pc.color.w, tpl) + sampleHeightDelta(wr);
    bU = mix(rm_world_h(rm_shape(a + ga.y * e, wu, oct_f), wu), pc.color.w, tpl) + sampleHeightDelta(wu);
    return mix(rm_world_h(rm_shape(a, wp, oct_f), wp), pc.color.w, tpl) + sampleHeightDelta(wp);
}
float terrainH_lod_pw(vec2 p, float ray_t, float tpl) {
    float h = terrainB_lod_pw(p, ray_t, tpl);
    return h + rm_ground_detail(p, ray_t, h) * (1.0 - tpl);
}
// The slope of the landscape under the relief at the pixel being shaded, left by calcNormal:
// (-dh/dx, -dh/dz), its height, and whether it is valid for this pixel.
vec2  g_nb = vec2(0.0);
float g_nb_h = 0.0;
bool  g_nb_ok = false;
// ... and the normal of the surface itself there (landscape + relief, sand, snow), before
// stones, grit and the paving's own relief are added to it: what the sand and the snow are
// shaded with. It is the same three taps the pixel's normal is made from - they used to be
// worked out a second time, four more taps of the whole landscape, for every sand or snow pixel.
vec3  g_nrm_plain = vec3(0.0, 1.0, 0.0);
// The surface's slope terms (h(x-e) - h(x+e), h(z-e) - h(z+e)) at spacing `e`, from the base
// slope above plus four taps of the relief alone - what four taps of the whole landscape
// gave (the sand's and the snow's own normals), without working the landscape out four times.
vec2 rm_detail_slope(vec2 wp, float t, float e, float tpl) {
    float k = 1.0 - tpl;
    float dx0 = rm_ground_detail(wp - vec2(e, 0.0), t, g_nb_h), dx1 = rm_ground_detail(wp + vec2(e, 0.0), t, g_nb_h);
    float dz0 = rm_ground_detail(wp - vec2(0.0, e), t, g_nb_h), dz1 = rm_ground_detail(wp + vec2(0.0, e), t, g_nb_h);
    return vec2(dx0 - dx1, dz0 - dz1) * k + g_nb * (2.0 * e);
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
    p = rm_flat_p0(p);
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
        if (i >= oct || !rm_flat_on()) break;
        vec3 n = rm_noised_i(p, i);
        float w = (i == oct - 1) ? frac : 1.0;
        d += n.yz * w;
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    a = rm_shape(a, wp, oct_f);
    float h = rm_world_h(a, wp);
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
// Steps the last raymarch() took (profiling view, VIBE_RM_SKIP bit 64).
int g_rm_steps = 0;
// How many times the settings' step count a march may take: see raymarch().
// (up to eight times: the factor goes down to 0.05 since 2026-11-24 - at the three times that
// served a factor of 0.15, a ray at 0.05 would give up a third as far out)
float rm_step_budget(float kStep) { return clamp(0.55 / max(kStep, 0.05), 1.0, 8.0); }
// `t_start`: a distance known to be clear of the ground (from the low-resolution march,
// terrain_raymarch.frag main), 0 = none.
float raymarch(vec3 ro, vec3 rd, float t_max, float t_start) {
    g_rm_steps = 0;
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
    float kStep = abs(pc.emissive.x);   // sign = full-resolution pass flag (see main)
    // The step budget goes with the step factor. A step is `kStep` x the clearance, so with a
    // small factor (asked for to keep steep rock faces whole) a ray that skims the ground gets
    // a fraction as far on the same number of steps: at 0.15 and 192 steps the rays near the
    // horizontal gave up a few hundred metres out, and a strip of sky lay between the near
    // ground and the far land - only from the ground: from 100 m up the clearance, and so
    // every step, is large. The setting's count is for a factor of 0.55; a smaller factor
    // gets proportionally more (rm_step_budget). Only rays that need them walk them.
    int kSteps = int(pc.tex_params.x * rm_step_budget(kStep));
    bool relax = pc.grass_params.y > 0.5;
    // Tiny initial offset; max half-cell so the surface hit shifts
    // sub-pixel only.
    t += jitter * 0.5;
    float maxH_cap = terrainMaxHeight();
#ifndef VIBE_WATER
    // Eye at or under the marched surface. The full FBM sits up to ~0.5 m
    // above the baked heightmap the player stands on (plus the close-range
    // ground detail), so a crouching / crawling eye can be INSIDE it: every
    // ray then "hit" at once and the ground in front turned to noise. March
    // the baked heightmap for this pixel instead - the eye is always above that.
    // (First against the baked height, one fetch: the full surface is never more than about a
    // metre above it, and an eye standing is. The full evaluation - the same answer for
    // every pixel of the frame, at the price of a march step each - is only for an eye that low.)
    if (all(lessThan(abs(ro.xz), vec2(kHeightmapSide * 0.5 - 2.0))) &&
        ro.y - meshHeight(ro.xz) < 1.5 && ro.y - terrainM_lod(ro.xz, 0.0) < 0.12) {
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
    // Start where the low-resolution march got to (a little before it, by this pixel's own
    // jitter, so the pixels of one of its texels do not share an error pattern).
    float t_first = t;
    if (t_start > 0.0) t = max(t, t_start - jitter * 0.5);
    t_prev = t;
    // Fast final approach (settings): see the step below.
    float conv = scene.rm_perf.z;
    float h_last = 0.0;
    bool  real_last = false;
    for (int i = 0; i < kSteps; i++) {

        vec3 pos = ro + t * rd;
        g_rm_steps = i + 1;
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
#ifndef VIBE_WATER
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
            // (beyond the home tile there is no bake: every step reads the real surface. This
            // stood at "far above the ground", which made the first step past the tile's edge
            // leap out of range - seen from inside the tile, the land beyond it was missing
            // until the far march picked it up 1.4 km out.)
            float hm = -1.0;
            if (all(lessThan(abs(pos.xz), vec2(kHeightmapSide * 0.5 - 2.0))))
                hm = pos.y - meshHeight(pos.xz) - kHmMargin;
            else {
                float nh = nf_height(pos.xz);
                if (nh > -1e8) hm = pos.y - nh - kHmMargin;
            }
            coarse = hm > 0.75;
            if (!coarse) g_rm_ray_y = pos.y;
            h = coarse ? hm : pos.y - terrainM_lod(pos.xz, t);
            g_rm_ray_y = -1e9;
            // The low-resolution march is a bound, not a promise: should this pixel's ray
            // already be in the ground at its start, march it from the beginning.
            if (i == 0 && t_start > 0.0 && !coarse && h < 0.0) {
                t = t_first; t_prev = t; t_start = 0.0;
                continue;
            }
            // On the road the marched surface IS the baked (levelled) one.
            // The FBM's extra octaves stand up to ~0.5 m proud of it; seen
            // from a low eye along the road they formed a lumpy, spiky
            // hump across the paving a few metres ahead.
            if (!coarse && h < 0.65 && t < 90.0 && rd.y > -0.16 && !rm_far_chart()) {   // (home only: a site's road lies on formula ground, there is no baked surface) grazing rays only: steep ones are fixed by the slide in main()
                float keep = scene.road_info.y;
                float rdist = rm_road_dist_thr(pos.xz, keep + 0.9);
                if (rdist < keep + 0.9)
                    h = mix(pos.y - meshHeight(pos.xz), h, smoothstep(keep, keep + 0.9, rdist));
            }
        }
#else /* VIBE_WATER */
        g_rm_ray_y = pos.y;
        float h = pos.y - terrainM_lod(pos.xz, t);
        g_rm_ray_y = -1e9;
#endif /* VIBE_WATER */
        // P7: loosen the hit threshold with distance. At grazing angle a
        // typical pixel covers 0.3-1 m of world space past 200 m, so
        // demanding sub-millimetre convergence wastes ~30 % of the
        // march steps on far rays. mix 0.0015 → 0.005 over [150, 600] m.
        float hit_eps = mix(0.0015, 0.0025, smoothstep(150.0, 600.0, t)) * t;
#ifndef VIBE_WATER
        if (!coarse && abs(h) < hit_eps) return t;   // only real-surface steps may hit (was: break + a full re-evaluation below)
#else /* VIBE_WATER */
        if (abs(h) < hit_eps) break;
#endif /* VIBE_WATER */
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
#ifndef VIBE_WATER
        // Fast final approach. Close over the ground a step of `kStep * h` shrinks with the
        // clearance: a ray skimming the ground took dozens of ever smaller steps - most of
        // this pass. Two steps on the real surface tell how fast the clearance is being lost;
        // aim (a fraction `conv` of the way) at where it runs out. Limits: never more than a
        // dozen clearances (the ground is not a plane), a few clearances only down among the
        // stones, and an absolute cap that grows with distance. Overshooting into the ground
        // is caught by the bisection below.
        if (conv > 0.0 && !coarse && real_last && h > 0.0) {
            float s = (h_last - h) / max(t - t_prev, 1e-4);
            if (s > 1e-3) {
                float k_rel = (t < 26.0) ? mix(3.0, 12.0, smoothstep(0.04, 0.30, h)) : 12.0;
                advance = max(advance, min(min(conv * h / s, k_rel * h), 0.35 + 0.03 * t));
            }
        }
        real_last = !coarse; h_last = h;
#endif
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
#ifndef VIBE_WATER
            if (conv > 0.0 && real_last && h_last > 0.0) {
                // With the fast approach nearly every ray ends here, so the search has to
                // be cheap: the clearance is known at both ends (above at ta, below at tb)
                // and is close to a straight line between them - three steps of "where does
                // that line cross zero" land within millimetres (seven halvings did the
                // same). Read as the march reads it (the marching relief), so the hit lies
                // on the surface the march was walking over.
                float ha = h_last, hb = h;
                for (int k = 0; k < 3; ++k) {
                    float tm = ta + (tb - ta) * clamp(ha / (ha - hb), 0.05, 0.95);
                    vec3  pm = ro + tm * rd;
                    g_rm_ray_y = pm.y;
                    float hm2 = pm.y - terrainM_lod(pm.xz, tm);
                    g_rm_ray_y = -1e9;
                    // (above the ground and as close as the march itself calls a hit: there.
                    //  The other evaluations of the three would only move it by less than that)
                    if (hm2 > 0.0 && hm2 < hit_eps && !RMDBG(8192)) return tm;
                    if (hm2 > 0.0) { ta = tm; ha = hm2; } else { tb = tm; hb = hm2; }
                }
                return ta + (tb - ta) * clamp(ha / (ha - hb), 0.0, 1.0) * 0.98;      // (just short of the crossing: never inside the ground)
            }
#endif
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

#ifdef VIBE_MARCH_ONLY
// The low-resolution march (terrain_march_lo.frag.spv; frame.cpp "march pre-pass"). One ray
// for a block of pixels: it walks until the ground comes within the width of that block
// (`cone` = its radius per metre of distance), over an upper bound of the surface (no close
// relief to work out), and returns the last point that was still clear by that margin. The
// full-resolution pass starts its own march there - the walk through the open air and down
// to the ground is done once for the block instead of once for every pixel.
// Returns 0 = nothing known (march from the eye), 1e9 = no ground within the march's range.
// `t_grass`: the last point at which the ray was still more than a blade's height above the
// ground BEFORE it first came lower (0 = not known, 1e9 = it never comes that low): where the
// grass pass starts its own march (grass_raymarch.frag), instead of walking through the open
// air again. (Before it FIRST came lower: a ray that skims a crest and then has the ground
// fall away under it is high again beyond the crest. Counted on from there, the grass march
// started behind the blades on the crest - the far edge of a meadow came out in steps the
// size of this march's blocks.)
float raymarch_lo(vec3 ro, vec3 rd, float t_max, float cone, out float t_grass) {
    t_grass = 0.0;
    float maxH = terrainMaxHeight();
    float t = 0.0;
    if (ro.y > maxH && rd.y >= 0.0)  { t_grass = 1e9; return 1e9; }
    if (ro.y > maxH) t = (ro.y - maxH) / (-rd.y);
    float t_limit = (t_max > 0.0) ? min(t_max, 1500.0) : 1500.0;
    if (t > t_limit)  { t_grass = 1e9; return 1e9; }
    // (the eye in the ground: the full pass has its own march for that)
    if (all(lessThan(abs(ro.xz), vec2(kHeightmapSide * 0.5 - 2.0))) && ro.y - meshHeight(ro.xz) < 1.5 && ro.y - terrainM_lod(ro.xz, 0.0) < 0.12) return 0.0;
    float kStep  = abs(pc.emissive.x);
    int   kSteps = int(pc.tex_params.x * rm_step_budget(kStep));      // (as the full march: see there)
    t_grass = t;
    float t_safe = t;
    // (the same fast approach as the full march, aimed at where the clearance reaches the
    // margin - without it a ray skimming the ground crept toward it in dozens of steps)
    float conv = scene.rm_perf.z;
    float h_last = 0.0, t_last = t;
    bool  real_last = false;
    bool  been_low = false;
    for (int i = 0; i < kSteps; i++) {
        vec3 pos = ro + t * rd;
        if (rd.y > 0.0 && pos.y > maxH) return 1e9;
        float hm = -1.0;
        if (all(lessThan(abs(pos.xz), vec2(kHeightmapSide * 0.5 - 2.0))))
            hm = pos.y - meshHeight(pos.xz) - 2.0;
        else {
            float nh = nf_height(pos.xz);
            if (nh > -1e8) hm = pos.y - nh - 2.0;
        }
        float h = hm;
        if (hm <= 0.75) {
            g_rm_ray_y = 1e8;                         // far above: the relief's upper bound, not the relief
            h = pos.y - terrainM_lod(pos.xz, t);
            g_rm_ray_y = -1e9;
        }
        float thr = 3.0 * cone * t + 0.06;
        if (h >= thr + 1.25) { if (!been_low) t_grass = t; } else been_low = true;
        if (h < thr) return t_safe;
        if (t > t_limit) return 1e9;
        t_safe = t;
        // The stride lengthens with distance, as the full march's does (its "relaxation": a far
        // pixel covers metres of ground, so the same care as close by is wasted there) - but
        // never beyond 0.55 x the clearance (what a factor of 0.55 always took; the full march
        // goes on to several times the clearance): a block's ray that jumps a ridge leaves a
        // block-sized hole in it. A factor of 0.55 or more marches exactly as it did. Without this the
        // march had no way to reach the far land over flat ground on a small step factor
        // (0.15 x 3 m of clearance = 45 cm a step), and the full march, started where this one
        // stopped, ran out of steps in its turn.
        float k_far = kStep * mix(1.0, 2.5, smoothstep(200.0, 600.0, t));
        float adv = max(k_far, min(k_far * max(t * 0.02, 1.0) * 0.7, 0.55)) * h;
        bool real = hm <= 0.75;
        if (conv > 0.0 && real && real_last) {
            float s = (h_last - h) / max(t - t_last, 1e-4);
            if (s > 1e-3) adv = max(adv, min(min(0.7 * conv * (h - thr) / s, 10.0 * h), 0.5 + 0.04 * t));
        }
        real_last = real; h_last = h; t_last = t;
        t += adv;
    }
    return t_safe;
}
#endif

#ifndef VIBE_WATER
// How far the landscape is marched. 1500 m on the ground and up to 1 km
// altitude (unchanged); from there it grows with height, so that flying up
// the real landscape stays under the camera out to tens of kilometres (the
// planet's shaded far field in compose.frag takes over beyond it).
float rm_far_limit(float eye_y) {
    // (and from 25 km up it draws in again: by 60 km the planet's painted far field, with
    // its weather, has taken over - compose.frag has the same formula for the cross-fade)
    return clamp(1500.0 + max(eye_y, 0.0) * 100.0, 1500.0, 80000.0) * (1.0 - smoothstep(25000.0, 60000.0, eye_y));
}
// The march for that: only used above 1 km, for rays the normal march
// cannot reach. Looking down from the air there is nothing near the eye,
// so this is a plain walk through the slab that holds the landscape.
// `whole`: this march finds the land from the eye on (from the air, and anywhere outside the
// home tile); otherwise it continues where the normal march stopped.
float raymarch_far_impl(vec3 ro, vec3 rd, float t_end, bool whole) {
    // (from the ground too: the land does not end 1.5 km away, it only is not marched in detail
    // beyond that. From below the highest ground a ray may also climb to it.)
    float maxH = terrainMaxHeight();
    if (rd.y > -1e-4 && (ro.y > maxH || rd.y > 0.35)) return -1.0;
    float t = (ro.y > maxH) ? (ro.y - maxH) / (-rd.y) : (whole ? 0.05 : 1400.0);      // (the normal march has done the near part)
    float t_prev = t;
    // From the ground a level ray runs a long way close over the land: more steps, and strides
    // that lengthen with distance (the bisection below catches what a stride steps into).
    int n_steps = (whole && ro.y <= maxH) ? 300 : 160;
    for (int i = 0; i < n_steps; ++i) {
        // The end of the range (the water plane, an object, the far limit): look AT it before
        // giving up. A stride may pass the ground and the end in one go - low land just above
        // the water, seen from steeply above, was stepped over and the water showed through it
        // as a round hole under the camera that grew and shrank with the camera's height.
        bool last = false;
        if (t > t_end) {
            if (t_prev >= t_end) return -1.0;
            t = t_end; last = true;
        }
        vec3 p = ro + rd * t;
        if (rd.y > 0.0 && p.y > maxH) return -1.0;
        float h = p.y - terrainM_lod(p.xz, t);
        if (last && h >= 0.0) return -1.0;   // as much detail as a pixel resolves there: with all of it, the part finer than a pixel showed as ruled lines
        if (h < 0.0012 * t) {
            if (h < 0.0) {
                // (to the millimetre: the stride that stepped in can be metres long, and six halvings
                // of that left centimetres - enough for the ground's fine grain to show each ray's
                // error as a streak along the line of sight)
                float ta = t_prev, tb = t;
                for (int k = 0; k < 14; ++k) {
                    float tm = 0.5 * (ta + tb);
                    vec3  pm = ro + tm * rd;
                    if (pm.y - terrainM_lod(pm.xz, tm) > 0.0) ta = tm; else tb = tm;
                }
                return 0.5 * (ta + tb);
            }
            // Accepted a little above the ground (within the tolerance): walk the rest of the way
            // down. Left as it was, each ray stopped at its own height above the surface and the
            // ground's pattern was smeared into straight streaks along the line of sight.
            for (int k = 0; k < 4; ++k) {
                t += h * 0.8 / max(-rd.y, 0.3);            // (h is height; along a slanting ray it is further)
                vec3 pr = ro + rd * t;
                h = pr.y - terrainM_lod(pr.xz, t);
            }
            return t;
        }
        t_prev = t;
        t += max(h * 0.55 * (1.0 + min(t * 0.006, 2.5)), 0.0025 * t);
    }
    // Out of steps. Just above the ground: that is where it is (never a hole to the water
    // plane). Still well clear of it (a level ray from the ground): open air.
    {
        vec3 p = ro + rd * t;
        if (rd.y > -0.02 && p.y - terrainM_lod(p.xz, t) > 0.02 * t) return -1.0;
    }
    return t;
}
float raymarch_far(vec3 ro, vec3 rd, float t_end, bool whole) {
    g_rm_fp_only = true;
    float t = raymarch_far_impl(ro, rd, t_end, whole);
    g_rm_fp_only = false;
    return t;
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
