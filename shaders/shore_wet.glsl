// shore_wet.glsl - materials at the water, after the reference game's deferred wetness pass (its
// formulas; the code is ours; vk_engine/water_wet.cpp fills the switches, scene.water_wet):
// the wet band above a waterline, wet albedo and roughness, growth under water, the bed's tint by
// depth. ONE set of functions for the ground (terrain_raymarch.frag), the meshes (cube.frag) and
// the scattered stones (scatter.frag). Needs scene_ubo.glsl (tile_noise2). With SW_TERRAIN
// defined (the terrain shader, after water_solver.glsl) also the ground's own waterline:
// sw_terrain_band (the solver's surface, the stream, the sea's swash).

const vec3 kSwLum = vec3(0.2126, 0.7152, 0.0722);
float g_sw_mat_h = 0.3;      // (the terrain: its material's height 0..1 at the pixel, for the puddles in the hollows)

// The band: 1 at and under the waterline wl, falling linearly to 0 at wl + W; its top is ragged
// by a 100 m noise (where it is near 1 the full band shows, where it is small the band shrinks to
// nothing; the edge itself is sharp): w' = saturate((sqrt(w) - 1 + n) 1.25 / n).
float sw_band(float y, float wl, float W, vec2 xz) {
    float w = 1.0 - clamp((y - wl) / max(W, 1e-3), 0.0, 1.0);
    if (w <= 0.0) return 0.0;
    if (w >= 1.0) return 1.0;
    float n = tile_noise2(xz, 0.01, vec2(17.0, 5.0));
    return clamp((sqrt(w) - 1.0 + n) * 1.25 / max(0.001, n), 0.0, 1.0);
}
// The wet value from a band / puddle / dampness: m = saturate(max(...) - 0.1) / 0.9.
float sw_wet_m(float v) { return clamp(v - 0.1, 0.0, 1.0) / 0.9; }
// Wet albedo: a -> lerp(a, a saturate(0.5 - 0.5 luma), m) (soil of luma 0.25 goes to 37 %).
vec3 sw_wet_albedo(vec3 a, float m) {
    return mix(a, a * clamp(0.5 - 0.5 * dot(a, kSwLum), 0.0, 1.0), m);
}
// Wet roughness: r -> lerp(r, (1 - v) r, m) - a mirror at the waterline and in a puddle.
float sw_wet_rough(float r, float v, float m) { return mix(r, (1.0 - clamp(v, 0.0, 1.0)) * r, m); }

// Growth under water ("weed"): on up-facing submerged ground, deeper than 0.35 m below the line,
// in the material's hollows (h: its height 0..1), clumpy and nearly binary.
//   M = saturate(rate (10 - 20 h) depthK n1 n2 N.y^3), kept where M >= 0.75 (0.75..1 -> 0..1 here).
float sw_growth(vec2 xz, float h, float ny, float below) {
    if (!SW_ON(8) || below < 0.35 || ny < 0.2) return 0.0;
    float dk = clamp((below - 0.35) / 0.5, 0.0, 1.0);
    float n1 = max(tile_noise2(xz, 0.13 * 1.6, vec2(3.0)), tile_noise2(xz, 0.13 * 1.6, vec2(41.0, 9.0)));
    float n2 = max(tile_noise2(xz, 0.37 * 1.6, vec2(7.0)), tile_noise2(xz, 0.37 * 1.6, vec2(13.0, 29.0)));
    float M = clamp(scene.water_wet.y * (10.0 - 20.0 * clamp(h, 0.0, 1.0)) * dk * n1 * n2 * ny * ny * ny, 0.0, 1.0);
    return smoothstep(0.75, 0.85, M);
}
// The growth's material at xz: colour, and its own height 0..1 (lumps of weed), in code - two
// scales against tiling, as the reference's 8 m and 3.15 m reads of its texture set.
vec3 sw_growth_col(vec2 xz, out float gh) {
    float a = tile_noise2(xz, 6.0, vec2(5.0)), b = tile_noise2(xz, 17.0, vec2(2.0)), c = tile_noise2(xz, 0.9, vec2(8.0));
    gh = clamp(0.55 * a + 0.45 * b, 0.0, 1.0);
    vec3 dark = vec3(0.030, 0.042, 0.016), olive = vec3(0.070, 0.078, 0.028), rust = vec3(0.075, 0.058, 0.026);
    return mix(mix(dark, olive, smoothstep(0.25, 0.8, gh)), rust, 0.45 * smoothstep(0.55, 0.85, c)) * mix(0.8, 1.15, b);
}
// The bed's tint by depth d under the waterline (the reference's two ramps): the albedo is pulled
// toward the water's own albedo raised to a depth power, the lost brightness given back (hue
// shift; the water's extinction does the darkening). Returns the albedo; rough_k = how far the
// roughness goes to its fixed value.
vec3 sw_bed_tint(vec3 a, float d, vec3 water_alb, out float rough_k) {
    rough_k = 0.0;
    if (!SW_ON(16) || d <= 0.0) return a;
    float t = smoothstep(0.0, 1.0, d / 0.5);
    float e = smoothstep(0.0, 1.0, (d - 0.3) / 3.7);
    vec3  a2 = a * pow(clamp(water_alb, vec3(0.02), vec3(1.0)), vec3(e));
    float k = 1.0 + (dot(a, kSwLum) / max(dot(a2, kSwLum), 1e-4) - 1.0) * 1.0;
    rough_k = t;
    return mix(a, k * a2, t);
}

#ifdef SW_TERRAIN
// The sea's swash on the beach (m above the water level): the surf's four fronts on its 20 s
// cycle (terrain_rm_main_water.glsl, the same mask and phase) run up the beach after they reach
// the coast and fall back; 0.12 m in still air to ~0.6 m in a gale.
float sw_swash(vec2 xz, float t) {
    float m = -noise2(xz * 0.0225 + scene.water_swell.yz * 0.03);
    float jit = (noise2(xz * 0.021 + 3.7) - 0.5) * 0.25;
    float lvl = 0.0;
    for (int j = 0; j < 4; ++j) {
        float c = (float(j) + 0.5) * 0.25;
        float msk = min(2.0 * clamp(1.0 - 8.0 * abs(m + c), 0.0, 1.0), 1.0);
        if (msk <= 0.0) continue;
        float fs = 1.0 - 2.0 * fract(0.05 * t + jit + c);          // the front's place: 1 .. -1 (up the beach)
        if (fs >= 0.0) continue;
        float u = -fs;                                              // 0 at the coast .. 1 at the cycle's end
        float run = u < 0.22 ? u / 0.22 : 1.0 - (u - 0.22) / 0.78;   // up fast, back slowly
        lvl = max(lvl, msk * run * run * (3.0 - 2.0 * run));
    }
    return (0.12 + 0.48 * clamp(scene.water_wind.z / 20.0, 0.0, 1.0)) * lvl;
}
// The ground's wet band at p: the waterline is the higher of the authored top (the water level, or
// the stream's surface) and the solver's surface less 0.1 W; at the sea the swash's height, never
// below a dry line under the high water (0.55 of the swash's reach): below it the sand stays wet.
// line: the waterline used (m), swash_edge: 0..1 at the swash's moving edge (its foam line).
float sw_terrain_band(vec3 p, out float line, out float swash_edge) {
    line = -1e9; swash_edge = 0.0;
    if (!SW_ON(1) || scene.water_params.x < 0.5) return 0.0;
    float wl = scene.water_params.y;
#ifdef VIBE_WATER_LEVEL_GLSL
    if (SW_ON(8192) && p.y < wl + 2.5) wl = wl_at(p.xz);      // (the sea under the datum, the river falling: water_level.glsl)
#endif
    float st = wc_stream_surface(p.xz);
    float top = max(wl, st);
    if (p.y > top + 1.5) return 0.0;
    float W = st > wl ? scene.water_wet.w : scene.water_wet.z;
    line = top + max(ws_dh(p.xz) - 0.1 * W, 0.0);
    if (st <= wl && SW_ON(2) && p.y < wl + 1.2) {
        float sk = wt_sea_k(p.xz);
        if (sk > 0.0) {
            W = mix(W, 0.30, sk);
            float R = 0.12 + 0.48 * clamp(max(scene.water_wind.z, scene.water_sun.z) / 20.0, 0.0, 1.0);      // (the open sea's wind: environment.cpp sea_wind)
            float sw = sw_swash(p.xz, scene.water_params.w) * sk;
            float dry = wl + 0.55 * R * sk;
            line = max(line, max(wl + sw, dry));
            swash_edge = sk * (1.0 - smoothstep(0.0, 0.03, abs(p.y - (wl + sw) - 0.01))) * step(0.02, sw);
        }
    }
    return sw_band(p.y, line, W, p.xz);
}
#endif
