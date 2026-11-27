// water_level.glsl - the water's level is not one number any more (2026-11-27, the owner: "make
// the big river a little bit elevated from castle to sea; lower the sea entirely with 1 metre and
// gradually go up to the castle"). The scene's water level (water_params.y) is the DATUM: home's
// castle lake and the river down to below the stone bridge's pool stand at it. The OPEN SEA
// stands 1 m under it, and the river falls from the datum to the sea's level down its course:
// evenly (0.31 m over ~2 km) with three short rapids (0.23 m each over 8 m - 2026-11-27: they were 0.12 m over 20 m and did not read). Lakes away from the
// river stand at the datum. CPU twin: vk_engine/water_fx.cpp water_level_at (keep in step).
// Needs world_map.glsl (wm_riv, wm_riv_box) and terrain_rm_water.glsl (kRivCum, wt_sea_k).
// scene.water_wet bit 8192 (VIBE_NO_SLOPED_RIVER=1: one level everywhere, as before).
#ifndef VIBE_WATER_LEVEL_GLSL
#define VIBE_WATER_LEVEL_GLSL

const float kWlSeaDrop = 1.0;          // the open sea under the datum (m)

// The river's surface under the datum at `along` metres down its course (<= 0).
float wl_profile(float s) {
    float g = clamp((s - 350.0) / 2050.0, 0.0, 1.0);
    float r = smoothstep(696.0, 704.0, s) + smoothstep(1246.0, 1254.0, s) + smoothstep(1796.0, 1804.0, s);
    return -(0.31 * g + 0.23 * r);
}

// The metres down the river's course nearest p, and the distance to its line (no bend: the level
// changes slowly, the wander's metres do not matter to it). d = 1e9 away from the river.
float wl_river_along(vec2 p, out float d) {
    d = 1e9;
    if (rm_far_chart()) return 0.0;
    if (p.x < -200.0 || p.x > 2250.0 || p.y < -250.0 || p.y > 900.0) return 0.0;
    float d2 = 1e12, along = 0.0;
    for (int i = 0; i < 25; ++i) {
        vec2 a = wm_riv[i], ab = wm_riv[i + 1] - a;
        float u = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
        vec2 dv = p - (a + ab * u);
        float e2 = dot(dv, dv);
        if (e2 < d2) { d2 = e2; along = kRivCum[i] + u * length(ab); }
    }
    d = sqrt(d2);
    return along;
}

// The water's level at p, under the datum (<= 0, m).
float wl_off(vec2 p) {
    if (!SW_ON(8192)) return 0.0;
    float sk = wt_sea_k(p);
    if (sk >= 0.999) return -kWlSeaDrop;
    float d;
    float s = wl_river_along(p, d);
    float rk = 1.0 - smoothstep(90.0, 170.0, d);
    float riv = rk > 0.0 ? rk * wl_profile(s) : 0.0;
    return mix(riv, -kWlSeaDrop, sk);
}
float wl_at(vec2 p) { return scene.water_params.y + wl_off(p); }

#endif
