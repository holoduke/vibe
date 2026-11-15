// The lawn's blade field as a distance function, shared by every shader
// that needs blade shadows (the raymarched terrain, the lawn itself and the
// deferred pass for objects standing in the grass). It is the same field
// grass_raymarch.frag draws: one blade per 17 cm cell, hashed position,
// rotation, height and wind phase.
//
// Requires: scene_ubo.glsl, and a sampler2D `u_grass_mask` (the grass
// eligibility mask over the 2048 m map) declared before this include.

vec4 gs_hash42(vec2 p) {
    vec4 p4 = fract(vec4(p.xyxy) * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}
float gs_blade2d(vec2 p) {
    float d = length(p - vec2(1.7, -1.3)) - 2.0;
    d = max(d, -(length(p - vec2(1.7, -1.0)) - 1.8));
    d = max(d, -(p.y + 1.0));
    d = max(d, -(-p.x + 1.7));
    return d;
}
float gs_blade(vec3 p, float th) {
    p.y -= 1.0;
    float d2 = max(0.0, gs_blade2d(p.xy));
    return sqrt(d2 * d2 + p.z * p.z) - th;
}
// Distance to the nearest blade; `gy` = ground height (blade roots).
float gs_map(vec3 p, float gy) {
    const float kPeriod = 0.17;
    const float kPi2 = 6.28318530718;
    vec3 lp = p; lp.y -= gy;
    if (lp.y > 0.9) return lp.y - 0.6;
    vec2 cellId = floor((lp.xz + 0.5 * kPeriod) / kPeriod);
    lp.xz = mod(lp.xz + 0.5 * kPeriod, kPeriod) - 0.5 * kPeriod;
    float bh = clamp(scene.grass_rm_params.x, 0.05, 3.0) * 0.36;
    float d = 1e3;
    // Eligibility once, at the centre cell (as grass_raymarch.frag does):
    // the mask is 2 m per texel, the 3x3 block spans 0.5 m.
    {
        vec2 mg = textureLod(u_grass_mask, (cellId * kPeriod / 2048.0) + vec2(0.5), 0.0).rg;
        float sm = scene.grass_rm_params2.x;
        if (mg.r * (1.0 - smoothstep(sm * 0.7, sm * 1.2, mg.g * 2.0)) < 0.10) return 1e3;
    }
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            vec2 nId = cellId + vec2(float(dx), float(dy));
            vec4 r = gs_hash42(nId);
            if (r.x > scene.grass_rm_params.w) continue;
            vec3 np = lp - vec3(float(dx), 0.0, float(dy)) * kPeriod;
            float a = r.z * kPi2;
            float c = cos(a), s = sin(a);
            np.xz = mat2(c, -s, s, c) * np.xz;
            np.xz += (r.xy - 0.5) * kPeriod;
            float wind_t = scene.grass_rm_params.z + r.w * kPi2;
            np.x += sin(wind_t * 1.4) * scene.grass_rm_params.y * 0.025 *
                    smoothstep(0.0, 1.7 * bh, np.y);
            float sc = max(0.65, sqrt(r.w)) * bh;
            d = min(d, gs_blade(np / sc, 0.002 / sc) * sc);
        }
    }
    return d;
}
// How much light from direction L (toward the light) gets through the
// blades to point p, 0..1. `gy` = ground height under p, `t0` = where the
// march starts (a few cm when p sits on a blade itself, so it does not
// shadow itself). Sphere-traced with a wide penumbra: hair-thin shadows
// would otherwise alias away.
float grass_blade_vis(vec3 p, float gy, vec3 L, float t0) {
    float bh = clamp(scene.grass_rm_params.x, 0.05, 3.0) * 0.36;
    float top = gy + bh * 1.05;
    if (p.y >= top && L.y >= 0.0) return 1.0;            // above the blades, light from above
    // Until the ray leaves the blade layer (or 2 m: a light low in the grass).
    float t_max = L.y > 0.02 ? min((top - p.y) / L.y, 2.0) : 1.5;
    if (t_max <= t0) return 1.0;
    float res = 1.0;
    float t = t0;
    for (int i = 0; i < 12; ++i) {
        if (t > t_max) break;
        vec3 q = p + L * t;
        float d = gs_map(q, gy);
        res = min(res, 7.0 * max(d, 0.0) / t);
        if (res < 0.02) break;
        t += clamp(d, 0.015, 0.14);
    }
    return smoothstep(0.0, 1.0, res);
}
