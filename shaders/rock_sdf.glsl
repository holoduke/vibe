// Rock detail + small-stone shape.
//
// Rocks are REAL MESHES (world_props.cpp builds the formations at up to
// 20k triangles; scatter.vert shapes the small stones from a unit
// icosphere). The earlier version ray-marched an implicit surface inside
// a bounding hull; that field was not a true distance (noise-displaced
// sphere, hard strata), so the sphere tracer overstepped - rocks deformed
// with the view angle and whole rocks vanished when grazing rays ran out
// of steps - and the ray-traced shadow / collider used a different,
// coarser mesh than the surface that was drawn.
//
// What stays per pixel is only what no mesh LOD carries: cracks, pitting
// and grain, as a small height field over the surface. Its gradient bends
// the mesh normal, its cavity term darkens the cracks.
#ifndef VIBE_ROCK_GLSL
#define VIBE_ROCK_GLSL

// Object space of the formations: unit rock centred at kRockC, y up.
const vec3 kRockC = vec3(0.0, 0.35, 0.0);

float rk_h3(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float rk_vnoise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(rk_h3(i), rk_h3(i + vec3(1, 0, 0)), f.x),
                   mix(rk_h3(i + vec3(0, 1, 0)), rk_h3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(rk_h3(i + vec3(0, 0, 1)), rk_h3(i + vec3(1, 0, 1)), f.x),
                   mix(rk_h3(i + vec3(0, 1, 1)), rk_h3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float rk_fbm(vec3 p, int oct) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < oct; ++i) { s += a * rk_vnoise(p); p = p * 2.03 + vec3(5.1, 1.7, 9.3); a *= 0.5; }
    return s;
}
float rk_h1(float x) { float p = fract(x * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }

// Fine surface relief at object-space point p, in object units (negative =
// into the rock). `fine` 0..1 fades it with distance; `grike` adds the
// straight-ish fissures of a limestone pavement (slab formations, on the
// flat tops only: pass the up-facing weight). `cav` = how deep in a crack
// or pit the point is (0..1), for the albedo.
float rock_detail(vec3 p, float seed, float fine, float grike, out float cav) {
    cav = 0.0;
    float h = 0.0;
    if (grike > 0.01) {
        float gr = abs(rk_vnoise(vec3(p.xz * 2.3 + seed, 0.0)) - 0.5);
        float g = (1.0 - smoothstep(0.0, 0.05, gr)) * grike;
        h -= 0.05 * g;
        cav = g * 0.9;
    }
    if (fine > 0.01) {
        // Cracks: thin ridged valleys. Pitting: fine grain.
        float cr = abs(rk_fbm(p * 6.3 - seed, 3) - 0.5);
        float crack = 1.0 - smoothstep(0.0, 0.024, cr);
        float grain = rk_fbm(p * 27.0 + seed * 2.0, 3) - 0.5;
        h += fine * (-0.022 * crack + 0.012 * grain);
        cav = max(cav, fine * (crack * 0.55 + max(-grain, 0.0) * 0.5));
    }
    return h;
}
// Shading normal: the mesh normal `n` (object space, unit) bent by the
// gradient of rock_detail (4-tap tetrahedron, step `e`).
vec3 rock_detail_normal(vec3 p, vec3 n, float seed, float fine, float grike, float e, out float cav) {
    float c;
    const vec2 k = vec2(1.0, -1.0);
    float h0 = rock_detail(p + k.xyy * e, seed, fine, grike, c);
    float h1 = rock_detail(p + k.yyx * e, seed, fine, grike, c);
    float h2 = rock_detail(p + k.yxy * e, seed, fine, grike, c);
    float h3 = rock_detail(p + k.xxx * e, seed, fine, grike, c);
    rock_detail(p, seed, fine, grike, cav);
    vec3 g = (k.xyy * h0 + k.yyx * h1 + k.yxy * h2 + k.xxx * h3) / (4.0 * e);
    g -= n * dot(g, n);                         // tangential part only
    return normalize(n - g);
}

// ---- Small stones (scatter.vert): radius of the stone along unit
// direction d. A boulder is what is left of a block after it has been
// broken along a handful of planes: the radius is the soft minimum of the
// distances to `kStonePlanes` random planes (soft = worn edges), with a
// little low-frequency swelling. Star-shaped by construction, so a unit
// icosphere pushed out to it is always a valid mesh.
const int kStonePlanes = 9;
float stone_radius(vec3 d, float seed) {
    const float ks = 11.0;                      // edge sharpness (higher = crisper)
    float acc = exp(-ks * 1.25);                // bounding sphere
    for (int i = 0; i < kStonePlanes; ++i) {
        float fi = float(i);
        vec3 n = normalize(vec3(rk_h1(seed + fi * 3.1), rk_h1(seed + fi * 5.7 + 1.3), rk_h1(seed + fi * 7.9 + 2.1)) * 2.0 - 1.0 +
                           vec3(0.0, i < 2 ? 0.9 : 0.0, 0.0));        // the first two lean upward: a top face
        float hgt = 0.60 + 0.36 * rk_h1(seed + fi * 11.3 + 4.0);
        float c = dot(d, n);
        if (c > 0.05) acc += exp(-ks * (hgt / c));
    }
    float r = -log(acc) / ks;
    return r * (1.0 + 0.16 * (rk_fbm(d * 1.5 + seed, 3) - 0.5));
}
#endif
