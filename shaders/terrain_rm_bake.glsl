// terrain_rm_bake.glsl - part of terrain_raymarch.frag: the baked ground texture.
//
// The ground's material (terrain_rm_main_ground.glsl and the colour part of
// terrain_rm_main_light.glsl) does not change from frame to frame, yet it was worked out for
// every pixel of every frame. vk_engine/terrain_bake.cpp keeps it in a texture round the
// camera instead: kBakeLevels square pictures of kBakeSize texels, each level's texel twice
// the last (kBakeTexel0 = 7.5 mm at level 0), all centred on the camera and addressed by the
// world position itself (the sampler repeats: a texel's place in the picture is its world
// index modulo the picture's size, so when the camera moves only the strip that came into
// the window is drawn anew). They are drawn by a build of this same shader (VIBE_BAKE: main()
// stands over each texel, looking straight down from the level's own distance, runs the same
// material code and writes what the lighting needs instead of lighting it).
//
// Three pictures per level:
//   u_bake_a  rgb = sqrt(albedo), a = the height for the screen-space displacement / kGndSsdmMax
//   u_bake_n  the normal's x and z (0..1)
//   u_bake_m  r = highway asphalt share, g = sand cover, b = the asphalt's gloss,
//             a = 1 where the texel may be used (0: a steep face - a picture laid out from
//                 above has too few texels down a wall -, ground at or under the water, the
//                 castle's floor)
// The level goes by the DISTANCE (level 0 to g_bake_t0, each further level twice as far, the
// two nearest blended): everything in the material that fades with distance is then read at
// very nearly the distance it was baked for.
const int   kBakeLevels = 7;
const int   kBakeSize   = 2048;
const float kBakeTexel0 = 0.0075;
float g_bake_t0 = 0.0;          // the distance level 0 is made for (0 = no baked texture this frame); main()
float g_bake_lmin = 0.0;        // the first level in use (the finest are left alone while the camera moves fast)
#if !defined(VIBE_WATER) && !defined(VIBE_BAKE) && !defined(VIBE_MARCH_ONLY) && !defined(VIBE_LITE)
layout(set = 0, binding = 46) uniform sampler2DArray u_bake_a;
layout(set = 0, binding = 47) uniform sampler2DArray u_bake_n;
layout(set = 0, binding = 48) uniform sampler2DArray u_bake_m;
// The material at `pos`, `t` from the eye, from the baked pictures. False: not there (too far,
// a texel that may not be used, footprints near by - they change from step to step): the
// caller works it out as before.
bool rm_bake_fetch(vec3 pos, float t, out vec3 nor, out vec3 mate, out float hw_pave, out float hw_gloss, out float sand_cov) {
    nor = vec3(0.0, 1.0, 0.0); mate = vec3(0.0); hw_pave = 0.0; hw_gloss = 0.0; sand_cov = 0.0;
    float lf = log2(max(t / g_bake_t0, 1.0));
    if (lf > float(kBakeLevels - 1)) return false;
    if (g_foot_r2 > 0.0 && dot(pos.xz - g_foot_c, pos.xz - g_foot_c) < g_foot_r2 + 1.0) return false;
    { vec2 tr = track_dr(pos.xz); if (tr.x + tr.y > 0.002) return false; }   // (tracks: per pixel, post_fx.cpp)
    lf = max(lf, g_bake_lmin);
    float l0 = floor(lf), f = lf - l0;
    float span0 = float(kBakeSize) * kBakeTexel0 * exp2(l0);
    vec3 u0 = vec3(pos.xz / span0, l0);
    vec4 m = textureLod(u_bake_m, u0, 0.0);
    vec4 a = textureLod(u_bake_a, u0, 0.0);
    vec2 n = textureLod(u_bake_n, u0, 0.0).rg;
    if (f > 0.02 && l0 < float(kBakeLevels - 1) - 0.5) {
        vec3 u1 = vec3(pos.xz / (span0 * 2.0), l0 + 1.0);
        m = mix(m, textureLod(u_bake_m, u1, 0.0), f);
        a = mix(a, textureLod(u_bake_a, u1, 0.0), f);
        n = mix(n, textureLod(u_bake_n, u1, 0.0).rg, f);
    }
    if (m.a < 0.995) return false;
    n = n * 2.0 - 1.0;
    nor = vec3(n.x, sqrt(max(1.0 - dot(n, n), 0.04)), n.y);
    nor = normalize(nor);
    mate = a.rgb * a.rgb;
    g_gnd_ssdm_h = a.a * kGndSsdmMax;
    hw_pave = m.r; sand_cov = m.g; hw_gloss = m.b;
    return true;
}
// How much of the sun the raised ground around leaves the point at `pos` (1 = all): the stones
// the screen-space displacement lifts shade each other - cobbles, the soil's stones, kerbs. A
// short march toward the sun over the baked height (u_bake_a.a), the ray starting on the
// displaced surface and rising with the sun; `amp` = the pass's amplitude factor at this distance.
// (Ours: the reference game gets this from its screen-space shadow pass on the displaced depth;
//  our ground is lit before the displacement runs.)
float rm_bake_relief_shadow(vec3 pos, float t, vec3 L, float amp) {
    if (g_bake_t0 <= 0.0 || L.y < 0.03) return 1.0;
    float lf = max(log2(max(t / g_bake_t0, 1.0)), g_bake_lmin);
    if (lf > 3.0) return 1.0;                                   // (texels of 6 cm and more: no stones in them)
    float l0 = floor(lf);
    float span0 = float(kBakeSize) * kBakeTexel0 * exp2(l0);
    float hmax = kGndSsdmMax * amp;
    float h0 = textureLod(u_bake_a, vec3(pos.xz / span0, l0), 0.0).a * hmax;
    float lxz = length(L.xz);
    if (lxz < 1e-3) return 1.0;
    float rise = L.y / lxz;                                     // metres up per metre along
    float smax = min((hmax - h0) / max(rise, 0.05), 0.9);       // until the ray is over the tallest stone
    if (smax < 0.004) return 1.0;
    vec2  dir = L.xz / lxz;
    float j = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float occ = 0.0;
    const int kN = 8;
    for (int i = 0; i < kN; ++i) {
        float s = smax * (float(i) + 0.35 + 0.65 * j) / float(kN);
        float hs = textureLod(u_bake_a, vec3((pos.xz + dir * s) / span0, l0), 0.0).a * hmax;
        occ = max(occ, hs - (h0 + s * rise));
    }
    return 1.0 - smoothstep(0.0, 0.025 * amp, occ);
}
#endif
