// ground_lsd.glsl - the far ground carries the colour of the objects placed on it (the reference
// game's "large scale detail": a top-down capture of its placed objects painted onto the far
// terrain, colour only). vk_engine/ground_fx.cpp draws the placed meshes (the castle, the
// sites' and the station's buildings, bridges, roadside objects, the rock formations) from
// above into a picture of 1024 x 1024 texels of 4 m round the camera (binding 61; rgb =
// sqrt(colour), a = covered), redrawing only what came into the window or changed. The
// ground under them takes that colour from `lsd_fade.x` (where the meshes are still drawn and
// large: nothing, so nothing is doubled) to `lsd_fade.y` (all). The far woods are not in it:
// the ground's own canopy colour covers them (terrain_rm_main_light.glsl, beyond 1.2 km).
// Applied per pixel in the terrain pass beyond the baked ground texture (which ends at
// 64 x t0, about 200 m) - never in the bake. Needs ground_fx.glsl before it.
#ifndef GROUND_LSD_GLSL
#define GROUND_LSD_GLSL
layout(set = 0, binding = 61) uniform sampler2D u_gfx_lsd;

// The colour to lay on the ground at `p` seen from `t` metres (rgb), and how much (a).
vec4 gfx_lsd(vec3 p, float t) {
    if (gfx.lsd.x < 0.5 || (t < gfx.lsd_fade.x && gfx.lsd_fade.w < 0.5)) return vec4(0.0);
    vec2 l = (p.xz - gfx.lsd.yz) / gfx.lsd.w;            // 0..1 across the window
    if (any(lessThan(l, vec2(0.004))) || any(greaterThan(l, vec2(0.996)))) return vec4(0.0);
    // (the picture repeats: a texel's place is its world index modulo the picture's size)
    vec4 s = textureLod(u_gfx_lsd, p.xz / gfx.lsd.w, 0.0);
    if (gfx.lsd_fade.w > 0.5) return vec4(1.0, 0.0, 1.0, s.a);          // (VIBE_TEST_LSD=1: what is covered, magenta, at every distance)
    float edge = 1.0 - smoothstep(0.42, 0.496, max(abs(l.x - 0.5), abs(l.y - 0.5)));
    return vec4(s.rgb * s.rgb, s.a * edge * smoothstep(gfx.lsd_fade.x, gfx.lsd_fade.y, t));
}
#endif
