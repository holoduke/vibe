// Screen-space contact shadow toward the sun - the reference game's screen-space shadow pass,
// ported (docs/ARCHITECTURE.md "Lighting ports"). A short march over a depth picture of the
// frame from a surface point toward the sun: what lies just in front of the march (within a
// thickness) is a blocker. It catches what the ray-traced shadow cannot see: grass, flowers,
// pebbles, the screen-space-displaced relief, far trees drawn as cards, people outside the
// traced set. Needs scene_ubo.glsl (lp_vp, lp_params) before it.
//
// Six steps (the reference: 6, 10 on characters); length 0.2 .. 1.2 m by how much the surface
// faces up, growing with distance (3 % of it, so that whole trees stand on the ground far
// away); the receiver lifted by clamp(0.002 x distance, 5 mm, 0.5 m). Steps that leave the
// screen end the march with "lit" (no dark rim along the screen's edges). The view model has
// no depth in front of the world by less than a thickness and so never shadows it.
#ifndef VIBE_CONTACT_SHADOW_GLSL
#define VIBE_CONTACT_SHADOW_GLSL

// View depth (the clip w) of a depth-picture value.
float lp_view_depth(float z) { return scene.lp_params.x / (z + scene.lp_params.y); }

// How much of the sun the march finds blocked (0 = none, 1 = all). p: the receiver point,
// already lifted off its surface; L: toward the sun; len: the march's length (m);
// jit: 0 .. 1, a new value every frame (the temporal filter averages the steps' placement).
// hit_uv: where the blocker was found (for the caller to weigh by what it is).
float contact_shadow(sampler2D dep, vec3 p, vec3 L, float len, float jit, out vec2 hit_uv) {
    hit_uv = vec2(-1.0);
    vec4 c0 = scene.lp_vp * vec4(p, 1.0);
    vec4 c1 = scene.lp_vp * vec4(p + L * len, 1.0);
    const float kNear = 0.06;
    if (c0.w < kNear) return 0.0;
    if (c1.w < kNear) c1 = mix(c0, c1, (c0.w - kNear) / (c0.w - c1.w));
    const int kSteps = 6;
    // Thickness: the march's own change of depth over a step and a half, at least 2 cm plus
    // 0.4 % of the distance (the reference: half the depth slope over the steps, and a floor).
    float thick = max(abs(c1.w - c0.w) * (1.5 / float(kSteps)), 0.02 + 0.004 * c0.w);
    vec2 sz = vec2(textureSize(dep, 0));
    for (int i = 0; i < kSteps; ++i) {
        float t = (float(i) + jit) / float(kSteps);
        vec4 c = mix(c0, c1, t);
        vec2 uv = (c.xy / c.w) * 0.5 + 0.5;
        if (uv.x <= 0.0 || uv.y <= 0.0 || uv.x >= 1.0 || uv.y >= 1.0) break;
        float zs = texelFetch(dep, ivec2(uv * sz), 0).r;
        if (zs >= 0.999999) continue;
        float ws = lp_view_depth(zs);
        float dz = c.w - ws;                          // > 0: the picture's surface is in front of the march
        if (dz > 0.002 + 0.0005 * c.w && dz < thick) {
            hit_uv = uv;
            // (a blocker near the march's far end fades: the shadow has no hard end)
            return clamp((1.0 - t) * 4.0, 0.0, 1.0);
        }
    }
    return 0.0;
}
#endif
