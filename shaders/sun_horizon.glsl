// Whole-map terrain sun shadow (engine: vk_engine/sun_horizon.cpp).
//
// The map covers the 2048 m terrain. R = height of the terrain's shadow
// volume above that point for the current light direction (the sun is
// visible from p iff p.y is above it), G = that height minus the ground
// height (0 on lit ground).
//
// Returns 1 where the terrain does not block the sun, 0 where it does.
// Ground pixels use G: the rendered surface and the 2 m bake never agree to
// the centimetre, so comparing p.y with R directly would put shadow acne on
// every lit slope. Anything standing clear of the shadow volume (tree tops,
// towers, fog) is lit through R.
float sun_horizon_lit(sampler2D hz, vec3 p) {
    vec2 uv = p.xz / 2048.0 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    vec2 sd = textureLod(hz, uv, 0.0).rg;
    float ground = 1.0 - smoothstep(0.6, 3.5, sd.g);
    float air = smoothstep(-0.5, 2.0, p.y - sd.r);
    return max(ground, air);
}
