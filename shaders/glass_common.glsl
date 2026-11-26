// What a sheet of glass shows (glass.frag, glass_pane.frag). Needs before it: scene_ubo.glsl,
// u_scene_depth (binding 29) and u_scene_copy (binding 38: the lit frame, copied before the glass).

// THROUGH IT. The ray is bent on the way in and back on the way out: it leaves parallel to how
// it came, moved sideways by the pane's thickness x a factor that grows with the angle. The
// scene behind is therefore seen from a point shifted along the pane - drawn here as the
// background looked up a little to the side. (Stronger than a 6 mm pane would give: glass that
// bends nothing reads as a hole.) `extra`: a further shift (a crack's facet). `to_clip` takes
// P's space to clip space; `uv` is the pixel's own place on the screen.
vec4 glass_through(vec3 P, vec3 N, vec3 V, mat4 to_clip, vec2 uv, float frag_z, float dist, vec3 extra) {
    vec3 r = refract(-V, N, 1.0 / 1.52);
    vec3 side = (r + V);                                   // the sideways part of the bend
    vec4 cl = to_clip * vec4(P + side * (0.10 + 0.012 * min(dist, 30.0)) + extra, 1.0);
    vec2 uv_r = cl.w > 1e-4 ? (cl.xy / cl.w) * 0.5 + 0.5 : uv;
    uv_r = clamp(uv_r, vec2(0.001), vec2(0.999));
    // (not what stands in FRONT of the pane: there the background is taken straight)
    if (textureLod(u_scene_depth, uv_r, 0.0).r < frag_z - 1e-6) uv_r = uv;
    return textureLod(u_scene_copy, uv_r, 0.0);
}

// Soda-lime glass is faintly green, more so the longer the path through it.
vec3 glass_absorb(float ndv) {
    return mix(vec3(0.93, 0.965, 0.95), vec3(0.74, 0.87, 0.83), pow(1.0 - ndv, 1.5));
}

// ON IT: the sky and the ground mirrored (no sun).
vec3 glass_mirror(vec3 R) {
    vec3 sky = scene.sky_color.rgb;
    return R.y > 0.0 ? sky * mix(0.75, 1.15, smoothstep(0.0, 0.6, R.y))
                     : mix(sky * 0.55, scene.ambient.rgb * 0.5 + sky * 0.18, smoothstep(0.0, 0.35, -R.y));
}

// The sun's own light (0 at night).
vec3 glass_sunlight() {
    vec3 sun_dir = normalize(scene.sun_direction.xyz);
    return scene.sun_color.rgb * scene.sun_color.a * step(0.0, sun_dir.y);
}

// The sun's glint in a mirror direction R: a tight spot and a wide glow.
vec3 glass_glint(vec3 R, float tight, float tight_gain, float wide, float wide_gain) {
    float sd = max(dot(R, normalize(scene.sun_direction.xyz)), 0.0);
    return glass_sunlight() * (pow(sd, tight) * tight_gain + pow(sd, wide) * wide_gain);
}

float glass_fresnel(float ndv) { return 0.04 + 0.96 * pow(1.0 - ndv, 5.0); }
