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

// (2026-11-27, "windows should have slightly more reflection"): 7.5 % head-on - a double-glazed
// window has four faces that each give back 4 % (the first two dominate) - rising to a mirror at
// a grazing angle.
float glass_fresnel(float ndv) { return 0.075 + 0.925 * pow(1.0 - ndv, 5.0); }

// ON IT, from what is on the screen: the mirrored line of sight marched through the depth
// buffer in world space, each step projected by `vp` (world -> clip), up to ~70 m. A hit is a
// step that went behind what is drawn there by less than the step's own length; it is refined
// by halving. rgb: the lit scene there (the frame copy), a: how sure (0 = nothing found: off the
// screen, the sky, or it went behind the eye - then the caller's fallback shows).
vec4 glass_ssr(vec3 P, vec3 R, mat4 vp) {
    float t = 0.12, prev = 0.0;
    for (int i = 0; i < 30; ++i) {
        vec4 c = vp * vec4(P + R * t, 1.0);
        if (c.w < 0.08) break;                                         // (behind the eye)
        vec2 s = c.xy / c.w * 0.5 + 0.5;
        if (s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) break;
        float zs = textureLod(u_scene_depth, s, 0.0).r;
        float ds = scene.lp_params.x / (zs + scene.lp_params.y);       // view depth of what is drawn there
        if (c.w > ds + 0.02 && ds > 0.05) {
            if (c.w - ds > (t - prev) * 1.5 + 0.25) break;             // went behind something thick: hidden, not hit
            float lo = prev, hi = t;
            for (int k = 0; k < 5; ++k) {
                float m = 0.5 * (lo + hi);
                vec4 cm = vp * vec4(P + R * m, 1.0);
                vec2 sm = cm.xy / cm.w * 0.5 + 0.5;
                float dm = scene.lp_params.x / (textureLod(u_scene_depth, sm, 0.0).r + scene.lp_params.y);
                if (cm.w > dm) hi = m; else lo = m;
            }
            vec4 ch = vp * vec4(P + R * hi, 1.0);
            vec2 sh = ch.xy / ch.w * 0.5 + 0.5;
            vec2 e = min(sh, 1.0 - sh);
            float conf = smoothstep(0.0, 0.08, min(e.x, e.y)) * (1.0 - smoothstep(40.0, 70.0, t));
            return vec4(textureLod(u_scene_copy, clamp(sh, vec2(0.001), vec2(0.999)), 0.0).rgb, conf);
        }
        prev = t;
        t = t * 1.22 + 0.05;
    }
    return vec4(0.0);
}

// A room seen in its own window (the eye indoors): what is behind the eye is not on the screen,
// but a room is much the same all round - its own picture turned about the middle of the screen,
// softened. Dark outside, this is what the glass shows (as in life, a window at night is a
// mirror of the room); by day the bright view through it outshines it.
vec3 glass_room(vec2 uv) {
    vec2 m = vec2(1.0 - uv.x, uv.y);
    vec3 a = vec3(0.0);
    a += textureLod(u_scene_copy, clamp(m, vec2(0.001), vec2(0.999)), 0.0).rgb * 0.36;
    a += textureLod(u_scene_copy, clamp(m + vec2( 0.035, 0.0), vec2(0.001), vec2(0.999)), 0.0).rgb * 0.16;
    a += textureLod(u_scene_copy, clamp(m + vec2(-0.035, 0.0), vec2(0.001), vec2(0.999)), 0.0).rgb * 0.16;
    a += textureLod(u_scene_copy, clamp(m + vec2(0.0,  0.05), vec2(0.001), vec2(0.999)), 0.0).rgb * 0.16;
    a += textureLod(u_scene_copy, clamp(m + vec2(0.0, -0.05), vec2(0.001), vec2(0.999)), 0.0).rgb * 0.16;
    return a * 0.9;
}

// What the pane mirrors: the scene on the screen where the mirrored ray finds it, else the
// room (eye indoors: `indoor` 1) or the sky and the ground (outdoors), and - outdoors - the sun.
// (scene.lp_params: this frame's projection, filled every frame by light_port.cpp.)
vec3 glass_reflection(vec3 P, vec3 R, mat4 vp, vec2 uv, float indoor) {
    vec4 ssr = glass_ssr(P, R, vp);
    vec3 fall = mix(glass_mirror(R) + glass_glint(R, 900.0, 60.0, 24.0, 0.12), glass_room(uv), clamp(indoor, 0.0, 1.0));
    return mix(fall, ssr.rgb, ssr.a);
}
