// terrain_rm_water.glsl - part of terrain_raymarch.frag (included there, in place; not a standalone shader).
// Water: flow, normals, foam pattern, ripples

// Cheap animated ocean normal — two scrolling sin/cos directions
// summed at slightly off-axis frequencies. `scale` multiplies the
// base frequencies (1.0 = ~30 m wavelengths; 2.0 = half-wavelength
// finer ripples; 0.5 = bigger sweeping waves).
// Lake-style multi-octave wave normal. Combines a low-frequency
// directional swell (wind blowing along +X) with three cross-direction
// chops, then layers two high-frequency ripples for the close-up
// detail. Each octave is biased toward sharper crests via a sin²-ish
// shape (we use cos for the analytical derivative, equivalent up to
// a phase shift). The result reads as proper lake water rather than
// the previous uniform sinusoidal grid.
// ---- Alternative water style: P_Malin "Where the River Goes" port -----
// Advected FBM-with-derivative normal, blended over two time slices for
// stable directional flow. Flow direction is sampled from the terrain
// gradient at the water plane (water flows downhill). Used when
// rt_.water_style >= 1.
//
// Original Shadertoy: https://www.shadertoy.com/view/4dlGW2 by P_Malin
float pm_hash(float p) {
    vec2 p2 = fract(vec2(p) * vec2(4.438975, 3.972973));
    p2 += dot(p2.yx, p2.xy + 19.19);
    return fract(p2.x * p2.y);
}
vec2 pm_hash2(float p) {
    vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xx + p3.yz) * p3.zy);
}
// Value noise + analytic XY derivative in one call (P_Malin's
// SmoothNoise_DXY). Returns (dx, dy, val).
vec3 pm_noise_dxy(vec2 o) {
    vec2 p = floor(o);
    vec2 f = fract(o);
    float n = p.x + p.y * 57.0;
    float a = pm_hash(n);
    float b = pm_hash(n + 1.0);
    float c = pm_hash(n + 57.0);
    float d = pm_hash(n + 58.0);
    vec2 f2 = f * f, f3 = f2 * f;
    vec2 t  = 3.0 * f2 - 2.0 * f3;
    vec2 dt = 6.0 * f  - 6.0 * f2;
    float u = t.x, v = t.y;
    float res = a + (b - a) * u + (c - a) * v + (a - b + d - c) * u * v;
    float dx  = (b - a) * dt.x + (a - b + d - c) * dt.x * v;
    float dy  = (c - a) * dt.y + (a - b + d - c) * u * dt.y;
    return vec3(dx, dy, res);
}
// 4-octave FBM with per-octave flow advection (P_Malin's FBM_DXY).
vec3 pm_fbm_dxy(vec2 p, vec2 flow, float ps, float df) {
    vec3 f = vec3(0.0);
    float tot = 0.0;
    float a = 1.0;
    for (int i = 0; i < 4; ++i) {
        p += flow;
        flow *= -0.75;
        vec3 v = pm_noise_dxy(p);
        f += v * a;
        p += v.xy * df;
        p *= 2.0;
        tot += a;
        a *= ps;
    }
    return f / tot;
}
// Two-slice time blend of an advected normal (P_Malin's
// SampleFlowingNormal). Returns vec4(normal_xyz, foam_amount).
vec4 pm_flowing_normal(vec2 uv, vec2 flow_rate, float foam, float time) {
    float mag = 2.5 / (1.0 + dot(flow_rate, flow_rate) * 5.0);
    float t0 = fract(time);
    float t1 = fract(time + 0.5);
    float i0 = floor(time);
    float i1 = floor(time + 0.5);
    float o0 = t0 - 0.5;
    float o1 = t1 - 0.5;
    vec2 uv0 = uv + pm_hash2(i0);
    vec2 uv1 = uv + pm_hash2(i1);
    float ga = 0.25 + (foam * -1.5);   // gradient ascent factor
    vec3 d0 = pm_fbm_dxy(uv0 * 20.0, flow_rate * 20.0 * o0, 0.75 + foam * 0.25, ga);
    vec3 d1 = pm_fbm_dxy(uv1 * 20.0, flow_rate * 20.0 * o1, 0.75 + foam * 0.25, ga);
    float fade = max(0.25, 1.0 - foam * 5.0);
    vec3 n0 = mix(vec3(0.0, 1.0, 0.0),
                   normalize(vec3(d0.x, mag, d0.y)), fade);
    vec3 n1 = mix(vec3(0.0, 1.0, 0.0),
                   normalize(vec3(d1.x, mag, d1.y)), fade);
    float w = abs(t0 - 0.5) * 2.0;
    vec3 n = normalize(mix(n0, n1, w));
    float foam_tex = mix(d0.z, d1.z, w) * fade;
    return vec4(n, foam_tex);
}
// Underwater depth at world XZ (P_Malin's GetFlowDistance, adapted).
// Returns 0 outside the water plane; positive metres below the surface.
float pm_flow_depth(vec2 wp, float water_y, float ray_t) {
    // Mesh-terrain mode: river flow follows the real mesh bed (texture
    // lookup) instead of the FBM, so flow/foam agree with the ground.
    if (scene.terrain_local_info.x > 0.5)
        return max(0.0, water_y - meshHeight(wp));
    return max(0.0, water_y - terrainM_lod(wp, ray_t));
}
// Full P_Malin GetFlowRate port. Returns vec3(flow.x, flow.y, foam).
// Flow is the local water velocity in XZ; foam is a 0..1 amount that
// rises in shallow / turbulent zones. The engine's water plane is
// flat so we treat +X as the nominal downstream and let the terrain
// gradient distort the flow from there.
vec3 pm_get_flow_rate(vec2 wp, float water_y, float ray_t) {
    // Sample depth at centre + 4 taps for the gradient.
    const float eps_g = 0.5;
    float d_c  = pm_flow_depth(wp, water_y, ray_t);
    float d_xp = pm_flow_depth(wp + vec2(eps_g, 0.0), water_y, ray_t);
    float d_xn = pm_flow_depth(wp - vec2(eps_g, 0.0), water_y, ray_t);
    float d_zp = pm_flow_depth(wp + vec2(0.0, eps_g), water_y, ray_t);
    float d_zn = pm_flow_depth(wp - vec2(0.0, eps_g), water_y, ray_t);
    // P_Malin's GetGradient is the derivative of (-terrainH) so it
    // points toward deeper water. Use that sign convention.
    vec2 grad = vec2(d_xp - d_xn, d_zp - d_zn) / (2.0 * eps_g);

    // Nominal downstream + gradient-driven push toward deeper water.
    // Gradient term falls off where it's already deep so eddies form
    // in shallow regions, not in the middle of the lake.
    vec2 base = vec2(1.0, 0.0);
    vec2 flow = base + grad * 40.0 / (1.0 + d_c * 1.5);
    // Slow down with depth — surface velocity is highest in shallows.
    flow *= 1.0 / (1.0 + d_c * 0.5);

    // Slow down BEHIND obstacles. dot(normalised_grad, -normalised_flow)
    // is +1 when flow runs into a rising bed → big slowdown; -1 when
    // the bed slopes away ahead → no slowdown. behind ∈ [0..1].
#ifndef QLIKE_WATER
    // Compare squared lengths to avoid two sqrt() in the guard; if
    // both pass, use inversesqrt() to normalize and dot in one pass.
    float grad_l2 = dot(grad, grad);
    float flow_l2 = dot(flow, flow);
#else /* QLIKE_WATER */
    float grad_len = length(grad);
    float flow_len = length(flow);
#endif /* QLIKE_WATER */
    float behind = 0.0;
#ifndef QLIKE_WATER
    if (grad_l2 > 1e-6 && flow_l2 > 1e-6) {
        float inv_g = inversesqrt(grad_l2);
        float inv_f = inversesqrt(flow_l2);
        behind = 0.5 - dot(grad * inv_g, -flow * inv_f) * 0.5;
#else /* QLIKE_WATER */
    if (grad_len > 1e-3 && flow_len > 1e-3) {
        behind = 0.5 - dot(grad / grad_len,
                            -flow / flow_len) * 0.5;
#endif /* QLIKE_WATER */
    }
    float slow = clamp(d_c * 5.0, 0.0, 1.0);
    slow = mix(slow * 0.9 + 0.1, 1.0, behind * 0.9);
    slow = 0.5 + slow * 0.5;
    flow *= slow;

    // Foam amount — concentrates where flow is fast over shallow
    // water (rapids/eddies). P_Malin's exact mapping.
#ifndef QLIKE_WATER
    // Recover the (pre-`flow *= slow`) length from the squared length
    // computed above — one sqrt vs the previous length()+sqrt pair we
    // already paid; behind-test now uses inversesqrt so flow_len wasn't
    // computed there anymore.
    float flow_len = sqrt(flow_l2);
#else /* QLIKE_WATER */
    // flow_len was already computed above; after `flow *= slow` the new
    // length is flow_len * slow. Skips a second length() (sqrt+dot) per
    // pixel. abs() of a non-negative length was a no-op — dropped.
#endif /* QLIKE_WATER */
    float foam = flow_len * slow * 0.5;
    foam += clamp(foam - 0.4, 0.0, 1.0);
    foam = 1.0 - pow(max(0.0, d_c), foam * 0.35);

    return vec3(flow * 0.6, foam);
}
// Foam pattern (P_Malin's SampleWaterFoam). Returns 0..1 mask.
float pm_foam_pattern(vec2 uv, vec2 flow_rate, float time) {
    float t0 = fract(time);
    float t1 = fract(time + 0.5);
    float o0 = t0 - 0.5;
    float o1 = t1 - 0.5;
    vec2 uv0 = uv + pm_hash2(floor(time));
    vec2 uv1 = uv + pm_hash2(floor(time + 0.5));
    float f0 = pm_fbm_dxy(uv0 * 30.0, flow_rate * 50.0 * 0.25 * o0, 0.8, -0.5).z;
    float f1 = pm_fbm_dxy(uv1 * 30.0, flow_rate * 50.0 * 0.25 * o1, 0.8, -0.5).z;
    float w  = abs(t0 - 0.5) * 2.0;
    return mix(f0, f1, w);
}

// ---- Alternative water style: misty-lake-style with bump fade -------
// Inspired by the distance-based bump falloff trick from Reinder
// Nijhoff's "Misty Lake" (Shadertoy, CC BY-NC-SA 4.0). The defining
// visual element is that the water surface goes glassy at distance
// instead of staying chaotic — far water becomes a pure mirror, near
// water keeps its ripples. Implementation uses the engine's noise2
// directly; animation comes from time-shifted FBM samples.
//
// Returns a world-space normal at world XZ, distance-faded.
//   uv_scale  scales sample frequency
//   strength  bump amplitude
//   t         seconds for animation
//   cam_dist  distance from camera to the surface point
//   fade_dist metres at which bumps fade to flat (mirror)
vec3 lakeWaterNormal(vec2 worldXZ, float uv_scale, float strength,
                     float t, float cam_dist, float fade_dist) {
    // Distance falloff: 1 close to camera, 0 past fade_dist.
    float bump_w = 1.0 - smoothstep(0.0, max(0.5, fade_dist), cam_dist);
    if (bump_w < 1e-3) return vec3(0.0, 1.0, 0.0);

    vec2 p = worldXZ * uv_scale;
    // 3-octave FBM with time as a phase offset across octaves so the
    // surface evolves smoothly over time without UV advection.
    // Centred-FD normal extraction — sample 4 neighbours of p.
    const float dx_step = 0.05;
    const float inv_2_dx = 1.0 / (2.0 * 0.05);   // 1/(2*dx_step), folded
    // Octave-1 + octave-2 time offsets are loop-invariants — compute once
    // (was: 4 + 4 redundant vec2(t*a, t*b) builds in the noise calls).
    vec2 t_off_1 = vec2(t * 0.07,  t * 0.04);
    vec2 t_off_2 = vec2(t * 0.13, -t * 0.09);
    vec2 p_o1 = p + t_off_1;
    vec2 p_o2 = p * 2.7 + t_off_2;
    float n_xp = noise2(p_o1 + vec2(dx_step, 0.0));
    float n_xn = noise2(p_o1 - vec2(dx_step, 0.0));
    float n_zp = noise2(p_o1 + vec2(0.0, dx_step));
    float n_zn = noise2(p_o1 - vec2(0.0, dx_step));
    // Add a second octave (higher freq) for ripple texture.
    n_xp += 0.5 * noise2(p_o2 + vec2(dx_step, 0.0));
    n_xn += 0.5 * noise2(p_o2 - vec2(dx_step, 0.0));
    n_zp += 0.5 * noise2(p_o2 + vec2(0.0, dx_step));
    n_zn += 0.5 * noise2(p_o2 - vec2(0.0, dx_step));
    float dx = (n_xp - n_xn) * inv_2_dx;
    float dz = (n_zp - n_zn) * inv_2_dx;
    return normalize(vec3(-dx * strength * bump_w,
                           1.0,
                          -dz * strength * bump_w));
}

vec3 waterNormal(vec2 worldXZ, float t, float strength, float scale) {
    // Wind axis biased on +X — most lakes have a dominant direction.
    vec2 wind  = normalize(vec2(1.0, 0.35));
    vec2 cross1 = normalize(vec2(0.7, -0.7));
    vec2 cross2 = normalize(vec2(-0.5, 0.85));

    // Octaves: low (long swell) → high (ripple) wave numbers.
    // amp[i] decays so high frequencies don't dominate the normal.
    const int kOct = 6;
    float k_base[6]   = float[6](0.06, 0.11, 0.18, 0.27, 0.42, 0.65);
    float spd_base[6] = float[6](0.55, 0.70, 0.95, 1.20, 1.55, 2.10);
    float amp[6]      = float[6](1.00, 0.75, 0.55, 0.40, 0.28, 0.18);

    float dx = 0.0;
    float dz = 0.0;
    // Two octaves go along the wind axis; rest distributed across
    // the cross directions for a non-grid wave field.
    vec2  dir;
    for (int i = 0; i < kOct; ++i) {
        if (i < 2)        dir = wind;
        else if (i < 4)   dir = cross1;
        else              dir = cross2;
        float k = k_base[i] * scale;
        // Phase: dir·worldXZ * k - speed * t.
        float ph = (worldXZ.x * dir.x + worldXZ.y * dir.y) * k -
                   spd_base[i] * t;
        // d/dx of sin(ph) wrt worldXZ = cos(ph) * k * dir.x. Sharpen
        // by raising to a 1.4 power on the absolute term — keeps the
        // sign while pinching crests.
        float c = cos(ph);
        float abs_c = abs(c);
        float sharp = sign(c) * pow(abs_c, 1.4);
        dx += amp[i] * k * dir.x * sharp;
        dz += amp[i] * k * dir.y * sharp;
    }
    return normalize(vec3(-dx * strength, 1.0, -dz * strength));
}

// Splash rings (water_fx.cpp): each impact starts a wave PACKET - a few
// crests travelling outward behind a front, spreading and fading - like
// the capillary-gravity rings a stone makes. Returns the summed surface
// slope (xy, added to the wave normal) and a foam amount (z) for the
// churned patch at the impact point.
vec3 water_ripples(vec2 p) {
    int n = int(scene.ripple_info.x + 0.5);
    vec2 g = vec2(0.0);
    float foam = 0.0;
    for (int i = 0; i < n; ++i) {
        vec4 r = scene.ripples[i];
        float s = clamp(r.w * 28.5, 1.0, 6.0);          // size class: bullet 1 .. bomb 6
        float age = r.z;
        float c = 1.15 * sqrt(s);                       // front speed (m/s)
        float front = c * age + 0.12 * s;
        vec2 d = p - r.xy;
        float d2 = dot(d, d);
        float reach = front + 0.5 * s;
        if (d2 > reach * reach) continue;
        float dist = sqrt(d2);
        float life = 2.5 + 1.5 * s;
        float fade = clamp(1.0 - age / life, 0.0, 1.0);
        float k = 11.0 / s;                             // wavelength ~0.57 m x size
        float back = max(front - dist, 0.0);
        float wid = s * (0.55 + 0.45 * age);            // packet lengthens as it travels
        float packet = smoothstep(front + 0.35 * s, front - 0.10 * s, dist) * exp(-back * back / (wid * wid));
        float att = r.w * fade * sqrt(fade) / (1.0 + 0.30 * dist / s);
        g += d / max(dist, 1e-3) * (att * packet * k * cos(k * (dist - c * age)));
        foam += smoothstep(0.015, 0.035, r.w) *                // droplet rings carry no foam
                (1.0 - smoothstep(0.06 * s, 0.26 * s + 0.10 * s * age, dist)) *
                clamp(1.0 - age / (0.6 + 0.35 * s), 0.0, 1.0);
    }
    // Rain: every cell of a jittered grid gets drops at random moments;
    // each makes one small ring that widens and dies in under a second.
    float rain = scene.ripple_info.y;
    if (rain > 0.01) {
        float tm = scene.water_params.w;
        for (int L = 0; L < 3; ++L) {
            float sc = 1.9 + 1.3 * float(L);
            vec2 q = p * sc + vec2(17.3, 5.1) * float(L);
            vec2 id = floor(q);
            vec2 fq = fract(q) - 0.5;
            float h1 = fract(sin(dot(id, vec2(127.1, 311.7)) + float(L) * 3.7) * 43758.5453);
            float h2 = fract(sin(dot(id, vec2(269.5, 183.3)) + float(L) * 9.1) * 43758.5453);
            float ph = tm * (0.9 + 0.5 * h1) + h2 * 11.0;
            float cyc = floor(ph);
            float tt = fract(ph);
            // a new drop position each cycle; light rain skips most cycles
            float h3 = fract(sin(dot(id + cyc, vec2(41.3, 289.1))) * 43758.5453);
            if (h3 > rain * 0.9 + 0.08) continue;
            vec2 c = (vec2(fract(h3 * 17.0), fract(h3 * 53.0)) - 0.5) * 0.5;
            vec2 d = fq - c;
            float r = length(d);
            float rr = tt * 0.42;
            float band = smoothstep(0.07, 0.0, abs(r - rr));
            float wv = sin((r - rr) * 55.0) * band * (1.0 - tt) * (1.0 - tt);
            g += d / max(r, 1e-3) * (wv * 0.30);
        }
    }
    return vec3(g, clamp(foam, 0.0, 1.0));
}

