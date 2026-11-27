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
    // (in one loop: one copy of the ground's height in the compiled shader, not five)
    float d_c = 0.0, d_xp = 0.0, d_xn = 0.0, d_zp = 0.0, d_zn = 0.0;
    RM_ONE_COPY for (int i = 0; i < 5; ++i) {
        vec2 o = i < 3 ? vec2(eps_g, 0.0) : vec2(0.0, eps_g);
        float d = pm_flow_depth(i == 0 ? wp : ((i & 1) == 1 ? wp + o : wp - o), water_y, ray_t);
        if (i == 0) d_c = d; else if (i == 1) d_xp = d; else if (i == 2) d_xn = d; else if (i == 3) d_zp = d; else d_zn = d;
    }
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
#ifndef VIBE_WATER
    // Compare squared lengths to avoid two sqrt() in the guard; if
    // both pass, use inversesqrt() to normalize and dot in one pass.
    float grad_l2 = dot(grad, grad);
    float flow_l2 = dot(flow, flow);
#else /* VIBE_WATER */
    float grad_len = length(grad);
    float flow_len = length(flow);
#endif /* VIBE_WATER */
    float behind = 0.0;
#ifndef VIBE_WATER
    if (grad_l2 > 1e-6 && flow_l2 > 1e-6) {
        float inv_g = inversesqrt(grad_l2);
        float inv_f = inversesqrt(flow_l2);
        behind = 0.5 - dot(grad * inv_g, -flow * inv_f) * 0.5;
#else /* VIBE_WATER */
    if (grad_len > 1e-3 && flow_len > 1e-3) {
        behind = 0.5 - dot(grad / grad_len,
                            -flow / flow_len) * 0.5;
#endif /* VIBE_WATER */
    }
    float slow = clamp(d_c * 5.0, 0.0, 1.0);
    slow = mix(slow * 0.9 + 0.1, 1.0, behind * 0.9);
    slow = 0.5 + slow * 0.5;
    flow *= slow;

    // Foam amount — concentrates where flow is fast over shallow
    // water (rapids/eddies). P_Malin's exact mapping.
#ifndef VIBE_WATER
    // Recover the (pre-`flow *= slow`) length from the squared length
    // computed above — one sqrt vs the previous length()+sqrt pair we
    // already paid; behind-test now uses inversesqrt so flow_len wasn't
    // computed there anymore.
    float flow_len = sqrt(flow_l2);
#else /* VIBE_WATER */
    // flow_len was already computed above; after `flow *= slow` the new
    // length is flow_len * slow. Skips a second length() (sqrt+dot) per
    // pixel. abs() of a non-negative length was a no-op — dropped.
#endif /* VIBE_WATER */
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
    // (the weather's wind: scene.water_wind; the cross trains keep their angles to it)
    vec2 wind  = scene.water_wind.xy;
    vec2 cross1 = vec2(0.7071 * (wind.x + wind.y), 0.7071 * (wind.y - wind.x));
    vec2 cross2 = vec2(-0.5 * wind.x - 0.85 * wind.y, 0.85 * wind.x - 0.5 * wind.y) / 0.9862;

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
    // (inside the ripple field's window the field carries the rings - water_sim.cpp feeds every
    //  new ring into it; here only their churned foam then)
    vec2 rip_uv = scene.water_rip.w > 0.5 ? (p - scene.water_rip.xy) / scene.water_rip.z : vec2(-1.0);
    bool in_field = all(greaterThan(rip_uv, vec2(0.12))) && all(lessThan(rip_uv, vec2(0.88)));
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
        if (!in_field) g += d / max(dist, 1e-3) * (att * packet * k * cos(k * (dist - c * age)));
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
            // (a ring 2 - 3 crests wide, its slope up to 0.8: at 0.3 and one crest the rings did not
            //  read against the mirror in heavy rain at two metres - the rain fell on a still lake)
            float band = 1.0 - smoothstep(0.0, 0.10, abs(r - rr));
            float wv = sin((r - rr) * 55.0) * band * (1.0 - tt) * (1.0 - tt);
            g += d / max(r, 1e-3) * (wv * 0.80);
        }
    }
    return vec3(g, clamp(foam, 0.0, 1.0));
}


// ==== The wind on the water, the swell, the surf, the caustics, the river's current ==========
// After the reference game's sea (its formulas and constants; the code is ours).
// scene.water_wind: xy = the way the wind blows, z = speed m/s, w = the swell's height scale
// pow(saturate(speed / 30), 2.2) (environment.cpp water_wind_now).

// How much of the sea's wind waves the wind raises (1 at 8 m/s; a glassy sea in still air).
float wt_chop() { return clamp(0.12 + scene.water_wind.z / 9.0, 0.12, 1.8); }
// ... and of a lake's or the river's: little fetch, so they stay calm longer and rise less.
float wt_lake() { return clamp(0.10 + scene.water_wind.z * 0.11, 0.10, 1.5); }

// The open sea: beyond home's lakes and not land-locked water (CPU twin: water_fx.cpp).
float wt_sea_k(vec2 xz) {
    float k = smoothstep(1350.0, 1750.0, rm_home_dist(xz));
    if (k > 0.0) k *= 1.0 - smoothstep(0.0, 0.02, rm_continent(xz));
    return k;
}
// The swell fades out toward the coast by the depth under the water level, always faster than
// the water gets shallow: a trough never drops below the bed and a crest never floods the beach
// (with the swell's whole height Hn = 32.5 S, a trough reaches 0.43 Hn below the level).
float wt_swell_shore(float depth, float S) { return smoothstep(0.0, 1.0, depth / (2.0 + 39.0 * S)); }

// The five swell waves (lengths 400 / 180 / 150 / 130 / 120 m; CPU twin water_fx.cpp
// water_swell_height - keep the constants and the clock in step). Returns the height about
// the mean level for S = 1 and its slope (xz).
const float kSwK[5]    = float[5](0.015708, 0.034907, 0.041888, 0.048332, 0.052360);
const float kSwF[5]    = float[5](0.0937143, 0.2142082, 0.3162719, 0.3835667, 0.4790747);
const float kSwA[5]    = float[5](13.0, 8.0, 7.0, 2.5, 2.0);
const float kSwWarp[5] = float[5](50.0, 10.0, 100.0, 200.0, 180.0);
const vec2  kSwRot[5]  = vec2[5](vec2(1.0, 0.0), vec2(0.7660444, 0.6427876), vec2(0.8910065, -0.4539905),
                                 vec2(0.9612617, 0.2756374), vec2(0.9781476, -0.2079117));
const float kSwMean    = 0.30547;      // the mean of 1.3 exp(sin p - 1) - 0.3: taken off, the sea's mean level stays put
float wt_swell_noise(vec2 x) {
    vec2 q = x * 0.005 - scene.water_swell.yz * 0.125;       // (drifting down the wind: water_swell.yz, the wind summed over time)
    float s = 0.5 * sin(1.5 * q.x) * sin(1.5 * q.y);
    vec2 r = vec2(0.8 * q.x - 0.6 * q.y, 0.6 * q.x + 0.8 * q.y);
    s += 0.25 * sin(3.03 * r.x) * sin(3.03 * r.y);
    r = 2.02 * vec2(0.8 * r.x - 0.6 * r.y, 0.6 * r.x + 0.8 * r.y); s += 0.125 * sin(3.045 * r.x) * sin(3.045 * r.y);
    r = 2.03 * vec2(0.8 * r.x - 0.6 * r.y, 0.6 * r.x + 0.8 * r.y); s += 0.0625 * sin(3.015 * r.x) * sin(3.015 * r.y);
    return abs(s * 1.032258);
}
vec3 wt_swell(vec2 x, float t) {
    vec2 wd = scene.water_wind.xy;
    float n = wt_swell_noise(x);
    float h = 0.0;
    vec2 g = vec2(0.0);
    for (int i = 0; i < 5; ++i) {
        vec2 c = kSwRot[i];
        vec2 d = -vec2(wd.x * c.x - wd.y * c.y, wd.x * c.y + wd.y * c.x);       // (minus the wind, turned: the crests run with the wind)
        float ph = 6.2831853 * fract(kSwF[i] * t) + kSwK[i] * (dot(d, x) + kSwWarp[i] * n);
        float e = 1.3 * exp(sin(ph) - 1.0);
        h += kSwA[i] * (e - 0.3 - kSwMean);
        g += d * (kSwA[i] * kSwK[i] * e * cos(ph));
    }
    return vec3(h, g);
}

// Caustics on the bed (the same pattern as compose.frag uw_caustic, the view from below;
// after Dave Hoskins' "Tileable Water Caustic"): thin bright lines, ~0..1.5.
float wt_caustic(vec2 uv, float t) {
    const float TAU = 6.28318530718;
    vec2 p = mod(uv * TAU, TAU) - 250.0;
    vec2 i = p;
    float c = 1.0;
    const float inten = 0.005;
    for (int n = 0; n < 4; ++n) {
        float tt = t * (1.0 - (3.5 / float(n + 1)));
        i = p + vec2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
        c += 1.0 / length(vec2(p.x / (sin(i.x + tt) / inten), p.y / (cos(i.y + tt) / inten)));
    }
    c /= 4.0;
    c = 1.17 - pow(c, 1.4);
    return clamp(pow(abs(c), 8.0), 0.0, 1.5);
}

// The river's course at xz (world_map.glsl wm_river's polyline and its bend): xy = the way it
// flows (unit, toward the sea), z = the distance from the centre line (m), w = the channel's
// half width there. z < 0: not by the river (home chart only).
const float kRivCum[26] = float[26](0.0, 41.6, 85.9, 144.0, 206.1, 270.6, 339.6, 401.3, 503.5, 617.4, 733.9, 847.1, 941.3, 1056.7, 1158.5, 1272.8, 1370.5, 1466.5, 1572.9, 1668.9, 1791.4, 1907.9, 2009.1, 2146.5, 2299.0, 2497.0);   // (metres down the course to each point of wm_riv)
// `along`: metres down the course from the spring by the castle (the nearest reach's).
// `frame`: metres down the course and across it, in the bent coordinate the channel is cut along. Both are blended from the
// nearest reaches with the same weights as the direction, so they are smooth across a bend
// (the nearest reach's own values jump on the bisector). They are the frame the river's patterns
// are laid in: a pattern laid by projecting the absolute position on the flow's direction
// (dot(wp, dir)) has a lever of a kilometre - the direction turning by a hundredth of a radian
// over a metre moved it by ten metres: the streaks broke into thin lines a few pixels apart
// (the white lines across the river by the confluence).
vec4 wt_river(vec2 wp, out float along, out vec2 frame) {
    along = 0.0; frame = vec2(0.0);
    if (rm_far_chart() || wp.x < -85.0 || wp.x > 2095.0 || wp.y < -110.0 || wp.y > 780.0) return vec4(0.0, 0.0, -1.0, 0.0);
    vec2 q = wp + 12.0 * vec2(sin(wp.y * 0.043 + 1.3), sin(wp.x * 0.037 + 0.4)) + 5.0 * vec2(sin(wp.y * 0.113), sin(wp.x * 0.097 + 2.0));
    float pool = 1.0 - smoothstep(22.0, 55.0, distance(wp, vec2(43.4, 297.4)));
    q -= pool * 6.0 * vec2(0.719, 0.695);
    float d2 = 1e12;
    vec2 tan_ = vec2(1e-4, 0.0);
    float fw = 1e-9, fal = 0.0, flat_ = 0.0;
    for (int c = 0; c < 5; ++c) {
        vec4 b = wm_riv_box[c];
        if (q.x < b.x || q.y < b.y || q.x > b.z || q.y > b.w) continue;
        for (int i = c * 5; i < c * 5 + 5; ++i) {
            vec2 a = wm_riv[i], ab = wm_riv[i + 1] - a;
            vec2 dv = q - (a + ab * clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0));
            float e2 = dot(dv, dv);
            if (e2 < d2) along = kRivCum[i] + clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0) * length(ab);
            d2 = min(d2, e2);
            float wq = 1.0 / (1.0 + e2 * (1.0 / 400.0));          // (the nearest reaches, weighted: no seam where two meet at a bend)
            tan_ += normalize(ab) * (wq * wq);
            float w4 = wq * wq * wq * wq;      // (the frame: sharper, so far reaches do not pull it)
            fw += w4;
            fal += w4 * (kRivCum[i] + clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0) * length(ab));
            vec2 abn = normalize(ab);
            flat_ += w4 * (abn.x * (q - a).y - abn.y * (q - a).x);
        }
    }
    frame = vec2(fal, flat_) / fw;
    if (d2 > 60.0 * 60.0) return vec4(0.0, 0.0, -1.0, 0.0);
    // The channel is cut along the BENT coordinate q: its way in the world is the polyline's
    // way taken back through the bend (q = wp + wander: the inverse of its Jacobian - the
    // wander turns the channel by up to 45 degrees from its polyline).
    float ja = 0.516 * cos(wp.y * 0.043 + 1.3) + 0.565 * cos(wp.y * 0.113);      // d q.x / d z
    float jb = 0.444 * cos(wp.x * 0.037 + 0.4) + 0.485 * cos(wp.x * 0.097 + 2.0); // d q.y / d x
    vec2 tq = normalize(tan_);
    tan_ = vec2(tq.x - ja * tq.y, tq.y - jb * tq.x) * sign(1.0 - ja * jb);
    tan_ = dot(tan_, tan_) > 1e-12 ? normalize(tan_) : tq;       // (a singular bend: no NaN - water_fx.cpp river_at, the same)
    return vec4(tan_, sqrt(d2), 7.0 + 9.0 * smoothstep(1450.0, 1900.0, wp.x) + 4.0 * pool);
}

// The river's water along its course (the reference game paints it per vertex; here a table by
// the metres down the course, wt_river's `along`): rgb = the colour it scatters back, w = how
// fast light dies in it (/m). Clear at the spring by the castle, green in the pool under the
// stone bridge, a greenish brown down the valley, silty toward the mouth.
vec4 wt_river_murk(float s) {
    const vec4 kSpring = vec4(0.06, 0.11, 0.10, 0.22);
    const vec4 kPool   = vec4(0.05, 0.12, 0.06, 0.85);
    const vec4 kValley = vec4(0.08, 0.11, 0.06, 0.60);
    const vec4 kMouth  = vec4(0.15, 0.13, 0.08, 1.60);
    vec4 m = mix(kSpring, kPool, smoothstep(150.0, 250.0, s));
    m = mix(m, kValley, smoothstep(330.0, 480.0, s));
    return mix(m, kMouth, smoothstep(1700.0, 2150.0, s));
}

// One octave of value noise with its slope (pm_noise_dxy), as a surface normal's tilt: the
// river's near layers (the reference game: 1.4 and 1.05 /m, each drifting at its own speed).
vec2 wt_near_slope(vec2 p) {
    vec3 n = pm_noise_dxy(p);
    return n.xy;
}

// The water round the piers and rocks standing in it (scene.river_piers: xy = the middle,
// z = its radius in the current; scene.water_swell.w = how many - water_fx.cpp river_piers,
// the rocks from rock_formations.cpp). In the current's own frame (fdir, speed m/s):
//   * the current parts round it: potential flow round a cylinder (dflow; the CPU twin bends
//     the same way: water_fx.cpp obstacle_bend);
//   * a pillow of water heaped against the upstream face, white;
//   * a V-shaped wake from its sides and a trail of foam in its lee;
//   * EDDIES: vortices shed left and right in turn (a vortex street: one every D / (0.21 U)
//     seconds), carried downstream at 0.8 U, growing and fading over a few seconds - they turn
//     the surface's normal round their centres (slope) and draw foam spirals.
// Everything is stronger where the current is faster.
struct WtObs { vec2 dflow; vec2 slope; float foam; float hole; };
WtObs wt_obstacles(vec2 p, vec2 fdir, float speed, float t) {
    WtObs o;
    o.dflow = vec2(0.0); o.slope = vec2(0.0); o.foam = 0.0; o.hole = 0.0;
    int n = int(scene.water_swell.w + 0.5);
    vec2 ac = vec2(-fdir.y, fdir.x);
    float U = max(speed, 0.15);
    float vis = smoothstep(0.2, 0.9, speed);
    for (int i = 0; i < n; ++i) {
        vec4 P = scene.river_piers[i];
        vec2 d = p - P.xy;
        float a = max(P.z, 0.2);
        float reach = max(a * (5.0 + 4.0 * U) + 5.0, 2.0 * (a * (6.0 + 4.0 * U) + 4.0));     // (the foam lines run to twice their length scale)
        float r2 = dot(d, d);
        if (r2 > reach * reach) continue;
        float r = sqrt(r2);
        float down = dot(d, fdir), lat = dot(d, ac);
        // (potential flow round a cylinder)
        if (r > a) {
            float k = a * a / r2;
            float c2 = (down * down - lat * lat) / r2, s2 = 2.0 * down * lat / r2;
            o.dflow += (fdir * (-k * c2) + ac * (-k * s2)) * speed;
        }
        float ring = r - a;
        // the pillow on the upstream face
        float upf = 1.0 - smoothstep(-0.2 * a, 0.5 * a, down);
        float pillow = upf * exp(-max(ring, 0.0) / (0.25 + 0.12 * a + 0.15 * U)) * step(-0.25 * a, ring);
        o.foam = max(o.foam, pillow * vis);
        o.slope += (d / max(r, 1e-3)) * (pillow * 0.30 * U);
        if (down > 0.0) {
            // The wake's white water dies within a few of the stone's own widths (it was 6 - 30 m
            // whatever the stone: in the stream, with a boulder every few metres, the trails and V
            // arms of the 24 nearest covered the whole water in opaque froth - the stream read as
            // a pale sheet; on the river they were long thin white arcs).
            // the V of the wake (its arms open at about 20 degrees): mostly a standing wave, a little foam
            float arm = abs(abs(lat) - (a * 0.9 + down * 0.36));
            float vw = exp(-arm * arm / (0.06 + 0.012 * down)) * exp(-down / (1.0 + 1.5 * a + 1.5 * U));
            // (white on its arms only in fast water - in the river's slow reaches a thin white V
            //  drawn tens of metres long read as ruled lines on the water)
            o.foam = max(o.foam, vw * 0.35 * vis * smoothstep(1.2, 2.5, U));
            o.slope += ac * (sign(lat) * vw * 0.12 * U);
            // the trail in the lee
            float wl = a * (0.9 + 0.06 * down);
            float tail = exp(-lat * lat / (wl * wl)) * exp(-down / (a * (2.0 + 1.2 * U) + 0.8)) * smoothstep(1.2 * a, 3.0 * a, down);
            o.foam = max(o.foam, tail * 0.6 * vis);
            // the foam LINES off its flanks (2026-11-27): on a real river the shear lines behind a
            // stone or pier draw its white water out into two thin ribbons that drift on for many
            // of its widths, closing slowly, wandering, broken into lengths. The noise is laid in a
            // frame moving down the current at 0.9 U, so they drift at the surface speed. (Inside
            // the solver's window the caller fades this out: the solver's drift foam does it there.)
            float Ll = a * (6.0 + 4.0 * U) + 4.0;
            if (down < 2.0 * Ll) {
                float sd = down - 0.9 * U * t;
                float off = a * (1.0 - 0.35 * smoothstep(0.0, Ll, down));
                float wv = (noise2(vec2(sd * 0.25, float(i) * 3.1)) - 0.5) * (0.3 + 0.05 * down);
                float lw = 0.10 + 0.02 * down + 0.05 * a;
                float e1 = lat - off - wv, e2 = lat + off - wv;
                float lv = min(exp(-e1 * e1 / (lw * lw)) + exp(-e2 * e2 / (lw * lw)), 1.0);
                float brk = smoothstep(0.35, 0.65, noise2(vec2(sd * 0.4 + sign(lat) * 13.0, float(i) * 7.7)));
                o.foam = max(o.foam, lv * brk * exp(-down / Ll) * smoothstep(0.5 * a, 1.5 * a, down) * 0.5 * vis);
            }
            // the HOLE right behind it: the water drops into the lee and turns back - dark, no
            // white (it breaks a white sheet into a dark gap), before the trail begins
            float hl = a * 0.85;
            o.hole = max(o.hole, exp(-lat * lat / (hl * hl)) * exp(-down / (1.2 * a + 0.6 * U)) * smoothstep(0.0, 0.35 * a, down) * vis);
            o.slope -= fdir * (exp(-lat * lat / (hl * hl)) * exp(-down / (1.2 * a + 0.6 * U)) * 0.25 * U);      // (the surface dips into it)
        }
        // the eddies: a vortex street behind it
        float D = 2.0 * a;
        float Tsh = D / (0.21 * U);
        float ph = t / Tsh;
        float base = floor(ph);
        for (int j = 0; j < 6; ++j) {
            float nj = base - float(j);
            float age = (ph - nj) * Tsh;
            float side = (mod(nj, 2.0) < 0.5) ? 1.0 : -1.0;
            vec2  c = P.xy + fdir * (a + 0.8 * U * age) + ac * (side * 0.33 * D);
            vec2  e = p - c;
            float R = 0.55 * a + 0.12 * U * age + 0.2;
            float er2 = dot(e, e);
            if (er2 > 9.0 * R * R) continue;
            float er = sqrt(er2);
            float str = side * exp(-age / (2.5 + 1.5 * a)) * smoothstep(0.0, 0.5, age);
            float sw = exp(-er2 / (R * R));
            vec2  tang = vec2(-e.y, e.x) / max(R, 0.1);
            o.slope += tang * (sw * str * 0.35 * U);
            // (the eddy's spin is NOT added to the flow that carries the ripple pattern: the
            //  pattern is carried for up to 4 s, and a spin that fast wound it into rings of fine
            //  stripes round every stone. The eddy shows through the slope and its foam spiral.)
            float ang = atan(e.y, e.x);
            float spiral = 0.5 + 0.5 * sin(2.0 * ang + side * (er / R) * 5.0 - side * t * 2.5 * U);
            o.foam = max(o.foam, sw * abs(str) * smoothstep(0.45, 0.9, spiral) * 0.55 * vis);
        }
    }
    return o;
}


// FOAM AS LACE (2026-11-26, the owner's "huge white spots"): the reference draws white water as
// amount x a foam TEXTURE - bubble walls with open water between, thick foam with narrow holes,
// thin foam mostly holes. The stream's and the solver's white was drawn as the amount alone
// (smooth noises of 0.4 - 12 m), mixed in at 85 % opacity: smooth opaque blobs and veils. This
// turns an amount into coverage: two sizes of cells (38 and 14 cm), carried with the current in
// two phases (2 s cycle), broken into patches where thin, and the mean coverage where a pixel is
// larger than the cells (no sparkle far off).
float wt_foam_lace_at(vec2 q, float thick) {
    float c1 = foam_cells(q * 2.6);
    float c2 = foam_cells(q * 7.0 + 7.3);
    float wall1 = 1.0 - smoothstep(0.04 + 0.22 * thick, 0.16 + 0.55 * thick, c1);
    float wall2 = 1.0 - smoothstep(0.03 + 0.15 * thick, 0.14 + 0.40 * thick, c2);
    return max(wall1, 0.75 * wall2 * mix(0.4, 1.0, clamp(wall1 + thick, 0.0, 1.0)));
}
float wt_foam_lace(vec2 p, vec2 flow, float amt, float t, float px) {
    if (amt <= 0.002) return 0.0;
    float thick = clamp(amt, 0.0, 1.0);
    float cyc = t * 0.5, ph0 = fract(cyc), ph1 = fract(cyc + 0.5);
    float w0 = 1.0 - abs(1.0 - 2.0 * ph0);
    vec2 q0 = p - flow * (ph0 * 2.0) + pm_hash2(floor(cyc) + 41.0) * 9.0;
    vec2 q1 = p - flow * (ph1 * 2.0) + pm_hash2(floor(cyc + 0.5) + 53.0) * 9.0;
    float far = smoothstep(0.04, 0.12, px);          // (a pixel over a third of the small cells)
    float lace = 0.30 + 0.55 * thick;                // (about the lace's mean coverage)
    if (far < 1.0) lace = mix(mix(wt_foam_lace_at(q1, thick), wt_foam_lace_at(q0, thick), w0), lace, far);
    // (thin foam comes in patches that break up; thick foam is nearly whole)
    float brk = mix(smoothstep(0.30, 0.70, mix(noise2(q1 * 0.55 + 3.0), noise2(q0 * 0.55 + 3.0), w0)), 1.0, thick * thick);
    return clamp(min(amt * 1.4, 1.0) * mix(lace, 1.0, 0.12 * thick * thick) * brk, 0.0, 1.0);
}
