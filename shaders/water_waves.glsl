// water_waves.glsl - the water's one set of wave functions. Every body of water - the open sea, the
// lakes, the river, the stream - takes its fine waves, its ripples and its distance to the shore
// from these, scaled by fetch, depth and flow (no separate code per kind of water).
// Their data come from vk_engine/water_sim.cpp (shaders/water_sim.comp): the wind sea's four FFT
// cascades (slope, fold, height with mips; the displacement) and the ripple field round the camera.
//
// The includer declares its samplers and names them before including (after scene_ubo.glsl):
//   #define WW_GRAD <sampler2DArray>   slope xy, fold z, height w (scene binding 51)
//   #define WW_DISP <sampler2DArray>   choppy x, height, choppy z (52)          (optional)
//   #define WW_RIP  <sampler2D>        the ripple field: x height, y last step's (53) (optional)
#ifndef VIBE_WATER_WAVES_GLSL
#define VIBE_WATER_WAVES_GLSL

bool ww_fft_on() { return scene.water_fft.x > 0.5; }

// How much of each cascade (41, 13.7, 4.4, 1.37 m) a body of water raises: a wave does not grow
// longer than about a thirtieth of the stretch of water the wind blows over (its fetch) - a pond
// has only the short ones, the open sea all of them.
vec4 ww_fetch_w(float fetch) {
    return clamp(vec4(fetch) / (30.0 * scene.water_fft_L), vec4(0.0), vec4(1.0));
}

#ifdef WW_GRAD
// Slope (xy, rise per metre along x and z) and fold (z; > 0 where the choppy surface crowds
// together, > 1 where it folds over: whitecaps) of the cascades at p, each weighted by w.
// gx, gy: the pixel's footprint on the water (m) along its two axes - the mips (and the
// anisotropic filter along a grazing view) average the slopes, so far water flattens; that is the
// reference's only distance filter. `rot`: the frame the waves are laid in (unit; (1, 0) = the
// world's: the wind sea; a river passes its course's frame so the waves run with the current).
vec3 ww_fft_slope(vec2 p, vec2 gx, vec2 gy, vec4 w) {
    vec3 s = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        if (w[i] <= 0.001) continue;
        float L = scene.water_fft_L[i];
        vec4 g = textureGrad(WW_GRAD, vec3(p / L, float(i)), gx / L, gy / L);
        s += w[i] * g.xyz;
    }
    return s;
}
// Running water: the cascades laid in the course's own frame (fp = metres down the course and
// across it, carried with the current as the caller's two phases do) so that their waves run
// DOWN the current instead of with the wind - the spectrum's wind axis is put along the course.
// ax: the way the water runs (world, unit); gx, gy: the footprint (world). Slope back in world x z.
vec3 ww_fft_slope_flow(vec2 fp, vec2 ax, vec2 gx, vec2 gy, vec4 w) {
    vec2 ay = vec2(-ax.y, ax.x);
    vec2 wd = scene.water_wind.xy, wp = vec2(-wd.y, wd.x);
    vec2 p  = fp.x * wd + fp.y * wp;                                   // frame -> the cascades' own place
    vec2 fx = vec2(dot(gx, ax), dot(gx, ay)), fy = vec2(dot(gy, ax), dot(gy, ay));
    vec3 s = ww_fft_slope(p, fx.x * wd + fx.y * wp, fy.x * wd + fy.y * wp, w);
    vec2 sf = vec2(dot(s.xy, wd), dot(s.xy, wp));                      // the cascades' slope -> the frame's
    return vec3(ax * sf.x + ay * sf.y, s.z);
}
#endif

#ifdef WW_DISP
// The height of the cascades at p (m about the level), each cascade faded out where a pixel
// (`foot`, m) is wider than its waves. (CPU twin: water_sim.cpp water_fft_height - with every
// cascade: the twin serves the player's own surroundings.)
float ww_fft_height(vec2 p, float foot, vec4 w) {
    float h = 0.0;
    for (int i = 0; i < 4; ++i) {
        float L = scene.water_fft_L[i];
        float k = w[i] * (1.0 - smoothstep(0.03, 0.12, foot / L));
        if (k <= 0.001) continue;
        h += k * textureLod(WW_DISP, vec3(p / L, float(i)), 0.0).y;
    }
    return h;
}
#endif

#ifdef WW_RIP
// The ripple field (wakes, rings off the banks): its slope (xy) and height (z) at p; nothing
// outside its window round the camera, faded over its last metres.
vec3 ww_ripple(vec2 p) {
    if (scene.water_rip.w < 0.5) return vec3(0.0);
    vec2 uv = (p - scene.water_rip.xy) / scene.water_rip.z;
    vec2 eu = min(uv, 1.0 - uv);
    float e = smoothstep(0.0, 0.08, min(eu.x, eu.y));
    if (e <= 0.0) return vec3(0.0);
    const float tx = 1.0 / 512.0;
    float cell = scene.water_rip.z * tx;
    float hc = textureLod(WW_RIP, uv, 0.0).x;
    float hr = textureLod(WW_RIP, uv + vec2(tx, 0.0), 0.0).x, hl = textureLod(WW_RIP, uv - vec2(tx, 0.0), 0.0).x;
    float hu = textureLod(WW_RIP, uv + vec2(0.0, tx), 0.0).x, hd = textureLod(WW_RIP, uv - vec2(0.0, tx), 0.0).x;
    return vec3(vec2(hr - hl, hu - hd) / (2.0 * cell), hc) * e;
}
bool ww_ripple_covers(vec2 p) {
    if (scene.water_rip.w < 0.5) return false;
    vec2 uv = (p - scene.water_rip.xy) / scene.water_rip.z;
    return all(greaterThan(uv, vec2(0.1))) && all(lessThan(uv, vec2(0.9)));
}
#endif

// The distance to the shore (m), from the depth and the slope of the bed (the first-order
// distance to where the depth is 0: depth / |grad depth|). The reference keeps a signed distance
// field per region made by jump flooding; here the analytic ground answers "how deep" anywhere, and
// along a coast this is that distance where it matters (within a few tens of metres; further out
// it only has to be large). `bed_g`: the bed's gradient (rise per metre).
float ww_shore_dist(float depth, vec2 bed_g) {
    return depth / max(length(bed_g), 0.004);
}

#endif
