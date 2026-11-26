// The post-processing parameters and the metered exposure (vk_engine/post_fx.cpp).
// One storage buffer: the first vec4 is written by exposure.comp on the GPU, the rest by the
// CPU every frame (PostFxGpu in vk_engine/state/post_fx.h - keep the two in step).
struct PostFxData {
    float exposure;     // adapted metered exposure (multiplier before the curve; strength not applied)
    float avg;          // this frame's metered average luminance (scene units)
    float target;       // the exposure the metering aims for this frame
    float valid;        // 0 until the first frame was metered
    float focus;        // depth of field: the smoothed autofocus distance (m), GPU (exposure.comp)
    float trk_x, trk_z, trk_on;   // the tracks' window corner (world x, z), 1 = there are tracks (track_stamp.comp)
    vec4 look;          // x: AgX look on, y: the night factor n (0.5 by day .. 1 at night), z: look L per unit of avg, w: colour-blind mode (0 off, 1 protan, 2 deutan, 3 tritan)
    vec4 expo;          // x: auto-exposure strength (0 = off), y: compensation (multiplier), z: enclosed 0..1, w: metering scale (reference-units avg = avg * w)
    vec4 expo2;         // x: low percentile, y: high percentile, z: speed toward brighter exposure (1/s), w: speed toward darker
    vec4 expo3;         // x: dt (s), y: sky luminance for unfilled sky pixels, z: 1 = the sky is in the colour, w: the key (post_fx.cpp kKey)
    vec4 ltm;           // x: strength (0 off), y: shadows gain, z: highlights gain, w: sigma
    vec4 ltm2;          // x: tone mode, y: inverse level-0 size x, z: y, w: guided-fit epsilon
    vec4 glare;         // x: strength (0 off), y: threshold scale, z: max sprite radius (px), w: frame number
    vec4 glare2;        // x: output width, y: height, z: 1/w, w: 1/h
    vec4 mb;            // x: strength (0 off), y: max length (px), z: camera-rotation blur on, w: dead zone (px)
    vec4 dof;           // x: amount 0..1 (0 = off), y: max radius (px at 1080), z: focus distance (m, smoothed), w: aperture
    mat4 cur_rot_ivp;   // inverse of (projection x the camera's rotation) this frame (unjittered)
    mat4 prev_rot_vp;   // (projection x the camera's rotation) the frame before
    vec4 proj;          // x: P[3][2], y: P[2][2] of the projection (pfx_lin_depth), z: 1/render w, w: 1/render h
    // Smoke lit and fogged from the froxel volume (fx_fog.glsl): what the fog pass injects with
    // (atmosphere.cpp dispatch_fog) and the frame it was made in.
    vec4 fog0;          // rgb: the fog's sun (engine E x pi), w: its anisotropy g
    vec4 fog1;          // rgb: its ambient, w: its far end (m)
    vec4 fog2;          // xyz: the eye of that frame, w: 1 = on
    vec4 fog3;          // xyz: the camera's forward (this frame)
    mat4 fog_vp;        // that frame's view-projection
    vec4 sun_d;         // xyz: toward the sun (or the moon at night), w: 1 = it is the sun, above the horizon (glare)
};

// AgX look (the reference game): x' = pow(x * mix(g, 1, sat(x^3)), p); then saturation s about
// luma - in the sigmoid's output space, before the outset. Day: g lifts the shadows a little in
// bright scenes, saturation 1.4. Night (dark scene and night hour): deepest shadows x0.4, power
// 0.6, saturation 1.1. L = scene luminance in the reference's units (post_fx.cpp calibrates).
vec3 pfx_agx_look(vec3 x, float L, float n) {
    float a = clamp((L - 0.1) * 2.0, 0.0, 1.0), b = clamp((L - 0.6) / 9.4, 0.0, 1.0);
    float g = mix(mix(1.6 - 1.2 * n, 1.0, a), 1.2, b);
    float p = mix(mix(1.4 - 0.8 * n, 1.0, a), 1.0, b);
    float s = mix(mix(0.9 + 0.2 * n, 1.4, a), 1.4, b);
    x = max(x, vec3(0.0));
    x = pow(x * mix(vec3(g), vec3(1.0), clamp(x * x * x, 0.0, 1.0)), vec3(p));
    float y = dot(x, vec3(0.2126, 0.7152, 0.0722));
    return y + (x - y) * s;
}

// The exposure the compose pass applies before the curve: the metered one raised to the
// auto-exposure strength (0 = off), times the manual compensation.
float pfx_exposure(PostFxData d) {
    float e = (d.valid > 0.5 && d.expo.x > 0.0) ? pow(max(d.exposure, 1e-4), d.expo.x) : 1.0;
    return e * d.expo.y;
}
// Scene luminance for the AgX look, in the reference game's units (post_fx.cpp calibrates).
float pfx_look_L(PostFxData d) { return (d.valid > 0.5) ? d.avg * d.look.z : 5.0; }

// Perceptual luma of a display-linear colour: the space the exposure fusion works in.
float pfx_ltm_luma(vec3 c) { return pow(max(dot(c, vec3(0.2126, 0.7152, 0.0722)), 0.0), 1.0 / 2.2); }
// Local tone mapping gain from the guided fit ab = (a, b): fused = a * luma + b (perceptual),
// gain = fused / luma, faded to 1 in the deepest black (the reference: below 0.007), raised to
// the strength and turned back to display-linear.
vec3 pfx_ltm_apply(vec3 mapped, vec2 ab, float strength) {
    float l = pfx_ltm_luma(mapped);
    float f = max(ab.x * l + ab.y, 0.0);
    float g = f / max(l, 1e-4);
    float lo = clamp(l / 0.007, 0.0, 1.0);
    g = 1.0 + lo * lo * (g - 1.0);
    g = clamp(g, 0.25, 4.0);
    return clamp(mapped * pow(g, 2.2 * strength), 0.0, 1.0);
}

// Colour-blind view: the reference game's three matrices (protan, deutan, tritan), applied in
// display-linear (rows below are output r, g, b).
vec3 pfx_colorblind(vec3 c, int mode) {
    if (mode == 1) return vec3(0.8 * c.r + 0.2 * c.g, 0.25833 * c.r + 0.74167 * c.g, 0.14167 * c.g + 0.85833 * c.b);
    if (mode == 2) return vec3(0.81667 * c.r + 0.18333 * c.g, 0.33333 * c.r + 0.66667 * c.g, 0.125 * c.g + 0.875 * c.b);
    if (mode == 3) return vec3(0.96667 * c.r + 0.03333 * c.g, 0.73333 * c.g + 0.26667 * c.b, 0.18333 * c.g + 0.81667 * c.b);
    return c;
}

// Distance along the view axis (m) from a depth-buffer value (0 near .. 1 far); proj.x =
// P[3][2], proj.y = P[2][2] of the frame's projection (post_fx.cpp).
float pfx_lin_depth(PostFxData d, float z) { return d.proj.x / min(z + d.proj.y, -1e-7); }

// Motion blur: this pixel's blur velocity in output pixels (post_fx.cpp, mblur_*.comp).
// mv: the motion vector (current uv - previous uv, unjittered); dz: depth. With the camera's
// own turning left out (the default) the turn's part - what a point at infinity in this
// direction moved - is taken off; a pixel whose motion vector is already under the dead zone
// (the weapon, the cabin of the car one rides in: they turn with the camera) stays still.
// The sky has the turn only. Scaled by the strength, at most mb.y long.
vec2 pfx_velocity(PostFxData d, vec2 uv, vec2 mv, float dz, vec2 out_size) {
    bool sky = dz >= 0.99999;
    vec4 wf = d.cur_rot_ivp * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 pc = d.prev_rot_vp * vec4(wf.xyz / wf.w, 0.0);
    vec2 vrot = (pc.w > 1e-6) ? uv - (pc.xy / pc.w * 0.5 + 0.5) : vec2(0.0);
    vec2 v;
    if (d.mb.z > 0.5) {
        v = sky ? vrot : mv;
    } else {
        v = sky ? vec2(0.0) : mv - vrot;
        if (length(mv * out_size) < d.mb.w) v = vec2(0.0);
    }
    v *= out_size * d.mb.x;
    float l = length(v);
    return (l > d.mb.y) ? v * (d.mb.y / l) : v;
}
// The part of a velocity that blurs: the reference's 4-pixel dead zone taken off its length.
vec2 pfx_blur_vec(PostFxData d, vec2 v) {
    float l = length(v);
    return (l > 1e-4) ? v * (max(l - d.mb.w, 0.0) / l) : vec2(0.0);
}
