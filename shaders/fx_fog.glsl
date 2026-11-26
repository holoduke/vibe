// Smoke and dust lit and fogged from the froxel fog volume (post_fx.cpp; docs/ARCHITECTURE.md
// "Post effects"), after the reference game, whose particles take their light from a froxel
// light volume and are fogged like everything else. Included by fx_volume.frag / fx_ribbon.frag
// after scene_ubo.glsl. The FX are drawn before this frame's fog pass, so they read the volume
// the frame before made (its view-projection is in pfx.d.fog_vp).
//
//   light: the injected volume holds, per froxel, (in-scatter, extinction). In-scatter over
//   extinction is the light there: sun x visibility (the fog's own shadow rays through trees and
//   buildings, the cloud shadow) x phase + sky ambient + the local lights (lamps, car beams).
//   Taking off the ambient and dividing by the unshadowed sun gives the sun's visibility at the
//   particle; what is left over is the lamps' light.
//   fog: compose.frag fogs every pixel at the depth of the SURFACE behind it - smoke in front
//   of a far hill was fogged as if it stood at the hill. The smoke's colour is corrected so that,
//   after compose's fog, it carries the fog of its own distance:
//     c' = (c T_p + a (L_p - L_bg)) / T_bg.
// Off (pfx.d.fog2.w = 0: the setting, or no fog volume): the old lighting, nothing fogged here.

#include "post_fx.glsl"
layout(set = 0, binding = 56) uniform sampler3D u_pfx_integ;   // (L, T) from the eye to each slice
layout(set = 0, binding = 57) uniform sampler3D u_pfx_inj0;    // the two injected volumes (ping-pong;
layout(set = 0, binding = 58) uniform sampler3D u_pfx_inj1;    //  one is last frame's): averaged
layout(std430, set = 0, binding = 59) readonly buffer PfxFxB { PostFxData d; } pfx_fx;

bool fxf_on() { return pfx_fx.d.fog2.w > 0.5; }

// Froxel texture coordinate of a world point for the injected volume (slice centres), and the
// eye distance.
vec3 fxf_coord(vec3 p, out float dist) {
    vec4 c = pfx_fx.d.fog_vp * vec4(p, 1.0);
    vec2 uv = (c.w > 1e-4) ? clamp(c.xy / c.w * 0.5 + 0.5, vec2(0.0), vec2(1.0)) : vec2(0.5);
    dist = distance(p, pfx_fx.d.fog2.xyz);
    float s = sqrt(clamp(dist / max(pfx_fx.d.fog1.w, 1.0), 0.0, 1.0));
    return vec3(uv, s);
}
// The integrated fog (in-scatter L, transmittance T) from the eye to distance dist along screen uv.
vec4 fxf_integ(vec2 uv, float dist) {
    float s = sqrt(clamp(dist / max(pfx_fx.d.fog1.w, 1.0), 0.0, 1.0)) * 64.0 - 1.0;   // (compose.frag fog_at)
    vec4 f = textureLod(u_pfx_integ, vec3(uv, (max(s, 0.0) + 0.5) / 64.0), 0.0);
    return (s < 0.0) ? mix(vec4(0.0, 0.0, 0.0, 1.0), f, clamp(s + 1.0, 0.0, 1.0)) : f;
}

// The light at world point p for a view ray rd: the sun's visibility there, and the local
// lights' light (engine units, as scene.sun_color x intensity is).
void fxf_light(vec3 p, vec3 rd, out float vis, out vec3 local) {
    vis = 1.0; local = vec3(0.0);
    if (!fxf_on()) return;
    float dist;
    vec3 fc = fxf_coord(p, dist);
    if (dist > pfx_fx.d.fog1.w * 0.95) return;
    vec4 a = textureLod(u_pfx_inj0, fc, 0.0), b = textureLod(u_pfx_inj1, fc, 0.0);
    vec4 v = (a + b) * 0.5;
    if (!(v.a > 1e-6)) return;                       // (no fog there to carry the light)
    vec3 X = v.rgb / v.a;
    float g = pfx_fx.d.fog0.w;
    float c = dot(rd, normalize(scene.sun_direction.xyz));
    float ph = (1.0 - g * g) / (12.566371 * pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5));
    vec3 Sp = pfx_fx.d.fog0.rgb * ph;               // the unshadowed sun, as the fog injects it
    vec3 A = pfx_fx.d.fog1.rgb * (4.0 / 12.566371); // its ambient term
    const vec3 kL = vec3(0.2126, 0.7152, 0.0722);
    vis = clamp(dot(X - A, kL) / max(dot(Sp, kL), 1e-6), 0.0, 1.0);
    local = max(X - A - Sp * vis, vec3(0.0)) * 4.0;   // (x 4 pi / pi: isotropic, engine units)
    // (fade out toward the far end of the volume)
    float f = smoothstep(0.95, 0.75, dist / max(pfx_fx.d.fog1.w, 1.0));
    vis = mix(1.0, vis, f);
    local *= f;
}

// Premultiplied FX colour c with coverage a at world point p (eye distance d_p) over the pixel's
// surface: corrected so compose's fog at the surface's depth leaves the FX with its own fog.
vec3 fxf_fog(vec3 c, float a, float d_p) {
    if (!fxf_on()) return c;
    vec2 uv = gl_FragCoord.xy * scene.viewport.zw;
    float zs = texelFetch(u_scene_depth, ivec2(gl_FragCoord.xy), 0).r;
    float maxd = pfx_fx.d.fog1.w;
    float d_bg = maxd;
    if (zs < 0.99999) {
        // (eye distance from the depth: view-axis distance over the ray's cosine to the axis)
        vec4 wf = pfx_fx.d.cur_rot_ivp * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
        vec3 rdir = normalize(wf.xyz / wf.w);
        d_bg = pfx_lin_depth(pfx_fx.d, zs) / max(dot(rdir, pfx_fx.d.fog3.xyz), 0.05);
    }
    d_bg = min(d_bg, maxd);
    if (d_p >= d_bg) return c;
    vec4 fp = fxf_integ(uv, d_p), fb = fxf_integ(uv, d_bg);
    float tb = max(fb.a, 0.05);
    return max((c * fp.a + a * (fp.rgb - fb.rgb)) / tb, vec3(0.0));
}
