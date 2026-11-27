#version 460
// Glass (fx.cpp render_fx, "glass"; glass.cpp): window panes and windscreens, the pieces of a
// broken pane, and the lenses of the weapons' sights - drawn after the lighting.
//
// What is seen THROUGH the pane is the lit scene behind it, read from a copy of the frame made
// just before (u_scene_copy) - bent as the pane bends it (the line of sight is displaced
// sideways, more the flatter one looks along the glass), tinted by the glass it passed
// through. What is seen ON the pane is the sky and the ground mirrored in it, little head-on
// (4 %) and nearly all at a grazing angle (Fresnel), and the sun's glint.
//
// (It was a pattern of opaque dots - cube.frag - which neither upscaler blended: the panes
// boiled, and the monitors behind them flickered.)
//
// pc.emissive.w says what is drawn:
//   0  a pane (a static-mesh part with material -3)
//   1  a shard: a flying piece of a broken pane (uv = its corners (0,0) (1,0) (0,1); color.a = how much is left of it)
//   2  a lens of a sight on a held weapon. Positions are RELATIVE TO THE EYE here (the weapon is
//      drawn that way: props.cpp draw_viewmodel), prev_mvp takes them to clip space.
//      emissive.xyz = the lens's centre, tex_params = its axis (xyz) and radius (w),
//      grass_params = (aiming 0..1, magnification when aiming, reticle 0/1, 0)
// A cracked or broken pane has its own shader: glass_pane.frag.

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;      // here: the view-projection (world -> clip)
    vec4 color;         // rgb: the glass's own tint (dark = clear), a: a shard's size left
    vec4 emissive;      // w: what is drawn (above)
    vec4 tex_params;
    vec4 grass_params;
} pc;

#include "scene_ubo.glsl"

layout(set = 0, binding = 29) uniform sampler2D u_scene_depth;
layout(set = 0, binding = 38) uniform sampler2D u_scene_copy;

#include "glass_common.glsl"

// A sight's lens: clear, coated glass in a dark housing. The scene is seen through it (a little
// larger while aiming), a coloured sheen and the sun's glint lie on its curve toward the rim.
vec4 lens(vec2 uv) {
    vec3 P = vWorld;                                   // relative to the eye
    float dist = length(P);
    vec3 V = -P / max(dist, 1e-5);
    vec3 ax = normalize(pc.tex_params.xyz);
    // (the lens faces along its axis: the model's own normals are smoothed over the disc's edge)
    vec3 N = dot(ax, V) < 0.0 ? -ax : ax;
    float R = max(pc.tex_params.w, 1e-4);
    vec3 c = pc.emissive.xyz;
    vec3 d = P - c; d -= ax * dot(d, ax);              // from the axis, in the lens's plane
    float rr = length(d) / R;                          // 0 centre .. 1 rim
    float aim = clamp(pc.grass_params.x, 0.0, 1.0);

    // through it: about the lens's own centre on the screen, enlarged while aiming, and pulled
    // in a little more toward the rim (the curve of the glass)
    vec4 cc = pc.prev_mvp * vec4(c, 1.0);
    vec2 uv_c = cc.w > 1e-4 ? (cc.xy / cc.w) * 0.5 + 0.5 : uv;
    float mag = mix(1.0, max(pc.grass_params.y, 1.0), aim);
    vec2 uv_s = uv_c + (uv - uv_c) / mag * (1.0 - 0.07 * rr * rr);
    uv_s = clamp(uv_s, vec2(0.001), vec2(0.999));
    if (textureLod(u_scene_depth, uv_s, 0.0).r < gl_FragCoord.z - 1e-6) uv_s = uv;     // (not the gun's own parts in front of it)
    vec4 behind = textureLod(u_scene_copy, uv_s, 0.0);
    vec3 col = behind.rgb * mix(vec3(0.93, 0.97, 0.95), vec3(0.80, 0.90, 0.88), rr * rr);       // a subtle tint, deeper where the glass is thick

    // on it: the lens bulges - its normal leans outward toward the rim, so the sky's sheen
    // gathers there (and the coating colours it)
    vec3 radial = length(d) > 1e-6 ? d / length(d) : vec3(0.0);
    vec3 Nl = normalize(N + radial * (0.55 * rr * rr));
    float ndv = clamp(dot(Nl, V), 0.0, 1.0);
    float fres = 0.02 + 0.75 * pow(1.0 - ndv, 4.0);
    vec3 Rl = reflect(-V, Nl);
    vec3 coat = mix(vec3(0.70, 0.95, 1.25), vec3(1.15, 0.80, 1.20), smoothstep(0.3, 1.0, rr));   // blue-green to violet: an anti-reflection coating
    vec3 sheen = glass_mirror(Rl) * coat;
    col = mix(col, sheen, clamp(fres + 0.07 * smoothstep(0.60, 0.95, rr), 0.0, 0.6));
    col += glass_glint(Rl, 260.0, 14.0, 20.0, 0.10);

    // the reticle, etched in the glass and lit: a fine cross open in the middle, a dot, ticks
    if (pc.grass_params.z > 0.5) {
        vec3 e1 = cross(ax, vec3(0.0, 1.0, 0.0));
        e1 = dot(e1, e1) > 1e-4 ? normalize(e1) : vec3(1.0, 0.0, 0.0);
        vec3 e2 = cross(ax, e1);
        vec2 q = vec2(dot(d, e1), dot(d, e2)) / R;
        float px = max(length(dFdx(q)), length(dFdy(q)));
        float w = max(0.010, px * 0.9);
        vec2 aq = abs(q);
        float ln = 0.0;
        ln = max(ln, (1.0 - smoothstep(w * 0.5, w, aq.x)) * step(0.14, aq.y) * step(aq.y, 0.80));
        ln = max(ln, (1.0 - smoothstep(w * 0.5, w, aq.y)) * step(0.14, aq.x) * step(aq.x, 0.80));
        ln = max(ln, 1.0 - smoothstep(0.022, 0.022 + max(w, 0.012), length(q)));
        for (int k = 1; k <= 2; ++k) {                 // range ticks under the centre
            float y = -0.24 * float(k);
            ln = max(ln, (1.0 - smoothstep(w * 0.5, w, abs(q.y - y))) * step(aq.x, 0.10 - 0.03 * float(k)));
        }
        vec3 glow = max(pc.color.rgb, vec3(0.02)) / max(max(pc.color.r, pc.color.g), max(pc.color.b, 0.02));
        col = mix(col, glow * 2.6, ln * (0.25 + 0.75 * aim) * (1.0 - smoothstep(0.80, 0.86, rr)));
    }

    // the housing: the tube's shadow inside the rim, and a thin dark edge
    col *= 1.0 - 0.32 * smoothstep(0.70, 0.94, rr);
    col = mix(col, vec3(0.004, 0.004, 0.005) + sheen * 0.02, smoothstep(0.915, 0.97, rr));
    return vec4(col, behind.a);
}

void main() {
    vec2 uv = gl_FragCoord.xy * scene.viewport.zw;
    if (pc.emissive.w > 1.5) { outColor = lens(uv); return; }
    bool shard = pc.emissive.w > 0.5;

    vec3 V = normalize(scene.camera_pos.xyz - vWorld);
    vec3 N = normalize(vNormal);
    if (dot(N, V) < 0.0) N = -N;
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float dist = distance(scene.camera_pos.xyz, vWorld);

    // ---- through it ----
    vec4 behind = glass_through(vWorld, N, V, pc.prev_mvp, uv, gl_FragCoord.z, dist, vec3(0.0));
    vec3 col = behind.rgb * glass_absorb(ndv);

    // ---- on it ----
    float fres = glass_fresnel(ndv);
    vec3 R = reflect(-V, N);
    vec3 sky = scene.sky_color.rgb;
    vec3 env = glass_mirror(R);
    if (shard) {
        // A tumbling piece flashes as it turns through the sun, and its broken edge is pale green.
        env += glass_glint(R, 220.0, 26.0, 12.0, 0.35);
        col *= vec3(0.80, 0.93, 0.88);                         // (a piece is seen edge-on as often as not: greener than a pane)
        col = mix(col, env, clamp(fres + 0.14, 0.0, 1.0));
        float e = min(min(vUv.x, vUv.y), 1.0 - vUv.x - vUv.y);
        float edge = 1.0 - smoothstep(0.0, 0.085, e);
        col = mix(col, (sky * 0.75 + glass_sunlight() * 0.22) * vec3(0.80, 1.0, 0.93), edge * 0.7);
        outColor = vec4(col, behind.a);
        return;
    }
    // (the scene mirrored where it is on the screen; else the room - eye indoors, grass_params.x -
    //  or the sky, the ground and the sun: glass_common.glsl glass_reflection)
    env = glass_reflection(vWorld, R, pc.prev_mvp, uv, pc.grass_params.x);
    col = mix(col, env, fres);
    // A window is never spotless: a breath of dust, seen where the light rakes it.
    float dust = fract(sin(dot(floor(vUv * 37.0), vec2(12.9898, 78.233))) * 43758.5453);
    col += sky * 0.012 * dust * (0.4 + 0.6 * (1.0 - ndv));
    outColor = vec4(col, behind.a);
}
