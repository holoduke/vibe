#version 460
// Glass (fx.cpp render_fx, "glass"): window panes and windscreens, drawn after the lighting.
//
// What is seen THROUGH the pane is the lit scene behind it, read from a copy of the frame made
// just before (u_scene_copy) - bent as the pane bends it (the line of sight is displaced
// sideways, more the flatter one looks along the glass), tinted by the glass it passed
// through. What is seen ON the pane is the sky and the ground mirrored in it, little head-on
// (4 %) and nearly all at a grazing angle (Fresnel), and the sun's glint.
//
// (It was a pattern of opaque dots - cube.frag - which neither upscaler blended: the panes
// boiled, and the monitors behind them flickered.)

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;      // here: the view-projection (world -> clip)
    vec4 color;         // rgb: the glass's own tint (dark = clear), a: unused
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

#include "scene_ubo.glsl"

layout(set = 0, binding = 29) uniform sampler2D u_scene_depth;
layout(set = 0, binding = 38) uniform sampler2D u_scene_copy;

void main() {
    vec2 uv = gl_FragCoord.xy * scene.viewport.zw;
    vec3 V = normalize(scene.camera_pos.xyz - vWorld);
    vec3 N = normalize(vNormal);
    if (dot(N, V) < 0.0) N = -N;
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float dist = distance(scene.camera_pos.xyz, vWorld);

    // ---- through it ----
    // The ray is bent on the way in and back on the way out: it leaves parallel to how it
    // came, moved sideways by the pane's thickness x a factor that grows with the angle. The
    // scene behind is therefore seen from a point shifted along the pane - drawn here as the
    // background looked up a little to the side. (Stronger than a 6 mm pane would give: glass
    // that bends nothing reads as a hole.)
    vec3 r = refract(-V, N, 1.0 / 1.52);
    vec3 side = (r + V);                                   // the sideways part of the bend
    vec4 cl = pc.prev_mvp * vec4(vWorld + side * (0.10 + 0.012 * min(dist, 30.0)), 1.0);
    vec2 uv_r = cl.w > 1e-4 ? (cl.xy / cl.w) * 0.5 + 0.5 : uv;
    uv_r = clamp(uv_r, vec2(0.001), vec2(0.999));
    // (not what stands in FRONT of the pane: there the background is taken straight)
    if (textureLod(u_scene_depth, uv_r, 0.0).r < gl_FragCoord.z - 1e-6) uv_r = uv;
    vec4 behind = textureLod(u_scene_copy, uv_r, 0.0);
    // Soda-lime glass is faintly green, more so the longer the path through it.
    vec3 absorb = mix(vec3(0.93, 0.965, 0.95), vec3(0.74, 0.87, 0.83), pow(1.0 - ndv, 1.5));
    vec3 col = behind.rgb * absorb;

    // ---- on it ----
    float fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    vec3 R = reflect(-V, N);
    vec3 sun_dir = normalize(scene.sun_direction.xyz);
    vec3 sky = scene.sky_color.rgb;
    vec3 env = R.y > 0.0 ? sky * mix(0.75, 1.15, smoothstep(0.0, 0.6, R.y))
                         : mix(sky * 0.55, scene.ambient.rgb * 0.5 + sky * 0.18, smoothstep(0.0, 0.35, -R.y));
    vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
    float sd = max(dot(R, sun_dir), 0.0);
    env += sun * (pow(sd, 900.0) * 60.0 + pow(sd, 24.0) * 0.12) * step(0.0, sun_dir.y);
    col = mix(col, env, fres);
    // A window is never spotless: a breath of dust, seen where the light rakes it.
    float dust = fract(sin(dot(floor(vUv * 37.0), vec2(12.9898, 78.233))) * 43758.5453);
    col += sky * 0.012 * dust * (0.4 + 0.6 * (1.0 - ndv));
    outColor = vec4(col, behind.a);
}
