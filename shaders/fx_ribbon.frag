#version 460
#extension GL_GOOGLE_include_directive : require
// FX ribbons / billboards, premultiplied-alpha output (blend ONE,
// ONE_MINUS_SRC_ALPHA): alpha = 0 output is purely additive (glows).
//   pc.color.x = kind: 0 smoke trail, 1 glow beam (laser / afterburner),
//                      2 radial flash billboard
//   pc.color.y = global intensity, pc.color.z = time (s)

layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vTint;
layout(location = 2) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

#include "scene_ubo.glsl"
layout(set = 0, binding = 29) uniform sampler2D u_scene_depth;
// Light and fog from the froxel fog volume (post_fx.cpp; the setting particle_fog_light).
#include "fx_fog.glsl"

float h3(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(h3(i), h3(i + vec3(1,0,0)), f.x),
                   mix(h3(i + vec3(0,1,0)), h3(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(h3(i + vec3(0,0,1)), h3(i + vec3(1,0,1)), f.x),
                   mix(h3(i + vec3(0,1,1)), h3(i + vec3(1,1,1)), f.x), f.y), f.z);
}

void main() {
    int   kind  = int(pc.color.x + 0.5);
    float inten = pc.color.y;
    if (pc.color.w > 0.5) {
        // Mirrored draw (reflection in the water): keep it only where the
        // visible surface is the water plane, dimmed like a reflection and
        // broken up a little by the ripples.
        float zs = texelFetch(u_scene_depth, ivec2(gl_FragCoord.xy), 0).r;
        if (zs >= 0.99999) discard;
        vec2 suv = gl_FragCoord.xy * scene.viewport.zw;
        vec4 wp  = pc.prev_mvp * vec4(suv * 2.0 - 1.0, zs, 1.0);
        vec3 sp  = wp.xyz / wp.w;
        float sd = distance(sp, scene.camera_pos.xyz);
        if (abs(sp.y - scene.water_params.y) > 0.12 + 0.004 * sd) discard;
        float rip = 0.75 + 0.25 * sin(sp.x * 3.1 + pc.color.z * 1.7) * sin(sp.z * 2.7 - pc.color.z * 1.3);
        inten *= 0.55 * rip;
    }
    float time  = pc.color.z;
    if (kind == 0) {
        // Smoke / vapour trail: soft gaussian cross-section, broken up by
        // drifting noise so it reads as billowing vapour, lit by sun + sky.
        // The cross-section falls to nothing at the strip's border (it used to end there at a
        // tenth of its strength: on a trail hundreds of metres wide that drew an outline
        // against the sky). The border itself is eaten away by the noise - large billows,
        // smaller ones on them - so the edge of the smoke is a ragged cloud, not a line.
        float y2   = vUv.y * vUv.y;
        float prof = (1.0 - y2) * (1.0 - y2) * exp(-y2 * 1.1);
        vec3  q = vWorld * 0.22 + vec3(0.0, time * 0.25, time * 0.1);
        float n = vnoise(q * 0.21) * 0.45 + vnoise(q) * 0.36 + vnoise(q * 2.7) * 0.19;
        float a = vUv.x * prof * smoothstep(0.10, 0.85, n * 0.9 + 0.75 * prof);
        // Soft where it meets the ground or anything else: fades out over the last metres
        // in front of the surface behind it (a flat strip cut the terrain along a hard line).
        {
            float zs = texelFetch(u_scene_depth, ivec2(gl_FragCoord.xy), 0).r;
            float d_scene = 0.05 / max(1.0 - zs, 1e-7), d_here = 0.05 / max(1.0 - gl_FragCoord.z, 1e-7);
            a *= smoothstep(0.0, 3.0 + 0.03 * d_here, d_scene - d_here);
        }
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        // (the sun's visibility and the lamps' light from the fog volume; its own fog: fx_fog.glsl)
        float fvis; vec3 flocal;
        vec3 rdv = normalize(vWorld - scene.camera_pos.xyz);
        fxf_light(vWorld, rdv, fvis, flocal);
        vec3 lit = vTint * (sun * 0.75 * fvis + scene.sky_color.rgb * 0.55 + 0.05 + flocal * 0.75);
        a = clamp(a * inten, 0.0, 1.0);
        outColor = vec4(fxf_fog(lit * a, a, distance(vWorld, scene.camera_pos.xyz)), a);
    } else if (kind == 1) {
        // Glow beam: white-hot core + coloured halo, additive.
        float v = vUv.y;
        float core = exp(-v * v * 26.0);
        float halo = exp(-v * v * 3.0);
        // Strongly saturated tints (red AA tracers) keep a coloured core.
        vec3  corec = (vTint.g < 0.1) ? vec3(1.0, 0.35, 0.25) : vec3(1.0, 0.95, 1.0);
        vec3  c = vTint * (halo * 1.4) + corec * core * 3.2;
        outColor = vec4(c * vUv.x * inten, 0.0);
    } else if (kind == 6) {
        // Tyre mark (jeep.cpp): rubber laid on the road - a dark band lying on the ground,
        // soft at its edges, streaked along its length. uv = (strength, across).
        float edge = 1.0 - smoothstep(0.55, 1.0, abs(vUv.y));
        float streak = 0.70 + 0.30 * vnoise(vec3(vWorld.x * 7.0, vUv.y * 5.0, vWorld.z * 7.0));
        float a = clamp(vUv.x * edge * streak * inten, 0.0, 0.92);
        outColor = vec4(vec3(0.010, 0.010, 0.011) * a, a);
    } else if (kind == 4) {

        // Water droplet streak: a bright bead (droplets act as tiny lenses:
        // they show the sky and a sun glint), soft across, tapering back.
        float a = vUv.x * exp(-vUv.y * vUv.y * 3.5);
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        vec3 lit = sun * 0.30 + scene.sky_color.rgb * 0.55 + 0.02;
        a = clamp(a * inten, 0.0, 1.0) * 0.8;
        outColor = vec4(lit * a, a);
    } else if (kind == 5) {
        // Spray mist puff (uv in [-1,1]^2): soft ball eroded by noise so the
        // edge is ragged; forward-scattering white, lit by sun + sky.
        float r2 = dot(vUv, vUv);
        if (r2 > 1.0) discard;
        vec3  q = vWorld * 2.6 + vec3(vTint.z, -time * 1.3, vTint.z * 0.37);
        float n = vnoise(q) * 0.6 + vnoise(q * 2.9) * 0.4;
        float a = vTint.x * exp(-r2 * 2.4) * smoothstep(0.15, 0.7, n + 0.55 * (1.0 - r2));
        a *= 1.0 - smoothstep(0.7, 1.0, r2);
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        float fvis; vec3 flocal;
        fxf_light(vWorld, normalize(vWorld - scene.camera_pos.xyz), fvis, flocal);
        vec3 lit = vec3(0.95, 0.97, 1.0) * (sun * 0.20 * fvis + scene.sky_color.rgb * 0.42 + 0.02 + flocal * 0.2) * vTint.y;
        a = clamp(a * inten, 0.0, 1.0);
        outColor = vec4(fxf_fog(lit * a, a, distance(vWorld, scene.camera_pos.xyz)), a);
    } else if (kind == 8) {
        // Spray mist over a fall, a rock, a breaking crest (water_spray.cpp; uv in [-1,1]^2, tint =
        // alpha, brightness, seed): a soft thin sheet of droplets. It scatters the sun forward -
        // looking toward the sun through it, it glows (a Henyey-Greenstein lobe, g 0.6, with a
        // little back-scatter) - and the sky's light all round.
        float r2 = dot(vUv, vUv);
        if (r2 > 1.0) discard;
        vec3  q = vWorld * 1.7 + vec3(vTint.z, -time * 0.9, vTint.z * 0.61);
        float n = vnoise(q) * 0.55 + vnoise(q * 2.7) * 0.30 + vnoise(q * 6.3) * 0.15;
        float a = vTint.x * exp(-r2 * 2.0) * smoothstep(0.25, 0.75, n + 0.45 * (1.0 - r2));
        a *= 1.0 - smoothstep(0.75, 1.0, r2);
        vec3  rdv = normalize(vWorld - scene.camera_pos.xyz);
        float mu = dot(rdv, normalize(scene.sun_direction.xyz));
        float g = 0.6;
        float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5) * 0.25 + 0.12;
        float fvis; vec3 flocal;
        fxf_light(vWorld, rdv, fvis, flocal);
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        vec3 lit = vec3(0.96, 0.98, 1.0) * (sun * hg * fvis + scene.sky_color.rgb * 0.40 + flocal * 0.2) * vTint.y;
        a = clamp(a * inten, 0.0, 1.0);
        outColor = vec4(fxf_fog(lit * a, a, distance(vWorld, scene.camera_pos.xyz)), a);
    } else if (kind == 7) {
        // Blood (npc_world.cpp; uv in [-1,1]^2): a drop, a stain on the ground or a thin red
        // mist - a soft disc with a ragged edge. vTint = (alpha, brightness, seed).
        float r2 = dot(vUv, vUv);
        if (r2 > 1.0) discard;
        float n = vnoise(vec3(vUv * 2.3, vTint.z)) * 0.6 + vnoise(vec3(vUv * 6.1, vTint.z + 3.0)) * 0.4;
        float a = vTint.x * (1.0 - smoothstep(0.30, 1.0, r2 + (n - 0.5) * 0.7));
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        vec3 lit = vec3(0.34, 0.008, 0.006) * (sun * 0.55 + scene.sky_color.rgb * 0.45 + 0.03) * vTint.y;
        a = clamp(a * inten, 0.0, 1.0);
        outColor = vec4(lit * a, a);
    } else if (kind == 3) {
        // Cloth flag: flat normal from screen derivatives, two-sided
        // Lambert + sun shining THROUGH the cloth when back-lit (thin
        // fabric translucency) + sky fill. Procedural heraldry.
        vec3 Nf = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        vec3 V  = normalize(scene.camera_pos.xyz - vWorld);
        if (dot(Nf, V) < 0.0) Nf = -Nf;
        vec3 L  = normalize(scene.sun_direction.xyz);
        float ndl = dot(Nf, L);
        vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
        int design = int(vTint.x + 0.5);
        vec2 uv = vUv;
        vec3 base, accent;
        if (design == 0) { base = vec3(0.42, 0.03, 0.03); accent = vec3(0.85, 0.62, 0.12); }
        else             { base = vec3(0.04, 0.10, 0.36); accent = vec3(0.86, 0.84, 0.78); }
        vec3 alb = base;
        if (design == 2) {
            // Parachute canopy: alternating orange / off-white gores, darker
            // seams along the radial tapes and the hem (uv.x = gore, uv.y = crown->skirt).
            float gore = fract(uv.x * 8.0);
            alb = mix(vec3(0.80, 0.78, 0.72), vec3(0.85, 0.30, 0.05), step(0.5, gore));
            float seam = 1.0 - smoothstep(0.0, 0.035, min(fract(uv.x * 16.0), 1.0 - fract(uv.x * 16.0)));
            alb *= 1.0 - 0.35 * seam;
            alb *= 1.0 - 0.3 * smoothstep(0.94, 1.0, uv.y);
            alb *= 1.0 - 0.35 * (1.0 - smoothstep(0.0, 0.03, uv.y));      // vent band
            // Inside of the canopy (seen from below): shaded, cloth lit from behind.
            if (!gl_FrontFacing) alb *= 0.62;
            base = alb;
        } else if (design == 3) {
            alb = vec3(0.10, 0.09, 0.08);          // suspension line
        } else if (design == 0) {
            // Gold cross (Nordic offset) + fly-end swallowtail fringe.
            float cx = step(abs(uv.x - 0.36), 0.07), cy = step(abs(uv.y - 0.5), 0.09);
            alb = mix(alb, accent, max(cx, cy));
        } else {
            // Bend + diamond charge.
            float bend = step(abs((uv.x - uv.y * 1.6) + 0.3), 0.12);
            float dia  = step(abs(uv.x - 0.6) * 1.6 + abs(uv.y - 0.5), 0.22);
            alb = mix(alb, accent, max(bend, dia));
        }
        // Woven-fabric grain + soft dirt toward the fly end.
        float weave = 0.9 + 0.1 * sin(uv.x * 420.0) * sin(uv.y * 260.0);
        alb *= weave * (design >= 2 ? 1.0 : mix(1.0, 0.8, uv.x * uv.x));
        vec3 lit = alb * (sun * max(ndl, 0.0) * 1.05 +
                          sun * max(-ndl, 0.0) * 0.45 * (0.6 + 0.4 * base) +   // translucency
                          scene.sky_color.rgb * 0.45 + 0.03);
        outColor = vec4(lit, 1.0);
    } else {
        // Radial flash billboard (uv in [-1,1]^2), additive.
        float r2 = dot(vUv, vUv);
        float f = (exp(-r2 * 5.0) + 0.35 * exp(-r2 * 1.4)) * (1.0 - smoothstep(0.55, 1.0, sqrt(r2)));
        outColor = vec4(vTint * f * inten, 0.0);
    }
}
