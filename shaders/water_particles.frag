#version 460
// The shallow-water solver's foam particles (water_particles.vert): a lit white disc, premultiplied.
#extension GL_GOOGLE_include_directive : require
#include "scene_ubo.glsl"

layout(location = 0) in vec2 vUv;
layout(location = 1) in float vA;
layout(location = 2) in vec3 vWorld;
layout(location = 0) out vec4 outColor;

void main() {
    float r2 = dot(vUv, vUv);
    if (r2 > 1.0 || vA <= 0.0) discard;
    // (a sphere's normal from the disc, lit by the sun and the sky; albedo 0.8 - the reference's 0.8 r^2)
    vec3 n = normalize(vec3(vUv.x, sqrt(max(1.0 - r2, 0.0)), vUv.y));
    vec3 L = normalize(scene.sun_direction.xyz);
    vec3 sun = scene.sun_color.rgb * scene.sun_color.a;
    vec3 lit = vec3(0.8) * (sun * (0.35 + 0.65 * max(dot(n, L), 0.0)) * 0.6 * clamp(L.y * 3.0, 0.0, 1.0) + scene.sky_color.rgb * 0.55 + 0.01);
    float a = vA * (1.0 - smoothstep(0.6, 1.0, r2));
    outColor = vec4(lit * a, a);
}
