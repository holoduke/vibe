#version 460
// The shallow-water solver's foam particles (vk_engine/water_solver.cpp; shaders/water_solver.comp
// moves them): tiny camera-facing discs, six vertices an instance. The reference's are ~7 mm
// spheres written into its G-buffer - it is their number that makes the foam; here they are lit
// points blended into the lit picture (the water has no G-buffer pass of its own), drawn at least
// about a pixel wide with their coverage as alpha. Kept with a chance that falls from a quarter to
// half of the window from the eye.
#extension GL_GOOGLE_include_directive : require
#include "scene_ubo.glsl"

struct Part { vec4 p; vec4 v; };
layout(std430, set = 0, binding = 9) readonly buffer Parts { Part parts[]; };
layout(push_constant) uniform PC { mat4 vp; ivec4 m; ivec4 s; vec4 f; } pc;   // m.z the list, f.x radians a pixel, f.y particles a list, f.z the window (m)

layout(location = 0) out vec2 vUv;
layout(location = 1) out float vA;
layout(location = 2) out vec3 vWorld;

uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }

void main() {
    uint idx = uint(gl_InstanceIndex);
    Part q = parts[uint(pc.m.z) * uint(pc.f.y) + idx];
    vec3 c = q.p.xyz;
    float life = q.p.w;
    vec3 eye = scene.camera_pos.xyz;
    vec2 de = abs(c.xz - eye.xz);
    float L = pc.f.z;
    float keep = 1.0 - smoothstep(0.25 * L, 0.5 * L, max(de.x, de.y));
    float hsh = float(pcg(idx * 2654435761u) >> 8) * (1.0 / 16777216.0);
    const vec2 corner[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    vec2 cr = corner[gl_VertexIndex % 6];
    if (hsh > keep || life <= 0.0) { gl_Position = vec4(0.0, 0.0, -2.0, 1.0); vUv = cr; vA = 0.0; vWorld = c; return; }
    float dist = max(distance(c, eye), 0.05);
    // (the reference's radius 0.007 sqrt(sin(pi - life)): grows in, shrinks away; a little larger here -
    //  lit points, not spheres in a G-buffer)
    float r = 0.010 * sqrt(clamp(sin(3.14159265 - min(life, 2.0) * 1.5), 0.05, 1.0));
    float rpx = dist * pc.f.x * 0.7;
    float rr = max(r, rpx);
    vA = clamp((r * r) / (rr * rr), 0.08, 1.0);
    vec3 fw = (c - eye) / dist;
    vec3 rt = normalize(cross(fw, vec3(0.0, 1.0, 0.0)) + vec3(1e-5, 0.0, 0.0));
    vec3 up = cross(rt, fw);
    vec3 wp = c + (rt * cr.x + up * cr.y) * rr;
    vWorld = wp;
    vUv = cr;
    gl_Position = pc.vp * vec4(wp, 1.0);
}
