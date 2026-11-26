#version 460
// Screen-space displacement - the hardware path's pixels: our code for the reference game's
// HardwareRasterizeTrianglePS. The same 64-bit word as the compute rasterizer
// (ssdm_raster.comp), bit 0 set, kept by atomic max. No colour, no depth.
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
#define SSDM_RASTER_BUF
#include "ssdm_common.glsl"

layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vN;

uint ssdm_sm(int a) { return (uint(abs(a)) & 255u) | ((uint(a) >> 31) << 8); }

void main() {
    int dx = int(pc.size.x * vUv.x) - int(gl_FragCoord.x);
    int dy = int(pc.size.y * vUv.y) - int(gl_FragCoord.y);
    vec3 n = vN.xyz;
    vec2 o = n.xy / (abs(n.y) + abs(n.x) + abs(n.z));
    if (!(n.z >= 0.0)) o = (1.0 - abs(o.yx)) * vec2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
    uint ox = uint(roundEven(o.x * 255.5 + 255.5)), oy = uint(roundEven(o.y * 255.5 + 255.5));
    uint offs = (ssdm_sm(dy) | (ssdm_sm(dx) << 9)) ^ 131328u;
    float zf = 1.0 - gl_FragCoord.z;
    uint64_t key = (uint64_t(floatBitsToUint(zf) & 0xFFFFFF00u) << 32)
                 | (uint64_t(offs) << 20)
                 | (uint64_t(((oy << 9) | ox) & 0x3FFFFu) << 2)
                 | 274877906945UL;                                   // 1 << 38 | 1: background, hardware
    ivec2 p = ivec2(gl_FragCoord.xy);
    atomicMax(ssdm_r[uint(p.y) * uint(pc.size.x) + uint(p.x)], key);
}
