#version 460
// Screen-space displacement - the hardware path for the cells the compute rasterizer left
// (a triangle of more than 16 px^2): our code for the reference game's
// HardwareRasterizeTriangleMS. The game draws them with a mesh shader, 16 cells per group, four
// corners per cell; here the same corners are drawn as triangles from an indirect draw (six
// vertices per cell: (TL, TR, BL) and (TR, BR, BL)), with the same vertex code: the vertex
// shader's rules of ssdm_displace (nothing beyond the reach, a vertex cut at a depth jump is
// lost). As in the game the corners are always 2 px apart.
#extension GL_GOOGLE_include_directive : require
#include "ssdm_common.glsl"
#include "ssdm_displace.glsl"

layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 vN;

void main() {
    const uint kCorner[6] = uint[6](0u, 1u, 2u, 1u, 3u, 2u);
    uint cell = uint(gl_VertexIndex) / 6u, c = kCorner[uint(gl_VertexIndex) % 6u];
    uint w = ssdm_hw_idx[cell];
    vec2 vpix = vec2(float((w >> 16) + ((c << 1) & 2u)), float((w & 0xFFFFu) + ((c >> 1) << 1)));
    SsdmVtx v = ssdm_displace(vpix, true);
    gl_Position = v.clip;
    vUv = v.uv;
    vN  = vec4(v.N, v.state);
}
