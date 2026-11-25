#version 460
// Screen-space displacement mapping (SSDM), the way the reference game does it - the grid pass.
//
// THE METHOD. The stonework is drawn flat, like everything else. Then, before lighting:
//   1. ssdm_height   the displaced meshes once more, against the scene's depth: per visible
//                    pixel the surface normal and how far that point stands out (metres).
//   2. ssdm_solve    per pixel the vector its surface point moves by (the height, evened
//                    out with its surroundings so corners and borders stay closed).
//   3. ssdm_grid     (this) a grid with ONE VERTEX PER PIXEL is drawn over the part of the
//                    screen that holds such stonework. Each vertex is the surface point seen
//                    in its pixel, moved by its vector and projected again. The fragment
//                    stage copies the G-buffer of the pixel the point came from. The grid
//                    is depth-tested and writes depth.
// The result is a G-buffer and a depth buffer in which the stones really stand out: their
// outlines against the sky and round corners are displaced, the lighting and the shadows fall
// on the displaced surface, and whatever is drawn afterwards is tested against it.
// Where the grid leaves a pixel untouched the flat wall is still there underneath, so a
// fault shows as a flat patch, never as a hole.
//
// A vertex whose pixel is not displaced gets a not-a-number position: every triangle that
// uses it is dropped. That is how the grid ends at the border of the stonework.
//
// Drawn without buffers: instance = pixel row, 6 vertices per pixel of the row (two triangles
// between the centre of this pixel and those of its right / lower neighbours).
#extension GL_GOOGLE_include_directive : require
#include "ssdm_common.glsl"

layout(location = 0) noperspective out vec2 vSrc;     // the pixel this surface point came from
layout(location = 1) out vec4 vPrevClip;
layout(location = 2) out float vKeep;                 // the light the stones around leave this point (ssdm_solve)

void main() {
    const ivec2 kCorner[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1),
                                      ivec2(1, 0), ivec2(1, 1), ivec2(0, 1));
    ivec2 p = ivec2(pc.rect.xy) + ivec2(gl_VertexIndex / 6, gl_InstanceIndex) + kCorner[gl_VertexIndex % 6];
    vec4  o = texelFetch(u_off, p, 0);
    if (o.a <= 0.0) {
        gl_Position = vec4(uintBitsToFloat(0x7fc00000u));
        vSrc = vec2(0.0); vPrevClip = vec4(0.0); vKeep = 1.0;
        return;
    }
    float d;
    vec3  P = ssdm_rel(p, texelFetch(u_depth, p, 0).r, d) + o.xyz;
    gl_Position = pc.vp * vec4(P, 1.0);
    // Toward the camera by a hair: where a point is not moved at all the grid lies exactly in
    // the flat wall, and must win the depth test everywhere, not on every other pixel (its
    // albedo carries the shade of the stones, the flat wall's does not: stripes).
    gl_Position.z -= 4e-7 * gl_Position.w;
    vPrevClip   = pc.prev_vp * vec4(P, 1.0);
    vSrc        = vec2(p) + 0.5;
    vKeep       = o.a;
}
