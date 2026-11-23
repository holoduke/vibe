#version 460
#extension GL_GOOGLE_include_directive : require

// The ground's shading rate, worked out from the picture itself (frame.cpp, after the terrain's
// low-resolution march): one texel of the rate image per tile of pixels (8 x 8 or 16 x 16, the
// device's choice), written every frame.
//
// A tile is shaded once per 2 x 2 pixels where ALL the ground in it is near the camera, and at
// full rate everywhere else:
//   - Far ground and every outline (a ridge against the sky, a bump in front of the valley
//     behind it) keep every pixel: that is where a coarse rate shows as steps, and a far pixel
//     is cheap anyway (no close relief, few shadow steps).
//   - Near ground is where the cost is (relief, stones, shadow detail: about three times a far
//     pixel's) and where a surface feature covers many pixels.
// The distance "near" ends at grows with the camera's speed (pc.color.z): ground that crosses
// the screen quickly is not seen sharply.
//
// The distances come from the march's picture (binding 34: one distance per block of pixels).
// It used to be a fixed pattern - coarse toward the screen's edges whatever was there, which
// put the steps on far mountains at the sides of the view.

layout(location = 0) out uint outRate;

#include "scene_ubo.glsl"

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;        // x, y: a rate texel's size in pixels; z: "near" ends here (m)
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

layout(set = 0, binding = 34) uniform sampler2D u_march_lo;

void main() {
    vec2  tile = pc.color.xy;
    float dv   = max(scene.rm_perf.y, 1.0);                 // the march's block size in pixels
    ivec2 msz  = textureSize(u_march_lo, 0);
    vec2  p0   = floor(gl_FragCoord.xy) * tile / dv;        // this tile's corner, in the march's texels
    // (with a border of one block: a tile beside an outline stays at full rate too)
    ivec2 a = max(ivec2(floor(p0)) - 1, ivec2(0));
    ivec2 b = min(ivec2(ceil(p0 + tile / dv)), msz - 1);
    float tmin = 1e9, tmax = 0.0;
    bool  unknown = false;
    for (int y = a.y; y <= b.y; ++y)
        for (int x = a.x; x <= b.x; ++x) {
            float r = texelFetch(u_march_lo, ivec2(x, y), 0).r;
            if (r <= 0.0) unknown = true;                   // (nothing known there: no march was made)
            tmin = min(tmin, r);
            tmax = max(tmax, r);
        }
    // 2 x 2 (VkFragmentShadingRate texel: log2 width << 2 | log2 height) where everything in the
    // tile is near and of one piece - the far end no more than a third beyond the near end.
    uint rate = 0u;
    if (!unknown && tmax < pc.color.z && tmax < tmin * 1.35 + 0.4) rate = 5u;
    outRate = rate;
}
