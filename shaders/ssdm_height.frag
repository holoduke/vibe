#version 460
// Screen-space displacement mapping - the height pass (fragment stage).
//
// Written per visible pixel of the displaced stonework, for ssdm_solve.frag:
//   .rgb  the surface's own (geometric) normal, world space - the direction it is displaced in
//   .a    how far this point stands out from the drawn surface, in metres, plus a hair
//         (so .a > 0 marks "this pixel can be displaced", also in the joints where the
//         height is 0); 0 = nothing to displace here
//
// The height is the wall's own height texture, with exactly the UVs cube.frag shades the
// mesh with, so the stones that stand out are the drawn ones. Its 1 is the top of a stone,
// 0 the bottom of a joint: the drawn surface is the joints, the stones rise from it. Nothing
// is ever pushed below the drawn surface (the ray-traced shadows are cast by that surface;
// see docs/ARCHITECTURE.md).
#extension GL_GOOGLE_include_directive : require
#include "scene_ubo.glsl"

layout(set = 0, binding = 12) uniform sampler2D u_height[6];

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;    // x albedo index, y normal index, z uv scale
    vec4 grass_params;  // x displacement amplitude (m), y fade start (m), z fade end (m), w 1 = box brush (texture laid on by world position)
} pc;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec2 vUv;

layout(location = 0) out vec4 outHeight;

int height_idx_for_albedo(int a) {         // (cube.frag)
    if (a == 1) return 0;
    if (a == 4) return 1;
    if (a == 5) return 2;
    if (a == 6) return 3;
    return -1;
}

void main() {
    float fade = 1.0 - smoothstep(pc.grass_params.y, pc.grass_params.z,
                                  distance(vWorldPos, scene.camera_pos.xyz));
    int hix = height_idx_for_albedo(int(pc.tex_params.x + 0.5));
    if (fade <= 0.0 || hix < 0) { outHeight = vec4(0.0); return; }
    vec3  n = normalize(vNormal);
    vec2  muv;
    if (pc.grass_params.w > 0.5) {
        // Box brush: the texture is laid on by world position along the face's axis
        // (cube_surface_helpers.glsl spom_uv - keep the two the same).
        vec3 wp = vWorldPos * pc.tex_params.z, an = abs(n);
        if (an.x >= an.y && an.x >= an.z) muv = vec2(wp.z * sign(n.x), wp.y);
        else if (an.y >= an.z)            muv = vec2(wp.x, wp.z * sign(n.y));
        else                              muv = vec2(wp.x * sign(n.z), wp.y);
        // (floor tiles only have relief on their top: cube.frag spom_path)
        if ((hix == 2 || hix == 3) && n.y < 0.9) { outHeight = vec4(0.0); return; }
    } else {
        muv = vUv * pc.tex_params.z;                          // mesh part: its own UVs
    }
    if (hix == 0 || hix == 2) muv.x *= 0.5;                    // 2:1 masonry (as cube.frag)
    // The mip level of the LONG side of the pixel's footprint (the sampler's own choice follows
    // the short side): seen at a slant, neighbouring pixels are far apart on the wall, and
    // heights that jump from one pixel to the next comb the grid into stripes. Blurred like
    // this the stones still stand out of the outline, as one smooth swell each.
    vec2  tsz = vec2(textureSize(u_height[hix], 0));
    float fp  = max(length(dFdx(muv) * tsz), length(dFdy(muv) * tsz));
    float h   = textureLod(u_height[hix], muv, log2(max(fp, 1.0)) + 0.5).r;
    if (dot(n, scene.camera_pos.xyz - vWorldPos) < 0.0) n = -n;     // (the side that is seen)
    outHeight = vec4(n, h * pc.grass_params.x * fade + 1e-4);
}
