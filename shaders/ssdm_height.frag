#version 460
// Screen-space displacement mapping - the height pass (fragment stage).
//
// Written per visible pixel of the displaced stonework (the reference game writes the same
// into its G-buffer: the vertex normal, and the height as a byte):
//   .rgb  the surface's own (geometric) normal, world space - the direction it is displaced in
//   .a    the height 0..1 in 1/255 steps (the target is cleared to -1: "nothing fed here");
//         ssdm_displace.glsl turns it into metres, with the fade over distance
//
// The height is the wall's own height texture, with exactly the UVs cube.frag shades the
// mesh with, so the stones that stand out are the drawn ones. Its 1 is the top of a stone,
// 0 the bottom of a joint: the drawn surface is the joints, the stones rise from it. Nothing
// is ever pushed below the drawn surface (the ray-traced shadows are cast by that surface;
// see docs/ARCHITECTURE.md).
#extension GL_GOOGLE_include_directive : require
#include "scene_ubo.glsl"

layout(set = 0, binding = 12) uniform sampler2D u_height[6];
// Rock formations take their height from the landscape's rock scan (the normal's alpha, slot 7),
// laid on by world position: rock_form.glsl. (43 = cube.frag kTextureCount - keep the same.)
layout(set = 0, binding = 4) uniform sampler2D u_normal[43];
#define RF_HEIGHT_ONLY
#include "rock_form.glsl"

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;    // x albedo index, y normal index, z uv scale
    vec4 grass_params;  // x the material's height factor (0..1: its amplitude over the pass's full amplitude), w 1 = box brush (texture laid on by world position)
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

// The height byte as the reference game's G-buffer pass writes it: the height texture's value
// times the material's factor (at most 1), rounded to 1/255.
vec4 feed(vec3 n, float h) { return vec4(n, round(clamp(h * pc.grass_params.x, 0.0, 1.0) * 255.0) / 255.0); }

void main() {
    int hix = height_idx_for_albedo(int(pc.tex_params.x + 0.5));
    int rlook = int(pc.tex_params.y - 0.5);
    if (rlook == -9 || rlook == -10) {
        // A rock formation: the scan's own height. Chalk (-10): its blocks between their
        // partings and joints (rf_chalk_h), the scan's grain on top of them.
        vec3  rn = normalize(vNormal);
        RfMat rf = rf_material(vWorldPos, rn, dFdx(vWorldPos), dFdy(vWorldPos), distance(vWorldPos, scene.camera_pos.xyz));
        float rh = rf.h;
        if (rlook == -10) {
            float fl;
            rh = 0.8 * rf_chalk_h(vWorldPos, rn, fl) + 0.2 * rf.h;
        }
        if (dot(rn, scene.camera_pos.xyz - vWorldPos) < 0.0) rn = -rn;
        outHeight = feed(rn, rh);
        return;
    }
    if (hix < 0) {
        // Other materials with a height texture: their height is in the normal map's alpha
        // (world_textures.cpp: the service station's surfaces, stretched joint 0 .. top 1),
        // laid on with the mesh's own UVs.
        vec3 n = normalize(vNormal);
        int  ni = int(pc.tex_params.y + 0.5);
        if (pc.grass_params.x > 0.0 && ni >= 0 && ni < 43) {
            float h = texture(u_normal[ni], vUv * pc.tex_params.z).a;
            if (dot(n, scene.camera_pos.xyz - vWorldPos) < 0.0) n = -n;
            outHeight = feed(n, h);
            return;
        }
        outHeight = vec4(n, 0.0);
        return;
    }
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
        if ((hix == 2 || hix == 3) && n.y < 0.9) { outHeight = vec4(n, 0.0); return; }
    } else {
        muv = vUv * pc.tex_params.z;                          // mesh part: its own UVs
    }
    if (hix == 0 || hix == 2) muv.x *= 0.5;                    // 2:1 masonry (as cube.frag)
    // (the game samples its height map with the sampler's own level of detail)
    float h = texture(u_height[hix], muv).r;
    // Our height maps are photo scans with the joints at mid grey (the reference game's are made
    // for this pass: joint 0, stone top 1). Each is stretched to that: its 5th percentile (the
    // joints) to 0, its 98th (the tops) to 1 - measured from the files (2026-11-26):
    //   Bricks078 85..204, PaintedBricks001 105..223, Tiles130 108..226, Tiles074 60..224 (of 255).
    const vec2 kLevels[4] = vec2[4](vec2(85.0, 204.0), vec2(105.0, 223.0), vec2(108.0, 226.0), vec2(60.0, 224.0));
    const vec2 lv = kLevels[hix] / 255.0;
    h = clamp((h - lv.x) / (lv.y - lv.x), 0.0, 1.0);
    if (dot(n, scene.camera_pos.xyz - vWorldPos) < 0.0) n = -n;     // (the side that is seen)
    outHeight = feed(n, h);
}
