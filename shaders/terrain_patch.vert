#version 460
#extension GL_GOOGLE_include_directive : require
#define VIBE_PATCH_VS 1

// The patch terrain (vk_engine/terrain_patches.cpp; docs/ARCHITECTURE.md "Patch terrain"): the
// ground as ONE grid patch drawn many times. A patch is (N + 3)^2 vertices: N quads a side, and
// an outer ring that repeats the edge and hangs down from it as a skirt - the only thing that
// hides the seam between a patch and a coarser neighbour (no morphing, no stitching).
// Per instance: the patch's corner in the world, the spacing of its vertices (the level of
// detail: the CPU's quadtree picks it), and what is the same for all of them this frame.
// A vertex's height is the terrain formula itself - the very code the ray march tests against
// (terrain_rm_heightfield.glsl rm_patch_height), which is why the files below are included.
//
// Depth only: the patches are rasterised into a depth picture of their own (reversed depth, no
// far plane), which the terrain's full-screen pass reads in place of marching.

layout(location = 0) in vec4 inPatch;   // corner x, corner z, vertex spacing (m), level
layout(location = 1) in vec4 inView;    // the camera's height above the ground under it (m); spare

// Same block as terrain_raymarch.frag (the height formula reads the plateau, the octave count
// and the distances of its level of detail from it). `model` is, for this shader, the
// projection times the view's rotation: positions are worked out relative to the camera.
layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

layout(set = 0, binding = 8)  uniform sampler2D u_terrain_height;
layout(set = 0, binding = 17) uniform sampler2D u_terrain_height_full;
const int kRmTextureCount = 48;         // = terrain_raymarch.frag
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kRmTextureCount];
layout(set = 0, binding = 4) uniform sampler2D u_normal[kRmTextureCount];

#include "terrain_rm_height_tex.glsl"
#include "scene_ubo.glsl"
#include "terrain_rm_noise_road.glsl"
#include "terrain_rm_pebbles.glsl"
#include "terrain_rm_heightfield.glsl"

const int kPatchN = 64;                 // quads a side (= TerrainPatchState::kN)

invariant gl_Position;

void main() {
    g_far_chart = scene.globe_info.x > 0.5;
    foot_bounds_init();                 // (the snow amount and the footprints: both are in the height)
    const int row = kPatchN + 3;
    ivec2 ij = ivec2(gl_VertexIndex % row, gl_VertexIndex / row) - 1;
    ivec2 g  = clamp(ij, ivec2(0), ivec2(kPatchN));
    bool skirt = any(notEqual(ij, g));
    vec2 wxz = inPatch.xy + vec2(g) * inPatch.z;
    vec3 cam = scene.camera_pos.xyz;
    vec2 dxz = wxz - cam.xz;
    // Distance from the eye, for the formula's level of detail. From the camera's height above
    // its own ground, not this vertex's height: the same number for the vertices two patches
    // share, whatever their levels.
    float t = length(vec3(dxz, inView.x));
    float h = rm_patch_height(wxz, t, true);
    if (skirt) h -= 30.0 * clamp(t / 2048.0, 0.0, 1.0) + 0.05;
    gl_Position = pc.model * vec4(dxz.x, h - cam.y, dxz.y, 1.0);
}
