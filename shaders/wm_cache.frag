#version 460
#extension GL_GOOGLE_include_directive : require

// The world's slow fields round the camera, drawn into a texture (vk_engine/wm_cache.cpp):
// .r = land (> 0) or sea (< 0), .g = where the ranges stand - for every 4 m of the chart,
// by the terrain shader's own functions (rm_world_vr_f), so what is read back from the
// texture is what the formula gives there.
//   pc.emissive.xy  the chart position of texel (0, 0)'s centre
//   pc.emissive.z   metres a texel

layout(location = 0) out vec2 outVR;

#include "scene_ubo.glsl"

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

const int kRmTextureCount = 37;   // = cube.frag kTextureCount
layout(set = 0, binding = 4) uniform sampler2D u_normal[kRmTextureCount];

#define WM_CACHE_BAKE
#include "terrain_rm_noise_road.glsl"

void main() {
    vec2 wp = pc.emissive.xy + (gl_FragCoord.xy - 0.5) * pc.emissive.z;
    outVR = rm_world_vr_f(wp);
}
