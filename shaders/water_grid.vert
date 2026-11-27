#version 460
#extension GL_GOOGLE_include_directive : require
#define VIBE_PATCH_VS 1

// The open sea as a CAMERA-PROJECTED GRID (vk_engine/water_grid.cpp; after the reference game's
// sea: a grid laid out on the screen, each vertex's view ray dropped onto the sea plane, then
// displaced by the waves). Depth only, into a depth picture of its own (reversed depth, near plane
// kWgNear, no far plane - the patch terrain's technique, terrain_patch.vert); the terrain's
// full-screen pass reads the sea's hit from it instead of searching along the ray
// (terrain_raymarch.frag, the sea's hit).
//
// No vertex buffer: vertex i is corner (i % 6) of quad (i / 6) of a cols x rows grid that covers
// the screen plus a border of cells beyond it. The border's cells are STRETCHED (x 10 at the sides,
// x 20 at the bottom; not at the horizon) so that a vertex pushed sideways by a crest never opens
// a gap at the frame's edge. A ray at or above the horizon is laid at 20 km along its bearing.
// Displacement: the five swell waves (height; wt_swell) and the four FFT cascades (height and the
// horizontal "choppy" displacement - true sharp crests), both faded toward the coast exactly as
// the full-screen pass fades them, plus the foam's lift (up to water_grid.y m where the cascades
// fold over).

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;          // this pass: the grid's projection x the view's rotation (camera-relative)
    mat4 prev_mvp;
    vec4 color;          // x cols, y rows (quads, border included), z border cells at the sides, w at the bottom
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;
} pc;

layout(set = 0, binding = 8)  uniform sampler2D u_terrain_height;
layout(set = 0, binding = 17) uniform sampler2D u_terrain_height_full;
const int kRmTextureCount = 48;         // = terrain_raymarch.frag
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kRmTextureCount];
layout(set = 0, binding = 4) uniform sampler2D u_normal[kRmTextureCount];
layout(set = 0, binding = 51) uniform sampler2DArray u_ww_grad;
layout(set = 0, binding = 52) uniform sampler2DArray u_ww_disp;

#include "terrain_rm_height_tex.glsl"
#include "scene_ubo.glsl"
#include "terrain_rm_noise_road.glsl"
#include "terrain_rm_pebbles.glsl"
#include "terrain_rm_heightfield.glsl"
#include "terrain_rm_water.glsl"
#define WW_GRAD u_ww_grad
#define WW_DISP u_ww_disp
#include "water_waves.glsl"

invariant gl_Position;

void main() {
    g_far_chart = scene.globe_info.x > 0.5;
    int cols = int(pc.color.x + 0.5), rows = int(pc.color.y + 0.5);
    int quad = gl_VertexIndex / 6, corner = gl_VertexIndex % 6;
    const ivec2 kC[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 0), ivec2(1, 1), ivec2(0, 1));
    ivec2 ij = ivec2(quad % cols, quad / cols) + kC[corner];
    // the screen: x, y in -1 .. 1 inside; bx / by border cells beyond the sides and the bottom
    float bx = pc.color.z, by = pc.color.w;
    vec2 ndc = vec2((float(ij.x) - bx) / (float(cols) - 2.0 * bx), (float(ij.y)) / (float(rows) - by)) * 2.0 - 1.0;
    // (Vulkan's NDC: y down; the BOTTOM of the screen is y = +1 - the border rows are past it)
    if (abs(ndc.x) > 1.0) ndc.x = sign(ndc.x) * (1.0 + (abs(ndc.x) - 1.0) * 10.0);
    if (ndc.y > 1.0) ndc.y = 1.0 + (ndc.y - 1.0) * 20.0;
    vec3 cam = scene.camera_pos.xyz;
    float wl = scene.water_params.y - (SW_ON(8192) ? 1.0 : 0.0);      // (the open sea stands 1 m under the datum: water_level.glsl)
    vec3 rd = scene_ray(ndc);
    float t;
    vec2 xz;
    if (rd.y < -1e-4 && cam.y > wl) {
        t = min((wl - cam.y) / rd.y, 20000.0);
        xz = cam.xz + rd.xz * t;
    } else {
        vec2 hd = length(rd.xz) > 1e-5 ? normalize(rd.xz) : vec2(1.0, 0.0);
        t = 20000.0;
        xz = cam.xz + hd * t;
    }
    // The waves, faded toward the coast as the full-screen pass fades them (terrain_raymarch.frag:
    // the depth under the level, the distance to the coast from the bed's slope).
    float tw_t = scene.water_params.w;
    float sk = wt_sea_k(xz);
    vec3 P = vec3(xz.x, wl, xz.y);
    if (sk > 0.0 && t < 19000.0) {
        float bed0 = terrainM_lod(xz, t);
        float dep = wl - bed0;
        float dk = 1.0;
        if (scene.water_flags.z > 0.5 && dep < 60.0) {
            const float e = 2.0;
            float bx2 = terrainM_lod(xz + vec2(e, 0.0), t), bz2 = terrainM_lod(xz + vec2(0.0, e), t);
            dk = smoothstep(0.0, 1.0, ww_shore_dist(max(dep, 0.0), vec2(bx2 - bed0, bz2 - bed0) / e) / 150.0);
        }
        float sw_k = scene.water_wind.w * sk * min(dk, wt_swell_shore(dep, scene.water_wind.w));
        float fft_k = scene.water_fft.y > 0.5 ? sk * min(dk, smoothstep(0.0, 1.0, dep / (0.5 + 3.0 * scene.water_fft.z))) : 0.0;
        float h = sw_k > 1e-4 ? sw_k * wt_swell(xz, tw_t).x : 0.0;
        vec2 dxz = vec2(0.0);
        if (fft_k > 1e-4) {
            // (each cascade faded where a cell of the grid - about the pixel footprint here - is wider than its waves)
            float foot = max(t, 1.0) * 0.0025 / max(abs(rd.y), 0.12);
            vec3 d = vec3(0.0);
            float fold = 0.0;
            for (int i = 0; i < 4; ++i) {
                float L = scene.water_fft_L[i];
                float k = 1.0 - smoothstep(0.03, 0.12, foot / L);
                if (k <= 0.001) continue;
                d += k * textureLod(u_ww_disp, vec3(xz / L, float(i)), 0.0).xyz;
                fold += k * textureLod(u_ww_grad, vec3(xz / L, float(i)), 0.0).z;
            }
            h += fft_k * d.y;
            dxz = fft_k * d.xz;
            // THE FOAM'S LIFT: where the cascades fold over (whitecaps) the surface stands up to
            // water_grid.y m higher - the foam is a body, not only a colour.
            h += fft_k * scene.water_grid.y * pow(clamp(fold, 0.0, 1.0), 4.0);
        }
        P = vec3(xz.x + dxz.x, wl + h, xz.y + dxz.y);
    }
    gl_Position = pc.model * vec4(P - cam, 1.0);
}
