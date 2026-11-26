#version 460
#extension GL_GOOGLE_include_directive : require

// The grass placed on the GPU (tuft_place.comp: long tufts and turf), depth only, into the
// small plants' shadow maps (clutter_shadow.cpp). The same blade maths as the colour pass
// (tuft_cd.glsl), so a blade that lies down or leans in the wind casts the shadow of what is
// drawn. On the world pipeline layout: it reads the height map, the lawn's mask, the trail and
// the wind field.

layout(location = 0) in vec3 inPos;
layout(location = 2) in vec2 inUv;
layout(location = 3) in vec4 inPosScale;   // .xyz base, .w size (negative = placed on the GPU)
layout(location = 4) in vec4 inRotTint;    // .x turn (+ whole turns = lift)

#include "scene_ubo.glsl"

layout(push_constant) uniform PC {
    mat4 mvp;            // the light's view-projection
    mat4 model;          // [0] = the map's centre (xyz) and the radius it covers (w)
    mat4 prev_mvp;
    vec4 color;          // xy = wind direction (world x, z), z = wind strength 0..1
    vec4 emissive;       // the trail window
    vec4 tex_params;     // the mesh's shape (ScatterSet::kTuftShape / kTurfShape)
    vec4 grass_params;   // .z time, .w 1 = raymarched terrain (lift)
} pc;

layout(set = 0, binding = 17) uniform sampler2D u_heightmap_full;
layout(set = 0, binding = 13) uniform sampler2D u_grass_mask;
layout(set = 0, binding = 41) uniform sampler2D u_trail;
layout(set = 0, binding = 43) uniform sampler2D u_trail_h;
layout(set = 0, binding = 45) uniform sampler3D u_wind;

#include "tuft_cd.glsl"

void main() {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vec3  base = inPosScale.xyz;
    // Only what the placement made: hand-placed tufts are in the CPU cells' pass.
    // Small debris (parts 20..27: scatter_debris_meshes.inl, placed by tuft_place.comp): lying flat,
    // turned about the vertical - its small shadow in the near map (clutter_shadow.cpp, layer 1).
    if (inPosScale.w < 0.0 && int(inUv.x + 0.5) >= 20) {
        vec2 dd = base.xz - pc.model[0].xz;
        if (dot(dd, dd) > pc.model[0].w * pc.model[0].w) return;
        float c = cos(inRotTint.x), s = sin(inRotTint.x);
        vec3 lp = inPos * -inPosScale.w;
        vec3 w = base + vec3(c * lp.x - s * lp.z, lp.y, s * lp.x + c * lp.z);
        if (pc.grass_params.w > 0.5) w.y += floor(inRotTint.x / 6.2831853) * 0.004 * (1.0 - smoothstep(10.0, 26.0, distance(base, scene.camera_pos.xyz)));
        gl_Position = pc.mvp * vec4(w, 1.0);
        return;
    }
    if (inPosScale.w >= 0.0 || (int(inUv.x + 0.5) != 8 && int(inUv.x + 0.5) != 10)) return;
    float size = -inPosScale.w;
    vec2  d = base.xz - pc.model[0].xz;
    if (dot(d, d) > pc.model[0].w * pc.model[0].w || size < 0.05) return;
    float dist = distance(base, scene.camera_pos.xyz);
    float rot  = inRotTint.x;
    if (pc.grass_params.w > 0.5)
        base.y += floor(rot / 6.2831853) * 0.004 * (1.0 - smoothstep(10.0, 26.0, dist));
    float sway = clamp(fract(inUv.y * 32.0) / 0.98, 0.0, 1.0);
    vec3  n = vec3(0.0, 1.0, 0.0);
    bool  gone;
    vec3  world = tuft_cd(inPos, n, base, size, rot, sway, floor(inUv.y * 32.0), dist,
                          false, pc.tex_params.x < 0.5, gone);
    if (gone) return;
    gl_Position = pc.mvp * vec4(world, 1.0);
}
