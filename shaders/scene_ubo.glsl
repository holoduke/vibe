// Per-frame scene UBO (set 0, binding 0) shared by every world shader.
//
// This is the ONE GLSL declaration of the block. It mirrors
// `struct SceneUBO` in src/engine/vk_engine/internal.h member for member
// (std140; every member is a vec4 / ivec4 / mat4 / vec4[N], so C++ and
// GLSL offsets agree without manual padding). Field meanings are
// documented on the C++ side. When adding a field: append it to BOTH
// (at the end), and update the sizeof static_assert in helpers.cpp.
//
// Include after #version with GL_GOOGLE_include_directive enabled.
#ifndef QLIKE_SCENE_UBO_GLSL
#define QLIKE_SCENE_UBO_GLSL

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4   sun_direction;
    vec4   sun_color;
    vec4   ambient;
    vec4   sky_color;
    ivec4  rt_flags;
    vec4   rt_params;
    ivec4  rt_flags2;
    vec4   rt_params2;
    vec4   camera_pos;
    vec4   rt_lod;
    vec4   viewport;
    vec4   muzzle_pos;
    vec4   muzzle_color;
    vec4   terrain_params;
    vec4   terrain_h_low;
    vec4   terrain_h_high;
    vec4   grass_extra;
    vec4   grass_extra2;
    mat4   light_vp;
    vec4   terrain_extra;
    vec4   water_params;
    vec4   water_color;
    vec4   water_color_shallow;
    vec4   water_shore;
    vec4   water_foam_color;
    vec4   water_foam_params;
    vec4   fog_band;
    vec4   terrain_rt_extra;
    vec4   grass_color_top;
    vec4   grass_color_bottom;
    vec4   grass_color_ground;
    vec4   grass_color_ground_far;
    vec4   grass_shadow_params;
    vec4   grass_shore_color;
    vec4   grass_shore_params;
    vec4   terrain_shore_color;
    vec4   terrain_shore_params;
    vec4   distance_fog_color;
    vec4   distance_fog_params;
    vec4   terrain_shore_general_color;
    vec4   terrain_shore_general_params;
    vec4   terrain_sand_color;
    vec4   water_river_extra;
    vec4   water_river_extinct;
    vec4   restir_params;
    vec4   spom_params;
    vec4   terrain_local_info;
    vec4   terrain_max_grid[256];
    vec4   terrain_disp_params;
    vec4   terrain_antitile_params;
    vec4   grass_shadow_on_terrain_params;
    vec4   grass_side_lit_params;
    vec4   voxel_origin;
    vec4   voxel_dims;
    vec4   voxel_grid;
    vec4   road_info;
    vec4   road_bbox;
    vec4   road_pts[64];
    vec4   grass_rm_params;
    vec4   grass_rm_params2;
    mat4   clutter_vp;
    vec4   atmo_params;
    vec4   lamp_lights[8];   // road lanterns nearest the eye: xyz pos, w intensity (0 = off)
    vec4   rm_ground_dry;    // rgb, w = green amount
    vec4   rm_ground_green;  // rgb, w = red sand amount
    vec4   rm_red_sand;      // rgb, w = patch scale
    vec4   rm_rock;          // rgb, w = ground saturation
    vec4   craters[24];      // xyz centre, w bowl radius
    vec4   crater_info;      // x = count, y = cloud cover 0..1, z = cloud shadow contrast
    vec4   ripples[16];      // splash rings: xy = world xz, z = age (s), w = amplitude (m)
    vec4   ripple_info;      // x = count, y = rain 0..1, z = ground wetness 0..1, w = fx light count
    vec4   fx_lights[8];     // short-lived glow lights (gun flashes, tracers): xyz pos, w radius (m)
    vec4   fx_light_col[8];  // rgb = colour x intensity, a = 1: casts shadows (traced by the shader)
} scene;

// Light from muzzle flashes and tracer rounds: each is a sphere of light
// that fades to nothing at its radius. This sums the ones WITHOUT shadows
// (fx_light_col.a = 0); the shadow-casting ones (laser bolts, gun flashes)
// are added by each shader with its own shadow ray. A little wrap so surfaces
// turned partly away (and foliage) still catch the glow.
// Road lanterns without a shadow ray (for shaders that cannot trace one,
// e.g. the raymarched lawn): warm light, gone at 13 m.
vec3 lamp_lights_unshadowed(vec3 p, vec3 n) {
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        float li = scene.lamp_lights[i].w;
        if (li < 1e-3) continue;
        vec3 to = scene.lamp_lights[i].xyz - p;
        float d = length(to);
        if (d >= 13.0) continue;
        float fall = 1.0 - d / 13.0; fall *= fall;
        sum += vec3(1.0, 0.62, 0.28) * (li * fall * max(0.25 + 0.75 * dot(n, to / max(d, 1e-3)), 0.0));
    }
    return sum;
}

vec3 fx_lights_diffuse(vec3 p, vec3 n) {
    int cnt = int(scene.ripple_info.w + 0.5);
    vec3 sum = vec3(0.0);
    for (int i = 0; i < cnt; ++i) {
        if (scene.fx_light_col[i].a > 0.5) continue;
        vec3 to = scene.fx_lights[i].xyz - p;
        float r = scene.fx_lights[i].w;
        float d2 = dot(to, to);
        if (d2 >= r * r) continue;
        float d = sqrt(d2);
        float fall = 1.0 - d / r; fall *= fall;
        float ndl = dot(n, to / max(d, 1e-3));
        sum += scene.fx_light_col[i].rgb * (fall * max(0.3 + 0.7 * ndl, 0.0));
    }
    return sum;
}

// Cloud shadows on the ground: the cloud field is far above the scene, so
// its shadow is a 2D pattern projected down the sun direction - drifting
// value-noise FBM thresholded by the cloud cover (crater_info.y), with the
// contrast in crater_info.z (0 under a uniform overcast). Multiply the
// direct sun light by it.
float cs_hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float cs_noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(cs_hash(i), cs_hash(i + vec2(1.0, 0.0)), f.x),
               mix(cs_hash(i + vec2(0.0, 1.0)), cs_hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float cloud_shadow(vec3 wp) {
    float k = scene.crater_info.z;
    if (k <= 0.001) return 1.0;
    vec3 L = scene.sun_direction.xyz;
    vec2 p = wp.xz + L.xz / max(L.y, 0.2) * (1800.0 - wp.y);
    p = p * 0.0021 + scene.water_params.w * vec2(0.0090, 0.0033);
    float n = cs_noise(p) * 0.55 + cs_noise(p * 2.13 + 11.7) * 0.30 + cs_noise(p * 4.7 + 3.1) * 0.15;
    float cov = scene.crater_info.y;
    float c = smoothstep(1.0 - cov - 0.12, 1.0 - cov + 0.12, n);
    return 1.0 - k * c;
}

#endif
