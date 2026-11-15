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
    vec4   crater_info;      // x = count, y = cloud cover 0..1, z = cloud shadow contrast, w = haze light level (1 by day)
    vec4   ripples[16];      // splash rings: xy = world xz, z = age (s), w = amplitude (m)
    vec4   ripple_info;      // x = count, y = rain 0..1, z = ground wetness 0..1, w = fx light count
    vec4   fx_lights[8];     // short-lived glow lights (gun flashes, tracers): xyz pos, w radius (m)
    vec4   fx_light_col[8];  // rgb = colour x intensity, a = 1: casts shadows (traced by the shader)
    vec4   flash_pos;        // hand flashlight: xyz lens position, w range (m; 0 = off)
    vec4   flash_dir;        // xyz beam direction, w cos(outer cone angle)
    vec4   flash_col;        // rgb colour x intensity, w cos(inner cone angle)
    mat4   flash_vp;         // flashlight shadow map (small plants): world -> clip, [3][3] = 0 when off
    vec4   lamp_info;        // x = bit mask: lantern i has a small-plant shadow map
} scene;

// Light from muzzle flashes and tracer rounds: each is a sphere of light
// that fades to nothing at its radius. This sums the ones WITHOUT shadows
// (fx_light_col.a = 0); the shadow-casting ones (laser bolts, gun flashes)
// are added by each shader with its own shadow ray. A little wrap so surfaces
// turned partly away (and foliage) still catch the glow.
// Beam pattern of a reflector torch, by the cosine of the angle off the
// axis: a tight hotspot, a wider corona around it, a dim spill out to the
// edge of the cone, and the faint rings a smooth reflector throws.
float flashlight_pattern(float c) {
    float c_out = scene.flash_dir.w;
    if (c <= c_out) return 0.0;
    // 0 on the axis .. 1 at the edge of the cone (tangent of the angle).
    float x = (sqrt(max(1.0 - c * c, 0.0)) / max(c, 1e-3)) / (sqrt(max(1.0 - c_out * c_out, 0.0)) / c_out);
    float edge = 1.0 - smoothstep(0.86, 1.0, x);
    float spill = 0.16;
    float corona = 0.85 * exp(-x * x * 9.0);
    float hot = 2.6 * exp(-x * x * 55.0);
    float rings = 1.0 + 0.05 * cos(x * 46.0) * (1.0 - x) + 0.03 * cos(x * 17.0);
    return (spill + corona + hot) * rings * edge;
}

// Hand flashlight: light arriving at p from the beam, before N.L and the
// shadow ray (each shader traces its own). A cone with a soft edge and a
// brighter centre spot, falling off with distance and gone at its range.
// Returns the direction to the lamp and the distance for the shadow ray.
vec3 flashlight_at(vec3 p, out vec3 to_lamp, out float dist) {
    to_lamp = vec3(0.0, 1.0, 0.0); dist = 0.0;
    float range = scene.flash_pos.w;
    if (range <= 0.0) return vec3(0.0);
    vec3 to = scene.flash_pos.xyz - p;
    dist = length(to);
    if (dist >= range) return vec3(0.0);
    to_lamp = to / max(dist, 1e-3);
    float pat = flashlight_pattern(dot(-to_lamp, scene.flash_dir.xyz));
    if (pat <= 0.0) return vec3(0.0);
    float win = 1.0 - dist / range; win *= win;
    return scene.flash_col.rgb * (pat * win / (1.0 + 0.02 * dist * dist));
}

// The flashlight beam in the air: light scattered toward the eye by the
// dust and moisture inside the cone, integrated along the view ray up to
// the surface at t_max. Samples are packed toward the lamp (where the cone
// is narrow) and jittered per pixel and frame; the temporal filter turns
// that into a smooth volume. Scattering fades in over the first half
// metre, so there is no glowing ball at the lens.
vec3 flashlight_beam(vec3 ro, vec3 rd, float t_max, float jitter) {
    float range = scene.flash_pos.w;
    if (range <= 0.0) return vec3(0.0);
    // Rays pointing well away from the beam never enter it.
    if (dot(rd, scene.flash_dir.xyz) < scene.flash_dir.w - 0.45) return vec3(0.0);
    float t_end = min(t_max, range);
    const int N = 24;
    vec3 sum = vec3(0.0);
    float prev = 0.0;
    for (int i = 0; i < N; ++i) {
        float f = (float(i) + jitter) / float(N);
        float t = t_end * f * f;
        float dt = t - prev;
        prev = t;
        vec3 to = ro + rd * t - scene.flash_pos.xyz;
        float d = length(to);
        if (d >= range) break;
        vec3 ld = to / max(d, 1e-3);
        float pat = flashlight_pattern(dot(ld, scene.flash_dir.xyz));
        if (pat <= 0.0) continue;
        float win = 1.0 - d / range; win *= win;
        // Inverse-square, bounded near the lamp, and no scattering right at the lens.
        float att = win / (0.8 + d * d) * smoothstep(0.10, 0.55, d);
        // Back-scatter toward the eye, a little stronger looking along the beam.
        float mu = dot(ld, -rd);
        float phase = 0.055 + 0.030 * mu * mu;
        sum += scene.flash_col.rgb * (pat * att * phase * dt);
    }
    return sum * 0.20;
}

// Small plants (flowers, ferns, grass tufts, pebbles) are not in the ray-
// traced scene; a depth map rendered from the lamp (layer 2 of the small-
// object shadow array) gives their shadows in the flashlight beam.
// Returns visibility 0..1 (3x3 taps).
#define FLASH_PLANT_VIS(TEX, P, OUT)                                                          \
    {                                                                                         \
        OUT = 1.0;                                                                            \
        vec4 fc_ = scene.flash_vp * vec4(P, 1.0);                                             \
        if (fc_.w > 0.2) {                                                                    \
            vec3 fn_ = fc_.xyz / fc_.w;                                                       \
            vec2 fuv_ = fn_.xy * 0.5 + 0.5;                                                   \
            if (all(greaterThan(fuv_, vec2(0.002))) && all(lessThan(fuv_, vec2(0.998)))) {    \
                vec2 fts_ = 1.0 / vec2(textureSize(TEX, 0).xy);                               \
                float fb_ = 0.006 / (fc_.w * fc_.w) + 0.00002;   /* ~4 cm of depth */         \
                float fo_ = 0.0;                                                              \
                for (int fy_ = -1; fy_ <= 1; ++fy_)                                           \
                    for (int fx_ = -1; fx_ <= 1; ++fx_)                                       \
                        fo_ += (textureLod(TEX, vec3(fuv_ + vec2(fx_, fy_) * fts_, 2.0), 0.0).r < fn_.z - fb_) ? 1.0 : 0.0; \
                OUT = 1.0 - fo_ / 9.0;                                                        \
            }                                                                                 \
        }                                                                                     \
    }

// Small-plant shadows for road lantern I: each lantern has a quarter of an
// atlas layer (layers 3 and 4 of the small-object shadow array) holding the
// distance to the nearest plant in every direction below and around it
// (an equidistant fisheye about straight down; clutter_shadow.vert writes
// it with the same mapping). Returns visibility 0..1.
#define LAMP_PLANT_VIS(TEX, P, I, OUT)                                                        \
    {                                                                                         \
        OUT = 1.0;                                                                            \
        if (((int(scene.lamp_info.x + 0.5) >> (I)) & 1) != 0) {                               \
            vec3 ld_ = (P) - scene.lamp_lights[I].xyz;                                        \
            float ll_ = length(ld_);                                                          \
            float lt_ = acos(clamp(-ld_.y / max(ll_, 1e-4), -1.0, 1.0));                      \
            if (lt_ < 1.70 && ll_ < 13.8) {                                                   \
                vec2 lxy_ = normalize(ld_.xz + vec2(1e-6)) * (lt_ / 1.75) * 0.96;             \
                vec2 lq_ = vec2(float((I) & 1), float(((I) >> 1) & 1));                       \
                vec2 luv_ = (lxy_ * 0.5 + 0.5 + lq_) * 0.5;                                   \
                float llay_ = 3.0 + float((I) >> 2);                                          \
                vec2 lts_ = 1.0 / vec2(textureSize(TEX, 0).xy);                               \
                float lz_ = ll_ / 14.0 - 0.005;                                               \
                float lo_ = 0.0;                                                              \
                for (int ly_ = -1; ly_ <= 1; ++ly_)                                           \
                    for (int lx_ = -1; lx_ <= 1; ++lx_)                                       \
                        lo_ += (textureLod(TEX, vec3(luv_ + vec2(lx_, ly_) * lts_, llay_), 0.0).r < lz_) ? 1.0 : 0.0; \
                OUT = 1.0 - lo_ / 9.0;                                                        \
            }                                                                                 \
        }                                                                                     \
    }

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
