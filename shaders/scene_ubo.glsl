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
#ifndef VIBE_SCENE_UBO_GLSL
#define VIBE_SCENE_UBO_GLSL

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
    vec4   cloud_a;          // the cloud layer, as clouds.comp sees it: x coverage, y size, z base altitude (m), w tallest (m)
    vec4   cloud_b;          // xy = wind offset of the weather field (m), z = density
    vec4   footprints[32];   // prints in the snow: xy = world xz, z = heading (player yaw), w = depth 0..1 (0 = unused)
    vec4   season;           // how much of each season shows: x spring, y summer, z autumn, w winter
    // The globe (engine/globe.h): the chart the engine is in.
    mat4   globe_frame;      // columns: the chart's east, up, south at its origin (planet frame)
    vec4   globe_info;       // x: 1 = a far chart (not home's), y: the chart origin's local x (m), z: 1 = the near-field ground bake is valid, w: profiling mask (VIBE_RM_SKIP: terrain shader sections left out)
    ivec4  globe_cell[12];   // the landscape FBM's lattice at the chart's origin, per octave: cell ...
    vec4   globe_frac[12];   // ... and position in it (world_map.glsl wm_shape3)
    vec4   rm_perf;          // raymarched terrain, speed: x = bits (1: low-resolution march first, 2: simpler relief under grass, 4: sun shadows from that march's pass), y = that march's pixel size (full-res pixels), z = fast final approach 0..1, w = normal: drop detail finer than this many pixels (0 = keep all)
} scene;

// Leaves through the year. `h`: a number 0..1 of the tree (each tree turns
// its own colour and at its own pace in autumn). Returns the leaf colour to use.
vec3 season_leaf(vec3 col, float h) {
    float sp = scene.season.x, au = scene.season.z;
    if (sp + au < 0.004) return col;
    float lum = dot(col, vec3(0.3, 0.5, 0.2));
    // Spring: young leaves - lighter, yellower green.
    col = mix(col, lum * vec3(1.05, 1.42, 0.62), sp * 0.55);
    // Autumn: per tree yellow, orange or rust; some turn early, some late.
    float h1 = h, h2 = fract(h * 7.13 + 0.37);
    vec3 tone = h1 < 0.40 ? vec3(2.05, 1.45, 0.32) : (h1 < 0.75 ? vec3(2.15, 0.98, 0.26) : vec3(1.75, 0.55, 0.24));
    float turned = smoothstep(h2 * 0.55, h2 * 0.55 + 0.45, au);
    return mix(col, lum * tone, turned * 0.85);
}

// Weather on surfaces (packed in ripple_info.z by descriptors.cpp).
float scene_wet()  { return fract(scene.ripple_info.z); }
#ifdef VIBE_SUMMER
float scene_snow() { return 0.0; }       // the no-snow build of a shader: all winter code folds away
#else
float scene_snow() { return floor(scene.ripple_info.z) / 255.0; }
#endif
// Snow lying on a surface: on what faces up, patchy while it is still thin.
float snow_cover(vec3 n, vec3 wp) {
    float s = scene_snow();
    if (s < 0.004) return 0.0;
    float pn = fract(sin(dot(floor(wp.xz * 3.0), vec2(12.9898, 78.233))) * 43758.5453) * 0.25
             + 0.75 * (0.5 + 0.5 * sin(wp.x * 0.9) * sin(wp.z * 1.1 + 1.3));
    return smoothstep(0.25, 0.75, n.y) * smoothstep(pn * 0.8, pn * 0.8 + 0.25, s * 1.05);
}

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
// How much of the direct sunlight the clouds let through at `wp` (1 = open
// sky). The cloud pass (clouds.comp, mode 3) marches the real cloud layer
// toward the sun for every point of the land and stores the result in a
// map, so a shadow on the ground lies under the cloud that casts it.
// crater_info.z is the strength (0 when the sky is one grey deck: the light
// is uniformly dim then, and that is applied to the sun itself).
// A shader that wants it declares the map sampler and names it:
//   layout(set = 0, binding = N) uniform sampler2D u_cloud_shadow;
//   #define CLOUD_SHADOW_MAP u_cloud_shadow      (before this include)
#ifdef CLOUD_SHADOW_MAP
float cloud_shadow(vec3 wp) {
    float k = scene.crater_info.z;
    if (k <= 0.001) return 1.0;
    vec3 L = scene.sun_direction.xyz;
    // The map is laid out on the y = 0 plane: slide down the sun ray to it.
    vec2 g = wp.xz - L.xz / max(L.y, 0.10) * wp.y;
    vec2 uv = g / 6144.0 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 1.0;
    float t = textureLod(CLOUD_SHADOW_MAP, uv, 0.0).r;
    return 1.0 - k * (1.0 - clamp(t, 0.0, 1.0));
}
#endif

#endif
