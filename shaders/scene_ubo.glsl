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
    vec4   lamp_lights[24];  // lanterns and street lamps, nearest the eye first: xyz pos, w intensity (0 = off); lamp_info.y = how many slots are in use
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
    vec4   lamp_info;        // x = bit mask: lantern i has a small-plant shadow map; y = slots of lamp_lights in use (loop bound)
    vec4   cloud_a;          // the cloud layer, as clouds.comp sees it: x coverage, y size, z base altitude (m), w tallest (m)
    vec4   cloud_b;          // xy = wind offset of the weather field (m), z = density
    vec4   footprints[32];   // prints in the snow: xy = world xz, z = heading (player yaw), w = depth 0..1 (0 = unused)
    vec4   season;           // how much of each season shows: x spring, y summer, z autumn, w winter
    // The globe (engine/globe.h): the chart the engine is in.
    mat4   globe_frame;      // columns: the chart's east, up, south at its origin (planet frame)
    vec4   globe_info;       // x: 1 = a far chart (not home's), y: the chart origin's local x (m), z: 1 = the near-field ground bake is valid, w: profiling mask (VIBE_RM_SKIP: terrain shader sections left out)
    ivec4  globe_cell[12];   // the landscape FBM's lattice at the chart's origin, per octave: cell ...
    vec4   globe_frac[12];   // ... and position in it (world_map.glsl wm_shape3)
    vec4   rm_perf;          // raymarched terrain, speed: x = bits (1: low-resolution march first, 2: simpler relief under grass, 4: sun shadows from that march's pass, 8: start at that march's hit, 32: the low-resolution terrain pass starts at the march too, 64: the ground is displaced in screen space - write its height, no parallax of its own), y = that march's pixel size (full-res pixels), z = fast final approach 0..1, w = normal: drop detail finer than this many pixels (0 = keep all)
    // The sea's wave field far from the world's origin (terrain_raymarch.frag sea_map). xy of
    // sea_org: the point (m) the waves are worked out from, near the camera. Per octave of
    // the wave field, that point's coordinate in the octave's own space, split up so that no
    // large number reaches the shader: sea_a.xy = it modulo 2 pi (for the sines), .zw = its
    // fraction and sea_b.xy = its whole part modulo 4096 (for the noise lattice).
    // A site's shaped ground (engine/globe.h Site; terrain_rm_noise_road.glsl rm_site_ground):
    // xy = the centre, in this chart's coordinates, of the site that is within sight;
    // z = 1 when there is one; w = its number.
    vec4   site_ground;
    vec4   site_shape;       // that site's ground: x level within (m), y meets the natural ground at (m), z height above the water (m), w no woods within (m)
    vec4   sea_org;
    vec4   sea_a[5];
    vec4   sea_b[5];
    // Shelters: the roofed boxes nearest the camera - no rain, no wet, no snow inside them
    // (scene_sheltered below). a = min x, min z, max x, max z; b.x = the roof's height (y),
    // b.y = 1 when the entry is used (they are packed from 0).
    vec4   shelter_a[4];
    vec4   shelter_b[4];
    // The highways' field (engine/highway.h): hw_info = the window's centre x, z; 1 when there
    // is road in it; its side (m). hw_bbox = where the shelf is: min x, min z, max x, max z.
    vec4   hw_info;
    vec4   hw_bbox;
    // Home's flat landscape formula from a chart that is not home's (engine/globe.h Ring;
    // terrain_rm_noise_road.glsl rm_ring*): ring_info.x = 1 when it applies, .yz = the chart
    // point the mapping is taken at; home coordinates = ring_h0.xy + (dot(ring_j.xy, l),
    // dot(ring_j.zw, l)) for l = p - ring_info.yz. ring_cell[i]: octave i of the formula's FBM
    // at ring_h0 - xy = position in the lattice cell, zw = the cell.
    vec4   ring_info;
    vec4   ring_h0;
    vec4   ring_j;
    vec4   ring_cell[24];
    // The grass's blade lattice, from the camera (grass_raymarch.frag map): xy = the number of
    // the lattice cell the camera is in, zw = the camera's place in it (m). Worked out in doubles.
    vec4   grass_cam;
    // Screen to view ray, with the camera at the origin: inverse(projection x the view's
    // rotation), worked out in doubles (scene_ray below).
    mat4   ray_mat;
    // The world's slow fields round the camera as a texture (vk_engine/wm_cache.cpp;
    // terrain_rm_noise_road.glsl rm_world_vr): xy = the chart position of texel (0, 0)'s centre,
    // z = texels a metre, w = texels a side. wm_cache: the near picture, wm_cache1: the far one.
    vec4   wm_cache;
    vec4   wm_cache1;
    // Cars pressing the grass down (jeep.cpp jeep_grass_press; grass_raymarch.frag car_press):
    // two cars, two vec4 each. [0]: xy = the car's middle (world xz), zw = the way it faces (xz,
    // unit). [1]: x = half its width, y = half its length, z = how far behind it the grass is
    // still coming up (m), w = 1 when there is a car.
    vec4   car_press[4];
    // Snow on the cars (jeep.cpp jeep_snow_fill; snow_cover below): the eight cars nearest the eye that
    // have lost some of theirs by driving, two vec4 each. [0]: xy the middle (world xz), zw the way it
    // faces (unit xz). [1]: x half its width, y half its length, z its floor (world y), w the share of
    // its snow it still carries. x = 0: no more cars. A car not in the list carries all of it.
    vec4   car_snow[16];
    // The fields of the flowering meadow (the engine's flower_fields_, as tuft_place.comp gets
    // them): xy = the middle (world x, z), z = radius (0 = none), w = how frayed the edge is (m).
    // The ground inside one takes the field's colour with distance (terrain_rm_main_light.glsl).
    vec4   flower_field[4];
    // Where a side road leaves a highway (engine/highway.h Junction; terrain_rm_noise_road.glsl
    // rm_jn_*). [0]: xy = the point J on the highway's centre line (chart x, z), zw = the way the
    // side road leaves (unit). [1]: x = 1 when there is one, y = the side road's width class,
    // z = its metres at J as the field counts them. [2], [3]: its centre line heights at 0, 8 .. 56 m.
    vec4   hw_jn[4];
    // The lighting ports (light_port.cpp; deferred_lighting.frag, contact_shadow.glsl).
    // lp_flags.x = bits: 1 contact shadows, 2 far AO + bent cone, 4 leaf-tinted sun shadows,
    // 8 true colour at bounce hits, 16 one picked lamp per pixel, 32 screen march before the
    // reflection / bounce ray; .y = lamps in the lamp list (deferred pass binding 26); .z = the
    // terrain's slot in the materials table (-1: none); .w = this frame's first entry of the list.
    // lp_params: x, y = the projection's [3][2] and [2][2] (view depth from a depth value z:
    // x / (z + y)); z = how much the bounce-light history and caches are scaled this frame
    // (1 = not; a sudden drop of the sky's light); w = lamps looped over before the picked one.
    // lp_vp: this frame's view-projection (the one the deferred pass inverts).
    ivec4  lp_flags;
    vec4   lp_params;
    mat4   lp_vp;
    // The wind on the water (environment.cpp water_wind; terrain_rm_water.glsl wt_*):
    // xy = the way it blows (unit, world x z), z = speed (m/s), w = the swell's height scale
    // pow(saturate(speed / 30), 2.2). water_swell.x = the water surface at the eye about the
    // water level (m; water_fx.cpp water_swell_height, the CPU twin of wt_swell); yz = the
    // wind's direction summed over time (s): what the water's drifting patterns move by;
    // w = how many of river_piers are in use.
    vec4   water_wind;
    vec4   water_swell;
    // The piers standing in the river (water_fx.cpp river_piers): xy = the middle (world x z),
    // z = how wide it stands in the current (m). terrain_rm_water.glsl wt_obstacles. The rocks in the water too.
    vec4   river_piers[24];
    // The stream (engine/water_course.h; shaders/water_course.glsl): stream_info.x = points in use,
    // .y = the channel's depth at the spring, .z = at the mouth; stream_box = per 8 segments min x z,
    // max x z (padded by the carve's reach); stream_pts = x, z, water surface, half width.
    vec4   stream_info;
    vec4   stream_box[16];
    vec4   stream_pts[128];
    // The player's body (not the camera: third person, a car): xyz = its middle (feet 0.9 m lower) -
    // what the water lilies part round (scatter.vert). w unused.
    vec4   water_body;
    // The water's simulation (vk_engine/water_sim.cpp; shaders/water_waves.glsl):
    // water_fft: x 1 = the wind sea's four FFT cascades are on (VIBE_NO_FFT), y 1 = they also move
    //   the open sea's surface, z the height bound of that motion (m, for the hit), w the wind
    //   sea's slope variance it was normalised to (Cox-Munk, resolved part);
    // water_fft_L: the four cascades' sizes (m);
    // water_rip: xy the ripple field's corner (world x z), z its size (m), w 1 = on (VIBE_NO_RIPPLE_FIELD);
    // water_flags: x the refraction rule (VIBE_NO_WATER_REFRACT), y reflected caustics
    //   (VIBE_NO_REFL_CAUSTICS), z the shore distance (VIBE_NO_SHORE_SDF), w the sky ambient (VIBE_NO_WATER_SKY_AMB).
    vec4   water_fft;
    vec4   water_fft_L;
    vec4   water_rip;
    vec4   water_flags;
    // water_sun: x how much of the sun's DISC shows through the cloud deck (1 clear, ~0 under a closed
    //   overcast): the glint and the caustics are images of the disc, not diffuse light. z: the open
    //   sea's wind (m/s; environment.cpp sea_wind - the local wind with a sea breeze as its floor).
    vec4   water_sun;
    // The local shallow-water solver (vk_engine/water_solver.cpp; shaders/water_solver.glsl):
    // water_sw: xy its window's corner (world x z), z its size (m), w 1 = on (VIBE_NO_SOLVER);
    // water_sw2: x 1 = its foam particles are on.
    vec4   water_sw;
    vec4   water_sw2;
    // Materials at the water and the sea's surface (vk_engine/water_wet.cpp; shaders/shore_wet.glsl):
    // water_wet: x = bits (1 the wet band, 2 the sea's swash, 4 puddles in the hollows, 8 growth
    //   under water, 16 the bed's tint by depth, 32 the bed's relief march, 64 no snow in the band),
    //   y = the growth's rate, z = the band's height W at a lake / the river (m), w = at the stream;
    // water_grid: x 1 = the open sea from the projected grid's depth picture (scene binding 62),
    //   y = the foam's lift (m), z 1 = one picked lamp glints on the water, w 1 = the fog volume's
    //   in-scatter inside the water.
    vec4   water_wet;
    vec4   water_grid;
} scene;
#define SW_ON(bit) ((int(scene.water_wet.x + 0.5) & (bit)) != 0)

// The "shading test" (Graphics tab; gfx_quality.cpp apply_shade_test): measuring switches.
// A part that is "flat" is its own colour times one light for the whole picture - no
// direction, no shadow, no rays, no haze. What it costs then is geometry and marching.
#define FLAT_OBJECTS ((scene.rt_flags2.y & 2) != 0)
#define FLAT_GRASS   ((scene.rt_flags2.y & 4) != 0)
#define FLAT_GROUND  ((scene.rt_flags2.y & 8) != 0)
vec3 flat_light() { return 0.5 * scene.sun_color.rgb * scene.sun_color.a + 0.5 * scene.sky_color.rgb; }

// The view ray through a point of the screen (ndc: -1 .. 1), unit length. Use this - never
// inverse(view_proj) x ndc less the camera's position: that puts the camera's coordinates
// into the sum and takes them out again, and where they are large (a float resolves 1 mm at
// 10 km) the direction is left with two digits; neighbouring pixels then share a ray and
// whatever is ray-marched comes out in blocks. scene.ray_mat has no position in it.
vec3 scene_ray(vec2 ndc) {
    vec4 a = scene.ray_mat * vec4(ndc, 0.0, 1.0), b = scene.ray_mat * vec4(ndc, 1.0, 1.0);
    vec4 q = abs(a.w) > abs(b.w) ? a : b;          // (the near plane's point: the far plane may lie at infinity)
    return normalize(q.xyz / q.w);
}

// ---- Tiles ---------------------------------------------------------------------------------
// The engine's coordinates are small everywhere: far from home they are the land's own
// coordinates less a whole number of tiles (engine/globe.h, "Small coordinates"; kTile there).
// Whatever is worked out from a position and REPEATS ONCE A TILE is therefore the same before
// and after the engine re-centres - so the fine patterns (grass blades, ground textures, the
// ground's fine noise) are made to: a cell size or a frequency is snapped to the nearest that
// fits a whole number of times into a tile (a change of a few parts in 100,000), and a cell's
// number is wrapped at the tile before it is hashed. Use these for anything new that is drawn
// from world coordinates at a scale of metres or less.
const float kTile = 6400.0;
#define TILE_N(cell)    (max(round(kTile / (cell)), 1.0))                   /* cells of about `cell` metres in a tile */
#define TILE_CELL(cell) (kTile / TILE_N(cell))                               /* that cell's exact size */
#define TILE_FREQ(k)    (max(round(kTile * (k)), 1.0) / kTile)               /* cells per metre, snapped */
#define TILE_W(w)       (max(round(kTile * (w) / 6.28318530718), 1.0) * 6.28318530718 / kTile)   /* radians per metre, snapped */
// A cell's number, wrapped at the tile (n = cells in a tile). Numbers within half a tile of
// zero are left as they are.
vec2 tile_wrap(vec2 id, float n) { return id - n * floor(id / n + 0.5); }
float tile_h21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
// Value noise of the position wp at about k cells a metre (+ off, in cells), the same in every tile.
float tile_noise2(vec2 wp, float k, vec2 off) {
    float n = max(round(kTile * k), 1.0);
    vec2 p = wp * (n / kTile) + off;
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    vec2 i0 = tile_wrap(i, n), i1 = tile_wrap(i + 1.0, n);
    float a = tile_h21(i0), b = tile_h21(vec2(i1.x, i0.y)), c = tile_h21(vec2(i0.x, i1.y)), d = tile_h21(i1);
    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
}
// Value noise on a lattice TURNED against the world's axes, still the same in every tile.
// abc: a right triangle with whole sides (3,4,5 / 7,24,25 / 5,12,13 ...; a or b may be
// negative) - the lattice is turned by its angle, and only such an angle lets a tile's shift
// land on whole cells. k: cells per metre along the lattice's two axes (they may differ: a
// grain). off: an offset in cells.
float tile_noise2_rot(vec2 wp, vec3 abc, vec2 k, vec2 off) {
    vec2 n = max(round(kTile * k / abc.z), vec2(1.0));
    vec2 p = vec2(abc.x * wp.x + abc.y * wp.y, -abc.y * wp.x + abc.x * wp.y) * (n / kTile) + off;
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    vec2 i0 = i - n * floor(i / n + 0.5), i1 = (i + 1.0) - n * floor((i + 1.0) / n + 0.5);
    float a = tile_h21(i0), b = tile_h21(vec2(i1.x, i0.y)), c = tile_h21(vec2(i0.x, i1.y)), d = tile_h21(i1);
    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
}


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
// Under a roof: inside one of the shelter boxes (the engine lists what is roofed: fx.cpp
// keeps the rain itself out of the same boxes).
bool scene_sheltered(vec3 wp) {
    for (int i = 0; i < 4; ++i) {
        if (scene.shelter_b[i].y < 0.5) break;
        vec4 r = scene.shelter_a[i];
        if (wp.x > r.x && wp.z > r.y && wp.x < r.z && wp.z < r.w && wp.y < scene.shelter_b[i].x) return true;
    }
    return false;
}
float scene_wet()  { return fract(scene.ripple_info.z); }
#ifdef VIBE_SUMMER
float scene_snow() { return 0.0; }       // the no-snow build of a shader: all winter code folds away
#else
float scene_snow() { return floor(scene.ripple_info.z) / 255.0; }
#endif
// Snow lying on a surface: on what faces up, patchy while it is still thin.
// On a car that has driven some of its snow off (car_snow): the share it still carries, and the
// point in the car's own frame (so the patches it loses stay on the car, not in the world).
// The stream's upper reach (engine/water_course.cpp: it rises at ~106 m, above the summer snow line
// of 70 - 90 m) runs in a valley the snow has left (2026-11-27, the owner: it ran through snow in
// summer). 1 = snow may lie here, 0 = within 18 m of the stream's line, all of it again by 55 m.
// Only the first 48 points (the upper reach), only within their boxes. Used by every summer snow
// line: the ground (getMaterial), stones (scatter.frag), rock formations (rock_form.glsl), meshes'
// accumulation (cube_main_surfaces.glsl).
float stream_snow_clear(vec2 p) {
    int n = min(int(scene.stream_info.x + 0.5), 48);
    if (n < 2) return 1.0;
    float d2 = 1e12;
    for (int b = 0; b < 6; ++b) {
        vec4 B = scene.stream_box[b];
        if (p.x < B.x - 55.0 || p.y < B.y - 55.0 || p.x > B.z + 55.0 || p.y > B.w + 55.0) continue;
        for (int k = b * 8; k < min(b * 8 + 8, n - 1); ++k) {
            vec2 a = scene.stream_pts[k].xy, ab = scene.stream_pts[k + 1].xy - a;
            vec2 dv = p - (a + ab * clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-4), 0.0, 1.0));
            d2 = min(d2, dot(dv, dv));
        }
    }
    return smoothstep(18.0, 55.0, sqrt(d2));
}
float car_snow_at(vec3 wp, out vec2 local) {
    local = wp.xz;
    for (int i = 0; i < 8; ++i) {
        vec4 c = scene.car_snow[2 * i], e = scene.car_snow[2 * i + 1];
        if (e.x <= 0.0) break;
        vec2 d = wp.xz - c.xy;
        float along = dot(d, c.zw), across = dot(d, vec2(-c.w, c.z));
        if (abs(along) < e.y && abs(across) < e.x && wp.y > e.z + 0.35 && wp.y < e.z + 2.6) { local = vec2(across, along) + vec2(17.0, 31.0); return e.w; }
    }
    return 1.0;
}
float snow_cover(vec3 n, vec3 wp) {
    float s = scene_snow();
    if (s < 0.004 || scene_sheltered(wp)) return 0.0;
    vec2 sp;
    s *= car_snow_at(wp, sp);
    if (s < 0.004) return 0.0;
    float pn = fract(sin(dot(floor(sp * 3.0), vec2(12.9898, 78.233))) * 43758.5453) * 0.25
             + 0.75 * (0.5 + 0.5 * sin(sp.x * 0.9) * sin(sp.y * 1.1 + 1.3));
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

// A car's dipped beam (jeep.cpp jeep_headlights; flash_col.w > 1.5 says the light in the slot is
// one): not a round cone but what a dipped head lamp throws - a sharp cut-off a little under the
// level of the lamp (nothing goes higher: it does not dazzle, and it does not light the trees
// ahead), wide to both sides (some 35 degrees), the brightest part just under the cut, and less
// and less toward the ground before the car. `ld`: the way from the lamp to the point (unit).
float car_beam_pattern(vec3 ld) {
    vec3 f = scene.flash_dir.xyz;
    vec3 r = cross(f, vec3(0.0, 1.0, 0.0));
    float rl = length(r);
    if (rl < 1e-3) return 0.0;
    r /= rl;
    vec3 u = cross(r, f);
    float z = dot(ld, f);
    if (z <= 0.05) return 0.0;
    float h = dot(ld, r) / z, v = dot(ld, u) / z;            // (tangents across and up, from the beam's axis)
    // (the axis is aimed 1.7 degrees down: the cut-off at 1 degree above it is just under level)
    float cut = mix(0.012, 1.0, 1.0 - smoothstep(0.006, 0.022, v));
    float hs = h / 0.70, vs = max(-v, 0.0) / 0.30;           // (35 degrees across, 17 down: the road a few metres before the car)
    float e2 = hs * hs + vs * vs;
    float edge = 1.0 - smoothstep(0.45, 1.0, e2);
    float hot = 2.4 * exp(-(hs * hs * 9.0 + vs * vs * 60.0));
    float corona = 0.9 * exp(-(hs * hs * 2.2 + vs * vs * 7.0));
    float spill = 0.10;
    return (spill + corona + hot) * edge * cut;
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
    float pat = scene.flash_col.w > 1.5 ? car_beam_pattern(-to_lamp) : flashlight_pattern(dot(-to_lamp, scene.flash_dir.xyz));
    if (pat <= 0.0) return vec3(0.0);
    float win = 1.0 - dist / range; win *= win;
    // (a long beam - a car's - thins out as slowly as it reaches far: range 95 m is the old law)
    float fk = 0.02 * min(1.0, (95.0 * 95.0) / (range * range));
    return scene.flash_col.rgb * (pat * win / (1.0 + fk * dist * dist));
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
        float pat = scene.flash_col.w > 1.5 ? car_beam_pattern(ld) : flashlight_pattern(dot(ld, scene.flash_dir.xyz));
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
    int lamp_n = int(scene.lamp_info.y + 0.5);
    for (int i = 0; i < lamp_n; ++i) {
        float li = scene.lamp_lights[i].w;
        if (li < 1e-3) continue;
        vec3 to = scene.lamp_lights[i].xyz - p;
        float d = length(to);
        if (d >= 34.0) continue;
        float fall = 1.0 - d / 34.0; fall = fall * fall * fall * fall * fall;      // (as bright near the lamp as when it reached 13 m, and a long thin tail)
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
    // The map wraps and covers kCloudShadowExtent round the camera (clouds.comp shadow_map):
    // a point's texel is its place modulo the extent. Toward the map's edge the shadows fade
    // out (beyond it the same texels would stand for other ground).
    const float kCloudShadowExtent = 16384.0;      // m (keep in step: atmosphere.cpp kCloudShadowExtent)
    vec2 dc = abs(g - scene.camera_pos.xz) / (0.5 * kCloudShadowExtent);
    float edge = max(dc.x, dc.y);
    if (edge >= 0.98) return 1.0;
    float t = textureLod(CLOUD_SHADOW_MAP, fract(g / kCloudShadowExtent), 0.0).r;      // (fract: right whatever the sampler does outside 0..1)
    return 1.0 - k * (1.0 - smoothstep(0.78, 0.98, edge)) * (1.0 - clamp(t, 0.0, 1.0));
}
#endif

#endif
