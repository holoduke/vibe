// terrain_rm_pebbles.glsl - part of terrain_raymarch.frag (included there, in place; not a standalone shader).
// Procedural pebbles, lake-bed stones, foam cells

// ---- Procedural pebbles -------------------------------------------------
// Loose stones computed by formula (jittered cells, one lens-shaped stone
// per occupied cell), in three sizes: pebbles, gravel, grit. Each size
// fades in once its stones cover a few pixels, so the ground never runs
// out of detail however close the camera gets - there is always a finer
// layer underneath. They are real height on the raymarched ground (the
// ray lands ON the stones near the camera), not a picture.
vec3 peb_h3(vec2 c) {
    vec3 p3 = fract(vec3(c.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}
float g_peb_id = 0.0;    // random 0..1 of the stone at the last query
float g_peb_k  = 0.0;    // 0 at its rim .. 1 at its centre (0 = bare soil)
float g_peb_sz = 0.0;    // its layer: 1 pebble, 2 gravel, 3 grit, 4 big cobble
float peb_layer(vec2 p, float cell, float dens, float layer_id, float hf, inout float best) {
    // (a whole number of cells to the tile, and a cell's number wrapped there: the same stones
    // in every tile, so they lie where they lay when the engine re-centres - scene_ubo.glsl)
    float pn = max(round(kTile / cell), 1.0);
    vec2 q  = p * (pn / kTile);
    vec2 ip = floor(q), fq = fract(q);
    float h = 0.0;
    // A stone reaches at most 0.9 cell from its cell centre, so only the
    // 2x2 cells on this point's side of its own cell can cover it.
    vec2 gb = step(vec2(0.5), fq) - 1.0;
    for (int j = 0; j <= 1; ++j)
    for (int i = 0; i <= 1; ++i) {
        vec2 g = gb + vec2(float(i), float(j));
        vec3 r3 = peb_h3(tile_wrap(ip + g, pn) + layer_id * 71.3);
        if (r3.z > dens) continue;
        vec2  o  = g + 0.5 + (r3.xy - 0.5) * 0.55 - fq;
        float rr = mix(0.27, 0.50, fract(r3.z * 17.3));
        if (dot(o, o) > 1.55 * rr * rr) continue;             // cannot reach this point (cheap reject)
        float a  = r3.x * 6.2832;
        vec2  d  = vec2(cos(a) * o.x + sin(a) * o.y, -sin(a) * o.x + cos(a) * o.y);
        d.y *= mix(1.0, 1.75, r3.y);                         // oval stones
        // Shape: from round river pebbles to blocky broken stone.
        vec2  d2 = d * d;
        float bx = fract(r3.z * 29.1) * fract(r3.x * 3.3 + 0.37);
        float k  = 1.0 - mix(d2.x + d2.y, max(d2.x, d2.y) * 1.3, bx) / (rr * rr);
        if (k <= 0.0) continue;
        float hh = hf * cell * rr * mix(0.55, 1.05, fract(r3.x * 7.7)) * k * (1.75 - 0.75 * k);
        if (hh > h) h = hh;
        if (hh > best) { best = hh; g_peb_id = fract(r3.y * 13.7 + r3.x); g_peb_k = k; g_peb_sz = layer_id; }
    }
    return h;
}
// Foam lace: distance between the nearest and second-nearest cell point
// (small on the walls between cells). Foam sits on those walls and the
// cell centres are open water - the net of bubbles real surf leaves.
float foam_cells(vec2 p) {
    vec2 ip = floor(p), fp = fract(p);
    float f1 = 9.0, f2 = 9.0;
    for (int j = -1; j <= 1; ++j)
    for (int i = -1; i <= 1; ++i) {
        vec2 g = vec2(float(i), float(j));
        vec2 o = g + peb_h3(ip + g + 113.0).xy - fp;
        float d = dot(o, o);
        if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) f2 = d;
    }
    return sqrt(f2) - sqrt(f1);
}
// Lake / river bed: a bed of closely packed pebbles (after "Raymarching
// Pebbles", shadertoy wlsBRX: a pebble height field, shaded with its slope,
// darkened in the crevices between the stones, with a wet highlight).
// Two sizes, every cell occupied; the small ones fill the gaps.
float bed_h(vec2 wp, float wA, float wB, out float id, out float k) {
    float best = 0.0, h = 0.0;
    g_peb_id = 0.0; g_peb_k = 0.0;
    if (wA > 0.0) {
        h = wA * peb_layer(wp, 0.22, 1.0, 5.0, 1.0, best);
        h = max(h, wA * peb_layer(wp + vec2(0.11, 0.11), 0.22, 1.0, 7.0, 0.85, best));   // second set in the gaps: packed
    }
    if (wB > 0.0) h = max(h, wB * peb_layer(wp + 5.3, 0.075, 1.0, 6.0, 1.0, best));
    id = g_peb_id; k = g_peb_k;
    return h;
}
// One smooth random value per ~1/f metres (4 hashes; noise2 also computes
// derivatives nobody needs here).
// (of the position wp at about k cells a metre + off; the same in every tile)
float peb_vnoise(vec2 wp, float k, vec2 off) {
    float n = max(round(kTile * k), 1.0);
    vec2 p = wp * (n / kTile) + off;
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    vec2 i0 = tile_wrap(i, n), i1 = tile_wrap(i + 1.0, n);
    float a = peb_h3(i0).x, b = peb_h3(vec2(i1.x, i0.y)).x;
    float c = peb_h3(vec2(i0.x, i1.y)).x, d = peb_h3(i1).x;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
// 0 = everything (shading), 1 = while marching (only what moves the
// silhouette, near the camera), 2 = shadow walk (no grit), -1 = off.
int g_peb_mode = 0;
// The stone under the pixel being shaded (left by calcNormal's centre tap).
float g_pebN_h = 0.0, g_pebN_id = 0.0, g_pebN_k = 0.0, g_pebN_sz = 0.0;
// Height of the stones at wp (metres). t = distance along the view ray.
float rm_pebbles(vec2 wp, float t) {
    g_peb_k = 0.0; g_peb_sz = 0.0;
    if (g_peb_mode < 0 || t > 26.0) return 0.0;
    float fp = max(t, 0.05) * 2.0 * scene.viewport.w;        // ~metres per pixel
    float wA = smoothstep(2.5, 6.0, 0.110 / fp);
    float wB = smoothstep(2.5, 6.0, 0.034 / fp);
    float wC = smoothstep(2.5, 6.0, 0.010 / fp);
    float wD = 1.0;
    if (g_peb_mode == 1) {
        wA *= 1.0 - smoothstep(1.5, 2.5, t);
        wB = 0.0; wC = 0.0;
        wD = 1.0 - smoothstep(4.0, 6.0, t);
    } else if (g_peb_mode == 2) {
        wC = 0.0;
        wB *= 1.0 - smoothstep(2.0, 3.5, t);
    }
    float best = 0.0, h = 0.0;
    if (wA + wB + wC > 0.0) {
        // Stones lie in drifts: dense patches, thin scatter, bare soil.
        float dens = smoothstep(0.30, 0.75, peb_vnoise(wp, 0.45, vec2(7.1)));
        if (wA > 0.0) h = max(h, wA * peb_layer(wp, 0.110, 0.10 + 0.55 * dens, 1.0, 1.0, best));
        if (wB > 0.0) h = max(h, wB * peb_layer(wp + 3.7, 0.034, 0.15 + 0.60 * dens, 2.0, 1.0, best));
        if (wC > 0.0) h = max(h, wC * peb_layer(wp + 9.1, 0.010, 0.45 + 0.40 * dens, 3.0, 1.0, best));
    }
    // Big cobbles: fist- to head-sized stones, few, and in OTHER places
    // than the pebble drifts (their own patch pattern) - singles and
    // small groups half sunk in the soil.
    if (wD > 0.0) {
        float d2 = 0.03 + 0.30 * smoothstep(0.45, 0.80, peb_vnoise(wp, 0.30, vec2(31.7)));
        float sid = g_peb_id, sk = g_peb_k, ssz = g_peb_sz, bb = 0.0;
        float hb = wD * peb_layer(wp + 17.3, 0.420, d2, 4.0, 0.5, bb);
        if (hb > h) h = hb;
        else { g_peb_id = sid; g_peb_k = sk; g_peb_sz = ssz; }   // a smaller stone on top keeps its own look
    }
    return h;
}
// peb_layer with the slope of the stone it returns (d height / d p): the stone's shape is a
// formula, so its slope is too. The normal used three whole look-ups of every layer to get it.
float peb_layer_g(vec2 p, float cell, float dens, float layer_id, float hf, inout float best, out vec2 grad) {
    float pn = max(round(kTile / cell), 1.0);
    vec2 q  = p * (pn / kTile);
    vec2 ip = floor(q), fq = fract(q);
    float h = 0.0;
    grad = vec2(0.0);
    vec2 gb = step(vec2(0.5), fq) - 1.0;
    for (int j = 0; j <= 1; ++j)
    for (int i = 0; i <= 1; ++i) {
        vec2 g = gb + vec2(float(i), float(j));
        vec3 r3 = peb_h3(tile_wrap(ip + g, pn) + layer_id * 71.3);
        if (r3.z > dens) continue;
        vec2  o  = g + 0.5 + (r3.xy - 0.5) * 0.55 - fq;
        float rr = mix(0.27, 0.50, fract(r3.z * 17.3));
        if (dot(o, o) > 1.55 * rr * rr) continue;
        float a  = r3.x * 6.2832;
        float ca = cos(a), sa = sin(a), sy = mix(1.0, 1.75, r3.y);
        vec2  d  = vec2(ca * o.x + sa * o.y, (-sa * o.x + ca * o.y) * sy);
        vec2  d2 = d * d;
        float bx = fract(r3.z * 29.1) * fract(r3.x * 3.3 + 0.37);
        float k  = 1.0 - mix(d2.x + d2.y, max(d2.x, d2.y) * 1.3, bx) / (rr * rr);
        if (k <= 0.0) continue;
        float amp = hf * cell * rr * mix(0.55, 1.05, fract(r3.x * 7.7));
        float hh = amp * k * (1.75 - 0.75 * k);
        if (hh > h) {
            h = hh;
            float mx = step(d2.y, d2.x);
            vec2 dk = -mix(2.0 * d, 2.6 * d * vec2(mx, 1.0 - mx), bx) / (rr * rr);        // dk / dd
            vec2 ko = vec2(ca * dk.x - sa * sy * dk.y, sa * dk.x + ca * sy * dk.y);          // dk / do
            grad = ko * (-(pn / kTile) * amp * (1.75 - 1.5 * k));                             // (o moves against p)
        }
        if (hh > best) { best = hh; g_peb_id = fract(r3.y * 13.7 + r3.x); g_peb_k = k; g_peb_sz = layer_id; }
    }
    return h;
}
// rm_pebbles as the shading sees it (mode 0), with the slope of the stones' surface.
float rm_pebbles_g(vec2 wp, float t, out vec2 grad) {
    g_peb_k = 0.0; g_peb_sz = 0.0;
    grad = vec2(0.0);
    if (t > 26.0) return 0.0;
    float fp = max(t, 0.05) * 2.0 * scene.viewport.w;
    float wA = smoothstep(2.5, 6.0, 0.110 / fp);
    float wB = smoothstep(2.5, 6.0, 0.034 / fp);
    float wC = smoothstep(2.5, 6.0, 0.010 / fp);
    float best = 0.0, h = 0.0;
    vec2 g;
    if (wA + wB + wC > 0.0) {
        float dens = smoothstep(0.30, 0.75, peb_vnoise(wp, 0.45, vec2(7.1)));
        if (wA > 0.0) { float l = wA * peb_layer_g(wp, 0.110, 0.10 + 0.55 * dens, 1.0, 1.0, best, g); if (l > h) { h = l; grad = wA * g; } }
        if (wB > 0.0) { float l = wB * peb_layer_g(wp + 3.7, 0.034, 0.15 + 0.60 * dens, 2.0, 1.0, best, g); if (l > h) { h = l; grad = wB * g; } }
        if (wC > 0.0) { float l = wC * peb_layer_g(wp + 9.1, 0.010, 0.45 + 0.40 * dens, 3.0, 1.0, best, g); if (l > h) { h = l; grad = wC * g; } }
    }
    {
        float d2 = 0.03 + 0.30 * smoothstep(0.45, 0.80, peb_vnoise(wp, 0.30, vec2(31.7)));
        float sid = g_peb_id, sk = g_peb_k, ssz = g_peb_sz, bb = 0.0;
        float hb = peb_layer_g(wp + 17.3, 0.420, d2, 4.0, 0.5, bb, g);
        if (hb > h) { h = hb; grad = g; }
        else { g_peb_id = sid; g_peb_k = sk; g_peb_sz = ssz; }
    }
    return h;
}
float rm_pebbles_march(vec2 wp, float t) {
    if (t > 6.0 || g_peb_mode < 0) return 0.0;
    int m = g_peb_mode;
    g_peb_mode = 1;
    float h = rm_pebbles(wp, t);
    g_peb_mode = m;
    return h;
}

// Depth (m) of the deepest footprint at world xz `p` (scene.footprints):
// a sole - the ball of the foot and a heel - pressed up to 12 cm into the snow.
#ifdef VIBE_SUMMER
#define g_snow 0.0
#else
float g_snow = 0.0;              // snow amount 0..1, read once per pixel (foot_bounds_init, first thing in main)
#endif
vec2  g_foot_c = vec2(0.0);      // bounding circle of all prints (foot_bounds_init)
float g_foot_r2 = -1.0;          // < 0: no prints
void foot_bounds_init() {
#ifndef VIBE_SUMMER
    g_snow = scene_snow();
#endif
    g_foot_r2 = -1.0;
    vec2 c = vec2(0.0); float n = 0.0;
    for (int i = 0; i < 32; ++i) if (scene.footprints[i].w > 0.0) { c += scene.footprints[i].xy; n += 1.0; }
    if (n < 0.5) return;
    c /= n;
    float r2 = 0.0;
    for (int i = 0; i < 32; ++i) if (scene.footprints[i].w > 0.0) { vec2 q = scene.footprints[i].xy - c; r2 = max(r2, dot(q, q)); }
    g_foot_c = c;
    g_foot_r2 = (sqrt(r2) + 0.35) * (sqrt(r2) + 0.35);
}
// Tracks in snow and sand (post_fx.cpp post_fx_tracks; track_stamp.comp, track_rim.comp): the
// depth pressed in by wheels and walkers and the raised rim beside it, in a 64 m window round
// the player whose corner track_stamp.comp writes into the post-effects block. (depth, rim) m.
#include "post_fx.glsl"
layout(set = 0, binding = 55) uniform sampler2D u_tracks;
layout(std430, set = 0, binding = 59) readonly buffer PfxTrkB { PostFxData d; } pfx_trk;
vec2 track_dr(vec2 p) {
#ifdef VIBE_BAKE
    return vec2(0.0);                // (they change: never baked - rm_bake_fetch leaves them to the pixel)
#else
    if (pfx_trk.d.trk_on < 0.5) return vec2(0.0);
    vec2 uv = (p - vec2(pfx_trk.d.trk_x, pfx_trk.d.trk_z)) * (1.0 / 64.0);
    if (any(lessThan(uv, vec2(0.004))) || any(greaterThan(uv, vec2(0.996)))) return vec2(0.0);
    return textureLod(u_tracks, uv, 0.0).xy;
#endif
}
float foot_prints(vec2 p);
// Depth (m) pressed into snow / sand at `p`: the deepest print or track, less half the rim
// beside a track (negative there: the snow pushed out of a rut stands up - the reference's
// displacement -depth + 0.5 rim).
float foot_depth(vec2 p) {
    vec2 tr = track_dr(p);
    return max(foot_prints(p), tr.x) - 0.5 * tr.y;
}
float foot_prints(vec2 p) {
    float d = 0.0;
    if (g_foot_r2 < 0.0 || dot(p - g_foot_c, p - g_foot_c) > g_foot_r2) return 0.0;
    for (int i = 0; i < 32; ++i) {
        vec4 f = scene.footprints[i];
        if (f.w <= 0.0) continue;
        vec2 q = p - f.xy;
        if (dot(q, q) > 0.06) continue;
        float c = cos(f.z), s = sin(f.z);
        float lx = c * q.x + s * q.y;              // across the foot
        float lz = s * q.x - c * q.y;              // along it (toes forward)
        float ball = (length(vec2(lx / 0.058, (lz - 0.055) / 0.105)) - 1.0) * 0.058;
        float heel = length(vec2(lx, lz + 0.105)) - 0.047;
        float k = 0.03, h = clamp(0.5 + 0.5 * (heel - ball) / k, 0.0, 1.0);
        float sdf = mix(heel, ball, h) - k * h * (1.0 - h);          // smooth union
        d = max(d, f.w * 0.12 * smoothstep(0.014, -0.020, sdf));
    }
    return d;
}

// Blown sand on the road: it lies here and there in drifts a few cm thick on the paving - a
// layer of its own in the marched height (like the snow), so it has an outline, buries the
// cobbles and takes footprints. 0..1 where it lies; none under snow.
float sand_road_patch(vec2 wp) {
    float a = tile_noise2(wp, 0.085, vec2(53.0));                                   // stretches of road with sand
    if (a < 0.50) return 0.0;
    float b = tile_noise2(wp, 0.42, vec2(21.0)) * 0.72 + tile_noise2(wp, 1.7, vec2(8.0)) * 0.28;
    return smoothstep(0.50, 0.62, a) * smoothstep(0.43, 0.55, b) * (1.0 - smoothstep(0.05, 0.3, g_snow));    // (a drift ends in a lip, not a long fade)
}
// Its height (m) above the paving, prints pressed in.
float sand_road_h(vec2 wp) {
    float p = sand_road_patch(wp);
    if (p <= 0.0) return 0.0;
    float wn = tile_noise2(wp, 0.6, vec2(3.0));
    // drift, the lumps in it, and wind ripples with a sharp crest (a gentle windward side, a steep lee)
    float rs = fract((dot(wp, vec2(21.0, 9.0)) + 7.0 * wn) * 0.15915);
    float rr = rs < 0.72 ? rs / 0.72 : (1.0 - rs) / 0.28;
    float s = p * (0.030 + 0.055 * tile_noise2(wp, 1.1, vec2(5.0)) + 0.010 * tile_noise2(wp, 4.3, vec2(11.0)) + 0.007 * rr * rr);
    float fd = foot_depth(wp);
    if (fd > 0.0) s -= min(fd * 0.45, s * 0.85);
    return s;
}

// The fine wind ripples of the sand (about 9 cm apart, too fine for the march: they go into
// the normal and the colour). 0 in the trough .. 1 on the crest; they fork and die out.
float sand_fine_ripple(vec2 wp, float wn) {
    float ph = (dot(wp, vec2(64.0, 24.0)) + 10.0 * wn + 3.0 * tile_noise2(wp, 2.3, vec2(7.0))) * 0.15915;
    float rs = fract(ph);
    float rr = rs < 0.58 ? rs / 0.58 : (1.0 - rs) / 0.42;        // (a rounder ripple: the lee was a third of the windward side - a hard edge)
    return rr * rr * (3.0 - 2.0 * rr);
}

// ---- Lake ice (winter) --------------------------------------------------
// Fake depth as in the "parallax ice" material shaders (Paragon's ice,
// the Godot CC0 "parallax ice for fake depth"): a frosted surface layer,
// and under it layers of fracture planes looked up where the line of sight
// - bent a little by the surface normal - reaches each depth. Here the
// layers are procedural (cell-border cracks), four deep.
float ice_cracks(vec2 p) {                 // 1 on a crack, 0 away from it
    vec2 ip = floor(p), fp = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = vec2(hash21(ip + g), hash21(ip + g + 17.3));
            float d = length(g + o - fp);
            if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d;
        }
    return 1.0 - smoothstep(0.0, 0.028, d2 - d1);
}
vec3 ice_albedo(vec2 wp, vec3 rd, vec3 n, float water_depth) {
    vec2 shift = (rd.xz - n.xz * 0.35) / max(-rd.y, 0.22);        // metres sideways per metre of depth
    // The body of the ice: clear and dark over deep water, milky in the shallows.
    vec3 col = mix(vec3(0.30, 0.50, 0.58), vec3(0.035, 0.11, 0.17), smoothstep(0.2, 2.5, water_depth));
    // Fracture planes, deepest first (each layer hides part of the one below).
    for (int i = 3; i >= 0; --i) {
        float dpt = 0.05 + 0.13 * float(i);
        vec2  q = wp + shift * dpt;
        q += (vec2(noise2(q * 0.55 + 3.0), noise2(q * 0.55 + 41.0)) - 0.5) * 1.1;          // cracks wander, they are not ruled lines
        float c = ice_cracks(q * (0.42 + 0.17 * float(i)) + 9.7 * float(i));
        c = max(c, 0.5 * ice_cracks(q * 2.3 + 4.0 * float(i)) * smoothstep(0.58, 0.70, noise2(q * 0.4 + float(i))));   // fine crazing in places
        col = mix(col, mix(vec3(0.80, 0.93, 1.0), vec3(0.30, 0.55, 0.70), float(i) / 3.0), c * exp(-dpt * 4.5) * 0.75);
    }
    // Air bubbles frozen in, a little way down.
    {
        vec2 q = (wp + shift * 0.09) * 14.0;
        vec2 ip = floor(q);
        float h = hash21(ip);
        vec2 o = vec2(h, hash21(ip + 5.1));
        float bub = (1.0 - smoothstep(0.05, 0.16 * h + 0.06, length(fract(q) - o))) * step(0.72, hash21(ip + 9.3)) *
                    smoothstep(0.45, 0.7, tile_noise2(wp, 0.35, vec2(3.0)));
        col = mix(col, vec3(0.85, 0.95, 1.0), bub * 0.7);
    }
    // The surface: frost and scuffs, and wind-blown snow lying on it.
    float frost = tile_noise2(wp, 2.7, vec2(0.0)) * 0.5 + tile_noise2(wp, 9.0, vec2(0.0)) * 0.3 + tile_noise2(wp, 0.33, vec2(0.0)) * 0.2;
    col = mix(col, vec3(0.62, 0.78, 0.86), smoothstep(0.45, 0.85, frost) * 0.45);
    float snow = smoothstep(0.52, 0.66, tile_noise2(wp, 0.21, vec2(11.0)) * 0.7 + tile_noise2(wp, 1.3, vec2(0.0)) * 0.3);
    col = mix(col, vec3(0.90, 0.93, 0.97), snow * 0.9);
    return col;
}
