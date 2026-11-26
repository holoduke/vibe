// terrain_rm_noise_road.glsl - part of terrain_raymarch.frag (included there, in place; not a standalone shader).
// Noise, plateau weight, road distance

// === Hash + Value-Noise with analytical derivatives ===
// dot/fract pattern (avoids sin() precision artifacts on some GPUs).
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

// Value Noise + ∂n/∂x, ∂n/∂y. Hermite smoothstep `3t²-2t³` for C¹
// continuity; `du = 6t(1-t)` is the chain-rule derivative used by the
// FBM-erosion accumulator. (Reverted from a quintic experiment —
// kept the proven cubic that matches the look the project shipped
// with and stays consistent with the CPU rm_noised bake.)
vec3 noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = hash21(i + vec2(0.0, 0.0));
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    float value = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    vec2  deriv = du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    return vec3(value, deriv);
}

float noise2(vec2 p) { return noised(p).x; }
// noised() for octave `oct` of the flat FBM: at home as it is; from another chart with that
// octave's lattice cell and position added (cells wrapped to +-2^22: exact in a float, and
// the same cell for a place from whichever chart it is seen).
vec3 rm_noised_i(vec2 p, int oct) {
    if (scene.ring_info.x < 0.5) return noised(p);
    vec4 lc = scene.ring_cell[min(oct, 23)];
    p += lc.xy;
    vec2 i = floor(p) + lc.zw;
    i -= 8388608.0 * floor(i / 8388608.0 + 0.5);
    vec2 f = fract(p);
    vec2 u  = f * f * (3.0 - 2.0 * f);
    vec2 du = 6.0 * f * (1.0 - f);
    float a = hash21(i + vec2(0.0, 0.0));
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    float value = a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y;
    vec2  deriv = du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx);
    return vec3(value, deriv);
}

// rm_noised_i with the second derivatives as well (hxx, hyy, hxy): what the landscape's slope
// needs to be worked out in the same pass as its height (terrainB_slope).
vec3 rm_noised_h(vec2 p, int oct, out vec3 hess) {
    vec2 i;
    if (scene.ring_info.x < 0.5) i = floor(p);
    else {
        vec4 lc = scene.ring_cell[min(oct, 23)];
        p += lc.xy;
        i = floor(p) + lc.zw;
        i -= 8388608.0 * floor(i / 8388608.0 + 0.5);
    }
    vec2 f = fract(p);
    vec2 u   = f * f * (3.0 - 2.0 * f);
    vec2 du  = 6.0 * f * (1.0 - f);
    vec2 ddu = 6.0 - 12.0 * f;
    float a = hash21(i + vec2(0.0, 0.0));
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    float k = a - b - c + d;
    vec2 lin = vec2(b - a, c - a) + k * u.yx;
    hess = vec3(ddu * lin, du.x * du.y * k);
    return vec3(a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y, du * lin);
}

// Per-octave ~37° rotation breaks axis-aligned banding. Unit
// determinant = pure rotation, no scaling.
const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);

const float TERRAIN_SCALE  = 0.003;
const float TERRAIN_HEIGHT = 120.0;

// The shape of the world: shaders/world_map.glsl (shared with the planet's far field in
// compose.frag). Every height the landscape formula makes passes through it.
#define WM_NOISE(p) noise2(p)
#define WM_CELL(i) scene.globe_cell[i].xyz
#define WM_FRAC(i) scene.globe_frac[i].xyz
#include "world_map.glsl"
#include "water_course.glsl"     // the stream: water above the sea, its channel (wc_stream_carve in rm_world_h)
#ifndef RM_WATER_LEVEL
#define RM_WATER_LEVEL scene.water_params.y
#endif
// The globe (world_map.glsl, engine/globe.h). World positions here are the engine's chart
// coordinates: in home's chart they are home coordinates and everything is as it always was;
// in a far chart (another part of the planet) the world is made from the point on the sphere.
bool rm_far_chart() { return scene.globe_info.x > 0.5; }
// Home's flat landscape formula from a chart that is not home's (engine/globe.h Ring): the
// formula is a function of home coordinates, which are large out there - so they are never
// formed for its FBM. The FBM runs on the offset from the chart's anchor (small) and takes
// the anchor's share per octave as a lattice cell and a position in it (scene.ring_cell).
bool rm_ring() { return scene.ring_info.x > 0.5; }
vec2 rm_ring_l(vec2 wp) { vec2 l = wp - scene.ring_info.yz; return vec2(dot(scene.ring_j.xy, l), dot(scene.ring_j.zw, l)); }
// Home coordinates of a chart point (for the formula's slow parts: continents, ranges, woods).
vec2 rm_ring_hp(vec2 wp) { return scene.ring_h0.xy + rm_ring_l(wp); }
// Where the flat FBM starts for the point wp: home's scaled coordinates at home, the scaled
// offset from the anchor elsewhere.
vec2 rm_flat_p0(vec2 wp) { return (rm_far_chart() ? rm_ring_l(wp) : wp) * 0.003; }
// Whether the flat FBM is worked out here at all (home, or a chart within its reach).
bool rm_flat_on() { return !rm_far_chart() || rm_ring(); }
// Profiling: VIBE_RM_SKIP=<mask> leaves a section of this shader out at run time (a real
// branch on a uniform, so the compiler cannot fold the rest away - cutting the shader short
// with an early return measures nothing: what no longer reaches the output is removed).
//   1 close-up ground textures   2 stones in the normal      4 the ground's own sun shadow
//   8 grass blade shadows       16 small plants' shadow map  32 ray-traced object shadows
// 512 screen-space normal      1024 close-range ground relief   32768 stones in the march
// 128 the relief left out of the march (before any of its work)
// 65536 object shadows only where the ground faces the sun (as before 2026-11-24)
// 131072 water flat (the shading test)
// 524288 the normal's relief from the main rills alone   1048576 one tap of the relief in the normal, not three
// 2097152 view: the cloud shadow map on the ground (white = open sky)
// 4194304 no ground material (one colour, no road or highway surface; with 1 + 2 + 512: what the light alone costs)
// 8388608 view: the running water's white (r: round rocks and piers, g: all white water, b: the fast-water term B)
// 16777216 view: the running water's carried ripple pattern (its normal's x red, z green), to measure how fast it moves
// 33554432 view: what the water mirrors (before the Fresnel); 92274688 (= 67108864 + 25165824): the running water's pattern frame (r: fract of metres down the course / 12.5, g: across / 2.2) and streak noise (b)
#ifdef VIBE_LITE
// The "first start" build of this shader (see VIBE_LITE in terrain_raymarch.frag): the sections
// in VIBE_LITE_MASK are compiled out for good.
#define RMDBG(bit) (((VIBE_LITE_MASK) & (bit)) != 0)
#else
#define RMDBG(bit) ((int(scene.globe_info.w) & (bit)) != 0)
#endif
// Distance from home for the rules that tell home's lakes from the open sea (a far chart is far).
float rm_home_dist(vec2 wp) { return rm_far_chart() ? (scene.ring_info.x > 0.5 ? length(scene.ring_h0.xy + vec2(dot(scene.ring_j.xy, wp - scene.ring_info.yz), dot(scene.ring_j.zw, wp - scene.ring_info.yz))) : 1.0e9) : length(wp); }
// A chart position's ground point, as an offset (m) from the chart origin's ground point, in
// the chart's frame (east, up, south): the chart unrolls the sphere, this rolls it back.
vec3 rm_chart_local(vec2 wp) {
    vec2 l = vec2(wp.x - scene.globe_info.y, wp.y - scene.ring_info.w);
    float d = length(l);
    if (d < 1.0) return vec3(l.x, 0.0, l.y);
    float th = d / wm_R, sh = sin(0.5 * th), k = wm_R * sin(th) / d;
    return vec3(l.x * k, -2.0 * wm_R * sh * sh, l.y * k);
}
vec3 rm_planet_G(vec2 wp) { return normalize(mat3(scene.globe_frame) * (rm_chart_local(wp) + vec3(0.0, wm_R, 0.0))); }
// The landscape FBM's value: `a` as the flat formula made it, handed over to the sphere's
// between 40 and 90 km from home (a far chart is all sphere).
float rm_shape(float a, vec2 wp, float oct_f) {
    if (rm_far_chart()) {
        if (!rm_ring()) return wm_shape3(mat3(scene.globe_frame) * rm_chart_local(wp), oct_f);
        float rd2 = length(rm_ring_hp(wp));                 // (within reach of home's flat formula: `a` is that)
        if (rd2 < wm_blend0) return a;
        return mix(a, wm_shape3(mat3(scene.globe_frame) * rm_chart_local(wp), oct_f), smoothstep(wm_blend0, wm_blend1, rd2));
    }
    float d2 = dot(wp, wp);
    if (d2 < wm_blend0 * wm_blend0) return a;
    return mix(a, wm_shape3(mat3(scene.globe_frame) * rm_chart_local(wp), oct_f), smoothstep(wm_blend0, wm_blend1, sqrt(d2)));
}
// A site's ground (engine/globe.h SiteShape; terrain.cpp site_ground - keep in step): level
// round its centre at `rise` above the water, sloping to the natural ground by `slope`.
float rm_site_ground(float h, vec2 wp) {
    if (scene.site_ground.z < 0.5) return h;
    float r = distance(wp, scene.site_ground.xy);
    if (r >= scene.site_shape.y) return h;
    return mix(h, RM_WATER_LEVEL + scene.site_shape.z, 1.0 - smoothstep(scene.site_shape.x, scene.site_shape.y, r));
}
// A highway (engine/highway.h): the field round the camera holds, for every 4 m, the signed
// distance to the nearest road line, the road's height there and the distance along it; all
// three are linear across a straight stretch, so the bilinear read is exact.
bool rm_hw_in(vec2 wp) {
    return scene.hw_info.z > 0.5 && all(greaterThan(wp, scene.hw_bbox.xy)) && all(lessThan(wp, scene.hw_bbox.zw));
}
// The distance is in the ROAD'S OWN MEASURE: metres divided by the road's width against a
// highway's (highway.h nearest). 5.5 is the asphalt's edge and 8 the verge's on any road, so
// the shelf, the relief's cut-out and the surface's layout need not know how wide a road is;
// what is laid out in real metres (wheel tracks, paint, cracks) asks rm_hw_width.
float rm_hw_width(vec2 wp) {         // 1 a highway; 0.364 the castle lane
    return max(textureLod(u_normal[31], (wp - scene.hw_info.xy) / scene.hw_info.w + 0.5, 0.0).a, 0.2);
}
vec2 rm_hw_dh(vec2 wp) {             // x: signed distance to the centre line (the road's measure), y: the road's height
    vec2 uv = (wp - scene.hw_info.xy) / scene.hw_info.w + 0.5;
    vec4 f = textureLod(u_normal[30], uv, 0.0);
    vec2 dh = vec2((f.r * 65280.0 + f.g * 255.0) * (128.0 / 65535.0) - 64.0, (f.b * 65280.0 + f.a * 255.0) * (512.0 / 65535.0) - 64.0);
    // Under a bridge the field says "far" with the side's sign kept (highway.cpp bake): -64 on
    // one side of the centre line, +64 on the other. Read bilinearly that goes THROUGH ZERO
    // along the centre line - a strip of road two metres wide under the whole deck, shelf and
    // asphalt: a wall across the river. So a distance that reads near is checked against the
    // four texels it was made from: all four "far" (the top byte 0 or 255) is no road.
    // (highway.cpp ground() - keep in step.)
    if (abs(dh.x) < 34.0) {
        vec4 g = textureGather(u_normal[30], uv, 0);
        if (all(lessThan(min(g, 1.0 - g), vec4(0.002)))) dh.x = 64.0;
        // The same between two ROADS: where the nearest road changes from one texel to the next
        // (beside a junction: along a line some 20 degrees off the side road) the one's distance
        // may be +40 and the other's -40, and the read between them goes through zero - a streak
        // of asphalt, paint and shelf across the field. A real centre line has a texel within
        // 16 on each side of it (the road's measure; 5.7 m - a texel's diagonal - on a road as
        // narrow as the castle lane, the narrowest this holds for); four texels of both signs
        // that are all further off than that are two roads, not one.
        vec4 gd = g * 127.5 - 63.75;
        if (min(min(gd.x, gd.y), min(gd.z, gd.w)) < -16.5 && max(max(gd.x, gd.y), max(gd.z, gd.w)) > 16.5
            && all(greaterThan(abs(gd), vec4(16.5)))) dh.x = 64.0;
    }
    return dh;
}
float rm_hw_s(vec2 wp) {             // metres along the road (from a reference that keeps the dashes' place)
    vec3 f = textureLod(u_normal[31], (wp - scene.hw_info.xy) / scene.hw_info.w + 0.5, 0.0).rgb;
    return (f.r * 16711680.0 + f.g * 65280.0 + f.b * 255.0 - 8388608.0) * 0.001;
}
// ---- A junction: a side road leaving a highway (engine/highway.h Junction: the rule and the
// numbers are written there; highway.cpp ground(), junction_lane_h, junction_ad - keep in step).
// Round it the field holds the highway alone; the side road is worked out here from the
// record in scene.hw_jn and laid over the highway's ground.
bool  rm_jn_on() { return scene.hw_jn[1].x > 0.5; }
// u: metres from J along the side road, v: metres to the side of its axis.
vec2  rm_jn_uv(vec2 wp) { vec2 r = wp - scene.hw_jn[0].xy, w = scene.hw_jn[0].zw; return vec2(dot(r, w), dot(r, vec2(-w.y, w.x))); }
// (where the surface is the junction's: it hands back to the field at 43.5 m, where the field's side road is whole again)
bool  rm_jn_surf_in(vec2 uv) { return uv.x > -8.0 && uv.x < 43.5 && abs(uv.y) < 24.0; }
// Metres from the highway's centre line: the field's distance - not in the side road's corridor
// beyond 29 m, where the field's texels turn into the side road's (the highway has ended by there).
float rm_jn_xh(vec2 uv, float d) { return (uv.x > 29.0 && abs(uv.y) < 18.0) ? 64.0 : abs(d); }
float rm_jn_cr(float p0, float p1, float p2, float p3, float t) {
    float t2 = t * t, t3 = t2 * t;
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}
float rm_jn_hs(int i) { return i < 4 ? scene.hw_jn[2][i] : scene.hw_jn[3][i - 4]; }
// The side road's centre line height u metres from J: its line is a Catmull-Rom curve through
// points 8 m apart, cut into four straight pieces a span - the same pieces here.
float rm_jn_lane_h(float u) {
    u = clamp(u, 0.0, 47.999);
    int i = int(u / 8.0), a = max(i - 1, 0);
    float h0 = rm_jn_hs(a), h1 = rm_jn_hs(i), h2 = rm_jn_hs(i + 1), h3 = rm_jn_hs(i + 2);
    float u0 = 8.0 * float(a), u1 = 8.0 * float(i), u2 = u1 + 8.0, u3 = u1 + 16.0;
    float ua = u1, ha = h1;
    for (int k = 1; k <= 4; ++k) {
        float t = 0.25 * float(k);
        float ub = k == 4 ? u2 : rm_jn_cr(u0, u1, u2, u3, t), hb = k == 4 ? h2 : rm_jn_cr(h0, h1, h2, h3, t);
        if (u <= ub || k == 4) return mix(ha, hb, clamp((u - ua) / max(ub - ua, 1e-4), 0.0, 1.0));
        ua = ub; ha = hb;
    }
    return h2;
}
// The paving: the two roads' asphalt and the corners between them rounded to 4 m, as a
// distance in a road's measure (5.5 = the edge). xh: metres from the highway's centre line.
float rm_jn_ad(vec2 uv, float xh) {
    float k = scene.hw_jn[1].y, av = abs(uv.y);
    float a = xh - 5.5, b = uv.x > 0.0 ? av - 5.5 * k : 1.0e3, r = 4.0;
    float sd = max(r, min(a, b)) - length(max(vec2(r - a, r - b), vec2(0.0)));
    float lw = smoothstep(5.5, 9.5, xh) * (1.0 - smoothstep(8.0, 12.0, av));      // (the side road's measure where it is its edge)
    return 5.5 + sd / mix(1.0, k, lw);
}
// The ground: h the land, hr the highway's shelf on it (the field's), xh metres from the highway.
float rm_jn_ground(float h, float hr, float xh, vec2 uv) {
    float u = uv.x, av = abs(uv.y), k = scene.hw_jn[1].y;
    float hh = mix(hr, h, smoothstep(25.0, 29.0, u) * (1.0 - smoothstep(16.0, 20.0, av)));
    float res = hh;
    float adl = av / k;
    if (u > 0.0 && adl < 30.0) {
        // (nothing of the side road inside the highway's asphalt, all of it 8 m beyond: one surface)
        float m = u > 20.0 ? 1.0 : smoothstep(5.5, 13.5, xh);
        float fall = 0.02 * min(adl, 5.5) + 0.05 * clamp(adl - 5.5, 0.0, 2.5);
        res = mix(hh, rm_jn_lane_h(u) - fall, (1.0 - smoothstep(8.0, 30.0, adl)) * m);
    }
    return u > 41.0 ? mix(res, hr, min((u - 41.0) / 5.0, 1.0)) : res;
}
// The ground with the road's shelf: level to the verge (8 m), met by the natural ground at 30 m
// (highway.cpp ground() - keep in step).
float rm_highway(float h, vec2 wp) {
    if (!rm_hw_in(wp)) return h;
    vec2 dh = rm_hw_dh(wp);
    float w = 1.0 - smoothstep(8.0, 30.0, abs(dh.x));
    // (the crown: 2 % from the centre line to the asphalt's edge, 5 % over the verge - highway.cpp cross_fall)
    float ad = abs(dh.x);
    float fall = 0.02 * min(ad, 5.5) + 0.05 * clamp(ad - 5.5, 0.0, 2.5);
    float hr = w <= 0.0 ? h : mix(h, dh.y - fall, w);
    // Beside the service station (terrain.h kStationEdge; highway.cpp ground() - keep in step) the
    // shelf's 22 m of cut and fill would run through the forecourt. There the ground is the
    // terrace up to the plot's edge, and from that edge a straight ramp to the road's verge.
    // (Not where there is no road in the ground - under the bridge south of the station the
    //  distance reads "far": the ramp was then drawn toward a road 64 m off, a sunken flat
    //  with walls, while the CPU's ground had none: grass and stones hung over the hole. And
    //  the rule ends before the bridge begins (z 565..573; the span starts at 577).)
    if (wp.y > 503.0 && wp.y < 573.0 && wp.x < 1654.0 && ad > 8.0 && ad < 60.0) {
        float a = wp.x - 1642.6, b = ad - 8.0;
        float k = a <= 0.0 ? 0.0 : smoothstep(0.0, 1.0, a / (a + b));
        float hz = a <= 0.0 ? h : mix(41.825, dh.y - fall, k);
        hr = mix(hr, hz, smoothstep(503.0, 509.0, wp.y) * (1.0 - smoothstep(565.0, 573.0, wp.y)));
    }
    if (rm_jn_on()) {
        vec2 uv = rm_jn_uv(wp);
        if (uv.x > -8.0 && uv.x < 46.0 && abs(uv.y) < 24.0) hr = rm_jn_ground(h, hr, abs(dh.x), uv);
    }
    return hr;
}
// "How far from the asphalt", as a distance from a centre line in a road's measure (5.5 = the
// paving's edge, 8 = the verge's): the field's distance, and at a junction the two roads
// and the rounded corners between them. d: rm_hw_dh's distance here.
float rm_hw_ad(vec2 wp, float d) {
    if (!rm_jn_on()) return abs(d);
    vec2 uv = rm_jn_uv(wp);
    if (!rm_jn_surf_in(uv)) return abs(d);
    return rm_jn_ad(uv, rm_jn_xh(uv, d));
}
// The surface at a junction: whose asphalt this is, and the paint that belongs to the junction.
// In: d, s, wk as the field has them (the highway's, round a junction). Out: the same of the
// road whose surface is here - the highway's inside its own paving, the side road's beyond -
// and mk for rm_hw_asphalt: x = how much of the edge line is there (broken across the mouth),
// y = paint laid across (the stop line: a bar 45 cm wide a metre short of the highway's edge
// line, and the side road's own edge lines up to it).
void rm_hw_jn_surface(vec2 wp, float t, inout float d, inout float s, inout float wk, out vec2 mk) {
    mk = vec2(1.0, 0.0);
    if (!rm_jn_on()) return;
    vec2 uv = rm_jn_uv(wp);
    if (!rm_jn_surf_in(uv)) return;
    float u = uv.x, av = abs(uv.y), k = scene.hw_jn[1].y;
    float xh = rm_jn_xh(uv, d);
    float pw = max(t * 0.0022, 0.01);
    if (u <= 0.0) return;
    if (xh < 5.5) {
        float in_m = clamp((5.5 * k + 4.0 - av) / pw + 0.5, 0.0, 1.0);                       // (the mouth: the lane and its two corners)
        float dash = clamp((0.25 - abs(fract(uv.y * 0.5) - 0.5)) * 2.0 / pw + 0.5, 0.0, 1.0);    // (1 m of every 2)
        mk.x = 1.0 - in_m * (1.0 - dash);
        float bar  = clamp((0.225 - abs(xh - 5.025)) / pw + 0.5, 0.0, 1.0) * clamp((5.5 * k - 0.2 - av) / pw + 0.5, 0.0, 1.0) * min(1.0, 0.45 / pw);
        float stub = clamp((0.05 - abs(av - 1.74)) / pw + 0.5, 0.0, 1.0) * min(1.0, 0.10 / pw) * step(5.25, xh);
        mk.y = max(bar, stub);
        return;
    }
    if (av < 12.0) { d = uv.y / k; s = scene.hw_jn[1].z + u; wk = k; }
}
// Land or sea (.x, wm_continent's value) and the ranges (.y) at a chart position, as the
// height formula reads them - from the formula itself.
vec2 rm_world_vr_f(vec2 wp) {
    if (!rm_far_chart()) return vec2(wm_continent(wp * 0.001), wm_ranges(wp * 0.001));
    vec3 G = rm_planet_G(wp);
    vec2 mk = (rm_ring() ? rm_ring_hp(wp) : wm_az(G)) * 0.001;
    return vec2(wm_continent_globe(mk, G), wm_ranges_globe(mk, G));
}
// The same from the two pictures drawn round the camera (vk_engine/wm_cache.cpp: 4 m a texel
// out to 4 km, 160 m a texel out to 82 km, both made by rm_world_vr_f). Both fields change
// over hundreds of metres; the formula is 23 to 31 noise look-ups, and it was paid at every
// step of the march and three times for the normal. This shader no longer holds it.
// The four texels are summed here, not by the sampler: a filtered read moves in 1/256 steps
// of a texel, and the normal takes its slope over 2 cm.
#ifdef WM_CACHE_BAKE
vec2 rm_world_vr(vec2 wp) { return rm_world_vr_f(wp); }
#else
layout(set = 0, binding = 39) uniform sampler2D u_wm_cache;
layout(set = 0, binding = 40) uniform sampler2D u_wm_cache1;
vec2 rm_wm_tex(sampler2D s, vec4 c, vec2 wp) {
    vec2  q = clamp((wp - c.xy) * c.z, vec2(0.0), vec2(c.w - 1.001));
    ivec2 i = ivec2(q);
    vec2  f = q - vec2(i);
    vec2 a = texelFetch(s, i, 0).rg, b = texelFetch(s, i + ivec2(1, 0), 0).rg;
    vec2 d = texelFetch(s, i + ivec2(0, 1), 0).rg, e = texelFetch(s, i + ivec2(1, 1), 0).rg;
    return mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
}
vec2 rm_world_vr(vec2 wp) {
    // (how many texels inside the near picture's edge; over its last 64 - 256 m - the far
    // picture is blended in, so the two meet without a step)
    vec2  q = (wp - scene.wm_cache.xy) * scene.wm_cache.z;
    float edge = min(min(q.x, q.y), scene.wm_cache.w - 1.0 - max(q.x, q.y));
    if (edge > 65.0) return rm_wm_tex(u_wm_cache, scene.wm_cache, wp);
    vec2 far = rm_wm_tex(u_wm_cache1, scene.wm_cache1, wp);
    if (edge <= 1.0) return far;
    return mix(far, rm_wm_tex(u_wm_cache, scene.wm_cache, wp), (edge - 1.0) / 64.0);
}
#endif
// (wm_height and wm_height_globe of world_map.glsl, with the two slow fields from rm_world_vr)
float rm_world_h(float a, vec2 wp) {
    if (!rm_far_chart()) {
        float h = a * 120.0;
        float w = smoothstep(1450.0, 1900.0, length(wp));     // (as wm_height_land: home's own formula within)
        if (w > 0.0) { vec2 vr = rm_world_vr(wp); h = wm_height_core(h, a, w, vr.y, vr.x, RM_WATER_LEVEL); }
        return rm_highway(wc_stream_carve(wm_river(wm_pad(h, wp), wp, RM_WATER_LEVEL), wp), wp);      // (CPU: terrain.cpp world_ground_height_impl, rm_terrain)
    }
    vec2 vr = rm_world_vr(wp);
    return rm_site_ground(rm_highway(wm_height_core(a * 120.0, a, 1.0, vr.y, vr.x, RM_WATER_LEVEL), wp), wp);
}
float rm_continent(vec2 wp) { return rm_world_vr(wp).x; }
float rm_forest(vec2 wp, float h, float ny, float wl) {
    bool far = rm_far_chart();
    if (far && scene.site_ground.z > 0.5 && distance(wp, scene.site_ground.xy) < scene.site_shape.w) return 0.0;   // (a site's own ground: no woods)
    vec2 hpf = (far && rm_ring()) ? rm_ring_hp(wp) : wp;
    if ((!far || rm_ring()) && dot(hpf, hpf) < wm_blend0 * wm_blend0) return wm_forest(hpf, h, ny, wl);
    vec3 G = rm_planet_G(wp);
    return wm_forest_globe((far && !rm_ring()) ? wm_az(G) : hpf, G, h, ny, wl);
}


// High-frequency FBM-eroded noise used by sampleHeightDelta to paint
// procedural detail onto user-brushed cells. Same n.x / (1 + |∇d|²)
// erosion recipe as terrainM, but with a much higher base scale (0.05
// vs 0.003) so meter-scale ridges actually appear within the brush
// footprint. Centred ≈ 0 so it adds equal positive/negative variation
// — doesn't shift the mean brushed height. Forward-declared above
// sampleHeightDelta; the body lives here so it can use the noise
// helpers + m2 declared above.
float terrainDetailNoise(vec2 worldXZ) {
    vec2 p = worldXZ * 0.05;   // 20 m base; ≤ 30 cm at 6 octaves
    float a = 0.0, b = 1.0;
    vec2 d = vec2(0.0);
    for (int i = 0; i < 6; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        p = m2 * p * 2.0;
    }
    return a - 0.5;
}

// Plateau blend factor — 1 inside the gameplay plateau region, 0
// outside, smooth ramp across `kPlateauBlend` metres so the FBM
// surroundings ease into the flat castle pad. Mirrors what
// generate_heightmap() does in the rasterised path so the visible
// terrain shape matches what physics / castle assume.
const float kPlateauBlend = 20.0;  // smooth ramp from castle-flat to FBM mountains
float plateauWeight(vec2 worldXZ) {
    if (rm_far_chart()) return 0.0;                      // (home's castle site: not in another part of the planet)
    vec2 c = pc.color.xy;
    float ext = pc.color.z;
    vec2 d = abs(worldXZ - c) - vec2(ext);
    float dout = max(max(d.x, 0.0), max(d.y, 0.0));
    return 1.0 - smoothstep(0.0, kPlateauBlend, dout);
}

// Plateau ground-detail noise. The macro terrain FBM uses scales
// 0.003 → 0.096 (333 m → 10 m wavelengths) — its SMALLEST feature is
// ~10 m, so scaling it down for plateau use gives invisibly-small
// "detail" (a few cm). What the player perceives as "rocky ground"
// on the surrounding mountains is those 10 m ridges seen from far
// away; on the flat plateau they're not there at all.
//
// This separate noise fires AT FINER SCALES (2.5 m → 1.25 m
// wavelengths — safely above the ~0.8 m march-step alias floor)
// and adds visible "ground texture" to plateau cells without
// affecting anything outside. 2 octaves of plain noise (no
// derivative erosion damper — we WANT the high freqs to contribute
// fully). m2 (per-octave rotation declared further down) can't be
// forward-referenced — inlined.
float plateauGroundDetail(vec2 wp) {
    // 50/50 mix of plain noise (rolling) + soft ridge (1 - |2n-1|)
    // — gives natural rock character without the glass-flat smoothness
    // of pure FBM or the spike-needle look of multifractal-weighted
    // ridges. No squaring, no weight gating. 2 octaves at safe
    // wavelengths (2.5 m + 1.25 m).
    float a = 0.0, b = 1.0;
    vec2 p = wp * 0.4;
    for (int i = 0; i < 2; ++i) {
        float n = noised(p).x;
        float ridge  = 1.0 - abs(2.0 * n - 1.0);     // ∈ [0, 1]
        float smooth_n = n;                            // ∈ [0, 1]
        a += b * (mix(smooth_n, ridge, 0.5) - 0.5);  // re-centre on 0
        b *= 0.5;
        vec2 q = vec2( 0.8 * p.x - 0.6 * p.y,
                       0.6 * p.x + 0.8 * p.y);
        p = q * 2.0;
    }
    return a;
}

// Medium-detail FBM — used by the ray march. Derivative-erosion term
// `1/(1+dot(d,d))` suppresses high-frequency layers on steep slopes,
// producing realistic ridge/valley structure instead of fractal noise.
// Octave count is the per-frame quality knob (pc.tex_params.z, 4..9):
// last 3 of 9 contribute ~1.5 % of total amplitude so dropping them
// barely changes the shape but cuts ~30 % of the inner work.
// Close-range GEOMETRIC ground detail (real displacement of the marched
// surface, not a normal trick): erosion rills carved between raised
// ground, a finer second set, the scanned soil relief, lumps, and wind
// ripples on sand. Always >= 0 (the rendered surface must never drop
// below the baked heightmap that shadows / objects use) and faded out by
// 34 m, where it is sub-pixel anyway.
float rm_road_dist(vec2 q) {         // distance to the road centreline (1e9 outside its bounds)
    if (scene.road_info.x < 1.5) return 1e9;        // (no road in this chart)
    int rn = int(scene.road_info.x + 0.5);
    if (rn < 2 || any(lessThan(q, scene.road_bbox.xy)) || any(greaterThan(q, scene.road_bbox.zw))) return 1e9;
    float best = 1e9;
    for (int i = 0; i + 1 < rn && i < 63; ++i) {
        vec2 a = scene.road_pts[i].xz, ab = scene.road_pts[i + 1].xz - a;
        vec2 dv = q - (a + ab * clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0));
        best = min(best, dot(dv, dv));
    }
    return sqrt(best);
}
// Per-pixel cache: road distance is 1-Lipschitz, so d(q) >= d(q0) - |q - q0|.
// Callers only need the exact value below `thr`:
//  * far from the road: anything provably >= thr skips the loop;
//  * near / on the road (the eye standing on it: every near-ground march
//    step used to run all 63 segments): only the window of segments around
//    the last nearest one is tested. That is exact as long as it beats
//    the Lipschitz lower bound of every segment outside the window
//    (g_rd_out - |q - q0|); otherwise the full loop runs and re-anchors.
vec2  g_rd_q = vec2(1e9);
float g_rd_v = -1e9;
int   g_rd_i = -1;          // nearest segment at g_rd_q
float g_rd_out = 0.0;       // min distance at g_rd_q to segments outside the window
float rm_road_seg_d2(vec2 q, int i) {
    vec2 a = scene.road_pts[i].xz, ab = scene.road_pts[i + 1].xz - a;
    vec2 dv = q - (a + ab * clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0));
    return dot(dv, dv);
}
float rm_road_dist_thr(vec2 q, float thr) {
    if (scene.road_info.x < 1.5) return 1e9;        // (no road in this chart)
    float moved = length(q - g_rd_q);
    float lb = g_rd_v - moved;
    if (lb >= thr) return lb;
    int rn = int(scene.road_info.x + 0.5);
    const int kWin = 2;
    if (g_rd_i >= 0) {
        float d2 = 1e18;
        for (int i = max(g_rd_i - kWin, 0); i <= min(g_rd_i + kWin, rn - 2); ++i) d2 = min(d2, rm_road_seg_d2(q, i));
        float dl = sqrt(d2);
        if (dl <= g_rd_out - moved) return dl;        // provably the true nearest
    }
    g_rd_i = -1;
    if (rn < 2 || any(lessThan(q, scene.road_bbox.xy)) || any(greaterThan(q, scene.road_bbox.zw))) {
        // Outside the bbox: cache the distance to the box, a valid lower
        // bound for re-entry.
        vec2 ob = max(max(scene.road_bbox.xy - q, q - scene.road_bbox.zw), vec2(0.0));
        g_rd_q = q; g_rd_v = length(ob);
        return 1e9;
    }
    float best = 1e18; int bi = 0;
    for (int i = 0; i + 1 < rn && i < 63; ++i) {
        float d2 = rm_road_seg_d2(q, i);
        if (d2 < best) { best = d2; bi = i; }
    }
    float outd = 1e18;
    for (int i = 0; i + 1 < rn && i < 63; ++i)
        if (abs(i - bi) > kWin) outd = min(outd, rm_road_seg_d2(q, i));
    g_rd_q = q; g_rd_v = sqrt(best); g_rd_i = bi; g_rd_out = sqrt(outd);
    return g_rd_v;
}
float g_rm_ray_y = -1e9;   // see terrainM_lod


// Winter: where snow lies on the paving (0..1; irregular patches).
float snow_road_patch(vec2 wp) {
    float pn = noise2(wp * 0.8 + 4.0) * 0.55 + noise2(wp * 2.7 + 11.0) * 0.30 + noise2(wp * 9.0) * 0.15;
    return smoothstep(0.40, 0.56, pn);
}

// ---- A highway's asphalt (engine/highway.h) --------------------------------------------------
// What makes a road look used rather than drawn: it is laid in stretches that have aged
// differently, the wheels polish two tracks in each lane and oil darkens the strip between
// them, holes are cut out and patched with newer, blacker asphalt, cracks are sealed with tar,
// dust gathers toward the edge, and the paint is worn where tyres cross it.
//   d: signed distance from the centre line, s: metres along the road, t: distance from the eye,
//   tex: the scan's colour here, hgt: its height (0 pores .. 1 stone tops).
// Returns the colour; gloss: 0 matt .. 1 like tar; lvl: the surface's level as a factor on the
// asphalt's height for the screen-space displacement (1 the sheet, 0 the bottom of a crack,
// below 1 in the wheel ruts, above 1 on a patch laid on top).
float rm_hw_hash(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
// Plates: the plane cut into irregular cells (one a unit across). Returns how far p is from the
// nearest joint between two cells (0 on it), id: a number 0..1 for the cell p is in.
float rm_hw_plates(vec2 p, out float id) {
    vec2 n = floor(p), f = fract(p);
    float f1 = 8.0, f2 = 8.0; id = 0.0;
    for (int j = -1; j <= 1; ++j) for (int i = -1; i <= 1; ++i) {
        vec2 g = vec2(float(i), float(j)), c = n + g;
        float dd = length(g + vec2(hash21(c), hash21(c + 37.3)) - f);
        if (dd < f1) { f2 = f1; f1 = dd; id = hash21(c + 11.7); } else if (dd < f2) f2 = dd;
    }
    return f2 - f1;
}
// `wk`: the road's width (rm_hw_width). A road of one lane (wk < 0.75) has one pair of wheel
// tracks about its middle, an edge line a hand inside each edge, and no line down the middle.
// `mk`: a junction's paint (rm_hw_jn_surface): x = how much of the edge line is there, y = paint laid across.
vec3 rm_hw_asphalt(vec2 wp, float d, float s, float t, vec3 tex, float hgt, float wk, vec2 mk, out float gloss, out float lvl) {
    float ad = abs(d), lane = d < 0.0 ? -1.0 : 1.0;
    bool  one = wk < 0.75;
    float ar = ad * wk;                                   // metres from the centre line
    if (one) lane = 1.0;
    float near = 1.0 - smoothstep(4.0, 30.0, t), mid = 1.0 - smoothstep(60.0, 220.0, t);
    float fw = max(t * 0.0022, 0.004);                    // what a pixel covers, about
    vec3 a = tex * 0.60;
    a *= mix(0.84, 1.14, noise2(vec2(s * 0.011, d * 0.06 + lane * 3.0) + 3.0));
    a *= mix(0.93, 1.06, tile_noise2(wp, 0.9, vec2(0.0)));
    gloss = 0.22;
    float track = 1.0 - smoothstep(0.20, 0.62, one ? abs(ar - 0.80) : abs(abs(ad - 1.95) - 0.85));
    float oil = (1.0 - smoothstep(0.10, 0.50, one ? ar : abs(ad - 1.95))) * smoothstep(0.25, 0.7, noise2(vec2(s * 0.05, lane * 5.0)));
    a *= 1.0 + 0.11 * track - 0.20 * oil * mid;
    gloss += 0.16 * track + 0.10 * oil;
    lvl = 1.0 - 0.30 * track;
    {   // a patch: a rectangle of newer asphalt, in some 37 m stretches of a lane
        float cell = floor(s / 37.0), r = rm_hw_hash(cell * 1.7 + lane * 31.0);
        if (r > 0.70) {
            float r2 = rm_hw_hash(cell * 3.1 + 7.0), r3 = rm_hw_hash(cell * 5.3 + 2.0), r4 = rm_hw_hash(cell * 7.9 + 4.0);
            vec2 c = vec2((cell + 0.3 + 0.4 * r2) * 37.0, 0.9 + 2.2 * r3), hs = vec2(1.5 + 5.0 * r4, 0.7 + 0.9 * r2);
            vec2 q = abs(vec2(s, one ? ar + 0.9 : ad) - c) - hs;
            float sd = max(q.x, q.y);
            float in_p = 1.0 - smoothstep(-fw, fw, sd);
            a = mix(a, tex * 0.36 * vec3(0.96, 0.98, 1.04), in_p);
            gloss = mix(gloss, 0.30, in_p);
            lvl = mix(lvl, 1.35, in_p);
            float seam = (1.0 - smoothstep(0.012, 0.012 + fw, abs(sd))) * min(1.0, 0.03 / fw);     // its tarred joint
            a = mix(a, vec3(0.012), seam * 0.85);
            gloss = mix(gloss, 0.75, seam);
            lvl *= 1.0 - seam;
        }
    }
    {   // cracks across the lane, sealed with tar
        float per = 19.0, cell = floor(s / per + 0.5), r = rm_hw_hash(cell * 2.3 + lane * 11.0);
        if (r > 0.45) {
            float x = s - cell * per + (noise2(vec2(ar * 0.9, cell)) - 0.5) * 1.1 + (r - 0.7) * 6.0;
            float ext = 1.0 - smoothstep(3.2 + 2.0 * r, 3.6 + 2.0 * r, ad);
            float cr = (1.0 - smoothstep(0.016, 0.016 + fw, abs(x))) * min(1.0, 0.04 / fw) * ext;
            a = mix(a, vec3(0.013), cr * 0.9);
            gloss = mix(gloss, 0.8, cr);
            lvl *= 1.0 - cr;
        }
    }
    {   // a crack along the lane, wandering, in stretches
        float y = ad - (2.95 + 0.30 * (noise2(vec2(s * 0.045, lane * 9.0)) - 0.5) + 0.05 * (noise2(vec2(s * 0.9, lane)) - 0.5));
        float gate = smoothstep(0.55, 0.7, noise2(vec2(s * 0.013, lane * 4.0 + 8.0)));
        float cr = (1.0 - smoothstep(0.006, 0.006 + fw, abs(y))) * min(1.0, 0.02 / fw) * gate;
        a = mix(a, a * 0.25, cr);
        lvl *= 1.0 - cr;
    }
    // Wear that no texture carries, generated here (close by only; colour and level together, so
    // the screen-space displacement gives it depth): the sheet broken into plates where the
    // wheels run, and potholes.
    if (t < 40.0) {
        float cl = 1.0 - smoothstep(25.0, 40.0, t);
        vec2 p = vec2(s, d * wk);                          // metres along, metres across
        float gate = smoothstep(0.62, 0.78, noise2(vec2(s * 0.021, lane * 6.0 + 13.0))) * (0.35 + 0.65 * track) * cl;
        if (gate > 0.01) {
            float id;
            float e = rm_hw_plates(p * 5.5 + (noise2(p * 2.3) - 0.5) * 0.6, id);      // plates a hand or two across
            float w = 0.05 + fw * 5.5;
            float cr = (1.0 - smoothstep(w * 0.4, w, e)) * gate * min(1.0, 0.012 / fw);
            a *= mix(1.0, mix(0.86, 1.06, id), gate);       // (each plate its own shade, and settled its own way)
            a = mix(a, a * 0.22, cr);
            lvl = mix(lvl, lvl * mix(0.55, 1.05, id), gate) * (1.0 - cr);
            gloss *= 1.0 - 0.5 * cr;
        }
        float cell = floor(s / 53.0), r = rm_hw_hash(cell * 4.7 + lane * 17.0);
        if (r > 0.72) {                                     // a pothole, in some 53 m stretches: in a wheel track
            float r2 = rm_hw_hash(cell * 2.9 + 1.0), r3 = rm_hw_hash(cell * 6.1 + 3.0);
            float yc = one ? (d < 0.0 ? -0.80 : 0.80) : lane * (1.95 + (r3 > 0.5 ? 0.85 : -0.85));
            vec2 q = p - vec2((cell + 0.25 + 0.5 * r2) * 53.0, yc);
            float dist = length(q * vec2(0.7, 1.0)) + (noise2(q * 6.0 + cell) - 0.5) * 0.14 - (0.20 + 0.22 * r3);
            float in_h = (1.0 - smoothstep(-0.03 - fw, fw, dist)) * cl;
            float rim = (1.0 - smoothstep(0.0, 0.12, abs(dist))) * cl;
            a = mix(a, a * 0.70, rim * (1.0 - in_h));       // (the broken rim)
            a = mix(a, vec3(0.085, 0.078, 0.070) * (0.6 + 0.8 * hgt), in_h);          // the base course: grit
            gloss = mix(gloss, 0.05, in_h);
            lvl = mix(lvl, 0.0, in_h);
        }
    }
    // Dust and grit toward the edge: it lies in the pores.
    float dusty = smoothstep(4.2, 5.4, ad) * (0.45 + 0.55 * noise2(vec2(s * 0.08, lane * 2.0 + 1.0)));
    a = mix(a, vec3(0.21, 0.19, 0.16) * (1.25 - 0.6 * hgt), dusty * 0.55);
    gloss *= 1.0 - 0.6 * dusty;
    {   // Markings: thermoplastic, 15 cm - a solid line 3.75 m out on each side, a broken one
        // (3 m of every 12) down the middle; ragged at the edge, chipped off the stone tops and
        // worn where the wheels cross.
        float pw = max(fw, 0.01);
        float jit = (tile_noise2(wp, 11.0, vec2(0.0)) - 0.5) * 0.014 * near;
        float e = one ? clamp((0.05 + jit - abs(ar - 1.74)) / pw + 0.5, 0.0, 1.0) * min(1.0, 0.10 / pw)
                      : clamp((0.075 + jit - abs(ad - 3.75)) / pw + 0.5, 0.0, 1.0) * min(1.0, 0.15 / pw);
        float dx = abs(fract(s / 12.0) * 12.0 - 1.5);
        float c = clamp((0.075 + jit - ad) / pw + 0.5, 0.0, 1.0) * min(1.0, 0.15 / pw) * clamp((1.5 - dx) / (pw * 3.0) + 0.5, 0.0, 1.0);
        float paint = max(one ? e : max(e * mk.x, c), mk.y);
        float wear = smoothstep(0.42, 0.82, tile_noise2(wp, 1.7, vec2(7.0)) * 0.55 + tile_noise2(wp, 23.0, vec2(0.0)) * 0.25 * near + hgt * 0.35 * near + 0.10);
        paint *= 1.0 - mix(0.22, 0.80 * wear, near);
        a = mix(a, vec3(0.50, 0.50, 0.47) * mix(0.85, 1.0, tile_noise2(wp, 0.8, vec2(0.0))), paint);
        gloss = mix(gloss, 0.38, paint);
    }
    return a;
}
