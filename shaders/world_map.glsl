// The world, as one formula at every zoom level.
//
// Included by terrain_rm_noise_road.glsl (the raymarched landscape) and by
// compose.frag (the planet seen from high up and from space). It is the
// SAME text in both, so what is seen from orbit is what is found on the way
// down - there is no second map to keep in step.
//
// The includer defines WM_NOISE(p): 2D value noise 0..1 (both use the same
// hash).
//
// Zoom levels, coarse to fine:
//   continents and oceans      wm_continent: a warped fractal, 2400 .. 150 km
//   coasts, bays, capes        the same function's detail band, 90 .. 5 km
//   coastal plain / sea bed    wm_height: how the land meets the water
//   hills and mountain ranges  wm_height: where the land is low or high
//   the landscape itself       the caller's height formula `a` (0..1):
//                              ridges, valleys, down to metres
//
// Map position: km from home (east = +x). Near home the map is flat; on the
// planet it is the azimuthal map around home (compose.frag makes that).

float wm_fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; ++i) {
        s += a * WM_NOISE(p);
        p = mat2(0.8, -0.6, 0.6, 0.8) * p * 2.03 + 17.1;
        a *= 0.5;
    }
    return s / 0.96875;                                        // 0..1
}

float wm_fbm3(vec2 p) {
    return (0.5 * WM_NOISE(p) + 0.25 * WM_NOISE(mat2(0.8, -0.6, 0.6, 0.8) * p * 2.03 + 17.1) +
            0.125 * WM_NOISE(p * 4.1 - 9.4)) / 0.875;
}

// > 0: land, < 0: sea. The size says how far inland / out to sea, in
// "continent units" (0.01 is roughly a kilometre near a coast).
float wm_continent(vec2 mk) {
    // Continents: a fractal with its coordinates bent by a second noise, which
    // is what gives coastlines their peninsulas and gulfs instead of blobs.
    vec2 q = mk * 0.00042;
    vec2 bend = vec2(WM_NOISE(q * 2.3 + 5.2), WM_NOISE(q * 2.3 + 91.7)) - 0.5;
    float v = wm_fbm(q + bend * 0.55 + 21.3) - 0.5;
    // Home lies inside a continent (its heart is 600 km to the west), on its east coast:
    // the ocean begins just behind home's mountains, under 2 km east, and runs on for hundreds of km. Both are
    // laid out on a bent copy of the map (a few km of bend near home, tens of km far
    // away), or that coast would be a ruled line on the planet.
    vec2 wq = mk + (vec2(wm_fbm3(mk * 0.013 + 3.3), wm_fbm3(mk * 0.013 + 47.9)) - 0.5) * mix(3.0, 60.0, smoothstep(4.0, 90.0, length(mk)));
    // (broad and gentle, so the continent keeps the fractal's own outline)
    v += 0.20 * (1.0 - smoothstep(250.0, 1600.0, length(wq - vec2(-600.0, 0.0))));
    // (a round basin 650 km across whose rim passes just east of home - round, so that from orbit
    // it is a sea among the fractal's own, not a box)
    // Away from home its outline wanders by tens of km and its rim is broad: from orbit it is
    // an irregular sea, not a punched disc. At home the rim is where it was, and sharp.
    float dh = length(mk);
    vec2 wq2 = wq + (vec2(wm_fbm3(mk * 0.0035 + 13.1), wm_fbm3(mk * 0.0035 + 71.4)) - 0.5) * 170.0 * smoothstep(15.0, 250.0, dh);
    // Around home the land is firmly land (the fractal alone left it at the water's level
    // for kilometres: one flat sand table, home an island in it). The coast east of home is
    // then made by the basin alone, and its bays by the coast detail below.
    v = mix(v, max(v, 0.16), 1.0 - smoothstep(5.0, 25.0, dh));
    // (near home the shore wanders by a few hundred metres - bays and points a kilometre or
    // so across - instead of running dead straight behind the mountains)
    wq2.x += ((WM_NOISE(mk * 0.9 + 31.7) - 0.5) * 0.9 + (WM_NOISE(mk * 2.7 + 5.1) - 0.5) * 0.35) * (1.0 - smoothstep(8.0, 40.0, dh));
    float rim = mix(1.5, 45.0, smoothstep(10.0, 300.0, dh));
    v -= 0.46 * (1.0 - smoothstep(327.0 - rim, 327.0 + rim, length(wq2 - vec2(329.2, 0.0))));   // (its shore: about 1.9 km east of home, behind the mountains)
    // Far inland and far out to sea the coast's detail (at most 0.105) changes nothing that
    // is read from this value (everything saturates by 0.25): skip its eight noise lookups.
    if (abs(v) > 0.37) return v;
    // Coasts: bays, capes, islands (90 km down to 5 km).
    v += (wm_fbm(mk * 0.011 + 8.9) - 0.5) * 0.16;
    // The shoreline itself: coves and points (3 km down to 700 m).
    v += (wm_fbm3(mk * 0.35 + 61.0) - 0.5) * 0.05;
    return v;
}

// Where the high ranges stand (0 = low country, 1 = a range), at map position mk (km).
float wm_ranges(vec2 mk) {
    return smoothstep(0.60, 0.78, WM_NOISE(mk * 0.08 + 7.3) * 0.65 + WM_NOISE(mk * 0.21 + 3.1) * 0.35);
}

// The river: home's lake (the water around the castle) drains to the sea. Its course is laid
// through the low ground that was already there - past the chapel's east side, down the valley
// south of the castle, then east through two low passes - and the land is cut down to it: a
// channel below the water level, banks, and valley sides that open up where the ground is high.
// Bends: the course itself, and a slow sideways wander of the whole channel.
// (src/engine/terrain.cpp has the same course: keep the two in step.)
const vec2 wm_riv[26] = vec2[26](vec2(62.0, 36.0), vec2(86.0, 70.0), vec2(100.0, 112.0), vec2(96.0, 170.0), vec2(74.0, 228.0), vec2(56.0, 290.0), vec2(22.0, 350.0), vec2(48.0, 406.0), vec2(150.0, 412.0), vec2(248.0, 470.0), vec2(344.0, 536.0), vec2(410.0, 628.0), vec2(504.0, 634.0), vec2(600.0, 570.0), vec2(696.0, 536.0), vec2(792.0, 474.0), vec2(888.0, 456.0), vec2(984.0, 456.0), vec2(1080.0, 410.0), vec2(1176.0, 410.0), vec2(1272.0, 486.0), vec2(1368.0, 552.0), vec2(1464.0, 584.0), vec2(1600.0, 604.0), vec2(1752.0, 616.0), vec2(1950.0, 620.0));
const vec4 wm_riv_box[5] = vec4[5](vec4(-69.0, -89.0, 225.0, 415.0), vec4(-103.0, 165.0, 469.0, 661.0), vec4(219.0, 349.0, 917.0, 759.0), vec4(667.0, 285.0, 1397.0, 611.0), vec4(1147.0, 361.0, 2075.0, 745.0));     // (five stretches of five reaches, with the valley's reach around them)
// (2026-11-27) The river's water falls 1 m from the datum to the sea down its course
// (water_level.glsl wl_profile, water_fx.cpp river_level_profile - keep the three in step); its
// channel is cut that much deeper with it, so the lower river keeps its depth. Metres down the
// course to each point of wm_riv (terrain_rm_water.glsl kRivCum: the same numbers).
const float wm_riv_cum[26] = float[26](0.0, 41.6, 85.9, 144.0, 206.1, 270.6, 339.6, 401.3, 503.5, 617.4, 733.9, 847.1, 941.3, 1056.7, 1158.5, 1272.8, 1370.5, 1466.5, 1572.9, 1668.9, 1791.4, 1907.9, 2009.1, 2146.5, 2299.0, 2497.0);
float wm_riv_drop(float s) {
    float g = clamp((s - 350.0) / 2050.0, 0.0, 1.0);
    float r = smoothstep(696.0, 704.0, s) + smoothstep(1246.0, 1254.0, s) + smoothstep(1796.0, 1804.0, s);
    return 0.31 * g + 0.23 * r;
}
float wm_river(float h, vec2 wp, float wl) {
    if (wp.x < -85.0 || wp.x > 2095.0 || wp.y < -110.0 || wp.y > 780.0) return h;
    float bed = wl - 2.4;
    if (h <= bed - 5.0) return h;
    vec2 q = wp + 12.0 * vec2(sin(wp.y * 0.043 + 1.3), sin(wp.x * 0.037 + 0.4)) + 5.0 * vec2(sin(wp.y * 0.113), sin(wp.x * 0.097 + 2.0));
    // Under the stone bridge the river is a pool: its channel lies 6 m further across (along
    // the bridge), it is 4 m wider each side, and its banks are steep - so the water stands
    // under two of the arches instead of one, and the road's end on the near bank is not cut.
    float pool = 1.0 - smoothstep(22.0, 55.0, distance(wp, vec2(43.4, 297.4)));
    q -= pool * 6.0 * vec2(0.719, 0.695);
    float d2 = 1e12;
    float along = 0.0;
    for (int c = 0; c < 5; ++c) {
        vec4 b = wm_riv_box[c];
        if (q.x < b.x || q.y < b.y || q.x > b.z || q.y > b.w) continue;
        for (int i = c * 5; i < c * 5 + 5; ++i) {
            vec2 a = wm_riv[i], ab = wm_riv[i + 1] - a;
            float u = clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0);
            vec2 dv = q - (a + ab * u);
            float e2 = dot(dv, dv);
            if (e2 < d2) { d2 = e2; along = wm_riv_cum[i] + u * length(ab); }
        }
    }
    if (d2 > 120.0 * 120.0) return h;
    bed -= wm_riv_drop(along);
    float e = max(sqrt(d2) - (7.0 + 9.0 * smoothstep(1450.0, 1900.0, wp.x) + 4.0 * pool), 0.0);      // (wider toward its mouth)
    // (The valley's sides were 0.8 to 40 m out and 1.3 beyond, met by the land in a 4 m
    //  rounding: where the land is high - at the mouth, along the lower river - that left a
    //  rock ledge a few metres wide running along the valley's rim. Now one even slope and a
    //  12 m rounding: a valley's shoulder.)
    // (The cut ends 120 m from the channel. Whatever stood higher than the valley's side
    //  there was sliced off along that line - a wall in the mountain, with the uncut land
    //  overhanging it. From 70 m out the side now climbs ever faster, so that by 120 m it is
    //  above any mountain and the cut has run out into the land as it is.)
    float rim = max(e - 70.0, 0.0);
    float cap = bed + mix(e * 0.45 + max(e - 8.0, 0.0) * 0.35 + rim * rim * 0.15,
                          min(e, 6.0) * 0.45 + max(e - 6.0, 0.0) * 1.3, pool);
    float kr = mix(12.0, 4.0, pool);                                                      // smooth minimum: 12 m (4 at the pool)
    float t = clamp(0.5 + 0.5 * (cap - h) / kr, 0.0, 1.0);
    return mix(cap, h, t) - kr * t * (1.0 - t);
}

// Ground height (m) at world position wp (m from home) for the landscape
// formula's value `a` (0..1) there. `wl`: the water level.
// (`w`: how far the home formula has handed over, `rng`: the ranges there, `v`: land / sea)
float wm_height_core(float h, float a, float w, float rng, float v, float wl) {
    // Hills and ranges: beyond home the land is low (under the snow line), with a
    // high range only where a slow noise says so. Scaled about 30 m.
    float hl = 30.0 + (h - 30.0) * mix(0.42, 1.12, rng);
    // The coastal plain: toward the sea the land comes down to a beach just above the water.
    // (its hills keep half their height: dunes and low rises, not a table)
    hl = mix(wl + 0.7 + max(hl - 24.0, 0.0) * 0.5, hl, smoothstep(0.0, 0.03, v));
    // (and the ground climbs away from the beach: a few metres within the first few hundred)
    hl += 9.0 * smoothstep(0.0, 0.05, v) * (1.0 - smoothstep(0.05, 0.25, v));
    // The sea bed: shallows along the coast, a shelf, then the deep.
    float bed = wl - 1.5 - 7.0 * a - 14.0 * (1.0 - smoothstep(-0.012, 0.0, v)) - 45.0 * (1.0 - smoothstep(-0.09, -0.01, v));
    return mix(h, mix(bed, hl, smoothstep(-0.012, 0.012, v)), w);
}
float wm_height_land(float a, vec2 wp, float wl) {
    float h = a * 120.0;
    // Home: the landscape and its ring of white mountains, as the formula has it.
    float w = smoothstep(1450.0, 1900.0, length(wp));     // (from beyond the home tile's corners: the tile's bake is the home formula alone)
    if (w <= 0.0) return h;
    return wm_height_core(h, a, w, wm_ranges(wp * 0.001), wm_continent(wp * 0.001), wl);
}
// The launch site: a level terrace on the coast east of home (the rocket's pad stands on it;
// src/engine/terrain.cpp wm::pad and kLaunchSite* in terrain.h - keep in step).
const vec2  wm_pad_c = vec2(1600.0, 830.0);
const float wm_pad_h = 36.0;
float wm_pad(float h, vec2 wp) {
    vec2 d = wp - wm_pad_c;
    float r2 = dot(d, d);
    if (r2 > 270.0 * 270.0) return h;
    return mix(wm_pad_h, h, smoothstep(150.0, 270.0, sqrt(r2)));
}
// The service station on the coast highway north of the launch site: a level rectangle
// (src/engine/terrain.cpp wm::station and kStation* in terrain.h - keep in step). Toward the
// road, on its east side, the ground comes back to the natural slope within a few metres - the
// road's own shelf takes over there, and the ground under the road's centre line (which the
// road's height is taken from) is not touched.
const vec4  wm_st_rect = vec4(1586.0, 507.0, 1643.0, 583.0);       // x0, z0, x1, z1
const float wm_st_h = 41.825;
float wm_st_out(vec2 wp) {            // 0 on the rectangle .. 1 where the natural ground is back
    vec2 c = 0.5 * (wm_st_rect.xy + wm_st_rect.zw), e = 0.5 * (wm_st_rect.zw - wm_st_rect.xy);
    vec2 q = max(abs(wp - c) - e, vec2(0.0));
    float dx = q.x / (wp.x > c.x ? 5.0 : 22.0), dz = q.y / 22.0;
    return min(sqrt(dx * dx + dz * dz), 1.0);
}
float wm_station(float h, vec2 wp) {
    if (abs(wp.x - 1614.5) > 51.0 || abs(wp.y - 545.0) > 61.0) return h;
    return mix(wm_st_h, h, smoothstep(0.0, 1.0, wm_st_out(wp)));
}
float wm_height(float a, vec2 wp, float wl) { return wm_river(wm_station(wm_pad(wm_height_land(a, wp, wl), wp), wp), wp, wl); }

// Where woods grow (0 = open ground, 1 = closed forest) at world position wp (m), for ground
// height h, the ground normal's y, and the water level. One rule for the whole world: the far
// landscape is coloured by it and the trees that stand there are placed by it
// (src/engine/terrain.cpp wm::forest - keep in step), so the wood seen from the air is the wood
// found on the ground.
//   height:   from just above the shore up to a treeline (thinning out over the last 14 m)
//   slope:    not on steep ground
//   moisture: three scales of noise - woods a kilometre across, stands of a few hundred metres,
//             ragged edges - wetter low down by the water
//   country:  an 8 km pattern of wooded and open country
float wm_forest(vec2 wp, float h, float ny, float wl) {
    float alt  = h - wl;
    float band = smoothstep(2.0, 6.0, alt) * (1.0 - smoothstep(40.0, 54.0, alt)) * smoothstep(0.70, 0.84, ny);
    if (band <= 0.0) return 0.0;
    float m = WM_NOISE(wp * 0.0011 + 3.7) * 0.55 + WM_NOISE(wp * 0.0043 + 17.0) * 0.30 + WM_NOISE(wp * 0.017 + 5.0) * 0.15;
    m += 0.10 * (1.0 - smoothstep(4.0, 25.0, alt));
    m += (WM_NOISE(wp * 0.00013 + 41.0) - 0.5) * 0.35;
    if (abs(wp.x - 1614.5) < 51.0 && abs(wp.y - 545.0) < 61.0) band *= smoothstep(0.3, 1.0, wm_st_out(wp));      // (nor on the service station's plot)
    return band * smoothstep(0.37, 0.53, m) * smoothstep(150.0, 230.0, distance(wp, wm_pad_c));      // (no woods at the launch site)
}

// ======================================================================
// The globe.
//
// The planet is a sphere of radius wm_R. Home is its top: in the planet's own frame home is
// (0, 1, 0), east of home +x, south +z. Everything above is written on a flat map of the
// country round home ("home coordinates": metres east and south of home, which on the sphere
// are the azimuthal coordinates round home - wm_az). That map cannot go round a sphere, so
// away from home the world is made on the sphere itself, from 3D noise at the point:
//   the fine shape of the land (the FBM `a`)   40 - 90 km from home it changes over
//   continents and ranges                      1800 - 3200 km
// (src/engine/terrain.cpp + globe.cpp mirror all of it - keep in step.)
// ======================================================================
const float wm_R = 6360000.0;
const float wm_blend0 = 40000.0, wm_blend1 = 90000.0;      // (m from home: engine/globe.h kBlend0 / kBlend1)

// Home coordinates (m) of the point G (unit vector, planet frame).
vec2 wm_az(vec3 G) {
    float sth = length(G.xz);
    return wm_R * G.xz * (sth > 1e-6 ? atan(sth, G.y) / sth : 1.0);
}
float wm_hash3(ivec3 c) {
    uint h = uint(c.x) * 0x8da6b343u ^ uint(c.y) * 0xd8163841u ^ uint(c.z) * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return float(h & 0xFFFFFFu) * (1.0 / 16777216.0);
}
// 3D value noise at lattice cell `i`, position `f` in it: value and gradient.
vec4 wm_noised3(ivec3 i, vec3 f) {
    vec3 u = f * f * (3.0 - 2.0 * f), du = 6.0 * f * (1.0 - f);
    float a = wm_hash3(i), b = wm_hash3(i + ivec3(1, 0, 0)), c = wm_hash3(i + ivec3(0, 1, 0)), d = wm_hash3(i + ivec3(1, 1, 0));
    float e = wm_hash3(i + ivec3(0, 0, 1)), g = wm_hash3(i + ivec3(1, 0, 1)), k = wm_hash3(i + ivec3(0, 1, 1)), l = wm_hash3(i + ivec3(1, 1, 1));
    float v = mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, g, u.x), mix(k, l, u.x), u.y), u.z);
    vec3 gr = vec3(mix(mix(b - a, d - c, u.y), mix(g - e, l - k, u.y), u.z),
                   mix(mix(c - a, d - b, u.x), mix(k - e, l - g, u.x), u.z),
                   mix(mix(e - a, g - b, u.x), mix(k - c, l - d, u.x), u.y));
    return vec4(v, du * gr);
}
float wm_noise3(vec3 p) {
    vec3 fl = floor(p);
    return wm_noised3(ivec3(fl), p - fl).x;
}
float wm_fbm3d(vec3 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; ++i) { s += a * wm_noise3(p); p = p * 2.03 + 17.1; a *= 0.5; }
    return s / 0.96875;
}
// Land (> 0) and sea (< 0) on the sphere, as wm_continent has them on the map.
float wm_continent3(vec3 G) {
    vec3 q = G * 2.6712;                                                   // (the map's 0.00042 per km)
    vec3 bend = vec3(wm_noise3(q * 2.3 + 5.2), wm_noise3(q * 2.3 + 91.7), wm_noise3(q * 2.3 + 33.1)) - 0.5;
    float v = (wm_fbm3d(q + bend * 0.55 + 21.3) - 0.5) * 1.25;             // (3D value noise is a little flatter than 2D)
    if (abs(v) > 0.37) return v;
    vec3 k = G * 6360.0;                                                   // km
    v += (wm_fbm3d(k * 0.011 + 8.9) - 0.5) * 0.16;
    v += ((0.5 * wm_noise3(k * 0.35 + 61.0) + 0.25 * wm_noise3(k * 0.71 + 78.1) + 0.125 * wm_noise3(k * 1.435 - 9.4)) / 0.875 - 0.5) * 0.05;
    return v;
}
float wm_ranges3(vec3 G) {
    vec3 k = G * 6360.0;
    return smoothstep(0.56, 0.72, wm_noise3(k * 0.08 + 7.3) * 0.65 + wm_noise3(k * 0.21 + 3.1) * 0.35);
}
// Land / sea and ranges anywhere: the map's near home, the sphere's far away.
// (`mk`: home coordinates in km)
float wm_continent_globe(vec2 mk, vec3 G) {
    float wC = smoothstep(1800.0, 3200.0, length(mk));
    float v = wC < 1.0 ? wm_continent(mk) : 0.0;
    if (wC > 0.0) v = mix(v, wm_continent3(G), wC);
    return v;
}
float wm_ranges_globe(vec2 mk, vec3 G) {
    float wC = smoothstep(1800.0, 3200.0, length(mk));
    float v = wC < 1.0 ? wm_ranges(mk) : 0.0;
    if (wC > 0.0) v = mix(v, wm_ranges3(G), wC);
    return v;
}
// Ground height beyond the home country (no river, no launch terrace out here).
float wm_height_globe(float a, vec2 hp, vec3 G, float wl) {
    vec2 mk = hp * 0.001;
    return wm_height_core(a * 120.0, a, 1.0, wm_ranges_globe(mk, G), wm_continent_globe(mk, G), wl);
}
// Woods, anywhere.
float wm_forest_globe(vec2 hp, vec3 G, float h, float ny, float wl) {
    float wT = smoothstep(wm_blend0, wm_blend1, length(hp));
    float f2 = wT < 1.0 ? wm_forest(hp, h, ny, wl) : 0.0;
    if (wT <= 0.0) return f2;
    float alt  = h - wl;
    float band = smoothstep(2.0, 6.0, alt) * (1.0 - smoothstep(40.0, 54.0, alt)) * smoothstep(0.70, 0.84, ny);
    float f3 = 0.0;
    if (band > 0.0) {
        vec3 P = G * wm_R;
        float m = wm_noise3(P * 0.0011 + 3.7) * 0.55 + wm_noise3(P * 0.0043 + 17.0) * 0.30 + wm_noise3(P * 0.017 + 5.0) * 0.15;
        m += 0.10 * (1.0 - smoothstep(4.0, 25.0, alt));
        m += (wm_noise3(P * 0.00013 + 41.0) - 0.5) * 0.35;
        f3 = band * smoothstep(0.37, 0.53, m);
    }
    return mix(f2, f3, wT);
}
// The fine shape of the land on the sphere: the landscape's FBM made from 3D noise. A float
// cannot hold a position on the planet finely enough for its small octaves, so the includer
// supplies, per octave, the lattice cell and the position in it of a nearby origin
// (WM_CELL(i), WM_FRAC(i) - worked out on the CPU in double precision), and `D` is the point's
// offset from that origin in metres (planet frame).
#ifdef WM_CELL
float wm_shape3(vec3 D, float oct_f) {
    int   oct = int(ceil(oct_f));
    float fr  = 1.0 - (float(oct) - oct_f);
    float a = 0.0, b = 1.0, s = 0.003;
    vec3  d = vec3(0.0);
    for (int i = 0; i < 12; ++i) {
        if (i >= oct) break;
        vec3 c  = WM_FRAC(i) + D * s;
        vec3 fl = floor(c);
        vec4 n  = wm_noised3(WM_CELL(i) + ivec3(fl), c - fl);
        float w = (i == oct - 1) ? fr : 1.0;
        d += n.yzw * (0.8 * w);
        a += b * n.x * w / (1.0 + dot(d, d));
        b *= 0.5; s *= 2.0;
    }
    return a;
}
#endif
