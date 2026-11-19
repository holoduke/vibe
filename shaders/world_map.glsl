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
float wm_river(float h, vec2 wp, float wl) {
    if (wp.x < -85.0 || wp.x > 2095.0 || wp.y < -110.0 || wp.y > 780.0) return h;
    float bed = wl - 2.4;
    if (h <= bed - 4.0) return h;
    vec2 q = wp + 12.0 * vec2(sin(wp.y * 0.043 + 1.3), sin(wp.x * 0.037 + 0.4)) + 5.0 * vec2(sin(wp.y * 0.113), sin(wp.x * 0.097 + 2.0));
    // Under the stone bridge the river is a pool: its channel lies 6 m further across (along
    // the bridge), it is 4 m wider each side, and its banks are steep - so the water stands
    // under two of the arches instead of one, and the road's end on the near bank is not cut.
    float pool = 1.0 - smoothstep(22.0, 55.0, distance(wp, vec2(43.4, 297.4)));
    q -= pool * 6.0 * vec2(0.719, 0.695);
    float d2 = 1e12;
    for (int c = 0; c < 5; ++c) {
        vec4 b = wm_riv_box[c];
        if (q.x < b.x || q.y < b.y || q.x > b.z || q.y > b.w) continue;
        for (int i = c * 5; i < c * 5 + 5; ++i) {
            vec2 a = wm_riv[i], ab = wm_riv[i + 1] - a;
            vec2 dv = q - (a + ab * clamp(dot(q - a, ab) / dot(ab, ab), 0.0, 1.0));
            d2 = min(d2, dot(dv, dv));
        }
    }
    if (d2 > 120.0 * 120.0) return h;
    float e = max(sqrt(d2) - (7.0 + 9.0 * smoothstep(1450.0, 1900.0, wp.x) + 4.0 * pool), 0.0);      // (wider toward its mouth)
    float cap = bed + mix(e * 0.45 + max(e - 8.0, 0.0) * 0.35 + max(e - 40.0, 0.0) * 0.5,
                          min(e, 6.0) * 0.45 + max(e - 6.0, 0.0) * 1.3, pool);
    float t = clamp(0.5 + 0.125 * (cap - h), 0.0, 1.0);                                  // smooth minimum, 4 m
    return mix(cap, h, t) - 4.0 * t * (1.0 - t);
}

// Ground height (m) at world position wp (m from home) for the landscape
// formula's value `a` (0..1) there. `wl`: the water level.
float wm_height_land(float a, vec2 wp, float wl) {
    float h = a * 120.0;
    // Home: the landscape and its ring of white mountains, as the formula has it.
    float w = smoothstep(1450.0, 1900.0, length(wp));     // (from beyond the home tile's corners: the tile's bake is the home formula alone)
    if (w <= 0.0) return h;
    // Hills and ranges: beyond home the land is low (under the snow line), with a
    // high range only where a slow noise says so. Scaled about 30 m.
    float hl = 30.0 + (h - 30.0) * mix(0.42, 1.12, wm_ranges(wp * 0.001));
    float v = wm_continent(wp * 0.001);
    // The coastal plain: toward the sea the land comes down to a beach just above the water.
    // (its hills keep half their height: dunes and low rises, not a table)
    hl = mix(wl + 0.7 + max(hl - 24.0, 0.0) * 0.5, hl, smoothstep(0.0, 0.03, v));
    // (and the ground climbs away from the beach: a few metres within the first few hundred)
    hl += 9.0 * smoothstep(0.0, 0.05, v) * (1.0 - smoothstep(0.05, 0.25, v));
    // The sea bed: shallows along the coast, a shelf, then the deep.
    float bed = wl - 1.5 - 7.0 * a - 14.0 * (1.0 - smoothstep(-0.012, 0.0, v)) - 45.0 * (1.0 - smoothstep(-0.09, -0.01, v));
    return mix(h, mix(bed, hl, smoothstep(-0.012, 0.012, v)), w);
}
// The launch site: a level terrace on the coast east of home (the rocket's pad stands on it;
// src/engine/terrain.cpp wm::pad and kLaunchSite* in terrain.h - keep in step).
const vec2  wm_pad_c = vec2(1600.0, 830.0);
const float wm_pad_h = 36.0;
float wm_pad(float h, vec2 wp) {
    vec2 d = wp - wm_pad_c;
    float r2 = dot(d, d);
    if (r2 > 220.0 * 220.0) return h;
    return mix(wm_pad_h, h, smoothstep(100.0, 220.0, sqrt(r2)));
}
float wm_height(float a, vec2 wp, float wl) { return wm_river(wm_pad(wm_height_land(a, wp, wl), wp), wp, wl); }

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
    return band * smoothstep(0.37, 0.53, m) * smoothstep(150.0, 230.0, distance(wp, wm_pad_c));      // (no woods at the launch site)
}
