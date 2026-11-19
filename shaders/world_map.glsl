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
    // the ocean begins a few km east of home and runs on for hundreds of km. Both are
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
    float rim = mix(2.0, 45.0, smoothstep(10.0, 300.0, dh));
    v -= 0.46 * (1.0 - smoothstep(327.0 - rim, 327.0 + rim, length(wq2 - vec2(331.5, 0.0))));
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

// Ground height (m) at world position wp (m from home) for the landscape
// formula's value `a` (0..1) there. `wl`: the water level.
float wm_height(float a, vec2 wp, float wl) {
    float h = a * 120.0;
    // Home: the landscape and its ring of white mountains, as the formula has it.
    float w = smoothstep(1500.0, 2600.0, length(wp));
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
